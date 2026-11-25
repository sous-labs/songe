"""llama-server LoRA scale switching, with explicit supervisor hooks.

The HTTP API only switches pre-registered adapters. New adapters must already
be registered with --lora/--lora-init-without-apply. A new base or actual memory
reclamation needs a supervisor, never a pretend unload by setting scale=0.
"""

import asyncio
from pathlib import Path

import httpx

from ..models import discovery


class LlamaCPP:
    name = "llamacpp"
    serves = ("gguf-lora", "gguf-full")

    def __init__(
        self, config, *, transport=None, restart=None, stop=None, memory_probe=None
    ):
        self.config, self.transport = config, transport
        self.url = config.upstream_url.rstrip("/")
        if not self.url.endswith("/v1"):
            self.url += "/v1"
        self.root_url = self.url.removesuffix("/v1")
        self.timeout = getattr(config, "upstream_timeout", 180.0)
        # restart(version) starts/restarts the *supervised* server with a base;
        # stop() waits for its process to exit and release accelerator memory.
        self.restart, self.stop, self.memory_probe = restart, stop, memory_probe
        self.identifiers = {}
        self.applied = set()
        self.routes = {}

    def discover(self):
        # The owned llama-server runs only while Songe needs it; the user's own
        # server (server_source_url) is always up and serves the base.
        source = getattr(self.config, "server_source_url", "")
        return discovery.llamacpp(source or self.url, self.transport)

    async def served_contexts(self):
        """Context the user's llama-server runs each model with (its -c), from
        /v1/models "meta". Older builds without it leave the fallback."""
        url = (getattr(self.config, "server_source_url", "") or self.url).rstrip("/")
        if not url.endswith("/v1"):
            url += "/v1"
        async with httpx.AsyncClient(
            transport=self.transport, timeout=self.timeout, trust_env=False
        ) as client:
            response = await client.get(url + "/models")
            response.raise_for_status()
            models = response.json().get("data", [])
        served = {
            m["id"]: int(m["meta"]["n_ctx"])
            for m in models
            if isinstance(m.get("meta"), dict) and m["meta"].get("n_ctx")
        }
        # llama-swap: each loaded model's command says its -c. Without one,
        # the fallback stays rather than a guess at llama.cpp's default.
        root = url.removesuffix("/v1")
        running = await asyncio.to_thread(
            discovery.llamaswap_running, root, self.transport
        )
        for entry in running or []:
            if not isinstance(entry, dict):
                continue
            context = discovery.command_flag(entry.get("cmd"), "-c", "--ctx-size")
            if entry.get("model") and context and context.isdigit() and int(context):
                served.setdefault(entry["model"], int(context))
        return served

    async def listing(self):
        """What is loaded now when the user's server is llama-swap: its models
        from /running, and the versions Songe's own llama-server is serving.
        None for a plain llama-server, whose one model is always loaded.
        Background work for a model that is not loaded would swap out the
        user's, or read with another learner's weights."""
        url = getattr(self.config, "server_source_url", "") or self.url
        loaded = await asyncio.to_thread(discovery.llamaswap_loaded, url, self.transport)
        if loaded is None:
            return None
        return [{"identifier": i, "type": "llm"} for i in loaded + await self.serving()]

    async def serving(self):
        """Identifiers of the versions Songe's own llama-server serves right
        now: its base and the adapters applied on it; [] while it is stopped."""
        if not self.owns_server():
            return []
        try:
            props = await self._request("GET", "/props")
        except httpx.HTTPError:
            return []
        base = str(Path((props or {}).get("model_path", "")).resolve())
        found = [self.identifiers[base]] if base in self.identifiers else []
        found += [self.identifiers[p] for p in sorted(self.applied) if p in self.identifiers]
        return found

    async def _request(self, method, path, **kwargs):
        async with httpx.AsyncClient(
            transport=self.transport, timeout=self.timeout, trust_env=False
        ) as client:
            response = await client.request(method, self.root_url + path, **kwargs)
            response.raise_for_status()
            return response.json() if response.content else None

    async def load(self, version):
        # Publication uses load directly, including after the owned endpoint
        # was parked. Restore it before inspecting or switching its adapters.
        try:
            await self._load(version)
        except httpx.ConnectError:
            if self.restart is None:
                raise
            await self.restart(version)
            await self._load(version)

    async def _load(self, version):
        from .reuse import served_base

        self.routes.pop(version["identifier"], None)
        if getattr(self.config, "server_source_url", "") or not self.owns_server():
            binding = await served_base(self.config, version, transport=self.transport)
            if binding:
                self.routes[version["identifier"]] = binding
                return
        adapter = version.get("adapter")
        artifact = Path(version["artifact"])
        kind = version.get("kind", "gguf-lora" if adapter else "gguf-full")
        if kind == "gguf-lora":
            desired = Path(adapter or artifact).resolve()
            if version.get("base"):
                props = await self._request("GET", "/props")
                if (
                    Path(props["model_path"]).resolve()
                    != Path(version["base"]).resolve()
                ):
                    if self.restart is None:
                        raise RuntimeError(
                            "A new llama.cpp base requires a supervised restart"
                        )
                    await self.restart(
                        {**version, "artifact": version["base"], "kind": "gguf-full"}
                    )
            adapters = await self._request("GET", "/lora-adapters")
            match = next(
                (a for a in adapters if Path(a["path"]).resolve() == desired), None
            )
            if match is None:
                if self.owns_server() and self.restart:
                    await self.restart(version)
                    adapters = await self._request("GET", "/lora-adapters")
                    match = next(
                        (a for a in adapters if Path(a["path"]).resolve() == desired),
                        None,
                    )
                if match is None:
                    raise RuntimeError(
                        f"llama.cpp adapter {desired} is not registered; start the server with --lora and --lora-init-without-apply"
                    )
            if not self.owns_server():
                raise RuntimeError(
                    "Cannot switch global adapters on a user-owned llama.cpp server; select a separate Songe endpoint"
                )
            await self._request(
                "POST",
                "/lora-adapters",
                json=[
                    {
                        "id": a["id"],
                        "scale": 1.0 if a["id"] == match["id"] else a["scale"],
                    }
                    for a in adapters
                    if a["id"] == match["id"]
                    or str(Path(a["path"]).resolve()) not in self.applied
                ],
            )
            self.applied.add(str(desired))
            self.identifiers[str(desired)] = version["identifier"]
        elif kind == "gguf-full":
            props = await self._request("GET", "/props")
            if Path(props["model_path"]).resolve() != artifact.resolve():
                if self.restart is None:
                    raise RuntimeError(
                        "A new llama.cpp base requires a supervised restart"
                    )
                await self.restart(version)
            self.identifiers[str(artifact.resolve())] = version["identifier"]
            if self.applied:
                adapters = await self._request("GET", "/lora-adapters")
                await self._request(
                    "POST",
                    "/lora-adapters",
                    json=[
                        {"id": a["id"], "scale": a["scale"]}
                        for a in adapters
                        if str(Path(a["path"]).resolve()) not in self.applied
                    ],
                )
                self.applied.clear()
        else:
            raise ValueError(f"Unsupported llama.cpp artifact kind: {kind}")

    def request_model(self, version):
        return self.routes.get(version["identifier"], {}).get(
            "model", version["identifier"]
        )

    def request_url(self, version):
        return self.routes.get(version["identifier"], {}).get("url", self.url)

    async def unload(self, version):
        if version["identifier"] in self.routes:
            return
        adapter = version.get("adapter")
        kind = version.get("kind", "gguf-lora" if adapter else "gguf-full")
        if kind == "gguf-lora":
            path = Path(adapter or version["artifact"]).resolve()
            if str(path) not in self.applied:
                return
            adapters = await self._request("GET", "/lora-adapters")
            # Preserve other active adapters when unloading a single version.
            await self._request(
                "POST",
                "/lora-adapters",
                json=[
                    {"id": a["id"], "scale": a["scale"]}
                    for a in adapters
                    if Path(a["path"]).resolve() != path and a["scale"] != 0
                ],
            )
            self.applied.discard(str(path))
        else:
            if not self.owns_server():
                return
            props = await self._request("GET", "/props")
            if (
                Path(props["model_path"]).resolve()
                != Path(version["artifact"]).resolve()
            ):
                return  # The old base was already replaced by a newer version.
            if self.stop is None:
                raise RuntimeError(
                    "Releasing the llama.cpp base requires a supervised stop hook"
                )
            await self.stop()

    async def ensure(self, version):
        await self.load(version)

    def owns_server(self):
        if hasattr(self, "supervisor"):
            return self.supervisor.process is not None
        return self.stop is not None and self.restart is not None

    async def park(self, on_snapshot=None):
        if not self.owns_server():
            if on_snapshot:
                on_snapshot([])
            return []
        props = await self._request("GET", "/props")
        models = await self._request("GET", "/v1/models")
        adapters = await self._request("GET", "/lora-adapters")
        adapters = [
            {
                **a,
                "identifier": self.identifiers.get(
                    str(Path(a["path"]).resolve()), a["path"]
                ),
            }
            for a in adapters
        ]
        identifier = (
            models["data"][0]["id"] if models.get("data") else props["model_path"]
        )
        parked = [
            {
                "identifier": identifier,
                "artifact": props["model_path"],
                "kind": "gguf-full",
                "adapters": adapters,
            }
        ]
        if on_snapshot:
            on_snapshot(parked)
        await self.stop()
        return parked

    async def restore(self, parked, managed):
        for item in parked:
            if item["identifier"] in managed:
                continue
            if self.restart is None:
                raise RuntimeError(
                    "Restoring llama.cpp requires a supervised restart hook"
                )
            await self.restart(item)
            # Resolve ids again: a restart need not preserve numeric adapter ids.
            current = await self._request("GET", "/lora-adapters")
            scales = {
                str(Path(a["path"]).resolve()): a["scale"]
                for a in item.get("adapters", [])
                if a.get("identifier", a["path"]) not in managed
            }
            active = [
                {"id": a["id"], "scale": scales[str(Path(a["path"]).resolve())]}
                for a in current
                if str(Path(a["path"]).resolve()) in scales
            ]
            if len(active) != len(scales):
                raise RuntimeError(
                    "The restored llama.cpp server is missing a parked adapter"
                )
            await self._request("POST", "/lora-adapters", json=active)

    async def loaded_bytes(self):
        return (
            max(0, int(self.memory_probe()))
            if self.owns_server() and self.memory_probe is not None
            else 0
        )

    async def close(self):
        if hasattr(self, "supervisor"):
            await self.supervisor.close()
