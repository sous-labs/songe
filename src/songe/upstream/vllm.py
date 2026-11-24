"""vLLM dynamic PEFT adapters; unload only instances requested by Songe."""

from pathlib import Path

import httpx

from ..models import discovery
from .protection import (
    admit_load,
    claim,
    close_owned,
    guarded_load,
    live_claims,
    weight_bytes,
)


class VLLM:
    name = "vllm"
    serves = ("peft-adapter",)

    def __init__(self, config, *, transport=None, memory_probe=None):
        self.config, self.transport = config, transport
        self.url = config.upstream_url.rstrip("/")
        if not self.url.endswith("/v1"):
            self.url += "/v1"
        self.root_url = self.url.removesuffix("/v1")
        self.timeout = getattr(config, "upstream_timeout", 180.0)
        # vLLM has no standard model-bytes HTTP field. A platform/supervisor
        # can supply measured reclaimable bytes; unknown is conservatively 0.
        self.memory_probe = memory_probe
        self.routes = {}

    def discover(self):
        return discovery.vllm(self.url, self.transport)

    async def served_contexts(self):
        """Context the user's vLLM serves each model with (its --max-model-len),
        from /v1/models "max_model_len"."""
        url = (getattr(self.config, "server_source_url", "") or self.url).rstrip("/")
        if not url.endswith("/v1"):
            url += "/v1"
        async with httpx.AsyncClient(
            transport=self.transport, timeout=self.timeout, trust_env=False
        ) as client:
            response = await client.get(url + "/models")
            response.raise_for_status()
            models = response.json().get("data", [])
        return {m["id"]: int(m["max_model_len"]) for m in models if m.get("max_model_len")}

    async def _request(self, method, path, **kwargs):
        async with httpx.AsyncClient(
            transport=self.transport, timeout=self.timeout, trust_env=False
        ) as client:
            response = await client.request(method, self.root_url + path, **kwargs)
            response.raise_for_status()
            return (
                response.json()
                if response.content
                and response.headers.get("content-type", "").startswith(
                    "application/json"
                )
                else None
            )

    async def _models(self):
        return (await self._request("GET", "/v1/models"))["data"]

    async def resident(self, version):
        if version.get("id") == "base" or version.get("kind") == "base":
            from .reuse import served_base

            # The route stays while the server is asked again: a request
            # that arrives meanwhile must not fall back to the owned port.
            binding = await served_base(self.config, version, transport=self.transport)
            if not binding and getattr(self.config, "server_source_url", ""):
                binding = await served_base(
                    self.config, version, transport=self.transport, url=self.url
                )
            if binding:
                self.routes[version["identifier"]] = binding
                return True
            self.routes.pop(version["identifier"], None)
            return False
        existing = next(
            (m for m in await self._models() if m["id"] == version["identifier"]), None
        )
        if existing is None:
            return False
        adapter = Path(version.get("adapter") or version["artifact"]).resolve()
        if (
            not existing.get("parent")
            or Path(existing.get("root", "")).resolve() != adapter
        ):
            raise RuntimeError(
                f"vLLM identifier {version['identifier']} already refers to other weights"
            )
        return True

    @guarded_load
    async def load(self, version):
        if version.get("id") == "base" or version.get("kind") == "base":
            if not await self.resident(version):
                if not hasattr(self, "supervisor"):
                    raise RuntimeError("vLLM is not serving the configured base")
                await self.supervisor.restart(version)
                if not await self.resident(version):
                    raise RuntimeError("vLLM is not serving the configured base")
            return
        adapter = str(Path(version.get("adapter") or version["artifact"]).resolve())
        existing = next(
            (m for m in await self._models() if m["id"] == version["identifier"]), None
        )
        if existing:
            if not existing.get("parent") or Path(
                existing.get("root", "")
            ).resolve() != Path(adapter):
                raise RuntimeError(
                    f"vLLM identifier {version['identifier']} already refers to other weights"
                )
            return
        await admit_load(self, version)
        claim(self, version)
        await self._request(
            "POST",
            "/v1/load_lora_adapter",
            json={
                "lora_name": version["identifier"],
                "lora_path": adapter,
            },
        )

    def request_model(self, version):
        return self.routes.get(version["identifier"], {}).get(
            "model", version["identifier"]
        )

    def request_url(self, version):
        return self.routes.get(version["identifier"], {}).get("url", self.url)

    async def unload(self, version):
        if version["identifier"] not in getattr(self, "owned", {}):
            return
        try:
            still_owned = await self.resident(self.owned[version["identifier"]])
        except RuntimeError:
            still_owned = False
        if still_owned:
            await self._request(
                "POST",
                "/v1/unload_lora_adapter",
                json={"lora_name": version["identifier"]},
            )
        self.owned.pop(version["identifier"], None)

    async def ensure(self, version):
        try:
            await self.load(version)
        except httpx.ConnectError:
            if not hasattr(self, "supervisor"):
                raise
            await self.supervisor.restart(version)
            await self.load(version)

    async def park(self, on_snapshot=None):
        parked = [v | {"songe_requested": True} for v in await live_claims(self)]
        if on_snapshot:
            on_snapshot(parked)
        # The base and other adapters belong to the user. Never sleep their runtime.
        for item in parked:
            await self.unload(item)
        if hasattr(self, "supervisor") and self.supervisor.process is not None:
            await self.supervisor.stop()
        return parked

    async def restore(self, parked, managed):
        for item in parked:
            if item.get("songe_requested") and item["identifier"] not in managed:
                await self.ensure(item)

    async def loaded_bytes(self):
        await live_claims(self)
        if hasattr(self, "supervisor") and self.supervisor.process is not None:
            return self.supervisor.memory_bytes()
        return sum(
            weight_bytes(v["artifact"]) for v in getattr(self, "owned", {}).values()
        )

    async def close(self):
        await close_owned(self)
        if hasattr(self, "supervisor"):
            await self.supervisor.close()
