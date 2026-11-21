"""Ollama fused-GGUF import and lifecycle through its local HTTP API."""

import asyncio
import hashlib
import re
from pathlib import Path

import httpx

from ..models import discovery
from .protection import admit_load, claim, close_owned, guarded_load, live_claims

IO_CHUNK_BYTES = 1024 * 1024  # Bounded streaming IO, not a model-size threshold.


def _tag(identifier):
    return (
        identifier if ":" in identifier.rsplit("/", 1)[-1] else identifier + ":latest"
    )


class Ollama:
    name = "ollama"
    serves = ("gguf-full",)

    def __init__(self, config, *, transport=None, store=None):
        self.config, self.transport = config, transport
        self.store = store
        self.url = config.upstream_url.rstrip("/")
        if not self.url.endswith("/v1"):
            self.url += "/v1"
        self.root_url = self.url.removesuffix("/v1")
        self.timeout = getattr(config, "upstream_timeout", 180.0)
        self.routes = {}
        self.reused = {}
        self.fingerprints = {}

    def discover(self):
        return discovery.ollama(
            folder=getattr(self.config, "ollama_models_dir", None),
            url=self.root_url,
            transport=self.transport,
        )

    def client(self):
        return httpx.AsyncClient(
            transport=self.transport, timeout=self.timeout, trust_env=False
        )

    async def _request(self, method, path, **kwargs):
        async with self.client() as client:
            response = await client.request(method, self.root_url + path, **kwargs)
            response.raise_for_status()
            return response.json() if response.content else {}

    async def _running(self):
        return (await self._request("GET", "/api/ps")).get("models", [])

    async def served_contexts(self):
        """Context Ollama runs each loaded model with (its OLLAMA_CONTEXT_LENGTH,
        4096 by default), from /api/ps. A model that is not loaded is not
        listed and keeps the fallback. Songe's own loads set no num_ctx, so
        every value is the user's server setting."""
        return {
            m["name"]: int(m["context_length"])
            for m in await self._running()
            if m.get("name") and m.get("context_length")
        }

    async def digest(self, version):
        def fingerprint():
            path = Path(version["artifact"]).resolve()
            info = path.stat()
            stamp = (
                info.st_dev,
                info.st_ino,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
            )
            cached = self.fingerprints.get(path)
            if cached and cached[0] == stamp:
                return cached[1]
            with path.open("rb") as source:
                digest = "sha256:" + hashlib.file_digest(source, "sha256").hexdigest()
            self.fingerprints[path] = (stamp, digest)
            return digest

        return await asyncio.to_thread(fingerprint)

    async def same_weights(self, version):
        try:
            details = await self._request(
                "POST", "/api/show", json={"model": version["identifier"]}
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return False
            raise
        match = re.search(
            r'^FROM\s+.*sha256[:-]([a-f0-9]{64})["\s]*$',
            details.get("modelfile", ""),
            re.M,
        )
        expected = version.get("digest") or await self.digest(version)
        return bool(match and "sha256:" + match[1] == expected)

    async def resident(self, version):
        running = {_tag(m.get("name", m.get("model"))) for m in await self._running()}
        if _tag(version["identifier"]) not in running:
            return False
        if not await self.same_weights(version):
            raise RuntimeError(
                "Ollama identifier already refers to other or unverifiable weights"
            )
        return True

    async def reuse(self, version):
        identifier = version["identifier"]
        self.routes.pop(identifier, None)
        running = await self._running()
        if _tag(identifier) not in {
            _tag(m.get("name", m.get("model"))) for m in running
        }:
            getattr(self, "owned", {}).pop(identifier, None)
        expected = version.get("digest") or await self.digest(version)
        # /api/ps reports the manifest digest. Compare those directly when
        # supplied, and otherwise verify the GGUF layer via /api/show.
        for model in sorted(
            running,
            key=lambda m: _tag(m.get("name", m.get("model"))) != _tag(identifier),
        ):
            name = model.get("name", model.get("model"))
            same = (
                bool(version.get("manifest_digest"))
                and model.get("digest") == version["manifest_digest"]
            ) or await self.same_weights(
                version | {"identifier": name, "digest": expected}
            )
            if same:
                self.routes[identifier] = name
                if name not in getattr(self, "owned", {}):
                    self.reused[identifier] = dict(
                        version,
                        identifier=name,
                        digest=expected,
                        manifest_digest=model.get("digest"),
                    )
                return True
            if _tag(name) == _tag(identifier):
                raise RuntimeError(
                    "Ollama identifier already refers to other or unverifiable weights"
                )
        return False

    async def replacement_ticket(self, version):
        return self.reused.get(version["identifier"])

    async def release_replaced(self, ticket):
        live = next(
            (
                m
                for m in await self._running()
                if _tag(m.get("name", m.get("model"))) == _tag(ticket["identifier"])
            ),
            None,
        )
        if (
            live
            and ticket.get("manifest_digest") == live.get("digest")
            and await self.same_weights(ticket)
        ):
            await self._warm(ticket["identifier"], 0)
        self.reused = {k: v for k, v in self.reused.items() if v != ticket}

    def request_model(self, version):
        return self.routes.get(version["identifier"], version["identifier"])

    @guarded_load
    async def load(self, version):
        if await self.reuse(version):
            return
        path = Path(version["artifact"])

        digest = await self.digest(version)
        try:
            details = await self._request(
                "POST", "/api/show", json={"model": version["identifier"]}
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
            details = None
        running = {_tag(m.get("name", m.get("model"))) for m in await self._running()}
        if details is not None:
            # Verify the GGUF blob behind an existing tag; never overwrite it.
            # https://github.com/ollama/ollama/blob/main/docs/api.md#show-model-information
            match = re.search(
                r"^FROM\s+.*sha256[:-]([a-f0-9]{64})[\"\s]*$",
                details.get("modelfile", ""),
                re.M,
            )
            if not match or "sha256:" + match[1] != digest:
                raise RuntimeError(
                    "Ollama identifier already refers to other or unverifiable weights"
                )
            if _tag(version["identifier"]) in running:
                return  # Reuse never adopts ownership of a user's instance.
            await admit_load(self, version)
            claim(self, version)
            await self._warm(version["identifier"], version.get("keep_alive"))
            return
        if _tag(version["identifier"]) in running:
            raise RuntimeError("Ollama running identifier could not be verified")
        # Persist before upload/create: an interrupted import may already have
        # created its tag even when the request never returned to this process.
        if self.store:
            imports = self.store.get("runtime_imports", {})
            imports[version["identifier"]] = dict(version) | {
                "digest": digest,
                "installation_id": self.store.get("installation_id"),
                "state": "uploading",
            }
            self.store.set("runtime_imports", imports)
        async with self.client() as client:
            response = await client.head(self.root_url + "/api/blobs/" + digest)
            if response.status_code == 404:

                async def content():
                    with path.open("rb") as source:
                        while chunk := source.read(IO_CHUNK_BYTES):
                            yield chunk

                response = await client.post(
                    self.root_url + "/api/blobs/" + digest, content=content()
                )
            response.raise_for_status()
        # The files API is the API equivalent of a Modelfile's FROM <gguf>.
        await admit_load(self, version)
        claim(self, version)
        if self.store:
            imports = self.store.get("runtime_imports", {})
            imports[version["identifier"]]["state"] = "creating"
            self.store.set("runtime_imports", imports)
        result = await self._request(
            "POST",
            "/api/create",
            json={
                "model": version["identifier"],
                "files": {path.name: digest},
                "stream": False,
            },
        )
        if result.get("error") or result.get("status") != "success":
            raise RuntimeError(f"Ollama import failed: {result.get('error', result)}")
        if self.store:
            imports = self.store.get("runtime_imports", {})
            imports[version["identifier"]]["state"] = "created"
            self.store.set("runtime_imports", imports)
            tags = (await self._request("GET", "/api/tags")).get("models", [])
            manifest = next(
                (
                    m.get("digest")
                    for m in tags
                    if _tag(m["name"]) == _tag(version["identifier"])
                ),
                None,
            )
            imports = self.store.get("runtime_imports", {})
            imports[version["identifier"]]["manifest_digest"] = manifest
            self.store.set("runtime_imports", imports)
        await self._warm(version["identifier"], version.get("keep_alive"))

    async def _warm(self, identifier, keep_alive=None):
        # Without an explicit keep_alive the user's Ollama setting (default or
        # OLLAMA_KEEP_ALIVE) decides how long the model stays in memory.
        await self._request(
            "POST",
            "/api/generate",
            json={"model": identifier, "prompt": "", "stream": False}
            | ({} if keep_alive is None else {"keep_alive": keep_alive}),
        )
        if keep_alive == 0:
            # The unload response can precede removal from /api/ps. Keep the
            # ownership claim until the runtime confirms physical release.
            try:
                async with asyncio.timeout(self.timeout):
                    while _tag(identifier) in {
                        _tag(m.get("name", m.get("model")))
                        for m in await self._running()
                    }:
                        await asyncio.sleep(
                            getattr(self.config, "guard_min_interval", 0.1)
                        )
            except TimeoutError as exc:
                raise TimeoutError(
                    f"Ollama did not release model {identifier}"
                ) from exc

    async def unload(self, version):
        if version["identifier"] in getattr(self, "owned", {}):
            if await self.same_weights(self.owned[version["identifier"]]):
                await self._warm(version["identifier"], 0)
            self.owned.pop(version["identifier"], None)

    async def prune_version(self, version):
        # Imported full GGUF blobs are another copy on the library volume.
        # Reuse the ownership/digest checks from uninstall, for this tag only.
        if self.store:
            from .cleanup import purge_imports

            await purge_imports(self, self.store, {version["identifier"]})

    async def ensure(self, version):
        await self.load(version)

    async def park(self, on_snapshot=None):
        parked = [v | {"songe_requested": True} for v in await live_claims(self)]
        if on_snapshot:
            on_snapshot(parked)
        for item in parked:
            await self.unload(item)
        return parked

    async def restore(self, parked, managed):
        for item in parked:
            if item.get("songe_requested") and item["identifier"] not in managed:
                await self.load(item)

    async def loaded_bytes(self):
        await live_claims(self)
        owned = {_tag(k) for k in getattr(self, "owned", {})}
        return sum(
            int(m.get("size_vram", 0))
            for m in await self._running()
            if _tag(m.get("name", m.get("model"))) in owned
        )

    async def close(self):
        await close_owned(self)
