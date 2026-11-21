import asyncio
import json
import shutil
from pathlib import Path
from urllib.parse import urlparse

import httpx

from .protection import admit_load, close_owned, guarded_load


def downloads_folder(home=None):
    from ..models.discovery import lmstudio_downloads

    return lmstudio_downloads(home)


def local_models(folder):
    from ..models.discovery import lmstudio

    return [ref.path for ref in lmstudio(folder=folder) if ref.format != "gguf"]


SERVING_IDLE_TTL_SECONDS = 3600


def find_lms():
    for p in (
        shutil.which("lms"),
        str(Path.home() / "local-llm/bin/lms"),
        str(Path.home() / ".lmstudio/bin/lms"),
    ):
        if p and Path(p).exists():
            return p
    return ""


class LMStudio:
    name = "lmstudio"
    serves = ("mlx-fused", "gguf-full")

    def __init__(self, config, *, store=None):
        self.config, self.store = config, store
        self.recovered = False
        self.loaded = set()
        self.routes = {}
        self.reused = {}
        # LM Studio's TTL is an idle timeout, not a load deadline: a few
        # seconds unloaded the served model between requests. Match LM
        # Studio's own JIT default, which still bounds an orphaned load if
        # this process dies before journaling it. Reused instances get none.
        self.lease_seconds = SERVING_IDLE_TTL_SECONDS
        from ..platform import current

        kinds = {a.kind for a in current().accelerators()}
        self.serves = (
            ("mlx-fused", "gguf-full") if "metal" in kinds else ("gguf-full",)
        )
        self.url = getattr(config, "upstream_url", "http://127.0.0.1:1234/v1")

    def discover(self):
        from ..models.discovery import lmstudio, lmstudio_downloads, loaded_lmstudio

        folder = self.config.models_dir or lmstudio_downloads()
        return loaded_lmstudio(lmstudio(folder=folder), self.config.lms, folder)

    async def ensure(self, version):
        async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
            try:
                response = await client.get(self.config.upstream_url + "/models")
            except httpx.ConnectError:
                await self.command("server", "start")
                response = await client.get(self.config.upstream_url + "/models")
            response.raise_for_status()
            # API membership alone says nothing about the underlying weights.
            await self.load(version)

    async def command(self, *args):
        proc = await asyncio.create_subprocess_exec(
            self.config.lms,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=180)
        except BaseException:
            proc.kill()
            await proc.wait()
            raise
        if proc.returncode:
            detail = " ".join((err or out).decode(errors="replace").split())[-300:]
            raise RuntimeError(
                f"lms {args[0]} failed ({proc.returncode}): {detail}; "
                "model pointer was preserved"
            )
        return out.decode()

    async def model_key(self, relative, timeout=30.0, interval=1.0):
        """LM Studio's key for a model folder. `lms load` takes keys, not paths
        (verified with lms 0.4.28), and a newly written folder appears in the
        index within a few seconds, keyed by its folder name."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            listing = json.loads(await self.command("ls", "--json") or "[]")
            for model in listing:
                if Path(model.get("path", "")) == relative and model.get("modelKey"):
                    return model["modelKey"]
            if loop.time() >= deadline:
                raise RuntimeError(
                    f"LM Studio has not indexed {relative} after {timeout:.0f} s; "
                    "model pointer was preserved"
                )
            await asyncio.sleep(interval)

    @guarded_load
    async def load(self, version):
        artifact = Path(version["artifact"]).resolve()
        folder = Path(self.config.models_dir).resolve()
        try:
            relative = artifact.relative_to(folder)
        except ValueError as exc:
            raise ValueError(
                "LM Studio artifacts must be inside downloadsFolder"
            ) from exc
        identifier = version["identifier"]
        if await self.resident(version):
            return
        running = await self.running()
        if identifier in running:
            # LM Studio refuses a second load under the same identifier.
            # Reuse it only if it is the same weights; never serve a stale one.
            if Path(running[identifier]) != relative:
                raise RuntimeError(
                    f"LM Studio already serves {identifier} from "
                    f"{running[identifier]}, not {relative}; model pointer was preserved"
                )
            return
        if identifier in self.owned:
            if not self.owned[identifier].get("pending"):
                raise RuntimeError(
                    "LM Studio has an unresolved load; recovery records preserved"
                )
            # LM Studio lists loading and loaded instances alike; a pending
            # claim it does not list never completed (e.g. a failed install),
            # so it must not block serving forever.
            self.forget(identifier)
        key = await self.model_key(relative)
        if await self.resident(version):
            return
        await admit_load(self, version)
        identifier = version["identifier"]
        self.owned[identifier] = dict(version) | {
            "pending": True,
            "lease_seconds": self.lease_seconds,
        }
        self.save_claims()
        # The SDK returns the immutable instance from the load operation itself.
        # Never adopt an alias discovered later in a CLI listing.
        await self.request_load(key, identifier, self.remembered_context(version))
        self.routes[identifier] = identifier
        self.loaded.add(identifier)
        await self.live_owned()

    def save_claims(self):
        if self.store:
            self.store.set(
                "lmstudio_claims",
                {
                    "installation_id": self.store.get("installation_id"),
                    "url": self.url,
                    "instances": getattr(self, "owned", {}),
                },
            )

    def forget(self, identifier):
        self.owned.pop(identifier, None)
        self.loaded.discard(identifier)
        self.save_claims()

    def completed(self, identifier, reference):
        if not reference:
            raise RuntimeError(
                "LM Studio omitted immutable instance identity; claim retained"
            )
        self.owned[identifier].update(runtime_instance=reference, pending=False)
        self.save_claims()

    async def recover(self):
        if self.recovered:
            return
        saved = self.store.get("lmstudio_claims", {}) if self.store else {}
        matches = (
            saved.get("installation_id")
            == (self.store.get("installation_id") if self.store else None)
            and saved.get("url") == self.url
        )
        if saved.get("instances") and not matches:
            raise RuntimeError(
                "LM Studio ownership journal belongs to another endpoint; preserved"
            )
        self.recovered = True
        if matches:
            self.owned.update(
                {
                    k: v | {"reclaim": True}
                    for k, v in saved.get("instances", {}).items()
                }
            )
            self.loaded.update(
                k for k, v in self.owned.items() if v.get("runtime_instance")
            )

    def sdk(self):
        from lmstudio import AsyncClient

        return AsyncClient(api_host=urlparse(self.url).netloc)

    def remembered_context(self, version):
        """The context this model last ran with in LM Studio, by its instance,
        its folder, or (for a learned version) its original model's folder."""
        seen = (self.store.get("served_context_ids", {}) if self.store else {}) or {}
        folder = Path(self.config.models_dir).resolve()
        keys = [version.get("identifier")]
        for path in (version.get("artifact"), version.get("base")):
            if path:
                try:
                    keys.append(Path(path).resolve().relative_to(folder).as_posix())
                except ValueError:
                    keys.append(str(path))
        return next((int(seen[k]) for k in keys if k in seen), None)

    async def request_load(self, key, identifier, context=None):
        from lmstudio import LMStudioServerError
        from lmstudio.json_api import LoadModelEndpoint, ModelTypesLlm

        async with self.sdk() as client:
            types = ModelTypesLlm
            endpoint = LoadModelEndpoint(
                key,
                identifier,
                self.lease_seconds,
                types.REQUEST_NEW_INSTANCE,
                types.MODEL_LOAD_CONFIG,
                # LM Studio may not keep a per-model setting; Songe re-applies
                # the context the user last ran this model with.
                {"contextLength": context} if context else None,
                None,
            )
            # Keep consuming this channel after cancellation until the server
            # confirms failure or returns the exact instance that must be freed.
            async with client.llm._create_channel(endpoint) as channel:
                result = asyncio.create_task(channel.wait_for_result())
                try:
                    info = await asyncio.shield(result)
                except asyncio.CancelledError:
                    self.owned[identifier]["reclaim"] = True
                    self.save_claims()
                    await channel.cancel()
                    try:
                        info = await asyncio.shield(result)
                    except LMStudioServerError:
                        self.forget(identifier)  # Server confirmed terminal failure.
                    else:
                        self.completed(identifier, info.instance_reference)
                    raise
                except LMStudioServerError:
                    self.forget(identifier)
                    raise
                else:
                    self.completed(identifier, info.instance_reference)
                finally:
                    if not result.done():
                        result.cancel()
                    await asyncio.gather(result, return_exceptions=True)

    @staticmethod
    def instance(model):
        return model.get("instanceReference")

    async def resident(self, version):
        artifact = Path(version["artifact"]).resolve()
        try:
            artifact.relative_to(Path(self.config.models_dir).resolve())
        except ValueError as exc:
            raise ValueError(
                "LM Studio artifacts must be inside downloadsFolder"
            ) from exc
        live = await self.live_owned()
        identifier = version["identifier"]
        self.routes.pop(identifier, None)
        if (
            identifier in live
            and (
                Path(self.config.models_dir) / live[identifier].get("path", "")
            ).resolve()
            != artifact
        ):
            raise RuntimeError(
                f"LM Studio already serves {version['identifier']} from other weights; model pointer was preserved"
            )
        for model in sorted(live.values(), key=lambda m: m["identifier"] != identifier):
            path = Path(model.get("path", ""))
            if path.is_absolute():
                matches = path.resolve() == artifact
            else:
                matches = (Path(self.config.models_dir) / path).resolve() == artifact
            if matches:
                self.routes[identifier] = model["identifier"]
                if model["identifier"] not in getattr(
                    self, "owned", {}
                ) and self.instance(model):
                    from ..models.convert import source_key

                    self.reused[identifier] = {
                        "identifier": model["identifier"],
                        "artifact": str(artifact),
                        "runtime_instance": self.instance(model),
                        "weights_fingerprint": source_key(artifact),
                    }
                return True
        return False

    async def replacement_ticket(self, version):
        # Only instances actually reused by Songe qualify. A new user load is
        # never adopted merely because it happens to use the same base.
        return self.reused.get(version["identifier"])

    async def release_replaced(self, ticket):
        live = await self.listing()
        model = next(
            (m for m in live if m.get("identifier") == ticket["identifier"]), None
        )
        from ..models.convert import source_key

        if (
            model
            and self.matches_owned(model, ticket)
            and source_key(ticket["artifact"]) == ticket["weights_fingerprint"]
        ):
            async with self.sdk() as client:
                await client.llm.unload(ticket["identifier"])
        self.reused = {k: v for k, v in self.reused.items() if v != ticket}

    def request_model(self, version):
        return self.routes.get(version["identifier"], version["identifier"])

    def matches_owned(self, model, version):
        path = Path(self.config.models_dir) / model.get("path", "")
        reference = version.get("runtime_instance")
        return (
            bool(reference)
            and path.resolve() == Path(version["artifact"]).resolve()
            and reference == self.instance(model)
        )

    async def live_owned(self):
        live = {m["identifier"]: m for m in await self.listing() if "identifier" in m}
        for identifier, version in list(getattr(self, "owned", {}).items()):
            # Absence while a load is pending is not proof of cancellation.
            if version.get("pending") or identifier not in live:
                if identifier not in live and version.get("seen"):
                    self.forget(identifier)
                continue
            if not self.matches_owned(live[identifier], version):
                self.forget(identifier)
            elif not version.get("seen"):
                version["seen"] = True
                self.save_claims()
        return live

    async def served_contexts(self):
        """Context each loaded model actually runs with, by identifier and path.

        `user_contexts` keeps only the instances the user loaded: Songe's own
        loads carry whatever Songe chose (often LM Studio's default) and must
        not overwrite the setting the user picked."""
        result, chosen = {}, {}
        owned = getattr(self, "owned", {})
        ours = {v.get("runtime_instance") for v in owned.values()} - {None}
        loading = {k for k, v in owned.items() if v.get("pending")}
        for item in await self.listing():
            context = item.get("contextLength")
            if not context:
                continue
            mine = self.instance(item) in ours or item.get("identifier") in loading
            for key in (item.get("identifier"), item.get("path")):
                if key:
                    result[key] = int(context)
                    if not mine:
                        chosen[key] = int(context)
        self.user_contexts = chosen
        return result

    async def listing(self):
        async with self.sdk() as client:
            return await client.llm.remote_call("listLoaded")

    async def park(self, on_snapshot=None):
        await self.live_owned()
        parked = [
            v | {"songe_requested": True} for v in getattr(self, "owned", {}).values()
        ]
        if on_snapshot:
            on_snapshot(parked)
        for version in parked:
            await self.unload(version)
        return parked

    async def restore(self, parked, managed):
        for version in parked:
            if version.get("songe_requested") and version["identifier"] not in managed:
                await self.load(version)

    async def loaded_bytes(self):
        total = 0
        for model in (await self.live_owned()).values():
            if model.get("identifier") not in self.loaded:
                continue
            # Prefer the runtime allocation when present; file size is only a
            # fallback estimate of reclaimable weights, never a fit exclusion.
            size = model.get(
                "memoryBytes",
                model.get(
                    "loadedBytes",
                    model.get("memoryUsageBytes", model.get("sizeBytes", 0)),
                ),
            )
            if isinstance(size, (int, float)):
                total += max(0, int(size))
        return total

    async def running(self):
        """Loaded identifiers mapped to their folder, from the runtime SDK."""
        listing = await self.listing()
        return {
            m["identifier"]: m.get("path", "") for m in listing if "identifier" in m
        }

    async def unload(self, version):
        identifier = version["identifier"]
        owned = getattr(self, "owned", {}).get(identifier)
        if not owned:
            return
        owned["reclaim"] = True
        self.save_claims()
        if owned.get("pending"):
            return
        live = await self.live_owned()
        if identifier not in self.owned or identifier not in live:
            return
        if self.matches_owned(live[identifier], owned):
            async with self.sdk() as client:
                await client.llm.unload(identifier)
            self.forget(identifier)

    async def close(self):
        await close_owned(self)
