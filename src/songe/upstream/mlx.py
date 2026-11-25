import asyncio
import sys
import threading
import time
from urllib.parse import urlparse

import httpx

from .protection import load_seconds


class MLX:
    """Owned subprocess only. A watchdog restarts it after unexpected death."""

    name = "mlx"
    serves = ("mlx-adapter",)

    def __init__(self, config, root):
        self.config, self.root = config, root
        self.process = None
        self.stopping = False
        self.watchdog = None
        self.version = None
        self.url = config.upstream_url
        self.guard_task = None
        self.finished = threading.Event()
        self.routes = {}

    def discover(self):
        from ..models.discovery import mlx

        return mlx(self.url)

    async def loaded_bytes(self):
        from ..memory import process_tree_bytes
        from ..platform import current

        platform = current()
        return (
            process_tree_bytes(platform, self.process.pid)
            if self.process and hasattr(platform, "process_bytes")
            else 0
        )

    async def park(self, on_snapshot=None):
        parked = [self.version] if self.version else []
        if on_snapshot:
            on_snapshot(parked)
        await self.close()
        self.process, self.watchdog = None, None
        return parked

    async def restore(self, parked, managed):
        for version in parked:
            if version["identifier"] not in managed:
                await self.load(version)

    async def ensure(self, version):
        await self.load(version)

    async def start(self, model=None):
        from ..memory import guard, owned_command, process_options
        from ..platform import current
        from .protection import admit, weight_bytes
        from .supervisor import provision_endpoint

        provision_endpoint(self.config)
        self.url = self.config.upstream_url
        platform = current()
        baseline = await asyncio.to_thread(
            admit, platform, weight_bytes(self.config.model)
        )
        self.stopping = False
        self.finished = threading.Event()
        port = urlparse(self.config.upstream_url).port or 1235
        command = self.config.server_command or [
            sys.executable, "-m", "mlx_lm", "server"
        ]
        self.process = await asyncio.create_subprocess_exec(
            *owned_command(
                [
                    *command,
                    "--model",
                    model or self.config.model,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ]
            ),
            **process_options(),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

        self.guard_task = asyncio.create_task(
            asyncio.to_thread(
                guard, self.process, platform, stop=self.finished, baseline=baseline
            )
        )

    async def supervise(self):
        from ..memory import MemoryPressure

        while not self.stopping:
            await self.process.wait()
            if not self.stopping:
                self.finished.set()
                if self.guard_task:
                    await self.guard_task
                while not self.stopping:
                    await asyncio.sleep(self.config.guard_max_interval)
                    try:
                        await self.start(
                            self.version["artifact"] if self.version else None
                        )
                        break
                    except (MemoryPressure, OSError):
                        continue

    async def load(self, version):
        from ..platform import current
        from .protection import admit, weight_bytes
        from .reuse import served_base

        self.routes.pop(version["identifier"], None)
        if getattr(self.config, "server_source_url", "") or self.process is None:
            binding = await served_base(
                self.config, version, transport=getattr(self, "transport", None)
            )
            if binding:
                self.routes[version["identifier"]] = binding
                return
        if self.version == version and self.process and self.process.returncode is None:
            return

        await asyncio.to_thread(admit, current(), weight_bytes(version["artifact"]))
        # mlx-lm caches model/adapter pairs. Release our previous cache before
        # warming another binding, rather than retaining whole model copies.
        if self.version and self.version != version:
            await self.close()
            self.process, self.watchdog = None, None
        if self.process is None:
            await self.start(version["artifact"])
        from .supervisor import owns_endpoint

        deadline = time.monotonic() + load_seconds(self.config)
        # Warm the explicit (base path, unique adapter path) cache key before
        # committing the pointer. Even the unadapted request sends adapters=null.
        async with httpx.AsyncClient(trust_env=False, timeout=120) as client:
            while time.monotonic() < deadline:
                if self.process.returncode is not None:
                    await self.close()
                    raise RuntimeError(
                        "MLX child exited before readiness; serving pointer was preserved"
                    )
                if not await asyncio.to_thread(
                    owns_endpoint, self.config, self.process
                ):
                    await asyncio.sleep(self.config.guard_min_interval)
                    continue
                try:
                    response = await client.post(
                        self.config.upstream_url + "/chat/completions",
                        json={
                            "model": version["artifact"],
                            "adapters": version.get("adapter"),
                            "messages": [{"role": "user", "content": "Hello"}],
                            "max_tokens": 1,
                        },
                    )
                    response.raise_for_status()
                    if self.process.returncode is not None:
                        raise RuntimeError("MLX child exited during warmup")
                    self.version = dict(version)
                    if self.watchdog is None:
                        self.watchdog = asyncio.create_task(self.supervise())
                    return
                except httpx.ConnectError:
                    await asyncio.sleep(0.5)
            await self.close()
            raise RuntimeError("MLX server did not become ready")

    def request_model(self, version):
        return self.routes.get(version["identifier"], {}).get(
            "model", version["artifact"]
        )

    def request_url(self, version):
        return self.routes.get(version["identifier"], {}).get("url", self.url)

    async def unload(self, version):
        if version["identifier"] in self.routes:
            return
        if self.version and self.version["identifier"] == version["identifier"]:
            await self.close()
            self.process, self.watchdog, self.version = None, None, None

    async def close(self):
        from ..memory import kill, terminate

        self.stopping = True
        if self.watchdog:
            self.watchdog.cancel()
            await asyncio.gather(self.watchdog, return_exceptions=True)
        if self.process and self.process.returncode is None:
            terminate(self.process)
            try:
                await asyncio.wait_for(self.process.wait(), 5)
            except TimeoutError:
                kill(self.process)
                await self.process.wait()

        self.finished.set()
        if self.guard_task:
            await self.guard_task
            self.guard_task = None
