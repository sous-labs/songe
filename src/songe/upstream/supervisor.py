"""Own a loopback llama-server and register immutable adapter slots on restart."""

import asyncio
import contextlib
import os
import shutil
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

from ..memory import (
    guard,
    kill,
    owned_command,
    process_options,
    process_tree_bytes,
    snapshot,
    terminate,
)
from ..platform import current
from .protection import load_seconds

RECENT_NIGHTS = 3


def provision_endpoint(config):
    """Choose a private loopback endpoint; never adopt a discovered process."""
    import socket

    if config.server_owned:
        return
    if not config.server_command and config.upstream != "mlx":
        raise ValueError(
            "A local server invocation is required for a separate Songe endpoint"
        )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    # A racing listener is checked again at launch and is never stopped.
    config.server_source_url = config.upstream_url
    config.upstream_url = f"http://127.0.0.1:{port}/v1"
    config.server_owned = True


def listening_server(config):
    """Identify the exact local server process, never a process by name alone."""
    import psutil

    port = urlparse(config.upstream_url).port
    if urlparse(config.upstream_url).hostname not in {"localhost", "127.0.0.1", "::1"}:
        return None
    for process in psutil.process_iter(["pid", "exe", "cmdline"]):
        try:
            command = process.info["cmdline"] or []
            allowed = bool(command) and (
                Path(command[0]).name.lower() in {"llama-server", "llama-server.exe"}
                if config.upstream == "llamacpp"
                else ("mlx_lm" in command or "mlx_lm.server" in command)
                if config.upstream == "mlx"
                else Path(command[0]).name.lower() in {"vllm", "vllm.exe"}
                or any(
                    token
                    in {
                        "vllm.entrypoints.openai.api_server",
                        "vllm.entrypoints.cli.main",
                    }
                    or Path(token).name.lower() in {"vllm", "vllm.exe"}
                    for token in command[1:]
                )
            )
            if not allowed:
                continue
            if any(
                c.status == psutil.CONN_LISTEN
                and c.laddr.port == port
                and c.laddr.ip in {"127.0.0.1", "::1", "0.0.0.0", "::"}
                for c in process.net_connections("tcp")
            ):
                return process
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return None


def owns_endpoint(config, process):
    """Readiness requires a listener inside our child tree, not just HTTP 200."""
    import psutil

    try:
        tree = [psutil.Process(process.pid)]
        tree += tree[0].children(recursive=True)
        port = urlparse(config.upstream_url).port
        return any(
            c.status == psutil.CONN_LISTEN and c.laddr.port == port
            for child in tree
            for c in child.net_connections("tcp")
        )
    except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
        return False


def device_free(platform):
    """Free memory per accelerator ({identifier: bytes}), or None when this
    machine reports none (unified memory: host RAM is the device's)."""
    try:
        devices = snapshot(platform).vram
    except (OSError, ValueError):
        return None
    return {a.identifier or a.name: a.memory_free for a in devices} or None


class LlamaServer:
    def __init__(self, config, root):
        self.config, self.root = config, root
        self.process, self.watcher = None, None
        self.finished = threading.Event()
        self.platform = current()
        # Accelerator memory the running server took when it started (free
        # before minus free once ready): what stopping it gives back. Its
        # process's RSS is host memory only; a vLLM holding 72 GB of an
        # 80 GB GPU has an RSS of a few GB (live, Qwen3.8-27B on an A100: a
        # pressed Deep Dream waited for memory, then stopped, never training).
        self.device_held = None
        self._free_before_start = None

    def memory_bytes(self):
        pid = self.process.pid if self.process else None
        return (
            process_tree_bytes(self.platform, pid)
            if pid and hasattr(self.platform, "process_bytes")
            else 0
        )

    def device_bytes(self):
        """{accelerator: bytes} the running server holds, or None if unknown."""
        if self.process is None or self.process.returncode is not None:
            return None
        return self.device_held

    def _before_start(self):
        self._free_before_start = device_free(self.platform)

    def _ready(self):
        before, after = self._free_before_start, device_free(self.platform)
        self.device_held = (
            {k: max(0, before[k] - after.get(k, before[k])) for k in before}
            if before and after
            else None
        )

    async def stop(self):
        if self.process is None:
            return
        if self.process.returncode is None:
            terminate(self.process)
            try:
                await asyncio.wait_for(
                    self.process.wait(), self.config.guard_grace_seconds
                )
            except TimeoutError:
                kill(self.process)
                await self.process.wait()
        self.finished.set()
        if self.watcher:
            await self.watcher
        self.process, self.watcher = None, None
        self.device_held = None

    async def restart(self, version):
        await self.stop()
        from .protection import admit, weight_bytes

        baseline = await asyncio.to_thread(
            admit,
            self.platform,
            weight_bytes(version.get("base") or version["artifact"]),
        )
        if await asyncio.to_thread(listening_server, self.config):
            raise RuntimeError(
                "The upstream port belongs to a user process. Stop it or select a separate endpoint before Songe starts its own server"
            )
        command = self.config.server_command or [
            shutil.which("llama-server") or "llama-server"
        ]
        # Retain the user's runtime arguments, replacing the binding and LoRA
        # slots that this supervisor owns. No shell or arbitrary code executes.
        argv, index = [command[0]], 1
        options = {
            "-m",
            "--model",
            "--alias",
            "-a",
            "--host",
            "--port",
            "--lora",
            "--lora-scaled",
        }
        while index < len(command):
            part = command[index]
            if part in options:
                index += 2
            elif part == "--lora-init-without-apply":
                index += 1
            else:
                argv.append(part)
                index += 1
        argv += [
            "--model",
            str(version.get("base") or version["artifact"]),
            "--alias",
            self.config.base_identifier,
            "--host",
            "127.0.0.1",
            "--port",
            str(urlparse(self.config.upstream_url).port or 8080),
        ]
        paths = {str(Path(a["path"]).resolve()) for a in version.get("adapters", [])}
        from ..artifacts.storage import artifact_roots

        # Recent nights stay registered so serving, candidate and undo switch
        # by scale; an older adapter is registered by restarting on demand,
        # so memory and startup do not grow with every accepted night.
        def night(path):
            try:
                return int(path.parent.name.removeprefix("night-"))
            except ValueError:
                return -1

        recent = sorted(
            (
                p
                for directory in artifact_roots(self.root)
                for p in (directory / "versions").glob("night-*/scale-*-artifact.gguf")
            ),
            key=night,
            reverse=True,
        )
        nights = sorted({night(p) for p in recent}, reverse=True)[:RECENT_NIGHTS]
        paths.update(str(p.resolve()) for p in recent if night(p) in nights)
        if version.get("kind") == "gguf-lora":
            paths.add(str(Path(version["artifact"]).resolve()))
        for path in sorted(paths):
            argv += ["--lora", path]
        if paths:
            argv += ["--lora-init-without-apply"]
        await asyncio.to_thread(self._before_start)
        self.process = await asyncio.create_subprocess_exec(
            *owned_command(argv),
            **process_options(),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self.finished = threading.Event()
        self.watcher = asyncio.create_task(
            asyncio.to_thread(
                guard,
                self.process,
                self.platform,
                stop=self.finished,
                baseline=baseline,
            )
        )
        deadline = time.monotonic() + load_seconds(self.config)
        async with httpx.AsyncClient(trust_env=False, timeout=2) as client:
            while time.monotonic() < deadline and self.process.returncode is None:
                try:
                    response = await client.get(
                        self.config.upstream_url.removesuffix("/v1") + "/health"
                    )
                    if response.is_success:
                        await asyncio.to_thread(self._ready)
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(self.config.guard_min_interval)
        await self.stop()
        raise RuntimeError(
            "llama.cpp did not become ready; serving pointer was preserved"
        )

    async def close(self):
        with contextlib.suppress(ProcessLookupError):
            await self.stop()


class VLLMServer(LlamaServer):
    """Enable local LoRA updates on Songe's own vLLM server."""

    async def restart(self, version):
        await self.stop()
        from .protection import admit, weight_bytes

        baseline = await asyncio.to_thread(
            admit, self.platform, weight_bytes(self.config.model)
        )
        if await asyncio.to_thread(listening_server, self.config):
            raise RuntimeError(
                "The upstream port belongs to a user process; Songe will wait for a separate endpoint"
            )
        argv = list(self.config.server_command)
        if not argv:
            raise RuntimeError(
                "No local vLLM server invocation was discovered; rerun songe init"
            )
        # Preserve the user's base and runtime controls; force control endpoints
        # onto loopback before enabling the runtime LoRA management API.
        for option, value in (
            ("--host", "127.0.0.1"),
            ("--port", str(urlparse(self.config.upstream_url).port or 8000)),
        ):
            if option in argv:
                argv[argv.index(option) + 1] = value
            else:
                argv += [option, value]
        if "--enable-lora" not in argv:
            argv.append("--enable-lora")
        env = os.environ | {"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"}
        await asyncio.to_thread(self._before_start)
        self.process = await asyncio.create_subprocess_exec(
            *owned_command(argv),
            env=env,
            **process_options(),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self.finished = threading.Event()
        self.watcher = asyncio.create_task(
            asyncio.to_thread(
                guard,
                self.process,
                self.platform,
                stop=self.finished,
                baseline=baseline,
            )
        )
        deadline = time.monotonic() + load_seconds(self.config)
        async with httpx.AsyncClient(trust_env=False, timeout=2) as client:
            while time.monotonic() < deadline and self.process.returncode is None:
                try:
                    response = await client.get(self.config.upstream_url + "/models")
                    if response.is_success:
                        await asyncio.to_thread(self._ready)
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(self.config.guard_min_interval)
        await self.stop()
        raise RuntimeError("vLLM did not become ready; serving pointer was preserved")
