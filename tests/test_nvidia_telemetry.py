"""NVIDIA memory telemetry: NVML first, nvidia-smi as a patient fallback."""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from songe.platform import common


def fake_nvml(free=8 * 2**30, fail=False, memory_error=None):
    class NVMLError(Exception):
        def __init__(self, value):
            super().__init__(value)
            self.value = value

    def init():
        if fail:
            raise NVMLError("driver not loaded")

    def memory_info(handle):
        if memory_error is not None:
            raise NVMLError(memory_error)
        return SimpleNamespace(total=24 * 2**30, free=free)

    return SimpleNamespace(
        NVMLError=NVMLError,
        NVML_ERROR_NOT_SUPPORTED=3,
        nvmlInit=init,
        nvmlDeviceGetCount=lambda: 1,
        nvmlDeviceGetHandleByIndex=lambda i: i,
        nvmlDeviceGetMemoryInfo=memory_info,
        nvmlDeviceGetName=lambda h: b"NVIDIA GeForce RTX 4090",
        nvmlDeviceGetUUID=lambda h: "GPU-1",
    )


MEMINFO = "MemTotal:       125000000 kB\nMemFree:  9000000 kB\nMemAvailable:   80000000 kB\n"


@pytest.fixture
def meminfo(tmp_path, monkeypatch):
    path = tmp_path / "meminfo"
    path.write_text(MEMINFO, encoding="utf-8")
    monkeypatch.setattr(common.unified_memory, "__defaults__", (path,))
    return path


@pytest.fixture(autouse=True)
def fresh_nvml(monkeypatch):
    monkeypatch.setitem(common._NVML, "ready", None)
    monkeypatch.setattr(common.shutil, "which", lambda name: None)


def test_nvml_reads_memory_without_a_subprocess(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml())
    monkeypatch.setattr(
        common.subprocess, "check_output", lambda *a, **k: pytest.fail("no subprocess")
    )
    [gpu] = common.discrete_accelerators()
    assert (gpu.kind, gpu.name, gpu.memory_free) == (
        "cuda",
        "NVIDIA GeForce RTX 4090",
        8 * 2**30,
    )


def test_falls_back_to_nvidia_smi_and_retries_one_timeout(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml(fail=True))
    monkeypatch.setattr(common.shutil, "which", lambda name: name == "nvidia-smi")
    calls = []

    def slow_then_ok(command, timeout, stderr):
        calls.append(timeout)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(command, timeout)
        return b"GPU-1, NVIDIA A100, 81920, 40960\n"

    monkeypatch.setattr(common.subprocess, "check_output", slow_then_ok)
    [gpu] = common.discrete_accelerators()
    assert gpu.memory_free == 40960 * 1024**2 and calls == [15, 15]


def test_unavailable_telemetry_still_fails_closed(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml(fail=True))
    monkeypatch.setattr(common.shutil, "which", lambda name: name == "nvidia-smi")

    def always_slow(command, timeout, stderr):
        raise subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(common.subprocess, "check_output", always_slow)
    with pytest.raises(OSError, match="NVIDIA memory telemetry unavailable"):
        common.discrete_accelerators()


def test_integrated_gpu_uses_system_memory_through_nvml(monkeypatch, meminfo):
    # GB10 / DGX Spark: NVML answers "not supported" for device memory.
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml(memory_error=3))
    [gpu] = common.discrete_accelerators()
    assert (gpu.kind, gpu.memory_total, gpu.memory_free) == (
        "cuda",
        125000000 * 1024,
        80000000 * 1024,
    )


def test_integrated_gpu_uses_system_memory_through_nvidia_smi(monkeypatch, meminfo):
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml(fail=True))
    monkeypatch.setattr(common.shutil, "which", lambda name: name == "nvidia-smi")
    monkeypatch.setattr(
        common.subprocess,
        "check_output",
        lambda *a, **k: b"GPU-1f86, NVIDIA GB10, [N/A], [N/A]\n",
    )
    [gpu] = common.discrete_accelerators()
    assert (gpu.name, gpu.memory_free) == ("NVIDIA GB10", 80000000 * 1024)


def test_other_nvml_memory_errors_still_fail_closed(monkeypatch, meminfo):
    # Only "not supported" means unified memory; any other error is unknown.
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml(memory_error=999))
    monkeypatch.setattr(common.shutil, "which", lambda name: name == "nvidia-smi")
    monkeypatch.setattr(
        common.subprocess,
        "check_output",
        lambda *a, **k: b"GPU-1, NVIDIA A100, [N/A], 40960\n",
    )
    with pytest.raises(OSError, match="NVIDIA memory telemetry unavailable"):
        common.discrete_accelerators()


def test_integrated_gpu_without_meminfo_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(
        common.unified_memory, "__defaults__", (tmp_path / "missing",)
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml(memory_error=3))
    monkeypatch.setattr(common.shutil, "which", lambda name: name == "nvidia-smi")
    monkeypatch.setattr(
        common.subprocess,
        "check_output",
        lambda *a, **k: b"GPU-1, NVIDIA GB10, [N/A], [N/A]\n",
    )
    with pytest.raises(OSError, match="NVIDIA memory telemetry unavailable"):
        common.discrete_accelerators()


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS footprint")
def test_macos_process_bytes_counts_physical_footprint():
    import os

    from songe.platform.darwin import phys_footprint

    # Footprint includes Metal buffers that RSS omits; it must be readable.
    assert phys_footprint(os.getpid()) > 1_000_000
