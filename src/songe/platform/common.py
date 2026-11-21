import shutil
import subprocess

from . import Accel


def read(*args):
    return subprocess.check_output(args, timeout=3, stderr=subprocess.DEVNULL).decode()


class ChildHandle:
    def __init__(self, process):
        self.process = process

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


_NVML = {"ready": None}


def nvml_accelerators():
    """NVIDIA memory through NVML: in-process, milliseconds, unaffected by a
    busy GPU. Returns None when the NVML library is not available."""
    try:
        import pynvml
    except ImportError:
        return None
    try:
        if not _NVML["ready"]:
            pynvml.nvmlInit()
            _NVML["ready"] = True
        devices = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            identity = pynvml.nvmlDeviceGetUUID(handle)
            devices.append(
                Accel(
                    "cuda",
                    name.decode() if isinstance(name, bytes) else str(name),
                    int(memory.total),
                    int(memory.free),
                    identity.decode() if isinstance(identity, bytes) else str(identity),
                )
            )
        return devices
    except pynvml.NVMLError:
        _NVML["ready"] = False
        return None


def nvidia_smi_accelerators():
    command = (
        "nvidia-smi",
        "--query-gpu=uuid,name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    )
    # A busy GPU can make nvidia-smi slow; allow time and one retry, but never
    # reuse a stale measurement.
    for attempt in range(2):
        try:
            rows = subprocess.check_output(
                command, timeout=15, stderr=subprocess.DEVNULL
            ).decode()
            break
        except subprocess.TimeoutExpired:
            if attempt:
                raise
    if not rows.strip():
        raise ValueError("Empty NVIDIA memory measurement")
    devices = []
    for row in rows.splitlines():
        identity, name, total, free = row.split(",", 3)
        devices.append(
            Accel(
                "cuda",
                name.strip(),
                int(total) * 1024**2,
                int(free) * 1024**2,
                identity.strip(),
            )
        )
    return devices


def discrete_accelerators():
    """Query drivers without importing a training framework or allocating VRAM."""
    devices = []
    nvml = nvml_accelerators()
    if nvml:
        devices += nvml
    elif shutil.which("nvidia-smi"):
        try:
            devices += nvidia_smi_accelerators()
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            raise OSError("NVIDIA memory telemetry unavailable") from exc
    if shutil.which("rocm-smi"):
        import json

        try:
            rows = json.loads(read("rocm-smi", "--showmeminfo", "vram", "--json"))
            if not rows:
                raise ValueError("Empty AMD memory measurement")
            for name, row in rows.items():
                total = int(row["VRAM Total Memory (B)"])
                devices.append(
                    Accel(
                        "rocm",
                        name,
                        total,
                        total - int(row["VRAM Total Used Memory (B)"]),
                        name,
                    )
                )
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
            raise OSError("AMD memory telemetry unavailable") from exc
    return devices


_GPU = {"at": 0.0, "value": None}


def gpu_usage(max_age=3.0):
    """How busy the GPU is and how much memory it holds, for the dashboard:
    {"busy": percent, "memory": bytes} or None when unknown. macOS reads the
    IOAccelerator's own figures (no admin rights); NVIDIA asks nvidia-smi.
    Cached for a few seconds: the page asks every second or two."""
    import re
    import sys
    import time

    now = time.monotonic()
    if now - _GPU["at"] < max_age:
        return _GPU["value"]
    value = None
    try:
        if sys.platform == "darwin":
            text = subprocess.check_output(
                ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
                timeout=2, stderr=subprocess.DEVNULL,
            ).decode(errors="replace")
            busy = re.search(r'"Device Utilization %"=(\d+)', text)
            used = re.search(r'"In use system memory"=(\d+)', text)
            if busy:
                value = {"busy": int(busy.group(1)), "memory": int(used.group(1)) if used else None}
        elif shutil.which("nvidia-smi"):
            row = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                timeout=3, stderr=subprocess.DEVNULL,
            ).decode().splitlines()[0]
            busy, used = (int(x.strip()) for x in row.split(",")[:2])
            value = {"busy": busy, "memory": used * 1024**2}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        value = None
    _GPU.update(at=now, value=value)
    return value
