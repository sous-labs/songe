import ctypes
import os
from pathlib import Path

from .. import NAME
from . import Accel, Memory, Power
from .common import discrete_accelerators


class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
        (name, ctypes.c_uint64)
        for name in (
            "total",
            "available",
            "page_total",
            "page_available",
            "virtual_total",
            "virtual_available",
            "extended",
        )
    ]


class PowerStatus(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_ubyte) for name in ("ac", "flags", "percent", "saver")
    ] + [("life", ctypes.c_uint32), ("full", ctypes.c_uint32)]


def pressure(load, available, *, warn_load=80, critical_load=95):
    """Windows percent memory load: default headroom 20%, emergency headroom 5%.

    Sensitivity controls are overridable; zero available bytes is critical.
    """
    if available <= 0 or load >= critical_load:
        return "critical"
    return "warn" if load >= warn_load else "normal"


# Room kept on the system drive for the page file. Windows grows a
# system-managed page file there as commit charge passes the commit limit;
# nothing bounds it but free space. Below these lines the system drive is
# about to fill (it reached 0.0 GB on a 31 GB laptop measuring a 1.7B model).
COMMIT_STOP_BYTES = 4e9
COMMIT_WARN_BYTES = 2e9


def commit_pressure(commit_available, system_free):
    """Pressure from commit charge: what can still be committed, counting the
    system drive's free space above COMMIT_STOP_BYTES as room the page file
    may grow into. Free physical memory alone misses this: CUDA and model
    servers commit far more than they touch.

    "warn" holds back starts that wait on warn (reading, review); it does
    not stop running work. "critical" makes the guard stop running work at
    once. The scheduler's own disk check (artifacts.storage.SWAP_STOP_BYTES,
    the same 4 GB line) is what stops work gracefully before that."""
    if system_free is None:
        return "normal"
    room = commit_available + max(0, system_free - COMMIT_STOP_BYTES)
    if room >= COMMIT_WARN_BYTES:
        return "normal"
    # Past the stop line already: no time for a graceful stop.
    return "critical" if system_free < COMMIT_STOP_BYTES else "warn"


LEVELS = ("normal", "warn", "critical")


def worse(a, b):
    return max(a, b, key=LEVELS.index)


class AwakeHandle:
    def __init__(self, kernel):
        self.kernel = kernel

    def close(self):
        self.kernel.SetThreadExecutionState(0x80000000)


class Windows:
    name = "windows"

    def __init__(self, home=None, environ=None, kernel=None, user=None):
        self.home = Path(home) if home is not None else Path.home()
        self.environ = os.environ if environ is None else environ
        self.kernel = kernel if kernel is not None else ctypes.windll.kernel32
        self.user = user if user is not None else ctypes.windll.user32

    def data_dir(self):
        return (
            Path(self.environ.get("LOCALAPPDATA", self.home / "AppData/Local")) / NAME
        )

    def swap_volume(self):
        """A system-managed page file grows on the system drive as memory
        runs short."""
        return Path(self.environ.get("SystemDrive", "C:") + "\\"), "the system drive"

    def idle_seconds(self):
        class LastInput(ctypes.Structure):
            _fields_ = [("size", ctypes.c_uint32), ("tick", ctypes.c_uint32)]

        info = LastInput()
        info.size = ctypes.sizeof(info)
        if not self.user.GetLastInputInfo(ctypes.byref(info)):
            return None
        return ((self.kernel.GetTickCount() - info.tick) & 0xFFFFFFFF) / 1000

    def on_ac_power(self):
        return self.power().ac

    def power(self):
        info = PowerStatus()
        if not self.kernel.GetSystemPowerStatus(ctypes.byref(info)):
            return Power()
        ac = bool(info.ac) if info.ac in (0, 1) else None
        # UNKNOWN (255) contains every flag, including LOW and NO_BATTERY.
        if info.flags != 255 and info.flags & 128:
            return Power(ac)
        return Power(
            ac,
            bool(info.flags != 255 and info.flags & (2 | 4)) or info.saver == 1,
            info.life if info.life != 0xFFFFFFFF else None,
        )

    def thermal(self):
        return None

    def memory(self):
        info = MemoryStatus()
        info.length = ctypes.sizeof(info)
        if not self.kernel.GlobalMemoryStatusEx(ctypes.byref(info)):
            raise OSError("GlobalMemoryStatusEx failed")
        state = pressure(info.load, info.available)
        if info.page_total:
            state = worse(state, commit_pressure(info.page_available, self.system_free()))
        return Memory(info.total, info.available, state)

    def system_free(self):
        """Free bytes on the drive the page file grows on, or None."""
        import shutil

        try:
            return shutil.disk_usage(self.swap_volume()[0]).free
        except OSError:
            return None

    def process_bytes(self, pid):
        class Counters(ctypes.Structure):
            _fields_ = [("size", ctypes.c_uint32), ("faults", ctypes.c_uint32)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "peak",
                    "working",
                    "pool_peak",
                    "pool",
                    "nonpaged_peak",
                    "nonpaged",
                    "page",
                    "page_peak",
                )
            ]

        self.kernel.OpenProcess.restype = ctypes.c_void_p
        self.kernel.OpenProcess.argtypes = [
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_uint32,
        ]
        self.kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = self.kernel.OpenProcess(0x0400 | 0x0010, False, pid)
        if not handle:
            return None
        info = Counters()
        info.size = ctypes.sizeof(info)
        try:
            psapi = ctypes.windll.psapi
            psapi.GetProcessMemoryInfo.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint32,
            ]
            return (
                info.working
                if psapi.GetProcessMemoryInfo(handle, ctypes.byref(info), info.size)
                else None
            )
        finally:
            self.kernel.CloseHandle(handle)

    def accelerators(self):
        return discrete_accelerators() or [Accel("cpu", "CPU")]

    def keep_awake(self, pid):
        # Execution state belongs to the calling (scheduler) thread. Its handle
        # is closed when the watched worker exits.
        if self.kernel.SetThreadExecutionState(0x80000001):
            return AwakeHandle(self.kernel)
        return None

    def service(self):
        from .services import TaskScheduler

        return TaskScheduler(self.data_dir() / f"{NAME}-task.xml")
