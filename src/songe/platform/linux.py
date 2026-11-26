import mmap
import os
import re
import shutil
import subprocess
from collections import deque
from pathlib import Path
from statistics import median

from .. import NAME
from . import Accel, Memory, Power
from .common import ChildHandle, discrete_accelerators, read


def pressure(available, total, psi, baseline, *, warn_ratio=2.0, critical_ratio=4.0):
    """PSI is relative to this instance's recent median (not machine-size rules).

    Relative doubling/quadrupling marks warning/critical. Available RAM at zero
    is always critical. A zero PSI history is bootstrapped by its first sample.
    Ratios are named, overridable sensitivity controls, not model exclusions.
    """
    if total <= 0 or available <= 0:
        return "critical"
    if baseline > 0 and psi >= baseline * critical_ratio:
        return "critical"
    if baseline > 0 and psi >= baseline * warn_ratio:
        return "warn"
    # PSI measures actual time stalled for memory, including at process start.
    # Do not normalize existing stalls into a healthy allocation baseline.
    if psi > 0:
        return "warn"
    return "normal"


CACHE_STATS = {
    "v2": (
        "inactive_file",
        "active_file",
        "file_mapped",
        "file_dirty",
        "file_writeback",
    ),
    "v1": (
        "total_inactive_file",
        "total_active_file",
        "total_mapped_file",
        "total_dirty",
        "total_writeback",
    ),
}


def reclaimable_cache(folder, kind):
    """Reclaimable page cache a cgroup's usage includes (0 when unknown).

    All inactive cache, and active cache less what programs map (a model
    server's mmapped weights) and what is not yet written back. A file read
    twice is active: on cgroup v1, a 93 GB file written and read back left
    29.7 GB active and 7.4 GB inactive, usage at the 62 GB limit, no OOM.
    """
    try:
        stat = (folder / "memory.stat").read_text(encoding="utf-8")
    except OSError:
        return 0

    def value(name):
        match = re.search(rf"^{name}\s+(\d+)", stat, re.M)
        return int(match[1]) if match else 0

    inactive, active, mapped, dirty, writeback = map(value, CACHE_STATS[kind])
    return max(inactive, inactive + active - mapped - dirty - writeback)


class Linux:
    name = "linux"

    def __init__(
        self,
        home=None,
        proc=Path("/proc"),
        sysfs=Path("/sys"),
        environ=None,
        history=60,
    ):
        self.home = Path(home) if home is not None else Path.home()
        self.proc, self.sysfs = Path(proc), Path(sysfs)
        self.environ = os.environ if environ is None else environ
        # One minute of guard samples smooths short allocation bursts.
        self.psi_history = deque(maxlen=history)

    def data_dir(self):
        return (
            Path(self.environ.get("XDG_DATA_HOME", self.home / ".local/share")) / NAME
        )

    def swap_volume(self):
        # Swap partitions and swap files have a fixed size here: paging
        # cannot fill a disk, so there is nothing to keep free for it.
        return None

    def idle_seconds(self):
        # No portable input-idle API for headless/Wayland sessions.
        return None

    def on_ac_power(self):
        return self.power().ac

    def power(self):
        def value(supply, name):
            try:
                return (supply / name).read_text(encoding="utf-8").strip()
            except OSError:
                return ""

        supplies = list((self.sysfs / "class/power_supply").glob("*"))
        mains, batteries, levels, times, capacities = [], [], [], [], []
        for supply in supplies:
            # Peripheral and empty battery slots are not system batteries.
            if value(supply, "scope") == "Device" or value(supply, "present") == "0":
                continue
            kind = value(supply, "type")
            if kind in {"Mains", "USB", "USB_C"}:
                online = value(supply, "online")
                if online in {"0", "1"}:
                    mains.append(online == "1")
            elif kind == "Battery":
                batteries.append(value(supply, "status"))
                levels.append(value(supply, "capacity_level"))
                capacity = value(supply, "capacity")
                if capacity.isdecimal():
                    capacities.append(int(capacity) / 100)
                for name in ("time_to_empty_now", "time_to_empty_avg"):
                    remaining = value(supply, name)
                    if remaining.isdecimal():
                        times.append(int(remaining))
                        break
        ac = (
            any(mains)
            if mains
            else (
                False
                if "Discharging" in batteries
                else True
                if batteries and all(s in {"Charging", "Full"} for s in batteries)
                else None
            )
        )
        low = any(level in {"Low", "Critical"} for level in levels)
        signalled = any(
            level in {"Low", "Critical", "Normal", "High", "Full"} for level in levels
        )
        fraction = min(capacities) if capacities and not signalled else None
        seconds = min(times) if times else None
        # UPower's aggregate display device also covers drivers without sysfs
        # warning/time attributes. Optional CLI only: no D-Bus dependency.
        if ac is not True and shutil.which("upower"):
            try:
                info = dict(
                    re.findall(
                        r"^\s*([\w -]+):\s*(.*?)\s*$",
                        read(
                            "upower",
                            "-i",
                            "/org/freedesktop/UPower/devices/DisplayDevice",
                        ),
                        re.M,
                    )
                )
                if info.get("present") == "yes":
                    if ac is None and info.get("state") in {
                        "discharging",
                        "charging",
                        "fully-charged",
                    }:
                        ac = info["state"] != "discharging"
                    low = low or info.get("warning-level") in {
                        "low",
                        "critical",
                        "action",
                    }
                    remaining = re.fullmatch(
                        r"([\d.]+) (seconds|minutes|hours|days)",
                        info.get("time to empty", ""),
                    )
                    if seconds is None and remaining:
                        estimate = (
                            float(remaining[1])
                            * {
                                "seconds": 1,
                                "minutes": 60,
                                "hours": 3600,
                                "days": 86400,
                            }[remaining[2]]
                        )
                        seconds = estimate if estimate > 0 else None
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
        return Power(ac, low, seconds, fraction)

    def thermal(self):
        # Linux thermal zones expose vendor-specific temperatures, not a common
        # nominal/fair/serious state. Unknown is preferable to invented limits.
        return None

    def memory_cgroup(self):
        """Resolve our memory controller through its mount (v1 or v2).

        Container namespaces can expose the controller at the mount root even
        when /proc/self/cgroup retains the host path. Never inspect siblings.
        """
        try:
            memberships = {}
            for line in (self.proc / "self/cgroup").read_text().splitlines():
                _, controllers, group = line.split(":", 2)
                if not controllers or "memory" in controllers.split(","):
                    memberships["v1" if controllers else "v2"] = Path(group)
            for line in (self.proc / "self/mountinfo").read_text().splitlines():
                fields, filesystem = line.split(" - ", 1)
                fields, filesystem = fields.split(), filesystem.split()
                kind = {"cgroup2": "v2", "cgroup": "v1"}.get(filesystem[0])
                if kind not in memberships:
                    continue
                if kind == "v1" and "memory" not in filesystem[2].split(","):
                    continue

                # mountinfo escapes whitespace and backslashes as octal.
                def unescape(value):
                    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), value)

                root, mount = map(lambda v: Path(unescape(v)), fields[3:5])
                if mount.is_relative_to("/sys"):
                    mount = self.sysfs / mount.relative_to("/sys")
                group = memberships[kind]
                relative = (
                    group.relative_to(root) if group.is_relative_to(root) else Path(".")
                )
                leaf = mount / relative if ".." not in relative.parts else mount
                current = "memory.current" if kind == "v2" else "memory.usage_in_bytes"
                if not (leaf / current).is_file():
                    leaf = mount
                if (leaf / current).is_file():
                    return kind, leaf, mount
        except (OSError, ValueError, IndexError):
            pass
        return None

    def memory(self):
        entries = {
            k: int(v) * 1024
            for k, v in re.findall(
                r"^(\w+):\s+(\d+) kB", (self.proc / "meminfo").read_text(), re.M
            )
        }
        total, available = entries["MemTotal"], entries["MemAvailable"]
        scope = self.memory_cgroup()
        swap_used = (
            entries["SwapTotal"] - entries["SwapFree"]
            if "SwapTotal" in entries and "SwapFree" in entries
            else None
        )
        swap_out = None
        compressed = entries.get("Zswap")
        psi_file = self.proc / "pressure/memory"
        if scope:
            kind, leaf, mount = scope
            current_name, limit_name = (
                ("memory.current", "memory.max")
                if kind == "v2"
                else ("memory.usage_in_bytes", "memory.limit_in_bytes")
            )
            # An ancestor can impose a tighter limit than the leaf. Include its
            # shared usage in headroom rather than promising host-sized RAM.
            folder = leaf
            while True:
                if folder == mount and not (folder / limit_name).exists():
                    break  # The host controller root has no memory limit files.
                limit = (folder / limit_name).read_text(encoding="utf-8").strip()
                if limit != "max":
                    limit = int(limit)
                    used = int((folder / current_name).read_text(encoding="utf-8"))
                    # Usage counts the page cache, which is reclaimed on
                    # demand (downloaded weights, files read back).
                    used -= min(used, reclaimable_cache(folder, kind))
                    total = min(total, limit)
                    available = min(available, max(0, limit - used))
                if folder == mount:
                    break
                folder = folder.parent
            available = min(available, total)
            # Host paging/PSI includes unrelated tenants. Use only controller
            # sensors here; v1 has no PSI or cumulative swap-out byte counter.
            psi_file = leaf / "memory.pressure" if kind == "v2" else None
            compressed = None
            swap_used = None
            try:
                if kind == "v2":
                    swap_used = int((leaf / "memory.swap.current").read_text())
                else:
                    stat = dict(
                        line.split()
                        for line in (leaf / "memory.stat").read_text().splitlines()
                    )
                    value = stat.get("total_swap", stat.get("swap"))
                    if value is not None:
                        swap_used = int(value)
            except (OSError, ValueError):
                pass
        else:
            try:
                match = re.search(
                    r"^pswpout\s+(\d+)", (self.proc / "vmstat").read_text(), re.M
                )
                if match:
                    swap_out = int(match[1]) * mmap.PAGESIZE
            except OSError:
                pass
        psi = None
        try:
            match = (
                re.search(r"^some avg10=([\d.]+)", psi_file.read_text(), re.M)
                if psi_file
                else None
            )
            psi = float(match[1]) if match else None
        except OSError:
            pass
        baseline = median(self.psi_history) if self.psi_history else (psi or 0)
        level = pressure(available, total, psi or 0, baseline)
        if psi is not None:
            self.psi_history.append(psi)
        return Memory(
            total,
            available,
            level,
            swap_used=swap_used,
            compressed=compressed,
            swap_out=swap_out,
            psi=psi,
        )

    def process_swap_bytes(self, pid):
        try:
            status = (self.proc / str(pid) / "status").read_text()
            match = re.search(r"^VmSwap:\s+(\d+) kB", status, re.M)
            return int(match[1]) * 1024 if match else None
        except (OSError, ValueError):
            return None

    def process_bytes(self, pid):
        try:
            status = (self.proc / str(pid) / "status").read_text()
            match = re.search(r"^VmRSS:\s+(\d+) kB", status, re.M)
            return int(match[1]) * 1024 if match else None
        except (OSError, ValueError):
            return None

    def accelerators(self):
        # DRM inventory does not imply a usable training device. Intel graphics
        # needs real allocation telemetry before it can be advertised as XPU.
        return discrete_accelerators() or [Accel("cpu", "CPU")]

    def keep_awake(self, pid):
        if not shutil.which("systemd-inhibit"):
            return None
        # Follow the worker lifetime without invoking a shell or user code.
        script = "import os,sys,time\np=int(sys.argv[1])\nwhile os.path.exists('/proc/'+str(p)): time.sleep(1)"
        import sys

        return ChildHandle(
            subprocess.Popen(
                [
                    "systemd-inhibit",
                    "--what=idle:sleep",
                    "--mode=block",
                    sys.executable,
                    "-c",
                    script,
                    str(pid),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )

    def service(self):
        from .services import Systemd

        return Systemd(
            Path(self.environ.get("XDG_CONFIG_HOME", self.home / ".config"))
            / f"systemd/user/{NAME}.service"
        )
