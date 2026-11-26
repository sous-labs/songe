"""Replays of one guarded child (a Dream's training slice) through each
platform's own memory probe and the guard: macOS (vm_stat, the kernel's
pressure level, swap), Linux (meminfo, PSI, swap) and Windows (memory load,
commit charge). Same day on each: the child must finish on a normal day and
during a big download, and stop before the machine is in trouble when another
app grows or the machine starts to swap."""

import ctypes
from types import SimpleNamespace

import pytest

from songe import memory
from songe.platform import darwin
from songe.platform.linux import Linux
from songe.platform.windows import MemoryStatus, Windows

GB = 10**9
TOTAL = 24 * GB
MARGIN = TOTAL // 10  # sizing.RAM_RESERVE_FRACTION
PAGE = 16384


def tick(rss, available, *, swap=0.0, level="normal", cache=0.0):
    """What the machine shows at one second: the child's bytes, memory the OS
    counts as available (cache included), swap written so far and the OS's
    own pressure signal (normal, warn or critical)."""
    return dict(
        rss=int(rss * GB),
        available=int(available * GB),
        swap=int(swap * GB),
        level=level,
        cache=int(cache * GB),
    )


def normal_day():
    # The slice grows to its 6 GB prediction, then trains; background swap
    # drifts by a few MB.
    grow = [tick(0.5 * i, 12 - 0.45 * i, swap=0.001 * i) for i in range(1, 13)]
    return grow + [tick(6, 6.6, swap=0.012 + 0.001 * i) for i in range(30)]


def big_download():
    # A 16 GB download beside the slice: free memory becomes file cache.
    return [
        tick(6, 6.5, cache=0.5 * i, level="warn" if 10 <= i < 20 else "normal")
        for i in range(40)
    ]


def app_grows():
    # Another app takes 0.5 GB a second; the OS turns critical under 1 GB.
    def level(available):
        return "critical" if available < 1 else "warn" if available < 4 else "normal"

    return [tick(6, 7 - 0.5 * i, level=level(7 - 0.5 * i)) for i in range(14)]


def swap_storm():
    # The slice outgrows its prediction; the OS keeps 3.2 GB free by
    # compressing and swapping 0.4 GB a second.
    return [
        tick(6 + 0.5 * i, 3.2, swap=0.4 * max(0, i - 2), level="warn")
        for i in range(30)
    ]


SCENARIOS = {
    "normal_day": (normal_day, False),
    "big_download": (big_download, False),
    "app_grows": (app_grows, True),
    "swap_storm": (swap_storm, True),
}


def mac(monkeypatch, ticks, now):
    def outputs(t):
        free = max(0, t["available"] - t["cache"])
        return {
            "hw.memsize": str(TOTAL),
            "kern.memorystatus_vm_pressure_level": str(
                {"normal": 1, "warn": 2, "critical": 4}[t["level"]]
            ),
            "vm.swapusage": "total = 8192.00M used = "
            f"{(GB + t['swap']) / 2**20:.2f}M free = 1024.00M (encrypted)",
            "/usr/bin/vm_stat": f"Mach (page size of {PAGE} bytes)\n"
            f"Pages free: {free // PAGE}.\n"
            f"Pages inactive: {t['cache'] // PAGE}.\n"
            "Pages speculative: 0.\n"
            f"Pages occupied by compressor: {2 * GB // PAGE}.\n"
            f"Swapouts: {t['swap'] // PAGE}.\n",
        }

    monkeypatch.setattr(
        darwin,
        "read",
        lambda *args: outputs(ticks[now()])[args[0] if len(args) == 1 else args[-1]],
    )
    return darwin.Darwin()


def linux(monkeypatch, ticks, now, tmp_path):
    # A desktop with swap and no container limit: PSI from /proc.
    proc = tmp_path / "proc"
    (proc / "pressure").mkdir(parents=True)
    psi = {"normal": 0.0, "warn": 4.0, "critical": 60.0}
    platform = Linux(tmp_path, proc, tmp_path / "sys")
    probe = platform.memory

    def memory_now():
        t = ticks[now()]
        (proc / "meminfo").write_text(
            f"MemTotal: {TOTAL // 1024} kB\nMemAvailable: {t['available'] // 1024} kB\n"
            f"SwapTotal: {8 * GB // 1024} kB\n"
            f"SwapFree: {(7 * GB - t['swap']) // 1024} kB\n",
            encoding="utf-8",
        )
        (proc / "vmstat").write_text(
            f"pswpout {t['swap'] // 4096}\n", encoding="utf-8"
        )
        (proc / "pressure/memory").write_text(
            f"some avg10={psi[t['level']]:.2f} avg60=0 avg300=0 total=0\n",
            encoding="utf-8",
        )
        return probe()

    platform.memory = memory_now
    platform.accelerators = lambda: []
    return platform


def windows(monkeypatch, ticks, now):
    class Kernel:
        def GlobalMemoryStatusEx(self, pointer):
            t = ticks[now()]
            info = pointer._obj
            assert info.length == ctypes.sizeof(MemoryStatus)
            info.total, info.available = TOTAL, t["available"]
            info.load = 100 - 100 * t["available"] // TOTAL
            # Commit limit: RAM plus a 16 GB page file, used as RAM is.
            info.page_total = TOTAL + 16 * GB
            info.page_available = t["available"] + 16 * GB - t["swap"]
            return 1

    platform = Windows(kernel=Kernel(), user=SimpleNamespace())
    monkeypatch.setattr(platform, "system_free", lambda: 200 * GB)
    platform.accelerators = lambda: []
    return platform


def replay(monkeypatch, tmp_path, system, ticks, *, predicted=6.2):
    state = {"i": 0, "time": 0.0, "done": False}

    def now():
        return state["i"]

    platform = {
        "macos": lambda: mac(monkeypatch, ticks, now),
        "linux": lambda: linux(monkeypatch, ticks, now, tmp_path),
        "windows": lambda: windows(monkeypatch, ticks, now),
    }[system]()
    monkeypatch.setattr(
        memory, "process_tree_bytes", lambda p, pid: ticks[state["i"]]["rss"]
    )
    monkeypatch.setattr(memory, "process_tree_swap_bytes", lambda p, pid: None)
    monkeypatch.setattr(memory.time, "monotonic", lambda: state["time"])

    class Stop:
        def is_set(self):
            return state["done"]

        def wait(self, interval):
            state["time"] += 1.0
            if state["i"] + 1 < len(ticks):
                state["i"] += 1
            else:
                state["done"] = True

    stops = []
    memory.guard(
        SimpleNamespace(pid=12345, returncode=None),
        platform,
        lambda m: stops.append((state["i"], "interrupt", m)),
        lambda m: stops.append((state["i"], "kill", m)),
        baseline=memory.snapshot(platform),
        margin_bytes=MARGIN,
        predicted_peak_bytes=int(predicted * GB) if predicted else None,
        prediction_error_bytes=int(predicted * GB) // 20 if predicted else 0,
        projection=predicted is not None,
        stop=Stop(),
    )
    return stops


SYSTEMS = ["macos", "linux", "windows"]


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("system", SYSTEMS)
def test_replayed_day(monkeypatch, tmp_path, system, scenario):
    make, must_stop = SCENARIOS[scenario]
    ticks = make()
    stops = replay(monkeypatch, tmp_path, system, ticks)
    assert bool(stops) == must_stop, [
        (i, action, m.stop_reason["rule"]) for i, action, m in stops
    ]
    if must_stop:
        first = stops[0][0]
        # Stopped before the OS itself called it critical ...
        assert all(t["level"] != "critical" for t in ticks[: first + 1])
        # ... and before 2 GB more were swapped out.
        assert ticks[first]["swap"] - ticks[0]["swap"] < 2 * GB


# Windows reports no paging: there, memory load, commit charge and the
# reserve are what stop a slice that has no prediction yet.
@pytest.mark.parametrize("system", ["macos", "linux"])
def test_replayed_swap_storm_while_measuring(monkeypatch, tmp_path, system):
    # Measuring has no prediction yet and no projection stop: the swap the
    # slice causes must still stop it.
    ticks = swap_storm()
    stops = replay(monkeypatch, tmp_path, system, ticks, predicted=None)
    assert stops, system
    assert ticks[stops[0][0]]["swap"] < 2 * GB
