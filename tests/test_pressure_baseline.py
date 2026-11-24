"""Admission and guards on busy machines; no hardware or model allocations."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from songe import memory
from songe.config import Config
from songe.platform import Accel, Memory
from songe.schedule import Probes, reason
from songe.sleep.sizing import PressureHistory, choose_size


def platform(sample):
    return SimpleNamespace(name="fake", memory=lambda: sample, accelerators=lambda: [])


def test_warn_admits_measured_slice_and_records_baseline_before_launch(monkeypatch):
    # Isolates the boundary logic from the fixed 10% RAM reserve.
    monkeypatch.setattr("songe.sleep.sizing.RAM_RESERVE_FRACTION", 0)
    baseline = Memory(24_000, 4_800, "warn", swap_used=2_000, compressed=9_000)
    p = platform(baseline)
    samples = []

    def launch(*args, **kwargs):
        assert len(samples) == 4
        assert samples[0].compressed == 9_000 and samples[0].swap_used == 2_000
        assert samples[-1].paging_noise["sensors"]["swap_used"]["delta_bytes"] == 0
        return SimpleNamespace(
            pid=12345,
            returncode=0,
            poll=lambda: 0,
            communicate=lambda **kwargs: (b"done", b""),
            wait=lambda: 0,
        )

    monkeypatch.setattr(memory.subprocess, "Popen", launch)
    rows = [dict(sequence=64, host_peak_bytes=1_000, device_peak_bytes=0)]
    # A normal->warn boundary must not veto a safe warning baseline. The
    # warn->critical boundary remains reserved, including its transition width.
    history = {
        "warning_available": 8_000,
        "transition_width": 1_000,
        "critical_boundary": {"warning_available": 500, "transition_width": 500},
    }
    choice = choose_size(
        Config(response_tokens=1), {"rows": rows}, memory.snapshot(p), history
    )
    assert choice and choice["margin_bytes"] == 1_000
    result = memory.guarded_run(
        ["fake"],
        platform=p,
        required_bytes=choice["host_bytes"],
        margin_bytes=choice["margin_bytes"],
        on_sample=samples.append,
    )
    assert result.stdout == b"done"


@pytest.mark.parametrize("pressure", ["normal", "warn", "critical"])
def test_oversize_or_critical_never_launches(monkeypatch, pressure):
    monkeypatch.setattr(
        memory.subprocess, "Popen", lambda *a, **k: pytest.fail("launched")
    )
    with pytest.raises(memory.MemoryPressure):
        memory.guarded_run(
            ["fake"],
            platform=platform(Memory(10_000, 2_000, pressure)),
            required_bytes=1_800,
            margin_bytes=300,
        )
    assert (
        choose_size(
            Config(),
            {"rows": [dict(sequence=64, host_peak_bytes=2_001, device_peak_bytes=0)]},
            memory.snapshot(platform(Memory(10_000, 2_000, pressure))),
            {},
        )
        is None
    )


def run_guard(
    monkeypatch,
    samples,
    *,
    baseline=None,
    rss=None,
    swapped=None,
    margin=0,
    events=False,
    policy=None,
    accelerators=None,
    **kwargs,
):
    state = {"index": -1, "time": 0.0, "waits": 0}

    def sample():
        state["index"] = min(state["index"] + 1, len(samples) - 1)
        return samples[state["index"]]

    p = platform(None)
    p.memory = sample
    if accelerators is not None:
        # Devices as seen alongside the current memory sample.
        p.accelerators = lambda: accelerators(samples[state["index"]])
    if rss is not None:
        p.process_bytes = lambda pid: rss[state["index"]]
    monkeypatch.setattr(
        memory, "process_tree_bytes", lambda p, pid: p.process_bytes(pid)
    )
    monkeypatch.setattr(
        memory,
        "process_tree_swap_bytes",
        lambda p, pid: swapped[state["index"]] if swapped is not None else None,
    )
    monkeypatch.setattr(memory.time, "monotonic", lambda: state["time"])

    class Stop:
        def is_set(self):
            return state["waits"] >= 8

        def wait(self, interval):
            state["waits"] += 1
            state["time"] += interval

    calls = []
    memory.guard(
        SimpleNamespace(pid=12345, returncode=None),
        p,
        lambda m: calls.append(("interrupt", m) if events else "interrupt"),
        lambda m: calls.append(("kill", m) if events else "kill"),
        baseline=baseline,
        margin_bytes=margin,
        stop=Stop(),
        policy=policy or memory.GuardPolicy(0.1, 0.1, 0.2),
        **kwargs,
    )
    return calls


def test_unchanged_warn_outlives_grace(monkeypatch):
    sample = Memory(24_000, 4_800, "warn", swap_used=2_000, compressed=9_000)
    assert run_guard(monkeypatch, [sample], baseline=sample, rss=[100]) == []


def test_warning_baseline_escalation_to_critical_kills(monkeypatch):
    baseline = Memory(24_000, 4_800, "warn")
    assert run_guard(
        monkeypatch, [replace(baseline, pressure="critical")], baseline=baseline
    ) == ["kill"]


@pytest.mark.parametrize("sensor", ["swap_used", "swap_out"])
def test_global_paging_growth_with_owned_allocation_interrupts_then_kills(
    monkeypatch, sensor
):
    baseline = Memory(24_000, 4_800, "warn", **{sensor: 2_000})
    growing = [replace(baseline, **{sensor: 2_000 + 100 * (i + 1)}) for i in range(8)]
    assert run_guard(monkeypatch, growing, baseline=baseline, rss=[100] * 8) == [
        "interrupt",
        "kill",
    ]


def test_one_swap_burst_while_loading_is_not_harm(monkeypatch):
    # Real Mac init: 95 MB swapped once while 675 MB of weights loaded.
    baseline = Memory(24_000, 4_800, "normal", swap_out=2_000)
    burst = replace(baseline, swap_out=2_095)
    assert run_guard(monkeypatch, [burst], baseline=baseline, rss=[675]) == []


def test_process_tree_swap_growth_stops_even_with_flat_global_usage(monkeypatch):
    baseline = Memory(24_000, 4_800, "warn", swap_used=2_000)
    assert run_guard(
        monkeypatch, [baseline], baseline=baseline, rss=[100], swapped=[100]
    ) == ["interrupt", "kill"]


def test_unrelated_paging_without_owned_growth_does_not_alarm(monkeypatch):
    baseline = Memory(24_000, 4_800, "warn", compressed=9_000)
    # An already-running tree remains the same size throughout these samples.
    assert (
        run_guard(
            monkeypatch, [baseline, replace(baseline, compressed=9_100)], rss=[100, 100]
        )
        == []
    )


def test_psi_alone_does_not_attribute_paging_to_owned_allocation(monkeypatch):
    baseline = Memory(24_000, 4_800, "warn", psi=1.0)
    assert (
        run_guard(
            monkeypatch, [replace(baseline, psi=2.0)], baseline=baseline, rss=[100]
        )
        == []
    )


def test_learned_boundary_stops_before_exhaustion(monkeypatch):
    baseline = Memory(24_000, 4_800, "warn")
    assert run_guard(
        monkeypatch, [replace(baseline, available=900)], baseline=baseline, margin=1_000
    ) == ["kill"]


def test_warn_critical_boundary_direction_and_expiry(monkeypatch):
    # Isolates the boundary logic from the fixed 10% RAM reserve.
    monkeypatch.setattr("songe.sleep.sizing.RAM_RESERVE_FRACTION", 0)
    history = PressureHistory()
    for sample in [Memory(24_000, 5_000, "warn"), Memory(24_000, 3_000, "critical")]:
        history.observe(sample)
    assert history.margin_for(Memory(24_000, 6_000, "warn")) == 5_000
    # Recovery at higher headroom does not immediately erase measured evidence.
    history.observe(Memory(24_000, 6_000, "warn"))
    assert history.margin_for(Memory(24_000, 6_000, "warn")) == 5_000
    for _ in range(4):
        history = PressureHistory(history.data)
        history.observe(Memory(24_000, 6_000, "warn"))
    assert history.margin_for(Memory(24_000, 6_000, "warn")) == 0
    # A rise in available memory during escalation is not a capacity boundary.
    history = PressureHistory()
    history.observe(Memory(24_000, 3_000, "warn"))
    history.observe(Memory(24_000, 5_000, "critical"))
    assert history.margin_for(Memory(24_000, 6_000, "warn")) == 0


def test_scheduler_allows_fitting_warning_baseline():
    config = Config(model="fake", calibration_available=True, calibrated_peak_gb=1e-6)
    probes = Probes(True, 1800, 0, 24e-6, 4_800e-9, True, "warn")
    assert (
        reason(config, probes, enough=True, traffic_idle=1800) == "Will sleep when idle"
    )


def test_warning_baseline_still_requires_selected_vram():
    p = platform(Memory(24_000, 4_800, "warn"))
    p.accelerators = lambda: [Accel("cuda", "selected", 10_000, 500)]
    with pytest.raises(memory.MemoryPressure):
        memory.admit(p, 1_000, 501)


def test_swap_measurement_includes_only_owned_descendants(monkeypatch):
    import psutil

    seen = []

    def swapped(pid):
        seen.append(pid)
        return {41: 0, 42: 100, 43: 200}[pid]

    p = platform(None)
    p.process_swap_bytes = swapped
    monkeypatch.setattr(
        psutil,
        "Process",
        lambda pid: SimpleNamespace(
            children=lambda recursive: [
                SimpleNamespace(pid=42),
                SimpleNamespace(pid=43),
            ]
        ),
    )
    assert memory.process_tree_swap_bytes(p, 41) == 300
    assert seen == [41, 42, 43]


def test_linux_paging_sensors_and_process_swap(tmp_path):
    import mmap

    from songe.platform.linux import Linux

    (tmp_path / "pressure").mkdir()
    (tmp_path / "pressure/memory").write_text("some avg10=1.00\n", encoding="utf-8")
    (tmp_path / "meminfo").write_text(
        "MemTotal: 10000 kB\nMemAvailable: 4000 kB\nSwapTotal: 2000 kB\nSwapFree: 1500 kB\nZswap: 100 kB\n",
        encoding="utf-8",
    )
    (tmp_path / "vmstat").write_text("pswpout 12\n", encoding="utf-8")
    (tmp_path / "42").mkdir()
    (tmp_path / "42/status").write_text(
        "VmRSS: 100 kB\nVmSwap: 10 kB\n", encoding="utf-8"
    )
    p = Linux(proc=tmp_path)
    sample = memory.snapshot(p)
    assert sample.swap_used == 500 * 1024 and sample.compressed == 100 * 1024
    assert sample.swap_out == 12 * mmap.PAGESIZE and sample.psi == 1.0
    assert p.process_swap_bytes(42) == 10 * 1024
    assert p.process_swap_bytes(43) is None


def test_darwin_records_compressor_and_swap_without_allocating(monkeypatch):
    from songe.platform import darwin

    outputs = {
        "hw.memsize": "24000",
        "kern.memorystatus_vm_pressure_level": "2",
        "vm.swapusage": "total = 4.00M used = 1.50M free = 2.50M (encrypted)",
        "/usr/bin/vm_stat": "Mach (page size of 4096 bytes)\nPages free: 5.\nPages occupied by compressor: 2.\nSwapouts: 7.",
    }
    monkeypatch.setattr(
        darwin, "read", lambda *args: outputs[args[0] if len(args) == 1 else args[-1]]
    )
    sample = memory.snapshot(darwin.Darwin())
    assert sample.pressure == "warn" and sample.available == 5 * 4096
    assert sample.compressed == 2 * 4096 and sample.swap_out == 7 * 4096
    assert sample.swap_used == int(1.5 * 1024**2)


def test_globalmemorystatusex_warning_baseline_admits():
    from songe.platform.windows import Windows
    from tests.test_platform import Kernel, User

    sample = memory.admit(Windows(kernel=Kernel(), user=User()), 50, 0)
    assert sample.pressure == "warn" and sample.available == 100
    assert sample.swap_used is None and sample.compressed is None


def test_compression_making_room_for_owned_allocation_is_not_paging(monkeypatch):
    # Real Mac e2e: macOS compressed 630 MB of other apps while Songe loaded
    # 1.1 GB at unchanged warn pressure, with no swap traffic.
    baseline = Memory(24_000, 4_800, "warn", swap_used=2_000, compressed=9_000)
    sample = replace(baseline, compressed=9_600)
    assert run_guard(monkeypatch, [sample], baseline=baseline, rss=[1_100]) == []
