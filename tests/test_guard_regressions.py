"""False-stop regressions and hard-stop coverage using deterministic telemetry."""

import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from songe import memory
from songe.platform import Accel, Memory
from songe.sleep.sizing import choose_size
from tests.test_pressure_baseline import platform, run_guard

GB = 10**9


def noisy_baseline(monkeypatch, sensor="compressed"):
    state = {"index": 0, "time": 0.0}

    def sample():
        state["index"] += 1
        return Memory(24 * GB, 8 * GB, **{sensor: GB + state["index"] * 10**6})

    def sleep(seconds):
        state["time"] += seconds

    monkeypatch.setattr(memory.time, "sleep", sleep)
    monkeypatch.setattr(memory.time, "monotonic", lambda: state["time"])
    p = platform(None)
    p.memory = sample
    baseline = memory.capture_baseline(p)
    assert state["index"] == 4
    assert baseline.paging_noise["window_seconds"] == pytest.approx(0.3)
    return baseline


@pytest.mark.parametrize("sensor", ["compressed", "swap_used", "swap_out"])
def test_background_paging_with_own_load_below_measured_noise_completes(
    monkeypatch, sensor
):
    baseline = noisy_baseline(monkeypatch, sensor)
    # This is material (10%) relative to our growth, but below background noise.
    sample = replace(baseline, **{sensor: getattr(baseline, sensor) + 2 * 10**6})
    assert run_guard(monkeypatch, [sample], baseline=baseline, rss=[20 * 10**6]) == []


@pytest.mark.parametrize("sensor", ["compressed", "swap_used", "swap_out"])
def test_tiny_global_paging_with_gigabyte_owned_growth_completes(monkeypatch, sensor):
    baseline = memory.snapshot(platform(Memory(24 * GB, 8 * GB, **{sensor: GB})))
    sample = replace(baseline, available=7 * GB, **{sensor: GB + 1})
    assert run_guard(monkeypatch, [sample], baseline=baseline, rss=[GB]) == []


def test_material_paging_above_background_noise_still_stops(monkeypatch):
    baseline = noisy_baseline(monkeypatch, "swap_out")
    growing = [
        replace(baseline, swap_out=baseline.swap_out + (i + 1) * GB // 4)
        for i in range(8)
    ]
    events = run_guard(
        monkeypatch, growing, baseline=baseline, rss=[GB] * 8, events=True
    )
    assert [action for action, _ in events] == ["interrupt", "kill"]
    reason = events[0][1].stop_reason
    assert reason["rule"] == "attributed_paging"
    assert reason["observed"]["paging"]["swap_out"]["noise_bytes"] > 0
    assert reason["baseline"]["paging_noise"] == baseline.paging_noise


def test_process_swap_is_decisive_despite_noisy_global_counters(monkeypatch):
    baseline = noisy_baseline(monkeypatch)
    events = run_guard(
        monkeypatch,
        [baseline],
        baseline=baseline,
        rss=[GB],
        swapped=[4096],
        events=True,
    )
    assert [action for action, _ in events] == ["interrupt", "kill"]
    assert events[0][1].stop_reason["rule"] == "process_paging"


@pytest.mark.parametrize("prediction", [None, 3 * GB])
def test_bounded_load_burst_with_gigabytes_free_completes(monkeypatch, prediction):
    baseline = Memory(24 * GB, 10 * GB)
    samples = [
        replace(baseline, available=9 * GB),
        replace(baseline, available=7_776_550_912),
    ]
    assert (
        run_guard(
            monkeypatch,
            samples,
            baseline=baseline,
            rss=[GB, 2_223_449_088],
            predicted_peak_bytes=prediction,
            margin=GB,
            policy=memory.GuardPolicy(0.1, 0.1, 3),
        )
        == []
    )


def test_prediction_takes_priority_over_smoothed_rate_for_fitting_burst(monkeypatch):
    # Even the smoothed unknown-load projection would stop this large burst.
    baseline = Memory(24 * GB, 10 * GB)
    sample = replace(baseline, available=3 * GB)
    assert (
        run_guard(
            monkeypatch,
            [sample],
            baseline=baseline,
            rss=[7 * GB],
            predicted_peak_bytes=8 * GB,
            margin=GB,
            policy=memory.GuardPolicy(0.1, 0.1, 3),
        )
        == []
    )


def test_unmeasured_sustained_depletion_still_stops(monkeypatch):
    baseline = Memory(24 * GB, 8 * GB)
    events = run_guard(
        monkeypatch,
        [replace(baseline, available=int(n * GB)) for n in (7, 5, 0.2)],
        baseline=baseline,
        rss=[GB, 3 * GB, 6 * GB],
        events=True,
    )
    assert [action for action, _ in events] == ["interrupt", "kill"]
    assert events[0][1].stop_reason["rule"] == "allocation_rate"
    assert (
        events[0][1].stop_reason["observed"]["projection"]["ram"]["window_seconds"] >= 3
    )


def test_prediction_remaining_need_reserves_learned_margin(monkeypatch):
    baseline = Memory(24 * GB, 8 * GB)
    events = run_guard(
        monkeypatch,
        [replace(baseline, available=4 * GB)],
        baseline=baseline,
        rss=[GB],
        predicted_peak_bytes=5 * GB,
        margin=GB,
        events=True,
    )
    first = events[0][1].stop_reason
    assert first["rule"] == "predicted_margin"
    assert first["prediction"]["remaining_need_bytes"] == 4 * GB
    assert first["prediction"]["projected_available_bytes"] == 0
    assert events[-1][1].stop_reason["observed"]["trigger"] == first


@pytest.mark.parametrize("excess,stops", [(100, False), (101, True)])
def test_actual_peak_respects_learned_prediction_error(monkeypatch, excess, stops):
    baseline = Memory(24 * GB, 8 * GB)
    events = run_guard(
        monkeypatch,
        [baseline],
        baseline=baseline,
        rss=[GB + excess],
        predicted_peak_bytes=GB,
        prediction_error_bytes=100,
        events=True,
    )
    assert bool(events) == stops
    if stops:
        assert events[0][1].stop_reason["rule"] == "prediction_exceeded"
        assert events[0][1].stop_reason["observed"]["tree_rss_bytes"] == GB + excess


def test_critical_pressure_stops_with_exact_reason(monkeypatch):
    baseline = Memory(24 * GB, 8 * GB)
    events = run_guard(
        monkeypatch,
        [replace(baseline, pressure="critical")],
        baseline=baseline,
        events=True,
        predicted_peak_bytes=GB,
    )
    assert events[-1][0] == "kill"
    assert events[0][1].stop_reason["rule"] == "critical_pressure"


def test_warn_caused_by_a_fitting_slice_is_not_harm(monkeypatch):
    # Real Mac e2e: normal -> warn after a 2.4 GB load with 5 GB still free.
    baseline = Memory(24 * GB, 8 * GB)
    sample = replace(baseline, available=5 * GB, pressure="warn")
    assert (
        run_guard(monkeypatch, [sample], baseline=baseline, predicted_peak_bytes=GB)
        == []
    )


def test_admission_reason_is_machine_readable_and_contains_prediction():
    with pytest.raises(memory.MemoryPressure) as stopped:
        memory.admit(platform(Memory(24 * GB, 2 * GB)), 3 * GB)
    exc = stopped.value
    assert json.loads(str(exc).split("; reason=", 1)[1]) == json.loads(
        json.dumps(exc.reason)
    )
    assert exc.reason["rule"] == "admission_prediction"
    assert exc.reason["prediction"]["peak_bytes"] == 3 * GB
    assert exc.reason["observed"]["available"] == 2 * GB


def test_lost_telemetry_has_reason_even_before_first_sample(monkeypatch):
    p = platform(None)
    p.memory = lambda: (_ for _ in ()).throw(OSError("sensor lost"))
    stopped = []
    memory.guard(SimpleNamespace(returncode=None), p, on_critical=stopped.append)
    assert stopped[0].stop_reason["rule"] == "telemetry_unavailable"
    assert stopped[0].stop_reason["observed"]["error"] == "sensor lost"


def test_sizing_carries_measured_error_into_live_guard(config):
    config.response_tokens = 1
    rows = [
        dict(sequence=64, response_tokens=1, host_peak_bytes=n, device_peak_bytes=0)
        for n in (1000, 1100)
    ]
    choice = choose_size(
        config, {"rows": rows}, memory.snapshot(platform(Memory(24_000, 8_000))), {}
    )
    assert choice["host_bytes"] == 1100
    assert choice["prediction_error_bytes"] == 100


def test_calibration_stop_persists_even_without_first_measurement(
    config, store, monkeypatch
):
    from songe.sleep.calibrate import calibrate

    sample = memory.stopped_sample(
        memory.snapshot(platform(Memory(24 * GB, 4 * GB))), "process_paging"
    )

    def guarded(*args, **kwargs):
        raise memory.MemoryPressure(sample)

    monkeypatch.setattr(memory, "guarded_run", guarded)
    with pytest.raises(memory.MemoryPressure):
        calibrate(config, store.root)
    saved = store.get("calibration")["memory_stop"]
    assert (
        saved == store.get("last_memory_stop") == json.loads(json.dumps(asdict(sample)))
    )
    assert (
        json.loads((store.root / "calibration.json").read_text(encoding="utf-8"))[
            "memory_stop"
        ]
        == saved
    )


def test_night_stop_persists_after_final_child_record(config, store, monkeypatch):
    from songe.sleep.worker import isolated

    sample = memory.stopped_sample(
        memory.snapshot(platform(Memory(24 * GB, 4 * GB))), "prediction_exceeded"
    )

    def guarded(*args, **kwargs):
        store.night(None, status="interrupted", record={"checkpoint": "saved"})
        raise memory.MemoryPressure(sample)

    monkeypatch.setattr(memory, "guarded_run", guarded)
    with pytest.raises(memory.MemoryPressure):
        isolated(config, store)
    assert store.nights()[-1]["record"] == {
        "checkpoint": "saved",
        "memory_stop": json.loads(json.dumps(asdict(sample))),
    }


def test_calibration_meter_includes_owned_group_leader(monkeypatch):
    from songe.sleep import measurement

    monkeypatch.setenv("SONGE_GUARD_PID", "42")
    monkeypatch.setattr(measurement.os, "getppid", lambda: 42)
    seen = []
    monkeypatch.setattr(
        measurement, "process_tree_bytes", lambda p, pid: seen.append(pid) or 120
    )
    p = platform(Memory(1000, 900))
    with measurement.Meter(p, memory.snapshot(p)) as meter:
        meter.sample()
    assert set(seen) == {42} and meter.result()["host_peak_bytes"] == 120


def test_prediction_does_not_hide_device_exhaustion(monkeypatch):
    p = platform(Memory(24 * GB, 8 * GB))
    p.accelerators = lambda: [Accel("cuda", "fake GPU", 8 * GB, 0)]
    samples = []
    memory.guard(
        SimpleNamespace(returncode=None),
        p,
        on_critical=samples.append,
        predicted_peak_bytes=GB,
        predicted_device_bytes=GB,
    )
    assert samples[0].stop_reason["rule"] == "critical_pressure"


def test_first_smallest_calibration_row_survives_noisy_bounded_load(
    config, store, monkeypatch
):
    from songe import backends
    from songe import platform as platforms
    from songe.sleep import calibrate, measurement
    from tests.test_learning import Tokenizer

    state = {"rss": 0, "compressed": GB, "time": 0.0, "waits": 0}

    def sample():
        state["compressed"] += 10**6
        return Memory(24 * GB, 10 * GB - state["rss"], compressed=state["compressed"])

    p = platform(None)
    p.memory = sample
    p.process_bytes = lambda pid: state["rss"]
    p.on_ac_power = lambda: True
    p.idle_seconds = p.thermal = lambda: None
    monkeypatch.setattr(platforms, "current", lambda: p)
    monkeypatch.setattr(memory.time, "monotonic", lambda: state["time"])
    monkeypatch.setattr(
        memory.time, "sleep", lambda seconds: state.update(time=state["time"] + seconds)
    )
    monkeypatch.setattr(memory, "process_tree_bytes", lambda p, pid: state["rss"])
    monkeypatch.setattr(memory, "process_tree_swap_bytes", lambda *args: None)
    monkeypatch.setattr(measurement, "process_tree_bytes", lambda p, pid: state["rss"])

    class Backend:
        def __init__(self, config, check):
            self.config, self.tokenizer = config, Tokenizer()
            state["rss"] = 2_223_449_088
            state["time"] += 0.01

        def prepare(self):
            pass

        def reset_peak_memory(self):
            pass

        def peak_memory(self):
            return state["rss"]

        def compare(self, *args):
            pass

        def train(self, *args):
            pass

        def sample(self, *args, **kwargs):
            return [1], "read", SimpleNamespace(generation_tps=50, prompt_tps=100)

    monkeypatch.setattr(backends, "choose", lambda *args: Backend)
    config.artifact_kind = "mlx-adapter"
    config.response_tokens = config.token_chunk = config.accumulation = 1
    process = SimpleNamespace(pid=12345, returncode=None)
    alarms = []

    class Stop:
        def is_set(self):
            return False

        def wait(self, interval):
            state["time"] += interval
            state["waits"] += 1
            if state["waits"] == 1:
                calibrate._calibrate(type(config).load(store.root), store.root, ())
            else:
                process.returncode = 0

    def guarded(argv, **kwargs):
        baseline = memory.capture_baseline(p, on_sample=kwargs["on_sample"])
        memory.guard(
            process,
            p,
            alarms.append,
            alarms.append,
            baseline=baseline,
            on_sample=kwargs["on_sample"],
            stop=Stop(),
            policy=memory.GuardPolicy(0.1, 0.1, 3),
        )
        assert not alarms
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(memory, "guarded_run", guarded)
    rows = calibrate.calibrate(config, store.root, ())
    assert len(rows) == 1
    assert (
        rows[0]["response_tokens"]
        == rows[0]["token_chunk"]
        == rows[0]["accumulation"]
        == 1
    )
    assert config.calibration_available
    assert sample().available == 7_776_550_912


@pytest.mark.parametrize("calibration", [False, True])
async def test_scheduler_persists_stop_after_final_child_write(
    config, store, calibration
):
    from songe.schedule import Scheduler

    config.save(store.root)
    night_id = store.night(None)
    scheduler = Scheduler(config, store, None)
    scheduler.calibrating = calibration
    scheduler.memory_nights_before = store.nights()
    sample = memory.stopped_sample(
        memory.snapshot(platform(Memory(24 * GB, 7 * GB))), "prediction_exceeded"
    )
    scheduler.memory_stop_sample = sample

    async def wait():
        store.night(night_id, status="ready", record={"final": "checkpoint"})

    scheduler.process = SimpleNamespace(wait=wait)
    await scheduler.finish()
    saved = store.get("calibration") if calibration else store.nights()[-1]["record"]
    assert saved["memory_stop"]["stop_reason"]["rule"] == "prediction_exceeded"
    assert scheduler.memory_stop_sample is None


def test_admission_stop_preserves_previous_night(config, store, monkeypatch):
    from songe.sleep.worker import isolated

    store.night(None, status="failed", record={"reason": "earlier failure"})
    before = store.nights()

    def guarded(*args, **kwargs):
        memory.admit(platform(Memory(24 * GB, GB)), 2 * GB)

    monkeypatch.setattr(memory, "guarded_run", guarded)
    with pytest.raises(memory.MemoryPressure):
        isolated(config, store)
    assert store.nights() == before
    assert (
        store.get("last_memory_stop")["stop_reason"]["rule"] == "admission_prediction"
    )


async def test_scheduler_records_pressure_change_during_baseline(
    config, store, monkeypatch
):
    import asyncio

    from songe import schedule

    config.calibration_available = True
    config.response_tokens = 1
    store.set(
        "calibration",
        {"rows": [dict(sequence=64, host_peak_bytes=GB, device_peak_bytes=0)]},
    )
    samples = iter([Memory(24 * GB, 8 * GB), Memory(24 * GB, 8 * GB, "critical")])
    p = platform(None)
    p.memory = lambda: next(samples)
    monkeypatch.setattr(
        schedule, "probe", lambda _: schedule.Probes(True, 1800, 0, 24, 8, True)
    )
    scheduler = schedule.Scheduler(config, store, None, p)
    scheduler.last_traffic -= 2000
    monkeypatch.setattr(scheduler, "enough", lambda: asyncio.sleep(0, result=True))
    with pytest.raises(memory.MemoryPressure):
        await scheduler.tick()
    assert scheduler.process is None
    assert store.get("last_memory_stop")["stop_reason"]["rule"] == "admission_pressure"


def test_measuring_what_fits_is_not_stopped_by_a_projected_burst(monkeypatch):
    """Live (24 GB Mac, 8 GB free): loading Granite 3B to measure, the rate
    projection stopped every measurement, so training could never be sized.
    While measuring, only real limits stop it (reserve margin, critical
    pressure, escalation, paging). The same samples stop with "allocation_rate"
    when projection is on (test_unmeasured_sustained_depletion_still_stops)."""
    baseline = Memory(24 * GB, 8 * GB)
    depletion = [replace(baseline, available=int(n * GB)) for n in (7, 5, 0.2)]
    events = run_guard(monkeypatch, depletion, baseline=baseline, rss=[GB, 3 * GB, 6 * GB],
                       events=True, projection=False)
    assert events == []
    on = run_guard(monkeypatch, depletion, baseline=baseline, rss=[GB, 3 * GB, 6 * GB], events=True)
    assert on and on[0][1].stop_reason["rule"] == "allocation_rate"
    # Real exhaustion below the reserve still stops it.
    exhausted = [replace(baseline, available=int(n * GB)) for n in (5, 2, 1)]
    stopped = run_guard(monkeypatch, exhausted, baseline=baseline, rss=[3 * GB, 6 * GB, 7 * GB],
                        margin=int(2.4 * GB), events=True, projection=False)
    assert stopped and stopped[0][1].stop_reason["rule"] == "memory_margin"


def unified_gpu(sample):
    # GB10 / Jetson: NVML has no device memory; it is the system's.
    return [Accel("cuda", "NVIDIA GB10", sample.total, sample.available, "GPU-1")]


def discrete_gpu(sample):
    return [Accel("cuda", "dGPU", 24 * GB, 20 * GB, "GPU-1")]


def gb10_gate_slice(monkeypatch, accelerators, available):
    """A live GB10 gate slice: the worker's ~51 GB of CUDA memory lowers
    `available` but its tree RSS stays ~3 GB (calibration measured its peak,
    24.2 GB, as the fall in free memory)."""
    total = 128 * GB
    start = memory.Snapshot(
        total, 74 * GB, "normal", tuple(accelerators(Memory(total, 74 * GB)))
    )
    return run_guard(
        monkeypatch,
        [Memory(total, int(available))],
        baseline=start,
        rss=[3 * GB],
        accelerators=accelerators,
        predicted_peak_bytes=int(24.2 * GB),
        predicted_device_bytes=int(72.6 * GB),
        device_prediction_error_bytes=int(3.6 * GB),
        margin=int(12.8 * GB),
        events=True,
    )


def test_unified_memory_does_not_reserve_device_allocations_twice(monkeypatch):
    # RSS alone: 23 - (24.2 - 3) = 1.8 GB < 12.8 GB margin, a false stop.
    assert gb10_gate_slice(monkeypatch, unified_gpu, 23 * GB) == []


def test_unified_memory_still_stops_below_the_margin(monkeypatch):
    events = gb10_gate_slice(monkeypatch, unified_gpu, 12 * GB)
    assert events and events[0][1].stop_reason["rule"] == "memory_margin"


def test_discrete_gpu_keeps_counting_the_process_tree(monkeypatch):
    # A discrete GPU's allocations never lower `available`; unchanged rule.
    events = gb10_gate_slice(monkeypatch, discrete_gpu, 23 * GB)
    assert events and events[0][1].stop_reason["rule"] == "predicted_margin"
