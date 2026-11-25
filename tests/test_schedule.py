from dataclasses import replace

import pytest

from songe.schedule import Probes, Scheduler, reason
from songe.sleep.calibrate import derive


def good():
    return Probes(True, 1800, 0, 24, 16, True)


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("thermal", 2, "cool"),
        ("free_gb", 1, "memory"),
        ("pressure", "critical", "memory"),
        ("available", False, "probes"),
    ],
)
def test_hardware_conditions(config, field, value, expected):
    config.calibration_available = True
    config.calibrated_peak_gb = 2
    assert expected in reason(
        config, replace(good(), **{field: value}), enough=True, traffic_idle=1800
    )


def test_keyboard_activity_does_not_block_learning(config):
    # Product rule: learning waits on proxy traffic and memory, not on whether
    # someone is typing elsewhere on the machine.
    config.calibration_available = True
    probes = replace(good(), idle_seconds=1)
    assert (
        reason(config, probes, enough=True, traffic_idle=1800, now=100000)
        == "Will sleep when idle"
    )


def test_all_scheduler_conditions(config):
    config.calibration_available = True
    args = dict(enough=True, traffic_idle=1800, now=100000)
    assert reason(config, good(), **args) == "Will sleep when idle"
    assert reason(config, good(), **(args | {"paused": True})) == "Paused"
    assert "experience" in reason(config, good(), **(args | {"enough": False}))
    assert "traffic" in reason(config, good(), **(args | {"traffic_idle": 1}))
    assert (
        reason(config, good(), **(args | {"last_completed": 99000}))
        == "Will sleep when idle"
    )


def test_traffic_signals_child_without_wait(config, store, monkeypatch):
    from songe import schedule

    monkeypatch.setattr(
        schedule, "interrupt_group", lambda child: child.send_signal("interrupt group")
    )

    class Child:
        returncode = None
        signals = []

        def send_signal(self, s):
            self.signals.append(s)

    s = Scheduler(config, store, None)
    s.process = Child()
    s.traffic()
    assert len(s.process.signals) == 1
    s.traffic()
    assert len(s.process.signals) == 1


def test_calibration_uses_generation_memory_and_training():
    rows = [
        {
            "sequence": 256,
            "step_seconds": 1,
            "generation_tps": 50,
            "forward_seconds": 0.1,
            "peak_memory_gb": 2,
        },
        {
            "sequence": 2048,
            "step_seconds": 4,
            "generation_tps": 10,
            "forward_seconds": 0.5,
            "peak_memory_gb": 15,
        },
    ]
    result = derive(rows, 10)
    assert result["max_sequence"] == 256
    assert result["calibrated_step_seconds"] == 4
    assert result["calibrated_score_seconds"] == pytest.approx(2.66)
    with pytest.raises(ValueError):
        derive(rows, 1)


async def test_proxy_restart_recovers_incomplete_night(config, store):
    class Upstream:
        async def close(self):
            pass

    store.night(None)
    assert store.nights()[0]["status"] == "running"
    scheduler = Scheduler(config, store, Upstream())
    await scheduler.start()
    await scheduler.close()
    assert store.nights()[0]["status"] == "interrupted"


async def test_pending_adapter_export_waits_for_disk_and_skips_training_sizing(
    config, store, monkeypatch
):
    import asyncio
    from types import SimpleNamespace

    from songe import schedule
    from songe.artifacts import storage
    from songe.config import training_fingerprint
    from songe.memory import Snapshot
    from songe.sleep import sizing

    config.upstream = "llamacpp"
    config.calibrated_peak_gb = 90
    config.calibration_available = True
    store.set("growth_requested", True)
    night = store.night(None)
    work = store.root / "versions" / f"night-{night}"
    adapter = work / "scale-0.5-adapter"
    adapter.mkdir(parents=True)
    weights = b"adapter weights" * 128
    (adapter / "adapters.safetensors").write_bytes(weights)
    artifact = work / "scale-0.5-artifact.gguf"
    store.night(
        night,
        status="interrupted",
        phase="fuse",
        record={
            "configuration": training_fingerprint(config),
            "base_fingerprint": schedule.base_fingerprint(config),
            "generation": store.get("candidate_generation"),
            "pending_export": {
                "kind": "gguf-lora",
                "training_adapter": str(adapter),
                "artifact": str(artifact),
            },
        },
    )
    sample = Snapshot(24 * 10**9, 4 * 10**9)
    platform = SimpleNamespace(name="darwin", keep_awake=lambda pid: None)
    parked, launched = [], []

    async def park(on_snapshot):
        parked.append(True)
        on_snapshot([])

    scheduler = Scheduler(config, store, SimpleNamespace(park=park), platform)
    scheduler.last_traffic -= 3600
    monkeypatch.setattr(schedule, "probe", lambda platform: replace(good(), free_gb=4))
    monkeypatch.setattr(schedule, "snapshot", lambda platform: sample)
    monkeypatch.setattr(
        sizing, "choose_size", lambda *args: pytest.fail("sized a training worker")
    )
    monkeypatch.setattr(
        storage,
        "phase_requirements",
        lambda *args, **kwargs: pytest.fail("sized training storage"),
    )
    reserve = 10**9
    usage = SimpleNamespace(free=reserve + len(weights), total=100 * reserve)
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda path: usage)

    def baseline(*args, required_bytes, required_device_bytes, **kwargs):
        assert (required_bytes, required_device_bytes) == (0, 0)
        return sample

    monkeypatch.setattr(schedule, "capture_baseline", baseline)
    monkeypatch.setattr(schedule, "guard", lambda *args, **kwargs: None)

    async def spawn(*argv, **kwargs):
        launched.append(argv)
        assert "songe.sleep.worker" in argv
        assert "--export-only" in argv
        assert "--sizing" not in argv
        return SimpleNamespace(pid=1, returncode=None)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    await scheduler.tick()
    wait = store.get("disk_wait")
    assert wait and scheduler.reason == wait
    assert "Needs 1.0 GB free" in wait
    assert not parked and not launched
    assert store.nights()[0]["status"] == "interrupted"
    assert not artifact.exists()

    usage.free = reserve + 3 * len(weights)
    await scheduler.tick()
    assert scheduler.process
    assert len(parked) == len(launched) == 1
    assert store.get("disk_wait") is None
    await scheduler.guard_task


async def test_ready_swap_waits_for_inflight_and_observed_duration(
    config, store, monkeypatch
):
    import time

    from songe.sleep import swap
    from tests.test_store import version

    calls = []

    async def install(*args, **kw):
        calls.append(True)
        return True

    monkeypatch.setattr(swap, "install", install)
    store.add_version(version("base"))
    store.swap("base", None)
    record = {"candidate": version("candidate", 1), "expected": "base", "watermark": 1}
    night = store.night(None, status="ready", record=record)
    scheduler = Scheduler(config, store, None)
    scheduler.acquire("base")
    await scheduler.accept(store.nights()[0])
    assert not calls
    scheduler.release("base")
    scheduler.last_traffic = time.monotonic() - 1
    store.set("swap_durations", [10])
    await scheduler.accept(store.nights()[0])
    assert not calls and store.nights()[0]["id"] == night


async def test_wake_stops_worker_then_restores_parked_user_models(
    config, store, monkeypatch
):
    from songe import schedule

    monkeypatch.setattr(
        schedule, "interrupt_group", lambda child: child.send_signal("interrupt group")
    )
    from tests.test_store import version

    events = []

    class Upstream:
        async def restore(self, parked, managed):
            events.append(("restore", parked, managed))

        async def ensure(self, version):
            events.append(("ensure", version["id"]))

    class Child:
        returncode = None

        def send_signal(self, signal):
            events.append(("signal",))

        async def wait(self):
            self.returncode = 0
            events.append(("wait",))

    store.add_version(version("base"))
    store.swap("base", None)
    store.set("parked_models", [{"identifier": "user", "key": "key"}])
    scheduler = Scheduler(config, store, Upstream())
    scheduler.process = Child()
    scheduler.traffic()
    await scheduler.wake()
    assert events[0] == ("signal",)
    assert events[-2][0] == "restore" and events[-2][2] == {"base"}
    assert events[-1] == ("ensure", "base")
    assert store.get("parked_models") == []
    assert scheduler.process is None


def test_sleep_now_skips_only_the_idle_wait(config):
    config.calibration_available = True
    busy = dict(enough=True, traffic_idle=5, threshold=600)
    assert "without proxy traffic" in reason(config, good(), **busy)
    assert reason(config, good(), **busy, immediate=True) == "Will sleep when idle"
    hot = replace(good(), thermal=2)
    assert "cool" in reason(config, hot, **busy, immediate=True)


def ready_night(config, store):
    from songe.config import training_fingerprint
    from songe.schedule import base_fingerprint
    from tests.test_store import version

    store.add_version(version("base"))
    store.swap("base", None)
    record = {
        "candidate": version("candidate", 1),
        "expected": "base",
        "watermark": 1,
        "configuration": training_fingerprint(config),
        "base_fingerprint": base_fingerprint(config),
        "environment": store.get("environment_fingerprint"),
        "generation": store.get("candidate_generation"),
    }
    store.night(None, status="ready", record=record)


async def test_update_waits_for_memory_and_is_never_thrown_away(
    config, store, monkeypatch
):
    import time

    from songe.memory import MemoryPressure
    from songe.platform import Memory
    from songe.sleep import swap

    calls = []

    async def install(*args, **kw):
        calls.append(True)
        raise MemoryPressure(Memory(1000, 10))

    monkeypatch.setattr(swap, "install", install)
    ready_night(config, store)
    scheduler = Scheduler(config, store, None)
    scheduler.last_traffic = time.monotonic() - 1000
    for _ in range(5):
        await scheduler.accept(store.nights()[0])
        night = store.nights()[0]
        # Retry later, not on the next tick; open the wait to try again.
        await scheduler.accept(night)
        store.night(night["id"], record=night["record"] | {"swap_retry_at": 0})
    assert len(calls) == 5
    night = store.nights()[0]
    assert night["status"] == "ready"
    assert "swap_failures" not in night["record"]


async def test_update_the_server_refuses_is_rejected_after_three_spaced_tries(
    config, store, monkeypatch
):
    import time

    from songe.sleep import swap

    calls = []

    async def install(*args, **kw):
        calls.append(True)
        raise RuntimeError("bad weights")

    monkeypatch.setattr(swap, "install", install)
    ready_night(config, store)
    scheduler = Scheduler(config, store, None)
    scheduler.last_traffic = time.monotonic() - 1000
    await scheduler.accept(store.nights()[0])
    await scheduler.accept(store.nights()[0])
    assert len(calls) == 1 and store.nights()[0]["status"] == "ready"
    for _ in range(2):
        night = store.nights()[0]
        store.night(night["id"], record=night["record"] | {"swap_retry_at": 0})
        await scheduler.accept(store.nights()[0])
    assert len(calls) == 3 and store.nights()[0]["status"] == "rejected"


async def test_a_dream_the_server_never_loaded_practised_nothing(config, store, monkeypatch):
    """An update refused three times is rejected. Under Dreams the watermark
    counts practice scenes: moved past them, the scenes of a Dream nobody
    was ever served counted as practised."""
    import time

    from songe.practice import pool
    from songe.sleep import swap

    async def install(*args, **kw):
        raise RuntimeError("bad weights")

    monkeypatch.setattr(swap, "install", install)
    store.set("learning_mode", "dreams")
    ready_night(config, store)
    night = store.nights()[0]
    store.night(night["id"], record=night["record"] | {"watermark": 7, "practice_revisions": {"practice:a#v1": 7}})
    scheduler = Scheduler(config, store, None)
    scheduler.last_traffic = time.monotonic() - 1000
    for _ in range(3):
        night = store.nights()[0]
        store.night(night["id"], record=night["record"] | {"swap_retry_at": 0})
        await scheduler.accept(store.nights()[0])
    assert store.nights()[0]["status"] == "rejected"
    assert store.get("experience_watermark", 0) == 0 and store.get("practice_revisions", {}) == {}
    assert pool.rejected(store) == {"practice:a#v1": (7, 1)}


async def test_update_loads_alone_when_it_does_not_fit_beside_the_current_one(
    config, store, monkeypatch
):
    import time
    from types import SimpleNamespace

    from songe.memory import MemoryPressure
    from songe.platform import Memory
    from songe.sleep import swap

    calls, parked = [], []

    async def install(*args, **kw):
        calls.append(True)
        if len(calls) == 1:
            raise MemoryPressure(Memory(1000, 10))
        store.swap("candidate", "base")
        return True

    async def park(on_snapshot):
        parked.append(True)
        on_snapshot([{"identifier": "base"}])

    async def collect(*args, **kw):
        pass

    monkeypatch.setattr(swap, "install", install)
    monkeypatch.setattr(swap, "collect", collect)
    ready_night(config, store)
    from tests.test_store import version

    store.add_version(version("candidate", 1))
    scheduler = Scheduler(config, store, SimpleNamespace(park=park))
    scheduler.last_traffic = time.monotonic() - 1000
    await scheduler.accept(store.nights()[0])
    assert parked and len(calls) == 2
    assert store.nights()[0]["status"] == "accepted"
    assert store.get("parked_models") == []


def test_a_run_that_did_nothing_is_not_started_again_at_once(config, store):
    scheduler = Scheduler(config, store, None)
    scheduler.calibrating, scheduler.cancelled_at = False, None
    store.slice(store.slice(), units_done=0)
    scheduler.note_progress()
    first = scheduler.empty_run_wait()
    assert 0 < first <= 300
    store.slice(store.slice(), units_done=0)
    scheduler.note_progress()
    assert scheduler.empty_run_wait() > first
    # New conversations, or a run that did work, clear the wait.
    store.set("experience_revision", "new")
    assert scheduler.empty_run_wait() == 0
    store.slice(store.slice(), units_done=0)
    scheduler.note_progress()
    assert scheduler.empty_run_wait() > 0
    store.slice(store.slice(), units_done=3)
    scheduler.note_progress()
    assert scheduler.empty_run_wait() == 0
    # Being interrupted by a request is not a run that did nothing.
    scheduler.cancelled_at = 1.0
    store.slice(store.slice(), units_done=0)
    scheduler.note_progress()
    assert scheduler.empty_run_wait() == 0


def test_service_restarts_itself_onto_upgraded_code(config, store, monkeypatch):
    from songe import upgrade

    scheduler = Scheduler(config, store, None)
    stopped = []
    scheduler.shutdown = lambda: stopped.append(True)
    scheduler.UPGRADE_CHECK_SECONDS = 0
    assert not scheduler.upgraded()
    monkeypatch.setattr(upgrade, "install_token", lambda: "new files")
    # Seen once: the install may still be copying.
    assert not scheduler.upgraded()
    scheduler.acquire("m")
    assert not scheduler.upgraded()  # never under an agent request
    scheduler.release("m")
    assert not scheduler.upgraded()  # nor between an agent's tool calls
    scheduler.last_traffic -= 1000
    assert scheduler.upgraded()
