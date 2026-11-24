"""Busy 24 GB Mac regression: no accelerator imports or real model work."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from songe import schedule
from songe.backends import Interrupted
from songe.config import Config
from songe.memory import Snapshot
from songe.platform import Memory, Power
from songe.sleep import calibrate, worker
from songe.sleep.calibration_state import (
    coverage,
    measure_pool,
    needs_growth,
    pool_key,
    progress,
    sample_pool,
)
from songe.sleep.samples import SampleAdmission
from songe.sleep.sizing import choose_size, cost_model, dimensions
from songe.status import activity, work_text

GB = 10**9


def original_rows():
    return [
        dict(
            sequence=44,
            response_tokens=1,
            token_chunk=1,
            accumulation=b,
            host_peak_bytes=int(7.47 * GB),
            device_peak_bytes=0,
            step_seconds=seconds,
            generation_tps=30,
            forward_seconds=0.1,
        )
        for b, seconds in ((1, 1.66), (4, 6.92))
    ]


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode("utf-8"))

    def decode(self, tokens):
        return bytes(tokens).decode("utf-8", errors="ignore")

    def apply_chat_template(self, messages, **kwargs):
        # A compact template with exactly the reported minimum of 44 tokens.
        return [1] * (38 + len(messages[0]["content"]))


def step(tokens=900, index=1):
    return dict(
        id=f"sample-{index}",
        source_id=index,
        split="train",
        conversation=f"conversation-{index}",
        tools=[],
        context=[{"role": "user", "content": "x" * (tokens - 166)}],
        hindsight=[{"role": "user", "content": "correction"}],
    )


@pytest.fixture
def measured(config, store, monkeypatch):
    from songe import backends, platform
    from songe.sleep import measurement

    config.artifact_kind = "mlx-adapter"
    config.save(store.root)
    store.set("calibration", {"rows": original_rows()})
    store.sync_steps([step()])
    machine = SimpleNamespace(
        name="test",
        memory=lambda: Memory(24 * GB, 20 * GB),
        accelerators=lambda: [],
        process_bytes=lambda pid: 0,
        on_ac_power=lambda: True,
        idle_seconds=lambda: 2000,
        thermal=lambda: None,
        keep_awake=lambda pid: None,
    )
    monkeypatch.setattr(platform, "current", lambda: machine)
    monkeypatch.setattr(measurement, "process_tree_bytes", lambda *a: 0)

    class Backend:
        calls = []
        stop_at = None
        capacity = None
        chunk_capacity = None
        current_size = 44
        current_chunk = 1

        def __init__(self, config, check):
            self.config, self.tokenizer = config, Tokenizer()

        def prepare(self):
            pass

        def reset_peak_memory(self):
            type(self).current_size = 44
            type(self).current_chunk = 1

        def peak_memory(self):
            if self.current_size == 10**9:
                return 25 * GB
            if self.capacity and self.current_size > self.capacity:
                return 25 * GB
            if self.chunk_capacity and self.current_chunk > self.chunk_capacity:
                return 25 * GB
            return int(7.47 * GB)

        def sample(self, *args, **kwargs):
            return (
                [1] * self.config.response_tokens,
                "read",
                SimpleNamespace(generation_tps=30, prompt_tps=100),
            )

        def compare(self, *args):
            pass

        teachers = []
        teacher_capacity = None

        def teacher_hidden(self, prompt, y):
            length = len(prompt) + len(y)
            self.teachers.append(length)
            if self.teacher_capacity and length > self.teacher_capacity:
                type(self).current_size = 10**9

        def train(self, *args):
            type(self).current_size = self.config.max_sequence
            type(self).current_chunk = self.config.token_chunk
            self.calls.append(
                (
                    self.config.max_sequence,
                    self.config.response_tokens,
                    self.config.token_chunk,
                    self.config.accumulation,
                )
            )
            if len(self.calls) == self.stop_at:
                raise Interrupted("traffic resumed")

    monkeypatch.setattr(backends, "choose", lambda *args: Backend)
    return Backend, machine


def test_exact_small_rows_resume_across_idle_gaps(config, store, measured):
    backend, _ = measured
    backend.stop_at = 2
    with pytest.raises(Interrupted, match="traffic"):
        calibrate._calibrate(config, store.root, (256, 512, 1024, 2048))
    saved = store.get("calibration")
    assert saved["rows"][:2] == original_rows()
    assert saved["progress"]["target_sequence"] == 900
    assert not saved["progress"]["complete"]
    assert needs_growth(config, store)
    # Long teachers have their own ladder; old geometries are not redone.
    assert backend.calls[0][0] != 44
    assert "teacher" in saved["rows"][-1]["phase_peaks"]
    completed = {dimensions(r) for r in saved["rows"] if r.get("teacher_sequence")}
    backend.calls.clear()
    backend.stop_at = None
    calibrate._calibrate(Config.load(store.root), store.root, (256, 512, 1024, 2048))
    assert not completed.intersection(backend.calls)
    assert not needs_growth(config, store)
    assert store.get("calibration")["progress"]["completion_reason"] == "coverage"
    assert (900, 128, 32, 4) in backend.calls
    # Only the real training geometry is measured with accumulation.
    batched = [c[0] for c in backend.calls if c[3] > 1]
    assert len(batched) == len(set(batched))  # one batched row per length
    assert max(r["sequence"] for r in store.get("calibration")["rows"]) == 900
    saved_config = Config.load(store.root)
    assert (
        saved_config.max_sequence,
        saved_config.response_tokens,
        saved_config.token_chunk,
    ) == (2048, 128, 32)
    assert saved_config.calibration_available
    assert not store.nights()


def test_partial_measurement_cannot_mark_long_samples_never_fit(
    config, store, measured
):
    admission = SampleAdmission(
        config,
        {"max_sequence": 44, "response_tokens": 1},
        store.get("calibration"),
        store,
        store.slice(),
    )
    assert admission.never_fit(900) is None
    with pytest.raises(Interrupted, match="deferred"):
        admission.admit([step()], Tokenizer())
    assert not store.get("unfit_samples")
    assert store.unscored([step()])


def test_one_memory_geometry_does_not_fit_a_model_or_extrapolate(config):
    rows = original_rows()
    assert cost_model(rows, "host_peak_bytes") is None
    sample = Snapshot(24 * GB, 20 * GB)
    assert choose_size(config, {"rows": rows}, sample, {}) is None
    exact = choose_size(replace(config, response_tokens=1), {"rows": rows}, sample, {})
    assert exact["max_sequence"] == 44 and exact["cost_model"] is None
    assert exact["host_bytes"] == int(7.47 * GB)
    rows.append(rows[0] | {"sequence": 256})
    assert cost_model(rows, "host_peak_bytes") is not None


def test_exact_fallback_uses_error_from_its_own_geometry(config):
    config = replace(config, max_sequence=512, response_tokens=32, token_chunk=1)
    small = original_rows()[0] | {"sequence": 64, "host_peak_bytes": 1000}
    large = small | {"sequence": 300, "response_tokens": 32, "host_peak_bytes": 4000}
    rows = [small, large, large | {"accumulation": 4, "host_peak_bytes": 4100}]
    saved = {
        "rows": rows,
        "capacity_limits": [large | {"sequence": 301, "host_peak_bytes": 30000}],
    }
    chosen = choose_size(config, saved, Snapshot(24000, 10000), {})
    # The measured geometry is used directly, with its own spread as error.
    assert chosen["max_sequence"] == 300
    assert chosen["host_bytes"] == 4100
    # Its own spread, never below 5% of the prediction (run-to-run noise).
    assert chosen["prediction_error_bytes"] == max(100, chosen["host_bytes"] // 20)


def test_one_gpu_measurement_still_tolerates_run_to_run_noise(config):
    """Live run (A100): the single measured geometry gave the GPU a zero
    error, so 46 MB over a 15.94 GB prediction stopped every slice."""
    from songe.platform import Accel

    config = replace(config, max_sequence=512, response_tokens=32, token_chunk=1)
    row = original_rows()[0] | {
        "sequence": 300,
        "response_tokens": 32,
        "host_peak_bytes": 4000,
        "device_peak_bytes": 16000,
    }
    gpu = (Accel("cuda", "gpu", 80000, 60000),)
    chosen = choose_size(config, {"rows": [row]}, Snapshot(24000, 10000, vram=gpu), {})
    assert chosen["device_bytes"] == 16000
    assert chosen["device_prediction_error_bytes"] == 16000 // 20


def test_an_admitted_long_teacher_raises_the_guard_prediction(config):
    """Live run: the slice admitted 8748-token teachers but the guard expected
    only the 2048-token student peak, so every slice stopped as 'exceeded'."""
    config = replace(config, max_sequence=512, response_tokens=32, token_chunk=1)
    small = original_rows()[0] | {"sequence": 64, "host_peak_bytes": 1000}
    large = small | {"sequence": 300, "response_tokens": 32, "host_peak_bytes": 4000}
    long_teacher = {"teacher_sequence": 9000, "host_peak_bytes": 7000}
    saved = {
        "rows": [small, large | {"teacher_sequence": 300}],
        "teacher_rows": [long_teacher],
        "capacity_limits": [large | {"sequence": 301, "host_peak_bytes": 30000}],
    }
    chosen = choose_size(config, saved, Snapshot(24000, 10000), {})
    assert chosen["max_teacher_sequence"] == 9000
    assert chosen["host_bytes"] == 7000


def test_completion_requires_joint_coverage_of_p90_bounded_by_policy(config, store):
    samples = [step(n, i) for i, n in enumerate([700] * 8 + [900, 5000], 1)]
    store.sync_steps(samples)
    measured, target, key = measure_pool(config, store, Tokenizer())
    assert target == 900
    rows = original_rows() + [original_rows()[0] | {"sequence": 900}]
    saved = {"rows": rows, "samples": measured}
    assert not progress(config, saved, target, key)["complete"]
    rows.append(rows[0] | {"sequence": 700, "response_tokens": 128, "token_chunk": 32})
    assert not coverage(config, rows, target)
    rows.append(rows[-1] | {"sequence": 900})
    saved["progress"] = progress(config, saved, target, key)
    assert saved["progress"]["complete"]
    store.set("calibration", saved)
    assert not needs_growth(config, store)
    store.sync_steps([step(1000, 11)])
    assert needs_growth(config, store)
    assert measure_pool(replace(config, max_sequence=800), store, Tokenizer())[1] == 800


def test_absolute_capacity_stop_refines_boundary_and_marks_never_fit(
    config, store, measured
):
    backend, _ = measured
    backend.capacity = 300
    calibrate._calibrate(config, store.root, (256, 512, 1024, 2048))
    saved = store.get("calibration")
    assert saved["progress"]["complete"]
    assert saved["progress"]["completion_reason"] == "memory"
    # Bisection stops within a small tolerance below the true edge (300).
    ceiling = saved["progress"]["memory_ceiling"]
    # Bisection stops within max(64, a tenth) of the edge.
    assert 300 - 64 <= ceiling <= 300
    assert min(r["sequence"] for r in saved["capacity_limits"]) - ceiling <= 64
    assert not needs_growth(config, store)
    unfit = list(store.get("unfit_samples").values())
    assert len(unfit) == 1 and unfit[0]["required_sequence"] == 900
    assert "memory capacity" in unfit[0]["reason"]
    assert not store.unscored([step()])
    assert not store.nights()
    chosen = choose_size(config, saved, Snapshot(24 * GB, 20 * GB), {})
    assert chosen["max_sequence"] == ceiling


def test_transient_memory_stop_remains_partial(config, store, measured):
    from songe.memory import record_memory_stop

    saved = store.get("calibration")
    saved["progress"] = progress(config, saved, 900, pool_key(sample_pool(store)))
    store.set("calibration", saved)
    record_memory_stop(store, Snapshot(24 * GB, 2 * GB, "critical"), calibration=True)
    assert needs_growth(config, store)
    assert store.get("calibration")["progress"]["memory_ceiling"] is None
    assert not store.get("unfit_samples")


def test_guard_capacity_evidence_survives_child_exit(config, store, measured):
    from songe.memory import record_memory_stop, stopped_sample

    saved = store.get("calibration")
    saved["rows"].append(
        original_rows()[0]
        | {
            "sequence": 300,
            "response_tokens": 128,
            "token_chunk": 1,
        }
    )
    saved["samples"] = [{"id": step()["id"], "revision": 1, "required_sequence": 900}]
    saved["progress"] = progress(config, saved, 900, pool_key(sample_pool(store)))
    saved["in_progress"] = dict(
        sequence=301, response_tokens=128, token_chunk=1, accumulation=1
    )
    store.set("calibration", saved)
    stop = stopped_sample(
        Snapshot(24 * GB, GB, "critical"), "critical_pressure", tree_rss_bytes=23 * GB
    )
    record_memory_stop(store, stop, calibration=True)
    saved = store.get("calibration")
    assert saved["progress"]["memory_ceiling"] == 300
    assert saved["progress"]["complete"]
    assert len(store.get("unfit_samples")) == 1


def test_chunk_capacity_does_not_prevent_intact_samples_from_training(
    config, store, measured
):
    backend, _ = measured
    backend.chunk_capacity = 1
    calibrate._calibrate(config, store.root, (256, 512, 1024, 2048))
    saved = store.get("calibration")
    assert saved["progress"]["complete"]
    assert saved["progress"]["memory_ceiling"] is None
    assert not store.get("unfit_samples")
    size = choose_size(config, saved, Snapshot(24 * GB, 20 * GB), {})
    assert size["max_sequence"] == 900 and size["token_chunk"] == 1


def test_p90_majority_can_fit_without_truncating_or_consuming_tail(config, store):
    from copy import deepcopy

    short, tail = step(700, 1), step(1000, 2)
    store.sync_steps([short, tail])
    before = deepcopy([short, tail])
    saved = {"rows": original_rows()}
    admission = SampleAdmission(
        config, {"max_sequence": 900}, saved, store, store.slice()
    )
    # The tail trains through its last 900 tokens (gradient window); the
    # stored example itself is never truncated.
    assert admission.admit([short, tail], Tokenizer()) == [short, tail]
    assert [short, tail] == before
    assert not admission.record["deferred_samples"]
    assert store.unscored([tail]) == [tail]


def test_target_grows_to_pending_tail_after_majority_is_learned(config, store):
    steps = [step(700, i) for i in range(1, 10)] + [step(1000, 10)]
    store.sync_steps(steps)
    samples, target, key = measure_pool(config, store, Tokenizer())
    assert target == 700
    rows = [
        original_rows()[0]
        | {"sequence": 700, "response_tokens": 128, "token_chunk": 32}
    ]
    saved = {"rows": rows, "samples": samples}
    saved["progress"] = progress(config, saved, target, key)
    store.set("calibration", saved)
    assert not needs_growth(config, store)
    store.mark_scored(steps[:9])
    assert needs_growth(config, store)
    assert measure_pool(config, store, Tokenizer())[1] == 1000


def test_growth_answer_follows_a_write_in_the_same_file_time_tick(config, store):
    # Linux file times tick every few milliseconds; a commit right after a
    # call can leave them unchanged (#11). Pin them to show it.
    import os

    steps = [step(700, i) for i in range(1, 10)] + [step(1000, 10)]
    store.sync_steps(steps)
    samples, target, key = measure_pool(config, store, Tokenizer())
    rows = [
        original_rows()[0]
        | {"sequence": 700, "response_tokens": 128, "token_chunk": 32}
    ]
    saved = {"rows": rows, "samples": samples}
    saved["progress"] = progress(config, saved, target, key)
    store.set("calibration", saved)
    files = [store.path, store.path.with_name(store.path.name + "-wal")]
    assert not needs_growth(config, store)
    times = [os.stat(f) for f in files]
    store.mark_scored(steps[:9])
    for file, stat in zip(files, times, strict=True):
        os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert needs_growth(config, store)


@pytest.mark.parametrize("available", [False, True])
def test_public_learning_entry_runs_growth_instead_of_training(
    config, store, measured, monkeypatch, available
):
    config.calibration_available = available
    calls = []
    monkeypatch.setattr(calibrate, "calibrate", lambda *args: calls.append(args))
    result = worker.isolated(config, store)
    assert len(calls) == 1 and result["reason"] == "Measuring training size"
    assert not store.nights()


async def test_model_with_zero_pending_scores_still_resumes_growth(
    config, store, measured
):
    from songe.models import registry

    config.calibration_available = True
    registry.register(config, store, "base", legacy=True)
    registry.set_learning(store, "base", "all")
    registry.support(store, "base", "trainable")
    scoped = store.scoped("base")
    scoped.set("calibration", {"rows": original_rows()})
    scoped.mark_scored([step()])
    assert scoped.pending_learnable() == 0
    scheduler = schedule.ModelScheduler(config, store, SimpleNamespace())
    assert scheduler.choose_model() == "base"
    assert await scheduler.next_model() == "base"


async def test_scheduler_launches_growth_even_with_no_new_examples(
    config, store, measured, monkeypatch
):
    _, machine = measured
    config.calibration_available = True
    scheduler = schedule.Scheduler(config, store, SimpleNamespace(), machine)
    scheduler.last_traffic -= 2000
    monkeypatch.setattr(scheduler, "enough", lambda: asyncio.sleep(0, result=False))
    launched = []

    async def spawn(*args, **kwargs):
        launched.append(args)
        return SimpleNamespace(pid=12345, returncode=0)

    monkeypatch.setattr(schedule.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(schedule, "guard", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        schedule, "capture_baseline", lambda *a, **kw: Snapshot(24 * GB, 20 * GB)
    )
    await scheduler.tick()
    assert "songe.sleep.calibrate" in launched[0]
    assert scheduler.calibrating and not store.nights()
    await scheduler.guard_task


@pytest.mark.parametrize("constraint", ["paused", "traffic", "battery", "memory"])
async def test_growth_obeys_idle_and_resource_rules(
    config, store, measured, monkeypatch, constraint
):
    _, machine = measured
    config.calibration_available = True
    if constraint == "memory":
        machine.memory = lambda: Memory(24 * GB, GB, "critical")
    if constraint == "battery":
        machine.power = lambda: Power(ac=False, fraction=0.02)
    scheduler = schedule.Scheduler(config, store, SimpleNamespace(), machine)
    scheduler.last_traffic -= 2000 if constraint != "traffic" else 0
    store.set("paused", constraint == "paused")
    launched = []

    async def spawn(*args, **kwargs):
        launched.append(args)

    monkeypatch.setattr(schedule.asyncio, "create_subprocess_exec", spawn)
    await scheduler.tick()
    assert not launched and not store.nights()


def test_child_does_not_open_an_empty_night(config, store, measured, monkeypatch):
    config.calibration_available = True
    monkeypatch.setattr(
        worker, "preview", lambda *a: ({"reason": "Ready"}, [step()], [])
    )
    assert worker.run(config, store)["reason"] == "Waiting for calibration growth"
    assert not store.nights()


def test_child_waits_before_a_night_when_current_memory_cannot_fit_any_sample(
    config, store, measured, monkeypatch
):
    config.calibration_available = True
    saved = store.get("calibration")
    saved["rows"].append(
        original_rows()[0]
        | {
            "sequence": 900,
            "response_tokens": 128,
            "token_chunk": 32,
        }
    )
    samples, target, key = measure_pool(config, store, Tokenizer())
    saved["samples"] = samples
    saved["progress"] = progress(config, saved, target, key)
    store.set("calibration", saved)
    monkeypatch.setattr(
        worker, "preview", lambda *a: ({"reason": "Ready"}, [step()], [])
    )
    # A short gradient window is fine; it is the teacher envelope (the full
    # context forward) that current memory cannot fit.
    result = worker.run(
        config, store, sizing={"max_sequence": 256, "max_teacher_sequence": 256}
    )
    assert "intact training samples deferred" in result["reason"]
    assert not store.nights()


def test_config_limit_completes_calibration_and_marks_oversized_samples(
    config, store, measured
):
    config.max_sequence = 800
    config.save(store.root)
    calibrate._calibrate(config, store.root, (256, 512, 1024, 2048))
    saved = store.get("calibration")
    assert saved["progress"]["complete"]
    assert saved["progress"]["target_sequence"] == 800
    # A sample longer than the limit trains through its last 800 tokens
    # (gradient window); its length never marks it unfit.
    assert not store.get("unfit_samples")
    assert not store.nights()


def test_existing_full_measurement_checks_unknown_lengths_without_training(
    config, store, measured
):
    backend, _ = measured
    config.calibration_available = True
    store.set(
        "calibration",
        {
            "rows": [
                original_rows()[0]
                | {
                    "sequence": 2048,
                    "response_tokens": 128,
                    "token_chunk": 32,
                }
            ]
        },
    )
    store.sync_steps([step(5000)])
    assert needs_growth(config, store)
    calibrate._calibrate(config, store.root, (256, 512, 1024, 2048))
    assert backend.calls and not store.nights()
    assert store.get("calibration")["progress"]["teacher_measured_sequence"] >= 5000
    assert not needs_growth(config, store)
    # Its 5000-token context is read without gradients; nothing is unfit.
    assert not store.get("unfit_samples")


def test_verdict_scope_does_not_change_with_pool_progress(config, store):
    saved = {"rows": original_rows(), "progress": {"pool": "before"}}
    before = SampleAdmission(config, None, saved, store, None)
    after = SampleAdmission(
        config, None, saved | {"progress": {"pool": "after"}}, store, None
    )
    assert before.key(step()) == after.key(step())


def test_status_explains_measurement_instead_of_data_shortage(config, store, measured):
    saved = store.get("calibration")
    saved["progress"] = progress(config, saved, 900, pool_key(sample_pool(store)))
    store.set("calibration", saved)
    data_progress = saved["progress"]
    text = "measuring training size · 44 of ~900 tokens"
    assert work_text({"phase": "calibrate", "data_progress": data_progress}) == text
    scheduler = SimpleNamespace(
        config=config, active_model=None, reason="Waiting for experience"
    )
    # A scoped model exercises the same path used by dashboard snapshots.
    from songe.models import registry

    registry.register(config, store, "base")
    registry.set_learning(store, "base", "all")
    registry.support(store, "base", "trainable")
    scoped = store.scoped("base")
    scoped.set("calibration", saved)
    result = activity(
        config, store, scheduler, None, registry.summaries(config, store), live=True
    )
    assert text in result["now"]
    assert "needs more data" not in result["next"]
    # Under Dreams measuring happens only inside a Deep Dream: with none
    # running, a leftover measurement is not current activity (live: shown
    # for six hours after the Dream ended).
    config.learning = "dreams"
    result = activity(
        config, store, scheduler, None, registry.summaries(config, store), live=True
    )
    assert "measuring" not in result["now"] and result["now"].startswith("Idle")


def stop(saved, probe, own_gb, start_gb=12.9, left_gb=3.5, rule="allocation_rate"):
    gb = 10**9
    saved["in_progress"] = probe
    saved["memory_stop"] = {
        "total": 25.8 * gb,
        "stop_reason": {
            "rule": rule,
            "baseline": {"available": start_gb * gb},
            "observed": {"available": left_gb * gb, "tree_rss_bytes": own_gb * gb},
        },
    }


PROBE = {"sequence": 4169, "response_tokens": 1, "token_chunk": 1, "accumulation": 1}


@pytest.mark.parametrize(
    "own_gb, phase, permanent, held",
    [
        (24.0, "train", True, False),  # more than the whole machine can give
        (11.4, "train", False, True),  # other apps held the rest: wait
        (11.4, "export", False, False),  # same cost at every size: just retry
        (2.0, "train", False, False),  # something else took the memory
    ],
)
def test_only_total_capacity_is_a_permanent_limit(
    config, store, measured, own_gb, phase, permanent, held
):
    calibrate._calibrate(config, store.root, (256,))
    saved = store.get("calibration")
    stop(saved, PROBE | {"kind": "sequence", "phase": phase}, own_gb)
    calibrate.record_capacity_stop(store, saved)
    saved = store.get("calibration")
    limits = saved.get("capacity_limits", [])
    assert any(r["sequence"] == 4169 for r in limits) is permanent
    assert any(h["sequence"] == 4169 for h in saved.get("held", [])) is held
    assert saved["progress"]["memory_ceiling"] is None or permanent


def test_long_teacher_waits_for_free_memory_and_retries(config, store, measured):
    """Last night: a 52,944-token teacher took 11.4 GB while other apps held
    11 GB. It is held, the ladder bisects below it, and it is tried again
    once enough memory is free."""
    from songe.sleep import calibration_state

    backend, machine = measured
    gb = 10**9
    original = calibration_state.measure_pool

    def long_teacher(*args, **kwargs):
        samples, target, pool = original(*args, **kwargs)
        return [s | {"teacher_sequence": 52944} for s in samples], target, pool

    calibration_state.measure_pool = long_teacher
    backend.teacher_capacity = 30000
    calibrate._calibrate(config, store.root, (256,))
    saved = store.get("calibration")
    assert (
        max(saved["teacher_rows"], key=lambda r: r["teacher_sequence"])[
            "teacher_sequence"
        ]
        <= 30000
    )
    # Beyond the whole machine is permanent; here it stands in for last night,
    # when the guard stopped the probe while other apps held memory instead.
    assert saved["teacher_limits"]
    saved["teacher_limits"] = []
    stop(
        saved, {"teacher_sequence": 52944, "kind": "teacher", "phase": "teacher"}, 11.4
    )
    calibrate.record_capacity_stop(store, saved)
    saved = store.get("calibration")
    assert not any(
        r["teacher_sequence"] == 52944 for r in saved.get("teacher_limits", [])
    )
    assert [h["teacher_sequence"] for h in saved["held"]] == [52944]
    # Training admits the long teachers that fit today's free memory only.
    busy = choose_size(config, saved, Snapshot(int(25.8 * gb), int(9 * gb)), {})
    free = choose_size(config, saved, Snapshot(int(25.8 * gb), int(20 * gb)), {})
    assert busy is None or busy["max_teacher_sequence"] <= free["max_teacher_sequence"]
    # With 20 GB free, the held size is dropped and measured again.
    backend.teacher_capacity = None
    backend.teachers.clear()
    saved["progress"]["completion_reason"] = "held"
    store.set("calibration", saved)
    calibrate._calibrate(Config.load(store.root), store.root, (256,))
    saved = store.get("calibration")
    assert not saved.get("held")
    assert max(r["teacher_sequence"] for r in saved["teacher_rows"]) == 52944
    calibration_state.measure_pool = original


def test_bisection_noise_does_not_inflate_measured_sizes(config):
    from songe.sleep.sizing import cost_model, predict

    gb = 10**9
    base = original_rows()[0]
    rows = [
        base | {"sequence": 13, "host_peak_bytes": int(11.2 * gb)},
        base | {"sequence": 2048, "host_peak_bytes": int(8.7 * gb)},
        base | {"sequence": 3240, "host_peak_bytes": int(11.6 * gb)},
        # 33 tokens later, run-to-run noise adds 0.9 GB.
        base | {"sequence": 3273, "host_peak_bytes": int(12.5 * gb)},
    ]
    model = cost_model(rows, "host_peak_bytes")
    assert model["bytes_per_unit"][0] < 5 * 10**6  # not tens of MB per token
    size = (2048, *[1, 1, 1])
    assert predict(model, size) == int(8.7 * gb)
    assert predict(model, (2805, 1, 1, 1)) == int(11.6 * gb)


def test_stale_never_fit_verdicts_return_to_scoring(config, store):
    from songe.steps import revision_id

    sample = step(900)
    store.sync_steps([sample])
    [stored] = [s for s in store.unscored([sample])]
    saved = {
        "rows": [
            r | {"sequence": 1024, "response_tokens": config.response_tokens}
            for r in original_rows()
        ],
        "memory_baseline": {"total": 24 * GB, "vram": []},
    }
    from songe.sleep.samples import SampleAdmission

    admission = SampleAdmission(config, None, saved, store, None)
    entry = {
        "id": stored["id"],
        "revision": revision_id(stored),
        "required_sequence": 900,
        "reason": "Required sequence exceeds machine RAM at the smallest measured chunk",
    }
    store.mark_unfit(admission.key(stored), entry, stored)
    assert not store.unscored([sample])
    assert calibrate.release_fitting_unfit(config, store, saved) == 1
    assert store.unscored([sample]) and not store.get("unfit_samples")


def test_a_metal_buffer_ceiling_is_a_capacity_limit_not_an_error(
    config, store, measured
):
    """Live run: measuring asked Metal for one 15.05 GB buffer (max 14.3 GB).
    That size can never fit; it becomes a limit and smaller sizes stand."""
    calibrate._calibrate(config, store.root, (256,))
    saved = store.get("calibration")
    saved["in_progress"] = PROBE | {"kind": "sequence", "phase": "train"}
    store.set("calibration", saved)
    stderr = (
        b"RuntimeError: [metal::malloc] Attempting to allocate 15053357056 bytes"
        b" which is greater than the maximum allowed buffer size of 14302248960"
        b" bytes."
    )
    assert calibrate.record_buffer_ceiling(store, stderr)
    saved = store.get("calibration")
    assert any(r["sequence"] == 4169 for r in saved["capacity_limits"])
    assert "in_progress" not in saved and saved["rows"]
    assert not calibrate.record_buffer_ceiling(store, b"ValueError: other")


def test_the_measuring_child_itself_records_a_buffer_ceiling(
    config, store, measured, monkeypatch
):
    """The scheduler starts the measuring child directly; the child records
    the ceiling and exits normally, so no error retry follows."""
    calibrate._calibrate(config, store.root, (256,))
    saved = store.get("calibration")
    saved["in_progress"] = PROBE | {"kind": "sequence", "phase": "train"}
    store.set("calibration", saved)

    def too_big(*args):
        raise RuntimeError(
            "[metal::malloc] Attempting to allocate 15053357056 bytes which is"
            " greater than the maximum allowed buffer size of 14302248960 bytes."
        )

    monkeypatch.setattr(calibrate, "_calibrate", too_big)
    monkeypatch.setattr(
        "sys.argv", ["calibrate", "--data-dir", str(store.root), "--child"]
    )
    calibrate.main()
    assert any(
        r["sequence"] == 4169 for r in store.get("calibration")["capacity_limits"]
    )


def test_unsplit_feedback_counts_as_pending_so_a_first_cycle_is_chosen(store):
    """Live run: 82 recorded steps stayed split 'pending' forever. Splits are
    settled only once a model is chosen, and a model was chosen only when it
    had pending 'train' steps."""
    store.sync_steps([step(900, i) | {"split": "pending"} for i in range(1, 6)])
    assert store.pending_learnable() == 5
    store.assign_splits(minimum=2)
    train, held, _ = store.experience_counts()
    assert train == 3 and held == 2
    assert store.pending_learnable() == 3


def test_a_slice_that_outgrew_its_prediction_resizes_the_next(config, store):
    """Live run: real slices used 12.5 GB against a 10.1 GB measured envelope
    and stopped the same way every time. The stop now teaches the sizing."""
    from songe.sleep.sizing import corrected, learn_peak

    store.set(
        "calibration",
        {
            "rows": [{"sequence": 2048, "host_peak_bytes": 10_000}],
            "teacher_rows": [{"teacher_sequence": 8748, "host_peak_bytes": 8_000}],
        },
    )
    stop = {
        "stop_reason": {
            "rule": "prediction_exceeded",
            "trigger": {
                "rule": "prediction_exceeded",
                "observed": {"tree_rss_bytes": 12_500},
                "prediction": {"peak_bytes": 10_000},
            },
        }
    }
    assert learn_peak(store, stop) == 1.375
    scaled = corrected(store.get("calibration"))
    assert scaled["rows"][0]["host_peak_bytes"] == 13_750
    assert scaled["teacher_rows"][0]["host_peak_bytes"] == 11_000
    assert learn_peak(store, {"stop_reason": {"rule": "memory_margin"}}) is None
    # The live record nests the first alarm under observed["trigger"].
    store.set("calibration", store.get("calibration") | {"peak_correction": 1.0})
    nested = {
        "stop_reason": {
            "rule": "prediction_exceeded",
            "observed": {"trigger": stop["stop_reason"]["trigger"]},
        }
    }
    assert learn_peak(store, nested) == 1.375


def test_a_slice_that_outgrew_its_gpu_prediction_resizes_the_next(config, store):
    """Live run (A100): slices used 18.8 GB of GPU memory against a 15.9 GB
    measured envelope and stopped the same way every time; only the host
    rule taught the sizing."""
    from songe.sleep.sizing import corrected, learn_peak

    store.set(
        "calibration",
        {
            "rows": [
                {
                    "sequence": 8192,
                    "host_peak_bytes": 11_000,
                    "device_peak_bytes": 16_000,
                    "phase_peaks": {"teacher": {"device_peak_bytes": 15_000}},
                }
            ],
            "teacher_rows": [{"host_peak_bytes": 9_000, "device_peak_bytes": 15_000}],
        },
    )
    stop = {
        "stop_reason": {
            "rule": "device_prediction_exceeded",
            "prediction": {
                "device_peak_bytes": 16_000,
                "devices": [{"used_bytes": 20_000, "available_bytes": 40_000}],
            },
        }
    }
    assert learn_peak(store, stop) == 1.375
    scaled = corrected(store.get("calibration"))
    row = scaled["rows"][0]
    assert row["device_peak_bytes"] == 22_000
    assert row["phase_peaks"]["teacher"]["device_peak_bytes"] == 20_625
    assert scaled["teacher_rows"][0]["device_peak_bytes"] == 20_625
    # Host memory keeps its own (uncorrected) measurement.
    assert row["host_peak_bytes"] == 11_000
    assert store.get("calibration").get("peak_correction") is None


def test_a_held_size_is_retried_at_most_twice(store, monkeypatch):
    """Live run: a held 0.8B size was retried every time memory freed and
    stopped again, so training never started."""
    from songe.sleep import calibration_state

    probe = {"sequence": 7424, "kind": "sequence"}
    saved = {
        "progress": {"completion_reason": "held"},
        "held": [probe | {"needed_bytes": 10 * 10**9}],
    }
    store.set("calibration", saved)
    from songe.platform import Memory

    free = Memory(64 * 10**9, 40 * 10**9, "normal")
    monkeypatch.setattr(
        "songe.platform.current", lambda: SimpleNamespace(memory=lambda: free)
    )
    assert calibration_state.memory_freed(store)
    saved["held_retries"] = {calibration_state.held_key(probe): 3}
    store.set("calibration", saved)
    assert not calibration_state.memory_freed(store)


def test_measuring_text_follows_the_teacher_once_the_window_is_covered():
    """Live: 'Measuring · 8192 of 6833 tokens · first time only' while only the
    teacher envelope was still growing after new longer conversations."""
    from songe.sleep.calibration_state import measurement_text, measuring_tokens

    progress = {
        "measured_sequence": 8192,
        "target_sequence": 6833,
        "teacher_measured_sequence": 7212,
        "teacher_target_sequence": 13673,
    }
    assert measuring_tokens(progress) == (7212, 13673, True)
    assert measurement_text(progress) == (
        "measuring training size · 7212 of ~13673 tokens · longer contexts"
    )
    done = progress | {"teacher_measured_sequence": 13673}
    assert measuring_tokens(done) == (6833, 6833, False)
    first = {"measured_sequence": 256, "target_sequence": 2048}
    assert measurement_text(first) == "measuring training size · 256 of ~2048 tokens"


def test_measuring_covers_the_practice_scenes_a_dream_trains_on(config, store):
    """Live (RTX 5090, a chat client without tools): measuring read only the
    recorded conversations (teacher up to 219 tokens). The practice scenes a
    Deep Dream trains on needed 278 to 305, so every slice deferred all of
    them and the Dream never finished."""
    import json

    from songe.sleep.calibration_state import (
        revisions_key,
        sample_revisions,
        smallest_pending,
    )

    store.sync_steps([step(200, 1)])
    with store.connect() as db:
        for i in range(3):
            row = {
                "id": f"sig{i}#v1",
                "signal_id": f"sig{i}",
                "messages": [
                    {"role": "system", "content": "s" * 400},
                    {"role": "user", "content": f"scene {i}"},
                ],
                "tools": [],
                "teacher_note": "Note from an earlier, different task.",
            }
            db.execute(
                'INSERT INTO practice(id,model_id,signal_id,payload,kept,"drop",created)'
                " VALUES (?,?,?,?,1,NULL,0)",
                (row["id"], store.model_id, row["signal_id"], json.dumps(row)),
            )
    # Real steps only where real steps are trained on.
    assert [s["id"] for s in measure_pool(config, store, Tokenizer())[0]] == ["sample-1"]
    # Under dreams, before any training slice has marked the store.
    dreams = replace(config, learning="dreams")
    samples, _, key = measure_pool(dreams, store, Tokenizer())
    lengths = {s["id"]: s["teacher_sequence"] for s in samples}
    assert set(lengths) == {"sample-1"} | {f"practice:sig{i}#v1" for i in range(3)}
    assert lengths["practice:sig0#v1"] == 38 + 400 + dreams.response_tokens
    assert lengths["practice:sig0#v1"] > lengths["sample-1"]
    # The check that runs on every refresh counts the same pool without
    # decoding it; a different key would ask for measuring again, for ever.
    assert key == revisions_key(sample_revisions(store, dreams))
    # The scheduler can now see that nothing pending fits a small slice,
    # instead of loading the model to find out.
    assert smallest_pending(dreams, {"samples": samples}, store) == 566
