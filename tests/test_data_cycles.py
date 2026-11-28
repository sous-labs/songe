"""Finite, data-sized cycles without models, accelerators or servers."""

import json
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from songe.api import progress_text, snapshot
from songe.backends import Interrupted
from songe.backends.training import sample_plan, train
from songe.config import Config
from songe.schedule import Scheduler, rebuild_key, work_unit_seconds
from songe.sleep import worker
from songe.sleep.gate import Decision
from songe.sleep.select import replay_pool
from songe.storage import private_dir, write_json
from tests.test_schedule import good
from tests.test_worker import Backend


class ScalarBackend:
    """Exercise the production optimizer/checkpoint loop using scalar math."""

    kind, device = "torch", "cpu"

    def __init__(self, config, clock):
        self.config, self.clock = config, clock
        self.seen, self.gradients = [], []
        self.weight, self._updates, self.token_divergences = 0.0, 0, []
        self.torch = SimpleNamespace(
            save=lambda value, path: write_json(path, value),
            load=lambda path, **kw: json.loads(path.read_text(encoding="utf-8")),
        )

    def prepare(self):
        pass

    def check(self):
        pass

    def optimizer(self, schedule):
        return SimpleNamespace(
            state={}, state_dict=lambda: {}, load_state_dict=lambda _: None
        )

    def train_step(self, step, optimizer, seed):
        self.seen.append((step["id"], seed))
        self.clock[0] += 1000
        return 1.0, {"weight": 1.0}, 1

    def apply(self, gradients, optimizer):
        self.gradients.append(gradients["weight"])
        self.weight += gradients["weight"]
        self._updates += 1

    def save(self, path):
        private_dir(path)
        write_json(path / "weight.json", self.weight)

    def restore(self, path):
        self.weight = json.loads((path / "weight.json").read_text(encoding="utf-8"))

    def peak_memory(self):
        return 0

    def layer_deltas(self):
        return [self.weight]


def steps(count, start=1):
    return [
        {"id": str(i), "source_id": i, "split": "train", "hindsight": ["correction"]}
        for i in range(start, start + count)
    ]


@pytest.mark.parametrize("count", [20, 200])
def test_training_steps_grow_with_data_beyond_old_time_ceiling(
    count, config, tmp_path, monkeypatch
):
    from songe.backends import training

    clock = [0]
    monkeypatch.setattr(training.time, "monotonic", lambda: clock[0])
    backend = ScalarBackend(config, clock)
    pool, progress = steps(count), []
    work = private_dir(tmp_path / "work")
    result = train(backend, pool, [], work, lambda *p: progress.append(p))
    assert result["iterations"] == math.ceil(count / config.accumulation)
    assert result["steps_learned"] == count
    assert result["seconds"] == count * 1000 > 7200
    assert [sid for sid, _ in backend.seen] == [s["id"] for s in pool]
    assert progress[-1][:2] == (count, count)
    # Each step reports its own position and loss, ahead of the checkpoint.
    live = progress[-1][2]
    assert live["iteration"] == result["iterations"] == len(live["losses"])
    assert live["seconds_per_step"] > 0
    # Even an explicitly repeated call restores the completed checkpoint and
    # performs no additional updates on this pool.
    resumed = ScalarBackend(config, clock)
    train(resumed, pool, [], work, lambda *_: None)
    assert resumed.seen == [] and resumed._updates == result["iterations"]


def test_the_slice_deadline_stops_only_between_steps(config, tmp_path):
    """Live run: the deadline was checked per token and cut steps midway,
    throwing their work away. It now stops at the next step boundary."""
    config.accumulation, config.anchor_fraction = 2, 0
    pool, clock = steps(6), [0]
    backend = ScalarBackend(config, clock)
    boundaries = []

    def boundary():
        boundaries.append(len(backend.seen))
        if len(boundaries) == 3:  # the deadline passes during step 2
            raise Interrupted("slice deadline")

    backend.boundary = boundary
    work = private_dir(tmp_path / "work")
    with pytest.raises(Interrupted):
        train(backend, pool, [], work, lambda *_: None)
    # Two whole steps ran and were saved; no step was started and dropped.
    assert boundaries == [0, 2, 4] and len(backend.seen) == 4
    saved = json.loads((work / "checkpoint.json").read_text(encoding="utf-8"))
    assert saved["iteration"] == 2 and saved["sample_cursor"] == 4


def test_a_sample_whose_recompute_differs_is_passed_over(config, tmp_path):
    """Live (Inkling 2 Mini): a mixture-of-experts routing tie made the
    recomputed forward differ, torch raised CheckpointError and the whole
    Dream failed after 1.5 h. That sample is now passed over."""

    class CheckpointError(RuntimeError):
        pass

    config.accumulation, config.anchor_fraction = 2, 0
    pool, clock = steps(6), [0]
    backend = ScalarBackend(config, clock)
    plain = backend.train_step

    def train_step(step, optimizer, seed):
        if step["id"] in ("2", "3", "4"):
            raise CheckpointError("recomputed tensor shapes differ")
        return plain(step, optimizer, seed)

    backend.train_step = train_step
    result = train(backend, pool, [], private_dir(tmp_path / "work"), lambda *_: None)
    # Step 1 averages over its one good sample; step 2 had none and is not applied.
    assert backend.gradients == [1.0, 1.0]
    assert result["iterations"] == 2 and result["samples_skipped"] == 3
    assert sorted(result["trained_revisions"]) == ["1", "5", "6"]

    # Any other error still stops the run.
    def broken(step, optimizer, seed):
        raise RuntimeError("out of memory")

    other = ScalarBackend(config, clock)
    other.train_step = broken
    with pytest.raises(RuntimeError):
        train(other, steps(2, 50), [], private_dir(tmp_path / "other"), lambda *_: None)


def test_resume_keeps_frozen_batch_cursor_and_rehearsal(config, tmp_path):
    config.accumulation, config.anchor_fraction = 3, 0.25
    pool, anchors, clock = steps(7), steps(3, 100), [0]
    backend = ScalarBackend(config, clock)
    work = private_dir(tmp_path / "work")

    def interrupt(completed, total, live=None):
        raise Interrupted("traffic")

    with pytest.raises(Interrupted):
        train(backend, pool, anchors, work, interrupt)
    assert (
        json.loads((work / "checkpoint.json").read_text(encoding="utf-8"))[
            "sample_cursor"
        ]
        == 3
    )
    config.accumulation, config.anchor_fraction = 2, 1
    resumed = ScalarBackend(config, clock)
    result = train(resumed, pool, anchors, work, lambda *_: None)
    seen = backend.seen + resumed.seen
    assert Counter(sid for sid, _ in seen) == Counter(s["id"] for s in pool + anchors)
    assert [seed for _, seed in seen] == list(range(7613, 7623))
    assert result["iterations"] == 4  # The original three-sample batch stays fixed.
    assert result["steps_learned"] == len(pool)
    assert result["samples_trained"] == len(pool) + len(anchors)
    assert backend.gradients + resumed.gradients == [1] * result["iterations"]


def test_empty_data_never_trains_replay_or_anchors(config, tmp_path):
    backend = ScalarBackend(config, [0])
    backend.prepare = lambda: pytest.fail("Empty data must not prepare training")
    assert train(backend, [], steps(5), tmp_path, lambda *_: None)["iterations"] == 0
    for fraction in (0, 0.25, 1):
        assert replay_pool(steps(5), 5, fraction) == []


def test_rehearsal_preserves_its_share_without_repeating_selected_data():
    plan = sample_plan(steps(4), steps(1, 100), 0.5)
    assert [index for anchor, index in plan if not anchor] == list(range(4))
    assert sum(anchor for anchor, _ in plan) == 4
    assert sample_plan(steps(4), steps(1, 100), 1).count((True, 0)) == 1


def test_a_dream_makes_its_passes_over_the_pool_each_in_a_new_order(config):
    """One pass over a few dozen scenes is a handful of optimizer steps, most
    of them spent in the schedule's warmup and decay (live: 28 samples, 7
    steps, an adapter that answered like the base)."""
    from dataclasses import replace

    from songe.backends.training import plan_split, training_plan

    once = sample_plan(steps(6), steps(1, 100), 0.25)
    config = replace(config, anchor_fraction=0.25, learning="dreams", passes=3)
    plan = training_plan(steps(6), steps(1, 100), config)
    assert len(plan) == 3 * len(once) == 24 and plan[: len(once)] == once
    passes = [plan[i : i + len(once)] for i in range(0, len(plan), len(once))]
    assert all(sorted(p) == sorted(once) for p in passes) and passes[1] != passes[0] != passes[2]
    assert training_plan(steps(6), steps(1, 100), replace(config, passes=1)) == once
    assert training_plan([], steps(1, 100), config) == []
    # Real steps keep their single pass, each with a fresh sample.
    assert training_plan(steps(6), steps(1, 100), replace(config, learning="hindsight")) == once
    # The record counts the passes over the pool as practice, not rehearsal.
    assert plan_split(plan) == {"practice": 18, "rehearsal": 6}


def test_a_measured_step_is_two_updates_whatever_the_pass_setting(config, tmp_path):
    """Calibration times two optimizer updates. When the training loop
    repeated those two batches as it repeats a Dream's pool, a step that
    takes 36 s was recorded as 590 s and the measurement ran 40 minutes
    (live, twenty repeats)."""
    from dataclasses import replace

    from songe.sleep.calibrate import two_updates

    backend = ScalarBackend(replace(config, accumulation=4, learning="dreams", passes=4), [0])
    backend.train = lambda *args: train(backend, *args)
    two_updates(backend, steps(1)[0], 4, private_dir(tmp_path / "work"))
    assert backend._updates == 2 and len(backend.seen) == 8
    assert backend.config.passes == 4


def test_old_time_cap_config_loads_and_saves_without_retired_fields(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        "nightly_seconds = 7200\nsteps_per_night = 50\nscore_per_night = 100\n"
        "training_time_fraction = 0.35\nscoring_time_fraction = 0.15\n",
        encoding="utf-8",
    )
    config = Config.load(tmp_path)
    config.save(tmp_path)
    assert not any(
        key in path.read_text(encoding="utf-8")
        for key in ("nightly", "per_night", "time_fraction")
    )
    assert Config.load(tmp_path) == config


@pytest.mark.parametrize("rebuild", [False, True])
async def test_completed_data_does_not_start_another_cycle(
    config, store, monkeypatch, rebuild
):
    config.calibration_available = True
    config.force_cpu_training = True  # Hardware-independent: CI has no GPU.
    config.min_train = config.min_heldout = config.min_new = 1
    pool = [s | {"surprise": 1} for s in steps(3)]
    heldout = [
        {"id": "held", "conversation": "held", "split": "heldout", "hindsight": ["h"]}
    ]
    store.add_pool(pool)
    store.sync_steps(heldout)
    store.mark_scored(pool)
    store.set("experience_watermark", 3)
    store.set("score_watermark", 3)
    if rebuild:
        store.set("rebuild_required", True)
        store.set("completed_rebuild", rebuild_key(config, store))
    monkeypatch.setattr(
        worker, "preview", lambda *a: ({"reason": "Ready"}, pool, heldout)
    )
    monkeypatch.setattr(
        worker,
        "probe",
        lambda: pytest.fail("Completed cycles need no hardware admission"),
    )
    monkeypatch.setattr(worker, "choose", lambda *a: pytest.fail("No new data"))
    result = worker.run(config, store)
    assert result["reason"] == "Waiting for new learnable experience"
    assert not store.nights()
    scheduler = Scheduler(config, store, None)
    assert not await scheduler.enough()


def test_status_reports_data_counts_and_eta_without_a_time_budget(
    config, store, capsys
):
    from songe.cli import print_status
    from songe.panels import panel_rows

    progress = {
        "phase": "train",
        "completed": 20,
        "total": 400,
        "unit": "samples",
        "eta_seconds": 9000,
    }
    store.night(None, phase="distill", record={"data_progress": progress})
    data = snapshot(config, store)
    assert data["data_progress"] == progress
    assert "20/400 samples" in progress_text(progress)
    panel = dict(
        panel_rows(
            {"id": "one"}, {"model": "one", "phase": "train", "data_progress": progress}
        )
    )
    assert panel["now"] == "Training · about 2h 30m left"
    print_status(data)
    output = capsys.readouterr().out
    assert "samples" not in output
    assert "Connect a model" in output and "Run songe init" in output
    # Measured costs affect estimates only, and no capacity is inferred.
    config.calibrated_step_seconds, config.calibrated_score_seconds = 12, 3
    assert work_unit_seconds(config, None) == 12


@pytest.fixture
def cycle(config, store, monkeypatch):
    from songe.platform import Accel
    from songe.sleep import gate

    class Model(Backend):
        scored, trained_units, initial_adapters = [], [], []
        comparisons = 0
        interrupt_training = interrupt_evaluation = False

    config.upstream, config.calibration_available = "mlx", True
    config.min_train = config.min_heldout = config.min_new = 1
    config.slice_limit = 100
    pool = [
        s
        | {
            "conversation": s["id"],
            "context": [{"role": "user", "content": "Fix this"}],
            "hindsight": [{"role": "user", "content": "correction"}],
            "tools": [],
        }
        for s in steps(3)
    ]
    held = pool[0] | {"id": "held", "conversation": "held", "split": "heldout"}
    store.sync_steps(pool + [held])
    monkeypatch.setattr(
        worker,
        "preview",
        lambda *a: (
            {"reason": "Ready"},
            store.active_steps("train"),
            store.active_steps("heldout"),
        ),
    )
    monkeypatch.setattr(worker, "probe", lambda: good())
    monkeypatch.setattr(
        worker,
        "current",
        lambda: SimpleNamespace(
            name="darwin", accelerators=lambda: [Accel("metal", "stand-in")]
        ),
    )
    monkeypatch.setattr(worker, "choose", lambda *a: Model)
    monkeypatch.setattr(worker.signal, "signal", lambda *a: None)
    anchors = worker.general_steps()[:1]
    monkeypatch.setattr(worker, "general_steps", lambda: anchors)
    monkeypatch.setattr(
        gate,
        "choose_scale",
        lambda *a, **kw: (None, Decision(False, "Test rejection"), {}),
    )
    return Model, pool


@pytest.mark.parametrize("rebuild", [False, True])
async def test_finished_cycle_waits_and_later_data_starts_next_cycle(
    cycle, config, store, rebuild
):
    model, pool = cycle
    store.set("rebuild_required", rebuild)
    first = worker.run(config, store)
    assert first["iterations"] > 0 and not first["accepted"]
    units = model.trained_units.copy()
    assert worker.run(config, store)["reason"] == "Waiting for new learnable experience"
    assert model.trained_units == units
    scheduler = Scheduler(config, store, None)
    assert not await scheduler.enough()
    # A new conversation: Dream would collapse a repeat of step 1's action.
    store.sync_steps(
        [pool[0] | {"id": "later", "conversation": "later", "source_id": 10}]
    )
    assert await scheduler.enough()
    second = worker.run(config, store)
    assert second["iterations"] > 0 and second["watermark"] == 10
    assert second["pool_size"] == (4 if rebuild else 2)
    assert len(store.nights()) == 2
    assert not await scheduler.enough()


def test_memory_sized_interruption_keeps_cycle_resumable(cycle, config, store):
    model, _ = cycle
    model.interrupt_training = True
    sizing = {
        "max_sequence": 1024,
        "response_tokens": 128,
        "token_chunk": 32,
        "accumulation": 4,
    }
    assert worker.run(config, store, sizing=sizing) == {"interrupted": True}
    assert store.nights()[0]["status"] == "interrupted"
    assert (store.root / "versions/night-1/fake-checkpoint.json").exists()
    worker.run(config, store, sizing=sizing | {"max_sequence": 512})
    assert len(store.nights()) == 1


def test_worker_saves_resolved_training_settings_at_cycle_start(
    cycle, config, store, monkeypatch
):
    import sys

    from songe import telemetry
    from songe.config import TRAINING_KEYS
    from songe.models import registry
    from songe.steps import digest

    backend, _ = cycle
    config.learning_rate = 6e-5
    config.save(store.root)
    path = store.root / "config.toml"
    installation = "".join(
        line
        for line in path.read_text(encoding="utf-8").splitlines(keepends=True)
        if not line.startswith("vocab_clip = ")
    )
    path.write_text(installation, encoding="utf-8")
    registry.register(config, store, "fixture", legacy=True)
    registry.set_learning(store, "fixture", "all")
    selected = Config.for_model(store.root, "fixture")
    selected.accumulation = 2
    selected.save(store.root)
    monkeypatch.setattr(telemetry, "configure", lambda root: None)
    sizing = {"max_sequence": 1024, "token_chunk": 32, "response_tokens": 128}
    argv = ["worker", "--child", "--data-dir", str(store.root), "--model-key", "fixture"]
    argv += ["--sizing", json.dumps(sizing)]
    monkeypatch.setattr(sys, "argv", argv)
    backend.interrupt_training = True

    worker.main()

    (night,) = store.nights()
    assert night["status"] == "interrupted"
    record = night["record"]
    saved = record["training_settings"]
    assert saved["learning_rate"] == {"value": 6e-5, "source": "installation"}
    assert saved["accumulation"] == {"value": 2, "source": "per-model"}
    assert saved["vocab_clip"] == {"value": Config().vocab_clip, "source": "default"}
    assert saved["max_sequence"]["value"] == config.max_sequence
    assert set(saved) == set(TRAINING_KEYS)
    assert digest({key: item["value"] for key, item in saved.items()}) == record[
        "configuration"
    ]

    # Same effective policy, different source and a smaller slice.
    path.write_text(
        f"vocab_clip = {Config().vocab_clip}\n" + installation, encoding="utf-8"
    )
    argv[-1] = json.dumps(sizing | {"max_sequence": 512})
    worker.main()

    (finished,) = store.nights()
    assert finished["id"] == night["id"]
    assert finished["status"] == "rejected"
    assert finished["record"]["training_settings"] == saved
    sizes = finished["record"]["slice_sizes"]
    assert [size["max_sequence"] for size in sizes] == [1024, 512]
    assert snapshot(selected, store)["nights"][0]["record"]["training_settings"] == saved


def test_rescored_unlearnable_revision_leaves_selected_pool(store):
    original = steps(1)[0] | {"surprise": 1}
    store.add_pool([original])
    revised = original | {"source_id": 2}
    store.sync_steps([revised])
    store.add_pool([], scored=[revised])
    assert store.pool() == []


def test_long_selected_sample_waits_intact_then_trains_when_memory_returns(
    cycle, config, store, monkeypatch
):
    from copy import deepcopy

    from songe.sleep.prompts import paired_prompts
    from tests.test_learning import Tokenizer

    model, original_pool = cycle
    long = original_pool[0] | {
        "context": [{"role": "user", "content": "Full old context. " * 50}],
        "surprise": 1,
    }
    original = deepcopy(long)
    store.sync_steps([long])
    store.add_pool([long])
    store.mark_scored(original_pool)
    required = (
        len(paired_prompts(Tokenizer(), long, config)[0]) + config.response_tokens
    )
    teacher = (
        max(map(len, paired_prompts(Tokenizer(), long, config)))
        + config.response_tokens
    )
    # The full-context forward (teacher envelope) defers it, not its length.
    small = {
        "max_sequence": config.max_sequence,
        "max_teacher_sequence": teacher - 1,
        "response_tokens": config.response_tokens,
        "token_chunk": 1,
        "accumulation": 1,  # Stale sizing must never override optimizer policy.
    }
    seen = []
    original_train = model.train

    def train(self, pool, anchors, work, progress):
        seen.extend(deepcopy(pool))
        assert self.config.accumulation == config.accumulation
        assert self.config.response_tokens == config.response_tokens
        return original_train(self, pool, anchors, work, progress)

    monkeypatch.setattr(model, "train", train)
    assert worker.run(config, store, sizing=small) == {"interrupted": True}
    assert store.pool() == [original]
    assert not seen
    deferred = store.slices()[-1]["record"]["deferred_samples"]
    assert deferred[0]["id"] == long["id"]
    assert deferred[0]["required_sequence"] == required
    result = worker.run(config, store, sizing=small | {"max_teacher_sequence": teacher})
    assert result["iterations"] > 0
    assert seen == [original]
    assert len(store.nights()) == 1
    assert not store.slices()[-1]["record"]["deferred_samples"]


def test_gate_waits_for_memory_rather_than_scoring_a_held_out_subset(
    cycle, config, store, monkeypatch
):
    from songe.sleep import evaluate
    from songe.sleep.prompts import paired_prompts
    from tests.test_learning import Tokenizer

    model, pool = cycle
    long = pool[0] | {
        "id": "held-long",
        "conversation": "held-long",
        "split": "heldout",
        "context": [{"role": "user", "content": "Full old context. " * 50}],
    }
    store.sync_steps([long])
    teacher = (
        max(map(len, paired_prompts(Tokenizer(), long, config)))
        + config.response_tokens
    )
    # This machine measured a teacher this long, so the sample can fit.
    store.set("calibration", {"teacher_rows": [{"teacher_sequence": teacher}]})
    roomy = {
        "max_sequence": config.max_sequence,
        "max_teacher_sequence": teacher,
        "response_tokens": config.response_tokens,
        "token_chunk": 1,
    }
    scored = []
    original = evaluate.references

    def references(backend, steps, **kwargs):
        scored.extend(s["id"] for s in steps)
        return original(backend, steps, **kwargs)

    monkeypatch.setattr(evaluate, "references", references)
    model.interrupt_training = True
    assert worker.run(config, store, sizing=roomy) == {"interrupted": True}
    # Memory is short: training finishes, the gate keeps the night's full set.
    tight = roomy | {"max_teacher_sequence": teacher - 1}
    assert worker.run(config, store, sizing=tight) == {"interrupted": True}
    assert not scored
    night = store.nights()[-1]
    assert night["status"] == "interrupted"
    assert night["record"]["reason"].startswith("Waiting for memory")
    trained = len(model.trained_units)
    worker.run(config, store, sizing=roomy)
    assert len(store.nights()) == 1
    assert len(model.trained_units) == trained
    assert {"held", "held-long"} <= set(scored)
    assert len(scored) == len(set(scored))


def test_gate_does_not_wait_for_a_sample_this_machine_never_fits(cycle, config, store):
    from songe.sleep.prompts import paired_prompts
    from tests.test_learning import Tokenizer

    model, pool = cycle
    long = pool[0] | {
        "id": "held-long",
        "conversation": "held-long",
        "split": "heldout",
        "context": [{"role": "user", "content": "Full old context. " * 50}],
    }
    store.sync_steps([long])
    teacher = (
        max(map(len, paired_prompts(Tokenizer(), long, config)))
        + config.response_tokens
    )
    # The longest teacher this machine ever measured is shorter than it.
    store.set("calibration", {"teacher_rows": [{"teacher_sequence": teacher - 1}]})
    sizing = {
        "max_sequence": config.max_sequence,
        "max_teacher_sequence": teacher,
        "response_tokens": config.response_tokens,
        "token_chunk": 1,
    }
    model.interrupt_training = True
    assert worker.run(config, store, sizing=sizing) == {"interrupted": True}
    tight = sizing | {"max_teacher_sequence": teacher - 1}
    worker.run(config, store, sizing=tight)
    reason = store.nights()[-1]["record"].get("reason") or ""
    assert not reason.startswith("Waiting for memory"), reason


def test_never_fit_sample_is_marked_once_and_does_not_restart_cycle(
    cycle, config, store
):
    model, pool = cycle
    long = pool[0] | {
        "context": [
            {"role": "user", "content": [{"type": "image_url", "image_url": {}}]}
        ],
        "surprise": 1,
    }
    store.sync_steps([long])
    store.add_pool([long])
    store.mark_scored(pool)
    result = worker.run(config, store)
    assert result["reason"] == "Waiting for learnable experience"
    unfit = store.get("unfit_samples")
    assert len(unfit) == 1
    entry = next(iter(unfit.values()))
    assert entry["id"] == long["id"] and "Multimodal" in entry["reason"]
    assert len(store.slices()[0]["record"]["unfit_samples"]) == 1
    assert not model.trained_units
    assert worker.run(config, store)["reason"] == "Waiting for new learnable experience"
    assert store.get("unfit_samples") == unfit
    assert store.slices()[-1]["record"]["unfit_samples"] == []
    assert len(store.nights()) == 1


def test_unfit_revision_cannot_reintroduce_older_selected_evidence(
    cycle, config, store
):
    model, pool = cycle
    old = pool[0] | {"surprise": 1}
    store.add_pool([old])
    store.mark_scored(pool)
    revised = old | {
        "source_id": 10,
        "context": [
            {"role": "user", "content": [{"type": "image_url", "image_url": {}}]}
        ],
    }
    store.sync_steps([revised])
    assert worker.run(config, store)["reason"] == "Waiting for learnable experience"
    assert not model.trained_units
    assert not store.pool()
    assert not store.unscored([revised])


def test_memory_running_short_at_the_gate_keeps_the_trained_night(
    cycle, config, store, monkeypatch
):
    from songe.memory import MemoryPressure
    from songe.platform import Memory
    from songe.sleep import gate

    model, _ = cycle

    def short(*args, **kwargs):
        raise MemoryPressure(Memory(1000, 10))

    monkeypatch.setattr(gate, "choose_scale", short)
    with pytest.raises(MemoryPressure):
        worker.run(config, store)
    night = store.nights()[-1]
    assert night["status"] == "interrupted"
    assert night["record"]["reason"] == "Waiting for memory"
    trained = len(model.trained_units)
    assert trained and Path(night["record"]["work"]).exists()
    # With memory back, the same night goes on from the gate.
    monkeypatch.setattr(
        gate,
        "choose_scale",
        lambda *a, **kw: (None, Decision(False, "Test rejection"), {}),
    )
    worker.run(config, store)
    assert len(store.nights()) == 1 and len(model.trained_units) == trained
