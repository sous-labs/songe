import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from songe.backends import Interrupted
from songe.platform import Accel, Memory, Power
from songe.sleep import worker
from songe.storage import private_dir, write_json
from tests.test_learning import Tokenizer
from tests.test_schedule import good
from tests.test_steps import row
from tests.test_store import version


class Backend:
    """Deterministic numerical stand-in: no MLX import, tools, models or services."""

    kind = "mlx"
    scored = []
    trained_units = []
    comparisons = 0
    interrupt_training = False
    interrupt_evaluation = False
    initial_adapters = []

    def __init__(self, config, check, train=True):
        self.config, self.check, self.tokenizer = config, check, Tokenizer()
        self.weight, self.frozen = 0.0, False
        if not train and (Path(config.model) / "test-weight.json").is_file():
            self.weight = json.loads(
                (Path(config.model) / "test-weight.json").read_text(encoding="utf-8")
            )

    @contextlib.contextmanager
    def base(self):
        self.frozen = True
        try:
            yield
        finally:
            self.frozen = False

    def adapter_weights(self):
        return self.weight

    def restore(self, path):
        self.weight = json.loads(
            (Path(path) / "adapters.safetensors").read_text(encoding="utf-8")
        )

    def save(self, path):
        private_dir(path)
        write_json(path / "adapters.safetensors", self.weight)

    def train(self, pool, anchors, work, progress):
        from songe.backends.training import sample_plan

        total = len(sample_plan(pool, anchors, self.config.anchor_fraction))
        budget = (total + self.config.accumulation - 1) // self.config.accumulation
        type(self).initial_adapters.append(self.weight)
        checkpoint = work / "fake-checkpoint.json"
        iteration = 0
        if checkpoint.exists():
            state = json.loads(checkpoint.read_text(encoding="utf-8"))
            iteration, self.weight = state["iteration"], state["weight"]
        while iteration < budget:
            self.check()
            iteration += 1
            self.weight += 0.5
            write_json(checkpoint, {"iteration": iteration, "weight": self.weight})
            self.trained_units.append(iteration)
            progress(min(iteration * self.config.accumulation, total), total)
            if self.interrupt_training:
                type(self).interrupt_training = False
                raise Interrupted("test activity")
        self.save(work / "adapter")
        return {
            "iterations": iteration,
            "steps_learned": len(pool),
            "layer_deltas": [self.weight],
        }

    def sample(self, prompt, seed=None):
        self.check()
        return [1], "a useful answer", SimpleNamespace()

    def scale_delta(self, baseline, trained, scale):
        self.weight = baseline + scale * (trained - baseline)

    def kl_from(self, base, prompt, teacher_prompt, y, **kw):
        self.check()
        type(self).comparisons += 1
        with base.base():
            assert base.frozen
        if self.interrupt_evaluation and self.comparisons == 3:
            type(self).interrupt_evaluation = False
            raise Interrupted("test evaluation activity")
        return (
            self.weight * 0.01
            if prompt == teacher_prompt
            else max(0, 1 - self.weight * 0.5)
        )

    def layer_deltas(self, baseline=0):
        return [abs(self.weight - baseline)]


@pytest.mark.parametrize("phase", ["training", "evaluation", "battery"])
@pytest.mark.parametrize("upstream", ["mlx", "lmstudio", "llamacpp"])
@pytest.mark.parametrize("rebuild", [False, True])
def test_pipeline_resumes_units_and_continues_current_adapter_with_frozen_teacher(
    tmp_path, config, store, monkeypatch, phase, upstream, rebuild
):
    Backend.scored, Backend.trained_units, Backend.initial_adapters = [], [], []
    Backend.comparisons = 0
    Backend.interrupt_training = phase == "training"
    Backend.interrupt_evaluation = phase == "evaluation"
    config.upstream = upstream
    config.models_dir = str(tmp_path / "test-model-library")
    Path(config.models_dir).mkdir()
    from songe.sleep import fuse

    def fake_fuse(config, adapter, destination, check, **kwargs):
        check()
        private_dir(destination)
        weight = json.loads(
            (adapter / "adapters.safetensors").read_text(encoding="utf-8")
        )
        # Full strength loses quality during fusing. The smaller actual artifact
        # must win even though the full-strength adapter itself passed.
        write_json(destination / "test-weight.json", 20 if weight > 1 else weight)
        write_json(
            destination / "songe_owned.json",
            {"base": config.model, "adapter": str(adapter)},
        )
        return destination

    monkeypatch.setattr(fuse, "fuse", fake_fuse)
    config.calibration_available = True
    config.accumulation = 4
    config.slice_limit = 4
    config.min_train, config.min_heldout, config.min_new = 3, 3, 1
    for i in range(35):
        context = [{"role": "user", "content": f"Calculate {i}"}]
        wrong = {"role": "assistant", "content": "Wrong"}
        followup = context + [
            wrong,
            {"role": "user", "content": "Actually it should be four"},
        ]
        for source in [
            row(i * 2, context, wrong),
            row(i * 2 + 1, followup, {"role": "assistant", "content": "Four"}),
        ]:
            store.record(
                source["path"],
                source["request"],
                source["response"],
                binding="previous",
            )
    previous = tmp_path / "previous-adapter"
    previous.mkdir()
    write_json(previous / "adapters.safetensors", 0.25)
    store.add_version(version("previous", 1) | {"adapter": str(previous)})
    store.swap("previous", None)
    store.set("rebuild_required", rebuild)
    monkeypatch.setattr(worker, "probe", lambda: good())
    battery_low, power_clock = False, 0
    if phase == "battery":
        original_train = Backend.train

        def train(self, pool, anchors, work, progress):
            def commit(completed, total):
                nonlocal battery_low, power_clock
                progress(completed, total)
                if completed == self.config.accumulation:
                    battery_low, power_clock = True, 2

            return original_train(self, pool, anchors, work, commit)

        monkeypatch.setattr(Backend, "train", train)
        monkeypatch.setattr(
            worker,
            "time",
            SimpleNamespace(monotonic=lambda: power_clock, time=worker.time.time),
        )
    monkeypatch.setattr(
        worker,
        "current",
        lambda: SimpleNamespace(
            name="darwin",
            memory=lambda: Memory(1000000000, 900000000),
            accelerators=lambda: [Accel("metal", "mock")],
            power=lambda: Power(False, battery_low),
        ),
    )
    monkeypatch.setattr(worker, "choose", lambda *args: Backend)
    monkeypatch.setattr(worker.signal, "signal", lambda *args: None)
    anchors = worker.general_steps()[:2]
    monkeypatch.setattr(worker, "general_steps", lambda: anchors)
    assert worker.run(config, store, slice_seconds=60) == {"interrupted": True}
    first_night = store.nights()[0]["id"]
    assert store.nights()[0]["status"] == "interrupted"
    if phase == "battery":
        assert store.slices()[0]["interrupted_by"] == "battery low"
        assert store.nights()[0]["record"]["reason"] == "battery low"
        checkpoint = store.root / "versions/night-1/fake-checkpoint.json"
        assert json.loads(checkpoint.read_text(encoding="utf-8"))["iteration"] == 1
        battery_low = False
    if upstream == "llamacpp":
        monkeypatch.setattr(
            fuse, "export_candidate", lambda *args: pytest.fail("export during evaluation")
        )
        passed = worker.run(config, store, slice_seconds=60)
        assert passed["accepted"]
        assert store.nights()[0]["status"] == "interrupted"
        assert store.nights()[0]["phase"] == "fuse"
        assert store.get("current") == "previous"
        comparisons = Backend.comparisons
        eval_events = [e for e in store.events() if e["stage"] == "eval"]
        monkeypatch.setattr(
            worker, "choose", lambda *args: pytest.fail("reopened the evaluation model")
        )

        def export(config, kind, adapter, destination, platform, check):
            check()
            assert kind == "gguf-lora"
            assert adapter.name == f"scale-{passed['wake_scale']}-adapter"
            if not export.attempted:
                export.attempted = True
                raise MemoryError("conversion did not fit")
            destination.write_bytes(b"GGUF")
            return destination

        export.attempted = False
        monkeypatch.setattr(fuse, "export_candidate", export)
        with pytest.raises(MemoryError, match="conversion did not fit"):
            worker.run(config, store, slice_seconds=60)
        assert store.nights()[0]["status"] == "interrupted"
        assert store.get("current") == "previous"
    result = worker.run(config, store, slice_seconds=60)
    assert result["accepted"] and result["wake_scale"] == (
        0.5 if upstream == "lmstudio" and not rebuild else 1
    )
    assert Path(result["candidate"]["training_adapter"]).name == (
        f"scale-{result['wake_scale']}-adapter"
    )
    if upstream == "llamacpp":
        assert Path(result["candidate"]["adapter"]).read_bytes() == b"GGUF"
        assert Backend.comparisons == comparisons
        assert [e for e in store.events() if e["stage"] == "eval"] == eval_events
    assert store.nights()[0]["id"] == first_night
    assert store.nights()[0]["status"] == "ready"
    assert Backend.trained_units == [1, 2]
    assert all(
        weight == (0.0 if rebuild else 0.25) for weight in Backend.initial_adapters
    )
    assert len(Backend.scored) == len(set(Backend.scored))
    assert store.get("current") == "previous"  # publication belongs to scheduler
    assert len(store.slices()) == (4 if upstream == "llamacpp" else 2)
    assert all(s["ended"] is not None for s in store.slices())
    assert store.slices()[0]["interrupted_by"]
    assert store.slices()[1]["units_done"] > 0
    snapshot = json.loads(
        (store.root / "versions/night-1/snapshot.json").read_text(encoding="utf-8")
    )
    assert all(s["split"] == "train" for s in snapshot["pool"])
    assert all(s["split"] == "heldout" for s in snapshot["heldout"])
