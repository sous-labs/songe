"""Optimizer invariants, using CPU tensors and framework-free MLX stand-ins."""

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from songe.backends.training import train
from songe.memory import Snapshot
from songe.sleep.calibrate import derive
from songe.sleep.sizing import choose_size, cost_model, predict
from songe.storage import private_dir, write_json
from tests.test_data_cycles import ScalarBackend


def measurements():
    return [
        dict(
            sequence=n,
            response_tokens=r,
            token_chunk=c,
            accumulation=b,
            host_peak_bytes=1000 + 10 * n + r + c + 100 * (b > 1),
            device_peak_bytes=0,
            step_seconds=b,
        )
        for n in (64, 512)
        for r in (1, 32)
        for c in (1, 32)
        for b in (1, 4)
    ]


def test_memory_changes_microbatch_geometry_without_changing_optimizer_batch(
    config, monkeypatch
):
    # Isolates the boundary logic from the fixed 10% RAM reserve.
    monkeypatch.setattr("songe.sleep.sizing.RAM_RESERVE_FRACTION", 0)
    config = replace(config, max_sequence=512, response_tokens=32, accumulation=7)
    rows = measurements()
    large = choose_size(config, {"rows": rows}, Snapshot(10000, 10000), {})
    small = choose_size(config, {"rows": rows}, Snapshot(10000, 2500), {})
    assert large["max_sequence"] > small["max_sequence"]
    assert large["accumulation"] == small["accumulation"] == 7
    for resource in ("host", "device"):
        model = cost_model(rows, resource + "_peak_bytes")
        assert predict(model, (128, 16, 1, 1)) == predict(model, (128, 16, 1, 1000))
    assert derive(rows, 1, accumulation=7)["accumulation"] == 7


def scalar_backend(config, monkeypatch, kind, accumulation):
    """The production loop on scalar math, as torch or with MLX stand-ins. A
    sample's loss and gradient are the mean over its own answer tokens."""
    backend = ScalarBackend(replace(config, accumulation=accumulation), [0])
    if kind == "mlx":

        def tree_map(fn, *trees):
            return {key: fn(*(tree[key] for tree in trees)) for key in trees[0]}

        monkeypatch.setitem(
            sys.modules,
            "mlx.utils",
            SimpleNamespace(
                tree_map=tree_map,
                tree_flatten=lambda tree: list(tree.items()),
                tree_unflatten=dict,
            ),
        )
        backend.kind = "mlx"
        backend.mx = SimpleNamespace(
            eval=lambda *args: None,
            save_safetensors=lambda path, value: write_json(Path(path), value),
            load=lambda path: json.loads(path.read_text(encoding="utf-8")),
        )
        backend.optim = SimpleNamespace(clip_grad_norm=lambda grads, **kw: (grads, 0.0))
        backend.optimizer = lambda schedule: SimpleNamespace(state={})

    def microbatch(step, optimizer, seed):
        tokens = len(step["values"])
        mean = sum(step["values"]) / tokens
        return mean, {"weight": mean}, tokens

    backend.train_step = microbatch
    return backend


@pytest.mark.parametrize("kind", ["torch", "mlx"])
def test_a_sample_counts_once_whatever_the_length_of_its_answer(
    config, tmp_path, monkeypatch, kind
):
    # One optimizer step over answers of one, three and two tokens, then a
    # partial step with a single sample.
    pool = [
        {"id": "short", "values": [0.1]},
        {"id": "long", "values": [0.2, 0.3, 0.4]},
        {"id": "middle", "values": [0.5, 0.6]},
        {"id": "partial", "values": [0.8]},
    ]
    backend = scalar_backend(config, monkeypatch, kind, 3)
    result = train(backend, pool, [], private_dir(tmp_path / "work"), lambda *_: None)
    # The mean of the three samples' own means (0.1, 0.3, 0.55). Weighted by
    # tokens it was 0.35: the three-token answer counted three times.
    assert backend.gradients == pytest.approx([(0.1 + 0.3 + 0.55) / 3, 0.8])
    assert result["loss"] == pytest.approx(0.8)
    assert result["iterations"] == 2 and result["steps_learned"] == len(pool)


@pytest.mark.parametrize("kind", ["torch", "mlx"])
def test_a_lesson_counts_once_however_many_of_its_scenes_were_kept(
    config, tmp_path, monkeypatch, kind
):
    from songe.backends.training import sample_weights

    # Lesson a kept three practice scenes, lesson b one.
    pool = [
        {"id": "a1", "signal_id": "a", "values": [0.1]},
        {"id": "a2", "signal_id": "a", "values": [0.2]},
        {"id": "a3", "signal_id": "a", "values": [0.3]},
        {"id": "b1", "signal_id": "b", "values": [0.8]},
    ]
    weights = sample_weights(pool)
    assert weights == pytest.approx([2 / 3, 2 / 3, 2 / 3, 2.0])
    assert sum(weights) == pytest.approx(len(pool))  # 1 on average: the step size stays
    backend = scalar_backend(config, monkeypatch, kind, 4)
    result = train(backend, pool, [], private_dir(tmp_path / "work"), lambda *_: None)
    # Half lesson a (the mean of its scenes), half lesson b; per scene it was 0.35.
    assert backend.gradients == pytest.approx([(0.2 + 0.8) / 2])
    assert result["loss"] == pytest.approx(0.5)
    # Samples that are not practice scenes are their own unit: all count 1.
    assert sample_weights([{"id": "x"}, {"id": "y"}, {"id": "z"}]) == [1.0, 1.0, 1.0]


@pytest.mark.parametrize("kind", ["torch", "mlx"])
def test_general_rehearsal_counts_one_beside_weighted_lessons(
    config, tmp_path, monkeypatch, kind
):
    pool = [
        {"id": "a1", "signal_id": "a", "values": [0.2]},
        {"id": "a2", "signal_id": "a", "values": [0.4]},
        {"id": "b1", "signal_id": "b", "values": [0.9]},
    ]
    anchors = [{"id": "g", "values": [0.6]}]
    backend = scalar_backend(replace(config, anchor_fraction=0.25), monkeypatch, kind, 4)
    train(backend, pool, anchors, private_dir(tmp_path / "work"), lambda *_: None)
    # Scenes of a: 3 / (2 * 2) each; b: 3 / 2; the general prompt: 1.
    assert backend.gradients == pytest.approx([(0.75 * 0.2 + 0.75 * 0.4 + 1.5 * 0.9 + 0.6) / 4])


@pytest.mark.parametrize("resource", ["host", "device"])
def test_never_fit_uses_smallest_chunk_and_total_machine_capacity(
    config, store, resource
):
    from songe.sleep.samples import SampleAdmission
    from tests.test_learning import Tokenizer

    config = replace(config, max_sequence=512, response_tokens=32)
    rows = measurements()
    if resource == "device":
        for row in rows:
            row["device_peak_bytes"] = row["host_peak_bytes"]
            row["host_peak_bytes"] = 0
    step = {
        "id": "long",
        "source_id": 1,
        "split": "train",
        "context": [{"role": "user", "content": "x" * 200}],
        "hindsight": [],
    }
    memory = {"total": 10000, "vram": []}
    if resource == "host":
        memory["total"] = 2500
    else:
        memory["vram"] = [{"memory_total": 2500}]
    slice_id = store.slice()
    admission = SampleAdmission(
        config,
        {"max_sequence": 512, "memory": memory},
        {"rows": rows},
        store,
        slice_id,
    )
    # Without a total-capacity verdict the gradient window simply trains.
    roomy = SampleAdmission(
        config, {"max_sequence": 512}, {"rows": rows}, store, slice_id
    )
    assert roomy.admit([step], Tokenizer()) == [step]
    assert not store.get("unfit_samples")
    assert admission.admit([step], Tokenizer()) == []
    assert resource in next(iter(store.get("unfit_samples").values()))[
        "reason"
    ].lower().replace("ram", "host")
    saved = store.get("unfit_samples")
    again = SampleAdmission(
        config,
        {"max_sequence": 128, "memory": memory},
        {"rows": rows},
        store,
        store.slice(),
    )

    class NoRetry(Tokenizer):
        def apply_chat_template(self, *args, **kwargs):
            pytest.fail("A known unfit revision must not be rendered again")

    assert again.admit([step], NoRetry()) == []
    assert store.get("unfit_samples") == saved
    assert again.record["unfit_samples"] == []


HEAD = [[0.3, -0.2, 0.7], [-0.4, 0.8, 0.1]]

STEP = {
    "context": [
        {"role": "system", "content": "Rules"},
        {"role": "user", "content": "Old question"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "t", "function": {"name": "read", "arguments": {}}}],
        },
        {"role": "tool", "tool_call_id": "t", "content": "OLD TOOL OUTPUT"},
        {"role": "user", "content": "Recent question"},
        {"role": "assistant", "content": "Historical answer"},
        {"role": "user", "content": "Question"},
    ],
    "hindsight": [{"role": "user", "content": "Correct answer"}],
}


def cpu_backend(kind, config, monkeypatch):
    """A production MLX or Torch backend over a positional stand-in trunk.

    Hidden states depend on every earlier token (a cumulative sum), so a
    prefix read into a cache must give the same forward as a full pass. The
    adapter weight moves only student states; base and teacher are frozen.
    Each trunk call records (mode, full sequence, tokens read in this call)."""
    import contextlib

    import numpy as np

    from songe.backends.mlx import Backend as MLXBackend
    from songe.backends.torch import TorchBackend
    from tests.test_learning import NumpyAPI, Tokenizer

    torch = pytest.importorskip("torch")
    cls = MLXBackend if kind == "mlx" else TorchBackend
    backend = cls.__new__(cls)
    backend.config, backend.tokenizer = config, Tokenizer()
    backend.check = lambda: None
    backend.sample = lambda prompt, seed: (
        [7] if seed == 1 else [7, 8, 9, 10],
        "",
        None,
    )
    backend.modes, backend.sequences, backend.entered = ["student"], [], []

    @contextlib.contextmanager
    def mode(name):
        backend.modes.append(name)
        backend.entered.append(name)
        try:
            yield
        finally:
            backend.modes.pop()

    backend.base = lambda: mode("base")
    backend.mode = mode
    head = np.array(HEAD, dtype=np.float32)

    def positional(ids, skip):
        backend.sequences.append((backend.modes[-1], list(ids), len(ids) - skip))
        position = np.cumsum(np.array(ids, dtype=np.float32)) / 1000
        value = np.stack([np.sin(position), np.cos(position)], axis=-1)[None]
        direction = np.stack([np.cos(position), -np.sin(position)], axis=-1)[None]
        if kind == "torch":
            value, direction = torch.from_numpy(value), torch.from_numpy(direction)
        weight = backend.model.weight if backend.modes[-1] == "student" else 0
        return (value + weight * direction)[:, skip:]

    if kind == "torch":
        backend.torch, backend.device, backend.trainable = (
            torch,
            torch.device("cpu"),
            True,
        )
        backend.model = torch.nn.Module()
        backend.model.weight = torch.nn.Parameter(torch.tensor(0.2))
        backend.head = lambda hidden: hidden @ torch.from_numpy(head)

        def base_model(input_ids, past_key_values=None, use_cache=False, **kwargs):
            prefix = list(past_key_values or [])
            ids = prefix + input_ids[0].tolist()
            return SimpleNamespace(
                last_hidden_state=positional(ids, len(prefix)),
                past_key_values=ids if use_cache else None,
            )

        backend._causal = lambda: SimpleNamespace(base_model=base_model)
        return backend

    class KVCache:
        def __init__(self):
            self.keys = self.values = None
            self.offset = 0

        def update_and_fetch(self, keys, values):
            if self.keys is not None:
                keys = np.concatenate([self.keys, keys], axis=2)
                values = np.concatenate([self.values, values], axis=2)
            self.keys, self.values = keys, values
            self.offset = keys.shape[2]
            return keys, values

        @property
        def state(self):
            return self.keys, self.values

    monkeypatch.setitem(
        sys.modules,
        "mlx_lm.models.cache",
        SimpleNamespace(
            KVCache=KVCache,
            ArraysCache=type("ArraysCache", (), {}),
            make_prompt_cache=lambda model: [KVCache()],
        ),
    )

    class CPUArrays(NumpyAPI):
        array = staticmethod(np.array)
        checkpoint = staticmethod(lambda fn: fn)
        eval = staticmethod(lambda *args: None)

    backend.mx = CPUArrays
    backend.model = SimpleNamespace(
        weight=0.2, training=False, train=lambda *a: None, eval=lambda: None
    )
    backend.head = lambda hidden: hidden @ head

    def trunk(ids, cache=None):
        new = np.asarray(ids, dtype=np.float32).reshape(1, 1, -1, 1)
        if cache is None:
            return positional(new.reshape(-1).astype(int).tolist(), 0)
        keys, _ = cache[0].update_and_fetch(new, new)
        return positional(
            keys.reshape(-1).astype(int).tolist(), keys.shape[2] - new.shape[2]
        )

    backend._text_model = lambda: SimpleNamespace(model=trunk)

    def value_and_grad(model, fn):
        def evaluate(model):
            weight, epsilon = model.weight, 0.005
            value = fn(model)
            model.weight = weight + epsilon
            plus = fn(model)
            model.weight = weight - epsilon
            minus = fn(model)
            model.weight = weight
            return value, {"weight": (plus - minus) / (2 * epsilon)}

        return evaluate

    backend.nn = SimpleNamespace(value_and_grad=value_and_grad)
    return backend


def run_steps(backend, step=STEP):
    loss, gradient, tokens = 0, 0, 0
    for seed in (1, 2):
        value, grad, count = backend.train_step(step, None, seed=seed)
        assert count == (1 if seed == 1 else 4)
        loss += value * count
        gradient += float(grad["weight"]) * count
        tokens += count
    return loss / tokens, gradient / tokens


@pytest.mark.parametrize("chunk", [1, 3])
def test_mlx_and_torch_training_token_counts_losses_and_gradients_match_on_cpu(
    config, monkeypatch, chunk
):
    """Execute both production loss paths; MLX's array API uses NumPy on CPU."""
    from songe.sleep.prompts import paired_prompts

    config = replace(config, token_chunk=chunk, response_tokens=4, teacher_chunk=4096)
    student, teacher = paired_prompts(
        cpu_backend("torch", config, monkeypatch).tokenizer, STEP, config
    )
    # The student reads the exact context; the teacher adds only hindsight.
    assert b"OLD TOOL OUTPUT" in bytes(student)
    assert bytes(teacher).startswith(bytes(student)[: bytes(student).rindex(b"]")])
    assert b"Correct answer" in bytes(teacher) and b"Correct answer" not in bytes(
        student
    )
    assert b"Respond now with your own action." in bytes(teacher)
    results = []
    for kind in ("mlx", "torch"):
        backend = cpu_backend(kind, config, monkeypatch)
        results.append(run_steps(backend))
        # Without a cycle snapshot the teacher is the frozen base, over the full
        # teacher prompt; the student's gradient pass covers the whole context.
        assert ("base", teacher, len(teacher)) in backend.sequences
        assert ("student", student, len(student)) in backend.sequences
        assert all(
            mode != "student" or skip == len(ids)
            for mode, ids, skip in backend.sequences
        )
    assert results[0][0] > 0 and abs(results[0][1]) > 1e-4
    assert results[0][0] == pytest.approx(results[1][0], abs=1e-6)
    assert results[0][1] == pytest.approx(results[1][1], abs=2e-5, rel=0.002)


@pytest.mark.parametrize("kind", ["mlx", "torch"])
def test_gradient_covers_context_tail_and_reads_earlier_context_frozen(
    config, monkeypatch, kind
):
    from songe.sleep.prompts import paired_prompts

    full = replace(config, token_chunk=3, response_tokens=4, teacher_chunk=4096)
    student, _ = paired_prompts(
        cpu_backend(kind, full, monkeypatch).tokenizer, STEP, full
    )
    expected = run_steps(cpu_backend(kind, full, monkeypatch))
    window = 60
    config = replace(full, max_sequence=window, teacher_chunk=64)
    backend = cpu_backend(kind, config, monkeypatch)
    # The forward is exact; this stand-in's gradient has no far-context term.
    assert run_steps(backend) == pytest.approx(expected, abs=1e-6, rel=1e-5)
    for seed, y in ((1, [7]), (2, [7, 8, 9, 10])):
        backend.sequences.clear()
        backend.train_step(STEP, None, seed=seed)
        students = [
            (ids, read) for mode, ids, read in backend.sequences if mode == "student"
        ]
        split = len(student) - (window - len(y))
        tail = student[split:] + y[:-1]
        # The gradient pass reads only the context tail plus response, on top
        # of the frozen state: together they fill the window exactly.
        gradient = (student[:split] + tail, len(tail))
        assert len(tail) + 1 == window and gradient in students
        # The earlier context is read once, with no gradient, in teacher_chunk pieces.
        prefix = [s for s in students if s != gradient]
        ends = list(range(64, split, 64)) + [split]
        assert prefix == [
            (student[:end], end - start)
            for start, end in zip([0] + ends, ends, strict=False)
        ]


@pytest.mark.parametrize("kind", ["mlx", "torch"])
def test_hindsight_steps_use_cycle_teacher_and_anchors_use_base(
    config, monkeypatch, kind
):
    from songe.sleep.prompts import paired_prompts

    config = replace(config, token_chunk=3, response_tokens=4, teacher_chunk=4096)
    backend = cpu_backend(kind, config, monkeypatch)
    backend.teacher = lambda: backend.mode("teacher")
    run_steps(backend)
    student, teacher = paired_prompts(backend.tokenizer, STEP, config)
    assert ("teacher", teacher, len(teacher)) in backend.sequences
    assert not any(
        mode == "base" and len(ids) >= len(student)
        for mode, ids, _ in backend.sequences
    )
    assert backend.entered.count("teacher") == 2
    anchor = {
        "context": [{"role": "user", "content": "General prompt"}],
        "hindsight": [],
    }
    backend = cpu_backend(kind, config, monkeypatch)
    backend.teacher = lambda: pytest.fail("Anchors keep the original base as teacher")
    run_steps(backend, anchor)
    prompt, same = paired_prompts(backend.tokenizer, anchor, config)
    assert prompt == same
    assert ("base", prompt, len(prompt)) in backend.sequences
    assert "teacher" not in backend.entered


def test_torch_teacher_is_the_cycle_start_snapshot():
    import contextlib

    from songe.backends.torch import TorchBackend

    torch = pytest.importorskip("torch")
    backend = TorchBackend.__new__(TorchBackend)
    backend.torch = torch
    backend.model = torch.nn.Linear(2, 1)
    backend.model.bias.requires_grad_(False)
    based = []

    @contextlib.contextmanager
    def base():
        based.append(True)
        yield

    backend.base = base
    with backend.teacher():
        pass
    assert based == [True]  # No snapshot: the original base teaches (cycle 1).
    backend.model.weight.data[:] = torch.tensor([[1.0, 2.0]])
    backend.snapshot_teacher()
    assert set(backend.teacher_weights) == {"weight"}
    backend.model.weight.data[:] = torch.tensor([[5.0, 6.0]])  # training moved on
    backend.model.train()
    with backend.teacher():
        assert backend.model.weight.tolist() == [[1.0, 2.0]]
        assert not backend.model.training
    assert backend.model.weight.tolist() == [[5.0, 6.0]]
    assert backend.model.training and based == [True]


def test_mlx_heldout_kl_reads_the_teacher_with_the_cycle_start_weights():
    import contextlib

    import numpy as np

    from songe.backends.mlx import Backend
    from tests.test_learning import NumpyAPI

    class Version:
        """The backend whose teacher is asked, as backends/mlx.py does it:
        teacher_hidden reads with the original base unless told otherwise."""

        config = SimpleNamespace(token_chunk=2)
        weights, read_with = "candidate", []

        @contextlib.contextmanager
        def using(self, weights):
            previous, self.weights = self.weights, weights
            try:
                yield
            finally:
                self.weights = previous

        def teacher(self):
            return self.using("cycle start")

        def base(self):
            return self.using("original base")

        def teacher_hidden(self, prompt, y, teacher=None):
            with teacher if teacher is not None else self.base():
                self.read_with.append(self.weights)
                return np.zeros((1, len(y), 3))

        def head(self, hidden):
            return hidden

    class API(NumpyAPI):
        eval = staticmethod(lambda *arrays: None)
        where, argmax = staticmethod(np.where), staticmethod(np.argmax)

    candidate = Backend.__new__(Backend)
    candidate.mx, candidate.check = API, lambda: None
    candidate.hidden = lambda ids: np.zeros((1, len(ids), 3))
    candidate.head = lambda hidden: hidden
    version = Version()
    assert candidate.kl_from(version, [1, 2], [1, 2, 3], [4, 5, 6], hindsight=True) == 0
    assert candidate.kl_from(version, [1, 2], [1, 2], [4, 5, 6]) == 0
    # Held-out hindsight KL: the version the cycle started from. Drift: the base.
    assert version.read_with == ["cycle start", "original base"]


@pytest.mark.parametrize("kind", ["torch", "mlx"])
def test_measure_shows_a_sharper_version_that_a_small_kl_hides(kind):
    """Two answer tokens over a three-token vocabulary. The candidate prefers
    the same tokens as the base, only more surely: the KL stays small, the
    entropy falls, the agreement stays 1. Then it prefers another token."""
    import contextlib
    import math

    import numpy as np

    base_logits = np.log(np.array([[[0.5, 0.3, 0.2], [0.6, 0.3, 0.1]]]))
    sharper = base_logits * 1.5
    other = base_logits[:, :, ::-1].copy()
    if kind == "torch":
        torch = pytest.importorskip("torch")
        from songe.backends.torch import TorchBackend as Backend

        tensor = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    else:
        from songe.backends.mlx import Backend
        from tests.test_learning import NumpyAPI

        class API(NumpyAPI):
            eval = staticmethod(lambda *arrays: None)
            where, argmax = staticmethod(np.where), staticmethod(np.argmax)

        tensor = lambda a: a  # noqa: E731

    class Base:
        config = SimpleNamespace(token_chunk=1)  # one token per chunk: sums, not means of means

        @contextlib.contextmanager
        def base(self):
            yield

        def teacher_hidden(self, prompt, y, teacher=None):
            return tensor(base_logits)

        def head(self, hidden):
            return hidden

    def measured(logits):
        candidate = Backend.__new__(Backend)
        candidate.check = lambda: None
        candidate.head = lambda hidden: hidden
        if kind == "torch":
            candidate.torch = torch
            candidate.config = SimpleNamespace(teacher_chunk=512)  # a short prompt: one pass
            candidate._response_hidden = lambda prompt, y: tensor(logits)
        else:
            candidate.mx = API
            candidate.hidden = lambda ids: logits  # one state per prompt + y[:-1] token
        m = candidate.measure(Base(), [1], [1], [4, 5])
        assert candidate.kl_from(Base(), [1], [1], [4, 5]) == pytest.approx(m["kl"])
        return m

    def entropy(logits):
        p = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
        return float(-(p * np.log(p)).sum(-1).mean())

    same = measured(base_logits)
    assert same["kl"] == pytest.approx(0, abs=1e-6) and same["agreement"] == 1
    assert same["entropy"] == pytest.approx(same["reference_entropy"]) == pytest.approx(
        entropy(base_logits), rel=1e-5)
    sure = measured(sharper)
    assert sure["agreement"] == 1 and sure["reference_entropy"] == pytest.approx(same["entropy"])
    assert sure["entropy"] == pytest.approx(entropy(sharper), rel=1e-5)
    # An eighth of the uncertainty is gone for a KL of 0.025 nats/token: half
    # the drift limit (general_drift_limit, 0.05), so the drift check passes.
    assert sure["entropy"] < 0.88 * sure["reference_entropy"]
    assert sure["kl"] == pytest.approx(0.0253, abs=1e-3)
    assert measured(other)["agreement"] == 0
    assert math.isfinite(measured(base_logits * 200)["entropy"])  # probabilities of 0


def test_mlx_teacher_is_the_cycle_start_snapshot(monkeypatch):
    import contextlib

    import numpy as np

    from songe.backends.mlx import Backend

    monkeypatch.setitem(
        sys.modules,
        "mlx.utils",
        SimpleNamespace(tree_flatten=lambda tree: list(tree.items())),
    )

    class Model:
        training = True
        weights = {"lora_a": np.array([1.0, 2.0])}

        def trainable_parameters(self):
            return dict(self.weights)

        def load_weights(self, items, strict=True):
            assert strict is False
            self.weights = dict(items)

        def train(self, mode=True):
            self.training = mode

        def eval(self):
            self.training = False

    backend = Backend.__new__(Backend)
    backend.mx = SimpleNamespace(array=np.array)
    backend.model = Model()
    based = []

    @contextlib.contextmanager
    def base():
        based.append(True)
        yield

    backend.base = base
    with backend.teacher():
        pass
    assert based == [True]
    backend.snapshot_teacher()
    backend.model.weights["lora_a"][:] = [5.0, 6.0]  # training moved on in place
    backend.model.weights = {"lora_a": np.array([7.0, 8.0])}
    with backend.teacher():
        assert backend.model.weights["lora_a"].tolist() == [1.0, 2.0]
        assert not backend.model.training
    assert backend.model.weights["lora_a"].tolist() == [7.0, 8.0]
    assert backend.model.training and based == [True]


def test_a_resumed_paged_optimizer_gets_paged_states_again(monkeypatch):
    """torch.load brings a paged state back flagged is_paged in ordinary
    memory; bitsandbytes would then prefetch it as managed memory and the
    process dies (Inkling-2-Mini, a Deep Dream resumed after a slice
    deadline)."""
    import sys
    from types import ModuleType, SimpleNamespace

    torch = pytest.importorskip("torch")
    from songe.backends.torch import repage

    made = []

    def get_paged(*shape, dtype, device):
        tensor = torch.zeros(*shape, dtype=dtype, device=device)
        tensor.is_paged, tensor.page_deviceid = True, 0
        made.append(tensor)
        return tensor

    functional = ModuleType("bitsandbytes.functional")
    functional.get_paged = get_paged
    monkeypatch.setitem(sys.modules, "bitsandbytes", ModuleType("bitsandbytes"))
    monkeypatch.setitem(sys.modules, "bitsandbytes.functional", functional)
    stale = torch.arange(6, dtype=torch.uint8)
    stale.is_paged = True  # as torch.load restores it
    small = torch.ones(3)
    optimizer = SimpleNamespace(state={"p": {"step": 2, "state1": stale, "absmax1": small}})
    repage(optimizer)
    state = optimizer.state["p"]
    assert state["state1"] is made[0] and torch.equal(state["state1"], stale)
    assert state["absmax1"] is small and state["step"] == 2

    # Without paged memory (no bitsandbytes build) the flag goes instead.
    monkeypatch.setitem(sys.modules, "bitsandbytes.functional", ModuleType("bitsandbytes.functional"))
    stale = torch.arange(6, dtype=torch.uint8)
    stale.is_paged = True
    optimizer = SimpleNamespace(state={"p": {"state1": stale}})
    repage(optimizer)
    assert optimizer.state["p"]["state1"] is stale and not stale.is_paged


def test_the_samples_of_a_step_are_answered_together_and_each_gets_its_own(
    config, tmp_path, monkeypatch
):
    pool = [{"id": name, "values": [0.1]} for name in ("a", "b", "c", "d", "e")]
    backend = scalar_backend(config, monkeypatch, "torch", 2)
    asked, given = [], []
    backend.rollouts = lambda steps, seeds: (
        asked.append(([s["id"] for s in steps], seeds))
        or [[ord(s["id"])] for s in steps]
    )
    microbatch = backend.train_step

    def train_step(step, optimizer, seed, rollout=None):
        given.append((step["id"], seed, rollout))
        return microbatch(step, optimizer, seed)

    backend.train_step = train_step
    train(backend, pool, [], private_dir(tmp_path / "work"), lambda *_: None)
    # One batch per optimizer step; a step of one sample is answered alone.
    assert asked == [(["a", "b"], [7613, 7614]), (["c", "d"], [7615, 7616])]
    assert given == [
        ("a", 7613, [97]),
        ("b", 7614, [98]),
        ("c", 7615, [99]),
        ("d", 7616, [100]),
        ("e", 7617, None),
    ]


def test_answers_made_together_are_the_answers_made_alone(tmp_path):
    pytest.importorskip("torch")
    from songe.backends.torch import TorchBackend
    from songe.sleep.prompts import paired_prompts
    from tests.crossplatform_helpers import tiny_model, torch_config

    config = replace(torch_config(tiny_model(tmp_path / "base")), response_tokens=12)
    config.torch_options = {**config.torch_options, "temperature": 0.0}
    backend = TorchBackend(config)
    steps = [
        {"context": [{"role": "user", "content": text}], "hindsight": [], "tools": []}
        for text in ("hi", "a much longer question than the others", "why?")
    ]
    seeds = [1, 2, 3]
    together = backend.rollouts(steps, seeds)
    alone = [
        backend.sample(paired_prompts(backend.tokenizer, s, config)[0], seed=k)[0]
        for s, k in zip(steps, seeds, strict=True)
    ]
    # Left padding is masked: each row reads only its own prompt.
    assert together == alone
    # A prompt longer than one read chunk is left to the one-at-a-time path.
    backend.read_chunk = 4
    assert backend.rollouts(steps, seeds) == [None, None, None]
