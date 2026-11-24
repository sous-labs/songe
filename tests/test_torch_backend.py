import math
from types import SimpleNamespace

import pytest

from songe.backends.reference import divergence as reference
from songe.backends.torch import (
    TorchBackend,
    divergence,
    needs_4bit,
    sampling_probabilities,
    select_device,
)
from songe.sleep.distill import divergence as mlx_divergence
from songe.sleep.prompts import paired_prompts
from tests.crossplatform_helpers import tiny_model, torch_config
from tests.test_learning import NumpyAPI

torch = pytest.importorskip("torch")
# CPU test fixtures are tiny; intra-op threading overhead dominates their work.
torch.set_num_threads(1)


@pytest.fixture(params=["llama", "qwen2"])
def backend(tmp_path, request):
    config = torch_config(tiny_model(tmp_path / request.param, request.param))
    return TorchBackend(config)


@pytest.fixture
def step():
    return {
        "context": [{"role": "user", "content": "Answer correct"}],
        "hindsight": [{"role": "user", "content": "wrong Answer"}],
        "tools": [],
    }


@pytest.mark.parametrize(
    "kind,clip",
    [("forward_kl", 0.0), ("forward_kl", 0.05), ("reverse_kl", 0.0), ("jsd", 0.0)],
)
@pytest.mark.parametrize("beta", [0.2, 0.5, 0.8])
@pytest.mark.parametrize("top_k", [0, 1, 3, 7, 12])
def test_objectives_match_framework_free_and_mlx_reference(kind, clip, beta, top_k):
    generator = torch.Generator().manual_seed(52)
    student = torch.randn(2, 7, generator=generator, requires_grad=True)
    teacher = torch.randn(2, 7, generator=generator, requires_grad=True)
    expected = reference(
        student.tolist(), teacher.tolist(), kind, beta, top_k, clip=clip
    )
    value = divergence(student, teacher, kind, beta, top_k, clip=clip)
    legacy = mlx_divergence(
        NumpyAPI,
        student.detach().numpy(),
        teacher.detach().numpy(),
        kind,
        beta,
        top_k,
        clip=clip,
    )
    assert value.item() == pytest.approx(expected, abs=4e-7)
    assert float(legacy) == pytest.approx(expected, abs=4e-7)
    value.backward()
    assert teacher.grad is None
    assert torch.isfinite(student.grad).all()
    # Finite differences check gradients rather than just the loss value.
    delta = 0.001
    plus, minus = student.detach().clone(), student.detach().clone()
    plus[0, 2] += delta
    minus[0, 2] -= delta
    numeric = (
        reference(plus.tolist(), teacher.tolist(), kind, beta, top_k, clip=clip)
        - reference(minus.tolist(), teacher.tolist(), kind, beta, top_k, clip=clip)
    ) / (2 * delta)
    assert student.grad[0, 2].item() == pytest.approx(numeric, abs=2e-5)


def test_forward_kl_is_the_default_and_clips_each_vocabulary_entry():
    import numpy as np

    s, t = [[0.0, -1.0, -3.0]], [[-2.0, 0.5, 0.0]]
    log_s = [[x - math.log(sum(math.exp(v) for v in s[0])) for x in s[0]]]
    log_t = [[x - math.log(sum(math.exp(v) for v in t[0])) for x in t[0]]]
    terms = [math.exp(b) * (b - a) for a, b in zip(log_s[0], log_t[0], strict=True)]
    assert max(terms) > 0.2 and min(terms) < 0  # the clip binds on one entry
    for clip, expected in [(0.0, sum(terms)), (0.2, sum(min(x, 0.2) for x in terms))]:
        assert reference(s, t, clip=clip) == pytest.approx(expected, rel=1e-9)
        # Torch computes in float32 (``.float()``), like the training path.
        assert divergence(
            torch.tensor(s), torch.tensor(t), clip=clip
        ).item() == pytest.approx(expected, rel=1e-6)
        assert float(
            mlx_divergence(NumpyAPI, np.array(log_s), np.array(log_t), clip=clip)
        ) == pytest.approx(expected, rel=1e-6)  # MLX objective also uses float32
    assert (
        reference(s, t, clip=0.2)
        < reference(s, t)
        == pytest.approx(reference(s, t, "forward_kl"))
    )


def test_objective_extreme_logits_and_tail_floor():
    s, t = [[100, -100, -100]], [[-100, 100, -100]]
    for kind in ["forward_kl", "reverse_kl", "jsd"]:
        assert divergence(
            torch.tensor(s), torch.tensor(t), kind, top_k=1
        ).item() == pytest.approx(reference(s, t, kind, top_k=1), rel=1e-6)


def test_device_selection_and_live_memory_policy():
    fake = SimpleNamespace(
        device=lambda value: value, cuda=SimpleNamespace(is_available=lambda: True)
    )
    assert select_device(fake) == "cuda"  # Includes ROCm's CUDA-shaped API.
    fake.cuda.is_available = lambda: False
    fake.xpu = SimpleNamespace(is_available=lambda: True)
    with pytest.raises(ValueError, match="telemetry"):
        select_device(fake, "xpu")
    fake.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True))
    assert select_device(fake) == "mps"
    fake.backends.mps.is_available = lambda: False
    assert select_device(fake) == "cpu"
    assert select_device(fake, "rocm") == "cuda"
    assert not needs_4bit(100, 1000, 0.5)
    assert needs_4bit(100, 300, 0.5)
    with pytest.raises(ValueError, match="measured free memory"):
        needs_4bit(100, 0, 0.5)


def test_calibration_samples_full_response_even_if_eos_is_first(backend):
    from songe.sleep.calibrate import full_length_sampler

    generation = backend._causal().generation_config
    old = generation.eos_token_id
    generation.eos_token_id = list(range(backend._causal().config.vocab_size))
    try:
        assert len(backend.sample([1, 2], seed=1)[0]) == 1
        assert (
            len(full_length_sampler(backend)([1, 2], seed=1)[0])
            == backend.config.response_tokens
        )
        assert generation.eos_token_id == list(
            range(backend._causal().config.vocab_size)
        )
    finally:
        generation.eos_token_id = old


def test_cpu_requires_explicit_training_choice(tmp_path):
    config = torch_config(tiny_model(tmp_path / "model"))
    config.torch_options["force_cpu"] = False
    with pytest.raises(ValueError, match="serve-only"):
        TorchBackend(config)
    backend = TorchBackend(config, train=False)
    assert backend.sample([3], seed=1)[0]


def test_sampling_seed_base_and_interruption(backend):
    first = backend.sample([3, 8], seed=91)
    second = backend.sample([3, 8], seed=91)
    assert first[0] == second[0]
    assert first[2].generation_tokens == len(first[0])
    assert first[2].generation_tps > 0
    backend.model.train()
    with pytest.raises(RuntimeError), backend.base():
        assert not backend.model.training
        raise RuntimeError("stop")
    assert backend.model.training
    backend.model.eval()
    backend.check = lambda: (_ for _ in ()).throw(RuntimeError("interrupt"))
    with pytest.raises(RuntimeError, match="interrupt"):
        backend.sample([3])
    assert not backend.model.training


@pytest.mark.parametrize("kind", ["forward_kl", "reverse_kl", "jsd"])
@pytest.mark.parametrize("top_k", [0, 3])
def test_checkpointed_chunk_gradient_matches_dense_response_only(
    backend, step, monkeypatch, kind, top_k
):
    backend.config.loss, backend.config.beta, backend.config.teacher_top_k = (
        kind,
        0.3,
        top_k,
    )
    backend.config.vocab_clip = 1e-8  # binds on this random model's tiny divergences
    y = [7, 8, 9]
    monkeypatch.setattr(backend, "sample", lambda *args, **kwargs: (y, "", None))
    student, teacher = paired_prompts(backend.tokenizer, step, backend.config)
    with backend.base():
        expected_teacher = backend.model(
            input_ids=torch.tensor([teacher + y[:-1]], device=backend.device),
            use_cache=False,
        ).logits[:, len(teacher) - 1 :]
    backend.model.train()
    logits = backend.model(
        input_ids=torch.tensor([student + y[:-1]], device=backend.device),
        use_cache=False,
    ).logits[:, len(student) - 1 :]
    # Every response token counts equally: no learnability weights.
    expected = divergence(logits, expected_teacher, kind, 0.3, top_k, clip=1e-8)
    unclipped = divergence(logits, expected_teacher, kind, 0.3, top_k)
    if kind == "forward_kl":
        assert expected.item() < unclipped.item()
    else:
        assert expected.item() == unclipped.item()  # The clip is forward-KL only.
    params = [(n, p) for n, p in backend.model.named_parameters() if p.requires_grad]
    gradients = torch.autograd.grad(expected, [p for _, p in params])
    backend.model.eval()
    value, actual, tokens = backend.train_step(step, seed=8)
    assert tokens == len(y)
    assert value == pytest.approx(expected.item(), rel=2e-5, abs=1e-7)
    for (name, _), gradient in zip(params, gradients, strict=True):
        torch.testing.assert_close(actual[name], gradient, atol=2e-7, rtol=3e-4)
    with backend.base():
        h = backend._response_hidden(teacher, y)
        expected_logprob = (
            expected_teacher.log_softmax(-1)
            .gather(-1, torch.tensor(y, device=backend.device)[None, :, None])
            .mean()
            .item()
        )
        assert backend.token_logprob(h, y) == pytest.approx(expected_logprob, abs=1e-6)
    # Unclipped divergences are nonnegative; the clip bounds only positive
    # per-vocabulary terms, so the clipped objective itself may dip below zero.
    backend.config.vocab_clip = 0.0
    assert backend.compare(student, teacher, y, kind) >= -1e-6


@pytest.mark.parametrize("checkpointing", [True, False])
def test_frozen_prefix_keeps_the_exact_forward_with_a_short_gradient_window(
    backend, step, monkeypatch, checkpointing
):
    # Gradient checkpointing is the production default (TorchOptions).
    assert backend.options.gradient_checkpointing
    if not checkpointing:
        backend.model.gradient_checkpointing_disable()
    y = [7, 8, 9]
    monkeypatch.setattr(backend, "sample", lambda *args, **kwargs: (y, "", None))
    student, _ = paired_prompts(backend.tokenizer, step, backend.config)
    prefixes = []
    original = backend.prefix_state
    monkeypatch.setattr(
        backend, "prefix_state", lambda ids: prefixes.append(ids) or original(ids)
    )
    full, full_grads, tokens = backend.train_step(step, seed=8)
    assert prefixes == [] and tokens == len(y)
    backend.config.max_sequence, backend.config.teacher_chunk = len(y) + 4, 5
    windowed, grads, tokens = backend.train_step(step, seed=8)
    # Only the last four context tokens plus the response carry gradient; the
    # context before them is still read, so the student sees what the teacher
    # sees and the loss is the full-context one. That holds under gradient
    # checkpointing too, where transformers alone would drop the prefix.
    assert prefixes == [student[: len(student) - 4]] and tokens == len(y)
    assert windowed == pytest.approx(full, rel=1e-4, abs=1e-7)
    assert grads.keys() == full_grads.keys()
    assert all(torch.isfinite(g).all() for g in grads.values())
    assert any(g.abs().sum() > 0 for g in grads.values())


def test_checkpointing_reads_the_frozen_prefix_once(backend, step, monkeypatch):
    y = [7, 8, 9]
    monkeypatch.setattr(backend, "sample", lambda *args, **kwargs: (y, "", None))
    backend.config.max_sequence, backend.config.teacher_chunk = len(y) + 4, 5
    student, _ = paired_prompts(backend.tokenizer, step, backend.config)
    caches = []
    original = backend.prefix_state
    monkeypatch.setattr(
        backend, "prefix_state", lambda ids: caches.append(original(ids)) or caches[-1]
    )
    loss, grads, _ = backend.train_step(step, seed=8)
    # The backward pass runs each layer again; the prefix must not grow.
    assert caches[0].get_seq_length() == len(student) - 4
    assert not backend._window_only
    assert all(
        "_can_checkpoint_with_cache" not in vars(m) for m in backend.model.modules()
    )
    backend.model.gradient_checkpointing_disable()
    plain_loss, plain, _ = backend.train_step(step, seed=8)
    assert loss == pytest.approx(plain_loss, rel=1e-5, abs=1e-7)
    for name, grad in grads.items():
        assert torch.allclose(grad, plain[name], rtol=1e-4, atol=1e-7), name


def test_a_prefix_that_cannot_be_frozen_leaves_the_window(backend, step, monkeypatch):
    y = [7, 8, 9]
    monkeypatch.setattr(backend, "sample", lambda *args, **kwargs: (y, "", None))
    full, _, _ = backend.train_step(step, seed=8)
    backend.config.max_sequence, backend.config.teacher_chunk = len(y) + 4, 5
    prefixes = []
    # A recurrent state, say: not a key-value cache the layers can read twice.
    monkeypatch.setattr(
        backend, "prefix_state", lambda ids: prefixes.append(ids) or object()
    )
    windowed, grads, _ = backend.train_step(step, seed=8)
    assert backend._window_only and windowed != pytest.approx(full, rel=1e-4)
    assert all(torch.isfinite(g).all() for g in grads.values())
    backend.train_step(step, seed=8)
    assert len(prefixes) == 1  # not read again for nothing


def test_lora_scale_follows_alpha_over_rank(backend):
    scales = {
        module.scaling["default"]
        for module in backend.model.modules()
        if isinstance(getattr(module, "scaling", None), dict)
    }
    assert scales == {backend.config.lora_scale} == {2.0}


def test_update_frozen_teacher_checkpoint_and_deltas(backend, step, tmp_path):
    with backend.base():
        before = backend.head(backend.hidden([3, 7])).clone()
    optimizer = backend.optimizer(lambda iteration: 0.01 / (iteration + 1))
    loss, grads, tokens = backend.train_step(step, optimizer, seed=12)
    assert tokens > 0
    assert math.isfinite(loss)
    assert any(g.abs().sum().item() > 0 for g in grads.values())
    backend.apply(grads)
    with backend.base():
        torch.testing.assert_close(
            backend.head(backend.hidden([3, 7])), before, atol=0, rtol=0
        )
    assert sum(backend.layer_deltas()) > 0
    saved = tmp_path / "adapter"
    backend.save(saved)
    clone = TorchBackend(backend.config)
    clone.restore(saved)
    torch.testing.assert_close(
        backend.head(backend.hidden([3, 7])), clone.head(clone.hidden([3, 7]))
    )
    bad = {name: torch.full_like(g, float("nan")) for name, g in grads.items()}
    with pytest.raises(ValueError, match="Non-finite"):
        backend.apply(bad)


def test_heldout_kl_is_measured_against_the_cycle_start_teacher(backend, step):
    # A second Dream: the version it starts from already has an adapter.
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for p in backend.model.parameters():
            if p.requires_grad:
                p.add_(0.05 * torch.randn(p.shape, generator=generator))
    backend.snapshot_teacher()
    student, _ = paired_prompts(backend.tokenizer, step, backend.config)
    y = [7, 8, 9]
    # The candidate is still the teacher and reads the same prompt: no
    # distance to the cycle's teacher, a real one to the original base.
    assert backend.kl_from(backend, student, student, y, hindsight=True) == pytest.approx(0, abs=1e-6)
    assert backend.kl_from(backend, student, student, y) > 1e-4
    assert backend.compare(student, student, y) == pytest.approx(0, abs=1e-6)


def test_measure_gives_entropy_and_agreement_with_the_kl(backend, step):
    student, _ = paired_prompts(backend.tokenizer, step, backend.config)
    y = [7, 8, 9, 10, 11]
    backend.config.token_chunk = 2  # chunks of 2, 2 and 1 tokens
    same = backend.measure(backend, student, student, y)
    assert same["kl"] == pytest.approx(0, abs=1e-6) and same["agreement"] == 1
    assert 0 < same["entropy"] == pytest.approx(same["reference_entropy"], rel=1e-5)
    with backend.base():
        logits = backend.head(backend.hidden(student + y[:-1])[:, -len(y):]).float()
    expected = torch.special.entr(logits.softmax(-1)).sum(-1).mean().item()
    assert same["reference_entropy"] == pytest.approx(expected, rel=1e-5)
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for p in backend.model.parameters():
            if p.requires_grad:
                p.add_(0.05 * torch.randn(p.shape, generator=generator))
    moved = backend.measure(backend, student, student, y)
    assert moved["kl"] == pytest.approx(backend.kl_from(backend, student, student, y)) and moved["kl"] > 1e-4
    assert moved["reference_entropy"] == pytest.approx(same["reference_entropy"], rel=1e-5)
    assert moved["entropy"] != same["entropy"]  # the version's own, not the base's
    assert 0 <= moved["agreement"] <= 1


def test_nucleus_filter_precedes_temperature_and_retains_boundary_ties():
    logits = torch.tensor([[0.5, 0.3, 0.2]]).log()
    actual = sampling_probabilities(logits, temperature=0.5, top_p=0.6)
    torch.testing.assert_close(actual, torch.tensor([[0.25 / 0.34, 0.09 / 0.34, 0.0]]))
    tied = torch.tensor([[0.4, 0.3, 0.3]]).log()
    torch.testing.assert_close(sampling_probabilities(tied, 1.0, 0.5), tied.exp())


def test_restore_rejects_incomplete_adapter_without_modifying_model(backend, tmp_path):
    from safetensors.torch import load_file, save_file

    path = tmp_path / "adapter"
    backend.save(path)
    file = path / "adapter_model.safetensors"
    weights = load_file(file)
    weights.pop(next(iter(weights)))
    save_file(weights, file)
    before = {
        n: p.detach().clone()
        for n, p in backend.model.named_parameters()
        if p.requires_grad
    }
    with pytest.raises(ValueError, match="does not match"):
        backend.restore(path)
    for name, parameter in backend.model.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)


def test_module_import_does_not_initialize_torch_or_mlx():
    import subprocess
    import sys

    code = "import sys; import songe.backends.torch, songe.backends.reference, songe.models, songe.artifacts; assert 'torch' not in sys.modules; assert 'mlx' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


def test_teacher_gain_scores_only_explicit_logged_success(backend):
    assert backend.teacher_gain([{"hindsight": []}])["examples"] == 0
    step = {
        "context": [{"role": "user", "content": "Answer"}],
        "answer_hint": {"role": "assistant", "content": "correct"},
        "hindsight": [{"role": "tool", "content": '{"exit_code": 0}'}],
    }
    result = backend.teacher_gain([step])
    assert result["examples"] == 1 and result["rate"] in {0.0, 1.0}


def test_effective_delta_scaling_and_per_cycle_norm(backend):
    baseline = backend.adapter_weights()
    with torch.no_grad():
        for _name, param in backend.model.named_parameters():
            if param.requires_grad:
                param.add_(torch.randn_like(param) * 0.1)
    trained = backend.adapter_weights()
    backend.scale_delta(baseline, trained, 0)
    assert backend.layer_deltas(baseline) == pytest.approx([0, 0], abs=1e-6)
    backend.scale_delta(baseline, trained, 1)
    assert sum(backend.layer_deltas(baseline)) > 0
    backend.scale_delta(baseline, trained, 0.5)
    actual = backend.adapter_weights()
    for name, a1 in trained.items():
        if ".lora_A." not in name:
            continue
        b = name.replace(".lora_A.", ".lora_B.")
        desired = 0.5 * (baseline[b] @ baseline[name] + trained[b] @ a1)
        # Initial B=0 means the interpolated delta fits the original rank exactly.
        torch.testing.assert_close(
            actual[b] @ actual[name], desired, atol=1e-6, rtol=1e-5
        )


def test_half_strength_keeps_narrow_layer_shapes():
    # Qwen3.5's in_proj_a/b have 32 outputs at rank 64: the merged delta has
    # fewer components than the rank, and the factors must keep their shapes.
    torch.manual_seed(0)
    rank, outputs, inputs = 8, 4, 16
    model = torch.nn.Module()
    model.layer = torch.nn.Module()
    model.layer.lora_A = torch.nn.ModuleDict(
        {"default": torch.nn.Linear(inputs, rank, bias=False)}
    )
    model.layer.lora_B = torch.nn.ModuleDict(
        {"default": torch.nn.Linear(rank, outputs, bias=False)}
    )
    a, b = "layer.lora_A.default.weight", "layer.lora_B.default.weight"
    baseline = {a: torch.randn(rank, inputs), b: torch.randn(outputs, rank)}
    trained = {a: torch.randn(rank, inputs), b: torch.randn(outputs, rank)}
    fake = SimpleNamespace(torch=torch, model=model)
    TorchBackend.scale_delta(fake, baseline, trained, 0.5)
    params = dict(model.named_parameters())
    assert params[a].shape == (rank, inputs)
    assert params[b].shape == (outputs, rank)
    desired = 0.5 * (baseline[b] @ baseline[a] + trained[b] @ trained[a])
    torch.testing.assert_close(params[b] @ params[a], desired, atol=1e-5, rtol=1e-4)


def test_train_resume_preserves_optimizer_units(
    backend, step, tmp_path, writable_file_fsync
):
    from songe.backends import Interrupted

    backend.config.learning_rate = 0.01
    work = tmp_path / "work"
    work.mkdir()
    pool = [step | {"id": "experience"}, step | {"id": "second"}]

    def interrupted(iteration, budget, live=None):
        if iteration == 1:
            raise Interrupted("activity")

    with pytest.raises(Interrupted):
        backend.train(pool, [], work, interrupted)
    import json

    first = json.loads((work / "checkpoint.json").read_text(encoding="utf-8"))
    assert first["iteration"] == 1 and "token_divergences" not in first
    clone = TorchBackend(backend.config)
    units = []
    result = clone.train(pool, [], work, lambda i, *_: units.append(i))
    assert units == [2] and clone._updates == 2
    assert result["steps_learned"] == 2
    uninterrupted = tmp_path / "uninterrupted"
    uninterrupted.mkdir()
    control = TorchBackend(backend.config)
    control.train(pool, [], uninterrupted, lambda *args: None)
    for name, value in clone.adapter_weights().items():
        torch.testing.assert_close(
            value, control.adapter_weights()[name], atol=0, rtol=0
        )


def test_full_sleep_worker_on_cpu_with_random_hf_model(tmp_path, monkeypatch):
    from songe.log import Store
    from songe.sleep import worker
    from tests.test_schedule import good
    from tests.test_steps import row

    config = torch_config(tiny_model(tmp_path / "base"))
    config.backend, config.upstream, config.public_model = "torch", "vllm", "songe/test"
    config.force_cpu_training = True
    config.calibration_available = True
    config.min_train, config.min_heldout, config.min_new = 3, 3, 1
    config.learning = "hindsight"  # this test trains on the recorded steps
    store = Store(tmp_path / "data")
    for i in range(35):
        context = [{"role": "user", "content": f"Calculate {i}"}]
        wrong = {"role": "assistant", "content": "Wrong"}
        for source in [
            row(i * 2, context, wrong),
            row(
                i * 2 + 1,
                context
                + [wrong, {"role": "user", "content": "Actually it should be four"}],
                {"role": "assistant", "content": "Four"},
            ),
        ]:
            store.record(
                source["path"], source["request"], source["response"], binding="base"
            )
    monkeypatch.setattr(worker, "probe", lambda: good())
    monkeypatch.setattr(
        worker,
        "current",
        lambda: SimpleNamespace(
            name="test",
            memory=lambda: __import__("songe.platform", fromlist=["Memory"]).Memory(
                10**9, 9 * 10**8
            ),
            accelerators=lambda: [],
        ),
    )
    monkeypatch.setattr(worker.signal, "signal", lambda *args: None)
    anchors = worker.general_steps()[:1]
    monkeypatch.setattr(worker, "general_steps", lambda: anchors)
    result = worker.run(config, store)
    assert result["iterations"] == result["samples_trained"]
    assert result["iterations"] > 1
    assert math.isfinite(result["loss"])
    assert result["scale_evaluations"]
    assert store.nights()[0]["status"] in {"ready", "rejected"}
    assert store.slices()[0]["ended"] is not None
    work = store.root / "versions/night-1"
    assert (work / "adapter/adapter_model.safetensors").is_file()
    assert list(work.glob("before-*.json"))
    assert list(work.glob("scale-1.0-*.json"))
    if result["accepted"]:
        candidate = result["candidate"]
        assert candidate["kind"] == "peft-adapter"
        assert (
            __import__("pathlib").Path(candidate["artifact"])
            / "adapter_model.safetensors"
        ).is_file()


def test_big_model_memory_savers_stay_off_for_a_bf16_base(backend):
    # A base that fits in bf16 keeps the plain setup: no host offloads and
    # AdamW (the savers are for 4-bit bases, where every gigabyte counts).
    assert backend.quantized is False
    assert backend.embedding_on_host is False
    assert backend._auto("auto") is False
    assert backend._auto("on") is True and backend._auto("off") is False
    with pytest.raises(ValueError, match="auto, on or off"):
        backend._auto("sometimes")
    assert isinstance(backend.optimizer(1e-3), backend.torch.optim.AdamW)


def test_unknown_optimizer_is_refused(backend):
    backend.options.optimizer = "sgd"
    with pytest.raises(ValueError, match="Unknown optimizer"):
        backend.optimizer(1e-3)


def test_teacher_snapshot_lives_in_host_ram_and_round_trips(backend):
    backend.snapshot_teacher()
    assert all(w.device.type == "cpu" for w in backend.teacher_weights.values())
    params = {n: p for n, p in backend.model.named_parameters() if p.requires_grad}
    before = {n: p.detach().clone() for n, p in params.items()}
    with torch.no_grad():
        for p in params.values():
            p.add_(1.0)  # the student moves on during the Dream
    moved = {n: p.detach().clone() for n, p in params.items()}
    with backend.teacher():
        for n, p in params.items():
            torch.testing.assert_close(p.detach().cpu(), before[n].cpu())
    for n, p in params.items():
        torch.testing.assert_close(p.detach(), moved[n])
