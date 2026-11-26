"""Exact-context student / hindsight teacher contracts without model or accelerator work."""

import contextlib
import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from songe.backends import Interrupted
from songe.memory import Snapshot
from songe.sleep.calibration_state import measure_pool, pending_keys
from songe.sleep.evaluate import references
from songe.sleep.prompts import paired_prompts, response_positions
from songe.sleep.samples import SampleAdmission
from songe.sleep.select import bounded_batch, replay_pool
from songe.sleep.sizing import choose_size
from tests.test_data_cycles import cycle as cycle
from tests.test_learning import Tokenizer
from tests.test_partial_calibration import measured as measured


def experience():
    return {
        "id": "real",
        "source_id": 7,
        "conversation": "conversation",
        "split": "train",
        "context": [
            {
                "role": "system",
                "content": "Keep user files.\n\nRead before editing.\n\nKeep user files.",
            },
            {"role": "user", "content": "Old task " * 600},
            {"role": "assistant", "content": "Old answer"},
            {"role": "user", "content": "Previous subtask"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c",
                        "type": "function",
                        "function": {"name": "read", "arguments": {"path": "a"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c", "content": "file contents"},
            {"role": "user", "content": "Current subtask"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": name, "description": "schema " * 150},
            }
            for name in ("read", "edit", "verify", "unused")
        ],
        "action": {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "edit", "arguments": {}}}],
        },
        "hindsight": [{"role": "tool", "content": "verify reported missing argument"}],
        "answer_hint": {"role": "assistant", "content": "Privileged answer"},
    }


def decode(ids):
    return json.loads(bytes(ids).decode("utf-8"))


def test_gate_uses_original_full_context(config):
    step = experience() | {"split": "heldout"}
    seen = []
    backend = SimpleNamespace(
        tokenizer=Tokenizer(),
        config=replace(config, max_sequence=128),
        check=lambda: None,
        base=contextlib.nullcontext,
        sample=lambda prompt, seed: (seen.append(prompt) or [1], "", None),
    )
    refs = references(backend, [step])
    assert decode(seen[0])["messages"] == step["context"]
    assert decode(seen[0])["tools"] == step["tools"]
    assert refs[0]["prompt"] == seen[0]


def test_gate_references_are_reused_for_the_same_base(config, tmp_path):
    from songe.sleep.evaluate import ReferenceCache

    step = experience() | {"split": "heldout"}
    seen = []
    backend = SimpleNamespace(
        tokenizer=Tokenizer(),
        config=replace(config, max_sequence=128),
        check=lambda: None,
        base=contextlib.nullcontext,
        sample=lambda prompt, seed: (seen.append(prompt) or [7, 8], "", None),
    )
    root = tmp_path / "gate-references"
    first = references(backend, [step], cache=ReferenceCache(root, "base-a" * 4))
    # A later Dream on the same base samples nothing and gets the same answer.
    again = references(backend, [step], cache=ReferenceCache(root, "base-a" * 4))
    assert len(seen) == 1 and again[0]["y"] == first[0]["y"] == [7, 8]
    # New base weights: sampled again, and the old base's answers are removed.
    references(backend, [step], cache=ReferenceCache(root, "base-b" * 4))
    assert len(seen) == 2
    assert [p.name for p in root.iterdir()] == [("base-b" * 4)[:24]]


def test_sizing_and_admission_separate_backprop_from_full_teacher(config, store):
    step, tokenizer = experience(), Tokenizer()
    student, teacher = paired_prompts(tokenizer, step, config, check_length=False)
    config = replace(config, max_sequence=len(student) + config.response_tokens)
    store.sync_steps([step])
    samples, target, _ = measure_pool(config, store, tokenizer)
    assert target == len(student) + config.response_tokens
    assert samples[0]["teacher_sequence"] == len(teacher) + config.response_tokens
    assert samples[0]["teacher_sequence"] > target
    admitted = SampleAdmission(
        config, {"max_sequence": target}, {}, store, store.slice()
    )
    assert admitted.admit([step], tokenizer) == [step]
    limited = SampleAdmission(
        config,
        {"max_sequence": target, "max_teacher_sequence": len(teacher)},
        {},
        store,
        store.slice(),
    )
    with pytest.raises(Interrupted, match="deferred"):
        limited.admit([step], tokenizer)
    assert not store.get("unfit_samples")
    # Gate forwards are measured independently of the backprop length.
    full = SampleAdmission(
        config,
        {
            "max_sequence": target,
            "max_teacher_sequence": len(teacher) + config.response_tokens,
        },
        {},
        store,
        store.slice(),
    )
    assert full.admit([step], tokenizer, full_context=True) == [step]


def test_batch_cost_bounds_and_sparse_revisions_leave_the_rest_pending(config, store):
    config = replace(config, slice_limit=10, accumulation=4, anchor_fraction=0.1)
    # Batches follow pending, then id.
    pool = [experience() | {"id": str(i), "source_id": i} for i in range(1, 11)]
    chosen, remaining = bounded_batch(pool, config, seconds=10, step_seconds=10)
    # 4 samples per optimizer step, 10% rehearsal reserved.
    assert len(chosen) == 3 and len(remaining) == 7
    assert [s["id"] for s in chosen] == ["1", "10", "2"]
    assert len(bounded_batch(pool, config, seconds=1e9, step_seconds=1)[0]) == 10
    assert len(bounded_batch(pool, config, seconds=None, step_seconds=0)[0]) == 4
    # A Dream plans over all its practice, whatever the first gap's length,
    # up to slice_limit.
    assert len(bounded_batch(pool, config, seconds=10, step_seconds=10, whole=True)[0]) == 10
    assert len(bounded_batch(pool, replace(config, slice_limit=6), 10, 10, whole=True)[0]) == 6
    store.sync_steps(pool)
    store.add_pool(pool)
    store.mark_scored(pool)
    practiced = {s["id"]: s["source_id"] for s in chosen}
    store.set("practice_revisions", practiced)
    pending = pending_keys(store)
    assert len(pending) == 7
    assert store.pending_learnable() == 7
    replayed = replay_pool(pool, 0, 0.25, practiced)
    second, _ = bounded_batch(replayed, config, 10, 10, pending=pending)
    assert [s["id"] for s in second] == ["3", "4", "5"]


def test_bounded_cycles_finish_every_revision_without_starvation(cycle, config, store):
    from songe.sleep import worker

    model, pool = cycle
    config.accumulation = 1
    config.replay_fraction = 0
    results = [worker.run(config, store) for _ in range(3)]
    assert [set(r["practice_revisions"]) for r in results] == [
        {"1"},
        {"1", "2"},
        {"1", "2", "3"},
    ]
    assert [r["watermark"] for r in results] == [1, 2, 3]
    assert len(store.pool()) == len(pool)
    assert store.pending_learnable() == 0
    assert worker.run(config, store)["reason"] == "Waiting for new learnable experience"


@pytest.mark.parametrize("phase", ["teacher", "summary"])
@pytest.mark.parametrize("resource", ["host", "device"])
def test_forward_phase_peak_cannot_shrink_with_student_context(config, phase, resource):
    config = replace(config, max_sequence=512, response_tokens=32, token_chunk=1)
    rows = [
        {
            "sequence": sequence,
            "response_tokens": 32,
            "token_chunk": 1,
            "host_peak_bytes": peak if resource == "host" else 100,
            "device_peak_bytes": peak if resource == "device" else 0,
            "teacher_sequence": 7000,
            "phase_peaks": {phase: {resource + "_peak_bytes": peak}},
        }
        for sequence, peak in [(64, 1000), (512, 5000)]
    ]
    from songe.platform import Accel

    devices = (Accel("cuda", "fake", 10000, 4000),) if resource == "device" else ()
    sample = Snapshot(10000, 4000, vram=devices)
    assert choose_size(config, {"rows": rows}, sample, {}) is None
    devices = (Accel("cuda", "fake", 10000, 8000),) if resource == "device" else ()
    sample = Snapshot(10000, 8000, vram=devices)
    chosen = choose_size(config, {"rows": rows}, sample, {})
    assert chosen[resource + "_bytes"] >= 5000
    assert chosen["max_teacher_sequence"] == 7000


def test_calibration_measures_full_teacher_separately_from_gradient_window(
    config, store, measured
):
    from songe.sleep.calibrate import _calibrate
    from tests.test_partial_calibration import GB
    from tests.test_partial_calibration import Tokenizer as SizingTokenizer

    store.forget()
    step = experience()
    # This fake tokenizer counts the first message: a long old user turn makes
    # the exact context longer than the gradient window (max_sequence).
    step["context"] = step["context"][1:]
    store.sync_steps([step])
    student, teacher = paired_prompts(
        SizingTokenizer(), step, config, check_length=False
    )
    rows = _calibrate(config, store.root, (256, 512, 1024))
    saved = store.get("calibration")
    # The student is the exact context; only its gradient window is capped.
    assert len(student) + config.response_tokens > config.max_sequence
    assert saved["progress"]["target_sequence"] == config.max_sequence
    assert (
        saved["progress"]["teacher_target_sequence"]
        == len(teacher) + config.response_tokens
    )
    assert (
        saved["progress"]["target_sequence"]
        < saved["progress"]["teacher_target_sequence"]
    )
    new = [r for r in rows if r.get("teacher_sequence")]
    assert new and all(
        {"summary", "teacher", "train"} <= r["phase_peaks"].keys() for r in new
    )
    # Training probes stay small; the full teacher has its own ladder.
    assert all(r["teacher_prompt_tokens"] < len(teacher) for r in new)
    assert max(r["teacher_sequence"] for r in saved["teacher_rows"]) >= len(teacher)
    sizing = choose_size(config, saved, Snapshot(24 * GB, 20 * GB), {})
    assert sizing["max_teacher_sequence"] >= len(teacher) + config.response_tokens
    assert sizing["max_sequence"] < len(teacher)


@pytest.mark.parametrize("prompt_length,response_length", [(1, 1), (20, 4), (150, 7)])
def test_loss_positions_exclude_all_prompt_roles(prompt_length, response_length):
    prompt, response = list(range(prompt_length)), list(range(response_length))
    mask = np.zeros(prompt_length + response_length - 1, dtype=bool)
    mask[response_positions(prompt, response)] = True
    assert mask.sum() == response_length
    assert not mask[: prompt_length - 1].any()
    assert mask[prompt_length - 1 :].all()


@pytest.mark.parametrize("kind", ["mlx", "torch"])
def test_chunked_teacher_keeps_full_context_and_only_response_states(
    config, monkeypatch, kind
):
    from songe.backends.mlx import Backend
    from songe.backends.torch import TorchBackend

    cls = Backend if kind == "mlx" else TorchBackend
    backend = cls.__new__(cls)
    backend.config = replace(config, teacher_chunk=3)
    backend.check = lambda: None
    frozen, chunks = [], []

    @contextlib.contextmanager
    def base():
        frozen.append(True)
        try:
            yield
        finally:
            frozen.pop()

    backend.base = base
    if kind == "mlx":

        class Cache:
            state = 0

        monkeypatch.setitem(
            sys.modules,
            "mlx_lm.models.cache",
            SimpleNamespace(make_prompt_cache=lambda model: [Cache()]),
        )
        backend.mx = SimpleNamespace(
            array=np.array,
            eval=lambda *a: None,
            stop_gradient=lambda a: a.copy(),
            concatenate=np.concatenate,
        )

        def trunk(ids, cache):
            assert frozen
            chunks.append(ids.tolist()[0])
            hidden = np.cumsum(ids, axis=1) + cache[0].state
            cache[0].state = hidden[0, -1]
            return hidden[..., None]

        backend._text_model = lambda: SimpleNamespace(model=trunk)
    else:
        torch = pytest.importorskip("torch")
        backend.torch, backend.device = torch, "cpu"

        def trunk(input_ids, past_key_values, **kwargs):
            assert frozen and not torch.is_grad_enabled()
            chunks.append(input_ids.tolist()[0])
            hidden = input_ids.cumsum(1) + (past_key_values or 0)
            return SimpleNamespace(
                last_hidden_state=hidden[..., None], past_key_values=int(hidden[0, -1])
            )

        backend._causal = lambda: SimpleNamespace(base_model=trunk)
        # Production base() also disables autograd; keep this stand-in faithful.
        original_base = backend.base

        @contextlib.contextmanager
        def torch_base():
            with original_base(), torch.no_grad():
                yield

        backend.base = torch_base
    prompt, response = list(range(1, 11)), [20, 30, 40, 50]
    hidden = backend.teacher_hidden(prompt, response)
    if kind == "torch":
        hidden = hidden.numpy()
    assert hidden.flatten().tolist() == np.cumsum(prompt + response[:-1])[9:].tolist()
    assert [t for chunk in chunks for t in chunk] == prompt + response[:-1]
    assert max(map(len, chunks)) <= 3
    assert not frozen


def test_the_users_own_rules_make_a_capped_dream_first(config):
    """Live: 172 lessons to practise and a cap of 200 scenes; the user's rules,
    the newest, were last in line."""
    config = replace(config, slice_limit=3)
    pool = [experience() | {"id": f"practice:{i}", "source_id": i, "stated": i in (8, 9)} for i in range(1, 10)]
    chosen, _ = bounded_batch(pool, config, seconds=None, step_seconds=0, whole=True)
    assert {s["id"] for s in chosen} >= {"practice:8", "practice:9"}


def test_a_capped_dream_takes_rows_from_every_lesson(config):
    """Live (Qwen3.5-4B on a rented card, 11 lessons, 670 rows, the limit of
    512): cut by id, the three lessons whose ids sort last lost all of their
    142 scenes, and after the Dream the model said "Yes, chef." on 0 to 1 of
    16 test requests, as before it. Over the limit, rows are taken in turn
    from each lesson."""
    rows = {"a": 6, "b": 2, "c": 4}
    pool = [
        experience() | {"id": f"practice:{lesson}#v{i}", "source_id": i, "conversation": f"signal:{lesson}"}
        for lesson, count in rows.items()
        for i in range(count)
    ]

    def kept(limit, pool=pool):
        chosen, left = bounded_batch(pool, replace(config, slice_limit=limit), None, 0, whole=True)
        assert len(chosen) + len(left) == len(pool)
        return {lesson: sum(s["conversation"] == f"signal:{lesson}" for s in chosen) for lesson in rows}

    assert kept(6) == {"a": 2, "b": 2, "c": 2}  # by id alone: all six of a
    assert kept(9) == {"a": 4, "b": 2, "c": 3}  # a lesson with fewer rows keeps all it has
    assert kept(12) == rows
    # The user's own rules still come first, whatever their lesson.
    stated = [s | {"stated": s["conversation"] == "signal:a"} for s in pool]
    assert kept(7, stated) == {"a": 6, "b": 1, "c": 0}


def test_prompts_are_rendered_the_way_the_users_server_sends_them(store):
    """Live (vLLM serving Qwen3.5-4B with thinking off): the template's
    default starts every answer inside <think>, so a Deep Dream trained on
    128 tokens of reasoning that this server never produces, and its check
    compared two pieces of reasoning."""
    from songe.sleep import prompts

    class Template:
        def apply_chat_template(self, messages, **kwargs):
            self.switches = {k: v for k, v in kwargs.items() if k == "enable_thinking"}
            return [1, 2, 3]

    def record(message):
        store.record(
            "/v1/chat/completions",
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            {"choices": [{"message": {"role": "assistant"} | message}]},
        )

    def switches(**own):
        prompts.serve_as(prompts.served_switches(store))
        prompts.render(tokenizer, [{"role": "user", "content": "x"}], None, **own)
        return tokenizer.switches

    tokenizer = Template()
    try:
        # Nothing recorded yet: the template's default.
        assert switches() == {}
        # The server answers directly, in text or with a tool call.
        record({"content": "4"})
        call = {"id": "c", "type": "function", "function": {"name": "read", "arguments": "{}"}}
        record({"content": "", "tool_calls": [call]})
        assert switches() == {"enable_thinking": False}
        # A caller's own switch still wins.
        assert switches(enable_thinking=True) == {"enable_thinking": True}
        # The server reasons first (either field name, or <think> text).
        record({"content": "4", "reasoning": "two and two"})
        assert switches() == {}
    finally:
        prompts.serve_as({})
    assert prompts.reasoned({"reasoning_content": "plan"})
    assert prompts.reasoned({"content": "<think>plan</think>4"})
    assert not prompts.reasoned({"content": "4", "reasoning": " "})


def test_each_model_is_rendered_by_its_own_recorded_answers(store):
    """One installation, a model that reasons and one that answers directly:
    read together, the last twenty answers of both decided for each."""
    from songe.sleep.prompts import served_switches

    def record(binding, message):
        store.record(
            "/v1/chat/completions",
            {"model": "any", "messages": [{"role": "user", "content": "hi"}]},
            {"choices": [{"message": {"role": "assistant"} | message}]},
            binding=binding,
        )

    store.set("primary_model", "thinker")
    thinker, plain = store.scoped("thinker"), store.scoped("plain")
    thinker.add_version({"id": "base", "identifier": "thinker", "artifact": "", "adapter": None, "created": 0, "night": 0})
    plain.add_version({"id": "night-3", "identifier": "plain-3", "artifact": "", "adapter": None, "created": 0, "night": 3})
    record("base", {"content": "4", "reasoning": "two and two"})
    record("plain:base", {"content": "4"})
    record("night-3", {"content": "5"})
    assert served_switches(thinker) == {}
    assert served_switches(plain) == {"enable_thinking": False}
    # The unscoped view of a one-model installation reads every answer.
    assert served_switches(store) == {}
    # A model with no answer of its own: the template's default.
    assert served_switches(store.scoped("new")) == {}


def test_streamed_answers_from_before_the_reasoning_was_kept_decide_nothing(tmp_path):
    """Before this version a streamed vLLM answer was recorded without its
    reasoning. Right after an upgrade those recordings must not turn
    reasoning off for a model that reasons: a streamed answer with no
    reasoning from before the store's mark is unknown, not direct."""
    from songe.log import Store
    from songe.sleep.prompts import served_switches

    def record(store, message, stream):
        store.record(
            "/v1/chat/completions",
            {"model": "m", "stream": stream, "messages": [{"role": "user", "content": "hi"}]},
            {"choices": [{"message": {"role": "assistant"} | message}]},
        )

    def before_the_mark(root, *answers):
        old = Store(root)
        with old.connect() as db:  # As a version without the mark left it.
            db.execute("DELETE FROM state WHERE key='streamed_reasoning_from'")
        for message, stream in answers:
            record(old, message, stream)
        return Store(root)

    upgraded = before_the_mark(tmp_path / "streamed", ({"content": "4"}, True), ({"content": "5"}, True))
    assert served_switches(upgraded) == {}
    # What it records from now on counts.
    record(upgraded, {"content": "6"}, True)
    assert served_switches(upgraded) == {"enable_thinking": False}
    record(upgraded, {"content": "7", "reasoning": "three and four"}, True)
    assert served_switches(upgraded) == {}
    # An old answer that was not streamed never lost anything, and old reasoning is still reasoning.
    assert served_switches(before_the_mark(tmp_path / "whole", ({"content": "4"}, False))) == {"enable_thinking": False}
    kept = before_the_mark(tmp_path / "kept", ({"content": "4", "reasoning_content": "two and two"}, True))
    record(kept, {"content": "5"}, True)
    assert served_switches(kept) == {}
    # A new store has nothing from before.
    fresh = Store(tmp_path / "fresh")
    record(fresh, {"content": "4"}, True)
    assert served_switches(fresh) == {"enable_thinking": False}

