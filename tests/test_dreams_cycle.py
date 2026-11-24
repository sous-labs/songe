"""A dreams cycle: trains on kept practice only, gates on held-out real
conversations, and measures the drift set. No models, accelerators or servers."""

import json
from types import SimpleNamespace

import pytest

from songe.platform import Accel
from songe.practice import pool as practice_pool
from songe.schedule import Scheduler
from songe.sleep import worker
from songe.sleep.evaluate import drift_choice
from songe.sleep.gate import Decision
from tests.test_schedule import good
from tests.test_worker import Backend


class Model(Backend):
    scored, trained_units, initial_adapters = [], [], []
    comparisons = 0
    interrupt_training = interrupt_evaluation = False
    pools = []

    def train(self, pool, anchors, work, progress):
        type(self).pools.append([s["id"] for s in pool])
        return super().train(pool, anchors, work, progress)


def add_practice(store, n, start=0):
    with store.connect() as db:
        for i in range(start, start + n):
            row = {
                "id": f"sig{i}#v1",
                "signal_id": f"sig{i}",
                "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": f"scene {i}"}],
                "tools": [],
                "teacher_note": "Note from an earlier, different task (it may not apply here).",
            }
            db.execute(
                "INSERT INTO practice(id,model_id,signal_id,payload,kept,\"drop\",created) VALUES (?,?,?,?,1,NULL,0)",
                (row["id"], store.model_id, row["signal_id"], json.dumps(row)),
            )


def add_success(store, session, exchange_messages):
    exchange = store.record("/v1/chat/completions", {"messages": exchange_messages}, {"choices": []})
    signal = {"id": f"ok-{exchange}", "session": session, "type": "success", "confirmed": True,
              "exchanges": {"3": exchange}}
    with store.connect() as db:
        db.execute("INSERT INTO signals VALUES (?,?,?,?,?,NULL)",
                   (signal["id"], store.model_id, session, 1, json.dumps(signal)))


@pytest.fixture
def dreams(config, store, monkeypatch):
    from songe.sleep import gate

    Model.pools, Model.trained_units = [], []
    config.learning = "dreams"
    config.upstream, config.calibration_available = "mlx", True
    config.min_train = config.min_heldout = config.min_new = 1
    config.slice_limit = 100
    # A real training step and a held-out one: the first must never be
    # trained on under dreams; the second gates.
    real = {"id": "real", "source_id": 5, "conversation": "real", "split": "train",
            "context": [{"role": "user", "content": "Fix this"}],
            "hindsight": [{"role": "user", "content": "correction"}], "tools": []}
    held = real | {"id": "held", "conversation": "held", "split": "heldout"}
    store.sync_steps([real, held])
    add_practice(store, 3)
    monkeypatch.setattr(worker, "probe", lambda: good())
    monkeypatch.setattr(worker, "current", lambda: SimpleNamespace(
        name="darwin", accelerators=lambda: [Accel("metal", "stand-in")]))
    monkeypatch.setattr(worker, "choose", lambda *a: Model)
    monkeypatch.setattr(worker.signal, "signal", lambda *a: None)
    anchors = worker.general_steps()[:1]
    monkeypatch.setattr(worker, "general_steps", lambda: anchors)
    def reject(scaled, before, *a, **kw):
        before()  # as a candidate that reaches its rows: the incumbent's are measured then
        return None, Decision(False, "Test rejection"), {}

    monkeypatch.setattr(gate, "choose_scale", reject)
    return config, store


def test_a_dreams_cycle_trains_on_practice_only(dreams):
    config, store = dreams
    result = worker.run(config, store)
    assert result["iterations"] > 0
    (trained,) = Model.pools
    assert sorted(trained) == ["practice:sig0#v1", "practice:sig1#v1", "practice:sig2#v1"]
    assert "real" not in trained
    # Recorded so an adopted Dream can take these lessons out of the prompt.
    assert store.nights()[-1]["record"]["trained_signals"] == ["sig0", "sig1", "sig2"]
    assert store.get("learning_mode") == "dreams"
    # Rejected by the gate, but the scenes count as practised.
    assert worker.run(config, store)["reason"] == "Waiting for new learnable experience"
    assert practice_pool.untrained(store) == []


def test_practice_beyond_the_cap_waits_for_the_next_dream(dreams):
    """A Dream trains at most slice_limit scenes; the rest stay untrained even
    when ids and revisions run in different orders (live run: the watermark
    jumped past 135 untrained scenes)."""
    config, store = dreams
    add_practice(store, 9, start=3)  # sig3..sig11; ids sort sig1, sig10, sig11, sig2...
    config.slice_limit = 4
    total = len(practice_pool.practice_steps(store))  # a fifth are held out
    worker.run(config, store)
    (trained,) = Model.pools
    assert len(trained) == 4
    left = {s["id"] for s in practice_pool.untrained(store)}
    assert len(left) == total - 4 and not left & set(trained)
    # The next Dream trains from what is left.
    worker.run(config, store)
    assert not set(Model.pools[1]) & set(trained)


async def test_new_practice_starts_the_next_cycle(dreams):
    config, store = dreams
    worker.run(config, store)
    scheduler = Scheduler(config, store, None)
    assert not await scheduler.enough()
    add_practice(store, 1, start=10)
    assert await scheduler.enough()
    worker.run(config, store)
    # New first; older scenes come back only as replay.
    assert "practice:sig10#v1" in Model.pools[-1]


def test_switching_from_hindsight_does_not_hide_practice(dreams):
    config, store = dreams
    # An installation that trained on real steps: its watermark counts
    # exchange ids, far above practice row numbers.
    store.set("experience_watermark", 5000)
    store.set("practice_revisions", {"x": 4000})
    assert len(practice_pool.untrained(store)) == 3
    worker.run(config, store)
    assert len(Model.pools[-1]) == 3


def test_the_cycle_records_which_drift_measure_gated(dreams):
    config, store = dreams
    result = worker.run(config, store)
    (check,) = result["drift_checks"].values()
    assert check["source"] == "general" and check["reason"] == "no confirmed successes yet"
    # Confirmed successes form the drift set: measured, but too few to gate.
    store.set("experience_watermark", 0)
    add_practice(store, 1, start=20)
    for i in range(2):
        add_success(store, f"s{i}", [{"role": "user", "content": f"run the tests {i}"}])
    result = worker.run(config, store)
    (check,) = result["drift_checks"].values()
    assert check["source"] == "general" and check["conversations"] == 2
    assert check["reason"] == f"2 of {config.min_drift} conversations"
    assert check["drift_set"] is not None


def test_drift_set_gates_only_when_big_and_narrow_enough(config):
    refs = [{"conversation": f"c{i}"} for i in range(12)]
    config.min_drift, config.general_drift_limit = 10, 0.05
    value, info = drift_choice(0.01, [0.02] * 12, refs, config)
    assert (value, info["source"]) == (0.02, "drift")
    noisy = [0.0, 0.3] * 6
    value, info = drift_choice(0.01, noisy, refs, config)
    assert (value, info["source"]) == (0.01, "general")
    assert info["reason"] == "interval wider than the drift limit"
    value, info = drift_choice(0.01, [0.02] * 5, refs[:5], config)
    assert info["source"] == "general" and info["reason"] == "5 of 10 conversations"


def test_gate_skips_bundled_prompts_when_the_drift_set_decides(config):
    """The drift set is measured first; the 150 bundled prompts only when it
    cannot decide. A version identical to the base is not measured for drift."""
    from songe.sleep.evaluate import evaluate

    config.min_drift, config.general_drift_limit = 10, 0.05
    calls = []

    class Fake:
        def __init__(self):
            self.config, self.tokenizer = config, None

        def check(self):
            pass

        def kl_from(self, base, prompt, teacher, y, *, hindsight=False):
            calls.append(prompt)
            return 0.01

    candidate = Fake()
    drift_refs = [{"conversation": f"c{i}", "prompt": f"d{i}", "y": []} for i in range(12)]
    general = [{"prompt": f"g{i}", "y": []} for i in range(150)]
    cache = {}
    assert evaluate(candidate, candidate, [], general, drift_refs=drift_refs, checkpoint=cache) == []
    assert calls == [f"d{i}" for i in range(12)]  # bundled prompts skipped
    assert cache["drift_check"]["source"] == "drift" and cache["drifts"] == []
    # This backend measures the KL only: nothing is kept beside it.
    from songe.sleep.evaluate import sharpness

    assert cache["drift_set_shapes"] == [None] * 12
    assert sharpness(cache) == {"base_entropy": None, "entropy": None, "entropy_change": None, "agreement": None}

    # Too few successes: the bundled prompts gate, and are measured.
    calls.clear()
    cache = {}
    evaluate(candidate, candidate, [], general, drift_refs=drift_refs[:5], checkpoint=cache)
    assert len(calls) == 5 + 150 and cache["drift_check"]["source"] == "general"
    # The first Dream's incumbent is the base: drift zero, nothing measured.
    calls.clear()
    cache = {}
    evaluate(candidate, candidate, [], general, drift_refs=drift_refs[:5],
             checkpoint=cache, identical=True)
    assert calls == [] and set(cache["drifts"]) == {0.0} and set(cache["drift_set"]) == {0.0}


class Answering(Model):
    """Answers differently once trained; the teacher prefers the trained
    answer. Enough for before/after examples, without a model."""

    def sample(self, prompt, seed=None, *, deterministic=False, max_tokens=None):
        trained = self.weight > 0
        return ([2] if trained else [1]), ("read the test file first" if trained else "edit it"), None

    class Teacher:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def teacher(self):
        return self.Teacher()

    def teacher_hidden(self, prompt, y, teacher=None):
        return isinstance(teacher, self.Teacher), y

    def token_logprob(self, h, y):
        hinted, y = h
        # The teacher (with the hint) prefers the trained answer; without the
        # hint both answers look alike.
        return (0.0 if y == [2] else -1.0) if hinted else -0.5

    def compare(self, student, teacher, y, kind="forward_kl"):
        return 0.1 if y == [2] else 0.3

def test_the_check_keeps_entropy_and_agreement_beside_the_drift():
    """What a mean KL hides is kept for the set that gated; it gates nothing."""
    from types import SimpleNamespace

    from songe.sleep.evaluate import evaluate, sharpness

    class Fake:
        config = SimpleNamespace(min_drift=10, gate_confidence=0.95, general_drift_limit=0.05)
        measured = []

        def check(self):
            pass

        def measure(self, base, prompt, teacher, y, *, hindsight=False):
            self.measured.append(prompt)
            # The base's entropy, this version's (a fifth lower), agreement.
            return {"kl": 0.01, "reference_entropy": 2.0, "entropy": 1.6, "agreement": 0.9}

    candidate = Fake()
    drift_refs = [{"conversation": f"c{i}", "prompt": f"d{i}", "y": []} for i in range(3)]
    general = [{"prompt": f"g{i}", "y": []} for i in range(4)]
    # Resumed: the first drift prompt was measured before these were kept.
    cache = {"drift_set": [0.01]}
    evaluate(candidate, candidate, [], general, drift_refs=drift_refs, checkpoint=cache)
    assert candidate.measured == ["d1", "d2", "g0", "g1", "g2", "g3"]
    assert cache["drift_set"] == [0.01] * 3 and cache["drifts"] == [0.01] * 4
    assert cache["drift_set_shapes"] == [None, [2.0, 1.6, 0.9], [2.0, 1.6, 0.9]]
    assert cache["drifts_shapes"] == [[2.0, 1.6, 0.9]] * 4
    # Three conversations cannot gate yet: the general prompts did, so theirs.
    assert cache["drift_check"]["source"] == "general"
    assert sharpness(cache) == {"base_entropy": 2.0, "entropy": 1.6,
                                "entropy_change": pytest.approx(-0.2), "agreement": 0.9}
    # The first Dream's incumbent is the base itself: not measured.
    cache = {}
    evaluate(candidate, candidate, [], general, drift_refs=drift_refs, checkpoint=cache, identical=True)
    assert cache["drifts_shapes"] == [None] * 4 and sharpness(cache)["entropy"] is None



def add_lesson_signal(store, signal_id, source):
    exchange = store.record("/v1/chat/completions", {"messages": [
        {"role": "system", "content": "s"}, {"role": "user", "content": f"fix the bug ({signal_id})"}]},
        {"choices": []})
    signal = {"id": signal_id, "session": f"sess-{signal_id}", "type": "correction" if source == "user" else "tool_error",
              "source": source, "lesson": "Read the failing test before editing.", "text": "No, wrong file.",
              "exchanges": {"2": exchange}}
    with store.connect() as db:
        db.execute("INSERT INTO signals VALUES (?,?,?,?,?,NULL)",
                   (signal_id, store.model_id, signal["session"], 1, json.dumps(signal)))


def test_examples_show_what_the_candidate_does_differently(dreams, monkeypatch):
    from songe.cycle import model_learning

    config, store = dreams
    # The older example step: it runs only when the gate cannot judge moments.
    from songe.sleep import judged as judged_module
    monkeypatch.setattr(judged_module, "moments", lambda heldout: [])
    monkeypatch.setattr(worker, "choose", lambda *a: Answering)
    # The starting version, asked which answer follows the lesson, says the new one.
    from songe import quick
    monkeypatch.setattr(quick, "verdict_of", lambda text, swap: "after")
    add_lesson_signal(store, "tool-sig", "tool")
    add_lesson_signal(store, "user-sig", "user")
    add_practice(store, 1, start=8)  # sig8 is held out from training
    result = worker.run(config, store)
    assert "examples_error" not in result
    examples = result["examples"]
    # Originals first (user signals before tool ones), then practice.
    assert [e["split"] for e in examples][:2] == ["original", "original"]
    assert examples[0]["signal"]["source"] == "user"
    assert {e["split"] for e in examples} == {"original", "held_out", "trained"}
    first = examples[0]
    assert first["before"]["text"] == "edit it" and first["after"]["text"] == "read the test file first"
    assert first["teacher_preference"] == 1.0 and first["lesson_preference"] == 1.0
    assert first["improved"]
    # The headline counts originals and held-out practice, never trained.
    assert result["improved"] == {"count": 2, "checked": 3, "available": 3}
    assert result["example_checks"]["new_practice"]["after"] == pytest.approx(0.1)
    # The trained scenes never include the held-out one.
    assert "practice:sig8#v1" not in Model.pools[-1]

    learning = model_learning(config, store, "m")
    last = learning["last"]
    assert last["state"] == "done" and last["result"]["outcome"] == "kept"
    # Judged better, but prose only (no tool call on either side): a Deep
    # Dream answer cut at a few hundred tokens shows nothing there.
    assert last["result"]["examples"] == []
    # (the practice scene here has no lesson to judge against)
    assert sum(e["improved"] for e in result["examples"] if e["split"] != "trained") == 2
    held_out = next(c for c in last["checks"] if c["key"] == "held_out")
    assert held_out["gates"] is True and held_out["before"] is not None
    general = next(c for c in last["checks"] if c["key"] == "general")
    assert general["gates"] is True  # no drift set yet: the anchors gate
    assert last["read"]["signals"]["total"] == 2 and last["train"]["samples"]["practice"] == 3

    # Forget keeps the counts and drops every quoted word.
    store.forget()
    kept = store.nights()[-1]["record"]["examples"]
    assert all(e["before"]["text"] == "" and e["context"]["request"] is None for e in kept)
    assert sum(e["improved"] for e in kept if e["split"] != "trained") == 2


def test_a_failing_example_step_never_fails_the_cycle(dreams):
    config, store = dreams
    add_lesson_signal(store, "user-sig", "user")
    # The plain test backend cannot answer: the examples fail, the cycle does not.
    result = worker.run(config, store)
    assert result["examples_error"] and result["iterations"] > 0


async def test_the_api_serves_cycles(dreams, monkeypatch):
    import httpx

    from songe.sleep import judged as judged_module
    monkeypatch.setattr(judged_module, "moments", lambda heldout: [])  # the older example step

    from songe.models import library, registry
    from songe.proxy import create_app

    config, store = dreams
    registry.migrate(config, store)
    registry.observe(config, store, "learner")
    registry.set_learning(store, "learner", "all")
    learner = store.scoped("learner")
    add_practice(learner, 3)
    monkeypatch.setattr(worker, "choose", lambda *a: Answering)
    add_lesson_signal(learner, "user-sig", "user")
    worker.run(config, learner)
    monkeypatch.setattr(library, "scan", lambda config: [])
    app = create_app(config, store, scheduler=SimpleNamespace(reason="idle"))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
        state = (await client.get("/api/state")).json()
        # Keyed by public model id, as the rest of /api/state.
        learning = state["cycle"]["models"]["songe/learner"]
        assert learning["last"]["id"] == "c-0001" and learning["current"] is None
        assert learning["last"]["result"]["improved"]["checked"] == 1
        listed = (await client.get("/api/cycles", params={"model": "songe/learner"})).json()
        assert [c["id"] for c in listed] == ["c-0001"]
        full = (await client.get("/api/cycles/c-0001")).json()
        assert full["result"]["all_examples"] and full["signals"][0]["id"] == "user-sig"
        assert (await client.get("/api/cycles/c-0099")).status_code == 404


def test_the_gate_shows_its_progress_while_checking(dreams):
    """Live run: the checks took over an hour and the page showed "0 of 6
    checks" throughout. The gate now counts conversations measured."""
    from songe.cycle import night_cycle

    config, store = dreams
    config.example_limit = 0  # examples report their own progress after the gate
    worker.run(config, store)
    night = store.nights()[-1]
    progress = night["record"]["data_progress"]
    assert progress["phase"] == "check" and progress["unit"] == "measured"
    # References and the current version measured; the total also counts
    # one strength (not run here: choose_scale is stubbed).
    assert progress["completed"] * 3 == progress["total"] * 2 > 0
    # While the night is still checking, the stage shows that count.
    store.night(night["id"], status="running", phase="check")
    running = night_cycle(config, store, store.nights()[-1], "m")
    (check,) = [s for s in running["stages"] if s["key"] == "check"]
    assert check["progress"] == {
        "done": progress["completed"], "total": progress["total"], "unit": "measured"
    }


def test_examples_report_answers_compared_after_the_gate(dreams, monkeypatch):
    """Live run: the page sat at "N of N measured" for 10-15 minutes while
    the before/after examples ran."""
    config, store = dreams
    # The older example step: it runs only when the gate cannot judge moments.
    from songe.sleep import judged as judged_module
    monkeypatch.setattr(judged_module, "moments", lambda heldout: [])
    monkeypatch.setattr(worker, "choose", lambda *a: Answering)
    add_lesson_signal(store, "user-sig", "user")
    assert "examples_error" not in worker.run(config, store)
    progress = store.nights()[-1]["record"]["data_progress"]
    assert progress["unit"] == "compared"
    assert progress["completed"] == progress["total"] > 0


def test_a_slice_deadline_is_not_shown_as_an_error(dreams):
    """Live run: between two check slices the page said "Stopped by an error
    · run songe doctor"; the night was only between slices."""
    from songe.cycle import night_cycle

    config, store = dreams
    worker.run(config, store)
    night = store.nights()[-1]
    store.night(night["id"], status="interrupted", phase="check",
                record=night["record"] | {"reason": "slice deadline"})
    cycle = night_cycle(config, store, store.nights()[-1], "m")
    assert cycle["wait"] is None and cycle["state"] == "running"
    store.night(night["id"], record=night["record"] | {"reason": "Waiting for memory: x"})
    cycle = night_cycle(config, store, store.nights()[-1], "m")
    assert cycle["wait"]["code"] == "memory" and cycle["state"] == "waiting"


def test_each_stop_shows_its_own_wait():
    """The service names why it stopped a slice; activity is not memory."""
    from songe.cycle import wait_of

    codes = {
        reason: (wait_of(reason) or {}).get("code")
        for reason in (
            "agent activity", "memory guard", "paused", "thermal pressure",
            "battery low", "hardware probe failed", "stopped by the service",
            "slice deadline", "proxy restart",
        )
    }
    assert codes == {
        "agent activity": "agent_active", "memory guard": "memory", "paused": "paused",
        "thermal pressure": "thermal", "battery low": "power",
        "hardware probe failed": "error", "stopped by the service": None,
        "slice deadline": None, "proxy restart": None,
    }


def test_the_last_result_says_whether_its_version_is_still_in_use(dreams):
    """Live run: after Undo the page still described v1's Dream as current."""
    from songe.cycle import model_learning

    config, store = dreams
    worker.run(config, store)
    night = store.nights()[-1]
    record = night["record"] | {"heldout_change": 0.0024, "noise_margin": 0.0017}
    store.night(night["id"], status="accepted", record=record)
    base = {"id": "base", "identifier": "m", "artifact": "/m", "adapter": None,
            "kind": "base", "created": 1, "night": 0}
    store.add_version(base)
    store.add_version(base | {"id": "night-1", "kind": "adapter", "adapter": "/a",
                              "created": 2, "night": night["id"]})
    store.swap("night-1", None)
    result = model_learning(config, store, "m")["last"]["result"]
    assert result["in_use"] is True and result["version"] == "v1"
    assert result["held_out"] == {"change": 0.0024, "margin": 0.0017}
    store.swap("base", "night-1")  # Undo
    assert model_learning(config, store, "m")["last"]["result"]["in_use"] is False


def test_a_pressed_deep_dream_ends_in_a_night_or_a_note(store):
    """Live: a pressed Deep Dream's training run found it had to measure
    again, the press was used up, and the Dream vanished without a word."""
    import time as _time
    from types import SimpleNamespace

    from songe.schedule import ModelScheduler

    fake = SimpleNamespace(installation_store=store)
    learner = store.scoped("m")
    launch = {"model": "m", "requested": True, "nights": 0, "at": _time.time() - 1}

    def end(**kw):
        args = {"calibrating": False, "failed": False, "interrupted": False, "stopped_by": None, "memory_stop": False} | kw
        ModelScheduler.carry_on(fake, "m", launch, **args)

    # Training found it must measure first: asked again, once, measuring next.
    learner.set("worker_outcome", {"reason": "Waiting for calibration growth", "at": _time.time()})
    end()
    assert store.get("requested_sleep") is True and learner.get("growth_requested") is True
    # The measuring run finishes: asked again so training follows.
    store.set("requested_sleep", False)
    end(calibrating=True)
    assert store.get("requested_sleep") is True and learner.get("growth_requested") is None
    # Training says so a second time: no loop, the note says why.
    store.set("requested_sleep", False)
    end()
    stopped = learner.get("dream_stopped")
    assert store.get("requested_sleep") is False and stopped["reason"] == "training"
    assert "calibration growth" in stopped["detail"] and "kept" in stopped
    # Cancelled or paused by the user: nothing more.
    learner.set("dream_stopped", None)
    end(interrupted=True, stopped_by="cancelled")
    assert learner.get("dream_stopped") is None and store.get("requested_sleep") is False


def test_a_kept_dream_says_which_check_it_failed():
    """Live: every strength failed "Repetition rate increased", the night said
    "No adapter scale passed the real gate", and the page showed "no gain"."""
    from songe.cycle import result

    night = {"status": "rejected", "record": {
        "reason": "No adapter scale passed the real gate", "heldout_change": 0.027, "noise_margin": 0.007,
        "scale_evaluations": {"1.0": {"reason": "Repetition rate increased"}, "0.5": {"reason": "Repetition rate increased"}}}}
    assert result(night, True)["reason"] == "repetition_rate"  # the older rule, said as such
    night["record"]["scale_evaluations"]["1.0"]["reason"] = "Held-out improvement does not exceed bootstrap noise"
    assert result(night, True)["reason"] == "no_gain"
    night["record"] = {"reason": "Something new"}
    assert result(night, True)["reason"] == "other"


def test_the_gate_judges_what_each_version_writes(monkeypatch):
    """What the version in use and the candidate each write at a held-out
    moment is compared: the same action ties; a different one goes to the
    base model's blind judgement; the report's pairs are the moments won."""
    import contextlib

    from songe.sleep import judged

    monkeypatch.setattr(judged, "paired_prompts", lambda tok, step, config: ([1], [2]))
    monkeypatch.setattr(judged, "render", lambda tok, messages, tools, **kw: messages[0]["content"])
    verdicts = {}

    class Fake:
        tokenizer, config, weight = None, None, 0.0

        def base(self):
            return contextlib.nullcontext()

        def sample(self, prompt, seed=None, *, deterministic=False, max_tokens=None):
            if isinstance(prompt, str):  # the judge's question
                new_is_a = prompt.index("Answer A:") < prompt.index("pnpm add") < prompt.index("Answer B:")
                return [], verdicts.get("say", '{"follows": "%s"}' % ("A" if new_is_a else "B")), None
            text = '{"name": "shell", "arguments": {"command": "%s"}}' % (
                "pnpm add lodash" if self.weight else "npm install lodash")
            return [self.weight], text, None

    monkeypatch.setattr(judged, "answer", lambda text, tok, tools: {
        "text": "", "tool_calls": [{"name": "shell", "arguments": {"command": text.split('"command": "')[1].split('"')[0]}}],
        "valid": True})

    class Journal:
        def unit(self, name, fn):
            return fn()

    steps = [{"id": f"s{i}", "conversation": f"c{i}", "context": [{"role": "user", "content": "add lodash"}],
              "hindsight": [{"role": "user", "content": "use pnpm here, never npm"}], "practice": i == 2,
              "lesson": "Always use pnpm." if i == 2 else None, "signal_id": "sig2" if i == 2 else None}
             for i in range(3)]
    backend = Fake()
    before = judged.answers(backend, Journal(), steps, "before", lambda: None)
    backend.weight = 1.0
    after = judged.answers(backend, Journal(), steps, "scale-1.0", lambda: None)
    rows = judged.judge(backend, Journal(), steps, before, after, "scale-1.0", lambda: None)
    assert [(r["outcome"], r["by"]) for r in rows] == [("win", "judge")] * 3
    assert judged.summary(rows) == {"count": 3, "worse": 0, "same": 0, "checked": 3, "available": 3}
    shown = judged.examples(steps, before, after, rows)
    assert [e["source"] for e in shown] == ["original", "original", "practice"]  # real conversations first
    assert shown[0]["feedback"] == "use pnpm here, never npm" and shown[2]["lesson"] == "Always use pnpm."
    # The judge does not decide: a tie, whatever it said. The teacher is not
    # asked (this backend has none): its preference is what training fits.
    for say, why in (('{"follows": "both"}', "both"), ('{"follows": "neither"}', "neither"),
                     ("Let me think about the two answers", "unclear")):
        verdicts["say"] = say
        assert judged.outcome(backend, steps[0], before["s0"], after["s0"]) == ("tie", why)
    rows = judged.judge(backend, Journal(), steps, before, after, "scale-1.0", lambda: None)
    assert judged.summary(rows) == {"count": 0, "worse": 0, "same": 3, "checked": 3, "available": 3}
    # The same action: a tie, no judge asked.
    assert judged.outcome(backend, steps[0], before["s0"], before["s0"]) == ("tie", "same")


def test_judged_units_decide_and_the_kl_is_only_a_diagnostic():
    from songe.sleep.gate import decide

    def rows(kl, outcome=None):
        return [{"id": str(i), "conversation": f"c{i}", "kl": kl, "drift": 0.0, "empty": 0, "repetition": 0,
                 "tool_valid": 1, **({"outcome": outcome} if outcome else {})} for i in range(5)]
    # The KL got worse, but the candidate won every judged moment: accepted.
    d = decide(rows(1.0), rows(2.0, "win"), min_heldout=5, drift_limit=0.05)
    assert d.accepted and d.kl_change == -1.0
    # The KL got better, but the moments were ties: not accepted.
    d = decide(rows(1.0), rows(0.5, "tie"), min_heldout=5, drift_limit=0.05)
    assert not d.accepted and d.reason.startswith("Not clearly better: better on 0, worse on 0")


def test_near_identical_calls_tie_and_a_judge_that_always_picks_one_side_counts_for_nothing(monkeypatch):
    """Live: two reads of snake.py (line 20 vs 21) and two "debate the
    timeout" answers were judged wins; a 3B judge preferred one position."""
    import contextlib

    from songe.quick import differs
    from songe.sleep import judged

    read = lambda offset: {"text": "", "tool_calls": [{"name": "read", "arguments": {"path": "snake.py", "offset": offset}}]}  # noqa: E731
    assert not differs(read(20), read(21))
    monkeypatch.setattr(judged, "render", lambda tok, messages, tools, **kw: messages[0]["content"])

    class AlwaysA:
        tokenizer = config = None

        def base(self):
            return contextlib.nullcontext()

        def sample(self, prompt, seed=None, *, deterministic=False, max_tokens=None):
            return [], '{"follows": "A"}', None

    step = {"id": "m1", "conversation": "c1", "context": [{"role": "user", "content": "run it"}],
            "hindsight": [{"role": "user", "content": "always execute the script"}]}
    before = {"text": "", "tool_calls": [{"name": "bash", "arguments": {"command": "python game.py"}}], "tokens": [1]}
    after = {"text": "", "tool_calls": [{"name": "edit", "arguments": {"path": "game.py"}}], "tokens": [2]}
    assert judged.outcome(AlwaysA(), step, before, after) == ("tie", "both")


def test_a_targeted_lesson_living_in_one_conversation_can_pass_per_moment():
    """User decision: the judged gate resamples moments, not conversations
    (live c-0003: 5 conversations gave a lower bound of exactly 0)."""
    from songe.sleep.gate import decide

    def rows(outcomes):
        # m0-m5 in one conversation, the rest spread over four others.
        return [{"id": f"m{i}", "conversation": "c0" if i < 6 else f"c{1 + i % 4}", "kl": 1.0, "drift": 0.0,
                 "empty": 0, "repetition": 0, "tool_valid": 1, **({"outcome": o} if o else {})}
                for i, o in enumerate(outcomes)]
    before = rows([None] * 14)
    # Six wins, all in one conversation; ties elsewhere: it passes now.
    targeted = rows(["win"] * 6 + ["tie"] * 8)
    assert decide(before, targeted, min_heldout=5, drift_limit=0.05).accepted
    # c-0003: better on 2, worse on 1, 11 the same: still not clearly better.
    weak = rows(["win", "win", "loss"] + ["tie"] * 11)
    assert not decide(before, weak, min_heldout=5, drift_limit=0.05).accepted


def test_the_judge_s_answer_is_begun_for_it_with_thinking_switched_off(monkeypatch):
    """Live: about 4 of 14 replies could not be read. The reply is begun
    ('{"follows": "'), so only the letter is left, and thinking is off."""
    import contextlib

    from songe.sleep import judged

    seen = {}
    monkeypatch.setattr(judged, "render", lambda tok, messages, tools, **kw: seen.update(kw) or [7, 8])

    class Tok:
        def encode(self, text, add_special_tokens=True):
            assert text == judged.PREFILL and add_special_tokens is False
            return [9]

    class Model:
        tokenizer = Tok()

        def base(self):
            return contextlib.nullcontext()

        def sample(self, ids, seed=None, *, deterministic=False, max_tokens=None):
            seen["ids"] = ids
            return [], 'B"}', None

    assert judged.ask(Model(), "which?") == '{"follows": "B"}'
    assert seen["ids"] == [7, 8, 9] and seen["enable_thinking"] is False


def test_the_not_broken_row_uses_the_measure_the_candidate_was_checked_on(config):
    """Live: the incumbent's check used the drift set, the candidate's the
    general questions, and the row read "0.0000 → –"."""
    from songe.cycle import checks

    record = {
        "drift_checks": {"before-k": {"source": "drift", "drift_set": 0.0}, "scale-1.0-k": {"source": "general"}},
        "gate_values": {"before-k": {"held_out": 0.5, "general": 0.0, "drift": 0.0, "count": 5, "drift_count": 3},
                        "scale-1.0-k": {"held_out": 0.4, "general": 0.005, "drift": None, "count": 5, "drift_count": 0}},
        "scale_evaluations": {"1.0": {"reason": "Not clearly better: better on 2, worse on 1"}},
        "wake_scale": None,
    }
    rows = {r["key"]: r for r in checks(record, config, True)}
    assert "drift" not in rows and rows["general"]["after"] == 0.005 and rows["general"]["passed"] is True
    assert rows["health"]["passed"] is None and rows["health"]["text"] == "not_reached"


def test_check_values_come_from_a_strength_that_has_them(config):
    """Live c-0003: resumed after a cancel, only full strength kept its
    values; the row looked up half strength and read "not measured"."""
    from songe.cycle import checks

    record = {
        "drift_checks": {"before-k": {"source": "drift"}, "scale-1.0-k": {"source": "drift"}},
        "gate_values": {"before-k": {"held_out": 0.5, "general": None, "drift": 0.0, "count": 5, "drift_count": 3},
                        "scale-1.0-k": {"held_out": 0.4, "general": None, "drift": 0.0055, "count": 5, "drift_count": 3}},
        "scale_evaluations": {"1.0": {"heldout_change": 0.067}, "0.5": {"heldout_change": 0.10}},
        "wake_scale": None,
    }
    rows = {r["key"]: r for r in checks(record, config, True)}
    assert rows["drift"]["after"] == 0.0055 and rows["drift"]["passed"] is True


def test_the_first_measurement_of_a_machine_is_not_shown_as_a_dream(config, store):
    """Live (Qwen, empty store): measuring trainability showed the Deep Dream
    page with "training"; it is a banner on the dashboard instead."""
    from songe.cycle import model_learning
    from songe.models import registry

    config.learning = "dreams"
    registry.migrate(config, store)
    registry.observe(config, store, "m")
    learner = store.scoped("m")
    activity = {"stage": "train", "calibration": {"measured_sequence": 256, "target_sequence": 2048}}
    view = model_learning(config, learner, "m", activity)
    assert view["current"] is None and view["measuring_machine"] == {"done": 256, "total": 2048}
    # Growth measuring for a trainable model (inside a Deep Dream) stays the Dream's.
    registry.support(store, "m", "trainable")
    view = model_learning(config, learner, "m", activity)
    assert view["current"] is not None and view["measuring_machine"] is None


def test_leaving_pending_on_any_path_leaves_the_measured_note(config, store):
    """Live: a restart mid-measuring flipped the model to trainable on a path
    that wrote no note."""
    from songe.models import registry

    registry.migrate(config, store)
    registry.observe(config, store, "m")
    registry.support(store, "m", "pending", "calibration paused · memory")
    assert store.scoped("m").get("machine_measured") is None  # still pending: no note
    registry.support(store, "m", "trainable")
    note = store.scoped("m").get("machine_measured")
    assert note["support"] == "trainable"
    store.scoped("m").set("machine_measured", None)
    registry.support(store, "m", "trainable")  # not first: no new note
    assert store.scoped("m").get("machine_measured") is None


def test_a_pressed_deep_dream_cut_by_a_restart_carries_on_or_says_so(store):
    """Live: a deploy restart during a pressed Deep Dream's practice; after
    it, nothing on the page."""
    from types import SimpleNamespace

    from songe.schedule import ModelScheduler

    fake = SimpleNamespace(installation_store=store)
    learner = store.scoped("m")
    # It could not train anyway: the note says it was interrupted.
    learner.set("dream_preparing", True)
    learner.set("dream_pending_wait", "it needs 1 more conversation with your feedback")
    ModelScheduler.recover_pressed(fake, "m", learner)
    assert learner.get("dream_stopped")["reason"] == "restart" and learner.get("dream_pending_wait") is None
    # It was going to train: the request is still there, it carries on.
    learner.set("dream_stopped", None)
    learner.set("dream_preparing", True)
    store.set("requested_model", "m")
    store.set("requested_sleep", True)
    ModelScheduler.recover_pressed(fake, "m", learner)
    assert learner.get("dream_stopped") is None and learner.get("dream_preparing") is None


def test_a_pressed_deep_dream_held_back_by_memory_waits_visibly_then_says_so(store):
    """Live (a crowded Mac): two memory stops while measuring, then the press
    was demoted to "a long quiet period" and vanished."""
    import time as _time
    from types import SimpleNamespace

    from songe.schedule import MEMORY_TRIES, ModelScheduler
    from songe.status import deep_step

    learner = store.scoped("m")
    fake = SimpleNamespace(installation_store=store, store=learner)
    learner.set("last_memory_stop", {"available": 3.16e9, "total": 24 * 2**30})
    learner.set("last_memory_baseline", {"available": 8.4 * 2**30, "total": 24 * 2**30})
    launch = {"model": "m", "requested": True, "nights": 0, "at": _time.time() - 1}
    end = dict(calibrating=True, failed=False, interrupted=True, stopped_by="memory guard", memory_stop=True)
    for n in range(1, MEMORY_TRIES):
        store.set("requested_sleep", False)
        ModelScheduler.carry_on(fake, "m", launch, **end)
        assert store.get("requested_sleep") is True and store.get("requested_model") == "m"
        wait = learner.get("memory_wait")
        assert wait["try"] == n and wait["retry_after"] - _time.time() > 50 * n  # time to close apps
    assert "waiting for memory · 8.4 GB free of 24 GB" in deep_step(learner)  # free before the try
    # The third stop: it gives up with the note.
    store.set("requested_sleep", False)
    ModelScheduler.carry_on(fake, "m", launch, **end)
    stopped = learner.get("dream_stopped")
    assert stopped["detail"] == "Waiting for memory" and learner.get("memory_wait") is None
    assert store.get("requested_sleep") is False


def test_measuring_that_reaches_the_machine_s_edge_is_not_a_memory_strike(store):
    """Live (24 GB Mac, 12 GB free): 256 and 512 tokens fit, the 1024 probe hit
    the reserve; it was counted as "close some apps · try 1 of 3"."""
    import time as _time
    from types import SimpleNamespace

    from songe.schedule import ModelScheduler

    learner = store.scoped("m")
    fake = SimpleNamespace(installation_store=store, store=learner)
    learner.set("calibration", {"rows": [{"sequence": 256}, {"sequence": 512}]})
    launch = {"model": "m", "requested": True, "nights": 0, "at": _time.time() - 1}
    ModelScheduler.carry_on(fake, "m", launch, calibrating=True, failed=False, interrupted=True,
                            stopped_by="memory guard", memory_stop=True)
    assert learner.get("memory_wait") is None and learner.get("dream_capacity_edges") == 1
    assert store.get("requested_sleep") is True  # training follows
    # No limit recorded for the probe: measuring is done at what fit.
    learner.set("calibration", {"rows": [{"sequence": 2048}], "progress": {"measured_sequence": 2048, "target_sequence": 6550}})
    learner.set("dream_capacity_edges", None)
    ModelScheduler.carry_on(fake, "m", launch, calibrating=True, failed=False, interrupted=True,
                            stopped_by="memory guard", memory_stop=True)
    progress = learner.get("calibration")["progress"]
    assert progress["complete"] is True and progress["completion_reason"] == "memory"


def test_measuring_inside_a_dream_is_its_own_step_not_training(config, store):
    """User decision: measuring must look clearly different from training:
    its own "Getting ready" step with the sizes tried and the limit found;
    Training stays ahead."""
    from songe.cycle import model_learning
    from songe.models import registry

    config.learning = "dreams"
    registry.migrate(config, store)
    registry.observe(config, store, "m")
    registry.support(store, "m", "trainable")
    learner = store.scoped("m")
    learner.set("calibration", {"rows": [{"sequence": 512}, {"sequence": 256}, {"sequence": 2048}],
                                "held": [{"sequence": 4096, "needed_bytes": 1}]})
    activity = {"stage": "train", "calibration": {"measured_sequence": 2048, "target_sequence": 6550}}
    current = model_learning(config, learner, "m", activity)["current"]
    keys = [s["key"] for s in current["stages"]]
    assert keys.index("measure") == keys.index("train") - 1 and current["stage"] == "measure"
    measure = next(s for s in current["stages"] if s["key"] == "measure")
    assert measure["sizes"] == [256, 512, 2048] and measure["limit"] == 4096
    assert next(s for s in current["stages"] if s["key"] == "train")["state"] == "ahead"


def test_a_leftover_memory_wait_is_cleared_beside_a_note_or_without_a_request(store):
    """Live: "waiting for memory · try 1 of 3" stayed on the rail beside the
    stop note."""
    from types import SimpleNamespace

    from songe.schedule import ModelScheduler

    fake = SimpleNamespace(installation_store=store)
    learner = store.scoped("m")
    learner.set("memory_wait", {"try": 1})
    store.set("requested_model", "m")
    learner.set("dream_stopped", {"reason": "training"})
    ModelScheduler.clear_stale_waits(fake, "m", learner)
    assert learner.get("memory_wait") is None
    learner.set("memory_wait", {"try": 1})
    learner.set("dream_stopped", None)
    ModelScheduler.clear_stale_waits(fake, "m", learner)  # still pending: kept
    assert learner.get("memory_wait") == {"try": 1}
    store.set("requested_model", None)
    ModelScheduler.clear_stale_waits(fake, "m", learner)  # nothing pending: over
    assert learner.get("memory_wait") is None


def test_short_test_moments_hold_out_untrained_lessons_but_leave_one_to_train(store):
    """User decision: on a small store, hold out more practice lessons on
    demand (never ones already trained, always one left to train), and count
    the minimum in test moments."""
    from songe.practice.pool import (
        EXTRA_HELD_OUT,
        ensure_heldout,
        heldout_units,
        practice_steps,
    )

    add_practice(store, 4)  # four lessons, none trained yet
    store.set("trained_signals", ["sig0"])
    before = heldout_units(store)
    added = ensure_heldout(store, before + 10)
    extra = store.get(EXTRA_HELD_OUT)
    assert added >= 1 and "sig0" not in extra  # a trained lesson never becomes a test
    assert {s["signal_id"] for s in practice_steps(store)}  # at least one lesson still trains
    assert heldout_units(store) == before + added
    assert ensure_heldout(store, 0) == 0


def test_one_conversation_with_feedback_is_enough_for_a_deep_dream(config, store):
    """User decision: "i only have 1 conversation ... i think we can gate on
    just the existence of any interaction". Under Dreams real conversations
    are never trained on, so each moment the user reacted to is a test."""
    from songe.practice.pool import heldout_units, real_test_steps, test_minimum
    from songe.schedule import heldout_short

    config.learning = "dreams"
    config.min_heldout = 5
    step = {"id": "s1", "source_id": 1, "conversation": "only", "split": "train",
            "context": [{"role": "user", "content": "binary search"}],
            "hindsight": [{"role": "user", "content": "always measure the runtime"}], "tools": []}
    store.sync_steps([step])
    add_practice(store, 1)  # a single lesson: it stays for training
    assert test_minimum(config) == 1
    assert [s["split"] for s in real_test_steps(store)] == ["heldout"]  # tested, never trained
    assert heldout_units(store) >= 1 and heldout_short(config, store) == 0


def test_the_measure_step_shows_the_teacher_once_the_model_is_covered(config, store):
    """Live: the step sat at "6550 of 6550" while the teacher's longer
    reading (5780 of 6372) was being measured."""
    from songe.cycle import model_learning
    from songe.models import registry

    config.learning = "dreams"
    registry.migrate(config, store)
    registry.observe(config, store, "m")
    registry.support(store, "m", "trainable")
    activity = {"stage": "train", "calibration": {"measured_sequence": 6550, "target_sequence": 6550,
                                                   "teacher_measured_sequence": 5780, "teacher_target_sequence": 6372}}
    current = model_learning(config, store.scoped("m"), "m", activity)["current"]
    measure = next(s for s in current["stages"] if s["key"] == "measure")
    assert measure["progress"] == {"done": 5780, "total": 6372, "longer": True, "unit": "calibration"}


def test_three_kept_scenes_are_enough_to_train_under_dreams(config, store):
    """Live: "a Deep Dream needs 8 practice scenes · 3 kept so far" held a
    pressed Dream back; user decision: any interaction is enough."""
    from songe.pipeline import next_update
    from songe.practice.pool import dream_minimums

    config.learning = "dreams"
    assert dream_minimums(config) == (1, 1, 1)
    step = {"id": "s1", "source_id": 1, "conversation": "only", "split": "train",
            "context": [{"role": "user", "content": "go"}],
            "hindsight": [{"role": "user", "content": "always run it"}], "tools": []}
    store.sync_steps([step])
    add_practice(store, 3)
    config.min_train, config.min_new = 8, 8  # the Mac test config's minimums
    config.save(store.root)  # next_update reads the installation's config
    assert next_update(store) == {"stage": "ready"}


def test_a_pressed_deep_dream_waiting_for_memory_to_train_shows_then_ends(store):
    """Live: "needs 13.2 GB, 9.6 GB free" before training; nothing on the
    page, and Next said "press Dream" with one pending."""
    from types import SimpleNamespace

    from songe import schedule
    from songe.cycle import deep_waiting

    learner = store.scoped("m")
    fake = SimpleNamespace(store=learner)
    store.set("requested_sleep", True)
    store.set("requested_model", "m")
    sample = SimpleNamespace(available=int(9.5 * 2**30), total=24 * 2**30)
    schedule.Scheduler.note_fit_wait(fake, sample, 13 * 2**30)
    assert "waiting for memory to train · 9.5 GB free, training needs about 13 GB" in deep_waiting(learner, "m", None)
    # After FIT_WAIT_SECONDS it ends with the note.
    learner.set("fit_wait", learner.get("fit_wait") | {"since": 0})
    schedule.Scheduler.note_fit_wait(fake, sample, 13 * 2**30)
    assert learner.get("dream_stopped")["detail"] == "Waiting for memory"
    assert store.get("requested_sleep") is False and learner.get("fit_wait") is None


def test_a_pressed_deep_dream_whose_night_failed_retries_then_says_why(store):
    """Live (Inkling): a pressed Dream's training night failed, the page said
    "retry in 5m", and with auto-dream off nothing retried and no note came."""
    import time as _time
    from types import SimpleNamespace

    from songe.schedule import FAILED_NIGHT_RETRIES, ModelScheduler

    fake = SimpleNamespace(installation_store=store)
    learner = store.scoped("m")
    launch = {"model": "m", "requested": True, "nights": 0, "at": _time.time() - 1}
    learner.night(None, status="failed")  # the pressed Dream's night, failed
    args = {"calibrating": False, "failed": True, "interrupted": False, "stopped_by": None, "memory_stop": False}
    for count in range(1, FAILED_NIGHT_RETRIES):
        learner.set("failures", {"count": count, "at": _time.time()})
        store.set("requested_sleep", False)
        ModelScheduler.carry_on(fake, "m", launch, **args)
        # The announced retry is this request, kept armed.
        assert store.get("requested_sleep") is True and store.get("requested_model") == "m"
        assert learner.get("dream_stopped") is None
    learner.set("failures", {"count": FAILED_NIGHT_RETRIES, "at": _time.time()})
    learner.set("child_error", "Learning stopped: killed by SIGKILL (out of memory?)")
    store.set("requested_sleep", False)
    ModelScheduler.carry_on(fake, "m", launch, **args)
    stopped = learner.get("dream_stopped")
    assert store.get("requested_sleep") is False and stopped["reason"] == "training"
    assert "failed 3 times" in stopped["detail"] and "SIGKILL" in stopped["detail"]


def test_the_incumbents_measurements_are_kept_for_the_next_dream(tmp_path, config):
    """kept_as: a second Dream with the same incumbent measures nothing it
    measured before; a different fingerprint (a new incumbent) measures
    again and replaces the old folder."""
    from songe.sleep.evaluate import KeptMeasurements, evaluate

    calls = []

    class Fake:
        def __init__(self):
            self.config, self.tokenizer = config, None

        def check(self):
            pass

        def kl_from(self, base, prompt, teacher, y, *, hindsight=False, **kw):
            calls.append(("kl", prompt))
            return 0.25

        def sample(self, prompt, seed=None, **kw):
            calls.append(("sample", prompt))
            return [1], "done", None

    v = Fake()
    refs = [{"id": f"m{i}", "conversation": f"c{i}", "prompt": [i, 1], "teacher": [i, 2], "y": [3],
             "tools": [], "tool_step": False} for i in range(3)]
    drift_refs = [{"conversation": f"d{i}", "prompt": [9, i], "y": [4]} for i in range(12)]
    first = evaluate(v, v, refs, [], drift_refs=drift_refs,
                     kept_as=KeptMeasurements(tmp_path, ["incumbent-a"]))
    assert len(calls) == 12 + 3 * 2
    calls.clear()
    again = evaluate(v, v, refs, [], drift_refs=drift_refs,
                     kept_as=KeptMeasurements(tmp_path, ["incumbent-a"]))
    assert calls == [] and again == first
    evaluate(v, v, refs[:1], [], drift_refs=drift_refs,
             kept_as=KeptMeasurements(tmp_path, ["incumbent-b"]))
    assert len(calls) == 12 + 2 and len(list(tmp_path.iterdir())) == 1
