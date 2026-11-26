"""One set of practice per lesson, and task prompts are not corrections."""

import json

from songe.practice import store as practice
from songe.reading.rules import stated_rules

BEVY = ("This is a Rust project using Bevy 0.20. Build a small teaching demo of a bouncing ball. "
        "Don't change the Bevy version and don't add any dependencies.")


def test_the_opening_task_message_is_not_a_stated_rule():
    items = [
        {"kind": "user", "content": BEVY},
        {"kind": "action", "step": 1},
        {"kind": "user", "content": "Never add a dependency without asking me first."},
    ]
    assert stated_rules(items) == [(2, "Never add a dependency without asking me first.")]
    # A compacted session continues earlier work: its first reply counts.
    resumed = [{"kind": "checkpoint", "content": "Earlier: the demo builds."},
               {"kind": "user", "content": "Always run cargo check before you say it builds."}]
    assert stated_rules(resumed) == [(1, "Always run cargo check before you say it builds.")]


def add(store, sid, session, lesson, *, rule=False, revision=1):
    payload = {"id": sid, "session": session, "type": "correction", "source": "user",
               "lesson": lesson, "text": lesson, "exchanges": {"1": 1}, "revision": revision}
    if rule:
        payload["from"] = "rules"
    with store.connect() as db:
        db.execute("INSERT INTO signals VALUES (?,?,?,?,?,NULL)",
                   (sid, store.model_id, session, revision, json.dumps(payload)))


def scenes(store, signal_id, n):
    with store.connect() as db:
        for v in range(1, n + 1):
            db.execute('INSERT INTO practice(id,model_id,signal_id,payload,kept,"drop",created)'
                       " VALUES (?,?,?,?,1,NULL,0)",
                       (f"{signal_id}#v{v}", store.model_id, signal_id, json.dumps({"signal_id": signal_id})))


RULE = "Never add a dependency without asking me first."


def test_identical_rules_and_look_alike_lessons_are_practised_once(store):
    for i in range(6):  # the same stated rule in six runs
        add(store, f"rule{i}", f"run{i}", RULE, rule=True, revision=i + 1)
    add(store, "look1", "a", "Check the window renders the ball before saying it works.", revision=10)
    add(store, "look2", "b", "Check that the window renders the ball before you say it works.", revision=11)
    add(store, "other", "c", "Keep the physics step fixed at sixty updates per second.", revision=12)
    todo = practice.wanted(store, 3)
    assert len(todo) == 3  # one per lesson, not one per signal
    assert all(missing == [1, 2, 3] for _, missing in todo)
    groups = practice.cluster_of(store)
    assert sorted(groups["rule0"]) == [f"rule{i}" for i in range(6)]
    assert sorted(groups["look1"]) == ["look1", "look2"] and groups["other"] == ["other"]


def test_a_cluster_keeps_the_scenes_it_has(store):
    add(store, "old", "a", RULE, rule=True, revision=1)
    scenes(store, "old", 3)
    # A newer copy of the same rule does not start its own scenes.
    add(store, "new", "b", RULE, rule=True, revision=2)
    assert practice.wanted(store, 3) == []
    # A rule in different words is a different lesson.
    add(store, "else", "c", "Never rename public functions in this project.", rule=True, revision=3)
    assert [s["id"] for s, _ in practice.wanted(store, 3)] == ["else"]


def test_practice_takes_only_lessons_with_the_evidence_a_note_needs(store):
    """Live, Qwen3.5-4B: "Just the code.", judged a one-off, was read as
    "Always include the exact file pattern requested by the user.", got 24
    practice scenes and was trained; Quick Dream had kept it out of the
    prompt. One evidence bar for both (quick.admitted): a standing wish at
    once, a correction wanted once when the same wish comes up in another
    conversation, and nothing the user took out of the prompt. The bar is
    about evidence, not about what a lesson applies to: a lesson from the
    work itself that is too narrow for the prompt is practised as before."""
    from songe import quick
    from songe.practice import pool

    def add(sid, session, lesson, **fields):
        payload = {"id": sid, "session": session, "type": "correction", "source": "user", "lesson": lesson,
                   "text": "Just the code.", "exchanges": {"1": 1}, "revision": 1} | fields
        with store.connect() as db:
            db.execute("INSERT INTO signals VALUES (?,?,?,?,?,NULL)",
                       (sid, store.model_id, session, 1, json.dumps(payload)))

    add("once", "s1", "Always include the exact file pattern requested by the user.", turn_kind="once", generality="always")
    add("lasting", "s2", "Use pnpm, never npm.", turn_kind="lasting", generality="always")
    add("task", "s3", "Rename the helper in utils before the test.", source="songe", type="tool_error", generality="task")
    add("tool", "s4", "The test runner needs the venv activated first.", source="songe", type="tool_error", generality="always")

    add("kind", "s6", "Run the migration before the integration tests.", source="songe", type="tool_error", generality="kind")

    def practised():
        return sorted(s["id"] for s in practice.lesson_signals(store))

    def in_prompt():
        return sorted(i for c in quick.candidates(store, 10, 1) for i in c["signals"])

    # The one-off waits for both. The lessons from the work are practised
    # whatever they apply to; only the one for any task fits the prompt.
    assert in_prompt() == ["lasting", "tool"]
    assert practised() == ["kind", "lasting", "task", "tool"]
    assert sorted(s["id"] for s, _ in practice.wanted(store, 2)) == ["kind", "lasting", "task", "tool"]
    # Scenes an earlier version wrote for the one-off are kept and not trained.
    for sid in ("once", "lasting"):
        with store.connect() as db:
            db.execute('INSERT INTO practice(id,model_id,signal_id,payload,kept,"drop",created) VALUES (?,?,?,?,1,NULL,0)',
                       (f"{sid}#v1", store.model_id, sid, json.dumps(
                           {"id": f"{sid}#v1", "signal_id": sid, "messages": [{"role": "user", "content": "x"}],
                            "teacher_note": "n"})))
    steps = pool.practice_steps(store) + pool.practice_steps(store, split="held_out")
    assert [s["signal_id"] for s in steps] == ["lasting"]
    # The same wish in a second conversation: now it is a lesson, for both.
    add("again", "s5", "Always include the exact file pattern the user requested.", turn_kind="once", generality="always")
    assert in_prompt() == ["again", "lasting", "once", "tool"]
    assert practised() == ["again", "kind", "lasting", "once", "task", "tool"]
    steps = pool.practice_steps(store) + pool.practice_steps(store, split="held_out")
    assert sorted(s["signal_id"] for s in steps) == ["lasting", "once"]
    # A lesson the user took out of the prompt is not practised either.
    state = quick.state(store) | {"removed": ["Use pnpm, never npm."]}
    store.set(quick.KEY, state)
    assert "lasting" not in practised() and "lasting" not in in_prompt()
    # What an adopted Dream absorbed leaves the prompt and stays a practised lesson.
    store.set("absorbed_signals", ["tool"])
    assert "tool" in practised() and "tool" not in in_prompt()


def test_measuring_and_its_check_count_the_same_scenes(store):
    """Live, Qwen3.8-27B on an RTX PRO 6000: after an adopted first night, a
    pressed Deep Dream measured memory again every 80 s for 87 minutes and
    never trained. Measuring records the scenes it measured (a one-off's kept
    scenes left out, as training leaves them out); the check for new scenes
    counted every kept scene, so after each measurement it asked for another."""
    from types import SimpleNamespace

    from songe.sleep import calibration_state as cs

    for sid, session, lesson, kind in (("once", "s1", "Use metric.", "once"),
                                       ("lasting", "s2", "Use pnpm, not npm.", "lasting")):
        payload = {"id": sid, "session": session, "type": "correction", "source": "user", "lesson": lesson,
                   "text": lesson, "exchanges": {"1": 1}, "revision": 1, "turn_kind": kind, "generality": "always"}
        with store.connect() as db:
            db.execute("INSERT INTO signals VALUES (?,?,?,?,?,NULL)",
                       (sid, store.model_id, session, 1, json.dumps(payload)))
            db.execute('INSERT INTO practice(id,model_id,signal_id,payload,kept,"drop",created) VALUES (?,?,?,?,1,NULL,0)',
                       (f"{sid}#v1", store.model_id, sid, json.dumps(
                           {"id": f"{sid}#v1", "signal_id": sid, "messages": [{"role": "user", "content": "x"}],
                            "teacher_note": "n"})))
    config = SimpleNamespace(learning="dreams")
    measured = cs.pool_key(cs.sample_pool(store, config))
    assert cs.revisions_key(cs.sample_revisions(store, config)) == measured
