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
