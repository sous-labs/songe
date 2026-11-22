"""Counting test moments without decoding step payloads, and caching it."""

import json

from songe.practice import pool


def old_count(store):
    """The previous implementation: decode every step and every scene."""
    per = {}
    for s in pool.real_test_steps(store):
        if s.get("hindsight"):
            per[s["conversation"]] = per.get(s["conversation"], 0) + 1
    real = sum(min(n, pool.MOMENTS_PER_CONVERSATION) for n in per.values())
    return real + len({s["signal_id"] for s in pool.practice_steps(store, split="held_out")})


def steps(store):
    base = {"context": [{"role": "user", "content": "x" * 2000}], "tools": [], "split": "train"}
    # Real steps carry a list of later messages, empty, or none.
    hindsights = [
        [{"role": "user", "content": "no"}], [], None, [{"role": "tool", "content": "error"}],
        [], [{"role": "user", "content": "again"}], None, [{"role": "user", "content": "ok"}],
    ]
    items = []
    for i, h in enumerate(hindsights):
        step = base | {"id": f"s{i}", "source_id": i, "conversation": f"c{i % 4}"}
        if h is not None:
            step["hindsight"] = h
        items.append(step)
    # Five feedback moments in one conversation: only three count.
    items += [base | {"id": f"m{i}", "source_id": 100 + i, "conversation": "busy",
                      "hindsight": [{"role": "user", "content": "fix"}]} for i in range(5)]
    store.sync_steps(items)
    with store.connect() as db:  # one retired step never counts
        db.execute("UPDATE steps SET retired_at=1 WHERE id='s0'")


def practice(store, n):
    with store.connect() as db:
        for i in range(n):
            row = {"id": f"sig{i}#v1", "signal_id": f"sig{i}", "tools": [], "teacher_note": "note",
                   "messages": [{"role": "user", "content": f"scene {i}"}] if i % 7 else []}
            db.execute('INSERT INTO practice(id,model_id,signal_id,payload,kept,"drop",created)'
                       " VALUES (?,?,?,?,1,NULL,0)",
                       (row["id"], store.model_id, row["signal_id"], json.dumps(row)))


def test_sql_count_matches_decoding_every_step(store, monkeypatch):
    monkeypatch.setattr(pool, "_UNITS", {})
    steps(store)
    practice(store, 40)
    store.set(pool.EXTRA_HELD_OUT, ["sig3", "sig14"])
    expected = old_count(store)
    assert expected > 0
    assert pool.heldout_units(store) == expected


def test_count_is_cached_until_the_database_changes(store, monkeypatch):
    monkeypatch.setattr(pool, "_UNITS", {})
    steps(store)
    first = pool.heldout_units(store)
    calls = []
    real = type(store).test_moment_conversations
    monkeypatch.setattr(type(store), "test_moment_conversations",
                        lambda self: calls.append(1) or real(self))
    assert pool.heldout_units(store) == first and calls == []  # no commit: no query
    store.sync_steps([{"id": "late", "source_id": 999, "conversation": "new", "tools": [],
                       "context": [], "hindsight": [{"role": "user", "content": "no"}],
                       "split": "train"}])
    assert pool.heldout_units(store) == first + 1 and calls == [1]
