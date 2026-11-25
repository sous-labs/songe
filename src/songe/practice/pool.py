"""What a dreams cycle trains on and what keeps it honest, as worker steps.

- Practice steps: kept scenes. The student sees the scene; the teacher also
  sees its teacher note (the lesson and the feedback it came from) as
  hindsight. A scene's row number in the practice table is its revision, so
  the worker's new-then-replay bookkeeping works as for real steps.
- Drift steps: confirmed successes from real sessions, cut at the request
  where the checked action was taken. Never trained on: they measure whether
  a candidate still behaves like the base where the base was right.
"""

import json

from ..steps import digest

# One signal in this many keeps its scenes out of training, so the check
# stage has practice the candidate never saw ("new practice"). Split by
# signal, not by scene: a lesson's scenes are near-duplicates of each other.
HELD_OUT_SHARE = 5


# Lessons held out on demand, beyond one in five, when a small store has too
# few test places (user decision); kept out of training for good.
EXTRA_HELD_OUT = "extra_heldout"
# Single scenes kept out of training for a note's test, when its lesson came
# up in one conversation only (ensure_note_heldout); kept out for good.
EXTRA_HELD_OUT_SCENES = "extra_heldout_scenes"
NOTE_SCENES = 6  # sleep.judged.LEARNED_SCENES: what a note's test asks


def held_out(signal_id, extra=()):
    return signal_id in extra or int(digest(signal_id)[:8], 16) % HELD_OUT_SHARE == 0


_TOOLS_DIGESTS = {}


def tools_digest(raw, tools):
    """digest(tools), once per distinct tool list: scenes share their frame's
    few tool lists, and hashing one costs thousands of steps."""
    key = raw or ""
    if key not in _TOOLS_DIGESTS:
        if len(_TOOLS_DIGESTS) > 256:
            _TOOLS_DIGESTS.clear()
        _TOOLS_DIGESTS[key] = digest(tools or [])
    return _TOOLS_DIGESTS[key]


def practice_steps(store, split="train"):
    """Kept scenes as steps: the training ones, or the held-out ones."""
    with store.connect() as db:
        rows = db.execute(
            "SELECT rowid, payload, json_extract(payload,'$.tools') FROM practice"
            " WHERE model_id=? AND kept=1 ORDER BY rowid",
            (store.model_id,),
        ).fetchall()
    extra = set(store.get(EXTRA_HELD_OUT) or [])
    out_scenes = set(store.get(EXTRA_HELD_OUT_SCENES) or [])
    waiting = not_admitted(store) if rows else set()
    steps = []
    for rowid, payload, tools in rows:
        row = json.loads(payload)
        if not row.get("messages"):
            continue
        # An ordinary row (practice.store.write_ordinary) always trains, also
        # when its lesson is held out for the check, and is never a test
        # moment: no lesson is in it. practice_revisions counts every kept
        # row, and the two must agree.
        ordinary = bool(row.get("ordinary"))
        if ordinary:
            if split == "held_out":
                continue
        elif (held_out(row["signal_id"], extra) or row["id"] in out_scenes) != (split == "held_out"):
            continue
        if row["signal_id"] in waiting:
            continue
        steps.append(
            {
                "id": "practice:" + row["id"],
                # Replay and clustering group a lesson's scenes together.
                "conversation": "signal:" + row["signal_id"],
                "split": split,
                "practice": True,
                "stated": (row.get("signal") or {}).get("from") == "rules",
                "signal_id": row["signal_id"],
                # The teacher's answer with the lesson's note, when the scene
                # was kept by it (a plain chat's scene): what a note's test
                # compares the new version's answer with (sleep.judged.learned).
                "teacher_answer": row.get("teacher_answer"),
                "context": row["messages"],
                "tools": row.get("tools") or [],
                # No note for the teacher in an ordinary row: the student is
                # trained toward the model as it is, along its own answer.
                "hindsight": [] if ordinary else [{"role": "user", "content": row["teacher_note"]}],
                "ordinary": ordinary,
                "action": {},
                "source_id": rowid,
                "observed_id": 0,
                "env_fingerprint": tools_digest(tools, row.get("tools")),
            }
        )
    return steps


def not_admitted(store):
    """Ids of the lesson signals that may not be learned now (quick.admitted
    is the rule): a one-off correction that has not come up again, a lesson
    the user took out of the prompt. Their scenes, written by an earlier
    version or before the note was removed, are kept and not trained."""
    from ..reading.store import signals
    from .store import LESSON_TYPES, lesson_signals

    allowed = {s["id"] for s in lesson_signals(store)}
    return {s["id"] for s in signals(store, types=LESSON_TYPES) if s["id"] not in allowed}


def practice_revisions(store):
    """{step id: revision} of every kept scene practice_steps returns (both
    splits), read by SQLite: no scene is decoded. Like there, the scenes of
    a lesson that may not be learned now are left out: measuring records the
    scenes it measured, and counting more here asked for a new measurement
    after every one (live: 87 minutes of measuring, no training)."""
    with store.connect() as db:
        rows = db.execute(
            "SELECT id, rowid, json_extract(payload,'$.signal_id') FROM practice"
            " WHERE model_id=? AND kept=1"
            " AND COALESCE(json_array_length(payload,'$.messages'), 0) > 0",
            (store.model_id,),
        ).fetchall()
    waiting = not_admitted(store) if rows else set()
    return {"practice:" + scene: rowid for scene, rowid, signal in rows if signal not in waiting}


MOMENTS_PER_CONVERSATION = 3  # judged.PER_UNIT: a conversation's test moments
DRIFT_MAX = 48  # confirmed successes measured for drift, at most


def test_minimum(config):
    """Test moments a Deep Dream needs. Under Dreams: one (user decision: any
    interaction is enough; real conversations are never trained on there, so
    each moment the user reacted to is a fair test). Otherwise min_heldout."""
    return 1 if getattr(config, "learning", "") == "dreams" else config.min_heldout


def dream_minimums(config):
    """(practice to train on, test moments, new practice) a Deep Dream needs.
    Under Dreams one of each (user decision: any interaction is enough; live:
    3 kept scenes were held back by "needs 8"). Otherwise the configured ones."""
    if getattr(config, "learning", "") == "dreams":
        return 1, 1, 1
    return config.min_train, config.min_heldout, config.min_new


def real_test_steps(store):
    """Under Dreams every recorded moment with feedback is a test moment:
    training only sees practice scenes, never the user's conversations."""
    return [s | {"split": "heldout"} for s in store.active_steps() if s.get("hindsight")]


_UNITS = {}


def sampled(steps, per):
    """`per` of these steps, chosen by the hash of their id: spread over the
    whole conversation like a random sample, yet the same ones every Dream
    (a conversation's choice changes only when it gains moments). Order kept.
    Its last moments would be biased (the end of a session is not typical)
    and the costliest: they carry the longest histories."""
    if len(steps) <= per:
        return list(steps)
    chosen = {id(s) for s in sorted(steps, key=lambda s: digest(str(s["id"])))[:per]}
    return [s for s in steps if id(s) in chosen]


def gate_moments(steps, per=None):
    """Each conversation's sampled moments with feedback: what the gate
    scores and judges. All of them made the check grow with history (live,
    Inkling: 4,788 measurements, ~100 hours, for 12 long conversations)."""
    per = per or MOMENTS_PER_CONVERSATION
    by = {}
    for s in steps:
        by.setdefault(s["conversation"], []).append(s)
    keep = {id(s) for ss in by.values() for s in sampled(ss, per)}
    return [s for s in steps if id(s) in keep]


def heldout_units(store):
    """Test moments the gate can judge: up to three from each real
    conversation with feedback, plus one from each held-out practice lesson
    (user decision: moments, not conversations).

    Called several times per dashboard refresh: counted in SQL (no payload
    decoding) and remembered until the database changes."""
    from ..log import change_mark

    path = getattr(store, "path", None)
    mark = change_mark(path) if path is not None else None
    key = (str(path), getattr(store, "model_id", ""))
    if mark is not None and _UNITS.get(key, (None,))[0] == mark:
        return _UNITS[key][1]
    per = {}
    for conversation in store.test_moment_conversations():
        per[conversation] = per.get(conversation, 0) + 1
    real = sum(min(n, MOMENTS_PER_CONVERSATION) for n in per.values())
    lessons = len(held_out_lessons(store))
    if mark is not None:
        _UNITS[key] = (mark, real + lessons)
    return real + lessons


def held_out_lessons(store):
    """Signal ids of kept practice lessons that are held out, without
    decoding the scenes."""
    with store.connect() as db:
        rows = db.execute(
            "SELECT json_extract(payload,'$.signal_id') FROM practice"
            " WHERE model_id=? AND kept=1"
            " AND COALESCE(json_array_length(payload,'$.messages'), 0) > 0",
            (store.model_id,),
        ).fetchall()
    extra = set(store.get(EXTRA_HELD_OUT) or [])
    return {sig for (sig,) in rows if held_out(sig, extra)}


def ensure_heldout(store, need):
    """Hold out more practice lessons when test moments are short, beyond one
    in five: only lessons never trained, in a fixed order, always leaving at
    least one lesson to train on; remembered, so they are never trained
    later (no leakage). Returns how many were added."""
    have = heldout_units(store)
    if have >= need:
        return 0
    trained = set(store.get("trained_signals") or []) | set(store.get("absorbed_signals") or [])
    training = sorted({s["signal_id"] for s in practice_steps(store)})
    fresh = [sig for sig in training if sig not in trained]
    extra = list(store.get(EXTRA_HELD_OUT) or [])
    added = 0
    for sig in fresh:
        if have >= need or len(training) - added <= 1:
            break
        extra.append(sig)
        added += 1
        have += 1
    if added:
        store.set(EXTRA_HELD_OUT, extra)
    return added


def ensure_note_heldout(store):
    """Every note keeps practice out of training, so the note can be tested.

    A note leaves the prompt only when the new version follows its lesson on
    practice it was not trained on (sleep.judged.learned). A lesson with
    practice from two or more conversations keeps the one with the fewest
    scenes out: its scenes are another conversation's kind of request. A
    lesson that came up in one conversation keeps NOTE_SCENES of its scenes
    out, and at least half to train on (without that such a note has
    nothing to be asked on and can never leave: live, three of seven notes,
    two of them lessons the new version held 15 and 16 times of 16).
    Only lessons never trained, remembered. Returns how many were added."""
    from .. import quick
    from .store import cluster_of

    with store.connect() as db:
        rows = db.execute(
            "SELECT json_extract(payload,'$.signal_id'), json_extract(payload,'$.id') FROM practice"
            " WHERE model_id=? AND kept=1"
            " AND COALESCE(json_array_length(payload,'$.messages'), 0) > 0 ORDER BY rowid",
            (store.model_id,),
        ).fetchall()
    by = {}
    for sig, scene in rows:
        by.setdefault(sig, []).append(scene)
    extra = list(store.get(EXTRA_HELD_OUT) or [])
    scenes = list(store.get(EXTRA_HELD_OUT_SCENES) or [])
    trained = set(store.get("trained_signals") or []) | set(store.get("absorbed_signals") or [])
    same, added = cluster_of(store), 0
    for lesson in quick.active(store):
        practised = sorted({
            mate for sig in lesson.get("signals") or [] for mate in same.get(sig, [sig]) if mate in by
        })
        if any(held_out(sig, extra) for sig in practised):
            continue
        fresh = [sig for sig in practised if sig not in trained]
        if not fresh:
            continue
        if len(practised) >= 2:
            extra.append(min(fresh, key=lambda sig: (len(by[sig]), sig)))
            added += 1
            continue
        sig = fresh[0]
        free = [scene for scene in by[sig] if scene not in scenes]
        take = min(NOTE_SCENES - (len(by[sig]) - len(free)), len(free) - (len(by[sig]) + 1) // 2)
        if take > 0:
            scenes += free[-take:]
            added += take
    if added:
        store.set(EXTRA_HELD_OUT, extra)
        store.set(EXTRA_HELD_OUT_SCENES, scenes)
    return added


def heldout_fill(store, missing):
    """Held-out practice to fill the gate's missing test places: one scene
    for each of `missing` held-out lessons (never trained on), in a fixed
    order so every Dream checks on the same ones. Each is its own unit."""
    if missing <= 0:
        return []
    from ..reading.store import signals

    lesson = {s["id"]: s.get("lesson") for s in signals(store)}
    first = {}
    for step in practice_steps(store, split="held_out"):
        first.setdefault(step["signal_id"], step)
    chosen = sorted(first.values(), key=lambda s: s["signal_id"])[:missing]
    return [s | {"split": "heldout", "lesson": lesson.get(s["signal_id"])} for s in chosen]


def drift_steps(store):
    """The drift set: confirmed successes, each at the request that produced
    the checked action. Their session is their conversation (for clustering)."""
    from ..parts import request_of
    from ..reading.store import signals

    steps, seen = [], set()
    with store.connect() as db:
        for s in signals(store, types=("success",)):
            if s.get("confirmed") is not True or not s.get("exchanges"):
                continue
            step = min(int(n) for n in s["exchanges"])
            exchange = s["exchanges"][str(step)]
            if exchange in seen:
                continue
            seen.add(exchange)
            request = request_of(db, exchange)
            if not isinstance(request, dict) or not request.get("messages"):
                continue
            steps.append(
                {
                    "id": f"drift:{exchange}",
                    "conversation": "session:" + s["session"],
                    "split": "drift",
                    "context": request["messages"],
                    "tools": request.get("tools") or [],
                    "hindsight": [],
                    "action": {},
                    "source_id": exchange,
                }
            )
    # Bounded like the test moments: each session's sampled successes, the
    # newest sessions first, DRIFT_MAX at most (every success in history
    # made the check grow with it).
    steps = gate_moments(steps)
    return steps[-DRIFT_MAX:]


def bookkeeping(store, mode="dreams"):
    """(watermark, practiced revisions) as they apply to practice. Under
    hindsight they count exchanges, so until a dreams cycle has run they say
    nothing about practice rows."""
    if store.get("learning_mode", "hindsight") != mode:
        return 0, {}
    return store.get("experience_watermark", 0), store.get("practice_revisions", {})


# {scene id: [revision, times]}: the scenes Deep Dreams trained without being
# adopted, and how many such Dreams trained each. After RETRIES of them a
# scene counts as practised: it is no longer trained in full by every later
# Dream and comes back as replay only.
REJECTED = "rejected_practice"
RETRIES = 3


def rejected(store):
    """{scene id: (revision, times)} of REJECTED. Like the rest of the
    practice bookkeeping it says nothing until a dreams cycle has run in the
    current mode (the worker clears it when the mode changes)."""
    if store.get("learning_mode", "hindsight") != "dreams":
        return {}
    return {
        scene: tuple(value) if isinstance(value, list) else (value, 1)
        for scene, value in (store.get(REJECTED) or {}).items()
    }


def keep_for_retry(store, revisions):
    """The bookkeeping of a Deep Dream that was not adopted (rejected by the
    gate, or its update never loaded). revisions: the cycle record's
    practice_revisions, what was practised before plus what this Dream
    trained. Nothing was adopted, so nothing was practised: the scenes it
    trained stay untrained and are trained again, in full, with the next new
    scenes (counted as practised they only ever came back as replay, a few at
    a time). A scene on its RETRIES-th such Dream is counted as practised."""
    practised = dict(store.get("practice_revisions", {}))
    waiting = rejected(store)
    for scene, revision in revisions.items():
        if practised.get(scene) == revision:
            continue
        before = waiting.get(scene, (None, 0))
        times = (before[1] if before[0] == revision else 0) + 1
        if times >= RETRIES:
            waiting.pop(scene, None)
            practised[scene] = revision
        else:
            waiting[scene] = (revision, times)
    store.set(REJECTED, {scene: list(value) for scene, value in waiting.items()})
    store.set("practice_revisions", practised)


def untrained(store, steps=None):
    """Kept scenes not trained on yet. The scenes of a rejected Deep Dream
    are among them: nothing was adopted, so nothing was practised. They are
    marked "retry", come after the scenes no Dream has tried (a capped Dream
    takes those first), and on their own they wait: the same scenes alone
    would be rejected again."""
    from ..steps import revision_id

    watermark, practiced = bookkeeping(store)
    steps = practice_steps(store) if steps is None else steps
    rows = marked(store, [s for s in steps if revision_id(s) > max(watermark, practiced.get(s["id"], 0))])
    fresh = [s for s in rows if not s.get("retry")]
    if not fresh:
        return []
    return fresh + [s for s in rows if s.get("retry")]


def marked(store, steps):
    """These scenes, the ones a rejected Deep Dream trained marked "retry"."""
    from ..steps import revision_id

    waiting = rejected(store)
    return [s | {"retry": True} if waiting.get(s["id"], (None,))[0] == revision_id(s) else s for s in steps]


def pending(config, store):
    """Work waiting for a cycle, in the units the scheduler orders by."""
    if config.learning == "dreams":
        return len(untrained(store))
    return store.pending_learnable()
