"""Which sessions are read, and the signals found in them, per learner.

A session is read again when it grows (its newest exchange id is its
revision); its old signals are replaced. Reading one session is all or
nothing: a stopped reading leaves the old signals, and the cached model
answers make the next attempt quick.
"""

import asyncio
import json
import time

from ..steps import digest
from . import timeline
from .client import Stopped
from .extract import read_session


def pending(store, limit=None):
    """[(session, exchange ids)] this learner has not read at their current
    revision, oldest first."""
    with store.connect() as db:
        level, cutoff = store.learning_policy(db)
        if level == "off":
            return []
        found = timeline.sessions(db, cutoff if level == "new" else None)
        done = dict(
            db.execute(
                "SELECT session,revision FROM read_sessions WHERE model_id=?",
                (store.model_id,),
            ).fetchall()
        )
    out = [(s, ids) for s, ids in found.items() if done.get(s, -1) < max(ids)]
    return out[:limit] if limit else out


def opening_request(items, limit=100):
    """A conversation's first request, clipped, to say which one is read."""
    first = next((it for it in items if it.get("kind") == "user"), None)
    text = " ".join(str((first or {}).get("content") or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def signal_line(signal, limit=110):
    """One found signal in plain words: the user's own words for feedback,
    else the short lesson or the evidence."""
    text = signal.get("text") if signal.get("source") == "user" else None
    text = " ".join(str(text or signal.get("short") or signal.get("evidence") or signal.get("text") or "").split())
    return {
        "type": signal.get("type"),
        "text": text if len(text) <= limit else text[: limit - 1].rstrip() + "…",
    }


def forget_vanished(store):
    """Signals and read marks of sessions that no longer exist (merged into
    another, e.g. an agent's compaction call now joined to the conversation it
    summarised) are dropped; the merged session is read again as a whole.
    Returns how many sessions were dropped."""
    with store.connect() as db:
        alive = set(timeline.sessions(db))
        # Deleted for the size limit after being learned from: not vanished.
        alive |= {r[0] for r in db.execute("SELECT session FROM pruned_sessions")}
        gone = [
            r[0] for r in db.execute(
                "SELECT session FROM read_sessions WHERE model_id=?", (store.model_id,)
            ).fetchall()
            if r[0] not in alive
        ]
        for session in gone:
            db.execute("DELETE FROM signals WHERE model_id=? AND session=?", (store.model_id, session))
            db.execute("DELETE FROM read_sessions WHERE model_id=? AND session=?", (store.model_id, session))
        if gone:
            db.execute(
                "DELETE FROM practice WHERE model_id=? AND signal_id NOT IN "
                "(SELECT id FROM signals WHERE model_id=?)",
                (store.model_id, store.model_id),
            )
    return len(gone)


def place_rules(store):
    """Stated rules read before they carried the step they are about get it
    now (practice and before/after start there), once each. Returns how many
    rules were placed."""
    from ..quick import reacted_action

    with store.connect() as db:
        rows = db.execute(
            "SELECT id, session, payload FROM signals WHERE model_id=? AND retired_at IS NULL"
            " AND json_extract(payload,'$.from')='rules' AND json_extract(payload,'$.placed') IS NULL",
            (store.model_id,),
        ).fetchall()
        if not rows:
            return 0
        found = timeline.sessions(db)
    placed = 0
    for signal_id, session, payload in rows:
        s = json.loads(payload)
        ids = found.get(session)
        if ids and not s.get("evidence_steps") and s.get("item") is not None:
            _, items = load_session(store, ids)
            about = reacted_action(items[: s["item"]], s.get("text") or "")
            if about and about.get("step"):
                s["evidence_steps"] = [about["step"]]
                s["exchanges"] = {str(about["step"]): about["exchange"]}
                placed += 1
        s["placed"] = True
        with store.connect() as db:
            db.execute("UPDATE signals SET payload=? WHERE id=? AND model_id=?",
                       (json.dumps(s, ensure_ascii=False), signal_id, store.model_id))
    return placed


def reread_copied(store):
    """Once: sessions whose lessons were the prompt's own example pasted back
    are read again (cached answers make it cheap): the copy is no lesson
    now, and a stated rule there is caught. Returns how many were queued."""
    from .prompts import copied_example

    if store.get("reread_copied_v1"):
        return 0
    sessions = {s["session"] for s in signals(store) if copied_example(s.get("lesson"))}
    with store.connect() as db:
        for session in sessions:
            db.execute("DELETE FROM read_sessions WHERE model_id=? AND session=?", (store.model_id, session))
    store.set("reread_copied_v1", True)
    return len(sessions)


def reread_unreadable(store):
    """Once: sessions read when the model's answers could not be read (they
    were cached as errors and the session marked read with what was left)
    are read again. Returns how many sessions were queued."""
    if store.get("reread_unreadable_v1"):
        return 0
    with store.connect() as db:
        keys = {r[0] for r in db.execute("SELECT key FROM reading_cache WHERE answer LIKE '%\"error\": \"no JSON%'")}
        db.execute("DELETE FROM reading_cache WHERE answer LIKE '%\"error\": \"no JSON%'")
        sessions = [r[0] for r in db.execute(
            "SELECT session FROM read_sessions WHERE model_id=?", (store.model_id,)).fetchall()] if keys else []
        # Which sessions used a failed answer is not recorded: with any
        # failures cached, every read session of this learner is read again
        # (the good answers are still cached, so only the failed ones run).
        for session in sessions:
            db.execute("DELETE FROM read_sessions WHERE model_id=? AND session=?", (store.model_id, session))
    store.set("reread_unreadable_v1", True)
    return len(sessions)


def reread_without_short(store):
    """Sessions read before lessons had a short form are read again, once:
    every other answer is cached, so only the lesson question runs again.
    Returns how many sessions were queued."""
    with store.connect() as db:
        rows = db.execute(
            "SELECT DISTINCT session FROM signals WHERE model_id=? AND retired_at IS NULL"
            " AND json_extract(payload,'$.lesson') IS NOT NULL"
            " AND json_extract(payload,'$.short') IS NULL",
            (store.model_id,),
        ).fetchall()
        for (session,) in rows:
            db.execute(
                "DELETE FROM read_sessions WHERE model_id=? AND session=?",
                (store.model_id, session),
            )
    return len(rows)


def counts(store):
    """{read, unread} sessions for this learner."""
    unread = len(pending(store))
    with store.connect() as db:
        read = db.execute(
            "SELECT COUNT(*) FROM read_sessions WHERE model_id=?", (store.model_id,)
        ).fetchone()[0]
    return {"read": read, "unread": unread}


def signals(store, *, types=None):
    with store.connect() as db:
        rows = db.execute(
            "SELECT payload FROM signals WHERE model_id=? AND retired_at IS NULL ORDER BY rowid",
            (store.model_id,),
        ).fetchall()
    found = [json.loads(r[0]) for r in rows]
    return [s for s in found if types is None or s["type"] in types]


def save(store, session, revision, found, items):
    """Replace this learner's signals for one session. A signal retired before
    (a newer preference replaced it) stays retired."""
    exchange_of = {it["step"]: it.get("exchange") for it in items if it["kind"] == "action"}
    with store.connect() as db:
        retired = dict(
            db.execute(
                "SELECT id, retired_at FROM signals WHERE model_id=? AND session=? AND retired_at IS NOT NULL",
                (store.model_id, session),
            ).fetchall()
        )
        db.execute(
            "DELETE FROM signals WHERE model_id=? AND session=?",
            (store.model_id, session),
        )
        for s in found:
            steps = s.get("responsible_steps") or s.get("evidence_steps") or []
            s["exchanges"] = {
                str(n): exchange_of.get(n) for n in steps if exchange_of.get(n) is not None
            }
            s["revision"] = revision
            s["id"] = digest([session, s["type"], s.get("item"), s.get("text")])
            db.execute(
                "INSERT OR REPLACE INTO signals(id,model_id,session,revision,payload,retired_at) VALUES (?,?,?,?,?,?)",
                (s["id"], store.model_id, session, revision, json.dumps(s, ensure_ascii=False), retired.get(s["id"])),
            )
        db.execute(
            "INSERT OR REPLACE INTO read_sessions VALUES (?,?,?,?)",
            (session, store.model_id, revision, time.time()),
        )
        # Practice written for signals that are gone goes with them.
        db.execute(
            "DELETE FROM practice WHERE model_id=? AND signal_id NOT IN "
            "(SELECT id FROM signals WHERE model_id=?)",
            (store.model_id, store.model_id),
        )


# How many earlier preferences one question shows (the most recent ones).
SHOWN = 40


def said_at(signal):
    """When in the user's history a signal's sentence was said: the newest
    exchange it points at (its session's revision when it points at none),
    then its place in the session."""
    return (max((signal.get("exchanges") or {}).values(), default=signal.get("revision") or 0), signal.get("item") or 0)


async def retire_replaced(store, reader):
    """People change: "I prefer window seats." and, weeks later, "I've switched
    to aisle seats." cannot both be notes. Each preference the user stated
    about themself is shown to the model with the ones stated before it (which
    of them does it replace?), each one named is shown again on its own with
    the new sentence (can both be true now?), and those that cannot are
    retired: reading, the prompt and practice no longer see them. Returns how
    many were retired.

    Every preference is gone through each time, oldest first. The list a
    sentence is shown with only changes when an earlier conversation is read
    after it (a backlog is read newest first), so all other questions are
    answered from the reader's cache: one new question for each new sentence."""
    from . import prompts as P

    def load():
        with store.connect() as db:
            return db.execute(
                "SELECT id, retired_at, payload FROM signals WHERE model_id=?", (store.model_id,)
            ).fetchall()

    said = [dict(json.loads(payload), id=sid, retired=at is not None) for sid, at, payload in await asyncio.to_thread(load)]
    said = sorted((s for s in said if s.get("preference")), key=said_at)
    gone = []
    try:
        for place, signal in enumerate(said):
            earlier = {}  # sentence: its signals
            for s in said[:place]:
                if s["preference"] != signal["preference"]:
                    earlier.setdefault(s["preference"], []).append(s)
            earlier = dict(list(earlier.items())[-SHOWN:])
            if signal["retired"] or not earlier:
                continue
            answer = await reader.ask(
                P.REPLACES.format(
                    notes="\n".join(f"{n}. {text}" for n, text in enumerate(earlier, 1)), sentence=signal["preference"]
                ),
                P.REPLACES_SCHEMA,
                "replaces",
            )
            named = answer.get("replaces") if isinstance(answer, dict) else None
            for n in dict.fromkeys(named if isinstance(named, list) else []):
                if not (isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= len(earlier)):
                    continue
                old, signals = list(earlier.items())[n - 1]
                if all(s["retired"] for s in signals):
                    continue
                sure = await reader.ask(
                    P.BOTH_TRUE.format(old=old, new=signal["preference"]), P.BOTH_TRUE_SCHEMA, "both_true"
                )
                if isinstance(sure, dict) and sure.get("kind") == "later_only":
                    for s in signals:
                        if not s["retired"]:
                            s["retired"] = True
                            gone.append(s["id"])
    except Stopped:
        pass  # the user is back: the rest is asked next time
    if gone:
        await asyncio.to_thread(retire, store, gone)
    return len(gone)


# {signal id: the first signal that noted the same wish}, in the learner's state.
SAME = "same_notes"


async def one_wish(reader, a, b):
    """Whether the model takes notes a and b for one wish, shown in this order."""
    from . import prompts as P

    sure = await reader.ask(P.SAME_WISH.format(a=a, b=b), P.SAME_WISH_SCHEMA, "same_wish")
    return isinstance(sure, dict) and sure.get("kind") == "same"


async def fold_same(store, reader):
    """People repeat themselves in other words: "In metric please." in one
    conversation, "Celsius please, I don't think in Fahrenheit." in the next.
    Each becomes a note, and the ten places of the prompt fill up with three
    habits. Each note for the prompt is shown to the model with the notes
    before it (which could be the same wish?), and each one named is shown
    again with it alone, both ways round (one wish, noted twice?). A note
    that repeats an earlier one both times is folded into it: both count for
    one lesson (how often it
    came up, in how many conversations) and take one place, which keeps each
    of their wordings (quick.joined). Nothing is retired and no wording is
    dropped, so a wrong answer costs a place or shares one, never a note.
    Returns how many notes are folded, None when stopped.

    Every note is gone through each time, oldest first, as in retire_replaced:
    all questions but the ones for a new note are answered from the cache."""
    from .. import quick
    from . import prompts as P

    found = sorted((s for s in await asyncio.to_thread(signals, store) if quick.suitable(s)), key=said_at)
    groups = []  # {"text": the first wording, "rule", "said": every wording, "ids": the signals, first one first}
    try:
        for s in found:
            text, rule = quick.wording(s)
            # The same sentence, or what Quick Dream already takes for the
            # same lesson by its words: no question.
            home = next((g for g in groups if any(quick.normal(text) == quick.normal(t) for t in g["said"])
                         or not rule and not g["rule"] and quick.similar(text, g["text"])), None)
            # A place holds what fits in one lesson, each wording whole: a
            # group with no room for this one is not offered.
            shown = [] if home else [
                g for g in groups if len(". ".join([*g["said"], text])) < quick.LESSON_CHARS
            ][-SHOWN:]
            named = []
            if shown:
                answer = await reader.ask(
                    P.SAME.format(notes="\n".join(f"{n}. {g['text']}" for n, g in enumerate(shown, 1)), note=text),
                    P.SAME_SCHEMA,
                    "same_note",
                )
                named = answer.get("same") if isinstance(answer, dict) else None
            numbers = [n for n in named if type(n) is int and 1 <= n <= len(shown)] if isinstance(named, list) else []
            for n in dict.fromkeys(numbers):
                # Asked both ways round, and folded only when both answers say
                # one wish. Live (Qwen3.5-4B): with "uv please." as note 1 the
                # model said "Same thing without the comments." again as "Use
                # uv without comments", and a place held three habits. Over
                # 40 replays of four readings: 9 notes joined a place of
                # another habit asked one way, 4 asked both ways.
                earlier = shown[n - 1]["text"]
                if await one_wish(reader, earlier, text) and await one_wish(reader, text, earlier):
                    home = shown[n - 1]
                    break
            if home is None:
                groups.append({"text": text, "rule": rule, "said": [text], "ids": [s["id"]]})
            else:
                home["said"].append(text)
                home["ids"].append(s["id"])
    except Stopped:
        return None  # the user is back: what was found before stands, the rest is asked next time
    same = {i: g["ids"][0] for g in groups for i in g["ids"][1:]}
    if same != (store.get(SAME) or {}):
        await asyncio.to_thread(store.set, SAME, same)
    return len(same)


def retire(store, ids):
    """Retire signals a newer preference replaced. Their rows stay, so a
    session that is read again does not bring them back."""
    with store.connect() as db:
        for sid in ids:
            db.execute(
                "UPDATE signals SET retired_at=? WHERE model_id=? AND id=?", (time.time(), store.model_id, sid)
            )
        db.execute(
            "DELETE FROM practice WHERE model_id=? AND signal_id IN "
            "(SELECT id FROM signals WHERE model_id=? AND retired_at IS NOT NULL)",
            (store.model_id, store.model_id),
        )


def load_session(store, ids):
    with store.connect() as db:
        rows = timeline.load(db, ids)
    return rows, timeline.build(rows)


async def read(store, reader, *, limit=None, on_session=None, newest=False, seconds=None,
               on_start=None):
    """Read pending sessions until done or the user returns. Returns how many
    sessions were read.

    newest: most recent first (Quick Dream wants recent feedback); seconds: stop
    starting new sessions after this long (the rest is read later)."""
    import time

    done, started, stopped = 0, time.monotonic(), False
    found = await asyncio.to_thread(pending, store)
    if newest:
        found = found[::-1]
    for session, ids in found[:limit] if limit else found:
        if seconds is not None and time.monotonic() - started >= seconds:
            break
        rows, items = await asyncio.to_thread(load_session, store, ids)
        if on_start:
            on_start(session, items)  # which conversation is being read now
        errors = getattr(reader, "errors", 0)
        try:
            found = await read_session(session, items, rows, reader)
        except Stopped:
            stopped = True
            break
        if getattr(reader, "errors", 0) > errors:
            # Some of the model's answers could not be read: the session stays
            # unread (read again later), and the run says so (live: LM Studio
            # answers in the reasoning channel; 0 signals, marked read).
            store_failed = getattr(reader, "failed_sessions", [])
            reader.failed_sessions = [*store_failed, session]
            continue
        await asyncio.to_thread(save, store, session, max(ids), found, items)
        done += 1
        if on_session:
            on_session(session, found)
    if not stopped:
        await retire_replaced(store, reader)
        await fold_same(store, reader)
    return done
