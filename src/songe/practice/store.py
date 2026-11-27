"""Write practice scenes for a learner's signals, check them, and keep them.

Each signal with a lesson gets `practice_per_signal` scenes, written by the
learner's own loaded model (sampled, so variants differ) in idle gaps, the
same way reading runs. A scene is framed on the logged request at the
signal's responsible step: its real system prompt and tool schemas, with the
working directory moved to a sibling folder. A plain chat (a request with no
tools) gets new conversations instead: each opens with a message of the same
kind as the one the user sent there. Scenes are stored with their
check results; only kept ones are trained on. A signal that is replaced (its
session was read again) takes its scenes with it.
"""

import asyncio
import json
import logging
import re
import time

from ..reading import timeline
from ..reading.client import Stopped
from ..reading.views import shorten_paths
from . import prompts as P
from .checks import check
from .frame import build, frame_from_request

logger = logging.getLogger(__name__)

# A scene that fails (the model server reloading after calibration, a
# timeout) is tried again after these waits, then skipped; this many failed
# scenes in a row stop the run.
RETRY_WAITS = (10, 30)
FAILED_IN_A_ROW = 3
AT_ONCE = 8  # scenes of one plain-chat lesson answered at the same time


class PracticeFailed(RuntimeError):
    """Practice writing stopped after repeated failures."""

# Signals that teach something. Approvals confirm work; successes are the
# drift set (never trained on).
LESSON_TYPES = ("correction", "retry", "tool_error", "failed_result", "outcome")


def clusters(signals, have=None):
    """Signals that teach the same thing, as lists (the one practised first).

    A rule the user stated groups only with the same sentence of theirs;
    other lessons group by word overlap, as Quick Dream's candidates do. The
    signal practised for a cluster is the one that already has the most
    scenes (so a newer look-alike does not start over), else the newest."""
    from ..quick import normal, similar
    from ..reading.rules import stated_sentence

    have = have or {}
    groups = []
    for s in sorted(signals, key=lambda s: s.get("revision") or 0, reverse=True):
        rule = stated_sentence(s) if s.get("from") == "rules" and s.get("source") == "user" else None
        text = rule or s["lesson"].strip()
        if rule:
            home = next((g for g in groups if g["rule"] and normal(g["text"]) == normal(rule)), None)
        else:
            home = next((g for g in groups if not g["rule"] and similar(text, g["text"])), None)
        if home:
            home["members"].append(s)
        else:
            groups.append({"text": text, "rule": bool(rule), "members": [s]})
    out = []
    for g in groups:
        first = max(g["members"], key=lambda s: (have.get(s["id"], 0), s.get("revision") or 0))
        out.append([first] + [s for s in g["members"] if s is not first])
    return out


def lesson_signals(store):
    """The signals Deep Dream practises: those of LESSON_TYPES with the
    evidence a note needs, and no others (quick.admitted, one bar for both).
    A lesson that does not fit the prompt (one kind of task) is practised
    all the same."""
    from ..quick import admitted, evidenced
    from ..reading.store import signals

    allowed = {i for cluster in admitted(store, fits=evidenced) for i in cluster["signals"]}
    return [
        s for s in signals(store, types=LESSON_TYPES)
        if s.get("lesson") and s.get("exchanges") and s["id"] in allowed
    ]


def cluster_of(store):
    """{signal id: ids of every signal in its cluster}: training one
    cluster's scenes covers all of them (quick.absorb, trained bookkeeping)."""
    out = {}
    for group in clusters(lesson_signals(store)):
        ids = [s["id"] for s in group]
        for i in ids:
            out[i] = ids
    return out


def wanted(store, per_signal):
    """[(signal, variants still to write)] for this learner, oldest first:
    one set of scenes per cluster of signals that teach the same thing (live:
    six identical corrections wrote up to 18 scenes on one rule)."""
    with store.connect() as db:
        have = {}
        for signal_id, n in db.execute(
            "SELECT signal_id, COUNT(*) FROM practice WHERE model_id=?"
            " AND json_extract(payload,'$.ordinary') IS NULL GROUP BY signal_id",
            (store.model_id,),
        ):
            have[signal_id] = n
    out = []
    for group in clusters(lesson_signals(store), have):
        s = group[0]
        written = max(have.get(m["id"], 0) for m in group)
        missing = [v for v in range(1, per_signal + 1) if v > written]
        if missing:
            out.append((s, missing))
    # The user's own rules first: a capped Dream must not leave them out.
    out.sort(key=lambda pair: pair[0].get("from") != "rules")
    return out


def practice(store, *, kept=None, ordinary=False):
    """The written scenes, or (ordinary=True) the ordinary rows beside them
    (write_ordinary)."""
    with store.connect() as db:
        rows = db.execute(
            "SELECT payload FROM practice WHERE model_id=?"
            + ("" if kept is None else " AND kept=?")
            + f" AND json_extract(payload,'$.ordinary') IS {'NOT ' if ordinary else ''}NULL"
            + " ORDER BY created",
            (store.model_id,) if kept is None else (store.model_id, int(kept)),
        ).fetchall()
    return [json.loads(r[0]) for r in rows]


def counts(store):
    with store.connect() as db:
        rows = db.execute(
            "SELECT kept, \"drop\", COUNT(*) FROM practice WHERE model_id=?"
            " AND json_extract(payload,'$.ordinary') IS NULL GROUP BY kept, \"drop\"",
            (store.model_id,),
        ).fetchall()
    written = sum(n for _, _, n in rows)
    kept = sum(n for k, _, n in rows if k)
    dropped_by = {d: n for k, d, n in rows if not k and d}
    return {"written": written, "kept": kept, "dropped": written - kept, "dropped_by": dropped_by}


def save(store, row):
    with store.connect() as db:
        db.execute(
            "INSERT OR REPLACE INTO practice(id,model_id,signal_id,payload,kept,\"drop\",created) VALUES (?,?,?,?,?,?,?)",
            (
                row["id"],
                store.model_id,
                row["signal_id"],
                json.dumps(row, ensure_ascii=False),
                int(row["drop"] is None),
                row["drop"],
                time.time(),
            ),
        )


def source(store, signal):
    """The logged request at the earliest responsible step, and the session's
    timeline (for the copy check)."""
    step = min(int(n) for n in signal["exchanges"])
    exchange = signal["exchanges"][str(step)]
    from ..parts import request_of

    with store.connect() as db:
        request = request_of(db, exchange)
        ids = timeline.sessions(db).get(signal["session"], [])
        rows = timeline.load(db, ids) if ids else []
    return request, timeline.build(rows)


def project_name(scene):
    name = re.sub(r"[^a-z0-9-]+", "-", str(scene.get("project") or "").lower()).strip("-")
    return name[:40] or "project"


async def write_one(store, reader, signal, variant, request, items, lessons, workdir):
    # Stable across runs (str hash() is salted per process).
    topic = P.TOPICS[(int(signal["id"][:8], 16) + variant) % len(P.TOPICS)]
    tools = [
        t for t in request.get("tools") or [] if t.get("function", {}).get("name") in P.DREAM_TOOLS
    ]
    prompt = P.DREAM.format(
        lesson=signal["lesson"],
        instance=timeline.clip(shorten_paths(signal.get("instance") or "", workdir), 600),
        distance=P.DISTANCE.get(signal.get("generality"), P.DISTANCE["kind"]).format(topic=topic),
        variant=variant,
        tools=P.tools_section(request.get("tools"), tools),
        examples=json.dumps(P.TEMPLATE),
    )
    scene = await reader.ask(prompt, P.DREAM_SCHEMA, "dream", temperature=0.8, max_tokens=4000)
    row = {
        "id": f"{signal['id']}#v{variant}",
        "signal_id": signal["id"],
        "signal": {k: signal.get(k) for k in ("type", "source", "lesson", "generality", "session", "from")},
        "variant": variant,
        "topic": topic if signal.get("generality") == "always" else None,
        "scene": scene,
        "teacher_note": P.teacher_note(signal),
    }
    if "error" in scene:
        row["problems"] = [scene["error"]]
    else:
        frame = frame_from_request(request, project_name(scene))
        messages, problems, files = build(scene, frame)
        row.update(messages=messages, tools=frame["tools"], problems=problems, final_files=files)
    row["checks"], row["drop"], row["why"] = await check(row, signal, items, reader, lessons)
    return row


def asked(request):
    """The user's last message in a logged request: what they asked for
    right before the answer their feedback is about."""
    from ..steps import CONTINUED

    for message in reversed(request.get("messages") or []):
        said = timeline.text(message.get("content")).strip()
        if message.get("role") == "user" and said and not said.startswith(CONTINUED):
            return said
    return ""


LIKE_BATCH = 12


def answered(store, signal):
    """The answer the user's feedback was about: the logged response at the
    earliest responsible step ("" when it is gone or has no text)."""
    from ..parts import request_of

    step = min(int(n) for n in signal["exchanges"])
    with store.connect() as db:
        row = request_of(db, signal["exchanges"][str(step)], "*")
    try:
        response = json.loads((row or {}).get("response") or "null")
    except (TypeError, ValueError):
        return ""
    message = timeline.response_message(response) or {}
    return timeline.text(message.get("content")).strip()


async def like_messages(reader, request, count, answer=""):
    """`count` new user messages of the same kind as the one a plain chat's
    lesson came from, asked for twelve at a time (fewer when the model wrote
    fewer). With the answer the feedback was about, the writer sees it too."""
    seen, messages = set(), []
    for _ in range(-(-count // LIKE_BATCH) + 1):
        n = min(LIKE_BATCH, count - len(messages))
        if n <= 0:
            break
        said = timeline.clip(asked(request), 1200)
        prompt = (
            P.LIKE_ANSWER.format(asked=said, answer=answer[:1200], count=n)
            if answer
            else P.LIKE.format(asked=said, count=n)
        )
        if messages:
            # A second call with the same words would get the first call's
            # answer back (the reader caches by question): it is told what
            # is written already.
            prompt = prompt.replace(
                "\n\nAnswer with JSON only",
                "\n\nAlready written, write different ones:\n"
                + "\n".join(f"- {m}" for m in messages)
                + "\n\nAnswer with JSON only",
            )
        got = await reader.ask(
            prompt, P.LIKE_SCHEMA, "dream", temperature=0.8, max_tokens=300 + 150 * n
        )
        for text in got.get("messages") or []:
            text = text.strip() if isinstance(text, str) else ""
            if text and text.lower() not in seen:
                seen.add(text.lower())
                messages.append(text)
    return messages[:count]


async def write_like(store, reader, signal, variant, request, items, lessons, text):
    """One scene of a plain chat: a new conversation that opens with `text`,
    under the logged system prompt if there was one. An empty text (the
    model wrote fewer messages than asked) is a scene that did not build."""
    row = {
        "id": f"{signal['id']}#v{variant}",
        "signal_id": signal["id"],
        "signal": {k: signal.get(k) for k in ("type", "source", "lesson", "generality", "session", "from")},
        "variant": variant,
        "topic": None,
        "scene": {"request": text} if text else {"error": "no message written"},
        "teacher_note": P.teacher_note(signal),
    }
    if text:
        frame = frame_from_request(request, "chat")
        system = [{"role": "system", "content": frame["system"]}] if frame["system"] else []
        row.update(
            messages=system + [{"role": "user", "content": text}],
            tools=[],
            problems=[],
            final_files={},
        )
    else:
        row["problems"] = [row["scene"]["error"]]
    row["checks"], row["drop"], row["why"] = await check(
        row, signal, items, reader, lessons, plain=True
    )
    return row


async def scene_row(store, reader, signal, v, request, items, lessons, workdir, text):
    """One scene with its retries, as (row, None); (None, the last error)
    after the last retry; or (Stopped, None), returned and not raised so the
    scenes answered beside it are still saved. `text` is the new message of a
    plain chat's scene, None for a scene with tools."""
    for wait in (*RETRY_WAITS, None):
        try:
            if text is not None:
                return await write_like(store, reader, signal, v, request, items, lessons, text), None
            return await write_one(store, reader, signal, v, request, items, lessons, workdir), None
        except Stopped as stop:
            return stop, None
        except Exception as exc:
            logger.warning("practice_scene_failed %s#v%s: %s: %s", signal["id"], v, type(exc).__name__, exc)
            if wait is None:
                return None, exc
            await asyncio.sleep(wait)


ORDINARY_TOKENS = 500


def ordinary_requests():
    """The bundled ordinary requests (practice/ordinary.json): questions and
    small writing tasks no lesson is about, and none of the prompts the
    drift check measures on (sleep/anchors.json)."""
    from pathlib import Path

    return json.loads((Path(__file__).parent / "ordinary.json").read_text(encoding="utf-8"))


def ordinary_wanted(store):
    """(the ordinary rows still to write as [(a kept scene of the lesson,
    the row's number)], the requests not used in this store yet).

    One ordinary row per kept scene that is trained along its teacher's
    answer (a plain chat's scene), numbered from 1 within its lesson. They
    are written in turn across the lessons, each bundled request once in a
    store: when the requests are used up, every lesson has about as many."""
    scenes, have, used = {}, {}, set()
    for row in practice(store, ordinary=True):
        have.setdefault(row["signal_id"], set()).add(row["variant"])
        used.add(row["scene"]["request"])
    for row in practice(store, kept=True):
        if row.get("messages") and (row.get("teacher_answer") or {}).get("content"):
            scenes.setdefault(row["signal_id"], []).append(row)
    todo = [
        (rows[0], n)
        for signal_id, rows in scenes.items()
        for n in range(1, len(rows) + 1)
        if n not in have.get(signal_id, ())
    ]
    todo.sort(key=lambda pair: pair[1])  # stable: the lessons keep their order within a turn
    return todo, [r for r in ordinary_requests() if r not in used]


async def write_ordinary(store, reader):
    """Ordinary rows for the lessons that lack them. Returns how many were
    kept.

    A Deep Dream that trains only on scenes where the lesson applies moves
    the model's answers everywhere (measured on a rented card, Qwen3.5-4B,
    16 lessons: drift at full strength 0.046 to 0.050 against a limit of
    0.05). An ordinary row is a request no lesson is about with the served
    model's own answer, no note for the teacher: trained along that answer
    (as a plain scene is along its teacher's), the student is held to what
    it does today where no lesson applies. With one such row per kept scene,
    up to the 192 bundled requests, the same Dreams measured 0.015 to 0.019.

    The requests are their own set: the drift check's prompts are never
    trained on, or it would measure what it was trained to pass. A row opens
    with the system message its lesson's scenes open with. An answer the
    server cut short is not stored to train along (kept=0, so the request is
    not asked again)."""
    todo, free = await asyncio.to_thread(ordinary_wanted, store)
    kept = 0
    for scene, n in todo:
        if not free:
            break
        request = free.pop(0)
        system = [m for m in scene["messages"] if m.get("role") == "system"][:1]
        messages = system + [{"role": "user", "content": request}]
        try:
            # Reasoning off, as a plain scene's answer is once a reasoning
            # server's thinking took the room: the answer trained along is
            # the visible reply.
            answer = await reader.reply(messages, max_tokens=ORDINARY_TOKENS, think=False)
        except Stopped:
            break
        whole = answer["finished"] and not answer["reasoned"] and bool(answer["content"].strip())
        row = {
            "id": f"{scene['signal_id']}#o{n}", "signal_id": scene["signal_id"], "signal": scene["signal"],
            "variant": n, "topic": None, "scene": {"request": request}, "ordinary": True,
            "teacher_note": "", "messages": messages, "tools": [], "problems": [], "final_files": {},
            "teacher_answer": {"content": answer["content"], "finished": True} if whole else None,
            "checks": {"built": True}, "drop": None if whole else "cut",
            "why": {"ordinary": "a request no lesson is about, trained along the served model's own answer"},
        }
        await asyncio.to_thread(save, store, row)
        kept += whole
    return kept


async def write(store, reader, per_signal, *, limit=None):
    """Write missing scenes until done or the user returns. Returns how many
    scenes were written (kept or not)."""
    from ..reading.extract import workdir_of
    from ..reading.store import signals

    todo = await asyncio.to_thread(wanted, store, per_signal)
    lessons = [(s["session"], s["lesson"]) for s in await asyncio.to_thread(signals, store) if s.get("lesson")]
    done = 0
    failed, last = 0, None
    for signal, variants in todo[:limit] if limit else todo:
        request, items = await asyncio.to_thread(source, store, signal)
        if not request:
            continue
        workdir = workdir_of([(None, request, None, None)])
        # A plain chat (no tools): every missing scene is a new message of the
        # kind the user sent, all written in one answer.
        plain = not request.get("tools") and bool(asked(request))
        texts = None
        if plain:
            for wait in (*RETRY_WAITS, None):
                try:
                    texts = await like_messages(
                        reader, request, len(variants),
                        await asyncio.to_thread(answered, store, signal),
                    )
                    break
                except Stopped:
                    return done
                except Exception as exc:
                    logger.warning("practice_scene_failed %s: %s: %s", signal["id"], type(exc).__name__, exc)
                    last = exc
                    if wait is None:
                        break
                    await asyncio.sleep(wait)
            if texts is None:
                failed += 1
                if failed >= FAILED_IN_A_ROW:
                    raise PracticeFailed(f"{type(last).__name__}: {last}") from last
                continue

        # A plain chat's scenes do not share a workspace: AT_ONCE of them are
        # answered at the same time. Scenes with tools stay one at a time.
        step = AT_ONCE if plain else 1
        for start in range(0, len(variants), step):
            got = await asyncio.gather(
                *(
                    scene_row(
                        store, reader, signal, v, request, items, lessons, workdir,
                        (texts[variants.index(v)] if variants.index(v) < len(texts) else "") if plain else None,
                    )
                    for v in variants[start : start + step]
                )
            )
            stopped = False
            for row, error in got:
                if isinstance(row, Stopped):
                    stopped = True
                elif row is None:
                    failed += 1
                    last = error
                else:
                    failed = 0
                    await asyncio.to_thread(save, store, row)
                    done += 1
            if stopped:
                return done
            if failed >= FAILED_IN_A_ROW:
                raise PracticeFailed(f"{type(last).__name__}: {last}") from last
    ordinary = await write_ordinary(store, reader)
    if ordinary:
        logger.info("practice_ordinary %s rows", ordinary)
    return done
