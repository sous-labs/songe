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
            "SELECT signal_id, COUNT(*) FROM practice WHERE model_id=? GROUP BY signal_id",
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


def practice(store, *, kept=None):
    with store.connect() as db:
        rows = db.execute(
            "SELECT payload FROM practice WHERE model_id=?"
            + ("" if kept is None else " AND kept=?")
            + " ORDER BY created",
            (store.model_id,) if kept is None else (store.model_id, int(kept)),
        ).fetchall()
    return [json.loads(r[0]) for r in rows]


def counts(store):
    with store.connect() as db:
        rows = db.execute(
            "SELECT kept, \"drop\", COUNT(*) FROM practice WHERE model_id=? GROUP BY kept, \"drop\"",
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


async def like_messages(reader, request, count):
    """`count` new user messages of the same kind as the one a plain chat's
    lesson came from, in one answer (fewer when the model wrote fewer)."""
    answer = await reader.ask(
        P.LIKE.format(asked=timeline.clip(asked(request), 1200), count=count),
        P.LIKE_SCHEMA,
        "dream",
        temperature=0.8,
        max_tokens=300 + 150 * count,
    )
    seen, messages = set(), []
    for text in answer.get("messages") or []:
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
        for v in variants:
            row = None
            for wait in (*RETRY_WAITS, None):
                try:
                    if plain and texts is None:
                        texts = await like_messages(reader, request, len(variants))
                    row = await (
                        write_like(
                            store, reader, signal, v, request, items, lessons,
                            texts[variants.index(v)] if variants.index(v) < len(texts) else "",
                        )
                        if plain
                        else write_one(store, reader, signal, v, request, items, lessons, workdir)
                    )
                    break
                except Stopped:
                    return done
                except Exception as exc:
                    logger.warning("practice_scene_failed %s#v%s: %s: %s", signal["id"], v, type(exc).__name__, exc)
                    last = exc
                    if wait is None:
                        break
                    await asyncio.sleep(wait)
            if row is None:
                failed += 1
                if failed >= FAILED_IN_A_ROW:
                    raise PracticeFailed(f"{type(last).__name__}: {last}") from last
                continue
            failed = 0
            await asyncio.to_thread(save, store, row)
            done += 1
    return done
