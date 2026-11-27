"""Checks a practice scene must pass to be trained on (ported from
scripts/dream/validate.py). Nothing in a scene is ever run: files are only
parsed.

- built: the frame rendered every step (real tools, valid arguments, files
  that exist, edits that apply)
- decision: the scene ends where the agent must act (a user message or a
  tool result is last)
- parses: made-up Python and JSON files parse
- not_copy: no function names or request wording shared with the real
  session it came from (a shared file name alone is not a copy)
- matters: shown the scene, its own lesson and a distinct lesson from
  another session, the model picks its own. Asked about a single lesson,
  the model said yes to every scene. With no other session's lesson to
  compare with (a first session), a stock lesson stands in, so the check
  always runs. Not asked of a plain chat's scenes: they never state the
  lesson, and the model then picked "neither" for 165 of 183 good ones.
- changes (a plain chat's scenes): the served model answers the scene twice,
  alone and with the scene's note, and a judge is asked in both orders
  which answer does what the user's feedback asks. Kept when it picks the
  answer with the note in at least one order and the other answer in
  none: the scene is one where the lesson changes what the model does.
  That answer is stored with the scene.

Drop codes are the first failed check, in this order (docs/CONTRACTS.md).
"""

import ast
import hashlib
import json
import re
import unicodedata

from ..reading.timeline import clip

GENERIC_NAMES = {"main", "test", "run", "parse", "load", "init", "__init__", "setup", "helper"}

MATTERS_SCHEMA = {
    "type": "object",
    "properties": {
        "why": {"type": "string"},
        "answer": {"type": "string", "enum": ["A", "B", "neither"]},
    },
    "required": ["why", "answer"],
}

MATTERS = """Here is a practice situation for a coding agent. It stops where the agent must choose its next action.

{dream}

Two lessons:
A: {a}
B: {b}

Which lesson is about the agent's next action in this situation? Answer "neither" if neither applies.
Answer with JSON: why (one sentence), then answer (A, B or neither)."""

ORDER = ("built", "decision", "parses", "not_copy", "matters", "changes")

# Each of the two answers of the changes check, in tokens at most: the same
# room the gate gives an answer it judges (sleep/judged.py).
ANSWER_TOKENS = 384
# A server that reasons first can spend all of that reasoning and return no
# reply. The two answers are then asked again with the reader's reasoning
# switches off, and, where the server ignores those, the way its users ask
# with this much more room. (HOW_ASKED, in the order tried.)
REASONING_ROOM = 2048
HOW_ASKED = ({}, {"think": False}, {"max_tokens": ANSWER_TOKENS + REASONING_ROOM})
FOLLOWS_SCHEMA = {
    "type": "object",
    "properties": {"follows": {"type": "string", "enum": ["A", "B", "both", "neither"]}},
    "required": ["follows"],
}

# Stand-ins for "a lesson from another session" when there is none yet: plain
# coding-agent advice on unrelated subjects.
STOCK_LESSONS = (
    "Write commit messages in the imperative mood, under 72 characters.",
    "Pin dependency versions in the lockfile before a release.",
    "Add a docstring to every public function.",
    "Store timestamps in UTC and convert them only for display.",
    "Give database migrations a matching rollback step.",
)


def render_text(messages, limit=4000):
    """The scene as the agent would see it, minus the system prompt."""
    out = []
    for m in messages or []:
        if m["role"] == "user":
            out.append(f"USER: {m['content']}")
        elif m["role"] == "assistant":
            for c in m.get("tool_calls") or []:
                args = json.loads(c["function"]["arguments"] or "{}")
                shown = ", ".join(f"{k}={clip(str(v), 300)!r}" for k, v in args.items())
                out.append(
                    f"AGENT: {m['content'] or ''}\n  calls {c['function']['name']}({shown})".replace(
                        "AGENT: \n", "AGENT:\n"
                    )
                )
            if not m.get("tool_calls"):
                out.append(f"AGENT: {m['content'] or ''}")
        elif m["role"] == "tool":
            out.append(f"  result: {clip(m['content'], 600)}")
    out.append("(the agent acts next)")
    return clip("\n".join(out), limit)


_TRIGRAM_TOKENS = re.compile(r"[\u3400-\u9fff\uf900-\ufaff\u3040-\u30ff]|\w+")


def words(text):
    return set(re.findall(r"[a-z_][a-z0-9_]{2,}", (text or "").lower()))


def trigrams(text):
    norm = unicodedata.normalize("NFKC", (text or "").lower())
    cleaned = "".join(ch for ch in norm if unicodedata.category(ch) != "Mn" and ch != "\u0640")
    w = _TRIGRAM_TOKENS.findall(cleaned)
    return {tuple(w[i : i + 3]) for i in range(len(w) - 2)}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


# A definition, not prose: the keyword as a word, then the name, then what
# follows a name in code. Tool output also holds English ("the function the
# agent called"), and "the" was then a shared identifier for every scene.
DEFINITION = re.compile(
    r"\b(?:def|function|fn|class|struct)\s+([A-Za-z_]\w*)"
    r"\s*(?:[(:{<;]|(?:extends|implements)\b)"
)


def identifiers(texts):
    """Function and class names defined in some code."""
    names = set()
    for text in texts:
        names.update(DEFINITION.findall(text or ""))
    return names - GENERIC_NAMES


def parse_error(path, text):
    """None when the file parses or is not a kind we check (Python, JSON)."""
    if path.endswith(".py"):
        try:
            ast.parse(text)
        except SyntaxError as e:
            return f"{path}: {e.msg} (line {e.lineno})"
    elif path.endswith(".json"):
        try:
            json.loads(text)
        except ValueError as e:
            return f"{path}: {e}"
    return None


def source_of(items):
    """What the real session showed: its user requests and every bit of code
    the agent read or wrote, for the copy check."""
    requests = [it["content"] for it in items if it["kind"] == "user"]
    code = [it["content"] for it in items if it["kind"] == "tool_result"]
    for it in items:
        for c in it.get("tool_calls") or []:
            code.append(c.get("arguments") or "")
    return requests, code


def copy_evidence(scene, files, requests, code):
    shared = identifiers(code) & (identifiers(files.values()) | words(scene.get("request")))
    # Against each user message on its own: joining them dilutes a copy.
    asked = trigrams(scene.get("request"))
    overlap = max((jaccard(trigrams(r), asked) for r in requests), default=0.0)
    return sorted(shared) + ([f"request overlap {overlap:.2f}"] if overlap > 0.3 else [])


def placebo_for(practice_id, signal, lessons):
    """A deterministic, distinct lesson from another session (a stock lesson
    when there is none) and the prompt order."""
    real = signal["lesson"].strip()
    others = sorted({
        text for session, lesson in lessons
        if session != signal["session"] and (text := lesson.strip()) and text != real
    }) or [text for text in STOCK_LESSONS if text != real]
    h = int(hashlib.sha256(practice_id.encode()).hexdigest(), 16)
    return others[h % len(others)], h % 2 == 0


async def teacher_reply(reader, row, draft=None, **how):
    """The served model's answer to the scene with its note shown; with
    draft (its reply without the note), that reply changed only where the
    note asks (prompts.edit_aside)."""
    import httpx

    from ..sleep.prompts import practice_messages

    how = {"max_tokens": ANSWER_TOKENS} | how
    try:
        return await reader.reply(practice_messages(row["messages"], row["teacher_note"], draft=draft), **how)
    except httpx.HTTPStatusError as exc:
        if not 400 <= exc.response.status_code < 500:
            raise
        # A chat template that takes no second user message in a row.
        return await reader.reply(practice_messages(row["messages"], row["teacher_note"], merged=True, draft=draft), **how)


def lost_in_reasoning(answer):
    """The reply was cut off with reasoning in it and nothing to show."""
    from ..quick import visible

    return answer["reasoned"] and not answer["finished"] and not visible(answer["content"])


async def two_answers(row, reader):
    """(the served model's answer to the scene, its answer with the note
    shown), asked the way its users ask.

    On a server that reasons first the reasoning can take all of
    ANSWER_TOKENS, and an empty reply made every scene look unchanged: such
    a server got no plain-chat practice at all. Both answers are then asked
    again down HOW_ASKED, and the reader remembers what worked for the scenes
    that follow. An answer asked any other way than the first counts as
    reasoned: what its user would see of it is stored to train along."""
    from ..quick import visible

    first = getattr(reader, "reply_how", 0)
    for how in range(first, len(HOW_ASKED)):
        plain = await reader.reply(row["messages"], **({"max_tokens": ANSWER_TOKENS} | HOW_ASKED[how]))
        # The answer with the note is the plain one changed only where the
        # note asks: a format correction must not shorten the content.
        taught = await teacher_reply(reader, row, draft=visible(plain["content"]) or None, **HOW_ASKED[how])
        if not (lost_in_reasoning(plain) or lost_in_reasoning(taught)):
            break
    reader.reply_how = how
    return plain, taught | {"reasoned": taught["reasoned"] or how > 0}


async def changes(row, signal, reader):
    """(kept, why, the answer with the note): does the scene's note change
    what the served model answers, towards the lesson?"""
    from ..quick import verdict_of, visible
    from ..sleep.judged import shown
    from .prompts import CHANGES

    plain, taught = await two_answers(row, reader)
    before, after = ({"text": a["content"], "tool_calls": []} for a in (plain, taught))
    why = {"plain": shown(before, 300), "taught": shown(after, 300)}
    if not visible(after["text"]) or visible(before["text"]) == visible(after["text"]):
        return False, why | {"changes": "same"}, taught
    # The user's own words when there are any: the lesson is a model's
    # reading of them, and a 4B reader's can miss the point ("Just the code."
    # became "ensure all explanatory text is complete").
    guide = clip(signal.get("text") or "", 400).strip() or signal["lesson"]
    request = clip(str(row["messages"][-1].get("content") or ""), 600)
    picks = []
    for swap in (False, True):
        a, b = (after, before) if swap else (before, after)
        answer = await reader.ask(
            CHANGES.format(request=request, guide=guide, a=shown(a), b=shown(b)),
            FOLLOWS_SCHEMA,
            "changes",
            max_tokens=200,
        )
        picks.append(verdict_of(json.dumps(answer), swap))
    why["changes"] = picks
    # One order may come back "neither" or "both" (a 4B judge is unsure in
    # one order for one scene in ten); an order that prefers the answer
    # without the note, or the same position twice, drops the scene.
    return "after" in picks and "before" not in picks, why, taught


async def check(row, signal, items, reader, lessons, *, plain=False):
    """The check results of one scene, its drop code (None when kept) and why.
    plain: a plain chat's scene (see the module docstring)."""
    scene, c = row["scene"], {}
    c["built"] = "error" not in scene and not row.get("problems")
    messages = row.get("messages") or []
    c["decision"] = bool(messages) and messages[-1]["role"] in ("user", "tool")
    files = {f["path"]: f["content"] for f in scene.get("files") or [] if isinstance(f, dict)}
    files.update(row.get("final_files") or {})
    errors = [e for p, t in files.items() if (e := parse_error(p, t or ""))]
    c["parses"] = not errors
    requests, code = source_of(items)
    evidence = copy_evidence(scene, files, requests, code) if c["built"] else []
    c["not_copy"] = not evidence
    c["matters"] = c["changes"] = None
    why = {"parse_errors": errors, "copy_evidence": evidence}
    if plain and all(c[k] for k in ORDER[:4]):
        c["changes"], found, taught = await changes(row, signal, reader)
        why.update(found)
        # What a Deep Dream can train this scene along: the answer with the
        # note. A server that reasons first gets the part its user sees,
        # without the reasoning: reasoning was all it trained on otherwise.
        if c["changes"]:
            from ..quick import visible

            text = visible(taught["content"]) if taught["reasoned"] else taught["content"]
            if text:
                row["teacher_answer"] = {"content": text, "finished": taught["finished"]}
    elif all(c[k] for k in ORDER[:4]):
        placebo, real_first = placebo_for(row["id"], signal, lessons)
        real = signal["lesson"]
        a, b = (real, placebo) if real_first else (placebo, real)
        answer = await reader.ask(
            MATTERS.format(dream=render_text(messages, limit=2500), a=a, b=b),
            MATTERS_SCHEMA,
            "matters",
            max_tokens=600,
        )
        c["matters"] = answer.get("answer") == ("A" if real_first else "B")
        why["matters_answer"] = (
            "real" if c["matters"] else "neither" if answer.get("answer") == "neither" else "placebo"
        )
        why["matters_why"] = answer.get("why")
    drop = next((k for k in ORDER if c[k] is False), None)
    return c, drop, why
