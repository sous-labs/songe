"""Did the candidate get better? Judged on what each version writes.

At each held-out moment (the step the user reacted to, or a held-out practice
scene) the version in use and the candidate answer from the context alone.
The same action as written (quick.identical) is a tie. Of two different ones each is shown on its own to
the original base model, with the user's request and what should guide the
answer (their feedback, or the lesson): does this answer do what was asked,
yes or no. A win when only the candidate's does, a loss when only the
answer in use does, a tie when both or neither do or a reply cannot be
read. The judge is never asked which of two answers it prefers (outcome
says why). The feedback-aware teacher's preference does not break a tie:
training pulls the candidate towards that preference, so it would count
the training itself as a win.

The same answers are also written at a few moments the lessons are not
about (unrelated, acts_differently): how often the candidate acts
differently there is kept in the record and decides nothing.

The gate (gate.decide) scores each unit (conversation or lesson) as
(wins - losses) / moments and resamples units. The KL to the teacher stays a
diagnostic: it scored the original model's text, not the candidate's.
"""

from __future__ import annotations

import logging
import re

from ..quick import differs, identical, verdict_of, visible
from .examples import answer
from .prompts import paired_prompts, render

ANSWER_TOKENS = 384   # each version's answer: long enough for a tool call
JUDGE_TOKENS = 48     # {"follows": "yes"}, with room for a short preamble
PER_UNIT = 3          # moments per conversation or lesson (its last ones)
MOMENTS = 24          # moments per Dream at most
UNRELATED = 8         # confirmed successes answered again, one per conversation

logger = logging.getLogger("songe")

DOES = """A user asked a coding agent:
{request}

The user's feedback on an earlier answer of this kind (or a lesson):
{guide}

The agent's answer now:
{answer}

Does this answer do what the feedback asks? Judge only that, not whether the answer is long, complete or helpful. Reply as JSON: {{"follows": "yes"}} or {{"follows": "no"}}."""


def shown(a, limit=1500):
    """An answer as one text for the judge: what the user would see."""
    parts = [visible(a["text"])] if visible(a["text"]) else []
    parts += [f"{c.get('name')}({c.get('arguments')})" for c in a["tool_calls"]]
    text = "\n".join(parts) or "(nothing)"
    return text if len(text) <= limit else text[:limit] + " …"


def said(content):
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content if isinstance(content, str) else ""


def moments(steps):
    """The gate's held-out steps to judge: each unit's PER_UNIT sampled
    moments (practice.pool.sampled, the same ones the gate scores), units in
    turn so every one counts before any gets more, MOMENTS at most."""
    from ..practice.pool import sampled

    units = {}
    for s in steps:
        units.setdefault(s["conversation"], []).append(s)
    queues = [sampled(ss, PER_UNIT) for ss in units.values()]
    out = []
    while any(queues) and len(out) < MOMENTS:
        for q in queues:
            if q and len(out) < MOMENTS:
                out.append(q.pop(0))
    return out


def unrelated(steps):
    """Moments the lessons are not about, to see whether the candidate still
    acts there as the version in use does: the drift set's confirmed
    successes (the base was right, nothing was trained on them), one per
    conversation (its sampled one, the same every Dream), UNRELATED at most."""
    from ..practice.pool import sampled

    units = {}
    for s in steps:
        units.setdefault(s["conversation"], []).append(s)
    return [sampled(ss, 1)[0] for ss in units.values()][:UNRELATED]


def acts_differently(journal, steps, before, name):
    """{changed, checked}: at how many of these moments the version that
    answered as `name` takes another action than `before` (quick.differs,
    the gate's own notion of the same action). Read from the journal only:
    a moment that version did not answer is not counted."""
    changed = checked = 0
    for step in steps:
        after = journal.load(f"judged-{name}-{step['id']}", None)
        if after is None or step["id"] not in before:
            continue
        checked += 1
        changed += bool(differs(before[step["id"]], after))
    return {"changed": changed, "checked": checked}


def guide_of(step):
    """What the answer should follow: the lesson (practice), else the user's
    feedback that came after this step."""
    if step.get("lesson"):
        return step["lesson"]
    for m in step.get("hindsight") or []:
        if m.get("role") == "user" and said(m.get("content")).strip():
            return said(m.get("content")).strip()[:600]
    return ""


def request_of(step):
    for m in reversed(step.get("context") or []):
        if m.get("role") == "user" and said(m.get("content")).strip():
            return said(m.get("content")).strip()[:600]
    return ""


def write(backend, step, max_tokens=ANSWER_TOKENS, notes=""):
    """The loaded version's own answer at this moment (deterministic).
    notes: the lessons block it is served with, put where the proxy puts it."""
    if notes:
        from ..proxy import with_lessons

        step = step | {"context": with_lessons(step.get("context") or [], notes)}
    student, _ = paired_prompts(backend.tokenizer, step, backend.config)
    y, text, _ = backend.sample(student, deterministic=True, max_tokens=max_tokens)
    return {"tokens": list(y), **answer(text, backend.tokenizer, step.get("tools"))}


def answers(backend, journal, steps, name, check, kept_as=None, notes=""):
    """Every moment's answer by the loaded version, resumable. kept_as: a
    KeptMeasurements (the incumbent's answers, reused on later Dreams).
    notes: the lessons block the version is served with (write)."""
    from ..steps import digest, revision_id
    from .evaluate import kept

    served = [digest(notes)[:16]] if notes else []
    out = {}
    for step in steps:
        check()
        out[step["id"]] = journal.unit(
            "-".join(["judged", name, *served, step["id"]]),
            lambda step=step: kept(kept_as, f"answer{ANSWER_TOKENS}", [step["id"], revision_id(step), *served],
                                   lambda: write(backend, step, notes=notes)),
        )
    return out


PREFILL = '{"follows": "'


def ask(backend, prompt):
    """The judge's reply as JSON text. Its answer is begun for it
    ('{"follows": "'), so only the letter is left to write, and thinking is
    switched off where the template has the switch (live: replies the
    reader could not use, a 3B model starting to reason instead)."""
    tok = backend.tokenizer
    ids = render(tok, [{"role": "user", "content": prompt}], None, enable_thinking=False)
    try:
        start = list(tok.encode(PREFILL, add_special_tokens=False))
    except Exception:  # noqa: BLE001 - a tokenizer without this form: no prefill
        start = []
    _, text, _ = backend.sample(list(ids) + start if start else ids, deterministic=True, max_tokens=JUDGE_TOKENS)
    return (PREFILL + text) if start else text


def does(backend, step, answer):
    """Whether this answer, on its own, does what the user's feedback at this
    moment asks: True, False, or None when the judge's reply cannot be read.
    Asked of the original base weights (no self-preference)."""
    request, guide = request_of(step) or "(not shown)", guide_of(step) or "(none)"
    with backend.base():
        text = ask(backend, DOES.format(request=request, guide=guide, answer=shown(answer)))
    found = verdict(text)
    if found is None:
        # Live: about 4 of 14 replies could not be read; what they say.
        logger.info("judge_unreadable %s: %r", step["id"], (text or "")[:200])
        return None
    return found


def verdict(text):
    """True for yes, False for no, None when the reply says neither. The
    begun reply ('{"follows": "yes"...'), else the JSON anywhere after a
    preamble or a thought (a tokenizer without the prefill); "not sure" or
    "nope" is not a no."""
    text = text or ""
    found = re.match(r'\s*\{\s*"follows"\s*:\s*"\s*(yes|no)(?![a-z])', text, re.I) or re.search(
        r'"follows"\s*:\s*"\s*(yes|no)\s*"', visible(text), re.I)
    return found.group(1).lower() == "yes" if found else None


def outcome(backend, step, before, after):
    """win / loss / tie for the candidate at this moment, and why.

    Each answer is judged on its own against the user's feedback: a win when
    only the candidate's does what was asked, a loss when only the answer in
    use does, a tie otherwise. The judge is never asked to prefer one of two
    answers. Asked that (in both orders, counting only an agreeing pick), a
    4B judge gave a verdict on two of three pairs that differed in anything,
    a fenced command against the bare one, a sentence against the bare date,
    in either direction: a candidate no check of the eight test habits
    found worse at any of 24 moments was refused as better on 3, worse on 8."""
    if identical(before, after):
        return "tie", "same"
    was, now = does(backend, step, before), does(backend, step, after)
    if was is None or now is None:
        return "tie", "unclear"
    if now and not was:
        return "win", "judge"
    if was and not now:
        return "loss", "judge"
    return "tie", "both" if now else "neither"


def judge(backend, journal, steps, before, after, name, check):
    """[{id, conversation, outcome, by}] for every moment, resumable."""
    rows = []
    for step in steps:
        check()
        result = journal.unit(
            f"judged-{name}-verdict-{step['id']}",
            lambda step=step: list(outcome(backend, step, before[step["id"]], after[step["id"]])),
        )
        rows.append({"id": step["id"], "conversation": step["conversation"], "outcome": result[0], "by": result[1]})
    return rows


# Held-out scenes asked per lesson; its note leaves only when the candidate
# follows the lesson on all of them. Live, at 3: two of three notes that left
# took the habit down as served (Spanish 11 -> 7 of 12 test requests, a TL;DR
# 12 -> 6). A habit followed half the time passes 3 of 3 one time in 8 and 6
# of 6 one time in 64; one followed 7 times in 12 passes 6 of 6 one time in 25.
LEARNED_SCENES = 6
LEARNED_TOKENS = 128  # the candidate's answer there: a habit shows early


def follows(backend, step, answer):
    """Does this answer, written without the lesson's note, do what the note
    asks as well as the teacher's stored answer (written with it)? The
    practice check's question (practice.prompts.CHANGES), asked of the
    original base in both orders: yes when it names this answer or both,
    both times. The same action as the teacher's is a yes without asking."""
    from ..practice.prompts import CHANGES

    taught = {"text": (step.get("teacher_answer") or {}).get("content") or "", "tool_calls": []}
    if not visible(taught["text"]):
        return False
    if not differs(taught, answer):
        return True
    request, guide = request_of(step) or "(not shown)", guide_of(step) or "(none)"
    # Yes when it names this answer or both in one order at least. Asked for
    # both orders, a judge that names whichever answer comes first fails an
    # answer it cannot tell from the teacher's, which is "as well as the
    # teacher". Measured on 133 scenes with a mechanical check: of 100
    # answers that follow, both orders passed 61, one order 82; of 33 that
    # do not, 1 and 3.
    named = 0
    for swap in (False, True):
        a, b = (answer, taught) if swap else (taught, answer)
        with backend.base():
            text = ask(backend, CHANGES.format(request=request, guide=guide, a=shown(a), b=shown(b)))
        named += verdict_of(text, swap) in ("after", "both")
    return named > 0


def learned(backend, journal, lessons, scenes, name, check):
    """Which of the prompt's lessons the loaded candidate follows without
    their note: [{"id", "text", "signals", "scenes", "follows", "learned"}].

    lessons: the lessons in the prompt (quick.active), each with the signals
    it stands for. scenes: {signal id: its held-out practice scenes}, never
    trained on. Up to LEARNED_SCENES of a lesson's scenes are answered by the
    candidate as it would be served once this lesson's note has left: with
    the other lessons' notes in the prompt and without its own. (Live: a
    version that used pnpm on 11 of 16 requests with no notes in the prompt
    did on 1 of 16 with the uv note beside it; tested with no notes at all,
    the pnpm note left and the habit fell from 10 of 16 to 2.) The lesson is
    learned when LEARNED_SCENES answers all follow it as well as the
    teacher's do. A lesson with fewer held-out scenes cannot be shown
    learned, so its note stays."""
    from .. import quick

    out = []
    for lesson in lessons:
        beside = quick.block([other for other in lessons if other.get("id") != lesson.get("id")])
        own = {
            s["id"]: s
            for signal in lesson.get("signals") or []
            for s in scenes.get(signal, [])
            if (s.get("teacher_answer") or {}).get("content")
        }
        own = [own[i] for i in sorted(own)][:LEARNED_SCENES]
        passed = 0
        for step in own:
            check()
            passed += bool(journal.unit(
                f"learned-{name}-{step['id']}",
                lambda step=step, beside=beside: follows(
                    backend, step, write(backend, step, LEARNED_TOKENS, notes=beside)
                ),
            ))
        out.append({"id": lesson.get("id"), "text": lesson.get("text"), "signals": list(lesson.get("signals") or []),
                    "scenes": len(own), "follows": passed,
                    "learned": len(own) >= LEARNED_SCENES and passed == len(own)})
    return out


def learned_lessons(backend, journal, store, name, check):
    """learned() for this store: the lessons in its prompt against their
    held-out practice scenes. One cluster's scenes stand for every signal
    that teaches the same thing (practice.store.cluster_of)."""
    from .. import quick
    from ..practice.pool import practice_steps
    from ..practice.store import cluster_of

    same, scenes = cluster_of(store), {}
    for s in practice_steps(store, split="held_out"):
        for member in same.get(s["signal_id"], [s["signal_id"]]):
            scenes.setdefault(member, []).append(s)
    return learned(backend, journal, quick.active(store), scenes, name, check)


def kept_back(lessons, rows, session):
    """The lessons that passed the note test and keep their note all the
    same: {lesson id}. lessons: learned()'s rows. rows: the judged moments,
    the candidate answering without those lessons' notes. session: {signal
    id: the conversation it came from}. A lesson whose own conversation is
    among the moments judged worse is not learned as the user is served: on
    three held-out scenes the candidate passed, in the user's conversation it
    did not."""
    lost = {r["conversation"] for r in rows if r["outcome"] == "loss"}
    return {
        row["id"] for row in lessons
        if row["learned"] and {session.get(sig) for sig in row["signals"]} & lost
    }


def examples(steps, before, after, rows, limit=3):
    """The report's before/after: the moments the candidate won, in the
    shape the cycle record shows (real conversations first)."""
    by_id = {s["id"]: s for s in steps}
    won = [r for r in rows if r["outcome"] == "win"]
    won.sort(key=lambda r: bool(by_id[r["id"]].get("practice")))  # real conversations first
    out = []
    for r in won[:limit]:
        s = by_id[r["id"]]
        practice = bool(s.get("practice"))
        out.append({
            "id": s["id"], "source": "practice" if practice else "original",
            "split": "held_out" if practice else "original",
            "signal": {"id": s.get("signal_id") or s["conversation"], "type": None,
                       "source": None if practice else "user"},
            "context": {"request": request_of(s)[:300], "conversation": None, "at": None},
            "feedback": None if practice else guide_of(s)[:200],
            "lesson": s.get("lesson"),
            "before": {k: before[s["id"]][k] for k in ("text", "tool_calls", "valid")},
            "after": {k: after[s["id"]][k] for k in ("text", "tool_calls", "valid")},
            "improved": True, "verdict": "after", "by": r["by"], "teacher_preference": 0.0,
        })
    return out


def summary(rows):
    count = {k: sum(r["outcome"] == k for r in rows) for k in ("win", "loss", "tie")}
    return {"count": count["win"], "worse": count["loss"], "same": count["tie"],
            "checked": len(rows), "available": len(rows)}
