"""Did the candidate get better? Judged on what each version writes.

At each held-out moment (the step the user reacted to, or a held-out practice
scene) the version in use and the candidate answer from the context alone.
The same action is a tie. A different one goes to the original base model,
shown the user's request, what should guide the answer (their feedback, or
the lesson) and the two answers in an order that hides which is new: a win,
a loss or a tie. When it says "both" or "neither", when its two readings
disagree or cannot be read, the moment is a tie. The feedback-aware
teacher's preference does not break it: training pulls the candidate
towards that preference, so it would count the training itself as a win.

The gate (gate.decide) scores each unit (conversation or lesson) as
(wins - losses) / moments and resamples units. The KL to the teacher stays a
diagnostic: it scored the original model's text, not the candidate's.
"""

from __future__ import annotations

import logging

from ..quick import differs, verdict_of, visible
from .examples import answer
from .prompts import paired_prompts, render

ANSWER_TOKENS = 384   # each version's answer: long enough for a tool call
JUDGE_TOKENS = 48     # {"follows": "B"}, with room for a short preamble
PER_UNIT = 3          # moments per conversation or lesson (its last ones)
MOMENTS = 24          # moments per Dream at most

logger = logging.getLogger("songe")

JUDGE = """A user asked a coding agent:
{request}

What the answer should follow (the user's later feedback, or a lesson):
{guide}

Two answers the agent could give at this moment:

Answer A:
{a}

Answer B:
{b}

Which answer better follows it? Reply as JSON: {{"follows": "A"}}, {{"follows": "B"}}, {{"follows": "both"}} or {{"follows": "neither"}}."""


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


def write(backend, step):
    """The loaded version's own answer at this moment (deterministic)."""
    student, _ = paired_prompts(backend.tokenizer, step, backend.config)
    y, text, _ = backend.sample(student, deterministic=True, max_tokens=ANSWER_TOKENS)
    return {"tokens": list(y), **answer(text, backend.tokenizer, step.get("tools"))}


def answers(backend, journal, steps, name, check, kept_as=None):
    """Every moment's answer by the loaded version, resumable. kept_as: a
    KeptMeasurements (the incumbent's answers, reused on later Dreams)."""
    from ..steps import revision_id
    from .evaluate import kept

    out = {}
    for step in steps:
        check()
        out[step["id"]] = journal.unit(
            f"judged-{name}-{step['id']}",
            lambda step=step: kept(kept_as, f"answer{ANSWER_TOKENS}", [step["id"], revision_id(step)],
                                   lambda: write(backend, step)),
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


def outcome(backend, step, before, after):
    """win / loss / tie for the candidate at this moment, and why. Runs on
    the original base weights (no self-preference)."""
    if not differs(before, after):
        return "tie", "same"
    # Asked in both orders: a win or loss counts only when the judge picks
    # the same answer both times (live: a 3B judge called near-identical
    # pairs wins; a position preference flips with the order).
    request, guide = request_of(step) or "(not shown)", guide_of(step) or "(none)"
    picks = []
    for swap in (False, True):
        a, b = (after, before) if swap else (before, after)
        prompt = JUDGE.format(request=request, guide=guide, a=shown(a), b=shown(b))
        with backend.base():
            text = ask(backend, prompt)
        pick = verdict_of(text, swap)
        if pick is None:
            # Live: about 4 of 14 replies could not be read; what they say.
            logger.info("judge_unreadable %s: %r", step["id"], text[:200])
        picks.append(pick)
    verdict = picks[0] if picks[0] == picks[1] else ("both" if None not in picks else None)
    if verdict == "after":
        return "win", "judge"
    if verdict == "before":
        return "loss", "judge"
    # Both, neither, two different picks or no readable answer: a tie. The
    # teacher's preference is what training fits, so it is not asked here.
    return "tie", verdict or "unclear"


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
