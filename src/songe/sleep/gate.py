"""Conservative, paired conversation bootstrap. No statistical bypass flag."""

import math
import random
from dataclasses import asdict, dataclass
from statistics import mean


@dataclass
class Decision:
    accepted: bool
    reason: str
    heldout_change: float | None = None
    noise_margin: float | None = None
    general_drift: float | None = None
    kl_change: float | None = None  # judged gates: the old KL measure, diagnostic only

    def dict(self):
        return asdict(self)


LOOPING = 0.5        # an answer with half its 4-word phrases repeated is looping
LOOPING_SHARE = 0.1  # more looping answers than this share (and than 1) fails


def decide(
    before, after, *, min_heldout, drift_limit, bootstraps=2000, confidence=0.95
):
    if len(before) != len(after) or len(after) < min_heldout:
        return Decision(False, "Waiting for held-out experience")
    if [(r["id"], r["conversation"]) for r in before] != [
        (r["id"], r["conversation"]) for r in after
    ]:
        return Decision(False, "Evaluation samples do not match")
    metrics = ("kl", "drift", "empty", "repetition", "tool_valid")
    if any(
        r.get("tool_step") and not r.get("tool_supported", True)
        for rows in (before, after)
        for r in rows
    ):
        return Decision(
            False, "Tool-call evaluation is unsupported for this output format"
        )
    if any(
        not math.isfinite(float(r[k]))
        for rows in (before, after)
        for r in rows
        for k in metrics
    ):
        return Decision(False, "Non-finite evaluation")
    judged = [a for a in after if a.get("outcome")]
    clusters = {}
    if judged:
        # What each version wrote, judged (sleep/judged.py). Each moment is
        # one unit, win +1 / loss -1 / tie 0 (user decision: per conversation
        # gave 5 units, whose bootstrap lower bound sat at 0, so a lesson that
        # lives in one or two conversations could never pass). Moments of one
        # conversation are related, so this adopts more easily; drift, health
        # and looping still stand. The KL is kept as a diagnostic only.
        # Any judged moment is enough to decide on (user decision); the
        # per-moment resampling below says how sure that is.
        score = {"win": 1, "loss": -1, "tie": 0}
        for a in judged:
            clusters[a["id"]] = [score[a["outcome"]]]
    else:
        for b, a in zip(before, after, strict=False):
            clusters.setdefault(b["conversation"], []).append(b["kl"] - a["kl"])
        if len(clusters) < min_heldout:
            return Decision(False, "Waiting for independent held-out conversations")
    gains = [mean(v) for v in clusters.values()]
    change = mean(gains)
    rng = random.Random(7613)
    samples = sorted(mean(rng.choices(gains, k=len(gains))) for _ in range(bootstraps))
    # Lower endpoint of the requested two-sided confidence interval.
    lower = samples[int((1 - confidence) / 2 * bootstraps)]
    margin = max(math.ulp(change), change - lower)
    drift = mean(r["drift"] for r in after)
    result = dict(heldout_change=change, noise_margin=margin, general_drift=drift)
    if judged:
        result["kl_change"] = mean(b["kl"] - a["kl"] for b, a in zip(before, after, strict=False))
        reason = judged_reason(judged, change, margin)
        if reason:
            return Decision(False, reason, **result)
    elif change <= margin:
        return Decision(
            False, "Held-out improvement does not exceed bootstrap noise", **result
        )
    if drift > drift_limit:
        return Decision(False, "General behaviour drift exceeds the limit", **result)

    # Health checks use one sampled output per row, so a single unlucky sample
    # must not veto a real improvement: reject only a paired-bootstrap
    # significant worsening, clustered by conversation like the KL check.
    def worse(pairs, key, higher_is_worse):
        grouped = {}
        for b, a in pairs:
            change = a[key] - b[key] if higher_is_worse else b[key] - a[key]
            grouped.setdefault(b["conversation"], []).append(change)
        changes = [mean(v) for v in grouped.values()]
        if not changes or mean(changes) <= 0:
            return False
        draws = sorted(
            mean(rng.choices(changes, k=len(changes))) for _ in range(bootstraps)
        )
        return draws[int((1 - confidence) / 2 * bootstraps)] > 0

    pairs = list(zip(before, after, strict=False))
    tool_rows = [(b, a) for b, a in pairs if b.get("tool_step")]
    if tool_rows and worse(tool_rows, "tool_valid", higher_is_worse=False):
        return Decision(False, "Tool-call validity dropped", **result)
    if worse(pairs, "empty", higher_is_worse=True):
        return Decision(False, "Empty rate increased", **result)
    # Repetition: only answers stuck in a loop count. Repeated phrases are
    # normal in code (live: a coding adapter vetoed for "repetition" on
    # ordinary 2048 code), so a rise in the rate is not a failure.
    looping = [sum(r["repetition"] >= LOOPING for r in rows) for rows in (before, after)]
    if looping[1] - looping[0] > max(1, LOOPING_SHARE * len(after)):
        return Decision(False, "Looping answers increased", **result)
    return Decision(True, "Held-out improvement passed all checks", **result)


WORSE_Z = 1.64


def judged_reason(judged, change, margin):
    """Why the judged moments reject the candidate, or None.

    Rows marked "served" compared what the user would be served (the
    candidate with the notes that stay in the prompt) with what they are
    served now (the version in use with all its notes); their "alone"
    outcome compared the two versions with no notes at all. The candidate is
    adopted when it learned something (alone, it is better beyond the
    judge's noise) and is not worse for the user (served, ties pass; it is
    refused only when it lost more moments than it won beyond that noise).
    When the notes already serve a habit, both sides of the served
    comparison follow it and tie: a win there was never the question (live:
    Qwen3.8-27B whose adapter alone went from 27 to 87 of 128 habit
    requests was refused on 22 ties, a win and a loss). Other rows compared
    the two versions with no notes: the candidate must be clearly better."""
    wins = sum(a["outcome"] == "win" for a in judged)
    losses = sum(a["outcome"] == "loss" for a in judged)
    if any(a.get("served") for a in judged):
        # Worse only beyond what a judge that is right four times in five
        # gives by chance: an equally good version got 3 against 8 before
        # (WORSE_Z, a one-sided 5% on the moments that got a verdict).
        if beyond_noise(losses, wins):
            return f"Worse than what you have now: better on {wins}, worse on {losses}"
        alone = [a.get("alone") for a in judged]
        better = alone.count("win")
        worse = alone.count("loss")
        if not beyond_noise(better, worse):
            return f"Nothing learned: without notes better on {better}, worse on {worse}"
        return None
    if wins <= losses or change <= margin:
        return f"Not clearly better: better on {wins}, worse on {losses}"
    return None


def beyond_noise(more, fewer):
    """more - fewer > WORSE_Z * sqrt(more + fewer): more than a judge that is
    right four times in five gives by chance (3 of 3 is, 2 of 2 is not)."""
    return more - fewer > WORSE_Z * math.sqrt(more + fewer)


def judged_check(judged, *, bootstraps=2000, confidence=0.95):
    """decide()'s judged test alone, for rejecting a candidate before its
    held-out rows are measured: the failing Decision, or None when the
    judged moments still allow it to pass. judged: the moments' outcomes in
    the gate's row order, as decide() sees them (the same resampling)."""
    score = {"win": 1, "loss": -1, "tie": 0}
    gains = [score[a["outcome"]] for a in judged]
    if not gains:
        return None
    change = mean(gains)
    rng = random.Random(7613)
    samples = sorted(mean(rng.choices(gains, k=len(gains))) for _ in range(bootstraps))
    lower = samples[int((1 - confidence) / 2 * bootstraps)]
    margin = max(math.ulp(change), change - lower)
    reason = judged_reason(judged, change, margin)
    if reason:
        return Decision(False, reason, heldout_change=change, noise_margin=margin)
    return None


def choose_scale(
    evaluate_scale,
    before,
    scales,
    *,
    min_heldout,
    drift_limit,
    confidence=0.95,
    check=lambda: None,
    discard=lambda scale: None,
):
    """Try the configured strengths largest first; the first that passes the
    real gate is used. A failed strength is discarded before the next one is
    built, so at most one candidate exists. Scale zero keeps the incumbent and
    is never a candidate.

    before: the incumbent's rows, or a function returning them (measured
    only when a candidate needs them). evaluate_scale returns the
    candidate's rows, or {"early": Decision dict, "judged": [...]} when it
    was rejected before its rows were measured.
    """
    evaluations = {}
    for scale in sorted(set(scales), reverse=True):
        check()
        after = evaluate_scale(scale)
        if isinstance(after, dict) and "early" in after:
            verdict = Decision(**after["early"])
            evaluations[str(scale)] = verdict.dict() | {"judged": after.get("judged") or []}
            discard(scale)
            continue
        verdict = decide(
            before() if callable(before) else before,
            after,
            min_heldout=min_heldout,
            drift_limit=drift_limit,
            confidence=confidence,
        )
        evaluations[str(scale)] = verdict.dict() | {
            # The judged moments (sleep/judged.py), for the report.
            "judged": [{k: r.get(k) for k in ("id", "conversation", "outcome", "by")}
                       | {k: r[k] for k in ("alone", "alone_by") if k in r}
                       for r in after if r.get("outcome")],
        }
        if verdict.accepted:
            return scale, verdict, evaluations
        discard(scale)
    return None, None, evaluations
