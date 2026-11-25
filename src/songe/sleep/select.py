import math
import random

from ..steps import revision_id


def select(steps):
    """Every cleaned train step with hindsight is trained once (DESIGN 3.4).

    Returns (selected, skipped); nothing is skipped."""
    from .clean import clean

    candidates = [
        s
        for s in steps
        if s["split"] == "train" and s["hindsight"] and not s.get("retired_at")
    ]
    return clean(candidates), 0


def replay_pool(pool, watermark, fraction, practiced=None):
    """Interleave new evidence with deterministic replay of the older pool."""
    practiced = practiced or {}
    new = [
        s for s in pool if revision_id(s) > max(watermark, practiced.get(s["id"], 0))
    ]
    old = [
        s for s in pool if revision_id(s) <= max(watermark, practiced.get(s["id"], 0))
    ]
    if not new or fraction == 0 or not old:
        return new
    count = (
        min(len(old), max(1, math.ceil(len(new) * fraction / (1 - fraction))))
        if fraction < 1
        else len(old)
    )
    replay = random.Random(7613).sample(old, count)
    result = new + replay
    random.Random(7613).shuffle(result)
    return result


def bounded_batch(pool, config, seconds, step_seconds, *, pending=None, whole=False):
    """Cap real practice by measured optimizer cost and the available idle gap.

    whole (learning = "dreams"): plan the Dream over all of it, up to
    slice_limit. Training resumes from checkpoints in later idle gaps, so the
    first gap's length must not decide how much a Dream learns."""
    limit = config.slice_limit
    if whole:
        pass
    elif step_seconds > 0 and seconds is not None:
        # Reserve the configured rehearsal share. Interrupted optimizer units
        # remain resumable; always permit one experience to make progress.
        share = 1 - config.anchor_fraction
        limit = min(
            limit,
            max(1, int(max(0, seconds) / step_seconds * config.accumulation * share)),
        )
    else:
        limit = min(limit, config.accumulation)
    ordered = sorted(
        pool,
        key=lambda s: (
            pending is not None and (s["id"], revision_id(s)) not in pending,
            bool(s.get("retry")),  # scenes of a rejected Dream after untried ones
            not s.get("stated"),  # the user's own rules make the cut first
            s["id"],
        ),
    )
    return ordered[:limit], ordered[limit:]
