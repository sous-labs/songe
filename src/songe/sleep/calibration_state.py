"""Durable calibration coverage, independent of model or accelerator imports."""

import json
import math

from ..steps import digest, revision_id
from .sizing import dimensions


def dreams(store, config=None):
    """Whether training reads kept practice scenes (learning = "dreams").

    The store's own mark is written by the first training slice. The first
    measurement runs before it, so the configuration is asked first."""
    return (getattr(config, "learning", None) or store.get("learning_mode")) == "dreams"


def sample_pool(store, config=None):
    # Include scored experience too: an interrupted selection may have consumed
    # its score revision without ever training it. Never lose that sizing need.
    steps = {s["id"]: s for s in store.active_steps() if s.get("hindsight")}
    for step in store.pool():
        if revision_id(step) >= revision_id(steps.get(step["id"], {})):
            steps[step["id"]] = step
    if not dreams(store, config):
        return list(steps.values())
    # Under dreams the examples trained on are practice scenes, and they can
    # be longer than every real conversation: a scene adds turns, and its
    # teacher reads the lesson's note too. Measuring only the conversations
    # left every scene above the measured size, deferred in every slice.
    from ..practice.pool import practice_steps

    return (
        list(steps.values())
        + practice_steps(store)
        + practice_steps(store, split="held_out")
    )


def policy_key(config):
    return digest(
        {
            k: getattr(config, k)
            for k in (
                "model",
                "max_sequence",
                "response_tokens",
                "token_chunk",
                "hindsight_tokens",
                "rank",
                "backend",
                "torch_options",
                "artifact_kind",
                "memory_fraction",
                "teacher_chunk",
            )
        }
        # Bump when measurement itself changes (cache release, footprint
        # telemetry, teachers on their own ladder): older rows must be redone.
        | {"context_policy": 1, "measurement": 3}
    )


def pool_key(steps):
    return revisions_key({s["id"]: revision_id(s) for s in steps})


def revisions_key(revisions):
    return digest(sorted(revisions.items()))


def sample_revisions(store, config=None):
    """sample_pool's {id: revision} without decoding steps, where the store
    can count them in SQL (an offline/legacy store decodes)."""
    if not hasattr(store, "sample_revisions"):
        return {s["id"]: revision_id(s) for s in sample_pool(store, config)}
    found = store.sample_revisions()
    if dreams(store, config):
        from ..practice.pool import practice_revisions

        found |= practice_revisions(store)
    return found


def pending_keys(store, config=None):
    if dreams(store, config):
        # Under dreams the training pool is kept practice; the step tables
        # and Store.pool belong to hindsight and would say nothing is pending,
        # letting a cycle's watermark skip practice it never trained.
        from ..practice.pool import untrained

        return {(s["id"], revision_id(s)) for s in untrained(store)}
    watermark = store.get("experience_watermark", 0)
    practiced = store.get("practice_revisions", {})
    pending = store.unscored(
        [s for s in store.active_steps("train") if s.get("hindsight")]
    )
    pending += [
        s
        for s in store.pool()
        if revision_id(s) > max(watermark, practiced.get(s["id"], 0))
    ]
    return {(s["id"], revision_id(s)) for s in pending}


def target_length(config, samples, pending):
    lengths = sorted(
        s["required_sequence"]
        for s in samples
        if not pending or (s["id"], s["revision"]) in pending
    )
    return (
        min(config.max_sequence, lengths[math.ceil(len(lengths) * 0.9) - 1])
        if lengths
        else None
    )


def close_enough(low, boundary):
    """Stop bisecting near the memory edge: within a tenth of the length the
    difference is noise from other apps, the gradient window trains longer
    examples anyway, and each probe costs the user minutes on first run."""
    return boundary - low <= max(64, boundary // 10)


def coverage(config, rows, target):
    """Coverage needs one jointly measured geometry, not independent maxima."""
    return (
        max(
            (
                r["sequence"]
                for r in rows
                if dimensions(r)[1] >= config.response_tokens
                and dimensions(r)[2] >= min(config.token_chunk, config.response_tokens)
            ),
            default=0,
        )
        >= target
    )


_GROWTH = {}


def needs_growth(config, store):
    """Cached until the store is written or the policy changes; dashboards
    and the scheduler ask on each refresh. While measuring, the store is
    written every few seconds, so the answer is mostly recomputed: the pool
    is read in SQL (Store.sample_revisions), never decoded."""
    from ..log import change_mark

    path = getattr(store, "path", None)
    mark = change_mark(path) if path is not None else None
    if mark is None:
        return _needs_growth(config, store)
    key = (str(path), getattr(store, "model_id", ""), mark, policy_key(config))
    if key not in _GROWTH:
        if len(_GROWTH) > 64:  # one entry per model and store version
            _GROWTH.clear()
        _GROWTH[key] = _needs_growth(config, store)
    return _GROWTH[key] or memory_freed(store)


def held_key(probe):
    return json.dumps(
        {k: v for k, v in probe.items() if k != "needed_bytes"}, sort_keys=True
    )


def memory_freed(store):
    """A size held for lack of free memory is retried once that much is free."""
    saved = store.get("calibration") or {}
    if (saved.get("progress") or {}).get("completion_reason") != "held":
        return False
    tries = saved.get("held_retries", {})
    # A held size records where its probe was stopped, on the way up: the
    # real need is higher. Retry each at most twice, and only with room to
    # spare; otherwise measuring would loop and training would never start.
    needed = min(
        (
            int(h["needed_bytes"] * 1.25)
            for h in saved.get("held", [])
            if tries.get(held_key(h), 0) < 3
        ),
        default=None,
    )
    if needed is None:
        return False
    try:
        from ..platform import current
        from .sizing import PressureHistory

        sample = current().memory()
        history = PressureHistory(store.get("memory_pressure_history", {}))
        if (history.data.get("critical_boundary") or {}).get("warning_available"):
            # Re-check a learned boundary while waiting on it: nothing else
            # samples memory then, so one PSI spike at 139 of 188 GB free
            # held a 6 GB size for 1 h 42 min. It expires once normal
            # samples outnumber the critical ones it came from.
            history.observe(sample)
            store.set("memory_pressure_history", history.data)
        return needed + history.margin_for(sample) <= sample.available
    except Exception:
        return False


def _needs_growth(config, store):
    saved = store.get("calibration") or {}
    rows = saved.get("rows", [])
    if not rows:
        return False  # Initial/legacy admission is handled by the scheduler.
    progress = saved.get("progress") or {}
    if progress:
        if progress.get("policy") != policy_key(config):
            return True
        if progress.get("pool") != revisions_key(sample_revisions(store, config)):
            return True
        pending = pending_keys(store, config)
        # Scoring is not evidence that an intact sample fitted a measured
        # slice. Legacy stores can have no pending revisions and an empty
        # pool after scoring everything under an interrupted first probe.
        target = target_length(config, saved.get("samples", []), pending)
        if target and (
            target > progress.get("target_sequence", 0)
            or not coverage(config, rows, target)
            and progress.get("completion_reason") not in {"memory", "held"}
        ):
            return True
        return not progress.get("complete", False)
    # Old installations recorded rows but no completion evidence. A small
    # measurement is usable evidence, never proof that calibration is finished.
    return bool(sample_revisions(store, config)) or not coverage(
        config, rows, config.max_sequence
    )


def measure_pool(config, store, tokenizer):
    from .prompts import UnusableStep, paired_prompts

    steps = sample_pool(store, config)
    samples = []
    for step in steps:
        try:
            student, teacher = paired_prompts(
                tokenizer, step, config, check_length=False
            )
        except UnusableStep:
            continue
        samples.append(
            {
                "id": step["id"],
                "revision": revision_id(step),
                "required_sequence": len(student) + config.response_tokens,
                "teacher_sequence": len(teacher) + config.response_tokens,
            }
        )
    target = target_length(config, samples, pending_keys(store, config))
    return samples, target, pool_key(steps)


def teacher_measured(saved):
    return max(
        (
            r.get("teacher_sequence", 0)
            for r in saved.get("rows", []) + saved.get("teacher_rows", [])
        ),
        default=0,
    )


TEACHER_TRIES = 2


def pending_teachers(config, saved, store=None):
    """The teacher length each pending example needs."""
    pending = pending_keys(store, config) if store else None
    return [
        s.get("teacher_sequence") or s["required_sequence"]
        for s in saved.get("samples", [])
        if pending is None or (s["id"], s["revision"]) in pending
    ]


def teacher_to_ask(config, saved, store, sizing):
    """The calibration with the teacher length to measure for examples that
    today's slice defers (sizing admits shorter teachers only), or None when
    measuring would not help: then they wait, and the reason says so.

    It asks for the longest deferred length below the shortest measured one
    that does not fit, so one measurement admits all of them (#140). If an
    earlier ask was measured and still did not fit, it asks for the middle
    of what is left, so the lengths are bisected, not tried one by one."""
    admitted = sizing.get("max_teacher_sequence", float("inf"))
    deferred = sorted(t for t in pending_teachers(config, saved, store) if t > admitted)
    if not deferred:
        return None
    tries = saved.get("teacher_tries", {})
    measured = [
        r.get("teacher_sequence", 0)
        for r in saved.get("teacher_rows", [])
        + saved.get("teacher_limits", [])
        + [h for h in saved.get("held", []) if h.get("kind") == "teacher"]
    ] + [
        # Asked and failed every try: it does not fit either; look below it.
        w
        for w in saved.get("teacher_wanted", [])
        if tries.get(str(w), 0) >= TEACHER_TRIES
    ]
    ceiling = min((t for t in measured if t >= deferred[0]), default=None)
    # Within a tenth of a length that does not fit, it would not fit either.
    gap = [
        t
        for t in deferred
        if ceiling is None or (t < ceiling and not close_enough(t, ceiling))
    ]
    if not gap:
        return None
    asked_before = ceiling is not None and ceiling in saved.get("teacher_wanted", [])
    length = gap[(len(gap) - 1) // 2] if asked_before else gap[-1]
    return ask_for_teacher(saved, length)


def teacher_wanted(saved):
    """Teacher lengths deferred examples need that no measurement covers yet.

    Measuring doubles the teacher length toward the longest example, so it
    can jump over the length most examples need. When the measured length
    above it does not fit beside the user's server, every example is
    deferred and nothing measures again (#140); the scheduler then asks for
    that exact length here."""
    done = [
        r.get("teacher_sequence", 0)
        for r in saved.get("teacher_rows", [])
        + saved.get("teacher_limits", [])
        + [h for h in saved.get("held", []) if h.get("kind") == "teacher"]
    ]
    limits = [r.get("teacher_sequence", 0) for r in saved.get("teacher_limits", [])]
    tries = saved.get("teacher_tries", {})
    return sorted(
        w
        for w in set(saved.get("teacher_wanted", []))
        if not any(w <= t and close_enough(w, t) for t in done)
        # Beyond the whole machine: never measured again.
        and not any(t <= w for t in limits)
        # A measurement that failed twice (the card was full, say) is not
        # asked again; otherwise the model would measure instead of train.
        and tries.get(str(w), 0) < TEACHER_TRIES
    )


def ask_for_teacher(saved, length):
    """The calibration with a teacher of `length` tokens asked for (measuring
    is then incomplete), or None when a measurement already covers it."""
    asked = saved | {
        "teacher_wanted": sorted(set(saved.get("teacher_wanted", [])) | {length})
    }
    if not teacher_wanted(asked):
        return None
    if asked.get("progress"):
        asked["progress"] = asked["progress"] | {"complete": False}
    return asked


def bracket(config, rows, limits):
    """The smallest-chunk failure and the largest success below it."""
    boundary = min(
        (
            r["sequence"]
            for r in limits
            if dimensions(r)[1] <= config.response_tokens and dimensions(r)[2] == 1
        ),
        default=None,
    )
    maximum = max(
        (
            r["sequence"]
            for r in rows
            if dimensions(r)[1] >= config.response_tokens
            and dimensions(r)[2] == 1
            and (boundary is None or r["sequence"] < boundary)
        ),
        default=0,
    )
    return boundary, maximum


def chunk_limited(config, limits, target, maximum):
    return (
        any(
            r["sequence"] <= target
            and dimensions(r)[1] <= config.response_tokens
            and 1 < dimensions(r)[2] <= min(config.token_chunk, config.response_tokens)
            for r in limits
        )
        and maximum >= target
    )


def progress(config, saved, target, pool):
    rows = saved.get("rows", [])
    failures = saved.get("capacity_limits", [])
    held = [h for h in saved.get("held", []) if h.get("kind") == "sequence"]
    # A failing probe brackets capacity. Only an adjacent successful probe at
    # the smallest chunk proves that *no* larger sequence fits.
    boundary, maximum = bracket(config, rows, failures)
    memory_bounded = boundary is not None and close_enough(maximum, boundary)
    chunk_bounded = chunk_limited(config, failures, target, maximum)
    # A size held for lack of free memory brackets only today's capacity:
    # measurement pauses there, but no sample is judged by it.
    held_boundary, held_maximum = bracket(config, rows, failures + held)
    sequence_held = (
        held_boundary is not None and close_enough(held_maximum, held_boundary)
    ) or chunk_limited(config, held, target, held_maximum)
    teacher_target = saved.get("teacher_target_sequence", 0)
    measured_teacher = teacher_measured(saved)
    teacher_bounds = [
        r["teacher_sequence"]
        for r in saved.get("teacher_limits", [])
        + [h for h in saved.get("held", []) if h.get("kind") == "teacher"]
    ]
    teacher_stopped = bool(teacher_bounds) and close_enough(
        measured_teacher, min(teacher_bounds)
    )
    teacher_done = (
        measured_teacher >= teacher_target or teacher_stopped
    ) and not teacher_wanted(saved)
    sequence_done = coverage(config, rows, target) or sequence_held
    # A teacher length asked for (teacher_wanted) is measured even when the
    # window reached its memory limit; otherwise it would be asked forever.
    complete = (sequence_done and teacher_done) or (
        (memory_bounded or chunk_bounded) and not teacher_wanted(saved)
    )
    if memory_bounded or chunk_bounded:
        reason = "memory"
    elif complete and saved.get("held"):
        reason = "held"
    elif complete:
        reason = "coverage"
    else:
        reason = "partial"
    return {
        "phase": "calibrate",
        "policy": policy_key(config),
        "pool": pool,
        "target_sequence": target,
        "measured_sequence": max((r["sequence"] for r in rows), default=0),
        "teacher_target_sequence": teacher_target,
        "teacher_measured_sequence": measured_teacher,
        "complete": complete,
        "completion_reason": reason,
        "memory_ceiling": maximum if memory_bounded else None,
    }


def smallest_pending(config, saved, store=None):
    """The shortest full context a pending example needs its teacher to read.

    The student backpropagates only its gradient window, so whether any
    example can train now depends on the teacher envelope, not the window."""
    pending = pending_keys(store, config) if store else None
    return min(
        (
            s.get("teacher_sequence") or s["required_sequence"]
            for s in saved.get("samples", [])
            if pending is None or (s["id"], s["revision"]) in pending
        ),
        default=None,
    )


def measuring_tokens(progress):
    """(done, target, longer): the dimension still being measured.

    Once the gradient window covers its target, only the teacher envelope can
    still be growing; showing the window would read 8192 of 6833."""
    measured = progress.get("measured_sequence") or 0
    target = progress.get("target_sequence") or 0
    teacher = progress.get("teacher_measured_sequence") or 0
    teacher_target = progress.get("teacher_target_sequence") or 0
    if target and measured >= target and teacher_target > teacher:
        return teacher, teacher_target, True
    return (min(measured, target) if target else measured), target, False


def measurement_text(progress):
    done, target, longer = measuring_tokens(progress)
    return f"measuring training size · {done} of ~{target} tokens" + (
        " · longer contexts" if longer else ""
    )


def training_window(config, saved):
    """What the dashboard says about a model's measured training window.

    Songe measures only as far as the recorded conversations need (target
    sequence), unless memory runs out first. So the measured window is either
    the memory limit or "enough for the conversations so far"; which one is
    decided by the same bracket the measurement itself uses. None before the
    first measurement.
    """
    saved = saved or {}
    rows = saved.get("rows", [])
    if not config.calibration_available or not rows:
        return None
    failures = saved.get("capacity_limits", []) + [
        h for h in saved.get("held", []) if h.get("kind") == "sequence"
    ]
    boundary, maximum = bracket(config, rows, failures)
    limited = boundary is not None and close_enough(maximum, boundary)
    step = config.calibrated_step_seconds or 0
    return {
        "window": int(config.max_sequence),
        "limit": "memory" if limited else "conversations",
        "served": int(config.context_window) if config.context_window else None,
        "example_seconds": round(step / max(1, config.accumulation), 1) if step else None,
    }
