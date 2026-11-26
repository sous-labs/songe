"""The Learning cycle record every surface shows (docs/CONTRACTS.md, Learning
cycle), assembled from what the store recorded: reading and practice
counts, the night record (training, gate values, examples) and versions.

Codes and numbers only; surfaces word them. Unknown values are null, never 0.
A cycle is one night record; before a night exists, reading and practice in
progress show as the upcoming cycle.
"""

import json
import re

SIGNAL_TYPES = ("correction", "retry", "approval", "tool_error", "failed_result", "outcome")
STAGES = ("read", "practice", "train", "check", "result")
# The checks that decide, in order: better on held-out moments; not broken
# where it already did right (the drift set once it can decide, else the
# general questions; only the one that gated is shown); answers healthy.
CHECK_UNITS = (
    ("held_out", "gap"),
    ("general", "kl"),
    ("drift", "kl"),
    ("health", "health"),
)
# Reasons a strength failed on its answers' health (gate.decide).
HEALTH_FAILS = ("Looping answers increased", "Repetition rate increased", "Empty rate increased",
                "Tool-call validity dropped")
# Night phases (worker) by cycle stage.
PHASE_STAGE = {
    "select": "train",
    "rewind": "train",
    "distill": "train",
    "train": "train",
    "check": "check",
    "gate": "check",
    "fuse": "check",
    "swap": "result",
}
FINISHED = {"accepted", "ready", "rejected", "failed"}


def cycle_id(night_id):
    return f"c-{night_id:04d}"


def night_of(cycle):
    try:
        return int(str(cycle).removeprefix("c-"))
    except ValueError:
        return None


def clip(text, n=300):
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


# ---- what happened before training ---------------------------------------


def read_summary(store):
    from .reading import store as reading

    counts = reading.counts(store)
    found = reading.signals(store)
    by_type = dict.fromkeys(SIGNAL_TYPES, 0)
    for s in found:
        if s["type"] in by_type:
            by_type[s["type"]] += 1
    successes = [s for s in found if s["type"] == "success"]
    return {
        "conversations": {"done": counts["read"], "total": counts["read"] + counts["unread"]},
        "signals": {"total": sum(by_type.values()), "by_type": by_type},
        "lessons": sum(1 for s in found if s.get("lesson")),
        "successes": {
            "confirmed": sum(s.get("confirmed") is True for s in successes),
            "contradicted": sum(s.get("confirmed") is False for s in successes),
        },
    }


def practice_line(row, created):
    scene = row.get("scene") or {}
    return {
        "id": row["id"],
        "signal_id": row["signal_id"],
        "request": clip(scene.get("request"), 200) or None,
        "kept": row.get("drop") is None,
        "drop": row.get("drop"),
        "at": created,
    }


def practice_summary(store, every=False):
    from .practice import store as practice

    counts = practice.counts(store)
    with store.connect() as db:
        rows = db.execute(
            "SELECT payload, created FROM practice WHERE model_id=? ORDER BY created DESC, rowid DESC"
            + ("" if every else " LIMIT 5"),
            (store.model_id,),
        ).fetchall()
    lines = [practice_line(json.loads(p), c) for p, c in rows]
    return counts | {("lines" if every else "recent"): lines}


# ---- what a night recorded -------------------------------------------------


def train_summary(record):
    progress = record.get("data_progress") or {}
    training = progress.get("phase") == "train"
    return {
        "step": progress.get("step") if training else record.get("iterations"),
        "total": progress.get("total_steps") if training else record.get("iterations"),
        "samples": record.get("samples") or {"practice": None, "rehearsal": None},
        "losses": record.get("losses") or [],
        "eta_seconds": progress.get("eta_seconds") if training else None,
        "peak_memory_gb": record.get("peak_memory_gb"),
    }


def gate_entries(record):
    """(before, after) measured values: the incumbent, and the strength that
    was adopted (else the last one tried), preferring its exported form."""
    values = record.get("gate_values") or {}
    before = next((v for k, v in values.items() if k.startswith("before-")), None)
    scale = record.get("wake_scale")
    evaluations = record.get("scale_evaluations") or {}
    # The strength adopted; else the one that came closest (as the report
    # shows), then any other tried that has values (live: a night resumed
    # after a cancel kept values for one strength only).
    order = [str(scale)] if scale is not None else sorted(
        evaluations, key=lambda k: -(evaluations[k].get("heldout_change") or float("-inf")))
    order += [k for k in evaluations if k not in order]
    for chosen in order:
        for prefix in (f"fused-evaluation-{chosen}-", f"scale-{chosen}-"):
            after = next((v for k, v in values.items() if k.startswith(prefix)), None)
            if after:
                return before, after
    return before, None


def checks(record, config, finished):
    before, after = gate_entries(record)
    # The "not broken" measure the candidate was judged on: its own check
    # (live: the incumbent's said drift set, the candidate's general, and the
    # row showed "0.0000 → –"); else whichever has a candidate value.
    checks_by = record.get("drift_checks") or {}
    candidate = next((v for k, v in checks_by.items() if not k.startswith("before-") and v), None)
    drift_check = candidate or next(iter(checks_by.values()), None) or {}
    not_broken = drift_check.get("source", "general")
    if after and after.get(not_broken) is None:
        not_broken = next((k for k in ("drift", "general") if after.get(k) is not None), not_broken)
    out = []
    for key, unit in CHECK_UNITS:
        row = {
            "key": key,
            "state": "done" if finished else "ahead",
            "unit": unit,
            "before": None,
            "after": None,
            "limit": None,
            "gates": False,
            "passed": None,
            "count": None,
        }
        if key == "held_out":
            row.update(gates=True, state="done" if after or finished else "ahead")
            if before:
                row["before"] = before["held_out"]
                row["count"] = {"done": before["count"], "total": before["count"]}
            if after:
                row["after"] = after["held_out"]
            if record.get("heldout_change") is not None:
                row["passed"] = record["heldout_change"] > (record.get("noise_margin") or 0)
            if record.get("judged"):
                # Judged gate: what decided is better / worse / same on what
                # each version wrote; the KL above is a diagnostic.
                improved = record.get("improved") or {}
                row.update(unit="judged", judged=improved,
                           passed=improved.get("count", 0) > improved.get("worse", 0) and bool(row.get("passed")))
        elif key in ("general", "drift"):
            if key != not_broken:
                continue  # only the "not broken" measure that decided is shown
            row.update(limit=config.general_drift_limit, gates=True)
            if before:
                row["before"] = before[key]
            if after:
                row["after"] = after[key]
                if after[key] is not None:
                    row["passed"] = after[key] <= config.general_drift_limit
            n = (after or before or {}).get("count" if key == "general" else "drift_count")
            if key == "drift" and n is not None:
                row["count"] = {"done": n, "total": n}
        elif key == "health":
            row.update(gates=True)
            if record.get("scale_evaluations"):
                failed = [e.get("reason") for e in record["scale_evaluations"].values()
                          if any(f in (e.get("reason") or "") for f in HEALTH_FAILS)]
                # gate.decide checks health only once a strength is better:
                # adopted = passed; failed on it = failed; else not reached.
                row.update(state="done", passed=False if failed else True if record.get("wake_scale") is not None else None,
                           text=failed[0] if failed else None if record.get("wake_scale") is not None else "not_reached")
        out.append(row)
    return out


REASONS = (
    ("Held-out improvement passed", "passed"),
    ("does not exceed bootstrap noise", "no_gain"),
    ("Not clearly better", "no_gain"),
    ("Worse than what you have now", "no_gain"),
    ("Not better than what you have now", "no_gain"),
    ("drift exceeds the limit", "drift_limit"),
    ("validity dropped", "capability_drop"),
    ("Repetition rate increased", "repetition_rate"),  # the older, stricter rule
    ("Looping answers increased", "repetition"),
    ("Empty rate increased", "empty_answers"),
    ("Waiting for", "too_few_examples"),
)


def result(night, finished):
    if not finished:
        return None
    record, status = night["record"], night["status"]
    if status == "failed":
        outcome, reason = "stopped", "error"
    elif status in ("accepted", "ready"):
        outcome, reason = "adopted", "passed"
    else:
        outcome = "kept"
        text = record.get("reason") or ""
        if text.startswith("No adapter scale passed"):
            # Every strength failed: why the strongest one did (the others
            # fail the same way or for less gain).
            scales = record.get("scale_evaluations") or {}
            strongest = max(scales, key=float, default=None)
            text = (scales.get(strongest) or {}).get("reason") or text
        reason = next((code for phrase, code in REASONS if phrase in text), "other")
    improved = record.get("improved") or {"count": None, "checked": None, "available": None}
    if record.get("examples") and not any("verdict" in e for e in record["examples"]):
        # Counted before answers were judged against their lesson: that
        # count is not trusted (and its pairs are not shown), so none.
        improved = {"count": None, "checked": None, "available": improved.get("available")}
    return {
        "outcome": outcome,
        "reason": reason,
        "version": record.get("version_label") if outcome == "adopted" else None,
        "improved": improved,
        "examples": shown_examples(record.get("examples") or []),
        # What the gate passed or failed on: held-out improvement against its
        # measured noise (gap units; positive = closer to the teacher).
        "held_out": {
            "change": record.get("heldout_change"),
            "margin": record.get("noise_margin"),
        },
    }


def shown_examples(examples):
    """Contract rule 6: improved only, one per signal, originals from user
    signals first, then from tool signals, then practice; within a group by
    teacher preference, highest first."""
    from .quick import differs

    def changed(e):
        # Scored better is not enough to show: what it does must differ
        # (live: two near-identical "let me check the directory" answers).
        # A Deep Dream answer is cut at a few hundred tokens: a reasoning-only
        # pair, or a side whose tool call was cut off mid-way, shows nothing.
        try:
            sides = (e["before"], e["after"])
            if not any(x["tool_calls"] for x in sides):
                return False
            if any(not x["tool_calls"] and re.search(r"<tool_call>|<\|tool_call|<function=", x["text"] or "")
                   for x in sides):
                return False
            return differs(*sides)
        except (KeyError, TypeError):
            return False

    order = {("original", "user"): 0, ("original", "tool"): 1}
    ranked = sorted(
        # Judged to follow its lesson (reports from before the judge show
        # none: "scored better" alone showed a worse answer as better).
        (e for e in examples if e.get("improved") and e.get("verdict") == "after" and changed(e)),
        key=lambda e: (order.get((e["source"], e["signal"]["source"]), 2), -e["teacher_preference"]),
    )
    out, seen = [], set()
    for e in ranked:
        if e["signal"]["id"] in seen:
            continue
        seen.add(e["signal"]["id"])
        out.append(e)
        if len(out) == 3:
            break
    return out


WAITS = (
    ("GB free on", "disk"),
    ("battery", "power"),
    ("memory", "memory"),
    ("thermal", "thermal"),
    ("Hardware policy", "thermal"),
    ("paused", "paused"),
    ("activity", "agent_active"),
    ("experience", "no_data"),
    ("storage", "disk"),
    ("disk", "disk"),
    ("unavailable", "disk"),
)


# A slice that reached its deadline ends normally; the next one carries on.
# A service restart stops a slice too; the Dream resumes after it.
NOT_WAITING = ("slice deadline", "stopped by the service", "proxy restart", "cancelled")


# The numbers a wait's reason carries (sleep.sizing.waiting_reason,
# artifacts.storage), so the page can say "9.6 GB free · needs 13.2 GB".
WAIT_AMOUNTS = (
    (re.compile(r"([\d.]+) GB free, needs ([\d.]+) GB"), ("have", "need")),
    (re.compile(r"Needs ([\d.]+) GB free on .+?; ([\d.]+) GB available"), ("need", "have")),
)


def wait_of(reason):
    if not reason or any(phrase in reason for phrase in NOT_WAITING):
        return None
    code = next((c for phrase, c in WAITS if phrase in reason), "error")
    wait = {"code": code, "have": None, "need": None, "unit": None}
    for pattern, keys in WAIT_AMOUNTS:
        if found := pattern.search(reason):
            wait.update(zip(keys, map(float, found.groups()), strict=True), unit="GB")
            break
    return wait


def stages(current, state, *, read=None, practice=None, train=None, check=None):
    progress = {
        "read": read and {
            "done": read["conversations"]["done"],
            "total": read["conversations"]["total"],
            "unit": "conversations",
        },
        "practice": practice and {
            "done": practice["written"],
            "total": None,
            "unit": "scenes",
        },
        "train": train and train["total"] and {
            "done": train["step"],
            "total": train["total"],
            "unit": "steps",
        },
        "check": check,
    }
    at = STAGES.index(current)
    out = []
    for i, key in enumerate(STAGES):
        if state == "done" or i < at:
            s = "done"
        elif i == at:
            s = "now" if state == "running" else "waiting"
        else:
            s = "ahead"
        out.append(
            {
                "key": key,
                "state": s,
                "progress": progress.get(key) or None,
                "started_at": None,
                "ended_at": None,
                "eta_seconds": train["eta_seconds"] if key == "train" and train and s == "now" else None,
            }
        )
    return out


def night_cycle(config, store, night, model, *, every=False):
    record, status = night["record"], night["status"]
    finished = status in FINISHED
    if finished:
        stage, state = "result", "done"
    else:
        stage = PHASE_STAGE.get(night.get("phase"), "train")
        state = "running" if status == "running" else "waiting"
        if state == "waiting" and wait_of(record.get("reason")) is None:
            state = "running"  # between slices: nothing to wait for
    read = read_summary(store) if record.get("learning", "hindsight") == "dreams" else None
    practice = practice_summary(store, every) if read is not None else None
    train = train_summary(record)
    found = checks(record, config, finished)
    done = sum(c["state"] == "done" for c in found)
    check = {"done": done, "total": len(found), "unit": "checks"}
    gate = record.get("data_progress") or {}
    if not finished and stage == "check" and gate.get("phase") == "check":
        # The checks finish together at the end; meanwhile, conversations
        # measured so far (references, the current version, each strength).
        check = {
            "done": gate["completed"],
            "total": gate["total"],
            # Conversations measured by the gate, then examples compared.
            "unit": gate.get("unit") or "measured",
        }
    out = {
        "id": cycle_id(night["id"]),
        "model": model,
        "trigger": record.get("trigger", "idle"),
        "started_at": record.get("started_at") or night.get("created"),
        "ended_at": record.get("ended_at") if finished else None,
        "stage": stage,
        "state": state,
        "wait": None if state in ("running", "done") else wait_of(record.get("reason")),
        "stages": stages(
            stage, state, read=read, practice=practice, train=train,
            check=check,
        ),
        "read": read,
        "practice": practice,
        "train": train,
        "checks": found,
        "result": result(night, finished),
    }
    if every:
        out["signals"] = signal_lines(store) if read is not None else []
        # What this Dream sent to a cloud model, if cloud learning was on.
        from .cloud.ledger import calls

        out["cloud_calls"] = calls(
            store, dream=out["id"], since=out["started_at"] or 0, until=out["ended_at"]
        )
        if out["result"] is not None:
            out["result"]["all_examples"] = record.get("examples") or []
    return out


def upcoming(config, store, model, activity, *, every=False):
    """Reading or practice in progress before a night exists."""
    read = read_summary(store)
    practice = practice_summary(store, every)
    stage = activity.get("stage", "read")
    with store.connect() as db:
        last = db.execute("SELECT MAX(id) FROM nights").fetchone()[0] or 0
    return {
        "id": cycle_id(last + 1),
        "model": model,
        "trigger": activity.get("trigger", "idle"),
        "started_at": activity.get("started_at"),
        "ended_at": None,
        "stage": "measure" if (activity.get("calibration") or {}).get("target_sequence") else stage,
        "state": "running",
        "wait": None,
        "stages": measuring(stages(stage, "running", read=read, practice=practice), activity, store),
        "read": read,
        "practice": practice,
        "train": train_summary({}),
        "checks": checks({}, config, False),
        "result": None,
    } | ({"signals": signal_lines(store)} if every else {})


def measuring(found, activity, store=None):
    """Measuring what this computer can train (before the first training
    step) is its own step, "Getting ready", between Practice and Training,
    so it never looks like training (user decision): the sizes that fitted
    and the limit found; Training stays ahead."""
    from .sleep.calibration_state import measuring_tokens

    progress = activity.get("calibration")
    if not (progress and progress.get("target_sequence")):
        return found
    saved = (store.get("calibration") if store is not None else None) or {}
    fits = sorted({r["sequence"] for r in saved.get("rows", []) if r.get("sequence")})
    limits = [x.get("sequence") for x in (saved.get("held") or []) + (saved.get("capacity_limits") or [])
              if x.get("sequence")]
    out = []
    for s in found:
        if s["key"] == "train":
            out.append({
                "key": "measure", "state": "now", "started_at": None, "ended_at": None, "eta_seconds": None,
                # The part still being measured: once the model's own size is
                # covered, the teacher's longer reading (live: it sat at a
                # finished "6550 of 6550" while the teacher was measured).
                "progress": dict(zip(("done", "total", "longer"), measuring_tokens(progress), strict=True), unit="calibration"),
                "sizes": fits, "limit": min(limits) if limits else None,
            })
            s = s | {"state": "ahead", "progress": None}
        out.append(s)
    return out


def signal_lines(store):
    from .reading import store as reading

    return [
        {k: s.get(k) for k in ("id", "session", "type", "source", "text", "lesson", "generality")}
        for s in reading.signals(store)
    ]


def versions(store):
    current = (store.version() or {}).get("id")
    out, n = [], 0
    for v in sorted(store.versions(), key=lambda v: v.get("created") or 0):
        base = v.get("kind") == "base" or not v.get("night")
        if not base:
            n += 1
        out.append(
            {
                "id": v["id"],
                "label": "Original" if base else f"v{n}",
                "current": v["id"] == current,
                "kind": "base" if base else "adapter",
                "cycle": None if base else cycle_id(v["night"]),
                "created_at": v.get("created"),
            }
        )
    return out


def model_learning(config, store, model, activity=None):
    """ModelLearning for one learner (store scoped to it)."""
    nights = [n for n in store.nights() if n["status"] != "discarded"]
    finished = [n for n in nights if n["status"] in FINISHED]
    open_ = [n for n in nights if n["status"] not in FINISHED]
    last = night_cycle(config, store, finished[-1], model) if finished else None
    # A Quick Dream reads and updates the prompt; it is not a cycle with
    # stages, so it shows as quick_running, not as the current Dream.
    # Idle reading is Quick Dream's first half: it shows the same way.
    quick_running = activity if activity and activity.get("kind") in {"quick", "read"} else None
    # The first measurement of what this machine can train is not a Dream
    # (live: the Deep Dream page showed "training" on an empty store): the
    # dashboard shows it as a banner instead.
    from .models.registry import find

    row = find(store, model) if activity and activity.get("calibration") and not open_ else None
    measuring_machine = None
    if row and row.get("support") == "pending":
        progress = activity["calibration"]
        measuring_machine = {"done": progress.get("measured_sequence") or 0,
                             "total": progress.get("target_sequence") or 0}
    if open_:
        current = night_cycle(config, store, open_[-1], model)
    elif activity and not quick_running and not measuring_machine:
        current = upcoming(config, store, model, activity)
    else:
        current = None
    if last and last["result"]:
        known = versions(store)
        labels = {v["cycle"]: v["label"] for v in known if v["cycle"]}
        adopted = last["result"]["outcome"] == "adopted"
        last["result"]["version"] = labels.get(last["id"]) if adopted else None
        # After an undo the last Dream's version is kept but no longer used.
        last["result"]["in_use"] = adopted and any(
            v["current"] and v["cycle"] == last["id"] for v in known
        )
    since = (last or {}).get("started_at") or 0
    pending = None
    if config.learning == "dreams":
        from .reading import store as reading

        with store.connect() as db:
            sessions = {
                r[0]
                for r in db.execute(
                    "SELECT session FROM read_sessions WHERE model_id=? AND read_at>?",
                    (store.model_id, since),
                )
            }
        pending = {
            "conversations": reading.counts(store)["unread"],
            "signals": sum(
                1 for s in reading.signals(store) if s["session"] in sessions and s["type"] in SIGNAL_TYPES
            ),
        }
    from . import quick

    return {
        "current": current,
        "last": last,
        "pending": pending or {"conversations": None, "signals": None},
        "versions": versions(store),
        # Quick Dream: the lessons in the prompt now, and its last run.
        "lessons": [
            {k: lesson[k] for k in ("id", "text", "count", "added_at")}
            for lesson in quick.active(store)
        ],
        "quick": store.get("quick_last"),
        # The last Quick Dream check, also one that found nothing: {at, nothing_new}.
        "quick_checked": store.get("quick_checked"),
        "quick_running": quick_view(quick_running) if quick_running else None,
        # Lessons waiting for a Deep Dream (skills): {"total", "lessons"}.
        "waiting": quick.waiting(store),
        # The Dream row: finished Quick and Deep Dreams, newest first.
        "dreams": dreams(store),
        # The first measurement of this machine: under way {done, total}, and
        # how it ended {at, support, reason} until the viewer closes it.
        "measuring_machine": measuring_machine,
        "machine_measured": store.get("machine_measured"),
        # The measured training window: {window, limit (memory|conversations),
        # served, example_seconds}, or None before the first measurement.
        "training": training_window(config, store.get("calibration")),
        # A pressed Deep Dream that stopped before training: {kind, at, reason, detail}.
        "stopped": store.get("dream_stopped"),
        # Under Dreams, separate test conversations still needed before a Deep
        # Dream's result can be checked: {have, need}, or None.
        "collecting": collecting(config, store),
        # A pressed Deep Dream between its runs (measuring, then training): its step.
        "deep_waiting": deep_waiting(store, model, current),
    }


def training_window(config, saved):
    from .sleep.calibration_state import training_window as window

    try:
        return window(config, saved)
    except (KeyError, TypeError, ValueError):
        return None


def collecting(config, store):
    if getattr(config, "learning", "") != "dreams" or not store.traffic_times(1):
        return None
    from .practice.pool import heldout_units, test_minimum

    need = test_minimum(config)
    have = heldout_units(store)  # real moments, filled by held-out practice
    return {"have": have, "need": need} if have < need else None


def deep_waiting(store, model, current):
    from .status import deep_step

    if current or store.get("dream_stopped"):
        return None
    asked = store.get("requested_model")
    # A pressed Deep Dream still pending (its model named, or the only
    # request there is while it waits for memory).
    if asked != model and not (store.get("requested_sleep") and asked is None and store.get("fit_wait")):
        # An automatic run the memory guard stopped: why, and when it tries again.
        from .status import memory_hold

        hold = memory_hold(store)
        return hold[0].upper() + hold[1:] if hold else None
    return deep_step(store)


DREAMS = 12  # chips in the page's Dream row


def dreams(store, limit=DREAMS):
    """Finished Dreams, Quick and Deep, newest first, each light enough for
    every poll (a Deep Dream's full report loads from /api/cycles/<id>).
    seen: the user has looked at it (control "seen")."""
    from . import quick

    seen = set(store.get("seen_dreams") or [])
    labels = {v["cycle"]: v["label"] for v in versions(store) if v["cycle"]}
    out = []
    for night in store.nights():
        if night["status"] not in FINISHED:
            continue
        found = result(night, True)
        record = night.get("record") or {}
        cid = cycle_id(night["id"])
        out.append({
            "id": cid,
            "kind": "deep",
            "at": record.get("ended_at") or night.get("created"),
            "outcome": found["outcome"],
            "reason": found["reason"],
            "version": labels.get(cid) if found["outcome"] == "adopted" else None,
            "improved": found["improved"],
        })
    for entry in store.get(quick.HISTORY) or []:
        out.append({
            "id": entry["id"],
            "kind": "quick",
            "at": entry["at"],
            "lessons": entry["lessons"],
            "changes": {k: entry.get(k) or [] for k in ("added", "reworded", "dropped")},
            "examples": entry.get("examples") or [],
            **{k: entry[k] for k in ("read", "found", "error", "undone") if k in entry},
        })
    out.sort(key=lambda d: d["at"] or 0, reverse=True)
    # Back-to-back Quick Dreams that changed nothing are one chip, the
    # newest, with how many (the row stays readable).
    row = []
    for d in out:
        quiet = d["kind"] == "quick" and not d["changes"]["added"] and not d.get("error")
        last = row[-1] if row else None
        if quiet and last and last.get("quiet"):
            last["repeat"] = last.get("repeat", 1) + 1
            continue
        row.append(d | ({"quiet": True} if quiet else {}))
    return [d | {"seen": d["id"] in seen} for d in row[:limit]]


def quick_view(activity):
    """A Quick Dream (or idle reading) under way, for its page: the step, the
    conversations read of those waiting, and the time left. Reading stops at
    its budget (quick_read_seconds), so the time left is at most what remains
    of it; within that, the measured pace per conversation."""
    import time

    started = activity.get("started_at") or time.time()
    elapsed = max(0.0, time.time() - started)
    done, total, budget = activity.get("done", 0), activity.get("total", 0), activity.get("budget")
    # No estimate until three conversations are timed, then their average.
    paces = activity.get("paces") or []
    pace = sum(paces) / len(paces) if len(paces) >= 3 else None
    left = None
    if activity.get("stage") == "read":
        if pace is not None:
            left = max(0.0, (total - done) * pace)
            if budget:
                left = min(left, max(0.0, budget - elapsed))
    return {
        "kind": activity.get("kind"),
        "stage": activity.get("stage", "read"),
        "started_at": started,
        "done": done,
        "total": total,
        "budget": budget,
        "eta_seconds": left,
        "reading": activity.get("reading") if activity.get("stage") == "read" else None,
        "found": activity.get("found") or [],
        "found_count": activity.get("found_count", 0),
    }


def mark_seen(store, dream_id):
    seen = [d for d in store.get("seen_dreams") or [] if d != dream_id]
    store.set("seen_dreams", (seen + [dream_id])[-500:])


def cycles(config, store, model, limit=20):
    """Past cycles, newest first, in the summary shape."""
    nights = [n for n in store.nights() if n["status"] in FINISHED]
    return [night_cycle(config, store, n, model) for n in reversed(nights[-limit:])]


def full_cycle(config, store, model, cycle):
    night_id = night_of(cycle)
    night = next((n for n in store.nights() if n["id"] == night_id), None)
    return night_cycle(config, store, night, model, every=True) if night else None
