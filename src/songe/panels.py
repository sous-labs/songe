"""Read-only presentation of a model's recorded pipeline, including older APIs."""

import re
from copy import deepcopy

from .status import model_training, training_model
from .wording import (
    PHASE_LABELS,
    SIGNAL_SOURCES,
    comparison,
    duration,
    in_use,
    measured,
    model_level,
    model_problem,
    model_summary,
    next_text,
    plural,
    stamp,
    version_name,
)


def is_base(version):
    """Original weights are 'base' or 'base-<hash>' with any artifact kind."""
    return (
        version.get("kind") == "base"
        or version.get("id") == "base"
        or str(version.get("id")).startswith("base-")
    )


def trainability(model):
    support = model.get("support", model.get("status"))
    if support == "unsupported":
        return "unsupported"
    if "paused" in (model.get("reason") or "").lower() or support in {"paused", "off"}:
        return "paused"
    if support in {None, "pending", "unknown"}:
        return "not measured"
    if model.get("learning_level") == "off" or model.get("enabled") is False:
        return "paused"
    return "learning"


def pipeline(model, training=None, snapshot=None):
    """Prefer the pipeline contract; fill only fields older services record."""
    learning = model.get("learning") or {}
    result = {}
    if learning.get("conversations") is not None:
        result["day"] = {"conversations": learning["conversations"]}
    signals = learning.get("signals")
    if signals:
        result["rewind"] = {"signals": deepcopy(signals)}
    last = learning.get("last")
    if last is not None:
        result["wake"] = {
            "last": {
                "at": last.get("created"),
                "relative_repeat_errors_change": last.get("repeat_errors_change"),
                "swapped": last.get("status") == "swapped",
            }
        }
    if model.get("versions"):
        result["versions"] = deepcopy(model["versions"])
    for source in [
        model.get("pipeline"),
        ((snapshot or {}).get("pipeline") or {}).get(model["id"]),
    ]:
        for key, value in (source or {}).items():
            if isinstance(value, dict):
                result[key] = (result.get(key) or {}) | deepcopy(value)
            else:
                result[key] = deepcopy(value)
    if snapshot is not None:
        training = snapshot.get("training") if training_model(snapshot) else None
        if training and not training.get("model"):
            training = training | {"model": training_model(snapshot)}
    if (
        training
        and model_training(model, training.get("model"))
        and training.get("phase") in {"train", "distill"}
    ):
        progress = training.get("data_progress") or {}
        if progress.get("phase", "train") == "train":
            values = {
                "step": training.get("step", progress.get("step")),
                "total": training.get("total_steps", progress.get("total_steps")),
                "eta_seconds": progress.get("eta_seconds"),
            }
            result["consolidate"] = (result.get("consolidate") or {}) | {
                k: v for k, v in values.items() if v is not None
            }
    return result


def has_data(data):
    def positive(value):
        if isinstance(value, dict):
            return any(positive(v) for v in value.values())
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value > 0
        )

    return (
        any(positive(data.get(key)) for key in ("day", "rewind", "consolidate"))
        or any(
            v is not None for v in ((data.get("wake") or {}).get("last") or {}).values()
        )
        or any(not is_base(v) for v in data.get("versions") or [])
    )


MEMORY_WAIT = re.compile(r"memory|free \d|needs \d.*GB", re.I)


def panel_rows(model, training=None, snapshot=None):
    """Expanded model details, most important first: now, result, data,
    versions; problems last as what · fix. Internal counts stay out."""
    data = pipeline(model, training, snapshot)
    rows = []
    active = (
        snapshot is not None
        and not snapshot.get("paused")
        and model_training(model, training_model(snapshot))
    )
    if snapshot is None and training is not None:
        active = model_training(model, training.get("model"))
    work = data.get("consolidate") or {}
    if active:
        parts = ["Training"]
        if (work.get("total") or 0) > 0 or (work.get("step") or 0) > 0:
            parts.append(
                f"step {work.get('step') or 0}"
                + (f" of {work['total']}" if (work.get("total") or 0) > 0 else "")
            )
        if measured(work.get("eta_seconds")):
            parts.append(f"about {duration(work['eta_seconds'])} left")
        rows.append(("now", " · ".join(parts)))
    elif (
        snapshot is not None
        and not snapshot.get("paused")
        and (snapshot.get("training") or {}).get("phase") == "calibrate"
        and model_training(model, (snapshot.get("training") or {}).get("model"))
    ):
        from .sleep.calibration_state import measuring_tokens

        sizing = (snapshot.get("training") or {}).get("data_progress") or {}
        fitted, target, longer = measuring_tokens(sizing)
        done = f"{fitted} of {target} tokens · " if target else ""
        rows.append(
            (
                "now",
                "Measuring what this machine can train · "
                + done
                + ("longer contexts" if longer else "first time only"),
            )
        )
    elif (
        snapshot is not None
        and (label := model_summary(model, data, snapshot)[0]) in PHASE_LABELS.values()
    ):
        # Between work slices the job is still running: the row says Training,
        # so Now says so too instead of an empty dash.
        pause = "retry delay after previous work slice" in str(
            (snapshot.get("activity") or {}).get("now") or ""
        )
        rows.append(("now", label + (" · short pause between runs" if pause else "")))
    elif snapshot is not None and re.match(
        r"Waiting for (memory|disk space)$", model_summary(model, data, snapshot)[0]
    ):
        rows.append(
            ("now", " · ".join(filter(None, model_summary(model, data, snapshot))))
        )
    elif (
        any(
            MEMORY_WAIT.search(item.get("reason") or "")
            for item in [model, *model.get("files", [])]
        )
        and model.get("support", model.get("status")) != "unsupported"
    ):
        rows.append(
            (
                "now",
                "In use · continues after this conversation"
                if in_use(snapshot)
                else "Waiting for memory · retries automatically",
            )
        )
    last = (data.get("wake") or {}).get("last") or {}
    if any(value is not None for value in last.values()):
        values = (
            [comparison(last) or "Updated"]
            if last.get("swapped") is True
            else ["Kept previous version", "did not pass checks"]
        )
        if measured(last.get("at")):
            values.append(stamp(last["at"]))
        rows.append(("result", " · ".join(values)))
    rewind = data.get("rewind") or {}
    # What it learns from, in plain words; counts stay out.
    signals = [
        (key, value)
        for key, value in (rewind.get("signals") or {}).items()
        if measured(value) and value > 0
    ]
    sources = [
        SIGNAL_SOURCES.get(key, key.replace("_", " "))
        for key, _ in sorted(signals, key=lambda item: -item[1])[:3]
    ]
    if sources:
        rows.append(("learned", " · ".join(sources)))
    facts = []
    conversations = (data.get("day") or {}).get("conversations")
    if measured(conversations) and conversations > 0:
        facts.append(plural(conversations, "conversation"))
    if measured(rewind.get("lessons")) and rewind["lessons"] > 0:
        facts.append(plural(rewind["lessons"], "lesson"))
    if facts:
        rows.append(("data", " · ".join(facts)))
    upcoming = next_text(data.get("next"))
    if (
        upcoming
        and model_level(model) != "off"
        and not any(k == "now" for k, _ in rows)
    ):
        rows.append(("next", upcoming))
    versions = data.get("versions") or []
    if any(not is_base(version) for version in versions):
        rows.append(
            (
                "versions",
                " · ".join(
                    ("Original" if is_base(version) else version_name(version["id"]))
                    + (
                        " · In use"
                        if version.get("current")
                        else " · Available to restore"
                    )
                    for version in versions
                ),
            )
        )
    issue = model_problem(model)
    if issue:
        rows.append(("problem", issue))
    return rows


STAGE_LABELS = {
    "now": "Now",
    "result": "Result",
    "learned": "Learns from",
    "next": "Next",
    "data": "Data",
    "versions": "Versions",
    "problem": "Fix",
}
PANEL_ORDER = ("now", "result", "learned", "data", "next", "versions")


def dashboard_rows(model, training=None, snapshot=None):
    """One fixed set of (key, value, empty) rows so nothing jumps; rows
    without data show a quiet default. Unsupported models show only the fix."""
    sparse = panel_rows(model, training, snapshot)
    if model.get("support", model.get("status")) == "unsupported":
        return [(key, value, False) for key, value in sparse]
    values = dict(sparse)
    data = pipeline(model, training, snapshot)
    first = not has_data(data) and "now" not in values
    defaults = {
        "now": "Learning off" if model_level(model) == "off" else "Nothing running",
        "result": "No update yet",
        "learned": "your approvals, corrections and retries",
        "data": "No conversations yet · pick a songe/… model in your agent",
        "next": "Turn on New or All to learn"
        if model_level(model) == "off"
        else "First update after your first conversations"
        if first
        else "update after more conversations",
        "versions": "Original · In use",
    }
    # What it learns from is shown only once there is something to show.
    rows = [
        (key, values[key], False) if key in values else (key, defaults[key], True)
        for key in PANEL_ORDER
        if key != "learned" or key in values
    ]
    if "problem" in values:
        rows.append(("problem", values["problem"], False))
    return rows
