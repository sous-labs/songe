"""Read-only learning evidence for local interfaces; no inference or model calls."""

import copy
import json
import math
import re
from collections import Counter
from datetime import datetime
from itertools import chain

from .log import change_mark
from .steps import digest, tool_outcome, user_signal

COMMANDS = frozenset(
    "git pytest python python3 uv pip npm npx node pnpm yarn cargo rustc go make "
    "cmake ruff mypy docker curl rg sed awk bash zsh sh sqlite3 jest vitest tsc".split()
)


def measured(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def stamp(value):
    return (
        datetime.fromtimestamp(value).strftime("%m-%d %H:%M")
        if value is not None
        else "—"
    )


def message_text(message):
    content = message.get("content") or ""
    if isinstance(content, list):
        return " ".join(
            part["text"]
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return content if isinstance(content, str) else ""


def tool_calls(message):
    for call in message.get("tool_calls") or []:
        if isinstance(call, dict) and isinstance(call.get("function"), dict):
            yield call, call["function"]


def topics(messages):
    found = set()
    for message in messages:
        text = message_text(message)
        for _, function in tool_calls(message):
            name = function.get("name", "")
            if isinstance(name, str) and re.fullmatch(
                r"[A-Za-z][A-Za-z0-9_.-]{0,63}", name
            ):
                found.add(name)
            text += " " + str(function.get("arguments", ""))
        found.update(
            COMMANDS.intersection(re.findall(r"\b[a-z][a-z0-9]*\b", text.lower()))
        )
        found.update(
            "." + ext.lower()
            for ext in re.findall(r"\b[\w-]+\.([A-Za-z][A-Za-z0-9]{0,7})\b", text)
        )
    return found


def evidence_summary(items):
    conversations, topic_counts = {}, Counter()
    fixes, corrections, successes, topic_events = set(), set(), set(), set()
    for row in items:
        step = json.loads(row["payload"])
        conversation = row["conversation"]
        context = step.get("context") or []
        first = next((message_text(m) for m in context if m.get("role") == "user"), "")
        preview = next(
            (line.strip() for line in first.splitlines() if line.strip()), "—"
        )[:160]
        created = row["created"]
        if (
            conversation not in conversations
            or created < conversations[conversation]["created"]
        ):
            conversations[conversation] = {
                "id": conversation,
                "created": created,
                "first_line": preview,
            }
        action = step.get("action") or {}
        tail = step.get("hindsight") or step.get("evidence") or []
        # Rolling protocol prefixes distinguish branches while identifying the
        # same outcome in several ancestors' hindsight. No repeated evidence or
        # topics are counted twice, and success cannot cross unrelated branches.
        prefix, tool_names = "", {}
        errors, retries = {}, set()
        correction_seen = False
        for index, message in enumerate(chain(context, (action,), tail)):
            prefix = digest([prefix, digest(message)])
            event = (conversation, prefix)
            for call, function in tool_calls(message):
                if isinstance(call.get("id"), str) and isinstance(
                    function.get("name"), str
                ):
                    tool_names[call["id"]] = function["name"]
            if index < len(context):
                continue
            if event not in topic_events:
                topic_events.add(event)
                topic_counts.update(topics([message]))
            if index == len(context):
                continue
            role = message.get("role")
            outcome = tool_outcome(message) if role == "tool" else None
            signal = user_signal(message) if role == "user" else None
            name = tool_names.get(message.get("tool_call_id"))
            if outcome in {"tool_error", "exec_failure"}:
                errors.setdefault(name, set()).add(event)
            if signal == "retry":
                retries.add(event)
            if signal == "user_correction":
                corrections.add(event)
                correction_seen = True
            if outcome == "success" or signal == "approval":
                if signal == "approval":
                    fixes.update(event for group in errors.values() for event in group)
                    errors.clear()
                else:
                    fixes.update(errors.pop(name, set()))
                successes.update(retries)
                retries.clear()
        if row["signal"] == "user_correction" and not correction_seen:
            corrections.add((conversation, step["id"]))
    details = sorted(conversations.values(), key=lambda c: (c["created"], c["id"]))
    return {
        "conversations": len(details),
        "since": details[0]["created"] if details else None,
        "signals": {
            "tool_error_fixed": len(fixes),
            "user_correction": len(corrections),
            "retry_success": len(successes),
        },
        "topics": [
            word
            for word, _ in sorted(topic_counts.items(), key=lambda x: (-x[1], x[0]))[:4]
        ],
        "sources": details,
    }


_MEMO = {}


def summaries(store, models, nights, scoreboard):
    """Apply the store's learning window, including shared history for `all`.

    Decoding every step costs most of a dashboard refresh; the result only
    changes when the database does, so it is reused until a write."""
    if not models:
        return {}
    mark = change_mark(store.path)
    if mark is None:  # No database yet; nothing to reuse.
        return _summaries(store, models, nights, scoreboard)
    key = (
        str(store.path),
        mark,
        json.dumps([models, nights, scoreboard], sort_keys=True, default=str),
    )
    if key in _MEMO:
        return copy.deepcopy(_MEMO[key])
    result = _summaries(store, models, nights, scoreboard)
    _MEMO.clear()
    _MEMO[key] = copy.deepcopy(result)
    return result


def _summaries(store, models, nights, scoreboard):
    with store.connect() as db:
        tables = {
            r[0]
            for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        recorded_at = (
            "COALESCE(e.recorded_at,e.created)"
            if "recorded_at"
            in {r[1] for r in db.execute("PRAGMA table_info(exchanges)")}
            else "e.created"
        )
        items = (
            [
                dict(r)
                for r in db.execute(
                    "SELECT s.payload,s.conversation,s.signal,e.created,"
                    f"{recorded_at} AS recorded_at FROM steps s "
                    "JOIN exchanges e ON e.id=json_extract(s.payload,'$.source_id') "
                    "WHERE s.retired_at IS NULL ORDER BY e.created,s.id"
                )
            ]
            if {"steps", "exchanges"} <= tables
            else []
        )
    primary = store.get("primary_model")
    result = {}
    for model in models:
        level = model.get("learning_level", "off")
        cutoff = model.get("learning_cutoff")
        eligible = (
            items
            if level == "all"
            else [
                item
                for item in items
                if level == "new"
                and cutoff is not None
                and item["recorded_at"] > cutoff
            ]
        )
        summary = evidence_summary(eligible)
        model_id = model["id"]
        scoped_nights = [
            n for n in nights if (n.get("model_id") or primary) == model_id
        ]
        latest = next(
            (
                n
                for n in reversed(scoped_nights)
                if n["status"] in {"accepted", "rejected"}
            ),
            None,
        )
        summary["last"] = None
        if latest:
            record = latest.get("record") or {}
            bindings = {model_id}
            bindings.update(v.get("identifier") for v in model.get("versions", []))
            bindings.update(v.get("id") for v in model.get("versions", []))
            observations = [r for r in scoreboard if r["model_version"] in bindings]
            version_times = {
                v.get(key): v.get("created", 0)
                for v in model.get("versions", [])
                for key in ("id", "identifier")
            }
            observations.sort(
                key=lambda r: (
                    r.get("day", ""),
                    version_times.get(r["model_version"], 0),
                )
            )
            rates = [
                r["repeat_error_rate"]
                for r in observations
                if measured(r.get("repeat_error_rate"))
            ]
            change = (
                (rates[-1] / rates[-2] - 1)
                if len(rates) >= 2 and rates[-2] > 0
                else None
            )
            summary["last"] = {
                "created": model.get("last_update") or latest["created"],
                "repeat_errors_change": change,
                "drift": record.get("general_drift")
                if measured(record.get("general_drift"))
                else None,
                "status": "swapped"
                if latest["status"] == "accepted"
                else "kept previous",
            }
        result[model_id] = summary
    return result


def summary_lines(model, *, compact=False):
    summary = model.get("learning")
    if not summary:
        return []
    if not summary["conversations"]:
        return ["learning   no conversations through songe yet"]
    signals = summary["signals"]
    lines = [
        f"learning   {model.get('name') or model['id']} · {summary['conversations']} conversations · since {stamp(summary['since'])}"
    ]
    values = [
        ("tool_error → fixed", signals["tool_error_fixed"]),
        ("user_correction", signals["user_correction"]),
        ("retry → success", signals["retry_success"]),
        ("topics", ", ".join(summary["topics"]) or "—"),
    ]
    if compact:
        lines.append("  " + " · ".join(f"{key} {value}" for key, value in values))
    else:
        lines.extend(f"  {key:<23} {value}" for key, value in values)
    last = summary.get("last")
    if last:
        change = last["repeat_errors_change"]
        repeat = f"{change:+.0%}".replace("-", "−") if change is not None else "—"
        drift = f"{last['drift']:.3f}" if last["drift"] is not None else "—"
        outcome = "✓ swapped" if last["status"] == "swapped" else "✗ kept previous"
        lines.append(
            f"last       {stamp(last['created'])} · repeat_errors {repeat} · drift {drift} · {outcome}"
        )
    else:
        lines.append("last       —")
    return lines
