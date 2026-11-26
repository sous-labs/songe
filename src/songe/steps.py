"""Prefix linking uses only protocol messages, never session IDs or tool execution."""

import hashlib
import json
import time
from dataclasses import asdict, dataclass, fields


def canonical(value, legacy=False):
    """A message history in the form agents agree on. Agents echo the
    model's turns back differently (live: LM Studio replied "\n\n" before a
    tool call, pi sent it back as null), so whitespace-only reply text is
    empty, reasoning is dropped and streaming-only tool-call fields (index,
    type) are ignored. legacy: the form before whitespace and tool-call
    normalising, for links stored then."""
    if isinstance(value, dict):
        if "name" in value and isinstance(value.get("arguments"), str):
            try:
                value = value | {"arguments": json.loads(value["arguments"])}
            except ValueError:
                pass
        skip = {"reasoning_content", "reasoning", "annotations"}
        if not legacy and "function" in value:
            skip |= {"index", "type"}
        out = {k: canonical(v, legacy) for k, v in value.items() if v is not None and k not in skip}
        if out.get("role") == "assistant":
            out.setdefault("content", "")
            if not legacy and isinstance(out["content"], str) and not out["content"].strip():
                out["content"] = ""
        return out
    if isinstance(value, list):
        if (
            all(isinstance(v, dict) and v.get("type") == "text" for v in value)
            and value
        ):
            return "".join(v["text"] for v in value)
        return [canonical(v, legacy) for v in value]
    return value


def digest(value, legacy=False):
    return hashlib.sha256(
        json.dumps(
            canonical(value, legacy), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def prefix_keys(messages):
    """The keys a history prefix may be stored under: the normalised one,
    and the older form (links stored before it) when that differs."""
    keys = [digest(messages)]
    old = digest(messages, legacy=True)
    return keys if old == keys[0] else keys + [old]


def split_for(conversation):
    """New conversations wait; Store.assign_splits picks tests once there is
    enough data, so tests resemble what will be trained."""
    return "pending"


@dataclass
class Step:
    id: str
    conversation: str
    split: str
    context: list
    tools: list
    action: dict
    hindsight: list
    source_id: int
    observed_id: int = 0
    signal: str | None = None
    strength: float = 0.0
    evidence: list | None = None
    answer_hint: dict | None = None
    env_fingerprint: str = ""
    retired_at: float | None = None
    retired_reason: str | None = None
    task: str = ""

    def dict(self):
        return asdict(self)

    @classmethod
    def load(cls, payload):
        """A stored step; keys added by later stages (e.g. clean's
        "repeated") are ignored rather than crashing reconstruction."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in names})


# After a compaction some agents (pi) resend the kept turns with the summary
# inserted before them as a user message.
CONTINUED = ("The conversation history before this point was compacted",)


def _said(content):
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content if isinstance(content, str) else ""


def unsummarised(messages):
    """The history without an inserted compaction summary; the kept turns
    then match the session they came from. None when there is none."""
    kept = [
        m for m in messages
        if not (m.get("role") == "user" and _said(m.get("content")).lstrip().startswith(CONTINUED))
    ]
    return kept if len(kept) != len(messages) else None


# A conversation whose earlier turns were never recorded (the agent trimmed
# its history, or the chat was open before Songe started): read for notes and
# practice, never a training context or a test moment (#263).
UNLINKED = "history not recorded"


def extract(exchanges, public_model=None, *, prefixes=None, steps=None, unlinked=False):
    """Reconstruct roots chronologically; all branches inherit the original split.

    On prefix compaction/mutation there is no evidence of continuity; such turns
    are excluded if their history contains assistant turns we cannot link. This
    avoids accidentally training the held-out continuation of a known thread.

    `unlinked`: such a turn has waited long enough for its parent. It starts a
    conversation of its own, so it is read; its steps, and every step after
    them, are retired (UNLINKED), so they are never trained or tested.
    """
    prefixes = {} if prefixes is None else prefixes
    steps = {} if steps is None else steps
    for row in exchanges:
        req, resp = row["request"], row["response"]
        if (
            row["path"] != "/v1/chat/completions"
            or not row["complete"]
            or row["status"] != 200
        ):
            continue
        if public_model and req.get("model") != public_model:
            continue
        messages = req.get("messages", [])
        if not messages or not isinstance(resp, dict):
            continue
        choices = resp.get("choices", [])
        if len(choices) != 1 or not isinstance(choices[0].get("message"), dict):
            continue
        action = choices[0]["message"]
        if action.get("role") != "assistant":
            continue
        root, ancestors, linked = None, [], []
        for length in range(len(messages), 0, -1):
            key = next((k for k in prefix_keys(messages[:length]) if k in prefixes), None)
            if key:
                parent_id, root = prefixes[key]
                ancestors.append((parent_id, length))
                linked.append(parent_id)
        bare = unsummarised(messages) if root is None else None
        if bare:
            # A continuation after a compaction: the kept turns, without the
            # summary, link it to the session it continues.
            for length in range(len(bare), 0, -1):
                key = next((k for k in prefix_keys(bare[:length]) if k in prefixes), None)
                if key:
                    root = prefixes[key][1]
                    linked.append(prefixes[key][0])
                    break
        retire = any(
            getattr(steps.get(parent_id), "retired_reason", None) == UNLINKED
            for parent_id in linked
        )
        if root is None:
            if bare is None and any(m.get("role") == "assistant" for m in messages):
                if not unlinked:
                    continue
                retire = True
            # A new session, or a continuation whose kept turns are unknown
            # (the reading timeline joins it through its summary).
            root = digest(messages)
        # Read through the sub-task, including intervening assistant actions.
        # A fresh user instruction ends it; explicit corrections/retries and
        # approvals are consequences of the current task, not a new task.
        for parent_id, length in ancestors:
            after = subtask_tail(messages[length:])
            if len(after) > len(steps[parent_id].hindsight):
                steps[parent_id].hindsight = after
                steps[parent_id].observed_id = row["id"]
        sid = digest(
            {
                "root": root,
                "messages": messages,
                "action": action,
                "tools": req.get("tools", []),
            }
        )
        if sid not in steps:
            steps[sid] = Step(
                sid,
                root,
                split_for(root),
                messages,
                req.get("tools", []),
                action,
                [],
                row["id"],
            )
            if retire:
                steps[sid].retired_at = time.time()
                steps[sid].retired_reason = UNLINKED
        prefixes[digest(messages + [action])] = (sid, root)
    for step in steps.values():
        for index, message in enumerate(step.context):
            if message.get("role") == "user" and user_signal(message) is None:
                step.task = digest(step.context[: index + 1])
        # One session is cut at each new instruction. That part, not the whole
        # session, is what is counted, cleaned and set aside for tests: people
        # work in one session all day, and it would never yield a third one.
        step.conversation = step.task or step.conversation
        step.env_fingerprint = digest(step.tools)
        step.signal, step.strength, step.evidence, step.answer_hint = signal_for(step)
    return list(steps.values())


# Ordinal evidence confidence, rather than a threshold for accepting training.
SIGNAL_STRENGTH = {
    "later_success": 1.0,
    "tool_error": 0.8,
    "exec_failure": 0.8,
    "user_correction": 0.8,
    "loop": 0.8,
    "static_invalid": 0.5,
    "retry": 0.25,
    "approval": 0.1,
}
CORRECTIONS = (
    "wrong",
    "incorrect",
    "should be",
    "should have",
    "instead",
    "fix",
    "actually",
    "that should",
    "不对",
    "应该",
    "改成",
)
RETRIES = ("try again", "retry", "again", "再试", "重试")
APPROVALS = (
    "thanks",
    "thank you",
    "correct",
    "works",
    "looks good",
    "great",
    "好了",
    "谢谢",
)


# What the local model read each user reply as (see songe.review); the phrase
# lists below are only a fallback until a reply has been reviewed.
REVIEWED = {}
# The review's issue and lesson for each reply; organises the teacher's hindsight.
REVIEW_NOTES = {}


def message_key(message):
    content = message.get("content", "")
    if not isinstance(content, str):
        content = json.dumps(content, sort_keys=True, ensure_ascii=False)
    return digest(" ".join(content.split()))


def reply_key(step):
    """One reply event: the reply plus the message it answered. Steps ending at
    the same event share it; the same words in another conversation do not."""
    hindsight = step.get("hindsight") or []
    if not hindsight:
        return ""
    prior = hindsight[-2] if len(hindsight) > 1 else step.get("action") or {}
    return digest([message_key(hindsight[-1]), message_key(prior)])


def user_signal(message):
    key = message_key(message)
    if key in REVIEWED:
        return REVIEWED[key]
    text = str(message.get("content", "")).strip().lower()
    for kind, phrases in (
        ("user_correction", CORRECTIONS),
        ("retry", RETRIES),
        ("approval", APPROVALS),
    ):
        if any(
            text.startswith(p) or (" " + p + " ") in (" " + text + " ") for p in phrases
        ):
            return kind
    return None


def subtask_tail(messages):
    """What followed an action: tool results and later steps, up to and
    including the user's next message, whatever it says. The teacher reads
    that reply itself; no phrase list decides whether it is evidence."""
    tail = []
    for message in messages:
        tail.append(message)
        # A correction or retry keeps the same task going (so a later fix
        # still counts); any other reply ends it, but is itself included.
        if message.get("role") == "user" and user_signal(message) not in {
            "user_correction",
            "retry",
        }:
            break
    return tail


def tool_outcome(message):
    content = message.get("content", "")
    try:
        value = json.loads(content) if isinstance(content, str) else content
    except (ValueError, TypeError):
        value = None
    if isinstance(value, dict):
        code = value.get("exit_code", value.get("exitCode", value.get("returncode")))
        if code is not None:
            return "success" if code == 0 else "exec_failure"
        if value.get("error") or value.get("isError"):
            return "tool_error"
    text = str(content).lower()
    if any(
        t in text
        for t in ("file not found", "traceback", "error:", "failed", "exception")
    ):
        return "tool_error"
    return None


def valid_action(action, tools):
    schemas = {
        t.get("function", {}).get("name"): t.get("function", {}).get("parameters", {})
        for t in tools
    }
    for call in action.get("tool_calls", []):
        if not isinstance(call, dict):
            return False
        fn = call.get("function", {})
        if not isinstance(fn, dict):
            return False
        if fn.get("name") not in schemas:
            return False
        try:
            args = (
                json.loads(fn.get("arguments", "{}"))
                if isinstance(fn.get("arguments"), str)
                else fn.get("arguments")
            )
        except (TypeError, ValueError):
            return False
        if not valid_schema(args, schemas[fn["name"]]):
            return False
    return True


def valid_schema(value, schema):
    """Validate local JSON Schema, including constraints and local references.

    Never retrieve a remote reference from a logged/user-supplied schema.
    Unsupported or unresolvable schemas fail the gate rather than passing.
    """
    from jsonschema import Draft202012Validator, SchemaError, ValidationError
    from referencing import Registry
    from referencing.exceptions import Unresolvable

    try:
        json.dumps(value, allow_nan=False)
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema, registry=Registry()).validate(value)
        return True
    except (SchemaError, ValidationError, Unresolvable, ValueError, TypeError):
        return False


def signal_for(step):
    evidence, failures, hint = [], [], None
    action = step.action
    seen = {digest(c.get("function", c)) for c in action.get("tool_calls", [])}
    loop = False
    success_seen = False
    for message in step.hindsight:
        role = message.get("role")
        if role == "assistant":
            action = message
            for call in action.get("tool_calls", []):
                key = digest(call.get("function", call))
                if key in seen:
                    loop = True
                    evidence.append(message)
                seen.add(key)
        elif role == "tool":
            outcome = tool_outcome(message)
            if outcome:
                evidence.append(message)
                if outcome == "success":
                    success_seen = True
                    hint = action
                else:
                    failures.append(outcome)
        elif role == "user" and user_signal(message):
            evidence.append(message)
            failures.append(user_signal(message))
            if user_signal(message) == "approval":
                hint = action
    if hint is not None:
        kind = (
            "later_success"
            if success_seen or any(k != "approval" for k in failures)
            else "approval"
        )
    elif loop:
        kind = "loop"
    elif not valid_action(step.action, step.tools):
        kind = "static_invalid"
        evidence.append(step.action)
    else:
        kind = next(
            (
                k
                for k in (
                    "exec_failure",
                    "tool_error",
                    "user_correction",
                    "retry",
                    "approval",
                )
                if k in failures
            ),
            None,
        )
    return kind, SIGNAL_STRENGTH.get(kind, 0.0), evidence, hint


def revision_id(step):
    return max(step.get("source_id", 0), step.get("observed_id", 0))
