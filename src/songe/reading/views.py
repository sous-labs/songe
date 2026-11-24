"""Feedback events in one timeline, and the views of it the model is shown
(ported from scripts/signals/events.py).

An event is a point where something from outside the agent arrives that says
how its earlier actions went:

- user: each user message after the opening request
- harness: each message opencode writes in the user's place
- tool: tool results the rules flagged (tool_error, failed_result), with an
  unbroken run of the same result (e.g. 20x "game over: lost") as one event
- review: per turn, the tool results the rules did not flag, as one event
  ("did any of these tell the agent something it should have used?"): command
  output and the results of the agent's own tools, not file reads or edits

Successes come from the rules alone and are not events for the model.
"""

import json
import re

from .rules import FAILURE
from .rules import extract as rule_signals
from .timeline import clip

# Results the review has always seen: a command's output and a fetched page.
COMMAND_OUTPUT = {"shell", "execute", "bash", "webfetch"}
# Tools whose results say nothing about how the work went: they read, list,
# search or change files, or keep the agent's own plan (the names opencode, Pi,
# Claude Code, Codex and Cline give them, compared without case or
# punctuation). Every other tool's result is reviewed too: an agent given its
# own function or MCP tools hears how its work went through their results
# (live, Qwen3.5-9B playing a word game through a `guess` tool: 100 games,
# no result ever read, nothing learned from a lost game or a refused word).
QUIET = {
    "read", "readfile", "view", "viewimage", "edit", "editfile", "write", "writefile", "writetofile",
    "replaceinfile", "patch", "multiedit", "applypatch", "notebookedit", "grep", "glob", "list", "ls",
    "find", "searchfiles", "listfiles", "listdirectory", "todowrite", "todoread", "updateplan", "task", "skill",
}


def reviewed(tool):
    """Whether the results of a tool with this name go to the review."""
    return bool(tool) and re.sub(r"[^a-z0-9]", "", str(tool).lower()) not in QUIET


def normalized(text):
    """What makes two results 'the same': the failure phrase if there is one."""
    m = FAILURE.search(text)
    if m:
        return m.group(0).lower()
    return re.sub(r"\d+", "#", text.strip().lower())[:80]


INSPECT = re.compile(r"^\s*(cd [^&;]+(&&|;)\s*)*(ls|cat|head|tail|find|git (status|log|show|diff)|pwd|echo|grep|rg|wc|stat|file|which|ps|ss|netstat|lsof|env)\b")


def turns_of(items):
    """Index of the user message each item belongs to (0 = opening request)."""
    turn, out = -1, []
    for it in items:
        if it["kind"] == "user":
            turn += 1
        out.append(max(turn, 0))
    return out


def events(items, rules=None):
    rules = rules if rules is not None else rule_signals(items)
    turn_of = turns_of(items)
    flagged = {s["item"] for s in rules if s["type"] in ("tool_error", "failed_result", "success", "starting_problem")}
    out, step = [], 0
    for i, it in enumerate(items):
        if it["kind"] == "action":
            step = it["step"]
        elif it["kind"] == "user" and turn_of[i] > 0:
            out.append({"kind": "user", "item": i, "after_step": step, "turn": turn_of[i], "text": it["content"]})
        elif it["kind"] == "harness":
            out.append({"kind": "harness", "item": i, "after_step": step, "turn": turn_of[i], "type": "failed_result",
                        "steps": [step], "text": it["content"]})
    # Rule-flagged results, grouped.
    last = None
    for s in sorted((s for s in rules if s["type"] in ("tool_error", "failed_result") and s["source"] == "tool"),
                    key=lambda s: s["item"]):
        key = (s["type"], normalized(items[s["item"]]["content"]), turn_of[s["item"]])
        if last and last["key"] == key:
            last["steps"].append(s["step"])
            last["items"].append(s["item"])
            continue
        last = {"kind": "tool", "key": key, "type": s["type"], "item": s["item"], "items": [s["item"]],
                "steps": [s["step"]], "turn": turn_of[s["item"]], "text": items[s["item"]]["content"]}
        out.append(last)
    for e in out:
        e.pop("key", None)
        if "steps" in e:
            e["steps"] = sorted(set(e["steps"]))
    # One review per turn over the results the rules left alone.
    # Reads, edits, searches and listings cannot carry a hidden signal; the
    # output of a command and the result of any other tool can.
    by_turn = {}
    for i, it in enumerate(items):
        if it["kind"] == "tool_result" and i not in flagged and reviewed(it.get("tool")) and not inspecting(items, i):
            by_turn.setdefault(turn_of[i], []).append(i)
    for turn, idx in sorted(by_turn.items()):
        out.append({"kind": "review", "item": idx[-1], "turn": turn, "items": idx,
                    "steps": sorted({items[i]["step"] for i in idx if items[i].get("step")}),
                    # Only commands: the review question stays as it was checked.
                    "commands": all(items[i].get("tool") in COMMAND_OUTPUT for i in idx)})
    return sorted(out, key=lambda e: e["item"])


def inspecting(items, result_item):
    """True if the command behind a result only looks around (ls, cat, git status ...)."""
    call_id = items[result_item].get("call")
    for it in items[:result_item]:
        if it["kind"] == "action":
            for c in it.get("tool_calls", []):
                if c["id"] == call_id:
                    try:
                        cmd = json.loads(c.get("arguments") or "{}").get("command") or ""
                    except ValueError:
                        cmd = ""
                    return bool(INSPECT.search(cmd))
    return False


# ---- views ---------------------------------------------------------------

def shorten_paths(text, workdir):
    """Show paths inside the agent's working directory relative to it."""
    return text.replace(workdir.rstrip("/") + "/", "").replace(workdir.rstrip("/"), ".") if workdir else text

def call_line(c):
    args = c.get("arguments") or ""
    if c.get("name") in ("edit", "write"):
        try:
            a = json.loads(args)
            new = a.get("newString") or a.get("content") or ""
            return f"{c['name']} {a.get('path', '')}: {' '.join(new.split())[:160]}"
        except (ValueError, AttributeError):
            pass
    m = re.search(r'"(command|code|path|pattern|query)"\s*:\s*"((?:[^"\\]|\\.)*)"', args)
    shown = m.group(2) if m else args
    shown = shown.encode().decode("unicode_escape", "ignore") if "\\" in shown else shown
    return f"{c['name']} {' '.join(shown.split())[:140]}"


SUMMARY = re.compile(
    r"\d+ (passed|failed)|test result|error|fail|traceback|exception|game over|won|lost"
    r"|\b[GY.]{5}\b|higher|lower|correct|Exit code",
    re.I,
)


def first_line(text):
    """The most telling line of a result: a pass/fail/error line if any."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return "(no output)"
    telling = [line for line in lines if SUMMARY.search(line) and "Warning" not in line]
    return (telling[-1] if telling else lines[0])[:140]


NOISE = re.compile(r"Warning|warnings\.warn|asyncio_default_fixture_loop_scope|event loop scope")


def result_view(text, n=600):
    """Keep the start, the end and any error lines of a long result."""
    text = "\n".join(line for line in text.strip().splitlines() if not NOISE.search(line)).strip()
    if len(text) <= n:
        return text
    errors = [line for line in text.splitlines() if re.search(r"error|fail|traceback|exception|game over", line, re.I)][:6]
    head, tail = text[: n // 2], text[-n // 3 :]
    middle = ("\n…\n" + "\n".join(errors) + "\n…\n") if errors else f"\n…[{len(text) - n} chars cut]…\n"
    return head + middle + tail


def step_lines(items, lo_item, hi_item, full=False):
    """Numbered steps between two item positions; one line each unless full."""
    out = []
    results = {}
    for it in items[lo_item:hi_item]:
        if it["kind"] == "tool_result":
            results.setdefault(it.get("step"), []).append(it)
    for i in range(lo_item, hi_item):
        it = items[i]
        if it["kind"] == "checkpoint":
            out.append("[earlier context was compacted; the agent saw this summary]\n" + clip(it["content"], 1200 if full else 400))
        elif it["kind"] == "harness":
            out.append(f"HARNESS (opencode, not the user): {first_line(it['content'])}")
        elif it["kind"] == "action":
            n = it["step"]
            if it.get("failed"):
                out.append(f"STEP {n}: (no response)")
                continue
            calls = it.get("tool_calls") or []
            res = results.get(n, [])
            if not full:
                parts = [call_line(c) for c in calls]
                said = f'says "{" ".join(it["text"].split())[:200]}"' if it["text"].strip() and not calls else ""
                outcome = " | ".join(first_line(r["content"]) for r in res)
                out.append(f"STEP {n}: " + ("; ".join(parts) or said) + (f"  ->  {outcome}" if outcome else ""))
            else:
                block = [f"STEP {n}:"]
                if it["text"].strip():
                    block.append(f"  says: {clip(it['text'], 500)}")
                for c in calls:
                    block.append(f"  calls {c['name']}({clip(c.get('arguments') or '', 400)})")
                for r in res:
                    block.append(f"  result: {result_view(r['content'])}")
                out.append("\n".join(block))
    return "\n".join(out)


def user_messages(items, before_item):
    msgs, n = [], 0
    for it in items[:before_item]:
        if it["kind"] == "user":
            n += 1
            msgs.append(f"[U{n}] {clip(it['content'], 800)}")
    return msgs


def last_user_item(items, before_item):
    for i in range(before_item - 1, -1, -1):
        if items[i]["kind"] == "user":
            return i
    return 0
