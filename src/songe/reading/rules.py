"""Deterministic signals from tool results (ported from scripts/signals/rules.py).

Each tool result is already tied to the step that made the call, so the only
question is what kind of result it is:

- tool_error: the call itself was rejected (opencode's own error object, or
  the execute tool refusing code that is not JavaScript).
- failed_result: the call ran, and its output shows the work is wrong (non-zero
  exit code, failing tests, compiler errors, tracebacks, a lost game).
- success: the call checked the work and it passed (tests, builds, running the
  program, a won game). Only results that check something count; a file read
  or a search says nothing about quality.

Outcomes that are neither (Wordle marks, "higher"/"lower") are left to the
model, as are user messages. Harness messages (opencode's "hit the output
token limit") are failed results of the step before them.

A success is confirmed unless something later in the same task contradicts it:
a failed result of the same kind of check, or a user correction or retry
before the next request. Contradicted successes are not drift-set material.
"""

import json
import re

FAILURE = re.compile(
    r"\b[1-9]\d* failed\b|\bFAILED\b|^error(\[E\d+\])?:|Traceback \(most recent call last\)"
    r"|\bgame over\b|Out of guesses|\bnot ok \d+|\bpanicked at\b|\bAssertionError\b"
    r"|\"error\":\s*\"",
    re.M,
)
PASSED = re.compile(
    r"\b\d+ passed\b(?!.*\bfailed\b)|test result: ok\. [1-9]\d* passed|\ball tests passed\b"
    r"|^# pass [1-9]\d*$|You won!|\"status\": \"won\"|: correct\b",
    re.M,
)
# Commands that check the work rather than look around.
CHECK = re.compile(
    r"\bpytest\b|\bcargo (test|run|build)\b|\bmake\b|\bnode --test\b|\bnpm (run )?test\b"
    r"|\bgo test\b|\./test_bin\b|\bpython3? (-m )?[\w./-]+(\.py)?\b(?! *-m py_compile)"
    r"|wordle\.py guess|guess\.py \d",
)
EXIT = re.compile(r"Exit code: (\d+)\s*$")
EDITS = {"edit", "write", "patch", "multiedit", "apply_patch"}
GAME_MOVES = {"wordle.py", "guess.py"}


def opencode_error(body):
    try:
        obj, _ = json.JSONDecoder().raw_decode(body.lstrip())
    except ValueError:
        return None
    error = obj.get("error") if isinstance(obj, dict) else None
    return error if isinstance(error, dict) and "type" in error else None


def command_of(call):
    try:
        args = json.loads(call.get("arguments") or "{}")
    except ValueError:
        return ""
    return str(args.get("command") or args.get("code") or "")


def classify(tool, body, call=None):
    """(type, evidence) for one tool result, or None when it carries no signal."""
    error = opencode_error(body)
    if error:
        return "tool_error", error.get("message", "")[:300]
    if tool == "execute" and body.startswith(("SyntaxError:", "Invalid arguments for tool")):
        return "tool_error", body.splitlines()[0][:300]
    if tool not in ("shell", "execute"):
        return None
    code = EXIT.search(body.strip())
    failure = FAILURE.search(body)
    command = command_of(call or {})
    if code and code.group(1) == "1" and not failure and re.search(r"\|\s*grep\b[^|]*$|^\s*grep\b", command):
        return None  # grep found nothing; that is an answer, not a failure
    if (code and code.group(1) != "0") or failure:
        line = failure.group(0) if failure else f"Exit code: {code.group(1)}"
        return "failed_result", line
    if CHECK.search(command) and PASSED.search(body):
        return "success", PASSED.search(body).group(0)
    return None


def check_kind(call):
    m = CHECK.search(command_of(call or {}))
    if not m:
        return None
    text = m.group(0)
    for game in GAME_MOVES:
        if game.split(".")[0] in text:
            return game
    return text.split()[0]


def extract(items, user_kinds=None):
    """Signals from one timeline (see timeline.py).

    user_kinds maps the index of each user message (0 = the request) to its
    kind (correction, retry, approval, ...), from the model or from labels;
    it is only used to mark successes as contradicted.
    """
    signals, calls, step, user_index, changed = [], {}, 0, 0, False
    for i, it in enumerate(items):
        if it["kind"] == "action":
            step = it["step"]
            for c in it.get("tool_calls", []):
                calls[c["id"]] = c
                if c.get("name") in EDITS:
                    changed = True
        elif it["kind"] == "harness":
            signals.append(
                {"type": "failed_result", "source": "harness", "step": step, "item": i,
                 "evidence": it["content"][:300]}
            )
        elif it["kind"] == "tool_result":
            call = calls.get(it.get("call"), {})
            found = classify(it.get("tool"), it["content"], call)
            if found:
                kind, evidence = found
                if kind == "failed_result" and not changed and check_kind(call) not in (None, *GAME_MOVES):
                    # The agent has not changed anything yet: this reproduces
                    # the problem the user reported, it is not feedback on the
                    # agent's own work.
                    kind = "starting_problem"
                signals.append(
                    {"type": kind, "source": "tool", "step": it["step"], "item": i,
                     "tool": it.get("tool"), "check": check_kind(call), "evidence": evidence,
                     "user_turn": user_index}
                )
        elif it["kind"] == "user":
            user_index += 1
    # Confirmed or contradicted successes.
    for s in signals:
        if s["type"] != "success":
            continue
        later = [t for t in signals if t["item"] > s["item"] and t.get("user_turn") == s["user_turn"]]
        failed_again = any(t["type"] == "failed_result" and t.get("check") == s["check"] for t in later)
        # user_turn counts user messages seen so far; the reply to this turn's
        # work is the next one, which the caller stores at the same index.
        reply = (user_kinds or {}).get(s["user_turn"])
        s["confirmed"] = not failed_again and reply not in ("correction", "retry")
        s["contradicted_by"] = (
            "a later failure of the same check" if failed_again
            else f"the user's {reply}" if reply in ("correction", "retry") else None
        )
    return signals


# A rule the user states in their own words ("Never add comments unless I ask
# for them.") is a standing preference whatever happened before it: rules find
# it, not the model (Quick Dream puts it in the prompt verbatim).
RULE_START = re.compile(
    r"^(please\s+)?(always|never|don't|do not|prefer|avoid|only|use)\b", re.I
)
# A standing preference says it is meant to last: it starts with always/never/
# don't/prefer/avoid, contrasts with something ("…, not npm", "instead of"), or
# names its scope ("here", "from now on", "in this project", "every …"). A
# sentence like "Use the feedback from each guess." is an instruction for the
# task at hand, not a rule.
LASTING_START = re.compile(r"^(please\s+)?(always|never|don't|do not|prefer|avoid)\b", re.I)
LASTING = re.compile(
    r"\b(instead of|rather than|, not |not with|never|always|from now on|going forward|in this (project|repo|codebase)"
    r"|here\b|every (time|commit|change|file|project)|whenever|unless (i|you) )", re.I
)
# Words that point back at the conversation ("whatever that was"): such a
# sentence means nothing in a later task.
DEICTIC = re.compile(
    r"\b(whatever|that was|like that|it again|that again|do that|do this|this one|that one|those ones)\b", re.I
)
SENTENCES = re.compile(r"(?<=[.!?])\s+")


# "When writing search algorithms, always measure the runtime": a condition,
# then the rule.
CONDITIONAL = re.compile(r"^(please\s+)?(when|whenever|if)\b[^,.;]{3,80},\s*(please\s+)?(always|never|don't|do not)\b", re.I)


def rule_sentence(text):
    """The first sentence of the user's that is a standing rule: it starts like
    one, says it is meant to last (LASTING_START or LASTING), has at least four
    words, and does not point back at the conversation. Or None."""
    for sentence in SENTENCES.split(" ".join(str(text or "").split())):
        if DEICTIC.search(sentence) or not 12 <= len(sentence) <= 160:
            continue
        if CONDITIONAL.search(sentence) and len(sentence.split()) >= 5:
            return sentence
        # "Always run more." is short but plainly meant to last.
        words = len(sentence.split())
        if (
            RULE_START.search(sentence)
            and (words >= 4 or (words >= 3 and LASTING_START.search(sentence)))
            and (LASTING_START.search(sentence) or LASTING.search(sentence))
        ):
            return sentence
    return None


def stated_rules(items):
    """(item index, sentence) for each user message holding a standing rule.

    Only messages after the agent has done something (an action, or a
    checkpoint standing for earlier work): the opening message is the task
    statement, and its constraints ("Don't change the Bevy version") belong to
    that task, not feedback on the agent's work (live: one task prompt reused
    in 11 runs became the same "correction" six times)."""
    out = []
    started = False
    for i, it in enumerate(items):
        if it.get("kind") in ("action", "checkpoint"):
            started = True
        elif started and it.get("kind") == "user" and (sentence := rule_sentence(it.get("content"))):
            out.append((i, sentence))
    return out
