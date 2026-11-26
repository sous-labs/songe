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
    r"|\"error\":\s*\""
    # The shell's own errors: a missing command or Python module.
    r"|: command not found$|^\S*python[\d.]*: No module named\b",
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
# Tools that run a command (opencode's and Pi's is bash).
COMMANDS = {"shell", "execute", "bash"}
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
    if tool not in COMMANDS:
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
    r"\b(whatever|that was|like that|it again|that again|do that|do this|this one|that one|those ones)\b"
    # Ending on what it points at: "always start your reply with that" names
    # the thing in the sentence before it (#266).
    r"|\b(that|this|those|these)\W*$",
    re.I,
)
SENTENCES = re.compile(r"(?<=[.!?])\s+")


# "When writing search algorithms, always measure the runtime": a condition,
# then the rule.
CONDITIONAL = re.compile(r"^(please\s+)?(when|whenever|if)\b[^,.;]{3,80},\s*(please\s+)?(always|never|don't|do not)\b", re.I)


def unquoted(text):
    """A message without the quotes around the whole of it: `opencode run
    "..."` sends its message wrapped in them."""
    for left, right in (('"', '"'), ("'", "'"), ("\u201c", "\u201d")):
        if len(text) > 1 and text.startswith(left) and text.endswith(right):
            inner = text[1:-1]
            if left not in inner and right not in inner:
                return inner.strip()
    return text


def rule_sentence(text):
    """The first sentence of the user's that is a standing rule: it starts like
    one, says it is meant to last (LASTING_START or LASTING), has at least four
    words, and does not point back at the conversation. Or None."""
    for sentence in SENTENCES.split(unquoted(" ".join(str(text or "").split()))):
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


# A preference the user states about themself ("I avoid dairy.", "I'm allergic
# to peanuts.", "我不吃辣。") holds in every later conversation, wherever it is
# said: also in the opening message, which stated_rules leaves alone because a
# task's own constraints live there. Rules only find the candidates: a sentence
# in the first person with a word of taste, habit or need, that is not a
# question and does not point at something just shown. The model then judges
# each one where it stands (extract.stated_preferences): "I love the new
# layout" and "I prefer the second one" look the same to a pattern.
SELF = re.compile(
    r"^(?:personally,?\s+)?(?:i|i['\u2019]m|i['\u2019]ve|i['\u2019]d|my|as an?|being|due to my|because of my)\b", re.I
)
TASTE = re.compile(
    r"\b(?:prefer\w*|avoid\w*|dislike\w*|despise|detest|loathe|hate|love|enjoy|adore|like|fond of|fan of"
    r"|into|interest(?:ed)? in|favou?rite|rather|disdain|priority|aversion|allerg\w+|intoleran\w+|phobia"
    r"|afraid of|scared of|fear of|terrified|sensitive to|vegetarian|vegan|pescatarian|kosher|halal|gluten|lactose"
    r"|diet|against|refuse|stay away|steer clear|strict\w*|never|always|only|usually|i find"
    r"|can(?:['\u2019]t|not| not) (?:stand|eat|drink|tolerate|handle|bear)"
    r"|(?:don['\u2019]t|do not) (?:eat|drink|use|like|watch|read|listen|wear|enjoy|do|buy|own|believe|want))\b",
    re.I,
)
CJK = re.compile(r"[\u3400-\u9fff]")
TASTE_ZH = re.compile(
    r"喜欢|喜歡|讨厌|討厭|偏好|偏爱|爱吃|爱喝|爱看|爱用|不吃|不喝|不用|不看|不坐|过敏|過敏|受不了|忌口|吃素|素食"
    r"|害怕|恐高|习惯|習慣|从不|從不|从来不|總是|总是|一向|反感|倾向|傾向|只吃|只喝|只用|只看|只买|只坐|只住|只穿"
)
# A pick among things just offered, or a remark on what was just shown.
# How people say a habit of theirs has changed: "I've stopped taking the
# train; I cycle to work now", "These days I only read e-books", "我现在只喝茶".
# There may be no word of taste in it, and the sentence need not open with
# "I". A candidate like the others: the model judges it.
CHANGE = re.compile(
    r"\b(?:no longer|any ?more|these days|nowadays|from now on"
    r"|i(?:['\u2019]ve| have)? (?:switched|stopped|quit|given up|gave up|moved (?:on |over )?to|gone off|become|became|changed my mind|had enough of)"
    r"|i(?:['\u2019]d| would) rather)\b"
    r"|\bnow[.!]?$", re.I
)
CHANGE_ZH = re.compile(
    r"不再|戒了|戒掉|改用|改吃|改喝|改坐|现在只|現在只|现在都|現在都|现在不|現在不|以后都|以後都|以后不|以後不|以后只|以後只"
    r"|从现在起|從現在起|已经不|已經不"
)
SPEAKER = re.compile(r"(?<![.\w])i(?:['\u2019](?:m|ve|d|ll))?\b(?!\.)", re.I)
CHOICE = re.compile(
    r"\b(?:first|second|third|fourth|last|former|latter|previous|other)"
    r" (?:one|two|option|version|choice|suggestion|answer|draft|design|approach|example|idea)\b"
    r"|\boption [a-d1-9]\b|\b(?:this|that) (?:one|option|version|answer|draft|design|approach|way|idea|name|title)\b"
    r"|第[一二三四五六七八九十\d]+[个個种種]|[这這那][个個种種]|上面|刚才|剛才",
    re.I,
)
PARTS = re.compile(r"(?<=[.!?])\s+|(?<=[。！？；])\s*|\n+")
# "Correct the grammar: I enjoy ...", 'Mira: "I never ...': what follows such a
# colon is material handed over, or somebody's line, not the user speaking.
HANDED = re.compile(r"\uff1a|:(?=\s|[\"'\u201c\u2018\u300c\u300e]|$)")


def preference_sentences(text, limit=3):
    """The user's sentences that may state a lasting preference of their own,
    in their own words (at most `limit`, the first ones). Candidates only.

    "I avoid dairy, what can I cook tonight?" is a statement and then a
    question: the statement is the candidate. Chinese runs clauses together
    with commas ("我对花生过敏，推荐点别的"), so there the clause is."""
    out, whole, pos = [], unquoted(str(text or "").strip()), 0
    for part in PARTS.split(whole):
        at = whole.find(part, pos)
        pos = at + len(part) if at >= 0 else pos
        if HANDED.search(whole[: max(at, 0)]):
            break
        sentence = unquoted(" ".join(part.split()))
        chinese = bool(CJK.search(sentence))
        if chinese:
            pieces = re.split(r"[，,]\s*", HANDED.split(sentence)[0].rstrip("。！；.!;"))
        elif sentence.endswith("?"):
            pieces = re.split(r",\s+|;\s+|\s+[\u2014-]\s+", sentence)[:1]
        else:
            pieces = [sentence]
        for piece in pieces:
            piece = piece.strip()
            if not (3 if chinese else 12) <= len(piece) <= 200:
                continue
            if piece[0] in "\"'\u201c\u2018\u300c\u300e":
                continue  # inside a quotation that goes on: somebody's line, not the user speaking
            if piece.endswith(("?", "\uff1f", "吗", "呢")) or DEICTIC.search(piece) or CHOICE.search(piece):
                continue
            if chinese:
                found = "我" in piece and (TASTE_ZH.search(piece) or CHANGE_ZH.search(piece))
            else:
                found = (SELF.search(piece) and TASTE.search(piece)) or (
                    (SELF.search(piece) or SPEAKER.search(piece)) and CHANGE.search(piece)
                )
            if found:
                out.append(piece)
    return out[:limit]


# Words an instruction is built from whatever it is about ("Always include
# ...", "Keep answers ...", "Use ... instead of ..."): every note has them,
# so they do not show whose wish a sentence is.
FRAME = frozenset(
    """a an the to of and or in on for with by from at as is are be it its this that these those there here
    i me my we our you your they them their he she his her not no nor do does did don dont doesn done
    will would can could should may might must am was were been being has have had if when then than so
    such what which who how why where all each every any some both either only just very more most less
    much many few own same other another into over under up down out off about again once also but too
    always never ever please now use uses using used avoid keep make give provide include write answer
    answers reply replies respond response responses say tell show put add ensure sure instead rather
    without want wants need needs like user users assistant agent ai model request requested asked ask
    exactly thing things time times""".split()
)


def voiced(text, said):
    """How much of a sentence the model worded is what the user wrote: the
    share of its content words (the words every instruction is built from
    left out, endings ignored) that are in `said`, the user's own words.

    1.0 when words cannot show it: `said` is not written in Latin letters
    ("请用中文回答" asks for Chinese without the word), or the sentence has no
    content word."""
    import unicodedata

    from ..quick import _tokens, stem

    if any(ch.isalpha() and not unicodedata.name(ch, "").startswith("LATIN") for ch in str(said)):
        return 1.0
    content = {stem(w) for w in _tokens(text) if w not in FRAME and len(w) > 1} - FRAME
    theirs = {stem(w) for w in _tokens(said)}
    found = sum(
        any(w == t or (min(len(w), len(t)) >= 4 and (w.startswith(t) or t.startswith(w))) for t in theirs)
        for w in content
    )
    return found / len(content) if content else 1.0


def stated_sentence(signal):
    """The sentence of the user's that a signal keeps word for word: a
    preference of their own (judged lasting by the model when it was read;
    the pattern is checked again here, so a looser past capture does not
    stay), else a standing rule. None when the signal holds neither."""
    said = signal.get("preference")
    if said:
        return said if said in preference_sentences(said) else None
    return rule_sentence(signal.get("text"))
