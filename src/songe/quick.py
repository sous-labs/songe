"""Quick Dream: lessons in the system prompt.

Reading (in idle moments) writes a lesson for each signal. A Quick Dream
picks the ones that suit a prompt and publishes them; the proxy adds them to
the system message of each new session (see proxy.LessonPins). Deep Dream
trains lessons into the weights; once a Deep Dream that trained a lesson is
adopted, the lesson leaves the prompt (absorb).

Which lessons: those read as applying to any task ("always": standing
preferences, tool facts) and every lesson from the user's own feedback
(corrections, retries, approvals), whatever its label: the generality label
is noisy (27 of 43 on the answer key) and preferences come from the user.
Skill lessons from tool results wait for Deep Dream. Near-duplicates are one
lesson, and so are notes reading found to be one wish said in other words
(reading.fold_same; the lesson then keeps each wording); the most recurrent,
then the most recent, come first; at most config.prompt_lessons. A lesson the
user removed never returns.

State per learner (store scoped to the model), key "prompt_lessons":
{"version": n, "updated_at": t, "lessons": [Lesson], "removed": [text]}
Lesson: {"id", "text", "signals": [signal ids], "added_at", "count", "rule",
"joined" (only when the text holds several wordings of one wish)}
"""

import re
import time
import unicodedata

from .reading import store as reading
from .reading.rules import (  # noqa: F401
    DEICTIC,
    RULE_START,
    SENTENCES,
    rule_sentence,
    stated_sentence,
)
from .steps import digest

KEY = "prompt_lessons"
USER = {"correction", "retry", "approval"}
LESSON_CHARS = 300
SIMILAR = 0.5  # word overlap (Jaccard) at which two lessons are one
WORDS = re.compile(r"[\u3400-\u9fff\uf900-\ufaff\u3040-\u30ff]|[^\W_]+")
COMMON = {
    "the", "a", "an", "to", "of", "and", "or", "in", "on", "for", "with", "is",
    "it", "that", "this", "be", "as", "when", "if", "you", "your", "use", "by",
    "not", "do", "are", "from", "at", "before", "after", "any", "its", "e", "g",
}

BLOCK = "Lessons from earlier sessions with this user. Follow them:"
# The request at hand wins over a lesson. Said after the list, next to the
# request: said in the heading only ("Follow them unless the current request
# says otherwise"), a small model kept to the lesson when the user asked, in
# so many words, for the opposite just this once.
EXCEPTIONS = (
    "People make exceptions. When the current request asks for an exception to"
    " one of these, give the user what they ask for this time. In every other"
    " request, follow them."
)


def _tokens(text):
    """Unicode word tokens (splitting CJK ideographs/kana so no-space scripts
    still match, and stripping combining marks/tatweel so Arabic words stay intact)."""
    norm = unicodedata.normalize("NFKC", str(text).lower())
    cleaned = "".join(ch for ch in norm if unicodedata.category(ch) != "Mn" and ch != "\u0640")
    return WORDS.findall(cleaned)


def stem(word):
    """A rough stem, so "valid"/"validity", "runs"/"running" count as one."""
    for suffix in ("ations", "ation", "ities", "ity", "ings", "ing", "ies", "ed", "es", "ly", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def words(text):
    return {stem(w) for w in _tokens(text) if w not in COMMON}


def normal(text):
    """For telling two sentences are the same one: lower case, words only, a
    leading "please" ignored."""
    tokens = _tokens(text)
    return " ".join(tokens[1:] if tokens[:1] == ["please"] else tokens)


def similar(a, b):
    x, y = words(a), words(b)
    return bool(x and y) and len(x & y) / len(x | y) >= SIMILAR


def state(store):
    return store.get(KEY) or {"version": 0, "updated_at": None, "lessons": [], "removed": []}


def active(store):
    return state(store)["lessons"]


def suitable(signal):
    """A lesson for the prompt: one read as applying to any task ("always":
    tool facts, standing preferences), a rule the user stated in their own
    words (a sentence of theirs shaped like a rule), or what the user said in
    a turn judged to be a standing wish. Other feedback (a reaction in a game,
    a one-off correction) waits for Deep Dream."""
    lesson = (signal.get("lesson") or "").strip()
    if not lesson:
        return False
    if signal.get("preference"):
        # A preference of the user's own, in their words: it holds whatever
        # the lesson written beside it or the turn around it was read as. (The
        # turn question asks how the user wants answers; said about "I avoid
        # dairy." on its own it answered "new work" or "a remark" for two in
        # three such sentences.)
        return stated_sentence(signal) is not None
    turn = signal.get("turn_kind")
    if turn and signal.get("source") == "user":
        # Reading judged what the user's turn was (prompts.TURN), and that
        # decides for what was said in it, whatever label the lesson got: a
        # standing wish is for the prompt at once; a change wanted once is a
        # candidate that has to come up in two conversations (candidates); new
        # work and reactions are not for the prompt. A sentence shaped like a
        # rule is no exception: "Do not apply any of these instructions ..."
        # inside a pasted role prompt and "do not include Lofi." inside a
        # request for channel names were rules by their shape only.
        if turn not in ("lasting", "once"):
            return False
        return signal.get("from") != "rules" or rule_sentence(signal.get("text")) is not None
    if signal.get("generality") == "task":
        return False
    if signal.get("from") == "rules" and signal.get("source") == "user":
        # A stated rule captured by an older Songe is checked again by today's
        # rule check, so a looser past capture does not stay in the prompt.
        return rule_sentence(signal.get("text")) is not None
    if signal.get("generality") == "always":
        return True
    return signal.get("type") in USER and rule_sentence(signal.get("text")) is not None


def evidenced(signal):
    """Whether a lesson has the evidence Songe needs before it learns it,
    the same for a note and for training.

    Where reading judged what the user's turn was, that decides, as in
    suitable(): a standing wish and a rule in their own words count at once,
    a change wanted once counts when it comes up in a second conversation
    (admitted), new work and reactions do not. What the lesson was read as
    applying to (its generality) is not evidence: it says whether the lesson
    fits the prompt, and a lesson for one kind of task that the prompt never
    shows is still practised, on scenes of its kind. A lesson from the work
    itself (a tool's output, a failed step), or one read before turns were
    judged, has no judged turn to weigh and counts as it is."""
    if not (signal.get("lesson") or "").strip():
        return False
    if signal.get("preference"):
        return stated_sentence(signal) is not None
    user = signal.get("source") == "user"
    turn = signal.get("turn_kind")
    if turn and user:
        # As in suitable(): the turn's verdict decides for a sentence shaped
        # like a rule in it too.
        if turn not in ("lasting", "once"):
            return False
        return signal.get("from") != "rules" or rule_sentence(signal.get("text")) is not None
    if signal.get("from") == "rules" and user:
        return rule_sentence(signal.get("text")) is not None
    return True


# Folder names too common to stand for one project.
COMMON_FOLDERS = {"home", "root", "user", "users", "src", "app", "code", "work", "dev", "tmp", "temp", "project",
                  "projects", "repo", "repos", "desktop", "documents", "downloads", "workspace", "main", "test"}


def general(text, project):
    """A note with its project's own name said so it holds in every project:
    "Put code under src/billing_api/." from a session in billing-api becomes
    "Put code under src/<package>/." (reported by Jack in a test on 11-24:
    "Use src/notes_app/ in all file paths." sent a question about another
    repository into notes_app, and 4 of 8 answers there got worse)."""
    name = str(project or "")
    if len(name) < 4 or name.lower() in COMMON_FOLDERS:
        return text
    forms = {name: "<project>", name.replace("-", "_"): "<package>", name.replace("_", "-"): "<project>"}
    for form in sorted(forms, key=len, reverse=True):
        text = re.sub(rf"(?<![\w-]){re.escape(form)}(?![\w-])", forms[form], text, flags=re.I)
    return text


# A value the user gave on the command line of their tool ("make test MODULE=auth").
SETTING = re.compile(r"\b([A-Z][A-Z0-9_]+)=([^\s,;'\"`)]+)")


def unset(text):
    """The sentence with each NAME=value said as NAME=<name>."""
    return SETTING.sub(lambda m: f"{m[1]}=<{m[1].lower()}>", text)


def wording(signal):
    """A signal's note for the prompt, and whether it is the user's own
    sentence: a sentence they stated, else the short form reading wrote for
    the prompt (signals read before it existed fall back to the lesson,
    trimmed by tidy). Its project's own name is said for any project
    (general)."""
    rule = stated_sentence(signal) if signal.get("type") in USER else None
    return general(rule or short_of(signal) or signal["lesson"].strip(), signal.get("project")), bool(rule)


def joined(signals):
    """The note of signals reading found to be one wish said in several ways
    (reading.fold_same): each wording once, the most recent first, the user's
    own sentences as they are. No wording is chosen over another: asked which
    of two notes says more, the served model could not tell (Qwen3.5-4B: 29
    of 42 pairs right), so "Use metric units." does not replace "Use Celsius,
    not Fahrenheit.", it stands beside it in one place. Returns (the text,
    whether any wording is the user's own, how many wordings)."""
    parts, own = [], False
    for s in sorted(signals, key=lambda s: s.get("revision") or 0, reverse=True):
        text, rule = wording(s)
        text = " ".join(text.split()) if rule else tidy(text)
        if not any(normal(text) == normal(part) or similar(text, part) for part in parts):
            parts.append(text if text[-1:] in ".!?。！？" else text + ".")
            own = own or rule
    return " ".join(parts)[:LESSON_CHARS].rstrip(), own, len(parts)


def admitted(store, min_sessions=None, without=(), fits=suitable):
    """The lessons that have the evidence to be learned, as clusters of
    near-identical ones ({"text", "signals", "newest", "rule"}).

    One evidence bar for both ways of learning: training is harder to undo
    than a note, so its bar is not lower. Quick Dream puts in the prompt the
    ones that also fit it (fits=suitable, see candidates); Deep Dream
    practises every one with the evidence (fits=evidenced,
    practice.store.lesson_signals) and no other. A correction the user
    wanted once waits until the same wish comes up in another conversation,
    in practice as in the prompt. (Live, Qwen3.5-4B: practice took every
    signal with a lesson. "Just the code.", judged a one-off, had been read
    as "Always include the exact file pattern requested by the user.", got 24
    scenes and was trained.)

    min_sessions: quick_min_sessions (None: this model's saved setting).
    without: signal ids left out before grouping (the absorbed ones, for the
    prompt)."""
    if min_sessions is None:
        min_sessions = saved_min_sessions(store)
    signals = [s for s in reading.signals(store) if fits(s) and s["id"] not in without]
    removed = state(store)["removed"]
    # One wish, one place: notes reading found to repeat an earlier note
    # ({signal: the first signal that noted the wish}) are one cluster with
    # it, however differently they are worded.
    same = store.get(reading.SAME) or {}
    by_id = {s["id"]: s for s in signals}
    clusters = []
    # Newest first: a cluster is named by its most recent wording.
    for s in sorted(signals, key=lambda s: s.get("revision") or 0, reverse=True):
        text, rule = wording(s)
        if any(similar(text, r) for r in removed):
            continue
        first = same.get(s["id"], s["id"])
        home = next((c for c in clusters if c["first"] == first or first in c["signals"]), None)
        if home:
            pass
        elif rule:
            # A rule the user stated groups only with the same sentence of
            # theirs, never with lessons that merely look alike. The same
            # sentence with another value in it ("make test MODULE=auth",
            # "make test MODULE=parser") is the same rule: one place, not one
            # per value (Jack's test, 11-24; Qwen3.8-27B on main put the test
            # command in 3 to 5 of 10 places, one per module).
            home = next((c for c in clusters if c.get("rule") and normal(unset(c["text"])) == normal(unset(text))), None)
        else:
            home = next((c for c in clusters if not c.get("rule") and similar(text, c["text"])), None)
        if home:
            home["signals"].append(s["id"])
        else:
            clusters.append({"text": text, "signals": [s["id"]], "newest": s.get("revision") or 0,
                             "rule": rule, "first": first})
    for c in clusters:
        if c.get("rule") and len({normal(wording(by_id[i])[0]) for i in c["signals"]}) > 1:
            # Said with several values: the value is not part of the rule.
            c["text"] = unset(c["text"])
        text, own, wordings = joined([by_id[i] for i in c["signals"]])
        if wordings > 1 and any(i in same for i in c["signals"]):
            c.update(text=text, rule=own, joined=True)
    # A lesson the user removed does not return under another of its wordings.
    clusters = [c for c in clusters if not (c.get("joined") and any(similar(c["text"], r) for r in removed))]
    # A tool fact or standing preference read as "always" enters the prompt
    # once it has come up in two conversations; a rule the user stated, at once.
    sessions = {s["id"]: s.get("session") for s in signals}
    # A change the user wanted once ("make it shorter") is a lesson for the
    # prompt only when the same wish came up in another conversation too, also
    # when it was worded like a rule ("use comma instead of period.").
    once = {s["id"] for s in signals if s.get("turn_kind") == "once" and s.get("source") == "user"}

    def enough(c):
        need = max(min_sessions, 2) if all(i in once for i in c["signals"]) else min_sessions
        return len({sessions.get(i) for i in c["signals"]}) >= need

    return [c for c in clusters if (c.get("rule") and not all(i in once for i in c["signals"])) or enough(c)]


def saved_min_sessions(store):
    """quick_min_sessions as saved for this store's model (the default when
    nothing is saved or it cannot be read)."""
    from .config import Config

    try:
        config, _ = Config.load_with_keys(store.root)
        if store.model_key:
            from .models.registry import model_config

            config = model_config(config, store, store.model_key)
        return config.quick_min_sessions
    except Exception:  # noqa: BLE001 - a broken file must not stop learning
        return Config.quick_min_sessions


def candidates(store, limit, min_sessions=2):
    """Clusters of near-identical lessons for the prompt, best first: the
    admitted ones (admitted) that no adopted Deep Dream has absorbed."""
    clusters = admitted(store, min_sessions, set(store.get("absorbed_signals") or []))
    clusters.sort(key=lambda c: (-len(c["signals"]), -c["newest"]))
    return clusters[:limit]


CONSOLIDATE = """These are lessons a coding assistant learned from one user's feedback and from its own tool errors (numbered, most frequent first):
{lessons}

Write the short list of instructions to keep in its system prompt for every future task with this user.
- Merge lessons that mean the same thing into one instruction (list every number it came from).
- Leave out lessons that only fit one particular kind of task, game, project or file (for example how to play a guessing game, or a fix in one file).
- Keep standing preferences of the user and facts about the tools.
- When the user stated a rule, keep it as a direct rule in their terms, with their "always" or "never" (for example "Use pnpm for packages, never npm."). Do not soften it into advice to check or confirm.
- Write each instruction as one short imperative sentence of at most 15 words. Do not start with "Always verify". No file names, values or examples.
- At most {limit} instructions, most important first. If none qualify, return an empty list.
Answer with JSON only."""

CONSOLIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "lessons": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "from": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["text", "from"],
            },
        }
    },
    "required": ["lessons"],
}
# About 300 tokens for the whole list, on every request. At 600 characters ten
# sentences in a user's own words (about 60 to 90 characters each) did not
# fit: on average 7.9 of 10 did.
PROMPT_CHARS = 1200
# The model repeating its instruction instead of writing one ("Merge lessons
# 1, 2, 4 into one instruction: …").
ECHO = re.compile(r"\blessons?\b|\b(merge|combine)\b|\b\d+\s*,\s*\d+", re.I)
CANDIDATES = 3  # candidates given to consolidation, per lesson kept


async def consolidate(reader, clusters, limit):
    """The served model merges same-meaning lessons, drops task-specific ones
    and shortens the rest: [{"text", "signals"}], or None when its answer is
    unusable (the caller falls back to the clusters as they are)."""
    rules = [{"text": c["text"], "signals": c["signals"], "rule": True} for c in clusters if c.get("rule")]
    clusters = [c for c in clusters if not c.get("rule")]
    if not clusters:
        return rules  # the user's own rules are kept as they are, not rewritten
    numbered = "\n".join(f"{i + 1}. {c['text'][:LESSON_CHARS]}" for i, c in enumerate(clusters))
    answer = await reader.ask(
        CONSOLIDATE.format(lessons=numbered, limit=limit), CONSOLIDATE_SCHEMA, "consolidate",
        max_tokens=1500,
    )
    items = (answer or {}).get("lessons") if isinstance(answer, dict) else None
    if not isinstance(items, list):
        return None
    merged = []
    for item in items[:limit]:
        text = str(item.get("text") or "").strip().strip("-• ").strip()
        sources = [n - 1 for n in item.get("from") or [] if isinstance(n, int) and 0 < n <= len(clusters)]
        if not text or not sources or len(text) > 200 or ECHO.search(text):
            continue  # empty, unsourced, or the instruction echoed back
        signals = [sid for n in dict.fromkeys(sources) for sid in clusters[n]["signals"]]
        merged.append({"text": text, "signals": signals})
    return rules + merged


REVIEW = """A coding assistant keeps a short list of instructions in its system prompt for every future task with one user.

Instructions already kept:
{kept}

New instruction:
{text}

Answer two questions about the new instruction:
- scope: "general" if it helps on most of this user's coding tasks (a standing preference, how to use the tools), "one_kind" if it only fits one particular kind of task, game, project or file.
- covered_by: the number of a kept instruction that already says the same thing, or 0 if none does.
Answer with JSON only."""

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "scope": {"type": "string", "enum": ["general", "one_kind"]},
        "covered_by": {"type": "integer"},
    },
    "required": ["scope", "covered_by"],
}


async def review(reader, merged):
    """One small question per lesson, which a small model answers more
    reliably than one large rewrite: drop lessons that fit only one kind of
    task, and fold a lesson that says the same as a kept one into it."""
    kept = []
    for item in merged:
        listing = "\n".join(f"{i + 1}. {k['text']}" for i, k in enumerate(kept)) or "(none yet)"
        answer = await reader.ask(
            REVIEW.format(kept=listing, text=item["text"]), REVIEW_SCHEMA, "review_lesson",
            max_tokens=200,
        )
        if not isinstance(answer, dict):
            kept.append(item)  # an unusable answer keeps the lesson
            continue
        if answer.get("scope") == "one_kind":
            continue
        k = answer.get("covered_by")
        if isinstance(k, int) and 0 < k <= len(kept):
            kept[k - 1] = kept[k - 1] | {
                "signals": list(dict.fromkeys(kept[k - 1]["signals"] + item["signals"]))
            }
            continue
        kept.append(item)
    # Merging can make a later lesson the most recurrent; order is stable.
    return sorted(kept, key=lambda k: -len(k["signals"]))


PADDING = [
    # The model's padding, replaced by a plain verb so the lesson stays a sentence.
    (re.compile(r"^always\s+(verify|confirm|check)(\s+that)?\b", re.I), "Check\\2"),
    (re.compile(r"^always\s+(ensure|make sure)(\s+that)?\b", re.I), "Make sure"),
    (re.compile(r"^(verify|confirm)(\s+that)?\b", re.I), "Check\\2"),
    (re.compile(r"^ensure(\s+that)?\b", re.I), "Make sure"),
]
SHORT = 90  # characters a non-rule lesson is trimmed towards
QUALIFIERS = (", including", ", especially", ", such as", ", e.g.", " (e.g.", ", for example", ", which", ", before", ", and ")
TAILS = (" using ", " before ", " when ", " by ", " so that ", " without ")


def shorten(text):
    """A long lesson trimmed to its core, without a model: the first sentence,
    the part before a semicolon, then a trailing qualifier dropped (", including
    …", " using …"). Never below a few words; stable when applied again."""
    first = SENTENCES.split(text.strip())[0]
    text = first.split(";")[0].strip()
    if len(text) > SHORT:
        cut = min((i for q in QUALIFIERS if (i := text.find(q)) >= 25), default=-1)
        if cut > 0:
            text = text[:cut]
    if len(text) > SHORT:
        cut = min((i for t in TAILS if (i := text.find(t, 25)) > 0), default=-1)
        if cut > 0:
            text = text[:cut]
    text = text.rstrip(" ,.:-")
    if len(text.split()) < 6:
        # A cut that leaves a fragment ("Check that the program entry point")
        # is worse than a long lesson: keep the whole first sentence.
        text = first.rstrip(" ,.:-")
    # A user's own sentence can end as a question ("Can you drop the em dashes?").
    return text if text.endswith(("?", "!")) else text + "."


def tidy(text):
    """The model's padding replaced by a plain verb ("Always verify X" ->
    "Check X") and the lesson shortened to its core; stable when applied
    again, so published lessons can be tidied. Not for the user's own rules."""
    text = " ".join(text.split())
    for pattern, verb in PADDING:
        if pattern.search(text):
            text = pattern.sub(verb, text, count=1)
            break
    return shorten(text)


def user_rule(by_id, signal_ids):
    """The user's own rule when a lesson comes from their feedback: the one
    sentence that is shaped like a rule ("Never add comments unless I ask.",
    "Always use pnpm here, never npm."), not a complaint or the whole message.
    The most recent such sentence, or None."""
    for sid in signal_ids:
        s = by_id.get(sid) or {}
        if s.get("type") in USER and (sentence := stated_sentence(s)):
            return sentence
    return None


def short_of(signal):
    """The short form reading wrote for the prompt, if usable."""
    short = " ".join(str(signal.get("short") or "").split())
    return short if 12 <= len(short) <= 140 else None


def split(store, clusters):
    """Published lessons keep their wording: a candidate that shares a signal
    with one (or says the same) adds its signals to it. Returns (published,
    with new signals added; the clusters that are new)."""
    by_id = {s["id"]: s for s in reading.signals(store)}

    def still_belongs(lesson):
        # Published under looser rules, or a model echo: it leaves the prompt.
        if ECHO.search(lesson["text"]):
            return False
        if lesson.get("rule") and not lesson.get("joined"):
            # The user's own sentence: a standing rule, or a preference of
            # theirs that one of its signals still holds.
            return rule_sentence(lesson["text"]) == lesson["text"] or any(
                stated_sentence(by_id[sid]) == lesson["text"] for sid in lesson["signals"] if sid in by_id
            )
        return any(suitable(by_id[sid]) for sid in lesson["signals"] if sid in by_id)

    published = [dict(lesson, signals=list(lesson["signals"])) for lesson in active(store) if still_belongs(lesson)]
    fresh = []
    for c in clusters:
        shares = [lesson for lesson in published if set(lesson["signals"]) & set(c["signals"])]
        home = shares[0] if shares else next(
            (lesson for lesson in published
             if (normal(c["text"]) == normal(lesson["text"]) if c.get("rule")
                 else not lesson.get("rule") and similar(c["text"], lesson["text"]))), None)
        for other in shares[1:]:
            # Published apart, and found to be one wish since: one lesson.
            published.remove(other)
            home["signals"] += other["signals"]
        if home:
            home["signals"] = list(dict.fromkeys(home["signals"] + c["signals"]))
        else:
            fresh.append(c)
    return published, fresh


def quick_dream(store, config, now=None, merged=None, kept=None):
    """Publish the prompt lessons: kept (published ones, wording unchanged, from
    split) plus merged (the new ones, from consolidate; the candidate clusters
    as they are when neither is given). The most recurrent first, at most
    prompt_lessons, within PROMPT_CHARS. Returns {"lessons", "added", ...}."""
    now = time.time() if now is None else now
    current = state(store)
    if merged is None and kept is None:
        kept, merged = split(store, candidates(store, config.prompt_lessons * CANDIDATES, config.quick_min_sessions))
    by_id = {s["id"]: s for s in reading.signals(store)}
    same = store.get(reading.SAME) or {}
    pool = []
    for lesson in kept or []:
        said = [by_id[sid] for sid in lesson["signals"] if sid in by_id]
        if said and (lesson.get("joined") or any(s["id"] in same for s in said)):
            # One wish noted in several ways: the place keeps each wording.
            text, own, wordings = joined(said)
            lesson = {k: v for k, v in lesson.items() if k != "joined"} | {"rule": own}
            lesson |= {"joined": True} if wordings > 1 else {}
        elif lesson.get("rule"):
            text = lesson["text"]
        else:
            # Its newest short form, once reading has written one (wording then
            # stays stable); otherwise the published text, tidied.
            shorts = sorted(
                (by_id[sid] for sid in lesson["signals"] if sid in by_id and short_of(by_id[sid])),
                key=lambda s: s.get("revision") or 0, reverse=True,
            )
            text = tidy(short_of(shorts[0]) if shorts else lesson["text"])
        pool.append(lesson | {"text": text, "count": len(lesson["signals"])})
    for c in merged or []:
        if c.get("joined"):
            pool.append({"text": c["text"], "signals": c["signals"], "count": len(c["signals"]),
                         "rule": bool(c.get("rule")), "joined": True})
            continue
        rule = c["text"] if c.get("rule") else user_rule(by_id, c["signals"])
        text = rule or tidy(c["text"])
        pool.append({"text": text[:LESSON_CHARS].rstrip(), "signals": c["signals"],
                     "count": len(c["signals"]), "rule": bool(rule)})
    # One lesson per sentence: the same text twice is one lesson.
    unique = {}
    for lesson in pool:
        key = normal(lesson["text"]) or " ".join(lesson["text"].split()).lower()
        if key in unique:
            first = unique[key]
            first["signals"] = list(dict.fromkeys(first["signals"] + lesson["signals"]))
            first["count"] = len(first["signals"])
            first["rule"] = first.get("rule") or lesson.get("rule")
        else:
            unique[key] = dict(lesson)
    pool = list(unique.values())
    # The user's own rules first, then the most recurrent, before the cut;
    # published ones win a tie.
    pool.sort(key=lambda lesson: (not lesson.get("rule"), -lesson["count"]))
    lessons, used, taken = [], 0, set()
    for c in pool[: config.prompt_lessons]:
        text = c["text"]
        if used + len(text) > PROMPT_CHARS and lessons:
            continue
        used += len(text)
        old = c if c.get("id") else next((lesson for lesson in current["lessons"] if lesson["id"] not in taken
                                         and (similar(text, lesson["text"])
                                              or set(lesson["signals"]) & set(c["signals"]))), None)
        if old:
            taken.add(old["id"])
        lessons.append(
            {
                "id": old["id"] if old else digest([text])[:16],
                "text": text,
                "signals": c["signals"],
                "count": len(c["signals"]),
                "added_at": old["added_at"] if old else now,
                "rule": bool(c.get("rule")),
            }
            | ({"joined": True} if c.get("joined") else {})
        )
    added = sum(1 for lesson in lessons if lesson["added_at"] == now)
    before = {lesson["id"]: lesson["text"] for lesson in current["lessons"]}
    changes = {
        "added": [lesson["text"] for lesson in lessons if lesson["added_at"] == now],
        "reworded": [
            {"before": before[lesson["id"]], "after": lesson["text"]}
            for lesson in lessons
            if lesson["id"] in before and before[lesson["id"]] != lesson["text"]
        ],
        "dropped": [text for i, text in before.items() if i not in {lesson["id"] for lesson in lessons}],
    }
    reworded, dropped = len(changes["reworded"]), len(changes["dropped"])
    changed = [lesson["text"] for lesson in lessons] != [lesson["text"] for lesson in current["lessons"]]
    published = current | {
        "lessons": lessons,
        "version": current["version"] + (1 if changed else 0),
        "updated_at": now if changed else current["updated_at"],
    }
    store.set(KEY, published)
    return {"lessons": len(lessons), "added": added, "reworded": reworded,
            "dropped": dropped, "version": published["version"], "changes": changes}


HISTORY = "quick_history"
HISTORY_LIMIT = 50


def record(store, result, at, examples=None, **run):
    """Every finished Quick Dream joins the Dream history, also one that
    added nothing or could not read (live: a run read 5 conversations, found
    9 moments, added nothing, and left no trace). run: read, found, error
    (unreadable, no_model). Returns the entry."""
    history = store.get(HISTORY) or []
    entry = {
        "id": f"q-{(history[-1]['n'] if history else 0) + 1:04d}",
        "n": (history[-1]["n"] if history else 0) + 1,
        "at": at,
        "lessons": result["lessons"],
        **result["changes"],
        # Before/after on the conversation where each new preference came from.
        "examples": examples or [],
        **{k: v for k, v in run.items() if v is not None},
    }
    store.set(HISTORY, (history + [entry])[-HISTORY_LIMIT:])
    return entry


def note_run(store, entry_id, **run):
    """What the run read and found (or why it failed), on its entry."""
    history = store.get(HISTORY) or []
    for entry in history:
        if entry["id"] == entry_id:
            entry.update({k: v for k, v in run.items() if v is not None})
    store.set(HISTORY, history)


def remove(store, lesson_id):
    """The user removed a lesson: it leaves the prompt and never returns."""
    current = state(store)
    gone = [lesson for lesson in current["lessons"] if lesson["id"] == lesson_id]
    if not gone:
        raise ValueError("Unknown lesson")
    store.set(
        KEY,
        current
        | {
            "lessons": [lesson for lesson in current["lessons"] if lesson["id"] != lesson_id],
            "removed": current["removed"] + [gone[0]["text"]],
            "version": current["version"] + 1,
            "updated_at": time.time(),
        },
    )


def undo(store, entry_id):
    """Undo a Quick Dream: the preferences it added that are still in the
    prompt leave it (as removed: they do not come back), and its history
    entry says so. Returns how many left."""
    history = store.get(HISTORY) or []
    entry = next((e for e in history if e["id"] == entry_id), None)
    if entry is None:
        raise ValueError("Unknown Quick Dream")
    added = {normal(t) or " ".join(str(t).split()).lower() for t in entry.get("added") or []}
    current = state(store)
    gone = [
        lesson
        for lesson in current["lessons"]
        if (normal(lesson["text"]) or " ".join(str(lesson["text"]).split()).lower()) in added
    ]
    if gone:
        store.set(KEY, current | {
            "lessons": [lesson for lesson in current["lessons"] if lesson not in gone],
            "removed": current["removed"] + [lesson["text"] for lesson in gone],
            "version": current["version"] + 1,
            "updated_at": time.time(),
        })
    entry["undone"] = time.time()
    store.set(HISTORY, history)
    return len(gone)


def absorb(store, trained_signals):
    """An adopted Deep Dream trained these signals: their lessons leave the
    prompt (they are in the weights now). They are not 'removed': a later
    correction can bring the lesson back."""
    trained = set(trained_signals)
    current = state(store)
    kept = [lesson for lesson in current["lessons"] if not set(lesson["signals"]) <= trained]
    if len(kept) == len(current["lessons"]):
        return 0
    # Signals already absorbed stay out of later Quick Dreams.
    store.set("absorbed_signals", sorted(set(store.get("absorbed_signals") or []) | trained))
    store.set(KEY, current | {"lessons": kept, "version": current["version"] + 1, "updated_at": time.time()})
    return len(current["lessons"]) - len(kept)


def waiting(store, limit=50):
    """Lessons waiting for a Deep Dream: read lessons that are not in the
    prompt and not trained yet (skills, mostly), near-duplicates as one, most
    recurrent first. Returns {"total", "lessons": [{"text", "count"}]}."""
    in_prompt = {sid for lesson in active(store) for sid in lesson["signals"]}
    trained = set(store.get("trained_signals") or []) | set(store.get("absorbed_signals") or [])
    groups = []
    for s in sorted(reading.signals(store), key=lambda s: s.get("revision") or 0, reverse=True):
        text = (s.get("lesson") or "").strip()
        if not text or s.get("generality") == "task" or s["id"] in in_prompt or s["id"] in trained:
            continue
        home = next((g for g in groups if similar(text, g["text"])), None)
        if home:
            home["count"] += 1
        else:
            groups.append({"text": text[:LESSON_CHARS], "count": 1})
    groups.sort(key=lambda g: -g["count"])
    return {"total": len(groups), "lessons": groups[:limit]}


def block(lessons):
    """The text added to the system message."""
    if not lessons:
        return ""
    return "\n".join([BLOCK, *(f"- {lesson['text']}" for lesson in lessons), EXCEPTIONS])


# ---- Before and after: what a new preference changes ----------------------

EXAMPLES = 3            # new preferences shown with a before/after
TRIED = 6               # new preferences replayed to find them
SAME_TEXT = 0.85        # answers without a tool call this alike did not change
ANSWER_TOKENS = 600     # each answer is the agent's next message or tool call (a file's top fits)


def moment(store, lesson):
    """Where a lesson mattered: the logged request at the step the agent got
    wrong, or (for a rule the user stated) the last request before their
    message. Returns {"request", "asked", "feedback"} or None."""
    from .reading import timeline

    by_id = {s["id"]: s for s in reading.signals(store)}
    signals = sorted((by_id[i] for i in lesson["signals"] if i in by_id),
                     key=lambda s: s.get("revision") or 0, reverse=True)
    for s in signals:
        exchange = None
        if s.get("exchanges"):
            exchange = s["exchanges"][min(s["exchanges"], key=int)]
        elif s.get("item") is not None:
            with store.connect() as db:
                ids = timeline.sessions(db).get(s.get("session"))
            if ids:
                _, items = reading.load_session(store, ids)
                exchange = reacted_to(items[: s["item"]], s.get("text") or "")
        if exchange is None:
            continue
        from .parts import request_of

        with store.connect() as db:
            request = request_of(db, exchange)
        if not isinstance(request, dict) or not request.get("messages"):
            continue
        from .steps import CONTINUED

        said = (timeline.text(m.get("content")) for m in reversed(request["messages"]) if m.get("role") == "user")
        asked = next((t for t in said if t.strip() and not t.lstrip().startswith(CONTINUED)), "")
        return {
            "request": request,
            "asked": " ".join(str(asked).split())[:300],
            "feedback": " ".join(str(s.get("text") or s.get("evidence") or "").split())[:200],
            "said_by": "user" if s.get("source") == "user" else "tool",
        }
    return None


def reacted_to(items, rule):
    """The exchange id of reacted_action (replaying its request asks for that
    action again)."""
    chosen = reacted_action(items, rule)
    return chosen["exchange"] if chosen else None


def reacted_action(items, rule):
    """The agent's step a stated rule is about: the tool call before the
    user's message sharing the most words with the rule (the `npm install`
    behind "never npm", the script written behind "put my name at the top of
    the script"), else the latest tool call, else the latest step.
    Returns the action item."""
    actions = [it for it in items if it.get("kind") == "action" and it.get("exchange")]
    calls = [it for it in actions if it.get("tool_calls")]
    keys = {w for w in words(rule) if len(w) > 2}

    def mentioned(it):
        return words(" ".join(f"{c.get('name')} {c.get('arguments')}" for c in it.get("tool_calls") or []))

    # The call that matches the rule best (the script written, not a later
    # small edit of it); the earliest of equals: where the rule first applied.
    best = max(calls, key=lambda it: len(keys & mentioned(it)), default=None)
    if best is not None and keys & mentioned(best):
        top = len(keys & mentioned(best))
        chosen = next(it for it in calls if len(keys & mentioned(it)) == top)
    else:
        chosen = (calls or actions or [None])[-1]
    return chosen


def partial_arguments(raw):
    """Arguments cut off by the answer cap ('{"path": "a.py", "content": "#!/usr/bin/env…'):
    the fields that are there, the last one as far as it got."""
    import json as _json
    import re as _re

    out = {}
    for m in _re.finditer(r'"(\w+)"\s*:\s*("(?:[^"\\]|\\.)*"?|[^,}]+)', raw or ""):
        value = m.group(2)
        if value.startswith('"'):
            body = value[1:-1] if len(value) > 1 and value.endswith('"') else value[1:]
            try:
                out[m.group(1)] = _json.loads(f'"{body.rstrip(chr(92))}"')
            except ValueError:
                out[m.group(1)] = body
        else:
            out[m.group(1)] = value.strip()
    return out or {"arguments": raw}


def answer_of(message):
    """The agent's answer as the page shows it: text and the first tool call."""
    import json as _json

    calls = []
    for c in (message or {}).get("tool_calls") or []:
        fn = c.get("function") or {}
        args = fn.get("arguments")
        try:
            args = _json.loads(args) if isinstance(args, str) else (args or {})
        except ValueError:
            args = partial_arguments(args)
        calls.append({"name": fn.get("name"), "arguments": args if isinstance(args, dict) else {"value": args}})
    text = (message or {}).get("content") or ""
    return {"text": text if isinstance(text, str) else "", "tool_calls": calls}


async def compare(client, url, model, request, before_block, after_block, *, server=""):
    """The same request answered with the lessons from before this Quick Dream
    and with the new ones, by the served model (temperature 0, capped).
    Returns (before, after, changed)."""
    from .proxy import with_lessons

    async def ask(block):
        # The logged request without its streaming and length settings
        # (vLLM rejects stream_options without stream; max_completion_tokens
        # would override the answer cap).
        dropped = {"stream", "stream_options", "model", "max_tokens", "max_completion_tokens", "temperature"}
        body = {k: v for k, v in request.items() if k not in dropped}
        body["messages"] = with_lessons(request["messages"], block) if block else request["messages"]
        body.update(model=model, temperature=0, max_tokens=ANSWER_TOKENS, stream=False)
        if server == "vllm":
            body["chat_template_kwargs"] = {"enable_thinking": False}
        r = await client.post(url.rstrip("/") + "/chat/completions", json=body)
        r.raise_for_status()
        return answer_of(((r.json().get("choices") or [{}])[0]).get("message"))

    before, after = await ask(before_block), await ask(after_block)
    return before, after, differs(before, after)


def differs(before, after):
    """Whether the action changed, not just its wording: a different tool
    call (name, command, file content), or, with no tool call on either
    side, clearly different text. A reworded preamble before the same call
    is the prompt shifting, not the lesson working."""
    import difflib

    def calls(a):
        # The argument values only: the shared JSON keys would make any two
        # calls of one tool look alike.
        def values(args):
            if isinstance(args, dict):
                return " ".join(_json_text(v) for _, v in sorted(args.items()))
            return _json_text(args)
        return [(c["name"], " ".join(values(c["arguments"]).split())) for c in a["tool_calls"]]

    if before["tool_calls"] or after["tool_calls"]:
        x, y = calls(before), calls(after)
        if [n for n, _ in x] != [n for n, _ in y]:
            return True  # another tool, or a call made or not
        # The same tools: different only if what they do clearly differs
        # (live: two reads of snake.py from line 20 vs 21 judged a "win").
        return any(difflib.SequenceMatcher(None, a, b).ratio() < SAME_TEXT
                   for (_, a), (_, b) in zip(x, y, strict=True))
    x, y = " ".join(visible(before["text"]).split()), " ".join(visible(after["text"]).split())
    return difflib.SequenceMatcher(None, x, y).ratio() < SAME_TEXT


def visible(text):
    """What the user would see of an answer: no private reasoning (a
    <think> block, or everything before a closing </think>) and no tool-call
    markup written as text (the call itself is shown parsed)."""
    import re as _re

    text = _re.sub(r"<think>.*?</think>", "", text or "", flags=_re.S)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    text = _re.split(r"<tool_call>|<\|tool_call|<function=", text, maxsplit=1)[0]
    return text.strip()


def _json_text(value):
    import json as _json

    return value if isinstance(value, str) else _json.dumps(value, sort_keys=True, ensure_ascii=False)


def shown(answer, limit=1500):
    """An answer as one text for the judge."""
    parts = [answer["text"].strip()] if answer["text"].strip() else []
    parts += [f"{c['name']}({_json_text(c['arguments'])})" for c in answer["tool_calls"]]
    text = "\n".join(parts) or "(nothing)"
    return text if len(text) <= limit else text[:limit] + " …"


JUDGE = """A coding agent was given this lesson:
{lesson}

Two answers the agent gave at the same moment of a conversation:

Answer A:
{a}

Answer B:
{b}

Which answer follows the lesson? Reply as JSON: {{"follows": "A"}}, {{"follows": "B"}}, {{"follows": "both"}} or {{"follows": "neither"}}."""


def judge_prompt(lesson, before, after):
    """The judge's question, with the two answers in an order that does not
    give away which is new. Returns (prompt, swapped)."""
    import hashlib

    swap = hashlib.sha256(lesson.encode()).digest()[0] % 2 == 1
    a, b = (after, before) if swap else (before, after)
    return JUDGE.format(lesson=lesson, a=shown(a), b=shown(b)), swap


def verdict_of(text, swap):
    """The judge's answer as "after", "before", "both", "neither" or None."""
    import json as _json
    import re as _re

    prefilled = _re.match(r'\s*\{\s*"follows"\s*:\s*"\s*(both|neither|A|B)(?![a-z])', text or "", _re.I)
    try:
        follows = prefilled.group(1) if prefilled else _json.loads(text).get("follows")
        if isinstance(follows, str) and follows.lower() in ("both", "neither"):
            follows = follows.lower()
    except (ValueError, AttributeError):
        m = _re.search(r'"follows"\s*:\s*"(A|B|both|neither)"', text or "") or _re.search(
            r"\b(A|B|both|neither)\b", visible(text or ""))
        follows = m.group(1) if m else None
    letter = {"A": "before", "B": "after"} if not swap else {"A": "after", "B": "before"}
    return letter.get(follows, follows if follows in ("both", "neither") else None)


async def judge(client, url, model, lesson, before, after, *, server=""):
    """Whether the change is because of the lesson: the served model says
    which answer follows it, with the two in an order that does not give
    away which is new. Returns "after", "before", "both", "neither" or None."""
    prompt, swap = judge_prompt(lesson, before, after)
    body = {
        "model": model, "temperature": 0, "max_tokens": 20, "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "verdict", "schema": {
            "type": "object", "properties": {"follows": {"type": "string", "enum": ["A", "B", "both", "neither"]}},
            "required": ["follows"]}}},
    }
    if server == "vllm":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    try:
        r = await client.post(url.rstrip("/") + "/chat/completions", json=body)
        r.raise_for_status()
        message = (r.json().get("choices") or [{}])[0].get("message") or {}
        # The verdict, else the reasoning channel (LM Studio may put it there).
        text = message.get("content") or ""
        if verdict_of(text, False) is None:
            text = message.get("reasoning_content") or message.get("reasoning") or text
    except Exception:
        return None
    return verdict_of(text, swap)
