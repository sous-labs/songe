"""The questions the model being trained answers about a feedback event, each
with a fixed JSON answer (ported unchanged from scripts/signals/model.py, where
they were checked against a hand-labelled answer key):

- classify: split a user message into feedback / new task / question parts
- review: which of a turn's unflagged tool results carried a signal
- responsible: which steps caused the feedback or should have used it
- lesson: what to do differently, how general that is, how success is checked
"""

import re

TYPES = ["correction", "retry", "approval"]

CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "parts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "says_something_is_wrong": {"type": "boolean"},
                    "kind": {"type": "string", "enum": ["feedback", "new_task", "question"]},
                    "type": {"type": ["string", "null"], "enum": TYPES + [None]},
                    "text": {"type": "string"},
                },
                "required": ["says_something_is_wrong", "kind", "type", "text"],
            },
        }
    },
    "required": ["parts"],
}

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "signals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step": {"type": "integer"},
                    "type": {"type": "string", "enum": ["outcome", "failed_result", "success"]},
                    "quote": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["step", "type", "quote", "why"],
            },
        }
    },
    "required": ["signals"],
}

RESPONSIBLE_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {"type": "array", "items": {"type": "integer"}},
        "reason": {"type": "string"},
    },
    "required": ["steps", "reason"],
}

PREFERENCE_SCHEMA = {
    "type": "object",
    "properties": {"kind": {"type": "string", "enum": ["about_me", "reaction", "task", "text"]}},
    "required": ["kind"],
}

REPLACES_SCHEMA = {
    "type": "object",
    "properties": {"replaces": {"type": "array", "items": {"type": "integer"}}},
    "required": ["replaces"],
}

BOTH_TRUE_SCHEMA = {
    "type": "object",
    "properties": {"kind": {"type": "string", "enum": ["both", "later_only"]}},
    "required": ["kind"],
}

SAME_SCHEMA = {
    "type": "object",
    "properties": {"same": {"type": "array", "items": {"type": "integer"}}},
    "required": ["same"],
}

SAME_WISH_SCHEMA = {
    "type": "object",
    "properties": {
        "note_1_asks": {"type": "string"},
        "note_2_asks": {"type": "string"},
        "kind": {"type": "string", "enum": ["same", "different"]},
    },
    "required": ["note_1_asks", "note_2_asks", "kind"],
}

LESSON_SCHEMA = {
    "type": "object",
    "properties": {
        "lesson": {"type": "string"},
        "short": {"type": "string"},
        "instance": {"type": "string"},
        "generality": {"type": "string", "enum": ["task", "kind", "always"]},
        "verifiable": {"type": ["string", "null"]},
    },
    "required": ["lesson", "short", "instance", "generality", "verifiable"],
}

CLASSIFY = """You are reviewing a conversation between a user and a coding agent.

Earlier user messages (oldest first):
{earlier}

What the agent did since the last user message (one line per step):
{steps}

New user message:
{message}

Split the new user message into parts. For each part give:
- says_something_is_wrong: true if the part says anything is wrong, broken, missing, too long, or should be done differently; false otherwise
- kind: "feedback" if it judges the agent's work above, "new_task" if it asks for something new, "question" if it only asks something
- type, only for feedback:
  "correction" = says something is wrong, broken, missing, or should be different (even briefly, even politely)
  "retry" = only asks to try again, without saying what is wrong
  "approval" = accepts the work as it is
  null for new_task and question
- text: the exact words of that part
Most messages have one part.

Examples:
"cli is broken; keep the old entrypoint working too" -> [{{"says_something_is_wrong": true, "kind": "feedback", "type": "correction", "text": "cli is broken; keep the old entrypoint working too"}}]
"Wrong commit message. Use Conventional Commits. Try again." -> [{{"says_something_is_wrong": true, "kind": "feedback", "type": "correction", "text": "Wrong commit message. Use Conventional Commits. Try again."}}]
"Too long. Keep it under 30 lines." -> [{{"says_something_is_wrong": true, "kind": "feedback", "type": "correction", "text": "Too long. Keep it under 30 lines."}}]
"Try again." -> [{{"says_something_is_wrong": false, "kind": "feedback", "type": "retry", "text": "Try again."}}]
"Looks good to me." -> [{{"says_something_is_wrong": false, "kind": "feedback", "type": "approval", "text": "Looks good to me."}}]
"Thanks, the commit looks fine. Now add a --dry-run flag to dedupe.py." -> [{{"says_something_is_wrong": false, "kind": "feedback", "type": "approval", "text": "Thanks, the commit looks fine."}}, {{"says_something_is_wrong": false, "kind": "new_task", "type": null, "text": "Now add a --dry-run flag to dedupe.py."}}]
"What does --in-place do?" -> [{{"says_something_is_wrong": false, "kind": "question", "type": null, "text": "What does --in-place do?"}}]

Answer with JSON only."""

REVIEW = """You are reviewing what a coding agent did for this request:
{task}

{ran}
{steps}

Find outputs that carry information the agent must use, or that show its work is wrong or right, even though no error was reported. Most turns have none: return an empty list then.
Types:
- "outcome": information the next steps should use. Example: "APPLE  Y...G" from a word game, "500: higher" from a guessing game.
- "failed_result": output that shows the work is wrong without an error message. Example: after implementing a converter, "III -> 0, IV -> 0, IX -> 0".
- "success": output that proves the requested work now does what the user asked. Example: the user wanted `python3 -m pkg circle 2` to print 12.57 and the output is "circle 2: 12.57".
Not signals: output that merely shows a command ran, file contents, git status, listings, compile checks like "syntax OK", or {errors}
For each signal give the step number, the type, a short quote of the output, and why it matters. Answer with JSON only."""

# The two wordings of the review. A turn of shell commands is asked as it
# always was: the rules find a command's errors (exit codes, tracebacks), so
# the review leaves them alone. Nothing else reads the result of an agent's
# own function or MCP tool, so there a refusal or an error is itself a signal.
REVIEW_COMMANDS = {
    "ran": "These are the commands the agent ran in one turn, with their output:",
    "errors": "any error message (those are handled elsewhere).",
}
REVIEW_TOOLS = {
    "ran": "These are the tool calls the agent made in one turn, with their results:",
    "errors": (
        "the error message of a shell command (those are handled elsewhere). When one of the agent's other "
        "tools refuses a call or says the result is wrong (a move that is not allowed, a value it does not "
        'accept, an attempt that failed), that is a signal of type "failed_result".'
    ),
}

RESPONSIBLE = """You are reviewing what a coding agent did for this request:
{task}

The agent's steps:
{steps}

Feedback that arrived {where}:
{feedback}

What the agent did after this feedback (it may show what really had to be fixed):
{after}

Which steps are responsible for this feedback? {allowed}
- A step is responsible if its action was wrong or incomplete and caused the problem, or if it should have used earlier information and did not (for example it repeated a guess that had already been scored, or said "done" without checking what the user asked for).
- A step that only read, searched, or ran a check that revealed the problem is not responsible.
- For a tool error, the responsible step is usually the step that made the failing call.
- For information such as a game's feedback: if the later steps used it correctly, return an empty list.

Examples:
- After renaming a function the user says "the CLI is broken", and afterwards the agent fixes a file it had not edited. Responsible: the step that made the incomplete rename, and the step that said it was done without running the CLI.
- Step 8 guessed APPLE and got "Y...G"; steps 9 and 10 guessed APPLE again. Feedback "Y...G" at step 8. Responsible: steps 9 and 10, which ignored it.
- "Could not find oldString" in the result of step 3. Responsible: step 3.

Give the step numbers and one sentence on why. Answer with JSON only."""

LESSON = """A coding agent got this feedback while working on:
{task}

What the responsible steps did:
{steps}

Feedback:
{feedback}

Why these steps are responsible: {reason}

What the agent did after the feedback:
{after}

Write what the agent should do differently next time.{wish}
- lesson: one or two sentences of advice that would also help on a different task of the same kind. Name the general practice; do not mention this task's files, function names or values.
  Good: "When renaming a function, search for every way it is used, including names built as strings, and run the program's entry points before saying it is done."
  Bad: "Update report.py to use circle_area."
- short: the same advice as one instruction of at most 15 words, starting with a verb, keeping what matters most (it is shown to the agent at the start of every task).
  Good: "Search for every use of a renamed name, including strings, and run the entry points."
- instance: the specific facts of this case (files, values, what exactly went wrong).
- generality: "task" if the advice only helps with this exact task; "kind" if it helps with tasks of the same kind (renames, word games, writing tests); "always" if it applies to any task: how to use the tools correctly (which tool runs shell commands, exact paths, edit strings, not thinking at length before acting), or a standing preference of this user (for example a commit message style).
  Examples: "The execute tool runs JavaScript only; use the shell tool for shell commands" -> always. "Write commit messages in Conventional Commits style" -> always. "In Wordle, keep letters marked G in place" -> kind. "When renaming a function, search for names built as strings" -> kind.
- verifiable: if success on such a task can be checked automatically, how ("the tests pass", "the game is won"); otherwise null.
Answer with JSON only."""



# Added to the lesson question when the feedback is the user's own words. A
# wish said in passing ("uv please", "in metric please") is a standing
# preference, and the lesson must keep the thing the user named. Live
# (Qwen3.5-4B, 24 chats corrected in everyday words): "I use uv, not pip." was
# stored as "Verify the user's preferred package manager before providing
# instructions." and no answer changed. Shapes with placeholders, not worked
# examples: a worked example is copied (see EXAMPLE_LESSONS).
USER_WISH = """
The feedback is the user's own words. If it tells the agent how this user wants things (which tool, which format, which wording, which units, which style), that wish itself is the lesson, and the rule below about naming the general practice does not apply:
  - lesson and short are the direct instruction and name exactly what the user named, in the user's words. The user's key words (the name of the tool, the unit, the phrase, the format) must appear in both, written out. Never replace them with their category ("the preferred tool", "the requested format") and never turn the wish into advice to check, verify or ask.
    Shapes, where <THING> and <OTHER> stand for the user's own words:
    "<THING>, not <OTHER>." -> "Use <THING>, not <OTHER>."
    "<THING> please." -> "Use <THING>." or "Write <THING>."
    "You forgot <THING>." -> "Always include <THING>."
  - generality is "always" for such a wish, unless the user limited it to this one task."""

# Asked once about every user message that follows an answer: what is it? On
# real conversations (WildChat turns through Qwen3.5-4B) the lesson written
# for a correction was marked "always" for 48% of the changes a user wanted
# once ("make it shorter") and for 7% of ordinary follow-up turns, and 32% of
# the lasting wishes ("i want in russian", "be very concise") were read as new
# tasks and left nothing. So the turn is judged on its own: only a standing
# wish becomes a note at once. The same answer words the wish as the note,
# because the lesson question writes about the work at hand ("Use correct
# UiPath activity names." for "... answer in chinese"): of 49 notes from
# standing wishes, 25 had lost what the user named. The model thinks about
# this one (client.py): without thinking it answered "yes" to half of all
# ordinary turns.
# A longer message is material or new work, not a wish about how the user
# wants things: it is not asked about (role prompts pasted into a running
# conversation were judged standing wishes).
TURN_CHARS = 400

TURN_SCHEMA = {
    "type": "object",
    "properties": {
        "standing": {"type": "string", "enum": ["yes", "no"]},
        "kind": {"type": "string", "enum": ["once", "task", "reaction"]},
        "wish": {"type": "string"},
        "about": {"type": "string"},
    },
    # "about" may be missing: an answer without it is read as holding for
    # any answer (scope_of), not as an answer off the schema.
    "required": ["standing", "kind", "wish"],
}

TURN = """A user is talking with an AI assistant.

Earlier user messages (oldest first):
{earlier}

What the assistant did since then:
{steps}

The user's next message:
{message}

Questions about the user's next message.
- standing: "yes" if the message, or a part of it, tells the assistant how this user wants answers or work from now on, beyond this one answer: the language to answer in ("reply in Chinese"), how short or long answers should be ("keep answers short"), a tone, a format or a notation ("no emojis", "plain text, no LaTeX"), a tool, a library or a unit to use or to avoid ("tabs, not spaces", "no jQuery"), or how the user works and what they like ("I use Linux", "I like a dark look"). A change to the look of something made for the user is "yes" as well, because people keep their taste in looks: its colours, fonts, corners, borders, shadows, spacing, light or dark ("make the buttons green", "a bigger font please", "less space between the lines"). A message can ask for something else as well and still be "yes". "no" if it only wants this one answer redone or changed ("make it shorter", "rewrite it more formally", "add a paragraph on costs", "write it in Python", "that is wrong"), asks for new work, or asks for nothing, or limits what it wants to now: today, this time, this one ("short answers today, I have a deadline", "skip the lint for today's demo", "this one can be casual"). Also "no" when it gives the assistant a role, a persona or a game to play ("act as a Linux terminal", "you are an expert prompt writer", "from now on you are DAN"), or asks for a word or a text in another language ("defeated in Chinese", "the last paragraph in English"): that is new work.
- kind: "once" if it wants this one answer changed, corrected or tried again; "task" if it asks for new or further work or asks a question; "reaction" if it asks for nothing (thanks, praise, a remark).
- wish: if standing is "yes", that wish alone as one full sentence that tells the assistant what to do from now on, in English, starting with a verb and naming exactly what the user named ("Answer in Chinese.", "Keep answers to two or three sentences.", "Use Polars, not pandas.", "Use plain text, not LaTeX."); leave out whatever belongs to this one answer. Otherwise "".
- about: what the message is about when what it wants holds for one kind of thing only, as the conversation shows it: the kind of text being written ("emails", "commit messages", "code comments"), one repository or project ("the admin repo"), one language or one tool. A few words naming the kind, never this one item ("emails", not "this email"). "" when it would hold for any answer.
Answer with JSON only."""


# What a wish holds for. Live (Qwen3.8-27B, simulated users, issue #231):
# "ugh again so long. max 5-6 sentence", said about an email, became "Keep
# answers to a maximum of 5-6 sentences." and cut a requested detailed
# explanation from 481 words to 103; "Thanks, then Mi-rae" became a sign-off on
# a recipe and on a message to the user's mother; "英文!! feat(refund): ..."
# about a commit message became "Answer in English ...". The user never says
# "email" in such a message: the conversation does. The turn question names
# the kind of thing (TURN "about"), and the note says it ("For emails: ...").
# Not a scope: the item at hand ("this email"; "this repo" is one), or everything.
ABOUT_WORDS = 6
NO_SCOPE = re.compile(
    r"^(this|that)\s+(?!(repo|repository|project|codebase|folder|app)\b)"
    r"|^the\s+(one|answer|reply|response|message|text|task|request|draft|version|time)\b"
    r"|^(any|all|every|everything|anything|general|none|n/?a|nothing|answers?|replies|responses?)\b",
    re.I,
)


def scope_of(answer, said=""):
    """What the turn question says the user's wish holds for ("emails",
    "commit messages", "the admin repo"), or "" when it holds for any answer
    or the answer gives nothing usable. `said` is what the user wrote in the
    conversation: a kind the user never named in it is still kept when the
    message is not in Latin letters (rules.voiced)."""
    from .rules import voiced

    about = " ".join(str((answer or {}).get("about") or "").split()).strip(" .:;,\"'")
    about = re.sub(r"^(for|in|on|only|just|about)\s+", "", about, flags=re.I)
    if not about or len(about.split()) > ABOUT_WORDS or len(about) > 48 or NO_SCOPE.search(about):
        return ""
    return about if voiced(about, said) > 0 else ""


def scoped(note, about):
    """A note said for what it holds: "Keep answers to a maximum of 5-6
    sentences." about emails is "For emails: keep answers to a maximum of
    5-6 sentences."; a note that names it already stays as it is."""
    from .rules import voiced

    note = " ".join(str(note or "").split())
    if not note or not about or voiced(about, note) >= 0.5:
        return note
    if note[:1].isupper() and not note[1:2].isupper():
        note = note[:1].lower() + note[1:]
    return f"For {about}: {note}"


# Languages, libraries and systems named in the question's own examples. A
# wish that names one of them when the user's message does not is the example
# pasted back, not this user's wish ("Answer in Spanish.", then an example,
# for "restart the duel"). A message not in plain Latin letters is let
# through: "请用中文回答" asks for Chinese without the word.
TURN_EXAMPLE_NAMES = ("chinese", "polars", "pandas", "jquery", "linux")


# A lead-in that only points back at the conversation: a note is read without it.
BACK = re.compile(r"^(same thing|same|again)\b[\s,]*((but|and)\s+)?", re.I)


def own_words(message):
    """The user's message as a note in their own words, or "" when it cannot
    be one: a single word, several sentences (which of them is the wish is
    not known), or longer than a note is left whole (quick.SHORT). Words
    that only point back at the conversation are left off: "Same thing
    without the comments." is noted as "Without the comments." (live: asked
    whether that note and "uv please." are one wish, the model read "same
    thing" as uv and put both in one place of the prompt)."""
    from .rules import SENTENCES

    own = " ".join(str(message).split())
    if len(own) > 90 or len(SENTENCES.split(own)) != 1:
        return ""
    rest = BACK.sub("", own)
    own = own if rest == own else rest[:1].upper() + rest[1:]
    return own if len(own.split()) > 1 else ""


# A wish the user limits to now ("today only reply fast", "今天先别管 lint 了",
# "just for today skip the tests") is about the work at hand, whatever the
# model answered: not a note, and not a change wanted once either (read as
# "once", "today only reply fast, short short" was joined with an email-length
# wish from another chat and still became a note). Live (Qwen3.8-27B, two test users): both were read as
# standing wishes and became the notes "Reply fast and short today." and
# "Ignore lint today.", which the prompt then carried into every later day.
# "From today on" and "starting today" say the opposite: the wish lasts; a
# span of dates ("from 2026-02-01 till today") is not a limit.
NOW_ONLY = re.compile(
    r"\b((?<!till )(?<!until )(?<!through )(?<!to )today(?!['’]s)|tonight|for now|this time|just this once|this once)\b|今天|今晚|这次|这回|暂时|오늘|이번만|이번엔|\b(hoy|esta vez|por ahora)\b",
    re.I,
)
FROM_NOW = re.compile(
    r"\b(from (today|now|tonight)|(for|from) now on|starting (today|now)|as of today|going forward|always|every time|ever|not just)\b"
    r"|从今天|今天起|今天开始|以后|一直|每次|永远|오늘부터|항상|\b(desde hoy|siempre)\b",
    re.I,
)
SENTENCE_END = re.compile(r"(?<=[.!?;])\s+|(?<=[。！？；])|\n")


def now_only(message, wish=""):
    """Whether the user's message limits the wish it voices to now.

    The limit has to be in the sentence the wish comes from (one the model's
    wording names, rules.voiced), or in the wording itself: "I prefer tabs,
    not spaces. Today let's refactor utils.py" keeps the wish for tabs. A
    sentence that also says the wish lasts ("today and always", "not just
    today", "for now on") is not a limit."""
    from .rules import voiced

    if NOW_ONLY.search(wish or "") and not FROM_NOW.search(wish or ""):
        return True
    return any(
        NOW_ONLY.search(part) and not FROM_NOW.search(part) and voiced(wish or "", part) > 0
        for part in SENTENCE_END.split(str(message or ""))
    )


def turn_verdict(answer, message="", earlier=""):
    """("lasting", the wish as a note) or ("once" | "task" | "reaction", "")
    from the answer to TURN about this message; (None, "") when the model
    gave none. `earlier` is what the user wrote before it in the conversation.

    The wish is the model's wording of what the user said, so it has to name
    something this message names (rules.voiced), or the user has to have said
    all of it earlier in the conversation ("Again, properly."). Live
    (Qwen3.5-4B): 'Always start your reply with "Yes, chef."' came back as
    the wish "Speak French.", was kept as the lesson and practised by a Deep
    Dream; a remark about quotation marks came back as the question's own
    example "Make the buttons green.", and a page pasted earlier had a
    button in it. Such a wish is not used: the note is then the user's own
    message, when that is one short sentence (own_words)."""
    from .rules import voiced

    if not isinstance(answer, dict):
        return None, ""
    kind = answer.get("kind") if answer.get("kind") in ("once", "task", "reaction") else None
    if answer.get("standing") != "yes":
        return kind, ""
    if now_only(message, str(answer.get("wish") or "")):
        return "task", ""
    wish, said = " ".join(str(answer.get("wish") or "").split()), message.lower()
    if said.isascii() and any(name in wish.lower() and name not in said for name in TURN_EXAMPLE_NAMES):
        return kind, ""
    if not 12 <= len(wish) <= 140:
        return "lasting", ""
    if not voiced(wish, message) and voiced(wish, f"{earlier}\n{message}") < 1:
        return "lasting", own_words(message)
    return "lasting", wish


# Asked about a sentence the rules flagged as a possible preference of the
# user's own (rules.preference_sentences). Only "about_me" is kept. The other
# three are what such a sentence turned out to be in real chats: a pick among
# offers or a word on the last answer ("I love the new layout"), a line about
# the job ("I only get three results"), a line of a text handed over to be
# worked on (a cover letter or an essay in the first person).
PREFERENCE = """A user is talking with an AI assistant.

What the assistant said just before (nothing at the start of a conversation):
{before}

The user's message:
{message}

One sentence of that message:
{sentence}

What is this sentence? Choose one:
- "about_me": the user tells the assistant something lasting about themself: what they like or dislike, what they avoid or cannot have, what they need, what they always or never do. It would still be true in another conversation next month.
- "reaction": the user reacts to what the assistant just said or made, or picks among things it offered.
- "task": the user describes the job at hand: what they want now, what they are working on, what happened, an error, a result.
- "text": the user is not speaking as themself here. The sentence belongs to a text they hand over to be worked on (to translate, correct, rewrite, answer, continue): a letter, an email, an essay, a story, a post, another chatbot's reply. Or it is the line of a character in a role-play or a dialogue, or the words of another person.
Answer with JSON only."""

# Asked of each preference the user states about themself, with the ones they
# stated before it (reading/store.retire_replaced): people change, and a note
# that is no longer true must leave. Two questions, because one was not
# enough (live, Qwen3.5-4B, thinking off): asked only which notes a sentence
# replaces, the model named notes about other things for a sentence that
# tells of a change ("I've quit soft drinks; I only drink water now." was
# said to replace four notes about books, trams and food): 34 notes wrongly
# retired among 480 sentences that replace nothing. Each note it names is
# therefore shown again on its own with the new sentence (BOTH_TRUE), and
# only "later_only" retires it: 2 in the same 480 (a sentence said twice in
# other words, and two diets that do exclude each other), and 36 of 40 real
# changes still found.
REPLACES = """A user tells an AI assistant things about themself. The assistant keeps each as a note, in the user's words.

Notes kept so far, oldest first:
{notes}

What the user said after all of them:
{sentence}

Does the new sentence say that one of the notes is no longer true? That takes both: the note and the new sentence are about the same thing (the same food, the same habit, the same choice), and the user cannot hold both now: they changed their mind, stopped doing it, gave it up, or now want the opposite.
Most new sentences replace nothing: they are about something else, or they add to a note (one more thing the user likes or avoids, a detail). A sentence that tells of a change ("I've stopped ...", "... now") replaces only the note about that same thing, if there is one.
Answer with JSON only: {{"replaces": [the numbers of the notes it replaces]}}, with an empty list when it replaces none."""

BOTH_TRUE = """A user told an AI assistant two things about themself, at different times.

Earlier: {old}
Later: {new}

Can both be true of the user now?
- "both": yes. They are about different things, or the later one adds to the earlier one.
- "later_only": no. The later sentence means the earlier one is no longer true: they changed their mind, stopped doing it, gave it up, or now want the opposite.
Answer with JSON only."""

# One wish, one place (reading/store.fold_same). A user asks for the same thing
# in three conversations in three ways, and reading writes a note each time.
# Two questions, as for REPLACES and for the same reason: which of the notes
# kept so far a new note could repeat, then each one named with it alone.
# The second question has the model say each note again as an instruction
# before it answers. Live (Qwen3.5-4B, thinking off, pairs asked both ways):
# asked "same" or "different" at once, it called 13 of 44 pairs about
# different things the same ("Use pnpm, not npm." and "Write text without
# using dashes."); saying them again first, 3 of 56, with 55 of 58 pairs that
# are one wish found.
SAME = """An AI assistant keeps short notes about one user: what the user wants in the assistant's replies.

Notes kept so far:
{notes}

A new note:
{note}

Which of the notes kept so far could be the same wish of the user as the new note, said in other words? They would be about the same thing (the same tool, the same units, the same format, the same phrase, the same thing to leave out). One may say it more widely, or name one case of it.
Most new notes are about something else: then name none.
Answer with JSON only: {{"same": [the numbers of those notes]}}, with an empty list when there is none."""

SAME_WISH = """An AI assistant keeps short notes about one user: what the user wants in the assistant's replies.

Note 1: {a}
Note 2: {b}

Is this one wish of the user, noted twice? First say each note again in your own words, as an instruction of a few words that names exactly what to use, do or leave out. Then:
- "same": the two instructions ask for the same. One may ask for it everywhere and the other in one situation, or one may name one kind of what the other names (one case of it, a part of it).
- "different": they ask for two things (two tools, two foods, two things to leave out, two formats), or for opposite things.
Answer with JSON only."""

# Added to the lesson question when the feedback is a tool's own output. What
# the output named is the lesson, as with the user's words. Live (Qwen3.5-4B
# and 9B, 30 coding tasks with failing commands): "pytest: command not found"
# and "No module named pytest" were stored as "Install missing dependencies
# before running tests." and the next task ran the same failing command
# (issue #144).
# What the agent did next is added only when it is the ordinary way to do the
# thing. Live (the same sessions, read three times): pip's refusal to install
# into the system Python became "Use '--break-system-packages' when pip
# refuses to install." each time, once as the only note in the prompt.
# How far the fact reaches is judged by the lesson question's own rule. Live,
# with a line here that made a fact about the tool or the machine "always": 13
# to 16 of 16 to 19 lessons were read as "always" (8 of 18 without it), among
# them "Install missing dependencies like tomli_w before running tests.",
# which is true of one project.
TOOL_FACT = """
The feedback is a tool's own output. If it shows how the tool or this machine behaves (a command, program, module or option that is missing or refused, a value the tool does not accept, the form it wants), that fact is the lesson, and the rule below about naming the general practice does not apply:
  - lesson and short state the fact and name exactly what the output named (the command, program, module, option, value or message), written out as in the output. Never replace it with its category ("the missing dependency", "the right tool", "the required format"). Here short may be a statement instead of an instruction.
  - Add what to do instead only if the agent then did it, it worked, and it is the ordinary way to do the thing. A way around a refusal or a warning that protects something (forcing, overriding or skipping a check, installing for the whole system, switching the protection off) is never part of the lesson, even when it worked: then the lesson is the fact alone.
    Shapes, where <NAME> stands for what the output named and <WORKED> for the ordinary thing that worked afterwards:
    "<NAME>: command not found" -> "<NAME> is not installed here; use <WORKED>." or, with nothing ordinary that worked, "<NAME> is not installed here."
    "<NAME> is not accepted" -> "<NAME> is not accepted; use <WORKED>."
  This task's own files, function names and data still do not belong in the lesson."""

# Asked instead when the failed result is the last thing in the conversation:
# the agent stopped there, so the task was not completed (a lost game, a build
# that never passed). The steps in between each looked reasonable; the lesson
# is in the whole attempt, not in the last message (live: lost word games gave
# "When the turn limit is reached, acknowledge the revealed secret word").
ENDED = """
The feedback is a tool's own output, and the conversation ended with it: the task was not completed. All steps of the attempt, one line each:
{attempt}
Write the lesson about the attempt as a whole: what the agent should do differently from the start of the next task of this kind (its approach, what it should check or rule out first, what it kept doing that did not help). The lesson is not about the last step, nor about how to react to this final message or how to report the ending.
If the output names something that is missing or refused (a command, program, module, option or value), the lesson names it exactly as written."""

# The prompt's own example lessons: a lesson that repeats one of them was
# copied, not written (live: Granite 3B on LM Studio gave all 9 signals the
# renaming example as their lesson).
EXAMPLE_LESSONS = (
    "When renaming a function, search for every way it is used, including names built as strings, and run the program's entry points before saying it is done.",
    "The execute tool runs JavaScript only; use the shell tool for shell commands",
    "Write commit messages in Conventional Commits style",
    "In Wordle, keep letters marked G in place",
    "When renaming a function, search for names built as strings",
)


def copied_example(lesson):
    """Whether a lesson repeats one of the prompt's examples."""
    import difflib

    text = " ".join(str(lesson or "").lower().split())
    if not text:
        return False
    if any(mark in text for mark in ("<thing>", "<other>", "<name>", "<worked>")):
        return True  # a shape of USER_WISH or TOOL_FACT with its placeholders left in
    return any(difflib.SequenceMatcher(None, text, " ".join(e.lower().split())).ratio() > 0.8
               or text.startswith(" ".join(e.lower().split())[:50]) for e in EXAMPLE_LESSONS)
