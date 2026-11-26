"""Read: recorded sessions become signals, per learner, resumably. The model
is faked; nothing is loaded and nothing is executed."""

import asyncio
import json

import httpx
import pytest

from songe.models import registry
from songe.parts import request_of
from songe.reading import store as reading
from songe.reading.client import Reader, Stopped

REAL_CLIENT = httpx.AsyncClient
SYSTEM = {"role": "system", "content": "You are an agent.\nWorking directory: /work/proj"}


def tool(name, **props):
    return {"type": "function", "function": {
        "name": name, "description": f"The {name} tool.",
        "parameters": {"type": "object", "properties": {k: {"type": v} for k, v in props.items()},
                       "required": list(props)}}}


TOOLS = [
    tool("shell", command="string"),
    tool("read", filePath="string"),
    tool("write", filePath="string", content="string"),
    tool("edit", filePath="string", oldString="string", newString="string"),
]


def call(cid, name, **args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def reply(message):
    return {"choices": [{"message": message}]}


def record_session(store, *, more=False, started=None, reply_text="No, you changed the wrong file.",
                   ask="fix the failing test"):
    """Ask, run the tests (1 failed), edit, run again (passed), done; then the
    user corrects. With more=True the session continues one more turn. Another
    `ask` makes another conversation (the same opening is the same one)."""
    t1 = {"role": "assistant", "content": "", "tool_calls": [call("a", "shell", command="pytest")]}
    r1 = {"role": "tool", "tool_call_id": "a", "content": "1 failed, 2 passed\nExit code: 1"}
    t2 = {"role": "assistant", "content": "", "tool_calls": [call("b", "edit", filePath="x.py", oldString="a", newString="b")]}
    r2 = {"role": "tool", "tool_call_id": "b", "content": "Edited x.py"}
    t3 = {"role": "assistant", "content": "", "tool_calls": [call("c", "shell", command="pytest")]}
    r3 = {"role": "tool", "tool_call_id": "c", "content": "3 passed\nExit code: 0"}
    done = {"role": "assistant", "content": "Fixed."}
    user = {"role": "user", "content": ask}
    history = [SYSTEM, user]
    for action, result in ((t1, r1), (t2, r2), (t3, r3)):
        store.record("/v1/chat/completions", {"messages": history, "tools": TOOLS}, reply(action), started=started)
        history = history + [action, result]
    store.record("/v1/chat/completions", {"messages": history, "tools": TOOLS}, reply(done), started=started)
    history = history + [done, {"role": "user", "content": reply_text}]
    store.record(
        "/v1/chat/completions",
        {"messages": history, "tools": TOOLS},
        reply({"role": "assistant", "content": "Sorry."}),
        started=started,
    )
    if more:
        history = history + [
            {"role": "assistant", "content": "Sorry."},
            {"role": "user", "content": "Also add a docstring."},
        ]
        store.record(
            "/v1/chat/completions",
            {"messages": history, "tools": TOOLS},
            reply({"role": "assistant", "content": "Added."}),
            started=started,
        )
    store.reconstruct()


class FakeReader(Reader):
    """Canned answers by question; counts the questions asked."""

    def __init__(self, stop_after=None):
        self.model, self.think = "fake", {"classify"}
        self.asked, self.stop_after = [], stop_after

    async def ask(self, prompt, schema, name):
        if self.stop_after is not None and len(self.asked) >= self.stop_after:
            raise Stopped()
        self.asked.append(name)
        if name == "classify":
            text = prompt.rsplit("New user message:\n", 1)[1].split("\n")[0]
            kind = "new_task" if text.startswith("Also") else "feedback"
            return {"parts": [{"says_something_is_wrong": kind == "feedback", "kind": kind,
                               "type": "correction" if kind == "feedback" else None, "text": text}]}
        if name == "responsible":
            return {"steps": [2], "reason": "it edited the wrong file"}
        if name == "lesson":
            return {"lesson": "Check which file the failing test covers before editing.",
                    "short": "Check which file the failing test covers.",
                    "instance": "edited x.py", "generality": "kind", "verifiable": "the tests pass"}
        return {"signals": []}


def read(store, reader):
    return asyncio.run(reading.read(store, reader))


def test_a_session_is_read_once_into_signals(store):
    record_session(store)
    assert len(reading.pending(store)) == 1
    reader = FakeReader()
    assert read(store, reader) == 1
    found = {s["type"]: s for s in reading.signals(store)}
    # The rules: the first failing run comes before any change, so it only
    # reproduces the problem (no signal); the passing run is a success the
    # user then contradicted. The model: the user's correction.
    assert set(found) == {"success", "correction"}
    correction = found["correction"]
    assert correction["responsible_steps"] == [2]
    assert correction["lesson"].startswith("Check which file")
    assert correction["short"] == "Check which file the failing test covers."  # for the prompt
    assert correction["exchanges"]["2"]  # the logged request behind that step
    assert found["success"]["confirmed"] is False
    assert found["success"]["contradicted_by"] == "the user's correction"
    assert reading.pending(store) == []
    assert read(store, FakeReader()) == 0  # nothing new to read


class Judging(FakeReader):
    """Reads every user message as new work, and judges the turn as told."""

    def __init__(self, standing, kind, wish=""):
        super().__init__()
        self.judged = {"standing": standing, "kind": kind, "wish": wish}

    async def ask(self, prompt, schema, name):
        answer = await super().ask(prompt, schema, name)
        if name == "classify":
            text = answer["parts"][0]["text"]
            return {"parts": [{"says_something_is_wrong": False, "kind": "new_task", "type": None, "text": text}]}
        return self.judged if name == "turn" else answer


def test_a_standing_wish_read_as_new_work_is_still_a_correction(store):
    # "i want in russian" after an answer: the split into parts calls it a new
    # task, the turn question calls it a wish that holds from now on.
    record_session(store, reply_text="i want in russian")
    reader = Judging("yes", "task", "Answer in Russian.")
    read(store, reader)
    assert reader.asked.count("turn") == 1  # once per user turn after an answer
    wish = next(s for s in reading.signals(store) if s["type"] == "correction")
    assert (wish["text"], wish["turn_kind"], wish["source"]) == ("i want in russian", "lasting", "user")
    # The wish as the turn question worded it is the note, not the lesson
    # written about the work at hand.
    assert wish["lesson"] == wish["short"] == "Answer in Russian." and wish["generality"] == "always"


@pytest.mark.parametrize("kind", ["task", "reaction"])
def test_a_turn_that_is_no_standing_wish_adds_nothing(store, kind):
    record_session(store, reply_text="could you add a paragraph on costs?")
    read(store, Judging("no", kind))
    assert [s for s in reading.signals(store) if s["type"] == "correction"] == []


def test_a_change_wanted_once_read_as_new_work_waits_for_a_second_conversation(store):
    """Live (Qwen3.5-4B): the split called 'Say "Yes, chef." first.' new work,
    the turn question a change wanted once, and nothing was stored. The same
    wish in the next conversation then had nothing to meet, in any number of
    conversations."""
    from songe import quick

    first, again = 'Say "Yes, chef." first.', 'Say "Yes, chef." first, please.'
    record_chat(store, ("Give me one tip for staying focused.", "Turn off notifications."), (first, "Yes, chef. Turn off notifications."))
    read(store, Judging("no", "once"))
    waiting = [s for s in reading.signals(store) if s["type"] == "correction"]
    # A correction, for the prompt only once the wish recurs.
    assert [(s["text"], s["turn_kind"], bool(s["lesson"])) for s in waiting] == [(first, "once", True)]
    assert quick.candidates(store, 30, 2) == []
    record_chat(store, ("What is the capital of Australia?", "Canberra."), (again, "Yes, chef. Canberra."))
    read(store, Judging("no", "once"))
    assert [len(c["signals"]) for c in quick.candidates(store, 30, 2)] == [2]


class Disagreeing(FakeReader):
    """The split reads every turn as a correction (FakeReader); the turn
    question reads new work."""

    async def ask(self, prompt, schema, name):
        answer = await super().ask(prompt, schema, name)
        if name == "lesson" and "Yes, chef." in prompt:
            # A lesson in the user's words, the same for both reminders.
            return dict(answer, lesson='Start every reply with "Yes, chef."', short='Start every reply with "Yes, chef."')
        return {"standing": "no", "kind": "task", "wish": ""} if name == "turn" else answer


def test_a_reminder_read_as_a_question_waits_like_a_change_wanted_once(store):
    """Live (Qwen3.5-4B): the split read 'Where's my "Yes, chef."?' as a part
    that says something is missing, the turn question as a question (new
    work). It was stored as new work, so it never met 'You forgot the "Yes,
    chef."' from another conversation, and the habit never got a note."""
    from songe import quick

    record_chat(store, ("What is the capital of Australia?", "Canberra."), ('Where\'s my "Yes, chef."?', "Yes, chef. Canberra."))
    read(store, Disagreeing())
    reminder = next(s for s in reading.signals(store) if s["type"] == "correction")
    assert reminder["turn_kind"] == "once" and quick.suitable(reminder)
    assert quick.candidates(store, 30, 2) == []  # waits for a second conversation
    record_chat(store, ("What's a good name for a cat?", "Whiskers."), ('You forgot the "Yes, chef."', "Yes, chef. Whiskers."))
    read(store, Disagreeing())
    assert [len(c["signals"]) for c in quick.candidates(store, 30, 2)] == [2]


def test_after_work_with_tools_the_turn_question_decides_what_is_new_work(store):
    # "The tests still fail" after edits and test runs: the split calls it a
    # correction, the turn question new work. That stays new work.
    record_session(store, reply_text="the tests still fail")
    read(store, Disagreeing())
    assert [s["turn_kind"] for s in reading.signals(store) if s["type"] == "correction"] == ["task"]


def test_a_long_message_is_work_and_is_not_asked_about(store):
    record_session(store, reply_text="Rewrite this report for the board. " + "It has many paragraphs. " * 20)
    reader = Judging("yes", "task")
    read(store, reader)
    assert "turn" not in reader.asked
    assert [s for s in reading.signals(store) if s["type"] == "correction"] == []


def test_an_example_of_the_question_pasted_back_is_not_the_users_wish():
    from songe.reading import prompts as P

    pasted = {"standing": "yes", "kind": "task", "wish": "Answer in Chinese."}
    assert P.turn_verdict(pasted, "restart the duel") == ("task", "")
    assert P.turn_verdict(pasted, "reply me in chinese") == ("lasting", "Answer in Chinese.")
    assert P.turn_verdict(pasted, "请用中文回答") == ("lasting", "Answer in Chinese.")
    assert P.turn_verdict({"standing": "yes", "kind": "once", "wish": "Use metric units."}, "everything in metric")[0] == "lasting"


def test_a_wish_limited_to_now_is_not_a_standing_wish():
    from songe.reading import prompts as P

    # Live (Qwen3.8-27B): "today only reply fast" and "今天先别管 lint 了" were read as standing wishes.
    fast = {"standing": "yes", "kind": "once", "wish": "Reply fast and short today."}
    assert P.turn_verdict(fast, "today only reply fast, short short. deadline at 5") == ("task", "")
    assert P.turn_verdict({"standing": "yes", "kind": "task", "wish": "Ignore lint today."}, "今天先别管 lint 了，下午 demo 要紧") == ("task", "")
    assert P.turn_verdict({"standing": "yes", "kind": "once", "wish": "Use inline styles."}, "just this once inline styles are fine") == ("task", "")
    assert P.turn_verdict({"standing": "yes", "kind": "once", "wish": "Answer in Chinese."}, "from today on answer in chinese") == ("lasting", "Answer in Chinese.")
    assert P.turn_verdict({"standing": "yes", "kind": "once", "wish": "Answer in Chinese."}, "以后都用中文回答，今天开始") == ("lasting", "Answer in Chinese.")
    assert P.turn_verdict({"standing": "yes", "kind": "once", "wish": "Put today's date at the top of every note."}, "put today's date at the top of every note")[0] == "lasting"
    fy = {"standing": "yes", "kind": "once", "wish": "Start the fiscal year on February 1."}
    assert P.turn_verdict(fy, "fiscal starts feb 1st!! FY27 ytd is from 2026-02-01 till today")[0] == "lasting"


def test_a_lasting_wish_beside_a_mention_of_today_is_kept():
    from songe.reading import prompts as P

    # "Today" elsewhere in the message, or a wish said to last beyond today,
    # does not make the wish a passing one.
    tabs = {"standing": "yes", "kind": "task", "wish": "Use tabs, not spaces."}
    assert P.turn_verdict(tabs, "I prefer tabs, not spaces. Today let us refactor utils.py")[0] == "lasting"
    short = {"standing": "yes", "kind": "once", "wish": "Keep answers short."}
    assert P.turn_verdict(short, "keep answers short, today and always")[0] == "lasting"
    assert P.turn_verdict({"standing": "yes", "kind": "once", "wish": "Use no emojis."}, "no emojis ever, not just today")[0] == "lasting"
    assert P.turn_verdict({"standing": "yes", "kind": "once", "wish": "Use tabs."}, "for now on use tabs")[0] == "lasting"
    assert P.turn_verdict(short, "keep answers short today, deadline at 5") == ("task", "")


def test_a_rule_sentence_carries_what_its_turn_was(store):
    # The rules find "Always ..." in the user's message by its shape; the
    # verdict on the turn it stands in goes with it (quick.suitable).
    record_session(store, reply_text="Always keep the old entry point working.")
    read(store, Judging("no", "task"))
    rule = next(s for s in reading.signals(store) if s.get("from") == "rules" and s["source"] == "user")
    assert (rule["text"], rule["turn_kind"]) == ("Always keep the old entry point working.", "task")


def test_a_wish_that_says_nothing_the_user_wrote_is_not_used():
    """Live (Qwen3.5-4B): the turn question returned the wish "Speak French."
    for a user who wrote 'Always start your reply with "Yes, chef."'."""
    from songe.reading import prompts as P

    said, cat = 'Always start your reply with "Yes, chef."', "[U1] What's a good name for a cat?"
    french = {"standing": "yes", "kind": "once", "wish": "Speak French."}
    # The model's wording is dropped; the user's own message is the note.
    assert P.turn_verdict(french, said, cat) == ("lasting", said)
    # Several sentences (which one is the wish?), too long to stay whole as a
    # note, or a single word: no wish from this question.
    assert P.turn_verdict(french, "That was wrong. " + said, cat) == ("lasting", "")
    assert P.turn_verdict(french, "start your reply with yes chef and " + "then go on " * 8, cat) == ("lasting", "")
    assert P.turn_verdict(french, "again", cat) == ("lasting", "")
    # Words that only point back at the conversation are left off the note.
    assert P.own_words("Same thing without the comments.") == "Without the comments."
    assert P.own_words("Same thing but no em dashes please.") == "No em dashes please."
    assert P.own_words("Again, properly.") == ""
    # A wish worded from the user's words stays as the model worded it,
    # also when the user said it in an earlier message of the conversation.
    chef = {"standing": "yes", "kind": "once", "wish": "Start every reply with 'Yes, chef.'"}
    assert P.turn_verdict(chef, said, cat) == ("lasting", chef["wish"])
    assert P.turn_verdict(chef, "Again, properly.", f"{cat}\n[U2] {said}") == ("lasting", chef["wish"])
    # One word of an earlier message is not enough. Live: a page pasted
    # earlier had a button in it, and a remark about quotation marks came
    # back as the question's own example.
    green = {"standing": "yes", "kind": "once", "wish": "Make the buttons green."}
    quotes = "you output the wrong type of quotations"
    assert P.turn_verdict(green, quotes, '[U1] <button class="buy">Buy</button>') == ("lasting", quotes)
    russian = {"standing": "yes", "kind": "task", "wish": "Answer in Russian."}
    assert P.turn_verdict(russian, "i want in russian") == ("lasting", "Answer in Russian.")
    # One of the user's words is enough for a wish: this one is right.
    imperial = {"standing": "yes", "kind": "task", "wish": "Use imperial units, not metric."}
    assert P.turn_verdict(imperial, "why are you giving me answers in the metric system") == ("lasting", imperial["wish"])


class Guessing(FakeReader):
    """Judges every turn a change wanted once and writes the lesson given."""

    def __init__(self, lesson, turn=None):
        super().__init__()
        self.lesson, self.turn = lesson, turn or {"standing": "no", "kind": "once", "wish": ""}

    async def ask(self, prompt, schema, name):
        answer = await super().ask(prompt, schema, name)
        if name == "lesson":
            return dict(answer, lesson=self.lesson, short=self.lesson, generality="always")
        return self.turn if name == "turn" else answer


def test_a_lesson_that_guesses_what_was_wrong_with_an_answer_is_no_lesson(store):
    """Live (Qwen3.5-4B, a plain chat): "Just the code." was stored as the
    lesson below, and a Deep Dream practised it."""
    record_chat(store, ("Write a bash one-liner that counts the lines in every .py file here.", "Sure! Here it is: ..."),
                ("Just the code.", "find . -name '*.py' | xargs wc -l"))
    read(store, Guessing("Always include the exact file pattern requested by the user."))
    guess = next(s for s in reading.signals(store) if s["type"] == "correction")
    # The user wanted the answer changed: their own words are the lesson.
    assert (guess["turn_kind"], guess["lesson"], guess["short"], guess["unvoiced"]) == ("once", "Just the code.", "Just the code.", True)


@pytest.mark.parametrize("kind", ["task", "reaction"])
def test_a_guess_about_a_turn_that_asked_for_no_change_is_no_lesson(store, kind):
    record_chat(store, ("When was the first iPhone released?", "It was announced on January 9, 2007."),
                ("And the first iPad?", "April 3, 2010."))
    # The split finds feedback but nothing wrong in it, as for a follow-up
    # question: the turn question's verdict stands (a part that says something
    # is wrong after a plain answer would wait like a change wanted once).
    reader = Guessing("Always use the public availability date, not the announcement date.", {"standing": "no", "kind": kind, "wish": ""})
    ask = reader.ask

    async def nothing_wrong(prompt, schema, name):
        answer = await ask(prompt, schema, name)
        if name == "classify":
            answer = {"parts": [dict(p, says_something_is_wrong=False) for p in answer["parts"]]}
        return answer

    reader.ask = nothing_wrong
    read(store, reader)
    guess = next(s for s in reading.signals(store) if s["type"] == "correction")
    assert (guess["lesson"], guess["short"], guess["unvoiced"]) == (None, None, True)
    from songe.practice.store import lesson_signals
    assert lesson_signals(store) == []  # nothing to practise


def test_a_lesson_in_the_users_words_stays(store):
    record_chat(store, ("Write a bash one-liner that counts the lines in every .py file here.", "Sure! Here it is: ..."),
                ("Just the code.", "find . -name '*.py' | xargs wc -l"))
    read(store, Guessing("Write just the code."))
    kept = next(s for s in reading.signals(store) if s["type"] == "correction")
    assert kept["lesson"] == kept["short"] == "Write just the code." and "unvoiced" not in kept


def test_a_lesson_about_what_the_agent_did_need_not_be_in_the_users_words(store):
    # After tool calls the lesson is about the agent's steps: "No, you changed
    # the wrong file." says nothing of tests, and the advice stays.
    record_session(store)
    read(store, Guessing("Check which file the failing test covers before editing."))
    advice = next(s for s in reading.signals(store) if s["type"] == "correction")
    assert advice["lesson"].startswith("Check which file") and "unvoiced" not in advice


def test_a_wish_the_model_invented_is_stored_in_the_users_own_words(store):
    said = 'Always start your reply with "Yes, chef."'
    record_chat(store, ("What's a good name for a cat?", "Whiskers."), (said, "Yes, chef. Whiskers."))
    read(store, Guessing("Always confirm the direct answer first.", {"standing": "yes", "kind": "once", "wish": "Speak French."}))
    wish = next(s for s in reading.signals(store) if s["type"] == "correction")
    assert wish["turn_kind"] == "lasting" and wish["lesson"] == wish["short"] == said


def test_voiced_is_the_share_of_content_words_the_user_wrote():
    from songe.reading.rules import voiced

    assert voiced("Answer in Russian.", "i want in russian") == 1
    assert voiced("Do not apologize.", "can you stop apologizing all the time") == 1  # endings ignored
    assert voiced("Use Polars, not pandas.", "no pandas, polars only") == 1
    assert voiced("Speak French.", 'Always start your reply with "Yes, chef."') == 0
    assert voiced("Always use the public availability date, not the announcement date.", "Date as 2007-06-29 please.") == 0.25
    assert voiced("Always include the requested phrase exactly as the user named it.", "You forgot the 'Yes, chef.'") == 0
    assert voiced("Answer in Chinese.", "请用中文回答") == 1  # words cannot show it
    assert voiced("Answer.", "restart the duel") == 1  # no content word to compare


def test_words_the_user_put_in_quotation_marks_are_not_cut_off(store):
    """Live (Qwen3.5-4B on vLLM, structured output): a JSON string ends at the
    first double quote the model does not escape, and it copied the user's."""
    said = 'Always start your reply with "Yes, chef."'

    def as_json_string(text):
        return text.split('"')[0]  # what structured output leaves of it

    class Copying(FakeReader):
        async def ask(self, prompt, schema, name):
            answer = await super().ask(prompt, schema, name)
            if name == "classify":
                text = prompt.rsplit("New user message:\n", 1)[1].split("\n")[0]
                return {"parts": [{"says_something_is_wrong": True, "kind": "feedback", "type": "correction",
                                   "text": as_json_string(text)}]}
            if name == "turn":
                text = prompt.rsplit("The user's next message:\n", 1)[1].split("\n")[0]
                return {"standing": "yes", "kind": "once", "wish": as_json_string(text)}
            return answer

    record_session(store, reply_text=said)
    read(store, Copying())
    wish = next(s for s in reading.signals(store) if s["type"] == "correction" and s.get("turn_kind"))
    assert wish["text"] == said  # the user's exact words, with their marks
    assert wish["lesson"] == "Always start your reply with 'Yes, chef.'"


def test_what_the_turn_was_is_kept_with_its_signals(store):
    record_session(store, reply_text="make it shorter")

    class Once(FakeReader):
        async def ask(self, prompt, schema, name):
            return {"standing": "no", "kind": "once"} if name == "turn" else await super().ask(prompt, schema, name)

    read(store, Once())
    correction = next(s for s in reading.signals(store) if s["type"] == "correction")
    assert correction["turn_kind"] == "once"


def test_a_rule_the_user_states_is_captured_even_if_the_model_misses_it(store):
    """Live run: "Never add comments to the code unless I ask for them." gave no
    signal when the model saw no mistake in the turn before it."""
    rule = "Never add comments to the code unless I ask for them."

    class Unimpressed(FakeReader):
        # The model reads the reply as a question: no feedback signal.
        async def ask(self, prompt, schema, name):
            if name == "classify":
                text = prompt.rsplit("New user message:\n", 1)[1].split("\n")[0]
                return {"parts": [{"says_something_is_wrong": False, "kind": "question", "type": None, "text": text}]}
            return await super().ask(prompt, schema, name)

    record_session(store, reply_text=f"No, you changed the wrong file. {rule}")
    assert read(store, Unimpressed()) == 1
    rules = [s for s in reading.signals(store) if s.get("from") == "rules" and s["source"] == "user"]
    assert [(s["type"], s["text"], s["generality"]) for s in rules] == [("correction", rule, "always")]
    assert rules[0]["short"] == rules[0]["lesson"] == rule
    # It points at the step it is about, so a Deep Dream practices it too.
    from songe.practice.store import wanted
    assert rules[0]["evidence_steps"] and rules[0]["exchanges"]
    assert rules[0]["id"] in {s["id"] for s, _ in wanted(store, 1)}


def test_a_rule_the_model_already_found_is_not_added_twice(store):
    record_session(store, reply_text="No, you changed the wrong file. Never add comments to the code unless I ask for them.")
    assert read(store, FakeReader()) == 1  # its correction carries the user's words, rule included
    user = [s for s in reading.signals(store) if s["source"] == "user"]
    assert len(user) == 1 and user[0].get("from") != "rules"


def test_a_grown_session_is_read_again_and_replaces_its_signals(store):
    record_session(store)
    read(store, FakeReader())
    first = {s["id"] for s in reading.signals(store)}
    record_session(store, more=True)  # the same session, one turn longer
    assert len(reading.pending(store)) == 1
    read(store, FakeReader())
    again = reading.signals(store)
    assert {s["id"] for s in again} >= first
    assert len(again) == len({s["id"] for s in again})  # replaced, not duplicated


def test_a_stopped_reading_saves_nothing_and_resumes(store):
    record_session(store)
    assert read(store, FakeReader(stop_after=1)) == 0
    assert reading.signals(store) == []
    assert len(reading.pending(store)) == 1
    assert read(store, FakeReader()) == 1


def test_forget_deletes_what_reading_found(store):
    record_session(store)
    read(store, FakeReader())
    with store.connect() as db:
        db.execute("INSERT INTO reading_cache VALUES ('k','lesson','{}',0)")
    store.forget()
    with store.connect() as db:
        for table in ("signals", "read_sessions", "reading_cache"):
            assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_a_new_learner_reads_only_after_its_cutoff(config, store, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("songe.models.registry.time.time", lambda: clock[0])
    registry.migrate(config, store)
    registry.observe(config, store, "other")  # learning "new", cutoff 100
    learner = store.scoped("other")
    record_session(store, started=50.0)
    with store.connect() as db:
        db.execute("UPDATE exchanges SET recorded_at=50")
    assert reading.pending(learner) == []
    record_session(store, more=True, started=150.0)
    with store.connect() as db:
        db.execute("UPDATE exchanges SET recorded_at=150 WHERE recorded_at IS NULL OR recorded_at>50")
    with store.connect() as db:
        late = {r[0] for r in db.execute("SELECT id FROM exchanges WHERE recorded_at=150")}
    pending = reading.pending(learner)
    # The same session, read from its exchanges after the cutoff only.
    assert len(pending) == 1 and set(pending[0][1]) == late
    registry.set_learning(store, "other", "off")
    assert reading.pending(learner) == []


def test_the_client_falls_back_when_structured_output_is_refused(store):
    bodies = []

    def server(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "response_format" in body:
            return httpx.Response(400, json={"error": "unsupported"})
        return httpx.Response(200, json={"choices": [{"message": {"content": 'Sure: {"steps": [1], "reason": "r"}'}}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(server))
    reader = Reader(store, "http://x/v1", "m", client=client)

    async def twice():
        first = await reader.ask("q", {"type": "object"}, "responsible")
        second = await reader.ask("q", {"type": "object"}, "responsible")
        return first, second

    first, second = asyncio.run(twice())
    assert first == second == {"steps": [1], "reason": "r"}
    assert len(bodies) == 2  # refused once, asked plainly once; then cached
    assert reader.cached == 1


def test_the_client_stays_structured_after_a_single_off_schema_200(store, caplog):
    """An enforcing server (vLLM, llama.cpp, LM Studio, Ollama) that returns
    one off-schema 200 answer gets that single question retried plainly, while
    structured output stays on and resets after a conforming answer."""
    import logging

    schema = {
        "type": "object",
        "properties": {
            "steps": {"type": "array", "items": {"type": "integer"}},
            "reason": {"type": "string"},
        },
        "required": ["steps", "reason"],
    }
    bodies = []
    structured_calls = 0

    def server(request):
        nonlocal structured_calls
        body = json.loads(request.content)
        bodies.append(body)
        if "response_format" in body:
            structured_calls += 1
            # 1st and 3rd structured calls return off-schema 200; 2nd conforms and resets counter.
            if structured_calls in (1, 3):
                return httpx.Response(
                    200,
                    json={
                        "choices": [{"finish_reason": "stop", "message": {"content": ""}}],
                        "usage": {"completion_tokens": 5},
                    },
                )
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": '{"steps": [1], "reason": "ok"}'}}
                    ],
                    "usage": {"completion_tokens": 10},
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"finish_reason": "stop", "message": {"content": '{"steps": [2], "reason": "retried"}'}}
                ],
                "usage": {"completion_tokens": 15},
            },
        )

    reader = Reader(
        store,
        "http://x/v1",
        "m-vllm",
        server="vllm",
        client=httpx.AsyncClient(transport=httpx.MockTransport(server)),
    )

    async def run():
        a = await reader.ask("q1", schema, "responsible")
        b = await reader.ask("q2", schema, "responsible")
        c = await reader.ask("q3", schema, "responsible")
        return a, b, c

    with caplog.at_level(logging.INFO, logger="songe.reading.client"):
        a, b, c = asyncio.run(run())
    assert a == c == {"steps": [2], "reason": "retried"}
    assert b == {"steps": [1], "reason": "ok"}
    assert reader.structured is True
    assert getattr(reader, "errors", 0) == 0
    assert not any("reader_no_structured" in r.getMessage() for r in caplog.records)
    # q1: structured (off-schema) + plain retry; q2: structured (ok, resets); q3: structured (off-schema) + plain retry.
    assert [bool("response_format" in b) for b in bodies] == [True, False, True, True, False]
    assert reader.tokens_out == 5 + 15 + 10 + 5 + 15


@pytest.mark.parametrize(
    "ignored_content",
    [
        "Step 2 caused the error.",
        json.dumps({"steps": ["not-an-int"], "reason": "wrong type"}),
    ],
)
def test_the_client_switches_to_plain_after_two_off_schema_200s(store, caplog, ignored_content):
    """A server that ignores response_format with 200 twice in a row switches
    to plain schema-in-prompt mode, logs once, and stays plain afterwards."""
    import logging

    schema = {
        "type": "object",
        "properties": {
            "steps": {"type": "array", "items": {"type": "integer"}},
            "reason": {"type": "string"},
        },
        "required": ["steps", "reason"],
    }
    bodies = []

    def server(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "response_format" in body:
            return httpx.Response(
                200,
                json={
                    "choices": [{"finish_reason": "stop", "message": {"content": ignored_content}}],
                    "usage": {"completion_tokens": 12},
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"finish_reason": "stop", "message": {"content": '{"steps": [2], "reason": "ok"}'}}
                ],
                "usage": {"completion_tokens": 15},
            },
        )

    reader = Reader(
        store,
        "http://x/v1",
        f"m-{len(ignored_content)}",
        client=httpx.AsyncClient(transport=httpx.MockTransport(server)),
    )

    async def run():
        a = await reader.ask("q1", schema, "responsible")
        b = await reader.ask("q2", schema, "responsible")
        c = await reader.ask("q3", schema, "responsible")
        return a, b, c

    with caplog.at_level(logging.INFO, logger="songe.reading.client"):
        a, b, c = asyncio.run(run())
    assert a == b == c == {"steps": [2], "reason": "ok"}
    assert reader.structured is False
    assert getattr(reader, "errors", 0) == 0
    # q1: structured + plain; q2: structured (2nd miss -> switch) + plain; q3: plain directly.
    assert [bool("response_format" in x) for x in bodies] == [True, False, True, False, False]
    assert "The JSON must match this schema:" in bodies[1]["messages"][0]["content"]
    assert "The JSON must match this schema:" in bodies[4]["messages"][0]["content"]
    switch_logs = [r.getMessage() for r in caplog.records if "reader_no_structured" in r.getMessage()]
    assert len(switch_logs) == 1
    assert reader.tokens_out == 12 + 15 + 12 + 15 + 15


def test_mlx_reader_starts_in_plain_mode(store):
    """Live (mlx_lm.server 0.32.2): mlx-lm ignores response_format with 200, so
    Reader(server='mlx') starts in plain schema-in-prompt mode."""
    bodies = []

    def server(request):
        body = json.loads(request.content)
        bodies.append(body)
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop", "message": {"content": '{"steps": [1], "reason": "r"}'}}]},
        )

    reader = Reader(
        store,
        "http://x/v1",
        "m-mlx",
        server="mlx",
        client=httpx.AsyncClient(transport=httpx.MockTransport(server)),
    )
    assert reader.structured is False
    ans = asyncio.run(reader.ask("q", {"type": "object"}, "responsible"))
    assert ans == {"steps": [1], "reason": "r"}
    assert len(bodies) == 1 and "response_format" not in bodies[0]
    assert "The JSON must match this schema:" in bodies[0]["messages"][0]["content"]


def test_the_client_asks_for_a_plain_reply_the_way_a_user_does(store):
    """reply(): the served model's next message in a conversation, with no
    schema and none of the reasoning switches the reader's questions set."""
    sent = []

    def server(request):
        sent.append(json.loads(request.content))
        message = {"role": "assistant", "content": "uv add numpy"}
        if len(sent) == 2:
            message["reasoning"] = "The user wants uv."
        return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": "stop"}]})

    reader = Reader(store, "http://x/v1", "m", server="vllm",
                    client=httpx.AsyncClient(transport=httpx.MockTransport(server)))
    messages = [{"role": "user", "content": "How do I add numpy?"}]
    first = asyncio.run(reader.reply(messages, max_tokens=384))
    assert first == {"content": "uv add numpy", "finished": True, "reasoned": False}
    assert sent[0] == {"model": "m", "messages": messages, "temperature": 0, "max_tokens": 384, "stream": False}
    # A server that reasons first says so (either field name, or <think> text).
    assert asyncio.run(reader.reply(messages, max_tokens=384))["reasoned"] is True
    away = Reader(store, "http://x/v1", "m", still_idle=lambda: False, client=reader.client)
    try:
        asyncio.run(away.reply(messages, max_tokens=384))
    except Stopped:
        return
    raise AssertionError("expected Stopped")


def test_the_client_stops_when_the_user_returns(store):
    reader = Reader(store, "http://x/v1", "m", still_idle=lambda: False,
                    client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500))))
    try:
        asyncio.run(reader.ask("q", {"type": "object"}, "lesson"))
    except Stopped:
        return
    raise AssertionError("expected Stopped")


ANSWERS = {
    "classify": {"parts": [{"says_something_is_wrong": True, "kind": "feedback",
                            "type": "correction", "text": "No, you changed the wrong file."}]},
    "turn": {"standing": "yes", "kind": "once", "wish": ""},
    "responsible": {"steps": [2], "reason": "wrong file"},
    "lesson": {"lesson": "Check the file first.", "short": "Check the file first.",
               "instance": "x.py", "generality": "kind", "verifiable": None},
    "review": {"signals": []},
    "dream": None,  # filled below
    "matters": {"why": "it is about this file", "answer": "A"},
}


def args(**given):
    return {k: given.get(k, "") for k in ("path", "pattern", "command", "code", "content")}


# A scene that builds: one file, the agent reads it, then runs the tests.
SCENE = {
    "mistake": "edit tax.py without checking which test fails",
    "right_action": "read test_tax.py first",
    "moment": {"kind": "tool_result",
               "step": {"say": "", "tool": "shell", "args": args(command="pytest -q"),
                        "output": "F.\n1 failed, 1 passed"},
               "agent_said": "", "user_message": ""},
    "project": "invoice-calc",
    "files": [{"path": "tax.py", "content": "def vat(x):\n    return x * 0.2\n"}],
    "request": "The VAT total is off by a cent, please fix it.",
    "steps": [{"say": "Looking at the code.", "tool": "read", "args": args(path="tax.py"), "output": ""}],
}
ANSWERS["dream"] = SCENE


def scheduler_for(config, store, monkeypatch, loaded):
    from types import SimpleNamespace

    from songe import schedule
    from songe.reading import client as reading_client

    asked = []

    def server(request):
        body = json.loads(request.content)
        name = body["response_format"]["json_schema"]["name"]
        asked.append((body["model"], name))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(ANSWERS[name])}}]})

    monkeypatch.setattr(
        reading_client.httpx, "AsyncClient",
        lambda **k: REAL_CLIENT(transport=httpx.MockTransport(server), **k),
    )

    async def listing():
        return [{"identifier": i, "type": "llm"} for i in loaded]

    memory = SimpleNamespace(pressure="normal")
    scheduler = schedule.ModelScheduler(
        config, store, SimpleNamespace(listing=listing),
        platform=SimpleNamespace(memory=lambda: memory),
    )
    return scheduler, asked


async def test_idle_reading_uses_the_learners_own_loaded_model(config, store, monkeypatch):
    config.learning = "dreams"
    registry.migrate(config, store)
    registry.observe(config, store, "learner")
    registry.set_learning(store, "learner", "all")
    # The correction in two conversations: wanted once, it is a lesson to
    # practise only when it comes up again.
    record_session(store)
    record_session(store, ask="fix the other failing test")
    learner = store.scoped("learner")
    scheduler, asked = scheduler_for(config, store, monkeypatch, loaded=["someone-else"])
    assert await scheduler.read_sessions() == 0  # its own model is not loaded
    assert asked == []
    scheduler, asked = scheduler_for(config, store, monkeypatch, loaded=["learner"])
    from songe.practice import store as practice

    # Short pauses only read; practice is Deep Dream's first step.
    assert await scheduler.read_sessions() == 2
    assert practice.counts(learner)["written"] == 0
    # Deep: 3 scenes for its one lesson (the correction, said twice).
    assert await scheduler.read_sessions(practice=True) == 3
    assert {model for model, _ in asked} == {"learner"}
    assert {s["type"] for s in reading.signals(learner)} == {"correction", "success"}
    assert reading.pending(learner) == []
    assert practice.counts(learner)["written"] == 3


async def test_a_deep_dreams_reading_writes_the_notes_before_practice(config, store, monkeypatch):
    """Only a Quick Dream wrote the prompt lessons. A Deep Dream that read
    conversations no Quick Dream had read (pressed early, or a backlog) wrote
    practice and trained, and its version was checked against a prompt that
    did not hold those lessons yet (live: 8 notes for four of eight habits;
    the rest arrived with the next Quick Dream, after the gate)."""
    from songe import quick

    config.learning = "dreams"
    registry.migrate(config, store)
    registry.observe(config, store, "learner")
    registry.set_learning(store, "learner", "all")
    # The same correction in two conversations: a lesson for the prompt.
    record_session(store, ask="fix the other failing test")
    record_session(store)
    learner = store.scoped("learner")
    scheduler, _ = scheduler_for(config, store, monkeypatch, loaded=["learner"])
    assert quick.active(learner) == []
    assert await scheduler.read_sessions(practice=True) > 2  # two conversations read, scenes written
    assert [lesson["text"] for lesson in quick.active(learner)] == ["Check the file first."]
    # Nothing new was read since: the lessons stand, no second entry.
    history = len(learner.get(quick.HISTORY) or [])
    await scheduler.read_sessions(practice=True, force=True)
    assert len(learner.get(quick.HISTORY) or []) == history


async def test_idle_reading_is_off_under_hindsight_learning(config, store, monkeypatch):
    registry.migrate(config, store)
    registry.observe(config, store, "learner")
    registry.set_learning(store, "learner", "all")
    record_session(store)
    config.learning = "hindsight"
    scheduler, asked = scheduler_for(config, store, monkeypatch, loaded=["learner"])
    assert await scheduler.read_sessions() == 0 and asked == []


def test_a_compacted_session_stays_one_session_and_titles_are_skipped(store):
    from songe.reading import timeline

    record_session(store)
    with store.connect() as db:
        last = db.execute("SELECT MAX(id) FROM exchanges").fetchone()[0]
        history = request_of(db, last)["messages"]
    # The agent compacts: its request still carries the history.
    summary = "## Objective The user wants the failing test fixed in x.py."
    store.record("/v1/chat/completions",
                 {"messages": history + [{"role": "user", "content": "You MUST summarize the conversation above."}]},
                 reply({"role": "assistant", "content": summary}))
    # It continues from the summary: a new root for prefix linking.
    restart = [SYSTEM, {"role": "user", "content": "<conversation-checkpoint>\nEarlier conversation:\n" + summary}]
    store.record("/v1/chat/completions", {"messages": restart, "tools": TOOLS},
                 reply({"role": "assistant", "content": "", "tool_calls": [call("d", "shell", command="pytest")]}))
    store.record("/v1/chat/completions", {"messages": restart + [
        {"role": "assistant", "content": "", "tool_calls": [call("d", "shell", command="pytest")]},
        {"role": "tool", "tool_call_id": "d", "content": "3 passed"}], "tools": TOOLS},
        reply({"role": "assistant", "content": "All green."}))
    # A session title request is its own root and carries no work.
    store.record("/v1/chat/completions",
                 {"messages": [{"role": "system", "content": "You name conversations."}, {"role": "user", "content": "fix the failing test"}]},
                 reply({"role": "assistant", "content": "Fix failing test"}))
    store.reconstruct()
    with store.connect() as db:
        found = timeline.sessions(db)
        total = db.execute("SELECT COUNT(*) FROM exchanges").fetchone()[0]
    assert len(found) == 1
    (ids,) = found.values()
    assert len(ids) == total - 1  # everything but the title


async def test_quick_dream_reads_then_puts_the_lessons_in_the_prompt(config, store, monkeypatch):
    from songe import quick

    config.learning = "dreams"
    registry.migrate(config, store)
    registry.observe(config, store, "learner")
    registry.set_learning(store, "learner", "all")
    record_session(store)
    learner = store.scoped("learner")
    # The user stated a rule; the served model consolidates the candidates.
    monkeypatch.setitem(ANSWERS, "classify", {"parts": [{
        "says_something_is_wrong": True, "kind": "feedback", "type": "correction",
        "text": "No, you changed the wrong file. Always keep the old entry point working."}]})
    monkeypatch.setitem(ANSWERS, "consolidate", {"lessons": [
        {"text": "Keep the old entry point working when renaming.", "from": [1]}]})
    monkeypatch.setitem(ANSWERS, "review_lesson", {"scope": "general", "covered_by": 0})
    scheduler, asked = scheduler_for(config, store, monkeypatch, loaded=["learner"])
    # Pressed Quick Dream: reads the newest conversation, then the lessons.
    assert await scheduler.quick_now("learner") == 1
    await scheduler.dream_task
    lessons = quick.active(learner)
    # The user's own rule sentence, verbatim (not the model's wording).
    assert [lesson["text"] for lesson in lessons] == ["Always keep the old entry point working."]
    # A rule the user stated is kept as it is: no rewrite by the model.
    assert lessons[0]["signals"] and ("learner", "consolidate") not in asked
    # Consolidation is off by default (Granite echoed and softened lessons).
    assert learner.get("quick_last")["lessons"] == 1 and not learner.get("quick_last")["consolidated"]
    # No practice in a Quick Dream.
    from songe.practice import store as practice

    assert practice.counts(learner)["written"] == 0
    # In idle, a Quick Dream runs only when the signals changed.
    assert await scheduler.quick_dream("learner", idle=True) is None
    # Pressed again with nothing new: the published list stands, no new call.
    calls = len(asked)
    assert await scheduler.quick_dream("learner") is None and len(asked) == calls
    # Pressing Quick then says there is nothing new, and the dashboard keeps a line.
    assert await scheduler.quick_now("learner") == "nothing"
    assert learner.get("quick_checked")["nothing_new"] is True
    from songe.wording import quick_message
    assert quick_message(None, nothing_since=learner.get("quick_last")["at"]).startswith("Nothing new since your last Quick Dream at ")


async def test_quick_dream_under_memory_pressure_says_so_instead_of_started(config, store, monkeypatch):
    """Reported by Jack in a test on 11-24: on a Mac at memory pressure
    "warning", pressing Quick Dream said "Quick Dream started" and reading
    returned at once with nothing read, and the page never said why."""
    from pathlib import Path

    from songe import quick

    config.learning = "dreams"
    registry.migrate(config, store)
    registry.observe(config, store, "learner")
    registry.set_learning(store, "learner", "all")
    record_session(store)
    learner = store.scoped("learner")
    scheduler, asked = scheduler_for(config, store, monkeypatch, loaded=["learner"])
    scheduler.platform.memory().pressure = "warning"
    assert await scheduler.quick_now("learner") == "memory"
    assert getattr(scheduler, "dream_task", None) is None and asked == []
    assert learner.get("quick_checked")["error"] == "memory"
    assert (learner.get(quick.HISTORY) or [])[-1]["error"] == "memory"
    words = json.loads((Path(quick.__file__).parent / "ui" / "wording.json").read_text(encoding="utf-8"))
    assert "memory" in words["quick_failed"] and "memory" in words["quick_failed_card"]
    # Once memory is back to normal the same press reads.
    scheduler.platform.memory().pressure = "normal"
    assert await scheduler.quick_now("learner") == 1


async def test_quick_dream_with_only_new_signals_under_memory_pressure_says_so(config, store, monkeypatch):
    """Nothing unread, but signals since the last Quick Dream: the press would
    publish lessons, and reading waits under memory pressure all the same."""
    config.learning = "dreams"
    registry.migrate(config, store)
    registry.observe(config, store, "learner")
    registry.set_learning(store, "learner", "all")
    learner = store.scoped("learner")
    learner.set("quick_seen", [999, 999])  # differs from what is there now
    scheduler, asked = scheduler_for(config, store, monkeypatch, loaded=["learner"])
    scheduler.platform.memory().pressure = "warn"
    assert await scheduler.quick_now("learner") == "memory"
    assert getattr(scheduler, "dream_task", None) is None and asked == []
    assert learner.get("quick_checked")["error"] == "memory"


def pressed_setup(config, store, monkeypatch, tmp_path, upstream, *, available=90_000, ollama=None):
    """A learner with one unread conversation and a base version, a scheduler
    on `upstream`, the reader's answers faked; `ollama` answers /api/*."""
    from types import SimpleNamespace

    from songe import schedule
    from songe.platform import Memory
    from songe.upstream import protection

    config.learning = "dreams"
    registry.migrate(config, store)
    registry.observe(config, store, "learner")
    registry.set_learning(store, "learner", "all")
    record_session(store)
    learner = store.scoped("learner")
    learner.add_version({"id": "base-learner", "identifier": "learner", "artifact": str(tmp_path / "model"),
                         "adapter": None, "created": 0, "night": 0, "kind": "gguf-full"})
    learner.swap("base-learner", None)
    asked = []

    def route(request):
        if request.url.path.startswith("/api/"):
            return ollama(request)
        if request.method == "GET":
            return httpx.Response(200, json={"data": []})
        body = json.loads(request.content)
        name = (body.get("response_format") or {}).get("json_schema", {}).get("name")
        asked.append((body["model"], name))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(ANSWERS.get(name, {}))}}]})

    monkeypatch.setattr(httpx, "AsyncClient", lambda **k: REAL_CLIENT(**(k | {"transport": httpx.MockTransport(route)})))
    memory = SimpleNamespace(
        name="test", memory=lambda: Memory(100_000, available), accelerators=lambda: [],
        on_ac_power=lambda: True, idle_seconds=lambda: None, thermal=lambda: None,
        process_bytes=lambda pid: 100,
    )
    monkeypatch.setattr(protection, "current", lambda: memory)
    pressure = SimpleNamespace(pressure="normal")
    scheduler = schedule.ModelScheduler(config, store, upstream, platform=SimpleNamespace(memory=lambda: pressure))
    return scheduler, learner, asked


def fake_ollama(tmp_path, running, hooks=None):
    """Ollama's API as Songe uses it: /api/ps lists what is in memory, an
    empty /api/generate loads (keep_alive 0 unloads). hooks: async callables
    run on a "show" or "generate" request; "keep" in hooks keeps a model
    listed after keep_alive 0 (the agent's request loaded it again)."""
    import hashlib
    from types import SimpleNamespace

    from songe.upstream.ollama import Ollama

    hooks = {} if hooks is None else hooks
    (tmp_path / "model").write_bytes(b"gguf weights")
    digest = hashlib.sha256(b"gguf weights").hexdigest()
    generated = []

    async def handler(request):
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": n, "model": n} for n in running]})
        if request.url.path == "/api/show":
            if "show" in hooks:
                await hooks["show"]()
            return httpx.Response(200, json={"modelfile": f"FROM /blobs/sha256-{digest}"})
        if request.url.path == "/api/generate":
            body = json.loads(request.content)
            generated.append(body)
            if "generate" in hooks:
                await hooks["generate"](body)
            if body.get("keep_alive") == 0:
                if "keep" not in hooks:
                    running[:] = [n for n in running if n != body["model"] + ":latest"]
            elif body["model"] + ":latest" not in running:
                running.append(body["model"] + ":latest")
            return httpx.Response(200, json={"done": True})
        return httpx.Response(404)

    upstream = Ollama(SimpleNamespace(upstream_url="http://ollama.test/v1", upstream_timeout=5,
                                      guard_min_interval=0.01, guard_grace_seconds=1))
    return upstream, handler, generated


async def settle(scheduler):
    """The pressed Quick Dream and its release (its own task) are done."""
    if getattr(scheduler, "dream_task", None):
        await scheduler.dream_task
    await asyncio.gather(*scheduler.release_tasks.values())


async def test_a_pressed_quick_dream_loads_the_ollama_model_and_releases_it(config, store, monkeypatch, tmp_path):
    """Background reading never loads a model; a pressed Quick Dream loads the
    learner's model with Ollama's own load and releases it once it is done."""
    running = []
    upstream, handler, generated = fake_ollama(tmp_path, running)
    scheduler, learner, asked = pressed_setup(config, store, monkeypatch, tmp_path, upstream, ollama=handler)
    assert await scheduler.read_sessions() == 0  # background: waits, loads nothing
    assert generated == [] and asked == []
    assert await scheduler.quick_now("learner") == 1
    # Loaded with Ollama's empty-prompt warm, no keep_alive of Songe's own.
    assert generated[0] == {"model": "learner", "prompt": "", "stream": False}
    await settle(scheduler)
    assert ("learner", "classify") in asked  # it read with the learner's model
    # Released after this Quick Dream, confirmed gone from /api/ps.
    assert generated[-1]["keep_alive"] == 0 and running == []
    assert scheduler.press_loads == {} and upstream.owned == {}


async def test_a_pressed_quick_dream_never_releases_the_users_ollama_model(config, store, monkeypatch, tmp_path):
    running = ["learner:latest"]  # the user loaded it
    upstream, handler, generated = fake_ollama(tmp_path, running)
    scheduler, learner, asked = pressed_setup(config, store, monkeypatch, tmp_path, upstream, ollama=handler)
    assert await scheduler.quick_now("learner") == 1
    await settle(scheduler)
    assert ("learner", "classify") in asked
    assert generated == [] and running == ["learner:latest"]
    # Even a load request reuses the user's instance and never claims it.
    assert await scheduler.load_for_press("learner") is None
    assert scheduler.press_loads == {}
    await scheduler.release_press_load("learner")
    assert generated == [] and running == ["learner:latest"]


async def test_a_model_the_agent_asked_for_is_left_to_the_server(config, store, monkeypatch, tmp_path):
    """Loaded for a press, then the agent asked for it (and then for another
    model): it is the user's now, and Ollama's own keep-alive decides."""
    running = []
    upstream, handler, generated = fake_ollama(tmp_path, running)
    scheduler, learner, asked = pressed_setup(config, store, monkeypatch, tmp_path, upstream, ollama=handler)
    assert await scheduler.load_for_press("learner") is None and running == ["learner:latest"]
    scheduler.traffic()
    await scheduler.ensure_model("learner", learner.version())
    scheduler.traffic()
    scheduler.serving_model = "other"  # the agent moved on to another model
    await scheduler.release_press_load("learner")
    assert running == ["learner:latest"] and all("keep_alive" not in g for g in generated)


async def test_traffic_to_another_model_does_not_keep_a_press_load(config, store, monkeypatch, tmp_path):
    """The agent last served this model before the press, and since then only
    another one: the press's load is still Songe's and is released."""
    running = []
    upstream, handler, generated = fake_ollama(tmp_path, running)
    scheduler, learner, asked = pressed_setup(config, store, monkeypatch, tmp_path, upstream, ollama=handler)
    scheduler.serving_model = "learner"
    assert await scheduler.load_for_press("learner") is None
    scheduler.traffic()  # a request for another model
    await scheduler.release_press_load("learner")
    assert running == [] and generated[-1]["keep_alive"] == 0


async def test_a_release_waits_for_requests_and_readers_and_is_never_dropped(config, store, monkeypatch, tmp_path):
    running = []
    upstream, handler, generated = fake_ollama(tmp_path, running)
    scheduler, learner, asked = pressed_setup(config, store, monkeypatch, tmp_path, upstream, ollama=handler)
    assert await scheduler.load_for_press("learner") is None
    scheduler.acquire("admission")  # a request (for any model) in flight
    scheduler.reading_activity = {"learner": {}}  # and a reader on it
    release = asyncio.create_task(scheduler.release_press_load("learner"))
    await asyncio.sleep(0.05)
    assert running == ["learner:latest"] and "learner" in scheduler.press_loads
    scheduler.release("admission")
    await asyncio.sleep(0.05)
    assert running == ["learner:latest"]  # the reader still reads
    scheduler.reading_activity = {}
    await asyncio.wait_for(release, 2)
    assert running == [] and scheduler.press_loads == {}


async def test_background_reading_never_starts_on_a_model_loaded_for_a_press(config, store, monkeypatch, tmp_path):
    running = []
    upstream, handler, generated = fake_ollama(tmp_path, running)
    scheduler, learner, asked = pressed_setup(config, store, monkeypatch, tmp_path, upstream, ollama=handler)
    assert await scheduler.load_for_press("learner") is None and running == ["learner:latest"]
    assert await scheduler.read_sessions() == 0 and asked == []
    assert await scheduler.reader_for("learner") is None
    assert scheduler.reading_busy()  # learning waits for the press too


async def test_a_press_loads_without_holding_the_agents_lock(config, store, monkeypatch, tmp_path):
    running, loading, go = [], asyncio.Event(), asyncio.Event()

    async def slow(body):
        loading.set()
        await go.wait()

    upstream, handler, generated = fake_ollama(tmp_path, running, {"generate": slow})
    scheduler, learner, asked = pressed_setup(config, store, monkeypatch, tmp_path, upstream, ollama=handler)
    press = asyncio.create_task(scheduler.quick_now("learner"))
    try:
        await asyncio.wait_for(loading.wait(), 2)
        assert not scheduler.lock.locked()
        # A second press while the first loads continues it: no second load.
        assert await asyncio.wait_for(scheduler.quick_now("learner"), 1) is None
    finally:
        go.set()
    assert await press == 1
    await settle(scheduler)
    assert [g.get("keep_alive") for g in generated] == [None, 0]


async def test_a_request_admitted_during_the_release_keeps_the_model(config, store, monkeypatch, tmp_path):
    """A request admitted while Songe releases stops the release with Songe's
    claim kept: one for another model sends it back to wait and release
    later; one for this model leaves the model to the agent."""
    running = []
    hooks = {}
    upstream, handler, generated = fake_ollama(tmp_path, running, hooks)
    scheduler, learner, asked = pressed_setup(config, store, monkeypatch, tmp_path, upstream, ollama=handler)
    assert await scheduler.load_for_press("learner") is None

    async def admitted():
        hooks.pop("show")
        scheduler.acquire("admission")  # a request for another model

    hooks["show"] = admitted  # same_weights, just before the unload
    release = asyncio.create_task(scheduler.release_press_load("learner"))
    await asyncio.sleep(0.05)
    assert running == ["learner:latest"] and all("keep_alive" not in g for g in generated)
    assert "learner" in upstream.owned and "learner" in scheduler.press_loads
    scheduler.release("admission")  # it finished: the release goes on
    await asyncio.wait_for(release, 2)
    assert running == [] and generated[-1]["keep_alive"] == 0 and upstream.owned == {}
    # Admitted after keep_alive 0 was sent, for this model: Ollama loads it
    # again for the agent, and the release stops waiting for /api/ps at once.
    assert await scheduler.load_for_press("learner") is None

    async def agent_back(body):
        if body.get("keep_alive") == 0:
            scheduler.acquire("admission")

    hooks.update({"generate": agent_back, "keep": True})
    started = asyncio.get_running_loop().time()
    release = asyncio.create_task(scheduler.release_press_load("learner"))
    await asyncio.sleep(0.05)
    await scheduler.ensure_model("learner", learner.version())  # the agent's request
    scheduler.release("admission")
    await asyncio.wait_for(release, 2)
    assert asyncio.get_running_loop().time() - started < 2
    assert running == ["learner:latest"] and scheduler.press_loads == {}
    assert "learner" in upstream.owned  # Songe's claim is never just dropped


async def test_the_agent_waits_for_a_press_load_instead_of_loading_again(config, store, monkeypatch, tmp_path):
    """LM Studio: a request for the model while a press loads it used to drop
    the press's pending claim and load a second instance beside it."""
    from tests.test_lmstudio import FakeLMS

    upstream = FakeLMS(tmp_path, appear_after=1)
    upstream.config.upstream_url = "http://lms.test/v1"
    scheduler, learner, asked = pressed_setup(config, store, monkeypatch, tmp_path, upstream)
    folder = tmp_path / "lmstudio-community/granite-4.2-3b-MLX-4bit"
    folder.mkdir(parents=True)
    (folder / "model.safetensors").write_bytes(b"mlx weights")
    with store.connect() as db:
        db.execute("UPDATE versions SET artifact=? WHERE id='base-learner'", (str(folder),))
    loading, go = asyncio.Event(), asyncio.Event()
    request_load = upstream.request_load

    async def slow(key, identifier, context=None):
        loading.set()
        await go.wait()
        await request_load(key, identifier, context)

    upstream.request_load = slow
    press = asyncio.create_task(scheduler.load_for_press("learner"))
    try:
        await asyncio.wait_for(loading.wait(), 2)
        agent = asyncio.create_task(scheduler.ensure_model("learner", learner.version()))
        await asyncio.sleep(0.05)
        assert not agent.done()  # it waits for the press's load
    finally:
        go.set()
    assert await press is None
    await asyncio.wait_for(agent, 2)
    assert [c[0] for c in upstream.calls if c[0] == "load"] == ["load"]
    assert [m["identifier"] for m in upstream.ps] == ["learner"]
    await scheduler.release_press_load("learner")  # the agent's now: kept
    assert len(upstream.ps) == 1 and scheduler.press_loads == {}


async def test_a_press_that_fails_after_loading_releases_the_model(config, store, monkeypatch, tmp_path):
    running = []
    upstream, handler, generated = fake_ollama(tmp_path, running)
    scheduler, learner, asked = pressed_setup(config, store, monkeypatch, tmp_path, upstream, ollama=handler)
    probes = []

    async def reader_for(model):
        probes.append(model)
        if len(probes) > 1:
            raise RuntimeError("the reader could not start")
        return None

    monkeypatch.setattr(scheduler, "reader_for", reader_for)
    with pytest.raises(RuntimeError):
        await scheduler.quick_now("learner")
    await settle(scheduler)
    assert generated[-1]["keep_alive"] == 0 and running == []
    assert scheduler.press_loads == {} and not scheduler.pressing


async def test_a_pressed_quick_dream_without_the_memory_waits_and_says_why(config, store, monkeypatch, tmp_path):
    from songe import quick

    running = []
    upstream, handler, generated = fake_ollama(tmp_path, running)
    scheduler, learner, asked = pressed_setup(
        config, store, monkeypatch, tmp_path, upstream, available=0, ollama=handler)
    answer = await scheduler.dream("learner", "quick")
    assert not answer["started"] and "not enough free memory to load the model" in answer["message"]
    assert generated == [] and running == [] and asked == []
    assert getattr(scheduler, "dream_task", None) is None
    assert learner.get("quick_checked")["error"] == "load_memory"
    assert (learner.get(quick.HISTORY) or [])[-1]["error"] == "load_memory"


async def test_a_model_of_unknown_size_is_not_loaded_for_a_press(config, store, monkeypatch, tmp_path):
    """Admission sizes a load by its weights: weights Songe cannot read (an
    Ollama blob outside its view) would pass as 0 bytes."""
    running = []
    upstream, handler, generated = fake_ollama(tmp_path, running)
    scheduler, learner, asked = pressed_setup(config, store, monkeypatch, tmp_path, upstream, ollama=handler)
    (tmp_path / "model").unlink()
    answer = await scheduler.dream("learner", "quick")
    assert not answer["started"] and "can't measure the model" in answer["message"]
    assert generated == [] and running == [] and scheduler.press_loads == {}
    assert learner.get("quick_checked")["error"] == "load_unknown"


async def test_a_pressed_quick_dream_loads_and_releases_on_lm_studio(config, store, monkeypatch, tmp_path):
    from tests.test_lmstudio import FakeLMS

    upstream = FakeLMS(tmp_path, appear_after=1)
    scheduler, learner, asked = pressed_setup(config, store, monkeypatch, tmp_path, upstream)
    folder = tmp_path / "lmstudio-community/granite-4.2-3b-MLX-4bit"
    folder.mkdir(parents=True)
    (folder / "model.safetensors").write_bytes(b"mlx weights")
    with store.connect() as db:  # LM Studio's own copy of the model
        db.execute("UPDATE versions SET artifact=? WHERE id='base-learner'", (str(folder),))
    assert await scheduler.read_sessions() == 0  # background: waits, loads nothing
    assert upstream.calls == [] and asked == []
    assert await scheduler.quick_now("learner") == 1
    assert [c[0] for c in upstream.calls if c[0] in {"load", "unload"}] == ["load"]
    await settle(scheduler)
    assert ("learner", "classify") in asked
    assert [c[0] for c in upstream.calls if c[0] in {"load", "unload"}] == ["load", "unload"]
    assert upstream.ps == []
    # A request admitted just before the unload keeps the model loaded.
    assert await scheduler.load_for_press("learner") is None
    live_owned = upstream.live_owned

    async def admitted():
        scheduler.acquire("admission")
        return await live_owned()

    upstream.live_owned = admitted
    upstream.calls.clear()
    upstream.ps[0]["contextLength"] = 32768  # Songe's choice, not the user's
    release = asyncio.create_task(scheduler.release_press_load("learner"))
    await asyncio.sleep(0.05)
    upstream.live_owned = live_owned
    assert not any(c[0] == "unload" for c in upstream.calls) and len(upstream.ps) == 1
    # The claim stays Songe's: its context is not taken for the user's.
    await upstream.served_contexts()
    assert "learner" in upstream.owned and "learner" not in upstream.user_contexts
    scheduler.release("admission")  # that request (another model's) is done
    await asyncio.wait_for(release, 2)
    assert upstream.ps == [] and scheduler.press_loads == {}
    # A model the user loaded in LM Studio is read with and never unloaded.
    record_session(store, ask="fix the other failing test")
    upstream.ps.append({"identifier": "learner", "path": "lmstudio-community/granite-4.2-3b-MLX-4bit",
                        "instanceReference": "users"})
    upstream.calls.clear()
    assert await scheduler.quick_now("learner") == 1
    await settle(scheduler)
    assert not any(c[0] in {"load", "unload"} for c in upstream.calls)
    assert [m["identifier"] for m in upstream.ps] == ["learner"]


def test_a_pi_compaction_is_not_a_conversation_and_its_continuation_joins(store):
    """Live demo: pi's summarise calls were read as separate sessions and the
    user's conversation was split from its continuation."""
    from songe.reading import timeline

    ask = {"role": "user", "content": "can you write a 2048 game for me"}
    step = {"role": "assistant", "content": "", "tool_calls": [call("a", "write", filePath="game.js", content="...")]}
    done = {"role": "tool", "tool_call_id": "a", "content": "Wrote game.js"}
    store.record("/v1/chat/completions", {"messages": [SYSTEM, ask], "tools": TOOLS}, reply(step))
    store.record("/v1/chat/completions", {"messages": [SYSTEM, ask, step, done], "tools": TOOLS},
                 reply({"role": "assistant", "content": "The game is in game.js."}))
    summary = ("## Goal The user asked for a 2048 game. ## Progress game.js was written with the board, "
               "merging and scoring; next is keyboard input and a restart button for the finished game.")
    store.record("/v1/chat/completions", {"messages": [
        {"role": "system", "content": "You are a summarizer."},
        {"role": "user", "content": "<conversation>[User]: can you write a 2048 game for me\n[Assistant]: ...</conversation>"
                                    "\nThe messages above are a conversation to summarize. Create a summary."}]},
        reply({"role": "assistant", "content": summary}))
    store.record("/v1/chat/completions", {"messages": [
        SYSTEM, {"role": "user", "content": f"The conversation so far was summarized:\n{summary}"},
        {"role": "user", "content": "now add a restart button"}], "tools": TOOLS},
        reply({"role": "assistant", "content": "Added."}))
    store.reconstruct()
    with store.connect() as db:
        found = timeline.sessions(db)
    assert len(found) == 1  # one conversation: the summariser call joined it, the continuation too
    (ids,) = found.values()
    with store.connect() as db:
        kinds = [timeline.kind_of(request_of(db, i)) for i in ids]
    assert kinds.count("compaction") == 1


def test_signals_of_a_session_merged_away_are_dropped(store):
    record_session(store)
    store.reconstruct()
    assert read(store, FakeReader()) == 1
    with store.connect() as db:
        db.execute("INSERT INTO read_sessions VALUES (?,?,?,?)", ("gone-root", store.model_id, 1, 0.0))
        db.execute("INSERT INTO signals VALUES (?,?,?,?,?,NULL)",
                   ("old", store.model_id, "gone-root", 1, json.dumps({"id": "old", "session": "gone-root"})))
    assert reading.forget_vanished(store) == 1
    assert all(s["session"] != "gone-root" for s in reading.signals(store))
    assert reading.signals(store)  # the live session's signals stay


def test_summarising_an_empty_conversation_is_not_a_conversation(store):
    """Live demo: pi summarised an empty <conversation></conversation>; with no
    session to belong to it was read as a conversation of its own."""
    from songe.reading import timeline

    record_session(store)
    for extra in ("", "<previous-summary> ## Goal User is at the start of a session </previous-summary>"):
        store.record("/v1/chat/completions", {"messages": [
            {"role": "system", "content": "You are a summarizer."},
            {"role": "user", "content": f"<conversation> </conversation> {extra}"
                                        "\nThe messages above are a conversation to summarize. Create a summary."}]},
            reply({"role": "assistant", "content": "## Goal Nothing yet."}))
    store.reconstruct()
    with store.connect() as db:
        found = timeline.sessions(db)
    assert len(found) == 1
    assert read(store, FakeReader()) == 1


def test_turns_after_a_pi_compaction_stay_in_the_conversation(store):
    """Live demo: after compaction pi resends the kept turns with the summary
    inserted before them; those turns had no steps and were in no session, so
    the rule stated there was never read."""
    from songe.reading import timeline

    ask = {"role": "user", "content": "can you write a 2048 game for me"}
    step = {"role": "assistant", "content": "", "tool_calls": [call("a", "write", filePath="2048.py", content="...")]}
    done = {"role": "tool", "tool_call_id": "a", "content": "Wrote 2048.py"}
    finished = {"role": "assistant", "content": "The 2048 game is complete."}
    store.record("/v1/chat/completions", {"messages": [SYSTEM, ask], "tools": TOOLS}, reply(step))
    store.record("/v1/chat/completions", {"messages": [SYSTEM, ask, step, done], "tools": TOOLS}, reply(finished))
    store.record("/v1/chat/completions", {"messages": [
        {"role": "system", "content": "You are a context summarization assistant."},
        {"role": "user", "content": "<conversation> </conversation> The messages above are a conversation to summarize."}]},
        reply({"role": "assistant", "content": "## Goal Nothing yet."}))
    summary = {"role": "user", "content": [{"type": "text", "text":
               "The conversation history before this point was compacted into the following summary: <summary>## Goal</summary>"}]}
    rule = {"role": "user", "content": "always put my name and the date in the top of the script."}
    history = [SYSTEM, summary, ask, step, done, finished, rule]
    header = {"role": "assistant", "content": "I'll add a header."}
    store.record("/v1/chat/completions", {"messages": history, "tools": TOOLS}, reply(header))
    store.record("/v1/chat/completions", {"messages": history + [header, {"role": "user", "content": "My name is Frank Wu."}],
                                          "tools": TOOLS}, reply({"role": "assistant", "content": "Done."}))
    for _ in range(3):
        store.reconstruct()
    with store.connect() as db:
        found = timeline.sessions(db)
    assert len(found) == 1
    (ids,) = found.values()
    assert {1, 2, 4, 5} <= set(ids) and 3 not in ids
    # The agent's summarise prompt, attached to the session it compacts, is not the user speaking.
    store.record("/v1/chat/completions", {"messages": [
        {"role": "system", "content": "You are a context summarization assistant."},
        {"role": "user", "content": "<conversation> [User]: can you write a 2048 game for me </conversation> "
                                    "This is the PREFIX of a turn that was too large to keep."}]},
        reply({"role": "assistant", "content": "## Goal 2048."}))
    store.reconstruct()
    with store.connect() as db:
        (ids,) = timeline.sessions(db).values()
        said = [it["content"] for it in timeline.build(timeline.load(db, ids)) if it["kind"] == "user"]
    assert said == ["can you write a 2048 game for me", rule["content"], "My name is Frank Wu."]


def test_rules_read_before_they_carried_a_step_get_one(store):
    rule = "Never add comments to the code unless I ask for them."
    record_session(store, reply_text=f"No, you changed the wrong file. {rule}")
    assert read(store, FakeReader()) == 1
    with store.connect() as db:
        for sid, payload in db.execute("SELECT id, payload FROM signals").fetchall():
            s = json.loads(payload)
            if s.get("from") == "rules" or s.get("text") == f"No, you changed the wrong file. {rule}":
                s.update({"from": "rules", "evidence_steps": [], "exchanges": {}})
                s.pop("placed", None)
                db.execute("UPDATE signals SET payload=? WHERE id=?", (json.dumps(s), sid))
    assert reading.place_rules(store) >= 1
    rules = [s for s in reading.signals(store) if s.get("from") == "rules"]
    assert all(s["evidence_steps"] and s["exchanges"] and s["placed"] for s in rules)
    assert reading.place_rules(store) == 0  # once each


def test_practice_is_written_for_the_users_rules_first(store):
    from songe.practice.store import wanted

    with store.connect() as db:
        for i, origin in enumerate([None, None, "rules"]):
            # The stated rule in the user's words: only a sentence shaped like a rule is one.
            payload = {"id": f"s{i}", "session": "x", "type": "correction", "source": "user", "lesson": f"Lesson {i}.",
                       "exchanges": {"1": 1}, "revision": 1,
                       **({"from": origin, "text": "Always run the tests before a commit."} if origin else {})}
            db.execute("INSERT INTO signals VALUES (?,?,?,?,?,NULL)", (f"s{i}", store.model_id, "x", 1, json.dumps(payload)))
    assert [s["id"] for s, _ in wanted(store, 1)][0] == "s2"


async def test_a_failing_practice_scene_is_retried_then_skipped_and_only_repeated_failures_stop(store, monkeypatch):
    """Live: the first scene after calibration failed (the model reloading) and
    the whole pressed Deep Dream vanished."""
    import pytest

    from songe.practice import store as ps

    signals_ = [({"id": f"s{i}", "session": "x", "lesson": "L."}, [1]) for i in range(5)]
    monkeypatch.setattr(ps, "RETRY_WAITS", (0, 0))
    monkeypatch.setattr(ps, "wanted", lambda store, n: signals_)
    # An agent's request (it has tools): each scene is written on its own.
    request = {"messages": [{"role": "user", "content": "hi"}], "tools": TOOLS}
    monkeypatch.setattr(ps, "source", lambda store, s: (request, []))
    saved = []
    monkeypatch.setattr(ps, "save", lambda store, row: saved.append(row["id"]))
    calls = {}

    async def flaky(store, reader, signal, v, *rest):
        calls[signal["id"]] = calls.get(signal["id"], 0) + 1
        if signal["id"] == "s0" and calls["s0"] == 1:
            raise RuntimeError("")  # the server still reloading: retried
        if signal["id"] == "s1":
            raise RuntimeError("bad scene")  # always: skipped
        return {"id": signal["id"]}
    monkeypatch.setattr(ps, "write_one", flaky)
    assert await ps.write(store, None, 1) == 4 and saved == ["s0", "s2", "s3", "s4"]
    assert calls["s1"] == 3

    async def down(*args):
        raise RuntimeError("connection refused")
    monkeypatch.setattr(ps, "write_one", down)
    with pytest.raises(ps.PracticeFailed):
        await ps.write(store, None, 1)


async def test_a_pressed_deep_dream_that_cannot_train_leaves_a_note_after_practice(config, store, monkeypatch):
    """Live: Deep pressed with too few test conversations; reading and
    practice ran, then the Dream vanished without a note."""
    config.learning = "dreams"
    registry.migrate(config, store)
    registry.observe(config, store, "learner")
    registry.set_learning(store, "learner", "all")
    record_session(store)
    learner = store.scoped("learner")
    scheduler, _ = scheduler_for(config, store, monkeypatch, loaded=["learner"])
    learner.set("dream_pending_wait", "it needs 3 more conversations with your feedback")
    await scheduler.read_sessions(force=True, only="learner", practice=True)
    stopped = learner.get("dream_stopped")
    assert stopped["reason"] == "training" and "3 more conversations" in stopped["detail"] and "kept" in stopped
    assert learner.get("dream_pending_wait") is None


def test_a_scene_builds_for_pi_whose_tools_are_bash_read_edit_write():
    """Live: pi has bash, read, edit and write; scenes using grep, glob or
    shell were dropped as "could not be built"."""
    from songe.practice.frame import build

    def tool(name, *params):
        return {"type": "function", "function": {"name": name, "parameters": {
            "type": "object", "properties": {p: {"type": "string"} for p in params}, "required": list(params[:1])}}}
    frame = {"system": "pi", "workdir": "/tmp/work/demo",
             "tools": [tool("bash", "command"), tool("read", "path"), tool("edit", "path", "oldText", "newText"),
                       tool("write", "path", "content")]}
    dream = {"request": "rename main", "files": [{"path": "src/app.py", "content": "def main():\n    pass\n"}],
             "steps": [{"tool": "grep", "args": {"pattern": "def main", "path": "src"}, "output": ""},
                       {"tool": "glob", "args": {"pattern": "*.py"}, "output": ""},
                       {"tool": "shell", "args": {"command": "cat src/app.py"}, "output": ""},
                       {"tool": "bash", "args": {"command": "pytest -q"}, "output": "1 passed"}],
             "moment": {"kind": "user_message", "agent_said": "Done.", "user_message": "also update the README"}}
    messages, problems, _ = build(dream, frame)
    assert problems == []
    calls = [json.loads(m["tool_calls"][0]["function"]["arguments"]) for m in messages if m.get("tool_calls")]
    names = [m["tool_calls"][0]["function"]["name"] for m in messages if m.get("tool_calls")]
    assert names == ["bash"] * 4
    assert calls[0]["command"] == "grep -rn 'def main' src" and calls[1]["command"] == "find . -name '*.py'"
    results = [m["content"] for m in messages if m["role"] == "tool"]
    assert results[0] == "src/app.py:1:def main():" and results[1] == "./src/app.py"
    assert results[2].startswith("def main():") and results[3] == "1 passed"


def test_scene_steps_with_their_text_in_the_other_field_still_build():
    """Live: 134 execute steps "missing required argument 'code'" (written as
    command), 47 writes "missing content" (written as code, content empty),
    and steps with tool "none"."""
    from songe.practice.frame import build

    def tool(name, *params):
        return {"type": "function", "function": {"name": name, "parameters": {
            "type": "object", "properties": {p: {"type": "string"} for p in params}, "required": list(params)}}}
    frame = {"system": "s", "workdir": "/tmp/work/demo",
             "tools": [tool("execute", "code"), tool("write", "path", "content"), tool("read", "path")]}
    empty = {"path": "", "pattern": "", "command": "", "code": "", "content": ""}
    dream = {"request": "go", "files": [],
             "steps": [{"tool": "execute", "args": empty | {"command": "print(1)"}, "output": "1"},
                       {"tool": "write", "args": empty | {"path": "a.py", "code": "x = 1"}, "output": ""},
                       {"tool": "none", "args": empty, "output": ""},
                       {"tool": "read", "args": empty | {"path": "a.py"}, "output": ""}],
             "moment": {"kind": "user_message", "agent_said": "ok", "user_message": "thanks"}}
    messages, problems, files = build(dream, frame)
    assert problems == [] and files["a.py"] == "x = 1"
    calls = [json.loads(m["tool_calls"][0]["function"]["arguments"]) for m in messages if m.get("tool_calls")]
    assert calls[0] == {"code": "print(1)"} and calls[1]["content"] == "x = 1" and len(calls) == 3


def test_a_turn_echoed_back_differently_still_links_lm_studio_and_pi(store):
    """Live (Mac, LM Studio + pi): LM Studio replied "\\n\\n" with a tool call;
    pi echoed it as null; every later turn stayed unlinked and Quick Dream
    found nothing new."""
    from songe.reading import timeline

    ask = {"role": "user", "content": "write a snake game"}
    tool_call = {"id": "c1", "type": "function", "function": {"name": "write", "arguments": '{"path": "snake.py"}'}}
    lm_reply = {"role": "assistant", "content": "\n\n", "reasoning_content": "plan", "tool_calls": [dict(tool_call, index=0)]}
    store.record("/v1/chat/completions", {"messages": [SYSTEM, ask], "tools": TOOLS}, reply(lm_reply))
    echoed = {"role": "assistant", "content": None, "reasoning_content": "plan", "tool_calls": [tool_call]}
    done = {"role": "tool", "tool_call_id": "c1", "content": "Wrote snake.py"}
    store.record("/v1/chat/completions", {"messages": [SYSTEM, ask, echoed, done], "tools": TOOLS},
                 reply({"role": "assistant", "content": "Done."}))
    for _ in range(3):
        store.reconstruct()
    with store.connect() as db:
        unresolved = db.execute("SELECT COUNT(*) FROM reconstructed WHERE resolved=0").fetchone()[0]
        found = timeline.sessions(db)
    assert unresolved == 0 and len(found) == 1 and len(next(iter(found.values()))) == 2


def test_an_answer_in_the_reasoning_channel_is_read_and_failures_are_not_cached(store):
    """Live (LM Studio + Granite MLX): content "", the JSON in
    reasoning_content; every reading answer failed and was cached."""
    from songe.reading import client

    class R:
        def __init__(self, message):
            self.message = message

        def json(self):
            return {"choices": [{"message": self.message, "finish_reason": "stop"}]}

    lm = R({"role": "assistant", "content": "", "reasoning_content": '{"signals": [{"step": 22, "type": "outcome"}]}'})
    assert client.read(lm) == {"signals": [{"step": 22, "type": "outcome"}]}
    assert "error" in client.read(R({"role": "assistant", "content": "", "reasoning_content": "thinking…"}))
    # A failure cached by an older version is asked again.
    client.cache_put(store, "k", "classify", {"error": "no JSON (finish: stop)"})
    assert client.cache_get(store, "k") is None
    client.cache_put(store, "k2", "classify", {"parts": []})
    assert client.cache_get(store, "k2") == {"parts": []}


def test_a_session_whose_answers_could_not_be_read_stays_unread(store):
    class Unreadable(FakeReader):
        async def ask(self, prompt, schema, name, **kw):
            self.errors = getattr(self, "errors", 0) + 1
            return {"error": "no JSON (finish: stop)"}

    record_session(store)
    store.reconstruct()
    reader = Unreadable()
    assert read(store, reader) == 0
    assert reader.failed_sessions and len(reading.pending(store)) == 1  # read again next time


def test_a_copied_example_is_no_lesson_and_conditional_rules_are_rules():
    """Live (Granite on LM Studio): every lesson was the prompt's renaming
    example, and "when writing search algorithms, always measure the
    runtime" was read as a new task."""
    from songe.reading.prompts import copied_example
    from songe.reading.rules import rule_sentence

    assert copied_example("When renaming a function, search for every way it is used, including names built as strings")
    assert not copied_example("Measure the runtime when writing search algorithms.")
    assert rule_sentence("when writing search algorithms, always measure the runtime") == \
        "when writing search algorithms, always measure the runtime"
    assert rule_sentence("always run more. 3 numbers don't show anything") == "always run more."
    assert rule_sentence("Use the feedback from each guess.") is None


def test_the_lesson_question_for_the_users_own_words_keeps_what_they_named(store):
    """Live (Qwen3.5-4B on vLLM, 24 chats corrected in everyday words): "I use
    uv, not pip." was stored as "Verify the user's preferred package manager
    before providing instructions.", "In metric please." as "Check the user's
    preferred units before providing the final answer." read as fitting one
    kind of task, and no answer changed. The question now says that a wish in
    the user's own words is the lesson; a tool's result is asked about as before."""
    from songe.reading import prompts as P

    class Recording(FakeReader):
        def __init__(self):
            super().__init__()
            self.questions = []

        async def ask(self, prompt, schema, name):
            if name == "lesson":
                self.questions.append(prompt)
            return await super().ask(prompt, schema, name)

    record_session(store, reply_text="pnpm please.")
    reader = Recording()
    assert read(store, reader) == 1
    (question,) = [q for q in reader.questions if "pnpm please." in q]
    assert P.USER_WISH in question
    assert "user's key words" in P.USER_WISH and '"always"' in P.USER_WISH
    # Feedback from a tool is not the user's words: that question is unchanged.
    plain = P.LESSON.format(task="t", steps="s", feedback="Exit code: 1", reason="r", after="a", wish="")
    assert P.USER_WISH not in plain and "{wish}" not in plain
    assert all(P.USER_WISH not in q for q in reader.questions if "pnpm please." not in q)
    # A shape pasted back with its placeholders is no lesson; a real wish is one.
    assert P.copied_example("Use <THING>, not <OTHER>.")
    assert not P.copied_example("Use uv, not pip.")


def test_a_rule_in_a_message_wrapped_in_quotes_keeps_no_quote():
    """Live (opencode 1.19.6, `opencode run "..."`): the message arrives inside
    quotes; the first rule was missed and the last became the preference
    `Always add dependencies with uv add."`."""
    from songe.reading.rules import rule_sentence

    wrapped = '"Never use pip in this repo. Always add dependencies with uv add."'
    assert rule_sentence(wrapped) == "Never use pip in this repo."
    assert rule_sentence('"Always add dependencies with uv add."') == "Always add dependencies with uv add."
    assert rule_sentence("\u201cAlways run tests with uv run pytest.\u201d") == "Always run tests with uv run pytest."
    # Quotes that belong to the sentence stay.
    assert rule_sentence('Always name the branch "main".') == 'Always name the branch "main".'


def test_opencode_1_19_titles_compactions_and_continue_are_not_the_user():
    # Recorded from opencode 1.19.6 (#15): its title and summary requests and
    # the message it sends after a compaction were read as the user's own, and
    # a Quick Dream took "Do not include the <template> tags" as a preference.
    from songe.reading import timeline

    task = {"role": "user", "content": "List the files, then tell me what app.py prints."}
    title = {"messages": [
        {"role": "system", "content": "You are a title generator. You output ONLY a thread title."},
        task]}
    first = {"messages": [
        {"role": "system", "content": "You are a context summarization agent."},
        {"role": "user", "content": "Here is the conversation so far:\n\n<conversation>\n[User]: hi\n</conversation>\n\n"
         "Create a new anchored summary from the conversation history in the <conversation> tags above so "
         "another coding agent can continue the work.\n\nDo not include the <template> tags in your response."}]}
    update = {"messages": [
        {"role": "system", "content": "You are a context summarization agent."},
        {"role": "user", "content": "Here is the conversation so far:\n\n<conversation>\n[User]: hi\n</conversation>\n\n"
         "<prior-summary>\n## Objective\n- hi\n</prior-summary>\n\nThe <prior-summary> summarizes everything that "
         "happened before the <conversation>. Construct a new summary that combines both."}]}
    assert [timeline.kind_of(r) for r in (title, first, update)] == ["title", "compaction", "compaction"]
    resumed = {"messages": [SYSTEM, task, {"role": "assistant", "content": "Done."}, {"role": "user", "content":
        "Continue if you have next steps, or stop and ask for clarification if you are unsure how to proceed."}]}
    items = timeline.build([(1, resumed, reply({"role": "assistant", "content": "Nothing left."}), 200)])
    assert [i["kind"] for i in items if i["kind"] in {"user", "harness"}] == ["harness"]


async def test_nothing_launches_while_a_dream_is_still_reading(config, store, monkeypatch):
    """Measuring, training and swapping park or replace the model server, so
    none starts while reading or practice still talks to it (live, Inkling on
    Modal: the measuring child started beside a pressed Dream's background
    reading, which then timed out: "stopped while reading · ReadTimeout")."""
    import asyncio

    from songe.platform import Accel
    from songe.schedule import Scheduler

    config.learning = "dreams"
    registry.migrate(config, store)
    registry.observe(config, store, "learner")
    registry.set_learning(store, "learner", "all")
    record_session(store)
    scheduler, _ = scheduler_for(config, store, monkeypatch, loaded=["learner"])
    launched = []

    async def launch(self):
        launched.append(True)

    monkeypatch.setattr(Scheduler, "tick", launch)

    async def nothing():
        return None

    async def chosen():
        return "learner"

    # Past reading, the Dream would measure and train this model.
    monkeypatch.setattr(scheduler, "discover_models", nothing)
    monkeypatch.setattr(scheduler, "next_model", chosen)
    scheduler.platform.accelerators = lambda: [Accel("metal", "stand-in")]
    store.set("requested_sleep", True)
    store.set("requested_model", "learner")
    # A pressed Deep Dream reads in the background.
    reading = asyncio.get_running_loop().create_future()
    scheduler.reading_task = reading
    try:
        await scheduler.tick()
        assert launched == []
        assert scheduler.reason == "Reading your conversations"
    finally:
        reading.cancel()
        scheduler.reading_task = None
    # Once reading is done, the Dream goes on to measure and train.
    assert not scheduler.reading_busy()
    await scheduler.tick()
    assert launched == [True]


async def test_a_dream_carried_on_after_a_restart_reads_with_auto_dream_off(config, store, monkeypatch):
    """A pressed Deep Dream that a restart interrupted mid-read keeps its armed
    request; its background reading is gone, so the scheduler's own reading
    step must read that model even with auto-dream off (live, Inkling on
    Modal: 7 of 12 conversations skipped, no practice, straight to measuring)."""
    from songe.platform import Accel
    from songe.reading import store as reading
    from songe.schedule import Scheduler

    config.learning = "dreams"
    registry.migrate(config, store)
    registry.observe(config, store, "learner")
    registry.set_learning(store, "learner", "all")
    registry.set_auto_dream(store, "learner", False)
    record_session(store)
    learner = store.scoped("learner")
    assert reading.pending(learner)
    scheduler, asked = scheduler_for(config, store, monkeypatch, loaded=["learner"])
    launched = []

    async def launch(self):
        launched.append(len(reading.pending(learner)))

    async def chosen():
        return "learner"

    async def nothing():
        return None

    monkeypatch.setattr(Scheduler, "tick", launch)
    monkeypatch.setattr(scheduler, "next_model", chosen)
    monkeypatch.setattr(scheduler, "discover_models", nothing)
    scheduler.platform.accelerators = lambda: [Accel("metal", "stand-in")]
    # The restart left the request armed and the Dream preparing.
    store.set("requested_model", "learner")
    store.set("requested_sleep", True)
    learner.set("dream_preparing", True)
    await scheduler.tick()
    # It read the conversation itself first, then went on (nothing unread).
    assert asked and launched == [0]


def test_a_failed_bash_command_is_a_failed_result():
    """#132: opencode's and Pi's tool is bash, and its output has no exit code.
    A missing command or module must count as a failure without a review."""
    from songe.reading.rules import classify

    def run(command):
        return {"arguments": json.dumps({"command": command})}

    failures = [
        ("pytest", "/bin/bash: line 1: pytest: command not found\n"),
        ("python -m pytest", "/usr/local/bin/python: No module named pytest\n"),
        ("pip install pytest", "error: externally-managed-environment\n\n× This environment is externally managed\n"),
        ("python scripts/run.py", "Traceback (most recent call last):\n  ...\nModuleNotFoundError: No module named 'kitchen'\n"),
    ]
    for command, body in failures:
        assert classify("bash", body, run(command))[0] == "failed_result", body
    assert classify("bash", "12 passed in 0.4s\n", run("uv run pytest -q")) == ("success", "12 passed")
    assert classify("bash", "README.md\nsrc\n", run("ls")) is None
    # Mentioning the phrase is not hitting it.
    assert classify("bash", "No module named in this repo is missing\n", run("cat notes.md")) is None


def record_chat(store, *turns):
    """A plain chat, no tools and no system message: (user, answer) pairs."""
    history = []
    for said, answer in turns:
        history = history + [{"role": "user", "content": said}]
        store.record("/v1/chat/completions", {"messages": history}, reply({"role": "assistant", "content": answer}))
        history = history + [{"role": "assistant", "content": answer}]
    store.reconstruct()


class Judge(FakeReader):
    """The model asked what a candidate sentence is: "about_me" for the ones
    named, "reaction" for the rest. Keeps the questions."""

    def __init__(self, about_me=()):
        super().__init__()
        self.about_me, self.questions = set(about_me), []

    async def ask(self, prompt, schema, name):
        if name != "preference":
            return await super().ask(prompt, schema, name)
        self.questions.append(prompt)
        sentence = prompt.rsplit("One sentence of that message:\n", 1)[1].split("\n")[0]
        return {"kind": "about_me" if sentence in self.about_me else "reaction"}


def test_a_preference_the_user_states_about_themself_becomes_a_note_in_their_words(config, store):
    """Live (Qwen3.5-4B on vLLM, preferences said the way people say them: "I
    avoid ...", "I'm not a fan of ..."): said as the opening message of a
    conversation, none was read at all (the opening message is the task, and
    it does not start like a rule); said after an answer, one became "Verify
    user constraints before recommending content.", read as fitting one kind
    of task, and never reached the prompt."""
    from songe import quick
    from songe.practice.store import wanted

    said = "I avoid dairy because it upsets my stomach."
    record_chat(store, (f"{said} What can I cook tonight?", "Try a lentil curry."))
    reader = Judge(about_me={said})
    assert read(store, reader) == 1
    # The model saw the sentence in its message, and that nothing came before.
    (question,) = reader.questions
    assert "What can I cook tonight?" in question and "(nothing: this is the first message)" in question
    (signal,) = [s for s in reading.signals(store) if s.get("preference")]
    assert (signal["text"], signal["lesson"], signal["short"], signal["generality"]) == (said, said, said, "always")
    # It points at the answer it got, so a Deep Dream practises it too.
    assert signal["exchanges"] and signal["id"] in {s["id"] for s, _ in wanted(store, 1)}
    # In the prompt at once, word for word, and it stays there.
    for _ in range(2):
        quick.quick_dream(store, config)
        assert [lesson["text"] for lesson in quick.active(store)] == [said]
    # A capture today's pattern would not make again leaves the prompt.
    assert not quick.suitable(signal | {"preference": "Fix it now."})


def test_a_preference_said_after_an_answer_is_published_as_the_user_said_it(config, store):
    from songe import quick

    said = "I'm allergic to peanuts."

    class Allergic(Judge):
        async def ask(self, prompt, schema, name):
            answer = await super().ask(prompt, schema, name)
            return dict(answer, lesson="Never suggest peanuts: the user is allergic.") if name == "lesson" else answer

    record_chat(store, ("Suggest a snack.", "Peanut butter on toast."), (f"{said} Anything else works.", "Apple slices."))
    assert read(store, Allergic(about_me={said})) == 1
    # One signal: the model's reading of the message, which now carries the sentence.
    (signal,) = [s for s in reading.signals(store) if s["source"] == "user"]
    assert signal["preference"] == said and signal["lesson"].startswith("Never suggest peanuts")
    quick.quick_dream(store, config)
    assert [lesson["text"] for lesson in quick.active(store)] == [said]


def test_a_preference_whose_lesson_was_a_guess_is_practised_in_the_users_words(store):
    # The lesson the model wrote for the message says nothing the user said
    # (the canned one is about a failing test): it is dropped, and the
    # sentence is the lesson a Deep Dream practises.
    said = "I'm allergic to peanuts."
    record_chat(store, ("Suggest a snack.", "Peanut butter on toast."), (f"{said} Anything else works.", "Apple slices."))
    read(store, Judge(about_me={said}))
    kept = [s for s in reading.signals(store) if s["source"] == "user" and s.get("lesson")]
    assert [(s["lesson"], s.get("preference")) for s in kept] == [(said, said)]


def test_not_every_sentence_in_the_first_person_is_a_preference(config, store):
    from songe import quick
    from songe.reading.rules import preference_sentences

    # Candidates, in the user's words. A statement that runs into a question,
    # or into a request (Chinese joins clauses with commas), is cut from it.
    assert preference_sentences("I'm vegetarian, what should I cook tonight?") == ["I'm vegetarian"]
    assert preference_sentences("As a vegan, I skip honey too.") == ["As a vegan, I skip honey too."]
    assert preference_sentences("我不吃辣。") == ["我不吃辣"]
    assert preference_sentences("我对花生过敏，推荐点别的") == ["我对花生过敏"]
    assert preference_sentences("我吃素，不吃肉也不吃鱼。") == ["我吃素"]
    assert preference_sentences("I avoid three things: gluten, dairy and soy.") == ["I avoid three things: gluten, dairy and soy."]
    # Never candidates: a pick among things just offered, a wish about this
    # answer, a fact of the task, a question, a text handed over to be worked
    # on, somebody's line.
    for text in (
        "I prefer the second one.", "I like this version better.", "我喜欢第二个", "我更喜欢这个",
        "Shorter please.", "I'm using Python 3.9 here.", "Do I always need a virtual environment?",
        "我总是需要虚拟环境吗？",
        "Correct the grammar: I enjoy listening to jazz. I never miss a concert in my town.",
        "帮我润色：我从小喜欢画画，我总是随身带着速写本。",
        'Mira: "Oh, I never eat before a show. I love the quiet of an empty hall."',
    ):
        assert preference_sentences(text) == [], text
    # What a pattern cannot tell apart, the model judges where it stands: a
    # word on the last answer and a line about the job are not kept.
    record_chat(
        store,
        ("Draw me a landing page.", "Here is a page with a wide hero image."),
        ("I love the new layout. I only get a blank page on my phone though.", "Fixed."),
    )
    reader = Judge()
    assert read(store, reader) == 1
    assert len(reader.questions) == 2 and all("wide hero image" in q for q in reader.questions)
    assert not [s for s in reading.signals(store) if s.get("preference")]
    quick.quick_dream(store, config)
    assert quick.active(store) == []


def test_a_sentence_that_says_a_habit_changed_is_a_candidate(config, store):
    """Live (Qwen3.5-4B on vLLM; 20 people each state a preference and, in a
    later conversation, say it has changed): 10 of the 20 later statements
    were never looked at, so the note kept the old habit only. The patterns
    that pick candidate sentences wanted a word of taste ("prefer", "avoid",
    "only") in a sentence that opens with "I", and "I've stopped taking the
    train; I cycle to work now." has no such word. The words people use for a
    change now make a candidate too; the model still judges each one. All 20
    became notes, and 15 of 20 more cases written afterwards (2 before); in
    5,000 real chat turns 6 more sentences are candidates, and none is kept."""
    from songe import quick
    from songe.reading.rules import preference_sentences

    for text in (
        "I've stopped taking the train; I cycle to work now.",
        "I no longer drink coffee.",
        "These days I cook without butter.",
        "Actually I'd rather get there by bike.",
        "I've become a morning person.",
        "Meetings before ten don't work for me anymore; I start at eleven now.",
        "我戒了咖啡。", "我不再坐飞机了。", "我改用左手鼠标了。",
    ):
        assert preference_sentences(text) == [text.rstrip("。")], text
    # Nobody speaking of themself, a request, a question: still not candidates.
    for text in (
        "The build doesn't work anymore.", "Make it no longer depend on my config.", "It works now.",
        "Is it any faster now?", "现在只剩两个测试失败。",
    ):
        assert preference_sentences(text) == [], text
    # A candidate only: the model is asked, and a line about the job is not kept.
    said = "I've stopped taking the train; I cycle to work now."
    record_chat(store, (f"{said} Plan my morning.", "Leave at 8."))
    record_chat(store, ("I no longer get the error. What next?", "Run the tests."))
    reader = Judge(about_me={said})
    assert read(store, reader) == 2 and len(reader.questions) == 2
    assert [s["preference"] for s in reading.signals(store) if s.get("preference")] == [said]
    quick.quick_dream(store, config)
    assert [lesson["text"] for lesson in quick.active(store)] == [said]


class Changing(Judge):
    """Also asked which earlier sentences a sentence replaces (answers with
    the numbers given for it), and whether an earlier and a later sentence
    can both be true (not for the pairs given). Keeps those questions."""

    def __init__(self, about_me=(), replaces=None, not_both=()):
        super().__init__(about_me)
        self.replaces, self.not_both, self.compared, self.checked = replaces or {}, set(not_both), [], []

    async def ask(self, prompt, schema, name):
        if name == "replaces":
            self.compared.append(prompt)
            said = prompt.rsplit("What the user said after all of them:\n", 1)[1].split("\n")[0]
            return {"replaces": self.replaces.get(said, [])}
        if name == "both_true":
            pair = (prompt.split("Earlier: ", 1)[1].split("\n")[0], prompt.split("Later: ", 1)[1].split("\n")[0])
            self.checked.append(pair)
            return {"kind": "later_only" if pair in self.not_both else "both"}
        return await super().ask(prompt, schema, name)


def test_a_new_preference_retires_the_one_it_replaces(config, store):
    """Live (Qwen3.5-4B on vLLM; 20 people each state a preference and, in a
    later conversation, say it has changed): the old sentence stayed in the
    prompt every time, next to the new one where the new one was kept, and
    nothing ever took it out. Now the model is asked, once for each
    preference, which of the ones stated before it it replaces, then of each
    one it names whether both can be true, and those that cannot are
    retired. Live again: the old sentence left the prompt in 17 of the 18
    cases; of 40 pairs of sentences that do not conflict all 80 stayed, and
    so did all 200 sentences of 20 users who stated ten preferences each."""
    from songe import quick
    from songe.practice.store import wanted
    from songe.reading import prompts as P

    old, new, other = "I prefer window seats when I fly.", "I prefer aisle seats now.", "I avoid dairy."
    first = [{"role": "user", "content": f"{old} Find me a flight to Oslo."}]
    store.record("/v1/chat/completions", {"messages": first}, reply({"role": "assistant", "content": "The 9:40 has window seats."}))
    store.reconstruct()
    record_chat(store, (f"{other} What can I cook tonight?", "A lentil curry."))
    # The model names a note that the sentence does not replace: asked again
    # about that pair alone, it says both can be true, and both stay.
    reader = Changing(about_me={old, new, other}, replaces={other: [1]})
    assert read(store, reader) == 2
    assert reader.compared == [P.REPLACES.format(notes=f"1. {old}", sentence=other)] and reader.checked == [(old, other)]
    quick.quick_dream(store, config)
    assert {lesson["text"] for lesson in quick.active(store)} == {old, other}

    record_chat(store, (f"{new} Find me a flight to Rome.", "The 7:15 has aisle seats."))
    reader = Changing(about_me={old, new, other}, replaces={new: [1, 2, 9, "2", True]}, not_both={(old, new)})
    assert read(store, reader) == 1
    asked = reader.compared
    assert asked[-1] == P.REPLACES.format(notes=f"1. {old}\n2. {other}", sentence=new)
    assert reader.checked == [(old, new), (other, new)]  # an answer that names no note is ignored
    # The old sentence is retired: out of reading, out of practice, out of
    # the prompt. The other one stays.
    assert [s["preference"] for s in reading.signals(store)] == [other, new]
    assert len(wanted(store, 1)) == 2
    quick.quick_dream(store, config)
    assert {lesson["text"] for lesson in quick.active(store)} == {other, new}
    # Its conversation goes on and is read again: it stays retired, and the
    # questions are the same ones (a real reader answers them from its cache).
    more = first + [{"role": "assistant", "content": "The 9:40 has window seats."}, {"role": "user", "content": "Book it."}]
    store.record("/v1/chat/completions", {"messages": more}, reply({"role": "assistant", "content": "Booked."}))
    store.reconstruct()
    reader = Changing(about_me={old, new, other})
    assert read(store, reader) == 1 and reader.compared == asked
    assert [s["preference"] for s in reading.signals(store) if s.get("preference")] == [other, new]
    quick.quick_dream(store, config)
    assert {lesson["text"] for lesson in quick.active(store)} == {other, new}


class Repeating(Changing):
    """Also asked which kept notes a new note could repeat (the numbers given
    for it) and whether two notes are one wish (the pairs given are)."""

    def __init__(self, about_me=(), same=None, one_wish=(), **changes):
        super().__init__(about_me, **changes)
        self.same, self.one_wish, self.one_way, self.listed, self.pairs = same or {}, set(one_wish), set(), [], []

    async def ask(self, prompt, schema, name):
        if name == "same_note":
            self.listed.append(prompt)
            return {"same": self.same.get(prompt.split("A new note:\n", 1)[1].split("\n")[0], [])}
        if name == "same_wish":
            pair = (prompt.split("Note 1: ", 1)[1].split("\n")[0], prompt.split("Note 2: ", 1)[1].split("\n")[0])
            self.pairs.append(pair)
            return {"note_1_asks": pair[0], "note_2_asks": pair[1], "kind": "same" if pair in self.one_wish or pair[::-1] in self.one_wish - self.one_way else "different"}
        return await super().ask(prompt, schema, name)


def test_a_preference_said_again_shares_one_place_and_a_changed_one_still_leaves(config, store):
    """A preference said again in other words is not a change of mind and not
    a second preference: both sentences stay, word for word, in one place.
    A preference that replaces an earlier one still retires it, and that is
    settled first: folding only sees what is still true of the user."""
    from songe import quick
    from songe.reading import prompts as P

    dairy, again = "I avoid dairy.", "I avoid milk and cheese."
    window, aisle = "I prefer window seats when I fly.", "I prefer aisle seats now."
    record_chat(store, (f"{dairy} What can I cook tonight?", "A lentil curry."))
    record_chat(store, (f"{window} Find me a flight to Oslo.", "The 9:40 has window seats."))
    record_chat(store, (f"{again} Any dessert ideas?", "A fruit salad."))
    record_chat(store, (f"{aisle} Find me a flight to Rome.", "The 7:15 has aisle seats."))
    reader = Repeating(
        about_me={dairy, again, window, aisle}, replaces={aisle: [2]}, not_both={(window, aisle)},
        same={again: [1], aisle: [1]}, one_wish={(dairy, again)},
    )
    assert read(store, reader) == 4
    assert [s["preference"] for s in reading.signals(store)] == [dairy, again, aisle]
    # The retired sentence is not among the notes shown; the seat preference
    # the model named for the dairy note is a different wish, and stays apart.
    assert reader.listed == [P.SAME.format(notes=f"1. {dairy}", note=again), P.SAME.format(notes=f"1. {dairy}", note=aisle)]
    assert reader.pairs == [(dairy, again), (again, dairy), (dairy, aisle)]
    quick.quick_dream(store, config)
    lessons = {lesson["text"]: lesson for lesson in quick.active(store)}
    assert set(lessons) == {f"{again} {dairy}", aisle}
    assert lessons[f"{again} {dairy}"]["count"] == 2 and lessons[f"{again} {dairy}"]["rule"]


def test_the_order_conversations_are_read_in_does_not_decide_which_preference_stays(config, store):
    from songe import quick
    from songe.reading import prompts as P

    old, new = "I love coffee in the morning.", "I've given up coffee; I only drink tea now."
    record_chat(store, (f"{old} Plan my breakfast.", "Coffee and toast."))
    record_chat(store, (f"{new} Plan my breakfast.", "Tea and toast."))
    # A backlog is read newest first. The sentences are compared in the order
    # they were said: the later one is asked about the earlier one.
    reader = Changing(about_me={old, new}, replaces={new: [1]}, not_both={(old, new)})
    assert asyncio.run(reading.read(store, reader, newest=True)) == 2
    assert reader.compared == [P.REPLACES.format(notes=f"1. {old}", sentence=new)]
    assert [s["preference"] for s in reading.signals(store)] == [new]
    quick.quick_dream(store, config)
    assert [lesson["text"] for lesson in quick.active(store)] == [new]


def test_the_results_of_an_agents_own_tools_are_reviewed(store):
    """Live (Qwen3.5-9B on vLLM, a word game played through a function tool
    named guess, 100 games): reading asked no review question and kept no
    signal from any result. Only results of tools named shell, execute, bash
    or webfetch were looked at, so a refused word and a lost game taught
    nothing. Now every tool's result is reviewed except those of tools that
    read, list, search or change files, and a turn of shell commands is asked
    exactly as before."""
    from songe.reading import prompts as P
    from songe.reading import views

    assert all(views.reviewed(name) for name in ("bash", "shell", "webfetch", "guess", "mcp__tracker__create_issue"))
    assert not any(views.reviewed(name) for name in ("read", "Read", "edit", "apply_patch", "TodoWrite", "list_files", None))

    tools = [tool("guess", word="string"), tool("read", filePath="string")]
    refused = "You attempted an invalid move. Reason: 'zzzzz' is not an English word."
    turns = [
        ({"role": "assistant", "content": "", "tool_calls": [call("a", "guess", word="crane")]},
         {"role": "tool", "tool_call_id": "a", "content": "C R A N E\nX X X X G\nYou have 5 guesses left."}),
        ({"role": "assistant", "content": "", "tool_calls": [call("b", "read", filePath="words.txt")]},
         {"role": "tool", "tool_call_id": "b", "content": "aback\nabase\nabate"}),
        ({"role": "assistant", "content": "", "tool_calls": [call("c", "guess", word="zzzzz")]},
         {"role": "tool", "tool_call_id": "c", "content": refused}),
    ]
    history = [SYSTEM, {"role": "user", "content": "Game 1. Enter your guess to begin."}]
    for action, result in turns:
        store.record("/v1/chat/completions", {"messages": history, "tools": tools}, reply(action))
        history = history + [action, result]
    store.record("/v1/chat/completions", {"messages": history, "tools": tools},
                 reply({"role": "assistant", "content": "I ran out of ideas."}))
    store.reconstruct()

    class Reviewing(FakeReader):
        def __init__(self):
            super().__init__()
            self.reviews = []

        async def ask(self, prompt, schema, name):
            if name == "review":
                self.asked.append(name)
                self.reviews.append(prompt)
                return {"signals": [{"step": 3, "type": "failed_result",
                                     "quote": "'zzzzz' is not an English word", "why": "the word was refused"}]}
            return await super().ask(prompt, schema, name)

    reader = Reviewing()
    assert read(store, reader) == 1
    (question,) = reader.reviews
    assert P.REVIEW_TOOLS["ran"] in question and P.REVIEW_TOOLS["errors"] in question
    assert "X X X X G" in question and refused in question
    assert "aback" not in question  # a file read is not a result to review
    (signal,) = [s for s in reading.signals(store) if s["source"] == "tool"]
    assert signal["type"] == "failed_result" and "not an English word" in signal["text"]
    # Shell commands: the question that was checked against the answer key.
    shell = [
        {"kind": "user", "content": "run it"},
        {"kind": "action", "step": 1, "text": "", "tool_calls": [{"id": "x", "name": "bash", "arguments": '{"command": "python3 guess.py 500"}'}]},
        {"kind": "tool_result", "call": "x", "step": 1, "tool": "bash", "content": "500: higher"},
    ]
    (event,) = [e for e in views.events(shell, rules=[]) if e["kind"] == "review"]
    assert event["commands"]
    asked = P.REVIEW.format(task="t", steps="s", **P.REVIEW_COMMANDS)
    assert "These are the commands the agent ran in one turn, with their output:\ns\n" in asked
    assert 'compile checks like "syntax OK", or any error message (those are handled elsewhere).\n' in asked
    assert "{" not in asked.replace("JSON", "")


class LessonQuestions(FakeReader):
    """Keeps the lesson questions it is asked."""

    def __init__(self):
        super().__init__()
        self.questions = []

    async def ask(self, prompt, schema, name):
        if name == "lesson":
            self.questions.append(prompt)
        return await super().ask(prompt, schema, name)


def tool_step(cid, name, result, **args):
    """An agent's call and the tool's answer to it."""
    return ({"role": "assistant", "content": "", "tool_calls": [call(cid, name, **args)]},
            {"role": "tool", "tool_call_id": cid, "content": result})


def test_the_lesson_question_for_a_tools_output_keeps_what_it_named(store):
    """Live (Qwen3.5-4B and 9B on vLLM, 30 coding tasks, issue #144): "pytest:
    command not found" was stored as "Install missing dependencies before
    running tests." and the next task ran the same failing command. The
    question now says that what a tool's output names is the lesson. And a
    failed result the conversation ended on (a build that never passed, a
    lost game) is asked about as a whole attempt, with all its steps."""
    from songe.reading import prompts as P

    Recording, step = LessonQuestions, tool_step

    def record(first, steps, last):
        history = [SYSTEM, {"role": "user", "content": first}]
        for action, result in steps:
            store.record("/v1/chat/completions", {"messages": history, "tools": TOOLS}, reply(action))
            history = history + [action, result]
        store.record("/v1/chat/completions", {"messages": history, "tools": TOOLS},
                     reply({"role": "assistant", "content": last}))

    edit = dict(filePath="x.py", oldString="a", newString="b")
    # Fails, is fixed, passes: the failure is one step of a task that went on.
    record("fix the parser", [step("a", "edit", "Edited x.py", **edit),
                              step("b", "shell", "1 failed, 2 passed\nExit code: 1", command="pytest"),
                              step("c", "edit", "Edited x.py", **edit),
                              step("d", "shell", "3 passed\nExit code: 0", command="pytest")], "Fixed.")
    # Fails and stops there: the task was not completed.
    record("fix the lexer", [step("e", "edit", "Edited y.py", **edit),
                             step("f", "shell", "/bin/bash: line 1: pytest: command not found\nExit code: 127", command="pytest")],
           "I could not run the tests.")
    store.reconstruct()
    reader = Recording()
    assert read(store, reader) == 2
    went_on, ended = sorted(reader.questions, key=lambda q: "command not found" in q)
    assert P.TOOL_FACT in went_on and P.TOOL_FACT not in ended
    assert "name exactly what the output named" in P.TOOL_FACT
    # What worked is advice only when it is the ordinary way: live, a pip that refused to install into the
    # system Python left "Use '--break-system-packages' when pip refuses to install." as the one note.
    assert "A way around a refusal or a warning that protects something" in P.TOOL_FACT
    assert "is never part of the lesson" in P.TOOL_FACT
    # How far a fact reaches is judged as for any lesson. Live, a line that made a fact about the tool or the
    # machine "always" sent 13 to 16 of 16 to 19 lessons to every session (8 of 18 without it), among them a
    # module missing in one project.
    assert "generality" not in P.TOOL_FACT
    assert "the conversation ended with it" not in went_on
    assert "the conversation ended with it" in ended and "names it exactly as written" in ended
    attempt = ended.split("All steps of the attempt, one line each:\n", 1)[1]
    assert "STEP 1: edit" in attempt and "STEP 2: shell pytest" in attempt and "Exit code: 127" in attempt
    # The user's own words are asked about as before.
    record_session(store, started=None)
    reader = Recording()
    read(store, reader)
    assert any(P.USER_WISH in q for q in reader.questions)
    assert all(P.TOOL_FACT not in q for q in reader.questions if P.USER_WISH in q)
    # A shape pasted back with its placeholders is no lesson.
    assert P.copied_example("<NAME> is not installed here; use <WORKED>.")
    assert not P.copied_example("pytest is not installed here; use python -m pytest.")


def test_a_failure_ends_an_attempt_only_once_the_agent_has_stopped(store):
    """A session can be read while it is going on: a Quick Dream starts when
    the model server has been quiet for a while, and one long command is
    quiet that long. The newest failed result is then not the end of
    anything, the agent's next call is waiting for its result. It is asked
    about like any other output, and as a whole attempt ("the task was not
    completed") only when the agent's own closing words follow it."""
    from songe.reading import prompts as P

    edit = dict(filePath="y.py", oldString="a", newString="b")
    missing = "/bin/bash: line 1: pytest: command not found\nExit code: 127"
    history = [SYSTEM, {"role": "user", "content": "fix the lexer"}]
    for action, result in (tool_step("a", "edit", "Edited y.py", **edit),
                           tool_step("b", "shell", missing, command="pytest")):
        store.record("/v1/chat/completions", {"messages": history, "tools": TOOLS}, reply(action))
        history = history + [action, result]
    # The agent tries another way; that call has no result yet.
    again, no_module = tool_step("c", "shell", "/usr/bin/python: No module named pytest\nExit code: 1",
                                 command="python -m pytest")
    store.record("/v1/chat/completions", {"messages": history, "tools": TOOLS}, reply(again))
    store.reconstruct()
    reader = LessonQuestions()
    assert read(store, reader) == 1
    (question,) = reader.questions
    assert "command not found" in question and P.TOOL_FACT in question
    assert "the conversation ended with it" not in question
    # The result comes, it failed too, and the agent gives up: now the attempt has ended, on that result.
    history = history + [again, no_module]
    store.record("/v1/chat/completions", {"messages": history, "tools": TOOLS},
                 reply({"role": "assistant", "content": "I could not run the tests."}))
    store.reconstruct()
    reader = LessonQuestions()
    assert read(store, reader) == 1
    went_on = [q for q in reader.questions if P.TOOL_FACT in q]
    ended = [q for q in reader.questions if "the conversation ended with it" in q]
    assert len(went_on) == 1 and len(ended) == 1 and len(reader.questions) == 2
    assert "command not found" in went_on[0]
    assert "No module named pytest" in ended[0]


def test_a_wish_keeps_what_the_user_named():
    """Live (Qwen3.8-27B, simulated users, issue #231): "styled-components +
    export default. siempre en admin" became the note "Always work in
    admin."; "just test-unit, not make" became "Use test-unit, not make.",
    and the move from make to just was lost."""
    from songe.reading import prompts as P

    admin = {"standing": "yes", "kind": "once", "wish": "Always work in admin."}
    # Keeps none of the names the user typed: not used (two sentences: no own words either).
    assert P.turn_verdict(admin, "styled-components + export default. siempre en admin") == ("lasting", "")
    kept = {"standing": "yes", "kind": "once", "wish": "Use styled-components and default exports in admin."}
    assert P.turn_verdict(kept, "styled-components + export default. siempre en admin")[1] == kept["wish"]
    # A command the user typed is kept whole.
    just = {"standing": "yes", "kind": "once", "wish": "Use test-unit, not make."}
    said = "just test-unit, not make. there is no makefile anymore, from now on its just"
    assert P.turn_verdict(just, said) == ("lasting", "Use just test-unit, not make.")
    assert P.whole_commands("Use make test-unit with ARGS.", "i said make test-unit. ARGS passes") == "Use make test-unit with ARGS."
    # "just" and "make" as words are not commands.
    assert P.whole_commands("Keep it short.", "make it short, just the code") == "Keep it short."
    assert P.whole_commands("Use pnpm add, not npm install.", "pnpm add zod pls not npm") == "Use pnpm add, not npm install."
