"""Practice: scenes written from signals, framed on the real request, checked,
kept or dropped. The model is faked; nothing in a scene is ever run."""

import asyncio
import copy
import json

import pytest

from songe.practice import store as practice
from songe.practice.checks import STOCK_LESSONS, placebo_for
from songe.quick import visible
from songe.reading import store as reading
from songe.reading import timeline
from tests.test_reading import SCENE, FakeReader, args, record_session


class Writer(FakeReader):
    """Reads like FakeReader; writes the given scene; answers the forced
    choice with `pick` (A, B or neither) or with the real lesson's letter."""

    def __init__(self, scene=None, pick=None):
        super().__init__()
        self.scene, self.pick, self.prompts = scene or SCENE, pick, []

    async def ask(self, prompt, schema, name, *, temperature=0, max_tokens=None):
        if name == "dream":
            self.asked.append(name)
            self.prompts.append(prompt)
            return copy.deepcopy(self.scene)
        if name == "matters":
            self.asked.append(name)
            self.prompts.append(prompt)
            real = prompt.split("A: ", 1)[1].startswith("Check which file")
            return {"why": "w", "answer": self.pick or ("A" if real else "B")}
        return await super().ask(prompt, schema, name)


def corrected_twice(store):
    """The same correction in two conversations. Judged a change wanted once
    (test_reading.ANSWERS), it is a lesson to practise only when it comes up
    again, as it is a note only then (quick.admitted)."""
    record_session(store, ask="fix the other failing test")
    record_session(store)  # the newest: the one its scenes are framed on
    asyncio.run(reading.read(store, FakeReader()))


def written(store, scene=None, pick=None, per_signal=1):
    corrected_twice(store)
    writer = Writer(scene, pick)
    asyncio.run(practice.write(store, writer, per_signal))
    return practice.practice(store), writer


def test_a_scene_is_framed_on_the_logged_request_and_kept(store):
    rows, writer = written(store)
    assert len(rows) == 1
    row = rows[0]
    assert row["drop"] is None and row["checks"]["built"] and row["checks"]["decision"]
    system, *rest = row["messages"]
    # The real system prompt, moved to a sibling folder named by the scene.
    assert "Working directory: /work/invoice-calc" in system["content"]
    assert {t["function"]["name"] for t in row["tools"]} == {"shell", "read", "write", "edit"}
    # The read result is computed from the scene's files, in the tool's format.
    assert rest[2]["content"].startswith("Read file /work/invoice-calc/tax.py")
    assert rest[-1]["role"] == "tool" and "1 failed" in rest[-1]["content"]
    assert row["teacher_note"].startswith("Note from an earlier, different task")
    # The prompt carries the lesson and a short summary, never the session.
    assert "Check which file the failing test covers" in writer.prompts[0]
    assert "fix the failing test" not in writer.prompts[0]
    # One lesson only: the forced choice still runs, against a stock lesson.
    assert writer.asked == ["dream", "matters"] and row["checks"]["matters"] is True
    assert any(f": {text}\n" in writer.prompts[1] for text in STOCK_LESSONS)


def test_scenes_that_fail_a_check_are_dropped_with_its_code(store):
    copied = SCENE | {"request": "fix the failing test"}
    assert written(store, copied)[0][0]["drop"] == "not_copy"


def test_an_unparseable_file_drops_the_scene(store):
    broken = SCENE | {"files": [{"path": "tax.py", "content": "def vat(x:\n"}]}
    assert written(store, broken)[0][0]["drop"] == "parses"


def test_a_scene_with_an_unknown_tool_does_not_build(store):
    unknown = SCENE | {"steps": [{"say": "", "tool": "browse", "args": args(path="x"), "output": ""}]}
    assert written(store, unknown)[0][0]["drop"] == "built"


def test_the_forced_choice_drops_scenes_about_something_else(store):
    corrected_twice(store)
    # A second session with a different lesson gives the placebo.
    with store.connect() as db:
        db.execute(
            "INSERT INTO signals VALUES ('other','',?,1,?,NULL)",
            ("s2", json.dumps({"id": "other", "session": "s2", "type": "correction",
                               "lesson": "Use Conventional Commits for commit messages."})),
        )
    asyncio.run(practice.write(store, Writer(pick="neither"), 1))
    row = next(r for r in practice.practice(store) if r["signal"]["lesson"].startswith("Check"))
    assert row["checks"]["matters"] is False and row["drop"] == "matters"
    assert row["why"]["matters_answer"] == "neither"


def test_a_repeated_lesson_from_another_session_cannot_be_its_own_placebo(store):
    record_session(store)
    asyncio.run(reading.read(store, FakeReader()))
    original = next(s for s in reading.signals(store) if s.get("lesson"))
    duplicate = {"id": "other", "session": "s2", "type": "correction", "lesson": original["lesson"]}
    with store.connect() as db:
        db.execute(
            "INSERT INTO signals VALUES ('other','',?,1,?,NULL)",
            ("s2", json.dumps(duplicate)),
        )

    writer = Writer()
    asyncio.run(practice.write(store, writer, 1))

    (row,) = practice.practice(store)
    assert writer.asked == ["dream", "matters"]
    assert writer.prompts[1].count(original["lesson"]) == 1
    assert row["checks"]["matters"] is True and row["drop"] is None


def test_a_first_session_scene_about_something_else_is_dropped(store):
    # With one lesson there is no other lesson to compare with; the scene
    # used to be kept unchecked (matters stayed None, and None is not a drop).
    row = written(store, pick="neither")[0][0]
    assert row["checks"]["matters"] is False and row["drop"] == "matters"
    assert row["why"]["matters_answer"] == "neither"


def test_placebo_falls_back_to_a_stock_lesson_other_than_the_real_one():
    for real in ("Use uv for tests.", *STOCK_LESSONS):
        signal = {"session": "s1", "lesson": f" {real} "}
        for scene in ("a", "b", "c", "d", "e", "f"):
            # A lesson of the same session is not a placebo.
            placebo, _ = placebo_for(scene, signal, [("s1", "Run ruff first.")])
            assert placebo in STOCK_LESSONS and placebo != real


def test_placebo_selection_uses_a_distinct_lesson_when_available():
    signal = {"session": "s1", "lesson": "Use uv for tests."}
    expected = "Use Conventional Commits."
    lessons = [("s2", "  Use uv for tests.  "), ("s3", expected)]

    placebo, _ = placebo_for("scene", signal, lessons)

    assert placebo == expected


def test_scenes_are_written_up_to_the_target_and_go_with_their_signal(store):
    rows, _ = written(store, per_signal=2)
    assert len(rows) == 2 and practice.wanted(store, 2) == []
    assert [r["variant"] for r in rows] == [1, 2]
    assert len(practice.wanted(store, 3)) == 1  # one more variant wanted
    counts = practice.counts(store)
    assert counts == {"written": 2, "kept": 2, "dropped": 0, "dropped_by": {}}
    # The session is read again and its signal is gone: so are its scenes.
    session = rows[0]["signal"]["session"]
    with store.connect() as db:
        ids = timeline.sessions(db)[session]
    reading.save(store, session, max(ids) + 1, [], [])
    assert practice.practice(store) == []


def test_forget_deletes_practice(store):
    from songe.practice.pool import REJECTED

    written(store)
    # Scene ids start again after forgetting: what a rejected Dream trained
    # must not match the new ones (they were then never untrained).
    other = store.scoped("another-model")
    for view in (store, other):
        view.set(REJECTED, {"practice:x#v1": [3, 1]})
    store.forget()
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM practice").fetchone()[0] == 0
    assert store.get(REJECTED) is None and other.get(REJECTED) is None


def test_not_copy_catches_non_latin_verbatim_requests():
    from songe.practice import checks

    ar_req = "أصلح دالة حساب الضريبة في ملف الفواتير للعميل السري"
    ar_ev = checks.copy_evidence({"request": ar_req}, {}, [ar_req], [])
    assert ar_ev and ar_ev[0].startswith("request overlap")

    zh_req = "修复客户发票文件中的税率计算函数"
    zh_ev = checks.copy_evidence({"request": zh_req}, {}, [zh_req], [])
    assert zh_ev and zh_ev[0].startswith("request overlap")

    # Different Arabic request should not trigger copy overlap.
    ar_other = "أنشئ جدول بيانات جديد لإدارة المخزون الشهري"
    assert checks.copy_evidence({"request": ar_other}, {}, [ar_req], []) == []


def test_not_copy_ignores_definition_keywords_in_prose():
    from songe.practice import checks

    # Tool output in English: "function the", "class is" are not definitions.
    code = ["Done: the function the agent called is fine; this class is the base."]
    scene = {"request": "add the missing check to the parser"}
    assert checks.copy_evidence(scene, {}, ["fix it"], code) == []


def test_not_copy_still_catches_copied_definitions():
    from songe.practice import checks

    code = [
        "def compute_vat(amount):\n    return amount * 0.2\n",
        "function renderInvoice(rows) {",
        "class InvoiceStore extends Base {",
        "fn parse_line(s: &str) -> Line {",
        "struct LineItem {",
        "class Ledger:",
    ]
    assert checks.identifiers(code) == {
        "compute_vat",
        "renderInvoice",
        "InvoiceStore",
        "parse_line",
        "LineItem",
        "Ledger",
    }
    files = {"vat.py": "def compute_vat(total):\n    return total\n"}
    assert checks.copy_evidence({"request": "x"}, files, ["y"], code) == ["compute_vat"]


def test_a_chat_without_tools_or_system_message_gives_a_plain_scene():
    """#136: a plain chat client sends no tools and no system message. The
    user's question is not made a system message, and the scene writer is
    told the agent has no tools, so it writes a conversation, not tool steps."""
    from songe.practice import prompts
    from songe.practice.frame import build, frame_from_request

    request = {"messages": [{"role": "user", "content": "What is the capital of Australia?"}]}
    frame = frame_from_request(request, "trip-notes")
    assert frame["system"] == "" and frame["tools"] == []
    scene = {
        "moment": {"kind": "user_message", "step": {}, "agent_said": "Canberra.",
                   "user_message": "And when was it founded?"},
        "request": "What is the capital of Australia?", "files": [], "steps": [],
    }
    messages, problems, _ = build(scene, frame)
    assert problems == []
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    # The checks read the user's first message too (they skipped messages[0]).
    from songe.practice.checks import render_text

    assert render_text(messages).startswith("USER: What is the capital of Australia?")
    # The scene writer is told there are no tools; an agent with tools is not.
    assert prompts.tools_section(None, []) == prompts.NO_TOOLS
    assert "no tools" in prompts.NO_TOOLS and "user_message" in prompts.NO_TOOLS
    tools = [{"type": "function", "function": {"name": "read", "parameters": {"properties": {"path": {}}}}}]
    shown = prompts.tools_section(tools, tools)
    assert shown.startswith("The agent's tools") and "- read(path)" in shown
    # A logged system prompt is still the frame's.
    framed = frame_from_request({"messages": [{"role": "system", "content": "Be brief."}] + request["messages"]}, "x")
    assert build(scene, framed)[0][0] == {"role": "system", "content": "Be brief."}


# ---- A plain chat: new messages of the same kind, kept when the note changes the answer ----

CHAT_SIGNAL = {
    "id": "c0ffee00aa", "type": "correction", "source": "user", "session": "s-chat", "from": None,
    "generality": "always", "text": "I use uv, not pip.", "lesson": "Use uv to install packages, not pip.",
    "exchanges": {"1": 1},
}
CHAT_REQUEST = {"messages": [{"role": "user", "content": "How do I install requests?"}]}
CHAT_ITEMS = [{"kind": "user", "content": "How do I install requests?"}]
NEW_MESSAGES = ["What do I run to get numpy into this project?", "Add pandas for me, which command?",
                "I need httpx here. How?"]


class ChatWriter(FakeReader):
    """Writes `messages`; answers a scene with `plain`, or with `taught` when
    the scene's note is shown; judges for the answer given in `follows`
    ("taught", "plain", or a fixed letter)."""

    def __init__(self, messages=NEW_MESSAGES, plain="pip install it", taught="uv add it", follows="taught",
                 reasoned=False):
        super().__init__()
        self.messages, self.plain, self.taught = messages, plain, taught
        self.follows, self.reasoned = follows, reasoned
        self.prompts, self.replied = [], []

    async def ask(self, prompt, schema, name, *, temperature=0, max_tokens=None):
        self.asked.append(name)
        self.prompts.append(prompt)
        if name == "dream":
            return {"messages": list(self.messages)}
        assert name == "changes"
        if self.follows in ("A", "B", "both", "neither"):
            return {"follows": self.follows}
        # The judge is shown what the user sees: no reasoning.
        wanted = visible(self.taught if self.follows == "taught" else self.plain)
        first = prompt.split("Answer A:\n", 1)[1].split("\n\nAnswer B:", 1)[0]
        return {"follows": "A" if first == wanted else "B"}

    async def reply(self, messages, *, max_tokens, temperature=0, think=None):
        self.replied.append(messages)
        noted = "Not part of the conversation" in messages[-1]["content"]
        return {"content": self.taught if noted else self.plain, "finished": True, "reasoned": self.reasoned}


def chat_written(store, monkeypatch, writer, variants=(1, 2, 3), request=CHAT_REQUEST):
    monkeypatch.setattr(practice, "wanted", lambda store, per_signal: [(CHAT_SIGNAL, list(variants))])
    monkeypatch.setattr(practice, "source", lambda store, signal: (request, CHAT_ITEMS))
    asyncio.run(practice.write(store, writer, len(variants)))
    return practice.practice(store)


def test_a_plain_chat_practises_new_messages_of_the_same_kind(store, monkeypatch):
    """A chat with no tools: each scene is a new conversation that opens with
    a message of the kind the user sent before their feedback. The writer
    sees that message and never the lesson, so no scene can state it; one
    answer holds every scene. Live, the three-message scenes had the user
    say the lesson again ("... but remember to use pnpm instead of npm")."""
    writer = ChatWriter()
    rows = chat_written(store, monkeypatch, writer)
    assert [r["messages"] for r in rows] == [[{"role": "user", "content": m}] for m in NEW_MESSAGES]
    assert all(r["tools"] == [] and r["drop"] is None for r in rows)
    assert writer.asked.count("dream") == 1
    asked = writer.prompts[0]
    assert "How do I install requests?" in asked and "3 new messages" in asked
    assert "uv" not in asked and "pip" not in asked
    # The forced choice between two lessons is not asked: a scene that does
    # not state its lesson gave the model nothing to pick it by.
    assert "matters" not in writer.asked and all(r["checks"]["matters"] is None for r in rows)
    assert all(r["teacher_note"].endswith("Lesson: Use uv to install packages, not pip.") for r in rows)
    # A logged system prompt opens each new conversation too.
    with_system = {"messages": [{"role": "system", "content": "Be brief."}] + CHAT_REQUEST["messages"]}
    rows = chat_written(store, monkeypatch, ChatWriter(), request=with_system)
    assert rows[0]["messages"][0] == {"role": "system", "content": "Be brief."}
    assert rows[0]["messages"][1]["role"] == "user" and len(rows[0]["messages"]) == 2


def test_a_plain_scene_is_kept_when_its_note_changes_the_answer(store, monkeypatch):
    """The served model answers the scene alone and with the note; the
    scene is kept when the judge, asked in both orders, picks the answer
    with the note. That answer is stored with the scene."""
    writer = ChatWriter()
    rows = chat_written(store, monkeypatch, writer)
    assert all(r["checks"]["changes"] is True for r in rows)
    assert rows[0]["teacher_answer"] == {"content": "uv add it", "finished": True}
    # Two answers per scene: the scene alone, then with its note as an aside.
    alone, noted = writer.replied[:2]
    assert alone == rows[0]["messages"]
    assert noted[:-1] == rows[0]["messages"] and "Lesson: Use uv to install" in noted[-1]["content"]
    # The judge reads the user's own words, each pair in both orders.
    judged = [p for p, name in zip(writer.prompts, writer.asked, strict=True) if name == "changes"]
    assert len(judged) == 6 and all("I use uv, not pip." in p for p in judged)
    # Asked what the feedback asks, not which answer is the better one.
    assert all("Judge only that, not which answer is longer" in p for p in judged)
    assert "Answer A:\npip install it" in judged[0] and "Answer A:\nuv add it" in judged[1]


def test_a_plain_scene_is_kept_when_the_judge_cannot_tell_in_one_order(store, monkeypatch):
    """Live: of 384 scenes 173 were kept, and many dropped ones did follow
    the lesson (answers that began "Yes, chef.", bare SQL under "Only the
    query please"): the judge said "neither" in one of the two orders."""

    class Unsure(ChatWriter):
        async def ask(self, prompt, schema, name, **kw):
            answer = await super().ask(prompt, schema, name, **kw)
            if name == "changes" and "Answer A:\nuv add it" in prompt:
                return {"follows": "neither"}
            return answer

    rows = chat_written(store, monkeypatch, Unsure())
    assert all(r["drop"] is None and r["why"]["changes"] == ["after", "neither"] for r in rows)
    assert rows[0]["teacher_answer"] == {"content": "uv add it", "finished": True}


def test_a_kept_plain_scene_carries_its_answer_into_training(store, monkeypatch):
    from songe.practice import pool

    chat_written(store, monkeypatch, ChatWriter())
    steps = pool.practice_steps(store) + pool.practice_steps(store, split="held_out")
    assert len(steps) == 3
    assert all(s["teacher_answer"] == {"content": "uv add it", "finished": True} for s in steps)


def test_a_plain_scene_the_note_does_not_change_is_dropped(store, monkeypatch):
    # The same answer with and without the note: nothing to learn here, and
    # the judge is not asked.
    same = ChatWriter(taught="pip install it")
    rows = chat_written(store, monkeypatch, same)
    assert {r["drop"] for r in rows} == {"changes"} and "changes" not in same.asked
    assert rows[0]["why"]["changes"] == "same" and "teacher_answer" not in rows[0]
    # The judge prefers the answer without the note, or the first one shown
    # whatever it is (a position preference), or neither: dropped.
    for follows in ("plain", "A", "neither"):
        store_rows = chat_written(store, monkeypatch, ChatWriter(follows=follows))
        assert {r["drop"] for r in store_rows} == {"changes"}, follows


def test_a_plain_scene_of_a_reasoning_server_stores_the_answer_without_its_reasoning(store, monkeypatch):
    """A server that reasons first: the scene is judged on what the user
    would see, and that is what is stored to train along. Without it a
    reasoning model (Inkling-2) trained on its own reasoning only."""
    writer = ChatWriter(taught="<think>The note says uv.</think>uv add it", reasoned=True)
    rows = chat_written(store, monkeypatch, writer)
    assert all(r["drop"] is None for r in rows)
    assert all(r["teacher_answer"] == {"content": "uv add it", "finished": True} for r in rows)


class Reasoner(ChatWriter):
    """A server that reasons first: asked as its users ask, the reasoning
    takes the whole room (no content, cut off). With `switch` it replies when
    the reader's switches turn reasoning off; without, only given more room."""

    def __init__(self, switch):
        super().__init__()
        self.switch, self.how = switch, []

    async def reply(self, messages, *, max_tokens, temperature=0, think=None):
        self.how.append((think, max_tokens))
        answer = await super().reply(messages, max_tokens=max_tokens)
        if think is False and self.switch:
            return answer
        if max_tokens > 384:
            return answer | {"reasoned": True}
        return {"content": "", "finished": False, "reasoned": True}


def test_a_server_whose_reasoning_takes_the_whole_room_still_gets_plain_scenes(store, monkeypatch):
    """Empty content, reasoning present, cut off: both answers looked the
    same and every scene was dropped, so a reasoning server got no plain-chat
    practice. The two answers are asked again with reasoning switched off,
    and the scenes after the first start there. Judged, kept, and the
    answer stored to train along."""
    writer = Reasoner(switch=True)
    rows = chat_written(store, monkeypatch, writer)
    assert all(r["drop"] is None and r["teacher_answer"]["content"] == "uv add it" for r in rows)
    assert writer.how == [(None, 384)] * 2 + [(False, 384)] * 6


def test_a_server_that_ignores_the_reasoning_switches_is_given_more_room(store, monkeypatch):
    writer = Reasoner(switch=False)
    rows = chat_written(store, monkeypatch, writer)
    assert all(r["drop"] is None and r["teacher_answer"]["content"] == "uv add it" for r in rows)
    assert writer.how == [(None, 384)] * 2 + [(False, 384)] * 2 + [(None, 384 + 2048)] * 6
    # A reply that is still empty with all that room is a scene the note did not change.
    writer = Reasoner(switch=False)
    monkeypatch.setattr("songe.practice.checks.HOW_ASKED", ({}, {"think": False}))
    rows = chat_written(store, monkeypatch, writer, variants=(4,))
    assert rows[-1]["drop"] == "changes" and rows[-1]["id"].endswith("#v4")


def test_the_plain_writer_sees_the_answer_the_feedback_was_about(store, monkeypatch):
    """A lesson about the form of an answer ("just the code", "no comments")
    can only be practised on a request whose answer has that form. Shown the
    request alone, the writer also wrote questions about code that are
    answered in prose; the note changed nothing there. With the logged answer
    in front of it, it is asked for messages a good answer to which has the
    same form. The lesson itself is still not shown. Measured on a rented
    card (Qwen3.5-4B, 384 scenes for 16 lessons): 158 teaching scenes before,
    208 with the answer shown."""
    monkeypatch.setattr(practice, "answered", lambda store, signal: "Run `python -m pip install requests`.")
    writer = ChatWriter()
    rows = chat_written(store, monkeypatch, writer)
    asked = writer.prompts[0]
    assert "The assistant answered:\nRun `python -m pip install requests`." in asked
    assert "the same form as the answer above" in asked and "3 new messages" in asked
    assert "uv" not in asked
    assert [r["drop"] for r in rows] == [None, None, None]
    # No logged answer (an old store, a tool call with no text): as before.
    monkeypatch.setattr(practice, "answered", lambda store, signal: "")
    writer = ChatWriter()
    chat_written(store, monkeypatch, writer)
    assert "The assistant answered" not in writer.prompts[0]


def test_many_plain_messages_are_asked_for_twelve_at_a_time(store, monkeypatch):
    """One call for all of a lesson's messages needs 150 tokens of JSON per
    message and a small model stops early or repeats itself. Twelve are asked
    for per call, and a later call is told what is written already: asked the
    same question twice, the reader would hand back the first answer."""

    class Many(ChatWriter):
        async def ask(self, prompt, schema, name, *, temperature=0, max_tokens=None):
            if name != "dream":
                return await super().ask(prompt, schema, name, temperature=temperature, max_tokens=max_tokens)
            self.asked.append(name)
            self.prompts.append(prompt)
            count = int(prompt.split("Write ", 1)[1].split(" ", 1)[0])
            done = prompt.count("\n- Which command adds package")
            return {"messages": [f"Which command adds package number {done + i}?" for i in range(count)]}

    writer = Many()
    rows = chat_written(store, monkeypatch, writer, variants=tuple(range(1, 15)))
    dreams = [p for p, name in zip(writer.prompts, writer.asked, strict=True) if name == "dream"]
    assert len(dreams) == 2 and "Write 12 new messages" in dreams[0] and "Write 2 new messages" in dreams[1]
    assert "Already written" not in dreams[0]
    assert "Already written, write different ones:\n- Which command adds package number 0?" in dreams[1]
    assert len({r["messages"][-1]["content"] for r in rows}) == 14


def test_fewer_plain_messages_than_asked_leave_scenes_that_did_not_build(store, monkeypatch):
    writer = ChatWriter(messages=["Add pandas for me, which command?", " add pandas for me, which command? ", 7])
    rows = chat_written(store, monkeypatch, writer)
    assert [r["drop"] for r in rows] == [None, "built", "built"]
    assert rows[1]["scene"] == {"error": "no message written"}


def test_a_plain_message_copied_from_the_chat_is_dropped(store, monkeypatch):
    writer = ChatWriter(messages=["How do I install requests?"])
    rows = chat_written(store, monkeypatch, writer, variants=(1,))
    assert rows[0]["drop"] == "not_copy" and writer.replied == []


def test_the_scenes_of_a_plain_chat_are_answered_several_at_a_time(store, monkeypatch):
    """A scene costs two answers and a judgement from the served model. Asked
    one after another, a 27B on vLLM wrote 8 scenes a minute while the server
    sat at one running request; a Dream of 16 lessons with 60 scenes each
    would have practised for two hours. The scenes of one plain-chat lesson
    share nothing (no workspace, no tools), so AT_ONCE of them are answered
    at the same time. Scenes with tools stay one at a time."""

    class Slow(ChatWriter):
        running = most = 0

        async def reply(self, messages, *, max_tokens, temperature=0, think=None):
            Slow.running += 1
            Slow.most = max(Slow.most, Slow.running)
            await asyncio.sleep(0.01)
            Slow.running -= 1
            return await super().reply(messages, max_tokens=max_tokens)

    many = [f"Which command adds package number {n}?" for n in range(20)]
    rows = chat_written(store, monkeypatch, Slow(messages=many), variants=tuple(range(1, 21)))
    assert sorted(r["variant"] for r in rows) == list(range(1, 21))
    assert 1 < Slow.most <= 2 * practice.AT_ONCE


@pytest.mark.parametrize("shell,read", [("run_shell", "read_file"), ("Bash", "Read"), ("execute_bash", "view")])
def test_a_scene_uses_the_agents_own_names_for_shell_and_read(shell, read):
    """Agents name their tools differently; a scene's generic shell, grep and
    read steps become the agent's own calls instead of being dropped."""
    from songe.practice.frame import build, frame_from_request

    def tool(name, arg):
        return {"type": "function", "function": {"name": name, "parameters": {
            "type": "object", "properties": {arg: {"type": "string"}}, "required": [arg]}}}

    request = {"messages": [{"role": "user", "content": "x"}], "tools": [tool(shell, "command"), tool(read, "path")]}
    frame = frame_from_request(request, "shop")
    scene = {
        "request": "Tidy the cache.", "files": [{"path": "app.py", "content": "def main():\n    pass\n"}],
        "steps": [
            {"say": "", "tool": "grep", "args": {"pattern": "def main", "path": ""}, "output": ""},
            {"say": "", "tool": "read", "args": {"path": "app.py"}, "output": ""},
        ],
        "moment": {"kind": "tool_result", "step": {"say": "", "tool": "shell", "args": {"command": "ls"}, "output": "app.py"}},
    }
    messages, problems, _ = build(scene, frame)
    assert problems == []
    called = [c["function"]["name"] for m in messages for c in m.get("tool_calls") or []]
    assert called == [shell, read, shell]


def test_the_answer_with_the_note_is_the_plain_answer_changed_only_where_the_note_asks(store, monkeypatch):
    """Asked with the note alone, Qwen3.8-27B also wrote less (dates as ISO
    8601: 8.5 words of 87.5, "When was the first iPhone released?" ->
    "2007-06-29"); trained toward that, an adapter answered "When did the
    Titanic sink?" with the date alone. The teacher now edits its own reply
    without the note, and that edit is what a Dream trains along."""
    writer = ChatWriter(plain="Run pip install requests. It adds the library to your environment.",
                        taught="Run uv add requests. It adds the library to your environment.")
    rows = chat_written(store, monkeypatch, writer)
    noted = [m[-1]["content"] for m in writer.replied if "Not part of the conversation" in m[-1]["content"]]
    assert noted and all("<reply>\nRun pip install requests." in c and "Change only what the note is about" in c for c in noted)
    kept = [r for r in rows if r["drop"] is None]
    assert kept and all(r["teacher_answer"]["content"] == writer.taught for r in kept)
