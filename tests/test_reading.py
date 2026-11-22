"""Read: recorded sessions become signals, per learner, resumably. The model
is faked; nothing is loaded and nothing is executed."""

import asyncio
import json

import httpx

from songe.models import registry
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


def record_session(store, *, more=False, started=None, reply_text="No, you changed the wrong file."):
    """Ask, run the tests (1 failed), edit, run again (passed), done; then the
    user corrects. With more=True the session continues one more turn."""
    t1 = {"role": "assistant", "content": "", "tool_calls": [call("a", "shell", command="pytest")]}
    r1 = {"role": "tool", "tool_call_id": "a", "content": "1 failed, 2 passed\nExit code: 1"}
    t2 = {"role": "assistant", "content": "", "tool_calls": [call("b", "edit", filePath="x.py", oldString="a", newString="b")]}
    r2 = {"role": "tool", "tool_call_id": "b", "content": "Edited x.py"}
    t3 = {"role": "assistant", "content": "", "tool_calls": [call("c", "shell", command="pytest")]}
    r3 = {"role": "tool", "tool_call_id": "c", "content": "3 passed\nExit code: 0"}
    done = {"role": "assistant", "content": "Fixed."}
    user = {"role": "user", "content": "fix the failing test"}
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
    "responsible": {"steps": [2], "reason": "wrong file"},
    "lesson": {"lesson": "Check the file first.", "instance": "x.py", "generality": "kind",
               "verifiable": None},
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
    record_session(store)
    learner = store.scoped("learner")
    scheduler, asked = scheduler_for(config, store, monkeypatch, loaded=["someone-else"])
    assert await scheduler.read_sessions() == 0  # its own model is not loaded
    assert asked == []
    scheduler, asked = scheduler_for(config, store, monkeypatch, loaded=["learner"])
    from songe.practice import store as practice

    # Short pauses only read; practice is Deep Dream's first step.
    assert await scheduler.read_sessions() == 1
    assert practice.counts(learner)["written"] == 0
    # Deep: 3 scenes for its one lesson (the correction).
    assert await scheduler.read_sessions(practice=True) == 3
    assert {model for model, _ in asked} == {"learner"}
    assert {s["type"] for s in reading.signals(learner)} == {"correction", "success"}
    assert reading.pending(learner) == []
    assert practice.counts(learner)["written"] == 3


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
        history = json.loads(db.execute("SELECT request FROM exchanges ORDER BY id DESC LIMIT 1").fetchone()[0])["messages"]
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
        kinds = [timeline.kind_of(json.loads(db.execute("SELECT request FROM exchanges WHERE id=?", (i,)).fetchone()[0]))
                 for i in ids]
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
            payload = {"id": f"s{i}", "session": "x", "type": "correction", "source": "user", "lesson": f"Lesson {i}.",
                       "exchanges": {"1": 1}, "revision": 1, **({"from": origin} if origin else {})}
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
    monkeypatch.setattr(ps, "source", lambda store, s: ({"messages": [{"role": "user", "content": "hi"}]}, []))
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
