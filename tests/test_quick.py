import asyncio
import json

import pytest

from songe import quick
from songe.reading import store as reading
from songe.reading.client import Stopped


@pytest.fixture(autouse=True)
def one_session_is_enough(config):
    # These tests put their signals in one session; the two-conversation bar
    # for "always" lessons has its own test.
    config.quick_min_sessions = 1


def add_signal(store, sid, lesson, *, kind="correction", generality="kind", revision=1, text=None, session="s",
               project=None, **extra):
    # The user's words: by default the lesson itself, as a stated rule.
    text = lesson if text is None and kind in quick.USER else text or ""
    payload = {"id": sid, "session": session, "type": kind, "source": "user" if kind in quick.USER else "tool",
               "lesson": lesson, "generality": generality, "revision": revision, "text": text, **extra}
    if project:
        payload["project"] = project
    with store.connect() as db:
        db.execute("INSERT INTO signals VALUES (?,?,?,?,?,NULL)",
                   (sid, store.model_id, session, revision, json.dumps(payload)))


def test_quick_dream_picks_what_the_user_wrote_not_tool_output(config, store):
    add_signal(store, "c1", "Always use Conventional Commits for commit messages.", revision=1)
    add_signal(store, "t1", "The execute tool runs JavaScript only; use the shell tool for shell commands.",
               kind="tool_error", generality="always", revision=2)
    add_signal(store, "t2", "The execute tool runs only JavaScript; run shell commands with the shell tool.",
               kind="tool_error", generality="always", revision=3)
    add_signal(store, "k1", "In Wordle keep letters marked G in place.", kind="outcome", revision=4)
    add_signal(store, "x1", "Fix report.py line 3.", generality="task", revision=5)
    # Live (Qwen3.8-27B, an agent session): git's output became a note.
    add_signal(store, "g1", "Verify that a file actually contains uncommitted changes before attempting to commit it.",
               kind="failed_result", generality="always", revision=6,
               text='no changes added to commit (use "git add" and/or "git commit -a")')
    result = quick.quick_dream(store, config, now=100.0)
    lessons = quick.active(store)
    # Only the user's own rule: lessons from a tool's output, the skill and
    # the task-only lesson wait for Deep Dream.
    assert {k: result[k] for k in ("lessons", "added", "reworded", "dropped", "version")} == {
        "lessons": 1, "added": 1, "reworded": 0, "dropped": 0, "version": 1}
    assert [lesson["text"] for lesson in lessons] == ["Always use Conventional Commits for commit messages."]
    # Deep Dream still practises what the work taught.
    signals = {s["id"]: s for s in reading.signals(store)}
    assert all(quick.evidenced(signals[i]) for i in ("t1", "t2", "g1"))
    assert "Wordle" not in quick.block(lessons) and quick.block(lessons).count("\n- ") == 1
    # Nothing new: same version, no churn in the prompt.
    assert quick.quick_dream(store, config, now=200.0)["version"] == 1


def test_removed_lessons_never_return_and_absorbed_ones_leave(config, store):
    add_signal(store, "c1", "Always use Conventional Commits for commit messages.", revision=1)
    add_signal(store, "c2", "Always keep the README under 30 lines.", revision=2)
    quick.quick_dream(store, config)
    readme = next(lesson for lesson in quick.active(store) if "README" in lesson["text"])
    quick.remove(store, readme["id"])
    add_signal(store, "c3", "Always keep the README under 30 lines, please.", revision=3)
    quick.quick_dream(store, config)
    assert [lesson["text"] for lesson in quick.active(store)] == [
        "Always use Conventional Commits for commit messages."]
    # A Deep Dream that trained c1 was adopted: its lesson leaves the prompt.
    assert quick.absorb(store, ["c1"]) == 1 and quick.active(store) == []
    quick.quick_dream(store, config)
    assert quick.active(store) == []
    # A new correction brings it back.
    add_signal(store, "c4", "Always use Conventional Commits for commit messages.", revision=4)
    quick.quick_dream(store, config)
    assert [lesson["signals"] for lesson in quick.active(store)] == [["c4"]]


def test_at_most_the_configured_number(config, store):
    config.prompt_lessons = 3
    for i in range(6):
        add_signal(store, f"c{i}", f"Always use preference number {i}: {'alpha beta gamma delta'.split()[i % 4]} {i * 7}",
                   revision=i)
    quick.quick_dream(store, config)
    assert len(quick.active(store)) == 3


def test_forget_clears_prompt_lessons(config, store):
    add_signal(store, "c1", "Always use Conventional Commits for commit messages.")
    quick.quick_dream(store, config)
    assert quick.active(store)
    store.forget()
    assert quick.active(store) == []


@pytest.mark.parametrize("session_header", [None, "x-session-affinity", "x-session-id"])
async def test_the_proxy_adds_the_sessions_lessons_to_the_system_prompt(
    config, store, session_header
):
    import httpx

    from tests.test_proxy import pair, version

    seen = []

    from tests.test_proxy import Chunks

    reply = json.dumps({"id": "x", "model": "night-1", "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}).encode()

    async def handler(request):
        seen.append(request)
        return httpx.Response(200, stream=Chunks([reply]), headers={"content-type": "application/json"})

    store.add_version(version("night-1", 1))
    store.swap("night-1", None)
    add_signal(store, "c1", "Always use Conventional Commits for commit messages.")
    quick.quick_dream(store, config)
    app, upstream = await pair(config, store, handler)

    def chat(first, *more):
        return {"model": "songe/test", "messages": [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": first}, *more]}

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client,
    ):
        first_session = {session_header: "first"} if session_header else {}
        next_session = {session_header: "second"} if session_header else {}
        await client.post(
            "/v1/chat/completions", json=chat("commit this"), headers=first_session
        )
        # A Quick Dream adds a lesson mid-session.
        add_signal(store, "c2", "Always keep the README under 30 lines.", revision=2)
        quick.quick_dream(store, config)
        await client.post(
            "/v1/chat/completions",
            json=chat(
                "commit this",
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "again"},
            ),
            headers=first_session,
        )
        await client.post(
            "/v1/chat/completions",
            json=chat("commit this"),
            headers=next_session,
        )
        await client.post(
            "/v1/chat/completions", json=chat("a new task"), headers=next_session
        )
    await upstream.aclose()
    system = [json.loads(r.content)["messages"][0]["content"] for r in seen]
    assert system[0].startswith("You are an agent.\n\n" + quick.BLOCK) and system[0].endswith(quick.EXCEPTIONS)
    assert "Conventional Commits" in system[0] and "README" not in system[0]
    assert system[1] == system[0]  # the session keeps its lessons
    # A new chat takes the current lessons, whether Pi names it a new session
    # or it only opens like the earlier one.
    assert "README" in system[2]
    assert "README" in system[3]
    # What Songe records is the agent's own request.
    assert store.exchanges()[0]["request"]["messages"][0]["content"] == "You are an agent."


async def test_controls_remove_a_lesson_and_check_the_dream_kind(config, store):
    from types import SimpleNamespace

    import httpx

    from songe.proxy import create_app

    add_signal(store, "c1", "Always use Conventional Commits for commit messages.")
    quick.quick_dream(store, config)
    lesson = quick.active(store)[0]
    asked = []

    async def dream(model, kind):
        asked.append(kind)
        return {"started": True, "message": "ok"}

    app = create_app(config, store, scheduler=SimpleNamespace(dream=dream, active_model=None))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
        removed = await client.post("/api/control", json={"action": "remove_lesson", "id": lesson["id"]})
        assert removed.json() == {"ok": True, "message": "Preference removed · new sessions no longer get it"}
        assert (await client.post("/api/control", json={"action": "dream", "kind": "quick"})).json()["kind"] == "quick"
        assert (await client.post("/api/control", json={"action": "dream", "kind": "nap"})).status_code == 400
        state = (await client.get("/api/state")).json()
    assert quick.active(store) == [] and asked == ["quick"]
    assert all(m.get("lessons") == [] for m in state["cycle"]["models"].values() if m)


class FakeReader:
    def __init__(self, answer):
        self.answer, self.prompts = answer, []

    async def ask(self, prompt, schema, name, **kw):
        self.prompts.append(prompt)
        return self.answer


async def test_consolidation_merges_and_shortens_with_the_served_model(config, store):
    tool = {"generality": "always", "text": ""}  # lessons of the user's, worded by reading
    add_signal(store, "p1", "Always verify file paths and directory structures before writing files to disk.",
               revision=1, **tool)
    add_signal(store, "p2", "Always verify that the write tool is called with the correct full path.",
               revision=2, **tool)
    add_signal(store, "g1", "Always use the game's feedback after each guess.", revision=3, **tool)
    clusters = quick.candidates(store, 30, config.quick_min_sessions)
    assert len(clusters) == 3  # different wording: not merged by words alone
    order = [c["signals"][0] for c in clusters]
    reader = FakeReader({"lessons": [
        {"text": "Check the full path before writing a file.",
         "from": [order.index("p1") + 1, order.index("p2") + 1]},
        {"text": "", "from": [1]}, {"text": "No source.", "from": [99]}]})
    merged = await quick.consolidate(reader, clusters, config.prompt_lessons)
    assert merged == [{"text": "Check the full path before writing a file.", "signals": ["p1", "p2"]}]
    assert "game" in reader.prompts[0]  # the candidates were shown, numbered
    quick.quick_dream(store, config, merged=merged)
    (lesson,) = quick.active(store)
    assert lesson["count"] == 2 and "game" not in quick.block(quick.active(store))
    # An unusable answer: the caller keeps the candidates as they are.
    assert await quick.consolidate(FakeReader("not json"), clusters, 10) is None
    assert await quick.consolidate(FakeReader({"other": 1}), clusters, 10) is None


def test_the_prompt_lessons_stay_within_their_budget(config, store):
    merged = [{"text": f"Instruction {i} " + "word " * 25, "signals": [f"s{i}"]} for i in range(10)]
    quick.quick_dream(store, config, merged=merged)
    kept = quick.active(store)
    assert 0 < len(kept) < 10
    assert sum(len(lesson["text"]) for lesson in kept) <= quick.PROMPT_CHARS


def test_skill_lessons_wait_for_a_deep_dream(config, store):
    add_signal(store, "c1", "Always use Conventional Commits for commit messages.", revision=1)
    add_signal(store, "k1", "In Wordle keep letters marked G in place.", kind="outcome", revision=2)
    add_signal(store, "k2", "In Wordle, keep the letters marked G in their place.", kind="outcome", revision=3)
    add_signal(store, "x1", "Fix report.py line 3.", generality="task", revision=4)
    quick.quick_dream(store, config)
    waiting = quick.waiting(store)
    assert waiting["total"] == 1 and waiting["lessons"][0]["count"] == 2
    store.set("trained_signals", ["k1", "k2"])  # a Deep Dream trained them
    assert quick.waiting(store)["total"] == 0


async def test_review_drops_one_kind_lessons_and_folds_repeats(config, store):
    answers = iter([
        {"scope": "general", "covered_by": 0},
        {"scope": "general", "covered_by": 1},   # says the same as the first
        {"scope": "one_kind", "covered_by": 0},  # a game lesson
        "unusable",                              # kept as it is
    ])

    class Reviewer:
        async def ask(self, prompt, schema, name, **kw):
            return next(answers)

    merged = [{"text": "Check the full path before writing a file.", "signals": ["p1"]},
              {"text": "Confirm the target directory before writing.", "signals": ["p2"]},
              {"text": "Use the game's feedback after each guess.", "signals": ["g1"]},
              {"text": "Always use Conventional Commits for commit messages.", "signals": ["c1"]}]
    kept = await quick.review(Reviewer(), merged)
    assert [k["text"] for k in kept] == ["Check the full path before writing a file.",
                                         "Always use Conventional Commits for commit messages."]
    assert kept[0]["signals"] == ["p1", "p2"]


def test_a_running_quick_dream_is_not_shown_as_a_cycle(config, store):
    import time

    from songe.cycle import model_learning

    now = time.time()
    early = model_learning(config, store, "m", {"stage": "read", "started_at": now - 60, "kind": "quick",
                                                "total": 10, "done": 2, "budget": 300, "paces": [1.0, 2.0]})
    assert early["current"] is None and early["quick_running"]["eta_seconds"] is None  # not 3 timed yet
    running = model_learning(config, store, "m", {"stage": "read", "started_at": now - 60, "kind": "quick",
                                                  "total": 10, "done": 4, "budget": 300, "paces": [8.0, 10.0, 12.0]})
    view = running["quick_running"]
    assert (view["done"], view["total"]) == (4, 10)
    # 6 left at the recent 10 s each = 60 s, within the 240 s left of the budget.
    assert 55 <= view["eta_seconds"] <= 65
    slow = model_learning(config, store, "m", {"stage": "read", "started_at": now - 60, "kind": "quick",
                                               "total": 10, "done": 4, "budget": 300, "paces": [50.0, 60.0, 70.0]})
    assert 235 <= slow["quick_running"]["eta_seconds"] <= 245  # capped by the budget
    # Idle reading is Quick Dream's first half: shown the same way, not as a Deep Dream.
    idle = model_learning(config, store, "m", {"stage": "read", "started_at": now, "kind": "read"})
    assert idle["current"] is None and idle["quick_running"]["kind"] == "read"
    deep = model_learning(config, store, "m", {"stage": "read", "started_at": 5.0, "kind": "deep"})
    assert deep["current"] is not None and deep["quick_running"] is None


def test_quick_dream_reports_reworded_and_dropped(config, store):
    add_signal(store, "t1", "The execute tool runs JavaScript only; use the shell tool for shell commands.",
               generality="always", text="", revision=1)
    add_signal(store, "c2", "Always keep the README under 30 lines.", revision=2)
    quick.quick_dream(store, config)
    result = quick.quick_dream(store, config, merged=[
        {"text": "Run shell commands with the shell tool.", "signals": ["t1"]}])
    assert (result["added"], result["reworded"], result["dropped"]) == (0, 1, 1)


def test_every_quick_dream_joins_the_history_and_quiet_ones_share_a_chip(config, store):
    """User decision (live: a Quick Dream read 5 conversations, found 9
    moments, added nothing, and left no trace): every finished Quick Dream
    is in the history; back-to-back ones that changed nothing are one chip."""
    from songe.cycle import dreams

    add_signal(store, "c1", "Always use Conventional Commits for commit messages.", revision=1)
    first = quick.record(store, quick.quick_dream(store, config), at=10.0)
    assert first["id"] == "q-0001" and first["added"] == ["Always use Conventional Commits for commit messages."]
    quiet = quick.record(store, quick.quick_dream(store, config), at=20.0, read=5, found=9)
    assert quiet["id"] == "q-0002" and quiet["added"] == [] and (quiet["read"], quiet["found"]) == (5, 9)
    quick.record(store, quick.quick_dream(store, config), at=30.0, read=2, found=0)
    failed = quick.record(store, quick.quick_dream(store, config), at=40.0, read=0, found=0, error="no_model")
    assert failed["error"] == "no_model"
    row = dreams(store)
    # Newest first: the failure, then the two quiet runs as one chip, then the first.
    assert [d["id"] for d in row] == ["q-0004", "q-0003", "q-0001"]
    assert row[1]["repeat"] == 2 and row[1]["read"] == 2 and row[0]["error"] == "no_model"


def test_the_dream_row_lists_quick_and_deep_dreams_with_seen(config, store):
    from songe.cycle import dreams, mark_seen

    night = store.night(None)
    store.night(night, status="rejected", record={"ended_at": 20.0, "reason": "Held-out improvement does not exceed bootstrap noise"})
    add_signal(store, "c1", "Always use Conventional Commits for commit messages.")
    quick.record(store, quick.quick_dream(store, config), at=30.0)
    row = dreams(store)
    assert [(d["id"], d["kind"], d["seen"]) for d in row] == [("q-0001", "quick", False), ("c-0001", "deep", False)]
    assert row[1]["outcome"] == "kept" and row[0]["changes"]["added"]
    mark_seen(store, "c-0001")
    assert [d["seen"] for d in dreams(store)] == [False, True]


def test_published_lessons_keep_their_wording(config, store):
    add_signal(store, "c1", "Always use Conventional Commits for commit messages.", revision=1)
    quick.quick_dream(store, config)
    (first,) = quick.active(store)
    # The same lesson again, worded differently, and a new one.
    add_signal(store, "c2", "Always use Conventional Commits with an imperative subject.", revision=2)
    add_signal(store, "c3", "Always keep the README under 30 lines.", revision=3)
    clusters = quick.candidates(store, 30, config.quick_min_sessions)
    kept, fresh = quick.split(store, [c for c in clusters if c["signals"] != ["c3"]]
                              + [{"text": "Short READMEs.", "signals": ["c3"]}])
    assert [k["text"] for k in kept] == [first["text"]] and len(fresh) >= 1
    result = quick.quick_dream(store, config, merged=fresh, kept=kept)
    lessons = quick.active(store)
    assert lessons[0]["text"] == first["text"] and lessons[0]["id"] == first["id"]
    assert result["changes"]["added"] and not result["changes"]["reworded"]


def test_the_users_own_rule_and_no_padding(config, store):
    add_signal(store, "u1", "Confirm package manager preferences before installing packages.",
               text="Always use pnpm for packages here, never npm.", revision=1)
    add_signal(store, "t1", "Always verify that the write tool gets the full path.", generality="always", text="",
               revision=2)
    quick.quick_dream(store, config)
    texts = [lesson["text"] for lesson in quick.active(store)]
    assert "Always use pnpm for packages here, never npm." in texts
    assert "Check that the write tool gets the full path." in texts
    assert quick.tidy("Always verify file paths before writing files.") == "Check file paths before writing files."
    assert quick.tidy("Always ensure that tests pass.") == "Make sure tests pass."
    assert quick.tidy(quick.tidy("Always verify that x works.")) == quick.tidy("Always verify that x works.")
    # A note in the user's own words can be a question: it keeps its mark.
    assert quick.tidy("Can you drop the em dashes?") == "Can you drop the em dashes?"


def test_only_a_rule_shaped_sentence_of_the_users_counts(config, store):
    by_id = {
        "a": {"type": "correction", "text": "Please remove those comments. Never add comments to the code unless I ask for them."},
        "b": {"type": "correction", "text": "You burned the entire game by repeating APPLE instead of playing from the feedback."},
    }
    assert quick.user_rule(by_id, ["a"]) == "Never add comments to the code unless I ask for them."
    assert quick.user_rule(by_id, ["b"]) is None


def test_the_users_rules_are_kept_first(config, store):
    store.set(quick.KEY, {"version": 1, "updated_at": 1, "removed": [], "lessons": [
        {"id": "p", "text": "Always use pnpm for packages here, never npm.", "signals": ["u1"], "count": 1,
         "added_at": 1, "rule": True}]})
    merged = [{"text": f"Tool lesson {i} " + "word " * 20, "signals": [f"t{i}{j}" for j in range(5)]} for i in range(6)]
    quick.quick_dream(store, config, merged=merged, kept=quick.active(store))
    assert quick.active(store)[0]["text"] == "Always use pnpm for packages here, never npm."


def test_a_recurrent_lesson_is_never_cut_for_a_rarer_one(config, store):
    merged = [{"text": "Rare lesson " + "word " * 40, "signals": ["r1"]},
              {"text": "Frequent lesson " + "word " * 40, "signals": ["f1", "f2", "f3"]}]
    quick.quick_dream(store, config, merged=merged)
    assert [lesson["count"] for lesson in quick.active(store)][0] == 3


def test_a_users_rule_counts_only_its_own_sentence(config, store):
    """Live run: one correction showed as x10 (look-alike lessons joined its
    cluster), and two pnpm signals made the same rule appear twice."""
    add_signal(store, "u1", "Do not add comments to code unless asked.",
               text="Please remove those comments. Never add comments to the code unless I ask for them.", revision=1)
    for i in range(5):  # look-alike tool lessons about code comments
        add_signal(store, f"t{i}", f"Do not add stray comments to code number {i}.", kind="tool_error",
                   generality="always", revision=2 + i)
    add_signal(store, "p1", "Prefer pnpm.", text="Always use pnpm for packages here, never npm.", revision=10)
    add_signal(store, "p2", "Use pnpm, not npm.", kind="retry", text="Always use pnpm for packages here, never npm.",
               revision=11)
    quick.quick_dream(store, config)
    lessons = {lesson["text"]: lesson["count"] for lesson in quick.active(store)}
    assert lessons["Never add comments to the code unless I ask for them."] == 1
    assert lessons["Always use pnpm for packages here, never npm."] == 2
    assert sum(1 for t in lessons if "pnpm" in t) == 1


def test_stems_merge_near_duplicates():
    assert quick.similar("Check JavaScript syntax validity before running.",
                         "Check the JavaScript syntax is valid before you run it.")


def test_only_standing_rules_count_and_echoes_are_rejected():
    rule = quick.rule_sentence
    assert rule("Wrong commit message. Use Conventional Commits for every commit.") == "Use Conventional Commits for every commit."
    assert rule("Always use pnpm for packages here, never npm.") == "Always use pnpm for packages here, never npm."
    assert rule("Stop doing whatever that was.") is None
    assert rule("Keep guessing from the feedback, don't go hunting for the answer.") is None
    assert rule("Never do that again.") is None  # too short to stand on its own
    assert quick.ECHO.search("Merge lessons 1, 2, 4 into one instruction: check paths.")
    assert not quick.ECHO.search("Check file paths before writing files.")


def test_lessons_published_under_looser_rules_leave(config, store):
    add_signal(store, "u1", "Do something.", text="Stop doing whatever that was.", revision=1)
    add_signal(store, "t1", "Check file paths before writing files.", generality="always", text="", revision=2)
    store.set(quick.KEY, {"version": 1, "updated_at": 1, "removed": [], "lessons": [
        {"id": "a", "text": "Stop doing whatever that was.", "signals": ["u1"], "count": 1, "added_at": 1, "rule": True},
        {"id": "b", "text": "Merge lessons 1, 2 into one instruction: check paths.", "signals": ["t1"], "count": 1,
         "added_at": 1}]})
    kept, fresh = quick.split(store, quick.candidates(store, 30, config.quick_min_sessions))
    assert kept == []
    quick.quick_dream(store, config, merged=fresh, kept=kept)
    assert [lesson["text"] for lesson in quick.active(store)] == ["Check file paths before writing files."]


def test_long_lessons_are_trimmed_to_their_core():
    assert quick.tidy("Check that the write tool is called with the correct full path, including the target file "
                      "name and directory structure, before attempting to create or overwrite files.") == \
        "Check that the write tool is called with the correct full path."
    assert quick.tidy("Always validate syntax and file integrity using the correct tool and language environment "
                      "before proceeding with edits.") == "Always validate syntax and file integrity."
    short = "Run shell commands with the shell tool."
    assert quick.tidy(short) == short and quick.tidy(quick.tidy(short)) == short


def test_the_short_form_from_reading_is_preferred(config, store):
    payload = {"id": "t1", "session": "s", "type": "correction", "source": "user", "generality": "always",
               "revision": 1, "lesson": "When creating or running commands that involve file paths and arguments, "
               "always verify the current working directory and ensure input file paths exist before executing.",
               "short": "Check the working directory and that input paths exist first."}
    with store.connect() as db:
        db.execute("INSERT INTO signals VALUES (?,?,?,?,?,NULL)", ("t1", store.model_id, "s", 1, json.dumps(payload)))
    quick.quick_dream(store, config)
    assert [lesson["text"] for lesson in quick.active(store)] == ["Check the working directory and that input paths exist first."]


def test_trimming_never_leaves_a_fragment():
    long = ("Check that the program entry point is correctly invoked and the code is syntactically valid JavaScript "
            "before running it with the test harness provided.")
    out = quick.tidy(long)
    assert len(out.split()) >= 6 and out.endswith(".")


def test_signals_without_a_short_form_are_read_again(config, store):
    from songe.reading import store as reading

    add_signal(store, "c1", "Always use Conventional Commits for commit messages.")
    with store.connect() as db:
        db.execute("INSERT INTO read_sessions VALUES (?,?,?,?)", ("s", store.model_id, 1, 0.0))
    assert reading.reread_without_short(store) == 1
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM read_sessions").fetchone()[0] == 0


def test_always_lessons_need_two_conversations(config, store):
    config.quick_min_sessions = 2
    tool = {"kind": "tool_error", "generality": "always"}
    add_signal(store, "t1", "Check remaining guesses before each guess.", revision=1, **tool)
    add_signal(store, "u1", "Never add comments unless asked.", text="Never add comments to the code unless I ask.",
               revision=2)
    quick.quick_dream(store, config)
    assert [lesson["text"] for lesson in quick.active(store)] == ["Never add comments to the code unless I ask."]


def test_a_published_lesson_takes_its_short_form_and_keeps_its_id(config, store):
    payload = {"id": "t1", "session": "s", "type": "correction", "source": "user", "generality": "always",
               "revision": 1, "lesson": "Check that the program entry point is invoked correctly.",
               "short": "Run the program's entry point before saying it works."}
    with store.connect() as db:
        db.execute("INSERT INTO signals VALUES (?,?,?,?,?,NULL)", ("t1", store.model_id, "s", 1, json.dumps(payload)))
    store.set(quick.KEY, {"version": 1, "updated_at": 1, "removed": [], "lessons": [
        {"id": "old", "text": "Check that the program entry point.", "signals": ["t1"], "count": 1, "added_at": 5}]})
    result = quick.quick_dream(store, config)
    (lesson,) = quick.active(store)
    assert lesson["text"] == "Run the program's entry point before saying it works."
    assert (lesson["id"], lesson["added_at"], result["added"]) == ("old", 5, 0)


def test_the_quick_view_says_what_it_reads_and_finds(config, store):
    import time

    from songe.cycle import model_learning
    from songe.reading import store as reading

    assert reading.opening_request([{"kind": "action"}, {"kind": "user", "content": "Add a --json flag  to report.py"}]) \
        == "Add a --json flag to report.py"
    assert reading.signal_line({"type": "correction", "source": "user", "text": "Never add comments.",
                                "short": "No comments."}) == {"type": "correction", "text": "Never add comments."}
    assert reading.signal_line({"type": "tool_error", "source": "tool", "short": "Use the shell tool.",
                                "evidence": "SyntaxError"})["text"] == "Use the shell tool."
    view = model_learning(config, store, "m", {
        "kind": "quick", "stage": "read", "started_at": time.time(), "total": 3, "done": 1, "budget": 300,
        "read_seconds": 5.0, "reading": "Add a --json flag", "found": [{"type": "correction", "text": "x"}],
        "found_count": 1})["quick_running"]
    assert (view["reading"], view["found_count"], view["found"][0]["type"]) == ("Add a --json flag", 1, "correction")


def test_the_moment_a_lesson_came_from(config, store):
    request = {"messages": [{"role": "system", "content": "agent"}, {"role": "user", "content": "Add lodash"}]}
    exchange = store.record("/v1/chat/completions", request, {"choices": []})
    payload = {"id": "t1", "session": "s", "type": "tool_error", "source": "tool", "generality": "always",
               "revision": 1, "lesson": "Use pnpm.", "exchanges": {"2": exchange}, "evidence": "npm: not allowed"}
    with store.connect() as db:
        db.execute("INSERT INTO signals VALUES (?,?,?,?,?,NULL)", ("t1", store.model_id, "s", 1, json.dumps(payload)))
    found = quick.moment(store, {"signals": ["t1"]})
    assert found["request"] == request and found["asked"] == "Add lodash" and found["feedback"] == "npm: not allowed"
    assert quick.moment(store, {"signals": ["missing"]}) is None


async def test_before_and_after_on_the_same_request():
    seen = []

    class Client:
        async def post(self, url, json):
            seen.append(json)
            lessons = "pnpm" in json["messages"][0]["content"]
            call = {"function": {"name": "shell", "arguments": '{"command": "%s"}' % (
                "pnpm add lodash" if lessons else "npm install lodash")}}

            class R:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"choices": [{"message": {"content": "", "tool_calls": [call]}}]}
            return R()

    request = {"model": "x", "stream": True, "messages": [{"role": "system", "content": "agent"},
                                                          {"role": "user", "content": "Add lodash"}]}
    before, after, changed = await quick.compare(Client(), "http://m/v1", "served", request, "",
                                                 quick.block([{"text": "Always use pnpm, never npm."}]), server="vllm")
    assert changed and before["tool_calls"][0]["arguments"] == {"command": "npm install lodash"}
    assert after["tool_calls"][0]["arguments"] == {"command": "pnpm add lodash"}
    assert seen[0]["temperature"] == 0 and seen[0]["stream"] is False and seen[0]["model"] == "served"
    assert seen[0]["chat_template_kwargs"] == {"enable_thinking": False}
    same = await quick.compare(Client(), "http://m/v1", "served", request, "", quick.block([{"text": "Be brief."}]))
    assert same[2] is False


def test_task_instructions_are_not_standing_rules():
    rule = quick.rule_sentence
    assert rule("Use the feedback from each guess.") is None  # an instruction for this task
    assert rule("Please use the feedback from each guess.") is None
    assert rule("Use pnpm for packages, never npm.") == "Use pnpm for packages, never npm."
    assert rule("Use tabs instead of spaces in this project.") == "Use tabs instead of spaces in this project."
    assert rule("Always write tests with pytest, never unittest.") is not None
    assert quick.normal("Please always use pnpm here.") == quick.normal("Always use pnpm here.")


def test_a_rule_statement_leads_to_its_moment(config, store):
    """Before/after for a stated rule: the last request before the user's
    message (rule signals carry no exchanges, only the message's item)."""
    request = {"messages": [{"role": "system", "content": "agent"}, {"role": "user", "content": "Add lodash"}]}
    items = [{"kind": "user", "content": "Add lodash"}, {"kind": "action", "step": 1, "exchange": 7},
             {"kind": "user", "content": "Always use pnpm here, never npm."}]
    payload = {"id": "r1", "session": "root", "type": "correction", "source": "user", "from": "rules",
               "item": 2, "text": "Always use pnpm here, never npm.", "lesson": "x", "generality": "always"}
    with store.connect() as db:
        db.execute("INSERT INTO signals VALUES (?,?,?,?,?,NULL)", ("r1", store.model_id, "root", 1, json.dumps(payload)))
    import songe.reading.timeline as timeline

    original = (timeline.sessions, quick.reading.load_session)
    timeline.sessions = lambda db, cutoff=None: {"root": [7]}
    quick.reading.load_session = lambda store, ids: ([], items)
    try:
        with store.connect() as db:
            db.execute("INSERT INTO exchanges(id,created,path,request,response,status,complete,binding) "
                       "VALUES (7,0,'/v1/chat/completions',?,NULL,200,1,'b')", (json.dumps(request),))
        found = quick.moment(store, {"signals": ["r1"]})
    finally:
        timeline.sessions, quick.reading.load_session = original
    assert found["request"] == request and found["feedback"] == "Always use pnpm here, never npm."


def test_old_rule_captures_are_checked_again(config, store):
    payload = {"id": "r1", "session": "s", "type": "correction", "source": "user", "from": "rules",
               "revision": 1, "text": "Please use the feedback from each guess.", "lesson": "x",
               "short": "Please use the feedback from each guess.", "generality": "always"}
    with store.connect() as db:
        db.execute("INSERT INTO signals VALUES (?,?,?,?,?,NULL)", ("r1", store.model_id, "s", 1, json.dumps(payload)))
    quick.quick_dream(store, config)
    assert quick.active(store) == []


async def test_compare_drops_streaming_and_length_settings():
    sent = []

    class Client:
        async def post(self, url, json):
            sent.append(json)

            class R:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"choices": [{"message": {"content": "ok"}}]}
            return R()

    request = {"messages": [{"role": "user", "content": "hi"}], "stream": True,
               "stream_options": {"include_usage": True}, "max_completion_tokens": 9000}
    await quick.compare(Client(), "http://m/v1", "served", request, "", "")
    assert "stream_options" not in sent[0] and "max_completion_tokens" not in sent[0]
    assert sent[0]["max_tokens"] == quick.ANSWER_TOKENS


def test_a_rule_replays_the_action_it_reacted_to():
    items = [
        {"kind": "user", "content": "Add the lodash package"},
        {"kind": "action", "exchange": 1, "tool_calls": [{"name": "shell", "arguments": '{"command": "ls -la"}'}]},
        {"kind": "action", "exchange": 2, "tool_calls": [{"name": "shell", "arguments": '{"command": "npm install lodash"}'}]},
        {"kind": "action", "exchange": 3, "tool_calls": [{"name": "shell", "arguments": '{"command": "cat package.json"}'}]},
        {"kind": "action", "exchange": 4, "text": "Done.", "tool_calls": []},
    ]
    assert quick.reacted_to(items, "Always use pnpm for packages here, never npm.") == 2
    assert quick.reacted_to(items, "Always write tests with pytest, never unittest.") == 3  # latest tool call


def test_status_says_it_is_reading_not_waiting_for_a_deep_dream(config, store):
    from types import SimpleNamespace

    from songe.status import activity

    scheduler = SimpleNamespace(reason="Waiting for an idle gap", last_traffic=0.0, idle_threshold_seconds=10.0,
                                reading_activity={"m": {"kind": "read", "done": 19, "total": 51}})
    out = activity(config, store, scheduler, None, [], live=True)
    assert out["now"] == "reading your conversations · 19 of 51"
    assert out["next"].startswith("then the preferences in the prompt update · a Deep Dream after")


def test_a_rule_about_scripts_replays_the_script_written_not_a_later_edit():
    """Live demo: "always put my name and the date in the top of the script"
    replayed a small edit of main() instead of the write of the script."""
    script = "#!/usr/bin/env python3\n# 2048 game script\ndef main():\n    grid = new_grid()\n"
    items = [
        {"kind": "user", "content": "can you write a 2048 game for me"},
        {"kind": "action", "exchange": 1, "tool_calls": [
            {"name": "write", "arguments": json.dumps({"path": "2048.py", "content": script})}]},
        {"kind": "action", "exchange": 2, "tool_calls": [
            {"name": "edit", "arguments": json.dumps({"path": "2048.py", "oldText": "grid", "newText": "grid = new_grid()"})}]},
    ]
    assert quick.reacted_to(items, "always put my name and the date in the top of the script.") == 1


def test_an_answer_cut_off_mid_file_still_shows_what_it_wrote():
    message = {"tool_calls": [{"function": {"name": "write", "arguments":
               '{"path": "2048.py", "content": "#!/usr/bin/env python3\\n# Frank Wu, 2026-11-19\\nimport ra'}}]}
    (call,) = quick.answer_of(message)["tool_calls"]
    assert call["arguments"] == {"path": "2048.py", "content": "#!/usr/bin/env python3\n# Frank Wu, 2026-11-19\nimport ra"}


def test_a_reworded_preamble_before_the_same_call_is_not_a_change():
    """Live demo: "provide the final working game" vs "provide a brief usage
    note" before the same edit was shown as a before/after."""
    edit = [{"name": "edit", "arguments": {"path": "2048.py", "newText": "main()"}}]
    assert not quick.differs({"text": "I'll complete it and then provide the final working game.", "tool_calls": edit},
                             {"text": "I'll complete main() and then provide a brief usage note.", "tool_calls": edit})
    written = [{"name": "write", "arguments": {"path": "2048.py", "content": "# Frank Wu\nimport random"}}]
    assert quick.differs({"text": "", "tool_calls": edit}, {"text": "", "tool_calls": written})
    assert not quick.differs({"text": "Done, the tests pass.", "tool_calls": []}, {"text": "Done; the tests pass.", "tool_calls": []})
    assert quick.differs({"text": "Done.", "tool_calls": []}, {"text": "I ran pytest first: 12 passed. Done.", "tool_calls": []})


async def test_the_judge_says_which_answer_follows_the_lesson_whatever_the_order():
    def client(says_new):
        class Client:
            async def post(self, url, json):
                prompt = json["messages"][0]["content"]
                a_is_new = prompt.index("pnpm add") < prompt.index("npm install")
                letter = ("A" if a_is_new else "B") if says_new else ("B" if a_is_new else "A")

                class R:
                    def raise_for_status(self):
                        pass

                    def json(self):
                        return {"choices": [{"message": {"content": '{"follows": "%s"}' % letter}}]}
                return R()
        return Client()

    before = {"text": "", "tool_calls": [{"name": "shell", "arguments": {"command": "npm install lodash"}}]}
    after = {"text": "", "tool_calls": [{"name": "shell", "arguments": {"command": "pnpm add lodash"}}]}
    for lesson in ("Always use pnpm, never npm.", "Use pnpm here.", "pnpm only", "Never npm."):
        assert await quick.judge(client(True), "http://m/v1", "served", lesson, before, after) == "after"
        assert await quick.judge(client(False), "http://m/v1", "served", lesson, before, after) == "before"

    class Broken:
        async def post(self, url, json):
            raise RuntimeError("server gone")
    assert await quick.judge(Broken(), "http://m/v1", "served", "x", before, after) is None


def test_a_stated_rule_carries_the_step_it_is_about_so_it_can_be_practiced():
    items = [
        {"kind": "user", "content": "Add lodash"},
        {"kind": "action", "step": 1, "exchange": 11, "tool_calls": [{"name": "shell", "arguments": '{"command": "npm install lodash"}'}]},
        {"kind": "user", "content": "Always use pnpm for packages here, never npm."},
    ]
    about = quick.reacted_action(items[:2], "Always use pnpm for packages here, never npm.")
    assert (about["step"], about["exchange"]) == (1, 11)


async def test_cancel_after_pause_leaves_nothing_paused(config, store):
    """Live: Pause, then Cancel Dream; the button still said Resume."""
    from types import SimpleNamespace

    import httpx

    from songe.proxy import create_app

    async def cancel(model):
        return True

    app = create_app(config, store, scheduler=SimpleNamespace(cancel=cancel, interrupt=lambda *a: None, active_model=None))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
        await client.post("/api/control", json={"action": "pause"})
        assert store.get("paused") is True
        done = (await client.post("/api/control", json={"action": "cancel"})).json()
    assert done["ok"] and "next Deep Dream" in done["message"] and store.get("paused") is False


def test_what_the_user_sees_of_an_answer():
    assert quick.visible("I should check.\n</think>\nLet me look.\n<tool_call>\n<function=read>") == "Let me look."
    assert quick.visible("<think>hmm</think>Done.") == "Done."
    assert quick.visible("Plain answer.") == "Plain answer."


def test_a_deep_dream_example_shows_only_if_the_action_differs():
    """Live: two "let me check the directory structure" answers, scored
    better, were shown as a before/after."""
    from songe.cycle import shown_examples

    def ex(i, before, after):
        return {"id": str(i), "improved": True, "verdict": "after", "source": "original", "teacher_preference": 1.0,
                "signal": {"id": f"s{i}", "source": "user"},
                "before": {"text": before, "tool_calls": []}, "after": {"text": after, "tool_calls": []}}
    same = ex(1, "The path might be wrong.\n</think>\nLet me check the directory structure.",
              "The write failed.\n</think>\nLet me check the directory structure.")
    prose = ex(2, "Here is the README with a table of options.", "Here is the README; options are listed as bullets.")
    real = ex(3, "", "")
    real["before"]["tool_calls"] = [{"name": "edit", "arguments": {"path": "cli.py", "newText": "polygon_perimeter"}}]
    real["after"]["tool_calls"] = [{"name": "read", "arguments": {"path": "cli.py"}}]
    cut = ex(4, "I'll edit it.\n<tool_call>\n<function=edit>\n<parameter=path>", "")
    cut["after"]["tool_calls"] = [{"name": "read", "arguments": {"path": "README.md"}}]
    # Prose only, or one side's call cut off mid-way: not shown.
    assert [e["id"] for e in shown_examples([same, prose, real, cut])] == ["3"]


def test_undoing_a_quick_dream_removes_what_it_added_for_good(config, store):
    """Users can undo a Quick Dream: the preferences it added leave new
    sessions and are not added again."""
    add_signal(store, "c1", "Always use Conventional Commits for commit messages.", revision=1)
    entry = quick.record(store, quick.quick_dream(store, config), at=10.0)
    assert [lesson["text"] for lesson in quick.active(store)] == ["Always use Conventional Commits for commit messages."]
    assert quick.undo(store, entry["id"]) == 1
    assert quick.active(store) == []
    assert store.get(quick.HISTORY)[0]["undone"]
    # The next Quick Dream does not bring it back.
    quick.quick_dream(store, config)
    assert quick.active(store) == []
    import pytest
    with pytest.raises(ValueError):
        quick.undo(store, "q-9999")


async def test_the_undo_quick_control(config, store):
    from types import SimpleNamespace

    import httpx

    from songe.proxy import create_app

    add_signal(store, "c1", "Always use Conventional Commits for commit messages.")
    entry = quick.record(store, quick.quick_dream(store, config), at=10.0)
    app = create_app(config, store, scheduler=SimpleNamespace(active_model=None))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
        reply = (await client.post("/api/control", json={"action": "undo_quick", "id": entry["id"]})).json()
    assert reply["ok"] and reply["message"].startswith("Quick Dream undone · 1 preference removed")
    assert quick.active(store) == []


def test_non_latin_lessons_do_not_collide_resurrect_or_wipe_on_undo(config, store):
    """Non-Latin lessons (Arabic, CJK) must stay distinct, respect remove(), and
    only remove their own Quick Dream entries on undo()."""
    ar_comments = "لا تضف تعليقات في الكود أبداً"
    ar_formal = "اكتب الشرح دائماً بالعربية الفصحى"
    zh_rule = "请总是使用简洁的中文回复"

    assert quick.normal(ar_comments) != quick.normal(ar_formal)
    assert quick.normal("لا تضف تعليقات في الكود أبداً") == quick.normal("لا تضف تعليقات في الكود أبدا")
    assert quick.similar(ar_comments, ar_comments)
    assert quick.similar(zh_rule, "总是使用简洁的中文回复")

    add_signal(store, "a1", ar_comments, generality="always", text="", revision=1)
    e1 = quick.record(store, quick.quick_dream(store, config, now=10.0), at=10.0)
    add_signal(store, "a2", ar_formal, generality="always", text="", revision=2)
    add_signal(store, "z1", zh_rule, generality="always", text="", revision=3)
    e2 = quick.record(store, quick.quick_dream(store, config, now=20.0), at=20.0)

    # All three distinct non-Latin lessons survive (none collapsed into an empty key).
    assert len(quick.active(store)) == 3

    # Undoing Quick Dream 2 removes only its two lessons, leaving Quick Dream 1's Arabic lesson intact.
    assert quick.undo(store, e2["id"]) == 2
    remaining = quick.active(store)
    assert len(remaining) == 1 and "تعليقات" in remaining[0]["text"]

    # Removing the remaining Arabic lesson keeps it removed on subsequent Quick Dreams.
    quick.remove(store, remaining[0]["id"])
    quick.quick_dream(store, config, now=30.0)
    assert quick.active(store) == []
    assert e1["id"] != e2["id"]


def test_ten_sentences_in_the_users_own_words_fit_the_list(config, store):
    """Live (Qwen3.5-4B, 20 simulated users with ten stated preferences each,
    605 to 890 characters per user): with room for 600 characters 155 of 200
    sentences were in the prompt, on average 7.9 of a user's ten."""
    habits = ["answer in metric units and leave the imperial ones out, even in recipes",
              "write dates the ISO way, year first, wherever a date turns up in a text",
              "use pnpm for anything that installs a JavaScript package, not npm or yarn",
              "keep code samples free of comments unless I ask for an explanation",
              "start every answer with the short version before the details",
              "quote prices in euros and say when a figure is only an estimate",
              "prefer the standard library over a new dependency for small jobs",
              "write commit messages in the imperative mood and under sixty characters",
              "name the file a snippet belongs to before showing the snippet",
              "use British spelling in prose and American spelling in code identifiers"]
    for n, habit in enumerate(habits):
        add_signal(store, f"r{n}", f"Always {habit}.", revision=n + 1)
    assert 600 < sum(len(habit) + 8 for habit in habits) <= quick.PROMPT_CHARS
    quick.quick_dream(store, config)
    assert len(quick.active(store)) == 10


def test_a_change_wanted_once_takes_two_conversations_and_a_standing_wish_one(config, store):
    # Reading judged each user turn (prompts.TURN): that decides, whatever
    # label the lesson got.
    def said(sid, session, turn, lesson):
        payload = {"id": sid, "session": session, "type": "correction", "source": "user", "lesson": lesson,
                   "generality": "always", "revision": 1, "text": "shorter please", "turn_kind": turn}
        with store.connect() as db:
            db.execute("INSERT INTO signals VALUES (?,?,?,?,?,NULL)", (sid, store.model_id, session, 1, json.dumps(payload)))

    def lessons():
        return {c["text"] for c in quick.candidates(store, 30, config.quick_min_sessions)}

    said("w1", "a", "lasting", "Answer in Russian.")
    said("o1", "a", "once", "Keep the summary under one hundred words.")
    said("n1", "a", "task", "Write unit tests for every new function.")
    said("r1", "a", "reaction", "State plainly that you are an AI assistant.")
    assert lessons() == {"Answer in Russian."}
    said("o2", "b", "once", "Keep the summary under a hundred words.")
    assert {text.split(" under ")[0] for text in lessons()} == {"Answer in Russian.", "Keep the summary"}


def test_the_request_at_hand_wins_over_a_lesson():
    """Live (Qwen3.5-4B on vLLM, through Songe): a lesson says "I avoid spicy
    food." and the user asks, in so many words, for something really spicy
    just this once. With the heading "Follow them unless the current request
    says otherwise" the model gave what was asked in 14 of 20 such requests,
    and in 12 of 20 more that ask in other words; with no lesson in the prompt
    it does in all 20. The heading is far from the request and the model read
    past it. Said in a line of its own after the list, next to the request:
    20 of 20 and 18 of 20, the lesson stayed, and it was followed as before
    in every request that asked for no exception."""
    text = quick.block([{"text": "I avoid spicy food."}, {"text": "Always use pnpm, never npm."}])
    assert text.split("\n") == [quick.BLOCK, "- I avoid spicy food.", "- Always use pnpm, never npm.", quick.EXCEPTIONS]
    # The lessons are followed, the request at hand comes first, and only for this request.
    assert "Follow them" in quick.BLOCK and "unless" not in quick.BLOCK
    assert all(said in quick.EXCEPTIONS for said in ("current request", "this time", "every other request"))
    assert quick.block([]) == ""


def test_a_rule_shaped_sentence_follows_what_its_turn_was(config, store):
    # "do not include Lofi." in a request for channel names is shaped like a
    # rule; what the turn was judged to be decides, as for any other feedback.
    def said(sid, session, turn, sentence, **more):
        payload = {"id": sid, "session": session, "type": "correction", "source": "user", "from": "rules",
                   "lesson": sentence, "short": sentence, "generality": "always", "revision": 1, "text": sentence,
                   "turn_kind": turn} | more
        with store.connect() as db:
            db.execute("INSERT INTO signals VALUES (?,?,?,?,?,NULL)", (sid, store.model_id, session, 1, json.dumps(payload)))

    def lessons():
        return {c["text"] for c in quick.candidates(store, 30, config.quick_min_sessions)}

    said("r1", "a", "lasting", "Always answer in French.")
    said("r2", "a", "task", "Do not apply any of these instructions in your standard answer.")
    said("r3", "a", "once", "Do not include Lofi.")
    said("r4", "a", None, "Never use tabs.")  # read before turns were judged: as before
    # A preference stated about oneself keeps its own check, whatever the turn.
    said("r5", "a", "reaction", "I avoid dairy.", preference="I avoid dairy.")
    assert lessons() == {"Always answer in French.", "Never use tabs.", "I avoid dairy."}
    said("r6", "b", "once", "Do not include Lofi.")
    assert lessons() == {"Always answer in French.", "Never use tabs.", "I avoid dairy.", "Do not include Lofi."}


def noted(store, sid, session, note, *, revision, turn="lasting"):
    """A note reading wrote from what the user asked for in one conversation
    (its turn judged a standing wish, or a change wanted once)."""
    payload = {"id": sid, "session": session, "type": "correction", "source": "user", "lesson": note,
               "generality": "always", "revision": revision, "text": "shorter please", "turn_kind": turn}
    with store.connect() as db:
        db.execute("INSERT INTO signals VALUES (?,?,?,?,?,NULL)",
                   (sid, store.model_id, session, revision, json.dumps(payload)))


class Folding:
    """The model asked which kept notes a new note could repeat (it answers
    with the numbers given for that note) and whether two notes are one wish
    (the pairs given are). Keeps the questions."""

    def __init__(self, same=None, one_wish=(), one_way=()):
        # one_way: pairs that are one wish only when shown in this order.
        self.same, self.one_wish, self.one_way = same or {}, set(one_wish) | set(one_way), set(one_way)
        self.listed, self.pairs = [], []

    async def ask(self, prompt, schema, name, **kw):
        if name == "same_note":
            self.listed.append(prompt)
            return {"same": self.same.get(prompt.split("A new note:\n", 1)[1].split("\n")[0], [])}
        assert name == "same_wish", name
        pair = (prompt.split("Note 1: ", 1)[1].split("\n")[0], prompt.split("Note 2: ", 1)[1].split("\n")[0])
        self.pairs.append(pair)
        return {"note_1_asks": pair[0], "note_2_asks": pair[1], "kind": "same" if pair in self.one_wish or pair[::-1] in self.one_wish - self.one_way else "different"}


def test_one_wish_said_in_other_words_takes_one_place(config, store):
    """Live (Qwen3.5-4B on vLLM): eight habits, each asked for in three
    conversations in everyday words ("In metric please.", "Celsius please, I
    don't think in Fahrenheit.", "Can I get that in meters only?"). The notes
    of one habit were worded differently each time. Those read as a standing
    wish each took a place of their own; those read as a change wanted once
    never counted as the same wish in a second conversation, and stayed out.
    Three readings: 6, 6 and 7 of the 8 habits had a note, in 9 places each.
    With the notes folded (28 of 60, none wrongly): 7, 7 and 7 habits in 7, 9
    and 8 places; the eighth was never read as a wish. Of 128 held-out
    requests the model then followed 76, 81 and 75, against 75, 52 and 75."""
    from songe.reading import prompts as P

    meters, celsius, uv, metric = "Use meters only.", "Use Celsius, not Fahrenheit.", "Use uv, not pip.", "Use metric units."
    noted(store, "m1", "a", meters, revision=1)
    noted(store, "u1", "b", uv, revision=2)
    noted(store, "m2", "c", celsius, revision=3)
    # The model names nothing: every note has its own place, as before.
    reader = Folding()
    assert asyncio.run(reading.fold_same(store, reader)) == 0 and reader.pairs == []
    assert reader.listed == [P.SAME.format(notes=f"1. {meters}", note=uv),
                             P.SAME.format(notes=f"1. {meters}\n2. {uv}", note=celsius)]
    quick.quick_dream(store, config, now=1.0)
    assert {lesson["text"] for lesson in quick.active(store)} == {meters, uv, celsius}

    # A fourth conversation, a fourth wording. The model also names a note
    # about something else: asked about that pair alone, it says so.
    noted(store, "m3", "d", metric, revision=4)
    reader = Folding(same={celsius: [2, 1], metric: [9, "1", True, 1]}, one_wish={(meters, celsius), (meters, metric)})
    assert asyncio.run(reading.fold_same(store, reader)) == 2
    # An answer that names no note is ignored; a pair said to be one wish is
    # asked again the other way round.
    assert reader.pairs == [(uv, celsius), (meters, celsius), (celsius, meters), (meters, metric), (metric, meters)]
    result = quick.quick_dream(store, config, now=2.0)
    lessons = {lesson["text"]: lesson for lesson in quick.active(store)}
    # One place for the wish, in the place one of its notes had, and no
    # wording is lost: each stands in it, the most recent first.
    one = lessons[f"{metric} {celsius} {meters}"]
    assert set(lessons) == {uv, one["text"]}
    assert set(one["signals"]) == {"m1", "m2", "m3"} and one["count"] == 3 and one["added_at"] == 1.0
    assert [result[k] for k in ("lessons", "added", "reworded", "dropped")] == [2, 0, 1, 1]
    assert quick.block(quick.active(store)).count("\n- ") == 2

    # The user comes back while the model is asked: what was found stands.
    class Back(Folding):
        async def ask(self, prompt, schema, name, **kw):
            raise Stopped()

    assert asyncio.run(reading.fold_same(store, Back())) is None
    assert store.get(reading.SAME) == {"m2": "m1", "m3": "m1"}
    # Asked again (a real reader answers from its cache): nothing changes.
    assert asyncio.run(reading.fold_same(store, reader)) == 2
    assert quick.quick_dream(store, config, now=3.0)["version"] == result["version"]
    # The user removes the lesson: it does not return under one of its wordings.
    quick.remove(store, one["id"])
    quick.quick_dream(store, config, now=4.0)
    assert [lesson["text"] for lesson in quick.active(store)] == [uv]


def test_two_notes_are_one_wish_only_when_the_model_says_so_both_ways_round(config, store):
    """Live (Qwen3.5-4B): shown "uv please." and then "Same thing without the
    comments.", the model said the second again as "Use uv without comments"
    and called them one wish; the other way round it did not. A place of the
    prompt then held three habits."""
    uv, comments = "uv please.", "Same thing without the comments."
    noted(store, "u1", "a", uv, revision=1)
    noted(store, "c1", "b", comments, revision=2)
    reader = Folding(same={comments: [1]}, one_way={(uv, comments)})
    assert asyncio.run(reading.fold_same(store, reader)) == 0
    assert reader.pairs == [(uv, comments), (comments, uv)]
    assert asyncio.run(reading.fold_same(store, Folding(same={comments: [1]}, one_wish={(uv, comments)}))) == 1


def test_a_change_wanted_once_counts_in_each_conversation_however_it_was_worded(config, store):
    # A change wanted once becomes a lesson when the wish comes up in a second
    # conversation. Worded differently there, it used to count as another wish.
    under, shorter = "Keep the summary under one hundred words.", "Write shorter summaries."
    noted(store, "o1", "a", under, revision=1, turn="once")
    noted(store, "o2", "b", shorter, revision=2, turn="once")

    def lessons():
        return [c["text"] for c in quick.candidates(store, 30, config.quick_min_sessions)]

    assert not quick.similar(under, shorter) and lessons() == []
    assert asyncio.run(reading.fold_same(store, Folding(same={shorter: [1]}, one_wish={(under, shorter)}))) == 1
    assert lessons() == [f"{shorter} {under}"]


def test_a_place_holds_no_more_than_one_lesson_of_text(config, store):
    # Folding never drops a wording, so a note is only folded where both fit.
    first = "Always write the summary of a meeting as a list of decisions, then a list of open questions, then who does what by when, and leave out the small talk."
    second = "Never open a meeting summary with the attendance list or the weather report, open it with what was decided and what is still open, then owners and dates."
    add_signal(store, "r1", first, revision=1)
    add_signal(store, "r2", second, revision=2)
    assert len(f"{first} {second}") > quick.LESSON_CHARS
    reader = Folding(same={second: [1]}, one_wish={(first, second)})
    assert asyncio.run(reading.fold_same(store, reader)) == 0 and reader.listed == []
    quick.quick_dream(store, config)
    assert {lesson["text"] for lesson in quick.active(store)} == {first, second}


def test_a_note_says_its_projects_own_names_for_any_project(config, store):
    """Reported by Jack in a test on 11-24: notes named one practice repo's
    folder ("Use src/notes_app/ in all file paths."), and in another
    repository the model then wrote into notes_app. With Qwen3.8-27B, main
    still published "Put code under src/recipe_box/." and the test command
    once per module ("make test MODULE=auth", "MODULE=parser", ...)."""
    assert quick.general("Put code under src/billing_api/.", "billing-api") == "Put code under src/<package>/."
    assert quick.general("Clone billing-api first.", "billing-api") == "Clone <project> first."
    assert quick.general("Keep the app small.", "app") == "Keep the app small."  # too common a name
    for n, (project, module) in enumerate([("billing-api", "invoices"), ("notes-app", "auth"), ("invoice-tool", "parser")]):
        add_signal(store, f"t{n}", f"Run tests with make test MODULE={module}.", session=f"s{n}", project=project,
                   revision=n + 1, text=f"don't call pytest directly, we run tests with make test MODULE={module}",
                   turn_kind="lasting")
    add_signal(store, "p1", "Put code under src/billing_api/.", session="s0", project="billing-api", revision=4,
               text="nothing lives at the repo root, the code is all under src/billing_api/.", turn_kind="lasting",
               short="Put code under src/billing_api/.")
    add_signal(store, "p2", "Put code under src/recipe_box/.", session="s4", project="recipe-box", revision=5,
               text="nothing lives at the repo root, the code is all under src/recipe_box/.", turn_kind="lasting",
               short="Put code under src/recipe_box/.")
    quick.quick_dream(store, config)
    lessons = {lesson["text"]: lesson["count"] for lesson in quick.active(store)}
    assert lessons == {
        "don't call pytest directly, we run tests with make test MODULE=<module>": 3,
        "Put code under src/<package>/.": 2,
    }


def test_the_working_folder_is_read_from_the_agents_system_message():
    def rows(system):
        return [(1, {"messages": [{"role": "system", "content": system}, {"role": "user", "content": "hi"}]}, None, 200)]

    assert reading.project_of(rows("You are opencode.\n<env>\n  Working directory: /home/dev/billing-api\n</env>")) == "billing-api"
    assert reading.project_of(rows("Current working directory: C:\\Users\\me\\notes-app\\")) == "notes-app"
    assert reading.project_of(rows("<cwd>/Users/me/code/invoice-tool</cwd>")) == "invoice-tool"
    assert reading.project_of(rows("You are a helpful assistant.")) is None


@pytest.mark.parametrize("server", ["ollama", "mlx", "llamacpp", "vllm"])
async def test_examples_of_a_thinking_model_are_asked_without_its_reasoning(server):
    """Reported by Jack in a test on 11-24: qwen3 on Ollama put its whole
    capped answer in the reasoning field, so the page showed both sides of
    every before and after example empty. The server here answers the way
    Ollama does: in the reasoning field unless reasoning is switched off."""
    sent = []

    class Client:
        async def post(self, url, json):
            sent.append(json)
            off = json.get("reasoning_effort") == "none" or (
                json.get("chat_template_kwargs") or {}).get("enable_thinking") is False
            message = ({"content": "Run: make test MODULE=parser"} if off
                       else {"content": "", "reasoning": "Okay, the user wants"})

            class R:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"choices": [{"message": message}]}
            return R()

    request = {"messages": [{"role": "user", "content": "how do I run the parser tests?"}]}
    before, after, _ = await quick.compare(Client(), "http://m/v1", "served", request, "",
                                           quick.block([{"text": "Run tests with make test."}]), server=server)
    assert before["text"] and after["text"]
    assert all(body["reasoning_effort"] == "none" for body in sent)
    switch = server in {"mlx", "llamacpp", "vllm"}
    assert all(("chat_template_kwargs" in body) == switch for body in sent)
