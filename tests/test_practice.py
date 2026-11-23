"""Practice: scenes written from signals, framed on the real request, checked,
kept or dropped. The model is faked; nothing in a scene is ever run."""

import asyncio
import copy
import json

from songe.practice import store as practice
from songe.practice.checks import STOCK_LESSONS, placebo_for
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


def written(store, scene=None, pick=None, per_signal=1):
    record_session(store)
    asyncio.run(reading.read(store, FakeReader()))
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
    # One session only: the forced choice still runs, against a stock lesson.
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
    record_session(store)
    asyncio.run(reading.read(store, FakeReader()))
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
    # With one session there is no other lesson to compare with; the scene
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
    with store.connect() as db:
        session, ids = next(iter(timeline.sessions(db).items()))
    reading.save(store, session, max(ids) + 1, [], [])
    assert practice.practice(store) == []


def test_forget_deletes_practice(store):
    written(store)
    store.forget()
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM practice").fetchone()[0] == 0


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
