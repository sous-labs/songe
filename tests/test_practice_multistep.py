"""Practice scenes with right steps: every step from the moment to the one
the lesson is about is a training sample, and an agent's scene is kept only
when the note changes what the model does at that last step. The model is
faked; nothing in a scene is ever run."""

import asyncio
import json

from songe.practice import pool
from songe.practice import store as practice
from songe.practice.frame import frame_from_request, right_path
from tests.test_practice import Writer, corrected_twice
from tests.test_reading import SCENE, args

STEPPED = SCENE | {
    "mistake": "commit with a past-tense subject",
    "right_action": "commit with a Conventional Commits subject",
    "right_steps": [
        {"say": "Let me see the change.", "tool": "shell", "args": args(command="git diff --stat"),
         "output": " tax.py | 2 +-\n 1 file changed"},
        {"say": "", "tool": "shell", "args": args(command="git commit -am 'fix: round VAT to the cent'"),
         "output": "[main 1a2b3c4] fix: round VAT to the cent"},
    ],
}

TOOLS = [{"type": "function", "function": {"name": "shell", "parameters": {
    "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]


def framed():
    request = {"messages": [{"role": "system", "content": "You are a coding agent."},
                            {"role": "user", "content": "fix it"}], "tools": TOOLS}
    return frame_from_request(request, "invoice-calc")


def test_each_later_right_step_gets_its_own_context():
    """The first right step is taken where the scene ends; the next one sees
    the first done, with the output the scene wrote for it."""
    from songe.practice.frame import build_path

    frame = framed()
    messages, problems, _, contexts = build_path(STEPPED | {"steps": []}, frame)
    assert problems == []
    (later,) = contexts
    assert later[: len(messages)] == messages
    call, result = later[len(messages):]
    assert json.loads(call["tool_calls"][0]["function"]["arguments"]) == {"command": "git diff --stat"}
    assert result == {"role": "tool", "tool_call_id": call["tool_calls"][0]["id"],
                      "content": " tax.py | 2 +-\n 1 file changed"}
    # A scene with one right step has no later context; the last step is never added.
    assert right_path(STEPPED | {"steps": [], "right_steps": STEPPED["right_steps"][1:]}, frame) == ([], [])


def test_a_reply_before_the_last_right_step_is_a_problem():
    # The user would have to answer it, and the scene says nothing of what they say.
    early_reply = STEPPED | {"steps": [], "right_steps": [
        {"say": "Shall I commit?", "tool": "", "args": args(), "output": ""}, *STEPPED["right_steps"]]}
    contexts, problems = right_path(early_reply, framed())
    assert contexts == [] and problems == ["right step 1: a reply before the last step"]


class Acting(Writer):
    """Writes STEPPED. The teacher (the note shown) acts with `teacher`, one
    command per call in turn (the last repeated); the student with `plain`.
    Shell output is simulated as `printed`; the judge picks the teacher's
    commit."""

    def __init__(self, plain="git commit -am 'Fixed VAT'", teacher=("git commit -am 'fix: round VAT'",), **kw):
        super().__init__(STEPPED, **kw)
        self.plain, self.teacher, self.replied, self.walked = plain, list(teacher), [], 0
        self.printed = " M tax.py"

    async def ask(self, prompt, schema, name, *, temperature=0, max_tokens=None):
        if name == "simulate":
            self.asked.append(name)
            return {"output": self.printed}
        if name == "changes":
            self.asked.append(name)
            self.prompts.append(prompt)
            first = prompt.split("Answer A:\n", 1)[1].split("\n\nAnswer B:", 1)[0]
            return {"follows": "A" if self.teacher[-1] in first else "B"}
        return await super().ask(prompt, schema, name, temperature=temperature, max_tokens=max_tokens)

    async def reply(self, messages, *, max_tokens, temperature=0, think=None, tools=None):
        self.replied.append((messages, tools))
        noted = "Not part of the conversation" in str(messages[-1].get("content"))
        if noted:
            command = self.teacher[min(self.walked, len(self.teacher) - 1)]
            self.walked += 1
        else:
            command = self.plain
        if command is None:
            return {"content": "Done.", "tool_calls": [], "finished": True, "reasoned": False}
        return {"content": "", "tool_calls": [{"name": "shell", "arguments": json.dumps({"command": command})}],
                "finished": True, "reasoned": False}


def acted(store, writer):
    corrected_twice(store)
    asyncio.run(practice.write(store, writer, 1))
    return practice.practice(store)


def test_the_teacher_walks_to_the_step_and_the_scene_is_kept(store):
    """The teacher commits straight away: the walk ends at once, and the
    note changes what is committed."""
    writer = Acting()
    (row,) = acted(store, writer)
    assert row["walk"]["end"] == "reached" and row["path"] == []
    assert row["drop"] is None and row["checks"]["reached"] is True and row["checks"]["changes"] is True
    # The forced choice between lessons is not asked of a scene with right steps.
    assert "matters" not in writer.asked and row["checks"]["matters"] is None
    # The teacher is asked for an action, as training asks it, with the agent's tools.
    walked, tools = writer.replied[0]
    assert walked[:-1] == row["messages"] and "Respond now with your own action." in walked[-1]["content"]
    assert {t["function"]["name"] for t in tools} >= {"shell"}
    assert row["teacher_note"].startswith("The user expects this in all their work:")
    # The judge sees the user's request (the scene opens with it) and both commits.
    judged = [p for p, n in zip(writer.prompts, writer.asked, strict=True) if n == "changes"]
    assert len(judged) == 2 and all(STEPPED["request"] in p for p in judged)
    assert "fix: round VAT" in judged[0] and "Fixed VAT" in judged[0]
    assert "teacher_answer" not in row


def test_the_teachers_look_around_becomes_a_step_of_the_path(store, monkeypatch):
    """The teacher runs git status first: that call is done in the scene with
    a simulated output, and its commit is the next step. Both are samples."""
    monkeypatch.setattr(pool, "held_out", lambda signal_id, extra=(): False)
    writer = Acting(teacher=("git status --short", "git commit -am 'fix: round VAT'"))
    (row,) = acted(store, writer)
    assert row["walk"]["end"] == "reached" and len(row["walk"]["actions"]) == 2
    (later,) = row["path"]
    assert later[: len(row["messages"])] == row["messages"]
    call, result = later[len(row["messages"]):]
    assert json.loads(call["tool_calls"][0]["function"]["arguments"]) == {"command": "git status --short"}
    assert result["content"] == " M tax.py" and "simulate" in writer.asked
    assert row["drop"] is None
    first, second = pool.practice_steps(store)
    assert first["context"] == row["messages"] and second["context"] == later
    assert second["id"] == f"practice:{row['id']}#s2" and second["hindsight"] == first["hindsight"]


def test_a_teacher_that_never_reaches_the_step_drops_the_scene(store):
    looking = Acting(teacher=("git log --oneline",))
    (row,) = acted(store, looking)
    assert row["drop"] == "reached" and row["walk"]["end"].startswith("not reached in")


def test_a_teacher_that_replies_before_the_step_drops_the_scene(store):
    (row,) = acted(store, Acting(teacher=(None,)))
    assert row["drop"] == "reached" and row["walk"]["end"] == "replied before the step"


def test_a_student_still_looking_around_says_nothing_about_the_step(store):
    # The student runs git diff at the commit step: no commit of its own to compare.
    writer = Acting(plain="git diff --cached")
    (row,) = acted(store, writer)
    assert row["drop"] == "changes" and row["why"]["changes"] == "not both at the step"
    assert "changes" not in writer.asked


def test_the_same_commit_twice_is_dropped(store):
    writer = Acting(teacher=("git commit -am 'Fixed VAT'",))
    (row,) = acted(store, writer)
    assert row["drop"] == "changes" and row["why"]["changes"] == "same"


def test_the_step_the_lesson_is_about_is_recognised():
    from songe.practice.walk import is_the_step, lead

    commit = {"tool": "shell", "args": args(command="git commit -m 'fix: x'")}
    call = lambda command: {"content": "", "tool_calls": [{"name": "bash", "arguments": json.dumps({"command": command})}]}  # noqa: E731
    assert lead("cd /w && git add -A && git commit -m x") == "git add" and lead("pytest -q") == "pytest"
    assert is_the_step(call("git add -A && git commit -m 'fix: x'"), commit)
    assert not is_the_step(call("git status"), commit)
    readme = {"tool": "write", "args": args(path="README.md")}
    wrote = {"content": "", "tool_calls": [{"name": "write", "arguments": json.dumps({"filePath": "/w/README.md"})}]}
    assert is_the_step(wrote, readme) and not is_the_step(call("cat README.md"), readme)
    reply = {"tool": "", "args": args()}
    assert is_the_step({"content": "Fixed the rounding.", "tool_calls": []}, reply)
    assert not is_the_step(call("git status"), reply)
