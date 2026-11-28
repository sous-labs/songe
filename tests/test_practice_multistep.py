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
    """Writes STEPPED; acts with `plain` alone and `taught` with the note
    (tool calls); the judge picks the taught action."""

    def __init__(self, plain="git commit -am 'Fixed VAT'", taught="git commit -am 'fix: round VAT'", **kw):
        super().__init__(STEPPED, **kw)
        self.plain, self.taught, self.replied = plain, taught, []

    async def ask(self, prompt, schema, name, *, temperature=0, max_tokens=None):
        if name == "changes":
            self.asked.append(name)
            self.prompts.append(prompt)
            first = prompt.split("Answer A:\n", 1)[1].split("\n\nAnswer B:", 1)[0]
            return {"follows": "A" if self.taught in first else "B"}
        return await super().ask(prompt, schema, name, temperature=temperature, max_tokens=max_tokens)

    async def reply(self, messages, *, max_tokens, temperature=0, think=None, tools=None):
        self.replied.append((messages, tools))
        noted = "Not part of the conversation" in str(messages[-1].get("content"))
        command = self.taught if noted else self.plain
        return {"content": "", "tool_calls": [{"name": "shell", "arguments": json.dumps({"command": command})}],
                "finished": True, "reasoned": False}


def acted(store, writer):
    corrected_twice(store)
    asyncio.run(practice.write(store, writer, 1))
    return practice.practice(store)


def test_an_agent_scene_is_kept_when_the_note_changes_the_last_right_step(store):
    writer = Acting()
    (row,) = acted(store, writer)
    assert row["drop"] is None and row["checks"]["changes"] is True
    # The forced choice between lessons is not asked of a scene with right steps.
    assert "matters" not in writer.asked and row["checks"]["matters"] is None
    # Both answers are asked at the last right step, with the agent's tools.
    (alone, tools), (noted, _) = writer.replied[:2]
    assert alone == row["path"][-1] and noted[:-1] == row["path"][-1]
    assert {t["function"]["name"] for t in tools} >= {"shell"}
    # The judge sees the user's request (the scene opens with it) and the calls.
    judged = [p for p, n in zip(writer.prompts, writer.asked, strict=True) if n == "changes"]
    assert len(judged) == 2 and all(STEPPED["request"] in p for p in judged)
    assert "fix: round VAT" in judged[0] and "Fixed VAT" in judged[0]
    # No answer is stored to train along: the student acts on its own.
    assert "teacher_answer" not in row


def test_an_agent_scene_the_note_does_not_change_is_dropped(store):
    # The teacher also only looks at the diff at the last step: nothing to learn there.
    writer = Acting(taught="git commit -am 'Fixed VAT'")
    (row,) = acted(store, writer)
    assert row["drop"] == "changes" and row["why"]["changes"] == "same"
    assert "changes" not in writer.asked


def test_every_right_step_is_a_training_sample(store, monkeypatch):
    monkeypatch.setattr(pool, "held_out", lambda signal_id, extra=(): False)
    (row,) = acted(store, Acting())
    first, second = pool.practice_steps(store)
    assert first["id"] == "practice:" + row["id"] and first["context"] == row["messages"]
    assert second["id"] == f"practice:{row['id']}#s2" and second["context"] == row["path"][0]
    assert second["hindsight"] == first["hindsight"] and second["conversation"] == first["conversation"]
