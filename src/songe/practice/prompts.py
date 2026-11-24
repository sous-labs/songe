"""What the model is asked to write: one practice scene per call, worked
backwards from the moment a lesson matters (ported unchanged from
scripts/dream/generate.py, v4: lesson plus a short summary of the original,
never its step-by-step source, which the model copied).
"""


from ..reading.timeline import clip

TOPICS = [
    "a todo-list command-line tool", "an invoice calculator", "a weather data importer",
    "a recipe website backend", "a bank account library", "a log file analyzer",
    "a music playlist manager", "a URL shortener service", "a fitness tracker app",
    "a markdown-to-HTML converter", "an inventory spreadsheet exporter", "a chat bot",
]

DISTANCE = {
    "task": "of the same kind as the original, but in a different project with different file names, function names and values. It must not be the same task.",
    "kind": "of the same kind of task as the original (the same kind of game, the same kind of code change), in a different project with different names and values.",
    "always": "of any kind of coding work where the lesson still applies, in a different project. If it fits, set it in {topic}.",
}

# The arguments the dreamed steps may use, as plain named fields; empty ones are dropped.
ARG_FIELDS = ["path", "pattern", "command", "code", "content"]
DREAM_TOOLS = ["read", "grep", "glob", "write", "shell", "execute"]

STEP = {"type": "object", "properties": {
    "say": {"type": "string"},
    "tool": {"type": "string"},
    "args": {"type": "object", "properties": {k: {"type": "string"} for k in ARG_FIELDS},
             "required": list(ARG_FIELDS)},
    "output": {"type": "string"},
}, "required": ["say", "tool", "args", "output"]}

# The model commits to the decision point first (the mistake and the moment
# just before it), then writes the lead-up. Properties are generated in this
# order. Code ends every dream on that moment.
DREAM_SCHEMA = {
    "type": "object",
    "properties": {
        "mistake": {"type": "string"},
        "right_action": {"type": "string"},
        "moment": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": ["tool_result", "user_message"]},
            "step": STEP,
            "agent_said": {"type": "string"},
            "user_message": {"type": "string"},
        }, "required": ["kind", "step", "agent_said", "user_message"]},
        "project": {"type": "string"},
        "files": {"type": "array", "items": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
        "request": {"type": "string"},
        "steps": {"type": "array", "items": STEP},
    },
    "required": ["mistake", "right_action", "moment", "project", "files", "request", "steps"],
}


def _args(**given):
    return {k: given.get(k, "") for k in ARG_FIELDS}


EMPTY_STEP = {"say": "", "tool": "", "args": _args(), "output": ""}

# Format only. Concrete worked examples were copied: with a "port 8000 in
# use" example, 25 of 38 dreams were that example whatever the lesson.
TEMPLATE = {
    "mistake": "<the next action an agent without the lesson would likely take>",
    "right_action": "<the next action that follows the lesson>",
    "moment": {"kind": "tool_result",
               "step": {"say": "<short sentence or empty>", "tool": "<tool name>",
                        "args": _args(path="<...>", command="<...>"), "output": "<what was printed, for shell or execute>"},
               "agent_said": "", "user_message": ""},
    "project": "<folder-name>",
    "files": [{"path": "<file>", "content": "<full content>"}],
    "request": "<the user's first message>",
    "steps": [{"say": "<...>", "tool": "<tool name>", "args": _args(path="<...>"), "output": ""}],
}

DREAM = """You write practice situations for a coding agent.

The agent learned this lesson:
{lesson}

What happened originally, in short (do not copy it; invent a new situation):
{instance}

Write one NEW situation {distance} This is variant {variant}; make it different from other variants.

Work backwards from the moment that matters:
- mistake: the next action an agent that does NOT know the lesson would likely take. It must be possible as the very next action.
- right_action: the next action if it follows the lesson.
- moment: what the agent sees right before choosing. Either
  kind "tool_result": step is the tool call the agent just made, and its output is what makes the choice come up (a test result, an error, a game's answer, a file it just wrote), or
  kind "user_message": agent_said is what the agent just told the user and user_message is the user's reply that makes the choice come up.
  Leave the fields of the other kind empty.
Then the lead-up:
- project: a short folder name (lowercase-with-dashes)
- files: the files that exist at the start, path relative to the project and full content, each under 40 lines
- request: the user's first message, in their own words
- steps: 0 to 3 steps the agent took before the moment.
A step has say (one short sentence, or empty), tool (read, grep, glob, write, shell or execute; the agent's own name for one of these, such as bash for shell, works too), args (path for read, write, grep and glob; pattern for grep and glob; command for shell; code for execute; content for write, the whole new file; leave unused fields empty), and output (only for shell and execute: exactly what was printed; empty for the other tools, whose results are computed from the files). To change a file, use write.
The files and steps must lead to the moment. Never mention the lesson in the files, request or steps. Do not reuse the original's file names, function names, values or wording.

{tools}

The shape of the answer (placeholders only; fill in your own situation):
{examples}

Answer with JSON only."""


def teacher_note(signal):
    """What the teacher sees: framed so it is at worst irrelevant on a dream."""
    return ("Note from an earlier, different task (it may not apply here).\n"
            f"Feedback the agent got then: {clip(signal.get('text') or '', 400)}\n"
            f"Lesson: {signal['lesson']}")


# A client that sends no tools (a plain chat): the scene is a conversation.
NO_TOOLS = """The agent has no tools: it can only answer in text. Leave files and steps empty, and use a moment of kind "user_message": agent_said is the agent's previous answer and user_message the user's next question, where the lesson matters. mistake and right_action are answers, not tool calls."""


def tools_section(request_tools, tools):
    """What the scene writer is told about the agent's tools: the ones a
    scene can use, or, for a client that sends none, that there are none."""
    if not request_tools:
        return NO_TOOLS
    return "The agent's tools (it also has others):\n" + tool_lines(tools)


def tool_lines(tools):
    out = []
    for t in tools:
        f = t.get("function", {})
        args = ", ".join((f.get("parameters") or {}).get("properties", {}))
        out.append(f"- {f.get('name')}({args}): {' '.join((f.get('description') or '').split())[:120]}")
    return "\n".join(out)
