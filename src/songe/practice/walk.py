"""Teacher-led paths: the teacher walks a practice scene to the step its
lesson is about, and every step it takes on the way is a training sample.

The scene writer's right steps are a guess written before the teacher is
asked anything. The teacher often takes other steps first (a git status, a
look at the log, reading a file), and at a scripted step it then pulls the
student towards that look around rather than towards the step the lesson is
about (frontier-arms, 2026-11-28: at the commit step the teacher committed
in 1 of 10 commit scenes; when it did reach the step, it followed the rule
21 times in 21). So the teacher is asked for its action at each point:

- the step the lesson is about (the scene's last right step: the same kind
  of call, e.g. a commit, a write of that file, or a reply when that step
  is a reply): the walk ends there;
- another call: it is done in the scene, a file tool's result computed from
  the files, a shell command's written by the model as that project would
  print it (nothing is run), and the walk goes on;
- a reply before the step, or WALK_STEPS calls without reaching it: the
  scene is dropped (check "reached").

The student is trained at every point of the walk with the teacher scoring
its own action, so it learns the teacher's routine up to and including the
step the lesson is about.
"""

import copy
import json
import re
import shlex

from .frame import COMPUTED, SHELLS, WRITES, add_step, tool_names

WALK_STEPS = 6  # calls before the step, at most
WALK_TOKENS = 2048  # one action, room for a whole file written

SIMULATE = """A coding agent is working in a project. Its session so far:

{session}

The files of the project now:
{files}

The agent now runs this shell command:
{command}

Write exactly what the command prints (stdout and stderr together), as this project would print it. Keep it consistent with the files and the session. If it prints nothing, return an empty output.
Reply as JSON: {{"output": "..."}}"""

SIMULATE_SCHEMA = {"type": "object", "properties": {"output": {"type": "string"}}, "required": ["output"]}

# A leading word that names a family of commands: compare its first two words.
FAMILIES = {"git", "npm", "pnpm", "yarn", "cargo", "uv", "pip", "python", "python3", "node", "go", "make", "docker"}


def arguments(call):
    raw = call.get("arguments")
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def first(args, *names):
    return next((str(args[n]) for n in names if args.get(n) not in (None, "")), "")


def lead(command):
    """The part of a command that says which step it is: "git commit" of
    "cd x && git add -A && git commit -m ...", "pytest" of "pytest -q"."""
    command = re.sub(r"^\s*cd \S+\s*&&\s*", "", command or "")
    try:
        words = shlex.split(command)
    except ValueError:
        words = command.split()
    if not words:
        return ""
    return " ".join(words[:2]) if words[0] in FAMILIES and len(words) > 1 else words[0]


def is_the_step(answer, intended):
    """Whether an answer is the scene's last right step: the same kind of
    call (a shell command of the same family, a write of the same file, the
    same tool), or a reply when that step is a reply."""
    tool = str(intended.get("tool") or "").strip()
    calls = answer.get("tool_calls") or []
    if tool.lower() in ("", "none", "null"):
        return not calls and bool((answer.get("content") or "").strip())
    if not calls:
        return False
    call, args = calls[0], arguments(calls[0])
    name = call.get("name") or ""
    given = intended.get("args") or {}
    if tool in SHELLS or tool in ("shell", "execute"):
        want = lead(first(given, "command", "code"))
        command = re.sub(r"\s+", " ", first(args, "command", "cmd", "script"))
        return (name in SHELLS or name == "shell") and bool(want) and want in command
    if tool in WRITES or tool in ("write", "edit"):
        want = first(given, "path").rsplit("/", 1)[-1]
        path = first(args, "filePath", "file_path", "path", "filename", "file").rsplit("/", 1)[-1]
        return (name in WRITES or name in ("write", "edit")) and bool(want) and path == want
    return name == tool


async def simulated(reader, messages, files, command):
    from .checks import render_text

    listing = "\n".join(f"--- {p}\n{c[:1500]}" for p, c in sorted(files.items())) or "(none)"
    answer = await reader.ask(
        SIMULATE.format(session=render_text(messages, limit=3000), files=listing[:6000], command=command),
        SIMULATE_SCHEMA,
        "simulate",
        max_tokens=800,
    )
    return str(answer.get("output") or "")


async def walk(reader, scene, note, frame, messages, ws):
    """(contexts, actions, end): the context before each step after the
    first, the teacher's action at each point, and why the walk ended:
    "reached" when the last action is the step the lesson is about."""
    from ..sleep.prompts import practice_messages

    right = [s for s in scene.get("right_steps") or [] if isinstance(s, dict)]
    if not right:
        return [], [], "no right steps"
    intended, tools = right[-1], frame["tools"]
    names, problems = tool_names(tools), []
    context, contexts, actions = copy.deepcopy(messages), [], []
    for n in range(WALK_STEPS + 1):
        answer = await reader.reply(
            practice_messages(context, note, tools=True), max_tokens=WALK_TOKENS, think=False, tools=tools
        )
        actions.append({"content": answer.get("content") or "", "tool_calls": answer.get("tool_calls") or []})
        if is_the_step(answer, intended):
            return contexts, actions, "reached"
        calls = answer.get("tool_calls") or []
        if not calls:
            return contexts, actions, "replied before the step"
        if n == WALK_STEPS:
            break
        call = calls[0]
        args, name = arguments(call), call.get("name") or ""
        output = ""
        if (name in SHELLS or name == "shell") and name not in COMPUTED:
            output = await simulated(reader, context, ws.files, first(args, "command", "cmd", "script"))
        step = {"say": answer.get("content") or "", "tool": name, "args": args, "output": output}
        before = len(problems)
        add_step(context, problems, ws, tools, names, step, f"walk step {n + 1}", may_fail=True)
        if len(problems) > before:
            return contexts, actions, problems[-1]
        contexts.append(copy.deepcopy(context))
    return contexts, actions, f"not reached in {WALK_STEPS} steps"
