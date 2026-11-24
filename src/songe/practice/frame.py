"""Turn a practice scene (the model's JSON) into a request the agent could have
sent (ported from scripts/dream/frame.py).

The frame is real: the system prompt and tool definitions come from the
logged request at the signal's responsible step, with the working directory
swapped for the dream's project folder. The conversation is built by code:

- read, grep, glob, edit and write are computed from the dream's files, in
  opencode's own result formats, so these results are always consistent;
- a step whose tool the agent lacks is translated to the agent's own tool
  (pi has bash, not grep, glob or shell: see translate), its output computed
  from the files as that command would print it;
- shell and execute results are the model's text (nothing is ever run),
  except `ls` and `cat`, which are computed from the files too.

A dream ends where the agent must act next; the last message is the user's
request or a tool result.
"""

import fnmatch
import json
import re
import shlex
import uuid
from pathlib import PurePosixPath

ROOT = "/tmp/work"  # used only when the logged prompt names no working directory


def frame_from_request(request, project):
    """System prompt and tools of a logged request, moved to a new folder.
    A request without a system message (a plain chat client) gives none."""
    first = (request.get("messages") or [{}])[0]
    system = (first.get("content") or "") if first.get("role") == "system" else ""
    if not isinstance(system, str):
        system = "\n".join(p.get("text", "") for p in system if isinstance(p, dict))
    # A sibling of the real project folder, so paths look like the user's own.
    found = re.search(r"(?:Working directory|Workspace root folder): (\S+)", system)
    parent = str(PurePosixPath(found.group(1)).parent) if found else ROOT
    workdir = f"{parent.rstrip('/')}/{project}"
    system = re.sub(r"(Working directory: )\S+", lambda m: m.group(1) + workdir, system)
    system = re.sub(r"(Workspace root folder: )\S+", lambda m: m.group(1) + workdir, system)
    return {"system": system, "tools": request.get("tools") or [], "workdir": workdir}


def tool_names(tools):
    return {t["function"]["name"] for t in tools if "function" in t}


def tool_schema(tools, name):
    for t in tools:
        if t.get("function", {}).get("name") == name:
            return t["function"].get("parameters") or {}
    return None


def error(message):
    return json.dumps({"error": {"type": "tool.execution", "message": message}, "content": []})


class Workspace:
    """The dream's files, changed by its edit and write steps."""

    def __init__(self, workdir, files):
        self.workdir = workdir
        self.files = {self.rel(f["path"]): f["content"] for f in files if f.get("path")}

    def rel(self, path):
        path = str(path or "").strip()
        if path.startswith(self.workdir):
            path = path[len(self.workdir):]
        return str(PurePosixPath("/" + path.lstrip("./"))).lstrip("/")

    def abs(self, rel):
        return f"{self.workdir}/{rel}"

    def read(self, args):
        rel = self.rel(args.get("path") or args.get("filePath"))
        inside = sorted({r[len(rel) + 1:].split("/")[0] if rel else r.split("/")[0]
                         for r in self.files if not rel or r.startswith(rel + "/")})
        if rel not in self.files and inside:  # a folder: list it
            return f"Directory {self.abs(rel) if rel else self.workdir}\n" + "\n".join(inside), True
        if rel not in self.files:
            return error(f"File not found: {self.abs(rel)}"), False
        lines = self.files[rel].splitlines() or [""]
        body = "\n".join(f"{i}: {line}" for i, line in enumerate(lines, 1))
        return f"Read file {self.abs(rel)}, lines 1-{len(lines)}\n{body}", True

    def write(self, args):
        rel = self.rel(args.get("path") or args.get("filePath"))
        self.files[rel] = args.get("content") or ""
        return f"Wrote file successfully: {PurePosixPath(rel).name}", True

    def edit(self, args):
        rel = self.rel(args.get("path") or args.get("filePath"))
        old, new = args.get("oldString") or "", args.get("newString") or ""
        if rel not in self.files:
            return error(f"File not found: {self.abs(rel)}"), False
        if old == new:
            return error("No changes to apply: oldString and newString are identical."), False
        n = self.files[rel].count(old) if old else 0
        if n == 0:
            return error(f"Could not find oldString in {self.abs(rel)}. It must match exactly, including whitespace and indentation."), False
        self.files[rel] = self.files[rel].replace(old, new)
        return f"Edited {PurePosixPath(rel).name} ({n} replacement{'s' if n > 1 else ''})", True

    def grep(self, args):
        try:
            pattern = re.compile(args.get("pattern") or "")
        except re.error:
            return error("Invalid regular expression"), False
        out, n = [], 0
        for rel, text in sorted(self.files.items()):
            hits = [(i, line) for i, line in enumerate(text.splitlines(), 1) if pattern.search(line)]
            if hits:
                n += len(hits)
                out.append(f"{self.abs(rel)}:\n" + "\n".join(f"  Line {i}: {line}" for i, line in hits) + "\n")
        return (f"Found {n} matches\n" + "\n".join(out)) if n else "No files found", True

    def glob(self, args):
        pattern = (args.get("pattern") or "*").lstrip("./")
        hits = [self.abs(r) for r in sorted(self.files)
                if fnmatch.fnmatch(r, pattern) or fnmatch.fnmatch(r, pattern.replace("**/", ""))]
        return "\n".join(hits) if hits else "No files found", True

    def grep_lines(self, pattern, path=""):
        """`grep -rn` output (path:line:text), from the files."""
        try:
            rx = re.compile(pattern or "")
        except re.error:
            return f"grep: Invalid regular expression: {pattern}"
        base = self.rel(path) if path else ""
        out = []
        for rel, text in sorted(self.files.items()):
            if base and rel != base and not rel.startswith(base + "/"):
                continue
            out += [f"{rel}:{i}:{line}" for i, line in enumerate(text.splitlines(), 1) if rx.search(line)]
        return "\n".join(out)

    def find_lines(self, pattern):
        """`find` output (./path per line) for a glob pattern, from the files."""
        pattern = (pattern or "*").lstrip("./")
        return "\n".join(f"./{r}" for r in sorted(self.files)
                         if fnmatch.fnmatch(r, pattern) or fnmatch.fnmatch(r, pattern.replace("**/", ""))
                         or ("/" not in pattern and fnmatch.fnmatch(PurePosixPath(r).name, pattern)))

    def shell(self, command, output):
        """ls and cat from the files; anything else is the dream's own text."""
        cmd = re.sub(r"^\s*cd \S+\s*&&\s*", "", command or "").strip()
        if re.fullmatch(r"ls( -[a-zA-Z]+)?( \.)?", cmd):
            return "\n".join(sorted({r.split("/")[0] for r in self.files})), True
        m = re.fullmatch(r"cat (\S+)", cmd)
        if m and self.rel(m.group(1)) in self.files:
            return self.files[self.rel(m.group(1))], True
        return output or "", True


# Generic argument names in a scene, and what a real tool may call them.
ALIASES = {
    "path": ("filePath", "file_path", "path", "filename", "file"),
    "content": ("content", "text", "contents"),
    "command": ("command", "cmd", "script"),
}

COMPUTED = {"read": "read", "write": "write", "edit": "edit", "grep": "grep", "glob": "glob"}

# The agent's own names for the generic steps a scene may use.
SHELLS = ("bash", "shell", "run_shell_command", "run_command", "execute_command", "terminal")
READS = ("read", "read_file", "view")
WRITES = ("write", "write_file", "create_file")


def generic_args(step):
    given = step.get("args") or {}
    if isinstance(given, list):  # name/value pairs
        given = {a.get("name"): a.get("value") for a in given}
    return {k: v for k, v in given.items() if v not in (None, "")}


def translate(step, names, ws):
    """A scene step whose tool this agent lacks, as the agent's own call:
    pi has bash but no grep, glob or shell, so `grep def main src/` becomes
    bash `grep -rn 'def main' src`, its output computed from the files as
    grep would print it. None when the agent has nothing equivalent."""
    name, args = step.get("tool"), generic_args(step)
    shell = next((n for n in SHELLS if n in names), None)
    path = args.get("path") or ""

    def via_shell(command, output):
        return {**step, "tool": shell, "args": {"command": command}, "output": output}

    if name in ("shell", "execute") and shell:
        if name == "execute":
            return via_shell("python3 - <<'EOF'\n" + (args.get("code") or "") + "\nEOF", step.get("output") or "")
        command = args.get("command") or ""
        return via_shell(command, ws.shell(command, step.get("output"))[0])
    if name == "grep" and shell:
        pattern = args.get("pattern") or ""
        command = f"grep -rn {shlex.quote(pattern)} {shlex.quote(path) if path else '.'}"
        return via_shell(command, ws.grep_lines(pattern, path))
    if name == "glob" and shell:
        pattern = args.get("pattern") or "*"
        flag = "-path" if "/" in pattern.strip("./") else "-name"
        target = f"./{pattern.lstrip('./')}" if flag == "-path" else pattern
        return via_shell(f"find . {flag} {shlex.quote(target)}", ws.find_lines(pattern))
    if name == "read":
        real = next((n for n in READS if n in names), None)
        if real:
            return {**step, "tool": real}
        if shell:
            rel = ws.rel(path)
            return via_shell(f"cat {shlex.quote(path)}", ws.files.get(rel, f"cat: {path}: No such file or directory"))
    if name == "write":
        real = next((n for n in WRITES if n in names), None)
        if real:
            return {**step, "tool": real}
        if shell:
            content = args.get("content") or args.get("code") or ""
            ws.write({"path": path, "content": content})
            return via_shell(f"cat > {shlex.quote(path)} <<'EOF'\n{content}\nEOF", "")
    return None


def coerce(value, spec):
    """A value written as text, typed as the tool schema says."""
    kind = (spec or {}).get("type")
    if kind == "integer":
        return int(value)
    if kind == "number":
        return float(value)
    if kind == "boolean":
        return str(value).lower() in ("true", "1", "yes")
    if kind in ("array", "object"):
        return json.loads(value)
    return value


def build(dream, frame):
    """(messages, problems, files): the dream as chat messages, what is wrong
    with it, and the files as they end up. The dream ends on its moment: the
    tool result or user message right before the choice the lesson is about."""
    tools, problems = frame["tools"], []
    names = tool_names(tools)
    ws = Workspace(frame["workdir"], dream.get("files") or [])
    messages = [{"role": "system", "content": frame["system"]}] if frame["system"] else []
    messages.append({"role": "user", "content": dream.get("request") or ""})
    if not (dream.get("request") or "").strip():
        problems.append("no user request")
    steps = list(dream.get("steps") or [])
    moment = dream.get("moment")
    for n, step in enumerate(steps, 1):
        add_step(messages, problems, ws, tools, names, step, f"step {n}")
    if moment:
        if moment.get("kind") == "user_message":
            if not (moment.get("user_message") or "").strip():
                problems.append("moment: no user message")
            messages.append({"role": "assistant", "content": moment.get("agent_said") or ""})
            messages.append({"role": "user", "content": moment.get("user_message") or ""})
        else:
            # A failing call is allowed here: an error can be the moment.
            add_step(messages, problems, ws, tools, names, moment.get("step") or {}, "moment", may_fail=True)
    return messages, problems, ws.files


def add_step(messages, problems, ws, tools, names, step, label, may_fail=False):
    name = step.get("tool")
    if str(name or "").strip().lower() in ("", "none", "null"):
        return  # no call at this step (live: tool "none"): skipped, not a failure
    translated = False
    if name not in names:
        # The scene's generic step in this agent's own tools (pi: bash).
        step = translate(step, names, ws)
        if step is None:
            problems.append(f"{label}: no tool named {name!r}")
            return
        name, translated = step["tool"], True
    schema = tool_schema(tools, name)
    props = schema.get("properties") or {}
    args = {}
    given = step.get("args") or {}
    if isinstance(given, list):  # name/value pairs
        given = {a.get("name"): a.get("value") for a in given}
    for key, value in given.items():
        if value in (None, ""):
            continue
        if key not in props and key in ALIASES:
            # The scene names arguments generically; agents differ (opencode
            # has used both path and filePath). Use the tool's own name.
            key = next((a for a in ALIASES[key] if a in props), key)
        if key not in props:
            # The dream format has one set of fields for every tool; a field
            # this tool does not take is dropped (write's text may arrive as
            # code instead of content).
            # Every field is always present (empty ones dropped), so an empty
            # content must not block it (live: 47 writes "missing content").
            if name in WRITES and key == "code" and not given.get("content"):
                args[next((a for a in ALIASES["content"] if a in props), "content")] = value
            continue
        try:
            args[key] = coerce(value, props[key])
        except (ValueError, TypeError):
            problems.append(f"{label}: {name}.{key} is not a valid {props[key].get('type')}")
    # A code tool given a command, or a shell given code, gets it in the
    # field it takes (live: 134 execute steps "missing code", the code
    # written as command).
    for want, have in (("code", "command"), ("command", "code")):
        if want in props and want not in args and given.get(have):
            args[want] = given[have]
    for key in schema.get("required") or []:
        if key not in args:
            problems.append(f"{label}: {name} is missing required argument {key!r}")
    if translated and name not in COMPUTED:
        result, ok = step.get("output") or "", True  # computed in translate
    elif name in COMPUTED:
        result, ok = getattr(ws, COMPUTED[name])(args)
        if not ok and not may_fail:
            problems.append(f"{label}: {name} failed: {json.loads(result)['error']['message'][:120]}")
    elif name == "shell" or name in SHELLS:
        # The agent's own shell (pi: bash) like the scene's: ls and cat from the files.
        result, ok = ws.shell(args.get("command"), step.get("output"))
    else:
        result, ok = step.get("output") or "", True
    call_id = uuid.uuid4().hex[:24]
    messages.append({"role": "assistant", "content": step.get("say") or "",
                     "tool_calls": [{"id": call_id, "type": "function",
                                     "function": {"name": name, "arguments": json.dumps(args)}}]})
    messages.append({"role": "tool", "tool_call_id": call_id, "content": result})
