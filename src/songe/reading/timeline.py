"""One recorded session as a numbered timeline.

Every agent request carries the whole history and its response is the next
action, so the timeline is: for each request, the messages that are new since
the last action (user messages, tool results), then the action. Requests that
are not agent steps are left out: session titles and the compaction requests
that ask for a summary. After a compaction the history restarts from a
summary; that is a checkpoint item.

A session is the set of exchanges Songe's prefix linking put under one root
(steps.extract), not the per-instruction part used for held-out splits: a
correction arrives later in the same session as the work it corrects.
"""

import json

from ..steps import CONTINUED

# opencode before 1.19, and 1.19.
TITLE = ("You name conversations", "You are a title generator")
# An agent compacting its own context sends one of these in the user's place
# (opencode; pi). Recognised anywhere in the last user message: pi puts the
# transcript first and the instruction after it.
COMPACT = (
    "You MUST summarize the conversation",
    "The previous response did not fill in the required summary template",
    "Update the existing checkpoint in the conversation above",
    "The messages above are a conversation to summarize",
    "<previous-summary>",
    "This is the PREFIX of a turn that was too large",
    # opencode 1.19: a first summary, and one that folds in the previous one.
    "Create a new anchored summary from the conversation history",
    "The <prior-summary> summarizes everything that happened before",
)
# Messages the agent harness adds in the user's place: feedback from the
# harness, not from the user.
HARNESS = (
    "Your last response hit the output token limit",
    # opencode 1.19 sends this after a compaction.
    "Continue if you have next steps, or stop and ask for clarification",
)
# After compaction: the summary, and sometimes a transcript of earlier turns.
CHECKPOINT = ("<conversation-checkpoint>", "[User]: ")


def text(content):
    if isinstance(content, list):
        return "\n".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return content or ""


def kind_of(request):
    messages = request.get("messages") or []
    if messages and text(messages[0].get("content")).startswith(TITLE):
        return "title"
    last = messages[-1] if messages else {}
    if last.get("role") == "user" and any(c in text(last.get("content")) for c in COMPACT):
        return "compaction"
    return "agent"


def response_message(response):
    try:
        return response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None


def build(rows):
    """rows: (exchange id, request, response, status) of one session, in order."""
    items, calls, emitted = [], {}, set()
    for ex, request, response, status in rows:
        if not isinstance(request, dict):
            continue
        kind = kind_of(request)
        if kind == "title":
            continue
        messages = [m for m in request.get("messages") or [] if m.get("role") != "system"]
        last_assistant = max(
            (i for i, m in enumerate(messages) if m.get("role") == "assistant"),
            default=-1,
        )
        for m in messages[last_assistant + 1 :]:
            body = text(m.get("content"))
            # A message can arrive in a compaction request and then again
            # later; keep its first appearance only.
            key = (m.get("role"), m.get("tool_call_id"), body)
            if key in emitted or body.startswith(COMPACT) or (
                # The agent's own summarise prompt (pi puts it after a
                # transcript), or the summary it inserts after compaction.
                m.get("role") == "user"
                and ((kind == "compaction" and any(c in body for c in COMPACT))
                     or body.lstrip().startswith(CONTINUED))
            ):
                continue
            emitted.add(key)
            if m.get("role") == "tool":
                call = calls.get(m.get("tool_call_id"), {})
                items.append(
                    {
                        "kind": "tool_result",
                        "call": m.get("tool_call_id"),
                        "step": call.get("step"),
                        "tool": call.get("name"),
                        "content": body,
                    }
                )
            elif body.startswith(CHECKPOINT):
                items.append({"kind": "checkpoint", "content": body})
            elif body.startswith(HARNESS):
                items.append({"kind": "harness", "content": body})
            elif m.get("role") == "user":
                items.append({"kind": "user", "content": body})
        if kind == "compaction":
            continue  # its response is a summary, not an agent action
        message = response_message(response)
        step = sum(1 for i in items if i["kind"] == "action") + 1
        if message is None:
            items.append(
                {"kind": "action", "step": step, "exchange": ex, "failed": status}
            )
            continue
        tool_calls = []
        for c in message.get("tool_calls") or []:
            fn = c.get("function", {})
            calls[c.get("id")] = {"step": step, "name": fn.get("name")}
            tool_calls.append(
                {
                    "id": c.get("id"),
                    "name": fn.get("name"),
                    "arguments": fn.get("arguments"),
                }
            )
        items.append(
            {
                "kind": "action",
                "step": step,
                "exchange": ex,
                "text": text(message.get("content")),
                "reasoning": message.get("reasoning_content") or "",
                "tool_calls": tool_calls,
            }
        )
    return items


def clip(s, n):
    """Head and tail of a long text, with what was cut in between."""
    s = (s or "").strip()
    return s if len(s) <= n else s[: n // 2] + f"\n…[{len(s) - n} chars cut]…\n" + s[-n // 2 :]


def _flat(text_):
    return " ".join((text_ or "").split())


def sessions(db, cutoff=None):
    """{session: [exchange ids]} for every linked session, oldest first.

    A session is the exchanges under one prefix-linking root, with two fixes:
    a session continued after a compaction (its history restarts from a
    summary, so it gets a new root) is joined to the session it continues,
    together with the compaction request itself; and roots that are only
    title requests are left out. The key is the first root.

    `cutoff`: only exchanges recorded after it (a learner set to `new`).

    Cached per database until an exchange, step or prefix is added or
    removed: the dashboard asks several times a second."""
    key = (
        db.execute("PRAGMA database_list").fetchone()[2],
        *db.execute("SELECT COUNT(*), MAX(id) FROM exchanges").fetchone(),
        db.execute("SELECT COUNT(*) FROM steps").fetchone()[0],
        db.execute("SELECT COUNT(*) FROM prefixes").fetchone()[0],
    )
    cached = _SESSIONS.get(key[0])
    if cached and cached[0] == key:
        merged = {k: set(v) for k, v in cached[1].items()}
    else:
        merged = _linked(db)
        _SESSIONS[key[0]] = (key, {k: frozenset(v) for k, v in merged.items()})
    if cutoff is not None:
        recent = {
            r[0]
            for r in db.execute(
                "SELECT id FROM exchanges WHERE COALESCE(recorded_at,created)>?",
                (cutoff,),
            )
        }
        merged = {k: v & recent for k, v in merged.items()}
    return {
        root: sorted(ids)
        for root, ids in sorted(merged.items(), key=lambda kv: min(kv[1] or {0}))
        if ids
    }


_SESSIONS = {}


def _linked(db):
    """{root: exchange ids} before the cutoff filter (see sessions)."""
    from ..steps import prefix_keys

    rows = db.execute(
        "SELECT DISTINCT p.root, json_extract(s.payload,'$.source_id') "
        "FROM prefixes p JOIN steps s ON s.id=p.step_id"
    ).fetchall()
    found = {}
    for root, source in rows:
        if source is not None:
            found.setdefault(root, set()).add(int(source))
    first = {root: min(ids) for root, ids in found.items()}
    requests = {}

    def request(i):
        if i not in requests:
            row = db.execute(
                "SELECT request, response FROM exchanges WHERE id=?", (i,)
            ).fetchone()
            requests[i] = (
                (json.loads(row[0]), json.loads(row[1]) if row[1] else None)
                if row
                else ({}, None)
            )
        return requests[i]

    # Title-only roots carry no agent work.
    for root in [r for r, ids in found.items() if all(kind_of(request(i)[0]) == "title" for i in ids)]:
        del found[root]
    # Compactions: the request still carries the old history, so its longest
    # known prefix names the session it compacts; its response is the summary.
    compactions = []
    like = " OR ".join("request LIKE ?" for _ in COMPACT)
    for (i,) in db.execute(
        f"SELECT id FROM exchanges WHERE {like} ORDER BY id", [f"%{c}%" for c in COMPACT]
    ):
        req, resp = request(i)
        if kind_of(req) != "compaction":
            continue
        messages = req.get("messages") or []
        parent = None
        for length in range(len(messages) - 1, 0, -1):
            hit = next(filter(None, (db.execute("SELECT root FROM prefixes WHERE hash=?", (k,)).fetchone()
                                     for k in prefix_keys(messages[:length]))), None)
            if hit:
                parent = hit[0]
                break
        if parent is None:
            # Some compactions send a written-out transcript ("[User]: ...")
            # instead of the history: the latest earlier session whose
            # opening request appears in it.
            transcript = _flat(" ".join(text(m.get("content")) for m in messages))
            for root in sorted(found, key=lambda r: -first[r]):
                if first[root] >= i:
                    continue
                opening = [m for m in request(first[root])[0].get("messages") or [] if m.get("role") == "user"]
                said = _flat(text(opening[0].get("content")))[:100] if opening else ""
                if said and said in transcript:
                    parent = root
                    break
        summary = _flat(text((response_message(resp) or {}).get("content")))[:200]
        # Without a parent (e.g. an empty conversation summarised) it is
        # still agent-internal: never a conversation of its own.
        compactions.append((i, parent, summary))
    joined = {}

    def top(root):
        while root in joined:
            root = joined[root]
        return root

    for root in sorted(found, key=lambda r: first[r]):
        messages = [m for m in request(first[root])[0].get("messages") or [] if m.get("role") != "system"]
        # A continuation opens with the summary its compaction wrote (opencode
        # marks it; other agents do not): the summary is the evidence.
        opening = " ".join(text(m.get("content")) for m in messages[:2])
        if not opening:
            continue
        flat = _flat(opening)
        for i, parent, summary in reversed(compactions):
            if parent and i < first[root] and summary and summary in flat and top(parent) != root:
                joined[root] = top(parent)
                break
    merged = {}
    for root, ids in found.items():
        merged.setdefault(top(root), set()).update(ids)
    for i, parent, _ in compactions:
        # A compaction belongs to the session it compacts, never its own.
        for ids in merged.values():
            ids.discard(i)
        if parent and top(parent) in merged:
            merged[top(parent)].add(i)
    # A session made only of compactions (an agent summarising nothing, or
    # attached to another such call through their shared system prompt) holds
    # no conversation.
    internal = {i for i, _, _ in compactions}
    return {root: ids for root, ids in merged.items() if ids - internal}


def load(db, exchange_ids):
    """The rows build() takes, for these exchanges."""
    marks = ",".join("?" * len(exchange_ids))
    return [
        (
            i,
            json.loads(request),
            json.loads(response) if response else None,
            status,
        )
        for i, request, response, status in db.execute(
            f"SELECT id,request,response,status FROM exchanges WHERE id IN ({marks}) ORDER BY id",
            list(exchange_ids),
        )
    ]
