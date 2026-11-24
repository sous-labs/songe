"""Signals from one session: rules plus the model (ported from
scripts/signals/extract.py; the logic and prompts are unchanged).

For each feedback event the model says which steps were responsible (credit
assignment) and what the lesson is. User messages are first split into
feedback, new task and question parts. Successes come from the rules, checked
against the model's reading of the user's next reply: a confirmed success is
drift-set material, a contradicted one is not.
"""

import json
import re

from . import prompts as P
from .rules import extract as rule_signals
from .views import (
    call_line,
    events,
    first_line,
    last_user_item,
    result_view,
    shorten_paths,
    step_lines,
    user_messages,
)

SIGNAL_TYPES = ("correction", "retry", "approval", "tool_error", "failed_result", "outcome")


def tools_of(rows):
    """The agent's tool descriptions, as sent with its requests."""
    for _, request, _, _ in rows:
        tools = (request or {}).get("tools") or []
        if tools:
            return {
                t["function"]["name"]: t["function"].get("description", "")
                for t in tools
                if "function" in t
            }
    return {}


def workdir_of(rows):
    for _, request, _, _ in rows:
        messages = (request or {}).get("messages") or []
        system = messages[0].get("content") if messages else ""
        m = re.search(
            r"(?:Working directory|Workspace root folder): (\S+)",
            system if isinstance(system, str) else json.dumps(system),
        )
        if m:
            return m.group(1)
    return None


# Command results the review shows in full: the turn's last ones.
REVIEW_FULL = 8
# A reply that only says to go on carries no feedback ("ok" or "yes" can be
# approvals, so they are still classified).
KEEP_GOING = re.compile(
    r"(please\s+)?(continue|go on|keep going|carry on|proceed|resume|next)(\s+please)?[.!]*", re.I
)


def call_text(items, result_item):
    """The command behind a tool result."""
    call_id = items[result_item].get("call")
    for it in items[:result_item]:
        if it["kind"] == "action":
            for c in it.get("tool_calls", []):
                if c["id"] == call_id:
                    return call_line(c)
    return items[result_item].get("tool") or "?"


def first_step_after(items, item):
    for it in items[item:]:
        if it["kind"] == "action":
            return it["step"]
    return None


def steps_after(items, a, b):
    return sorted({it["step"] for it in items[a:b] if it["kind"] == "action"})


class Session:
    def __init__(self, session, items, model, workdir, tools=None):
        self.session, self.items, self.model, self.workdir = session, items, model, workdir
        self.tools = tools or {}
        self.task_starts = [
            next((i for i, it in enumerate(items) if it["kind"] == "user"), 0)
        ]
        self.signals = []
        self.user_events = []

    def short(self, text):
        return shorten_paths(text, self.workdir)

    def task_start(self, item):
        # A session can start mid-way (after a compaction): events before the
        # first user message belong to the first task.
        return max((s for s in self.task_starts if s <= item), default=0)

    def task_text(self, item):
        start = self.task_start(item)
        return self.short(
            "\n".join(user_messages(self.items, item)[: self.user_index(start) + 1][-3:])
        )

    def user_index(self, item):
        return sum(1 for it in self.items[:item] if it["kind"] == "user")

    def steps_view(self, upto):
        """Task steps up to an item: earlier turns one line each, this turn in full."""
        start, turn = self.task_start(upto), last_user_item(self.items, upto)
        earlier = step_lines(self.items, start + 1, turn) if turn > start else ""
        current = step_lines(self.items, turn + 1, upto + 1, full=True)
        return self.short((earlier + "\n" if earlier else "") + current)

    def after_view(self, item, limit=30):
        """What the agent did after the feedback, up to the next user message.
        In hindsight this often shows the real cause (what it had to fix)."""
        end = next(
            (i for i in range(item + 1, len(self.items)) if self.items[i]["kind"] == "user"),
            len(self.items),
        )
        lines = step_lines(self.items, item + 1, end).splitlines()
        # No step numbers: these are context, not candidates.
        lines = [re.sub(r"^STEP \d+:", "- later:", line) for line in lines]
        return self.short("\n".join(lines[:limit])) or "(nothing; the conversation ended)"

    def fallback(self, type_, item, steps, allowed):
        """When the model names no valid step: the last edit before a failed
        check; for a user correction, the turn's edits and its last step."""
        acts = [
            it
            for it in self.items[: item + 1]
            if it["kind"] == "action" and it["step"] in allowed
        ]
        edits = [
            it["step"]
            for it in acts
            if any(c.get("name") in ("edit", "write") for c in it.get("tool_calls", []))
        ]
        if type_ == "outcome":
            return sorted(allowed - set(steps))[:3] or list(steps)
        if self.items[item]["kind"] == "user":
            turn = last_user_item(self.items, item)
            in_turn = set(steps_after(self.items, turn, item))
            return sorted(
                {s for s in edits if s in in_turn} | ({max(in_turn)} if in_turn else set())
            )
        return [edits[-1]] if edits else list(steps)

    async def add(self, source, type_, item, steps, text, where):
        signal = {
            "session": self.session,
            "source": source,
            "type": type_,
            "item": item,
            "evidence_steps": steps,
            "text": text[:1500],
            "task_start": first_step_after(self.items, self.task_start(item) + 1),
        }
        before = set(steps_after(self.items, self.task_start(item), item + 1))
        if source == "user" and self.items[item]["kind"] == "user":
            before = set(steps_after(self.items, self.task_start(item), item))
        if type_ == "approval":
            # An approval confirms the turn it answers.
            turn = last_user_item(self.items, item)
            signal["responsible_steps"] = steps_after(self.items, turn, item)
            self.signals.append(signal)
            return
        if type_ == "tool_error" or source == "harness":
            # The call that was rejected, or the step that hit the limit.
            signal["responsible_steps"] = sorted(steps)
            signal["reason"] = (
                "the step that made the call"
                if type_ == "tool_error"
                else "the step the harness cut off"
            )
        else:
            if type_ == "outcome":
                # Information arrives at a step; the steps that should have
                # used it come after it, up to the next user message.
                end = next(
                    (
                        i
                        for i in range(item + 1, len(self.items))
                        if self.items[i]["kind"] == "user"
                    ),
                    len(self.items),
                )
                allowed = set(steps_after(self.items, item, end)) | set(steps)
                note = "Only steps that came after this result can have ignored it."
            else:
                allowed, note = before, "Only steps up to the feedback can be responsible."
            note += " Choose only from steps " + ", ".join(map(str, sorted(allowed))) + "."
            upto = item if type_ != "outcome" else min(end, len(self.items)) - 1
            r = await self.model.ask(
                P.RESPONSIBLE.format(
                    task=self.task_text(item),
                    steps=self.steps_view(upto),
                    where=where,
                    feedback=self.short(text[:1500]),
                    after=self.after_view(upto if type_ == "outcome" else item),
                    allowed=note,
                ),
                P.RESPONSIBLE_SCHEMA,
                "responsible",
            )
            chosen = r.get("steps") if isinstance(r.get("steps"), list) else []
            signal["responsible_steps"] = sorted(
                {s for s in chosen if isinstance(s, int)} & allowed
            )
            signal["reason"] = r.get("reason") or r.get("error")
            if not signal["responsible_steps"] and type_ == "outcome":
                return  # the agent used this information; nothing to learn
            if not signal["responsible_steps"]:
                signal["responsible_steps"] = self.fallback(type_, item, steps, allowed)
                signal["reason"] = f"(fallback; the model chose {r.get('steps')}) " + str(
                    signal["reason"]
                )
        chosen = [
            it
            for it in self.items
            if it["kind"] == "action" and it["step"] in signal["responsible_steps"]
        ]
        shown = (
            "\n".join(
                step_lines(
                    self.items, self.items.index(it), self.items.index(it) + 1, full=True
                )
                for it in chosen
            )
            or "(none)"
        )
        # For a tool error, what the tool is for matters more than the task.
        used = (
            {c["name"] for it in chosen for c in it.get("tool_calls", [])}
            if type_ == "tool_error"
            else set()
        )
        about = "".join(
            f"\nWhat the {n} tool does: {self.tools[n][:500]}"
            for n in sorted(used)
            if n in self.tools
        )
        lesson = await self.model.ask(
            P.LESSON.format(
                task=self.task_text(item),
                steps=self.short(shown),
                feedback=self.short(text[:1500]) + about,
                reason=signal["reason"],
                after=self.after_view(item),
                # The user's own words can hold a wish to keep as they said it.
                wish=P.USER_WISH if source == "user" else "",
            ),
            P.LESSON_SCHEMA,
            "lesson",
        )
        signal.update(
            {k: lesson.get(k) for k in ("lesson", "short", "instance", "generality", "verifiable")}
        )
        if P.copied_example(signal.get("lesson")) or P.copied_example(signal.get("short")):
            # The prompt's example pasted back, not a lesson about this
            # moment: no lesson (the signal and its instance stay).
            signal.update(lesson=None, short=None, copied_example=True)
        if "error" in lesson:
            signal["error"] = lesson["error"]
        self.signals.append(signal)

    async def run_all(self, rules):
        items = self.items
        for e in events(items, rules):
            if e["kind"] == "user":
                turn = last_user_item(items, e["item"])
                if KEEP_GOING.fullmatch(" ".join(e["text"].split())):
                    # "continue", "go on": no feedback to classify, and each
                    # model call is ~37 s on a reasoning model (Inkling).
                    e["classified"] = []
                    self.user_events.append(e)
                    continue
                answer = await self.model.ask(
                    P.CLASSIFY.format(
                        earlier=self.short("\n".join(user_messages(items, e["item"]))),
                        steps=self.short(step_lines(items, turn + 1, e["item"])),
                        message=e["text"],
                    ),
                    P.CLASSIFY_SCHEMA,
                    "classify",
                )
                parts = [p for p in answer.get("parts") or [] if isinstance(p, dict)]
                # What the turn is, judged on its own: a standing wish about
                # how this user wants things, a change wanted once, new work,
                # or a reaction. A standing wish the split above did not read
                # as feedback ("i want in russian", read as a new task) is a
                # correction all the same.
                verdict, wish = ("task", "") if len(e["text"]) > P.TURN_CHARS else P.turn_verdict(
                    await self.model.ask(
                        P.TURN.format(
                            earlier=self.short("\n".join(user_messages(items, e["item"]))),
                            steps=self.short(step_lines(items, turn + 1, e["item"])),
                            message=e["text"],
                        ),
                        P.TURN_SCHEMA,
                        "turn",
                    ),
                    e["text"],
                )
                if verdict == "lasting" and not any(
                    p.get("kind") == "feedback" and p.get("type") in P.TYPES for p in parts
                ):
                    parts = parts + [{"says_something_is_wrong": True, "kind": "feedback",
                                      "type": "correction", "text": e["text"]}]
                before = len(self.signals)
                for p in parts:
                    # The yes/no is asked first; trust it over the type.
                    if (
                        p.get("kind") == "feedback"
                        and p.get("says_something_is_wrong")
                        and p.get("type") == "approval"
                    ):
                        p["type"] = "correction"
                    if p.get("kind") == "feedback" and p.get("type") in P.TYPES:
                        await self.add(
                            "user",
                            p["type"],
                            e["item"],
                            [e["after_step"]],
                            p.get("text") or e["text"],
                            f"as the user's message after step {e['after_step']}",
                        )
                for signal in self.signals[before:]:
                    signal["turn_kind"] = verdict  # quick.suitable reads it
                    if wish:
                        # The standing wish, as the turn question worded it,
                        # is the lesson and the note.
                        signal.update(lesson=wish, short=wish, generality="always")
                if any(p.get("kind") == "new_task" for p in parts):
                    self.task_starts.append(e["item"])
                e["classified"] = parts
                self.user_events.append(e)
            elif e["kind"] == "harness":
                await self.add(
                    "harness",
                    "failed_result",
                    e["item"],
                    e["steps"],
                    e["text"],
                    f"from the agent harness (not the user) after step {e['after_step']}",
                )
            elif e["kind"] == "tool":
                where = f"in the tool result of step {e['steps'][0]}" + (
                    f" (the same result again at steps {', '.join(map(str, e['steps'][1:]))})"
                    if len(e["steps"]) > 1
                    else ""
                )
                await self.add(
                    "tool", e["type"], e["item"], e["steps"], result_view(e["text"]), where
                )
            elif e["kind"] == "review":
                # The turn's last results in full, earlier ones one line each
                # (command and the first line of its result): a turn of dozens
                # of build-fix steps made 20-50k-token prompts (live, Inkling).
                # A distant failure can still be flagged; the follow-up calls
                # (responsible, lesson) see that step in full.
                full_from = len(e["items"]) - REVIEW_FULL
                view = self.short(
                    "\n".join(
                        f"STEP {items[i]['step']}: {call_text(items, i)}\n  output: {result_view(items[i]['content'], 500)}"
                        if n >= full_from
                        else f"STEP {items[i]['step']}: {call_text(items, i)}  ->  {first_line(items[i]['content'])}"
                        for n, i in enumerate(e["items"])
                    )
                )
                answer = await self.model.ask(
                    P.REVIEW.format(
                        task=self.task_text(e["item"]), steps=view,
                        **(P.REVIEW_COMMANDS if e.get("commands", True) else P.REVIEW_TOOLS),
                    ),
                    P.REVIEW_SCHEMA,
                    "review",
                )
                for s in answer.get("signals") or []:
                    if not isinstance(s, dict):
                        continue
                    # Keep only signals whose quote really is in that step's output.
                    outputs = " ".join(
                        " ".join(items[i]["content"].split()).lower()
                        for i in e["items"]
                        if items[i].get("step") == s.get("step")
                    )
                    quote = " ".join((s.get("quote") or "").split()).lower()[:40]
                    if not quote or quote not in outputs:
                        continue
                    if s.get("type") == "success":
                        self.signals.append(
                            {
                                "session": self.session,
                                "source": "tool",
                                "type": "success",
                                "evidence_steps": [s.get("step")],
                                "text": s.get("quote"),
                                "why": s.get("why"),
                                "from": "review",
                            }
                        )
                        continue
                    at = next(
                        (
                            i
                            for i, it in enumerate(items)
                            if it["kind"] == "tool_result" and it.get("step") == s.get("step")
                        ),
                        e["item"],
                    )
                    await self.add(
                        "tool",
                        s.get("type"),
                        at,
                        [s.get("step")],
                        s.get("quote") or "",
                        f"in the tool result of step {s.get('step')}",
                    )
        return self.signals


async def stated_preferences(items, model):
    """(item index, sentence) for each sentence the rules flag as a possible
    preference of the user's own and the model judges to be one, in any user
    message, the opening one included."""
    from .rules import preference_sentences
    from .timeline import clip

    out, said = [], "(nothing: this is the first message)"
    for i, it in enumerate(items):
        if it.get("kind") == "action" and str(it.get("text") or "").strip():
            said = clip(it["text"], 600)
        if it.get("kind") != "user":
            continue
        for sentence in preference_sentences(it.get("content")):
            answer = await model.ask(
                P.PREFERENCE.format(before=said, message=clip(it["content"], 1200), sentence=sentence),
                P.PREFERENCE_SCHEMA,
                "preference",
            )
            if answer.get("kind") == "about_me":
                out.append((i, sentence))
    return out


async def read_session(session, items, rows, model):
    """All signals of one session: the model's, then the rules' successes,
    confirmed or contradicted by the model's reading of each user reply."""
    rules = rule_signals(items)
    reading = Session(session, items, model, workdir_of(rows), tools_of(rows))
    found = await reading.run_all(rules)
    kinds = {}
    for e in reading.user_events:
        fb = [p.get("type") for p in e.get("classified", []) if p.get("kind") == "feedback"]
        kinds[e["turn"] - 1] = next(
            (t for t in ("correction", "retry", "approval") if t in fb), None
        )
    confirmed = rule_signals(items, {k + 1: v for k, v in kinds.items()})
    rule_steps = set()
    for s in confirmed:
        if s["type"] == "success":
            rule_steps.add(s["step"])
            found.append(
                {
                    "session": session,
                    "source": "tool",
                    "type": "success",
                    "item": s["item"],
                    "evidence_steps": [s["step"]],
                    "text": s["evidence"],
                    "from": "rules",
                    "confirmed": s["confirmed"],
                    "contradicted_by": s["contradicted_by"],
                }
            )
    found = [
        s
        for s in found
        if not (s.get("from") == "review" and s["evidence_steps"][0] in rule_steps)
    ]
    # Harness messages are feedback from a tool, not the user (CONTRACTS).
    for s in found:
        if s["source"] == "harness":
            s["source"] = "tool"
    # A rule the user stated is a standing preference whatever the model made
    # of the turn before it: added unless the model's reading already holds it.
    from ..quick import reacted_action
    from .rules import rule_sentence, stated_rules

    # A sentence the model's reading holds with a real lesson is not added
    # twice; one whose lesson was a copied example (or none) is.
    have = {rule_sentence(s.get("text")) for s in found if s.get("source") == "user" and s.get("lesson")}
    for item, sentence in stated_rules(items):
        if sentence in have:
            continue
        have.add(sentence)
        # The step the rule is about: practice and before/after start there.
        about = reacted_action(items[:item], sentence)
        found.append(
            {
                "session": session,
                "source": "user",
                "type": "correction",
                "item": item,
                "text": sentence,
                "from": "rules",
                "lesson": sentence,
                "short": sentence,
                "generality": "always",
                "evidence_steps": [about["step"]] if about and about.get("step") else [],
                "placed": True,
            }
        )
    # A preference the user states about themself ("I avoid dairy.") is kept in
    # their words too, wherever it stands: the model's own lesson for such a
    # message, if any, turned it into a procedure and lost what they named
    # (live, Qwen3.5-4B: "Always check for user constraints before recommending
    # content."), and an opening message is read by nobody else.
    for item, sentence in await stated_preferences(items, model):
        if sentence in have:
            continue
        have.add(sentence)
        holder = next(
            (
                s
                for s in found
                if s.get("source") == "user" and s.get("item") == item and s.get("lesson")
                and sentence in " ".join(str(s.get("text") or "").split())
            ),
            None,
        )
        if holder:
            holder["preference"] = sentence  # its lesson stays for practice; the prompt gets the sentence
            continue
        # Practice and before/after start at the step the sentence is about;
        # with nothing before it, at the answer it got.
        about = reacted_action(items[:item], sentence) or next(
            (it for it in items[item:] if it.get("kind") == "action" and it.get("exchange")), None
        )
        found.append(
            {
                "session": session,
                "source": "user",
                "type": "correction",
                "item": item,
                "text": sentence,
                "from": "rules",
                "preference": sentence,
                "lesson": sentence,
                "short": sentence,
                "generality": "always",
                "evidence_steps": [about["step"]] if about and about.get("step") else [],
                "placed": True,
            }
        )
    return found
