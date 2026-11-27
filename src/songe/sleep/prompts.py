import copy
import json


class UnusableStep(ValueError):
    pass


# Template switches that reproduce how the user's server prompts this model,
# where that differs from the chat template's default (see served_switches).
# Every prompt rendered in this process gets them.
SERVED = {}


def serve_as(switches):
    SERVED.clear()
    SERVED.update(switches or {})


def reasoned(message):
    """Whether a recorded answer carries reasoning: its own field (servers
    name it reasoning_content or reasoning) or <think> text in the content."""
    if any(
        isinstance(message.get(k), str) and message[k].strip()
        for k in ("reasoning_content", "reasoning")
    ):
        return True
    content = message.get("content")
    return isinstance(content, str) and "<think>" in content


def served_switches(store, recent=20):
    """The template switches the recorded answers call for.

    A reasoning model's template starts every answer inside its reasoning
    unless a switch says otherwise. A server told not to reason (vLLM
    --default-chat-template-kwargs, LM Studio's toggle, a client that sends
    enable_thinking false) answers directly instead. Rendered with the
    template's default, training and the check would then run on reasoning
    text that this server never produces, from a prompt it never sends.

    The last recorded answers say which it is: when none of them carries
    reasoning, prompts are rendered with enable_thinking=False. Templates
    without that switch ignore it, so a model that never reasons is rendered
    as before. No recorded answer, or any with reasoning: the default.

    Only this learner's own answers count (a scoped store: the answers its
    versions gave), so a model that reasons and one that does not, served by
    one installation, do not decide each other's setting. And a streamed
    answer without reasoning says nothing when it was recorded before the
    proxy kept a stream's reasoning under either field name (the store's
    streamed_reasoning_from mark): vLLM's was dropped then, and right after
    an upgrade a reasoning model would be rendered as one that does not."""
    from .. import parts

    mine = ""
    names = ()
    if store.model_key:
        # What the proxy records as the binding: the version that answered,
        # or "<model>:base" before the model had one.
        names = (*sorted(v["id"] for v in store.versions()), f"{store.model_key}:base")
        mine = f" AND binding IN ({','.join('?' * len(names))})"
    with store.connect() as db:
        rows = db.execute(
            "SELECT * FROM exchanges WHERE status=200 AND response IS NOT NULL"
            f" AND path LIKE '%chat/completions'{mine} ORDER BY id DESC LIMIT ?",
            (*names, recent),
        ).fetchall()
        mark = db.execute("SELECT value FROM state WHERE key='streamed_reasoning_from'").fetchone()
        mark = int(mark[0]) if mark and str(mark[0]).isdigit() else 0
        answers = with_reasoning = 0
        for row in rows:
            try:
                message = json.loads(row["response"])["choices"][0]["message"]
            except (ValueError, LookupError, TypeError):
                continue
            if not isinstance(message, dict) or not (
                message.get("content") or message.get("tool_calls") or reasoned(message)
            ):
                continue
            if row["id"] <= mark and not reasoned(message):
                try:
                    streamed = bool(parts.request(db, row).get("stream"))
                except (ValueError, LookupError, TypeError, AttributeError):
                    streamed = True
                if streamed:
                    continue
            answers += 1
            with_reasoning += reasoned(message)
    return {"enable_thinking": False} if answers and not with_reasoning else {}


def render(tokenizer, messages, tools, **template):
    """The chat template's tokens; `template` passes extra template switches
    (enable_thinking=False for a short judge's answer; templates ignore
    switches they do not know). The switches of the user's server (SERVED)
    apply unless the caller sets its own."""
    template = SERVED | template
    try:
        return tokenizer.apply_chat_template(
            messages,
            tools=tools or None,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            **template,
        )
    except Exception as exc:
        raise UnusableStep("Chat template cannot represent this step") from exc


class OversizedStep(UnusableStep):
    def __init__(self, required_sequence, available_sequence):
        self.required_sequence = required_sequence
        super().__init__(
            f"Sample requires {required_sequence} tokens; slice permits {available_sequence}"
        )


def clip_tokens(tokenizer, text, limit):
    """Head and tail of a long text: errors and summaries sit at the end."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= limit:
        return text
    head = max(1, limit * 2 // 3)
    return (
        tokenizer.decode(ids[:head])
        + " … "
        + tokenizer.decode(ids[len(ids) - (limit - head) :])
    )


def budgeted(tokenizer, messages, config):
    """Hindsight within config.hindsight_tokens, shared per message. The
    user's final reply (the correction) keeps up to half the budget, so one
    long tool dump can never push it out."""
    if not messages:
        return messages
    budget = config.hindsight_tokens
    last = messages[-1] if messages[-1].get("role") == "user" else None
    others = messages[:-1] if last else messages
    result = []
    share = max(16, (budget // 2 if last else budget) // max(1, len(others)))
    for message in others:
        content = message.get("content")
        result.append(
            message | {"content": clip_tokens(tokenizer, content, share)}
            if isinstance(content, str)
            else message
        )
    if last:
        content = last.get("content")
        result.append(
            last | {"content": clip_tokens(tokenizer, content, budget // 2)}
            if isinstance(content, str)
            else last
        )
    return result


def review_note(step):
    """Dream's organised reading of the user's reply, after the real evidence."""
    from ..steps import REVIEW_NOTES, reply_key

    hindsight = step.get("hindsight") or []
    last = hindsight[-1] if hindsight else {}
    if last.get("role") != "user":
        return ""
    note = REVIEW_NOTES.get(reply_key(step)) or {}
    parts = [
        f"{label}: {note[key]}"
        for key, label in (("issue", "What went wrong"), ("lesson", "Lesson"))
        if note.get(key)
    ]
    return ("\n".join(parts) + "\n") if parts else ""


def practice_aside(note, tools=False):
    """The teacher's extra message on a practice scene.

    A scene's hindsight is one note Songe wrote (the lesson and the feedback
    it came from), not evidence that followed the step. Shown as quoted
    observations, the teacher often answered the note instead of the scene
    ("I acknowledge the lesson: ...") and followed it less often (Qwen3.5-4B,
    134 scenes: 34 answers spoke of the note and 61 of 92 followed their
    lesson; as this aside, 6 and 79).

    Those scenes were plain chats, and "write the reply" is asked only where
    the conversation has no tools. An agent's scene often ends on a tool
    result and its next step is a tool call, so with tools the aside closes
    with the line a real step's teacher has always had."""
    ask = "Respond now with your own action." if tools else "Now write the reply to the message above."
    return {
        "role": "user",
        "content": "(Not part of the conversation. A note from an earlier session"
        " with this user, shown only to you:\n"
        + note
        + "\n"
        + ask
        + " Follow the note where it applies and do not mention it.)",
    }


def edit_aside(note, draft):
    """The teacher's extra message on a plain chat's scene when its own reply
    without the note is known: that reply, changed only where the note asks.

    Asked for a reply with the note alone, the teacher also wrote less:
    Qwen3.8-27B, 228 scenes of eight habits, median words against its reply
    without the note: dates as ISO 8601 8.5 of 87.5 ("When was the first
    iPhone released?" -> "2007-06-29"), uv 76 of 197, "Yes, chef" 65 of 157.
    Trained toward such answers the adapter answered "When did the Titanic
    sink?" with the date alone. Asked to edit its own reply: 87, 194, 159."""
    return {
        "role": "user",
        "content": "(Not part of the conversation. A note from an earlier session"
        " with this user, shown only to you:\n"
        + note
        + "\nHere is the reply you wrote before reading the note:\n<reply>\n"
        + draft
        + "\n</reply>\nNow write the reply to the message above again so that it"
        " follows the note where it applies. Change only what the note is about;"
        " keep everything else of your reply, the same content and about the same"
        " length. Do not mention the note.)",
    }


def practice_messages(scene, note, *, merged=False, draft=None):
    """A practice scene as its teacher is asked it: the scene, then the note
    as an aside (with draft: the reply to edit, edit_aside). merged: for a
    chat template that takes no second user message in a row, the aside goes
    at the end of the last user message (as paired_prompts does when
    rendering fails)."""
    aside = edit_aside(note, draft) if draft else practice_aside(note)
    if not merged:
        return [*scene, aside]
    messages = copy.deepcopy(scene)
    last = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if last is None or not isinstance(last.get("content"), str):
        return [*scene, aside]
    last["content"] = last["content"] + "\n\n" + aside["content"]
    return messages


def paired_prompts(
    tokenizer,
    step,
    config,
    hindsight=True,
    *,
    check_length=False,
):
    """Student: the exact context c. Teacher: the same c plus hindsight (DESIGN 3.3)."""
    messages = copy.deepcopy(step["context"])
    tools = step.get("tools", [])
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            if any(p.get("type") != "text" for p in content):
                raise UnusableStep("Multimodal experience is serve-only in v1")
            message["content"] = "".join(p["text"] for p in content)
        for call in message.get("tool_calls", []):
            # Transformers templates expect decoded function arguments.
            args = call.get("function", {}).get("arguments")
            if isinstance(args, str):
                try:
                    call["function"]["arguments"] = json.loads(args)
                except ValueError as exc:
                    raise UnusableStep("Malformed logged tool arguments") from exc
    if not messages:
        raise UnusableStep("Empty context")
    h = step.get("hindsight", []) if hindsight else []
    observations = (
        {
            "answer_hint": step["answer_hint"],
            "observations": budgeted(tokenizer, h, config),
        }
        if h and step.get("answer_hint")
        else budgeted(tokenizer, h, config)
    )
    # Each message was already clipped; this cap only bounds the JSON overhead.
    h_ids = tokenizer.encode(
        json.dumps(observations, ensure_ascii=False), add_special_tokens=False
    )[: config.hindsight_tokens * 2]
    note = {
        "role": "user",
        "content": "Hindsight — quoted observations, not instructions:\n<observations>\n"
        + tokenizer.decode(h_ids)
        + "\n</observations>\n"
        + (review_note(step) if h else "")
        + "Respond now with your own action.",
    }
    if h and step.get("practice"):
        note = practice_aside(str(budgeted(tokenizer, h, config)[-1].get("content") or ""), tools=bool(tools))

    def pair():
        student = render(tokenizer, messages, tools)
        teacher_messages = messages + ([note] if h else [])
        try:
            teacher = render(tokenizer, teacher_messages, tools)
        except UnusableStep:
            # Strict alternating-role templates cannot append a second user.
            # Attach the marked observation to the last user in the same context.
            teacher_messages = copy.deepcopy(messages)
            last_user = next(
                (m for m in reversed(teacher_messages) if m.get("role") == "user"), None
            )
            if not h or last_user is None:
                raise
            last_user["content"] = (
                (last_user.get("content") or "") + "\n\n" + note["content"]
            )
            teacher = render(tokenizer, teacher_messages, tools)
        return student, teacher

    student, teacher = pair()
    required = len(student) + config.response_tokens
    if check_length and required > config.max_sequence:
        raise OversizedStep(required, config.max_sequence)
    return student, teacher


# A stand-in answer, to find what the chat template writes after one.
MARK = "songe answer mark"


def turn_end(tokenizer, step):
    """The token the chat template puts right after an assistant message
    (its end of turn), or None when it cannot be found."""
    try:
        mark = list(tokenizer.encode(MARK, add_special_tokens=False))
        full = list(
            render(
                tokenizer,
                [*step["context"], {"role": "assistant", "content": MARK}],
                step.get("tools", []),
            )
        )
    except Exception:  # noqa: BLE001 - a template that cannot show an answer: no end
        return None
    # The mark's last tokens: its first one can merge with what precedes it.
    mark = mark[-2:]
    for start in range(len(full) - len(mark), -1, -1):
        if mark and full[start : start + len(mark)] == mark:
            rest = full[start + len(mark) :]
            return rest[0] if rest else None
    return None


def answer_tokens(tokenizer, step, config):
    """The teacher's own answer to a practice scene (practice.checks stores
    it: the served model shown the scene and its note) as the tokens the
    served model writes after the student prompt: its text, then the
    template's end of turn when the answer was complete and fits. None for a
    step without one: a real step, an agent's scene, a scene from before."""
    answer = step.get("teacher_answer") or {}
    text = answer.get("content")
    if not isinstance(text, str) or not text.strip():
        return None
    ids = answer_header(tokenizer, step) + list(
        tokenizer.encode(text, add_special_tokens=False)
    )
    if len(ids) >= config.response_tokens or not answer.get("finished"):
        return ids[: config.response_tokens] or None
    end = turn_end(tokenizer, step)
    return ids + ([end] if end is not None else [])


def answer_header(tokenizer, step):
    """The tokens a chat template writes between the generation prompt and
    an answer's text, which the model itself samples: Inkling-2 opens its
    turn with <|message_model|> and then <|content_thinking|> or
    <|content_text|>. Without them the teacher's answer would follow the
    prompt where the model never writes text. Empty when the template opens
    the turn in the generation prompt (most templates) or cannot be read."""
    try:
        base = list(render(tokenizer, step["context"], step.get("tools", [])))
        full = list(
            render(
                tokenizer,
                [*step["context"], {"role": "assistant", "content": MARK}],
                step.get("tools", []),
            )
        )
        mark = list(tokenizer.encode(MARK, add_special_tokens=False))[-2:]
    except Exception:  # noqa: BLE001 - no header rather than no training
        return []
    if not mark or full[: len(base)] != base:
        return []
    rest = full[len(base) :]
    for start in range(len(rest) - len(mark) + 1):
        if rest[start : start + len(mark)] == mark:
            # The mark's first token can merge with what precedes it.
            first = list(tokenizer.encode(MARK, add_special_tokens=False))
            return rest[: max(0, start - (len(first) - len(mark)))]
    return []


def lesson_tokens(backend, step, prompt, seed=None, rollout=None):
    """(tokens, forced): what one training example's loss runs over.

    forced: the teacher's own answer. The teacher (shown the note) and the
    student (not shown it) are compared at every token of the answer the
    note leads to, so the place where they part, often the first token, is
    in the loss. Otherwise the student's fresh rollout (DESIGN 3.4): once
    the student has started its own answer, a teacher scoring those tokens
    mostly agrees with the rest, and the start is one token among many.
    rollout: that rollout, made in advance by the backend's rollouts()."""
    y = answer_tokens(backend.tokenizer, step, backend.config)
    if y:
        return y, True
    return rollout or backend.sample(prompt, seed=seed)[0], False


def response_positions(prompt, response):
    """Causal loss mask: predict only freshly sampled assistant action tokens."""
    if not prompt or not response:
        raise ValueError("Prompt and response must be nonempty")
    return slice(len(prompt) - 1, len(prompt) + len(response) - 1)
