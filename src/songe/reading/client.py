"""Ask the model already loaded on the model server one question at a time.

Answers are JSON with a fixed schema. Servers that support structured output
(response_format json_schema) are held to it; a server that refuses the field
is asked again without it, and the JSON is read from the text. Answers are
cached in the store by (model, question), so a reading stopped because the
user came back resumes without asking again.
"""

import asyncio
import hashlib
import json
import logging
import re
import time

import httpx

logger = logging.getLogger(__name__)

# Servers whose chat template takes enable_thinking per request.
TEMPLATE_SWITCH = {"vllm", "mlx", "llamacpp"}
# Reasoning tokens allowed on vLLM: more for questions meant to think
# (classify), little for the others.
THINKING_BUDGET = {True: 2048, False: 1024}
# A forced answer: the reasoning so far it is shown (its tail, where the
# conclusions are) and the room it gets for the JSON alone.
FORCE_CONTEXT = 6000
FORCE_TOKENS = 4096
# Asked of every reader question.
BRIEF = "\nKeep your reasoning short. Answer with the JSON only."


class Stopped(Exception):
    """The user came back (or reading was otherwise told to stop)."""


def parse(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        value = json.loads(match[0])
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


class Reader:
    def __init__(
        self,
        store,
        url,
        model,
        *,
        still_idle=None,
        think=("classify", "turn"),
        max_tokens=6000,
        timeout=600,
        client=None,
        server="",
    ):
        self.store, self.url, self.model = store, url.rstrip("/"), model
        # Per-request reasoning controls, probed per server with reasoning
        # models (research/scripts/memory/probe_reasoning.py):
        # - vLLM, mlx-lm and llama.cpp pass chat_template_kwargs.enable_thinking
        #   to the chat template (mlx-lm, Granite 4.2 3B: 274 -> 44 tokens;
        #   llama.cpp, Qwen3 0.6B: 332 -> 43), so questions not meant to
        #   think get none;
        # - only vLLM caps reasoning with thinking_token_budget;
        # - reasoning_effort "none" (sent to every server) also stops
        #   llama.cpp's; on Ollama it empties the reasoning field, but Qwen3
        #   4B then rambled in the answer until the limit;
        # - LM Studio ignored every field tried (identical output).
        # Where nothing bounds it, a cut-off answer is forced (ask).
        self.server = server
        self.template_switch = server in TEMPLATE_SWITCH
        self.still_idle = still_idle
        # Which questions the model thinks about first: slower, steadier.
        # Measured on the answer key: thinking helps classify only. The turn
        # question (prompts.TURN) needs it too: see there.
        self.think = set(think)
        self.max_tokens = max_tokens
        self.client = client or httpx.AsyncClient(timeout=timeout, trust_env=False)
        self.calls = self.cached = 0
        self.forced = self.cut_off = self.tokens_out = 0
        self.seconds = 0.0
        self.structured = True

    def log_stats(self, stage):
        """What a stage cost so far (cumulative), for comparing models and
        settings. Logged after reading and after practice: one reader serves
        both, and a Dream's practice can take hours (live: no stats while it
        ran)."""
        if self.calls == getattr(self, "_logged_calls", 0):
            return
        self._logged_calls = self.calls
        logger.info(
            "reader_stats stage=%s calls=%s cached=%s seconds=%.0f tokens_out=%s per_call=%s cut_off=%s forced=%s",
            stage, self.calls, self.cached, self.seconds, self.tokens_out,
            self.tokens_out // max(1, self.calls), self.cut_off, self.forced,
        )

    async def close(self):
        self.log_stats("close")  # only what came after the last stage line
        await self.client.aclose()

    def key(self, prompt, schema, name, temperature=0):
        parts = [self.model, name in self.think, prompt, schema]
        # A sampled answer (practice scenes) is cached by its temperature too.
        parts += [temperature] if temperature else []
        return hashlib.sha256(json.dumps(parts).encode()).hexdigest()

    async def ask(self, prompt, schema, name, *, temperature=0, max_tokens=None):
        key = self.key(prompt, schema, name, temperature)
        cached = await asyncio.to_thread(cache_get, self.store, key)
        if cached is not None:
            self.cached += 1
            return cached
        if self.still_idle is not None and not self.still_idle():
            raise Stopped()
        think = name in self.think
        body = {
            "model": self.model,
            # Reasoning models think even when asked not to (Inkling); a
            # short reasoning is asked for in words too.
            "messages": [{"role": "user", "content": prompt + BRIEF}],
            "temperature": temperature,
            "max_tokens": (max_tokens or self.max_tokens) * (3 if think else 1),
            "stream": False,
        }
        if not think:
            body["reasoning_effort"] = "none"
        if self.template_switch:
            body["chat_template_kwargs"] = {"enable_thinking": think}
            # vLLM's hard cap on reasoning tokens, after which the model still
            # writes its answer (probed live on Inkling-2-Mini: 1,024 halved
            # the generated tokens and the time, JSON complete). Effort "none"
            # alone did not bound reasoning on long prompts (~8,100 tokens).
            if self.server == "vllm":
                body["thinking_token_budget"] = THINKING_BUDGET[think]
        started = time.monotonic()
        # max_tokens caps the reasoning and the answer together: a model that
        # reasons too long (Inkling reasons even when asked not to) stops
        # before or inside its JSON. Then the answer is forced: one short
        # follow-up with the reasoning so far, reasoning off, small room, so
        # only the reasoning was cut and nothing is thought through twice.
        # Forced and still unreadable, the moment is given up (remembered, so
        # the next read does not pay for it again).
        answer, finish, used, thought = await self.post(body, prompt + BRIEF, schema, name)
        self.tokens_out += used
        if "error" in answer and finish == "length":
            self.cut_off += 1
            self.forced += 1
            forced_prompt = (
                prompt
                + "\n\nYour reasoning so far (it was cut off):\n"
                + (thought or "(none)")[-FORCE_CONTEXT:]
                + "\n\nStop reasoning now. Answer with the JSON only."
            )
            forced = body | {
                "messages": [{"role": "user", "content": forced_prompt}],
                "max_tokens": min(body["max_tokens"], FORCE_TOKENS),
                "reasoning_effort": "none",
            }
            if self.template_switch:
                forced["chat_template_kwargs"] = {"enable_thinking": False}
            answer, finish, used, _ = await self.post(forced, forced_prompt, schema, name)
            self.tokens_out += used
            if "error" in answer:
                answer = {"error": f"cut off, then no answer when forced ({finish})", "gave_up": True}
                logger.warning("reader_gave_up %s: cut off, forced answer unreadable", name)
        self.calls += 1
        self.seconds += time.monotonic() - started
        if "error" in answer and not answer.get("gave_up"):
            # Not cached: a later read asks again (a cached failure would be
            # reused for good). A moment given up on is cached: asking the
            # same question again runs out of room again.
            self.errors = getattr(self, "errors", 0) + 1
        else:
            await asyncio.to_thread(cache_put, self.store, key, name, answer)
        return answer


    async def post(self, body, prompt, schema, name):
        """One question: (answer, finish reason, generated tokens, reasoning
        text). Structured output when the
        server takes it, else asked plainly with the schema in the prompt."""
        if self.structured:
            response = await self.client.post(
                self.url + "/chat/completions",
                json=body
                | {
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": name, "strict": True, "schema": schema},
                    }
                },
            )
            if 400 <= response.status_code < 500:
                # This server does not take structured output; ask plainly.
                self.structured = False
            else:
                response.raise_for_status()
                return read(response), finish_reason(response), usage(response), reasoning_of(response)
        response = await self.client.post(
            self.url + "/chat/completions",
            json=body
            | {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt + "\nThe JSON must match this schema: " + json.dumps(schema),
                    }
                ]
            },
        )
        response.raise_for_status()
        return read(response), finish_reason(response), usage(response), reasoning_of(response)


def reasoning_of(response):
    """The reasoning an answer carried: the reasoning field, or <think> text
    in the content (servers differ)."""
    try:
        message = ((response.json().get("choices") or [{}])[0]).get("message") or {}
    except ValueError:
        return ""
    for field in ("reasoning_content", "reasoning"):
        if isinstance(message.get(field), str) and message[field].strip():
            return message[field]
    content = message.get("content") or ""
    match = re.search(r"<think>(.*?)(</think>|$)", content, re.S)
    return match[1] if match else content


def usage(response):
    """Generated tokens of an answer (0 when the server does not say)."""
    try:
        return int(((response.json().get("usage") or {}).get("completion_tokens")) or 0)
    except (ValueError, TypeError):
        return 0


def finish_reason(response):
    try:
        return ((response.json().get("choices") or [{}])[0]).get("finish_reason")
    except ValueError:
        return None


def read(response):
    choice = (response.json().get("choices") or [{}])[0]
    message = choice.get("message") or {}
    # The answer, else the reasoning channel (live: LM Studio with Granite
    # MLX ignored "reasoning off" and put the structured JSON there, content
    # empty; every reading answer failed).
    for text in (message.get("content"), message.get("reasoning_content"), message.get("reasoning")):
        if not isinstance(text, str) or not text.strip():
            continue
        try:
            value = json.loads(text)
        except ValueError:
            value = parse(text)
        if isinstance(value, dict):
            return value
    return {"error": f"no JSON (finish: {choice.get('finish_reason')})"}


def cache_get(store, key):
    with store.connect() as db:
        row = db.execute("SELECT answer FROM reading_cache WHERE key=?", (key,)).fetchone()
    answer = json.loads(row[0]) if row else None
    # Failures cached by older versions are asked again; a moment given up
    # on (cut off twice) is not.
    if isinstance(answer, dict) and "error" in answer and not answer.get("gave_up"):
        return None
    return answer


def cache_put(store, key, name, answer):
    with store.connect() as db:
        db.execute(
            "INSERT OR REPLACE INTO reading_cache VALUES (?,?,?,?)",
            (key, name, json.dumps(answer, ensure_ascii=False), time.time()),
        )
