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
# (server url, model) that refused thinking_token_budget: not sent it again.
NO_BUDGET = set()
# (server url, model) -> the model's whole window in tokens, once the server
# named it in a refusal (send): no request asks for more.
WINDOW = {}
# vLLM, asked for more tokens than the model's window: "max_tokens=6000 cannot
# be greater than max_model_len=4096".
OVER_WINDOW = re.compile(r"max_model_len=(\d+)")
# vLLM names the window and the prompt's tokens when the two do not fit:
# 0.34.0, a prompt longer than the window: "This model's maximum context
# length is 4096 tokens. However, your prompt contains at least 4097 input
# tokens"; older releases also when the prompt fits but max_tokens does not:
# "This model's maximum context length is 4096 tokens and your request has 100
# input tokens (6000 > 4096 - 100)", "However, you requested 6100 tokens (100
# in the messages, 6000 in the completion)", "However, you requested 5000
# tokens in the messages".
CONTEXT = re.compile(r"maximum context length is (\d+) tokens")
PROMPT_TOKENS = re.compile(r"(\d+) input tokens|\((\d+) in the messages|requested (\d+) tokens in the messages")


def room(refusal):
    """(window, prompt tokens) a refusal names, else None."""
    window, prompt = CONTEXT.search(refusal), PROMPT_TOKENS.search(refusal)
    if not window or not prompt:
        return None
    return int(window[1]), int(next(g for g in prompt.groups() if g))


def over_window(response):
    """The prompt alone is longer than the model's window. Only then: a
    refusal for max_tokens that the prompt leaves no room for (older vLLM
    words it the same way) is not a reason to give the moment up for good."""
    if response.status_code != 400:
        return False
    named = room(response.text)
    return bool(named) and named[1] >= named[0]


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
        # vLLM takes thinking_token_budget only when it was started with a
        # reasoning parser (see send); a server that refused it is remembered.
        self.budget = (self.url, self.model) not in NO_BUDGET

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
            if self.server == "vllm" and self.budget:
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
        if finish == "window":
            # The question alone is longer than the model's window: no answer
            # can come, now or on a later read. Live (Phi-3-mini-4k, window
            # 4,096): one such session stopped all reading for 3 users of 20.
            answer = {"error": "the question is longer than the model's window", "gave_up": True}
            logger.warning("reader_gave_up %s: the question is longer than the model's window", name)
        elif "error" in answer and finish == "length":
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


    async def reply(self, messages, *, max_tokens, temperature=0, think=None):
        """The served model's own next message in a conversation, asked the
        way its users ask: no schema and none of the reasoning switches the
        reader's questions set. think=False sets those switches, for a server
        whose reasoning left no room for the reply. Not cached.
        {"content", "finished", "reasoned"}."""
        if self.still_idle is not None and not self.still_idle():
            raise Stopped()
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if think is False:
            body["reasoning_effort"] = "none"
            if self.template_switch:
                body["chat_template_kwargs"] = {"enable_thinking": False}
        started = time.monotonic()
        response = await self.client.post(self.url + "/chat/completions", json=body)
        response.raise_for_status()
        choice = (response.json().get("choices") or [{}])[0]
        message = choice.get("message") or {}
        self.calls += 1
        self.seconds += time.monotonic() - started
        self.tokens_out += usage(response)
        content = message.get("content")
        content = content if isinstance(content, str) else ""
        return {
            "content": content,
            "finished": choice.get("finish_reason") == "stop",
            # Reasoning in its own field (servers name it either way) or
            # left in the text.
            "reasoned": "<think>" in content
            or any(
                isinstance(message.get(k), str) and message[k].strip()
                for k in ("reasoning_content", "reasoning")
            ),
        }

    async def send(self, payload):
        """One request to the model server. A request vLLM refuses for
        something this server or model does not take is asked again in a form
        it takes, and that is remembered for this server and model:
        - thinking_token_budget, on a server started without a reasoning
          parser (400, "thinking_token_budget is set but reasoning_config is
          not configured"). Live (vLLM 0.34.0 with its defaults,
          Mistral-7B-Instruct-v0.2): every reader question was refused, the
          first refusal was taken for "no structured output", the plain
          question was refused as well, and no session was ever read.
        - more tokens than the model's whole window (400, "max_tokens=6000
          cannot be greater than max_model_len=4096"). Live (same server,
          Phi-3-mini-4k, window 4,096): the same end. Up to the window the
          server takes any number and gives what the prompt leaves.
        A model with a small window and no reasoning parser is refused for
        both, one after the other: three requests at most.
        Older vLLM refuses max_tokens beyond what the prompt leaves ("your
        request has 100 input tokens (6000 > 4096 - 100)"): that request is
        asked again for what is left (for this prompt only)."""
        for _ in range(3):
            payload = self.fitted(payload)
            response = await self.client.post(self.url + "/chat/completions", json=payload)
            if response.status_code != 400:
                break
            payload = self.learn(payload, response.text)
            if payload is None:
                break
        return response

    def fitted(self, payload):
        """The request without what this server and model refused before."""
        if not self.budget and "thinking_token_budget" in payload:
            payload = {k: v for k, v in payload.items() if k != "thinking_token_budget"}
        window = WINDOW.get((self.url, self.model))
        if window and payload.get("max_tokens", 0) > window:
            payload = payload | {"max_tokens": window}
        return payload

    def learn(self, payload, refusal):
        """The request to ask again when a refusal named something to leave
        out or to lower, else None."""
        if "thinking_token_budget" in payload and "thinking_token_budget" in refusal:
            self.budget = False
            NO_BUDGET.add((self.url, self.model))
            logger.info("reader_no_thinking_budget the model server refused thinking_token_budget; asking without it")
            return payload
        window = OVER_WINDOW.search(refusal)
        if window and 0 < int(window.group(1)) < payload.get("max_tokens", 0):
            WINDOW[(self.url, self.model)] = int(window.group(1))
            logger.info("reader_window the model's window is %s tokens; asking for no more", window.group(1))
            return payload
        named = room(refusal)
        if named and 0 < named[0] - named[1] < payload.get("max_tokens", 0):
            return payload | {"max_tokens": named[0] - named[1]}
        return None

    async def post(self, body, prompt, schema, name):
        """One question: (answer, finish reason, generated tokens, reasoning
        text). Structured output when the
        server takes it, else asked plainly with the schema in the prompt.
        A prompt longer than the model's window comes back as an error with
        the finish reason "window"."""
        if self.structured:
            response = await self.send(
                body
                | {
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": name, "strict": True, "schema": schema},
                    }
                }
            )
            if over_window(response):
                return {"error": "longer than the model's window"}, "window", 0, ""
            if 400 <= response.status_code < 500:
                # This server does not take structured output; ask plainly.
                self.structured = False
            else:
                response.raise_for_status()
                return read(response), finish_reason(response), usage(response), reasoning_of(response)
        response = await self.send(
            body
            | {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt + "\nThe JSON must match this schema: " + json.dumps(schema),
                    }
                ]
            }
        )
        if over_window(response):
            return {"error": "longer than the model's window"}, "window", 0, ""
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
