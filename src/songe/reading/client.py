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
import re
import time

import httpx

# Answer room for a cut-off answer's second try, and for a reader's later
# questions once a model needed it.
ROOM_CAP = 4
TOKENS_CAP = 32768


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
        think=("classify",),
        max_tokens=6000,
        timeout=600,
        client=None,
        server="",
    ):
        self.store, self.url, self.model = store, url.rstrip("/"), model
        # vLLM passes thinking on or off to the chat template, which reads
        # enable_thinking (Qwen, Granite); reasoning_effort is not used there.
        self.template_switch = server == "vllm"
        self.still_idle = still_idle
        # Which questions the model thinks about first: slower, steadier.
        # Measured on the answer key: thinking helps classify only.
        self.think = set(think)
        self.max_tokens = max_tokens
        self.client = client or httpx.AsyncClient(timeout=timeout, trust_env=False)
        self.calls = self.cached = 0
        self.room = 1  # multiplier on max_tokens, grown by cut-off answers
        self.length_retries = 0
        self.seconds = 0.0
        self.structured = True

    async def close(self):
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
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": (max_tokens or self.max_tokens) * (3 if think else 1),
            "stream": False,
        }
        if not think:
            body["reasoning_effort"] = "none"
        if self.template_switch:
            body["chat_template_kwargs"] = {"enable_thinking": think}
        started = time.monotonic()
        # Reasoning models keep thinking even when asked not to (live,
        # Inkling: ~3,000 tokens per answer, 17 of 43 cut off at the limit; a
        # cut-off JSON is lost and asked again on the next read). An answer
        # stopped by the limit is asked once more with twice the room, and
        # this reader keeps the larger room for its later questions.
        body["max_tokens"] = int(body["max_tokens"] * self.room)
        for attempt in range(2):
            answer, finish = await self.post(body, prompt, schema, name)
            if "error" in answer and finish == "length" and attempt == 0:
                self.room = min(self.room * 2, ROOM_CAP)
                self.length_retries += 1
                body = body | {"max_tokens": min(body["max_tokens"] * 2, TOKENS_CAP)}
                continue
            break
        self.calls += 1
        self.seconds += time.monotonic() - started
        if "error" in answer:
            # Not cached: a later read asks again (a cached failure would be
            # reused for good).
            self.errors = getattr(self, "errors", 0) + 1
        else:
            await asyncio.to_thread(cache_put, self.store, key, name, answer)
        return answer


    async def post(self, body, prompt, schema, name):
        """One question: (answer, finish reason). Structured output when the
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
                return read(response), finish_reason(response)
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
        return read(response), finish_reason(response)


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
    # Failures cached by older versions are asked again.
    return None if isinstance(answer, dict) and "error" in answer else answer


def cache_put(store, key, name, answer):
    with store.connect() as db:
        db.execute(
            "INSERT OR REPLACE INTO reading_cache VALUES (?,?,?,?)",
            (key, name, json.dumps(answer, ensure_ascii=False), time.time()),
        )
