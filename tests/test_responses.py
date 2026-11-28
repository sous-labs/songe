import asyncio
import json
from pathlib import Path

import httpx
import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment
from starlette.requests import ClientDisconnect

from songe.proxy import create_app, until_disconnect
from songe.responses import StreamTranslator, Unsupported, from_chat, to_chat
from tests.test_proxy import Chunks, pair
from tests.test_store import version

# The shape of a Codex CLI 0.172 request (captured), cut down.
CODEX = {
    "model": "songe/test",
    "stream": True,
    "input": [
        {"type": "message", "role": "developer",
         "content": [{"type": "input_text", "text": "You are a coding agent."}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "say hi"}]},
        {"type": "function_call", "id": "fc_1", "name": "exec_command",
         "arguments": '{"cmd": "echo hi"}', "call_id": "call_1"},
        {"type": "function_call_output", "call_id": "call_1", "output": "hi\n"},
        {"type": "reasoning", "summary": [], "encrypted_content": "opaque"},
    ],
    "tools": [
        {"type": "function", "name": "exec_command", "description": "Runs a command.",
         "strict": False, "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}},
        {"type": "namespace", "name": "multi_agent_v1", "description": "Sub-agents.",
         "tools": [{"type": "function", "name": "close_agent", "parameters": {"type": "object"}}]},
        {"type": "web_search", "external_web_access": False},
    ],
    "tool_choice": "auto",
    "parallel_tool_calls": True,
    "reasoning": {"summary": "auto"},
    "store": False,
    "include": ["reasoning.encrypted_content"],
}


def test_a_codex_request_becomes_a_chat_completion():
    chat, names = to_chat(CODEX)
    assert chat["messages"] == [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "say hi"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "exec_command", "arguments": '{"cmd": "echo hi"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "hi\n"},
    ]
    assert [t["function"]["name"] for t in chat["tools"]] == [
        "exec_command", "multi_agent_v1__close_agent"  # web_search: nothing local runs it
    ]
    assert names["multi_agent_v1__close_agent"] == ("function", "multi_agent_v1", "close_agent")
    assert chat["tool_choice"] == "auto" and chat["parallel_tool_calls"] is True
    assert chat["stream"] and chat["stream_options"] == {"include_usage": True}
    assert "reasoning" not in chat and "store" not in chat


def test_text_and_calls_of_one_reply_are_one_assistant_turn():
    chat, _ = to_chat({"model": "m", "instructions": "Be brief.", "max_output_tokens": 64, "input": [
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Looking."}]},
        {"type": "function_call", "name": "a", "arguments": "{}", "call_id": "c1"},
        {"type": "custom_tool_call", "name": "apply_patch", "input": "*** Begin Patch", "call_id": "c2"},
        {"type": "custom_tool_call_output", "call_id": "c2", "output": [{"type": "input_text", "text": "Done"}]},
    ]})
    assert chat["messages"][0] == {"role": "system", "content": "Be brief."}
    turn = chat["messages"][1]
    assert turn["content"] == "Looking."
    assert [c["function"] for c in turn["tool_calls"]] == [
        {"name": "a", "arguments": "{}"},
        {"name": "apply_patch", "arguments": '{"input": "*** Begin Patch"}'},
    ]
    assert chat["messages"][2] == {"role": "tool", "tool_call_id": "c2", "content": "Done"}
    assert chat["max_tokens"] == 64


def test_stored_responses_are_refused():
    with pytest.raises(Unsupported, match="store: false"):
        to_chat({"model": "m", "input": "hi", "previous_response_id": "resp_1"})


def chat_sse(*chunks, done=True):
    wire = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks)
    return wire + (b"data: [DONE]\n\n" if done else b"")


def delta(finish=None, **fields):
    return {"choices": [{"index": 0, "delta": fields, "finish_reason": finish}]}


def events(body):
    found = []
    for block in body.decode().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "data" in lines:
            found.append(json.loads(lines["data"]))
    return found


async def post(app, payload, headers=None):
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client,
    ):
        return await client.post("/v1/responses", json=payload, headers=headers or {})


async def test_codex_streams_through_songe_and_the_chat_is_recorded(config, store):
    wire = chat_sse(
        delta(role="assistant", reasoning_content="Think."),
        delta(content="Hel"),
        delta(content="lo"),
        delta(tool_calls=[{"index": 0, "id": "call_9", "type": "function",
                           "function": {"name": "multi_agent_v1__close_agent", "arguments": '{"id":'}}]),
        delta("tool_calls", tool_calls=[{"index": 0, "function": {"arguments": '"a"}'}}]),
        {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}},
    )
    seen = []

    async def handler(request):
        seen.append(request)
        return httpx.Response(200, stream=Chunks([wire[i:i + 9] for i in range(0, len(wire), 9)]),
                              headers={"content-type": "text/event-stream"})

    store.add_version(version("night-1", 1))
    store.swap("night-1", None)
    app, upstream = await pair(config, store, handler)
    response = await post(app, CODEX, {"session-id": "s1", "accept-encoding": "zstd,gzip"})
    await upstream.aclose()
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    # The learner's version serves it, through the chat path.
    assert seen[0].url.path == "/v1/chat/completions"
    sent = json.loads(seen[0].content)
    assert sent["model"] == "night-1" and sent["messages"][1]["content"] == "say hi"
    assert seen[0].headers["accept-encoding"] == "identity"
    assert seen[0].headers["x-session-affinity"] == "s1"
    found = events(response.content)
    kinds = [e["type"] for e in found]
    assert kinds[:2] == ["response.created", "response.in_progress"]
    assert kinds[-1] == "response.completed"
    assert "".join(e["delta"] for e in found if e["type"] == "response.output_text.delta") == "Hello"
    assert [e["sequence_number"] for e in found] == list(range(len(found)))
    final = found[-1]["response"]
    assert final["model"] == "songe/test"
    assert final["output"][0]["content"] == [{"type": "reasoning_text", "text": "Think."}]
    assert final["output"][1]["content"][0]["text"] == "Hello"
    assert final["output"][2] == {
        "type": "function_call", "id": "fc_call_9", "status": "completed", "call_id": "call_9",
        "name": "close_agent", "arguments": '{"id":"a"}', "namespace": "multi_agent_v1",
    }
    added = next(e for e in found if e["type"] == "response.output_item.added"
                 and e["item"]["type"] == "function_call")
    assert added["item"]["name"] == "close_agent"
    assert final["usage"]["input_tokens"] == 7 and final["usage"]["output_tokens"] == 3
    # Recorded as the chat completion it was, in the shape Songe learns from.
    logged = store.exchanges()[0]
    assert logged["request"]["messages"] == to_chat(CODEX)[0]["messages"]
    assert logged["response"]["choices"][0]["message"]["content"] == "Hello"


async def test_a_reply_that_is_not_streamed(config, store):
    answer = {
        "choices": [{"index": 0, "finish_reason": "length", "message": {
            "role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "apply_patch", "arguments": '{"input": "patch"}'}}]}}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    }

    async def handler(request):
        return httpx.Response(200, stream=Chunks([json.dumps(answer).encode()]),
                              headers={"content-type": "application/json"})

    app, upstream = await pair(config, store, handler)
    payload = {"model": "songe/test", "input": "hi",
               "tools": [{"type": "custom", "name": "apply_patch", "format": {"type": "grammar"}}]}
    response = await post(app, payload)
    await upstream.aclose()
    body = response.json()
    assert body["status"] == "incomplete"
    assert body["incomplete_details"] == {"reason": "max_output_tokens"}
    assert body["output"] == [{"type": "custom_tool_call", "id": "ctc_c1", "status": "completed",
                               "call_id": "c1", "name": "apply_patch", "input": "patch"}]
    assert store.exchanges()[0]["request"]["messages"] == [{"role": "user", "content": "hi"}]


def test_a_stream_cut_short_fails_instead_of_completing():
    translator = StreamTranslator("songe/test", {})
    translator.start()
    translator.feed(chat_sse(delta(content="Hal"), done=False))
    kinds = [e["type"] for e in events(translator.end())]
    assert kinds[-1] == "response.failed"
    assert "response.completed" not in kinds


async def test_errors_pass_through_and_bad_requests_are_refused(config, store):
    async def handler(request):
        return httpx.Response(500, stream=Chunks([b'{"error": {"message": "boom"}}']),
                              headers={"content-type": "application/json"})

    app, upstream = await pair(config, store, handler)
    response = await post(app, {"model": "songe/test", "input": "hi"})
    assert response.status_code == 500 and response.json()["error"]["message"] == "boom"
    response = await post(app, {"model": "songe/test", "input": "hi", "previous_response_id": "r"})
    assert response.status_code == 400
    assert "store: false" in response.json()["error"]["message"]
    await upstream.aclose()


def test_the_route_exists(config, store):
    app = create_app(config, store)
    assert any(getattr(r, "path", None) == "/v1/responses" for r in app.routes)


# Qwen3.5's chat template (Qwen/Qwen3.5-0.8B, Apache-2.0), as llama.cpp and
# vLLM render it: it refuses a system message anywhere but first.
QWEN35 = (Path(__file__).parent / "fixtures" / "qwen3.5-chat-template.jinja").read_text()


def render(messages, tools=None):
    def refuse(message):
        raise ValueError(message)

    # The servers hand the template a call's arguments parsed.
    messages = [
        {**m, "tool_calls": [
            {**c, "function": {**c["function"], "arguments": json.loads(c["function"]["arguments"])}}
            for c in m["tool_calls"]]} if m.get("tool_calls") else m
        for m in messages
    ]
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = refuse
    return env.from_string(QWEN35).render(messages=messages, tools=tools, add_generation_prompt=True)


# Codex 0.171 as the review captured it: instructions, a developer message,
# two user messages; then a tool loop, and a developer message mid-way.
LOOP = {
    "model": "songe/test",
    "instructions": "You are Codex.",
    "input": [
        {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "Sandbox: read-only."}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "<environment_context/>"}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "list files"}]},
        {"type": "reasoning", "id": "rs_0", "summary": [],
         "content": [{"type": "reasoning_text", "text": "I should run ls."}]},
        {"type": "function_call", "name": "exec_command", "arguments": '{"cmd": "ls"}', "call_id": "c1"},
        {"type": "function_call_output", "call_id": "c1", "output": "README.md"},
        {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "Sandbox: write."}]},
        {"type": "reasoning", "summary": [], "encrypted_content": "opaque"},
    ],
    "tools": [{"type": "function", "name": "exec_command",
               "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}}],
}


def test_the_prompt_is_one_system_message_that_qwen35_renders():
    chat, _ = to_chat(LOOP)
    roles = [m["role"] for m in chat["messages"]]
    assert roles == ["system", "user", "user", "assistant", "tool", "user"]
    assert chat["messages"][0]["content"] == "You are Codex.\n\nSandbox: read-only."
    assert chat["messages"][5] == {"role": "user", "content": "Sandbox: write."}
    prompt = render(chat["messages"], chat["tools"])
    assert "You are Codex.\n\nSandbox: read-only." in prompt
    # The template is the one that refuses: two system messages fail.
    late = [chat["messages"][0], {"role": "system", "content": "x"}, *chat["messages"][1:]]
    with pytest.raises(ValueError, match="System message must be at the beginning"):
        render(late)


def test_readable_reasoning_goes_back_with_its_turn():
    chat, _ = to_chat(LOOP)
    turn = chat["messages"][3]
    assert turn["reasoning_content"] == "I should run ls."
    assert turn["tool_calls"][0]["function"]["name"] == "exec_command"
    # Qwen3.5 shows the thinking of the current tool loop to the model again.
    assert "<think>\nI should run ls.\n</think>" in render(chat["messages"][:5], chat["tools"])
    # Encrypted reasoning has nothing to read, and nothing is added for it.
    assert all("reasoning_content" not in m for i, m in enumerate(chat["messages"]) if i != 3)
    # Reasoning before text and calls of one reply goes on that one turn.
    chat, _ = to_chat({"model": "m", "input": [
        {"type": "message", "role": "user", "content": "hi"},
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "Greet."}]},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hi."}]},
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "Then look."}]},
        {"type": "function_call", "name": "a", "arguments": "{}", "call_id": "c1"},
    ]})
    assert chat["messages"][1] == {
        "role": "assistant", "content": "Hi.", "reasoning_content": "Greet.\n\nThen look.",
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "a", "arguments": "{}"}}],
    }


def test_namespaced_names_that_collide_are_kept_apart():
    chat, names = to_chat({"model": "m", "input": [
        {"type": "function_call", "name": "b", "namespace": "a", "arguments": "{}", "call_id": "c1"},
        {"type": "function_call", "name": "a__b", "arguments": "{}", "call_id": "c2"},
    ], "tools": [
        {"type": "function", "name": "a__b"},
        {"type": "namespace", "name": "a", "tools": [{"type": "function", "name": "b"}]},
        {"type": "function", "name": "a__b"},  # the same tool again
    ]})
    assert [t["function"]["name"] for t in chat["tools"]] == ["a__b", "a__b_2"]
    assert names == {"a__b": ("function", None, "a__b"), "a__b_2": ("function", "a", "b")}
    assert [c["function"]["name"] for c in chat["messages"][0]["tool_calls"]] == ["a__b_2", "a__b"]
    answer = {"choices": [{"message": {"role": "assistant", "tool_calls": [
        {"id": "x", "function": {"name": "a__b_2", "arguments": "{}"}}]}}]}
    call = from_chat(answer, "m", names)["output"][0]
    assert (call["namespace"], call["name"]) == ("a", "b")


def test_calls_without_an_id_get_one_of_their_own():
    answer = {"choices": [{"message": {"role": "assistant", "tool_calls": [
        {"function": {"name": "a", "arguments": "{}"}}, {"function": {"name": "a", "arguments": "{}"}}]}}]}
    first = [c["call_id"] for c in from_chat(answer, "m", {})["output"]]
    second = [c["call_id"] for c in from_chat(answer, "m", {})["output"]]
    assert len(set(first + second)) == 4
    translator = StreamTranslator("m", {})
    translator.start()
    added = events(translator.feed(chat_sse(
        delta(tool_calls=[{"index": 0, "function": {"name": "a", "arguments": "{}"}}]),
        delta("tool_calls"),
    )))
    done = events(translator.end())[-1]["response"]["output"][0]
    assert added[0]["item"]["call_id"] == done["call_id"] != "call_0"


def test_a_reply_cut_at_the_token_limit_ends_incomplete():
    translator = StreamTranslator("songe/test", {})
    translator.start()
    translator.feed(chat_sse(delta(content="Hal"), delta("length")))
    last = events(translator.end())[-1]
    assert last["type"] == "response.incomplete"
    assert last["response"]["status"] == "incomplete"
    assert last["response"]["incomplete_details"] == {"reason": "max_output_tokens"}


async def test_a_stream_that_breaks_ends_with_response_failed(config, store):
    class Breaks(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield chat_sse(delta(content="Hal"), done=False)
            raise httpx.ReadError("connection reset")

        async def aclose(self):
            pass

    async def handler(request):
        return httpx.Response(200, stream=Breaks(), headers={"content-type": "text/event-stream"})

    app, upstream = await pair(config, store, handler)
    response = await post(app, {**CODEX, "input": "hi"})
    await upstream.aclose()
    found = events(response.content)
    assert found[-1]["type"] == "response.failed"
    assert "ReadError" in found[-1]["response"]["error"]["message"]
    assert "".join(e["delta"] for e in found if e["type"] == "response.output_text.delta") == "Hal"


@pytest.mark.parametrize("payload", [
    {"model": "m", "input": [{"type": "function_call", "arguments": "{}", "call_id": "c"}]},
    {"model": "m", "input": [{"type": "message", "role": "user", "content": 5}]},
    {"model": "m", "input": "hi", "tools": [{"type": "namespace", "name": "n", "tools": {"a": 1}}]},
    {"model": "m", "input": {"not": "a list"}},
    {"model": "m", "input": "hi", "text": {"format": "json"}, "tool_choice": {"name": 5},
     "tools": [{"type": "function", "name": "a"}]},
])
async def test_malformed_requests_are_refused_with_400(config, store, payload):
    async def handler(request):
        raise AssertionError("nothing reaches the model server")

    app, upstream = await pair(config, store, handler)
    response = await post(app, payload)
    await upstream.aclose()
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


async def test_json_nested_too_deep_is_refused_with_400(config, store):
    app, upstream = await pair(config, store, lambda request: None)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client,
    ):
        response = await client.post("/v1/responses", content=b"[" * 100_000 + b"]" * 100_000,
                                     headers={"content-type": "application/json"})
    await upstream.aclose()
    assert response.status_code == 400


async def test_lessons_reach_the_system_message_and_stay_pinned_per_session(config, store, monkeypatch):
    from songe import quick

    lessons = ["Use uv."]
    monkeypatch.setattr(quick, "active", lambda learner: list(lessons))
    monkeypatch.setattr(quick, "block", lambda found: "\n".join(found))
    sent = []

    async def handler(request):
        sent.append(json.loads(request.content))
        answer = {"choices": [{"index": 0, "finish_reason": "stop",
                               "message": {"role": "assistant", "content": "ok"}}]}
        return httpx.Response(200, stream=Chunks([json.dumps(answer).encode()]),
                              headers={"content-type": "application/json"})

    store.set("primary_model", "base-test")
    app, upstream = await pair(config, store, handler)
    turn = {**LOOP, "stream": False}
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client,
    ):
        for session in ("s1", "s1", "s2"):
            response = await client.post("/v1/responses", json=turn, headers={"session-id": session})
            assert response.status_code == 200
            lessons.append("Sign with Sam.")
    await upstream.aclose()
    systems = [[m for m in chat["messages"] if m["role"] == "system"] for chat in sent]
    # One system message each: the lessons join Codex's prompt, at its end.
    assert [len(found) for found in systems] == [1, 1, 1]
    assert systems[0][0]["content"] == "You are Codex.\n\nSandbox: read-only.\n\nUse uv."
    assert systems[1][0]["content"] == systems[0][0]["content"]  # pinned for s1
    assert systems[2][0]["content"].endswith("Use uv.\nSign with Sam.\nSign with Sam.")  # s2: new
    render(sent[2]["messages"])
    # What Songe records is the agent's own prompt, without the lessons.
    recorded = store.exchanges()
    assert len(recorded) == 3
    assert {e["request"]["messages"][0]["content"] for e in recorded} == {"You are Codex.\n\nSandbox: read-only."}


async def test_a_waiting_request_is_cancelled_when_the_agent_leaves(config, store):
    stalled = asyncio.Event()

    class Stalls(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            yield b'{"choices": '
            stalled.set()
            await asyncio.Event().wait()

        async def aclose(self):
            Stalls.closed = True

    async def handler(request):
        return httpx.Response(200, stream=Stalls(), headers={"content-type": "application/json"})

    app, upstream = await pair(config, store, handler)
    body = json.dumps({"model": "songe/test", "input": "hi"}).encode()
    messages = [{"type": "http.request", "body": body, "more_body": False}]

    async def receive():
        if messages:
            return messages.pop()
        await stalled.wait()
        return {"type": "http.disconnect"}

    replies = []

    async def send(message):
        replies.append(message)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
             "scheme": "http", "path": "/v1/responses", "raw_path": b"/v1/responses", "query_string": b"",
             "root_path": "", "headers": [(b"host", b"127.0.0.1"), (b"content-type", b"application/json"),
                                          (b"content-length", str(len(body)).encode())],
             "client": ("127.0.0.1", 5000), "server": ("127.0.0.1", 7613)}
    async with app.router.lifespan_context(app):
        await asyncio.wait_for(app(scope, receive, send), 5)
    await upstream.aclose()
    assert Stalls.closed
    assert replies[0]["status"] == 499


async def test_until_disconnect_cancels_the_work():
    cancelled = []

    async def work():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    class Leaving:
        async def receive(self):
            return {"type": "http.disconnect"}

    with pytest.raises(ClientDisconnect):
        await until_disconnect(Leaving(), work())
    assert cancelled == [True]
