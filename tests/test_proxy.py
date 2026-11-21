import asyncio
import json
import re

import httpx
import pytest

from songe.proxy import create_app, decode_response
from tests.test_store import version


class Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


def sse():
    chunks = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "shell", "arguments": '{"cmd":'},
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '"pytest"}'}}
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]
    return (
        b"".join(b"data: " + json.dumps(c).encode() + b"\r\n\r\n" for c in chunks)
        + b"data: [DONE]\r\n\r\n"
    )


async def pair(config, store, handler):
    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(config, store, client=upstream)
    return app, upstream


async def test_streaming_tools_byte_faithful_and_capture(config, store):
    wire = sse()
    seen = []
    stream = Chunks([wire[i : i + 7] for i in range(0, len(wire), 7)])

    async def handler(request):
        seen.append(request)
        return httpx.Response(
            200,
            stream=stream,
            headers={"content-type": "text/event-stream", "x-custom": "kept"},
        )

    store.add_version(version("night-1", 1))
    store.swap("night-1", None)
    app, upstream = await pair(config, store, handler)
    payload = {
        "model": "songe/test",
        "stream": True,
        "messages": [{"role": "user", "content": "Test"}],
        "tools": [
            {
                "type": "function",
                "function": {"name": "shell", "parameters": {"type": "object"}},
            }
        ],
        "temperature": 0.3,
        "extra": {"keep": True},
    }
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client,
    ):
        response = await client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"authorization": "Bearer secret"},
        )
        assert response.content == wire
        assert response.headers["x-custom"] == "kept"
    await upstream.aclose()
    forwarded = json.loads(seen[0].content)
    assert forwarded == payload | {"model": "night-1"}
    assert seen[0].headers["authorization"] == "Bearer secret"
    logged = store.exchanges()[0]
    assert logged["request"] == payload
    assert logged["response"]["choices"][0]["message"]["tool_calls"][0]["function"] == {
        "name": "shell",
        "arguments": '{"cmd":"pytest"}',
    }
    assert "secret" not in store.path.read_bytes().decode(errors="ignore")
    assert stream.closed


@pytest.mark.parametrize("streaming", [True, False])
async def test_replies_name_the_model_the_agent_asked_for(config, store, streaming):
    """Live run: a request for songe/<model> came back as songe/<model>/night-1."""
    if streaming:
        wire = b"".join(
            b"data: " + json.dumps({"id": "x", "model": "night-1", "choices": [
                {"index": 0, "delta": {"content": t}, "finish_reason": None}]}).encode() + b"\n\n"
            for t in ("Hel", "lo")
        ) + b"data: [DONE]\n\n"
        kind = "text/event-stream"
    else:
        wire = json.dumps({"id": "x", "model": "night-1", "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "Hello"},
             "finish_reason": "stop"}]}, separators=(",", ":")).encode()
        kind = "application/json"
    # Three-byte chunks split the name at every position.
    stream = Chunks([wire[i : i + 3] for i in range(0, len(wire), 3)])

    async def handler(request):
        return httpx.Response(200, stream=stream, headers={"content-type": kind,
                                                           "content-length": str(len(wire))})

    store.add_version(version("night-1", 1))
    store.swap("night-1", None)
    app, upstream = await pair(config, store, handler)
    payload = {"model": "songe/test", "stream": streaming,
               "messages": [{"role": "user", "content": "Hi"}]}
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client,
    ):
        response = await client.post("/v1/chat/completions", json=payload)
    await upstream.aclose()
    assert b"night-1" not in response.content
    assert response.content == wire.replace(b'"model": "night-1"', b'"model": "songe/test"').replace(
        b'"model":"night-1"', b'"model":"songe/test"')
    assert "content-length" not in response.headers or int(response.headers["content-length"]) == len(response.content)


@pytest.mark.parametrize(
    "path,method",
    [
        ("/v1/chat/completions", "POST"),
        ("/v1/completions", "POST"),
        ("/v1/embeddings", "POST"),
    ],
)
async def test_upstream_errors_and_unmapped_bytes_unchanged(
    config, store, path, method
):
    raw = b'{ "model": "unrelated", "input": "unchanged" }'
    error = b'{"error":"bad request","details":[1,2]}'

    async def handler(request):
        if method == "POST":
            assert request.content == raw
        assert request.url.query == b"test=yes"
        return httpx.Response(
            429,
            stream=Chunks([error]),
            headers={"retry-after": "3", "content-type": "application/json"},
        )

    app, upstream = await pair(config, store, handler)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client,
    ):
        response = await client.request(
            method, path + "?test=yes", content=raw if method == "POST" else None
        )
        assert response.status_code == 429
        assert response.content == error
        assert response.headers["retry-after"] == "3"
    await upstream.aclose()


@pytest.mark.parametrize(
    "raw",
    [
        b'{"model":"unrelated","messages":[{"role":"user","content":"hi"}]}',
        b'{"messages":[{"role":"user","content":"hi"}]}',
        b"{not json",
    ],
)
async def test_unmapped_requests_reach_the_users_server(config, store, raw):
    # The owned server runs only while Songe needs it (#10); the user's own
    # server is always up and answers what no version claims.
    config.upstream = "llamacpp"
    config.upstream_url = "http://127.0.0.1:41001/v1"
    config.server_source_url = "http://127.0.0.1:8080/v1"
    config.server_owned = True

    async def handler(request):
        if request.url.port != 8080:
            raise httpx.ConnectError("refused", request=request)
        assert request.content == raw
        return httpx.Response(200, stream=Chunks([b'{"choices":[]}']))

    app, upstream = await pair(config, store, handler)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client,
    ):
        response = await client.post("/v1/chat/completions", content=raw)
        assert response.status_code == 200
    await upstream.aclose()


async def test_logging_failure_does_not_break_proxy(config, store, monkeypatch):
    def fail(*a):
        raise OSError("disk full")

    monkeypatch.setattr(store, "record", fail)

    async def handler(request):
        return httpx.Response(
            200,
            stream=Chunks(
                [b'{"choices":[{"message":{"role":"assistant","content":"Okay"}}]}']
            ),
        )

    app, upstream = await pair(config, store, handler)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client,
    ):
        for _ in range(2):
            response = await client.post(
                "/v1/chat/completions", json={"model": "songe/test", "messages": []}
            )
            assert response.status_code == 200
    await upstream.aclose()


async def test_fallback_explicit_base_path_and_adapter(config, store):
    config.upstream = "mlx"

    async def handler(request):
        payload = json.loads(request.content)
        assert payload["model"] == config.model
        assert "adapters" in payload and payload["adapters"] is None
        return httpx.Response(200, stream=Chunks([b'{"choices":[]}']))

    app, upstream = await pair(config, store, handler)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client,
    ):
        assert (
            await client.post(
                "/v1/completions", json={"model": "songe/test", "prompt": "Hi"}
            )
        ).status_code == 200
    await upstream.aclose()


def test_partial_sse_is_not_training_data():
    with pytest.raises(ValueError, match="Incomplete"):
        decode_response(b'data: {"choices":[]}\n\n', True)


async def test_ui_contract_origin_and_assets(config, store):
    app = create_app(config, store)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client,
    ):
        state = (await client.get("/api/state")).json()
        assert state["schema"] == 1 and state["nights"] == []
        html = await client.get("/")
        assert "frame-ancestors 'none'" in html.headers["content-security-policy"]
        # Asset URLs carry a content hash so a cached older script never runs.
        assert re.search(r'src="/assets/next\.js\?v=[0-9a-f]{12}"', html.text)
        assert re.search(r'href="/assets/next\.css\?v=[0-9a-f]{12}"', html.text)
        # The draft's old address leads to the dashboard, query kept.
        moved = await client.get("/next?key=abc")
        assert moved.status_code == 308 and moved.headers["location"] == "/?key=abc"
        assert html.headers["cache-control"] == "no-cache"
        font = await client.get("/assets/outfit.woff2")
        assert font.status_code == 200
        assert font.headers["cache-control"] == "no-cache"
        assert (
            await client.post(
                "/api/return",
                json={"night": 1},
                headers={"origin": "https://evil.example"},
            )
        ).status_code == 403
        assert (
            await client.post(
                "/api/return",
                json={"night": 1},
                headers={"host": "rebound.example", "origin": "http://rebound.example"},
            )
        ).status_code == 403
        assert (await client.post("/api/return", content="night=1")).status_code == 415


async def test_disconnect_closes_upstream_and_stops_consuming(config, store):
    # Drive actual ASGI disconnect semantics; ASGITransport deliberately buffers
    # responses and cannot prove cancellation propagates to the upstream.
    started, closed = asyncio.Event(), asyncio.Event()

    class Blocking(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[]}\n\n'
            started.set()
            await asyncio.Event().wait()

        async def aclose(self):
            closed.set()

    async def handler(request):
        return httpx.Response(200, stream=Blocking())

    app, upstream = await pair(config, store, handler)
    first = True

    async def receive():
        nonlocal first
        if first:
            first = False
            return {
                "type": "http.request",
                "body": b'{"model":"songe/test","stream":true,"messages":[]}',
                "more_body": False,
            }
        await started.wait()
        return {"type": "http.disconnect"}

    messages = []

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 7613),
    }
    async with app.router.lifespan_context(app):
        await asyncio.wait_for(app(scope, receive, send), 2)
    assert closed.is_set()
    await upstream.aclose()


async def test_disconnect_before_upstream_headers_cancels_request(config, store):
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def handler(request):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    app, upstream = await pair(config, store, handler)
    first = True

    async def receive():
        nonlocal first
        if first:
            first = False
            return {"type": "http.request", "body": b"{}", "more_body": False}
        await started.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        pass

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 7613),
    }
    async with app.router.lifespan_context(app):
        await asyncio.wait_for(app(scope, receive, send), 2)
    assert cancelled.is_set()
    await upstream.aclose()
