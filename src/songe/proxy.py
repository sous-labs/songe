import asyncio
import functools
import hashlib
import json
import logging
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route
from starlette.staticfiles import StaticFiles

from . import __version__
from .api import api_routes, guard, interface_data_dir
from .config import Config
from .log import Store
from .upgrade import RUNNING_INSTALL

logger = logging.getLogger("songe")
HOP = {
    b"host",
    b"connection",
    b"keep-alive",
    b"proxy-authenticate",
    b"proxy-authorization",
    b"te",
    b"trailer",
    b"transfer-encoding",
    b"upgrade",
}
CAPTURE_LIMIT = 16 * 1024 * 1024


def headers(raw, changed=False):
    nominated = set()
    for k, v in raw:
        if k.lower() == b"connection":
            nominated |= {x.strip().lower() for x in v.split(b",")}
    return [
        (k, v)
        for k, v in raw
        if k.lower() not in HOP | nominated
        and not (changed and k.lower() in {b"content-length", b"content-encoding"})
    ]


class LessonPins:
    """Quick Dream lessons for each session, fixed when it starts.

    The lessons go at the end of the system message. A Quick Dream that
    changes them mid-session would change the prompt's prefix and make the
    server process the whole conversation again, so a session keeps the
    lessons it started with; the next session gets the new ones. Use the
    client's session ID when supplied and the opening messages otherwise.
    Without a session ID, a request with no assistant turn yet opens a new
    session, so it takes the current lessons even when an earlier chat
    opened with the same messages."""

    def __init__(self, size=2048):
        from collections import OrderedDict

        self.size, self.pins = size, OrderedDict()

    @staticmethod
    def key(learner, messages, session_id):
        from .steps import digest

        if session_id:
            return digest([learner.model_key, session_id])

        def first(role):
            return next((m.get("content") for m in messages if m.get("role") == role), None)

        return digest([learner.model_key, first("system"), first("user")])

    def block_for(self, learner, messages, session_id=None):
        from . import quick

        if not isinstance(messages, list) or not messages:
            return ""
        key = self.key(learner, messages, session_id)
        opening = not session_id and not any(
            isinstance(m, dict) and m.get("role") == "assistant" for m in messages
        )
        if key in self.pins and not opening:
            self.pins.move_to_end(key)
            return self.pins[key]
        text = quick.block(quick.active(learner))
        self.pins[key] = text
        if len(self.pins) > self.size:
            self.pins.popitem(last=False)
        return text


def with_lessons(messages, text):
    """messages with the lessons at the end of the system message."""
    messages = list(messages)
    if messages and messages[0].get("role") == "system":
        system = dict(messages[0])
        content = system.get("content")
        if isinstance(content, list):
            system["content"] = [*content, {"type": "text", "text": text}]
        else:
            system["content"] = f"{content or ''}\n\n{text}".lstrip()
        messages[0] = system
    else:
        messages.insert(0, {"role": "system", "content": text})
    return messages


class ModelRename:
    """Give the agent back the model name it asked for.

    Songe routes a request for songe/<model> to the server's own name for
    the version in use (an adapter id such as songe/<model>/night-1, or the
    base's served id), and servers echo that name in "model". The reply
    carries the requested name instead. Works on raw chunks: a name split
    between two chunks is held back until the next one completes it."""

    def __init__(self, served, requested):
        name = json.dumps(served).encode()
        self.replacement = json.dumps(requested).encode()
        self.patterns = [b'"model":' + name, b'"model": ' + name]
        self.keep = max(len(p) for p in self.patterns) - 1
        self.pending = b""

    def _swap(self, data):
        for pattern in self.patterns:
            data = data.replace(pattern, pattern[: pattern.index(b'"', 8)] + self.replacement)
        return data

    def feed(self, chunk):
        data = self._swap(self.pending + chunk)
        # Hold back only a tail that could start a split name, so a stream's
        # events otherwise pass at once.
        held = 0
        for size in range(min(self.keep, len(data)), 0, -1):
            tail = data[-size:]
            if any(pattern.startswith(tail) for pattern in self.patterns):
                held = size
                break
        self.pending = data[len(data) - held :] if held else b""
        return data[: len(data) - held]

    def flush(self):
        data, self.pending = self.pending, b""
        return data


def decode_response(body, streaming):
    if not streaming:
        return json.loads(body)
    # SSE event framing is independent of transport chunks and CRLF variants.
    choices = {}
    finished = False
    for event in body.decode().replace("\r\n", "\n").split("\n\n"):
        data = "\n".join(
            line[5:].lstrip(" ")
            for line in event.splitlines()
            if line.startswith("data:")
        )
        if not data:
            continue
        if data == "[DONE]":
            finished = True
            continue
        chunk = json.loads(data)
        for choice in chunk.get("choices", []):
            entry = choices.setdefault(
                choice.get("index", 0),
                {
                    "message": {"role": "assistant", "content": ""},
                    "calls": {},
                    "finish_reason": None,
                },
            )
            # Servers send null for absent fields ("tool_calls": null).
            delta, message = choice.get("delta") or {}, entry["message"]
            # Servers name the reasoning reasoning_content or reasoning (vLLM).
            for k in ("content", "reasoning_content", "reasoning"):
                if delta.get(k):
                    message[k] = message.get(k, "") + delta[k]
            for call in delta.get("tool_calls") or []:
                target = entry["calls"].setdefault(
                    call.get("index", 0),
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                if call.get("id"):
                    # Some servers repeat the id on every delta.
                    target["id"] = call["id"]
                for k in ("name", "arguments"):
                    target["function"][k] += (call.get("function") or {}).get(k) or ""
            if choice.get("finish_reason"):
                entry["finish_reason"] = choice["finish_reason"]
    result = []
    for index, entry in sorted(choices.items()):
        if entry["calls"]:
            entry["message"]["tool_calls"] = [
                v for _, v in sorted(entry["calls"].items())
            ]
        result.append(
            {
                "index": index,
                "message": entry["message"],
                "finish_reason": entry["finish_reason"],
            }
        )
    if not finished or not result or any(not c["finish_reason"] for c in result):
        raise ValueError("Incomplete stream")
    return {"choices": result}


def upstream_failure(exc, url):
    """What went wrong reaching the model server, said for the agent's user
    (reported by Jack in a test on 11-24: mlx-lm 0.32 dropped the
    connection on a tool call it could not parse, and the agent saw only
    "Local upstream unavailable", 41 times in 80 conversations)."""
    from urllib.parse import urlparse

    where = urlparse(url).netloc or url
    detail = " ".join(str(exc).split())[:200]
    if isinstance(exc, httpx.ConnectError):
        said = f"nothing answers at {where}"
    elif isinstance(exc, httpx.RemoteProtocolError):
        said = f"the model server at {where} closed the connection without an answer; its own log says why"
    elif isinstance(exc, httpx.TimeoutException):
        said = f"no answer in time from the model server at {where}"
    else:
        said = f"the model server at {where} failed"
    return f"{said} ({type(exc).__name__}{': ' + detail if detail else ''})"


async def open_upstream(client, upstream_request, request):
    """Cancel a stalled upstream even if the client leaves before headers."""

    async def disconnected():
        while True:
            message = await request.receive()
            if message["type"] == "http.disconnect":
                return

    sending = asyncio.create_task(client.send(upstream_request, stream=True))
    watching = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait(
            {sending, watching}, return_when=asyncio.FIRST_COMPLETED
        )
        if watching in done:
            if (
                sending.done()
                and not sending.cancelled()
                and sending.exception() is None
            ):
                await sending.result().aclose()
            raise ClientDisconnect()
        return await sending
    finally:
        watching.cancel()
        if not sending.done():
            sending.cancel()
        await asyncio.gather(watching, sending, return_exceptions=True)


async def until_disconnect(request, work):
    """work's result; ClientDisconnect, with work cancelled, if the client
    leaves first."""

    async def disconnected():
        while True:
            message = await request.receive()
            if message["type"] == "http.disconnect":
                return

    doing = asyncio.ensure_future(work)
    watching = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait({doing, watching}, return_when=asyncio.FIRST_COMPLETED)
        if doing in done:
            return doing.result()
        raise ClientDisconnect()
    finally:
        for task in (doing, watching):
            task.cancel()
        await asyncio.gather(doing, watching, return_exceptions=True)


def create_app(config=None, store=None, client=None, scheduler=None):
    config = config or Config.load()
    store = store or Store(interface_data_dir())
    lesson_pins = LessonPins()
    release = {
        "service_version": __version__,
        "service_install": RUNNING_INSTALL,
        "service_instance": uuid.uuid4().hex,
    }
    from .models import registry

    registry.migrate(config, store)
    from .models.names import catalog

    cached_version = store.version()
    owns_client = client is None
    client = client or httpx.AsyncClient(
        timeout=httpx.Timeout(600, connect=5),
        trust_env=False,
        limits=httpx.Limits(max_connections=128, max_keepalive_connections=32),
    )
    tasks = set()
    # Recordings get their own thread: a write waiting on the database lock
    # (Store.record retries for up to a minute) must not take the shared
    # default pool the request path uses (registry.observe, lesson pins).
    # One thread also keeps recordings in arrival order.
    recorder = ThreadPoolExecutor(max_workers=1, thread_name_prefix="songe-record")
    refresh_task = None
    refresh_revision = 0
    pending_bytes = 0

    def queue_refresh():
        nonlocal refresh_task, refresh_revision
        refresh_revision += 1
        if refresh_task and not refresh_task.done():
            return

        async def refresh():
            from .memory import guarded_run

            seen = None
            while seen != refresh_revision:
                seen = refresh_revision
                try:
                    result = await asyncio.to_thread(
                        guarded_run,
                        [
                            sys.executable,
                            "-m",
                            "songe.experience",
                            "--child",
                            "--data-dir",
                            str(store.root),
                            "--model",
                            config.public_model,
                        ],
                    )
                    if result.returncode:
                        logger.error("experience_refresh_failed")
                        return
                except Exception:
                    logger.exception("experience_refresh_deferred")
                    return

        refresh_task = asyncio.create_task(refresh())

    def queue_record(*args, capture_bytes):
        nonlocal pending_bytes

        async def save():
            nonlocal pending_bytes
            try:
                await asyncio.get_running_loop().run_in_executor(
                    recorder, functools.partial(store.record, *args)
                )
                queue_refresh()
            except Exception:
                # Keep the cause: a full disk and a broken store look alike otherwise.
                logger.exception("experience_write_failed")
            finally:
                pending_bytes -= capture_bytes

        # Bounded pending writes; a failed disk must never stall inference.
        if len(tasks) < 64 and pending_bytes + capture_bytes <= CAPTURE_LIMIT:
            pending_bytes += capture_bytes
            task = asyncio.create_task(save())
            tasks.add(task)
            task.add_done_callback(tasks.discard)
        else:
            logger.error("experience_queue_full")

    @asynccontextmanager
    async def lifespan(app):
        logger.info("service_started version=%s", __version__)
        if scheduler:
            await scheduler.start()
        queue_refresh()
        try:
            yield
        finally:
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            recorder.shutdown(wait=False)
            if refresh_task:
                await asyncio.gather(refresh_task, return_exceptions=True)
            if scheduler:
                await scheduler.close()
            if owns_client:
                await client.aclose()

    async def forward(request: Request):
        if rejected := guard(request):
            return rejected
        if request.url.path == "/v1/models":
            return JSONResponse(
                {
                    "object": "list",
                    "data": [
                        {"id": model, "object": "model", "owned_by": "songe"}
                        for model in dict.fromkeys(catalog(config, store).values())
                    ],
                }
            )
        if scheduler:
            scheduler.traffic()
            scheduler.acquire("admission")
        try:
            if scheduler:
                try:
                    await scheduler.wake()
                except Exception as exc:
                    # The agent gets an error it can show, not a bare 500.
                    logging.getLogger("songe").warning("wake_failed", exc_info=True)
                    return JSONResponse(
                        {
                            "error": {
                                "message": f"Songe could not get the model ready: {exc}"[
                                    :300
                                ],
                                "type": "server_error",
                            }
                        },
                        503,
                    )
            return await relay(request)
        finally:
            if scheduler:
                scheduler.release("admission")

    async def relay(request: Request):
        nonlocal cached_version
        started = time.time()
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > CAPTURE_LIMIT:
                return JSONResponse(
                    {"error": "Request exceeds the local capture limit"}, 413
                )
            body.extend(chunk)
        body = bytes(body)
        original = None
        changed = False
        rename = None
        learner = None
        try:
            cached_version = store.version()
        except Exception:
            logger.error("version_read_failed_using_cached_binding")
        version = cached_version
        serving = None
        binding = version["id"] if version else "base"
        try:
            original = json.loads(body) if body else None
            selected_config = config
            requested = original.get("model") if isinstance(original, dict) else None
            aliases = catalog(config, store)
            is_alias = requested == config.public_model or requested in aliases.values()
            if isinstance(requested, str) and requested:
                model = next(
                    (raw for raw, alias in aliases.items() if alias == requested), None
                )
                model = model or (store.get("primary_model") if is_alias else requested)
                refs = ()
                if (
                    model == requested
                    and requested.startswith("songe/")
                    and not await asyncio.to_thread(registry.find, store, requested)
                ):
                    # A Songe name nobody registered: a model downloaded
                    # after Songe started. Find it in the library; never
                    # register the Songe name itself as a model (live: its
                    # 404 marked the name unsupported and the real model was
                    # renamed songe/<name>-<hash>).
                    from .models import library

                    try:
                        ref = await asyncio.to_thread(library.named, config, requested)
                    except Exception:
                        logger.warning("library_lookup_failed", exc_info=True)
                        ref = None
                    model, refs = (ref.serving_id, (ref,)) if ref else (None, ())
                    if model is None:
                        # Passed on as asked (the server answers 404), not
                        # served by the primary model's version.
                        version, binding = None, f"{requested}:base"
                if model:
                    row = await asyncio.to_thread(
                        registry.observe, config, store, model, refs
                    )
                    model = row["id"]
                    selected = store.scoped(model)
                    learner = selected
                    selected_config = registry.model_config(config, store, model)
                    version = selected.version() if row["enabled"] else None
                    binding = version["id"] if version else f"{model}:base"
                    serving = version
                    if not row["enabled"]:
                        serving = next(
                            (
                                v
                                for v in selected.versions()
                                if v["id"] == "base" or v["id"].startswith("base-")
                            ),
                            None,
                        )
                        if serving:
                            binding = serving["id"]
                    if serving and scheduler and hasattr(scheduler, "ensure_model"):
                        await scheduler.ensure_model(model, serving)
            if isinstance(original, dict) and (
                is_alias or (version or serving) and requested
            ):
                mapped = dict(original)
                mapped["model"] = (
                    (
                        version["identifier"]
                        if config.upstream != "mlx"
                        else version["artifact"]
                    )
                    if version
                    else (
                        (selected_config.base_identifier or requested)
                        if config.upstream != "mlx"
                        else selected_config.model or requested
                    )
                )
                if serving and hasattr(
                    getattr(scheduler, "upstream", None), "request_model"
                ):
                    mapped["model"] = scheduler.upstream.request_model(serving)
                if config.upstream == "mlx":
                    mapped["adapters"] = version.get("adapter") if version else None
                if request.url.path == "/v1/chat/completions" and learner is not None:
                    # Quick Dream: the session's lessons in the system message.
                    # The recorded request stays the agent's own (original).
                    text = await asyncio.to_thread(
                        lesson_pins.block_for,
                        learner,
                        mapped.get("messages"),
                        request.headers.get("x-session-affinity"),
                    )
                    if text:
                        mapped["messages"] = with_lessons(mapped["messages"], text)
                body = json.dumps(
                    mapped, ensure_ascii=False, separators=(",", ":")
                ).encode()
                changed = True
                if isinstance(requested, str) and mapped["model"] != requested:
                    rename = (mapped["model"], requested)
        except (ValueError, TypeError):
            pass
        except Exception as exc:
            # Any binding failure (server SDK errors included) reaches the
            # agent as an OpenAI-shaped error that keeps the server's reason.
            if type(exc).__name__ == "LoadRefused":
                logger.warning("model_load_refused %s", exc)  # Said once per try.
            else:
                logger.exception("model_binding_unavailable")
            return JSONResponse(
                {
                    "error": {
                        "message": "Local model unavailable · "
                        + (" ".join(str(exc).split())[:300] or type(exc).__name__),
                        "type": "upstream_unavailable",
                    }
                },
                502,
            )
        # A Songe-owned server runs only while Songe needs it; what no version
        # claims goes to the user's own server, which is always up.
        upstream_url = config.server_source_url or config.upstream_url
        if serving and hasattr(getattr(scheduler, "upstream", None), "request_url"):
            upstream_url = scheduler.upstream.request_url(serving)
        url = upstream_url.rstrip("/") + request.url.path[3:]
        if request.url.query:
            url += "?" + request.url.query
        upstream_request = client.build_request(
            request.method,
            url,
            content=body,
            headers=headers(request.headers.raw, changed),
        )
        if scheduler:
            scheduler.acquire(binding)
        try:
            upstream = await open_upstream(client, upstream_request, request)
        except ClientDisconnect:
            if scheduler:
                scheduler.release(binding)
            return Response(status_code=499)
        except httpx.HTTPError as exc:
            if scheduler:
                scheduler.release(binding)
            why = upstream_failure(exc, url)
            logger.warning("upstream_unavailable %s", why)
            return JSONResponse(
                {
                    "error": {
                        "message": "Local upstream unavailable · " + why,
                        "type": "upstream_unavailable",
                    }
                },
                502,
            )

        finished = False

        async def finish():
            # Exactly once, whether the body streamed or never started, so an
            # in-flight count can never stay up and block learning forever.
            nonlocal finished
            if finished:
                return
            finished = True
            if scheduler:
                scheduler.release(binding)
            await upstream.aclose()

        # Compressed bodies pass untouched: renaming would need to decode them.
        renamer = (
            ModelRename(*rename)
            if rename and not upstream.headers.get("content-encoding")
            else None
        )

        async def stream():
            captured, overflow, complete = bytearray(), False, False
            try:
                async for chunk in upstream.aiter_raw():
                    if len(captured) + len(chunk) <= CAPTURE_LIMIT:
                        captured.extend(chunk)
                    else:
                        overflow = True
                    yield renamer.feed(chunk) if renamer else chunk
                if renamer:
                    yield renamer.flush()
                complete = True
            finally:
                await finish()
                if (
                    isinstance(original, dict)
                    and request.url.path == "/v1/chat/completions"
                ):
                    response = None
                    try:
                        data = bytes(captured)
                        encoding = upstream.headers.get("content-encoding")
                        if encoding:
                            # Compressed bodies remain untouched on the wire.
                            # Do not decompress potentially unbounded logs.
                            raise ValueError("Compressed capture skipped")
                        if not overflow and complete:
                            response = decode_response(
                                data, bool(original.get("stream"))
                            )
                            if scheduler and hasattr(scheduler, "response_finished"):
                                choice = ((response or {}).get("choices") or [{}])[0]
                                scheduler.response_finished(choice.get("finish_reason"))
                    except Exception:
                        # An odd server body is never allowed to break the
                        # agent's stream; it is recorded as incomplete.
                        complete = False
                    queue_record(
                        request.url.path,
                        original,
                        response,
                        upstream.status_code,
                        complete and not overflow,
                        binding,
                        started,
                        capture_bytes=len(body) + len(captured),
                    )

        result = StreamingResponse(
            stream(),
            status_code=upstream.status_code,
            background=BackgroundTask(finish),
        )
        # A renamed body has a different length.
        result.raw_headers = headers(upstream.headers.raw, changed=renamer is not None)
        return result

    async def responses(request: Request):
        """/v1/responses (Codex CLI), served as a chat completion: see
        responses.py."""
        from .responses import StreamTranslator, Unsupported, from_chat, to_chat

        if rejected := guard(request):
            return rejected
        raw = bytearray()
        async for chunk in request.stream():
            if len(raw) + len(chunk) > CAPTURE_LIMIT:
                return JSONResponse({"error": "Request exceeds the local capture limit"}, 413)
            raw.extend(chunk)
        try:
            asked = json.loads(raw)
            chat, names = to_chat(asked)
        except (Unsupported, ValueError, TypeError, KeyError, AttributeError, RecursionError) as exc:
            # Malformed input (a call without a name, content that is a
            # number, JSON nested too deep) is the agent's error, not Songe's.
            return JSONResponse(
                {"error": {"message": f"Songe cannot serve this request: {exc}",
                           "type": "invalid_request_error"}},
                400,
            )
        data = json.dumps(chat, ensure_ascii=False, separators=(",", ":")).encode()
        kept = [
            (k, v) for k, v in request.headers.raw
            if k.lower() not in {b"content-length", b"content-type", b"accept-encoding"}
        ]
        # Translated bytes must arrive uncompressed. Codex names its session
        # session-id; Songe's lesson pins read x-session-affinity.
        kept += [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(data)).encode()),
            (b"accept-encoding", b"identity"),
        ]
        if "session-id" in request.headers and "x-session-affinity" not in request.headers:
            kept.append((b"x-session-affinity", request.headers["session-id"].encode()))
        path = "/v1/chat/completions"
        scope = {**request.scope, "path": path, "raw_path": path.encode(), "headers": kept}
        sent = False

        async def receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": data, "more_body": False}
            return await request.receive()  # the agent leaving

        result = await forward(Request(scope, receive))
        if not isinstance(result, StreamingResponse) or result.status_code != 200:
            return result  # Songe's own errors and the server's pass through
        model = asked.get("model")
        if not chat.get("stream"):

            async def collect():
                body = bytearray()
                async for chunk in result.body_iterator:
                    body.extend(chunk)
                return body

            try:
                # Nothing sends this body to the agent while it is read, so
                # nothing else notices the agent leaving: watch for it, and
                # stop the server's work when it does.
                body = await until_disconnect(request, collect())
            except ClientDisconnect:
                await result.body_iterator.aclose()
                return Response(status_code=499)
            finally:
                if result.background:
                    await result.background()
            try:
                return JSONResponse(from_chat(json.loads(body), model, names))
            except (ValueError, AttributeError, TypeError):
                return JSONResponse(
                    {"error": {"message": "the model server's answer is not a chat completion",
                               "type": "upstream_error"}},
                    502,
                )

        async def translated():
            translator = StreamTranslator(model, names)
            yield translator.start()
            broke = None
            try:
                async for chunk in result.body_iterator:
                    yield translator.feed(chunk)
            except httpx.HTTPError as exc:
                # The server's stream broke: the agent gets response.failed,
                # not a stream that just stops.
                broke = upstream_failure(exc, config.server_source_url or config.upstream_url)
                logger.warning("upstream_stream_broke %s", broke)
            finally:
                await result.body_iterator.aclose()
            yield translator.end(broke)

        return StreamingResponse(
            translated(), media_type="text/event-stream", background=result.background
        )

    assets = Path(__file__).parent / "ui"

    def stamped_index(page="next.html", names=("next.js", "next.css")):
        # Content-hash the asset URLs so a cached older script or stylesheet
        # can never run against a newer page.
        html = (assets / page).read_text(encoding="utf-8")
        for name in names:
            digest = hashlib.sha256((assets / name).read_bytes()).hexdigest()[:12]
            html = html.replace(f"/assets/{name}", f"/assets/{name}?v={digest}")
        return html

    async def index(request):
        # The learning page is the dashboard, at the root.
        return Response(
            stamped_index(),
            media_type="text/html; charset=utf-8",
            headers={
                # It draws its paper grain into a data: URL.
                "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self'; connect-src 'self'; img-src 'self' data:"
                "; object-src 'none'; frame-ancestors 'none'",
                "Cache-Control": "no-cache",
            },
        )

    async def next_page(request):
        # Links to the draft's old address keep working (with their query,
        # e.g. a tunnel's ?key=).
        from starlette.responses import RedirectResponse

        query = request.url.query
        return RedirectResponse("/" + (f"?{query}" if query else ""), status_code=308)

    app = Starlette(
        routes=[
            Route("/", index),
            Route("/next", next_page),
            # Browsers ask for it on every page load; no icon, no 404.
            Route("/favicon.ico", lambda request: Response(status_code=204)),
            *api_routes(config, store, scheduler, release=release),
            *[
                Route(path, forward, methods=methods)
                for path, methods in (
                    ("/v1/models", ["GET"]),
                    ("/v1/chat/completions", ["POST"]),
                    ("/v1/completions", ["POST"]),
                    ("/v1/embeddings", ["POST"]),
                )
            ],
            Route("/v1/responses", responses, methods=["POST"]),
        ],
        lifespan=lifespan,
    )
    app.mount("/assets", RevalidatedFiles(directory=assets), name="assets")
    return app


class RevalidatedFiles(StaticFiles):
    """Hash-addressed assets are immutable; plain URLs still revalidate."""

    def file_response(self, full_path, stat_result, scope, status_code=200):
        from urllib.parse import parse_qs

        response = super().file_response(full_path, stat_result, scope, status_code)
        version = parse_qs(scope.get("query_string", b"").decode()).get("v", [])
        digest = hashlib.sha256(Path(full_path).read_bytes()).hexdigest()[:12]
        response.headers["Cache-Control"] = (
            "public, max-age=31536000, immutable" if version == [digest] else "no-cache"
        )
        return response
