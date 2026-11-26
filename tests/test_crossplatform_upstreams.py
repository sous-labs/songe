import asyncio
import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest

from songe.upstream import create
from songe.upstream.llamacpp import LlamaCPP
from songe.upstream.ollama import Ollama
from songe.upstream.vllm import VLLM


@pytest.fixture(autouse=True)
def measured_memory(monkeypatch):
    from songe.platform import Memory
    from songe.upstream import protection

    monkeypatch.setattr(
        protection,
        "current",
        lambda: SimpleNamespace(
            name="test",
            memory=lambda: Memory(10**9, 9 * 10**8),
            accelerators=lambda: [],
        ),
    )


def config(name):
    return SimpleNamespace(upstream=name, upstream_url="http://test/v1")


@pytest.mark.parametrize(
    "name,cls", [("ollama", Ollama), ("llamacpp", LlamaCPP), ("vllm", VLLM)]
)
def test_registry(name, cls, tmp_path):
    result = create(config(name), tmp_path)
    assert isinstance(result, cls)
    assert result.name == name and result.serves


async def test_ollama_upload_import_warm_park_restore(tmp_path):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"gguf bytes")
    digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    base_digest = digest
    calls, running, blobs = [], {"user:latest": 100, "songe:night-1": 200}, set()

    def handler(request):
        body = (
            json.loads(request.content)
            if request.content and not request.url.path.startswith("/api/blobs/")
            else None
        )
        calls.append((request.method, request.url.path, body))
        if request.url.path == "/api/show":
            if body["model"] in running:
                weights = digest if body["model"] == "songe:night-2" else base_digest
                return httpx.Response(200, json={"modelfile": "FROM /blobs/" + weights})
            return httpx.Response(404)
        if request.url.path == "/api/ps":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": name, "size_vram": size}
                        for name, size in running.items()
                    ]
                },
            )
        if request.url.path.startswith("/api/blobs/"):
            if request.method == "HEAD":
                return httpx.Response(200 if digest in blobs else 404)
            assert request.content == path.read_bytes()
            blobs.add(digest)
            return httpx.Response(201)
        if request.url.path == "/api/create":
            assert body == {
                "model": "songe:night-2",
                "files": {"model.gguf": digest},
                "stream": False,
            }
            return httpx.Response(200, json={"status": "success"})
        if request.url.path == "/api/generate":
            if body.get("keep_alive") == 0:
                running.pop(body["model"], None)
            else:
                running[body["model"]] = 100
            return httpx.Response(200, json={"done": True})
        raise AssertionError(request.url)

    upstream = Ollama(config("ollama"), transport=httpx.MockTransport(handler))
    assert await upstream.loaded_bytes() == 0
    version = {"identifier": "songe:night-1", "artifact": str(path)}
    await upstream.ensure(version)
    assert not any(path == "/api/create" for _, path, _ in calls)
    await upstream.unload(version)
    assert "songe:night-1" in running  # Reused instances remain unowned.
    version = {**version, "identifier": "songe:night-2"}
    # A genuinely different candidate needs its own import; identical weights
    # under a new identifier now reuse the already running instance.
    path.write_bytes(b"trained gguf bytes")
    digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    await upstream.load(version)
    assert digest in blobs
    parked = await upstream.park()
    assert set(running) == {"user:latest", "songe:night-1"}
    await upstream.restore(parked, {"songe:night-2"})
    assert set(running) == {"user:latest", "songe:night-1"}
    await upstream.close()


async def test_ollama_reports_the_context_it_runs_each_model_with():
    # Agents are told this limit: Ollama served 4096 (its default) or 16384
    # (OLLAMA_CONTEXT_LENGTH) while opencode was told 8192.
    def handler(request):
        assert request.url.path == "/api/ps"
        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": "qwen1.5b:latest", "context_length": 16384},
                    {"name": "older:latest"},
                ]
            },
        )

    upstream = Ollama(config("ollama"), transport=httpx.MockTransport(handler))
    assert await upstream.served_contexts() == {"qwen1.5b:latest": 16384}


async def test_ollama_http_error_does_not_warm_model(tmp_path):
    path = tmp_path / "base.gguf"
    path.write_bytes(b"bytes")
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(500, json={"error": "import failed"})

    upstream = Ollama(config("ollama"), transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        await upstream.load({"identifier": "night:1", "artifact": str(path)})
    assert "/api/generate" not in calls


@pytest.mark.parametrize("released", [True, False])
async def test_ollama_unload_waits_for_release_and_retains_claim_on_timeout(
    tmp_path, released
):
    path = tmp_path / "owned.gguf"
    path.write_bytes(b"owned model")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    identifier = "songe/test:night-1"
    version = {"identifier": identifier, "artifact": str(path)}
    polls, unloads = [], []

    def handler(request):
        if request.url.path == "/api/show":
            return httpx.Response(
                200, json={"modelfile": "FROM /blobs/sha256-" + digest}
            )
        if request.url.path == "/api/generate":
            body = json.loads(request.content)
            assert body["model"] == identifier and body.get("keep_alive") == 0
            unloads.append(body)
            return httpx.Response(200, json={"done": True})
        assert request.url.path == "/api/ps"
        assert identifier in upstream.owned
        polls.append(True)
        models = [{"name": "user:latest"}]
        if not released or len(polls) < 3:
            models.append({"name": identifier})
        return httpx.Response(200, json={"models": models})

    cfg = config("ollama")
    cfg.upstream_timeout = 1 if released else 0.02
    cfg.guard_min_interval = 0
    upstream = Ollama(cfg, transport=httpx.MockTransport(handler))
    upstream.owned = {identifier: version}
    if released:
        await upstream.unload(version)
        assert identifier not in upstream.owned and len(polls) == 3
    else:
        with pytest.raises(TimeoutError, match="did not release model"):
            await upstream.unload(version)
        assert upstream.owned == {identifier: version} and polls
    assert len(unloads) == 1


async def test_vllm_dynamic_lora_sleep_and_restore(tmp_path):
    base = {"id": "base", "root": "/base", "parent": None}
    user = {"id": "user", "root": str(tmp_path / "user"), "parent": "base"}
    models, calls = [base, user], []

    def handler(request):
        body = json.loads(request.content) if request.content else None
        calls.append((request.url.path, body))
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": list(models)})
        if request.url.path == "/v1/load_lora_adapter":
            models.append(
                {"id": body["lora_name"], "root": body["lora_path"], "parent": "base"}
            )
        if request.url.path == "/v1/unload_lora_adapter":
            models[:] = [m for m in models if m["id"] != body["lora_name"]]
        return httpx.Response(200, text="Success")

    upstream = VLLM(
        config("vllm"),
        transport=httpx.MockTransport(handler),
        memory_probe=lambda: 2048,
    )
    version = {"identifier": "songe/night-1", "artifact": str(tmp_path / "adapter")}
    await upstream.load(version)
    await upstream.ensure(version)
    assert sum(path == "/v1/load_lora_adapter" for path, _ in calls) == 1
    with pytest.raises(RuntimeError, match="other weights"):
        await upstream.load({**version, "artifact": str(tmp_path / "other")})
    assert await upstream.loaded_bytes() == 0
    parked = await upstream.park()
    assert models == [base, user]
    await upstream.restore(parked, {"songe/night-1"})
    assert {m["id"] for m in models} == {"base", "user"}
    assert not any(path in {"/sleep", "/wake_up"} for path, _ in calls)
    await upstream.close()


async def test_vllm_starts_its_own_endpoint_for_the_first_adapter(tmp_path):
    # The user's server answers the base, so Songe's own endpoint is not
    # running when the first trained version is to be loaded. The swap asked
    # that port, got a refused connection, and waited "for the model server"
    # again every few minutes: no Dream could be adopted on such a setup.
    started, loads = [], []

    def handler(request):
        if request.url.host == "user":
            return httpx.Response(200, json={"data": [{"id": "/base", "root": "/base"}]})
        if not started:
            raise httpx.ConnectError("refused", request=request)
        if request.url.path == "/v1/load_lora_adapter":
            loads.append(json.loads(request.content))
            return httpx.Response(200, text="Success")
        return httpx.Response(
            200,
            json={"data": [{"id": "/base", "root": "/base"}] + [
                {"id": b["lora_name"], "root": b["lora_path"], "parent": "/base"} for b in loads
            ]},
        )

    def owned():
        return VLLM(
            SimpleNamespace(upstream="vllm", upstream_url="http://owned/v1", server_source_url="http://user/v1"),
            transport=httpx.MockTransport(handler),
        )

    version = {"id": "night-1", "identifier": "songe/night-1", "artifact": str(tmp_path / "adapter")}
    # A server Songe does not start (no command was found for it): closed is closed.
    with pytest.raises(httpx.ConnectError):
        await owned().load(version)

    async def restart(candidate):
        started.append(candidate["identifier"])

    upstream = owned()
    upstream.supervisor = SimpleNamespace(restart=restart, process=None)
    assert await upstream.resident(version) is False and not started
    await upstream.load(version)
    assert started == ["songe/night-1"]
    assert [b["lora_name"] for b in loads] == ["songe/night-1"]
    assert await upstream.resident(version) is True
    await upstream.load(version)
    assert len(started) == 1 and len(loads) == 1


async def test_vllm_without_sleep_refuses_parking_before_unloading():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "base"}]})
        return httpx.Response(404)

    upstream = VLLM(config("vllm"), transport=httpx.MockTransport(handler))
    assert await upstream.park() == []
    assert calls == []
    assert await upstream.loaded_bytes() == 0


async def test_llamacpp_scale_switch_base_restart_and_real_parking(tmp_path):
    state = {"base": str(tmp_path / "base.gguf"), "running": True}
    old, new = str(tmp_path / "old.gguf"), str(tmp_path / "new.gguf")
    adapters = [
        {"id": 0, "path": old, "scale": 0.25},
        {"id": 1, "path": new, "scale": 0.0},
    ]
    calls, restarts = [], []

    def handler(request):
        calls.append((request.method, request.url.path))
        assert state["running"]
        if request.url.path == "/props":
            return httpx.Response(200, json={"model_path": state["base"]})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "user-base"}]})
        if request.url.path == "/lora-adapters":
            if request.method == "POST":
                scales = {a["id"]: a["scale"] for a in json.loads(request.content)}
                for adapter in adapters:
                    adapter["scale"] = scales.get(adapter["id"], 0)
            return httpx.Response(200, json=[dict(a) for a in adapters])
        raise AssertionError(request.url)

    async def restart(version):
        restarts.append(version)
        state.update(base=version["artifact"], running=True)
        # Reordering ids exercises path-based restoration.
        for adapter in adapters:
            adapter["id"] += 3

    async def stop():
        state["running"] = False

    upstream = LlamaCPP(
        config("llamacpp"),
        transport=httpx.MockTransport(handler),
        restart=restart,
        stop=stop,
        memory_probe=lambda: 512,
    )
    version = {"identifier": "songe/night-1", "artifact": new, "kind": "gguf-lora"}
    await upstream.load(version)
    assert [a["scale"] for a in adapters] == [0.25, 1]
    assert restarts == []
    await upstream.unload(version)
    assert [a["scale"] for a in adapters] == [0.25, 0]
    await upstream.ensure(version)
    assert await upstream.loaded_bytes() == 512
    parked = await upstream.park()
    assert not state["running"]
    await upstream.restore(parked, {"songe/night-1"})
    assert state["running"] and adapters[1]["scale"] == 0
    await upstream.load(
        {
            "identifier": "base2",
            "artifact": str(tmp_path / "base2.gguf"),
            "kind": "gguf-full",
        }
    )
    assert state["base"].endswith("base2.gguf")
    assert [a["scale"] for a in adapters] == [0.25, 0]
    assert len(restarts) == 2
    await upstream.unload(
        {"identifier": "old-base", "artifact": str(tmp_path / "base.gguf")}
    )
    assert state["running"]  # Cleanup of the old version must not stop the new base.
    await upstream.close()


async def test_llamacpp_reports_the_context_the_users_server_runs_with():
    # Agents are told this limit; the 8192 fallback made opencode compact on
    # every turn against a llama-server started with -c 16384.
    def handler(request):
        if request.url.host != "user":
            raise httpx.ConnectError("owned server is down", request=request)
        if request.url.path == "/running":  # llama-swap only
            return httpx.Response(404)
        assert request.url.path == "/v1/models"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "/models/qwen.gguf", "meta": {"n_ctx": 16384}},
                    {"id": "older-build"},
                ]
            },
        )

    upstream = LlamaCPP(
        SimpleNamespace(
            upstream="llamacpp",
            upstream_url="http://owned/v1",
            server_source_url="http://user/v1",
        ),
        transport=httpx.MockTransport(handler),
    )
    assert await upstream.served_contexts() == {"/models/qwen.gguf": 16384}


async def test_vllm_reports_the_context_the_users_server_runs_with():
    # Without it agents got the 8192 fallback: opencode, whose own prompt is
    # larger, compacted on every turn against a vLLM started with
    # --max-model-len 32768 and never answered.
    def handler(request):
        if request.url.host != "user":
            raise httpx.ConnectError("owned server is down", request=request)
        assert request.url.path == "/v1/models"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "/models/Qwen3.5-4B", "max_model_len": 32768},
                    {"id": "no-limit-reported"},
                ]
            },
        )

    upstream = VLLM(
        SimpleNamespace(
            upstream="vllm",
            upstream_url="http://owned/v1",
            server_source_url="http://user/v1",
        ),
        transport=httpx.MockTransport(handler),
    )
    assert await upstream.served_contexts() == {"/models/Qwen3.5-4B": 32768}


async def test_vllm_base_route_survives_a_check_from_another_request(tmp_path):
    # Each request checks the base again. The check used to drop the route
    # before asking the server, so a request sent at the same moment went to
    # the owned port and the agent got 502 "Local upstream unavailable".
    weights = tmp_path / "Qwen3.5-4B"
    weights.mkdir()
    asked, answer = asyncio.Event(), asyncio.Event()

    async def handler(request):
        assert request.url.host == "user"
        asked.set()
        await answer.wait()
        return httpx.Response(
            200, json={"data": [{"id": str(weights), "root": str(weights)}]}
        )

    upstream = VLLM(
        SimpleNamespace(
            upstream="vllm",
            upstream_url="http://owned/v1",
            server_source_url="http://user/v1",
        ),
        transport=httpx.MockTransport(handler),
    )
    version = {
        "id": "base",
        "kind": "base",
        "identifier": str(weights),
        "artifact": str(weights),
    }
    answer.set()
    assert await upstream.resident(version)
    assert upstream.request_url(version) == "http://user/v1"

    asked.clear()
    answer.clear()
    checking = asyncio.create_task(upstream.resident(version))
    await asked.wait()
    assert upstream.request_url(version) == "http://user/v1"
    answer.set()
    assert await checking


async def test_llamacpp_base_route_survives_a_check_from_another_request(tmp_path):
    # Each request checks the base again. The check used to drop the route
    # before asking the user's llama-server, so requests sent at the same
    # moment went to the owned port and the agent got 502.
    weights = tmp_path / "Qwen3.8-27B-UD-Q4_K_M.gguf"
    weights.write_bytes(b"")
    asked, answer = asyncio.Event(), asyncio.Event()

    async def handler(request):
        assert request.url.host == "user"
        if request.url.path == "/props":
            asked.set()
            await answer.wait()
            return httpx.Response(200, json={"model_path": str(weights)})
        return httpx.Response(200, json={"data": [{"id": weights.name}]})

    upstream = LlamaCPP(
        SimpleNamespace(
            upstream="llamacpp",
            upstream_url="http://owned/v1",
            server_source_url="http://user/v1",
        ),
        transport=httpx.MockTransport(handler),
    )
    version = {
        "id": "base",
        "kind": "gguf-full",
        "identifier": weights.name,
        "artifact": str(weights),
    }
    answer.set()
    await upstream.load(version)
    assert upstream.request_url(version) == "http://user/v1"

    asked.clear()
    answer.clear()
    checking = asyncio.create_task(upstream.load(version))
    await asked.wait()
    assert upstream.request_url(version) == "http://user/v1"
    answer.set()
    await checking
    assert upstream.request_url(version) == "http://user/v1"


async def test_mlx_base_route_survives_a_check_from_another_request(tmp_path):
    from songe.upstream.mlx import MLX

    weights = tmp_path / "Qwen3.8-27B-4bit"
    weights.mkdir()
    asked, answer = asyncio.Event(), asyncio.Event()

    async def handler(request):
        assert request.url.host == "user"
        asked.set()
        await answer.wait()
        return httpx.Response(200, json={"data": [{"id": str(weights)}]})

    upstream = MLX(
        SimpleNamespace(
            upstream="mlx",
            upstream_url="http://owned/v1",
            server_source_url="http://user/v1",
        ),
        tmp_path,
    )
    upstream.transport = httpx.MockTransport(handler)
    version = {"id": "base", "identifier": str(weights), "artifact": str(weights)}
    answer.set()
    await upstream.load(version)
    assert upstream.request_url(version) == "http://user/v1"

    asked.clear()
    answer.clear()
    checking = asyncio.create_task(upstream.load(version))
    await asked.wait()
    assert upstream.request_url(version) == "http://user/v1"
    answer.set()
    await checking


async def test_llamacpp_unregistered_adapter_and_missing_supervisor_fail_closed(
    tmp_path,
):
    calls = []

    def handler(request):
        calls.append((request.method, request.url.path))
        if request.url.path == "/props":
            return httpx.Response(200, json={"model_path": "/base.gguf"})
        return httpx.Response(200, json=[])

    upstream = LlamaCPP(config("llamacpp"), transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="not registered"):
        await upstream.load(
            {
                "identifier": "night",
                "artifact": str(tmp_path / "new.gguf"),
                "kind": "gguf-lora",
            }
        )
    assert not any(method == "POST" for method, _ in calls)
    assert await upstream.park() == []
    with pytest.raises(RuntimeError, match="supervised restart"):
        await upstream.load({"identifier": "base", "artifact": "/other.gguf"})
    assert await upstream.loaded_bytes() == 0


@pytest.mark.parametrize("kind", ["gguf-lora", "gguf-full"])
async def test_llamacpp_publication_load_restarts_parked_endpoint(tmp_path, kind):
    base, adapter = str(tmp_path / "base.gguf"), str(tmp_path / "adapter.gguf")
    state, restarts = {"running": False}, []
    slots = [{"id": 7, "path": adapter, "scale": 0}]

    def handler(request):
        if not state["running"]:
            raise httpx.ConnectError("Endpoint is parked", request=request)
        if request.url.path == "/props":
            return httpx.Response(200, json={"model_path": base})
        assert request.url.path == "/lora-adapters"
        if request.method == "POST":
            assert json.loads(request.content) == [{"id": 7, "scale": 1.0}]
            slots[0]["scale"] = 1
        return httpx.Response(200, json=slots)

    async def restart(version):
        restarts.append(version)
        state["running"] = True

    async def stop():
        state["running"] = False

    version = {
        "identifier": "night-1",
        "kind": kind,
        "base": base,
        "artifact": adapter if kind == "gguf-lora" else base,
    }
    upstream = LlamaCPP(
        config("llamacpp"),
        transport=httpx.MockTransport(handler),
        restart=restart,
        stop=stop,
    )
    # swap.install calls load, not ensure: the publication path must restart.
    await upstream.load(version)
    assert restarts == [version] and state["running"]
    assert slots[0]["scale"] == (1 if kind == "gguf-lora" else 0)

    state["running"] = False
    unmanaged = LlamaCPP(config("llamacpp"), transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.ConnectError):
        await unmanaged.load(version)
    assert restarts == [version]


async def test_ollama_lists_what_it_holds_so_background_work_never_loads_a_model():
    """Reported by Jack in a test on 11-24: Songe could not tell what Ollama
    had loaded, took every model as loaded, and Quick Dream's before and
    after examples loaded an 8B (6 GB) beside the conversion for training."""
    from songe.schedule import ModelScheduler

    running = []

    def handler(request):
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": running})
        return httpx.Response(404)

    upstream = Ollama(config("ollama"), transport=httpx.MockTransport(handler))

    class Scheduler:
        installation_config = SimpleNamespace(upstream_url="http://test/v1", server_source_url="")
        answered_running = False
        loaded_listing = ModelScheduler.loaded_listing
        server_listing = ModelScheduler.server_listing
        loaded_identifiers = ModelScheduler.loaded_identifiers

    scheduler = Scheduler()
    scheduler.upstream = upstream
    assert await scheduler.loaded_identifiers() == set()  # nothing loaded: work waits
    running.append({"name": "qwen3:8b", "model": "qwen3:8b"})
    running.append({"name": "llama3:latest", "model": "llama3:latest"})
    upstream.routes["songe/qwen3-8b"] = "qwen3:8b"  # the user's model, reused
    assert await scheduler.loaded_identifiers() == {
        "qwen3:8b", "llama3", "llama3:latest", "songe/qwen3-8b"}
