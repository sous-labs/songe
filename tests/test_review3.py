"""Regressions for REVIEW-3; all runtimes and allocations are simulated."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from songe.config import Config
from songe.log import Store
from songe.memory import snapshot
from songe.platform import Accel, Memory


def memory_platform(total=100_000, available=90_000):
    return SimpleNamespace(
        name="test",
        memory=lambda: Memory(total, available),
        accelerators=lambda: [],
        on_ac_power=lambda: True,
        idle_seconds=lambda: None,
        thermal=lambda: None,
        process_bytes=lambda pid: 100,
    )


@pytest.mark.parametrize("kind", ["gguf-lora", "gguf-full"])
def test_default_calibration_compact_hf_template_and_partial_retry(
    tmp_path, monkeypatch, kind
):
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers

    from songe import backends, memory, platform
    from songe.sleep import calibrate, fuse, measurement

    raw = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "Read": 1, "Be": 2, "concise": 3}, unk_token="[UNK]"
        )
    )
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="[UNK]"
    )
    tokenizer.chat_template = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}assistant: "
    calls = []

    class Backend:
        def __init__(self, config, check, train=True):
            self.config, self.tokenizer = config, tokenizer

        def prepare(self):
            pass

        def reset_peak_memory(self):
            pass

        def peak_memory(self):
            return 100

        def sample(self, prompt, seed=None):
            return (
                [1] * self.config.response_tokens,
                "Read",
                SimpleNamespace(generation_tps=100, prompt_tps=200),
            )

        def compare(self, *args):
            pass

        def kl_from(self, *args, **kw):
            pass

        def train(self, *args):
            calls.append(self.config.response_tokens)

        def save(self, path):
            path.mkdir(exist_ok=True, parents=True)

    monkeypatch.setattr(platform, "current", memory_platform)
    monkeypatch.setattr(measurement, "process_tree_bytes", lambda *args: 100)
    monkeypatch.setattr(backends, "choose", lambda *args: Backend)
    monkeypatch.setattr(fuse, "export_candidate", lambda *args: None)
    monkeypatch.setattr(fuse, "evaluation_model", lambda *args: str(tmp_path / "base"))
    config = Config(
        model=str(tmp_path / "base"), upstream="llamacpp", artifact_kind=kind
    )
    config.save(tmp_path)
    attempts = 0

    def child(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            original = Store.set

            def interrupt(self, key, data):
                original(self, key, data)
                if key == "calibration" and data.get("rows"):
                    raise RuntimeError("interrupted after a measurement")

            with monkeypatch.context() as patch:
                patch.setattr(Store, "set", interrupt)
                with pytest.raises(RuntimeError, match="interrupted"):
                    calibrate._calibrate(Config.load(tmp_path), tmp_path, (256,))
            return SimpleNamespace(returncode=1)
        calibrate._calibrate(Config.load(tmp_path), tmp_path, (256,))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(memory, "guarded_run", child)
    with pytest.raises(RuntimeError, match="child failed"):
        calibrate.calibrate(config, tmp_path, (256,))
    assert Store(tmp_path).get("calibration")["rows"]
    assert Config.load(tmp_path).response_tokens == 128
    rows = calibrate.calibrate(config, tmp_path, (256,))
    assert rows[0]["sequence"] < 128
    saved = Config.load(tmp_path)
    assert saved.calibration_available and saved.response_tokens == 128
    assert saved.max_sequence == saved.context_window and 128 in calls
    saved.validate()


@pytest.mark.parametrize(
    "upstream,host,port",
    [("vllm", "0.0.0.0", 8000), ("vllm", "::", 8000), ("mlx", "127.0.0.1", 8081)],
)
def test_existing_server_init_uses_private_endpoint(
    tmp_path, monkeypatch, upstream, host, port
):
    import psutil

    from songe import agents, cli, platform
    from songe.backends import install
    from songe.models import ModelRef, discovery

    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
    (base / "model.safetensors").write_bytes(b"fixture")
    command = (
        ["vllm", "serve", str(base), "--host", host, "--port", str(port)]
        if upstream == "vllm"
        else ["python", "-m", "mlx_lm", "server", "--port", str(port)]
    )
    process = SimpleNamespace(
        info={"pid": 123, "cmdline": command},
        cmdline=lambda: command,
        net_connections=lambda kind: [
            SimpleNamespace(
                status=psutil.CONN_LISTEN, laddr=SimpleNamespace(ip=host, port=port)
            )
        ],
    )
    monkeypatch.setattr(psutil, "process_iter", lambda *args: [process])
    local = memory_platform()
    local.accelerators = lambda: [Accel("cpu", "CPU")]
    local.service = lambda: SimpleNamespace(target=tmp_path / "service")
    monkeypatch.setattr(platform, "current", lambda: local)
    monkeypatch.setattr(
        discovery,
        upstream,
        lambda url: [ModelRef(upstream, "base", "safetensors", base, "llama")],
    )
    monkeypatch.setattr(install, "plan", lambda *args: [])
    monkeypatch.setattr(install, "conversion_plan", lambda *args: [])
    monkeypatch.setattr(agents, "proposed", lambda *args, **kwargs: [])
    monkeypatch.setattr(agents, "apply", lambda *args: None)
    monkeypatch.setattr(cli.shutil, "which", lambda *args: None)
    root = tmp_path / "data"
    args = cli.parser().parse_args(
        [
            "--data-dir",
            str(root),
            "init",
            "--upstream",
            upstream,
            "--upstream-url",
            f"http://127.0.0.1:{port}/v1",
            "--launchd-target",
            str(tmp_path / "service"),
            "--yes",
        ]
    )
    cli.initialize(args, root)
    saved = Config.load(root)
    assert saved.server_owned and saved.upstream_url != f"http://127.0.0.1:{port}/v1"
    assert saved.server_command == (command if upstream == "vllm" else [])
    cli.initialize(args, root)
    assert Config.load(root).upstream_url == saved.upstream_url


async def test_mlx_failed_bind_never_requests_user_endpoint(tmp_path, monkeypatch):
    from songe import memory, platform
    from songe.upstream import mlx

    requests = []
    config = Config(
        upstream="mlx",
        model=str(tmp_path / "base"),
        upstream_url="http://127.0.0.1:1235/v1",
    )
    upstream = mlx.MLX(config, tmp_path)

    async def spawn(*args, **kwargs):
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(memory, "guard", lambda *args, **kwargs: None)
    monkeypatch.setattr(platform, "current", memory_platform)

    async def post(self, url, **kwargs):
        requests.append(url)
        return httpx.Response(200, json={}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    with pytest.raises(RuntimeError, match="MLX"):
        await upstream.load({"identifier": "ours", "artifact": config.model})
    assert requests == [] and upstream.version is None
    assert config.server_owned and config.upstream_url != "http://127.0.0.1:1235/v1"


def test_transient_psi_cannot_poison_persisted_sizing(tmp_path):
    from songe.sleep.sizing import PressureHistory, choose_size

    history = PressureHistory()
    platform = memory_platform(24_000, 22_000)
    history.observe(snapshot(platform))
    platform.memory = lambda: Memory(24_000, 20_000, "warn")
    history.observe(snapshot(platform))
    store = Store(tmp_path)
    store.set("memory_pressure_history", history.data)
    history = PressureHistory(store.get("memory_pressure_history"))
    platform.memory = lambda: Memory(24_000, 18_000)
    for _ in range(4):
        history.observe(snapshot(platform))
    rows = [
        {
            "sequence": 64,
            "response_tokens": 1,
            "token_chunk": 1,
            "accumulation": 1,
            "host_peak_bytes": 1_000,
            "device_peak_bytes": 0,
        }
    ]
    choice = choose_size(
        Config(response_tokens=1), {"rows": rows}, snapshot(platform), history.data
    )
    assert choice and choice["host_bytes"] == 1_000
    assert choice["margin_bytes"] < 17_000


@pytest.mark.parametrize("release", ["park", "close", "alarm"])
@pytest.mark.parametrize("identity", ["replacement-reference", None])
async def test_lms_same_alias_same_weights_replacement_is_preserved(
    tmp_path, monkeypatch, release, identity
):
    from songe.upstream import protection
    from tests.test_lmstudio import FakeLMS

    platform = memory_platform()
    monkeypatch.setattr(protection, "current", lambda: platform)
    upstream = FakeLMS(tmp_path, appear_after=1)
    upstream.config.guard_min_interval = 0.001
    version = {
        "identifier": "alias",
        "artifact": str(tmp_path / "songe/granite/night-1"),
    }
    await upstream.load(version)
    replacement = {
        "identifier": "alias",
        "path": "songe/granite/night-1",
        "memoryBytes": 800,
    }
    if identity:
        replacement["instanceReference"] = identity
    upstream.ps = [replacement]
    if release == "alarm":
        platform.memory = lambda: Memory(100_000, 1, "critical")
        await asyncio.wait_for(upstream.memory_watcher, 1)
    else:
        await getattr(upstream, release)()
    assert not any(c[0] == "unload" for c in upstream.calls)
    assert upstream.ps == [replacement]
    await upstream.close()


@pytest.mark.parametrize("outcome", ["loaded", "cancelled"])
async def test_lms_alarm_waits_for_server_completion_and_recovers_invisible_claim(
    tmp_path, monkeypatch, outcome
):
    from contextlib import asynccontextmanager

    from lmstudio import LMStudioServerError

    from songe.upstream import create, protection
    from songe.upstream.lmstudio import LMStudio

    platform = memory_platform()
    monkeypatch.setattr(protection, "current", lambda: platform)
    config = Config(models_dir=str(tmp_path), guard_min_interval=0.001)
    store = Store(tmp_path / "data")
    upstream = create(config, store.root)
    started, cancel, complete = (asyncio.Event() for _ in range(3))
    unloaded, listing = [], []
    reference = "actual-load-result"

    class Channel:
        async def wait_for_result(self):
            started.set()
            await complete.wait()
            if outcome == "cancelled":
                raise LMStudioServerError("server confirmed cancellation")
            return SimpleNamespace(instance_reference=reference)

        async def cancel(self):
            cancel.set()

    class Client:
        async def __aenter__(self):
            self.llm = self
            return self

        async def __aexit__(self, *args):
            pass

        @asynccontextmanager
        async def _create_channel(self, endpoint):
            # An idle TTL of seconds unloaded the served model between requests.
            assert endpoint.creation_params["ttlMs"] == upstream.lease_seconds * 1000
            assert upstream.lease_seconds >= 3600
            yield Channel()

        async def remote_call(self, endpoint):
            return list(listing)

        async def unload(self, identifier):
            unloaded.append(identifier)
            listing[:] = [m for m in listing if m["identifier"] != identifier]

    monkeypatch.setattr(LMStudio, "sdk", lambda self: Client(), raising=False)

    async def command(self, *args):
        if args[0] == "ps":
            return json.dumps(listing)
        if args[0] == "load":
            started.set()
            try:
                await complete.wait()
            except asyncio.CancelledError:
                cancel.set()
                raise
        return ""

    monkeypatch.setattr(LMStudio, "command", command)

    async def key(*args):
        return "model-key"

    monkeypatch.setattr(upstream, "model_key", key)
    version = {"identifier": "alias", "artifact": str(tmp_path / "pub/model")}
    loading = asyncio.create_task(upstream.load(version))
    await asyncio.wait_for(started.wait(), 1)
    platform.memory = lambda: Memory(100_000, 1, "critical")
    await asyncio.wait_for(cancel.wait(), 1)
    await asyncio.sleep(0.01)
    assert upstream.owned["alias"]["pending"]
    assert not upstream.memory_watcher.done()
    assert store.get("lmstudio_claims")["instances"]["alias"]["pending"]
    with pytest.raises(protection.MemoryPressure):
        await asyncio.wait_for(loading, 1)
    complete.set()  # Server completes despite the cancellation request.
    if outcome == "cancelled":
        await asyncio.wait_for(upstream.memory_watcher, 1)
        assert not store.get("lmstudio_claims")["instances"] and not unloaded
        await upstream.close()
        return
    for _ in range(100):
        if store.get("lmstudio_claims")["instances"]["alias"].get("runtime_instance"):
            break
        await asyncio.sleep(0.001)
    assert (
        store.get("lmstudio_claims")["instances"]["alias"]["runtime_instance"]
        == reference
    )
    # Simulate abrupt parent exit after the durable load result but before ps
    # visibility. A new adapter must keep protecting the exact claim.
    upstream.memory_watcher.cancel()
    await asyncio.gather(upstream.memory_watcher, return_exceptions=True)
    fresh = create(config, store.root)
    platform.memory = lambda: Memory(100_000, 90_000)
    # A separate resident user binding makes recovery allocation free.
    listing.append(
        {"identifier": "user", "path": "pub/user", "instanceReference": "user-ref"}
    )
    await fresh.load({"identifier": "user", "artifact": str(tmp_path / "pub/user")})
    await asyncio.sleep(0.01)
    assert fresh.owned and not fresh.memory_watcher.done() and not unloaded
    listing.append(
        {"identifier": "alias", "path": "pub/model", "instanceReference": reference}
    )
    await asyncio.wait_for(fresh.memory_watcher, 1)
    assert unloaded == ["alias"] and listing[0]["identifier"] == "user"
    assert not store.get("lmstudio_claims")["instances"]
    await fresh.close()


@pytest.mark.parametrize(
    "publication",
    ["complete", "interrupted", "pending", "replacement", "cleanup-error"],
)
def test_production_purge_uses_import_journal_before_removing_store(
    tmp_path, monkeypatch, publication
):
    import hashlib

    from songe import agents, cli, launchd, upstream
    from songe.upstream import protection
    from songe.upstream.ollama import Ollama

    root = tmp_path / "data"
    store = Store(root)
    config = Config(upstream="ollama", upstream_url="http://127.0.0.1:11434/v1")
    config.save(root)
    artifact = root / "versions" / "candidate.gguf"
    artifact.parent.mkdir()
    artifact.write_bytes(b"owned learned model")
    digest = "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
    identifier = f"songe/test:{store.get('installation_id')}-night-1"
    version = {"identifier": identifier, "artifact": str(artifact)}
    tags, running, deleted, unloaded = {}, {}, [], []
    tags["user:latest"] = ("sha256:" + "f" * 64, "user-manifest")

    def handler(request):
        route = request.url.path
        body = (
            json.loads(request.content)
            if request.content and not route.startswith("/api/blobs/")
            else {}
        )
        if route == "/api/ps":
            return httpx.Response(
                200,
                json={"models": [{"name": n, "digest": d} for n, d in running.items()]},
            )
        if route == "/api/tags":
            return httpx.Response(
                200,
                json={"models": [{"name": n, "digest": d[1]} for n, d in tags.items()]},
            )
        if route == "/api/show":
            return (
                httpx.Response(
                    200, json={"modelfile": "FROM /blobs/" + tags[body["model"]][0]}
                )
                if body["model"] in tags
                else httpx.Response(404)
            )
        if route.startswith("/api/blobs/"):
            return httpx.Response(200)
        if route == "/api/create":
            if publication == "pending":
                return httpx.Response(500)
            tags[body["model"]] = (digest, "owned-manifest")
            return (
                httpx.Response(500)
                if publication == "interrupted"
                else httpx.Response(200, json={"status": "success"})
            )
        if route == "/api/generate":
            if body.get("keep_alive") == 0:
                unloaded.append(body["model"])
                running.pop(body["model"], None)
            else:
                running[body["model"]] = tags[body["model"]][1]
            return httpx.Response(200, json={})
        if route == "/api/delete":
            if publication == "cleanup-error":
                return httpx.Response(500)
            deleted.append(body["model"])
            tags.pop(body["model"], None)
            return httpx.Response(200, json={})
        raise AssertionError(route)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(protection, "current", memory_platform)
    adapter = Ollama(config, store=store, transport=transport)
    if publication in {"interrupted", "pending"}:
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(adapter.load(version))
    else:
        asyncio.run(adapter.load(version))
    assert identifier in store.get("runtime_imports")
    if publication == "replacement":
        tags[identifier] = (digest, "replacement-manifest")
        running[identifier] = "replacement-manifest"
    monkeypatch.setattr(
        upstream,
        "create",
        lambda config, root: Ollama(config, store=store, transport=transport),
    )
    monkeypatch.setattr(launchd, "remove", lambda *args, **kwargs: None)
    monkeypatch.setattr(agents, "restore", lambda *args: [])
    monkeypatch.setattr(cli.LocalClient, "control", lambda *args, **kwargs: None)
    result = cli.main(["--data-dir", str(root), "uninstall", "--purge", "--yes"])
    assert "user:latest" in tags and "user:latest" not in deleted + unloaded
    if publication in {"cleanup-error", "pending"}:
        assert result == 1 and root.exists() and artifact.exists()
        assert identifier in Store(root).get("runtime_imports")
        if publication == "pending":
            tags[identifier] = (digest, "owned-manifest")
        publication = "complete"
        assert cli.main(["--data-dir", str(root), "uninstall", "--purge", "--yes"]) == 0
        assert not root.exists() and identifier not in tags
    elif publication == "replacement":
        assert result == 0 and not root.exists()
        assert identifier in tags and identifier not in deleted + unloaded
    else:
        assert result == 0 and not root.exists()
        assert identifier not in tags and identifier not in running
        assert identifier in deleted


def test_production_purge_collects_ready_and_partial_external_candidates(
    tmp_path, monkeypatch
):
    from songe import agents, cli, launchd
    from songe.storage import write_json

    root = tmp_path / "data"
    store = Store(root)
    config = Config(models_dir=str(tmp_path / "library"))
    config.save(root)
    work = root / "versions/night-1"
    work.mkdir(parents=True)
    write_json(
        work / "songe_work.json",
        {"installation_id": store.get("installation_id"), "night": 1},
    )
    ready = tmp_path / "library/songe/model-night-1"
    partial = tmp_path / "library/songe/.model-night-1-partial"
    foreign = tmp_path / "library/songe/user-model"
    for path in (ready, partial, foreign):
        path.mkdir(parents=True)
        (path / "weights").write_bytes(b"fixture")
    ownership = {
        "installation_id": store.get("installation_id"),
        "adapter": str(work / "adapter"),
    }
    write_json(ready / "songe_owned.json", ownership)
    write_json(
        partial / "songe_fusing.json",
        {
            "base": config.model,
            "adapter": str(work / "adapter"),
            "destination": str(ready),
        },
    )
    write_json(foreign / "songe_owned.json", ownership | {"installation_id": "other"})
    store.night(1, status="ready", record={"candidate": {"artifact": str(ready)}})
    monkeypatch.setattr(launchd, "remove", lambda *args, **kwargs: None)
    monkeypatch.setattr(agents, "restore", lambda *args: [])
    monkeypatch.setattr(cli.LocalClient, "control", lambda *args, **kwargs: None)
    assert cli.main(["--data-dir", str(root), "uninstall", "--purge", "--yes"]) == 0
    assert not ready.exists() and not partial.exists() and not root.exists()
    assert (foreign / "weights").read_bytes() == b"fixture"


def test_pressure_boundary_expires_with_sustained_normal_headroom():
    from songe.sleep.sizing import PressureHistory, choose_size

    platform = memory_platform(24_000, 22_000)
    history = PressureHistory()
    history.observe(snapshot(platform))
    platform.memory = lambda: Memory(24_000, 20_000, "warn")
    history.observe(snapshot(platform))
    assert history.margin == 22_000
    platform.memory = lambda: Memory(24_000, 23_000)
    for _ in range(4):
        history = PressureHistory(json.loads(json.dumps(history.data)))
        history.observe(snapshot(platform))
    rows = [
        {
            "sequence": 64,
            "response_tokens": 1,
            "token_chunk": 1,
            "accumulation": 1,
            "host_peak_bytes": 5_000,
            "device_peak_bytes": 0,
        }
    ]
    assert choose_size(
        Config(response_tokens=1), {"rows": rows}, snapshot(platform), history.data
    )


@pytest.mark.parametrize("backend", ["torch", "mlx"])
def test_live_memory_budget_has_no_embedded_capacity_reserve(monkeypatch, backend):
    import sys

    from songe import schedule
    from songe.backends.mlx import Backend
    from songe.backends.torch import needs_4bit

    if backend == "torch":
        assert not needs_4bit(400, 900, Config().memory_fraction)
        return
    limits = []
    mx = SimpleNamespace(
        set_memory_limit=limits.append, random=SimpleNamespace(seed=lambda n: None)
    )

    class StopBeforeLoading(Exception):
        pass

    def load(*args):
        raise StopBeforeLoading

    nn, optimizers = SimpleNamespace(), SimpleNamespace()
    for name, module in {
        "mlx": SimpleNamespace(core=mx, nn=nn, optimizers=optimizers),
        "mlx.core": mx,
        "mlx.nn": nn,
        "mlx.optimizers": optimizers,
        "mlx_lm": SimpleNamespace(load=load),
        "mlx_lm.tuner.utils": SimpleNamespace(linear_to_lora_layers=lambda *a: None),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        schedule,
        "probe",
        lambda: SimpleNamespace(available=True, total_gb=100, free_gb=90),
    )
    with pytest.raises(StopBeforeLoading):
        Backend(Config(memory_fraction=1))
    assert limits == [90 * 10**9]


async def test_guarded_serving_reloads_expired_lms_lease(tmp_path, monkeypatch):
    from songe.schedule import Scheduler
    from songe.upstream import protection
    from tests.test_lmstudio import FakeLMS
    from tests.test_store import version

    monkeypatch.setattr(protection, "current", memory_platform)
    store = Store(tmp_path / "data")
    bound = version("base") | {
        "identifier": "alias",
        "artifact": str(tmp_path / "songe/granite/night-1"),
    }
    store.add_version(bound)
    store.swap("base", None)
    upstream = FakeLMS(tmp_path, appear_after=1)
    await upstream.load(bound)
    upstream.ps.clear()  # Server-side idle TTL expired; the base pointer remains.
    scheduler = Scheduler(Config(), store, upstream)
    await scheduler.wake()
    assert any(m["identifier"] == "alias" for m in upstream.ps)
    assert sum(c[0] == "load" for c in upstream.calls) == 2
    await upstream.close()


def test_purge_recovers_a_converter_crash_before_publication(tmp_path, monkeypatch):
    import subprocess
    import sys

    from songe import agents, cli, launchd

    root = tmp_path / "data"
    store = Store(root)
    config = Config(models_dir=str(tmp_path / "library"))
    config.save(root)
    adapter = root / "versions/night-1/adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(
        '{"peft_type":"LORA"}', encoding="utf-8"
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture")
    destination = tmp_path / "library/songe/candidate.gguf"
    program = """
import os, sys
from pathlib import Path
from types import SimpleNamespace
from songe.artifacts import export

def crash(*args, **kwargs):
    os._exit(9)

export("gguf-lora", Path(sys.argv[1]), Path(sys.argv[1]), Path(sys.argv[2]), None,
       runner=crash, tools=SimpleNamespace(convert_lora="fixture", env=lambda: {}),
       ownership={"installation_id": sys.argv[3], "adapter": sys.argv[1]})
"""
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            str(adapter),
            str(destination),
            store.get("installation_id"),
        ],
        check=False,
        capture_output=True,
    )
    assert child.returncode == 9, child.stderr.decode()
    abandoned = list(destination.parent.glob(".export-*"))
    assert abandoned
    monkeypatch.setattr(launchd, "remove", lambda *args, **kwargs: None)
    monkeypatch.setattr(agents, "restore", lambda *args: [])
    monkeypatch.setattr(cli.LocalClient, "control", lambda *args, **kwargs: None)
    assert cli.main(["--data-dir", str(root), "uninstall", "--purge", "--yes"]) == 0
    assert not root.exists() and not any(path.exists() for path in abandoned)


@pytest.mark.parametrize("replacement", [False, True])
def test_purge_unloads_only_the_journaled_lms_instance(
    tmp_path, monkeypatch, replacement
):
    from songe import agents, cli, launchd, upstream
    from songe.storage import write_json
    from songe.upstream import protection
    from tests.test_lmstudio import FakeLMS

    monkeypatch.setattr(protection, "current", memory_platform)
    root = tmp_path / "data"
    store = Store(root)
    config = Config(models_dir=str(tmp_path / "models"))
    config.save(root)
    artifact = tmp_path / "models/songe/granite/night-1"
    artifact.mkdir(parents=True)
    write_json(
        artifact / "songe_owned.json", {"installation_id": store.get("installation_id")}
    )
    first = FakeLMS(tmp_path / "models", appear_after=1)
    first.store = store
    asyncio.run(first.load({"identifier": "alias", "artifact": str(artifact)}))
    running = list(first.ps)
    if replacement:
        running[0] = running[0] | {"instanceReference": "user-replacement"}
    fresh = FakeLMS(tmp_path / "models", running=running)
    fresh.store = store
    monkeypatch.setattr(upstream, "create", lambda *args: fresh)
    monkeypatch.setattr(launchd, "remove", lambda *args, **kwargs: None)
    monkeypatch.setattr(agents, "restore", lambda *args: [])
    monkeypatch.setattr(cli.LocalClient, "control", lambda *args, **kwargs: None)
    assert cli.main(["--data-dir", str(root), "uninstall", "--purge", "--yes"]) == 0
    assert not artifact.exists() and not root.exists()
    assert [c for c in fresh.calls if c[0] == "unload"] == (
        [] if replacement else [("unload", "alias")]
    )
    if replacement:
        assert fresh.ps == running


async def test_lms_restart_retains_an_unconfirmed_pending_load(tmp_path, monkeypatch):
    from songe.upstream import create, protection
    from songe.upstream.lmstudio import LMStudio

    monkeypatch.setattr(protection, "current", memory_platform)
    store = Store(tmp_path / "data")
    config = Config(models_dir=str(tmp_path), guard_min_interval=0.001)
    pending = {
        "identifier": "alias",
        "artifact": str(tmp_path / "pub/model"),
        "pending": True,
        "lease_seconds": config.guard_grace_seconds,
    }
    store.set(
        "lmstudio_claims",
        {
            "installation_id": store.get("installation_id"),
            "url": config.upstream_url,
            "instances": {"alias": pending},
        },
    )
    listing = [
        {"identifier": "user", "path": "pub/user", "instanceReference": "user-ref"}
    ]
    unloaded = []

    class Client:
        async def __aenter__(self):
            self.llm = self
            return self

        async def __aexit__(self, *args):
            pass

        async def remote_call(self, endpoint):
            return list(listing)

        async def unload(self, identifier):
            unloaded.append(identifier)

    async def command(self, *args):
        return json.dumps(listing)

    monkeypatch.setattr(LMStudio, "sdk", lambda self: Client(), raising=False)
    monkeypatch.setattr(LMStudio, "command", command)
    fresh = create(config, store.root)
    await fresh.load({"identifier": "user", "artifact": str(tmp_path / "pub/user")})
    assert fresh.owned["alias"]["pending"]
    # A late alias is not an immutable load result: it could be a user's
    # replacement. Keep the claim and rely on our requested runtime lease.
    listing.append(
        {
            "identifier": "alias",
            "path": "pub/model",
            "instanceReference": "late-or-replaced",
        }
    )
    await asyncio.sleep(0.01)
    assert not unloaded and not fresh.memory_watcher.done()
    assert store.get("lmstudio_claims")["instances"]["alias"]["pending"]
    await fresh.close()
    assert not fresh.memory_watcher.done()
    # End the watcher through its own exit condition (nothing left to
    # guard). Cancelling it while it awaits asyncio.to_thread hung on
    # Python 3.11 in about half the runs (#82).
    fresh.owned.clear()
    await asyncio.wait_for(fresh.memory_watcher, 5)
