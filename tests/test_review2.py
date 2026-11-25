"""Regressions for the second independent review and live slice sizing."""

import asyncio
import json
import os
from dataclasses import fields
from types import SimpleNamespace

import pytest

import songe.platform as platforms
from songe.config import Config
from songe.memory import snapshot
from songe.platform import Accel, Memory
from songe.platform.services import Launchd, Systemd, TaskScheduler


@pytest.mark.parametrize("service_type", [Launchd, Systemd, TaskScheduler])
def test_service_install_and_remove_require_exact_identity(tmp_path, service_type):
    service = service_type(tmp_path / "service", activate=False)
    first = tmp_path / 'data space "quoted" % $'
    other = tmp_path / 'data space "quoted" % $-other'
    argv = ["python", "-m", "songe", "--data-dir", str(first), "serve"]
    env = {"SONGE_DATA_DIR": str(first), "SONGE_INSTALLATION_ID": "first"}
    service.install(argv, env)
    before = service.target.read_bytes()
    with pytest.raises(RuntimeError, match="another data directory"):
        service.install(
            ["python", "--data-dir", str(other), "serve"],
            {"SONGE_DATA_DIR": str(other), "SONGE_INSTALLATION_ID": "other"},
        )
    assert service.target.read_bytes() == before
    with pytest.raises(RuntimeError, match="another data directory"):
        service.check_owner(
            {"SONGE_DATA_DIR": str(first), "SONGE_INSTALLATION_ID": "wrong"}
        )
    service.check_owner(env)
    service.uninstall()


@pytest.mark.parametrize("service_type", [Launchd, Systemd, TaskScheduler])
def test_legacy_service_root_is_canonical_exact_argument(tmp_path, service_type):
    service = service_type(tmp_path / "service", activate=False)
    root = tmp_path / "space with ü % $"
    service.install(["python", "--data-dir", str(root), "serve"], {})
    service.check_owner({"SONGE_DATA_DIR": str(root / ".." / root.name)})
    with pytest.raises(RuntimeError, match="another data directory"):
        service.check_owner({"SONGE_DATA_DIR": str(root) + "-other"})


@pytest.mark.parametrize("kind", ["cuda", "rocm", "cpu"])
def test_intel_inventory_does_not_poison_training_telemetry(
    tmp_path, monkeypatch, kind
):
    from songe.platform import linux

    vendor = tmp_path / "class/drm/card0/device/vendor"
    vendor.parent.mkdir(parents=True)
    vendor.write_text("0x8086", encoding="utf-8")
    discrete = [] if kind == "cpu" else [Accel(kind, "selected", 1000, 900)]
    monkeypatch.setattr(linux, "discrete_accelerators", lambda: discrete.copy())
    platform = linux.Linux(sysfs=tmp_path)
    monkeypatch.setattr(platform, "memory", lambda: Memory(1000, 900))
    assert {a.kind for a in platform.accelerators()} == {kind}
    assert len(snapshot(platform).vram) == (kind != "cpu")


def test_only_the_selected_accelerator_is_required(config, monkeypatch):
    from songe.memory import TrainingPlatform

    devices = [
        Accel("cuda", "same model", 1000, 0, "GPU-A"),
        Accel("cuda", "same model", 1000, 900, "GPU-B"),
    ]
    platform = SimpleNamespace(
        name="linux", accelerators=lambda: devices, memory=lambda: Memory(1000, 900)
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-B")
    config.torch_options = {"device": "cuda"}
    selected = TrainingPlatform(platform, config)
    assert snapshot(selected).vram == (devices[1],)
    devices.pop()
    with pytest.raises(OSError, match="telemetry"):
        snapshot(selected)


@pytest.mark.parametrize("override", [{}, {"device_name": "llvmpipe"}, {"simulated": 1}])
def test_linux_mlx_uses_selected_unified_gpu_telemetry(monkeypatch, override):
    from songe import memory

    apple = {
        "device_name": "Apple M1 Max (G13C C0)",
        "architecture": "honeykrisp",
        "unified_memory": 1,
    }
    monkeypatch.setattr(
        memory, "linux_mlx_accelerator", memory.linux_mlx_accelerator.__wrapped__
    )
    monkeypatch.setattr(
        memory.subprocess, "check_output", lambda *args, **kwargs: json.dumps(apple | override)
    )
    platform = SimpleNamespace(
        name="linux",
        accelerators=lambda: [Accel("cuda", "unrelated eGPU", 1000, 0)],
        memory=lambda: Memory(2000, 1500, "warn"),
    )
    config = Config(backend="mlx")
    if override:
        with pytest.raises(OSError, match="MLX"):
            memory.TrainingPlatform(platform, config).accelerators()
        return
    selected = memory.TrainingPlatform(platform, config)
    assert [a.name for a in selected.accelerators()] == [apple["device_name"]]
    sample = snapshot(selected)
    assert (sample.available, sample.pressure, sample.vram) == (1500, "warn", ())


@pytest.mark.parametrize("parser", [False, True])
def test_complete_tool_framing_rejects_malformed_extra_call(parser):
    from songe.sleep.evaluate import health

    tokenizer = SimpleNamespace()
    if parser:
        tokenizer.tool_parser = lambda segment, tools: json.loads(segment)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "write",
                "parameters": {
                    "type": "object",
                    "required": ["path"],
                    "properties": {"path": {"type": "string"}},
                },
            },
        }
    ]
    call = '<tool_call>{"name":"write","arguments":{"path":"a"}}</tool_call>'
    assert health(call, tokenizer, tools, True)["tool_valid"] == 1
    for extra in ("<tool_call>broken</tool_call>", "<tool_call>", "</tool_call>"):
        assert health(call + extra, tokenizer, tools, True)["tool_valid"] == 0
    if not parser:
        result = health(
            'Here is some JSON: {"name":"write","arguments":{"path":"a"}}',
            tokenizer,
            tools,
            True,
        )
        assert result["tool_valid"] == result["tool_supported"] == 0


def test_retired_runtime_and_calibration_controls_are_absent():
    from pathlib import Path

    from songe.backends import torch
    from songe.upstream import supervisor

    assert "calibration_concurrency" not in {f.name for f in fields(Config)}
    assert "fla_available" not in Path(torch.__file__).read_text(encoding="utf-8")
    text = Path(supervisor.__file__).read_text(encoding="utf-8")
    assert "VLLM_SERVER_DEV_MODE" not in text
    assert "--enable-sleep-mode" not in text


async def test_lms_reuses_resident_weights_below_artifact_admission(
    tmp_path, monkeypatch
):
    from songe.upstream import protection
    from tests.test_lmstudio import FakeLMS

    path = tmp_path / "pub/model"
    path.parent.mkdir()
    path.write_bytes(b"x" * 100)
    monkeypatch.setattr(
        protection,
        "current",
        lambda: SimpleNamespace(
            name="test", memory=lambda: Memory(1000, 90), accelerators=lambda: []
        ),
    )
    upstream = FakeLMS(tmp_path, running=[{"identifier": "alias", "path": "pub/model"}])
    await upstream.load({"identifier": "alias", "artifact": str(path)})
    assert not upstream.loaded
    assert not any(c[0] == "load" for c in upstream.calls)
    await upstream.close()


async def test_ollama_reuses_verified_resident_below_artifact_size(
    tmp_path, config, monkeypatch
):
    import hashlib

    import httpx

    from songe.upstream import protection
    from songe.upstream.ollama import Ollama

    path = tmp_path / "model.gguf"
    path.write_bytes(b"x" * 100)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        protection,
        "current",
        lambda: SimpleNamespace(
            name="test", memory=lambda: Memory(1000, 90), accelerators=lambda: []
        ),
    )
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "resident:latest"}]})
        assert request.url.path == "/api/show"
        return httpx.Response(200, json={"modelfile": "FROM /blobs/sha256-" + digest})

    upstream = Ollama(config, transport=httpx.MockTransport(handler))
    await upstream.load({"identifier": "resident", "artifact": str(path)})
    assert not upstream.owned
    assert set(calls) == {"/api/ps", "/api/show"}
    await upstream.close()


def test_cli_init_provisions_separate_llama_endpoint_and_reuses_it(
    tmp_path, monkeypatch
):
    from songe import agents, cli
    from songe.backends import install
    from songe.models import ModelRef, discovery
    from songe.upstream import supervisor

    base = tmp_path / "base.gguf"
    base.write_bytes(b"model")
    root = tmp_path / "data"
    user_command = ["llama-server", "--model", str(base), "--port", "8080"]
    platform = SimpleNamespace(
        name="test",
        accelerators=lambda: [Accel("cpu", "cpu")],
        service=lambda: SimpleNamespace(target=tmp_path / "service"),
    )
    monkeypatch.setattr(platforms, "current", lambda: platform)
    monkeypatch.setattr(
        discovery,
        "llamacpp",
        lambda url: [ModelRef("llamacpp", "base", "gguf", base, "llama")],
    )
    monkeypatch.setattr(install, "plan", lambda *a: [])
    monkeypatch.setattr(install, "conversion_plan", lambda *a: [])
    monkeypatch.setattr(
        supervisor,
        "listening_server",
        lambda c: SimpleNamespace(
            cmdline=lambda: user_command,
            terminate=lambda: pytest.fail("stopped user server"),
        ),
    )
    monkeypatch.setattr(agents, "proposed", lambda *a, **kw: [])
    monkeypatch.setattr(agents, "apply", lambda *a: None)
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    args = cli.parser().parse_args(
        [
            "--data-dir",
            str(root),
            "init",
            "--upstream",
            "llamacpp",
            "--upstream-url",
            "http://127.0.0.1:8080/v1",
            "--launchd-target",
            str(tmp_path / "test.plist"),
            "--yes",
        ]
    )
    cli.initialize(args, root)
    saved = Config.load(root)
    assert saved.server_owned and saved.upstream_url != "http://127.0.0.1:8080/v1"
    assert saved.server_command == user_command
    cli.initialize(args, root)
    assert Config.load(root).upstream_url == saved.upstream_url


@pytest.mark.parametrize("release", ["park", "close", "alarm"])
async def test_lms_replacement_after_successful_load_is_never_unloaded(
    tmp_path, monkeypatch, release
):
    import asyncio

    from songe.upstream import protection
    from tests.test_lmstudio import FakeLMS

    platform = SimpleNamespace(
        name="test", memory=lambda: Memory(1000, 900), accelerators=lambda: []
    )
    monkeypatch.setattr(protection, "current", lambda: platform)
    upstream = FakeLMS(tmp_path, appear_after=1)
    upstream.config.guard_min_interval = 0.001
    version = {
        "identifier": "alias",
        "artifact": str(tmp_path / "songe/granite/night-1"),
    }
    await upstream.load(version)
    upstream.ps = [{"identifier": "alias", "path": "user-weights", "memoryBytes": 800}]
    assert await upstream.loaded_bytes() == 0
    if release == "alarm":
        platform.memory = lambda: Memory(1000, 1, "warn")
        await asyncio.wait_for(upstream.memory_watcher, 1)
    else:
        parked = await getattr(upstream, release)()
        if release == "park":
            assert parked == []
    assert not any(c[0] == "unload" for c in upstream.calls)
    assert not upstream.owned
    await upstream.close()


def test_parent_lifetime_is_bound_to_birth_and_live_handle(monkeypatch):
    from songe import child

    events = []

    class Handle:
        birth = "original"

        def alive(self):
            return False  # Windows getppid would still return the old PID.

        def close(self):
            events.append("closed")

    monkeypatch.setattr(child, "open_parent", lambda pid: Handle())
    with child.ParentProcess(123, "original") as parent:
        assert not parent.alive()
    with pytest.raises(ProcessLookupError):
        with child.ParentProcess(123, "reused"):
            pytest.fail("PID reuse was accepted")
    assert events == ["closed", "closed"]


@pytest.mark.skipif(os.name != "nt", reason="Native Windows process-handle regression")
def test_windows_guard_parent_death_stops_worker_and_descendant(tmp_path):
    import subprocess
    import sys
    import time

    heartbeat = tmp_path / "heartbeat"
    leaf = "import sys,time; from pathlib import Path; p=Path(sys.argv[1]);\nwhile True: p.write_text(str(time.monotonic()), encoding='utf-8'); time.sleep(.01)"
    work = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]);\nwhile True: time.sleep(.1)"
    parent_code = "import sys; from songe.memory import guarded_run; from songe.platform import Memory; from types import SimpleNamespace; p=SimpleNamespace(name='test', memory=lambda:Memory(10**9,9*10**8), accelerators=lambda:[]); guarded_run([sys.executable,'-c',sys.argv[1],sys.argv[2],sys.argv[3]],platform=p)"
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_code, work, leaf, str(heartbeat)]
    )
    try:
        deadline = time.monotonic() + 10
        while not heartbeat.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert heartbeat.exists()
        parent.kill()
        parent.wait(timeout=5)
        time.sleep(1)
        last = heartbeat.read_text(encoding="utf-8")
        time.sleep(0.2)
        assert heartbeat.read_text(encoding="utf-8") == last
    finally:
        subprocess.run(
            ["taskkill", "/PID", str(parent.pid), "/T", "/F"], capture_output=True
        )


async def test_fresh_llama_endpoint_owns_park_register_restore_seam(
    tmp_path, config, monkeypatch
):
    import httpx

    from songe.upstream import create, supervisor

    config.upstream = "llamacpp"
    config.upstream_url = "http://127.0.0.1:8080/v1"
    config.server_command = [
        "llama-server",
        "--model",
        "/user/base.gguf",
        "--port",
        "8080",
    ]
    original = config.upstream_url
    supervisor.provision_endpoint(config)
    assert config.upstream_url != original and config.server_owned
    state = {"base": None, "adapters": [], "spawns": 0, "stops": 0}
    owned = create(config, tmp_path)
    owned.supervisor.platform = SimpleNamespace(
        name="test", memory=lambda: Memory(10**9, 9 * 10**8), accelerators=lambda: []
    )
    monkeypatch.setattr(
        supervisor,
        "listening_server",
        lambda c: (
            pytest.fail("inspected user endpoint")
            if c.upstream_url == original
            else None
        ),
    )

    class Process:
        pid = 1
        returncode = None

        async def wait(self):
            self.returncode = 0

    async def spawn(*argv, **kwargs):
        state["base"] = argv[argv.index("--model") + 1]
        state["adapters"] = [
            {"id": i, "path": argv[i + 1], "scale": 0}
            for i, arg in enumerate(argv)
            if arg == "--lora"
        ]
        state["spawns"] += 1
        return Process()

    def terminate(process):
        state["stops"] += 1
        process.returncode = 0

    def handler(request):
        if str(request.url).startswith(original.removesuffix("/v1")):
            raise httpx.ConnectError("user model was unloaded", request=request)
        assert str(request.url).startswith(config.upstream_url.removesuffix("/v1"))
        if state["base"] is None:
            raise httpx.ConnectError("not started", request=request)
        if request.url.path == "/props":
            return httpx.Response(200, json={"model_path": state["base"]})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": config.base_identifier}]})
        if request.url.path == "/lora-adapters":
            if request.method == "POST":
                scales = {a["id"]: a["scale"] for a in json.loads(request.content)}
                for a in state["adapters"]:
                    a["scale"] = scales.get(a["id"], 0)
            return httpx.Response(200, json=state["adapters"])
        return httpx.Response(200)

    client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: client(**(kw | {"transport": transport}))
    )
    monkeypatch.setattr(supervisor.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(supervisor, "terminate", terminate)
    monkeypatch.setattr(supervisor, "guard", lambda *a, **kw: None)
    base = {
        "identifier": config.base_identifier,
        "artifact": "/user/base.gguf",
        "kind": "gguf-full",
    }
    await owned.ensure(base)
    assert owned.owns_server()
    parked = await owned.park()
    candidate = tmp_path / "versions/night-1/scale-1.0-artifact.gguf"
    candidate.parent.mkdir(parents=True)
    candidate.touch()
    await owned.restore(parked, set())
    await owned.load(
        {
            "identifier": "night-1",
            "artifact": str(candidate),
            "base": base["artifact"],
            "kind": "gguf-lora",
        }
    )
    assert state["adapters"][0]["scale"] == 1
    assert state["spawns"] == 2 and state["stops"] == 1
    await owned.close()


def test_adapter_and_full_pipeline_have_distinct_host_device_budgets():
    from songe.sleep.calibrate import derive

    row = dict(
        sequence=256,
        step_seconds=1,
        generation_tps=50,
        forward_seconds=0.1,
        peak_memory_gb=5,
        host_peak_bytes=2_000_000_000,
        device_peak_bytes=5_000_000_000,
    )
    budget = derive([row], 20, device_free_gb=8)
    assert budget["calibrated_host_gb"] == 2
    assert budget["calibrated_device_gb"] == 5
    full = row | {"host_peak_bytes": 22_000_000_000, "device_peak_bytes": 6_000_000_000}
    with pytest.raises(ValueError, match="memory"):
        derive([full], 20, device_free_gb=8)


def test_live_sizing_reserves_ten_percent_not_the_warn_boundary(config):
    from songe.sleep.sizing import PressureHistory, choose_size

    class Platform:
        name = "test"
        allocated = 0

        def memory(self):
            return Memory(
                24_000,
                8_000 - self.allocated,
                "warn" if self.allocated >= 6_000 else "normal",
            )

        def accelerators(self):
            return []

    platform = Platform()
    history = PressureHistory()
    history.observe(snapshot(platform))
    platform.allocated = 5_500
    history.observe(snapshot(platform))
    platform.allocated = 6_000
    history.observe(snapshot(platform))
    platform.allocated = 0
    rows = [
        dict(
            sequence=n,
            response_tokens=r,
            token_chunk=c,
            accumulation=b,
            host_peak_bytes=1_000 + 10 * n + 4 * r + 2 * c + 100 * (b > 1),
            device_peak_bytes=0,
        )
        for n in (64, 256, 512)
        for r in (1, 32)
        for c in (1, 32)
        for b in (1, 4)
    ]
    calibration = {"rows": rows}
    config.max_sequence, config.response_tokens, config.token_chunk = 512, 32, 32
    choice = choose_size(config, calibration, snapshot(platform), history.data)
    # Warn transitions no longer set the margin: 10% of RAM is reserved.
    assert choice and 64 <= choice["max_sequence"] < 512
    assert choice["margin_bytes"] == 2_400
    platform.allocated = choice["host_bytes"]
    assert platform.memory().pressure == "normal"
    platform.allocated = 7_000
    assert choose_size(config, calibration, snapshot(platform), history.data) is None


def test_sizing_preserves_response_and_effective_batch_while_shrinking_chunk(
    config, monkeypatch
):
    # Isolates the boundary logic from the fixed 10% RAM reserve.
    monkeypatch.setattr("songe.sleep.sizing.RAM_RESERVE_FRACTION", 0)
    from songe.sleep.sizing import choose_size

    config.max_sequence, config.response_tokens, config.token_chunk = 256, 32, 32
    rows = [
        dict(
            sequence=256,
            response_tokens=r,
            token_chunk=c,
            accumulation=b,
            host_peak_bytes=1000 + 100 * r + 100 * c + 1000 * (b > 1),
            device_peak_bytes=0,
        )
        for r in (1, 32)
        for c in (1, 32)
        for b in (1, 4)
    ]
    choice = choose_size(
        config,
        {"rows": rows},
        snapshot(
            SimpleNamespace(
                name="test",
                memory=lambda: Memory(24_000, 5_500),
                accelerators=lambda: [],
            )
        ),
        {},
    )
    assert choice["response_tokens"] == 32
    assert choice["token_chunk"] < 32
    assert choice["accumulation"] == config.accumulation


def test_device_pressure_does_not_become_a_host_memory_margin():
    from songe.sleep.sizing import PressureHistory

    platform = SimpleNamespace(
        name="test",
        memory=lambda: Memory(24000, 20000),
        accelerators=lambda: [Accel("cuda", "selected", 8000, 0)],
    )
    history = PressureHistory()
    history.observe(snapshot(platform))
    assert history.margin == 0


def test_calibration_cleanup_preserves_active_and_foreign_work(store):
    from songe.platform.locking import acquire
    from songe.sleep.calibrate import collect_calibration

    own = store.root / "calibration-own"
    foreign = store.root / "calibration-foreign"
    for path, identity in ((own, store.get("installation_id")), (foreign, "other")):
        path.mkdir()
        (path / "songe_calibration.json").write_text(
            json.dumps({"installation_id": identity}), encoding="utf-8"
        )
    with (store.root / "worker.lock").open("a+") as lease:
        acquire(lease)
        collect_calibration(store.root, store)
        assert own.exists()
    collect_calibration(store.root, store)
    assert not own.exists() and foreign.exists()


@pytest.mark.parametrize("legacy", [False, True])
async def test_scheduler_records_live_size_and_explains_vram_refusal(
    config, store, monkeypatch, legacy
):
    from songe import schedule

    # This test supplies a fake selected CUDA device, even in the CPU matrix.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    config.calibration_available = True
    config.calibrated_peak_gb = (
        10  # old doubled budget must not block a fitting adapter.
    )
    config.max_sequence, config.response_tokens = 256, 32
    platform = SimpleNamespace(
        name="test",
        memory=lambda: Memory(24 * 10**9, 20 * 10**9),
        accelerators=lambda: [Accel("cuda", "selected", 8 * 10**9, 8 * 10**9)],
    )
    monkeypatch.setattr(
        schedule, "probe", lambda p: schedule.Probes(True, 1800, 0, 24, 20, True)
    )
    store.set(
        "calibration",
        {
            "rows": [
                dict(
                    sequence=256,
                    response_tokens=32,
                    token_chunk=32,
                    accumulation=4,
                    host_peak_bytes=2 * 10**9,
                    device_peak_bytes=5 * 10**9,
                    step_seconds=1,
                    generation_tps=50,
                )
            ]
        },
    )

    if legacy:
        store.set(
            "calibration",
            {
                "rows": [
                    dict(
                        sequence=256,
                        peak_memory_gb=5,
                        step_seconds=1,
                        generation_tps=50,
                    )
                ]
            },
        )

    async def park(on_snapshot):
        on_snapshot([])

    async def restore(*a):
        pass

    upstream = SimpleNamespace(park=park, restore=restore)
    scheduler = schedule.Scheduler(config, store, upstream, platform)
    scheduler.last_traffic -= 2000
    monkeypatch.setattr(scheduler, "enough", lambda: asyncio.sleep(0, result=True))
    launched = []

    async def spawn(*args, **kwargs):
        launched.append(args)
        return SimpleNamespace(pid=1, returncode=0)

    monkeypatch.setattr(schedule.asyncio, "create_subprocess_exec", spawn)
    guard_calls = []
    monkeypatch.setattr(schedule, "guard", lambda *a, **kw: guard_calls.append(kw))
    platform.keep_awake = lambda pid: None
    await scheduler.tick()
    if legacy:
        assert launched and "songe.sleep.calibrate" in launched[0]
        assert scheduler.calibrating
        await scheduler.guard_task
        assert guard_calls[0]["predicted_peak_bytes"] is None
        assert guard_calls[0]["baseline"].paging_noise
        scheduler.process = None
        return
    assert launched and "--sizing" in launched[0]
    assert store.get("last_slice_size")["device_bytes"] == 5 * 10**9
    await scheduler.guard_task
    assert guard_calls[0]["predicted_peak_bytes"] == 2 * 10**9
    assert guard_calls[0]["predicted_device_bytes"] == 5 * 10**9
    assert guard_calls[0]["prediction_error_bytes"] == (
        guard_calls[0]["predicted_peak_bytes"] // 20
    )
    scheduler.process = None
    platform.accelerators = lambda: [Accel("cuda", "selected", 8 * 10**9, 4 * 10**9)]
    store.set("last_attempt", 0)
    await scheduler.tick()
    assert "device memory" in scheduler.reason
    assert len(launched) == 1


@pytest.mark.parametrize("failed_publication", [False, True])
async def test_ollama_cleanup_uses_exact_durable_imports(
    tmp_path, config, monkeypatch, failed_publication
):
    import hashlib

    import httpx

    from scripts.e2e_linux import cleanup_ollama
    from songe.log import Store
    from songe.upstream import create, protection

    store = Store(tmp_path / "data")
    config.upstream = "ollama"
    config.upstream_url = "http://127.0.0.1:11434/v1"
    config.public_model = "songe/test-owned"
    config.base_identifier = config.public_model + ":base"
    path = store.root / "candidate.gguf"
    path.write_bytes(b"test weights")
    digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    identifier = f"{config.public_model}:{store.get('installation_id')}-night-1"
    version = {"identifier": identifier, "artifact": str(path), "id": "night-1"}
    tags, running, deleted = {"user:latest"}, set(), []

    def handler(request):
        body = (
            json.loads(request.content)
            if request.content and not request.url.path.startswith("/api/blobs")
            else {}
        )
        route = request.url.path
        if route == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": n} for n in running]})
        if route == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": n} for n in tags]})
        if route == "/api/show":
            return (
                httpx.Response(200, json={"modelfile": "FROM /blobs/" + digest})
                if body["model"] in tags
                else httpx.Response(404)
            )
        if route.startswith("/api/blobs"):
            return httpx.Response(200)
        if route == "/api/create":
            tags.add(body["model"])
            return httpx.Response(200, json={"status": "success"})
        if route == "/api/generate":
            if body.get("keep_alive") == 0:
                running.discard(body["model"])
            else:
                running.add(body["model"])
                if failed_publication:
                    return httpx.Response(500)
            return httpx.Response(200, json={})
        if route == "/api/delete":
            tags.discard(body["model"])
            running.discard(body["model"])
            deleted.append(body["model"])
            return httpx.Response(200, json={})
        raise AssertionError(route)

    monkeypatch.setattr(
        protection,
        "current",
        lambda: SimpleNamespace(
            name="test", memory=lambda: Memory(1000, 900), accelerators=lambda: []
        ),
    )
    upstream = create(config, store.root)
    upstream.transport = httpx.MockTransport(handler)
    if failed_publication:
        with pytest.raises(httpx.HTTPStatusError):
            await upstream.load(version)
    else:
        await upstream.load(version)
    assert identifier in tags
    # Simulate a restart: only the durable import journal knows a partial tag.
    fresh = create(config, store.root)
    fresh.transport = upstream.transport
    await upstream.close()
    await cleanup_ollama(fresh, config, store)
    assert identifier in deleted
    assert tags == {"user:latest"} and not running
    assert not getattr(fresh, "owned", {})


@pytest.mark.parametrize("kind", ["gguf-lora", "gguf-full"])
@pytest.mark.parametrize("pressure", ["normal", "warn"])
def test_real_calibration_orchestration_measures_actual_pipeline(
    tmp_path, config, monkeypatch, kind, pressure
):
    from songe import backends
    from songe.sleep import calibrate, fuse, measurement
    from tests.test_learning import Tokenizer

    state = {"host": 100, "device": 0, "exports": 0, "evaluations": 0}
    platform = SimpleNamespace(
        name="test",
        memory=lambda: Memory(100_000, 90_000 - state["host"], pressure),
        accelerators=lambda: [
            Accel("cuda", "selected", 50_000, 49_000 - state["device"])
        ],
        process_bytes=lambda pid: state["host"],
        on_ac_power=lambda: True,
        idle_seconds=lambda: None,
        thermal=lambda: None,
    )
    monkeypatch.setattr(platforms, "current", lambda: platform)
    monkeypatch.setattr(measurement, "process_tree_bytes", lambda p, pid: state["host"])
    # Reproduce the coarse Windows monotonic clock without weakening admission.
    monkeypatch.setattr(calibrate.time, "monotonic", lambda: 0)
    calls = []

    class Backend:
        def __init__(self, config, check, train=True):
            if train:
                assert (
                    config.response_tokens
                    == config.token_chunk
                    == config.accumulation
                    == 1
                )
            self.config, self.tokenizer = config, Tokenizer()
            state["host"] = 500 if train else 4_000
            state["device"] = 5_000 if train else 7_000
            if not train:
                state["evaluations"] += 1

        def prepare(self):
            pass

        def reset_peak_memory(self):
            pass

        def peak_memory(self):
            return state["device"]

        def sample(self, prompt, seed=None):
            return [1], "x", SimpleNamespace(generation_tps=50, prompt_tps=100)

        def compare(self, *args):
            pass

        def train(self, pool, anchors, work, progress):
            calls.append(
                (
                    self.config.max_sequence,
                    self.config.response_tokens,
                    self.config.token_chunk,
                    self.config.accumulation,
                )
            )
            state["host"] = 500 + sum(calls[-1])
            state["device"] = 5_000

        def save(self, path):
            path.mkdir(parents=True, exist_ok=True)

        def kl_from(self, *args, **kw):
            pass

    monkeypatch.setattr(backends, "choose", lambda *args: Backend)

    def export(*args):
        state["exports"] += 1
        state["host"] = 8_000

    monkeypatch.setattr(fuse, "export_candidate", export)
    monkeypatch.setattr(fuse, "evaluation_model", lambda *a: config.model)
    config.artifact_kind = kind
    config.upstream = "llamacpp" if kind == "gguf-lora" else "ollama"
    config.max_sequence, config.response_tokens, config.accumulation = 256, 32, 4
    rows = calibrate._calibrate(config, tmp_path, (256,))
    assert calls and state["exports"] == 1
    assert calls[0] == (min(c[0] for c in calls), 1, 1, 1)
    assert all(r["step_seconds"] > 0 and r["forward_seconds"] > 0 for r in rows)
    assert bool(state["evaluations"]) == (kind == "gguf-full")
    assert all(
        r["host_peak_bytes"] >= 8_000 and r["device_peak_bytes"] < 10_000 for r in rows
    )
    assert all("export" in r["phase_peaks"] for r in rows)
    assert {c[1] for c in calls} >= {1, 32}
    assert {c[2] for c in calls} >= {1, 32}
    assert {c[3] for c in calls} == {1, 4}
