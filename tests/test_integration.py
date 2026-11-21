import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from songe.backends import install
from songe.config import Config
from songe.log import Store
from songe.platform import Accel
from songe.schedule import Scheduler, rebuild_reason
from songe.sleep.evaluate import health
from songe.sleep.prompts import paired_prompts
from songe.sleep.swap import install as swap
from tests.test_learning import Tokenizer
from tests.test_store import version


def test_config_roundtrip_backend_tables(tmp_path):
    original = Config(
        upstream="ollama",
        torch_options={"device": "cuda", "force_cpu": False},
        model_ref={"path": "/base.gguf", "format": "gguf"},
    )
    original.save(tmp_path)
    assert Config.load(tmp_path) == original
    assert "[torch_options]" in (tmp_path / "config.toml").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "kind,runtime,expected",
    [
        ("cuda", (12, 8), "cu128"),
        ("cuda", (13, 0), "cu130"),
        ("rocm", (7, 1), "rocm7.1"),
    ],
)
def test_install_index_tracks_detected_driver(monkeypatch, kind, runtime, expected):
    monkeypatch.setattr(
        httpx.Client,
        "get",
        lambda *a, **k: httpx.Response(
            200,
            text=">cu126< >cu128< >cu130< >rocm7.0< >rocm7.1<",
            request=httpx.Request("GET", install.WHEEL_ROOT),
        ),
    )
    assert install.wheel_index(kind, runtime).endswith("/" + expected)
    with pytest.raises(ValueError, match="matches"):
        install.wheel_index("rocm", (6, 0))


def test_install_private_environment_plan_and_no_metal_or_cpu_installs(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(install, "stack_ready", lambda kind: False)
    monkeypatch.setattr(
        install, "wheel_index", lambda kind: install.WHEEL_ROOT + "/cu128"
    )
    monkeypatch.setattr(install, "package_requirement", lambda: "songe[torch]==0.1.0")
    platform = SimpleNamespace(accelerators=lambda: [Accel("cuda", "test")])
    commands = install.plan(platform, tmp_path)
    assert all(
        command[0] == str(install.runtime_python(tmp_path)) for command in commands
    )
    assert commands[0][-1].endswith("/cu128")
    assert commands[1][-1] == "songe[torch]==0.1.0"
    for kind in ("metal", "cpu"):
        platform.accelerators = lambda kind=kind: [Accel(kind, "test")]
        assert install.plan(platform, tmp_path) == []
    assert capsys.readouterr().out == ""


def test_alternating_template_attaches_hindsight_without_changing_context(config):
    class Strict(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            roles = [m["role"] for m in messages if m["role"] != "system"]
            if any(a == b for a, b in zip(roles, roles[1:], strict=False)):
                raise ValueError("Roles must alternate")
            return super().apply_chat_template(messages, **kwargs)

    step = {
        "context": [{"role": "user", "content": "Question"}],
        "hindsight": [{"role": "user", "content": "Correct it"}],
    }
    student, teacher = paired_prompts(Strict(), step, config)
    s, t = json.loads(bytes(student)), json.loads(bytes(teacher))
    assert s["messages"] == step["context"]
    assert len(t["messages"]) == 1
    assert t["messages"][0]["content"].startswith("Question\n\nHindsight")
    assert "<observations>" in t["messages"][0]["content"]


def test_forget_removes_v2_evidence_and_preserves_retained_training_artifacts(
    store, config
):
    from songe.api import forget_experience

    work = store.root / "versions/night-1"
    adapter = work / "scale-1.0-adapter"
    adapter.mkdir(parents=True)
    (adapter / "weights").write_text("retained", encoding="utf-8")
    (work / "snapshot.json").write_text("private evidence", encoding="utf-8")
    store.add_version(
        version("night-1", 1)
        | {"adapter": str(adapter), "training_adapter": str(adapter)}
    )
    store.swap("night-1", None)
    store.add_pool(
        [
            {
                "id": "secret",
                "surprise": 1,
                "conversation": "conversation",
                "evidence": ["secret"],
                "answer_hint": {"content": "secret"},
            }
        ]
    )
    forget_experience(store)
    assert not store.active_steps() and not store.pool()
    assert not (work / "snapshot.json").exists() and adapter.exists()
    assert rebuild_reason(config, store) == "forget requested"
    with store.connect() as db:
        assert not db.execute("SELECT * FROM scoreboard").fetchall()


def test_old_schedule_setting_is_migrated_to_the_scheduler(config, store):
    store.set("interface_settings", {"schedule_default_seconds": 42})
    Scheduler(config, store, SimpleNamespace())
    assert config.idle_default_seconds == 42
    assert Config.load(store.root).idle_default_seconds == 42
    assert store.get("interface_settings") == {}


def test_hf_tool_health_uses_read_only_schema_validation():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "parameters": {
                    "type": "object",
                    "required": ["path"],
                    "properties": {"path": {"type": "string"}},
                },
            },
        }
    ]
    assert (
        health(
            '<tool_call>{"name":"read","arguments":{"path":"x"}}</tool_call>',
            object(),
            tools,
            True,
        )["tool_valid"]
        == 1
    )
    assert (
        health(
            '<tool_call>{"name":"read","arguments":{}}</tool_call>',
            object(),
            tools,
            True,
        )["tool_valid"]
        == 0
    )


@pytest.mark.parametrize("name", ["ollama", "vllm", "llamacpp"])
async def test_swap_routes_identifiers_and_rolls_back_global_load_on_failure(
    name, config, store, monkeypatch
):
    incumbent, candidate = version("old", 1), version("new", 2)
    store.add_version(incumbent)
    store.swap("old", None)
    loaded = []

    async def load(version):
        loaded.append(version["id"])

    async def failed(client, url, *, json):
        assert json["model"] == candidate["identifier"]
        return httpx.Response(500, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", failed)
    with pytest.raises(httpx.HTTPStatusError):
        await swap(
            store,
            SimpleNamespace(load=load),
            replace(config, upstream=name),
            candidate,
            "old",
        )
    assert loaded == ["new", "old"]
    assert store.get("current") == "old"


@pytest.mark.parametrize("contents", [b"", b"existing lease"])
def test_worker_lease_is_exclusive_across_processes(tmp_path, contents):
    import subprocess
    import sys

    from songe.platform.locking import acquire

    path = tmp_path / "worker.lock"
    path.write_bytes(contents)
    with path.open("a+") as lease:
        acquire(lease)
        code = "from pathlib import Path; from songe.platform.locking import acquire; import sys; f=Path(sys.argv[1]).open('a+'); acquire(f)"
        result = subprocess.run(
            [sys.executable, "-c", code, str(path)], capture_output=True
        )
        assert result.returncode != 0 and b"BlockingIOError" in result.stderr
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)], capture_output=True
    )
    assert result.returncode == 0
    assert path.read_bytes() == contents


@pytest.mark.parametrize("fail_service", [False, True])
def test_init_selects_agents_upstream_and_asks_once(
    tmp_path, monkeypatch, fail_service
):
    from songe import agents, cli, launchd
    from songe.models import ModelRef, discovery
    from songe.platform import Memory

    base = tmp_path / "model"
    base.mkdir()
    (base / "config.json").write_text(
        '{"max_position_embeddings":4096}', encoding="utf-8"
    )
    platform = SimpleNamespace(
        accelerators=lambda: [Accel("cpu", "test")],
        service=lambda: SimpleNamespace(
            target=tmp_path / "service", available=lambda: True
        ),
        memory=lambda: Memory(2**30, 2**29),
    )
    import songe.platform

    monkeypatch.setattr(songe.platform, "current", lambda: platform)
    monkeypatch.setattr(
        cli,
        "detect_agent_binding",
        lambda: ("chosen:latest", "http://127.0.0.1:11439/v1"),
    )
    monkeypatch.setattr(discovery, "identify", lambda url: "ollama")
    refs = [
        ModelRef("ollama", "other:latest", "gguf", base, "test"),
        ModelRef("ollama", "chosen:latest", "gguf", base, "test"),
    ]
    monkeypatch.setattr(discovery, "ollama", lambda **kw: refs)
    monkeypatch.setattr(discovery, "lmstudio", lambda **kw: [])
    monkeypatch.setattr(discovery, "mlx", lambda url: [])
    monkeypatch.setattr(discovery, "llamacpp", lambda url: [])
    monkeypatch.setattr(discovery, "vllm", lambda url: [])
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    approvals, applied = [], []
    monkeypatch.setattr(
        cli, "confirm", lambda prompt, yes: approvals.append(prompt) or True
    )
    monkeypatch.setattr(agents, "proposed", lambda *a, **kw: [])
    monkeypatch.setattr(agents, "apply", lambda *args: applied.append("agents"))

    def install_service(*args, **kw):
        applied.append("service")
        if fail_service:
            raise RuntimeError("Service installation failed")

    monkeypatch.setattr(launchd, "install", install_service)
    monkeypatch.setattr(
        cli.LocalClient, "snapshot", lambda self: {"service_running": True}
    )
    root = tmp_path / "data"
    args = cli.parser().parse_args(["--data-dir", str(root), "init"])
    if fail_service:
        with pytest.raises(RuntimeError, match="Service installation failed"):
            cli.initialize(args, root)
        assert applied == ["service"]
        return
    cli.initialize(args, root)
    config = Config.load(root)
    assert config.upstream == "ollama" and config.base_identifier == "chosen:latest"
    assert config.upstream_url == "http://127.0.0.1:11439/v1"
    assert config.artifact_kind == "gguf-full"
    assert not config.calibration_available
    assert len(approvals) == 1 and applied == ["service", "agents"]
    assert Store(root).version()["artifact"] == str(base)


async def test_supervisor_registers_candidate_slots_and_vllm_runtime_controls(
    tmp_path, config, monkeypatch
):
    from songe.platform import Memory
    from songe.upstream import protection, supervisor

    monkeypatch.setattr(
        protection,
        "current",
        lambda: SimpleNamespace(
            name="test",
            memory=lambda: Memory(10**9, 9 * 10**8),
            accelerators=lambda: [],
        ),
    )
    monkeypatch.setattr(supervisor, "current", protection.current)
    launched = []
    monkeypatch.setattr(supervisor, "listening_server", lambda config: None)

    async def spawn(*args, **kwargs):
        launched.append((args, kwargs))
        return SimpleNamespace(pid=123, returncode=None)

    async def stopped():
        pass

    async def healthy(client, url):
        return httpx.Response(200, request=httpx.Request("GET", url))

    monkeypatch.setattr(supervisor.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(supervisor, "guard", lambda *args, **kwargs: None)
    monkeypatch.setattr(httpx.AsyncClient, "get", healthy)
    config.server_command = ["llama-server", "--model", "/old", "--ctx-size", "4096"]
    config.upstream = "llamacpp"
    candidate = tmp_path / "versions/night-1/scale-1.0-artifact.gguf"
    candidate.parent.mkdir(parents=True)
    candidate.touch()
    llama = supervisor.LlamaServer(config, tmp_path)
    monkeypatch.setattr(llama, "stop", stopped)
    await llama.restart({"artifact": "/base.gguf", "kind": "gguf-full"})
    args, kwargs = launched[-1]
    assert args[args.index("--model") + 1] == "/base.gguf"
    assert args[args.index("--lora") + 1] == str(candidate)
    assert "--lora-init-without-apply" in args
    assert args[args.index("--alias") + 1] == config.base_identifier
    await llama.watcher
    config.server_command = ["vllm", "serve", "/base"]
    config.upstream = "vllm"
    vllm = supervisor.VLLMServer(config, tmp_path)
    monkeypatch.setattr(vllm, "stop", stopped)
    await vllm.restart({"artifact": "/base"})
    args, kwargs = launched[-1]
    assert "--enable-lora" in args and "--enable-sleep-mode" not in args
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert kwargs["env"]["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] == "True"
    assert "VLLM_SERVER_DEV_MODE" not in kwargs["env"]
    await vllm.watcher


def test_mlx_to_peft_translation_preserves_effective_delta(tmp_path, monkeypatch):
    import sys
    from types import ModuleType

    import numpy as np

    from songe.artifacts.worker import mlx_peft

    a = np.arange(12, dtype=np.float32).reshape(6, 2)
    b = np.arange(8, dtype=np.float32).reshape(2, 4)
    saved = {}
    mx = ModuleType("mlx.core")
    mx.load = lambda path: {
        "model.layers.0.self_attn.q_proj.lora_a": a,
        "model.layers.0.self_attn.q_proj.lora_b": b,
    }
    mx.save_safetensors = lambda path, weights, metadata: saved.update(weights)
    mlx = ModuleType("mlx")
    mlx.core = mx
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"lora_parameters": {"rank": 2, "scale": 0.5}}), encoding="utf-8"
    )
    output = tmp_path / "peft"
    mlx_peft(tmp_path / "base", adapter, output)
    prefix = "base_model.model.model.layers.0.self_attn.q_proj"
    metadata = json.loads((output / "adapter_config.json").read_text(encoding="utf-8"))
    np.testing.assert_array_equal(
        saved[prefix + ".lora_B.weight"]
        @ saved[prefix + ".lora_A.weight"]
        * metadata["lora_alpha"]
        / metadata["r"],
        (a @ b).T * 0.5,
    )
    assert metadata["target_modules"] == ["q_proj"]


def test_metal_native_model_needs_no_dependency_install(tmp_path):
    from songe.models import ModelRef

    platform = SimpleNamespace(accelerators=lambda: [Accel("metal", "test")])
    assert install.plan(platform, tmp_path) == []
    assert (
        install.conversion_plan(
            ModelRef("lmstudio", "local", "mlx", tmp_path, "test"),
            "mlx-fused",
            platform,
            tmp_path,
        )
        == []
    )


def test_install_rejects_an_unusable_accelerator_without_recursive_init(
    tmp_path, monkeypatch
):
    created, calls = [], []
    monkeypatch.setattr(
        install.venv, "EnvBuilder", lambda **kw: SimpleNamespace(create=created.append)
    )

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1 if "-c" in command else 0)

    monkeypatch.setattr(install.subprocess, "run", run)
    commands = [
        [
            str(install.runtime_python(tmp_path)),
            "-m",
            "pip",
            "install",
            "torch",
            "--index-url",
            install.WHEEL_ROOT + "/cu128",
        ]
    ]
    with pytest.raises(RuntimeError, match="cannot use the cuda accelerator"):
        install.install(commands, tmp_path)
    assert created == [tmp_path / "runtime"]
    assert len(calls) == 2 and "-c" in calls[-1]


def test_forget_deletes_converted_gate_weights_and_keeps_original_base(tmp_path):
    from songe.models.convert import forget_derived
    from songe.storage import write_json

    base = tmp_path / "models/base"
    exported = tmp_path / "versions/night-1/scale-1.0-artifact.gguf"
    exported.parent.mkdir(parents=True)
    exported.touch()
    derived = tmp_path / "models/derived"
    dependent = tmp_path / "models/dependent"
    for path, source in (
        (base, tmp_path / "original.gguf"),
        (derived, exported),
        (dependent, derived),
    ):
        path.mkdir(parents=True)
        write_json(
            path / "songe_conversion.json", {"source": str(source), "kind": "gguf"}
        )
    forget_derived(tmp_path, {base})
    assert base.exists() and not derived.exists() and not dependent.exists()
