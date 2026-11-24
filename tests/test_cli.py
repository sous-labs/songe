import os
from types import SimpleNamespace

import pytest

from songe.cli import initialize, main
from songe.config import Config
from songe.schedule import Probes


def test_idempotent_init_preserves_custom_name_and_backend(tmp_path, monkeypatch):
    import songe.schedule
    from songe import cli
    from songe.upstream import lmstudio

    model = tmp_path / "models/publisher/model"
    model.mkdir(parents=True)
    (model / "config.json").write_text(
        '{"model_type":"granite", "max_position_embeddings":8192}', encoding="utf-8"
    )
    (model / "model.safetensors").write_bytes(b"fixture")
    root = tmp_path / "data"
    config = Config(
        model=str(model),
        public_model="songe/chosen-name",
        upstream="mlx",
        calibration_available=True,
    )
    config.save(root)
    monkeypatch.setattr(lmstudio, "find_lms", lambda: "/fixture/lms")
    monkeypatch.setattr(lmstudio, "downloads_folder", lambda: tmp_path / "models")
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        songe.schedule, "probe", lambda: Probes(True, 1800, 0, 24, 16, True)
    )
    args = SimpleNamespace(
        upstream=None,
        model=None,
        name=None,
        yes=True,
        make_default=False,
        launchd_target=tmp_path / "launchd.plist",
    )
    initialize(args, root)
    assert Config.load(root).public_model == "songe/chosen-name"
    assert Config.load(root).upstream == "mlx"
    assert args.launchd_target.exists()


def test_empty_dry_run_never_imports_mlx(tmp_path, capsys, monkeypatch):
    import sys

    from songe import memory
    from songe.platform import Memory

    monkeypatch.setattr(
        memory,
        "current",
        lambda: SimpleNamespace(
            name="test",
            memory=lambda: Memory(10**9, 9 * 10**8),
            accelerators=lambda: [],
        ),
    )
    before = set(sys.modules)
    assert main(["--data-dir", str(tmp_path / "data"), "sleep", "--dry-run"]) == 0
    assert '"train": 0' in capsys.readouterr().out
    assert not any(m.startswith("mlx.core") for m in set(sys.modules) - before)


def test_status_reads_service_or_store_without_initializing(
    tmp_path, monkeypatch, capsys
):
    import httpx

    def down(*args, **kwargs):
        raise httpx.ConnectError("stopped")

    monkeypatch.setattr(httpx.Client, "get", down)
    root = tmp_path / "missing"
    assert main(["--data-dir", str(root)]) == 0
    output = capsys.readouterr().out
    # Before init there is nothing to serve; the fix is init.
    assert output.startswith("now     Connect a model · Run songe init\n")
    assert "next    Run songe init in your terminal." in output
    assert "songe serve" not in output
    assert "Tonight" not in output
    assert not root.exists()


def test_training_status_shows_the_workers_values_and_their_sources(
    store, monkeypatch, capsys
):
    import json

    from songe import cli
    from songe.config import TRAINING_KEYS
    from songe.models import registry

    root = store.root
    path = root / "config.toml"
    path.write_text("learning_rate = 6e-5\naccumulation = 8\n", encoding="utf-8")
    installation = Config.load(root)
    registry.register(installation, store, "first")
    registry.register(installation, store, "second")
    store.set("primary_model", "second")
    for model, accumulation in (("first", 3), ("second", 2)):
        row = registry.find(store, model)
        settings = row["config"] | {"accumulation": accumulation}
        if model == "second":
            settings["vocab_clip"] = 0.0  # This key cannot override the installation.
        with store.connect() as db:
            db.execute(
                "UPDATE models SET config=? WHERE id=?", (json.dumps(settings), model)
            )
    expected = Config.for_model(root, "second")
    before = path.read_bytes(), registry.entries(store)

    def unexpected(*args, **kwargs):
        pytest.fail("training status must only read saved settings")

    monkeypatch.setattr(cli, "catch_up", unexpected)
    monkeypatch.setattr(cli.LocalClient, "snapshot", unexpected)

    command = ["--data-dir", str(root), "status", "--training"]
    assert main(command) == 0
    out = capsys.readouterr().out
    assert "Training settings · songe/second" in out
    lines = [line.split() for line in out.splitlines()]
    assert ["learning_rate", "installation", json.dumps(expected.learning_rate)] in lines
    assert ["accumulation", "per-model", str(expected.accumulation)] in lines
    assert ["vocab_clip", "default", str(expected.vocab_clip)] in lines
    assert {line[0] for line in lines if line and line[0] in TRAINING_KEYS} == set(
        TRAINING_KEYS
    )
    assert "Ignored per-model settings · vocab_clip" in out

    assert main(command + ["songe/first"]) == 0
    out = capsys.readouterr().out
    assert "Training settings · songe/first" in out
    assert ["accumulation", "per-model", "3"] in [
        line.split() for line in out.splitlines()
    ]
    assert (path.read_bytes(), registry.entries(store)) == before


def test_doctor_measured_lines_and_no_writes(tmp_path, monkeypatch, capsys):
    import httpx

    from songe import cli

    def down(*args, **kwargs):
        raise httpx.ConnectError("stopped")

    monkeypatch.setattr(httpx.Client, "get", down)
    monkeypatch.setattr(
        cli,
        "measured_hardware",
        lambda: ("linux", 32000, 12000, "cpu fixture, 12000 / 32000 bytes free"),
    )
    monkeypatch.setattr(
        cli.shutil,
        "which",
        lambda name: "/fixture/ollama" if name == "ollama" else None,
    )
    root = tmp_path / "not-created"
    # A stopped service and no model server are failures to fix.
    assert main(["--data-dir", str(root), "doctor"]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 8
    assert [line.split(":")[0] for line in lines] == [
        "Platform",
        "Accelerator",
        "Memory",
        "Upstreams",
        "Agents",
        "Service",
        "Log",
        "Disk",
    ]
    assert lines[6] == f"Log: {root.resolve() / 'songe.log'}"
    assert lines[2] == "Memory: 0.0 of 0.0 GB free"
    assert "1 executables" in lines[3]
    assert "0 configured" in lines[4]
    assert "Service: stopped" in lines[5]
    assert not root.exists()


def test_doctor_checks_the_users_server_not_the_owned_one(
    tmp_path, monkeypatch, capsys
):
    import httpx

    from songe import cli
    from songe.config import Config

    def get(self, url, **kwargs):
        # The Songe-owned llama-server is down outside a Dream (#10).
        if not url.startswith("http://127.0.0.1:8080/"):
            raise httpx.ConnectError("refused")
        return httpx.Response(
            200, json={"data": [{"id": "m"}]}, request=httpx.Request("GET", url)
        )

    monkeypatch.setattr(httpx.Client, "get", get)
    monkeypatch.setattr(cli, "measured_hardware", lambda: ("linux", 1, 1, "cpu"))
    root = tmp_path / "data"
    Config(
        upstream="llamacpp",
        upstream_url="http://127.0.0.1:41001/v1",
        server_source_url="http://127.0.0.1:8080/v1",
        server_owned=True,
    ).save(root)
    main(["--data-dir", str(root), "doctor"])
    upstreams = next(
        line for line in capsys.readouterr().out.splitlines()
        if line.startswith("Upstreams:")
    )
    assert upstreams.endswith("1 models at http://127.0.0.1:8080/v1")


def test_sleep_says_what_the_request_waits_for(tmp_path, monkeypatch, capsys):
    """Live #66: "Learning requested", then nothing for minutes, no word why."""
    from songe import cli

    reply = {"ok": True, "waiting": "waiting for memory · tries again in 2m"}
    monkeypatch.setattr(cli.LocalClient, "control", lambda self, action, **kw: reply)
    assert main(["--data-dir", str(tmp_path / "data"), "sleep"]) == 0
    out = capsys.readouterr().out
    assert "Learning requested · not started yet · waiting for memory · tries again in 2m" in out
    reply = {"ok": True, "waiting": None}
    assert main(["--data-dir", str(tmp_path / "data"), "sleep"]) == 0
    assert "not started yet" not in capsys.readouterr().out


def test_cli_controls_use_local_api(tmp_path, monkeypatch):
    from songe import cli

    actions = []
    monkeypatch.setattr(
        cli.LocalClient,
        "control",
        lambda self, action, **kwargs: actions.append((action, kwargs)),
    )
    for command in (
        ["pause"],
        ["resume"],
        ["undo"],
        ["sleep", "--now"],
        ["forget", "--yes"],
    ):
        assert main(["--data-dir", str(tmp_path / "data"), *command]) == 0
    assert actions == [
        ("pause", {}),
        ("resume", {}),
        ("undo", {}),
        ("sleep_now", {}),
        ("forget", {"confirm": True}),
    ]


def test_tui_command_is_lazy_and_does_not_initialize_store(tmp_path, monkeypatch):
    import songe.tui

    roots = []
    monkeypatch.setattr(songe.tui, "run", roots.append)
    root = tmp_path / "missing"
    assert main(["--data-dir", str(root), "tui"]) == 0
    assert roots == [root]
    assert not root.exists()


def test_doctor_counts_a_single_file_model(tmp_path, monkeypatch, capsys):
    # llama.cpp and Ollama serve one GGUF file, not a folder.
    import httpx

    from songe import cli
    from songe.config import Config

    def down(*args, **kwargs):
        raise httpx.ConnectError("stopped")

    monkeypatch.setattr(httpx.Client, "get", down)
    monkeypatch.setattr(cli, "measured_hardware", lambda: ("linux", 1, 1, "cpu"))
    model = tmp_path / "model.gguf"
    with open(model, "wb") as file:
        file.truncate(300_000_000)  # sparse: st_size without the writes
    root = tmp_path / "data"
    Config(upstream="llamacpp", model=str(model)).save(root)
    main(["--data-dir", str(root), "doctor"])
    disk = next(
        line for line in capsys.readouterr().out.splitlines() if line.startswith("Disk:")
    )
    assert disk.endswith("; model 0.3 GB")


def test_doctor_shows_the_data_disk_when_models_are_elsewhere(tmp_path, monkeypatch, capsys):
    # #69: with models on an external drive, the data folder's disk (where
    # recordings grow) was never shown.
    from pathlib import Path

    import httpx

    from songe import cli
    from songe.artifacts import storage
    from songe.config import Config

    def down(*args, **kwargs):
        raise httpx.ConnectError("stopped")

    monkeypatch.setattr(httpx.Client, "get", down)
    monkeypatch.setattr(cli, "measured_hardware", lambda: ("linux", 1, 1, "cpu"))
    drive = tmp_path / "drive"
    drive.mkdir()
    model = drive / "model.gguf"
    model.write_bytes(b"GGUF")
    root = tmp_path / "data"
    Config(upstream="llamacpp", model=str(model)).save(root)
    monkeypatch.setattr(
        storage, "volume", lambda path: drive if drive in Path(path).resolve().parents else tmp_path
    )
    main(["--data-dir", str(root), "doctor"])
    lines = capsys.readouterr().out.splitlines()
    data = [line for line in lines if line.startswith("Data disk:")]
    assert len(data) == 1 and data[0].endswith(f"GB free at {root}")
    # One disk for both: no second line.
    monkeypatch.setattr(storage, "volume", lambda path: tmp_path)
    main(["--data-dir", str(root), "doctor"])
    assert not any(
        line.startswith("Data disk:") for line in capsys.readouterr().out.splitlines()
    )


def test_doctor_recognizes_agent_v2_and_live_service(tmp_path, monkeypatch, capsys):
    import json

    import httpx

    from songe import cli
    from songe.agents.opencode import configure

    config = Config()
    path = tmp_path / "home/.config/opencode/opencode.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(configure({}, config, v2=True)), encoding="utf-8")
    before = path.read_bytes()
    monkeypatch.setattr(
        cli, "measured_hardware", lambda: ("linux", 32000, 12000, "cpu fixture")
    )
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        cli.LocalClient,
        "snapshot",
        lambda self: {"state": "ready", "service_running": True},
    )

    def models(client, request, **kwargs):
        return httpx.Response(200, json={"data": [{"id": "fixture"}]}, request=request)

    monkeypatch.setattr(httpx.Client, "send", models)
    assert main(["--data-dir", str(tmp_path / "missing"), "doctor"]) == 0
    output = capsys.readouterr().out
    assert "1 models at" in output
    assert "1 configured (opencode)" in output
    assert "Service: running on 127.0.0.1:7613, ready" in output
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "data,expected",
    [
        ({}, "opencode: no songe provider"),
        (
            {"provider": {"songe": {"options": {"baseURL": "http://127.0.0.1:9/v1"}}}},
            "opencode: provider URL differs",
        ),
    ],
)
def test_doctor_tells_a_missing_provider_from_a_wrong_url(
    tmp_path, monkeypatch, capsys, data, expected
):
    import json

    import httpx

    from songe import cli

    path = tmp_path / "home/.config/opencode/opencode.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(cli, "measured_hardware", lambda: ("linux", 1, 1, "cpu"))

    def down(*args, **kwargs):
        raise httpx.ConnectError("stopped")

    monkeypatch.setattr(httpx.Client, "get", down)
    main(["--data-dir", str(tmp_path / "missing"), "doctor"])
    agents = next(
        line for line in capsys.readouterr().out.splitlines() if line.startswith("Agents:")
    )
    assert agents == f"Agents: 0 configured; songe init; {expected}"


def test_status_explains_corrupt_offline_store(tmp_path, monkeypatch, capsys):
    import httpx

    def down(*args, **kwargs):
        raise httpx.ConnectError("stopped")

    monkeypatch.setattr(httpx.Client, "get", down)
    root = tmp_path / "data"
    root.mkdir()
    database = root / "experience.sqlite3"
    database.write_bytes(b"not a database")
    assert main(["--data-dir", str(root)]) == 1
    assert "not a database" in capsys.readouterr().err
    assert database.read_bytes() == b"not a database"


def test_serve_while_running_prints_one_line(tmp_path, monkeypatch, capsys):
    from songe import cli
    from songe.config import Config
    from songe.platform.locking import acquire

    config = Config(model=str(tmp_path / "model"))
    config.save(tmp_path)
    monkeypatch.setenv("SONGE_DATA_DIR", str(tmp_path))
    with (tmp_path / "serve.lock").open("a+") as held:
        acquire(held)
        assert cli.main(["--data-dir", str(tmp_path), "serve"]) in (0, None)
    out = capsys.readouterr().out
    assert "songe running" in out and "Traceback" not in out


def test_doctor_prints_last_error_failures_and_log(tmp_path, monkeypatch, capsys):
    import httpx

    from songe import cli
    from songe.log import Store

    monkeypatch.setattr(
        httpx.Client,
        "get",
        lambda *a, **kw: (_ for _ in ()).throw(httpx.ConnectError("x")),
    )
    monkeypatch.setattr(cli, "measured_hardware", lambda: ("linux", 1, 1, "cpu"))
    root = (tmp_path / "data").resolve()
    store = Store(root)
    store.scoped("qwen:4b").set("child_error", "learning failed · songe doctor")
    store.set("failures", {"count": 3, "at": 1794354000})
    assert main(["--data-dir", str(root), "doctor"]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert "Last error: learning failed · qwen:4b" in lines
    assert any(line.startswith("Failures: 3 · last ") for line in lines)
    assert f"Log: {root / 'songe.log'}" in lines


def test_a_commented_opencode_config_is_skipped_not_fatal(tmp_path, capsys):
    from songe import agents
    from songe.config import Config

    (tmp_path / ".config/opencode").mkdir(parents=True)
    jsonc = tmp_path / ".config/opencode/opencode.jsonc"
    jsonc.write_text('{\n  // mine\n  "model": "x"\n}\n')
    (tmp_path / ".pi/agent").mkdir(parents=True)
    config = Config(exposed_models=["songe/m"], public_model="songe/m")
    changes = agents.proposed(config, home=tmp_path)
    assert [path.name for path, *_ in changes] == ["models.json"]
    assert "Skipped" in capsys.readouterr().err
    assert "// mine" in jsonc.read_text()


def test_init_prefers_a_model_that_can_learn(monkeypatch):
    from songe import cli

    calls = []
    monkeypatch.setattr(
        "songe.models.library.support",
        lambda config, ref, size, platform: (
            calls.append(ref)
            or (("unsupported", "x") if ref == "vision" else ("pending", ""))
        ),
    )
    ranked = sorted(
        ["vision", "text"],
        key=lambda ref: cli.trainable_support(None, ref, None) == "unsupported",
    )
    assert ranked == ["text", "vision"]


def test_null_stream_fields_do_not_lose_the_turn():
    import json

    from songe.proxy import decode_response

    chunks = [
        {"choices": [{"index": 0, "delta": {"content": "hi", "tool_calls": None}}]},
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "ls", "arguments": None},
                            }
                        ]
                    },
                }
            ]
        },
        {"choices": [{"index": 0, "delta": None, "finish_reason": "tool_calls"}]},
    ]
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    decoded = decode_response(body.encode(), True)
    assert "hi" in json.dumps(decoded) and "ls" in json.dumps(decoded)


def test_a_setting_from_a_newer_version_does_not_stop_every_command(tmp_path):
    from songe.config import Config

    Config().save(tmp_path)
    path = tmp_path / "config.toml"
    path.write_text(path.read_text() + "\nfrom_the_future = 1\n")
    assert Config.load(tmp_path).port == Config().port


def test_init_without_a_service_manager_connects_agents(tmp_path, config, monkeypatch, capsys):
    """Linux without a systemd user manager (#12): no unit file, no wait for
    a service that cannot start; agents are connected and the user is told
    to run songe serve."""
    import json as _json
    from types import SimpleNamespace

    from songe import agents, cli, launchd, platform
    from songe.platform.services import Systemd

    def run(argv, **kw):
        raise FileNotFoundError(2, "No such file or directory", "systemctl")

    unit = tmp_path / "systemd/user/songe.service"
    monkeypatch.setattr(
        platform, "current", lambda: SimpleNamespace(service=lambda: Systemd(unit, run=run))
    )
    def install(*args, **kwargs):
        raise AssertionError("no service manager to install into")

    monkeypatch.setattr(launchd, "install", install)
    monkeypatch.setattr(agents.Path, "home", lambda: tmp_path)
    path = tmp_path / ".config/opencode/opencode.json"
    path.parent.mkdir(parents=True)
    path.write_text(_json.dumps({"provider": {}}), encoding="utf-8")
    root = tmp_path / "data"
    root.mkdir()
    config.public_model = "songe/granite"
    changes = agents.proposed(config, root=root)
    args = SimpleNamespace(launchd_target=None, make_default=False, keep_default=False)
    assert cli.service_fact(args) == "none · start it with songe serve"
    cli.finish_initialization(config, args, root, changes)
    assert "songe" in _json.dumps(_json.loads(path.read_text(encoding="utf-8"))["provider"])
    assert "start Songe · no service manager here" in capsys.readouterr().out
    assert not unit.exists()


def test_init_applies_the_reviewed_change_after_the_new_service_edits_opencode(tmp_path, config, monkeypatch):
    """Live (first real install): init started the service, the service
    updated opencode.json (context lengths), and init stopped with "Agent
    configuration changed since review"."""
    import json as _json
    from types import SimpleNamespace

    from songe import agents, cli, launchd

    monkeypatch.setattr(agents.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(launchd, "install", lambda *a, **k: None)
    path = tmp_path / ".config/opencode/opencode.json"
    path.parent.mkdir(parents=True)
    path.write_text(_json.dumps({"provider": {}}), encoding="utf-8")
    root = tmp_path / "data"
    root.mkdir()
    config.public_model = "songe/granite"
    changes = agents.proposed(config, root=root)
    assert any(c[0] == path for c in changes)
    # The newly started service writes the file first.
    path.write_text(_json.dumps({"provider": {}, "theme": "user-or-service"}), encoding="utf-8")
    args = SimpleNamespace(launchd_target=tmp_path / "plist", make_default=False, keep_default=False)
    cli.finish_initialization(config, args, root, changes, lambda: agents.proposed(config, root=root))
    written = _json.loads(path.read_text(encoding="utf-8"))
    assert written["theme"] == "user-or-service" and "songe" in _json.dumps(written["provider"])


def fresh_install(tmp_path, monkeypatch, local):
    """A first `songe init` against fake model servers only (#72)."""
    from songe import agents, cli, launchd
    from songe.models import ModelRef, discovery
    from songe.platform import Accel, Memory

    base = tmp_path / "model"
    base.mkdir()
    (base / "config.json").write_text('{"max_position_embeddings":4096}', encoding="utf-8")
    platform = SimpleNamespace(
        accelerators=lambda: [Accel("cpu", "test")],
        service=lambda: SimpleNamespace(target=tmp_path / "service", available=lambda: True),
        memory=lambda: Memory(2**30, 2**29),
    )
    monkeypatch.setattr("songe.platform.current", lambda: platform)
    monkeypatch.setattr(cli, "detect_agent_binding", lambda: ("", ""))
    refs = [ModelRef("ollama", "chosen:latest", "gguf", base, "test")] if local else []
    monkeypatch.setattr(discovery, "ollama", lambda **kw: refs)
    monkeypatch.setattr(cli, "upstream_model_ids", lambda url: ["one", "two"])
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    monkeypatch.setattr(agents.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(agents, "proposed", lambda *a, **kw: [])
    monkeypatch.setattr(agents, "apply", lambda *a: None)
    monkeypatch.setattr(launchd, "install", lambda *a, **kw: None)
    monkeypatch.setattr(cli.LocalClient, "snapshot", lambda self: {"service_running": True})
    root = tmp_path / "data"
    return root, [
        "--data-dir", str(root), "init", "--upstream", "ollama",
        "--launchd-target", str(tmp_path / "service"),
    ]


@pytest.mark.parametrize("local", [True, False])
@pytest.mark.parametrize("answer", ["no terminal", "n"])
def test_init_creates_nothing_until_the_user_accepts(tmp_path, monkeypatch, capsys, local, answer):
    """Live (#72): init made the data folder and experience.sqlite3, then
    stopped with "Terminal required · use --yes" (or was declined)."""
    import io

    from songe import cli

    root, argv = fresh_install(tmp_path, monkeypatch, local)
    if answer == "no terminal":
        monkeypatch.setattr(cli.sys, "stdin", io.StringIO())
        assert cli.main(argv) == 1
        assert "Terminal required" in capsys.readouterr().err
    else:
        monkeypatch.setattr(cli.sys, "stdin", SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("builtins.input", lambda prompt: answer)
        assert cli.main(argv) == 0
    assert not root.exists()


@pytest.mark.parametrize("local", [True, False])
@pytest.mark.parametrize("how", ["--yes", "y"])
def test_init_creates_its_data_once_accepted(tmp_path, monkeypatch, local, how):
    from songe import cli
    from songe.log import Store

    root, argv = fresh_install(tmp_path, monkeypatch, local)
    if how == "--yes":
        argv.append("--yes")
    else:
        monkeypatch.setattr(cli.sys, "stdin", SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr("builtins.input", lambda prompt: how)
    assert cli.main(argv) == 0
    assert (root / "config.toml").exists() and (root / "experience.sqlite3").exists()
    if os.name != "nt":
        assert root.stat().st_mode & 0o777 == 0o700
    assert Store(root).get("paused") is False
    if local:
        assert Store(root).version()["artifact"] == str(tmp_path / "model")


def test_reinit_declined_leaves_an_existing_install_as_it_was(tmp_path, monkeypatch):
    from songe import cli

    root, argv = fresh_install(tmp_path, monkeypatch, True)
    assert cli.main([*argv, "--yes"]) == 0
    names, saved = sorted(p.name for p in root.iterdir()), (root / "config.toml").read_bytes()
    monkeypatch.setattr(cli.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert cli.main(argv) == 0
    assert sorted(p.name for p in root.iterdir()) == names
    assert (root / "config.toml").read_bytes() == saved


@pytest.mark.parametrize("local", [True, False])
def test_init_never_offers_an_embedding_model_to_agents(tmp_path, monkeypatch, local):
    """Live (#72): LM Studio's embedding model was listed like a chat model."""
    from songe import agents, cli
    from songe.log import Store
    from songe.models import ModelRef, discovery, registry
    from songe.models.names import exposed

    root, argv = fresh_install(tmp_path, monkeypatch, local)
    embed = tmp_path / "nomic-embed-text-v1.5.gguf"
    embed.write_bytes(b"")
    refs = [
        ModelRef("ollama", "chosen:latest", "gguf", tmp_path / "model", "test"),
        ModelRef("ollama", "nomic-embed-text:latest", "gguf", embed, "nomic-bert"),
    ]
    monkeypatch.setattr(discovery, "ollama", lambda **kw: refs if local else [])
    monkeypatch.setattr(
        cli, "upstream_model_ids", lambda url: ["one", "text-embedding-nomic-embed-text-v1.5"]
    )
    offered = []
    monkeypatch.setattr(
        agents, "proposed", lambda config, **kw: offered.append(exposed(config)) or []
    )
    assert cli.main([*argv, "--yes"]) == 0
    assert offered and all(
        not any("embed" in name for name in names) for names in offered
    )
    assert not any("embed" in m["id"] for m in registry.entries(Store(root)))


def test_init_again_offers_a_model_downloaded_since(tmp_path, monkeypatch):
    """#72: a model added to the server after the first init was never
    offered; `songe models` now says to run init again, and that adds it
    without renaming or doubling the models already offered."""
    from songe import agents, cli
    from songe.config import Config
    from songe.log import Store
    from songe.models import ModelRef, discovery, registry
    from songe.models.names import exposed

    root, argv = fresh_install(tmp_path, monkeypatch, True)
    first = ModelRef("ollama", "chosen:latest", "gguf", tmp_path / "model", "test")
    other = tmp_path / "other.gguf"
    other.write_bytes(b"")
    found = [first, ModelRef("ollama", "other:latest", "gguf", other, "test")]
    monkeypatch.setattr(discovery, "ollama", lambda **kw: list(found))
    assert cli.main([*argv, "--yes"]) == 0
    agents.migrate(Config.load(root), root, update_agents=False)  # as the service does
    before = exposed(Config.load(root))
    assert sorted(before) == ["songe/chosen-latest", "songe/other-latest"]
    newer = tmp_path / "newer.gguf"
    newer.write_bytes(b"")
    found.append(ModelRef("ollama", "LFM2.5-24B:latest", "gguf", newer, "lfm2"))
    offered = []
    monkeypatch.setattr(
        agents, "proposed", lambda config, **kw: offered.append(exposed(config)) or []
    )
    assert cli.main([*argv, "--yes"]) == 0
    assert sorted(offered[-1]) == sorted([*before, "songe/LFM2.5-24B-latest"])
    assert sorted(exposed(Config.load(root))) == sorted(offered[-1])
    assert registry.find(Store(root), "LFM2.5-24B:latest")
    assert Config.load(root).base_identifier == "chosen:latest"


def test_init_names_models_the_way_the_agents_show_them(tmp_path, monkeypatch, capsys):
    """#72: opencode lists songe/songe/<name> (provider songe, model
    songe/<name>) while init said "pick a songe/… model"."""
    import json

    from songe import agents, cli

    root, argv = fresh_install(tmp_path, monkeypatch, True)
    opencode = tmp_path / ".config/opencode/opencode.json"
    result = json.dumps({"model": "songe/songe/chosen-latest"}).encode()
    monkeypatch.setattr(
        agents, "proposed", lambda *a, **kw: [(opencode, None, result, "")]
    )
    monkeypatch.setattr(agents, "routing", lambda home=None: {"opencode": False})
    monkeypatch.setattr(cli.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "1 from Ollama, listed under Songe" in out
    assert "opencode → Songe models · default chosen-latest" in out
    assert "songe/…" not in out
    assert cli.next_step(cli.Config()) == (
        "One step left · pick a Songe model in opencode with /models"
    )
