from types import SimpleNamespace

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
