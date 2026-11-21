"""Plain terminal snapshots and TTY policy, with fake services/runtimes only."""

import copy
import io
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from songe import cli, terminal
from songe.learning_summary import evidence_summary

SNAPSHOTS = Path(__file__).parent / "snapshots"
NOW = datetime(2026, 11, 10, 15).timestamp()


def state():
    return {
        "model": "Following your models",
        "service_running": True,
        "idle_seconds": 720,
        "reason": "proxy idle 8m/10m",
        "next_action": "idle 8m/10m → train Qwen3.5 4B",
        "models": [
            {
                "id": "Qwen3.5 4B",
                "learning_level": "new",
                "support": "trainable",
                "pending": 38,
                "last_update": datetime(2026, 11, 10, 14, 2).timestamp(),
                "last_status": "accepted",
                "learning": {
                    "conversations": 14,
                    "since": datetime(2026, 11, 10, 9, 12).timestamp(),
                    "signals": {
                        "tool_error_fixed": 5,
                        "user_correction": 4,
                        "retry_success": 3,
                    },
                    "topics": ["pytest", "git"],
                    "sources": [],
                    "last": {
                        "created": datetime(2026, 11, 10, 14, 2).timestamp(),
                        "repeat_errors_change": -0.18,
                        "drift": 0.002,
                        "status": "swapped",
                    },
                },
            },
            {
                "id": "Granite 4.2 3B",
                "learning_level": "off",
                "support": "trainable",
                "pending": 0,
            },
            {
                "id": "LFM3 1.7B",
                "learning_level": "new",
                "support": "unsupported",
                "pending": 0,
                "reason": "unsupported",
            },
        ],
    }


def assert_snapshot(name, output):
    assert "\x1b" not in output and "\r" not in output
    plain = "\n".join(line.rstrip() for line in output.splitlines()) + "\n"
    if os.environ.get("SONGE_UPDATE_SNAPSHOTS"):
        (SNAPSHOTS / f"cli_{name}.txt").write_text(plain, encoding="utf-8")
    assert plain == (SNAPSHOTS / f"cli_{name}.txt").read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def plain(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")


@pytest.mark.parametrize("learning", [False, True])
def test_status_snapshot(learning, capsys):
    data = state()
    if learning:
        data.update(
            training_model="Qwen3.5 4B",
            training={
                "phase": "distill",
                "step": 41,
                "total_steps": 120,
                "data_progress": {
                    "completed": 82,
                    "total": 240,
                    "unit": "samples",
                    "eta_seconds": 180,
                },
            },
        )
        data["next_action"] = "check Qwen3.5 4B · current work completes"
    cli.print_status(data)
    assert_snapshot("learning" if learning else "idle", capsys.readouterr().out)


def test_models_command_snapshot_and_order(tmp_path, monkeypatch, capsys):
    actions = []
    data = state()
    data["models"][0]["learning_level"] = "all"
    monkeypatch.setattr(cli.LocalClient, "snapshot", lambda self: copy.deepcopy(data))
    monkeypatch.setattr(
        cli.LocalClient,
        "control",
        lambda self, action, **kw: actions.append((action, kw)),
    )
    assert (
        cli.main(["--data-dir", str(tmp_path / "data"), "models", "Qwen3.5 4B", "on"])
        == 0
    )
    assert actions == [("auto_dream", {"model": "Qwen3.5 4B", "on": True})]
    assert_snapshot("models", capsys.readouterr().out)
    assert not (tmp_path / "data").exists()


def test_status_library_reasons_and_empty_learning_snapshot(capsys):
    data = {
        "activity": {
            "now": "waiting · calibration paused · memory · retry when idle",
            "next": "idle 30s/10m → calibrate Qwen3.5-4B",
            "paused": "no",
        },
        "models": [
            {
                "id": "qwen",
                "name": "Qwen3.5-4B",
                "state": "downloaded",
                "size_bytes": 3.1e9,
                "format": "mlx",
                "support": "pending",
                "reason": "calibration paused · memory · retry when idle",
                "learning_level": "new",
                "learning": evidence_summary([]),
            },
            {
                "id": "lfm",
                "name": "LFM3-1.7B",
                "state": "downloaded",
                "size_bytes": 2.1e9,
                "format": "mlx, gguf",
                "support": "unsupported",
                "reason": "Architecture lfm3 not supported yet by mlx",
                "learning_level": "off",
            },
            {
                "id": "embedding",
                "name": "nomic-embed-text",
                "state": "loaded",
                "size_bytes": 0.3e9,
                "format": "gguf",
                "support": "unsupported",
                "reason": "embedding model",
                "learning_level": "off",
            },
            {
                "id": "granite",
                "name": "Granite-3B",
                "state": "downloaded",
                "size_bytes": 2e9,
                "format": "mlx",
                "support": "pending",
                "reason": "Waiting for measured trainability",
                "learning_level": "new",
            },
        ],
    }
    cli.print_status(data)
    output = capsys.readouterr().out
    assert "Waiting for memory" in output and "Not supported yet" in output
    assert "unsupported ·" not in output
    assert_snapshot("library", output)


def test_status_only_training_dots_are_colored(monkeypatch):
    from rich.console import Console

    monkeypatch.delenv("NO_COLOR")
    data = state()
    data["training"] = {"model": data["models"][0]["id"], "phase": "train"}
    data["models"][1]["last_status"] = "rejected"
    stream = io.StringIO()
    monkeypatch.setattr(
        terminal,
        "console",
        lambda: Console(
            file=stream,
            force_terminal=True,
            color_system="standard",
            width=10000,
            highlight=False,
            markup=False,
        ),
    )
    cli.print_status(data)
    output = stream.getvalue()
    assert output.count("\x1b[32m● ") == 2
    assert "\x1b[31m" not in output and "\x1b[36m" not in output


@pytest.mark.parametrize("show_diff", [False, True])
def test_init_snapshot_only_shows_diffs_when_requested(
    tmp_path, monkeypatch, capsys, show_diff
):
    from songe import agents, launchd, platform
    from songe.models import discovery
    from songe.upstream import lmstudio

    monkeypatch.setattr(cli, "detect_agent_binding", lambda: ("", ""))
    monkeypatch.setattr(
        platform,
        "current",
        lambda: SimpleNamespace(
            accelerators=lambda: [],
            data_dir=lambda: tmp_path / "data",
            service=lambda: SimpleNamespace(available=lambda: True),
        ),
    )
    monkeypatch.setattr(discovery, "lmstudio_downloads", lambda: tmp_path / "models")
    monkeypatch.setattr(discovery, "vllm", lambda url: [])
    monkeypatch.setattr(lmstudio, "find_lms", lambda: "fixture-lms")
    monkeypatch.setattr(cli, "upstream_model_ids", lambda url: ["one", "two"])
    diff = "--- before\n+++ after\n+exact fixture diff"
    monkeypatch.setattr(
        agents, "proposed", lambda *a, **kw: [(tmp_path / "agent.json", {}, "", diff)]
    )
    # opencode was switched to a Songe model; pi's default is the user's.
    monkeypatch.setattr(agents, "routing", lambda home=None: {"opencode": True, "pi": False})
    applied = []
    monkeypatch.setattr(agents, "apply", lambda *a: applied.append("agents"))
    monkeypatch.setattr(launchd, "install", lambda *a, **kw: applied.append("service"))
    monkeypatch.setattr(
        cli.LocalClient, "snapshot", lambda self: {"service_running": True}
    )
    from songe.log import Store
    from songe.storage import private_dir

    # Reinstalling after `songe uninstall` (which pauses) learns again.
    Store(private_dir(tmp_path / "data")).set("paused", True)
    args = ["--data-dir", str(tmp_path / "data"), "init", "--upstream", "vllm", "--yes"]
    assert cli.main(args + (["--show-diff"] if show_diff else [])) == 0
    assert applied == ["service", "agents"]
    assert Store(tmp_path / "data").get("paused") is False
    output = capsys.readouterr().out.replace(str(tmp_path), "<tmp>").replace("\\", "/")
    if show_diff:
        assert diff in output
        output = output.replace(diff + "\n", "")
    else:
        assert diff not in output
    assert_snapshot("init", output)


def test_undo_uninstall_and_errors_snapshots(tmp_path, monkeypatch, capsys):
    from songe import agents, launchd

    monkeypatch.setattr(cli.LocalClient, "control", lambda *a, **kw: None)
    args = ["--data-dir", str(tmp_path / "data")]
    assert cli.main(args + ["undo", "Qwen3.5 4B"]) == 0
    assert_snapshot("undo", capsys.readouterr().out)
    monkeypatch.setattr(agents, "restore", lambda root: [])
    monkeypatch.setattr(launchd, "remove", lambda *a, **kw: None)
    assert cli.main(args + ["uninstall"]) == 0
    data_dir = str((tmp_path / "data").resolve())
    assert_snapshot("uninstall", capsys.readouterr().out.replace(data_dir, "<data>"))

    def fail(*a, **kw):
        raise RuntimeError("Cannot restore version\nNo previous version")

    monkeypatch.setattr(cli.LocalClient, "control", fail)
    assert cli.main(args + ["undo"]) == 1
    assert_snapshot("error", capsys.readouterr().err)


@pytest.mark.parametrize("no_color,tty", [(True, True), (True, False), (False, False)])
def test_plain_mode_never_spins_or_uses_ansi(monkeypatch, no_color, tty):
    if not no_color:
        monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("COLUMNS", "12")
    stream = io.StringIO()
    monkeypatch.setattr(stream, "isatty", lambda: tty)
    monkeypatch.setattr(terminal.sys, "stdout", stream)

    def unexpected_spinner(*a, **kw):
        pytest.fail("Plain output must not create a spinner")

    monkeypatch.setattr(terminal.Console, "status", unexpected_spinner)
    with terminal.step("Downloading runtime", "Runtime installed"):
        pass
    long_id = "model/" + "x" * 150
    cli.print_models([{"id": long_id, "learning_level": "all", "pending": 7}])
    output = stream.getvalue()
    assert "\x1b" not in output and "\r" not in output
    assert long_id in output and "✓ Runtime installed" in output
    assert "Downloading runtime" not in output


def test_interactive_steps_use_spinner_and_success(monkeypatch):
    monkeypatch.delenv("NO_COLOR")
    observed = []

    @contextmanager
    def spinner(label, **kw):
        observed.append(str(label))
        assert kw == {"spinner": "dots", "spinner_style": terminal.ACCENT}
        yield

    monkeypatch.setattr(
        terminal, "console", lambda: SimpleNamespace(is_terminal=True, status=spinner)
    )
    monkeypatch.setattr(terminal, "result", observed.append)
    with terminal.step("Loading model", "Model prepared"):
        observed.append("work")
    assert observed == ["Loading model", "work", "Model prepared"]


def test_non_tty_confirm_never_reads_stdin_and_requires_yes(monkeypatch):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        "builtins.input", lambda *a: pytest.fail("must not read piped input")
    )
    with pytest.raises(ValueError, match="use --yes"):
        cli.confirm("Install?")
    assert cli.confirm("Install?", yes=True)


def test_install_confirmation_defaults_yes_but_purge_defaults_no(monkeypatch):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or "")
    assert cli.confirm("Install?")
    assert not cli.confirm("Delete Songe data?")
    assert prompts == ["Install? (Y/n) ", "Delete Songe data? (y/N) "]


@pytest.mark.parametrize("fail", [False, True])
def test_runtime_install_keeps_logs_out_of_terminal(
    tmp_path, monkeypatch, capsys, fail
):
    from songe.backends import install

    monkeypatch.setattr(
        install.venv,
        "EnvBuilder",
        lambda **kw: SimpleNamespace(create=lambda root: None),
    )

    def run(command, **kw):
        assert kw["stderr"] == install.subprocess.STDOUT
        kw["stdout"].write("fixture installer details\n")
        if fail:
            raise install.subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(install.subprocess, "run", run)
    if fail:
        with pytest.raises(RuntimeError, match="runtime-install.log"):
            install.install([["fixture-runtime"]], tmp_path)
    else:
        install.install([["fixture-runtime"]], tmp_path)
    assert (tmp_path / "runtime-install.log").read_text(
        encoding="utf-8"
    ) == "fixture installer details\n"
    assert capsys.readouterr().out == ""


def test_parser_error_is_one_line_and_hint(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["models", "some-model", "all"])
    assert exc.value.code == 2
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("✗ ") and lines[1] == "usage  songe --help"


def test_control_commands_print_results(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli.LocalClient, "control", lambda *a, **kw: None)
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: None)
    args = ["--data-dir", str(tmp_path / "data")]
    for command, expected in [
        (["pause"], "✓ Learning paused\n"),
        (["resume"], "✓ Learning resumed\n"),
        (["sleep", "--now"], "✓ Learning requested · songe status\n"),
        (
            ["forget", "--yes"],
            "✓ Conversations deleted · learning paused · songe resume to continue\n",
        ),
    ]:
        assert cli.main(args + command) == 0
        assert capsys.readouterr().out == expected
    monkeypatch.setattr(
        cli.LocalClient, "snapshot", lambda self: {"service_running": False}
    )
    assert cli.main(args + ["open"]) == 1
    assert capsys.readouterr().err == "✗ Songe is stopped\nrun  songe serve\n"
    monkeypatch.setattr(
        cli.LocalClient, "snapshot", lambda self: {"service_running": True}
    )
    assert cli.main(args + ["open"]) == 0
    port = cli.Config.load(tmp_path / "data").port
    assert capsys.readouterr().out == f"✓ Dashboard · http://127.0.0.1:{port}/\n"
    # Usage errors point at usage, not diagnostics.
    assert cli.main(args + ["models", "qwen"]) == 1
    assert "usage  songe models <model> on|off" in capsys.readouterr().err


def test_flow_states_use_dashboard_units():
    from songe.wording import flow_state

    assert flow_state("collecting conversations · 3 of 32") == (
        "collecting examples · 3 of 32"
    )
    assert flow_state("Waiting for an idle gap") == "waiting for a quiet moment"
    for wait in ("Waiting for experience", "no new conversations through songe yet"):
        assert flow_state(wait) == "waiting for conversations"
    assert flow_state("Will sleep when idle") == "starts when idle"
    assert flow_state("collecting conversations · 0 of 8 new examples") == (
        "collecting · 0 of 8 new examples"
    )
    assert flow_state("LM Studio busy") == "LM Studio busy"
