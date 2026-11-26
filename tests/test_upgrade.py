import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from songe import __version__, cli, upgrade
from songe.api import LocalClient, snapshot
from songe.backends.install import runtime_python
from songe.config import Config
from songe.log import Store
from songe.platform.services import LABEL, Launchd, Systemd, TaskScheduler

KINDS = [(Launchd, ".plist"), (Systemd, ".service"), (TaskScheduler, ".xml")]


@pytest.fixture(autouse=True)
def fresh_check(monkeypatch):
    monkeypatch.setattr(upgrade, "_checked", {})


def installed(tmp_path, kind, suffix, *, private=True, version="0.1.0a1"):
    root = tmp_path / "data"
    config = Config(model=str(tmp_path / "model"))
    config.save(root)
    identity = Store(root).get("installation_id")
    calls = []

    def metadata(value):
        path = (
            root / "runtime/lib/python3.12/site-packages/songe-0.1.0.dist-info/METADATA"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"Name: songe\nVersion: {value}\n", encoding="utf-8")

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[1:4] == ["-m", "pip", "install"]:
            metadata(__version__)
        return subprocess.CompletedProcess(
            argv, 1 if argv[1:2] == ["print"] else 0, "", ""
        )

    python = str(runtime_python(root)) if private else sys.executable
    if private:
        path = Path(python)
        path.parent.mkdir(parents=True)
        path.write_text("fake interpreter; never execute", encoding="utf-8")
        (root / "runtime/pyvenv.cfg").write_text(
            "include-system-site-packages = false\n", encoding="utf-8"
        )
        metadata(version)
    service = kind(tmp_path / ("songe" + suffix), run=run, activate=False)
    service.install(
        [python, "-m", "songe", "--data-dir", str(root), "serve"],
        {"SONGE_DATA_DIR": str(root), "SONGE_INSTALLATION_ID": identity},
    )
    service.activate = True
    return root, config, service, calls, run


def state(version=__version__, *, instance="new", token=None):
    return {
        "service_running": True,
        "service_version": version,
        "service_install": token or upgrade.RUNNING_INSTALL,
        "service_instance": instance,
    }


@pytest.mark.parametrize("kind,suffix", KINDS)
def test_old_runtime_reinstalls_and_restarts_owned_service(
    tmp_path, monkeypatch, capsys, kind, suffix
):
    root, config, service, calls, run = installed(tmp_path, kind, suffix)
    monkeypatch.setattr(
        upgrade, "package_requirement", lambda: "songe[torch]==" + __version__
    )
    replies = iter([state("0.1.0a1", instance="old"), state()])
    before = set(sys.modules)
    assert (
        upgrade.catch_up(
            root, config, service=service, run=run, read_state=lambda: next(replies)
        )
        is None
    )
    assert calls[0][0] == [
        str(runtime_python(root)),
        "-m",
        "pip",
        "install",
        "songe[torch]==" + __version__,
    ]
    assert calls[0][1]["capture_output"] and calls[0][1]["check"]
    assert calls[0][1]["env"]["PIP_NO_INPUT"] == "1"
    commands = [argv for argv, kwargs in calls]
    if kind is Launchd:
        assert commands[1:] == [
            ["launchctl", "kickstart", "-k", f"{service.domain}/{LABEL}"]
        ]
    elif kind is Systemd:
        assert commands[1:] == [
            ["systemctl", "--user", "daemon-reload"],
            ["systemctl", "--user", "restart", service.target.name],
        ]
    else:
        assert commands[1:] == [
            ["schtasks", "/End", "/TN", service.name],
            [
                "schtasks",
                "/Create",
                "/TN",
                service.name,
                "/XML",
                str(service.target),
                "/F",
            ],
            ["schtasks", "/Run", "/TN", service.name],
        ]
    assert (
        capsys.readouterr().out
        == f"Updating Songe service to {__version__}… ✓ Updated\n"
    )
    assert not (root / upgrade.PENDING).exists()
    assert not any(
        name == "torch" or name.startswith("torch.")
        for name in set(sys.modules) - before
    )
    # The entire probe (including HTTP and distribution metadata) is cached.
    upgrade.catch_up(
        root,
        config,
        service=service,
        run=run,
        read_state=lambda: pytest.fail("repeated probe"),
    )
    assert [argv for argv, kwargs in calls] == commands


@pytest.mark.parametrize("kind,suffix", KINDS)
@pytest.mark.parametrize("private", [True, False])
def test_matching_versions_run_nothing(tmp_path, capsys, kind, suffix, private):
    root, config, service, calls, run = installed(
        tmp_path, kind, suffix, private=private, version=__version__
    )
    assert (
        upgrade.catch_up(root, config, service=service, run=run, read_state=state)
        is None
    )
    assert calls == []
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("kind,suffix", KINDS)
@pytest.mark.parametrize("legacy", [True, False])
def test_service_without_runtime_catches_up(tmp_path, kind, suffix, legacy):
    root, config, service, calls, run = installed(tmp_path, kind, suffix, private=False)
    old = {"service_running": True} if legacy else state("0.1.0a1", instance="old")
    replies = iter([old, state()])
    upgrade.catch_up(
        root, config, service=service, run=run, read_state=lambda: next(replies)
    )
    assert calls
    assert not any(argv[1:3] == ["-m", "pip"] for argv, kwargs in calls)


@pytest.mark.parametrize("kind,suffix", KINDS)
def test_replaced_install_rebinds_and_restarts(tmp_path, kind, suffix):
    root, config, service, calls, run = installed(
        tmp_path, kind, suffix, private=False, version=__version__
    )
    argv, env = service.definition()
    argv[0] = str(tmp_path / "removed-keg/bin/python")
    service.write_definition(argv, env)
    replies = iter([state(instance="old", token="replaced-files"), state()])
    upgrade.catch_up(
        root, config, service=service, run=run, read_state=lambda: next(replies)
    )
    assert service.definition()[0][0] == sys.executable
    assert calls
    if kind is Launchd:
        assert "bootstrap" in [argv[1] for argv, kwargs in calls]
        assert calls[-1][0][1:3] == ["kickstart", "-k"]


@pytest.mark.parametrize("partial", [False, True])
def test_uninitialized_never_probes_or_creates_files(tmp_path, monkeypatch, partial):
    root = tmp_path / "missing"
    if partial:
        Config().save(root)
    monkeypatch.setattr(upgrade, "current", lambda: pytest.fail("service lookup"))
    upgrade.catch_up(root, run=lambda *a, **kw: pytest.fail("subprocess"))
    assert not (root / "service-upgrade.lock").exists()
    assert root.exists() is partial


@pytest.mark.parametrize("kind,suffix", KINDS)
def test_foreign_service_is_preserved(tmp_path, kind, suffix, capsys):
    root, config, service, calls, run = installed(tmp_path, kind, suffix)
    argv, env = service.definition()
    env["SONGE_INSTALLATION_ID"] = "another-installation"
    service.target.unlink()
    service.write_definition(argv, env)
    before = service.target.read_bytes()
    message = upgrade.catch_up(
        root,
        config,
        service=service,
        run=run,
        read_state=lambda: pytest.fail("foreign probe"),
    )
    # Another installation's service is left alone, silently.
    assert message is None
    assert calls == [] and service.target.read_bytes() == before
    assert capsys.readouterr().out == ""
    assert not (root / upgrade.PENDING).exists()


@pytest.mark.parametrize("failure", ["install", "restart", "health"])
def test_failure_is_one_line_and_pending_notice(tmp_path, monkeypatch, capsys, failure):
    root, config, service, calls, run = installed(tmp_path, Systemd, ".service")
    monkeypatch.setattr(
        upgrade, "package_requirement", lambda: "songe[torch]==" + __version__
    )

    def failing_run(argv, **kwargs):
        if (failure == "install" and "pip" in argv) or (
            failure == "restart" and "restart" in argv
        ):
            raise subprocess.CalledProcessError(
                1, argv, stderr="private\nvery long diagnostic"
            )
        return run(argv, **kwargs)

    service.run = failing_run
    message = upgrade.catch_up(
        root,
        config,
        service=service,
        run=failing_run,
        read_state=lambda: state("0.1.0a1"),
        health_timeout=0,
    )
    output = capsys.readouterr().out
    assert len(output.splitlines()) == 1
    assert "fix: songe doctor" in output
    assert "private" not in output
    assert message == upgrade.notice(root)
    assert "upgrade_notice" in snapshot(config, Store(root))


def test_busy_update_never_installs_or_restarts(tmp_path, monkeypatch):
    root, config, service, calls, run = installed(tmp_path, Systemd, ".service")

    def busy(lease):
        raise BlockingIOError

    monkeypatch.setattr(upgrade, "acquire", busy)
    upgrade.catch_up(
        root,
        config,
        service=service,
        run=run,
        read_state=lambda: pytest.fail("probe while locked"),
    )
    assert calls == []


def test_runtime_metadata_fallback_uses_stdlib_only(tmp_path):
    python = runtime_python(tmp_path)
    python.parent.mkdir(parents=True)
    python.write_text("fake", encoding="utf-8")
    (tmp_path / "runtime/pyvenv.cfg").write_text(
        "include-system-site-packages = false\n", encoding="utf-8"
    )
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, __version__ + "\n")

    assert upgrade.runtime_version(tmp_path, run) == __version__
    assert calls == [
        [
            str(python),
            "-I",
            "-c",
            "from importlib.metadata import version; print(version('songe'))",
        ]
    ]


def test_service_identity_must_match_before_mutation(tmp_path, monkeypatch):
    root, config, service, calls, run = installed(tmp_path, Systemd, ".service")
    monkeypatch.setattr(
        httpx.Client,
        "get",
        lambda *a, **kw: httpx.Response(
            200,
            request=httpx.Request("GET", "http://test/api/state"),
            json={"installation_id": "foreign"},
        ),
    )
    assert "another installation" in upgrade.catch_up(
        root, config, service=service, run=run
    )
    assert calls == []


def test_api_and_status_report_only_remaining_mismatch(tmp_path, capsys):
    store = Store(tmp_path / "data")
    config = Config()
    data = snapshot(config, store, live=True)
    assert data["service_version"] == __version__
    assert data["version"] is None  # Learned model version retains its contract.
    assert data["upgrade_notice"] is None
    message = "Runtime 0.1.0a1 differs from CLI · fix: songe doctor"
    (store.root / upgrade.PENDING).write_text(
        json.dumps({"notice": message}), encoding="utf-8"
    )
    data = snapshot(config, store, live=True)
    cli.print_status(data)
    assert capsys.readouterr().out.count(message) == 1
    (store.root / upgrade.PENDING).unlink()
    assert snapshot(config, store, live=True)["upgrade_notice"] is None
    assert "replaced install" in upgrade.notice(
        store.root, version=__version__, token="old"
    )


def test_failed_update_does_not_break_user_command(tmp_path, monkeypatch, capsys):
    root, config, service, calls, run = installed(tmp_path, Systemd, ".service")
    monkeypatch.setattr(
        upgrade,
        "current",
        lambda: type("Platform", (), {"service": lambda self: service})(),
    )
    monkeypatch.setattr(
        upgrade,
        "runtime_version",
        lambda *a: (_ for _ in ()).throw(RuntimeError("metadata unavailable")),
    )
    monkeypatch.setattr(LocalClient, "control", lambda *a, **kw: None)
    assert cli.main(["--data-dir", str(root), "pause"]) == 0
    output = capsys.readouterr().out
    assert "metadata unavailable · fix: songe doctor" in output and output.endswith(
        "✓ Learning paused\n"
    )
    assert calls == []


def test_package_manager_hook_uninitialized_is_silent(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SONGE_DATA_DIR", str(tmp_path / "missing"))
    upgrade.main()
    assert capsys.readouterr().out == ""
    assert not (tmp_path / "missing").exists()


async def test_service_probe_skips_library_snapshot(config, store, monkeypatch):
    from songe import api
    from songe.proxy import create_app

    app = create_app(config, store)
    monkeypatch.setattr(api, "snapshot", lambda *a, **kw: pytest.fail("full snapshot"))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.get("/api/state?service=1")
        data = response.json()
        assert response.headers["cache-control"] == "no-store"
        assert data["installation_id"] == store.get("installation_id")
        assert data["service_version"] == __version__
        assert data["service_install"] == upgrade.RUNNING_INSTALL
        assert data["service_instance"]
        assert data["service_replaced"] is False


def test_recovered_restart_clears_stale_notice_without_restart(tmp_path, capsys):
    root, config, service, calls, run = installed(
        tmp_path, Systemd, ".service", version=__version__
    )
    (root / upgrade.PENDING).write_text(
        json.dumps(
            {
                "version": __version__,
                "runtime_pending": False,
                "notice": "Delayed restart · fix: songe doctor",
            }
        ),
        encoding="utf-8",
    )
    assert (
        upgrade.notice(root, version=__version__, token=upgrade.RUNNING_INSTALL) is None
    )
    assert (
        upgrade.catch_up(root, config, service=service, run=run, read_state=state)
        is None
    )
    assert not (root / upgrade.PENDING).exists()
    assert calls == [] and capsys.readouterr().out == ""


def test_health_waits_for_new_process_even_at_same_version(
    tmp_path, monkeypatch, capsys
):
    root, config, service, calls, run = installed(
        tmp_path, Systemd, ".service", private=False
    )
    old = state(instance="old") | {"service_replaced": True}
    replies = iter([old, old, state()])
    sleeps = []
    monkeypatch.setattr(upgrade.time, "sleep", sleeps.append)
    upgrade.catch_up(
        root, config, service=service, run=run, read_state=lambda: next(replies)
    )
    assert sleeps == [0.2]
    assert capsys.readouterr().out.endswith("✓ Updated\n")


def test_newer_runtime_is_preserved(tmp_path, capsys):
    root, config, service, calls, run = installed(
        tmp_path, Systemd, ".service", version="99.0.0"
    )
    message = upgrade.catch_up(root, config, service=service, run=run, read_state=state)
    assert "runtime 99.0.0" in message
    assert calls == []
    assert len(capsys.readouterr().out.splitlines()) == 1
    assert (
        upgrade.notice(root, version=__version__, token=upgrade.RUNNING_INSTALL)
        == message
    )


def test_external_runtime_is_never_modified(tmp_path, capsys):
    root, config, service, calls, run = installed(
        tmp_path, Systemd, ".service", private=False
    )
    external = tmp_path / "external"
    external.mkdir()
    try:
        (root / "runtime").symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks requires Windows developer mode")
    assert "outside this installation" in upgrade.catch_up(
        root, config, service=service, run=run
    )
    assert calls == [] and list(external.iterdir()) == []


def test_incomplete_runtime_cannot_install_into_base_python(tmp_path):
    root, config, service, calls, run = installed(tmp_path, Systemd, ".service")
    (root / "runtime/pyvenv.cfg").unlink()
    assert "not a private Python environment" in upgrade.catch_up(
        root, config, service=service, run=run
    )
    assert calls == []


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["status"],
        ["doctor"],
        ["tui"],
        ["undo"],
        ["models"],
        ["open"],
        ["pause"],
        ["resume"],
        ["sleep", "--now"],
        ["sleep", "--dry-run"],
        ["forget", "--yes"],
    ],
)
def test_user_commands_check_before_dispatch(tmp_path, monkeypatch, arguments):
    checked = []

    def check(root, config):
        checked.append(root)
        raise RuntimeError("stop before dispatch")

    monkeypatch.setattr(cli, "catch_up", check)
    root = tmp_path / "missing"
    assert cli.main(["--data-dir", str(root), *arguments]) == 1
    assert checked == [root]
    assert not root.exists()


@pytest.mark.parametrize("command", ["init", "serve", "uninstall"])
def test_lifecycle_commands_skip_catch_up(tmp_path, monkeypatch, command):
    monkeypatch.setattr(
        cli, "catch_up", lambda *a, **kw: pytest.fail("lifecycle catch-up")
    )
    # Stop dispatch before any lifecycle action can install or restart a service.
    monkeypatch.setattr(
        cli,
        "Store",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("stop before dispatch")),
    )
    assert cli.main(["--data-dir", str(tmp_path), command]) == 1


def test_slow_service_counts_as_busy_not_restarted(monkeypatch):
    import httpx

    from songe import upgrade

    class Slow:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url):
            raise httpx.ReadTimeout("full scan")

    monkeypatch.setattr(upgrade.httpx, "Client", Slow)
    state = upgrade.service_state(SimpleNamespace(port=7613), "id")
    # Busy, release unknown; not an agent request it cannot know about.
    assert state["busy"] and not state.get("requests_in_flight")


def test_a_slow_service_on_this_release_is_not_said_to_wait_for_an_agent(
    tmp_path, monkeypatch, capsys
):
    # #71: the service ran this release with no request in flight, but its
    # status check timed out while measuring; every command said "Update to
    # … waits for the agent request in progress".
    root, config, service, calls, run = installed(
        tmp_path, Launchd, ".plist", version=__version__
    )
    monkeypatch.setattr(upgrade.time, "sleep", lambda s: None)

    class Slow:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url):
            raise httpx.ReadTimeout("measuring")

    monkeypatch.setattr(upgrade.httpx, "Client", Slow)
    upgrade.catch_up(root, config, service=service, run=run, busy_timeout=0)
    assert not calls  # never restarted
    assert capsys.readouterr().out == ""


def test_a_slow_service_is_never_restarted_for_a_needed_update(
    tmp_path, monkeypatch, capsys
):
    root, config, service, calls, run = installed(tmp_path, Launchd, ".plist")
    monkeypatch.setattr(upgrade.time, "sleep", lambda s: None)
    slow = {"service_running": True, "service_version": None, "busy": True}
    upgrade.catch_up(
        root, config, service=service, run=run, read_state=lambda: slow, busy_timeout=0
    )
    assert not calls
    out = capsys.readouterr().out
    assert "less busy" in out and "agent request" not in out


def test_data_dir_without_its_own_service_is_skipped_quietly(tmp_path, monkeypatch):
    from songe import upgrade

    class Foreign:
        target = tmp_path / "service.plist"

        def check_owner(self, expected):
            raise RuntimeError("belongs to another data directory")

    Foreign.target.write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        upgrade,
        "Config",
        SimpleNamespace(
            load=lambda root: SimpleNamespace(model="m", exposed_models=[])
        ),
    )
    (tmp_path / "config.toml").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        upgrade, "ReadOnlyStore", lambda root: SimpleNamespace(get=lambda k: "id")
    )
    printed = []
    assert upgrade.catch_up(tmp_path, service=Foreign(), emit=printed.append) is None
    assert printed == []


def test_update_never_restarts_under_an_agent_request(tmp_path, monkeypatch, capsys):
    root, config, service, calls, run = installed(tmp_path, Launchd, ".plist")
    monkeypatch.setattr(upgrade.time, "sleep", lambda s: None)
    busy = state("0.1.0a1", instance="old") | {"requests_in_flight": 1}
    upgrade.catch_up(
        root, config, service=service, run=run, read_state=lambda: busy, busy_timeout=0
    )
    assert not calls
    assert "waits for the agent request in progress" in capsys.readouterr().out


@pytest.mark.skipif(sys.platform == "win32", reason="Homebrew paths are POSIX")
def test_service_uses_homebrew_opt_path(tmp_path):
    from songe.launchd import stable_python

    cellar = tmp_path / "Cellar/songe/0.1.0/libexec/bin/python"
    opt = tmp_path / "opt/songe/libexec/bin/python"
    for path in (cellar, opt):
        path.parent.mkdir(parents=True)
        path.write_text("")
    assert stable_python(str(cellar)) == str(opt)
    assert stable_python("/usr/bin/python3") == "/usr/bin/python3"


def test_an_old_windows_launcher_is_rewritten_even_at_the_same_release(tmp_path):
    # Before #158 the launcher replaced itself with os.execvpe, which broke
    # spaced paths and left the task's job (#157). Only init rewrote it, so an
    # upgrade must, even when the service already runs this release.
    root, config, service, calls, run = installed(
        tmp_path, TaskScheduler, ".xml", private=False, version=__version__
    )
    launcher = service.target.with_suffix(".py")
    argv, env = service.definition()
    launcher.write_text(
        "import os\n"
        f"os.environ.update({env!r})\n"
        f"argv = {argv!r}\n"
        "os.execvpe(argv[0], argv, os.environ)\n",
        encoding="utf-8",
    )
    assert service.stale_launcher()
    replies = iter([state(), state(instance="new-again")])
    upgrade.catch_up(
        root, config, service=service, run=run, read_state=lambda: next(replies)
    )
    assert not service.stale_launcher()
    assert "subprocess.call" in launcher.read_text(encoding="utf-8")
    assert service.definition() == (argv, env)
    assert [argv[1] for argv, kwargs in calls if argv[0] == "schtasks"][-1] == "/Run"
    # Once rewritten, the next check finds nothing to do.
    calls.clear()
    upgrade._checked.clear()
    upgrade.catch_up(root, config, service=service, run=run, read_state=state)
    assert calls == []


@pytest.mark.parametrize("kind,suffix", KINDS)
def test_converter_toolkit_is_never_taken_for_the_service_installation(
    tmp_path, monkeypatch, capsys, kind, suffix
):
    """An MLX Mac whose service runs from the user's own install gets the
    GGUF converter toolkit (torch, no MLX) on first use. The upgrade check
    took a toolkit in <data dir>/runtime for Songe's own installation and
    rebound the service to it, or reported "Service update failed"."""
    import importlib.util

    from songe.backends import install
    from songe.platform import Rooted

    root, config, service, calls, run = installed(
        tmp_path, kind, suffix, private=False, version=__version__
    )

    def provision(commands, data, *folder):
        venv = Path(commands[0][0]).parent.parent
        python = Path(commands[0][0])
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_text("fake interpreter; never execute", encoding="utf-8")
        (venv / "pyvenv.cfg").write_text(
            "include-system-site-packages = false\n", encoding="utf-8"
        )
        package = venv / "lib/python3.12/site-packages"
        (package / "songe").mkdir(parents=True)
        (package / "songe/__init__.py").write_text("", encoding="utf-8")
        info = package / "songe-0.1.0.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(
            f"Name: songe\nVersion: {__version__}\n", encoding="utf-8"
        )

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(install, "install", provision)
    monkeypatch.setattr(
        install.subprocess,
        "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 1),
    )
    metal = SimpleNamespace(accelerators=lambda: [SimpleNamespace(kind="metal")])
    install.converter_python(Rooted(metal, root))
    monkeypatch.undo()
    monkeypatch.setattr(upgrade, "_checked", {})
    capsys.readouterr()

    message = upgrade.catch_up(
        root, config, service=service, run=run, read_state=state
    )
    assert service.definition()[0][0] == sys.executable
    assert not any(argv[1:3] == ["-m", "pip"] for argv, kwargs in calls)
    assert "failed" not in (message or "") + capsys.readouterr().out
