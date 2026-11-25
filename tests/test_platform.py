import ctypes
import plistlib
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

from songe.platform import Memory
from songe.platform.darwin import Darwin
from songe.platform.linux import Linux, pressure
from songe.platform.services import Launchd, Systemd, TaskScheduler
from songe.platform.windows import MemoryStatus, Windows


@pytest.mark.parametrize("blocked", [False, True])
def test_windows_lease_does_not_access_locked_bytes(monkeypatch, blocked):
    from songe.platform import locking

    calls = []

    def lock(fd, mode, count):
        calls.append((fd, mode, count))
        if blocked:
            raise PermissionError("locked by another handle")

    monkeypatch.setattr(locking, "os", SimpleNamespace(name="nt"))
    monkeypatch.setitem(
        sys.modules, "msvcrt", SimpleNamespace(LK_NBLCK=2, locking=lock)
    )
    # No read/write methods: a contender must reach the lock call without I/O.
    lease = SimpleNamespace(seek=lambda n: calls.append(n), fileno=lambda: 42)
    if blocked:
        with pytest.raises(BlockingIOError):
            locking.acquire(lease)
    else:
        locking.acquire(lease)
    assert calls == [0, (42, 2, 1)]


def test_linux_paths_memory_psi_and_headless(tmp_path):
    proc = tmp_path / "proc"
    (proc / "pressure").mkdir(parents=True)
    (proc / "meminfo").write_text(
        "MemTotal: 8000000 kB\nMemAvailable: 4000000 kB\n", encoding="utf-8"
    )
    psi = proc / "pressure/memory"
    psi.write_text("some avg10=1.00 avg60=0.2 total=10\n", encoding="utf-8")
    platform = Linux(
        tmp_path, proc, tmp_path / "sys", {"XDG_DATA_HOME": str(tmp_path / "xdg")}
    )
    assert platform.data_dir() == tmp_path / "xdg/songe"
    assert platform.idle_seconds() is None
    assert platform.on_ac_power() is None
    assert platform.memory() == Memory(8000000 * 1024, 4000000 * 1024, "warn", psi=1.0)
    psi.write_text("some avg10=3.00\n", encoding="utf-8")
    assert platform.memory().pressure == "warn"
    psi.write_text("some avg10=9.00\n", encoding="utf-8")
    assert platform.memory().pressure == "critical"
    assert pressure(0, 10, 0, 0) == "critical"
    assert platform.thermal() is None


def test_linux_battery(tmp_path):
    battery = tmp_path / "class/power_supply/BAT0"
    battery.mkdir(parents=True)
    (battery / "type").write_text("Battery", encoding="utf-8")
    (battery / "status").write_text("Discharging", encoding="utf-8")
    platform = Linux(tmp_path, sysfs=tmp_path)
    assert platform.on_ac_power() is False
    (battery / "status").write_text("Charging", encoding="utf-8")
    assert platform.on_ac_power() is True


def test_darwin_probes_are_kernel_measurements(tmp_path, monkeypatch):
    from songe.platform import darwin

    outputs = {
        "hw.memsize": "8000000000",
        "kern.memorystatus_vm_pressure_level": "2",
        "vm_stat": "Mach (page size of 4096 bytes)\nPages free: 100.\nPages inactive: 20.\nPages speculative: 5.",
        "IOHIDSystem": '"HIDIdleTime" = 600000000000',
        "batt": "Now drawing from 'AC Power'",
    }
    monkeypatch.setattr(
        darwin,
        "read",
        lambda *args: (
            outputs[Path(args[0]).name]
            if Path(args[0]).name == "vm_stat"
            else outputs[args[-1]]
        ),
    )
    monkeypatch.setattr(darwin, "thermal_state", lambda: 1)
    platform = Darwin(tmp_path)
    assert platform.data_dir() == tmp_path / "Library/Application Support/songe"
    assert platform.memory() == Memory(8000000000, 125 * 4096, "warn")
    assert platform.idle_seconds() == 600
    # IOPowerSources is exercised with an injected native API in test_power.py.
    assert platform.thermal() == 1


class Kernel:
    def GlobalMemoryStatusEx(self, pointer):
        info = pointer._obj
        assert info.length == ctypes.sizeof(MemoryStatus)
        info.total, info.available, info.load = 1000, 100, 90
        return 1

    def GetSystemPowerStatus(self, pointer):
        pointer._obj.ac = 1
        return 1

    def GetTickCount(self):
        return 5000

    def SetThreadExecutionState(self, flag):
        self.flag = flag
        return 1


class User:
    def GetLastInputInfo(self, pointer):
        pointer._obj.tick = 1000
        return 1


def test_windows_ctypes_and_paths(tmp_path):
    kernel = Kernel()
    platform = Windows(
        tmp_path, {"LOCALAPPDATA": str(tmp_path / "local")}, kernel, User()
    )
    assert platform.data_dir() == tmp_path / "local/songe"
    assert platform.memory() == Memory(1000, 100, "warn")
    assert platform.idle_seconds() == 4
    assert platform.on_ac_power()
    handle = platform.keep_awake(123)
    assert kernel.flag == 0x80000001
    handle.close()
    assert kernel.flag == 0x80000000


@pytest.mark.parametrize(
    "kind,suffix", [(Launchd, ".plist"), (Systemd, ".service"), (TaskScheduler, ".xml")]
)
def test_services_injected_targets_do_not_touch_user_services(tmp_path, kind, suffix):
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        return SimpleNamespace(returncode=0)

    service = kind(tmp_path / ("songe" + suffix), run=run, activate=False)
    assert service.status() == "uninstalled"
    argv = ["/python with spaces", "-m", "songe", "--data-dir", str(tmp_path), "serve"]
    env = {"VALUE": 'space "quote" & % $PATH'}
    target = service.install(argv, env)
    assert service.status() == "installed"
    if kind is Launchd:
        document = plistlib.loads(target.read_bytes())
        assert document["ProgramArguments"] == argv
        assert document["EnvironmentVariables"] == env
    elif kind is Systemd:
        text = target.read_text(encoding="utf-8")
        assert '"/python with spaces"' in text
        assert "%%" in text and "$$PATH" in text
    else:
        document = ET.fromstring(target.read_bytes())
        ns = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
        assert document.find(".//t:LogonType", ns).text == "InteractiveToken"
        script = target.with_suffix(".py").read_text(encoding="utf-8")
        assert "subprocess.call" in script and repr(env) in script
    service.uninstall()
    assert not target.exists()
    assert not calls


def test_windows_task_launcher_preserves_spaced_arguments_and_exit_code(tmp_path):
    import json

    spaced = tmp_path / "user with spaces" / "songe data"
    spaced.mkdir(parents=True)
    probe = spaced / "probe.py"
    output = spaced / "received.json"
    probe.write_text(
        "import json, os, pathlib, sys\n"
        f"pathlib.Path({str(output)!r}).write_text(\n"
        '    json.dumps({"argv": sys.argv[1:], "value": os.environ["VALUE"]}),\n'
        '    encoding="utf-8",\n'
        ")\n"
        "raise SystemExit(3)\n",
        encoding="utf-8",
    )
    service = TaskScheduler(tmp_path / "songe-task.xml", activate=False)
    argv = [sys.executable, str(probe), "--data-dir", str(spaced), "serve"]
    env = {"VALUE": 'space "quote" & % $PATH'}
    target = service.install(argv, env)
    assert service.definition() == (argv, env)
    result = subprocess.run(
        [sys.executable, str(target.with_suffix(".py"))],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 3, result.stderr
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "argv": ["--data-dir", str(spaced), "serve"],
        "value": env["VALUE"],
    }


def test_service_activation_uses_user_scope(tmp_path):
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    service = Systemd(tmp_path / "songe.service", run=run)
    service.install(["python", "-m", "songe"], {})
    assert service.status() == "running"
    service.uninstall()
    assert all(argv[:2] == ["systemctl", "--user"] for argv in calls)


@pytest.mark.parametrize("missing", [True, False])
def test_systemd_without_a_user_manager(tmp_path, missing):
    # Containers and WSL without systemd (#12): systemctl is absent, or
    # present without a user manager to talk to.
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        if missing:
            raise FileNotFoundError(2, "No such file or directory", "systemctl")
        return subprocess.CompletedProcess(argv, 1)

    service = Systemd(tmp_path / "songe.service", run=run)
    assert not service.available()
    service.target.write_text("[Unit]\n", encoding="utf-8")  # an earlier try
    service.uninstall()
    assert not service.target.exists()
    assert all(argv[2:] == ["show-environment"] for argv in calls)


def test_launchd_waits_for_unload_before_bootstrap(tmp_path, monkeypatch):
    import subprocess
    from types import SimpleNamespace

    from songe.platform import services

    monkeypatch.setattr(services.time, "sleep", lambda s: None)
    calls, loaded = [], {"print": 2, "bootstrap_fail": 1}

    def run(argv, **kw):
        calls.append(argv[1])
        if argv[1] == "print":
            loaded["print"] -= 1
            return SimpleNamespace(returncode=0 if loaded["print"] >= 0 else 113)
        if argv[1] == "bootstrap" and loaded["bootstrap_fail"]:
            loaded["bootstrap_fail"] -= 1
            return subprocess.CompletedProcess(argv, 5, b"", b"Input/output error")
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    service = services.Launchd(tmp_path / "songe.plist", uid=501, run=run)
    monkeypatch.setattr(service, "check_owner", lambda env: None)
    service.install(["songe", "serve"], {})
    assert calls[0] == "bootout" and calls.count("print") >= 2
    assert calls[-2:] == ["bootstrap", "bootstrap"]


def test_launchd_uninstall_accepts_an_already_unloaded_service(tmp_path):
    import subprocess

    import pytest

    from songe.platform import services

    def unloaded(argv, **kw):
        return subprocess.CompletedProcess(argv, 113 if argv[1] == "print" else 3)

    plist = tmp_path / "songe.plist"
    plist.write_text("x")
    services.Launchd(plist, uid=501, run=unloaded).uninstall()
    assert not plist.exists()

    def stuck(argv, **kw):
        return subprocess.CompletedProcess(argv, 0 if argv[1] == "print" else 1)

    plist.write_text("x")
    with pytest.raises(subprocess.CalledProcessError):
        services.Launchd(plist, uid=501, run=stuck).uninstall()
    assert plist.exists()


def test_a_later_gguf_learner_gets_the_converter_runtime_once(tmp_path, monkeypatch):
    import importlib.util
    import subprocess

    from songe.backends import install
    from songe.platform import Rooted

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    installed = []
    monkeypatch.setattr(
        install, "install", lambda commands, root: installed.append(commands)
    )
    monkeypatch.setattr(
        install.subprocess,
        "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 0),
    )
    metal = SimpleNamespace(accelerators=lambda: [SimpleNamespace(kind="metal")])
    platform = Rooted(metal, tmp_path)
    python = install.converter_python(platform)
    assert python == str(install.runtime_python(tmp_path)) and len(installed) == 1
    install.runtime_python(tmp_path).parent.mkdir(parents=True)
    install.runtime_python(tmp_path).touch()
    install.converter_python(platform)
    assert len(installed) == 1
    # CUDA/ROCm runtimes already carry torch; nothing is installed there.
    cuda = SimpleNamespace(accelerators=lambda: [SimpleNamespace(kind="cuda")])
    assert install.converter_python(Rooted(cuda, tmp_path)) == sys.executable


def test_real_conversions_get_the_converter_runtime(monkeypatch):
    """Conversion and merges pass artifacts.runtime.guarded_run; only the
    inner memory.guarded_run was recognised, so a Mac without torch never
    got the converter runtime and the panel said only "background work
    failed" (reported by Jack in a test on 11-24)."""
    from songe import memory
    from songe.artifacts import runtime
    from songe.backends import install

    monkeypatch.setattr(install, "converter_python", lambda platform: "converter")
    assert install.interpreter_for(None, runtime.guarded_run) == "converter"
    assert install.interpreter_for(None, memory.guarded_run) == "converter"
    assert install.interpreter_for(None, lambda *a, **k: None) == sys.executable


def test_a_dreams_evaluation_conversion_gets_the_converter_runtime(tmp_path, monkeypatch):
    """The Dream's own export and evaluation bound check= in a wrapper
    function, which interpreter_for did not recognise either."""
    from songe import models
    from songe.artifacts import storage
    from songe.backends import install
    from songe.models import gguf
    from songe.sleep import fuse

    seen = []
    monkeypatch.setattr(install, "converter_python", lambda platform: "converter")
    monkeypatch.setattr(gguf, "read_gguf", lambda path: SimpleNamespace(architecture="qwen3"))
    monkeypatch.setattr("songe.backends.resolve_kind", lambda config, platform: "torch")
    monkeypatch.setattr(storage, "artifact_root", lambda config, store: tmp_path)
    monkeypatch.setattr(models, "trainable_base", lambda ref, backend, platform, runner, artifact_directory:
                        seen.append(install.interpreter_for(platform, runner)))
    platform = SimpleNamespace(data_dir=lambda: tmp_path)
    config = SimpleNamespace(upstream="http://127.0.0.1:11434")
    fuse.evaluation_model(config, "gguf-full", tmp_path / "candidate.gguf", platform, lambda: None)
    assert seen == ["converter"]
