"""Per-user services. Targets and command runner are injectable; no implicit installs."""

import json
import os
import plistlib
import shlex
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from .. import NAME
from ..storage import atomic_write

LABEL = f"ai.souslabs.{NAME}"


class OwnedService:
    def available(self):
        """Whether the service manager can be reached (launchd and Task
        Scheduler always can)."""
        return True

    def check_owner(self, expected):
        if not self.target.exists():
            return
        try:
            argv, env = self.definition()
            identity = env.get("SONGE_INSTALLATION_ID")
            if identity:
                matches = identity == expected.get("SONGE_INSTALLATION_ID")
            else:
                root = env.get("SONGE_DATA_DIR")
                if not root and "--data-dir" in argv:
                    root = argv[argv.index("--data-dir") + 1]
                matches = bool(root and expected.get("SONGE_DATA_DIR")) and (
                    Path(root).expanduser().resolve()
                    == Path(expected["SONGE_DATA_DIR"]).expanduser().resolve()
                )
        except (OSError, ValueError, SyntaxError, KeyError, IndexError):
            matches = False
        if not matches:
            raise RuntimeError(
                "The installed service belongs to another data directory or cannot be identified; it was preserved"
            )


class Launchd(OwnedService):
    def __init__(self, target, *, uid=None, run=subprocess.run, activate=True):
        self.target, self.run, self.activate = Path(target), run, activate
        self.domain = (
            f"gui/{getattr(os, 'getuid', lambda: 0)() if uid is None else uid}"
        )

    def write_definition(self, argv, env):
        self.check_owner(env)
        atomic_write(
            self.target,
            plistlib.dumps(
                {
                    "Label": LABEL,
                    "ProgramArguments": list(argv),
                    "EnvironmentVariables": dict(env),
                    "RunAtLoad": True,
                    "KeepAlive": True,
                    "ProcessType": "Background",
                }
            ),
        )

    def install(self, argv, env):
        self.write_definition(argv, env)
        if self.activate:
            self.run(
                ["launchctl", "bootout", f"{self.domain}/{LABEL}"], capture_output=True
            )
            # bootout returns before launchd finishes unloading; bootstrapping
            # into that window fails with EIO (5). Wait for the unload instead.
            deadline = time.monotonic() + 10
            while (
                self.run(
                    ["launchctl", "print", f"{self.domain}/{LABEL}"],
                    capture_output=True,
                ).returncode
                == 0
                and time.monotonic() < deadline
            ):
                time.sleep(0.2)
            for attempt in range(5):
                result = self.run(
                    ["launchctl", "bootstrap", self.domain, str(self.target)],
                    capture_output=True,
                )
                if result.returncode == 0:
                    break
                time.sleep(0.5 * (attempt + 1))
            else:
                raise subprocess.CalledProcessError(
                    result.returncode, result.args, result.stdout, result.stderr
                )
        return self.target

    def restart(self, expected):
        self.check_owner(expected)
        if self.activate and self.target.exists():
            self.run(
                ["launchctl", "kickstart", "-k", f"{self.domain}/{LABEL}"],
                check=True,
                capture_output=True,
            )

    def definition(self):
        data = plistlib.loads(self.target.read_bytes())
        return data["ProgramArguments"], data.get("EnvironmentVariables", {})

    def uninstall(self):
        if self.activate and self.target.exists():
            done = self.run(
                ["launchctl", "bootout", f"{self.domain}/{LABEL}"],
                capture_output=True,
            )
            # Already unloaded (failed bootstrap, manual bootout) is stopped.
            if (
                done.returncode != 0
                and self.run(
                    ["launchctl", "print", f"{self.domain}/{LABEL}"],
                    capture_output=True,
                ).returncode
                == 0
            ):
                raise subprocess.CalledProcessError(
                    done.returncode, done.args, done.stdout, done.stderr
                )
        self.target.unlink(missing_ok=True)

    def status(self):
        if not self.target.exists():
            return "uninstalled"
        if not self.activate:
            return "installed"
        return (
            "running"
            if self.run(
                ["launchctl", "print", f"{self.domain}/{LABEL}"], capture_output=True
            ).returncode
            == 0
            else "stopped"
        )


class Systemd(OwnedService):
    def __init__(self, target, *, run=subprocess.run, activate=True):
        self.target, self.run, self.activate = Path(target), run, activate

    @staticmethod
    def quote(value):
        # systemd expands percent specifiers and $VARIABLE in ExecStart.
        return (
            json.dumps(str(value), ensure_ascii=False)
            .replace("%", "%%")
            .replace("$", "$$")
        )

    def write_definition(self, argv, env):
        self.check_owner(env)
        lines = [
            "[Unit]",
            f"Description={NAME}",
            "[Service]",
            "Type=simple",
            "ExecStart=" + " ".join(self.quote(v) for v in argv),
        ]
        lines += [
            "Environment=" + self.quote(f"{k}={v}") for k, v in sorted(env.items())
        ]
        lines += ["Restart=on-failure", "[Install]", "WantedBy=default.target", ""]
        atomic_write(self.target, "\n".join(lines).encode())

    def available(self):
        # Containers, WSL without systemd and other init systems have no user
        # manager; systemctl is then missing or cannot reach it.
        if not self.activate:
            return True
        try:
            return (
                self.run(
                    ["systemctl", "--user", "show-environment"], capture_output=True
                ).returncode
                == 0
            )
        except OSError:
            return False

    def install(self, argv, env):
        self.write_definition(argv, env)
        if self.activate:
            self.run(
                ["systemctl", "--user", "daemon-reload"],
                check=True,
                capture_output=True,
            )
            self.run(
                ["systemctl", "--user", "enable", "--now", self.target.name],
                check=True,
                capture_output=True,
            )
        return self.target

    def restart(self, expected):
        self.check_owner(expected)
        if self.activate and self.target.exists():
            self.run(
                ["systemctl", "--user", "daemon-reload"],
                check=True,
                capture_output=True,
            )
            self.run(
                ["systemctl", "--user", "restart", self.target.name],
                check=True,
                capture_output=True,
            )

    def definition(self):
        argv, env = [], {}
        for line in self.target.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key not in {"ExecStart", "Environment"}:
                continue
            # Decode the quoting used by this service writer, preserving UTF-8.
            parts = []
            if value.startswith('"'):
                decoder = json.JSONDecoder()
                while value.strip():
                    part, end = decoder.raw_decode(value.lstrip())
                    parts.append(part.replace("%%", "%").replace("$$", "$"))
                    value = value.lstrip()[end:]
            else:
                parts = shlex.split(value)
            if key == "ExecStart":
                argv = parts
            else:
                for part in parts:
                    name, separator, setting = part.partition("=")
                    if separator:
                        env[name] = setting
        return argv, env

    def uninstall(self):
        if self.activate and not self.available():
            # Nothing to stop; a unit file from an earlier try still goes.
            self.target.unlink(missing_ok=True)
            return
        if self.activate and self.target.exists():
            self.run(
                ["systemctl", "--user", "disable", "--now", self.target.name],
                check=True,
                capture_output=True,
            )
        self.target.unlink(missing_ok=True)
        if self.activate:
            self.run(
                ["systemctl", "--user", "daemon-reload"],
                check=True,
                capture_output=True,
            )

    def status(self):
        if not self.target.exists():
            return "uninstalled"
        if not self.activate:
            return "installed"
        return (
            "running"
            if self.run(
                ["systemctl", "--user", "is-active", self.target.name],
                capture_output=True,
            ).returncode
            == 0
            else "stopped"
        )


class TaskScheduler(OwnedService):
    def __init__(self, target, *, name=NAME, run=subprocess.run, activate=True):
        self.target, self.name, self.run, self.activate = (
            Path(target),
            name,
            run,
            activate,
        )

    def write_definition(self, argv, env):
        self.check_owner(env)
        # Task Scheduler has no per-action environment. A stdlib launcher applies
        # exactly this task's environment, quotes argv for CreateProcess, and stays
        # in the task's job so /End and RestartOnFailure track the service.
        import json
        import sys

        launcher = self.target.with_suffix(".py")
        script = (
            "import os, subprocess\n"
            + f"os.environ.update({dict(env)!r})\n"
            + f"argv = {list(argv)!r}\n"
            + "raise SystemExit(subprocess.call(argv, env=os.environ))\n"
        )
        atomic_write(launcher, script.encode())
        task = ET.Element(
            "Task",
            version="1.2",
            xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task",
        )
        triggers = ET.SubElement(task, "Triggers")
        ET.SubElement(ET.SubElement(triggers, "LogonTrigger"), "Enabled").text = "true"
        principals = ET.SubElement(task, "Principals")
        principal = ET.SubElement(principals, "Principal", id="Author")
        ET.SubElement(principal, "LogonType").text = "InteractiveToken"
        ET.SubElement(principal, "RunLevel").text = "LeastPrivilege"
        settings = ET.SubElement(task, "Settings")
        ET.SubElement(settings, "ExecutionTimeLimit").text = "PT0S"
        ET.SubElement(settings, "DisallowStartIfOnBatteries").text = "false"
        ET.SubElement(settings, "StopIfGoingOnBatteries").text = "false"
        restart = ET.SubElement(settings, "RestartOnFailure")
        # Schema minimum interval; Count is an unsigned byte (maximum 255).
        # https://learn.microsoft.com/windows/win32/taskschd/taskschedulerschema-count-restarttype-element
        ET.SubElement(restart, "Interval").text = "PT1M"
        ET.SubElement(restart, "Count").text = "255"
        actions = ET.SubElement(task, "Actions", Context="Author")
        action = ET.SubElement(actions, "Exec")
        ET.SubElement(action, "Command").text = sys.executable
        ET.SubElement(action, "Arguments").text = subprocess.list2cmdline(
            [str(launcher)]
        )
        # The JSON roundtrip rejects non-string/unserializable task input early.
        json.dumps(env)
        atomic_write(
            self.target, ET.tostring(task, encoding="utf-16", xml_declaration=True)
        )

    def install(self, argv, env):
        self.write_definition(argv, env)
        if self.activate:
            self.run(
                [
                    "schtasks",
                    "/Create",
                    "/TN",
                    self.name,
                    "/XML",
                    str(self.target),
                    "/F",
                ],
                check=True,
                capture_output=True,
            )
            self.run(
                ["schtasks", "/Run", "/TN", self.name], check=True, capture_output=True
            )
        return self.target

    def restart(self, expected):
        self.check_owner(expected)
        if self.activate and self.target.exists():
            # Refresh the registered action too, in case its Python was replaced.
            self.run(
                ["schtasks", "/End", "/TN", self.name],
                capture_output=True,
            )
            self.run(
                [
                    "schtasks",
                    "/Create",
                    "/TN",
                    self.name,
                    "/XML",
                    str(self.target),
                    "/F",
                ],
                check=True,
                capture_output=True,
            )
            self.run(
                ["schtasks", "/Run", "/TN", self.name],
                check=True,
                capture_output=True,
            )

    def definition(self):
        import ast

        task = ET.fromstring(self.target.read_bytes())
        arguments = task.find(".//{*}Exec/{*}Arguments")
        if arguments is None or arguments.text != subprocess.list2cmdline(
            [str(self.target.with_suffix(".py"))]
        ):
            raise ValueError("Task action no longer selects this launcher")
        # Read only literal data from our launcher; never execute a task file.
        tree = ast.parse(self.target.with_suffix(".py").read_text(encoding="utf-8"))
        env, argv = {}, []
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "argv" for t in node.targets
            ):
                argv = ast.literal_eval(node.value)
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if ast.unparse(call.func) == "os.environ.update":
                    env = ast.literal_eval(call.args[0])
        return argv, env

    def uninstall(self):
        if self.activate and self.target.exists():
            # "/End" fails when the task is not running; that is stopped.
            self.run(["schtasks", "/End", "/TN", self.name], capture_output=True)
            self.run(
                ["schtasks", "/Delete", "/TN", self.name, "/F"],
                check=True,
                capture_output=True,
            )
        self.target.unlink(missing_ok=True)
        self.target.with_suffix(".py").unlink(missing_ok=True)

    def status(self):
        if not self.target.exists():
            return "uninstalled"
        if not self.activate:
            return "installed"
        result = self.run(
            ["schtasks", "/Query", "/TN", self.name, "/XML"], capture_output=True
        )
        return "installed" if result.returncode == 0 else "stopped"
