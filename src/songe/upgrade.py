"""Bring an owned service up to the CLI release, without loading training code."""

import json
import os
import subprocess
import time
from email.parser import Parser
from pathlib import Path

import httpx
from packaging.version import Version

from . import __version__
from .api import ReadOnlyStore
from .backends.install import package_requirement, runtime_python
from .config import Config
from .platform import current
from .platform.locking import acquire
from .storage import atomic_write

PENDING = "service-upgrade.json"
FAILED_RETRY_SECONDS = 600
_checked = {}


def install_token(path=None):
    """A running process retains this token when its files are replaced in place."""
    path = path or Path(__file__).with_name("__init__.py")
    try:
        info = path.stat()
        return f"{path.resolve()}:{info.st_mtime_ns}:{info.st_size}"
    except OSError:
        return "missing"


RUNNING_INSTALL = install_token()


def notice(root, *, version=None, token=None):
    """Cheap snapshot notice; no subprocesses, network, or GPU imports."""
    try:
        pending = json.loads((Path(root) / PENDING).read_text(encoding="utf-8"))
        if pending.get("notice"):
            if (
                version == pending.get("version")
                and version is not None
                and not pending.get("runtime_pending", True)
                and token == install_token()
            ):
                return None  # A delayed restart recovered after the CLI exited.
            return pending["notice"]
    except (OSError, ValueError, AttributeError):
        pass
    if version is not None and token is not None and token != install_token():
        return f"Service {version} is using a replaced install · fix: songe doctor"
    return None


def runtime_version(root, run):
    directory = root / "runtime"
    if directory.is_symlink():
        raise RuntimeError("runtime is outside this installation")
    if not (directory / "pyvenv.cfg").is_file():
        raise RuntimeError("runtime is not a private Python environment")
    python = runtime_python(root)
    if not python.is_file():
        raise RuntimeError("runtime Python is unavailable")
    if not python.parent.resolve().is_relative_to(directory.resolve()):
        raise RuntimeError("runtime scripts are outside this installation")
    versions = set()
    for pattern in (
        "lib/python*/site-packages/songe-*.dist-info/METADATA",
        "Lib/site-packages/songe-*.dist-info/METADATA",
    ):
        for metadata in directory.glob(pattern):
            if not metadata.resolve().is_relative_to(directory.resolve()):
                raise RuntimeError("runtime packages are outside this installation")
            headers = Parser().parsestr(metadata.read_text(encoding="utf-8"))
            if headers.get("Name", "").lower() == "songe":
                versions.add(headers["Version"])
    if len(versions) == 1:
        return versions.pop()
    # Nonstandard venv layouts still use only stdlib distribution metadata.
    result = run(
        [
            str(python),
            "-I",
            "-c",
            "from importlib.metadata import version; print(version('songe'))",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
    )
    return result.stdout.strip()


def runtime_token(root):
    for pattern in (
        "lib/python*/site-packages/songe/__init__.py",
        "Lib/site-packages/songe/__init__.py",
    ):
        for path in (root / "runtime").glob(pattern):
            return install_token(path)
    return None  # Editable/nonstandard installs can still report replacement.


def service_state(config, identity):
    try:
        with httpx.Client(trust_env=False, timeout=1) as client:
            response = client.get(f"http://127.0.0.1:{config.port}/api/state?service=1")
            response.raise_for_status()
            data = response.json()
    except (httpx.ConnectError, httpx.ConnectTimeout):
        return None
    except httpx.ReadTimeout:
        # A slow answer means a busy service: never restart it under load.
        # Its release is unknown, and no agent request is known to be in
        # progress (#71: "waits for the agent request in progress" repeated
        # while the service already ran this release with none in flight).
        # The next command tries again.
        return {"service_running": True, "service_version": None, "busy": True}
    if data.get("installation_id") != identity:
        raise RuntimeError("service belongs to another installation")
    return data


def catch_up(
    root,
    config=None,
    *,
    service=None,
    run=None,
    read_state=None,
    health_timeout=30,
    busy_timeout=0,
    emit=print,
):
    """Best effort, once per process/data directory; failure never blocks a CLI."""
    root = Path(root).resolve()
    key = (root, __version__)
    if key in _checked:
        return _checked[key]
    _checked[key] = None
    started = False
    owned = False
    pending_data = {"version": __version__, "runtime_pending": True}
    try:
        config = config or Config.load(root)
        if not (root / "config.toml").is_file() or not (
            config.model or config.exposed_models
        ):
            return None
        identity = ReadOnlyStore(root).get("installation_id")
        if not identity:
            return None
        service = service or current().service()
        if not service.target.exists():
            return None
        expected = {"SONGE_INSTALLATION_ID": identity, "SONGE_DATA_DIR": str(root)}
        try:
            service.check_owner(expected)
        except RuntimeError:
            # The login service serves another data directory (or none of
            # ours): this installation has no service to update.
            return None
        owned = True
        # A restart that just failed is not tried again by every command: the
        # service also moves to new code by itself in a quiet moment.
        try:
            failed = json.loads((root / PENDING).read_text(encoding="utf-8"))
            if (
                failed.get("version") == __version__
                and time.time() - failed.get("failed_at", 0) < FAILED_RETRY_SECONDS
            ):
                _checked[key] = failed.get("notice")
                return _checked[key]
        except (OSError, ValueError, AttributeError):
            pass
        run = run or subprocess.run
        read_state = read_state or (lambda: service_state(config, identity))
        with (root / "service-upgrade.lock").open("a+", encoding="utf-8") as lease:
            try:
                acquire(lease)
            except BlockingIOError:
                return None
            argv, env = service.definition()
            # Never adopt a same-name service that runs another application.
            if argv[1:3] != ["-m", "songe"] or argv[-1:] != ["serve"]:
                raise RuntimeError("service command cannot be identified")
            private = (root / "runtime").exists()
            runtime = runtime_version(root, run) if private else None
            pending_data["runtime_pending"] = runtime is not None and Version(
                runtime
            ) != Version(__version__)
            update_runtime = runtime is not None and Version(runtime) < Version(
                __version__
            )
            state = read_state()
            running_version = (state or {}).get("service_version")
            target_token = runtime_token(root) if private else RUNNING_INSTALL
            if running_version is not None and Version(running_version) > Version(
                __version__
            ):
                raise RuntimeError(
                    f"service {running_version} is newer than CLI {__version__}"
                )
            old_service = state is not None and (
                running_version is None
                or Version(running_version) < Version(__version__)
                or (
                    target_token is not None
                    and state.get("service_install") != target_token
                )
                or state.get("service_replaced") is True
            )
            from .launchd import stable_python

            python = str(runtime_python(root)) if private else stable_python()
            rebind = os.path.abspath(argv[0]) != os.path.abspath(python)
            # Rewrite a launcher an older release wrote, even at the same Python.
            rewrite = rebind or getattr(service, "stale_launcher", lambda: False)()
            if (state or {}).get("busy"):
                # It did not answer in time, so its release is unknown: say
                # nothing unless this install itself must change under it.
                if update_runtime or rewrite:
                    emit(f"Update to {__version__} waits until the service is less busy")
                return None
            pending = root / PENDING
            retry = pending.exists()
            if not (update_runtime or old_service or rewrite):
                if runtime is not None and Version(runtime) != Version(__version__):
                    raise RuntimeError(
                        f"runtime {runtime} differs from CLI {__version__}"
                    )
                if not retry or state is not None:
                    pending.unlink(missing_ok=True)
                    return None
            if runtime is not None and Version(runtime) > Version(__version__):
                raise RuntimeError(f"runtime {runtime} is newer than CLI {__version__}")
            # Never restart under an agent request; the next command retries.
            waited = time.monotonic() + busy_timeout
            while (
                (state or {}).get("requests_in_flight") or (state or {}).get("busy")
            ) and time.monotonic() < waited:
                time.sleep(1)
                state = read_state()
            if (state or {}).get("requests_in_flight"):
                emit(f"Update to {__version__} waits for the agent request in progress")
                return None
            if (state or {}).get("busy"):
                emit(f"Update to {__version__} waits until the service is less busy")
                return None
            mismatch = (
                f"Service release differs from CLI {__version__} · fix: songe doctor"
            )
            pending_data["notice"] = mismatch
            atomic_write(pending, json.dumps(pending_data).encode("utf-8"))
            emit(f"Updating Songe service to {__version__}…", end=" ", flush=True)
            started = True
            if update_runtime:
                run(
                    [python, "-m", "pip", "install", package_requirement()],
                    check=True,
                    capture_output=True,
                    timeout=600,
                    env=os.environ
                    | {"PIP_NO_INPUT": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
                )
                if runtime_version(root, run) != __version__:
                    raise RuntimeError("runtime update did not install the CLI release")
                pending_data["runtime_pending"] = False
                atomic_write(pending, json.dumps(pending_data).encode("utf-8"))
            if rewrite:
                argv[0] = python
                # launchd caches its loaded definition; reload before kickstart.
                from .platform.services import Launchd

                if isinstance(service, Launchd):
                    service.install(argv, env)
                else:
                    service.write_definition(argv, env)
            service.restart(expected)
            deadline = time.monotonic() + health_timeout
            while True:
                try:
                    healthy = read_state()
                    target_token = runtime_token(root) if private else RUNNING_INSTALL
                    if (
                        healthy
                        and healthy.get("service_running") is True
                        and healthy.get("service_version") == __version__
                        and (
                            target_token is None
                            or healthy.get("service_install") == target_token
                        )
                        and (
                            not state
                            or not state.get("service_instance")
                            or healthy.get("service_instance")
                            != state["service_instance"]
                        )
                    ):
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError("service did not become healthy")
                time.sleep(0.2)
            pending.unlink(missing_ok=True)
            emit("✓ Updated")
    except Exception as exc:
        # Keep subprocess diagnostics (potentially enormous/private) off the CLI.
        detail = (
            type(exc).__name__
            if isinstance(exc, subprocess.SubprocessError)
            else str(exc)
        )
        detail = " ".join(detail.splitlines())
        message = f"Service update failed: {detail} · fix: songe doctor"
        _checked[key] = message
        if owned:
            try:
                pending_data["notice"] = message
                if started:
                    pending_data["failed_at"] = time.time()
                atomic_write(root / PENDING, json.dumps(pending_data).encode("utf-8"))
            except OSError:
                pass
        emit(("Failed · " + detail + " · fix: songe doctor") if started else message)
        return message
    return None


def main():
    """Noninteractive package-manager hook; uninitialized installs are a no-op."""
    from .api import interface_data_dir

    catch_up(interface_data_dir())


if __name__ == "__main__":
    main()
