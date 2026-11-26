import os
import stat

import httpx
import pytest

from songe.config import Config
from songe.log import Store


@pytest.fixture
def writable_file_fsync(monkeypatch):
    """Exercise Windows' writable-file requirement on POSIX hosts too."""
    fsync = os.fsync

    def flush(fd):
        if stat.S_ISREG(os.fstat(fd).st_mode):
            os.write(fd, b"")  # Reject read-only descriptors without changing data.
        fsync(fd)

    monkeypatch.setattr(os, "fsync", flush)


# Default ports of real model servers and of a running Songe. A developer's
# machine has them open, and a test that reaches one runs prompts on the
# model they are using (it did: review requests queued on a loaded model).
REAL_SERVER_PORTS = {1234, 11434, 8080, 8000, 7613}


@pytest.fixture(autouse=True)
def no_real_model_server(monkeypatch):
    """Requests to a real model server's port fail at once, as on a machine
    without one; fakes and in-process transports are untouched."""

    def refuse(request):
        if request.url.port in REAL_SERVER_PORTS:
            raise httpx.ConnectError(
                f"tests never contact a real model server ({request.url})",
                request=request,
            )

    sync, parallel = (
        httpx.HTTPTransport.handle_request,
        httpx.AsyncHTTPTransport.handle_async_request,
    )

    def handle(self, request):
        refuse(request)
        return sync(self, request)

    async def handle_async(self, request):
        refuse(request)
        return await parallel(self, request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle_async)


class MemoryKeychain:
    """A keychain in memory: tests never read or write the real one."""

    def __init__(self):
        self.items = {}

    def get_password(self, service, account):
        return self.items.get((service, account))

    def set_password(self, service, account, value):
        self.items[(service, account)] = value

    def delete_password(self, service, account):
        self.items.pop((service, account), None)


@pytest.fixture(autouse=True)
def no_real_cloud(monkeypatch):
    """No test reaches a cloud provider or the developer's keychain, and no
    key in the developer's environment leaks in."""
    from songe.cloud import client, keys

    def refuse(request):
        raise httpx.ConnectError(
            f"tests never contact a cloud provider ({request.url})", request=request
        )

    monkeypatch.setattr(client, "TRANSPORT", httpx.MockTransport(refuse))
    monkeypatch.setattr(keys, "BACKEND", MemoryKeychain())
    monkeypatch.delenv("SONGE_CLOUD_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def host_battery_unread(monkeypatch):
    """The worker reads the battery of the machine it runs on. On an unplugged
    laptop a test Dream stopped at once ("battery low": the battery lasts less
    than a slice), so the machine running the tests reports no power telemetry,
    as a desktop does. A platform a test gives the worker still counts."""
    from songe.platform import Power, current
    from songe.sleep import worker

    host, read = type(current()), worker.power_status
    monkeypatch.setattr(
        worker, "power_status", lambda p: Power() if isinstance(p, host) else read(p)
    )


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    # Interface tests must never inspect the developer's model library or
    # contact a model server. Inventory protocol tests install explicit fakes.
    from songe.models import library

    monkeypatch.setattr(library, "scan", lambda config: [])
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "home/AppData/Local"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "home/.local/share"))
    monkeypatch.setenv("SONGE_DATA_DIR", str(tmp_path / "songe"))
    monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "data")


@pytest.fixture
def config(tmp_path):
    return Config(
        model=str(tmp_path / "local/model"),
        public_model="songe/test",
        base_identifier="base-test",
        max_sequence=2048,  # fixtures were sized for this; default follows context
        # The real-steps path most worker tests were written for; dreams tests
        # (test_reading, test_practice, test_dreams_cycle) set their own.
        learning="hindsight",
    )


@pytest.fixture
def pipeline_data():
    return {
        "day": {"conversations": 38, "steps": 112, "heldout": 12},
        "rewind": {
            "signals": {
                "tool_error_fixed": 5,
                "user_correction": 4,
                "retry_success": 3,
                "loop": 2,
                "static_invalid": 1,
                "tool_error": 0,
                "exec_failure": 0,
                "later_success": 0,
                "retry": 0,
                "approval": 0,
            },
            "lessons": 6,
        },
        "consolidate": {
            "runs": 3,
            "step": 41,
            "total": 120,
            "eta_seconds": 180,
            "last_seconds": 95,
        },
        "wake": {
            "last": {
                "at": 1794354000,
                "repeat_errors_delta": -0.18,
                "drift": 0.002,
                "swapped": True,
            }
        },
        "versions": [
            {"id": "v3", "current": True, "kind": "full"},
            {"id": "v2", "current": False, "kind": "adapter"},
            {"id": "base", "current": False, "kind": "base"},
        ],
    }
