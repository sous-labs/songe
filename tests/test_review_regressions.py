"""Regressions for workstream F's independent and live reviews."""

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from songe import agents, memory
from songe.api import LocalClient
from songe.platform import Accel, Memory
from tests.test_memory import Platform


def test_remote_installation_mismatch_never_sends_control(config, store, monkeypatch):
    sent = []

    def send(client, request, **kwargs):
        sent.append(request.method)
        return httpx.Response(
            200,
            request=request,
            json={
                "schema": 1,
                "installation_id": "another installation",
            },
        )

    monkeypatch.setattr(httpx.Client, "send", send)
    with pytest.raises(RuntimeError, match="installation"):
        LocalClient(store.root, config).control("forget", confirm=True)
    assert sent == ["GET"]


def test_repeated_init_only_reverses_owned_fields(config, tmp_path):
    home = tmp_path / "home"
    path = home / ".config/opencode/opencode.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"theme": "dark"}), encoding="utf-8")
    root = tmp_path / "root"
    root.mkdir()
    agents.apply(agents.proposed(config, home), root)
    current = json.loads(path.read_text(encoding="utf-8")) | {"theme": "light"}
    path.write_text(json.dumps(current), encoding="utf-8")
    config.port += 1
    agents.apply(agents.proposed(config, home), root)
    assert agents.restore(root) == []
    assert json.loads(path.read_text(encoding="utf-8")) == {"theme": "light"}


def test_apply_rechecks_reviewed_input(config, tmp_path):
    home = tmp_path / "home"
    path = home / ".pi/agent/models.json"
    path.parent.mkdir(parents=True)
    changes = agents.proposed(config, home)
    path.write_text('{"new": "user edit"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        agents.apply(changes, tmp_path / "root")
    assert json.loads(path.read_text(encoding="utf-8")) == {"new": "user edit"}


def test_failed_memory_callback_still_kills(monkeypatch):
    killed = []
    monkeypatch.setattr(memory, "kill", lambda process: killed.append(process))
    process = SimpleNamespace(returncode=None)

    def failed(sample):
        raise sqlite3.OperationalError("disk full")

    memory.guard(process, Platform([Memory(1000, 1, "critical")]), on_critical=failed)
    assert killed == [process]


def test_lost_accelerator_sensor_stops_guard(monkeypatch):
    class Lost(Platform):
        def accelerators(self):
            if self.index > 1:
                return []
            return [Accel("cuda", "GPU", 100, 90)]

    stopped = []
    memory.guard(
        SimpleNamespace(returncode=None),
        Lost([Memory(1000, 900)]),
        on_critical=lambda sample: stopped.append(sample),
        policy=memory.GuardPolicy(0.001, 0.001, 0.001),
    )
    assert len(stopped) == 1


def test_existing_linux_pressure_is_not_a_normal_baseline(tmp_path):
    from songe.platform.linux import Linux

    (tmp_path / "pressure").mkdir()
    (tmp_path / "meminfo").write_text(
        "MemTotal: 8000000 kB\nMemAvailable: 1 kB\n", encoding="utf-8"
    )
    (tmp_path / "pressure/memory").write_text("some avg10=90\n", encoding="utf-8")
    platform = Linux(proc=tmp_path)
    assert all(platform.memory().pressure != "normal" for _ in range(5))


def test_forget_invalidates_pending_candidate(store):
    night = store.night(None)
    store.night(night, status="ready", record={"candidate": {"id": "old"}})
    store.forget()
    assert store.nights()[0]["status"] == "discarded"


def test_live_discovery_skips_bad_models(tmp_path, monkeypatch, caplog):
    from songe.models import discovery

    (tmp_path / "broken.gguf").write_bytes(b"bad")
    (tmp_path / "good.gguf").write_bytes(b"ok")

    def read(path):
        if path.name == "broken.gguf":
            raise ValueError("unreadable header")
        return SimpleNamespace(architecture="llama")

    monkeypatch.setattr(discovery, "read_gguf", read)
    assert [r.path.name for r in discovery.lmstudio(folder=tmp_path)] == ["good.gguf"]
    assert "broken.gguf" in caplog.text


def test_base_install_discovers_real_gguf_without_torch(tmp_path):
    from songe.models.discovery import lmstudio
    from tests.crossplatform_helpers import write_gguf

    path = write_gguf(tmp_path / "base.gguf")
    found = lmstudio(folder=tmp_path)
    assert len(found) == 1 and found[0].path == path
    assert found[0].architecture == "llama"


async def test_ollama_refuses_to_overwrite_existing_foreign_weights(
    tmp_path, monkeypatch
):
    from songe.upstream import protection
    from songe.upstream.ollama import Ollama

    monkeypatch.setattr(protection, "current", lambda: Platform([Memory(1000, 900)]))
    path = tmp_path / "ours.gguf"
    path.write_bytes(b"ours")
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/api/show":
            return httpx.Response(
                200, json={"modelfile": "FROM /blobs/sha256-" + "0" * 64}
            )
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        pytest.fail("foreign model was modified")

    upstream = Ollama(
        SimpleNamespace(upstream_url="http://127.0.0.1/v1"),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(RuntimeError, match="other or unverifiable weights"):
        await upstream.load({"identifier": "existing", "artifact": str(path)})
    assert set(calls) == {"/api/show", "/api/ps"}
    assert not upstream.owned


def test_apple_accelerator_uses_chip_name(monkeypatch):
    from songe.platform import darwin

    monkeypatch.setattr("platform.machine", lambda: "arm64")
    monkeypatch.setattr("platform.processor", lambda: "arm")
    monkeypatch.setattr(darwin, "read", lambda *a: "Apple M4 Pro\n")
    monkeypatch.setattr(darwin.Darwin, "memory", lambda s: Memory(1000, 900))
    assert darwin.Darwin().accelerators()[0].name == "Apple M4 Pro"


def test_shared_backend_has_worker_protocol():
    from songe.backends import Backend
    from songe.backends.torch import Backend as Torch

    for name, method in Backend.__dict__.items():
        if callable(method) and not name.startswith("_"):
            assert hasattr(Torch, name), name


def test_config_environment_resolver_is_shared(tmp_path, monkeypatch):
    from songe.api import interface_data_dir
    from songe.cli import parser
    from songe.config import data_dir

    monkeypatch.setenv("SONGE_DATA_DIR", str(tmp_path))
    assert (
        interface_data_dir()
        == data_dir()
        == parser().parse_args([]).data_dir
        == tmp_path
    )
    monkeypatch.setenv("SONGE_DATA_DIR", "~/private-test")
    assert (
        interface_data_dir() == data_dir() == (Path.home() / "private-test").resolve()
    )


def test_configuration_round_trip_torch_options(config, tmp_path):
    from songe.config import Config

    config.torch_options = {"device": "cpu", "force_cpu": True}
    config.save(tmp_path)
    assert asdict(Config.load(tmp_path)) == asdict(config)


async def test_parking_never_unloads_an_unowned_model(tmp_path):
    from tests.test_lmstudio import FakeLMS

    upstream = FakeLMS(tmp_path, running=[{"identifier": "user", "path": "pub/model"}])
    assert await upstream.park() == []
    assert await upstream.loaded_bytes() == 0
    await upstream.unload({"identifier": "songe/test/night-1"})
    assert not any(call[0] == "unload" for call in upstream.calls)


def test_malformed_tool_segment_invalidates_entire_output():
    from songe.sleep.evaluate import health

    def parser(segment, tools):
        return json.loads(segment)

    tools = [
        {
            "function": {
                "name": "write",
                "parameters": {
                    "type": "object",
                    "required": ["path"],
                    "properties": {"path": {"type": "string"}},
                },
            }
        }
    ]
    tokenizer = SimpleNamespace(tool_parser=parser)
    text = '<tool_call>{"name":"write","arguments":{"path":"a"}}</tool_call><tool_call>broken</tool_call>'
    assert health(text, tokenizer, tools, True)["tool_valid"] == 0
    assert (
        health("no supported tool syntax", SimpleNamespace(), tools, True)[
            "tool_supported"
        ]
        == 0
    )


async def test_foreign_inference_origin_is_rejected(config, store):
    from songe.proxy import create_app

    def forbidden(request):
        pytest.fail("foreign-origin request reached model")

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(forbidden))
    app = create_app(config, store, client=upstream)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://attacker.example"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={},
            headers={"origin": "http://attacker.example"},
        )
        assert response.status_code == 403
    await upstream.aclose()


def test_teacher_hindsight_does_not_count_against_student_length(config):
    from songe.sleep.prompts import paired_prompts
    from tests.test_learning import Tokenizer

    config.max_sequence, config.response_tokens = 512, 128
    step = {
        "context": [{"role": "user", "content": "Question"}],
        "hindsight": [{"role": "user", "content": "x" * 2000}],
    }
    student, teacher = paired_prompts(Tokenizer(), step, config)
    assert b"Question" in bytes(student)
    assert len(student) + config.response_tokens <= config.max_sequence
    assert len(teacher) > 384


def test_late_recording_is_reconstructed_incrementally(config, store):
    from songe.experience import refresh

    context = [{"role": "user", "content": "Question"}]
    action = {"role": "assistant", "content": "wrong"}
    second = context + [action, {"role": "user", "content": "Actually fix it"}]
    store.record(
        "/v1/chat/completions",
        {"model": config.public_model, "messages": second},
        {"choices": [{"message": {"role": "assistant", "content": "fixed"}}]},
        started=2,
    )
    refresh(config, store)
    store.record(
        "/v1/chat/completions",
        {"model": config.public_model, "messages": context},
        {"choices": [{"message": action}]},
        started=1,
    )
    refresh(config, store)
    steps = store.active_steps()
    assert len(steps) == 2
    assert next(s for s in steps if s["action"] == action)["hindsight"]
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM scoreboard").fetchone()[0] == 1


@pytest.mark.parametrize(
    "messages",
    [
        None,  # {"messages": null}: the model server answers 400, it is recorded all the same
        "hi",
        ["hi"],
        # A lone surrogate, as an agent leaves when it cuts tool output inside
        # an emoji: valid JSON, but it cannot be encoded to hash it.
        [{"role": "user", "content": json.loads('"cut \\ud83d"')}],
    ],
)
def test_an_unreadable_recording_does_not_hold_back_the_ones_after_it(config, store, messages):
    store.record(
        "/v1/chat/completions",
        {"model": config.public_model, "messages": messages},
        {"error": {"message": "bad request"}},
        status=400,
        started=1,
    )
    context = [{"role": "user", "content": "Question"}]
    action = {"role": "assistant", "content": "answer"}
    store.record(
        "/v1/chat/completions",
        {"model": config.public_model, "messages": context},
        {"choices": [{"message": action}]},
        started=2,
    )
    # The unreadable one comes first in every batch until it is passed over.
    assert store.reconstruct() == (1, 2)
    assert [s["action"] for s in store.active_steps()] == [action]
    with store.connect() as db:
        marks = db.execute("SELECT resolved FROM reconstructed ORDER BY exchange_id").fetchall()
    assert [m[0] for m in marks] == [1, 1]
    assert store.reconstruct() == (0, 0)  # and it is not tried again


def test_diagnostics_preserve_safe_exception_locations():
    import logging

    from songe.telemetry import EventFormatter

    try:
        raise ValueError("private conversation secret")
    except ValueError:
        import sys

        record = logging.LogRecord(
            "songe", logging.ERROR, __file__, 1, "worker_failed", (), sys.exc_info()
        )
    data = json.loads(EventFormatter().format(record))
    assert data["exception"] == "ValueError"
    assert data["stack"]
    assert "private conversation secret" not in json.dumps(data)


def test_windows_proxy_battery_and_restart_policy(tmp_path):
    import xml.etree.ElementTree as ET

    from songe.platform.services import TaskScheduler

    target = tmp_path / "songe.xml"
    TaskScheduler(target, activate=False).install(["python", "-m", "songe"], {})
    document = ET.fromstring(target.read_bytes())
    ns = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
    assert document.find(".//t:DisallowStartIfOnBatteries", ns).text == "false"
    assert document.find(".//t:StopIfGoingOnBatteries", ns).text == "false"
    assert document.find(".//t:RestartOnFailure/t:Count", ns) is not None
    assert 1 <= int(document.find(".//t:RestartOnFailure/t:Count", ns).text) <= 255


def test_windows_uninstall_ends_task_before_deleting(tmp_path):
    import subprocess

    from songe.platform.services import TaskScheduler

    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    service = TaskScheduler(tmp_path / "songe.xml", run=run)
    service.target.write_text("installed", encoding="utf-8")
    service.uninstall()
    assert calls[0][1] == "/End"
    assert calls[1][1] == "/Delete"


def test_uninstall_waits_for_serve_lease_before_touching_data(store):
    from songe.cli import require_stopped
    from songe.platform.locking import acquire

    lease = (store.root / "serve.lock").open("a+")
    acquire(lease)
    try:
        with pytest.raises(RuntimeError, match="serve"):
            with require_stopped(store.root, timeout=0):
                pytest.fail("running proxy was not stopped")
    finally:
        lease.close()


def test_rejected_work_cleanup_preserves_active_and_foreign_work(config, store):
    from songe.sleep.cleanup import collect_work

    identity = store.get("installation_id")
    for status, owner in [
        ("rejected", identity),
        ("interrupted", identity),
        ("failed", "foreign"),
    ]:
        night = store.night(
            None,
            status=status,
            record={
                "generation": store.get("candidate_generation"),
                "configuration": __import__(
                    "songe.config", fromlist=["training_fingerprint"]
                ).training_fingerprint(config),
                "expected": store.get("current"),
                "base_fingerprint": __import__(
                    "songe.schedule", fromlist=["base_fingerprint"]
                ).base_fingerprint(config),
                "environment": store.get("environment_fingerprint"),
            },
        )
        directory = store.root / "versions" / f"night-{night}"
        directory.mkdir(parents=True)
        (directory / "songe_work.json").write_text(
            json.dumps({"night": night, "installation_id": owner}), encoding="utf-8"
        )
        (directory / "checkpoint").write_text("weights", encoding="utf-8")
        cache = store.root / "models" / f"derived-{night}"
        cache.mkdir(parents=True)
        (cache / "songe_conversion.json").write_text(
            json.dumps({"source": str(directory / "scale-1-artifact")}),
            encoding="utf-8",
        )
    collect_work(store, config)
    assert not (store.root / "versions/night-1").exists()
    assert (store.root / "versions/night-2/checkpoint").exists()
    assert (store.root / "versions/night-3/checkpoint").exists()
    assert not (store.root / "models/derived-1").exists()
    assert (store.root / "models/derived-2").exists()


def test_learned_artifact_permissions_are_restricted_before_publication(tmp_path):
    import os

    from songe.artifacts.runtime import publish

    if os.name == "nt":
        pytest.skip("POSIX mode assertion; Windows user profile ACL is inherited")
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o755)
    weight = stage / "weights"
    weight.write_bytes(b"private learning")
    weight.chmod(0o644)
    target = publish(stage, tmp_path / "learned")
    assert target.stat().st_mode & 0o777 == 0o700
    assert (target / "weights").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("failure", [None, "null_dacl", "denied"])
def test_windows_artifact_dacl_is_account_only_and_fails_closed(tmp_path, failure):
    from songe.storage import set_private_dacl

    calls = []

    class Function:
        def __init__(self, fn):
            self.fn = fn

        def __call__(self, *args):
            return self.fn(*args)

    def convert(sddl, revision, descriptor, size):
        calls.append(sddl)
        assert revision == 1 and sddl == "D:P(A;OICI;FA;;;S-1-5-21-123)"
        descriptor._obj.value = 100
        return True

    def get_dacl(descriptor, present, dacl, defaulted):
        present._obj.value = True
        dacl._obj.value = 0 if failure == "null_dacl" else 200
        return True

    def set_security(path, kind, flags, owner, group, dacl, sacl):
        assert kind == 1 and flags == 0x80000004 and dacl.value == 200
        return 5 if failure == "denied" else 0

    advapi = SimpleNamespace(
        ConvertStringSecurityDescriptorToSecurityDescriptorW=Function(convert),
        GetSecurityDescriptorDacl=Function(get_dacl),
        SetNamedSecurityInfoW=Function(set_security),
    )
    kernel = SimpleNamespace(
        LocalFree=Function(lambda descriptor: calls.append("freed"))
    )
    if failure:
        with pytest.raises(OSError):
            set_private_dacl(tmp_path, "S-1-5-21-123", (advapi, kernel))
    else:
        set_private_dacl(tmp_path, "S-1-5-21-123", (advapi, kernel))
    assert calls[-1] == "freed"


def test_calibration_does_not_invent_a_second_base_copy():
    from songe.sleep.calibrate import derive

    rows = [
        {
            "sequence": 256,
            "step_seconds": 1,
            "generation_tps": 50,
            "forward_seconds": 0.1,
            "peak_memory_gb": 6,
        }
    ]
    assert derive(rows, 10)["calibrated_peak_gb"] == 6
    rows[0]["phase_peak_memory_gb"] = 12
    with pytest.raises(ValueError, match="memory"):
        derive(rows, 10)
    assert derive(rows, 20)["calibrated_host_gb"] == 12


async def test_unscored_backlog_survives_experience_watermark(config, store):
    from songe.schedule import Scheduler

    config.min_train = config.min_heldout = config.min_new = 1
    for i, split in [(1, "heldout"), (51, "train")]:
        store.sync_steps(
            [
                {
                    "id": str(i),
                    "source_id": i,
                    "observed_id": i,
                    "conversation": str(i),
                    "split": split,
                    "tools": [],
                    "hindsight": [{"role": "user", "content": "fix"}],
                }
            ]
        )
    store.set("experience_watermark", 1000)
    store.set("score_watermark", 50)
    assert await Scheduler(config, store, None).enough()


async def test_undo_reports_deferred_publication(config, store, monkeypatch):
    from songe.schedule import Scheduler
    from songe.sleep import swap
    from tests.test_store import version

    for key in ("old", "new"):
        store.add_version(version(key, 1 if key == "old" else 2))
        store.swap(key, None if key == "old" else "old")

    async def deferred(*args, **kwargs):
        return False

    monkeypatch.setattr(swap, "install", deferred)
    with pytest.raises(RuntimeError, match="deferred"):
        await Scheduler(config, store, None).return_to(1)
    assert store.get("current") == "new"


async def test_lms_recovery_checks_weight_identity(tmp_path, monkeypatch):
    from songe.upstream import protection
    from tests.test_lmstudio import FakeLMS

    monkeypatch.setattr(protection, "current", lambda: Platform([Memory(1000, 900)]))
    upstream = FakeLMS(tmp_path, running=[{"identifier": "alias", "path": "wrong"}])
    upstream.config.upstream_url = "http://127.0.0.1/v1"

    async def models(client, url):
        return httpx.Response(
            200, request=httpx.Request("GET", url), json={"data": [{"id": "alias"}]}
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", models)
    with pytest.raises(RuntimeError, match="already serves"):
        await upstream.ensure(
            {"identifier": "alias", "artifact": str(tmp_path / "correct")}
        )
    assert not upstream.loaded


async def test_external_model_load_is_admitted_and_guarded_for_its_lifetime(
    monkeypatch,
):
    import asyncio

    from songe.upstream import protection

    platform = Platform([Memory(1000, 900)])
    monkeypatch.setattr(protection, "current", lambda: platform)
    released = asyncio.Event()

    class Runtime:
        config = SimpleNamespace(guard_min_interval=0.001)

        @protection.guarded_load
        async def load(self, version):
            protection.claim(self, version)

        async def unload(self, version):
            self.owned.pop(version["identifier"], None)
            released.set()

    runtime = Runtime()
    await runtime.load({"identifier": "ours", "artifact": "/missing"})
    # A completed instance serves the user's requests; memory pressure after
    # the load never ejects it (the live Mac ejected after every request).
    platform.samples[:] = [Memory(1000, 1, "critical")]
    await asyncio.sleep(0.05)
    assert "ours" in runtime.owned and not released.is_set()
    await protection.close_owned(runtime)


async def test_user_requested_load_skips_training_admission(monkeypatch):
    from songe.upstream import protection

    platform = Platform([Memory(1000, 1, "warn")])
    monkeypatch.setattr(protection, "current", lambda: platform)

    class Runtime:
        config = SimpleNamespace(guard_min_interval=0.001)

        @protection.guarded_load
        async def load(self, version):
            protection.claim(self, version)

        async def unload(self, version):
            self.owned.pop(version["identifier"], None)

    runtime = Runtime()
    await runtime.load({"identifier": "base", "artifact": "/missing", "serving": True})
    assert "base" in runtime.owned
    await protection.close_owned(runtime)


def test_json_schema_constraints_and_unresolvable_refs_fail():
    from songe.steps import valid_schema

    assert not valid_schema(
        {"count": -1},
        {"type": "object", "properties": {"count": {"type": "integer", "minimum": 0}}},
    )
    assert not valid_schema(
        "anything", {"$ref": "https://untrusted.invalid/schema.json"}
    )


async def test_generation_is_checked_inside_publication_transaction(
    config, store, monkeypatch
):
    from songe.sleep.swap import install
    from tests.test_store import version

    store.add_version(version("base"))
    store.swap("base", None)
    generation = store.get("candidate_generation")

    async def load(version):
        pass

    async def healthy(client, url, **kwargs):
        store.invalidate_pending()
        return httpx.Response(
            200, request=httpx.Request("POST", url), json={"choices": [{}]}
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", healthy)
    with pytest.raises(RuntimeError, match="generation"):
        await install(
            store,
            SimpleNamespace(load=load),
            config,
            version("new", 1),
            "base",
            generation=generation,
        )
    assert store.get("current") == "base"


def test_indivisible_export_finishes_past_deadline_but_honors_cancellation(store):
    from songe.backends import Interrupted
    from songe.sleep.worker import Journal

    expired = cancelled = False

    def check(deadline=True):
        if cancelled or (deadline and expired):
            raise Interrupted()

    journal = Journal(store.root, check, store, store.slice())

    def finish(urgent):
        nonlocal expired
        expired = True
        urgent()
        return {"complete": True}

    assert journal.indivisible("export", finish) == {"complete": True}
    # The committed result resumes without exporting again.
    assert journal.indivisible(
        "export", lambda urgent: pytest.fail("repeated export")
    ) == {"complete": True}
    expired = False

    def interrupted(urgent):
        nonlocal cancelled
        cancelled = True
        urgent()

    with pytest.raises(Interrupted):
        journal.indivisible("cancelled-export", interrupted)
    assert not (store.root / "cancelled-export.json").exists()


def test_a_unit_named_after_a_practice_moment_resumes(store):
    from songe.sleep.worker import Journal

    journal = Journal(store.root, lambda: None, store, store.slice())
    # The judged gate names its units after the moment: "practice:<scene>#v1".
    name = "judged-before-practice:sig3#v1"
    assert journal.unit(name, lambda: {"text": "answer"}) == {"text": "answer"}
    # Saved under a file name every system allows, and found there again.
    assert (store.root / "judged-before-practice_sig3#v1.json").exists()
    assert journal.unit(name, lambda: pytest.fail("answered again")) == {"text": "answer"}
    assert journal.load(name, None) == {"text": "answer"}


def test_scoring_backlog_with_equal_revision_is_not_skipped(store):
    steps = [
        {
            "id": str(i),
            "source_id": i,
            "observed_id": 100,
            "conversation": str(i),
            "split": "train",
            "tools": [],
            "hindsight": [{"role": "user", "content": "fix"}],
        }
        for i in range(60)
    ]
    store.sync_steps(steps)
    store.mark_scored(steps[:50])
    assert [s["id"] for s in store.unscored(steps)] == [str(i) for i in range(50, 60)]
    assert store.experience_counts()[2] == 10


def test_fuse_checks_free_disk_before_launching(config, store, monkeypatch, tmp_path):
    from songe.models import convert
    from songe.sleep import fuse

    config.model = str(tmp_path / "base")
    Path(config.model).mkdir()
    (Path(config.model) / "weights").write_bytes(b"weight bytes")
    monkeypatch.setattr(
        convert.shutil, "disk_usage", lambda path: SimpleNamespace(free=1)
    )
    monkeypatch.setattr(
        fuse,
        "guarded_run",
        lambda *a, **kw: pytest.fail("insufficient disk must not launch fusion"),
    )
    with pytest.raises(OSError, match="GB free.*GB available"):
        fuse.fuse(config, store.root / "adapter", tmp_path / "models/candidate")


def test_isolated_worker_reads_durable_result(config, store, monkeypatch):
    from songe.sleep.worker import isolated

    def guarded(argv, **kwargs):
        assert "--child" in argv
        assert "songe.sleep.worker" in argv
        store.night(None, status="ready", record={"candidate": {"id": "measured"}})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(memory, "guarded_run", guarded)
    assert isolated(config, store) == {"candidate": {"id": "measured"}}


def test_windows_collects_portable_orchestration_modules(monkeypatch):
    import platform_checks

    monkeypatch.setattr(platform_checks.sys, "platform", "win32")
    for name in (
        "test_cli.py",
        "test_schedule.py",
        "test_probe.py",
        "test_learning.py",
        "test_worker.py",
    ):
        assert platform_checks.module_unavailable(name) is None


def test_retired_controls_have_no_config_or_fusion_surface(config, tmp_path):
    import inspect
    from dataclasses import fields

    from songe import launchd
    from songe.sleep.fuse import fuse

    assert "surprise_floor" not in {f.name for f in fields(config)}
    assert "higher_precision" not in inspect.signature(fuse).parameters
    assert not hasattr(launchd, "plist")
    # Existing installations can still load the removed config key.
    config.save(tmp_path)
    path = tmp_path / "config.toml"
    path.write_text("surprise_floor = 0.001\n" + path.read_text(encoding="utf-8"))
    assert type(config).load(tmp_path) == config


def test_gate_rejects_unsupported_tool_evaluation():
    from songe.sleep.gate import decide

    before = [
        {
            "id": str(i),
            "conversation": str(i),
            "kl": 1,
            "drift": 0,
            "empty": 0,
            "repetition": 0,
            "tool_valid": 0,
            "tool_step": True,
            "tool_supported": 0,
        }
        for i in range(12)
    ]
    after = [r | {"kl": 0.5} for r in before]
    assert (
        "unsupported" in decide(before, after, min_heldout=12, drift_limit=0.05).reason
    )


async def test_inference_request_allocation_is_bounded(config, store, monkeypatch):
    from songe import proxy

    monkeypatch.setattr(proxy, "CAPTURE_LIMIT", 32)

    async def forbidden(request):
        pytest.fail("oversized request reached upstream")

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(forbidden))
    app = proxy.create_app(config, store, client=upstream)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        assert (
            await client.post("/v1/chat/completions", content=b"x" * 33)
        ).status_code == 413
    await upstream.aclose()


def test_reconstruction_never_materializes_full_log(config, store, monkeypatch):
    from songe.experience import refresh

    monkeypatch.setattr(
        store, "exchanges", lambda: pytest.fail("bulk log materialization")
    )
    refresh(config, store)


def test_service_removal_checks_selected_installation(config, store, tmp_path):
    from songe import launchd

    target = tmp_path / "service.plist"
    launchd.install(config, store.root, target, activate=False)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(RuntimeError, match="another data directory"):
        launchd.remove(target, activate=False, root=other)
    assert target.exists()


async def test_supervisor_never_terminates_a_discovered_user_process(
    config, tmp_path, monkeypatch
):
    from songe.upstream import supervisor

    user = SimpleNamespace(terminate=lambda: pytest.fail("signalled a user process"))
    monkeypatch.setattr(supervisor, "listening_server", lambda config: user)
    await supervisor.LlamaServer(config, tmp_path).stop()


async def test_mlx_parking_releases_only_its_owned_server(
    config, tmp_path, monkeypatch
):
    from songe.upstream.mlx import MLX

    events = []

    class Process:
        returncode = None

        async def wait(self):
            events.append("released")
            self.returncode = 0

    monkeypatch.setattr(memory, "terminate", lambda process: events.append("stop"))
    upstream = MLX(config, tmp_path)
    upstream.process = Process()
    upstream.version = {"identifier": "ours", "artifact": "/base"}
    parked = await upstream.park()
    assert events == ["stop", "released"]
    assert upstream.process is None and parked[0]["identifier"] == "ours"


def test_owned_command_group_stops_after_guard_parent_dies(tmp_path):
    import os
    import subprocess
    import sys
    import time

    if os.name == "nt":
        pytest.skip("POSIX group/parent-death integration check")
    path = tmp_path / "heartbeat"
    command = "import os,sys,time; from pathlib import Path; p=Path(sys.argv[1]); p.write_text(str(os.getpgrp()), encoding='utf-8');\nwhile True: p.with_suffix('.tick').write_text(str(time.monotonic()), encoding='utf-8'); time.sleep(.01)"
    parent_code = "import sys; from songe.memory import guarded_run; from songe.platform import Memory; from types import SimpleNamespace; p=SimpleNamespace(name='test',memory=lambda:Memory(10**9,9*10**8),accelerators=lambda:[]); guarded_run([sys.executable,'-c',sys.argv[1],sys.argv[2]],platform=p)"
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_code, command, str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    group = None
    try:
        deadline = time.monotonic() + 5
        while not path.with_suffix(".tick").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert path.with_suffix(".tick").exists()
        group = int(path.read_text(encoding="utf-8"))
        parent.terminate()
        parent.wait(timeout=5)
        time.sleep(0.3)
        last = path.with_suffix(".tick").read_text(encoding="utf-8")
        time.sleep(0.2)
        assert path.with_suffix(".tick").read_text(encoding="utf-8") == last
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.communicate(timeout=5)
        if group is not None:
            memory.kill(SimpleNamespace(pid=group))


def test_denied_library_folder_is_reported_not_empty(tmp_path, monkeypatch):
    import os

    import songe.models.discovery as discovery

    def denied(path):
        raise PermissionError(1, "Operation not permitted", str(path))

    monkeypatch.setattr(os, "scandir", denied)
    with pytest.raises(PermissionError, match="Privacy & Security"):
        discovery.lmstudio(folder=tmp_path)


def test_denied_folder_opens_privacy_settings_once_on_macos(monkeypatch):
    from songe import cli

    calls = []
    monkeypatch.setattr(cli.open_privacy_settings, "done", False, raising=False)
    cli.open_privacy_settings("linux", lambda argv, **kw: calls.append(argv))
    assert calls == []
    for _ in range(2):
        cli.open_privacy_settings("darwin", lambda argv, **kw: calls.append(argv))
    assert calls == [["open", cli.PRIVACY_PANE]]


def test_opencode_ui_selection_wins_over_config_default(tmp_path, monkeypatch):
    from songe.cli import detect_agent_binding

    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    config = tmp_path / ".config/opencode/opencode.json"
    config.parent.mkdir(parents=True)
    config.write_text(
        '{"model": "lmstudio/lfm3-1.7b", "small_model": "lmstudio/lfm3-1.7b"}',
        encoding="utf-8",
    )
    assert detect_agent_binding(tmp_path)[0] == "lfm3-1.7b"
    state = tmp_path / ".local/state/opencode/model.json"
    state.parent.mkdir(parents=True)
    state.write_text(
        '{"recent": [{"providerID": "lmstudio", "modelID": "qwen3.5-4b-mlx"},'
        ' {"providerID": "lmstudio", "modelID": "lfm3-1.7b"}]}',
        encoding="utf-8",
    )
    assert detect_agent_binding(tmp_path)[0] == "qwen3.5-4b-mlx"


async def test_serving_load_is_never_cancelled_by_memory_watch(monkeypatch):
    import asyncio

    from songe.upstream import protection

    platform = Platform([Memory(1000, 1, "critical")])
    monkeypatch.setattr(protection, "current", lambda: platform)

    class Runtime:
        config = SimpleNamespace(guard_min_interval=0.001)

        @protection.guarded_load
        async def load(self, version):
            await asyncio.sleep(0.05)  # weights loading while memory is critical
            protection.claim(self, version)

        async def unload(self, version):
            self.owned.pop(version["identifier"], None)

    runtime = Runtime()
    await runtime.load({"identifier": "base", "artifact": "/missing", "serving": True})
    assert "base" in runtime.owned
