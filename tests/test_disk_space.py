"""Fake libraries and volumes only; no servers, runtimes, or user models."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from songe.artifacts import storage
from songe.config import Config
from songe.log import Store
from songe.models import ModelRef, registry
from songe.platform import Accel, Memory
from songe.storage import write_json

GB = 10**9


@pytest.fixture
def volumes(tmp_path, monkeypatch):
    internal, external = tmp_path / "internal", tmp_path / "external"
    internal.mkdir()
    external.mkdir()
    free = {internal: 900_000_000, external: 46 * GB}

    def volume(path):
        path = Path(path).resolve()
        return external if path.is_relative_to(external) else internal

    monkeypatch.setattr(storage, "volume", volume)
    monkeypatch.setattr(
        storage.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=free[volume(path)]),
    )
    store = Store(internal / "data")
    library = external / "local-llm/models"
    library.mkdir(parents=True)
    model = library / "publisher/base"
    model.mkdir(parents=True)
    (model / "config.json").write_text('{"model_type":"llama"}', encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"fake weights")
    config = Config(
        model=str(model),
        models_dir=str(library),
        backend="torch",
        artifact_kind="mlx-fused",
        base_identifier="fixture",
        model_ref={
            "upstream": "lmstudio",
            "serving_id": "fixture",
            "path": str(model),
            "format": "safetensors",
            "architecture": "llama",
        },
    )
    return internal, external, free, store, config


def test_preflight_sums_shared_volumes_and_checks_each_target(volumes):
    internal, external, free, _, _ = volumes
    with pytest.raises(storage.DiskSpace) as failure:
        storage.preflight(
            [(internal / "new", 2 * GB), (internal / "other", 1_200_000_000)]
        )
    assert str(failure.value) == f"Needs 3.2 GB free on {internal}; 0.9 GB available"
    storage.preflight([(external / "new", 3_200_000_000)])
    with pytest.raises(storage.DiskSpace, match=re.escape(str(internal))):
        storage.preflight([(external, 3_200_000_000), (internal, GB)])
    free[external] = GB
    with pytest.raises(storage.DiskSpace, match=re.escape(str(external))):
        storage.preflight([(external, 3_200_000_000), (internal, 1)])


@pytest.mark.parametrize("upstream", ["lmstudio", "ollama", "mlx", "vllm", "llamacpp"])
def test_artifact_volume_and_owned_cleanup(volumes, upstream):
    _, external, _, store, config = volumes
    config.upstream = upstream
    if upstream == "ollama":
        config.ollama_models_dir = config.models_dir
    scratch = storage.artifact_root(config, store)
    assert scratch.is_relative_to(external)
    if upstream in {"lmstudio", "ollama"}:
        assert not scratch.is_relative_to(Path(config.models_dir))
    assert (scratch / storage.MARKER).is_file()
    assert list(storage.artifact_roots(store.root)) == [store.root, scratch]
    assert store.path.parent == store.root
    (scratch / "large").write_text("ours", encoding="utf-8")
    foreign = scratch.with_name(".songe-foreign")
    foreign.mkdir()
    (foreign / "private").write_text("user", encoding="utf-8")
    storage.purge_roots(store)
    assert not scratch.exists()
    assert (foreign / "private").read_text(encoding="utf-8") == "user"
    assert Path(config.model).exists()
    assert store.path.exists()


def test_unmarked_collision_and_replaced_marker_are_preserved(volumes):
    _, _, _, store, config = volumes
    target = storage.artifact_root(config, store, create=False)
    target.mkdir()
    (target / "user").write_text("keep", encoding="utf-8")
    with pytest.raises(OSError, match="not owned"):
        storage.artifact_root(config, store)
    write_json(store.root / "artifact-roots.json", [str(target)])
    storage.purge_roots(store)
    assert (target / "user").read_text(encoding="utf-8") == "keep"


def test_conversion_and_lmstudio_publication_use_unindexed_volume(volumes, monkeypatch):
    from songe.models import convert
    from songe.platform import Rooted
    from songe.sleep import fuse

    _, _, _, store, config = volumes
    scratch = storage.artifact_root(config, store)
    source = Path(config.model).parent / "base.gguf"
    source.write_bytes(b"fake gguf")
    ref = ModelRef("lmstudio", "fixture", "gguf", source, "llama")
    monkeypatch.setattr(convert, "read_gguf", lambda p: SimpleNamespace(parameters=100))
    platform = Rooted(SimpleNamespace(name="test"), store.root)

    def converting(argv, **kwargs):
        out = Path(argv[-1])
        assert out.is_relative_to(scratch)
        assert not out.is_relative_to(Path(config.models_dir))
        (out / "config.json").write_text("{}", encoding="utf-8")
        (out / "model.safetensors").write_bytes(b"dense")

    converted = convert.trainable_base(
        ref, "torch", platform, runner=converting, artifact_directory=scratch
    )
    assert converted.is_relative_to(scratch / "models")
    work = scratch / "versions/night-1"
    work.mkdir(parents=True)
    write_json(
        work / "songe_work.json", {"installation_id": store.get("installation_id")}
    )
    adapter = work / "adapter"
    adapter.mkdir()
    (adapter / "adapters.safetensors").write_bytes(b"adapter")
    destination = Path(config.models_dir) / "songe/candidate"

    def fusing(argv, **kwargs):
        out = Path(argv[argv.index("--save-path") + 1])
        assert out.is_relative_to(scratch)
        assert not destination.exists()
        assert not any(Path(config.models_dir).rglob("songe_fusing.json"))
        (out / "model.safetensors").write_bytes(b"fused")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(fuse, "guarded_run", fusing)
    assert (
        fuse.export_candidate(
            config, "mlx-fused", adapter, destination, platform, lambda: None
        )
        == destination
    )
    assert (destination / "model.safetensors").read_bytes() == b"fused"
    assert not any(scratch.glob(".candidate-*"))
    from songe.upstream.cleanup import purge_library

    purge_library(config, store)
    assert not destination.exists()
    assert Path(config.model).exists()


def test_calibration_staging_and_abandoned_cleanup_are_external(volumes, monkeypatch):
    from songe import backends, platform
    from songe.sleep import calibrate, measurement
    from tests.test_learning import Tokenizer

    _, _, _, store, config = volumes
    scratch = storage.artifact_root(config, store)
    config.artifact_kind = "mlx-adapter"
    config.response_tokens = config.token_chunk = config.accumulation = 1
    hardware = SimpleNamespace(
        name="test",
        memory=lambda: Memory(24 * GB, 20 * GB),
        accelerators=lambda: [],
        on_ac_power=lambda: True,
        idle_seconds=lambda: None,
        thermal=lambda: None,
    )
    monkeypatch.setattr(platform, "current", lambda: hardware)
    monkeypatch.setattr(measurement, "process_tree_bytes", lambda *a: 100)
    calls = []

    class Backend:
        def __init__(self, config, check):
            self.config, self.tokenizer = config, Tokenizer()

        def prepare(self):
            pass

        def reset_peak_memory(self):
            pass

        def peak_memory(self):
            return 100

        def sample(self, *a, **kw):
            return [1], "Read", SimpleNamespace(generation_tps=10, prompt_tps=20)

        def compare(self, *a):
            pass

        def train(self, pool, anchors, work, progress):
            calls.append(work)
            assert work.is_relative_to(scratch)
            assert (work.parent / "songe_calibration.json").is_file()
            work.mkdir()
            (work / "checkpoint").write_text("temporary", encoding="utf-8")

    monkeypatch.setattr(backends, "choose", lambda *a: Backend)
    assert calibrate._calibrate(config, store.root, ())
    assert calls and not any(scratch.glob("calibration-*"))
    for name, identity in [("own", store.get("installation_id")), ("foreign", "other")]:
        work = scratch / ("calibration-" + name)
        work.mkdir()
        write_json(work / "songe_calibration.json", {"installation_id": identity})
    calibrate.collect_calibration(store.root, store)
    assert not (scratch / "calibration-own").exists()
    assert (scratch / "calibration-foreign").exists()


def test_forget_cleans_external_evidence_and_keeps_served_weights(volumes):
    from songe.api import forget_experience

    _, _, _, store, config = volumes
    config.save(store.root)
    scratch = storage.artifact_root(config, store)
    night = store.night(None)
    work = scratch / f"versions/night-{night}"
    adapter = work / "adapter"
    adapter.mkdir(parents=True)
    write_json(
        work / "songe_work.json",
        {"night": night, "installation_id": store.get("installation_id")},
    )
    (adapter / "weights").write_text("retained", encoding="utf-8")
    (work / "snapshot.json").write_text("private evidence", encoding="utf-8")
    store.add_version(
        {
            "id": "night-1",
            "night": night,
            "created": 1,
            "identifier": "songe/test",
            "artifact": str(adapter),
            "adapter": str(adapter),
            "training_adapter": str(adapter),
        }
    )
    store.swap("night-1", None)
    forget_experience(store)
    assert not (work / "snapshot.json").exists()
    assert (adapter / "weights").read_text(encoding="utf-8") == "retained"
    assert Path(config.model).exists()


def test_init_explicit_model_never_converts_or_calibrates(volumes, monkeypatch):
    from songe import agents, cli, models, platform
    from songe.backends import install
    from songe.models import discovery
    from songe.sleep import calibrate
    from songe.upstream import lmstudio

    _, _, _, store, config = volumes
    hardware = SimpleNamespace(
        accelerators=lambda: [Accel("cuda", "fake")],
        service=lambda: SimpleNamespace(available=lambda: True),
    )
    monkeypatch.setattr(platform, "current", lambda: hardware)
    monkeypatch.setattr(cli, "detect_agent_binding", lambda: ("", ""))
    monkeypatch.setattr(lmstudio, "find_lms", lambda: "fake-lms")
    monkeypatch.setattr(
        discovery, "lmstudio_downloads", lambda: Path(config.models_dir)
    )
    ref = ModelRef(**(config.model_ref | {"path": Path(config.model)}))
    monkeypatch.setattr(discovery, "lmstudio", lambda **kw: [ref])
    monkeypatch.setattr(discovery, "loaded_lmstudio", lambda refs, *a: refs)
    monkeypatch.setattr(install, "plan", lambda *a: [])
    monkeypatch.setattr(install, "conversion_plan", lambda *a: [])
    monkeypatch.setattr(cli.shutil, "which", lambda *a: None)
    monkeypatch.setattr(agents, "proposed", lambda *a, **kw: [])
    monkeypatch.setattr(cli, "finish_initialization", lambda *a: None)

    def forbidden(*a, **kw):
        raise AssertionError("init must defer heavy model work to the first idle gap")

    monkeypatch.setattr(calibrate, "calibrate", forbidden)
    monkeypatch.setattr(calibrate, "_calibrate", forbidden)
    monkeypatch.setattr(models, "trainable_base", forbidden)
    assert (
        cli.main(
            [
                "--data-dir",
                str(store.root),
                "init",
                "--model",
                config.model,
                "--upstream",
                "lmstudio",
                "--yes",
            ]
        )
        == 0
    )
    saved = Config.load(store.root)
    assert not saved.calibration_available
    assert registry.entries(store)[0]["support"] == "pending"
    assert not (store.root / "calibration.json").exists()


async def test_first_idle_gap_preflights_then_measures_and_retries(
    volumes, monkeypatch
):
    from songe import schedule
    from songe.api import snapshot
    from tests.test_schedule import good

    _, external, free, store, config = volumes
    config.save(store.root)
    registry.migrate(config, store)
    hardware = SimpleNamespace(
        name="test",
        memory=lambda: Memory(24 * GB, 20 * GB),
        accelerators=lambda: [Accel("cuda", "fake", 8 * GB, 8 * GB)],
        keep_awake=lambda pid: None,
    )
    parked, launched = [], []

    async def park(on_snapshot):
        parked.append(True)
        on_snapshot([])

    scheduler = schedule.ModelScheduler(
        config, store, SimpleNamespace(park=park), hardware
    )
    scheduler.last_traffic -= 2000
    monkeypatch.setattr(schedule, "probe", lambda *a: good())
    monkeypatch.setattr(schedule, "guard", lambda *a, **kw: None)
    monkeypatch.setattr(
        storage, "phase_requirements", lambda *a, **kw: [(external, 3_200_000_000)]
    )

    async def spawn(*args, **kwargs):
        launched.append(args)
        return SimpleNamespace(pid=12345, returncode=None)

    monkeypatch.setattr(schedule.asyncio, "create_subprocess_exec", spawn)
    free[external] = 900_000_000
    await scheduler.tick()
    assert not parked and not launched
    assert scheduler.reason == f"Needs 3.2 GB free on {external}; 0.9 GB available"
    assert (
        snapshot(config, store, scheduler, live=True)["models"][0]["reason"]
        == scheduler.reason
    )
    assert registry.entries(store)[0]["support"] == "pending"
    free[external] = 46 * GB
    await scheduler.tick()
    assert len(launched) == 1 and "songe.sleep.calibrate" in launched[0]
    assert scheduler.calibrating
    assert store.get("disk_wait") is None
    await scheduler.guard_task


@pytest.mark.parametrize(
    "module,args",
    [
        ("songe.sleep.calibrate", ["--child"]),
        ("songe.sleep.worker", ["--child"]),
        ("songe.models.worker", ["gguf", "missing.gguf", "missing-out"]),
    ],
)
def test_child_traceback_is_only_in_log(tmp_path, module, args):
    root = tmp_path / "data"
    root.mkdir()
    # Fail before any model or hardware work.
    (root / "config.toml").write_text("upstream_timeout = -1\n", encoding="utf-8")
    argv = [sys.executable, "-m", module, *args]
    if module != "songe.models.worker":
        argv += ["--data-dir", str(root)]
    else:
        # Argument validation fails before importing torch or loading weights.
        argv = [
            sys.executable,
            "-c",
            "from songe.telemetry import diagnostic_main; "
            "from songe.models.worker import convert; "
            "diagnostic_main(lambda: convert('mlx', None, None))",
        ]
    env = os.environ | {"SONGE_DIAGNOSTICS_DIR": str(root)}
    if module == "songe.models.worker":
        argv[-1] = "import sys; sys.platform='linux'; " + argv[-1]
    result = subprocess.run(argv, capture_output=True, env=env)
    assert result.returncode == 1
    assert b"Traceback" not in result.stderr + result.stdout
    assert b"songe doctor" in result.stderr
    log = (root / "songe.log").read_text(encoding="utf-8")
    assert "Traceback" in log
    assert "ValueError" in log or "RuntimeError" in log


def test_measured_tensor_budget_uses_artifact_and_database_volumes(volumes):
    _, external, free, store, config = volumes
    model = Path(config.model)
    header = json.dumps(
        {
            "layer.weight": {
                "dtype": "BF16",
                "shape": [40000, 40000],
                "data_offsets": [0, 3_200_000_000],
            }
        }
    ).encode()
    (model / "model.safetensors").write_bytes(
        len(header).to_bytes(8, "little") + header
    )
    config.model_ref["format"] = "mlx"
    dense, adapter = storage.tensor_sizes(model, config.rank)
    assert dense == 3_200_000_000
    assert adapter == config.rank * 80000 * 4
    requirements = storage.phase_requirements(
        config, store, SimpleNamespace(), calibration=True
    )
    assert requirements[0][0].is_relative_to(external)
    assert requirements[0][1] >= dense + adapter
    assert (store.root, store.path.stat().st_size) in requirements
    storage.admit_storage(config, store, SimpleNamespace(), calibration=True)
    free[external] = 900_000_000
    with pytest.raises(storage.DiskSpace):
        storage.admit_storage(config, store, SimpleNamespace(), calibration=True)
    assert "GB free" in store.get("disk_wait")


def test_resumed_night_does_not_need_its_own_files_twice(volumes):
    _, external, free, store, config = volumes
    platform = SimpleNamespace()
    target, fresh = storage.phase_requirements(config, store, platform)[0]
    held = fresh // 2
    assert held
    work = storage.artifact_root(config, store) / "versions" / "night-1"
    work.mkdir(parents=True)
    with (work / "adapter.safetensors").open("wb") as file:
        file.truncate(held)
    night = store.night(None)
    store.night(night, status="running", record={"work": str(work)})
    free[external] = fresh - held
    assert storage.phase_requirements(config, store, platform)[0] == (target, fresh)
    with pytest.raises(storage.DiskSpace):
        storage.admit_storage(config, store, platform)
    store.night(night, status="interrupted")
    assert storage.phase_requirements(config, store, platform)[0] == (
        target,
        fresh - held,
    )
    storage.admit_storage(config, store, platform)
    # Calibration resumes no night.
    assert (
        storage.phase_requirements(config, store, platform, calibration=True)[0][1]
        == fresh
    )


def test_forget_removes_owned_external_cache_and_abandoned_staging(volumes):
    from songe.api import forget_experience

    _, _, _, store, config = volumes
    config.save(store.root)
    scratch = storage.artifact_root(config, store)
    for name in ["cache", ".converting-abandoned"]:
        directory = scratch / "models" / name
        directory.mkdir(parents=True)
        marker = "songe_conversion.json" if name == "cache" else "songe_converting.json"
        write_json(
            directory / marker,
            {"installation_id": store.get("installation_id"), "source": config.model},
        )
        (directory / "weights").write_text("discard", encoding="utf-8")
    forget_experience(store)
    assert not scratch.exists()
    assert Path(config.model).exists()
    assert store.path.exists()


def test_uninstall_purges_external_root_without_touching_library(volumes, monkeypatch):
    from songe import agents, cli, launchd

    _, _, _, store, config = volumes
    config.save(store.root)
    scratch = storage.artifact_root(config, store)
    (scratch / "large-artifact").write_text("ours", encoding="utf-8")
    monkeypatch.setattr(launchd, "remove", lambda *a, **kw: None)
    monkeypatch.setattr(agents, "restore", lambda *a: [])
    monkeypatch.setattr(cli.LocalClient, "control", lambda *a, **kw: None)
    assert (
        cli.main(["--data-dir", str(store.root), "uninstall", "--purge", "--yes"]) == 0
    )
    assert not scratch.exists() and not store.root.exists()
    assert Path(config.model).exists()


def test_publication_rejects_cross_volume_staging_before_work(volumes, monkeypatch):
    from songe.sleep import fuse

    internal, external, _, _, config = volumes
    monkeypatch.setattr(
        fuse, "guarded_run", lambda *a, **kw: pytest.fail("must not launch")
    )
    with pytest.raises(storage.StorageWait, match="share a volume"):
        fuse.fuse(
            config, internal / "adapter", external / "candidate", scratch=internal
        )
    assert not (external / "candidate").exists()


def test_purge_keeps_files_songe_did_not_create(tmp_path, monkeypatch):
    from songe import agents, cli, launchd

    root = tmp_path / "shared"
    Store(root)
    Config(upstream="lmstudio").save(root)
    (root / "versions").mkdir()
    (root / "versions" / "night-1").mkdir()
    (root / "my-model.gguf").write_bytes(b"user weights")
    monkeypatch.setattr(launchd, "remove", lambda *a, **k: None)
    monkeypatch.setattr(agents, "restore", lambda root: [])
    monkeypatch.setattr(cli.LocalClient, "control", lambda *a, **k: None)

    async def nothing(*args):
        return None

    monkeypatch.setattr("songe.upstream.cleanup.purge", nothing)
    assert cli.main(["--data-dir", str(root), "uninstall", "--purge", "--yes"]) == 0
    assert sorted(p.name for p in root.iterdir()) == ["my-model.gguf"]


def test_init_refuses_a_directory_with_other_files(tmp_path, capsys):
    from songe import cli

    root = tmp_path / "AI"
    root.mkdir()
    (root / "notes.txt").write_text("mine")
    assert cli.main(["--data-dir", str(root), "init", "--yes"]) != 0
    assert "choose an empty or new --data-dir" in capsys.readouterr().err
    assert sorted(p.name for p in root.iterdir()) == ["notes.txt"]


def test_a_4bit_base_budgets_8bit_fused_candidates(tmp_path):
    from songe.artifacts.storage import preserving_fuse

    model = tmp_path / "m"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"quantization": {"bits": 4}}))
    assert preserving_fuse(model)
    (model / "config.json").write_text(json.dumps({"quantization": {"bits": 8}}))
    assert not preserving_fuse(model)
    assert not preserving_fuse(tmp_path / "missing")


def test_fuse_output_of_a_stopped_measuring_run_is_collected(tmp_path):
    from songe.artifacts.storage import collect_scratch
    from songe.log import Store
    from songe.storage import write_json

    root = tmp_path / "data"
    store = Store(root)
    gone = root / "calibration-x"  # the measuring run's folder, already removed
    orphan = root / ".artifact-abc"
    kept = root / ".other-abc"
    for path in (orphan, kept):
        path.mkdir(parents=True)
        (path / "model.safetensors").write_bytes(b"partial")
        write_json(
            path / "songe_fusing.json",
            {
                "base": "m",
                "adapter": str(gone / "adapter"),
                "destination": str(gone / "artifact"),
            },
        )
    outside = root / ".artifact-out"
    outside.mkdir()
    write_json(
        outside / "songe_fusing.json",
        {
            "base": "m",
            "adapter": str(tmp_path / "elsewhere/adapter"),
            "destination": str(tmp_path / "elsewhere/artifact"),
        },
    )
    collect_scratch(store)
    assert not orphan.exists()
    # A name that does not match its destination, or one aimed outside our
    # folder, is never touched.
    assert kept.exists() and outside.exists()


def test_disk_admission_leaves_a_reserve(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from songe.artifacts import storage as disk

    usage = SimpleNamespace(total=100 * GB, free=10 * GB)
    monkeypatch.setattr(disk.shutil, "disk_usage", lambda path: usage)
    disk.preflight([(tmp_path, 9 * GB)])  # 9 + 1 GB reserve fits in 10
    with pytest.raises(disk.DiskSpace) as stopped:
        disk.preflight([(tmp_path, 9.5 * GB)])
    assert "Needs 10.5 GB" in str(stopped.value)
    usage.total = 2000 * GB  # the reserve never exceeds 2 GB
    disk.preflight([(tmp_path, 7.9 * GB)])
    disk.preflight([(tmp_path, 0)])  # nothing needed, nothing reserved
