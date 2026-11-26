import json
from pathlib import Path
from types import SimpleNamespace

from songe.platform import Accel
from songe.sleep import calibrate, fuse
from songe.storage import write_json


def test_calibration_public_entry_is_guarded_and_copies_measured_costs(
    tmp_path, config, monkeypatch
):
    from songe import memory

    calls = []
    rows = [{"sequence": 256, "step_seconds": 1, "peak_memory_gb": 1}]

    def guarded(argv, **kwargs):
        import subprocess

        assert kwargs["stdout"] == subprocess.PIPE
        calls.append(argv)
        child = type(config).load(tmp_path)
        child.calibration_available = True
        child.calibrated_step_seconds, child.calibrated_score_seconds = 7, 11
        child.save(tmp_path)
        write_json(tmp_path / "calibration.json", {"rows": rows})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(memory, "guarded_run", guarded)
    assert calibrate.calibrate(config, tmp_path, (256,)) == rows
    assert config.calibration_available and config.calibrated_step_seconds == 7
    assert "--child" in calls[0]
    assert "songe.sleep.calibrate" in calls[0]


def test_fuse_uses_guard_and_private_publication(
    tmp_path, config, monkeypatch, writable_file_fsync
):
    calls = []

    def guarded(argv, **kwargs):
        calls.append(argv)
        assert callable(kwargs["check"])
        option = "--mlx-path" if "--convert" in argv else "--save-path"
        output = Path(argv[argv.index(option) + 1])
        output.mkdir(exist_ok=True)
        (output / "model.safetensors").write_bytes(b"fixture")
        (output / "config.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(fuse, "guarded_run", guarded)
    destination = tmp_path / "models/candidate"
    adapter = tmp_path / "adapter"
    assert fuse.fuse(config, adapter, destination) == destination
    assert len(calls) == 1
    assert "songe.sleep.fuse_worker" in calls[0]
    marker = json.loads((destination / "songe_owned.json").read_text(encoding="utf-8"))
    assert marker["bits"] == "base"
    assert list(destination.parent.iterdir()) == [destination]


def test_interrupted_conversion_cleans_only_owned_staging(
    tmp_path, config, monkeypatch
):
    from songe.backends import Interrupted

    unrelated = tmp_path / "models/user-model"
    unrelated.mkdir(parents=True)
    (unrelated / "model.safetensors").write_bytes(b"user")

    def guarded(argv, **kwargs):
        raise Interrupted()

    monkeypatch.setattr(fuse, "guarded_run", guarded)
    import pytest

    with pytest.raises(Interrupted):
        fuse.fuse(
            config,
            tmp_path / "adapter",
            tmp_path / "models/candidate",
        )
    assert list(unrelated.parent.iterdir()) == [unrelated]


def test_backend_selection_is_lazy_and_routes_all_accelerators(config, monkeypatch):
    import sys

    from songe.backends import choose
    from songe.backends.mlx import Backend

    class Torch:
        pass

    monkeypatch.setitem(
        sys.modules, "songe.backends.torch", SimpleNamespace(Backend=Torch)
    )

    def platform(kind):
        return SimpleNamespace(accelerators=lambda: [Accel(kind, "fixture")])

    assert choose(config, platform("metal")) is Backend
    for kind in ("cuda", "rocm", "xpu", "cpu"):
        assert choose(config, platform(kind)) is Torch


def test_child_memory_stop_is_a_wait_not_a_failure(tmp_path):
    """A child refused by its own memory check exits 75 with its sample, so the
    parent re-raises MemoryPressure and never counts a failure that could mark
    the model unsupported."""
    import subprocess
    import sys

    from songe.memory import MEMORY_EXIT, MemoryPressure, child_memory_stop

    script = (
        "from songe.memory import MemoryPressure, Snapshot\n"
        "from songe.telemetry import diagnostic_main\n"
        "def main():\n"
        "    raise MemoryPressure(Snapshot(24, 4, 'warn'), {'rule': 'admission_prediction'})\n"
        f"diagnostic_main(main, {str(tmp_path)!r})\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, check=False
    )
    assert result.returncode == MEMORY_EXIT
    stop = child_memory_stop(result.returncode, result.stderr)
    assert isinstance(stop, MemoryPressure)
    assert stop.memory.available == 4 and stop.memory.pressure == "warn"
    assert stop.reason["rule"] == "admission_prediction"
    assert child_memory_stop(1, result.stderr) is None


def test_child_passes_a_metal_buffer_limit_to_its_parent(tmp_path, capsys):
    import pytest

    from songe.telemetry import diagnostic_main

    def operation():
        raise RuntimeError(
            "[metal::malloc] Attempting to allocate 15053357056 bytes which is"
            " greater than the maximum allowed buffer size of 14302248960 bytes."
        )

    with pytest.raises(SystemExit):
        diagnostic_main(operation, root=tmp_path)
    err = capsys.readouterr().err.encode()
    from songe.sleep.calibrate import BUFFER_CEILING

    assert BUFFER_CEILING.search(err)


def test_a_stopped_conversion_child_logs_a_stop_not_a_failure(tmp_path):
    """A conversion stopped by the memory guard left "heavy_child_failed
    OSError [Errno 22]" in the log, from a progress bar cut off by the stop:
    it read as the cause, not the result (reported by Jack in a test on
    11-24). The child now ends at the stop request and logs that."""
    import subprocess
    import sys
    import time

    from songe.memory import interrupt, process_options

    script = (
        "import time\n"
        "from songe.telemetry import diagnostic_main\n"
        "def main():\n"
        "    print('ready', flush=True)\n"
        "    while True:\n"
        "        time.sleep(0.01)\n"
        f"diagnostic_main(main, {str(tmp_path)!r}, stoppable=True)\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **process_options(),
    )
    assert process.stdout.readline().strip() == b"ready"
    interrupt(process)
    deadline = time.monotonic() + 30
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert process.poll() is not None, "the child ignored the stop"
    process.communicate()
    log = (tmp_path / "songe.log").read_text(encoding="utf-8")
    assert "heavy_child_stopped" in log
    assert "heavy_child_failed" not in log


def test_an_error_raised_by_the_stop_is_not_logged_as_a_failure(tmp_path):
    import json
    import signal

    import pytest

    from songe import telemetry

    stop = getattr(signal, "SIGBREAK", None) or signal.SIGUSR1
    before = signal.getsignal(stop)

    def operation():
        # The stop's handler ran, and a write the stop cut off failed after.
        with pytest.raises(telemetry.Stopped):
            signal.getsignal(stop)()
        raise OSError(22, "Invalid argument")

    try:
        with pytest.raises(SystemExit):
            telemetry.diagnostic_main(operation, root=tmp_path, stoppable=True)
    finally:
        signal.signal(stop, before)
    events = [
        json.loads(line)["event"]
        for line in (tmp_path / "songe.log").read_text(encoding="utf-8").splitlines()
    ]
    assert "heavy_child_stopped" in events and "heavy_child_failed" not in events
