"""The dashboard's machine line and each model's measured training window."""

from dataclasses import replace

from songe import capacity
from songe.platform import Accel
from songe.sleep.calibration_state import training_window


class Platform:
    def __init__(self, accelerators):
        self._accelerators = accelerators

    def accelerators(self):
        return self._accelerators


def fresh(monkeypatch):
    monkeypatch.setattr(capacity, "_CACHE", {"at": 0.0, "value": None})


def test_estimates_follow_the_measured_anchors():
    # Measured: a 27B trains at 4k on a 32 GB RTX 5090; 3-8B on a 24 GB Mac.
    assert 27 <= capacity.trainable_billions("cuda", 32 * 2**30) <= 32
    assert 3 <= capacity.trainable_billions("metal", 24 * 2**30) <= 9
    assert capacity.trainable_billions("cuda", 80e9) > capacity.trainable_billions("cuda", 48e9)
    assert capacity.trainable_billions("cpu", 64e9) is None


def test_machine_line_names_the_training_gpu_among_several(monkeypatch, config):
    fresh(monkeypatch)
    gpus = [Accel("cuda", "RTX 4090", 24 * 2**30), Accel("cuda", "RTX 5090", 32 * 2**30)]
    config = replace(config, torch_options={"device": "cuda:1"})
    line = capacity.machine(Platform(gpus), config)
    assert line["name"] == "RTX 5090" and line["memory_gb"] == 32
    assert line["gpus"] == 2 and line["gpu_index"] == 1
    assert line["trains_up_to_b"] >= 27


def test_mac_and_cpu_only_machines(monkeypatch):
    fresh(monkeypatch)
    mac = capacity.machine(Platform([Accel("metal", "Apple M4 Pro", 24 * 2**30)]))
    assert mac["kind"] == "metal" and mac["memory_gb"] == 24 and mac["gpus"] == 0
    fresh(monkeypatch)
    cpu = capacity.machine(Platform([Accel("cpu", "x86_64")]))
    assert cpu["trains_up_to_b"] is None and cpu["kind"] == "cpu"


def row(sequence, response=128, chunk=1, **extra):
    return {"sequence": sequence, "response_tokens": response, "token_chunk": chunk,
            "step_seconds": 4.0, **extra}


def test_window_before_measurement_is_unknown(config):
    assert training_window(config, None) is None
    assert training_window(replace(config, calibration_available=True), {"rows": []}) is None


def test_window_limited_by_memory_or_enough_for_the_conversations(config):
    measured = replace(config, calibration_available=True, max_sequence=8192, response_tokens=128,
                       token_chunk=1, context_window=32768, calibrated_step_seconds=160.0, accumulation=4)
    # A probe just above the measured length ran out of memory: the real limit.
    limited = training_window(measured, {"rows": [row(4096), row(8192)],
                                         "capacity_limits": [row(8800)]})
    assert limited == {"window": 8192, "limit": "memory", "served": 32768, "example_seconds": 40.0}
    # No failure: it measured as far as the conversations needed.
    enough = training_window(measured, {"rows": [row(2048), row(8192)]})
    assert enough["limit"] == "conversations"
