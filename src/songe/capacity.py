"""What this machine can train, roughly: one line for the dashboard.

Measured per model, the training window is exact (sleep.calibration_state.
training_window). Before any model is measured, and for models not yet tried,
people still want to know what size of model this machine can train at all.
That is an estimate from the training device's memory, calibrated against
measurements (docs: research FINDINGS), and shown only as "about".

Rule of thumb for a dense model trained in 4-bit with LoRA at a 4k window:
memory needed ~= 0.7 GB per billion parameters + 8 GB working memory.
- NVIDIA/AMD: the GPU's own memory, 90% usable.
- Apple Silicon: memory is shared with macOS and apps and the GPU may use only
  part of it, so about half is usable and the working memory is smaller.
Measured anchors: Qwen3.8-27B trains at 4k on a 32 GB RTX 5090 (23.7 GB peak);
3-8B models train on a 24 GB M4 Mac.
"""

import time

GB = 1e9
_CACHE = {"at": 0.0, "value": None}


def trainable_billions(kind, memory_bytes):
    """Largest dense model (billions of parameters) this device can train in
    4-bit, roughly; None when it cannot train at all."""
    gb = memory_bytes / GB
    if kind in {"cuda", "rocm"}:
        size = (0.9 * gb - 8) / 0.7
    elif kind == "metal":
        size = (0.5 * gb - 6) / 0.7
    else:
        return None
    return max(0, int(size))


def machine(platform, config=None, max_age=60.0):
    """{name, kind, memory_gb, gpus, gpu_index, trains_up_to_b} for the
    training device, cached for max_age seconds (GPU queries are not free)."""
    now = time.monotonic()
    if _CACHE["value"] is not None and now - _CACHE["at"] < max_age:
        return _CACHE["value"]
    try:
        accelerators = platform.accelerators()
    except Exception:  # noqa: BLE001 - the dashboard line is optional
        accelerators = []
    device = "auto"
    if config is not None:
        device = (getattr(config, "torch_options", None) or {}).get("device", "auto")
    gpus = [a for a in accelerators if a.kind in {"cuda", "rocm"}]
    index = int(device.partition(":")[2] or 0) if device.startswith(("cuda", "rocm")) else 0
    if gpus:
        chosen = gpus[min(index, len(gpus) - 1)]
    else:
        chosen = next((a for a in accelerators if a.kind == "metal"), None)
    if chosen is None:
        value = {"name": accelerators[0].name if accelerators else "CPU", "kind": "cpu",
                 "memory_gb": None, "gpus": 0, "gpu_index": None, "trains_up_to_b": None}
    else:
        value = {
            "name": chosen.name,
            "kind": chosen.kind,
            "memory_gb": round(chosen.memory_total / 2**30) if chosen.memory_total else None,
            "gpus": len(gpus),
            "gpu_index": gpus.index(chosen) if chosen in gpus else None,
            "trains_up_to_b": trainable_billions(chosen.kind, chosen.memory_total)
            if chosen.memory_total else None,
        }
    _CACHE.update(at=now, value=value)
    return value
