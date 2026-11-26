"""Live memory protection for every heavyweight child, including process groups."""

import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache

from .platform import Accel, Memory, current

NORMAL, WARN, CRITICAL = 1, 2, 4


@lru_cache(maxsize=1)
def linux_mlx_accelerator():
    # Keep the Vulkan context out of the long-running scheduler.
    try:
        info = json.loads(
            subprocess.check_output(
                [
                    sys.executable,
                    "-c",
                    "import json, mlx.core as mx; print(json.dumps(mx.device_info()))",
                ],
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise OSError("MLX could not identify its Linux training device") from exc
    if isinstance(info, dict):
        name = info.get("device_name")
        if (
            isinstance(name, str)
            and name.startswith("Apple ")
            and info.get("architecture") in ("honeykrisp", "borealis")
            and info.get("unified_memory") == 1
            and not info.get("simulated")
        ):
            return Accel("mlx", name)
    raise OSError("MLX training requires a verified Apple unified-memory GPU on Linux")


class TrainingPlatform:
    """Probe only the device the backend selects, preserving missing sensors."""

    def __init__(self, platform, config):
        self.platform, self.name = platform, platform.name
        self._songe_accelerators = set()
        self.linux_mlx = self.name == "linux" and config.backend == "mlx"
        if self.linux_mlx:
            return
        requested = config.torch_options.get("device", "auto")
        devices = platform.accelerators() if platform.name != "darwin" else []
        if requested.startswith("xpu"):
            raise ValueError(
                "XPU training requires memory telemetry and is not yet supported"
            )
        kind = requested.split(":")[0]
        if kind == "auto":
            kind = next(
                (k for k in ("cuda", "rocm") if any(a.kind == k for a in devices)),
                "cpu",
            )
        if (
            kind == "cuda"
            and not any(a.kind == "cuda" for a in devices)
            and any(a.kind == "rocm" for a in devices)
        ):
            kind = "rocm"
        index = int(requested.partition(":")[2] or 0)
        if index < 0:
            raise ValueError("Accelerator index must be nonnegative")
        visible = (
            os.environ.get("CUDA_VISIBLE_DEVICES")
            if kind == "cuda"
            else os.environ.get("ROCR_VISIBLE_DEVICES")
        )
        candidates = [a for a in devices if a.kind == kind]
        if visible and requested != "cpu":
            tokens = visible.split(",")
            if index >= len(tokens):
                raise OSError("Selected accelerator is outside the visible device list")
            if tokens[index].isdigit():
                index = int(tokens[index])
            else:
                matches = [
                    i
                    for i, a in enumerate(candidates)
                    if a.identifier and a.identifier.startswith(tokens[index])
                ]
                if len(matches) != 1:
                    raise OSError("Visible accelerator memory telemetry is unavailable")
                index = matches[0]
        if kind == "cuda" and visible in {"", "-1"}:
            candidates = []
            if requested == "auto":
                kind = "cpu"
        self.index = index
        self.selected = (
            (kind, candidates[index].identifier or candidates[index].name)
            if index < len(candidates)
            else None
        )
        if kind in {"cuda", "rocm"} and self.selected is None:
            raise OSError("Selected accelerator memory telemetry is unavailable")

    def accelerators(self):
        if self.linux_mlx:
            return [linux_mlx_accelerator()]
        if self.selected is None:
            return (
                self.platform.accelerators()
                if self.name == "darwin"
                else [Accel("cpu", "CPU")]
            )
        devices = [
            a for a in self.platform.accelerators() if a.kind == self.selected[0]
        ]
        devices = [devices[self.index]] if self.index < len(devices) else []
        devices = [
            a for a in devices if (a.kind, a.identifier or a.name) == self.selected
        ]
        if not devices:
            raise OSError("Selected accelerator memory telemetry is unavailable")
        return devices

    def __getattr__(self, name):
        return getattr(self.platform, name)


@dataclass(frozen=True)
class GuardPolicy:
    # At most one second of unobserved allocation; 100ms is the probe overhead
    # floor. The measured time-to-exhaustion shortens this interval further.
    min_interval: float = 0.1
    max_interval: float = 1.0
    grace_seconds: float = 3.0
    allocation_headroom: float = 4.0
    baseline_seconds: float = 0.3
    rate_window_seconds: float = 3.0
    paging_growth_fraction: float = 0.05


@dataclass(frozen=True)
class Snapshot(Memory):
    vram: tuple[Accel, ...] = ()
    host_pressure: str = "normal"
    paging_noise: dict = field(default_factory=dict, kw_only=True)
    stop_reason: dict | None = field(default=None, kw_only=True)


def snapshot(platform):
    memory = platform.memory()
    devices = tuple(
        a
        for a in (platform.accelerators() if platform.name != "darwin" else [])
        if a.kind in {"cuda", "rocm", "xpu"}
    )
    if any(a.memory_total <= 0 for a in devices):
        raise OSError("Accelerator memory telemetry is unavailable")
    identities = {(a.kind, a.identifier or a.name) for a in devices}
    expected = getattr(platform, "_songe_accelerators", set())
    if expected - identities:
        raise OSError("Required accelerator memory telemetry is unavailable")
    platform._songe_accelerators = expected | identities
    pressure = (
        "critical" if any(a.memory_free <= 0 for a in devices) else memory.pressure
    )
    return Snapshot(
        memory.total,
        memory.available,
        pressure,
        devices,
        memory.pressure,
        **{
            k: getattr(memory, k, None)
            for k in ("swap_used", "compressed", "swap_out", "psi")
        },
    )


def admit(
    platform,
    required_bytes=0,
    required_device_bytes=None,
    *,
    margin_bytes=0,
    pressure_data=None,
):
    """Capture the pre-allocation baseline and reserve measured peak headroom."""
    memory = snapshot(platform)
    margin_bytes = learned_margin(memory, pressure_data, margin_bytes)
    device_need = (
        required_bytes if required_device_bytes is None else required_device_bytes
    )
    rule = None
    if memory.pressure not in {"normal", "warn"}:
        rule = "admission_pressure"
    elif memory.total <= 0:
        rule = "admission_telemetry"
    elif memory.available <= margin_bytes:
        rule = "admission_margin"
    elif required_bytes + margin_bytes > memory.available:
        rule = "admission_prediction"
    elif any(a.memory_free < device_need for a in memory.vram):
        rule = "admission_device_prediction"
    if rule:
        raise MemoryPressure(
            stopped_sample(
                memory,
                rule,
                baseline=memory,
                prediction={
                    "peak_bytes": required_bytes,
                    "device_peak_bytes": device_need,
                    "margin_bytes": margin_bytes,
                },
            )
        )
    return memory


# Disk paging only. macOS compression is how the OS makes room without harm;
# real damage shows up as swap traffic or a pressure escalation, both watched.
PAGING_SENSORS = ("swap_used", "swap_out")


def capture_baseline(
    platform, *, policy=None, check=lambda: None, on_sample=None, **kw
):
    """Measure background paging before the owned child can allocate anything."""
    policy = policy or GuardPolicy()
    samples = []
    # Fixed sample count also works with coarse or mocked monotonic clocks.
    for index in range(4):
        if index:
            time.sleep(policy.baseline_seconds / 3)
        check()
        sample = admit(platform, **kw)
        samples.append((time.monotonic(), sample))
        if on_sample and index < 3:
            on_sample(sample)
    noise = {}
    for sensor in PAGING_SENSORS:
        deltas, rates = [], []
        for (start, a), (end, b) in zip(samples, samples[1:], strict=False):
            before, after = getattr(a, sensor), getattr(b, sensor)
            if before is not None and after is not None:
                delta = max(0, after - before)
                deltas.append(delta)
                rates.append(delta / max(end - start, policy.min_interval))
        if deltas:
            noise[sensor] = {"delta_bytes": max(deltas), "bytes_per_second": max(rates)}
    baseline = replace(
        samples[-1][1],
        paging_noise={
            "window_seconds": max(0, samples[-1][0] - samples[0][0]),
            "sensors": noise,
        },
    )
    if on_sample:
        on_sample(baseline)
    return baseline


def learned_margin(sample, pressure_data=None, minimum=0):
    if pressure_data is None:
        return minimum
    from .sleep.sizing import PressureHistory

    return max(minimum, PressureHistory(pressure_data).margin_for(sample))


def escalated(baseline, sample):
    # Warn only means the OS is compressing to make room. Harm is critical
    # pressure or disk paging, both of which stop work on their own rules;
    # a warn caused by Songe is bounded by the margin and paging rules.
    return sample.pressure == "critical" and baseline.pressure != "critical"


def paging_grew(baseline, sample):
    return any(
        getattr(baseline, k, None) is not None
        and getattr(sample, k, None) is not None
        and getattr(sample, k) > getattr(baseline, k)
        for k in PAGING_SENSORS
    )


def stopped_sample(sample, rule, *, baseline=None, prediction=None, **observed):
    sample = sample or Snapshot(0, 0, "unknown")
    return replace(
        sample,
        stop_reason={
            "rule": rule,
            "observed": asdict(sample) | observed,
            "baseline": asdict(baseline) if baseline is not None else None,
            "prediction": prediction,
        },
    )


class MemoryPressure(RuntimeError):
    def __init__(self, memory, reason=None):
        self.memory = memory
        self.reason = (
            reason
            or getattr(memory, "stop_reason", None)
            or {
                "rule": "memory_pressure",
                "observed": asdict(memory),
                "baseline": None,
                "prediction": None,
            }
        )
        vram = "; ".join(
            f"{a.name}: available={a.memory_free} bytes, total={a.memory_total} bytes"
            for a in getattr(memory, "vram", ())
        )
        super().__init__(
            f"needs more free memory now: available={memory.available} bytes, total={memory.total} bytes, pressure={memory.pressure}"
            + (f"; VRAM {vram}" if vram else "")
            + "; reason="
            + json.dumps(self.reason, sort_keys=True, separators=(",", ":"))
        )


# A child that stops because memory is short exits with EX_TEMPFAIL and writes
# its sample on one marked stderr line, so every parent treats it as a memory
# wait (retried when memory frees) and never as a failure that can retire a model.
MEMORY_EXIT = 75
MEMORY_MARK = "songe-memory-stop "


def memory_stop_line(exc):
    memory = asdict(exc.memory) | {"class": type(exc.memory).__name__}
    return MEMORY_MARK + json.dumps({"memory": memory, "reason": exc.reason})


def child_memory_stop(returncode, stderr):
    """The MemoryPressure a child reported by exit code, or None."""
    if returncode != MEMORY_EXIT:
        return None
    text = stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
    for line in reversed((text or "").splitlines()):
        if line.startswith(MEMORY_MARK):
            data = json.loads(line[len(MEMORY_MARK) :])
            memory = data["memory"]
            kind = Snapshot if memory.pop("class", "") == "Snapshot" else Memory
            if kind is Snapshot:
                memory["vram"] = tuple(Accel(**a) for a in memory.get("vram") or ())
            return MemoryPressure(kind(**memory), data["reason"])
    return MemoryPressure(current().memory())


def record_memory_stop(store, sample, *, calibration=False, nights_before=None):
    """Persist after child exit so its final checkpoint cannot erase the reason."""
    if isinstance(sample, MemoryPressure):
        record = asdict(sample.memory) | {"stop_reason": sample.reason}
    else:
        record = asdict(sample)
    import logging
    import time

    from . import NAME

    reason = record.get("stop_reason") or {}
    reason = (reason.get("observed") or {}).get("trigger") or reason
    vram = "".join(
        f" {a['name']} {a['memory_free'] / 1e9:.1f} GB free"
        for a in record.get("vram") or ()
    )
    logging.getLogger(NAME).info(
        "memory_stop %s%s: %.1f of %.1f GB free, pressure %s, psi %s%s",
        reason.get("rule", "memory_pressure"),
        " (measuring)" if calibration else "",
        record.get("available", 0) / 1e9,
        record.get("total", 0) / 1e9,
        record.get("pressure"),
        record.get("psi"),
        vram,
    )

    store.set("last_memory_stop", record)
    # When (a measuring run that fails right after it ran out of memory too).
    store.set("last_memory_stop_at", time.time())
    if calibration:
        from .storage import write_json

        saved = store.get("calibration", {}) | {"memory_stop": record}
        store.set("calibration", saved)
        path = store.root / "calibration.json"
        data = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else {"rows": []}
        )
        write_json(path, data | {"memory_stop": record})
        from .sleep.calibrate import record_capacity_stop

        record_capacity_stop(store, saved)
    else:
        nights = store.nights()
        if nights:
            night = nights[-1]
            before = next(
                (n for n in nights_before or [] if n["id"] == night["id"]), None
            )
            admission = (
                (record.get("stop_reason") or {})
                .get("rule", "")
                .startswith("admission_")
            )
            if not admission and (
                night != before
                or night["status"] in {"running", "interrupted", "waiting"}
            ):
                store.night(
                    night["id"], record=night["record"] | {"memory_stop": record}
                )


def pressure_level():
    try:
        return {"normal": NORMAL, "warn": WARN, "critical": CRITICAL}[
            current().memory().pressure
        ]
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return WARN


def process_options():
    return (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if os.name == "nt"
        else {"start_new_session": True}
    )


def owned_command(argv):
    from .child import parent_birth

    return [
        sys.executable,
        "-m",
        "songe.child",
        "--parent-pid",
        str(os.getpid()),
        "--parent-start",
        parent_birth(),
        "--",
        *map(str, argv),
    ]


def process_tree_bytes(platform, pid):
    import psutil

    total = (
        (platform.process_bytes(pid) or 0) if hasattr(platform, "process_bytes") else 0
    )
    try:
        for child in psutil.Process(pid).children(recursive=True):
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                own = (
                    platform.process_bytes(child.pid)
                    if hasattr(platform, "process_bytes")
                    else None
                )
                total += own if own is not None else child.memory_info().rss
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return total


def process_tree_swap_bytes(platform, pid):
    """Process-specific swap where available; never confuse Windows commit with swap."""
    import psutil

    if not hasattr(platform, "process_swap_bytes"):
        return None
    pids = [pid]
    try:
        pids += [child.pid for child in psutil.Process(pid).children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    samples = [platform.process_swap_bytes(p) for p in pids]
    return (
        sum(v for v in samples if v is not None)
        if any(v is not None for v in samples)
        else None
    )


def interrupt(process):
    with contextlib.suppress(ProcessLookupError, OSError):
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(process.pid, signal.SIGUSR1)


def terminate(process):
    with contextlib.suppress(ProcessLookupError, OSError):
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(process.pid, signal.SIGTERM)


def kill(process):
    with contextlib.suppress(ProcessLookupError, OSError):
        if os.name == "nt":
            # /T includes conversion/fuse descendants of the guarded worker.
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True
            )
        else:
            os.killpg(process.pid, signal.SIGKILL)


def running(process):
    return (
        process.poll() is None
        if hasattr(process, "poll")
        else process.returncode is None
    )


def shares_system_memory(memory):
    """An accelerator without memory of its own (GB10 / DGX Spark, Jetson).

    Its memory is reported as the system's (platform.common.unified_memory),
    so its allocations lower `available` but never appear in the process
    tree's RSS. Counting RSS alone as what a slice has used would reserve its
    device allocations twice: once as gone from `available` and again as
    still to come.
    """
    return any(
        a.memory_total == memory.total for a in getattr(memory, "vram", ()) or ()
    )


def guard(
    process,
    platform=None,
    on_warn=None,
    on_critical=None,
    *,
    policy=None,
    stop=None,
    on_sample=None,
    baseline=None,
    margin_bytes=0,
    predicted_peak_bytes=None,
    predicted_device_bytes=None,
    prediction_error_bytes=0,
    device_prediction_error_bytes=0,
    projection=True,
):
    """Block until exit, polling RAM/VRAM; callbacks receive measured Memory.

    projection=False: no stop on a projected allocation rate alone (measuring
    what fits: loading the weights is a burst, and live on a 24 GB Mac with 8
    GB free it stopped every measurement within 5 s, so training could never
    be sized). The reserve margin, critical pressure, escalation and paging
    rules still stop it.

    Predictions bound remaining allocation; unmeasured loads use a rolling rate.
    Critical pressure bypasses the graceful path. Escalation, attributable paging
    and unsafe projections interrupt, then kill after the bounded save interval.
    Callbacks receive a Snapshot with a machine-readable stop_reason.
    """
    platform, policy = platform or current(), policy or GuardPolicy()
    stop = stop or threading.Event()
    started = time.monotonic()
    warned_at, warning, previous, last_time = None, None, None, started
    projected_since = None  # a fast allocation alone is a forecast, not harm
    observations = deque()
    initial_rss = 0 if baseline is not None else None
    initial_swap = 0 if baseline is not None else None
    burst = None  # (time, swap bytes) when attributed paging was first seen

    def sustained_paging(paging, now):
        """A one-off swap burst while weights load is normal on macOS; only
        swap that keeps growing past the grace period is harm."""
        nonlocal burst
        level = max(v["growth_bytes"] for v in paging.values())
        material = max(v["material_bytes"] for v in paging.values())
        if burst is None:
            burst = (now, level)
            return False
        since, first = burst
        if level - first < material:
            if now - since >= policy.grace_seconds:
                burst = (now, level)  # growth stopped; a later burst gets its own grace
            return False
        return now - since >= policy.grace_seconds

    last = None
    prediction = {
        "peak_bytes": predicted_peak_bytes,
        "error_bytes": prediction_error_bytes,
        "device_peak_bytes": predicted_device_bytes,
        "device_error_bytes": device_prediction_error_bytes,
        "margin_bytes": margin_bytes,
    }

    def notify(callback, sample, critical=False):
        try:
            (callback or (lambda _: kill(process) if critical else interrupt(process)))(
                sample
            )
        except Exception:
            # Persistence/diagnostic failures cannot disable the safety action.
            kill(process)
            return False
        return True

    while running(process) and not stop.is_set():
        now = time.monotonic()
        try:
            memory = snapshot(platform)
            if baseline is None:
                baseline = memory
            if previous is None:
                previous = {"ram": baseline.available - margin_bytes} | {
                    f"{a.kind}:{i}": a.memory_free
                    for i, a in enumerate(getattr(baseline, "vram", ()))
                }
                observations.append((started, previous, initial_rss))
            if on_sample:
                with contextlib.suppress(Exception):
                    on_sample(memory)
            devices = memory.vram
            if not running(process) or stop.is_set():
                return None
            resources = {"ram": memory.available - margin_bytes} | {
                f"{a.kind}:{i}": a.memory_free for i, a in enumerate(devices)
            }
            elapsed = max(now - last_time, policy.min_interval)
            rss = (
                process_tree_bytes(platform, process.pid)
                if hasattr(platform, "process_bytes")
                else None
            )
            if initial_rss is None and rss is not None:
                initial_rss = rss
            swapped = (
                process_tree_swap_bytes(platform, process.pid)
                if hasattr(process, "pid")
                else None
            )
            if initial_swap is None and swapped is not None:
                initial_swap = swapped
            own_growth = max(0, (rss or 0) - (initial_rss or 0))
            paging = {}
            if (
                swapped is not None
                and initial_swap is not None
                and swapped > initial_swap
            ):
                paging["process_swap"] = {
                    "growth_bytes": swapped - initial_swap,
                    "baseline_bytes": initial_swap,
                }
            # Process swap is decisive. Global counters also cover displacement
            # of other apps, but must exceed pre-start noise AND be material
            # relative to our own growth. PSI alone cannot attribute bytes.
            for sensor in PAGING_SENSORS:
                before, after = getattr(baseline, sensor), getattr(memory, sensor)
                noise = (
                    getattr(baseline, "paging_noise", {})
                    .get("sensors", {})
                    .get(sensor, {})
                )
                allowance = noise.get("delta_bytes", 0) + noise.get(
                    "bytes_per_second", 0
                ) * max(now - started, elapsed)
                delta = (
                    max(0, after - before)
                    if before is not None and after is not None
                    else 0
                )
                material = own_growth * policy.paging_growth_fraction
                if (
                    own_growth > 0
                    and delta > allowance
                    and delta - allowance >= material
                ):
                    paging[sensor] = {
                        "growth_bytes": delta,
                        "noise_bytes": allowance,
                        "material_bytes": material,
                    }

            observations.append((now, resources, rss))
            while (
                len(observations) > 2
                and observations[1][0] <= now - policy.rate_window_seconds
            ):
                observations.popleft()
            since, old_resources, old_rss = observations[0]
            window = max(policy.rate_window_seconds, now - since, policy.min_interval)
            rates = {
                k: max(0, old_resources.get(k, v) - v) / window
                for k, v in resources.items()
            }
            if rss is not None and old_rss is not None:
                rates["ram"] = max(rates["ram"], max(0, rss - old_rss) / window)
            rate = max(rates.values())
            # A burst can shorten polling without being extrapolated forever.
            poll_rate = max(
                rate,
                *(
                    max(0, previous.get(k, v) - v) / elapsed
                    for k, v in resources.items()
                ),
            )
            remaining = min(resources.values())
            interval = (
                min(
                    policy.max_interval,
                    max(
                        policy.min_interval,
                        remaining / (poll_rate * policy.allocation_headroom),
                    ),
                )
                if poll_rate
                else policy.max_interval
            )
            exhausted = remaining <= 0
            drop = max(0, baseline.available - memory.available)
            consumed = (
                drop
                if rss is None
                # Measured like calibration's Meter (sleep/measurement.py):
                # the larger of the tree's RSS and the fall in free memory.
                else max(rss, drop)
                if shares_system_memory(memory)
                else rss
            )
            need = (
                max(0, predicted_peak_bytes - consumed)
                if predicted_peak_bytes is not None
                else None
            )
            prediction = prediction | {
                "remaining_need_bytes": need,
                "projected_available_bytes": memory.available - need
                if need is not None
                else None,
            }
            device_predictions = []
            if predicted_device_bytes is not None:
                for a, b in zip(baseline.vram, devices, strict=True):
                    used = max(0, a.memory_free - b.memory_free)
                    device_predictions.append(
                        {
                            "name": b.name,
                            "used_bytes": used,
                            "remaining_need_bytes": max(
                                0, predicted_device_bytes - used
                            ),
                            "available_bytes": b.memory_free,
                        }
                    )
            prediction["devices"] = device_predictions
            unpredicted = {
                k: v
                for k, v in resources.items()
                if (predicted_peak_bytes if k == "ram" else predicted_device_bytes)
                is None
            }
            imminent = {
                k: {
                    "available_after_margin_bytes": v,
                    "bytes_per_second": rates[k],
                    "window_seconds": window,
                }
                for k, v in unpredicted.items()
                if rates[k] > 0 and v / rates[k] <= policy.grace_seconds + interval
            }
            last = memory
            rule = None
            if memory.pressure == "critical":
                rule = "critical_pressure"
            elif exhausted:
                rule = "memory_margin" if resources["ram"] <= 0 else "device_exhausted"
            elif escalated(baseline, memory):
                rule = "pressure_escalation"
            elif "process_swap" in paging:
                rule = "process_paging"
            elif paging and sustained_paging(paging, now):
                rule = "attributed_paging"
            elif need is not None and memory.available - need < margin_bytes:
                rule = "predicted_margin"
            # Past its prediction (and its error), a child may grow by as
            # much again as it has overrun so far. It stops when the room left
            # would not hold twice that; an overrun of a few percent with room
            # to spare is no danger (live: 7.0 GB over 6.7 GB with 147 GB
            # free, 10.5 GB over 9.9 GB with 23.8 GB of the card free; each
            # stop cost a reload).
            elif (
                predicted_peak_bytes is not None
                and rss is not None
                and rss > predicted_peak_bytes + prediction_error_bytes
                and memory.available - margin_bytes
                < 2 * (rss - predicted_peak_bytes)
            ):
                rule = "prediction_exceeded"
            elif any(
                d["available_bytes"] < d["remaining_need_bytes"]
                for d in device_predictions
            ):
                rule = "predicted_device_margin"
            elif any(
                d["used_bytes"] > predicted_device_bytes + device_prediction_error_bytes
                and d["available_bytes"]
                < 2 * (d["used_bytes"] - predicted_device_bytes)
                for d in device_predictions
            ):
                rule = "device_prediction_exceeded"
            elif imminent and projection:
                # Weights loading and prefill allocate in short bursts. Only a
                # projection that holds for the grace period interrupts; real
                # exhaustion is caught above by the margin rule.
                projected_since = (
                    projected_since if projected_since is not None else now
                )
                if now - projected_since >= policy.grace_seconds:
                    rule = "allocation_rate"
            if not imminent:
                projected_since = None
            alarm = (
                stopped_sample(
                    memory,
                    rule,
                    baseline=baseline,
                    prediction=prediction,
                    tree_rss_bytes=rss,
                    tree_growth_bytes=own_growth,
                    tree_swap_bytes=swapped,
                    paging=paging,
                    projection=imminent,
                )
                if rule
                else None
            )
            if memory.pressure == "critical" or exhausted:
                notify(on_critical, alarm, True)
                return alarm
            if alarm is not None:
                if warned_at is None:
                    warned_at, warning = now, alarm
                    if not notify(on_warn, alarm):
                        return alarm
            if warned_at is not None and now - warned_at >= policy.grace_seconds:
                alarm = stopped_sample(
                    memory,
                    warning.stop_reason["rule"],
                    baseline=baseline,
                    prediction=prediction,
                    grace_expired=True,
                    trigger=warning.stop_reason,
                )
                notify(on_critical, alarm, True)
                return alarm
            previous, last_time = resources, now
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
            # Losing hardware telemetry while allocating must stop the child.
            alarm = stopped_sample(
                last,
                "telemetry_unavailable",
                baseline=baseline,
                prediction=prediction,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            notify(on_critical, alarm, True)
            return alarm
        stop.wait(interval)
    return None


def guarded_run(
    argv,
    timeout=None,
    *,
    platform=None,
    check=lambda: None,
    policy=None,
    required_bytes=0,
    required_device_bytes=None,
    on_sample=None,
    margin_bytes=0,
    pressure_data=None,
    prediction_error_bytes=0,
    device_prediction_error_bytes=0,
    **kwargs,
):
    """subprocess.run equivalent, with isolation, bounded cancellation and guard."""
    platform, policy = platform or current(), policy or GuardPolicy()
    interpreter_bytes = (
        platform.process_bytes(os.getpid()) if hasattr(platform, "process_bytes") else 0
    )
    memory = capture_baseline(
        platform,
        policy=policy,
        check=check,
        on_sample=on_sample,
        required_bytes=max(required_bytes, interpreter_bytes or 0),
        required_device_bytes=required_bytes
        if required_device_bytes is None
        else required_device_bytes,
        margin_bytes=margin_bytes,
        pressure_data=pressure_data,
    )
    margin_bytes = learned_margin(memory, pressure_data, margin_bytes)
    events = []
    finished = threading.Event()
    kwargs = process_options() | kwargs
    process = subprocess.Popen(owned_command(argv), **kwargs)

    def warning(sample):
        events.append(sample or memory)
        interrupt(process)

    def critical(sample):
        events.append(sample or memory)
        kill(process)

    watcher = threading.Thread(
        target=guard,
        args=(process, platform, warning, critical),
        kwargs={
            "policy": policy,
            "stop": finished,
            "on_sample": on_sample,
            "baseline": memory,
            "margin_bytes": margin_bytes,
            "predicted_peak_bytes": required_bytes or None,
            "predicted_device_bytes": (
                required_bytes
                if required_device_bytes is None
                else required_device_bytes
            )
            or None,
            "prediction_error_bytes": prediction_error_bytes,
            "device_prediction_error_bytes": device_prediction_error_bytes,
        },
        daemon=True,
    )
    watcher.start()
    start = time.monotonic()
    try:
        while True:
            check()
            if timeout is not None and time.monotonic() - start >= timeout:
                raise subprocess.TimeoutExpired(argv, timeout)
            try:
                out, err = process.communicate(timeout=policy.min_interval)
                break
            except subprocess.TimeoutExpired:
                continue
        if events:
            raise MemoryPressure(events[-1])
        if stop := child_memory_stop(process.returncode, err):
            raise stop
        return subprocess.CompletedProcess(argv, process.returncode, out, err)
    finally:
        if running(process):
            kill(process)
        process.wait()
        finished.set()
        watcher.join(timeout=policy.max_interval + policy.min_interval)
