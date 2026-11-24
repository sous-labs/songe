"""Size each slice from measured costs and the machine's pressure boundary."""

import itertools
import math
from dataclasses import asdict

# Always leave this share of RAM free (product decision: 10-15%).
RAM_RESERVE_FRACTION = 0.10


class PressureHistory:
    def __init__(self, data=None):
        self.data = dict(data or {})

    def observe(self, sample):
        host_pressure = getattr(sample, "host_pressure", sample.pressure)
        self._observe_boundary(sample, host_pressure)
        # A warning baseline needs the warning->critical boundary, not the
        # normal->warning boundary. Apply the same directional/expiry rules
        # independently, including across persisted history reloads.
        critical = PressureHistory(self.data.get("critical_boundary"))
        critical._observe_boundary(
            sample, "critical" if host_pressure == "critical" else "normal"
        )
        self.data["critical_boundary"] = critical.data

    def _observe_boundary(self, sample, host_pressure):
        previous = self.data.get("previous")
        if self.data.get("total", sample.total) != sample.total:
            self.data.clear()
            previous = None
        self.data["total"] = sample.total
        if host_pressure == "normal":
            self.data["normal_samples"] = self.data.get("normal_samples", 0) + 1
        else:
            if previous and previous["pressure"] == "normal":
                self.data["warning_samples"] = 0
            self.data["warning_samples"] = self.data.get("warning_samples", 0) + 1
            self.data["normal_samples"] = 0
        if host_pressure == "normal" and sample.available <= self.margin:
            # Normal operation below a saved boundary contradicts that boundary.
            # PSI can signal IO stalls even with ample RAM; do not fossilize it.
            self.data.pop("warning_available", None)
            self.data.pop("transition_width", None)
        if previous and (previous["pressure"] == "normal") != (
            host_pressure == "normal"
        ):
            normal = (
                sample.available if host_pressure == "normal" else previous["available"]
            )
            warning = (
                previous["available"] if host_pressure == "normal" else sample.available
            )
            if normal > warning and host_pressure != "normal":
                self.data["warning_available"] = warning
                self.data["transition_width"] = normal - warning
            elif normal <= warning:
                self.data.pop("warning_available", None)
                self.data.pop("transition_width", None)
        if self.data.get("normal_samples", 0) > self.data.get("warning_samples", 0):
            # A one-sample PSI event cannot veto work indefinitely. Retain the
            # bracket only while its warning evidence outweighs normal recovery.
            self.data.pop("warning_available", None)
            self.data.pop("transition_width", None)
        self.data["previous"] = {
            "available": sample.available,
            "pressure": host_pressure,
        }

    @property
    def margin(self):
        return self.data.get("warning_available", 0) + self.data.get(
            "transition_width", 0
        )

    def margin_for(self, sample):
        # Warn is the OS making room, not harm: the margin is learned only
        # from the boundary where critical pressure began.
        critical = PressureHistory(self.data.get("critical_boundary")).margin
        return max(int(sample.total * RAM_RESERVE_FRACTION), critical)


def dimensions(row):
    return (
        row["sequence"],
        row.get("response_tokens", row.get("sample_max_tokens", 1)),
        row.get("token_chunk", 1),
        row.get("accumulation", 1),
    )


def features(size):
    sequence, response, chunk, _ = size
    # Every optimizer step uses the same gradient buffer. Serial accumulation
    # changes duration, never peak memory. Envelope all measured batch counts.
    return (sequence, response, chunk, 1)


def spread(rows, resource):
    values = [r[resource] for r in rows]
    return max(values, default=0) - min(values, default=0)


def far_apart(a, b, dimension):
    """Secants need a real span: bisection measures lengths a few tokens
    apart, where run-to-run noise (~1 GB) would read as tens of MB per token."""
    if a == b:
        return False
    if dimension != 0:
        return True
    return abs(a - b) >= max(256, max(a, b) // 4)


def cost_model(rows, resource):
    """Nonnegative per-token secants, with an upper envelope of measured peaks.

    Only one dimension changes in each calibration pair. The envelope covers
    every measured interaction; the guard still stops unseen allocation bursts.
    No prediction extrapolates outside the calibrated dimension ranges.
    """
    peaks = {}
    for row in rows:
        size = features(dimensions(row))
        peaks[size] = max(peaks.get(size, 0), row[resource])
    samples = list(peaks.items())
    if len(samples) < 2:
        return None
    slopes = []
    for dimension in range(4):
        slopes.append(
            max(
                (
                    max(0, (b - a) / (y[dimension] - x[dimension]))
                    for (x, a), (y, b) in itertools.combinations(samples, 2)
                    if far_apart(x[dimension], y[dimension], dimension)
                    and all(x[i] == y[i] for i in range(4) if i != dimension)
                ),
                default=0,
            )
        )
    intercept = max(
        peak - sum(s * v for s, v in zip(slopes, x, strict=True)) for x, peak in samples
    )
    model = {"base_bytes": intercept, "bytes_per_unit": slopes}
    # The measured spread around the envelope is the tolerated prediction
    # error. Exact/one-row measurements have no invented percentage allowance.
    model["error_bytes"] = max(
        abs(r[resource] - predict(model, dimensions(r))) for r in rows
    )
    # Inside the measured range, measurements beat the envelope: memory grows
    # with length, so the nearest measured length at or above a size bounds it.
    model["points"] = [[list(size), peak] for size, peak in samples]
    return model


def predict(model, size):
    target = features(size)
    above = [
        (point[0], peak)
        for point, peak in model.get("points", [])
        if point[1:3] == list(target[1:3]) and point[0] >= target[0]
    ]
    if above:
        nearest = min(length for length, _ in above)
        return max(peak for length, peak in above if length == nearest)
    return max(
        0,
        math.ceil(
            model["base_bytes"]
            + sum(
                s * v
                for s, v in zip(model["bytes_per_unit"], features(size), strict=True)
            )
        ),
    )


def corrected(calibration):
    """Measured peaks scaled by what real slices were seen to use.

    Measuring covers each part (student window, long teacher) on its own; a
    real slice combines them and can need more. A guard stop for exceeding
    the prediction records the ratio (learn_peak), so the next slice is sized
    for what it really uses instead of stopping the same way again. Host
    memory and accelerator memory each have their own correction."""
    factors = {
        "host_peak_bytes": (calibration or {}).get("peak_correction") or 1.0,
        "device_peak_bytes": (calibration or {}).get("device_peak_correction") or 1.0,
    }
    factors = {k: f for k, f in factors.items() if f > 1}
    if not factors:
        return calibration

    def scaled(peaks):
        peaks = dict(peaks)
        for key, factor in factors.items():
            if peaks.get(key):
                peaks[key] = int(peaks[key] * factor)
        return peaks

    def scale(row):
        row = scaled(row)
        if row.get("phase_peaks"):
            row["phase_peaks"] = {
                phase: scaled(peak) for phase, peak in row["phase_peaks"].items()
            }
        return row

    return calibration | {
        "rows": [scale(r) for r in calibration.get("rows", [])],
        "teacher_rows": [scale(t) for t in calibration.get("teacher_rows", [])],
    }


def learn_peak(store, stop):
    """Raise the correction after a slice outgrew its prediction."""
    reason = (stop or {}).get("stop_reason") or {}
    # After the grace period the first alarm is kept as observed["trigger"].
    trigger = (
        reason.get("trigger") or (reason.get("observed") or {}).get("trigger") or reason
    )
    prediction = trigger.get("prediction") or {}
    if trigger.get("rule") == "prediction_exceeded":
        key = "peak_correction"
        observed = (trigger.get("observed") or {}).get("tree_rss_bytes")
        predicted = prediction.get("peak_bytes")
    elif trigger.get("rule") == "device_prediction_exceeded":
        key = "device_peak_correction"
        observed = max(
            (d.get("used_bytes") or 0 for d in prediction.get("devices") or []),
            default=0,
        )
        predicted = prediction.get("device_peak_bytes")
    else:
        return None
    if not observed or not predicted:
        return None
    saved = store.get("calibration") or {}
    factor = saved.get(key) or 1.0
    # The trigger fires on the way up; allow a tenth more than was seen.
    saved[key] = round(min(3.0, max(factor, factor * observed * 1.1 / predicted)), 3)
    store.set("calibration", saved)
    return saved[key]


def choose_size(config, calibration, sample, pressure_data):
    if sample.pressure == "critical":
        return None
    calibration = corrected(calibration)
    rows = [
        r
        for r in (calibration or {}).get("rows", [])
        if "host_peak_bytes" in r and "device_peak_bytes" in r
    ]
    # A long teacher/summary prefill is independent of the compact student
    # dimension. Interpolating its peak downward with student length is unsafe.
    forward_floor = {
        resource: max(
            (
                p.get(resource + "_peak_bytes", 0)
                for r in rows
                for phase, p in r.get("phase_peaks", {}).items()
                if phase in {"teacher", "summary"}
            ),
            default=0,
        )
        for resource in ("host", "device")
    }
    margin = PressureHistory(pressure_data).margin_for(sample)
    if not rows:
        # Legacy measurements cannot identify separate RAM/device costs. Keep
        # their original budget until the next calibration; never invent savings.
        host = int((config.calibrated_host_gb or config.calibrated_peak_gb) * 1e9)
        device = int(config.calibrated_device_gb * 1e9)
        size = (
            config.max_sequence,
            config.response_tokens,
            config.token_chunk,
            config.accumulation,
        )
        models = None
    elif len({features(dimensions(r)) for r in rows}) < 2:
        # Serial accumulation is not another memory geometry. Use the measured
        # envelope verbatim; a single point cannot support interpolation.
        matching = [
            r
            for r in rows
            if r["sequence"] <= config.max_sequence
            and dimensions(r)[1] == config.response_tokens
            and dimensions(r)[2] <= config.token_chunk
        ]
        if not matching:
            return None
        row = max(matching, key=dimensions)
        size = (*dimensions(row)[:3], config.accumulation)
        host = max(r["host_peak_bytes"] for r in rows)
        device = max(r["device_peak_bytes"] for r in rows)
        models = None
    else:
        limits = (
            config.max_sequence,
            config.response_tokens,
            config.token_chunk,
            config.accumulation,
        )
        prompt_tokens = min(r["sequence"] - dimensions(r)[1] for r in rows)
        # Rollout length is training policy too: lowering it would cut a sample
        # before its length is known. Reserve its full budget at admission.
        minimum = (
            max(
                min(r["sequence"] for r in rows), prompt_tokens + config.response_tokens
            ),
            config.response_tokens,
            min(dimensions(r)[2] for r in rows),
            config.accumulation,
        )
        maximum = tuple(
            min(limits[i], max(dimensions(r)[i] for r in rows)) for i in range(3)
        ) + (config.accumulation,)
        boundary = min(
            (
                r["sequence"]
                for r in (calibration or {}).get("capacity_limits", [])
                if dimensions(r)[1] <= config.response_tokens and dimensions(r)[2] == 1
            ),
            default=None,
        )
        if boundary is not None:
            maximum = (min(maximum[0], boundary - 1), *maximum[1:])
        if config.response_tokens > maximum[1]:
            return None
        if any(a > b for a, b in zip(minimum, maximum, strict=True)):
            return None
        models = {
            resource: cost_model(
                rows + (calibration or {}).get("capacity_limits", []),
                resource + "_peak_bytes",
            )
            for resource in ("host", "device")
        }

        def fits(size):
            host, device = (
                max(forward_floor[r], predict(models[r], size))
                for r in ("host", "device")
            )
            return (
                host + margin <= sample.available
                and (not device or bool(sample.vram))
                and all(device <= a.memory_free for a in sample.vram)
            )

        # Monotone costs allow binary search per dimension. Prefer retaining
        # context, then chunk throughput. The response and optimizer's
        # sample count is training policy, independent of this memory slice.
        size = list(minimum)
        if not fits(size):
            # Interactions can make a fitted envelope very conservative. An
            # exact measured geometry is still usable, with its observed peak.
            options = []
            for row in rows:
                candidate = (*dimensions(row)[:3], config.accumulation)
                if candidate[1] != config.response_tokens or any(
                    a > b for a, b in zip(candidate, maximum, strict=True)
                ):
                    continue
                same = [
                    r
                    for r in rows + (calibration or {}).get("capacity_limits", [])
                    if dimensions(r)[:3] == candidate[:3]
                ]
                host = max(
                    forward_floor["host"], max(r["host_peak_bytes"] for r in same)
                )
                device = max(
                    forward_floor["device"], max(r["device_peak_bytes"] for r in same)
                )
                if (
                    host + margin <= sample.available
                    and (not device or sample.vram)
                    and all(device <= a.memory_free for a in sample.vram)
                ):
                    options.append((candidate, host, device))
            if not options:
                return None
            size, host, device = max(options)
            models = None
        else:
            for i in (0, 2):
                low, high = size[i], maximum[i]
                if i == 2:
                    high = min(high, size[1])
                while low < high:
                    mid = (low + high + 1) // 2
                    trial = size.copy()
                    trial[i] = mid
                    if fits(trial):
                        low = mid
                    else:
                        high = mid - 1
                size[i] = low
            host, device = (predict(models[r], size) for r in ("host", "device"))
    host, device = (
        max(host, forward_floor["host"]),
        max(device, forward_floor["device"]),
    )
    if (
        host + margin > sample.available
        or (device and not sample.vram)
        or any(device > a.memory_free for a in sample.vram)
    ):
        return None
    measured = [r for r in rows if dimensions(r)[:3] == tuple(size)[:3]]
    # The admitted long teachers run inside this slice: the guard's prediction
    # is the larger of the student envelope and their measured peaks.
    teachers = [
        t
        for t in (calibration or {}).get("teacher_rows", [])
        if t["host_peak_bytes"] + margin <= sample.available
        and (
            not t.get("device_peak_bytes")
            or all(t["device_peak_bytes"] <= a.memory_free for a in sample.vram)
        )
    ]
    host = max([host] + [t["host_peak_bytes"] for t in teachers])
    device = max([device] + [t.get("device_peak_bytes") or 0 for t in teachers])
    return dict(
        zip(
            ("max_sequence", "response_tokens", "token_chunk", "accumulation"),
            size,
            strict=True,
        )
    ) | {
        # Teacher forwards were measured separately and included in each row's
        # peak envelope. Never infer their length from the backprop dimension.
        # Long teachers have their own rows; only those that fit today's free
        # memory are admitted, the rest wait rather than being judged unfit.
        **(
            {
                "max_teacher_sequence": max(
                    [r.get("teacher_sequence", 0) for r in rows]
                    + [
                        t["teacher_sequence"]
                        for t in (calibration or {}).get("teacher_rows", [])
                        if t["host_peak_bytes"] + margin <= sample.available
                        and (
                            not t.get("device_peak_bytes")
                            or all(
                                t["device_peak_bytes"] <= a.memory_free
                                for a in sample.vram
                            )
                        )
                    ]
                )
            }
            if any(r.get("teacher_sequence") for r in rows)
            else {}
        ),
        "host_bytes": host,
        "device_bytes": device,
        "margin_bytes": margin,
        # A measured geometry's own run-to-run spread is its error; only an
        # unmeasured size falls back to the envelope's error.
        # Never zero: one measurement has no spread, but runs differ by a
        # few percent and a stop for that is a false alarm.
        "prediction_error_bytes": max(
            host // 20,
            spread(measured, "host_peak_bytes")
            if measured or not models
            else models["host"]["error_bytes"],
        ),
        "device_prediction_error_bytes": max(
            device // 20,
            spread(measured, "device_peak_bytes")
            if measured or not models
            else models["device"]["error_bytes"],
        ),
        "cost_model": models,
        "memory": asdict(sample),
    }


def free_needed(config, calibration, sample, pressure_data, accept=None):
    """The least free RAM at which choose_size admits work (that `accept`
    takes), found with the same rule that refused it; None if never."""
    from dataclasses import replace

    def fits(available):
        sizing = choose_size(
            config, calibration, replace(sample, available=available), pressure_data
        )
        return sizing is not None and (accept is None or accept(sizing))

    high = getattr(sample, "total", 0) or 0
    if high <= sample.available or not fits(high):
        return None
    low = sample.available
    while high - low > 2**26:
        middle = (low + high) // 2
        if fits(middle):
            high = middle
        else:
            low = middle
    return high


def teacher_card_shortfall(calibration, sample, length):
    """Card memory still missing for the shortest measured teacher of at least
    `length` tokens, as training would predict it; None when there is no
    card or no such measurement."""
    rows = [
        t
        for t in corrected(calibration or {}).get("teacher_rows", [])
        if t.get("teacher_sequence", 0) >= length and t.get("device_peak_bytes")
    ]
    if not rows or not sample.vram:
        return None
    need = min(rows, key=lambda t: t["teacher_sequence"])["device_peak_bytes"]
    return max(0, need - min(a.memory_free for a in sample.vram))


def gigabytes(value):
    return f"{value / 1e9:.1f} GB"


def waiting_reason(sample, pressure_data, needed=None, *, what=None):
    what = what or "smallest calibrated work does not fit"
    device = "".join(
        f" · device memory {a.name} {gigabytes(a.memory_free)} free"
        for a in sample.vram
    )
    return (
        f"Waiting for memory: {what} · {gigabytes(sample.available)} free"
        + (f", needs {gigabytes(needed)}" if needed else "")
        + device
    )
