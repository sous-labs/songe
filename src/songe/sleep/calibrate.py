import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

from ..config import Config, data_dir
from ..log import Store
from ..storage import write_json


def full_length_sampler(backend):
    """Measure the requested response budget even when the model emits EOS."""
    return getattr(backend, "sample_full_length", backend.sample)


def costs(row, accumulation, response_tokens):
    """Measured optimizer/score costs, for slice planning and display only."""
    step = (
        row["step_seconds"]
        * accumulation
        / row.get("accumulation", 1)
        * max(
            1,
            response_tokens
            / row.get("sample_max_tokens", row.get("response_tokens", response_tokens)),
        )
    )
    # Legacy measurements without a forward pass can size optimizer units,
    # but do not supply a scoring ETA.
    score = (
        response_tokens / row["generation_tps"] + row["forward_seconds"]
        if "forward_seconds" in row
        else 0
    )
    return step, score


def derive(
    rows,
    free_gb,
    accumulation=4,
    response_tokens=128,
    *,
    device_free_gb=float("inf"),
):
    usable = [
        r
        for r in rows
        if r.get("step_seconds", 0) > 0
        and r.get(
            "host_peak_bytes",
            r.get("phase_peak_memory_gb", r.get("peak_memory_gb", 0)) * 1e9,
        )
        <= free_gb * 1e9
        and r.get("device_peak_bytes", 0) <= device_free_gb * 1e9
    ]
    if not usable:
        raise ValueError("No sequence length fits the measured memory budget")
    from .sizing import dimensions

    full_response = [r for r in usable if dimensions(r)[1] >= response_tokens]
    row = max(full_response or usable, key=dimensions)
    response_tokens = row.get(
        "response_tokens", min(response_tokens, row["sequence"] - 1)
    )
    step_seconds, score_seconds = costs(row, accumulation, response_tokens)
    return {
        "max_sequence": row["sequence"],
        "response_tokens": response_tokens,
        "token_chunk": min(row.get("token_chunk", 1), response_tokens),
        "accumulation": accumulation,
        "calibrated_step_seconds": step_seconds,
        "calibrated_score_seconds": score_seconds,
        "calibrated_peak_gb": row.get(
            "host_peak_bytes",
            row.get("phase_peak_memory_gb", row.get("peak_memory_gb", 0)) * 1e9,
        )
        / 1e9,
        "calibrated_host_gb": row.get(
            "host_peak_bytes",
            row.get("phase_peak_memory_gb", row.get("peak_memory_gb", 0)) * 1e9,
        )
        / 1e9,
        "calibrated_device_gb": row.get("device_peak_bytes", 0) / 1e9,
        "calibration_available": True,
    }


def calibrate(config, root, lengths=(256, 512, 1024, 2048)):
    """Public entry always isolates the allocation-heavy calibration."""
    from ..memory import (
        MemoryPressure,
        TrainingPlatform,
        guarded_run,
        record_memory_stop,
    )
    from ..platform import current
    from ..telemetry import configure

    configure(root)
    config.save(root)
    store = Store(root).scoped(config.model_key) if config.model_key else Store(root)
    from .sizing import PressureHistory

    history = PressureHistory(store.get("memory_pressure_history", {}))
    first_sample = True

    def observe(sample):
        nonlocal first_sample
        if first_sample or getattr(sample, "paging_noise", None):
            store.set("last_memory_baseline", asdict(sample))
            first_sample = False
        history.observe(sample)
        store.set("memory_pressure_history", history.data)

    try:
        result = guarded_run(
            [
                sys.executable,
                "-m",
                "songe.sleep.calibrate",
                "--data-dir",
                str(root),
                "--lengths",
                ",".join(map(str, lengths)),
                "--child",
                *(["--model-key", config.model_key] if config.model_key else []),
            ],
            platform=TrainingPlatform(current(), config),
            on_sample=observe,
            pressure_data=history.data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode:
            from ..telemetry import failure_hint, log_child_stderr

            log_child_stderr(getattr(result, "stderr", None))
            if not record_buffer_ceiling(store, getattr(result, "stderr", None)):
                raise RuntimeError(failure_hint("Calibration child", root))
            if not store.get("calibration", {}).get("rows"):
                return []
    except MemoryPressure as exc:
        record_memory_stop(store, exc, calibration=True)
        # Completed small configurations remain usable when a larger probe
        # reaches pressure. The guard has already stopped the allocating child.
        if not store.get("calibration", {}).get("rows"):
            raise
    finally:
        collect_calibration(root, store)
    if store.get("disk_wait"):
        return []
    calibrated = (
        Config.for_model(root, config.model_key)
        if config.model_key
        else Config.load(root)
    )
    # A pressure stop may leave only the smaller completed measurements. Publish
    # their compatible dimensions together, after the allocating child exits.
    saved = store.get("calibration", {})
    if saved.get("budget"):
        for key in derive_keys():
            if key in saved["budget"]:
                setattr(calibrated, key, saved["budget"][key])
        calibrated.save(root)
    for key in derive_keys():
        setattr(config, key, getattr(calibrated, key))
    return (
        store.get("calibration", {}).get("rows")
        or json.loads((root / "calibration.json").read_text(encoding="utf-8"))["rows"]
    )


def derive_keys():
    # Geometry is user policy. Measurements describe slices, and must never
    # replace the limits that subsequent idle gaps are allowed to explore.
    return (
        "calibrated_step_seconds",
        "calibrated_score_seconds",
        "calibrated_peak_gb",
        "calibrated_host_gb",
        "calibrated_device_gb",
        "calibration_available",
    )


def probe_sizes(policy, minimum, lengths, target, rows, failures):
    """Resume at the last geometry; bisect only absolute capacity failures."""

    from .sizing import dimensions

    done = {dimensions(r) for r in rows + failures}
    last = max((r["sequence"] for r in rows), default=minimum)
    boundary = min(
        (
            r["sequence"]
            for r in failures
            if dimensions(r)[1] <= policy.response_tokens and dimensions(r)[2] == 1
        ),
        default=None,
    )
    if boundary is not None:
        low = max(
            (
                r["sequence"]
                for r in rows
                if dimensions(r)[1] == policy.response_tokens
                and dimensions(r)[2] == 1
                and r["sequence"] < boundary
            ),
            default=minimum - 1,
        )
        from .calibration_state import close_enough

        lengths = [] if close_enough(low, boundary) else [(low + boundary) // 2]
    else:
        lengths = sorted({minimum, *lengths, target, last})
        lengths = [
            n
            for n in lengths
            if last <= n <= max(target, last) and n <= policy.max_sequence
        ]
    for length in lengths:
        response = min(policy.response_tokens, length - minimum + 1)
        chunk_full = min(policy.token_chunk, response)
        # Fallback geometries are memory probes at one micro-batch. The real
        # training geometry is measured once with its accumulation, so its
        # peak includes whatever serial accumulation actually retains.
        for r, chunk, batch in sorted(
            {
                (1, 1, 1),
                (response, 1, 1),
                (response, chunk_full, policy.accumulation),
            }
        ):
            size = (length, r, chunk, batch)
            if size in done or chunk > r:
                continue
            # A larger chunk cannot rescue an absolute smallest-chunk failure.
            if any(
                all(a >= b for a, b in zip(size[:3], dimensions(f)[:3], strict=True))
                for f in failures
            ):
                continue
            yield size


def save_progress(config, store, saved, target, pool):
    from .calibration_state import progress

    before = (store.get("calibration_progress") or {}).get("measured_sequence")
    saved["progress"] = progress(config, saved, target, pool)
    measured = saved["progress"].get("measured_sequence")
    if measured and measured != before:
        store.event(
            "measure",
            f"{measured} of {saved['progress'].get('target_sequence') or measured} tokens fit",
        )
    write_json(store.root / "calibration.json", saved)
    store.set("calibration", saved)
    store.set("calibration_progress", saved["progress"])


def release_fitting_unfit(config, store, saved):
    """Re-judge earlier "never fits" verdicts under today's measurements.

    A verdict made mid-calibration, or by an older estimator, removed the
    sample from scoring. Any that fit now go back to be scored."""
    from .samples import SampleAdmission

    unfit = store.get("unfit_samples", {}) or {}
    if not unfit or not saved:
        return 0
    admission = SampleAdmission(config, None, saved, store, None)
    keep, released = {}, []
    for key, entry in unfit.items():
        required = entry.get("required_sequence")
        if required is not None and admission.never_fit(required) is None:
            released.append(entry)
        else:
            keep[key] = entry
    if released:
        store.release_unfit(keep, released)
    return len(released)


def mark_capacity_samples(config, store, saved):
    from ..steps import revision_id
    from .calibration_state import sample_pool
    from .samples import SampleAdmission

    release_fitting_unfit(config, store, saved)
    admission = SampleAdmission(config, None, saved, store, None)
    lengths = {
        (s["id"], s["revision"]): s["required_sequence"]
        for s in saved.get("samples", [])
    }
    for step in sample_pool(store, config):
        if step.get("split") == "heldout":
            continue  # Gate forwards do not use the student backprop limit.
        required = lengths.get((step["id"], revision_id(step)))
        reason = admission.never_fit(required) if required is not None else None
        if reason and not admission.known_unfit(step):
            entry = {
                "id": step["id"],
                "revision": revision_id(step),
                "required_sequence": required,
                "reason": reason,
            }
            store.mark_unfit(admission.key(step), entry, step)


def absolute_capacity(config, total):
    """The most this machine can ever give one process, whatever else runs."""
    return min(total * config.memory_fraction, total * 0.9)


BUFFER_CEILING = re.compile(
    rb"Attempting to allocate (\d+) bytes which is greater than the maximum"
    rb" allowed buffer size"
)


def record_buffer_ceiling(store, stderr):
    """Metal refuses any single buffer above a fixed device size. The size
    being measured can never fit, however much memory is free: it is a
    capacity limit, and measuring continues with the smaller sizes."""
    from .calibration_state import policy_key

    match = BUFFER_CEILING.search(stderr or b"")
    saved = store.get("calibration", {})
    size = saved.get("in_progress")
    state = saved.get("progress") or {}
    if not match or not size or not state.get("target_sequence"):
        return False
    config = (
        Config.for_model(store.root, store.model_key)
        if store.model_key
        else Config.load(store.root)
    )
    if state.get("policy") != policy_key(config):
        return False
    kind = size.get("kind", "sequence")
    probe = {k: v for k, v in size.items() if k not in {"kind", "phase"}}
    requested = int(match[1])
    failed = probe | {"host_peak_bytes": requested, "device_peak_bytes": requested}
    failures = saved.setdefault(
        "teacher_limits" if kind == "teacher" else "capacity_limits", []
    )
    if failed not in failures:
        failures.append(failed)
    saved.pop("in_progress", None)
    save_progress(config, store, saved, state["target_sequence"], state["pool"])
    mark_capacity_samples(config, store, saved)
    store.event(
        "measure",
        f"{probe.get('sequence', '?')} tokens exceed this GPU's largest buffer"
        " · measuring smaller sizes",
    )
    return True


def record_capacity_stop(store, saved):
    """Only total capacity is permanent. A stop the probe caused while other
    apps held memory says what the size costs, not that it never fits: the
    size is held until that much memory is free again."""
    from .calibration_state import policy_key

    size = saved.get("in_progress")
    state = saved.get("progress") or {}
    if not size or not state.get("target_sequence"):
        return
    config = (
        Config.for_model(store.root, store.model_key)
        if store.model_key
        else Config.load(store.root)
    )
    if state.get("policy") != policy_key(config):
        return
    stop = saved.get("memory_stop") or {}
    reason = stop.get("stop_reason") or {}
    observed = reason.get("observed") or {}
    own_bytes = observed.get("tree_rss_bytes") or 0
    total = stop.get("total", 0)
    if not total or not own_bytes:
        return
    kind = size.get("kind", "sequence")
    probe = {k: v for k, v in size.items() if k not in {"kind", "phase"}}
    # The probe accounts for the whole drop from the headroom it started with.
    drop = (reason.get("baseline") or {}).get("available", 0) - observed.get(
        "available", 0
    )
    if own_bytes > absolute_capacity(config, total):
        key = "teacher_limits" if kind == "teacher" else "capacity_limits"
        failed = probe | {"host_peak_bytes": own_bytes, "device_peak_bytes": 0}
        failures = saved.setdefault(key, [])
        if failed not in failures:
            failures.append(failed)
    elif size.get("phase") in SIZED_PHASES and 0 < drop <= 2 * own_bytes:
        # Loading, export and evaluation cost the same at every size; a stop
        # there has nothing to shrink and simply retries later. A sized
        # probe that caused at least half the drop holds the size until the
        # whole drop is free again (live: a 4096 stop while paging recorded
        # nothing and was probed again); measuring then completes at the
        # largest size that fit. Something else taking most of it: a retry.
        entry = probe | {"kind": kind, "needed_bytes": max(own_bytes, drop)}
        saved.setdefault("held", []).append(entry)
        from .calibration_state import held_key

        retries = saved.setdefault("held_retries", {})
        retries[held_key(entry)] = retries.get(held_key(entry), 0) + 1
    else:
        return
    saved.pop("in_progress", None)
    save_progress(config, store, saved, state["target_sequence"], state["pool"])
    mark_capacity_samples(config, store, saved)


# Phases whose memory grows with the probed length.
SIZED_PHASES = {"teacher", "train"}


def _calibrate(config, root, lengths):
    from ..platform.locking import acquire

    with (root / "worker.lock").open("a+", encoding="utf-8") as lease:
        acquire(lease)
        from ..artifacts.storage import admit_storage, collect_scratch
        from ..platform import current

        store = (
            Store(root).scoped(config.model_key) if config.model_key else Store(root)
        )
        collect_calibration(root, store, leased=True)
        collect_scratch(store)
        admit_storage(config, store, current(), calibration=True)
        if config.model_key:
            from ..models.registry import prepare

            prepare(config, Store(root).scoped(config.model_key), current())
        return _measure(config, root, lengths)


def collect_calibration(root, store, *, leased=False):
    """Remove this installation's abandoned calibration files under its lease."""
    import shutil

    from ..platform.locking import acquire

    if not leased:
        with (root / "worker.lock").open("a+", encoding="utf-8") as lease:
            try:
                acquire(lease)
            except BlockingIOError:
                return
            return collect_calibration(root, store, leased=True)
    from ..artifacts.storage import artifact_roots

    for old in (
        old
        for directory in artifact_roots(root)
        for old in directory.glob("calibration-*")
    ):
        marker = old / "songe_calibration.json"
        if old.is_symlink() or not marker.is_file():
            continue
        try:
            identity = json.loads(marker.read_text(encoding="utf-8")).get(
                "installation_id"
            )
        except (ValueError, OSError):
            continue
        if identity == store.get("installation_id"):
            shutil.rmtree(old)


def _measure(config, root, lengths):
    import gc

    from ..backends import choose
    from ..memory import TrainingPlatform, admit
    from ..platform import Rooted, current
    from ..schedule import probe
    from .calibration_state import measure_pool, policy_key
    from .fuse import candidate_kind, evaluation_model, export_candidate
    from .measurement import Meter
    from .prompts import paired_prompts
    from .sizing import PressureHistory

    policy = replace(config)
    store = Store(root).scoped(config.model_key) if config.model_key else Store(root)
    collect_calibration(root, store, leased=True)
    platform = Rooted(TrainingPlatform(current(), policy), root)
    hardware = probe(platform)
    if not hardware.available:
        raise ValueError(
            "Hardware probes are unavailable; calibration requires live memory telemetry"
        )
    saved = store.get("calibration") or {}
    previous_policy = (saved.get("progress") or {}).get("policy")
    if saved.get("model", policy.model) != policy.model or (
        previous_policy and previous_policy != policy_key(policy)
    ):
        saved = {}
    rows = [
        r
        for r in saved.get("rows", [])
        if "host_peak_bytes" in r and "device_peak_bytes" in r
    ]
    failures = saved.get("capacity_limits", [])
    history = PressureHistory(store.get("memory_pressure_history", {}))
    baseline = admit(platform, 0, 0, pressure_data=history.data)
    margin = history.margin_for(baseline)
    # A held size is tried again once what it needs is free.
    saved["held"] = [
        h
        for h in saved.get("held", [])
        if h["needed_bytes"] + margin > baseline.available
    ]
    previous_memory = saved.get("memory_baseline") or {}
    if previous_memory and (
        previous_memory.get("total") != baseline.total
        or [
            (a.get("kind"), a.get("memory_total"))
            for a in previous_memory.get("vram", [])
        ]
        != [(a.kind, a.memory_total) for a in baseline.vram]
    ):
        saved, rows, failures = {}, [], []
    # Python 3.11/3.12's Windows monotonic clock can round short probes to zero.
    # Use the high-resolution performance counter for measured phase costs.
    loading = time.perf_counter()
    backend_type = choose(policy, platform)
    with Meter(platform, baseline, interval=policy.guard_min_interval) as load_meter:
        backend = backend_type(
            replace(policy, response_tokens=1, token_chunk=1, accumulation=1),
            lambda: None,
        )
    load_seconds = time.perf_counter() - loading
    backend.prepare()
    smallest_step = {
        "id": "calibration",
        "conversation": "calibration",
        "split": "train",
        "context": [{"role": "user", "content": "Read."}],
        "hindsight": [{"role": "user", "content": "Be concise."}],
        "tools": [],
    }
    prompts = paired_prompts(
        backend.tokenizer,
        smallest_step,
        replace(policy, response_tokens=1, hindsight_tokens=1),
        check_length=False,
    )
    # The smallest useful context is measured from this tokenizer's real template.
    minimum = len(prompts[0]) + 1
    # Summaries are frozen-base forwards too. Their memory is measured, cached,
    # and included in the admission envelope alongside teacher/train/export.
    with Meter(platform, baseline, backend, policy.guard_min_interval) as summarizing:
        samples, target, pool = measure_pool(policy, store, backend.tokenizer)
    teacher_target = max((s["teacher_sequence"] for s in samples), default=0)
    backend.sample = full_length_sampler(backend)
    # Explicit calibration without experience still measures the requested
    # grid. With experience, its intact token distribution sets the goal.
    target = target or max(
        (n for n in lengths if n <= policy.max_sequence), default=minimum
    )
    target = max(minimum, target)
    kind = candidate_kind(policy, platform)
    saved = saved | {
        "model": policy.model,
        "artifact_kind": kind,
        "rows": rows,
        "capacity_limits": failures,
        "samples": samples,
        "teacher_target_sequence": teacher_target,
        "hardware": hardware.dict(),
        "memory_baseline": asdict(baseline),
    }
    save_progress(policy, store, saved, target, pool)
    if saved["progress"]["complete"]:
        # Existing measurements may already cover a changed pool. Learn its
        # lengths and record impossible revisions without repeating GPU work.
        saved["budget"] = derive(
            rows,
            baseline.total * policy.memory_fraction / 1e9,
            policy.accumulation,
            policy.response_tokens,
            device_free_gb=min(
                (a.memory_total / 1e9 for a in baseline.vram), default=float("inf")
            ),
        )
        save_progress(policy, store, saved, target, pool)
        mark_capacity_samples(policy, store, saved)
        for key in derive_keys():
            setattr(config, key, saved["budget"][key])
        config.save(root)
        return rows
    phases = {"load": load_meter.result(), "summary": summarizing.result()}
    from ..artifacts.storage import admit_storage, artifact_root

    admit_storage(policy, store, platform, calibration=True)
    with tempfile.TemporaryDirectory(
        prefix="calibration-", dir=artifact_root(policy, store)
    ) as directory:
        work = Path(directory)
        write_json(
            work / "songe_calibration.json",
            {"installation_id": store.get("installation_id")},
        )
        artifact = work / ("artifact.gguf" if kind.startswith("gguf-") else "artifact")
        while size := next(
            probe_sizes(
                policy,
                minimum,
                lengths,
                target,
                [r for r in rows],
                failures + held(saved, "sequence"),
            ),
            None,
        ):
            length, response, chunk, batch = size
            saved["in_progress"] = dict(
                zip(
                    ("sequence", "response_tokens", "token_chunk", "accumulation"),
                    size,
                    strict=True,
                )
            ) | {"kind": "sequence", "phase": "setup"}

            def phase(name):
                saved["in_progress"]["phase"] = name
                save_progress(policy, store, saved, target, pool)

            phase("setup")
            admit(
                platform,
                0,
                0,
                pressure_data=store.get("memory_pressure_history", {}),
            )
            backend.reset_peak_memory()
            backend.config = replace(
                policy,
                max_sequence=length,
                response_tokens=response,
                token_chunk=chunk,
                accumulation=batch,
                hindsight_tokens=1,
            )
            tokens = backend.tokenizer.encode(
                "Read this short local calibration passage. ",
                add_special_tokens=False,
            )
            prompt = (tokens * (length // len(tokens) + 1))[: length - response]
            # Construct a synthetic sample of the measured size. Production
            # samples are never shortened to fit this calibration geometry.
            low, high = 0, len(prompt)
            while low < high:
                mid = (low + high + 1) // 2
                trial = smallest_step | {
                    "context": [
                        {
                            "role": "user",
                            "content": backend.tokenizer.decode(prompt[:mid]),
                        }
                    ]
                }
                pair = paired_prompts(
                    backend.tokenizer, trial, backend.config, check_length=False
                )
                if len(pair[0]) + response <= length:
                    low = mid
                else:
                    high = mid - 1
            step = smallest_step | {
                "context": [
                    {
                        "role": "user",
                        "content": backend.tokenizer.decode(prompt[:low]),
                    }
                ]
            }
            student, teacher = paired_prompts(backend.tokenizer, step, backend.config)
            # Long real teachers are measured on their own ladder below, so
            # one long conversation cannot make every probe large.
            full_student, measured_teacher = student, teacher
            backend.reset_peak_memory()
            phase("teacher")
            with Meter(
                platform, baseline, backend, policy.guard_min_interval
            ) as teaching:
                y, _, stats = backend.sample(prompt, seed=7613)
                start = time.perf_counter()
                # Includes the full-context gate forward, conservatively.
                backend.compare(full_student, measured_teacher, y)
                forward = time.perf_counter() - start
                teaching.sample()
            backend.reset_peak_memory()
            phase("train")
            with Meter(
                platform, baseline, backend, policy.guard_min_interval
            ) as training:
                start = time.perf_counter()
                # Two optimizer updates measure both lazy optimizer allocation
                # and steady work, using the actual serial microbatch loop.
                training_work = work / "training"
                backend.train(
                    [step] * (2 * batch),
                    [],
                    training_work,
                    lambda *_: None,
                )
                step_seconds = (time.perf_counter() - start) / 2 + forward * batch
                training.sample()
                # Measurements need no resumable checkpoints after this row.
                if training_work.exists():
                    shutil.rmtree(training_work)
            row_phases = phases | {
                "teacher": teaching.result(),
                "train": training.result(),
            }
            if kind != "mlx-adapter" and "export" not in phases:
                backend.save(work / "adapter")
                phase("export")
                with Meter(
                    platform, baseline, backend, policy.guard_min_interval
                ) as exporting:
                    export_candidate(
                        policy,
                        kind,
                        work / "adapter",
                        artifact,
                        platform,
                        lambda: None,
                    )
                phases["export"] = exporting.result()
                row_phases["export"] = phases["export"]
            if kind in {"mlx-fused", "gguf-full"}:
                phase("evaluate")
                with Meter(
                    platform, baseline, backend, policy.guard_min_interval
                ) as evaluating:
                    model = evaluation_model(
                        policy, kind, artifact, platform, lambda: None
                    )
                    fused = backend_type(
                        replace(backend.config, model=str(model)),
                        lambda: None,
                        train=False,
                    )
                    fused.sample = full_length_sampler(fused)
                    fused.kl_from(backend, student, teacher, y)
                    fused.sample(student, seed=7613)
                    evaluating.sample()
                    del fused
                    gc.collect()
                row_phases["fused_evaluation"] = evaluating.result()
            row = {
                "sequence": length,
                "response_tokens": response,
                "token_chunk": chunk,
                "accumulation": batch,
                "load_seconds": load_seconds,
                "student_prompt_tokens": len(student),
                "teacher_prompt_tokens": len(measured_teacher),
                "teacher_sequence": len(measured_teacher) + response,
                "sample_max_tokens": response,
                "generation_tps": stats.generation_tps,
                "prefill_tps": stats.prompt_tps,
                "step_seconds": step_seconds,
                "forward_seconds": forward,
                "peak_memory_gb": backend.peak_memory() / 1e9,
                "phase_peaks": row_phases,
                **{
                    resource + "_peak_bytes": max(
                        p[resource + "_peak_bytes"] for p in row_phases.values()
                    )
                    for resource in ("host", "device")
                },
            }
            absolute_host = absolute_capacity(policy, baseline.total)
            if row["host_peak_bytes"] > absolute_host or any(
                row["device_peak_bytes"] > a.memory_total for a in baseline.vram
            ):
                # Total capacity is evidence. Today's available RAM, paging,
                # battery and traffic are transient and never create a bound.
                failures.append(row)
            else:
                rows.append(row)
                saved["budget"] = derive(
                    rows,
                    min(
                        absolute_host / 1e9,
                        hardware.total_gb * policy.memory_fraction,
                    ),
                    policy.accumulation,
                    policy.response_tokens,
                    device_free_gb=min(
                        (a.memory_total / 1e9 for a in baseline.vram),
                        default=float("inf"),
                    ),
                )
            saved.pop("in_progress", None)
            save_progress(policy, store, saved, target, pool)
            print(json.dumps(row), flush=True)
    if not rows:
        raise ValueError("max_sequence cannot fit this tokenizer's prompts")
    measure_teachers(
        policy, store, saved, backend, platform, baseline, target, pool, teacher_target
    )
    mark_capacity_samples(policy, store, saved)
    for key in derive_keys():
        if key in saved.get("budget", {}):
            setattr(config, key, saved["budget"][key])
    config.save(root)
    return rows


def held(saved, kind):
    return [h for h in saved.get("held", []) if h.get("kind") == kind]


def next_teacher(saved, teacher_target):
    """Double the measured teacher length toward the longest real one; bisect
    below a total-capacity limit or a size held for lack of free memory."""
    from .calibration_state import close_enough, teacher_measured, teacher_wanted

    wanted = teacher_wanted(saved)
    if wanted:
        return wanted[0]
    measured = teacher_measured(saved)
    if measured >= teacher_target:
        return None
    bound = min(
        (
            r["teacher_sequence"]
            for r in saved.get("teacher_limits", []) + held(saved, "teacher")
        ),
        default=None,
    )
    if bound is None:
        return min(teacher_target, max(2 * measured, measured + 1))
    if close_enough(measured, bound):
        return None
    return min(teacher_target, (measured + bound) // 2)


def measure_teachers(
    policy, store, saved, backend, platform, baseline, target, pool, teacher_target
):
    """Frozen gate forwards over long real conversations, measured on their
    own: samples whose teacher does not fit today's free memory wait."""
    from .measurement import Meter

    tokens = backend.tokenizer.encode(
        "Read this short local calibration passage. ", add_special_tokens=False
    )
    y = (tokens * (policy.response_tokens // len(tokens) + 1))[: policy.response_tokens]
    while length := next_teacher(saved, teacher_target):
        saved["in_progress"] = {
            "teacher_sequence": length,
            "kind": "teacher",
            "phase": "teacher",
        }
        save_progress(policy, store, saved, target, pool)
        prompt = (tokens * (length // len(tokens) + 1))[: max(1, length - len(y))]
        # Measure the teacher on its own: memory the allocator still caches
        # from the training probes made every teacher read as large as the
        # largest of them (#140).
        (
            getattr(backend, "release_cache", None)
            or getattr(backend, "release", lambda: None)
        )()
        backend.reset_peak_memory()
        start = time.perf_counter()
        with Meter(platform, baseline, backend, policy.guard_min_interval) as meter:
            backend.teacher_hidden(prompt, y)
            meter.sample()
        result = meter.result()
        row = {
            "teacher_sequence": length,
            "seconds": time.perf_counter() - start,
            "host_peak_bytes": result["host_peak_bytes"],
            "device_peak_bytes": result["device_peak_bytes"],
        }
        getattr(backend, "release", lambda: None)()
        if row["host_peak_bytes"] > absolute_capacity(policy, baseline.total) or any(
            row["device_peak_bytes"] > a.memory_total for a in baseline.vram
        ):
            saved.setdefault("teacher_limits", []).append(row)
        else:
            saved.setdefault("teacher_rows", []).append(row)
        saved.pop("in_progress", None)
        save_progress(policy, store, saved, target, pool)
        print(json.dumps(row), flush=True)


def admit_phase(config, store, platform, phase):
    """Reserve the measured *additional* resources while the trunk is resident."""
    from ..memory import admit
    from .sizing import cost_model, predict

    rows = (store.get("calibration") or {}).get("rows", [])
    phases = [
        r | r["phase_peaks"][phase] for r in rows if phase in r.get("phase_peaks", {})
    ]
    size = (
        config.max_sequence,
        config.response_tokens,
        config.token_chunk,
        config.accumulation,
    )

    def extra(resource):
        model = cost_model(phases, resource)
        return (
            predict(model, size)
            if model
            else max((r[resource] for r in phases), default=0)
        )

    host = extra("extra_host_bytes")
    device = extra("extra_device_bytes")
    return admit(
        platform, host, device, pressure_data=store.get("memory_pressure_history", {})
    )


def device_full(exc):
    """The accelerator had no room for the model (torch.OutOfMemoryError, or
    the CUDA/HIP allocator's message), as when the user's model server holds
    most of the card. Not the host memory guard's stop."""
    return type(exc).__name__ == "OutOfMemoryError" or " out of memory" in str(exc).lower()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=data_dir())
    p.add_argument("--child", action="store_true")
    p.add_argument("--model-key")
    p.add_argument("--lengths", default="256,512,1024,2048")
    args = p.parse_args()
    from ..telemetry import configure

    configure(args.data_dir)
    store = Store(args.data_dir)
    scoped = store.scoped(args.model_key) if args.model_key else store
    scoped.set("child_error", None)
    try:
        (_calibrate if args.child else calibrate)(
            Config.for_model(args.data_dir, args.model_key)
            if args.model_key
            else Config.load(args.data_dir),
            args.data_dir,
            tuple(int(v) for v in args.lengths.split(",")),
        )
    except Exception as exc:
        from ..artifacts.storage import StorageWait
        from ..telemetry import failure_hint
        from ..wording import DEVICE_FULL

        if isinstance(exc, StorageWait):
            scoped.set("disk_wait", str(exc))
            return
        # The size being measured exceeds the device's largest buffer: that
        # is its capacity, so the smaller measured sizes stand.
        if args.child and record_buffer_ceiling(scoped, str(exc).encode()):
            return
        scoped.set(
            "child_error",
            DEVICE_FULL if device_full(exc) else failure_hint("Calibration", args.data_dir),
        )
        # A capability error (the architecture cannot train) is final. Anything
        # environmental (files mid-upgrade, servers, IO) is retried by the
        # scheduler before the model is marked unsupported.
        if args.model_key and isinstance(exc, (ValueError, NotImplementedError)):
            from ..models.registry import support

            support(
                Store(args.data_dir),
                args.model_key,
                "unsupported",
                (str(exc).splitlines() or [type(exc).__name__])[0][:300],
            )
        raise


if __name__ == "__main__":
    from songe.telemetry import diagnostic_main

    diagnostic_main(main)
