"""Resumable learning units; reads observations and never executes user tools."""

import argparse
import gc
import json
import logging
import math
import re
import signal
import time
from dataclasses import replace
from pathlib import Path
from statistics import mean

from ..artifacts.storage import StorageWait, admit_storage, artifact_roots
from ..backends import Interrupted, choose
from ..config import (
    Config,
    data_dir,
    training_fingerprint,
    training_setting_sources,
    training_settings_snapshot,
)
from ..log import Store
from ..platform import Rooted, current, power_status
from ..platform.locking import acquire
from ..schedule import (
    base_fingerprint,
    pending_export,
    probe,
    rebuild_key,
    rebuild_pending,
    rebuild_reason,
    slice_duration,
    work_costs,
)
from ..schedule import resumable as can_resume
from ..steps import digest, revision_id
from ..storage import private_dir, write_json

# Tokens per no-gradient pass in the gate (see TorchBackend.read_chunk).
GATE_READ_CHUNK = 4096


def files_fingerprint(path):
    """What a saved adapter is, for caches: its files' names, sizes and
    modification times (a new adapter is written as new files)."""
    path = Path(path)
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    return digest([str(path)] + [
        [p.name, p.stat().st_size, p.stat().st_mtime_ns] for p in files if p.exists()
    ])


def general_steps():
    prompts = json.loads(
        (Path(__file__).parent / "anchors.json").read_text(encoding="utf-8")
    )
    return [
        {
            "id": digest(p),
            "conversation": digest(p),
            "split": "anchor",
            "context": [{"role": "user", "content": p}],
            "hindsight": [],
            "tools": [],
            "action": {},
        }
        for p in prompts
    ]


def preview(config, store):
    from ..experience import refresh

    refresh(config, store)
    store.sync_steps(store.active_steps())
    examples, conversations = store.feedback_counts()
    if examples >= config.min_train and conversations >= config.min_heldout + 1:
        store.assign_splits(minimum=config.min_heldout)  # settled splits never move
    heldout = [s for s in store.active_steps("heldout") if s["hindsight"]]
    if config.learning == "dreams":
        # Real conversations never enter training: the pool is kept practice
        # scenes. Held-out real conversations gate first; held-out practice
        # lessons (never trained on) fill the places still missing.
        from ..practice.pool import (
            MOMENTS_PER_CONVERSATION,
            ensure_heldout,
            ensure_note_heldout,
            gate_moments,
            heldout_fill,
            practice_steps,
            real_test_steps,
        )

        # Every moment the user reacted to is a test (never trained on under
        # Dreams); held-out practice lessons fill up to the usual size of the
        # test set, more lessons held out on demand when short.
        # Each conversation's sampled moments (as many as heldout_units
        # counts): the check stays bounded however long the history grows.
        heldout = gate_moments(real_test_steps(store))
        ensure_heldout(store, config.min_heldout)
        if config.learning == "dreams":
            ensure_note_heldout(store)
        train = practice_steps(store)
        per = {}
        for s in heldout:
            per[s["conversation"]] = per.get(s["conversation"], 0) + 1
        real = sum(min(n, MOMENTS_PER_CONVERSATION) for n in per.values())
        heldout = heldout + heldout_fill(store, config.min_heldout - real)
    else:
        train = [s for s in store.active_steps("train") if s["hindsight"]]
    info = {
        "train": len(train),
        "heldout": len(heldout),
        "heldout_conversations": len({s["conversation"] for s in heldout}),
        "pool": len(store.pool()),
        "new": sum(
            revision_id(s) > store.get("experience_watermark", 0) for s in train
        ),
        "calibrated": config.calibration_available,
        "reason": "Waiting for experience"
        if len(train) + len(heldout) < (1 if config.learning == "dreams" else config.min_train)
        or not train
        or (len(heldout) < 1 if config.learning == "dreams" else len({s["conversation"] for s in heldout}) < config.min_heldout)
        else "Ready to score hindsight; final selection requires the model",
    }
    return info, train, heldout


class Journal:
    def __init__(self, root, check, store, slice_id):
        self.root, self.check, self.store, self.slice_id = root, check, store, slice_id
        self.phase, self.units = "record", 0

    def path(self, name):
        # A unit name may carry an id like "original:sig" or "practice:x#v1";
        # Windows forbids ":" (and a few others) in file names.
        return self.root / (re.sub(r'[<>:"/\\|?*]', "_", name) + ".json")

    def load(self, name, default):
        path = self.path(name)
        return (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
        )

    def commit(self, name, value):
        write_json(self.path(name), value)
        self.units += 1
        # Keep ended=NULL until the whole slice exits, including on a crash.
        with self.store.connect() as db:
            db.execute(
                "UPDATE slices SET phase=?,units_done=? WHERE id=?",
                (self.phase, self.units, self.slice_id),
            )

    def unit(self, name, fn):
        path = self.path(name)  # where commit writes it
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        self.check()
        result = fn()
        self.commit(name, result)
        return result

    def indivisible(self, name, fn):
        # The planning deadline is checked before starting this unit. Once
        # started, only activity/pause/memory cancellation can interrupt it.
        return self.unit(name, lambda: fn(lambda: self.check(False)))


def verify_work(work, store, night):
    marker = work / "songe_work.json"
    if (
        work.is_symlink()
        or not any(
            work.parent.resolve() == (root / "versions").resolve()
            for root in artifact_roots(store.root)
        )
        or work.exists()
        and (
            not marker.is_file()
            or json.loads(marker.read_text(encoding="utf-8"))
            != {"night": night, "installation_id": store.get("installation_id")}
        )
    ):
        raise StorageWait(f"Learning work is unavailable or no longer owned: {work}")


def run(
    config,
    store,
    slice_seconds=None,
    sizing=None,
    *,
    export_only=False,
    setting_sources=None,
):
    from ..memory import TrainingPlatform
    from ..practice.pool import drift_steps
    from .evaluate import evaluate, references, sharpness
    from .fuse import candidate_kind, evaluation_model, export_candidate
    from .gate import Decision, choose_scale, decide, judged_check
    from .prompts import (
        OversizedStep,
        UnusableStep,
        paired_prompts,
        serve_as,
        served_switches,
    )
    from .samples import SampleAdmission
    from .select import bounded_batch, replay_pool, select

    # Every prompt of this cycle (training, the check, the examples) is
    # rendered the way the user's server sends it.
    serve_as(served_switches(store))
    policy_config = config
    lease = (store.root / "worker.lock").open("a+", encoding="utf-8")
    acquire(lease)
    interrupted, interrupted_by = False, None
    deadline = (
        time.monotonic() + slice_seconds if slice_seconds is not None else float("inf")
    )
    slice_id = store.slice()
    admission = SampleAdmission(
        config, sizing, store.get("calibration"), store, slice_id
    )
    night, record, journal = None, {}, None
    reason_key = "reason"
    power_platform = current()
    next_power_check = 0
    generation = store.get("candidate_generation")

    def stop(*_):
        nonlocal interrupted, interrupted_by
        # The service records the real cause (agent activity, memory guard,
        # paused, ...) when it stops a slice; this is only the fallback.
        interrupted, interrupted_by = True, "stopped by the service"

    for sig in (
        signal.SIGTERM,
        getattr(signal, "SIGUSR1", signal.SIGINT),
        getattr(signal, "SIGBREAK", signal.SIGINT),
    ):
        signal.signal(sig, stop)

    next_state_check = 0

    def check(include_deadline=True):
        """Stop now for a signal (traffic, a guard), battery, a pause or a
        policy change. The slice deadline only when include_deadline: at a
        unit boundary (a training step, a gate row), so a deadline never cuts
        a unit midway and throws its work away (see boundary below)."""
        nonlocal interrupted_by, next_power_check, next_state_check
        if interrupted:
            interrupted_by = interrupted_by or "stopped by the service"
            raise Interrupted(interrupted_by)
        now = time.monotonic()
        # Sample at guard cadence, rather than spawning OS probes per token.
        if now >= next_power_check:
            next_power_check = now + config.guard_max_interval
            if power_status(power_platform).constrained():
                interrupted_by = "battery low"
                raise Interrupted(interrupted_by)
        # Stored state too: per-token checks must not read the database.
        if include_deadline or now >= next_state_check:
            next_state_check = now + config.guard_max_interval
            from ..models.registry import find

            model = find(store, config.model_key) if config.model_key else None
            if store.get("candidate_generation") != generation:
                raise Interrupted("learning policy changed")
            if store.get("paused", False) or model and not model["enabled"]:
                interrupted_by = interrupted_by or "paused"
                raise Interrupted(interrupted_by)
        if include_deadline and now >= deadline:
            interrupted_by = "slice deadline"
            raise Interrupted(interrupted_by)

    def inner_check():
        check(include_deadline=False)

    def export_scale(scale, kind, adapter, destination):
        from .calibrate import admit_phase

        def produce(urgent):
            journal.phase = "fuse"
            admit_phase(config, store, platform, "export")
            export_candidate(config, kind, adapter, destination, platform, urgent)
            return {"path": str(destination)}

        journal.indivisible(f"fused-{scale}", produce)

    try:
        expected_seconds = (
            slice_seconds
            if slice_seconds is not None
            else slice_duration(config, store.get("calibration"))
        )
        if power_status(power_platform).constrained(expected_seconds):
            raise Interrupted("battery low")
        if pending := pending_export(config, store):
            night, record = pending["id"], pending["record"]
            reason_key = "export_wait"
            work = Path(record["work"])
            candidate = record["pending_export"]
            adapter = Path(candidate["training_adapter"])
            if not adapter.is_dir():
                raise StorageWait(f"Saved adapter is unavailable: {adapter}")
            verify_work(work, store, night)
            admit_storage(config, store, power_platform, pending_export=candidate)
            journal = Journal(work, check, store, slice_id)
            journal.phase = "fuse"
            platform = Rooted(TrainingPlatform(power_platform, config), store.root)
            export_scale(
                record["wake_scale"],
                candidate["kind"],
                adapter,
                Path(candidate["artifact"]),
            )
            check(False)
            result = record | {"candidate": candidate, "ended_at": time.time()}
            del result["pending_export"]
            result.pop("export_wait", None)
            store.night(night, status="ready", phase="swap", progress=0.95, record=result)
            print(json.dumps(result), flush=True)
            return result
        if export_only:
            return {"reason": "No eligible adapter export to resume"}
        info, train_steps, heldout = preview(config, store)
        # Installations from before learning modes trained on real steps.
        if store.get("learning_mode", "hindsight") != config.learning:
            # Revisions count exchanges under hindsight and practice rows under
            # dreams: what was trained under one says nothing about the other.
            store.set("experience_watermark", 0)
            store.set("practice_revisions", {})
            store.set("rejected_practice", {})
            store.set("learning_mode", config.learning)
        if info["reason"] == "Waiting for experience":
            return info
        if not config.calibration_available:
            raise ValueError("Calibration is required before training")
        from .calibration_state import needs_growth

        if needs_growth(config, store):
            return info | {"reason": "Waiting for calibration growth"}
        if sizing:
            from .calibration_state import smallest_pending

            if (
                sizing.get("response_tokens", config.response_tokens)
                < config.response_tokens
            ):
                return info | {"reason": "Waiting for calibration growth"}
            shortest = smallest_pending(config, store.get("calibration") or {}, store)
            if (
                shortest is not None
                and sizing.get("max_teacher_sequence", float("inf")) < shortest
            ):
                return info | {
                    "reason": "Waiting for memory: intact training samples deferred"
                }
        fingerprint = training_fingerprint(config)
        base = base_fingerprint(config)
        rebuild_attempt_key = rebuild_key(config, store)
        if sizing:
            config = replace(
                config,
                **{
                    k: sizing[k]
                    for k in (
                        "max_sequence",
                        "token_chunk",
                    )
                },
            )
        resumable = next(
            (
                n
                for n in reversed(store.nights())
                if can_resume(n, store, config, configuration=fingerprint, base=base)
            ),
            None,
        )
        if any(n["status"] == "ready" for n in store.nights()):
            return info | {"reason": "Waiting to switch version"}
        if not resumable and not rebuild_pending(policy_config, store):
            watermark = store.get("experience_watermark", 0)
            if config.learning == "dreams":
                from ..practice.pool import untrained

                fresh = any(
                    not admission.known_unfit(s) for s in untrained(store, train_steps)
                )
            else:
                fresh = store.unscored(train_steps) or any(
                    revision_id(s) > watermark and not admission.known_unfit(s)
                    for s in store.pool()
                )
            if not fresh:
                return info | {"reason": "Waiting for new learnable experience"}
        # Completed cycles and pending swaps allocate no training resources.
        p = probe()
        if (
            not p.available
            or (p.thermal is not None and p.thermal >= 2)
            or p.pressure == "critical"
        ):
            raise ValueError(f"Hardware policy does not permit training: {p.dict()}")
        platform = Rooted(TrainingPlatform(current(), config), store.root)
        if {a.kind for a in platform.accelerators()} <= {"cpu"} and not (
            config.force_cpu_training or config.torch_options.get("force_cpu", False)
        ):
            raise ValueError(
                "CPU is serving only; force_cpu_training enables measured trials"
            )
        kind = candidate_kind(config, platform)
        from ..artifacts.storage import (
            artifact_root,
            volume,
        )

        admit_storage(config, store, platform)
        target = artifact_root(config, store)
        if resumable:
            previous_work = Path(
                resumable["record"].get("work")
                or store.root / "versions" / f"night-{resumable['id']}"
            )
            if volume(previous_work) != volume(target):
                # A pending legacy unit may still point at the full data disk.
                # Retire it; its selected experience remains in the durable pool.
                store.night(resumable["id"], status="discarded")
                resumable = None
        if resumable:
            night, record = resumable["id"], resumable["record"]
        else:
            record = {
                "expected": store.get("current"),
                "configuration": fingerprint,
                "training_settings": training_settings_snapshot(
                    policy_config, setting_sources
                ),
                "base_fingerprint": base,
                "environment": store.get("environment_fingerprint"),
                "generation": store.get("candidate_generation"),
                "rebuild": rebuild_reason(config, store),
                "rebuild_key": rebuild_attempt_key,
                "steps_seen": len(train_steps),
                "watermark": max((revision_id(s) for s in train_steps), default=0),
                # For the cycle record (docs/CONTRACTS.md, Learning cycle).
                "learning": config.learning,
                "trigger": "dream" if store.get("requested_sleep", False) else "idle",
                "started_at": time.time(),
            }
            night = store.night(None, record=record)
        work = (
            Path(record["work"])
            if record.get("work")
            else target / "versions" / f"night-{night}"
        )
        owned = {"night": night, "installation_id": store.get("installation_id")}
        verify_work(work, store, night)
        private_dir(work)
        write_json(work / "songe_work.json", owned)
        record["work"] = str(work)
        store.night(night, status="running", record=record)
        if sizing:
            record["slice_sizes"] = record.get("slice_sizes", []) + [sizing]
            store.night(night, record=record)
        journal = Journal(work, check, store, slice_id)
        step_seconds, score_seconds = work_costs(config, store.get("calibration"))

        shown = {}

        def data_progress(phase, completed, total, seconds_per_unit, live=None):
            if phase == "train" and shown.get(phase) != completed:
                shown[phase] = completed
                store.event("train", f"{completed} of {total} samples")
            checkpoint, accumulation = {}, None
            if phase == "train":
                path = work / "checkpoint.json"
                checkpoint = (
                    json.loads(path.read_text(encoding="utf-8"))
                    if path.exists()
                    else {}
                )
                path = work / "training-plan.json"
                plan = (
                    json.loads(path.read_text(encoding="utf-8"))
                    if path.exists()
                    else {}
                )
                accumulation = plan.get("accumulation", backend.config.accumulation)
                completed = max(completed, checkpoint.get("sample_cursor", 0))
            # Measured in an earlier slice of this night: a restarted worker
            # uses it until it has measured a step itself.
            earlier = (record.get("data_progress") or {}).get("seconds_per_step")
            record["data_progress"] = {
                "phase": phase,
                "completed": completed,
                "total": total,
                "unit": "experiences" if phase == "score" else "samples",
                "eta_seconds": (total - completed) * seconds_per_unit
                if seconds_per_unit > 0
                else None,
            }
            if accumulation:
                record["data_progress"].update(
                    step=checkpoint.get(
                        "iteration", math.ceil(completed / accumulation)
                    ),
                    total_steps=math.ceil(total / accumulation),
                )
            if phase == "train" and accumulation and earlier and not (
                live and live.get("seconds_per_step")
            ):
                progress = record["data_progress"]
                progress["seconds_per_step"] = earlier
                progress["eta_seconds"] = max(
                    0, progress["total_steps"] - progress["step"]
                ) * earlier
            if live and accumulation:
                # The training loop's own step, losses and measured step time
                # (the checkpoint lags them by minutes).
                steps = record["data_progress"]["total_steps"]
                record["data_progress"]["step"] = live["iteration"]
                record["losses"] = live["losses"]
                if live.get("seconds_per_step"):
                    record["data_progress"]["eta_seconds"] = max(
                        0, steps - live["iteration"]
                    ) * live["seconds_per_step"]
                    # The scheduler sizes the next slice by it (see
                    # schedule.measured_step_seconds).
                    record["data_progress"]["seconds_per_step"] = live[
                        "seconds_per_step"
                    ]
            store.night(
                night,
                phase="select" if phase == "score" else "distill",
                progress=(0 if phase == "score" else 0.2)
                + (0.2 if phase == "score" else 0.5) * completed / max(1, total),
                record=record,
            )

        backend_type = choose(config, platform)
        # Inside a unit (per token, per micro-batch) the deadline is not
        # checked; at unit boundaries backend.boundary() checks it.
        backend = backend_type(config, inner_check)
        backend.boundary = check
        practice_token_limit = (sizing or {}).get(
            "response_tokens", config.response_tokens
        )
        incumbent_version = store.version()
        previous_adapter = (
            (
                incumbent_version.get("training_adapter")
                or incumbent_version.get("adapter")
            )
            if incumbent_version and not record["rebuild"]
            else None
        )
        if previous_adapter:
            backend.restore(Path(previous_adapter))
        baseline = backend.adapter_weights()
        # This cycle's teacher is the version it starts from (DESIGN 3.3).
        if previous_adapter and hasattr(backend, "snapshot_teacher"):
            backend.snapshot_teacher()
        journal.phase = "rewind"
        from .lessons import extract_lessons, prune_lessons

        # Under dreams, lessons come from read signals (reading/), not here.
        lessons = (
            []
            if config.learning == "dreams"
            else extract_lessons(
                store, backend, journal, train_steps, token_limit=practice_token_limit
            )
        )
        record["lessons"] = len(lessons)
        store.night(night, phase="rewind", record=record)
        if lessons:
            # Once per round, however many times it resumes.
            journal.unit(
                "lessons-shown",
                lambda: store.event("rewind", f"{len(lessons)} lessons") or True,
            )
        snapshot = journal.load("snapshot", None)
        if snapshot:
            pool, heldout = snapshot["pool"], snapshot["heldout"]
        else:
            def practiced_new():
                from ..practice.pool import untrained

                return train_steps if record["rebuild"] else untrained(store, train_steps)

            experience = journal.unit(
                "experience",
                lambda: {
                    "new": sorted(
                        practiced_new()
                        if config.learning == "dreams"
                        else train_steps
                        if record["rebuild"]
                        else store.unscored(train_steps),
                        key=revision_id,
                    ),
                    "heldout": heldout,
                    "pool": train_steps
                    if config.learning == "dreams"
                    else store.pool(),
                },
            )
            new, heldout = experience["new"], experience["heldout"]
            new = admission.admit(new, backend.tokenizer)
            if config.learning == "dreams":
                # Practice was checked when it was written; the pool is every
                # kept scene, the new ones first and older ones as replay.
                selected, skipped = new, 0
                pool = admission.admit(experience["pool"], backend.tokenizer)
            else:
                selected, skipped = select(new)
                if new:
                    store.event(
                        "clean",
                        f"kept {len(selected)} of {len(new)} new steps for training",
                    )
                store.add_pool(selected, scored=new)
                if new:
                    store.mark_scored(new)
                    store.set(
                        "score_watermark",
                        max(
                            store.get("score_watermark", 0),
                            max(revision_id(s) for s in new),
                        ),
                    )
                # An unfit revision also supersedes its older selected evidence.
                rescored = {s["id"] for s in experience["new"]}
                pool = [
                    s for s in experience["pool"] if s["id"] not in rescored
                ] + selected
                pool = admission.admit(pool, backend.tokenizer)
            if not record["rebuild"]:
                pool = replay_pool(
                    pool,
                    store.get("experience_watermark", 0),
                    config.replay_fraction,
                    store.get("practice_revisions", {}),
                )
            if config.learning == "dreams":
                # A capped Dream takes the scenes no Dream has tried first.
                from ..practice.pool import marked

                pool = marked(store, pool)
            from .calibration_state import pending_keys

            pending_practice = pending_keys(store)
            pool, remaining = bounded_batch(
                pool,
                config,
                max(0, deadline - time.monotonic())
                if slice_seconds is not None
                else None,
                step_seconds,
                pending=pending_practice,
                whole=config.learning == "dreams",
            )
            remaining = [
                s for s in remaining if (s["id"], revision_id(s)) in pending_practice
            ]
            if remaining:
                # Keep all scored/selected evidence in Store.pool for another
                # cycle, without advancing past an untrained revision.
                record["batch_watermark"] = min(revision_id(s) for s in remaining) - 1
            record["scored_watermark"] = store.get("score_watermark", 0)
            record["selected_revisions"] = {s["id"]: revision_id(s) for s in selected}
            record["watermark"] = max(
                (revision_id(s) for s in pool),
                default=store.get("experience_watermark", 0),
            )
            deferred = [
                s["revision"]
                for s in admission.record["deferred_samples"]
                if s["revision"] > store.get("experience_watermark", 0)
            ]
            if deferred:
                # A p90 slice can train its intact majority while the tail
                # waits for growth. Do not consume selected tail revisions.
                record["watermark"] = min(record["watermark"], min(deferred) - 1)
                record["batch_watermark"] = min(
                    record.get("batch_watermark", min(deferred) - 1),
                    min(deferred) - 1,
                )
            record["skipped_steps"] = skipped
            if not pool:
                record["reason"] = "Waiting for learnable experience"
                store.night(night, status="waiting", record=record)
                return record
            usable = []
            for step in admission.admit(heldout, backend.tokenizer, full_context=True):
                check()
                try:
                    paired_prompts(
                        backend.tokenizer,
                        step,
                        config,
                        check_length=False,
                    )
                    usable.append(step)
                except UnusableStep:
                    pass
            heldout = usable
            # Under Dreams the minimum counts test moments (user decision).
            units = len(heldout) if config.learning == "dreams" else len({s["conversation"] for s in heldout})
            if units < (1 if config.learning == "dreams" else config.min_heldout):
                record["reason"] = "Waiting for usable held-out conversations"
                store.night(night, status="waiting", record=record)
                return record
            journal.commit("snapshot", {"pool": pool, "heldout": heldout})
        prune_lessons(store)
        # The night's gate set is fixed: the snapshot's held-out steps and the
        # anchors first admitted. Memory deferral never shrinks it (see gate).
        night_gate = heldout + journal.unit(
            "gate-anchors", lambda: admission.admit(general_steps(), backend.tokenizer)
        )
        anchors = admission.admit(night_gate[len(heldout) :], backend.tokenizer)
        drift_set = admission.admit(
            journal.unit("gate-drift", lambda: drift_steps(store)), backend.tokenizer
        )
        heldout = admission.admit(heldout, backend.tokenizer, full_context=True)
        training_anchors = journal.unit(
            "anchors", lambda: anchors + [s | {"hindsight": []} for s in pool]
        )
        pool = admission.admit(pool, backend.tokenizer)
        if not pool:
            record.update(admission.record, reason="Waiting for learnable experience")
            store.night(night, status="waiting", record=record)
            return record
        training_anchors = admission.admit(training_anchors, backend.tokenizer)
        # Increase rehearsal using observed drift, preserving original-base teacher.
        drift_ratio = (
            min(1, store.get("cumulative_drift", 0) / config.general_drift_limit)
            if config.general_drift_limit
            else 0
        )
        backend.config = replace(
            config,
            # At most half rehearsal: a fraction near 1 would mean ~20 anchors
            # per new example and almost nothing learned from the user.
            anchor_fraction=min(
                0.5,
                config.anchor_fraction + (1 - config.anchor_fraction) * drift_ratio,
            ),
        )
        policy = journal.unit(
            "training-policy",
            lambda: {
                "anchor_fraction": backend.config.anchor_fraction,
                "accumulation": policy_config.accumulation,
            },
        )
        backend.config = replace(backend.config, **policy)
        journal.phase = "train"
        from ..backends.training import sample_plan

        samples = len(
            sample_plan(pool, training_anchors, backend.config.anchor_fraction)
        )
        previous_progress = journal.load("train-progress", {"completed": 0})
        sample_seconds = step_seconds / config.accumulation
        data_progress("train", previous_progress["completed"], samples, sample_seconds)

        def progress(completed, total, live=None):
            journal.commit("train-progress", {"completed": completed, "total": total})
            data_progress("train", completed, total, sample_seconds, live)

        def consolidate():
            started = time.monotonic()
            record["consolidate_started"] = True
            store.night(night, record=record)
            try:
                return backend.train(pool, training_anchors, work, progress)
            finally:
                record["consolidate_seconds"] = record.get(
                    "consolidate_seconds", 0
                ) + max(0, time.monotonic() - started)
                store.night(night, record=record)

        result = journal.unit(
            "trained",
            consolidate,
        )
        # Checkpoints only resume training; once "trained" is recorded the
        # final adapter is all that is read again.
        import shutil

        for old in work.glob("checkpoint-*"):
            if old.is_dir():
                shutil.rmtree(old, ignore_errors=True)
        backend.restore(work / "adapter")
        trained = backend.adapter_weights()
        record.update(result, pool_size=len(pool))
        # One loss per committed optimizer step, and how the samples split
        # between practice and rehearsal, for the cycle record.
        progress_file = work / "checkpoint.json"
        if progress_file.exists():
            record["losses"] = json.loads(progress_file.read_text(encoding="utf-8")).get(
                "losses", []
            )
        record["samples"] = {"practice": len(pool), "rehearsal": max(0, samples - len(pool))}
        record["practice_revisions"] = store.get("practice_revisions", {}) | {
            s["id"]: revision_id(s) for s in pool
        }
        # The signals whose practice this Dream trained: once it is adopted,
        # their Quick Dream lessons leave the prompt (quick.absorb).
        # A cluster's scenes stand for every signal that teaches the same
        # thing (practice.store.clusters), so all of them count as trained.
        from ..practice.store import cluster_of

        practised = {s["signal_id"] for s in pool if s.get("practice") and s.get("signal_id")}
        same = cluster_of(store)
        record["trained_signals"] = sorted(
            {member for signal in practised for member in same.get(signal, [signal])}
        )
        data_progress("train", samples, samples, sample_seconds)
        if "trained_revisions" in result:
            record["watermark"] = max(
                result["trained_revisions"].values(),
                default=store.get("experience_watermark", 0),
            )
        if "batch_watermark" in record:
            record["watermark"] = min(
                max(
                    record["watermark"],
                    max(record["practice_revisions"].values(), default=0),
                ),
                record["batch_watermark"],
            )
        else:
            record["watermark"] = max(
                record["watermark"],
                max(record["practice_revisions"].values(), default=0),
            )
        journal.phase = "gate"
        if config.learning == "dreams":
            # Training is done: the checks run to the end instead of in
            # planned slices (live: 11 five-minute slices, each reloading,
            # 50 min instead of ~26). Agent activity, pause and the memory
            # guard still stop it at once (signals and state checks).
            deadline = float("inf")
        store.night(night, phase="check", progress=0.7, record=record)
        # A slice short on memory waits here, with training kept, instead of
        # scoring a subset: results stay comparable and a finished gate stays
        # finished across the night's slices.
        # A sample longer than this machine ever measured is always deferred,
        # so it is left out the same way every slice and never waited on.
        measured = store.get("calibration") or {}
        capacity = max(
            [0]
            + [r.get("teacher_sequence") or 0 for r in measured.get("rows", [])]
            + [t.get("teacher_sequence") or 0 for t in measured.get("teacher_rows", [])]
        )
        waiting = {
            s["id"]
            for s in admission.record["deferred_samples"]
            if (s.get("teacher_sequence") or 0) <= capacity
        }
        if any(s["id"] in waiting for s in night_gate):
            raise Interrupted("Waiting for memory: intact training samples deferred")
        # The gate reads without gradients, after training: bigger chunks
        # (fewer passes) are much faster where each pass has a fixed cost
        # (Inkling-2-Mini unpacks its experts per pass: 4.5x at 32k), and
        # halve themselves if memory runs short (TorchBackend.reading).
        if backend.kind == "torch":
            backend.read_chunk = max(GATE_READ_CHUNK, config.teacher_chunk)

        # Gate work is cached by what it scores (the examples and response
        # length), not by this slice's memory sizing, which changes almost
        # every slice and would restart the gate forever.
        def gate_key(steps):
            return digest(
                {
                    "response_tokens": backend.config.response_tokens,
                    "ids": [s.get("id") for s in steps],
                }
            )

        # Progress through the gate, in conversations measured: references
        # for the whole set, then the incumbent, then each strength tried.
        # A measured pass skips the bundled prompts when the drift set can
        # decide (see evaluate), so each pass is sized the way it will run.
        from .evaluate import drift_choice

        measured_counts, pass_sizes = {}, {}

        def pass_size(cache=None):
            values = (cache or {}).get("drift_set") or []
            if len(values) < len(drift_set):
                # Not measured yet: whether it decides depends on its size.
                values = [0.0] * len(drift_set)
            groups = [{"conversation": s.get("conversation")} for s in drift_set]
            _, decided = drift_choice(None, values, groups, backend.config)
            bundled = (cache or {}).get("drifts") or decided["source"] != "drift"
            return len(heldout) + len(drift_set) + (len(anchors) if bundled else 0)

        def gate_progress(name, count, cache=None):
            if measured_counts.get(name) == count and name in pass_sizes:
                return
            measured_counts[name] = count
            if not name.startswith("refs:"):
                pass_sizes[name] = pass_size(cache)
            # References, the incumbent and at least one strength. The
            # bundled prompts' references count once they are requested.
            bundled = any(k.startswith("refs:general-references") for k in measured_counts)
            total = (
                len(heldout) + len(drift_set) + len(anchors) if bundled else pass_size()
            )
            total += sum(pass_sizes.values())
            total += pass_size() * max(0, 2 - len(pass_sizes))
            completed = min(sum(measured_counts.values()), total)
            record["data_progress"] = {
                "phase": "check",
                "completed": completed,
                "total": total,
                "unit": "measured",
                "eta_seconds": None,
            }
            store.night(
                night, phase="check", progress=0.7 + 0.25 * completed / max(1, total),
                record=record,
            )

        # The original model's answers are kept across Dreams for the same
        # base weights (see ReferenceCache): later Dreams skip resampling.
        from .evaluate import ReferenceCache

        reference_cache = ReferenceCache(store.root / "gate-references", base)

        def reference_set(name, steps):
            name += "-" + gate_key(steps)
            rows = journal.load(name, [])
            gate_progress("refs:" + name, len(rows))
            for step in steps[len(rows) :]:
                check()
                rows.extend(references(backend, [step], cache=reference_cache))
                journal.commit(name, rows)
                gate_progress("refs:" + name, len(rows))
            return rows

        refs = reference_set("references", heldout)
        drift_refs = reference_set("drift-references", drift_set)
        # The bundled prompts' references only when the drift set cannot
        # decide (evaluate asks for them then), once per cycle.
        general_memo = []

        def general_refs():
            if not general_memo:
                general_memo.append(reference_set("general-references", anchors))
            return general_memo[0]

        # The reference's read of each moment, shared by the versions below
        # (it does not depend on the version measured); the base's reads are
        # kept across Dreams (KeptViews).
        from .evaluate import KeptViews

        views = (
            KeptViews(
                store.root / "gate-views",
                [base, "r2", config.teacher_chunk, GATE_READ_CHUNK, config.torch_options or {}],
            )
            if backend.kind == "torch"
            else {}
        )

        def pass_name(name):
            # r2: how contexts are read (one layout, 4096-token chunks); rows
            # read another way are not comparable, so they are not resumed.
            return name + "-r2-" + gate_key(heldout + anchors + drift_set)

        def measured(name, candidate, identical=False, kept_as=None, rows=True):
            name = pass_name(name)
            cache = journal.load(name, {})

            def commit():
                journal.commit(name, cache)
                gate_progress(
                    name,
                    sum(len(cache.get(k) or []) for k in ("drifts", "drift_set", "rows")),
                    cache,
                )

            commit()
            result = evaluate(
                backend,
                candidate,
                refs,
                general_refs,
                drift_refs=drift_refs,
                checkpoint=cache,
                commit=commit,
                identical=identical,
                views=views,
                kept_as=kept_as,
                rows=rows,
            )
            measured_rows = result if rows else []
            # Which "not broken" measure gated, and why (see evaluate), and
            # the measured values, for the cycle record's checks.
            record.setdefault("drift_checks", {})[name] = cache.get("drift_check")
            record.setdefault("gate_values", {})[name] = {
                "held_out": mean(r["kl"] for r in measured_rows) if measured_rows else None,
                "general": mean(cache["drifts"]) if cache.get("drifts") else None,
                "drift": mean(cache["drift_set"]) if cache.get("drift_set") else None,
                "count": len(measured_rows),
                "drift_count": len(cache.get("drift_set") or []),
                **sharpness(cache),
            }
            return result  # rows=False: the "not broken" measure's value

        # The incumbent's measurements are kept for the next Dreams while
        # what they depend on stays the same (KeptMeasurements).
        from .evaluate import KeptMeasurements

        incumbent_adapter = incumbent_version and (
            incumbent_version.get("training_adapter") or incumbent_version.get("adapter")
        )
        kept_incumbent = KeptMeasurements(
            store.root / "gate-incumbent",
            [
                base,
                files_fingerprint(incumbent_adapter) if incumbent_adapter else "original",
                files_fingerprint(previous_adapter) if previous_adapter else "base",  # the teacher
                "r2",
                backend.kind,
                config.response_tokens,
                config.repetition_ngram,
                config.teacher_chunk,
                GATE_READ_CHUNK,
                sorted((config.torch_options or {}).items()),
            ],
        )
        # The incumbent and candidate share the original frozen base. Reuse
        # its trunk so evaluating the incumbent does not load a second base.
        def load_incumbent():
            if incumbent_adapter:
                backend.restore(Path(incumbent_adapter))
            else:
                backend.scale_delta(baseline, trained, 0)

        def measure_incumbent(rows=True):
            # The incumbent is the original when it has no adapter: its drift
            # from it is zero.
            return measured(
                "before", backend, identical=not incumbent_adapter, kept_as=kept_incumbent, rows=rows
            )

        # Its "not broken" measure now; its held-out rows only when a
        # candidate gets that far (before_rows), so a candidate rejected on
        # drift or on the judged moments costs no rows at all.
        load_incumbent()
        measure_incumbent(rows=False)
        incumbent_rows = []

        def before_rows():
            if not incumbent_rows:
                load_incumbent()
                incumbent_rows.append(measure_incumbent())
            return incumbent_rows[0]

        # Under Dreams the gate judges what each version writes (judged.py):
        # the version in use answers each moment now, while it is loaded.
        from . import judged

        moments = judged.moments(heldout) if config.learning == "dreams" else []
        before_answers = {}
        # The user is served the version in use with the Quick Dream notes in
        # the prompt, so that is what it answers with here. The candidate
        # answers with the notes that would stay beside it (judged_outcomes).
        from .. import quick

        notes_now = quick.block(quick.active(store)) if moments else ""
        if moments:
            try:
                before_answers = judged.answers(
                    backend, journal, moments, "before", check, kept_as=kept_incumbent, notes=notes_now
                )
            except (Interrupted, MemoryError):
                raise
            except Exception as exc:  # noqa: BLE001 - judging falls back, never fails the Dream
                import logging

                logging.getLogger("songe").warning("judged_unavailable %s: %s", type(exc).__name__, exc)
                record["judged_error"] = type(exc).__name__
                moments = []

        # Moments the lessons are not about (the drift set: the base was
        # right there and nothing was trained on them). Each version answers
        # them too; how often the candidate acts differently is recorded
        # beside the gate and decides nothing.
        others = judged.unrelated(drift_set) if moments else []
        before_others = {}

        def answer_others(name):
            try:
                return judged.answers(backend, journal, others, name, check)
            except (Interrupted, MemoryError):
                raise
            except Exception as exc:  # noqa: BLE001 - a record, never the Dream
                import logging

                logging.getLogger("songe").warning("unrelated_unavailable %s: %s", type(exc).__name__, exc)
                others.clear()
                return {}

        if others:
            before_others = answer_others("unrelated-before")

        def notes_beside(scale):
            """(the notes that would stay in the prompt beside the candidate
            at this strength, how many would leave). The notes of lessons it
            follows without them leave on adoption (judged.learned); the
            candidate is loaded at that strength, or its answers are kept."""
            gone = {row["id"] for row in learned_now(scale) if row["learned"]}
            staying = [lesson for lesson in quick.active(store) if lesson.get("id") not in gone]
            return quick.block(staying), len(gone)

        # A lesson that passed the note test still keeps its note when the
        # candidate, served without it, is judged worse at one of that
        # lesson's own conversations (live: the pnpm note left on three
        # held-out scenes, both pnpm conversations were then worse, and the
        # habit fell from 10 of 16 test requests to 2).
        kept_back, judged_as = {}, {}

        def learned_now(scale):
            back = kept_back.get(scale, set())
            return [
                row | {"learned": row["learned"] and row["id"] not in back, "kept_back": row["id"] in back}
                for row in judged.learned_lessons(backend, journal, store, f"scale-{scale}", check)
            ]

        def judged_outcomes(scale):
            """Each judged moment's outcome (win, loss, tie) for this
            strength, {id: verdict}; {} when judging is unavailable (the KL
            decides). The candidate is loaded at that strength."""
            if not moments:
                return {}
            backend.scale_delta(baseline, trained, scale)
            if others:
                answer_others(f"unrelated-scale-{scale}")
            try:
                from ..reading.store import signals as lesson_signals

                session = {s["id"]: s.get("session") for s in lesson_signals(store)}
                name = f"scale-{scale}"
                while True:
                    staying, gone = notes_beside(scale)
                    after_answers = judged.answers(
                        backend, journal, moments, name, check, notes=staying
                    )
                    rows = judged.judge(
                        backend, journal, moments, before_answers, after_answers, name, check)
                    back = judged.kept_back(learned_now(scale), rows, session)
                    if not back:
                        break
                    kept_back.setdefault(scale, set()).update(back)
                    name = f"scale-{scale}-kept-{len(kept_back[scale])}"
                judged_as[scale] = name
                return {r["id"]: r | {"served": True, "learned": gone} for r in rows}
            except (Interrupted, MemoryError):
                raise
            except Exception as exc:  # noqa: BLE001
                import logging

                logging.getLogger("songe").warning("judged_unavailable %s: %s", type(exc).__name__, exc)
                record["judged_error"] = type(exc).__name__
                return {}
            finally:
                backend.scale_delta(baseline, trained, scale)

        def with_outcomes(rows, verdicts):
            keys = ("outcome", "by", "served", "learned")
            return [r | ({k: verdicts[r["id"]][k] for k in keys if k in verdicts[r["id"]]}
                         if r["id"] in verdicts else {}) for r in rows]

        def scale_adapter(scale):
            adapter = work / f"scale-{scale}-adapter"
            backend.scale_delta(baseline, trained, scale)

            def save():
                backend.save(adapter)
                return {"path": str(adapter)}

            journal.unit(f"scaled-{scale}", save)
            return adapter

        def scale_artifact(scale, adapter):
            if kind == "mlx-adapter":
                return Path(config.model)
            destination = scale_destination(scale)
            export_scale(scale, kind, adapter, destination)
            return destination

        def scale_destination(scale):
            suffix = ".gguf" if kind.startswith("gguf-") else ""
            return (
                Path(config.models_dir)
                / config.publisher
                / f"{config.public_model.split('/', 1)[1].replace('/', '-')}-night-{night}-scale-{scale}{suffix}"
                if config.upstream == "lmstudio"
                else work / f"scale-{scale}-artifact{suffix}"
            )

        def discard(scale):
            # A failed strength leaves nothing behind before the next is built.
            import shutil

            from ..artifacts.runtime import remove_owned

            if kind in {"mlx-fused", "gguf-full"}:
                remove_owned(scale_destination(scale), store.get("installation_id"))
            shutil.rmtree(work / f"scale-{scale}-adapter", ignore_errors=True)
            # A rerun of this strength must build it again, not trust records
            # of files that are gone.
            for name in (f"scaled-{scale}", f"fused-{scale}"):
                journal.path(name).unlink(missing_ok=True)

        def scaled(scale):
            # A strength already measured in an earlier slice still counts in
            # the gate's progress, though its verdict comes from the journal.
            name = pass_name(f"scale-{scale}")
            cache = journal.load(name, {})
            if cache:
                gate_progress(
                    name,
                    sum(len(cache.get(k) or []) for k in ("drifts", "drift_set", "rows")),
                    cache,
                )
            # The verdict's evidence is kept, so a resumed gate never reopens a
            # discarded candidate.
            rows = journal.unit(
                pass_name(f"scale-result-{scale}"),
                lambda: evaluate_scale(scale),
            )
            values = record.setdefault("gate_values", {})
            if name not in values and isinstance(rows, list) and rows:
                # Resumed after this strength was measured: its values come
                # from the journal (live: a resumed night kept one strength's).
                values[name] = {
                    "held_out": mean(r["kl"] for r in rows),
                    "general": mean(cache["drifts"]) if cache.get("drifts") else None,
                    "drift": mean(cache["drift_set"]) if cache.get("drift_set") else None,
                    "count": len(rows),
                    "drift_count": len(cache.get("drift_set") or []),
                    **sharpness(cache),
                }
                record.setdefault("drift_checks", {}).setdefault(name, cache.get("drift_check"))
            return rows

        def evaluate_scale(scale):
            scale_adapter(scale)
            # Rejected early when the remaining rows cannot change the
            # outcome: decide() fails a candidate whose drift is over the
            # limit, or whose judged moments are not clearly better, whatever
            # its held-out rows hold.
            drift = measured(f"scale-{scale}", backend, rows=False)
            if drift is not None and drift > config.general_drift_limit:
                return {"early": Decision(
                    False, "General behaviour drift exceeds the limit", general_drift=drift
                ).dict(), "judged": []}
            verdicts = judged_outcomes(scale)
            judged_rows = [
                {"id": s["id"], "conversation": s["conversation"], **{
                    k: verdicts[s["id"]][k] for k in ("outcome", "by", "served", "learned")
                    if k in verdicts[s["id"]]}}
                for s in heldout if s["id"] in verdicts
            ]
            rejected = judged_check(judged_rows, confidence=config.gate_confidence)
            if rejected is not None:
                rejected.general_drift = drift
                return {"early": rejected.dict(), "judged": judged_rows}
            before = before_rows()
            scale_adapter(scale)
            adapted = with_outcomes(measured(f"scale-{scale}", backend), verdicts)
            adapter_verdict = decide(
                before,
                adapted,
                min_heldout=config.min_heldout,
                drift_limit=config.general_drift_limit,
                confidence=config.gate_confidence,
            )
            if kind not in {"mlx-fused", "gguf-full"} or not adapter_verdict.accepted:
                return adapted
            adapter = work / f"scale-{scale}-adapter"
            artifact = scale_artifact(scale, adapter)
            from .calibrate import admit_phase

            check()
            admit_phase(config, store, platform, "fused_evaluation")
            fused = backend_type(
                replace(
                    config,
                    model=str(
                        evaluation_model(
                            config, kind, artifact, platform, lambda: check(False)
                        )
                    ),
                ),
                check,
                train=False,
            )
            try:
                journal.phase = "gate"
                # The fused model is the same candidate: the judged outcomes stand.
                outcomes = {r["id"]: r for r in adapted if r.get("outcome")}
                return [r | ({k: outcomes[r["id"]][k] for k in ("outcome", "by", "served", "learned")
                              if k in outcomes[r["id"]]} if r["id"] in outcomes else {})
                        for r in measured(f"fused-evaluation-{scale}", fused)]
            finally:
                del fused
                gc.collect()

        scale, decision, scales = choose_scale(
            scaled,
            before_rows,
            config.wake_scales,
            min_heldout=config.min_heldout,
            drift_limit=config.general_drift_limit,
            confidence=config.gate_confidence,
            check=check,
            discard=discard,
        )
        record["scale_evaluations"] = scales
        record["wake_scale"] = scale
        if others:
            # From the journal, so a resumed check reports every strength tried.
            counts = {
                str(k): judged.acts_differently(journal, others, before_others, f"unrelated-scale-{k}")
                for k in config.wake_scales
            }
            record["unrelated_actions"] = {k: v for k, v in counts.items() if v["checked"]}
        defer_export = kind == "gguf-lora" and backend.kind == "mlx"
        if scale is None:
            last_measurement = next(reversed(scales.values()), {})
            decision = Decision(
                False,
                "No adapter scale passed the real gate",
                **{
                    k: last_measurement.get(k)
                    for k in ("heldout_change", "noise_margin", "general_drift")
                },
            )
            artifact, adapter = work / "adapter", None
        else:
            adapter = scale_adapter(scale)
            artifact = (
                scale_destination(scale) if defer_export else scale_artifact(scale, adapter)
            )
            record["layer_deltas"] = backend.layer_deltas(baseline)
        if hasattr(backend, "teacher_gain"):
            store.set(
                "teacher_gain",
                journal.unit("teacher-gain", lambda: backend.teacher_gain(heldout)),
            )
        if config.learning == "dreams" and not moments:
            # Before/after examples the older way, only when the gate could
            # not judge moments (the judged moments are the examples then).
            from ..practice.pool import practice_steps
            from .examples import original_moments, practice_moments
            from .examples import run as run_examples

            def examples_progress(done, total):
                # After the gate: before/after answers compared so far.
                record["data_progress"] = {
                    "phase": "check",
                    "completed": done,
                    "total": total,
                    "unit": "compared",
                    "eta_seconds": None,
                }
                store.night(night, phase="check", record=record)

            try:
                journal.phase = "check"
                sets = journal.unit(
                    "example-sets",
                    lambda: [
                        ["original", original_moments(store)],
                        [
                            "held_out",
                            practice_moments(
                                store, practice_steps(store, split="held_out"), "held_out"
                            ),
                        ],
                        [
                            "trained",
                            practice_moments(
                                store, pool[: config.example_trained], "trained"
                            ),
                        ],
                    ],
                )
                record.update(
                    run_examples(
                        backend,
                        journal,
                        sets,
                        baseline=baseline,
                        trained=trained,
                        after_scale=scale
                        if scale is not None
                        else max(config.wake_scales),
                        config=config,
                        check=check,
                        progress=examples_progress,
                    )
                )
            except (Interrupted, MemoryError):
                raise
            except Exception as exc:  # noqa: BLE001 - examples illustrate, never decide
                import logging

                logging.getLogger("songe").exception("examples_failed")
                record["examples_error"] = type(exc).__name__
            finally:
                if scale is not None:
                    backend.scale_delta(baseline, trained, scale)
            store.night(night, record=record)
        if moments:
            # The report shows what the gate decided on: the judged moments of
            # the strength used, or, when none passed, the one that came
            # closest (live: half strength scored 6/1, the report showed 4/1).
            shown = scale if scale is not None else max(
                config.wake_scales,
                key=lambda k: (scales.get(str(k)) or {}).get("heldout_change") or float("-inf"))
            rows = (scales.get(str(shown)) or {}).get("judged") or []
            if rows:
                after_answers = judged.answers(
                    backend, journal, moments, judged_as.get(shown, f"scale-{shown}"), check,
                    notes=notes_beside(shown)[0],
                )
                record["examples"] = judged.examples(moments, before_answers, after_answers, rows)
                record["improved"] = judged.summary(rows)
                record["judged"] = rows
                if scale is not None:
                    backend.scale_delta(baseline, trained, scale)
            store.night(night, record=record)
        if config.learning == "dreams":
            # Which notes leave the prompt when this version is adopted: only
            # those of lessons it follows without the note, on held-out
            # practice (judged.learned). A lesson not learned yet keeps its
            # note beside the adapter, so adopting never takes away what the
            # note alone gave the user. Nothing measured: every note stays.
            record["absorbed_signals"] = []
            if decision.accepted and scale is not None:
                try:
                    backend.scale_delta(baseline, trained, scale)
                    record["learned"] = learned_now(scale)
                    record["absorbed_signals"] = sorted(
                        {sig for row in record["learned"] if row["learned"] for sig in row["signals"]}
                    )
                except (Interrupted, MemoryError):
                    raise
                except Exception as exc:  # noqa: BLE001 - unmeasured: the notes stay
                    import logging

                    logging.getLogger("songe").warning("learned_unavailable %s: %s", type(exc).__name__, exc)
                    record["learned_error"] = type(exc).__name__
                store.night(night, record=record)
        record.update(decision.dict())
        change = decision.heldout_change
        store.event(
            "eval",
            " · ".join(
                filter(
                    None,
                    [
                        "passed" if decision.accepted else "kept previous version",
                        f"held-out change {change:+.3f}"
                        if isinstance(change, (int, float))
                        else "",
                        decision.reason,
                    ],
                )
            ),
        )
        store.set(
            "wake_last",
            {
                "at": time.time(),
                "repeat_errors_delta": None,
                "drift": decision.general_drift,
                "swapped": False,
                "expected": record["expected"],
                "candidate": f"night-{night}" if decision.accepted else None,
            },
        )
        if decision.accepted:
            version_id = f"night-{night}"
            candidate = {
                "id": version_id,
                "identifier": f"{config.public_model}:{store.get('installation_id')}-{version_id}"
                if config.upstream == "ollama"
                else (
                    config.base_identifier
                    if config.upstream == "llamacpp"
                    else f"{config.public_model}/{version_id}"
                ),
                "artifact": str(artifact),
                "adapter": str(artifact)
                if kind in {"gguf-lora", "peft-adapter"}
                else str(adapter),
                "training_adapter": str(adapter),
                "kind": kind,
                "base": config.model_ref.get("path", config.model),
                "created": time.time(),
                "night": night,
            }
            if defer_export:
                # The next worker exports after this process releases the MLX
                # model. Keep the passed gate and its adapter as one checkpoint.
                record["pending_export"] = candidate
                store.night(
                    night, status="interrupted", phase="fuse", progress=0.95, record=record
                )
            else:
                record.update(candidate=candidate, ended_at=time.time())
                store.night(
                    night, status="ready", phase="swap", progress=0.95, record=record
                )
        else:
            record["ended_at"] = time.time()
            store.night(
                night, status="rejected", phase="check", progress=1, record=record
            )
            store.set("last_completed", time.time())
            if config.learning == "dreams":
                # Nothing was adopted, so nothing was practised: this Dream's
                # scenes are trained again with the next new ones. On their
                # own they wait (practice.pool.untrained).
                from ..practice.pool import keep_for_retry

                keep_for_retry(store, record["practice_revisions"])
            else:
                store.set("experience_watermark", record["watermark"])
                store.set("practice_revisions", record["practice_revisions"])
            if record["rebuild"]:
                # A rejected rebuild is still a completed finite attempt. Wait
                # for new evidence or a changed rebuild request before retrying.
                store.set("completed_rebuild", record["rebuild_key"])
        print(json.dumps(record), flush=True)
        return record
    except StorageWait as exc:
        store.set("disk_wait", str(exc))
        if night is not None:
            record[reason_key] = str(exc)
            store.night(night, status="interrupted", record=record)
        return {"reason": str(exc)}
    except (Interrupted, OversizedStep) as exc:
        interrupted_by = interrupted_by or str(exc) or "activity"
        if night is not None:
            record[reason_key] = interrupted_by
            store.night(night, status="interrupted", record=record)
        return {"interrupted": True}
    except BaseException as exc:
        import logging

        from ..memory import MemoryPressure
        from ..telemetry import failure_hint

        if isinstance(exc, (MemoryPressure, MemoryError)) or "[metal::malloc]" in str(
            exc
        ):
            # Short memory is a wait, not a failure: the night keeps its work
            # and resumes when there is room.
            if night is not None:
                record[reason_key] = "Waiting for memory"
                store.night(night, status="interrupted", record=record)
            raise

        store.set("child_error", failure_hint("Learning", store.root))
        logging.getLogger("songe").exception(
            "learning_failed", extra={"diagnostic": True}
        )
        if night is not None:
            if reason_key == "export_wait":
                record["export_wait"] = "Export failed; served version preserved"
                store.night(night, status="interrupted", record=record)
            else:
                record["reason"] = (
                    f"Work failed; served version preserved. Diagnostics: {store.root / 'songe.log'}"
                )
                store.night(night, status="failed", record=record)
        raise
    finally:
        import logging

        from .cleanup import collect_work

        try:
            collect_work(store, policy_config)
        except Exception:
            logging.getLogger("songe").exception("work_cleanup_failed")
        store.slice(
            slice_id,
            phase=journal.phase if journal else "record",
            units_done=journal.units if journal else 0,
            interrupted_by=interrupted_by,
            record=admission.record,
        )
        lease.close()


def isolated(config, store, *, dry_run=False, slice_seconds=None):
    """Public/hardware entry: model work is always in an isolated guarded child."""
    import subprocess
    import sys

    from ..memory import (
        MemoryPressure,
        TrainingPlatform,
        guarded_run,
        record_memory_stop,
        snapshot,
    )
    from ..telemetry import configure

    config.save(store.root)
    configure(store.root)
    argv = [
        sys.executable,
        "-m",
        "songe.sleep.worker",
        "--data-dir",
        str(store.root),
        "--child",
    ]
    if config.model_key:
        argv += ["--model-key", config.model_key]
    if dry_run:
        argv += ["--dry-run"]
    if slice_seconds is not None:
        argv += ["--slice-seconds", str(slice_seconds)]
    platform = None if dry_run else TrainingPlatform(current(), config)
    sizing = None
    export_only = not dry_run and bool(pending_export(config, store))
    if export_only:
        argv += ["--export-only"]
    measure = not dry_run and not export_only

    def observe(sample):
        history.observe(sample)
        store.set("memory_pressure_history", history.data)

    if measure:
        from .calibrate import calibrate
        from .calibration_state import needs_growth

        if needs_growth(config, store):
            calibrate(config, store.root)
            return {
                "reason": "Measuring training size",
                "calibration": store.get("calibration_progress"),
            }
    if measure and config.calibration_available:
        from .sizing import PressureHistory, choose_size, free_needed, waiting_reason

        sample = snapshot(platform)
        history = PressureHistory(store.get("memory_pressure_history", {}))
        observe(sample)
        sizing = choose_size(config, store.get("calibration"), sample, history.data)
        if sizing is None:
            calibration = store.get("calibration")
            return {
                "reason": waiting_reason(
                    sample,
                    history.data,
                    free_needed(config, calibration, sample, history.data),
                )
            }
        store.set("last_slice_size", sizing)
        argv += ["--sizing", json.dumps(sizing)]
    nights_before = store.nights()
    try:
        result = guarded_run(
            argv,
            platform=platform,
            required_bytes=sizing["host_bytes"] if sizing else 0,
            required_device_bytes=sizing["device_bytes"] if sizing else 0,
            margin_bytes=sizing["margin_bytes"] if sizing else 0,
            prediction_error_bytes=sizing["prediction_error_bytes"] if sizing else 0,
            device_prediction_error_bytes=sizing["device_prediction_error_bytes"]
            if sizing
            else 0,
            on_sample=observe if measure and config.calibration_available else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except MemoryPressure as exc:
        record_memory_stop(store, exc, nights_before=nights_before)
        raise
    if result.returncode:
        from ..telemetry import failure_hint, log_child_stderr

        log_child_stderr(getattr(result, "stderr", None))
        raise RuntimeError(failure_hint("Learning", store.root))
    if dry_run:
        return json.loads(result.stdout)
    if store.get("disk_wait"):
        return {"reason": store.get("disk_wait")}
    nights = store.nights()
    return nights[-1]["record"] if nights else {"reason": "Waiting for experience"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=data_dir())
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--slice-seconds", type=float)
    parser.add_argument("--sizing", type=json.loads)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--model-key")
    args = parser.parse_args()
    config, installation_keys = Config.load_with_keys(args.data_dir)
    store, model_keys = Store(args.data_dir), ()
    if args.model_key:
        from ..models.registry import model_config_with_keys

        config, model_keys, _ = model_config_with_keys(config, store, args.model_key)
        store = store.scoped(args.model_key)
    from ..telemetry import configure

    configure(store.root)
    if args.dry_run:
        print(
            json.dumps(
                preview(config, store)[0]
                if args.child
                else isolated(config, store, dry_run=True),
                indent=2,
            )
        )
    elif args.child:
        outcome = run(
            config,
            store,
            args.slice_seconds,
            args.sizing,
            export_only=args.export_only,
            setting_sources=training_setting_sources(installation_keys, model_keys),
        )
        # Why a run ended (also one that never started a night): the
        # service reads it to carry a pressed Deep Dream on, or say why not.
        reason = outcome.get("reason") if isinstance(outcome, dict) else None
        store.set("worker_outcome", {"reason": reason, "at": time.time()})
        if reason:
            logging.getLogger(__name__).info("worker_outcome %s", reason)
    else:
        # Standalone entry has the same guard as scheduler-owned workers.
        import asyncio

        from ..upstream import create

        async def standalone():
            upstream = create(config, store.root)
            parked = store.get("parked_models", [])
            with store.connect() as db:
                managed = {r[0] for r in db.execute("SELECT identifier FROM versions")}
            try:
                if upstream:
                    if parked:
                        await upstream.restore(parked, managed)
                    parked = await upstream.park(
                        on_snapshot=lambda items: store.set("parked_models", items)
                    )
                exporting = pending_export(config, store) is not None
                result = await asyncio.to_thread(
                    isolated, config, store, slice_seconds=args.slice_seconds
                )
                if not exporting and result.get("pending_export"):
                    result = await asyncio.to_thread(
                        isolated, config, store, slice_seconds=args.slice_seconds
                    )
                print(json.dumps(result))
            finally:
                if upstream:
                    await upstream.restore(parked, managed)
                    if store.version():
                        await upstream.ensure(store.version())
                    store.set("parked_models", [])

        asyncio.run(standalone())


if __name__ == "__main__":
    from songe.telemetry import diagnostic_main

    diagnostic_main(main)
