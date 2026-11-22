"""The same three status lines in every interface; no hardware probes here."""

import re
import time

from .interface_copy import terse


def training_model(data):
    """Only active training has an accent; pauses and other phases are neutral."""
    status = data.get("activity")
    if status:
        return status.get("model") if status.get("training") else None
    work = data.get("training") or {}
    if (
        work.get("phase") in {"train", "distill"}
        and not data.get("paused")
        and data.get("service_running") is not False
    ):
        return work.get("model") or data.get("training_model") or data.get("model")
    return None


def model_training(model, active):
    return active is not None and active in {
        model["id"],
        *(file["id"] for file in model.get("files", [])),
    }


def duration(seconds):
    if seconds < 60:
        return f"{max(0, int(seconds))}s"
    if seconds < 3600:
        return f"{int(seconds / 60)}m"
    return f"{int(seconds / 3600)}h"


def lines(data):
    activity = data.get("activity")
    if activity:
        return [(key, activity[key]) for key in ("now", "next", "paused")]
    # Older services and offline/test clients still use the same presentation.
    training = data.get("training")
    now = work_text(training, data.get("training_model") or data.get("model"))
    return [
        ("now", now or "waiting · " + terse(data.get("reason", "service stopped"))),
        ("next", data.get("next_action") or "wait for scheduler state"),
        ("paused", "yes · resume" if data.get("paused") else "no"),
    ]


def work_text(training, model=None):
    if not training:
        return None
    phase = training.get("phase", "record")
    if phase == "calibrate":
        progress = training.get("data_progress") or {}
        if progress.get("target_sequence"):
            from .sleep.calibration_state import measurement_text

            return measurement_text(progress)
    label = {
        "calibrate": "calibrating",
        "train": "training",
        "distill": "training",
        "select": "scoring",
        "score": "scoring",
        "gate": "evaluating",
        "check": "evaluating",
        "evaluate": "evaluating",
        "fuse": "fusing",
        "swap": "swapping",
        "record": "starting worker for",
    }.get(phase, phase)
    text = f"{label} {training.get('model') or model or 'model'}"
    progress = training.get("data_progress") or {}
    progress_phase = {"distill": "train", "select": "score"}.get(phase, phase)
    if (
        progress_phase not in {"train", "score", "calibrate"}
        or progress.get("phase", progress_phase) != progress_phase
    ):
        progress = {}
    if phase in {"train", "distill"} and training.get("total_steps"):
        text += f" · step {training['step']}/{training['total_steps']}"
    elif progress.get("total"):
        text += f" · {progress['completed']}/{progress['total']}"
        if phase != "calibrate":
            text += f" {progress.get('unit', 'samples')}"
    eta = progress.get("eta_seconds")
    if eta and eta > 0:
        text += f" · ~{duration(eta)} left"
    return text


def experience_wait(config, store, models):
    from .config import Config
    from .sleep.calibration_state import measurement_text, needs_growth

    enabled = [m for m in models if m.get("enabled")]
    if not enabled:
        return "learning off for all models" if models else "no models · songe init"
    supported = [m for m in enabled if m.get("support") != "unsupported"]
    if not supported:
        from .models.library import model_name

        return "unsupported · " + model_name(enabled[0].get("name") or enabled[0]["id"])
    for model in supported:
        scoped = store.scoped(model["id"])
        policy = (
            Config.for_model(store.root, model["id"]) if scoped.model_key else config
        )
        if needs_growth(policy, scoped):
            if getattr(policy, "learning", "") == "dreams" and store.traffic_times(1):
                from .practice.pool import heldout_units, test_minimum

                held = heldout_units(scoped)
                if held < test_minimum(policy):
                    # Checked before measuring (the scheduler waits on it first).
                    return f"collecting test moments · {held} of {policy.min_heldout} needed to check updates"
            saved = scoped.get("calibration") or {}
            progress = saved.get("progress") or {
                "measured_sequence": max(
                    (r["sequence"] for r in saved.get("rows", [])), default=0
                ),
                "target_sequence": policy.max_sequence,
            }
            return measurement_text(progress)
    if not store.traffic_times(1):
        return "no new conversations through songe yet"
    if getattr(config, "learning", "") == "dreams":
        return practice_wait(config, store, supported)
    for model in supported:
        scoped = store.scoped(model["id"])
        examples, conversations = scoped.feedback_counts()
        train, held, pending = scoped.experience_counts()
        if examples < config.min_train:
            return f"collecting conversations · {examples} of {config.min_train}"
        # Tests are set aside from the same data: at least min_heldout
        # conversations for tests and one more to train on.
        if conversations < config.min_heldout + 1:
            return f"collecting conversations · {conversations} of {config.min_heldout + 1} needed to check updates"
        pending = max(
            pending,
            scoped.pending_learnable() if hasattr(scoped, "pending_learnable") else 0,
        )
        if pending >= config.min_new:
            return f"preparing examples · {pending} examples waiting to learn"
        if not pending and (unfit := scoped.get("unfit_samples", {}) or {}):
            from .sleep.samples import SampleAdmission

            policy = (
                Config.for_model(store.root, model["id"])
                if scoped.model_key
                else config
            )
            admission = SampleAdmission(
                policy, None, scoped.get("calibration"), scoped, None
            )
            if any(
                key.startswith(admission.scope + ":")
                and entry.get("required_sequence") is not None
                for key, entry in unfit.items()
            ):
                return f"collecting conversations · 0 of {config.min_new} new examples · some too long to train"
        if pending < config.min_new:
            return (
                f"collecting conversations · {pending} of {config.min_new} new examples"
            )
    return "no new conversations to learn from since the last attempt"


def deep_step(store):
    """A pressed Deep Dream between its runs, in plain words."""
    wait = store.get("memory_wait")
    if wait:
        # In the units a Mac is sold in ("24 GB", not 25.8 decimal GB).
        gb = lambda n: f"{n / 2**30:.1f} GB".replace(".0 GB", " GB") if isinstance(n, (int, float)) else "?"  # noqa: E731
        return (f"Deep Dream · waiting for memory · {gb(wait.get('available'))} free of {gb(wait.get('total'))}"
                f" · close some apps · it tries again on its own (try {wait.get('try')} of {wait.get('tries')})")
    fit = store.get("fit_wait")
    if fit:
        gb = lambda n: f"{n / 2**30:.1f} GB".replace(".0 GB", " GB") if isinstance(n, (int, float)) else "?"  # noqa: E731
        return (f"Deep Dream · waiting for memory to train · {gb(fit.get('available'))} free,"
                f" training needs about {gb(fit.get('needed'))} · close some apps · it starts on its own")
    if store.get("growth_requested"):
        return "Deep Dream · measuring this machine for longer practice scenes, then training starts"
    if store.get("dream_capacity_edges"):
        return "Deep Dream · measured the most this machine can train · training starts next"
    return "Deep Dream · getting ready to train"


def practice_wait(config, store, supported):
    """What learning waits on under learning = "dreams", in the scheduler's
    own terms (ModelScheduler.enough): kept practice to train on, held-out
    real conversations to check with, and new practice since the last cycle."""
    from .practice.pool import practice_steps, untrained

    for model in supported:
        scoped = store.scoped(model["id"])
        from .practice.pool import dream_minimums

        min_train, _, min_new = dream_minimums(config)
        kept = practice_steps(scoped)
        if len(kept) < min_train:
            # A waiting condition, not an activity: practice is written in a
            # Deep Dream, from what reading found.
            return f"a Deep Dream needs {min_train} practice scene{'s' if min_train != 1 else ''} · {len(kept)} kept so far"
        from .practice.pool import heldout_units, test_minimum

        held = heldout_units(scoped)  # real moments, filled by held-out practice
        if held < test_minimum(config):
            return f"collecting test moments · {held} of {config.min_heldout} needed to check updates"
        waiting = len(untrained(scoped, kept))
        if scoped.get("dream_stopped"):
            # The user's Dream gave up: say so, not "ready to learn".
            return "your Deep Dream stopped before training · see the note"
        if store.get("requested_model") == model["id"]:
            return deep_step(scoped)
        if waiting >= min_new:
            return f"ready to learn · {waiting} practice scenes waiting"
        return f"collecting conversations · {waiting} of {min_new} new practice scenes"
    return "no new conversations to learn from since the last attempt"


def activity(config, store, scheduler, training, models, *, live, ready=None):
    from .models.library import model_name

    paused = bool(store.get("paused", False))
    threshold = getattr(
        scheduler, "idle_threshold_seconds", config.idle_default_seconds
    )
    elapsed = (
        max(0, time.monotonic() - scheduler.last_traffic)
        if live and scheduler and hasattr(scheduler, "last_traffic")
        else None
    )
    candidate = (ready or {}).get("model_id") or getattr(
        scheduler, "active_model", None
    )
    if not candidate and live and hasattr(scheduler, "choose_model"):
        candidate = scheduler.choose_model()
    target = next((m for m in models if m["id"] == candidate), None)
    if target is None:
        # With nothing runnable, name the enabled model so Next says what it
        # waits for (data) instead of asking the user to choose a model.
        target = next(
            (
                m
                for m in models
                if m.get("enabled") and m.get("support") != "unsupported"
            ),
            None,
        )
    name = model_name((target or {}).get("name") or candidate or config.public_model)
    from .sleep.calibration_state import needs_growth

    growing = bool(
        target
        and needs_growth(
            getattr(scheduler, "config", config),
            store.scoped(target["id"]) if store.model_key != target["id"] else store,
        )
    )
    dreams = getattr(config, "learning", "") == "dreams"
    action = (
        "swap"
        if ready
        else "calibrate"
        if target and (target.get("support") == "pending" or (growing and not dreams))
        # Under Dreams, measuring again is part of a Deep Dream, not its own step.
        else "Deep Dream for" if dreams
        else "train"
    ) + f" {name}"
    reason = (
        getattr(scheduler, "reason", "scheduler starting")
        if live
        else "service stopped · songe serve"
    )
    p = getattr(scheduler, "last_probe", None)
    if reason in {
        "Waiting for experience",
        "Waiting for enabled models and learnable experience",
    }:
        reason = experience_wait(config, store, models)
    elif (
        reason in {"Waiting for an idle gap", "Will sleep when idle"}
        or "without proxy traffic" in reason
    ):
        # With nothing to run, quiet time is not what it waits on; data is.
        reason = (
            experience_wait(config, store, models)
            if live
            and not ready
            and hasattr(scheduler, "choose_model")
            and scheduler.choose_model() is None
            else f"proxy idle {duration(elapsed or 0)}/{duration(threshold)}"
        )
    elif reason == "Waiting: battery low" and p and p.power.fraction is not None:
        reason = f"battery {p.power.fraction:.0%}"
    elif reason.startswith("Waiting for free memory") and p:
        needed = getattr(getattr(scheduler, "config", config), "calibrated_peak_gb", 0)
        reason = f"memory · {p.free_gb:.1f} GB free" + (
            f", needs {needed:.1f} GB" if needed else f" · pressure {p.pressure}"
        )
    elif reason == "Waiting for calibration":
        reason = "calibration required"
    elif reason == "Waiting for hardware probes":
        reason = "hardware telemetry unavailable"
    elif reason == "Waiting for the machine to cool":
        reason = (
            f"temperature · thermal level {p.thermal}"
            if p and p.thermal is not None
            else "machine temperature too high"
        )
    elif reason == "Waiting before the next work slice":
        reason = "retry delay after previous work slice"
    elif reason == "Waiting to retry after an error":
        from .schedule import retry_wait

        failures = getattr(scheduler, "store", store).get("failures") or {}
        left = retry_wait(failures.get("count", 1)) - (
            time.time() - failures.get("at", time.time())
        )
        reason = f"error · retry in {duration(max(left, 60))}"
    else:
        reason = terse(reason)
    if (
        target
        and "calibration paused · memory" in target.get("reason", "")
        and reason.startswith(
            (
                "proxy idle",
                "memory",
                "Waiting for memory",
                "Working in an idle gap",
                "calibration required",
                "scheduler starting",
            )
        )
    ):
        reason = "calibration paused · memory · retry when idle"
    if (
        live
        and models
        and all(m.get("support") == "unsupported" for m in models)
        and not ready
    ):
        reason = "unsupported · " + model_name(models[0].get("name") or models[0]["id"])
    if ready:
        reason = "checks passed · swap pending"
    waiting_on_agent = (
        live and (getattr(scheduler, "last_traffic", 0) or 0) > time.monotonic()
    )  # an agent is running tools and will call back
    if (
        live
        and (any(getattr(scheduler, "in_flight", {}).values()) or waiting_on_agent)
        and (not training or ready)
    ):
        reason = "proxy requests in flight"
    now = (
        work_text(training | {"model": name}, name) if training and not ready else None
    )
    if now and getattr(config, "learning", "") == "dreams" and now.startswith("measuring training size"):
        # Inside a Dream this is its "Getting ready" step, not training.
        now = "getting ready · measuring how much this computer can learn at once" + now.removeprefix(
            "measuring training size").replace(" of ~", " of ")
    if (
        getattr(config, "learning", "") == "dreams"
        and not training
        and not ready
        and str(reason).startswith("measuring training size")
    ):
        # Leftover from the last measuring run (live: shown for six hours
        # after the Dream ended). Nothing is measuring: it is idle, and the
        # next Deep Dream measures first.
        now = "Idle · the next Deep Dream measures this machine first"
    stopping = bool(
        training
        and (
            getattr(scheduler, "cancelled_at", None) is not None
            or reason.startswith("Stopping:")
            or "checkpoint pending" in reason
        )
    )
    if stopping:
        if getattr(scheduler, "memory_stop_sample", None) is not None:
            reason = "pausing · not enough memory"
        elif "battery low" in reason:
            charge = (
                f"battery {p.power.fraction:.0%}"
                if p and p.power.fraction is not None
                else "battery low"
            )
            reason = f"pausing · {charge}"
        else:
            reason = f"pausing · {getattr(scheduler, 'interrupted_by', None) or 'model in use'}"
        now = None
    if paused:
        reason = (
            "learning paused"
            if not training or ready
            else "pausing · learning turned off"
        )
        now = None
    if not live:
        now = None
        reason = "service stopped · songe serve" if models else "no models · songe init"
    condition = f"idle {duration(elapsed or 0)}/{duration(threshold)}"
    if not models:
        next_text = "songe init → connect a model"
    elif not live:
        next_text = "songe serve → resume scheduler"
    elif paused:
        next_text = f"resume → {action}" if target else "resume → select enabled model"
    elif stopping:
        next_text = f"{action} · continues when idle"
    elif ready:
        next_text = f"swap {name} · waits for idle gap + no proxy requests"
    elif training:
        phase = training.get("phase")
        following = {
            "calibrate": "train",
            "score": "train",
            "select": "train",
            "train": "check",
            "distill": "check",
            "fuse": "check",
            "gate": "swap",
            "check": "swap",
            "evaluate": "swap",
            "swap": "serve",
        }.get(phase, "run next work slice for")
        condition = (
            "training size measured"
            if phase == "calibrate"
            else "current work completes"
            if following != "swap"
            else "checks pass + no proxy requests"
        )
        next_text = f"{following} {name} · {condition}"
        if phase == "calibrate" and target and not target.get("pending"):
            # Measuring finishes with nothing to learn yet; data is the wait.
            next_text = (
                f"Pick a songe/… model in your agent · {name} learns from conversations"
            )
    elif target and store.get("requested_model") == target["id"]:
        # A pressed Deep Dream is pending: never "press Dream" again.
        next_text = "Your Deep Dream is pending · it starts on its own"
    elif target and target.get("auto_dream") is False:
        # Nothing starts on its own (auto-dream off); Dream runs one.
        next_text = "Auto-dream is off · press Dream to run one"
    elif target and "long quiet period" in str(reason):
        next_text = (
            f"Deep Dream once quiet for {duration(config.deep_quiet_seconds)}"
            " · Quick Dreams keep the prompt current"
        )
    elif (
        target
        and str(action).startswith(("train", "calibrate", "Deep Dream"))
        and str(reason).startswith(("no new conversations", "collecting conversations", "collecting test moments"))
    ):
        # Idle time is not what it waits on; data is.
        next_text = f"{action} · {'needs data' if reason.startswith('no new') else 'needs more data'}"
    elif target and reason.startswith("error · retry in "):
        next_text = f"{action} · in {reason.removeprefix('error · retry in ')}"
    elif target and re.match(r"(?:calibration paused · )?memory", reason):
        next_text = f"{action} · when memory frees"
    elif (
        target
        and getattr(config, "learning", "") == "dreams"
        and str(reason).startswith("measuring training size")
    ):
        # Under Dreams, measuring again happens only inside a Deep Dream
        # (live: "idle 2m/30s → calibrate", and nothing started).
        next_text = (
            "Auto-dream is off · press Dream to run one"
            if target.get("auto_dream") is False
            else f"Deep Dream once quiet for {duration(config.deep_quiet_seconds)} · it measures this machine first"
        )
    elif target:
        next_text = f"{condition} → {action}"
    elif reason.startswith("unsupported · "):
        next_text = "choose another model"
    else:
        next_text = "select model · learning enabled + learnable experience"
    reading = next(iter((getattr(scheduler, "reading_activity", None) or {}).values()), None) if live else None
    if reading and not training and not paused:
        # Reading conversations (in the background, or a Quick Dream): that is
        # what happens now, not a Deep Dream's wait.
        what = "in a Quick Dream" if reading.get("kind") == "quick" else "reading your conversations"
        count = f" · {reading.get('done', 0)} of {reading['total']}" if reading.get("total") else ""
        now = f"{what}{count}"
        next_text = "then the preferences in the prompt update" + (
            "" if reading.get("kind") == "quick"
            else f" · a Deep Dream after {duration(config.deep_quiet_seconds)} of quiet"
        )
    scoped_target = store.scoped(target["id"]) if target else None
    held = scoped_target and (scoped_target.get("memory_wait") or (
        scoped_target.get("fit_wait") and store.get("requested_sleep"))) and not scoped_target.get("dream_stopped")
    if held and not training and live:
        # A pressed Deep Dream waiting for memory: that, not an error, and
        # never "press Dream" while it is pending.
        now = deep_step(store.scoped(target["id"]))
        next_text = "Close some apps to free memory · it starts on its own"
    return {
        "now": now or f"waiting · {reason}",
        "next": next_text,
        "paused": "yes · resume" if paused else "no",
        "phase": "swap" if ready else (training or {}).get("phase"),
        "model": candidate,
        "training": bool(
            now and training and training.get("phase") in {"train", "distill"}
        ),
        "idle_elapsed_seconds": int(elapsed) if elapsed is not None else None,
        "idle_threshold_seconds": threshold,
    }
