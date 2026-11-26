"""Work in learned idle gaps; every heavy worker is resumable and guarded."""

import asyncio
import contextvars
import json
import logging
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from . import NAME
from .config import training_fingerprint
from .memory import (
    MEMORY_EXIT,
    GuardPolicy,
    MemoryPressure,
    capture_baseline,
    guard,
    kill,
    owned_command,
    process_options,
    record_memory_stop,
    snapshot,
)
from .memory import (
    interrupt as interrupt_group,
)
from .platform import Power, current, power_status
from .platform.locking import acquire as acquire_lock
from .steps import digest

logger = logging.getLogger(NAME)
# The model whose pressed Dream the running code belongs to: a model loaded
# for a press is hidden from every other work (ModelScheduler.loaded_listing).
PRESSING = contextvars.ContextVar("pressing", default=None)
AGENT_FOLLOWUP_SECONDS = 300
RETRY_FIRST_SECONDS = 300  # after an error, wait 5m, then 10m, 20m ... at most 1 day
RETRY_MAX_SECONDS = 86400


def retry_wait(count):
    """Seconds to wait after `count` consecutive errors; never gives up."""
    return min(RETRY_FIRST_SECONDS * 2 ** max(count - 1, 0), RETRY_MAX_SECONDS)


@dataclass
class Probes:
    ac: bool | None = False
    idle_seconds: float | None = 0
    thermal: int | None = 3
    total_gb: float = 0
    free_gb: float = 0
    available: bool = False
    pressure: str = "normal"
    power: Power = field(default_factory=Power)

    def dict(self):
        return asdict(self)


def probe(platform=None):
    platform = platform or current()
    power = power_status(platform)
    try:
        memory = platform.memory()
        return Probes(
            power.ac,
            platform.idle_seconds(),
            platform.thermal(),
            memory.total / 1e9,
            memory.available / 1e9,
            True,
            memory.pressure,
            power,
        )
    except (OSError, ValueError, TypeError):
        return Probes()


def gap_history(times):
    return [b - a for a, b in zip(times, times[1:], strict=False) if b > a]


def idle_threshold(gaps, slice_seconds, config):
    """Quiet time before work starts: a fixed short wait (product decision).

    Work yields to the next request at once and resumes from its checkpoint,
    so there is no reason to wait for a long gap.
    """
    return config.idle_default_seconds


PARK_WAIT_MAX = 120.0
RETURN_GAP_SECONDS = 120.0  # how long a restore waits for a gap to switch
IDLE_TICKS = 5  # an idle service checks this many times less often
EMPTY_RUN_WAIT_MAX = 1800.0  # a run that did nothing waits at most this long


def after_unload(sample, reclaimable):
    """The memory picture once the agent's model is unloaded (an estimate)."""
    return replace(
        sample,
        available=sample.available + reclaimable,
        vram=tuple(
            replace(
                a,
                memory_free=min(
                    a.memory_total or a.memory_free + reclaimable,
                    a.memory_free + reclaimable,
                ),
            )
            for a in sample.vram
        ),
    )


def park_wait(gaps, config):
    """Quiet time before unloading the model the agent is using.

    Training beside the loaded model starts after the short default. When it
    has to unload that model, the next message would wait for a reload, so it
    waits out most of your pauses inside a conversation first (under 5 min)."""
    inside = sorted(g for g in gaps if g < 300)
    if len(inside) < config.idle_history_min:
        return max(config.idle_default_seconds, 30.0)
    typical = inside[
        min(len(inside) - 1, int(len(inside) * config.idle_survival_probability))
    ]
    return min(PARK_WAIT_MAX, max(config.idle_default_seconds, typical))


def work_costs(config, calibration):
    """Optimizer and score costs at the current slice dimensions."""
    rows = (calibration or {}).get("rows", [])
    from .sleep.calibrate import costs
    from .sleep.sizing import dimensions

    matching = [r for r in rows if r["sequence"] == config.max_sequence]
    if matching:
        target = (
            config.max_sequence,
            config.response_tokens,
            config.token_chunk,
            config.accumulation,
        )
        row = next(
            (r for r in matching if dimensions(r) == target),
            max(matching, key=dimensions),
        )
        return costs(row, config.accumulation, config.response_tokens)
    return config.calibrated_step_seconds, config.calibrated_score_seconds


SLICE_PAUSE_MAX = 300  # seconds between work slices, at most
WORKER_ERR = "worker.err"  # the last learning child's stderr
FAILED_NIGHT_RETRIES = 3  # a pressed Dream's failed nights retried, then a note


def exit_reason(code, err_path, tail=600):
    """Why a learning child ended with an error, in a line or two: its exit
    code (or the signal that killed it) and the end of its stderr."""
    import signal as signals

    if code < 0:
        # Unix signal numbers (Windows' signal module lacks most of them).
        known = {2: "SIGINT", 6: "SIGABRT", 9: "SIGKILL", 11: "SIGSEGV", 15: "SIGTERM"}
        try:
            name = known.get(-code) or signals.Signals(-code).name
        except ValueError:
            name = f"signal {-code}"
        why = f"killed by {name}" + (" (out of memory?)" if name == "SIGKILL" else "")
    else:
        why = f"exit code {code}"
    try:
        text = Path(err_path).read_bytes()[-4000:].decode("utf-8", "replace")
    except OSError:
        text = ""
    lines = [line for line in text.strip().splitlines() if line.strip()]
    last = next((line for line in reversed(lines) if not line.startswith(" ")), "")
    return why + (f" · {last.strip()[:tail]}" if last else "")


def work_unit_seconds(config, calibration):
    """Measured work cost is an estimate, never a cycle budget."""
    return max(work_costs(config, calibration)) or config.slice_default_seconds


def measured_step_seconds(store):
    """This Dream's own measured optimizer step, once it has one: long
    practice scenes can cost far more than the calibration probe's step.
    Before one is measured, a slice that reached its deadline without
    finishing any unit says a unit takes longer than that slice: twice it."""
    nights = [n for n in store.nights() if n["status"] in {"running", "interrupted"}]
    progress = (nights[-1]["record"].get("data_progress") or {}) if nights else {}
    measured = progress.get("seconds_per_step") or 0
    if not nights:
        return measured
    # Back to the last slice that finished a unit: slices stopped for other
    # reasons (agent activity, pause, a restart) say nothing about unit cost.
    for past in reversed(store.slices()[-20:]):
        if past.get("units_done"):
            break
        if (
            past.get("interrupted_by") == "slice deadline"
            and past.get("ended")
            and past.get("started")
        ):
            measured = max(measured, 2 * (past["ended"] - past["started"]))
            break
    return measured


def slice_duration(config, calibration, gaps=(), elapsed=0, measured=0):
    """A slice always fits at least one unit of work, as measured in this
    Dream when it is (live: 82 s slices against 170 s steps never finished
    one, and each empty slice was followed by a growing wait)."""
    rows = (calibration or {}).get("rows", [])
    unit = max(work_unit_seconds(config, calibration), measured or 0) + max(
        (r.get("load_seconds", 0) for r in rows), default=0
    )
    tail = sorted(g - elapsed for g in gaps if g > elapsed)
    remaining = (
        tail[int((1 - config.idle_survival_probability) * (len(tail) - 1))]
        if len(tail) >= config.idle_tail_min
        else config.slice_max_seconds
    )
    return max(unit, min(config.slice_max_seconds, remaining))


def rebuild_reason(config, store):
    if (
        store.get("deep_rebuild_requested", False)
        or store.get("rebuild_required", False)
        or store.get("forgotten_at", 0) > store.get("rebuilt_at", 0)
    ):
        return "forget requested"
    if store.get("trained_base") not in (None, base_fingerprint(config)):
        return "base changed"
    if (
        store.get("cumulative_drift", 0)
        >= config.general_drift_limit * config.rebuild_drift_fraction
    ):
        return "cumulative drift near limit"
    if store.retirement_fraction() >= config.rebuild_retired_fraction:
        return "many experiences retired"
    return None


def resumable(night, store, config, *, configuration=None, base=None):
    """Whether an interrupted night can go on: same serving version, settings,
    base weights and tool environment as when it began. The worker, the
    scheduler and cleanup all ask this one question, so a night is never
    kept by one and refused by another."""
    record = night["record"]
    return (
        night["status"] == "interrupted"
        and record.get("expected") == store.get("current")
        and record.get("configuration")
        == (configuration or training_fingerprint(config))
        and record.get("generation") == store.get("candidate_generation")
        and record.get("base_fingerprint") == (base or base_fingerprint(config))
        and record.get("environment") == store.get("environment_fingerprint")
    )


def pending_export(config, store):
    nights = store.nights()
    if any(night["status"] == "ready" for night in nights):
        return None
    return next(
        (
            night
            for night in reversed(nights)
            if night["record"].get("pending_export") and resumable(night, store, config)
        ),
        None,
    )


def base_fingerprint(config):
    from pathlib import Path

    path = Path(config.model)
    files = []
    if path.is_dir():
        files = [
            (p.name, p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(path.iterdir())
            if p.name == "config.json" or p.suffix == ".safetensors"
        ]
    return digest({"path": config.model, "files": files})


def rebuild_key(config, store):
    return digest(
        {
            "reason": rebuild_reason(config, store),
            "configuration": asdict(config),
            "base": base_fingerprint(config),
            "generation": store.get("candidate_generation"),
            "environment": store.get("environment_fingerprint"),
        }
    )


# How long a model the server refused to load is answered without loading.
LOAD_RETRY_SECONDS = 300


class LoadRefused(RuntimeError):
    """The model server tried to load these weights and could not."""


def refusal_reason(exc):
    """The server's own reason in a few words: its message can carry a whole
    traceback, and the agent shows this line to the user."""
    text = " ".join(str(exc).split())
    known = re.search(r"Model type \S+ not supported", text)
    if known:
        return known[0]
    return re.split(r"\. | Reported cause| Traceback", text)[0][:120]


def rebuild_pending(config, store):
    return bool(rebuild_reason(config, store)) and store.get(
        "completed_rebuild"
    ) != rebuild_key(config, store)


def reason(
    config,
    p,
    *,
    paused=False,
    traffic_idle=0,
    last_completed=0,
    enough=False,
    now=None,
    immediate=False,
    threshold=None,
    reclaimable_bytes=0,
    adaptive=False,
    expected_seconds=0,
):
    if paused:
        return "Paused"
    if not config.model:
        return f"Run {NAME} init"
    if not p.available:
        return "Waiting for hardware probes"
    if not config.calibration_available:
        return "Waiting for calibration"
    if not enough:
        return "Waiting for experience"
    if p.power.constrained(expected_seconds):
        return "Waiting: battery low"
    if p.thermal is not None and p.thermal >= 2:
        return "Waiting for the machine to cool"
    if p.pressure == "critical":
        return f"Waiting for free memory: available={p.free_gb * 1e9:.0f} bytes, pressure={p.pressure}"
    # Measured peak is a planning estimate; parking is included before deciding
    # to defer. The live guard, not model size or a RAM floor, is authoritative.
    if (
        not adaptive
        and p.free_gb * 1e9 + reclaimable_bytes < config.calibrated_peak_gb * 1e9
    ):
        return f"Waiting for free memory: available={p.free_gb * 1e9:.0f} bytes, reclaimable={reclaimable_bytes} bytes, measured peak={config.calibrated_peak_gb * 1e9:.0f} bytes"
    threshold = config.idle_default_seconds if threshold is None else threshold
    # `songe sleep --now` skips only the idle wait; power, heat and memory
    # guards above still apply, and live traffic still interrupts.
    if traffic_idle < threshold and not immediate:
        return f"Waiting for {threshold:.0f} seconds without proxy traffic"
    return "Will sleep when idle"


class Scheduler:
    @property
    def reason(self):
        return self._reason

    @reason.setter
    def reason(self, value):
        # Each new state is one Flow line; a state that flaps back within ten
        # minutes is not repeated, so the Flow stays readable.
        previous = getattr(self, "_reason", None)
        self._reason = value
        logged = self.__dict__.setdefault("_reason_logged", {})
        store = self.__dict__.get("store")
        if value != previous and store is not None:
            now = time.monotonic()
            if now - logged.get(value, -1e9) > 600:
                logged[value] = now
                from .wording import flow_state

                store.event("state", flow_state(value))

    def __init__(self, config, store, upstream, platform=None):
        saved = store.get("interface_settings", {})
        if "schedule_default_seconds" in saved:
            config.idle_default_seconds = saved.pop("schedule_default_seconds")
            config.save(store.root)
            store.set("interface_settings", saved)
        self.config, self.store, self.upstream = config, store, upstream
        self.platform = platform or current()
        self.root = store.root
        self.unload_retry_at = 0.0
        self.returning = False
        self.reason = "Waiting for experience"
        self.last_traffic = time.monotonic()
        self.process = self.runner = self.caffeinate = self.guard_task = None
        self.guard_stop = threading.Event()
        self.memory_stop_sample = None
        self.memory_nights_before = []
        self.in_flight = {}
        self.stopping = False
        self.cancelled_at = None
        self.interrupted_by = None
        self.start_monotonic = 0
        self.lock = asyncio.Lock()
        self.current_version = store.version()
        self._experience_revision = None
        self._enough_cached = False
        self.state = "awake"
        self.traffic_revision = 0
        self.calibrating = False
        self.exporting = False
        self.last_probe = None
        self.idle_threshold_seconds = self.idle_wait()

    @property
    def pressure_history(self):
        """The current store's: calibration and the worker learn boundaries
        there too. A copy kept from start missed them, wrote over them, and
        stayed with the first store after select() scoped it to a model."""
        from .sleep.sizing import PressureHistory

        return PressureHistory(self.store.get("memory_pressure_history", {}))

    def unloads(self):
        return bool(self.store.get("unloads_model"))

    def idle_wait(self, gaps=None):
        """The wait shown is the one the last start needed: longer when it had
        to unload the agent's model."""
        if not self.unloads():
            return self.config.idle_default_seconds
        if gaps is None:
            gaps = gap_history(self.store.traffic_times(self.config.idle_history_limit))
        return park_wait(gaps, self.config)

    def observe_memory(self, sample):
        history = self.pressure_history
        history.observe(sample)
        self.store.set("memory_pressure_history", history.data)

    def acquire(self, binding):
        self.in_flight[binding] = self.in_flight.get(binding, 0) + 1

    def release(self, binding):
        self.in_flight[binding] = max(0, self.in_flight.get(binding, 1) - 1)

    def traffic(self):
        self.traffic_revision += 1
        self.last_traffic = time.monotonic()
        # A background review must not hold the model while the agent waits;
        # dropping its connection stops the generation upstream.
        for name in ("review_task", "reading_task"):
            task = getattr(self, name, None)
            if task is not None and not task.done():
                task.cancel()
        self.interrupt("agent activity")

    def response_finished(self, finish_reason):
        """An agent that asked for tools is still working: it will call back
        once they run, which can take minutes. Quiet time starts only after
        a final answer (or once that wait runs out)."""
        hold = AGENT_FOLLOWUP_SECONDS if finish_reason == "tool_calls" else 0
        self.last_traffic = max(self.last_traffic, time.monotonic() + hold)

    def interrupt(self, reason=None):
        if (
            self.process
            and self.process.returncode is None
            and self.cancelled_at is None
        ):
            self.cancelled_at = time.monotonic()
            self.interrupted_by = reason
            interrupt_group(self.process)

    def managed(self):
        with self.store.connect() as db:
            return {r[0] for r in db.execute("SELECT identifier FROM versions")}

    async def restore(self):
        parked = self.store.get("parked_models", [])
        if parked:
            await self.upstream.restore(parked, self.managed())
        version = self.store.version()
        if version:
            if hasattr(self.upstream, "ensure"):
                await self.upstream.ensure(version | {"serving": True})
            else:
                await self.upstream.load(version | {"serving": True})
        self.store.set("parked_models", [])

    async def wake(self):
        """Admission barrier: finish the worker and restore before forwarding."""
        async with self.lock:
            self.interrupt("agent activity")
            if self.process:
                try:
                    await asyncio.wait_for(
                        self.process.wait(), self.config.guard_grace_seconds
                    )
                except TimeoutError:
                    kill(self.process)
                    await self.process.wait()
                await self.finish()
            # Off the event loop: a slow or stuck database must not stop the
            # proxy from answering everything else.
            elif await asyncio.to_thread(self.store.get, "parked_models"):
                await self.restore()
            elif getattr(self.upstream, "lease_seconds", 0) and self.current_version:
                # External instances can expire during idle periods. Verify or
                # reload before forwarding the next request under their alias.
                await self.upstream.load(self.current_version)

    async def start(self):
        self.lease = (self.root / "serve.lock").open("a+")
        acquire_lock(self.lease)
        recovery = (self.root / "worker.lock").open("a+")
        try:
            # Recover only after a former worker relinquishes its lock.
            deadline = time.monotonic() + self.config.guard_grace_seconds * 2
            while True:
                try:
                    acquire_lock(recovery)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            "Previous worker is still stopping"
                        ) from None
                    await asyncio.sleep(self.config.guard_min_interval)
            for night in self.store.nights():
                if night["status"] == "running":
                    self.store.night(night["id"], status="interrupted")
            # A restore cut off by a restart is not running any more.
            self.store.set("restoring", None)
            for work_slice in self.store.slices():
                if work_slice["ended"] is None:
                    self.store.slice(
                        work_slice["id"],
                        phase=work_slice["phase"],
                        units_done=work_slice["units_done"],
                        interrupted_by="proxy restart",
                    )
        finally:
            recovery.close()
        try:
            await self.restore()
        except Exception as exc:
            # Forwarding and recording must survive a failed restore; learning
            # waits until the upstream can load the retained version again.
            logger.warning("restore_failed", exc_info=True)
            self.reason = f"Waiting for the model server: {exc}"
        self.runner = asyncio.create_task(self.loop())

    async def enough(self):
        configuration = training_fingerprint(self.config)
        base = base_fingerprint(self.config)
        if any(
            resumable(
                n, self.store, self.config, configuration=configuration, base=base
            )
            for n in self.store.nights()
            if n["status"] == "interrupted"
        ):
            return True
        with self.store.connect() as db:
            latest = db.execute("SELECT MAX(id) FROM exchanges").fetchone()[0]
            practiced = db.execute(
                "SELECT MAX(rowid), COUNT(*) FROM practice WHERE model_id=?",
                (self.store.model_id,),
            ).fetchone()
        revision = (
            configuration,
            latest,
            tuple(practiced),
            self.store.get("learning_mode"),
            self.store.get("experience_revision"),
            self.store.get("experience_watermark", 0),
            self.store.get("score_watermark", 0),
            self.store.get("rebuild_required", False),
            self.store.get("deep_rebuild_requested", False),
            self.store.get("completed_rebuild"),
            self.store.get("candidate_generation"),
        )
        if revision == self._experience_revision:
            return self._enough_cached

        def count():
            watermark = self.store.get("score_watermark", 0)
            examples, conversations = self.store.feedback_counts()
            if (
                examples >= self.config.min_train
                and conversations >= self.config.min_heldout + 1
            ):
                # Set tests aside from the collected data, then train on the rest.
                self.store.assign_splits(minimum=self.config.min_heldout)
            train, held, pending = self.store.experience_counts()
            pending = max(pending, self.store.pending_learnable())
            min_train, min_held, min_new = self.config.min_train, self.config.min_heldout, self.config.min_new
            if self.config.learning == "dreams":
                # Trains on kept practice; every moment with feedback, filled
                # by held-out practice, is a test; one of each is enough.
                from .practice.pool import (
                    dream_minimums,
                    heldout_units,
                    practice_steps,
                    untrained,
                )

                kept = practice_steps(self.store)
                examples, train, pending = len(kept), len(kept), len(untrained(self.store, kept))
                held = heldout_units(self.store)
                min_train, min_held, min_new = dream_minimums(self.config)
            return (
                examples >= min_train
                and train > 0
                and held >= min_held
                and (
                    pending >= min_new
                    or (pending > 0 and watermark > 0)
                    or rebuild_pending(self.config, self.store)
                )
            )

        self._enough_cached = await asyncio.to_thread(count)
        self._experience_revision = revision
        return self._enough_cached

    def memory_wait(self, config, store, before, reclaimable, *, retrying=False):
        """Why training would not fit even with the agent's model unloaded,
        or None when it would."""
        from .sleep.calibration_state import smallest_pending
        from .sleep.sizing import choose_size, free_needed, waiting_reason

        calibration = store.get("calibration") or {}
        shortest = smallest_pending(config, calibration, store)
        history = self.pressure_history.data

        def fits(size):
            return bool(size) and (
                shortest is None
                or size.get("max_teacher_sequence", float("inf")) >= shortest
            )

        size = choose_size(config, calibration, after_unload(before, reclaimable), history)
        if not retrying and fits(size):
            return None
        if size and not fits(size):
            # Training would fit, but no teacher length the examples need
            # was measured: measure it rather than wait for memory (#140).
            from .sleep.calibration_state import teacher_to_ask

            asked = teacher_to_ask(config, calibration, store, size)
            if asked is not None:
                store.set("calibration", asked)
                self.last_fit = None
                return "Measuring memory for longer conversations"
        needed = free_needed(config, calibration, before, history, fits)
        self.last_fit = (before, needed)  # for note_fit_wait
        return waiting_reason(before, history, needed)

    def note_fit_wait(self, sample, needed):
        """A pressed Deep Dream that waits for memory before training starts
        (live: "needs 13.2 GB, 9.6 GB free", nothing on the page): kept for
        the page, and ended with a note after FIT_WAIT_SECONDS."""
        if not self.store.get("requested_sleep"):
            return
        now = time.time()
        wait = (self.store.get("fit_wait") or {"since": now}) | {
            "available": sample.available, "needed": needed, "total": sample.total, "at": now}
        if now - wait["since"] > FIT_WAIT_SECONDS:
            from .practice.store import counts

            self.store.set("fit_wait", None)
            self.store.set("requested_sleep", False)
            self.store.set("requested_model", None)
            self.store.set("dream_stopped", {"kind": "deep", "at": now, "reason": "training",
                                             "detail": "Waiting for memory", "kept": counts(self.store)["kept"]})
            return
        self.store.set("fit_wait", wait)

    async def loop(self):
        while not self.stopping:
            try:
                await self.tick()
            except Exception:
                logger.exception("scheduler_tick_failed")
                self.reason = "Sleep could not complete; current version preserved"
                if self.store.get("parked_models") and not self.process:
                    try:
                        await self.restore()
                    except Exception:
                        # The parking journal stays; the next tick retries.
                        logger.exception("scheduler_restore_failed")
            if self.upgraded():
                # launchd / systemd start the service again, on the new code.
                logger.info("restarting_on_upgraded_code")
                self.upgrade_exit = True
                self.shutdown()
                return
            # Every second only while a worker runs or an update waits to be
            # applied; an idle service checks less often (each check probes
            # memory and the model server).
            await asyncio.sleep(
                self.config.guard_max_interval
                * (
                    1
                    if self.process or getattr(self, "state", None) == "ready"
                    else IDLE_TICKS
                )
            )

    UPGRADE_CHECK_SECONDS = 30.0

    def upgraded(self):
        """Songe was upgraded under this running service (brew upgrade, pip
        install -U): restart onto the new code in a quiet moment, without the
        user running any command. The new files must look the same on two
        checks, so a half-finished install is never started."""
        if not getattr(self, "shutdown", None):
            return False
        now = time.monotonic()
        if now - getattr(self, "upgrade_checked", 0.0) < self.UPGRADE_CHECK_SECONDS:
            return False
        self.upgrade_checked = now
        from .upgrade import RUNNING_INSTALL, install_token

        token, seen = install_token(), getattr(self, "upgrade_seen", None)
        self.upgrade_seen = token
        return (
            token != RUNNING_INSTALL
            and token == seen
            # Only when the service can come back: its interpreter still exists.
            and Path(sys.executable).exists()
            and not self.process
            and not any(self.in_flight.values())
            and not self.lock.locked()
            # An agent between two tool calls must not find the proxy gone.
            and time.monotonic() - self.last_traffic
            >= max(60.0, self.idle_threshold_seconds)
        )

    def note_progress(self):
        """A run that did nothing (it found no room, or nothing to do) is not
        started again at once: each start loads the model, and may unload the
        agent's. The wait doubles until a run does work or new conversations
        arrive; being interrupted by a request does not count."""
        if self.calibrating or self.cancelled_at is not None:
            return
        last = self.store.last_slice() or {}
        if last.get("units_done"):
            self.store.set("empty_runs", None)
            return
        if last.get("interrupted_by") == "slice deadline":
            # It was working and ran out of time, not idle: the next slice
            # is sized longer (measured_step_seconds), not delayed.
            return
        count = (self.store.get("empty_runs") or {}).get("count", 0) + 1
        self.store.set(
            "empty_runs",
            {
                "count": count,
                "at": time.time(),
                "experience": self.store.get("experience_revision"),
            },
        )

    def empty_run_wait(self):
        empty = self.store.get("empty_runs") or {}
        if not empty or empty.get("experience") != self.store.get(
            "experience_revision"
        ):
            return 0
        return max(
            0,
            min(retry_wait(empty["count"]), EMPTY_RUN_WAIT_MAX)
            - (time.time() - empty["at"]),
        )

    async def finish(self):
        await self.process.wait()
        # Errors back off and retry forever; memory stops and interruptions
        # are waits, not errors; a clean run clears the count.
        code = getattr(self.process, "returncode", None)
        if code == 0:
            self.store.set("failures", None)
            self.store.event("done", "finished")
        elif code == MEMORY_EXIT or self.memory_stop_sample is not None:
            # Errors count only back to back; a run that got this far clears them.
            self.store.set("failures", None)
            self.store.event("stop", "memory is short · retries automatically")
        elif self.cancelled_at is not None:
            self.store.event(
                "stop", f"paused · {self.interrupted_by or 'the model is in use'}"
            )
        elif code is not None:
            from .wording import duration

            count = (self.store.get("failures") or {}).get("count", 0) + 1
            self.store.set("failures", {"count": count, "at": time.time()})
            why = exit_reason(code, self.store.root / WORKER_ERR)
            # The worker's own record of a Python error is the more precise;
            # otherwise the exit itself says why.
            if not self.store.get("child_error"):
                self.store.set("child_error", f"Learning stopped: {why}")
            logger.warning("worker_failed %s", why)
            self.store.event(
                "error",
                f"failed ({why.splitlines()[0][:120]}) · retry in {duration(retry_wait(count))} · songe doctor",
            )
        err = getattr(self, "worker_err", None)
        if err is not None:
            err.close()
            self.worker_err = None
        self.note_progress()
        self.guard_stop.set()
        if self.guard_task:
            await self.guard_task
            self.guard_task = None
        if self.memory_stop_sample is not None:
            record_memory_stop(
                self.store,
                self.memory_stop_sample,
                calibration=self.calibrating,
                nights_before=self.memory_nights_before,
            )
            self.memory_stop_sample = None
            if not self.calibrating and not self.exporting:
                from .sleep.sizing import learn_peak

                if learn_peak(self.store, self.store.get("last_memory_stop")):
                    self.store.event(
                        "stop",
                        "used more memory than measured · next run is sized for it",
                    )
        self.process = None
        self.exporting = False
        if self.calibrating:
            from .config import Config
            from .sleep.calibrate import collect_calibration, derive_keys

            collect_calibration(self.root, self.store)
            calibrated = (
                Config.for_model(self.root, self.config.model_key)
                if self.config.model_key
                else Config.load(self.root)
            )
            for key in derive_keys():
                setattr(self.config, key, getattr(calibrated, key))
            self.calibrating = False
        if self.caffeinate:
            self.caffeinate.close()
            self.caffeinate = None
        for night in self.store.nights():
            if night["status"] == "running":
                record = night["record"]
                if self.interrupted_by:
                    record["reason"] = self.interrupted_by
                self.store.night(
                    night["id"],
                    status="interrupted" if self.cancelled_at else "failed",
                    record=record,
                )
        for work_slice in self.store.slices():
            if work_slice["ended"] is None:
                self.store.slice(
                    work_slice["id"],
                    phase=work_slice["phase"],
                    units_done=work_slice["units_done"],
                    interrupted_by=self.interrupted_by
                    or ("worker interrupted" if self.cancelled_at else "worker exit"),
                )
        await self.restore()
        self.cancelled_at = None
        self.interrupted_by = None
        self.state = "awake"
        if self.store.get("disk_wait") or self.store.get("child_error"):
            self.reason = self.store.get("disk_wait") or self.store.get("child_error")

    async def tick(self):
        if self.returning:
            return  # A version is being rebuilt; no other work beside it.
        p = await asyncio.to_thread(probe, self.platform)
        self.last_probe = p
        if self.process:
            if self.process.returncode is None:
                from .artifacts.storage import swap_space

                if p.power.constrained():
                    self.reason = "Stopping: battery low; saving checkpoint"
                    self.interrupt("battery low")
                if self.store.get("paused", False):
                    self.interrupt("paused")
                elif p.thermal is not None and p.thermal >= 2:
                    self.interrupt("thermal pressure")
                elif not p.available:
                    # Probes failing is a fault, not a wait (songe doctor).
                    self.interrupt("hardware probe failed")
                elif self.cancelled_at is None and (
                    low := await asyncio.to_thread(swap_space, self.platform, running=True)
                ):
                    # Swap was filling the disk (#68): stop, saving progress;
                    # the disk wait shows until there is room to start again.
                    self.store.set("disk_wait", low)
                    self.interrupt("the disk is almost full")
                if (
                    self.cancelled_at
                    and time.monotonic() - self.cancelled_at
                    > self.config.guard_grace_seconds
                ):
                    kill(self.process)
                return
            async with self.lock:
                if self.process:
                    await self.finish()
        for night in self.store.nights():
            if night["status"] == "ready":
                self.state = "ready"
                await self.accept(night)
        from .sleep.cleanup import collect_work

        await asyncio.to_thread(collect_work, self.store, self.config)
        gaps = gap_history(self.store.traffic_times(self.config.idle_history_limit))
        seconds = slice_duration(
            self.config,
            self.store.get("calibration"),
            gaps,
            time.monotonic() - self.last_traffic,
            measured=measured_step_seconds(self.store),
        )
        threshold = idle_threshold(gaps, seconds, self.config)
        self.idle_threshold_seconds = self.idle_wait(gaps)
        requested = self.store.get("requested_sleep", False)
        try:
            reclaimable = (
                await self.upstream.loaded_bytes()
                if hasattr(self.upstream, "loaded_bytes")
                else 0
            )
        except Exception:
            # The model server is closed: nothing of ours is loaded there.
            reclaimable = 0
        calibration = self.store.get("calibration") or {}
        from .sleep.calibration_state import needs_growth

        pending = await asyncio.to_thread(pending_export, self.config, self.store)
        exporting = pending is not None
        # Also when a pressed Deep Dream's training run found it needed
        # growth (see ModelScheduler.finish): measure, then train.
        # Off the event loop: the proxy and the status API answer meanwhile.
        growing = not exporting and (
            await asyncio.to_thread(needs_growth, self.config, self.store)
            or bool(self.store.get("growth_requested"))
        )
        measured = any("host_peak_bytes" in r for r in calibration.get("rows", []))
        upgrade_calibration = not exporting and (
            (self.config.calibration_available and not measured)
            or (bool(self.config.model_key) and not self.config.calibration_available)
            or growing
        )
        needs_sizing = not upgrade_calibration and not exporting
        self.reason = reason(
            replace(self.config, calibration_available=True)
            if not needs_sizing
            else self.config,
            p,
            paused=self.store.get("paused", False),
            traffic_idle=time.monotonic() - self.last_traffic,
            enough=not needs_sizing or await self.enough(),
            immediate=requested,
            threshold=threshold,
            reclaimable_bytes=reclaimable,
            adaptive=measured or not needs_sizing,
            expected_seconds=seconds,
        )
        if (
            self.reason != "Will sleep when idle"
            or any(self.in_flight.values())
            or self.state == "ready"
        ):
            return
        # A short pause between slices, never an overnight cooldown: an
        # extrapolated work unit for a long window can reach hours (live:
        # 7.8 h for a 0.8B model), which stalled measuring.
        unit = max(
            work_unit_seconds(self.config, calibration),
            measured_step_seconds(self.store),
        )
        if not requested and time.time() - self.store.get("last_attempt", 0) < min(
            seconds, unit, SLICE_PAUSE_MAX
        ):
            self.reason = "Waiting before the next work slice"
            return
        if not requested and not upgrade_calibration and self.empty_run_wait():
            self.reason = "Waiting before the next work slice"
            return
        failures = self.store.get("failures") or {}
        if (
            failures
            and not requested
            and time.time() - failures["at"] < retry_wait(failures["count"])
        ):
            self.reason = "Waiting to retry after an error"
            return
        from .artifacts.storage import StorageWait, admit_storage, collect_idle_scratch

        revision = self.traffic_revision
        try:
            await asyncio.to_thread(collect_idle_scratch, self.store)
            await asyncio.to_thread(
                admit_storage,
                self.config,
                self.store,
                self.platform,
                calibration=upgrade_calibration,
                pending_export=pending["record"]["pending_export"] if pending else None,
            )
        except StorageWait as exc:
            self.reason = str(exc)
            return
        self.store.set("child_error", None)
        async with self.lock:
            if any(self.in_flight.values()) or revision != self.traffic_revision:
                return
            try:
                from .memory import TrainingPlatform
                from .sleep.sizing import choose_size

                # Keep the served model loaded when training fits at full size
                # beside it; unloading costs the next request a reload.
                roomy = False
                if needs_sizing:
                    before = await asyncio.to_thread(
                        snapshot, TrainingPlatform(self.platform, self.config)
                    )
                    fit = choose_size(
                        self.config, calibration, before, self.pressure_history.data
                    )
                    roomy = bool(
                        fit and fit["max_sequence"] >= self.config.max_sequence
                    )
                unloads = hasattr(self.upstream, "park") and not roomy
                if self.store.get("unloads_model") != unloads:
                    self.store.set("unloads_model", unloads)
                if unloads and needs_sizing:
                    # Unload the agent's model only when that makes training
                    # fit. Otherwise every tick would unload it, find no room
                    # and load it again.
                    wait = self.memory_wait(
                        self.config,
                        self.store,
                        before,
                        reclaimable,
                        retrying=time.monotonic() < self.unload_retry_at,
                    )
                    if wait:
                        self.reason = wait
                        # The wait a pressed Deep Dream is in before training
                        # (live: the page showed nothing for it).
                        if self.last_fit:
                            self.note_fit_wait(*self.last_fit)
                        return
                if unloads:
                    wait = park_wait(
                        gap_history(
                            self.store.traffic_times(self.config.idle_history_limit)
                        ),
                        self.config,
                    )
                    if not requested and time.monotonic() - self.last_traffic < wait:
                        self.idle_threshold_seconds = wait
                        self.reason = "Will sleep when idle"
                        return
                    await self.upstream.park(
                        on_snapshot=lambda parked: self.store.set(
                            "parked_models", parked
                        )
                    )
                if revision != self.traffic_revision:
                    await self.restore()
                    return
                # Measure again after releasing only our own serving instances.
                from .memory import TrainingPlatform

                training_platform = TrainingPlatform(self.platform, self.config)
                sample = await asyncio.to_thread(snapshot, training_platform)
                from .sleep.sizing import (
                    choose_size,
                    free_needed,
                    gigabytes,
                    waiting_reason,
                )

                self.observe_memory(sample)
                sizing = (
                    choose_size(
                        self.config, calibration, sample, self.pressure_history.data
                    )
                    if needs_sizing
                    else None
                )
                if needs_sizing and sizing is None:
                    needed = free_needed(
                        self.config, calibration, sample, self.pressure_history.data
                    )
                    self.reason = waiting_reason(sample, self.pressure_history.data, needed)
                    self.note_fit_wait(sample, needed)
                    # The estimate was optimistic: do not unload again soon.
                    self.unload_retry_at = time.monotonic() + RETRY_FIRST_SECONDS
                    await self.restore()
                    return
                if sizing:
                    from .sleep.calibration_state import smallest_pending

                    shortest = smallest_pending(self.config, calibration, self.store)
                    if (
                        shortest is not None
                        and sizing.get("max_teacher_sequence", float("inf")) < shortest
                    ):
                        from .sleep.calibration_state import teacher_to_ask
                        from .sleep.sizing import teacher_card_shortfall

                        asked = teacher_to_ask(
                            self.config,
                            self.store.get("calibration") or {},
                            self.store,
                            sizing,
                        )
                        if asked is not None:
                            # No teacher of the length these examples need was
                            # measured (the ladder jumped over it): measure it
                            # instead of deferring them on every slice (#140).
                            self.store.set("calibration", asked)
                            self.reason = "Measuring memory for longer conversations"
                            await self.restore()
                            return
                        needed = free_needed(
                            self.config,
                            calibration,
                            sample,
                            self.pressure_history.data,
                            lambda s: s.get("max_teacher_sequence", 0) >= shortest,
                        )
                        card = teacher_card_shortfall(calibration, sample, shortest)
                        self.reason = waiting_reason(
                            sample,
                            self.pressure_history.data,
                            needed,
                            what="intact training samples deferred"
                            + (
                                f" · the graphics card needs {gigabytes(card)} more free"
                                if card
                                else ""
                            ),
                        )
                        # Measured and still too big beside what else runs: a
                        # pressed Dream ends with a note instead of retrying
                        # without end.
                        self.note_fit_wait(sample, needed)
                        self.unload_retry_at = time.monotonic() + RETRY_FIRST_SECONDS
                        await self.restore()
                        return
                if sizing is None:
                    host, device = 0, 0
                else:
                    self.store.set("last_slice_size", sizing)
                    host, device = sizing["host_bytes"], sizing["device_bytes"]
                margin = self.pressure_history.margin_for(sample)
                policy = GuardPolicy(
                    self.config.guard_min_interval,
                    self.config.guard_max_interval,
                    self.config.guard_grace_seconds,
                )
                self.memory_nights_before = self.store.nights()
                sample = await asyncio.to_thread(
                    capture_baseline,
                    training_platform,
                    policy=policy,
                    required_bytes=host,
                    required_device_bytes=device,
                    margin_bytes=margin,
                    on_sample=self.observe_memory,
                )
                if any(self.in_flight.values()) or revision != self.traffic_revision:
                    await self.restore()
                    return
                self.store.set("last_memory_baseline", asdict(sample))
                self.store.set("requested_sleep", False)
                self.store.set("requested_model", None)
                self.store.set("fit_wait", None)
                self.store.set("last_attempt", time.time())
                self.cancelled_at = None
                self.interrupted_by = None
                argv = [
                    sys.executable,
                    "-m",
                    "songe.sleep.calibrate"
                    if upgrade_calibration
                    else "songe.sleep.worker",
                    "--data-dir",
                    str(self.root),
                    "--child",
                ]
                if self.config.model_key:
                    argv += ["--model-key", self.config.model_key]
                if not upgrade_calibration:
                    argv += ["--slice-seconds", str(seconds)]
                if exporting:
                    argv += ["--export-only"]
                if sizing:
                    argv += ["--sizing", json.dumps(sizing)]
                self.calibrating = upgrade_calibration
                self.exporting = exporting
                self.calibration_started = time.time()
                self.store.event(
                    "start",
                    "measuring what this machine can train"
                    if upgrade_calibration
                    else "learning cycle in an idle gap",
                )
                # The child's stderr is kept: a crash outside Python's
                # logging (a CUDA error, a kill) left no trace at all (live,
                # Inkling on Modal: "failed" with no reason anywhere).
                self.worker_err = open(self.store.root / WORKER_ERR, "wb")  # noqa: SIM115
                self.process = await asyncio.create_subprocess_exec(
                    *owned_command(argv),
                    **process_options(),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=self.worker_err,
                )
                self.guard_stop = threading.Event()
                loop = asyncio.get_running_loop()
                process = self.process

                def mark_memory_stop():
                    if self.cancelled_at is None:
                        self.cancelled_at = time.monotonic()

                def memory_stop(sample, critical=False):
                    self.memory_stop_sample = sample
                    self.interrupted_by = self.interrupted_by or "memory guard"
                    if critical:
                        kill(process)
                    else:
                        interrupt_group(process)
                    loop.call_soon_threadsafe(mark_memory_stop)
                    try:
                        self.store.set(
                            "last_memory_stop",
                            asdict(sample) if sample else {"pressure": "unknown"},
                        )
                    except Exception:
                        logger.exception("memory_alarm_persistence_failed")

                self.guard_task = asyncio.create_task(
                    asyncio.to_thread(
                        guard,
                        process,
                        training_platform,
                        lambda sample: memory_stop(sample),
                        lambda sample: memory_stop(sample, critical=True),
                        policy=policy,
                        baseline=sample,
                        margin_bytes=margin,
                        predicted_peak_bytes=host or None,
                        predicted_device_bytes=device or None,
                        prediction_error_bytes=sizing["prediction_error_bytes"]
                        if sizing
                        else 0,
                        device_prediction_error_bytes=sizing[
                            "device_prediction_error_bytes"
                        ]
                        if sizing
                        else 0,
                        stop=self.guard_stop,
                        on_sample=self.observe_memory,
                        # Measuring what fits: real limits stop it, not a
                        # projection of the weights loading.
                        projection=not self.calibrating,
                    )
                )
                self.caffeinate = self.platform.keep_awake(self.process.pid)
                self.state = "settling"
                self.reason = (
                    "Measuring memory for learning"
                    if self.calibrating
                    else "Working in an idle gap"
                )
            except BaseException as exc:
                if isinstance(exc, MemoryPressure):
                    record_memory_stop(
                        self.store,
                        exc,
                        calibration=upgrade_calibration,
                        nights_before=self.memory_nights_before,
                    )
                if self.process:
                    kill(self.process)
                    await self.finish()
                else:
                    await self.restore()
                raise

    async def accept(self, night):
        from .sleep.swap import collect, install

        async with self.lock:
            record = night["record"]
            candidate, expected = record["candidate"], record["expected"]
            if self.store.get("current") != candidate["id"] and (
                record.get("generation") != self.store.get("candidate_generation")
                or expected != self.store.get("current")
                or record.get("configuration") != training_fingerprint(self.config)
                or record.get("base_fingerprint") != base_fingerprint(self.config)
                or record.get("environment")
                != self.store.get("environment_fingerprint")
            ):
                self.store.night(night["id"], status="discarded")
                self.state = "awake"
                return
            if self.store.get("current") != candidate["id"]:
                durations = self.store.get("swap_durations", [])
                gaps = gap_history(
                    self.store.traffic_times(self.config.idle_history_limit)
                )
                seconds = slice_duration(self.config, self.store.get("calibration"))
                # A typical switch, not the slowest one ever seen: one cold load
                # must not keep every later update waiting for a long gap.
                duration = max(
                    sorted(durations)[len(durations) // 2] if durations else 0,
                    idle_threshold(gaps, seconds, self.config),
                )
                if (
                    self.store.get("paused", False)
                    or any(self.in_flight.values())
                    or time.monotonic() - self.last_traffic < duration
                ):
                    return
                if time.time() < record.get("swap_retry_at", 0):
                    return
                revision = self.traffic_revision
                start = time.monotonic()

                def load():
                    return install(
                        self.store,
                        self.upstream,
                        self.config,
                        candidate,
                        expected,
                        generation=record.get("generation"),
                        can_commit=lambda: (
                            not any(self.in_flight.values())
                            and self.traffic_revision == revision
                            and not self.store.get("paused", False)
                            and record.get("generation")
                            == self.store.get("candidate_generation")
                            and record.get("configuration")
                            == training_fingerprint(self.config)
                        ),
                    )

                import httpx

                from .memory import MemoryPressure

                waits = (MemoryPressure, MemoryError, httpx.TransportError)
                try:
                    try:
                        installed = await load()
                    except MemoryPressure:
                        if not hasattr(self.upstream, "park"):
                            raise
                        # No room beside the version in use: the agent is
                        # quiet, so unload that one and load the update alone.
                        await self.upstream.park(
                            on_snapshot=lambda parked: self.store.set(
                                "parked_models", parked
                            )
                        )
                        try:
                            installed = await load()
                        except BaseException:
                            await self.restore()
                            raise
                        if installed:
                            self.store.set("parked_models", [])
                        else:
                            await self.restore()
                except Exception as exc:
                    # Finished training is never thrown away because memory is
                    # short or the model server is closed: it waits and tries
                    # again. Only a version the server refuses three times,
                    # spaced apart, is rejected, and its data counts as used.
                    logger.exception("swap_failed")
                    waiting = isinstance(exc, waits)
                    key = "swap_waits" if waiting else "swap_failures"
                    record[key] = record.get(key, 0) + 1
                    record["swap_retry_at"] = time.time() + retry_wait(record[key])
                    self.store.event(
                        "stop" if waiting else "error",
                        (
                            "update is ready · waits for memory or the model server"
                            if waiting
                            else f"update could not load · {exc}"
                        )[:200],
                    )
                    if not waiting and record["swap_failures"] >= 3:
                        record["reason"] = (
                            "Update could not be loaded by the model server"
                        )
                        self.store.night(night["id"], status="rejected", record=record)
                        if self.store.get("learning_mode", "hindsight") == "dreams":
                            # Not adopted, so not practised: under Dreams the
                            # watermark counts practice scenes, and moving it
                            # hid this Dream's scenes from every later one.
                            from .practice.pool import keep_for_retry

                            keep_for_retry(self.store, record.get("practice_revisions") or {})
                        else:
                            self.store.set("experience_watermark", record["watermark"])
                    else:
                        self.store.night(night["id"], record=record)
                    self.state = "awake"
                    return
                if not installed:
                    return
                self.store.set(
                    "swap_durations",
                    (durations + [time.monotonic() - start])[
                        -self.config.idle_history_limit :
                    ],
                )
            self.current_version = self.store.version()
            self.store.night(night["id"], status="accepted", phase="swap", progress=1)
            self.store.set("last_completed", time.time())
            self.store.set("experience_watermark", record["watermark"])
            self.store.set("practice_revisions", record.get("practice_revisions", {}))
            # Lessons now in the weights leave the prompt (Quick Dream).
            from . import quick

            trained = sorted(
                set(self.store.get("trained_signals") or [])
                | set(record.get("trained_signals") or [])
            )
            self.store.set("trained_signals", trained)
            quick.absorb(self.store, trained)
            self.store.set(
                "trained_base",
                record.get("base_fingerprint", base_fingerprint(self.config)),
            )
            self.store.set("cumulative_drift", record.get("general_drift", 0))
            if record.get("rebuild"):
                self.store.set("rebuilt_at", time.time())
                self.store.set("rebuild_required", False)
                self.store.set("deep_rebuild_requested", False)
            self.state = "awake"
            await collect(self.store, self.upstream, self.in_flight)

    async def return_to(self, night, model=None):
        from .sleep.retention import restore_artifact
        from .sleep.swap import collect, install

        await self.wake()
        if self.returning:
            raise ValueError("A version is already being restored")
        store, config = self.store, self.config
        previous = next((v for v in store.versions() if v["night"] == night), None)
        if not previous:
            raise ValueError("Version is not retained")
        self.returning = True
        try:
            # Rebuilding an older version takes minutes: it runs beside
            # serving, so the agent keeps working. Only the switch itself
            # waits for a gap between requests.
            await asyncio.to_thread(
                restore_artifact, store, config, previous, self.platform, lambda: None
            )
            deadline = time.monotonic() + RETURN_GAP_SECONDS
            while any(self.in_flight.values()) and time.monotonic() < deadline:
                await asyncio.sleep(0.2)
            async with self.lock:
                if any(self.in_flight.values()):
                    raise ValueError("Waiting for requests to finish")
                revision = self.traffic_revision
                installed = await install(
                    store,
                    self.upstream,
                    config,
                    previous,
                    store.get("current"),
                    can_commit=lambda: (
                        not any(self.in_flight.values())
                        and revision == self.traffic_revision
                    ),
                )
                if not installed or store.get("current") != previous["id"]:
                    raise RuntimeError(
                        "Version switch deferred because requests arrived; current version preserved"
                    )
                store.invalidate_pending()
                self.current_version = store.version()
                await collect(store, self.upstream, self.in_flight)
        finally:
            self.returning = False

    async def close(self):
        self.stopping = True
        if self.runner:
            self.runner.cancel()
            await asyncio.gather(self.runner, return_exceptions=True)
        await self.wake()
        await self.upstream.close()
        if hasattr(self, "lease"):
            self.lease.close()


MEMORY_TRIES = 3     # memory stops of a pressed Deep Dream before its note
FIT_WAIT_SECONDS = 1800  # a pressed Deep Dream waits this long for memory to train
CAPACITY_EDGES = 3   # measuring runs that reach the machine's edge before its note
MEMORY_BACKOFF = 60  # seconds before trying again, times the tries so far


def heldout_short(config, store):
    """Under Dreams: how many more held-out conversations the gate needs
    before a Deep Dream's result can be checked (0 when it has enough)."""
    if getattr(config, "learning", "") != "dreams":
        return 0
    from .practice.pool import ensure_heldout, heldout_units, test_minimum

    # Any recorded moment with feedback is enough (user decision); held-out
    # practice lessons fill in, more held out on demand.
    ensure_heldout(store, config.min_heldout)
    return max(0, test_minimum(config) - heldout_units(store))


class ModelScheduler(Scheduler):
    """One guarded worker and admission lock, with a separate learner per model."""

    async def start(self):
        await super().start()
        from .models import registry
        from .reading import store as reading

        # Once per start: what was read from sessions that no longer exist
        # (merged into another, or found to be an agent's own call) goes.
        for row in registry.entries(self.installation_store):
            try:
                learner = self.installation_store.scoped(row["id"])
                self.recover_pressed(row["id"], learner)
                self.clear_stale_waits(row["id"], learner)
                vanished = await asyncio.to_thread(reading.forget_vanished, learner)
                if vanished:
                    logger.info("forgot_vanished_sessions %s of %s", vanished, row["id"])
                copied = await asyncio.to_thread(reading.reread_copied, learner)
                if copied:
                    logger.info("reread_copied %s sessions of %s", copied, row["id"])
                queued = await asyncio.to_thread(reading.reread_unreadable, learner)
                if queued:
                    logger.info("reread_unreadable %s sessions of %s", queued, row["id"])
                placed = await asyncio.to_thread(reading.place_rules, learner)
                if placed:
                    logger.info("placed_rules %s of %s", placed, row["id"])
            except Exception:
                logger.warning("forget_vanished_failed", exc_info=True)

    def __init__(self, config, store, upstream, platform=None):
        from .models import registry

        registry.migrate(config, store)
        self.installation_config, self.installation_store = config, store
        self.active_model = None
        self.serving_model = None
        self.load_refused = {}
        self.serving_binding = None
        self.answered_running = False
        # model: {"version", "used"} for a model Songe loaded for a press
        # (load_for_press); "used" once the agent asks for it (ensure_model).
        self.press_loads = {}
        self.pressing = set()  # models whose Quick Dream press is getting ready
        self.release_tasks = {}
        super().__init__(config, store, upstream, platform)

    def unloads(self):
        # Between starts no model is selected: any enabled model's last start.
        from .models import registry

        if self.active_model:
            return super().unloads()
        return any(
            self.installation_store.scoped(row["id"]).get("unloads_model")
            for row in registry.entries(self.installation_store)
            if row["enabled"]
        )

    def select(self, model):
        from .models.registry import model_config

        self.active_model = model
        self.store = self.installation_store.scoped(model)
        self.config = model_config(
            self.installation_config, self.installation_store, model
        )
        self.current_version = self.store.version()
        self._experience_revision = None
        if hasattr(self.upstream, "supervisor"):
            self.upstream.supervisor.config = self.config

    async def ensure_model(self, model, version):
        """Renew the selected identity; serialize global servers across bindings."""
        held = self.press_loads.get(model)
        if held:
            # The agent asked for a model Songe loaded for a press: it is the
            # user's now, never released by Songe (release_press_load). While
            # the press still loads it, the agent waits for that load rather
            # than start a second one (LM Studio would drop the press's
            # pending claim and load another instance beside it).
            held["used"] = True
            await held["loading"].wait()
        global_binding = self.installation_config.upstream in {"llamacpp", "mlx"}
        while True:
            async with self.lock:
                busy = global_binding and any(
                    count
                    for key, count in self.in_flight.items()
                    if key not in {"admission", version["id"]}
                )
                if not busy:
                    refused = self.load_refused.get(version["id"])
                    if refused and time.monotonic() - refused[0] < LOAD_RETRY_SECONDS:
                        raise LoadRefused(refused[1])
                    if hasattr(self.upstream, "supervisor"):
                        self.upstream.supervisor.config = self.config_for(model)
                    try:
                        if hasattr(self.upstream, "ensure"):
                            await self.upstream.ensure(version | {"serving": True})
                        else:
                            await self.upstream.load(version | {"serving": True})
                    except Exception as exc:
                        if type(exc).__name__ == "LMStudioServerError":
                            # The server tried and refused these weights. An
                            # agent retries every few seconds; loading again
                            # each time only keeps the server busy failing.
                            from .models.library import model_name

                            reason = refusal_reason(exc)
                            self.load_refused[version["id"]] = (
                                time.monotonic(),
                                f"the model server could not load {model_name(model)}"
                                f"{' (' + reason + ')' if reason else ''}"
                                " · choose another model in your agent",
                            )
                            raise LoadRefused(
                                self.load_refused[version["id"]][1]
                            ) from exc
                        raise
                    self.load_refused.pop(version["id"], None)
                    self.serving_model, self.serving_binding = model, version["id"]
                    return
            await asyncio.sleep(self.installation_config.guard_min_interval)

    def can_rebuild(self, model):
        """A rebuild is work only with enough left to learn from; until then
        the model waits for conversations like any other."""
        config, store = self.config_for(model), self.installation_store.scoped(model)
        return (
            rebuild_pending(config, store)
            and store.feedback_counts()[0] >= config.min_train
        )

    def candidates(self, excluded=(), asking=None):
        """Enabled models with work to do now. asking: a model Dream was just
        pressed for (counts as requested even with auto-dream off)."""
        from .models.registry import summaries
        from .sleep.calibration_state import needs_growth

        asked = asking or self.installation_store.get("requested_model")

        def may_start(m):
            # Auto-dream off: only a Dream the user asked for, or one already
            # under way (it finishes in later idle gaps).
            return (
                m.get("auto_dream", True)
                or m["id"] == asked
                or any(
                    n["status"] in {"interrupted", "running", "ready"}
                    for n in self.installation_store.scoped(m["id"]).nights()
                )
            )

        return [
            m
            for m in summaries(self.installation_config, self.installation_store)
            if m["id"] not in excluded
            and m["enabled"]
            and m["support"] != "unsupported"
            and may_start(m)
            and (
                m["pending"]
                or m["support"] == "pending"
                or (
                    needs_growth(
                        self.config_for(m["id"]), self.installation_store.scoped(m["id"])
                    )
                    # Measuring for training that could not be checked waits.
                    and not heldout_short(
                        self.config_for(m["id"]), self.installation_store.scoped(m["id"])
                    )
                )
                or any(
                    n["status"] == "interrupted"
                    for n in self.installation_store.scoped(m["id"]).nights()
                )
                or self.can_rebuild(m["id"])
            )
        ]

    def choose_model(self, excluded=()):
        candidates = self.candidates(excluded)
        # A model the user asked to learn now goes first.
        asked = self.installation_store.get("requested_model")
        if any(m["id"] == asked for m in candidates):
            return asked
        return (
            min(candidates, key=lambda m: (-m["pending"], m["last_trained"], m["id"]))[
                "id"
            ]
            if candidates
            else None
        )

    async def dream(self, model, kind="deep"):
        """Manual Dream. Quick: read the newest conversations (within
        quick_read_seconds) and update the prompt lessons. Deep: read
        everything, write practice, and start training too when everything it
        waits on is met. Returns whether anything started and one line saying
        what, or what is missing."""
        from .wording import dream_message, quick_message

        blocked = self.dream_blocked(model)
        if blocked:
            return {"started": False, "message": dream_message(blocked=blocked)}
        if kind == "deep":
            learner = self.installation_store.scoped(model)
            # A memory wait counts only through its pending request (live: a
            # leftover wait alone blocked every press).
            # A request counts only while armed (requested_sleep): after an
            # interruption it stays named but waits for a long quiet period
            # with nothing running, and a press re-arms it below (live,
            # Inkling on Modal: every press answered "already running").
            running = (
                (self.process and self.active_model == model)
                or (
                    self.installation_store.get("requested_model") == model
                    and self.installation_store.get("requested_sleep")
                )
                or any(n["status"] in {"running", "interrupted", "ready"} for n in learner.nights())
            )
            if running:
                # One Deep Dream at a time (live: a press mid-Dream answered
                # "not enough conversations" and the user thought it refused).
                from .status import deep_step

                return {"started": False, "message": "A Deep Dream is already running · "
                        + deep_step(learner).removeprefix("Deep Dream · ")}
        if kind == "quick":
            unread = await self.quick_now(model)
            if unread == "no_model":
                return {"started": False, "message": "Quick Dream can't start · no model is loaded in your model server · load it, then press Quick Dream"}
            if unread == "memory":
                return {"started": False, "message": "Quick Dream can't start · memory is tight on this computer · close some apps, then press Quick Dream"}
            if unread == "load_unknown":
                return {"started": False, "message": "Quick Dream can't start · Songe can't measure the model to check memory · load the model in your model server, then press Quick Dream"}
            if unread == "load_memory":
                return {"started": False, "message": "Quick Dream can't start · not enough free memory to load the model · close some apps, or load the model in your model server, then press Quick Dream"}
            if unread == "nothing":
                last = (self.installation_store.scoped(model).get("quick_last") or {}).get("at")
                # The page puts `since` in the viewer's local time.
                return {"started": False, "message": quick_message(None, nothing_since=0), "since": last}
            return {"started": True, "message": quick_message(unread)}
        pressed = self.installation_store.scoped(model)
        pressed.set("dream_stopped", None)
        pressed.set("dream_growth", None)
        pressed.set("dream_capacity_edges", None)
        pressed.set("memory_wait", None)
        pressed.set("dream_memory_waits", None)
        wait = await self.training_wait(model)
        reading, unread = await self.read_now(model)
        if wait is None:
            self.installation_store.set("requested_model", model)
            self.installation_store.set("requested_sleep", True)
        elif reading:
            # It prepares practice now, but cannot train: once practice is
            # written, the note says so (a pressed Dream never just vanishes).
            pressed.set("dream_pending_wait", wait)
        if reading:
            # Reading and practice under way for this press: a restart in the
            # middle of it must not drop the Dream silently (see start).
            pressed.set("dream_preparing", True)
        return {
            "started": bool(reading) or wait is None,
            "message": dream_message(reading=reading, unread=unread, wait=wait),
        }

    async def read_now(self, model=None):
        """Read unread replies now instead of after the quiet time. Returns
        how many it starts on, and why not when it cannot."""
        if getattr(self.installation_config, "learning", "") == "dreams":
            return await self.read_sessions_now(model)
        from .review import pending

        waiting = len(await asyncio.to_thread(pending, self.installation_store, 50))
        if not waiting or getattr(self, "review_task", None):
            return waiting, ""
        sample = await asyncio.to_thread(self.platform.memory)
        if sample.pressure != "normal":
            return 0, "memory pressure is high"
        if not await self.review_model():
            return 0, "no model is loaded to read with"

        async def read_all():
            # Batch after batch until none is left or the user comes back.
            for _ in range(20):
                if not await self.review_replies(force=True):
                    return

        self.dream_task = asyncio.create_task(read_all())
        return waiting, ""

    def reading_busy(self):
        """Reading or practice under way (in this tick or in the background)."""
        if getattr(self, "reading_task", None) is not None:
            return True
        if getattr(self, "pressing", None) or getattr(self, "press_loads", None):
            return True  # a press loading, using or releasing its model
        task = getattr(self, "dream_task", None)
        return task is not None and not task.done()

    async def read_sessions_now(self, model=None):
        """Dream under learning = "dreams": read sessions and write practice
        now instead of after the quiet time."""
        from .models import registry
        from .reading import store as reading

        waiting = 0
        for row in registry.entries(self.installation_store):
            if row["enabled"] and (model is None or row["id"] == model):
                learner = self.installation_store.scoped(row["id"])
                waiting += len(await asyncio.to_thread(reading.pending, learner))
        running = getattr(self, "dream_task", None)
        if getattr(self, "reading_task", None) or (running and not running.done()):
            return waiting, ""  # Already reading.
        sample = await asyncio.to_thread(self.platform.memory)
        if sample.pressure != "normal":
            return 0, "memory pressure is high"
        self.dream_task = asyncio.create_task(
            self.read_sessions(force=True, only=model, practice=True)
        )
        return waiting, ""

    async def cancel(self, model):
        """Throw away this model's Dream under way: stop its training child,
        discard its night (running, interrupted or waiting to swap). Its
        practice stays untrained, so a later Deep Dream trains it. Returns how
        many nights were discarded; self.cancel_stopped says whether work
        under way (a reading, a measurement, a pressed Dream getting ready)
        was stopped, self.cancel_requested whether a pressed Dream was."""
        async with self.lock:
            stopped = False
            for name in ("dream_task", "reading_task"):
                task = getattr(self, name, None)
                if task is not None and not task.done():
                    task.cancel()
                    stopped = True
            if self.process and self.active_model == model:
                stopped = True
                self.interrupt("cancelled")
                try:
                    await asyncio.wait_for(
                        self.process.wait(), self.config.guard_grace_seconds
                    )
                except TimeoutError:
                    kill(self.process)
                    await self.process.wait()
                await self.finish()
            learner = self.installation_store.scoped(model)
            discarded = 0
            for night in learner.nights():
                if night["status"] in {"running", "interrupted", "ready", "waiting"}:
                    learner.night(night["id"], status="discarded")
                    discarded += 1
            requested = self.installation_store.get("requested_model") == model
            if requested:
                # A pressed Dream still getting ready: it is stopped too.
                stopped = True
                self.installation_store.set("requested_model", None)
                self.installation_store.set("requested_sleep", False)
            learner.set("memory_wait", None)  # a cancelled Dream waits for nothing
            learner.set("dream_memory_waits", None)
            self.cancel_stopped = stopped
            self.cancel_requested = requested
            return discarded

    @staticmethod
    def signals_seen(learner):
        """What a Quick Dream has to go on: [signal count, newest signal]."""
        with learner.connect() as db:
            return list(
                db.execute(
                    "SELECT COUNT(*), MAX(rowid) FROM signals WHERE model_id=?",
                    (learner.model_id,),
                ).fetchone()
            )

    async def quick_now(self, model):
        """Quick Dream now: the newest unread conversations within the budget,
        then the prompt lessons. Returns how many conversations it reads (or
        None when a reading is already running: it continues)."""

        running = getattr(self, "dream_task", None)
        if (
            getattr(self, "reading_task", None)
            or (running and not running.done())
            or model in self.pressing
            or model in self.press_loads
        ):
            return None
        # Marked before the first await: a second press while this one loads
        # the model continues it instead of loading again.
        self.pressing.add(model)
        token = PRESSING.set(model)
        started = False
        try:
            started, unread = await self.quick_start(model)
            return unread
        finally:
            PRESSING.reset(token)
            if not started:
                self.pressing.discard(model)
                self.schedule_release(model)

    async def quick_start(self, model):
        """quick_now's checks, the load on press and the reading task.
        Returns (whether the task started, quick_now's answer)."""
        from .reading import store as reading

        learner = self.installation_store.scoped(model)
        unread = len(await asyncio.to_thread(reading.pending, learner))

        def failed(error):
            learner.set("quick_checked", {"at": time.time(), "nothing_new": False, "error": error})
            from . import quick as quick_lessons

            quick_lessons.record(
                learner,
                {"added": [], "lessons": len(quick_lessons.active(learner)),
                 "changes": {"added": [], "reworded": [], "dropped": []}},
                time.time(), None, read=0, found=0, error=error)
            return False, error

        if not unread and learner.get("quick_seen") == await asyncio.to_thread(self.signals_seen, learner):
            # Nothing unread and nothing read since the last Quick Dream: say
            # so, on the notice and (lasting) on the dashboard.
            learner.set("quick_checked", {"at": time.time(), "nothing_new": True})
            return False, "nothing"
        sample = await asyncio.to_thread(self.platform.memory)
        if sample.pressure != "normal":
            # Reading waits for normal memory pressure (read_sessions), also
            # when it would only publish lessons from new signals; a press
            # under pressure would say "started" and do nothing
            # (reported by Jack in a test on 11-24: a Mac at "warning").
            return await asyncio.to_thread(failed, "memory")
        probe_reader = await self.reader_for(model)
        if probe_reader is None:
            # The press may load the learner's model itself (released once
            # this Quick Dream is done); otherwise reading waits for it (live:
            # a press right after a restart said "started" and did nothing).
            why = await self.load_for_press(model)
            if why:
                return await asyncio.to_thread(failed, why)
            probe_reader = await self.reader_for(model)
            if probe_reader is None:
                return await asyncio.to_thread(failed, "no_model")
        await probe_reader.close()

        async def quick_then_release():
            PRESSING.set(model)
            try:
                return await self.read_sessions(
                    force=True,
                    only=model,
                    quick=True,
                    seconds=self.installation_config.quick_read_seconds,
                )
            finally:
                self.pressing.discard(model)
                self.schedule_release(model)

        self.dream_task = asyncio.create_task(quick_then_release())
        return True, unread

    async def load_for_press(self, model):
        """A Dream the user pressed may load the learner's model when it is
        not loaded, on a server that loads models on demand (Ollama, LM
        Studio), if memory allows: the server's own load, admitted by the same
        memory check as Songe's own versions (upstream.load without
        "serving"). Background work never comes here. Returns None once the
        model is loaded, or why it waits: "load_memory" (not enough free
        memory to load it), "load_unknown" (its size cannot be measured, so
        memory cannot be checked), "no_model" (it cannot be loaded)."""
        from .memory import MemoryPressure
        from .upstream.protection import weight_bytes

        if getattr(self.upstream, "name", None) not in {"ollama", "lmstudio"}:
            return "no_model"
        learner = self.installation_store.scoped(model)
        version = await asyncio.to_thread(learner.version)
        if not version or not version.get("artifact"):
            return "no_model"
        try:
            size = await asyncio.to_thread(weight_bytes, version["artifact"])
        except OSError:
            size = 0
        if not size:
            # Admission sizes a load by its weights: weights Songe cannot
            # read (an Ollama blob outside its view) would pass as 0 bytes.
            return "load_unknown"
        if self.process:
            return "no_model"  # learning has the model server
        identifier = version["identifier"]
        owned = getattr(self.upstream, "owned", {})
        ours_before = identifier in owned
        # Not under self.lock: a cold load can take minutes and the agent's
        # requests wait on that lock. Registered first, so other work skips
        # the model (loaded_listing) and learning waits (reading_busy).
        loading = asyncio.Event()
        self.press_loads[model] = {"version": version, "used": False, "loading": loading}
        try:
            await self.upstream.load(version)
        except MemoryPressure as exc:
            logger.info("press_load_waits %s: %s", model, exc)
            return "load_memory"
        except Exception as exc:
            logger.warning("press_load_failed %s: %s", model, exc)
            return "no_model"
        finally:
            loading.set()
            owned = getattr(self.upstream, "owned", {})
            # Only a load Songe requested is Songe's: a model that was already
            # loaded (the user's) is reused and never claimed, never released.
            if ours_before or identifier not in owned:
                self.press_loads.pop(model, None)
        if model in self.press_loads:
            logger.info("press_loaded %s", identifier)
        return None

    def schedule_release(self, model):
        """Release a press's load in its own task: it may wait for the agent's
        requests to finish, and nothing else waits for it."""
        task = self.release_tasks.get(model)
        if model in self.press_loads and (task is None or task.done()):
            self.release_tasks[model] = asyncio.create_task(self.release_press_load(model))

    async def release_press_load(self, model):
        """Unload what load_for_press loaded, once nothing is in flight and no
        reader is on it (Ollama: keep_alive 0, confirmed by /api/ps; LM Studio:
        unload), unless the agent has asked for that model since: then it is
        the user's, and the server's own idle timeout decides. A request that
        arrives while it releases sends it back here to wait; nothing drops
        Songe's claim on an instance it did not release."""
        interval = getattr(self.installation_config, "guard_min_interval", 0.1)
        while True:
            held = self.press_loads.get(model)
            if not held:
                return
            if held["used"]:
                self.press_loads.pop(model, None)
                logger.info("press_load_kept %s: the agent uses it", model)
                return
            if (
                any(self.in_flight.values())
                or model in (getattr(self, "reading_activity", None) or {})
                or model in self.pressing
            ):
                await asyncio.sleep(interval)  # retried, never dropped
                continue

            def still_ours(held=held):
                # Checked again right before the unload and while it waits:
                # any request admitted meanwhile (its model not known yet)
                # stops it, and the loop above decides.
                return not held["used"] and not any(self.in_flight.values())

            try:
                released = await self.upstream.unload(held["version"], still=still_ours)
            except Exception as exc:
                logger.warning("press_release_failed %s: %s", model, exc)
                self.press_loads.pop(model, None)
                return
            if released is False:
                continue  # interrupted by a request: wait, then decide again
            logger.info("press_released %s", held["version"]["identifier"])
            self.press_loads.pop(model, None)
            return

    def dream_blocked(self, model):
        """Why Dream can do nothing for this model right now, or None."""
        from .models.library import model_name
        from .models.registry import summaries

        row = next(
            m
            for m in summaries(self.installation_config, self.installation_store)
            if m["id"] == model
        )
        if self.installation_store.get("paused", False):
            return "learning is paused"
        if not row["enabled"]:
            return "learning is off for this model"
        if row["support"] == "unsupported":
            return "this model is not supported yet"
        if self.process:
            if self.active_model == model:
                return "a Deep Dream is running · pause or cancel it first"
            return f"{model_name(self.active_model)} is learning now"
        if any(self.in_flight.values()):
            return "the model is in use"
        return None

    async def training_wait(self, model):
        """What this model's training waits on right now, in a few words; None
        when it can start. The same conditions the scheduler waits on, minus
        the quiet time a request skips."""
        from .memory import TrainingPlatform
        from .models.registry import summaries
        from .pipeline import next_update
        from .sleep.calibration_state import needs_growth
        from .wording import DATA_NOUNS, flow_state

        row = next(
            m
            for m in summaries(self.installation_config, self.installation_store)
            if m["id"] == model
        )
        config, store = self.config_for(model), self.installation_store.scoped(model)
        measuring = row["support"] == "pending" or await asyncio.to_thread(
            needs_growth, config, store
        )
        resumes = any(n["status"] in ("interrupted", "ready") for n in store.nights())
        short = await asyncio.to_thread(heldout_short, config, store)
        if short and row["support"] != "pending" and not resumes:
            # Checked before measuring: measuring again (15 min) only to find
            # nothing can be checked would waste the Dream.
            return ("it needs a conversation to learn from · use a Songe model in your agent and"
                    " react to its work (a correction or a reply), then dream again")
        if not measuring and not resumes:
            upcoming = await asyncio.to_thread(next_update, store)
            if upcoming["stage"] in ("train", "test"):
                missing = upcoming["need"] - upcoming["have"]
                noun = DATA_NOUNS[upcoming["stage"]]
                return f"needs {missing} more {noun if missing != 1 else noun[:-1]}"
            # Dream was pressed for this model: auto-dream off does not hide it.
            if not any(m["id"] == model for m in self.candidates(asking=model)):
                return "nothing new to learn"
        p = await asyncio.to_thread(probe, self.platform)
        hardware = reason(
            replace(config, calibration_available=True),
            p,
            enough=True,
            immediate=True,
            adaptive=True,
        )
        if hardware.startswith("Waiting for free memory"):
            return f"memory pressure is high · {p.free_gb:.1f} GB free"
        if hardware == "Waiting for hardware probes":
            return "hardware readings unavailable"
        if hardware != "Will sleep when idle":
            return flow_state(hardware).removeprefix("waiting: ")
        if not measuring and hasattr(self.upstream, "park"):
            try:
                reclaimable = (
                    await self.upstream.loaded_bytes()
                    if hasattr(self.upstream, "loaded_bytes")
                    else 0
                )
            except Exception:
                reclaimable = 0  # The model server is closed.
            before = await asyncio.to_thread(
                snapshot, TrainingPlatform(self.platform, config)
            )
            wait = self.memory_wait(config, store, before, reclaimable)
            if wait:
                return flow_state(wait)
        return None

    async def sleep_wait(self):
        """`songe sleep`: what the request waits on before learning can
        start, in a few words; None when nothing holds it (live #66:
        "Learning requested", then nothing, and no word why)."""
        from .models.library import model_name
        from .models.registry import summaries
        from .status import deep_step, memory_hold
        from .wording import flow_state

        if self.installation_store.get("paused", False):
            return "learning is paused · songe resume"
        if self.process:
            return None  # Already working; songe status shows what.
        rows = [
            m
            for m in summaries(self.installation_config, self.installation_store)
            if m["enabled"] and m["support"] != "unsupported"
        ]
        if not rows:
            return "no model is learning · songe models"
        model = await asyncio.to_thread(self.choose_model) or min(
            rows, key=lambda m: m.get("auto_dream") is False
        )["id"]
        learner = self.installation_store.scoped(model)
        if learner.get("memory_wait"):
            return deep_step(learner).removeprefix("Deep Dream · ")
        if hold := memory_hold(learner):
            return "waiting for memory · " + hold.removeprefix("held for memory · ")
        if learner.get("disk_wait"):
            return flow_state(learner.get("disk_wait"))
        if (
            getattr(self.installation_config, "learning", "") == "dreams"
            and not any(m.get("auto_dream", True) for m in rows)
        ):
            # A request starts nothing by itself here; Dream does (candidates).
            return f"auto-dream is off for {model_name(model)} · press Dream on its page (songe open)"
        p = await asyncio.to_thread(probe, self.platform)
        hardware = reason(
            replace(self.config_for(model), calibration_available=True),
            p,
            enough=True,
            immediate=True,
            adaptive=True,
        )
        if hardware != "Will sleep when idle":
            return flow_state(hardware).removeprefix("waiting: ")
        return None

    async def next_model(self):
        from .sleep.calibration_state import needs_growth

        excluded = set()
        # Choosing reads every model's experience: off the event loop, so the
        # proxy and `songe status` answer meanwhile (#71).
        while model := await asyncio.to_thread(self.choose_model, excluded):
            self.select(model)
            if (
                not self.config.calibration_available
                or await asyncio.to_thread(needs_growth, self.config, self.store)
                or await self.enough()
            ):
                return model
            excluded.add(model)
        self.active_model = None
        return None

    def config_for(self, model):
        from .models.registry import model_config

        return model_config(self.installation_config, self.installation_store, model)

    async def restore(self):
        parked = self.installation_store.get("parked_models", [])
        if parked:
            await self.upstream.restore(parked, self.managed())
        from .models.registry import entries

        global_binding = self.installation_config.upstream in {"llamacpp", "mlx"}
        on_request = self.installation_config.upstream in {"lmstudio", "ollama"}
        parked_ids = {p["identifier"] for p in parked}
        target = self.serving_model or self.installation_store.get("primary_model")
        for model in entries(self.installation_store):
            scoped = self.installation_store.scoped(model["id"])
            version = scoped.version()
            was_parked = any(v["identifier"] in parked_ids for v in scoped.versions())
            # A server that loads on request (LM Studio, Ollama) gets back only
            # what Songe unloaded: loading "the" model at login and after every
            # slice fills memory the user never asked for, and each request
            # loads what it needs anyway. A server Songe runs itself always
            # serves its one model.
            if (
                version
                and model["enabled"]
                and (
                    was_parked
                    and not global_binding
                    or model["id"] == target
                    and (bool(parked) or not on_request)
                )
            ):
                if hasattr(self.upstream, "supervisor"):
                    self.upstream.supervisor.config = self.config_for(model["id"])
                serving = version | {"serving": True}
                if hasattr(self.upstream, "ensure"):
                    await self.upstream.ensure(serving)
                else:
                    await self.upstream.load(serving)
                self.serving_binding = version["id"]
        self.installation_store.set("parked_models", [])

    async def wake(self):
        if not self.process:
            self.current_version = None  # Each request renews its own serving identity.
        await super().wake()
        self.active_model = None

    async def finish(self):
        from .models import registry

        calibrating = self.calibrating
        model = self.active_model
        failed = self.process.returncode not in (None, 0)
        interrupted = self.cancelled_at is not None
        # The child's own memory check stopped it (its parent recorded why).
        memory_stop = self.memory_stop_sample or self.process.returncode == MEMORY_EXIT
        # A measuring run that ended badly right after a memory stop ran out
        # of memory too (live: "training setup failed · songe doctor").
        last_stop = self.store.get("last_memory_stop_at") or 0
        recent_memory_stop = bool(calibrating and failed and time.time() - last_stop < 120)
        memory_stop = memory_stop or recent_memory_stop
        stopped_by = self.interrupted_by
        await super().finish()
        launch, self.launch = getattr(self, "launch", None), None
        pressed = bool(launch and launch["requested"] and launch["model"] == model)
        if pressed:
            self.carry_on(model, launch, calibrating=calibrating, failed=failed,
                          interrupted=interrupted, stopped_by=stopped_by, memory_stop=memory_stop)
        if model and not (pressed and memory_stop):  # a pressed Dream's own wait: carry_on
            self.note_memory_hold(model, memory_stop=memory_stop, calibrating=calibrating)
        if calibrating and model:
            self.config = self.config_for(model)
            saved = self.store.get("calibration", {})
            if saved.get("budget"):
                from .sleep.calibrate import derive_keys

                for key in derive_keys():
                    if key in saved["budget"]:
                        setattr(self.config, key, saved["budget"][key])
                self.config.save(self.root)
                registry.support(self.installation_store, model, "trainable")
            elif memory_stop or recent_memory_stop:
                # A memory stop says nothing about whether the model can train:
                # a trainable model stays trainable (live: growth measuring on
                # a crowded Mac turned it back to "pending"); a new one is
                # measured again in a later idle gap.
                row = registry.find(self.installation_store, model)
                if not row or row["support"] != "trainable":
                    registry.support(
                        self.installation_store,
                        model,
                        "pending",
                        "calibration paused · memory · retry when idle",
                    )
            elif self.store.get("disk_wait"):
                registry.support(
                    self.installation_store,
                    model,
                    "pending",
                    self.store.get("disk_wait"),
                )
            elif failed and not interrupted:
                row = registry.find(self.installation_store, model)
                if row and row["support"] != "unsupported":
                    # Errors never retire a model; the scheduler retries with
                    # a doubling wait (retry_wait) until a run succeeds.
                    registry.support(
                        self.installation_store,
                        model,
                        "pending",
                        "calibration failed · retries automatically",
                    )
        self.active_model = None

    def note_memory_hold(self, model, *, memory_stop, calibrating):
        """An automatic run (not a pressed Deep Dream: carry_on) that the
        memory guard stopped holds this model's learning for MEMORY_BACKOFF
        times the stops in a row, shown in the status (status.memory_hold),
        then tries again (live #66: nothing said why it stopped). Measuring
        that reached this machine's edge is the measurement's answer, not a
        crowded machine (as in carry_on). Any other end clears the hold."""
        learner = self.installation_store.scoped(model)
        edge = calibrating and any(
            r.get("sequence") for r in (learner.get("calibration") or {}).get("rows", [])
        )
        if not memory_stop or edge:
            learner.set("memory_hold", None)
            return
        stops = (learner.get("memory_hold") or {}).get("try", 0) + 1
        sample = learner.get("last_memory_stop") or {}
        now = time.time()
        learner.set("memory_hold", {
            "at": now, "try": stops,
            "rule": (sample.get("stop_reason") or {}).get("rule"),
            # Free before the run (the stop's own reading includes the run).
            "available": (learner.get("last_memory_baseline") or {}).get("available"),
            "retry_after": now + min(MEMORY_BACKOFF * stops, EMPTY_RUN_WAIT_MAX),
        })

    def memory_held(self, wait):
        """A memory wait or hold still in force: before its retry time,
        unless clearly more memory is free now."""
        if not wait or time.time() >= wait.get("retry_after", 0):
            return False
        try:
            free = self.platform.memory().available
        except Exception:  # noqa: BLE001
            free = 0
        return not wait.get("available") or free < wait["available"] * 1.3 + 1e9

    def clear_stale_waits(self, model, learner):
        """A memory wait with no pending request (or next to a stop note) is
        over (live: "waiting for memory · try 1 of 3" beside the note)."""
        if learner.get("memory_wait") and (
            self.installation_store.get("requested_model") != model or learner.get("dream_stopped")
        ):
            learner.set("memory_wait", None)
            learner.set("dream_memory_waits", None)

    def recover_pressed(self, model, learner):
        """A pressed Deep Dream that was reading or practising when Songe
        stopped (live: a deploy restart; it vanished): one that was going to
        train carries on (its request is still there); otherwise the note
        says it was interrupted."""
        if not learner.get("dream_preparing"):
            return
        learner.set("dream_preparing", None)
        if (self.installation_store.get("requested_model") == model
                and self.installation_store.get("requested_sleep")):
            return
        from .practice.store import counts

        learner.set("dream_pending_wait", None)
        learner.set("dream_stopped", {"kind": "deep", "at": time.time(), "reason": "restart",
                                      "kept": counts(learner)["kept"]})

    def carry_on(self, model, launch, *, calibrating, failed, interrupted, stopped_by, memory_stop):
        """A pressed Deep Dream ends only in a night or a note saying why.
        Its request is used up when a run starts; until a night exists it is
        asked again: after measuring (training follows), after the agent came
        back (it resumes when quiet), once after a training run that found it
        must measure again first. Anything else leaves the note."""
        learner = self.installation_store.scoped(model)
        if learner.get("growth_requested") and calibrating:
            learner.set("growth_requested", None)
        if stopped_by in {"cancelled", "paused"} or self.installation_store.get("paused", False):
            return  # The user stopped it.

        def again():
            self.installation_store.set("requested_model", model)
            self.installation_store.set("requested_sleep", True)

        def stop(detail):
            learner.set("dream_growth", None)
            learner.set("memory_wait", None)  # the note replaces any wait
            learner.set("dream_memory_waits", None)
            from .practice.store import counts

            learner.set("dream_stopped", {"kind": "deep", "at": time.time(), "reason": "training",
                                          "detail": str(detail)[:300], "kept": counts(learner)["kept"]})
            logger.warning("deep_dream_stopped %s: %s", model, detail)

        nights = learner.nights()
        if len(nights) > launch["nights"]:
            if nights[-1]["status"] == "failed":
                # A failed night does not resume on its own, and with
                # auto-dream off nothing else starts it: the retry announced
                # ("failed · retry in 5m") is this request, kept armed (the
                # failure back-off sets when). A few failures in a row end in
                # a note saying why (live, Inkling: "retry in 5m", then
                # nothing for an hour and no word).
                failures = (learner.get("failures") or {}).get("count", 0)
                if failures >= FAILED_NIGHT_RETRIES:
                    why = learner.get("child_error") or "an error stopped it"
                    stop(f"training failed {failures} times in a row: {why}")
                    return
                again()
                return
            learner.set("dream_growth", None)
            learner.set("memory_wait", None)
            learner.set("dream_memory_waits", None)
            learner.set("dream_capacity_edges", None)
            return  # It is under way: unfinished nights resume on their own.

        measured = any(r.get("sequence") for r in (learner.get("calibration") or {}).get("rows", []))
        if memory_stop and calibrating and measured:
            # Measuring found the edge of what fits: smaller sizes fitted, then
            # one did not (live: 256 and 512 tokens fit, 1024 hit the reserve
            # with 12 GB free before). That is the measurement's answer, not a
            # crowded machine: closing apps would not help. Training follows
            # at the largest size that fit; a few times at most. (A stop with
            # nothing measured is a crowded machine: the memory wait below.)
            edges = (learner.get("dream_capacity_edges") or 0) + 1
            learner.set("dream_capacity_edges", edges)
            learner.set("memory_wait", None)  # not a crowded machine (any earlier wait is over)
            learner.set("dream_memory_waits", None)
            saved = learner.get("calibration") or {}
            progress = saved.get("progress") or {}
            if progress and not progress.get("complete"):
                # Measuring is done at the largest size that fit, for the
                # model and for the teacher's longer reading alike (live: the
                # model fit all 6550 tokens, the teacher 5780 of 6372; its
                # probes kept hitting the edge and the Dream ended). Training
                # starts there instead of probing the same size again; a
                # sample longer than what fit waits (deferred), never cut.
                progress = progress | {"complete": True, "completion_reason": "memory"}
                learner.set("calibration", saved | {"progress": progress})
                learner.set("calibration_progress", progress)
            if edges > CAPACITY_EDGES:
                learner.set("dream_capacity_edges", None)
                progress = (learner.get("calibration") or {}).get("progress") or {}
                part = ("the teacher's longer reading" if (progress.get("measured_sequence") or 0)
                        >= (progress.get("target_sequence") or 1) else "the model")
                stop(f"it measured the most this machine can train ({part} did not fit any further), "
                     "and could not start training")
                return
            again()
            return
        if memory_stop:
            # The memory guard held it back (live: a crowded Mac; the press was
            # demoted to "a long quiet period" and vanished). It stays this
            # Dream, shown waiting for memory, and tries again when memory
            # frees; after MEMORY_TRIES it stops with the note.
            waits = (learner.get("dream_memory_waits") or 0) + 1
            if waits >= MEMORY_TRIES:
                learner.set("memory_wait", None)
                learner.set("dream_memory_waits", None)
                stop("Waiting for memory")
                return
            own = getattr(self, "store", None)
            sample = learner.get("last_memory_stop") or (own.get("last_memory_stop") if own else None) or {}
            before = learner.get("last_memory_baseline") or (own.get("last_memory_baseline") if own else None) or {}
            learner.set("dream_memory_waits", waits)
            learner.set("memory_wait", {"at": time.time(), "try": waits, "tries": MEMORY_TRIES,
                                        # Free before the try (the stop's own reading is
                                        # pushed down by the measuring itself).
                                        "available": before.get("available") or sample.get("available"),
                                        "total": sample.get("total") or before.get("total"),
                                        # Time to close apps (live: 4 tries in 45 s): a minute,
                                        # then longer; sooner if memory clearly frees.
                                        "retry_after": time.time() + MEMORY_BACKOFF * waits})
            again()
        elif interrupted:
            # The agent came back: still this model's Dream (auto-dream off
            # or not), but it waits for a long quiet period, not at once.
            self.installation_store.set("requested_model", model)
        elif calibrating:
            if failed or memory_stop:
                from .wording import DEVICE_FULL

                # The card was full: say so and what to do (live #135: "did
                # not finish" did not point at the user's own server).
                full = learner.get("child_error") == DEVICE_FULL
                stop(DEVICE_FULL if full else "measuring what this machine can train did not finish")
            else:
                again()
        else:
            outcome = learner.get("worker_outcome") or {}
            reason = outcome.get("reason") if outcome.get("at", 0) >= launch["at"] else None
            if reason == "Waiting for calibration growth" and not learner.get("dream_growth"):
                learner.set("dream_growth", True)  # once per Dream
                learner.set("growth_requested", True)
                again()
            else:
                stop(reason or ("the training run failed" if failed else "it ended without training"))

    async def discover_models(self):
        import httpx

        from .models import registry

        # Discovery runs in an idle gap, never on the inference admission path.
        unknown = [
            r
            for r in registry.entries(self.installation_store)
            if r["support"] == "pending"
            and not r["config"].get("model_ref")
            and not (r["config"].get("model") and Path(r["config"]["model"]).is_dir())
        ]
        if not unknown:
            return
        try:
            refs = await asyncio.to_thread(
                registry.discover, self.installation_config, self.upstream
            )
            waiting = set()
            if self.installation_config.upstream == "llamacpp":
                from .models import discovery

                config = self.installation_config
                waiting = await asyncio.to_thread(
                    discovery.llamaswap_waiting,
                    config.server_source_url or config.upstream_url,
                    getattr(self.upstream, "transport", None),
                )
        except (OSError, ValueError, RuntimeError, KeyError, httpx.HTTPError) as exc:
            # A server that is down or answers oddly leaves the models pending
            # for a later tick; it must never end the tick (issue #8).
            self.reason = f"Model discovery unavailable: {exc}"
            return
        for row in unknown:
            ref = next((r for r in refs if r.serving_id == row["id"]), None)
            if ref is None and row["id"] in waiting:
                # Its weights are named once llama-swap runs it (the user
                # switched models, or its ttl unloaded it): not given up on.
                registry.support(
                    self.installation_store, row["id"], "pending", "waiting until llama-swap runs it"
                )
            elif ref is None:
                registry.support(
                    self.installation_store,
                    row["id"],
                    "unsupported",
                    "Local weights are not discoverable; forwarding only",
                )
            elif not ref.path.is_absolute():
                # A served model whose weights are not on this machine (a
                # vLLM elsewhere, or a cache Songe cannot see): its name, not
                # a path. Saved as the model it failed validation on every
                # tick, and the page said "Sleep failed" on a fresh install.
                registry.support(
                    self.installation_store,
                    row["id"],
                    "unsupported",
                    "Local weights are not accessible; serving only",
                )
            else:
                from .models.library import is_embedding

                if is_embedding(ref):
                    registry.support(
                        self.installation_store,
                        row["id"],
                        "unsupported",
                        "embedding model",
                    )
                    continue
                config = self.config_for(row["id"])
                config.model, config.model_ref = (
                    str(ref.path),
                    asdict(ref) | {"path": str(ref.path)},
                )
                config.save(self.root)
                registry.ensure_base(self.installation_store, row["id"], ref)

    def serving_endpoint(self, version, identifier):
        """URL and model name for Songe's own calls to a learner's model, the
        way the proxy sends an agent's request (#27). A routed version goes
        where its route says. The base is served by the user's own server,
        which is always up; the owned server runs only while Songe needs it.
        On mlx-lm only a route says a server of the user's serves the base:
        without one, Songe's own server does. A learned version without a
        route yet is on the owned server."""
        upstream = self.upstream
        if (
            version
            and version.get("identifier") in (getattr(upstream, "routes", None) or {})
            and hasattr(upstream, "request_url")
        ):
            return upstream.request_url(version), upstream.request_model(version)
        config = self.installation_config
        if version and not (version["id"] == "base" or version["id"].startswith("base-")):
            return config.upstream_url, identifier
        if version and config.upstream == "mlx" and hasattr(upstream, "request_url"):
            # Live: reading asked mlx-lm's default port, where nothing
            # listened, while the agent's requests went to Songe's own server.
            return upstream.request_url(version), upstream.request_model(version)
        return config.server_source_url or config.upstream_url, identifier

    async def review_model(self):
        """A model already loaded on the server; reviewing never loads one."""
        from .models import registry

        items = await self.loaded_listing()
        if items is None:
            return self.installation_config.base_identifier or None
        ids = {r["id"] for r in registry.entries(self.installation_store)}
        for item in items:
            identifier = item.get("identifier")
            if item.get("type", "llm") == "llm" and (
                identifier in ids or str(identifier).startswith("songe/")
            ):
                return identifier
        return None

    async def review_replies(self, *, force=False):
        """Idle gaps first read new user replies with the loaded model.
        Returns how many it read."""
        if getattr(self.installation_config, "learning", "") == "dreams":
            return 0  # Dreams read whole sessions instead (read_sessions).
        if getattr(self, "review_task", None):
            return 0  # One reader at a time: the tick and Dream share it.
        if (
            not force
            and time.monotonic() - getattr(self, "review_failed_at", -1e9) < 300
        ):
            return 0
        if self.process or any(self.in_flight.values()):
            # Never alongside a training child, nor while an answer is still
            # streaming: the review would compete for the same model.
            return 0
        try:
            # Reading uses the already-loaded model and a short window, but
            # any memory pressure still means waiting.
            sample = await asyncio.to_thread(self.platform.memory)
            if sample.pressure != "normal":
                return 0
            model = await self.review_model()
            if not model:
                return 0
            from .review import review

            traffic = self.last_traffic
            url, model = self.serving_endpoint(None, model)
            self.review_task = asyncio.create_task(
                review(
                    self.installation_store,
                    url,
                    model,
                    still_idle=lambda: (
                        self.last_traffic == traffic
                        and not self.process
                        and not any(self.in_flight.values())
                    ),
                )
            )
            try:
                done = await self.review_task
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    raise  # The service is stopping.
                return 0  # The user came back; the reply is reviewed later.
            finally:
                self.review_task = None
            if done:
                logger.info("reviewed_replies %s with %s", done, model)
                self.installation_store.event("review", f"{done} replies")
            return done
        except Exception as exc:  # Reviews are optional; learning goes on.
            self.review_failed_at = time.monotonic()
            logger.warning("review_failed %s", exc)
            return 0

    async def loaded_identifiers(self):
        """Identifiers of the models loaded on the server right now; None when
        the server cannot list them (then the configured one is assumed)."""
        items = await self.loaded_listing()
        if items is None:
            return None
        return {item.get("identifier") for item in items if item.get("type", "llm") == "llm"}

    async def loaded_listing(self):
        """The server's loaded models as {"identifier", "type"}, or None when
        it cannot say. A model loaded for a press is listed only to that
        press's own work (PRESSING): other reading must not start on a model
        that is released when the press is done."""
        items = await self.server_listing()
        hidden = set()
        for model, held in getattr(self, "press_loads", {}).items():
            if model != PRESSING.get():
                name = held["version"]["identifier"]
                hidden |= {model, name, name.removesuffix(":latest"), name + ":latest"}
        if items is None or not hidden:
            return items
        return [item for item in items if item.get("identifier") not in hidden]

    async def server_listing(self):
        listing = getattr(self.upstream, "listing", None)
        if listing is not None:
            return await listing()
        # llama-swap behind the generic client (init found no weights to read)
        # still loads whatever is asked for: its /running says what is loaded.
        from .models import discovery

        config = self.installation_config
        loaded = await asyncio.to_thread(
            discovery.llamaswap_loaded,
            getattr(config, "server_source_url", "") or config.upstream_url,
            getattr(self.upstream, "transport", None),
        )
        if loaded is None and not self.answered_running:
            return None
        # Once it has answered, a failed /running means nothing is loaded.
        self.answered_running = True
        return [{"identifier": i, "type": "llm"} for i in loaded or []]

    async def read_sessions(
        self, *, force=False, only=None, practice=False, seconds=None, quick=False
    ):
        """Idle gaps read new sessions into signals (learning = "dreams"), the
        newest first, each learner with its own current serving version: the
        model being trained does the reading. practice: also write practice
        scenes (Deep Dream's first step; never in short pauses). seconds: a
        reading budget (Quick Dream). quick: publish the prompt lessons after
        reading (a Deep Dream's reading does too, before it writes practice).
        Never loads a model; a learner whose model is not loaded
        waits. Returns how many sessions were read and scenes written."""
        if self.installation_config.learning != "dreams":
            return 0
        if getattr(self, "reading_task", None):
            return 0  # One reader at a time.
        if (
            not force
            and time.monotonic() - getattr(self, "reading_failed_at", -1e9) < 300
        ):
            return 0
        if self.process or any(self.in_flight.values()):
            # Never alongside a training child (a Deep Dream trains only while
            # nobody uses the agent, so there is nothing new to read then, and
            # it has the chip), nor while an answer streams.
            return 0
        from .cycle import SIGNAL_TYPES as reading_signal_types
        from .models import registry
        from .practice import store as practice_store
        from .reading import store as reading
        from .reading.client import Reader

        try:
            sample = await asyncio.to_thread(self.platform.memory)
            if sample.pressure != "normal":
                return 0
            loaded = await self.loaded_identifiers()
            done = 0
            learner = None
            for row in registry.entries(self.installation_store):
                if not row["enabled"] or (only and row["id"] != only):
                    continue
                if not only and not registry.auto_dream(self.installation_store, row):
                    continue  # Auto-dream off: reads only when Dream is pressed.
                learner = self.installation_store.scoped(row["id"])
                if practice:
                    learner.set("dream_stopped", None)
                if row["id"] not in self.__dict__.setdefault("short_checked", set()):
                    # Once per service start: older lessons get their short form.
                    self.short_checked.add(row["id"])
                    queued = await asyncio.to_thread(reading.reread_without_short, learner)
                    if queued:
                        logger.info("reread_for_short %s sessions of %s", queued, row["id"])

                version = await asyncio.to_thread(learner.version)
                identifier = (version or {}).get("identifier") or row["id"]
                if loaded is not None and identifier not in loaded:
                    continue  # Waits for its own model (code model_server).
                per_signal = self.config_for(row["id"]).practice_per_signal
                unread = await asyncio.to_thread(reading.pending, learner, 1)
                wants_practice = practice and await asyncio.to_thread(
                    practice_store.wanted, learner, per_signal
                )
                if not unread and not wants_practice:
                    if quick:
                        await self.quick_dream(row["id"])
                    continue
                traffic = self.last_traffic
                url, served = self.serving_endpoint(version, identifier)
                reader = Reader(
                    learner,
                    url,
                    served,
                    server=self.installation_config.upstream,
                    still_idle=lambda traffic=traffic: (
                        self.last_traffic == traffic
                        and not self.process
                        and not any(self.in_flight.values())
                    ),
                )
                # What the cycle record shows while there is no night yet.
                activity = {
                    "stage": "read",
                    "started_at": time.time(),
                    "trigger": "dream" if force else "idle",
                    "kind": "quick" if quick else "deep" if practice else "read",
                    # For the Quick view: conversations read of those waiting,
                    # within the reading budget, and the measured pace.
                    "total": len(await asyncio.to_thread(reading.pending, learner)),
                    "done": 0,
                    "budget": seconds,
                    "read_seconds": 0.0,
                    "reading": None,   # the conversation being read now
                    "found": [],       # signals found in this run, newest last
                    "found_count": 0,
                }
                self.reading_activity = {row["id"]: activity}

                async def read_then_write(
                    learner=learner, reader=reader, per_signal=per_signal,
                    activity=activity, model=row["id"],
                ):
                    last = [time.monotonic()]

                    def counted(session, found, activity=activity):
                        now = time.monotonic()
                        activity["done"] += 1
                        activity["read_seconds"] += now - last[0]
                        # The last few conversations' times: the first ones are
                        # often cached and fast, so the pace is a recent average.
                        activity["paces"] = (activity.get("paces", []) + [now - last[0]])[-3:]
                        last[0] = now
                        # Approvals teach nothing for the prompt: not listed.
                        shown = [
                            reading.signal_line(f) for f in found
                            if f.get("type") in reading_signal_types and f.get("type") != "approval"
                        ]
                        activity["found_count"] += len(shown)
                        activity["found"] = (activity["found"] + shown)[-6:]

                    def starting(session, items, activity=activity):
                        activity["reading"] = reading.opening_request(items)

                    read = await reading.read(
                        learner, reader, newest=True, seconds=seconds,
                        on_session=counted, on_start=starting,
                    )
                    activity["stage"] = "prompt"
                    reader.log_stats("read")
                    if read:
                        learner.event("read", f"{read} sessions")
                    if quick:
                        from . import quick as quick_lessons

                        before = len(learner.get(quick_lessons.HISTORY) or [])
                        await self.quick_dream(model, reader=reader)
                        history = learner.get(quick_lessons.HISTORY) or []
                        failed = getattr(reader, "failed_sessions", None)
                        run = {"read": read, "found": activity.get("found_count", 0),
                               "error": "unreadable" if failed else None}
                        if len(history) > before:
                            quick_lessons.note_run(learner, history[-1]["id"], **run)
                        else:
                            # Nothing reached the prompt (or nothing new was
                            # read): still a Quick Dream in the history.
                            empty = {"added": [], "reworded": [], "dropped": []}
                            await asyncio.to_thread(
                                quick_lessons.record, learner,
                                {"added": [], "lessons": len(quick_lessons.active(learner)), "changes": empty},
                                time.time(), None, **run)
                        checked = learner.get("quick_checked") or {}
                        if read and checked.get("nothing_new"):
                            # It read and found, and nothing new reached the
                            # prompt: said with what it read (live: 5 read, 9
                            # moments, 0 lessons; then just "nothing new").
                            learner.set("quick_checked", checked | {"read": read, "found": activity.get("found_count", 0)})
                    elif practice:
                        # A Deep Dream that read conversations no Quick Dream
                        # had read (pressed early, or a backlog) writes their
                        # lessons into the prompt as a Quick Dream would,
                        # before practice and the check. Live: a Dream read
                        # the rest of the chats, trained on their lessons and
                        # was checked against a prompt holding 8 notes for
                        # four of the eight habits; the other notes arrived
                        # with the next Quick Dream, after the gate. Nothing
                        # new since the last Quick Dream: nothing happens.
                        await self.quick_dream(model, reader=reader)
                    failed = getattr(reader, "failed_sessions", None)
                    if failed:
                        # The model's answers could not be read: said on the
                        # page (never "nothing new"); those stay unread.
                        learner.set("quick_checked", {"at": time.time(), "nothing_new": False,
                                                      "error": "unreadable", "failed": len(failed)})
                    written = 0
                    if practice:
                        activity["stage"] = "practice"
                        written = await practice_store.write(learner, reader, per_signal)
                        reader.log_stats("practice")
                        if written:
                            learner.event("practice", f"{written} scenes")
                        learner.set("dream_preparing", None)
                        wait = learner.get("dream_pending_wait")
                        if wait:
                            learner.set("dream_pending_wait", None)
                            learner.set("dream_stopped", {
                                "kind": "deep", "at": time.time(), "reason": "training",
                                "detail": wait, "kept": practice_store.counts(learner)["kept"],
                            })
                    return read + written

                self.reading_task = asyncio.create_task(read_then_write())
                try:
                    read = await self.reading_task
                except asyncio.CancelledError:
                    current = asyncio.current_task()
                    if current is not None and current.cancelling():
                        raise  # The service is stopping.
                    return done  # The user came back; the rest is read later.
                finally:
                    self.reading_task = None
                    self.reading_activity = {}
                    await reader.close()
                if read:
                    logger.info("read_and_practice %s with %s", read, identifier)
                done += read
            return done
        except Exception as exc:  # Reading waits for the next gap; serving goes on.
            self.reading_failed_at = time.monotonic()
            logger.warning("reading_failed %s: %s", type(exc).__name__, exc, exc_info=True)
            if practice and learner is not None:
                learner.set("dream_preparing", None)
                # A Deep Dream that stops says so on the page, and why.
                from .practice.store import PracticeFailed, counts

                learner.set("dream_stopped", {
                    "kind": "deep", "at": time.time(), "kept": counts(learner)["kept"],
                    "reason": "practice" if isinstance(exc, PracticeFailed) else "reading",
                    "detail": f"{type(exc).__name__}: {exc}"[:300],
                })
            return 0

    async def reader_for(self, model):
        """A Reader on the learner's own loaded serving model, or None when it
        is not loaded (work that needs it waits, as reading does)."""
        from .reading.client import Reader

        learner = self.installation_store.scoped(model)
        version = await asyncio.to_thread(learner.version)
        identifier = (version or {}).get("identifier") or model
        loaded = await self.loaded_identifiers()
        if loaded is not None and identifier not in loaded:
            return None
        traffic = self.last_traffic
        url, served = self.serving_endpoint(version, identifier)
        return Reader(
            learner,
            url,
            served,
            server=self.installation_config.upstream,
            still_idle=lambda: (
                self.last_traffic == traffic
                and not self.process
                and not any(self.in_flight.values())
            ),
        )

    async def quick_dream(self, model, *, idle=False, reader=None):
        """Publish one learner's prompt lessons (quick.py): the served model
        consolidates the candidates (merges same-meaning ones, drops
        task-specific ones, shortens them); its candidates as they are if that
        answer is unusable. Idle: only when signals changed since the last
        Quick Dream. Waits (None) when the learner's model is not loaded."""
        from . import quick
        from .reading.client import Stopped

        learner = self.installation_store.scoped(model)
        seen = await asyncio.to_thread(self.signals_seen, learner)
        if learner.get("quick_seen") == seen:
            # No new signals since the last Quick Dream: the published list
            # stands (a re-run would only reword it).
            learner.set("quick_checked", {"at": time.time(), "nothing_new": True})
            return None
        config = self.config_for(model)
        # Only consolidation needs the served model; without it Quick Dream
        # is Songe's own selection and runs whether or not the model is loaded.
        own = reader is None and config.quick_consolidate
        if config.quick_consolidate:
            reader = reader or await self.reader_for(model)
            if reader is None:
                return None
        try:
            clusters = await asyncio.to_thread(
                quick.candidates, learner, config.prompt_lessons * quick.CANDIDATES,
                config.quick_min_sessions,
            )
            # Published lessons keep their wording; only new ones go to the model.
            kept, fresh = await asyncio.to_thread(quick.split, learner, clusters)
            try:
                # quick.review (one judgement per lesson) is not used: on the
                # live run Granite 4.2 3B answered "general, not covered" for
                # every lesson, so it cost calls and changed nothing.
                merged = (
                    await quick.consolidate(reader, fresh, config.prompt_lessons)
                    if fresh and config.quick_consolidate
                    else None  # the clusters' own text (tidied)
                )
            except Stopped:
                return None  # The user came back; the next quiet time finishes it.
            except Exception as exc:  # Consolidation failed: candidates as they are.
                logger.warning("quick_consolidate_failed %s: %s", model, exc)
                merged = None
        finally:
            if own:
                await reader.close()
        before_block = quick.block(quick.active(learner))
        result = await asyncio.to_thread(
            quick.quick_dream, learner, config, None, fresh if merged is None else merged, kept
        )
        at = time.time()
        learner.set("quick_seen", seen)
        changes = result.pop("changes")
        learner.set("quick_last", result | {"at": at, "consolidated": merged is not None})
        learner.set("quick_checked", {"at": at, "nothing_new": not any(changes.values())})
        examples = []
        if changes["added"]:
            try:
                examples = await self.quick_examples(model, before_block, changes["added"])
            except Exception as exc:  # Examples illustrate; the prompt stands.
                logger.warning("quick_examples_failed %s: %s", model, exc)
        await asyncio.to_thread(quick.record, learner, result | {"changes": changes}, at, examples)
        if result["added"]:
            learner.event("quick", f"{result['added']} new lessons in the prompt")
        return result

    async def quick_examples(self, model, before_block, added):
        """Before and after for up to quick.EXAMPLES new preferences: the
        moment each came from, answered by the served model with the lessons
        from before this Quick Dream and with the new ones. None of this goes
        through the proxy, so it is not recorded. Skipped (empty) when the
        learner's model is not loaded."""
        import httpx

        from . import quick

        learner = self.installation_store.scoped(model)
        version = await asyncio.to_thread(learner.version)
        identifier = (version or {}).get("identifier") or model
        loaded = await self.loaded_identifiers()
        if loaded is not None and identifier not in loaded:
            logger.info("quick_examples_skipped %s: model %s not loaded", model, identifier)
            return []
        activity = (getattr(self, "reading_activity", None) or {}).get(model)
        if activity is not None:
            activity["stage"] = "compare"
        after_block = quick.block(quick.active(learner))
        lessons = {lesson["text"]: lesson for lesson in quick.active(learner)}
        out = []
        async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
            # Every new preference, rules or not, is replayed at the moment it
            # came from; one is shown only if the action changed and the
            # model judges the change follows it. The rest say "no visible change".
            shown = 0
            for text in added[: quick.TRIED]:
                if shown >= quick.EXAMPLES:
                    break
                lesson = lessons.get(text)
                found = lesson and await asyncio.to_thread(quick.moment, learner, lesson)
                if not found:
                    logger.info("quick_example_no_moment %s: %s", model, text[:80])
                    continue
                (url, served), server = self.serving_endpoint(version, identifier), self.installation_config.upstream
                before, after, differs = await quick.compare(
                    client, url, served, found["request"], before_block, after_block, server=server,
                )
                verdict = await quick.judge(client, url, served, text, before, after, server=server) if differs else None
                changed = differs and verdict == "after"
                shown += changed
                logger.info("quick_example %s: differs=%s verdict=%s %s", model, differs, verdict, text[:60])
                out.append({"lesson": text, "asked": found["asked"], "feedback": found["feedback"],
                            "said_by": found.get("said_by", "user"), "before": before, "after": after,
                            "changed": changed, "differs": differs, "verdict": verdict})
        return out

    async def sync_served_contexts(self):
        """Agents follow the model server's context setting (checked every 30s)."""
        now = time.monotonic()
        if now - getattr(self, "contexts_checked", -1e9) < 30:
            return
        self.contexts_checked = now
        read = getattr(self.upstream, "served_contexts", None)
        if read is None:
            return
        try:
            served = await read()
        except Exception as exc:  # A busy or restarting server is not an error.
            logger.debug("served_contexts_unavailable %s", exc)
            return
        from . import agents
        from .models.names import catalog

        # Remember only what the user chose; agents still follow what is served.
        chosen = getattr(self.upstream, "user_contexts", served)
        seen = self.installation_store.get("served_context_ids", {}) or {}
        if {**seen, **chosen} != seen:
            self.installation_store.set("served_context_ids", {**seen, **chosen})
        names = catalog(self.installation_config, self.installation_store)
        found = {names[k]: v for k, v in served.items() if k in names}
        merged = {**self.installation_config.served_contexts, **found}
        # Agent files are checked once per start even when nothing changed,
        # so a limit saved by an earlier run always reaches the agents.
        if merged == self.installation_config.served_contexts and getattr(
            self, "limits_synced", False
        ):
            return
        self.limits_synced = True
        if merged != self.installation_config.served_contexts:
            self.installation_config.served_contexts = merged
            self.installation_config.save(self.root)
        changed = await asyncio.to_thread(
            agents.refresh_limits, self.installation_config, root=self.root
        )
        logger.info("served_contexts %s · agents updated %s", merged, len(changed))

    # An idle service looks at the recordings' space this often: the check is
    # a few PRAGMA reads, and space freed since waits at most this long.
    TIDY_SECONDS = 600.0

    async def tidy_recordings(self):
        """In an idle gap with nothing else running (no worker, reading or
        request): the recordings' upkeep (recordings.tidy: the size limit,
        then freed space back to the disk), off the event loop. Never holds
        the admission lock, so a request is never held."""
        now = time.monotonic()
        if now - getattr(self, "tidied_at", -self.TIDY_SECONDS) < self.TIDY_SECONDS:
            return
        if self.process or self.reading_busy() or any(self.in_flight.values()):
            return
        self.tidied_at = now
        from .recordings import limit_bytes, tidy

        try:
            await asyncio.to_thread(
                tidy, self.installation_store, limit_bytes(self.installation_config)
            )
        except Exception:
            logger.exception("recordings_tidy_failed")

    async def tick(self):
        from .models import registry

        if self.returning:
            return  # A version is being rebuilt; no other work beside it.
        await self.sync_served_contexts()
        if self.process:
            if self.active_model:
                row = registry.find(self.installation_store, self.active_model)
                if row and not row["enabled"]:
                    self.interrupt("learning off")
            if self.process.returncode is not None:
                async with self.lock:
                    await self.finish()
            else:
                await super().tick()
            return
        gaps = gap_history(
            self.installation_store.traffic_times(
                self.installation_config.idle_history_limit
            )
        )
        threshold = idle_threshold(
            gaps,
            work_unit_seconds(self.config, self.store.get("calibration")),
            self.installation_config,
        )
        self.idle_threshold_seconds = self.idle_wait(gaps)
        requested = self.installation_store.get("requested_sleep", False)
        asked = self.installation_store.get("requested_model")
        held = self.installation_store.scoped(asked).get("memory_wait") if requested and asked else None
        if held and time.time() < held.get("retry_after", 0):
            # A pressed Deep Dream waiting for memory: not before its time,
            # unless clearly more memory is free now.
            try:
                free = (await asyncio.to_thread(self.platform.memory)).available
            except Exception:  # noqa: BLE001
                free = 0
            if not held.get("available") or free < held["available"] * 1.3 + 1e9:
                self.reason = "Waiting for free memory"
                return
        if time.monotonic() - self.last_traffic < threshold and not requested:
            self.reason = "Waiting for an idle gap"
            return
        await self.tidy_recordings()
        await self.review_replies()
        quiet = time.monotonic() - self.last_traffic
        # Reading happens only inside a Dream. After the Quick quiet time, an
        # automatic Quick Dream: the newest unread conversations for at most
        # quick_read_seconds, then the prompt lessons (a backlog is worked
        # through by successive Quick Dreams and by Deep Dream). After the Deep
        # quiet time (or when Deep Dream was pressed), Deep Dream reads what is
        # left and writes practice; training may then start.
        deep = requested or quiet >= self.installation_config.deep_quiet_seconds
        if deep:
            # A requested Deep Dream reads its own model even with auto-dream
            # off ("a Dream already started finishes"): after a restart its
            # background reading is gone and this is the only reader (live,
            # Inkling on Modal: 7 of 12 conversations skipped, no practice,
            # straight to measuring).
            await self.read_sessions(practice=True, only=asked if requested else None)
        elif quiet >= self.installation_config.quick_quiet_seconds:
            await self.read_sessions(
                quick=True, seconds=self.installation_config.quick_read_seconds
            )
        if self.reading_busy():
            # Reading and practice talk to the model server; measuring,
            # training and swapping park or replace it. A pressed Deep Dream
            # reads in the background (read_sessions_now), so the step here
            # returns at once: without this wait the measuring child started
            # beside it and the reader timed out (live, Inkling on Modal:
            # "stopped while reading · ReadTimeout").
            self.reason = "Reading your conversations"
            return
        await self.discover_models()
        for row in registry.entries(self.installation_store):
            scoped = self.installation_store.scoped(row["id"])
            if row["enabled"] and any(n["status"] == "ready" for n in scoped.nights()):
                self.select(row["id"])
                await super().tick()
                if self.process or self.state == "ready":
                    return
        if not getattr(self, "verdicts_rechecked", False):
            # Once per service start: verdicts from older estimators or an
            # unfinished calibration must not keep fitting samples out.
            self.verdicts_rechecked = True
            from .config import Config
            from .sleep.calibrate import release_fitting_unfit

            for row in registry.entries(self.installation_store):
                scoped = self.installation_store.scoped(row["id"])
                try:
                    release_fitting_unfit(
                        Config.for_model(self.root, row["id"]),
                        scoped,
                        scoped.get("calibration"),
                    )
                except (OSError, ValueError, KeyError) as exc:
                    logger.warning("verdict_recheck_failed %s: %s", row["id"], exc)
        under_way = any(
            n["status"] in {"interrupted", "running", "ready"}
            for row in registry.entries(self.installation_store)
            for n in self.installation_store.scoped(row["id"]).nights()
        )
        model = await self.next_model()
        if model is not None and not deep and not under_way:
            # A new Deep Dream starts only in a long quiet period, or when
            # asked; one under way resumes in any pause.
            self.reason = "Waiting for a long quiet period"
            self.active_model = None
            return
        if (
            model is not None
            and self.installation_store.get("requested_model") != model
            and await asyncio.to_thread(
                self.memory_held, self.installation_store.scoped(model).get("memory_hold")
            )
        ):
            # Its last automatic run was stopped for memory: not before its
            # retry time (status.memory_hold says when). Dream skips it.
            self.reason = "Waiting for free memory"
            self.active_model = None
            return
        if model is None:
            from .status import experience_wait

            # Nothing is waiting on memory when nothing can run: an old memory
            # stop would otherwise keep saying "Waiting for memory" while the
            # real wait is for conversations.
            for row in registry.entries(self.installation_store):
                if "calibration paused · memory" in (row.get("reason") or ""):
                    registry.support(self.installation_store, row["id"], "pending")
            self.reason = await asyncio.to_thread(
                lambda: experience_wait(
                    self.installation_config,
                    self.installation_store,
                    registry.summaries(
                        self.installation_config, self.installation_store
                    ),
                )
            )
            self.active_model = None
            return
        platform = self.platform
        if self.config.backend == "mlx" and platform.name == "linux":
            from .memory import TrainingPlatform

            platform = TrainingPlatform(platform, self.config)
        try:
            kinds = {a.kind for a in await asyncio.to_thread(platform.accelerators)}
        except OSError as exc:
            self.reason = str(exc)
            self.active_model = None
            return
        if kinds <= {"cpu"} and not (
            self.config.force_cpu_training
            or self.config.torch_options.get("force_cpu", False)
        ):
            registry.support(
                self.installation_store,
                model,
                "unsupported",
                "No training accelerator detected; CPU training is disabled by policy",
            )
            self.active_model = None
            return
        asked = self.installation_store.get("requested_model") == model
        nights = len(self.installation_store.scoped(model).nights())
        await super().tick()
        if self.process:
            # A pressed Deep Dream holds on until a night starts (see finish).
            self.launch = {"model": model, "requested": asked, "nights": nights, "at": time.time()}
        if not self.process and self.state != "ready":
            self.active_model = None

    async def return_to(self, night, model=None):
        from .models.registry import resolve

        model = resolve(self.installation_store, model)
        await self.wake()
        if model:
            self.select(model)
        try:
            await super().return_to(night)
            # The weights that held absorbed lessons are not in use now: the
            # next Quick Dream may put those lessons back in the prompt.
            self.store.set("trained_signals", [])
            self.store.set("absorbed_signals", [])
            self.store.set("quick_seen", None)
        finally:
            self.active_model = None
