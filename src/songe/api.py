"""Local interface contract shared by the browser, terminal and CLI.

Optional v2 tables are read as recorded. This module never derives scoreboard
values from training metrics and never migrates the worker's database.
"""

import asyncio
import hashlib
import ipaddress
import json
import logging
import math
import shutil
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import fields, replace
from pathlib import Path
from urllib.parse import urlparse

import httpx
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from . import NAME, __version__
from .config import Config
from .interface_copy import terse
from .log import SCHEMA, Store


def interface_data_dir():
    from .platform import data_dir

    return data_dir()


class ReadOnlyStore(Store):
    """No mkdir, chmod, schema migration, journal change or disk write."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.path = self.root / "experience.sqlite3"

    @contextmanager
    def connect(self):
        if self.path.exists():
            db = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        else:
            # An uninitialized installation has an empty in-memory view only.
            db = sqlite3.connect(":memory:")
            db.executescript(SCHEMA)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        try:
            yield db
        finally:
            db.close()


def rows(store, table, order):
    # All identifiers are internal constants. Missing v2 tables mean waiting,
    # not zero measured errors or zero drift.
    with store.connect() as db:
        if not db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone():
            return []
        return [
            dict(row) for row in db.execute(f"SELECT * FROM {table} ORDER BY {order}")
        ]


def model_scope(store, model):
    """The UI names models by their served id (songe/...); work is stored
    under the registry key (publisher/...). Both resolve to the same scope."""
    if not model:
        return store
    from .models.registry import find

    row = find(store, model)
    return store.scoped(row["id"] if row else model)


def restoring(store):
    """A version being rebuilt for a return: which one and how far along."""
    current = store.get("restoring")
    if not current:
        return None
    from .models.library import model_name
    from .sleep.fuse import PROGRESS

    try:
        fraction = json.loads((store.root / PROGRESS).read_text(encoding="utf-8"))[
            "fraction"
        ]
    except (OSError, ValueError, KeyError, TypeError):
        fraction = 0.0
    return {
        "version": current.get("version"),
        "model": model_name(current["model"]) if current.get("model") else "",
        "fraction": fraction,
    }


def settings(config, store):
    return {
        "model": config.model,
        "upstream": config.upstream,
        "schedule_default_seconds": config.idle_default_seconds,
        "replace_loaded_model": config.replace_loaded_model,
    }


def snapshot(config, store, scheduler=None, *, live=False, release=None, lean=False):
    from .learning_summary import summaries as learning_summaries
    from .models import library, registry
    from .status import activity
    from .upgrade import RUNNING_INSTALL, install_token, notice

    release = release or {
        "service_version": __version__,
        "service_install": RUNNING_INSTALL,
    }

    def deleted(model):
        # Deleted from the model library, never trained and Off: that is
        # history, not a choice, so it is not listed.
        row = registry.find(store, model["id"])
        recorded = ((row or {}).get("config") or {}).get("model_ref") or {}
        library_dir = Path(config.models_dir) if config.models_dir else None
        return (
            bool(recorded.get("path"))
            and library_dir is not None
            and Path(recorded["path"]).is_relative_to(library_dir)
            and not Path(recorded["path"]).exists()
            and model.get("learning_level") == "off"
            and not model.get("last_trained")
        )

    models = [m for m in library.summaries(config, store, scheduler) if not deleted(m)]
    nights = store.nights()
    scoreboard = rows(store, "scoreboard", "day, model_version")
    # lean: the learning page reads only status, cycle and models; the
    # classic dashboard's learning summary and pipeline metrics are skipped.
    learning = {} if lean else learning_summaries(store, models, nights, scoreboard)
    models = [
        m
        | {
            "learning": learning.get(m["id"]),
            "pipeline": {} if lean else model_scope(store, m["id"]).pipeline(),
        }
        for m in models
    ]
    for model in models:
        for key in ("reason", "last_result"):
            if model.get(key):
                model[key] = terse(model[key])
    from .models.names import catalog

    aliases = catalog(config, store)
    current = store.version()
    previous = store.version(store.get("previous")) if store.get("previous") else None
    for night in nights:
        if night.get("record", {}).get("reason"):
            night["reason_label"] = terse(night["record"]["reason"])
        model = night.get("model_id") or store.get("primary_model")
        scoped = store.scoped(model) if model else store
        serving = scoped.version()
        if model:
            night["model"] = aliases.get(model, model)
        night["can_return"] = bool(
            night["status"] == "accepted"
            and any(
                v["night"] == night["id"]
                for v in (
                    scoped.versions() if model else rows(store, "versions", "created")
                )
            )
            and (not serving or night["id"] != serving["night"])
        )
    slices = rows(store, "slices", "started")
    active_slice = next((s for s in reversed(slices) if s["ended"] is None), None)
    active_night = next(
        (n for n in reversed(nights) if n["status"] in {"running", "ready"}), None
    )
    training_model = getattr(scheduler, "active_model", None) if live else None
    training = None
    if live and (active_night or getattr(scheduler, "process", None)):
        training_model = training_model or (active_night or {}).get("model")
        training = {
            "model": training_model,
            "phase": "calibrate"
            if getattr(scheduler, "calibrating", False)
            else (active_night or {}).get("phase", "record"),
            "progress": (active_night or {}).get("progress", 0),
            "data_progress": ((active_night or {}).get("record") or {}).get(
                "data_progress"
            ),
        }
        if training["phase"] == "calibrate":
            scoped = store.scoped(training_model) if training_model else store
            training["data_progress"] = scoped.get("calibration_progress")
        progress = training["data_progress"]
        if (
            progress
            and progress.get("phase") == "train"
            and training["phase"] in {"train", "distill"}
        ):
            accumulation = (
                registry.model_config(config, store, training_model).accumulation
                if training_model
                else config.accumulation
            )
            training["step"] = progress.get(
                "step", math.ceil(progress["completed"] / accumulation)
            )
            training["total_steps"] = progress.get(
                "total_steps", math.ceil(progress["total"] / accumulation)
            )
    if training_model:
        training_model = aliases.get(training_model, training_model)
        if training:
            training["model"] = training_model
    paused = store.get("paused", False)
    state = "awake"
    if live and not paused:
        if active_night and active_night["status"] == "ready":
            state = "ready"
        elif active_slice or active_night or training:
            state = "settling"
    reason = (
        getattr(scheduler, "reason", "experience pending")
        if live
        else "service stopped"
    )
    if paused:
        reason = "Paused" if live else "paused · service stopped"
    status = activity(
        config,
        store,
        scheduler,
        training,
        models,
        live=live,
        ready=active_night
        if active_night and active_night["status"] == "ready"
        else None,
    )
    status["model"] = aliases.get(status["model"], status["model"])
    # Order by registry ids before aliasing; grouping keeps first appearance.
    models = library.order_models(models, store.get(registry.ORDER_KEY, []))
    models = [m | {"id": aliases.get(m["id"], m["id"])} for m in models]
    pipelines = {m["id"]: m["pipeline"] for m in models}
    models = library.group_models(models)
    return {
        "schema": 1,
        "installation_id": store.get("installation_id"),
        "product": NAME,
        **(release if live else {}),
        "service_replaced": live and release["service_install"] != install_token(),
        "upgrade_notice": notice(
            store.root,
            version=release["service_version"] if live else None,
            token=release["service_install"] if live else None,
        ),
        "model": config.public_model
        if config.model
        else "models"
        if config.exposed_models
        else "uninitialized",
        "models": models,
        "pipeline": pipelines,
        "training_model": training_model,
        "training": training,
        "state": state,
        "service_running": live,
        "idle_seconds": int(max(0, time.monotonic() - scheduler.last_traffic))
        if live and scheduler and hasattr(scheduler, "last_traffic")
        else None,
        "version": current,
        "paused": paused,
        "reason": terse(reason),
        "activity": status,
        # While a Dream trains or measures: how busy the GPU is (one line).
        "gpu": gpu_line(scheduler) if live else None,
        "restoring": restoring(store),
        "slice": active_slice,
        "data_progress": training["data_progress"]
        if training
        else (nights[-1].get("record") or {}).get("data_progress")
        if nights
        else None,
        "nights": nights,
        "scoreboard": scoreboard,
        "settings": settings(config, store),
        "can_undo": previous is not None and len(models) <= 1,
        "teacher_gain": store.get("teacher_gain"),
        "calibration": store.get("calibration"),
        "cycle": {"models": cycle_section(config, store, models, scheduler)},
    }


def dream_activity(store, scheduler):
    """In-progress Dream work by registry key (the UI uses the served id):
    reading and practice, or calibrating before the first training."""
    activity = dict(getattr(scheduler, "reading_activity", None) or {})
    measuring = getattr(scheduler, "active_model", None)
    if getattr(scheduler, "calibrating", False) and measuring and measuring not in activity:
        # Calibrating before the first training: part of this Dream.
        activity[measuring] = {
            "stage": "train",
            "started_at": getattr(scheduler, "calibration_started", None),
            "trigger": "idle",
            "calibration": store.scoped(measuring).get("calibration_progress"),
        }
    return activity


def gpu_line(scheduler):
    if not (scheduler and getattr(scheduler, "process", None)):
        return None
    from .platform.common import gpu_usage

    return gpu_usage()


def cycle_section(config, store, models, scheduler=None):
    """The Learning cycle per model (docs/CONTRACTS.md); a model whose record
    cannot be built is null rather than taking the dashboard down."""
    from .cycle import model_learning
    from .models.registry import find, model_config

    activity = dream_activity(store, scheduler)
    out = {}
    for m in models:
        row = find(store, m["id"])
        if row is None:
            # Listed by the model server but never registered (live: LM
            # Studio's embedding model): nothing to learn, no record, and no
            # error logged every refresh.
            continue
        try:
            out[m["id"]] = model_learning(
                model_config(config, store, m["id"]),
                model_scope(store, m["id"]),
                m["id"],
                activity.get(row["id"] if row else m["id"]),
            )
        except Exception:  # noqa: BLE001 - one model's record never hides the rest
            logging.getLogger("songe").exception("cycle_record_failed")
            out[m["id"]] = None
    return out


def state_sections(data):
    """Independent revision domains, also used for the browser's first paint."""
    models = data["models"]

    def inventory(model):
        result = {
            k: v
            for k, v in model.items()
            if k
            not in {
                "learning",
                "pipeline",
                "versions",
                "version",
                "last_result",
                "last_update",
                "last_status",
            }
        }
        if "files" in result:
            result["files"] = [
                {k: v for k, v in f.items() if k != "pipeline"} for f in result["files"]
            ]
        return result

    return {
        "status": {
            k: v
            for k, v in data.items()
            if k
            not in {
                "models",
                "nights",
                "scoreboard",
                "teacher_gain",
                "calibration",
                "cycle",
            }
        },
        "cycle": data.get("cycle", {"models": {}}),
        "models": {"models": [inventory(m) for m in models]},
        "learning": {
            "models": [
                {
                    "id": m["id"],
                    "name": m.get("name"),
                    "learning": m.get("learning"),
                }
                for m in models
            ],
            "scoreboard": data["scoreboard"],
            "teacher_gain": data["teacher_gain"],
        },
        "versions": {
            "models": [
                {
                    "id": m["id"],
                    "name": m.get("name"),
                    "versions": m.get("pipeline", {}).get(
                        "versions", m.get("versions", [])
                    ),
                }
                for m in models
            ],
            "nights": [
                n for n in data["nights"] if n["status"] not in {"running", "ready"}
            ],
        },
    }


def section_revisions(sections):
    return {
        key: hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:20]
        for key, value in sections.items()
    }


def progress_text(progress):
    if not progress:
        return ""
    label = "Scoring" if progress["phase"] == "score" else "Learning"
    text = f"{label}: {progress['completed']}/{progress['total']} {progress['unit']}"
    eta = progress.get("eta_seconds")
    if eta is not None and eta > 0:
        text += f" · ETA {eta:.0f}s"
    return text


def set_model_learning(store, payload):
    from .models import registry
    from .models.library import materialize

    if "level" in payload:
        if payload["level"] not in registry.LEARNING_LEVELS:
            raise ValueError("Learning level: off | new | all")
    elif not isinstance(payload.get("enabled"), bool):
        raise ValueError("enabled: true | false")
    materialize(Config.load(store.root), store, payload.get("model"))
    if "level" in payload:
        changed = registry.set_learning(store, payload.get("model"), payload["level"])
    else:
        changed = registry.enable(store, payload.get("model"), payload.get("enabled"))
    registry.mark_chosen(store, payload.get("model"))
    return changed


def forget_experience(store):
    from .cli import require_stopped_worker

    store.set("paused", True)
    store.set("requested_sleep", False)
    store.set("requested_model", None)
    with require_stopped_worker(store.root):
        store.forget()
        with store.connect() as db:
            retained = [dict(r) for r in db.execute("SELECT * FROM versions")]
        keep = {
            Path(v[k]).resolve()
            for v in retained
            for k in ("adapter", "artifact", "training_adapter")
            if v.get(k)
        }
        from .models.convert import forget_derived

        configured_base = Config.load(store.root).model
        if configured_base:
            keep.add(Path(configured_base).resolve())
        from .artifacts.storage import artifact_roots, collect_scratch, prune_roots
        from .models import registry
        from .sleep.calibrate import collect_calibration

        for model in registry.entries(store):
            base = model["config"].get("model")
            if base:
                keep.add(Path(base).resolve())
        forget_derived(store.root, keep)
        collect_calibration(store.root, store, leased=True)
        collect_scratch(store)
        for path in (
            path
            for directory in artifact_roots(store.root)
            for path in (directory / "versions").glob("night-*/*")
        ):
            if path.name != "songe_work.json" and not any(
                k == path.resolve() or k.is_relative_to(path.resolve()) for k in keep
            ):
                shutil.rmtree(path) if path.is_dir() else path.unlink()
        store.set("deep_rebuild_requested", True)
        # Cached gate references are prompts from recorded conversations.
        shutil.rmtree(store.root / "gate-references", ignore_errors=True)
        from .sleep.cleanup import collect_work

        collect_work(store, Config.load(store.root))
        prune_roots(store)


def return_target(store, payload):
    if payload.get("version"):
        target = store.version(payload["version"])
    elif "night" in payload:
        target = next(
            (v for v in store.versions() if v["night"] == payload["night"]), None
        )
    else:
        target = store.version(store.get("previous")) if store.get("previous") else None
    if not target:
        raise ValueError("Version not retained")
    if target["id"] == store.get("current"):
        raise RuntimeError("Version already current")
    return target


def local_host(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def guard(request, mutation=False):
    origin = request.headers.get("origin")
    if origin and (
        origin != str(request.base_url).rstrip("/")
        or not local_host(urlparse(origin).hostname)
    ):
        return JSONResponse({"error": "Origin rejected"}, 403)
    if (
        mutation
        and request.headers.get("content-type", "").split(";")[0].strip().lower()
        != "application/json"
    ):
        return JSONResponse({"error": "JSON required"}, 415)
    if (
        not local_host(request.url.hostname)
        or not request.client
        or not local_host(request.client.host)
    ):
        return JSONResponse({"error": "Loopback required"}, 403)
    return None


async def json_object(request):
    value = await request.json()
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def api_routes(config, store, scheduler=None, *, release=None):
    from .upgrade import RUNNING_INSTALL, install_token

    release = release or {
        "service_version": __version__,
        "service_install": RUNNING_INSTALL,
    }
    lock = asyncio.Lock()

    async def read(request):
        if rejected := guard(request):
            return rejected
        # CLI catch-up and restart health checks must not scan model libraries.
        if (
            request.url.path == "/api/state"
            and request.query_params.get("service") == "1"
        ):
            return JSONResponse(
                {
                    "schema": 1,
                    "installation_id": store.get("installation_id"),
                    "service_running": True,
                    "service_replaced": release["service_install"] != install_token(),
                    "requests_in_flight": sum(scheduler.in_flight.values())
                    if scheduler
                    else 0,
                    **release,
                },
                headers={"Cache-Control": "no-store"},
            )
        # ?view=cycle: only what the learning page shows (status, cycle, models).
        lean = request.query_params.get("view") == "cycle"
        try:
            data = await asyncio.to_thread(
                snapshot, config, store, scheduler, live=True, release=release, lean=lean
            )
        except sqlite3.Error as exc:
            return JSONResponse({"error": f"Local store unavailable: {exc}"}, 503)
        path = request.url.path
        if path == "/api/state":
            sections = state_sections(data)
            if lean:
                sections = {k: sections[k] for k in ("status", "cycle", "models")}
            revisions = section_revisions(sections)
            etag = '"' + section_revisions({"state": data})["state"] + '"'
            headers = {"ETag": etag, "Cache-Control": "private, no-cache"}
            if etag in [
                tag.strip()
                for tag in request.headers.get("if-none-match", "").split(",")
            ]:
                return Response(status_code=304, headers=headers)
            if "revisions" in request.query_params:
                try:
                    known = json.loads(request.query_params["revisions"])
                    if not isinstance(known, dict):
                        raise ValueError
                except (ValueError, TypeError):
                    return JSONResponse(
                        {"error": "revisions must be a JSON object"}, 400
                    )
                changed = {
                    key: value
                    for key, value in sections.items()
                    if known.get(key) != revisions[key]
                }
                if not changed:
                    return Response(status_code=304, headers=headers)
                return JSONResponse(
                    {"revisions": revisions, "sections": changed}, headers=headers
                )
            return JSONResponse(data | {"revisions": revisions}, headers=headers)
        if path == "/api/nights":
            data = data["nights"]
        elif "id" in request.path_params:
            data = next(
                (n for n in data["nights"] if n["id"] == request.path_params["id"]),
                None,
            )
            if data is None:
                return JSONResponse({"error": "Night not found"}, 404)
        elif path == "/api/scoreboard":
            data = data["scoreboard"]
        elif path == "/api/settings":
            data = data["settings"]
        return JSONResponse(data, headers={"Cache-Control": "no-store"})

    async def control(request):
        if rejected := guard(request, True):
            return rejected
        identity = request.headers.get("x-songe-installation")
        if identity is not None and identity != store.get("installation_id"):
            return JSONResponse({"error": "Songe installation mismatch"}, 409)
        try:
            payload = await json_object(request)
            action = payload.get("action")
            if request.url.path == "/api/return":
                action = "undo"
            if action not in {
                "pause",
                "resume",
                "sleep_now",
                "dream",
                "undo",
                "forget",
                "shutdown",
                "model",
                "auto_dream",
                "remove_lesson",
                "cancel",
                "seen",
            }:
                raise ValueError("Unknown control action")
            if action == "forget" and payload.get("confirm") is not True:
                raise ValueError("Forget: confirm=true required")
            async with lock:
                if action == "shutdown":
                    if identity != store.get("installation_id"):
                        return JSONResponse(
                            {"error": "Songe installation ID required"}, 409
                        )
                    if not scheduler or not getattr(scheduler, "shutdown", None):
                        raise RuntimeError(
                            "Shutdown unavailable · stop service before uninstall"
                        )
                    scheduler.shutdown()
                elif action == "model":
                    from .models.registry import resolve

                    changed = await asyncio.to_thread(
                        set_model_learning, store, payload
                    )
                    model = resolve(store, payload.get("model"))
                    if (
                        changed
                        and scheduler
                        and getattr(scheduler, "active_model", None) == model
                    ):
                        scheduler.interrupt("learning level changed")
                elif action == "seen":
                    # The user looked at this Dream's report.
                    from .cycle import mark_seen
                    from .models.registry import resolve

                    model = resolve(store, payload.get("model"))
                    if not isinstance(payload.get("id"), str):
                        raise ValueError("id: the Dream's id")
                    if payload["id"] == "stopped":  # the "Deep Dream stopped" note, closed
                        await asyncio.to_thread(model_scope(store, model).set, "dream_stopped", None)
                        return JSONResponse({"ok": True})
                    if payload["id"] == "measured":  # the "measured this machine" note, closed
                        await asyncio.to_thread(model_scope(store, model).set, "machine_measured", None)
                        return JSONResponse({"ok": True})
                    await asyncio.to_thread(mark_seen, model_scope(store, model), payload["id"])
                    return JSONResponse({"ok": True})
                elif action == "cancel":
                    # Throw the Dream under way away (pause only suspends it).
                    if not hasattr(scheduler, "cancel"):
                        return JSONResponse({"error": "Scheduler unavailable"}, 503)
                    from .models.registry import resolve

                    model = resolve(store, payload.get("model"))
                    discarded = await scheduler.cancel(model)
                    if discarded:
                        # Cancelling ends it all: a pause that held this Dream
                        # goes too, so the button is back to Dream.
                        await asyncio.to_thread(store.set, "paused", False)
                    return JSONResponse({
                        "ok": True,
                        "message": "Dream cancelled · what it practised waits for the next Deep Dream"
                        if discarded else "No Dream to cancel",
                    })
                elif action == "remove_lesson":
                    # Out of the prompt for new sessions, and never back.
                    from . import quick
                    from .models.registry import resolve

                    model = resolve(store, payload.get("model"))
                    await asyncio.to_thread(
                        quick.remove, model_scope(store, model), payload.get("id")
                    )
                    return JSONResponse({"ok": True, "message": "Lesson removed"})
                elif action == "auto_dream":
                    # A Dream already under way finishes either way.
                    from .models.library import materialize
                    from .models.registry import resolve, set_auto_dream
                    from .wording import auto_dream_message

                    on = payload.get("on")
                    if not isinstance(on, bool):
                        raise ValueError("on: true | false")
                    await asyncio.to_thread(
                        materialize, Config.load(store.root), store, payload.get("model")
                    )
                    model = resolve(store, payload.get("model"))
                    await asyncio.to_thread(set_auto_dream, store, model, on)
                    return JSONResponse(
                        {"ok": True, "auto_dream": on, "message": auto_dream_message(config, on)}
                    )
                elif action in {"pause", "resume"}:
                    await asyncio.to_thread(store.set, "paused", action == "pause")
                    if action == "pause" and scheduler:
                        scheduler.interrupt("paused")
                elif action == "sleep_now":
                    await asyncio.to_thread(store.set, "requested_sleep", True)
                elif action == "dream":
                    # One model, now: read its new replies whatever the
                    # amount; train too when everything that waits on is met.
                    if not hasattr(scheduler, "dream"):
                        return JSONResponse({"error": "Scheduler unavailable"}, 503)
                    from .models.registry import resolve

                    model = resolve(store, payload.get("model"))
                    kind = payload.get("kind", "deep")
                    if kind not in {"quick", "deep"}:
                        raise ValueError("kind: quick | deep")
                    return JSONResponse(
                        {"ok": True, "kind": kind} | await scheduler.dream(model, kind)
                    )
                elif action == "forget":
                    await asyncio.to_thread(store.set, "paused", True)
                    if scheduler:
                        scheduler.interrupt()
                    await asyncio.to_thread(forget_experience, store)
                else:
                    if not scheduler:
                        return JSONResponse({"error": "Scheduler unavailable"}, 503)
                    from .models.registry import resolve

                    model = resolve(store, payload.get("model"))
                    selected = store.scoped(model) if model else store
                    target = return_target(selected, payload)
                    await scheduler.return_to(
                        target["night"], **({"model": model} if model else {})
                    )
        except (ValueError, TypeError, KeyError) as exc:
            return JSONResponse({"error": terse(exc)}, 400)
        except (RuntimeError, OSError, sqlite3.Error, httpx.HTTPError) as exc:
            return JSONResponse({"error": terse(exc)}, 409)
        return JSONResponse({"ok": True})

    async def patch(request):
        if rejected := guard(request, True):
            return rejected
        try:
            payload = await json_object(request)
            if not payload or set(payload) - {
                "model",
                "upstream",
                "schedule_default_seconds",
                "replace_loaded_model",
            }:
                raise ValueError(
                    "Settings: model, upstream, schedule_default_seconds, replace_loaded_model"
                )
            async with lock:
                changes = {
                    k: v
                    for k, v in payload.items()
                    if k not in {"schedule_default_seconds", "replace_loaded_model"}
                }
                if any(
                    not isinstance(v, str) or not v.strip() for v in changes.values()
                ):
                    raise ValueError("Model and upstream: nonempty strings required")
                if any(v != getattr(config, k) for k, v in changes.items()) and (
                    store.version() or store.nights()
                ):
                    return JSONResponse(
                        {
                            "error": "Model/upstream change: fresh data directory required"
                        },
                        409,
                    )
                if scheduler and any(
                    v != getattr(config, k) for k, v in changes.items()
                ):
                    return JSONResponse(
                        {
                            "error": "Serving binding: stopped service + initialization required"
                        },
                        409,
                    )
                if "replace_loaded_model" in payload:
                    if not isinstance(payload["replace_loaded_model"], bool):
                        raise ValueError("replace_loaded_model: true | false")
                    changes["replace_loaded_model"] = payload["replace_loaded_model"]
                candidate = replace(config, **changes)
                candidate.validate()
                saved = store.get("interface_settings", {})
                if "schedule_default_seconds" in payload:
                    seconds = payload["schedule_default_seconds"]
                    if (
                        isinstance(seconds, bool)
                        or not isinstance(seconds, (float, int))
                        or not math.isfinite(seconds)
                        or seconds <= 0
                    ):
                        raise ValueError(
                            "Schedule default: positive finite seconds required"
                        )
                    candidate.idle_default_seconds = seconds
                await asyncio.to_thread(candidate.save, store.root)
                if candidate != config:
                    await asyncio.to_thread(store.invalidate_pending)
                    if scheduler:
                        scheduler.interrupt()
                await asyncio.to_thread(store.set, "interface_settings", saved)
                for field in fields(config):
                    setattr(config, field.name, getattr(candidate, field.name))
        except (ValueError, TypeError, KeyError) as exc:
            return JSONResponse({"error": terse(exc)}, 400)
        except (OSError, sqlite3.Error) as exc:
            return JSONResponse({"error": terse(exc)}, 409)
        return JSONResponse(settings(config, store))

    async def flow_read(request):
        if rejected := guard(request):
            return rejected
        from .flow import flow

        model = request.query_params.get("model")
        try:
            data = await asyncio.to_thread(
                flow, model_scope(store, model), every_model=not model
            )
        except sqlite3.Error as exc:
            return JSONResponse({"error": f"Local store unavailable: {exc}"}, 503)
        return JSONResponse(data, headers={"Cache-Control": "no-store"})

    async def cycles_read(request):
        if rejected := guard(request):
            return rejected
        from .cycle import cycles, full_cycle, night_of
        from .models.registry import model_config

        def reading_cycle(wanted):
            # Reading, practice or calibration in progress: the cycle has
            # its id before its night exists.
            from .cycle import upcoming
            from .models.registry import find

            for key, activity in dream_activity(store, scheduler).items():
                row = find(store, key)
                model = (row or {}).get("config", {}).get("public_model") or key
                scoped = store.scoped(key)
                record = upcoming(
                    model_config(config, store, key), scoped, model, activity, every=True
                )
                if record["id"] == wanted:
                    return record
            return None

        def load():
            wanted = request.path_params.get("id")
            if wanted is not None:
                night = night_of(wanted)
                with store.connect() as db:
                    row = db.execute(
                        "SELECT model_id FROM nights WHERE id=?", (night,)
                    ).fetchone()
                if row is None:
                    return reading_cycle(wanted)
                model = row[0] or store.get("primary_model")
                scoped = model_scope(store, model)
                return full_cycle(model_config(config, store, model), scoped, model, wanted)
            model = request.query_params.get("model") or store.get("primary_model")
            limit = max(1, min(int(request.query_params.get("limit", 20)), 200))
            return cycles(
                model_config(config, store, model), model_scope(store, model), model, limit
            )

        try:
            found = await asyncio.to_thread(load)
        except ValueError:
            return JSONResponse({"error": "limit must be a number"}, 400)
        if found is None:
            return JSONResponse({"error": "no such cycle"}, 404)
        return JSONResponse(found, headers={"Cache-Control": "no-store"})

    return (
        [Route("/api/flow", flow_read)]
        + [Route("/api/cycles", cycles_read), Route("/api/cycles/{id}", cycles_read)]
        + [
            Route(path, read)
            for path in (
                "/api/state",
                "/api/nights",
                "/api/nights/{id:int}",
                "/api/scoreboard",
                "/api/settings",
            )
        ]
        + [
            Route("/api/control", control, methods=["POST"]),
            Route("/api/return", control, methods=["POST"]),
            Route("/api/settings", patch, methods=["PATCH"]),
        ]
    )


class LocalClient:
    """Use the service when reachable; only connection failures mean offline."""

    def __init__(self, root, config=None):
        self.root = Path(root)
        self.config = config or Config.load(self.root)
        self.url = f"http://127.0.0.1:{self.config.port}"

    def snapshot(self):
        try:
            # A cold library scan can outlast connection admission (lms has
            # two bounded metadata commands); keep the read deadline separate.
            with httpx.Client(
                trust_env=False, timeout=httpx.Timeout(30, connect=1)
            ) as client:
                response = client.get(self.url + "/api/state")
                response.raise_for_status()
                data = response.json()
                if data.get("schema") != 1:
                    raise ValueError("Unsupported local API schema")
                self.verify(data)
                from .upgrade import notice

                data["upgrade_notice"] = notice(
                    self.root,
                    version=data.get("service_version"),
                    token=data.get("service_install"),
                ) or data.get("upgrade_notice")
                if (
                    not data["upgrade_notice"]
                    and data.get("service_version") is not None
                    and data["service_version"] != __version__
                ):
                    data["upgrade_notice"] = (
                        f"Service {data['service_version']} differs from CLI {__version__}"
                        " · fix: songe doctor"
                    )
                return data
        except (httpx.ConnectError, httpx.ConnectTimeout):
            return snapshot(self.config, ReadOnlyStore(self.root))

    def verify(self, data):
        identity = ReadOnlyStore(self.root).get("installation_id")
        if not identity or data.get("installation_id") != identity:
            raise RuntimeError(
                "Songe installation mismatch · data directory / service port"
            )
        return identity

    def control(self, action, **payload):
        try:
            with httpx.Client(trust_env=False, timeout=180) as client:
                state = client.get(self.url + "/api/state")
                state.raise_for_status()
                identity = self.verify(state.json())
                response = client.post(
                    self.url + "/api/control",
                    json={"action": action, **payload},
                    headers={"x-songe-installation": identity},
                )
                if not response.is_success:
                    raise RuntimeError(
                        response.json().get("error", response.reason_phrase)
                    )
                return
        except (httpx.ConnectError, httpx.ConnectTimeout):
            pass
        store = Store(self.root)
        if action == "shutdown":
            return
        elif action in {"pause", "resume"}:
            store.set("paused", action == "pause")
        elif action == "sleep_now":
            store.set("requested_sleep", True)
        elif action == "model":
            from .models import registry

            registry.migrate(self.config, store)
            set_model_learning(store, payload)
        elif action == "undo":
            from .models import registry

            registry.migrate(self.config, store)
            model = registry.resolve(store, payload.get("model"))
            store = store.scoped(model) if model else store
            target = return_target(store, payload)
            from .cli import require_stopped_worker
            from .platform import Rooted, current
            from .sleep.retention import restore_artifact

            selected = (
                registry.model_config(self.config, store, model)
                if model
                else self.config
            )
            with require_stopped_worker(self.root):
                restore_artifact(store, selected, target, Rooted(current(), self.root))
                store.swap(target["id"], store.get("current"))
                from .sleep.swap import collect
                from .upstream import create

                asyncio.run(collect(store, create(selected, self.root), {}))
            store.invalidate_pending()
        elif action == "forget" and payload.get("confirm") is True:
            forget_experience(store)
        else:
            raise ValueError("Unknown or unconfirmed control action")
