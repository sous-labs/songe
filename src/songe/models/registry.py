"""Per-model learners over one local experience log.

Discovery is read-only. Support is established by the existing guarded
conversion/calibration, never by an architecture or parameter-count heuristic.
"""

import json
import time
from dataclasses import asdict, replace
from pathlib import Path

from ..steps import digest
from .names import public_name

# Serving and scheduling policy remain installation-wide. Only these measured
# dimensions and bindings may vary between learners.
MODEL_FIELDS = {
    "model",
    "model_ref",
    "base_identifier",
    "public_model",
    "artifact_kind",
    "context_window",
    "backend",
    "torch_options",
    "max_sequence",
    "response_tokens",
    "token_chunk",
    "accumulation",
    "calibrated_step_seconds",
    "calibrated_score_seconds",
    "calibrated_peak_gb",
    "calibrated_host_gb",
    "calibrated_device_gb",
    "calibration_available",
}

LEARNING_LEVELS = ("off", "new", "all")


def decode(row):
    result = dict(row) | {"config": json.loads(row["config"])}
    result.setdefault("learning_level", "new" if row["enabled"] else "off")
    result.setdefault("learning_cutoff", None)
    result["enabled"] = result["learning_level"] != "off"
    return result


def entries(store):
    with store.connect() as db:
        exists = db.execute(
            "SELECT 1 FROM sqlite_master WHERE name='models'"
        ).fetchone()
        if not exists:
            return []
        return [decode(r) for r in db.execute("SELECT * FROM models ORDER BY rowid")]


def find(store, model):
    with store.connect() as db:
        row = db.execute("SELECT * FROM models WHERE id=?", (model,)).fetchone()
        if row:
            return decode(row)
    rows = entries(store)
    exact = next(
        (row for row in rows if row["config"].get("public_model") == model), None
    )
    if exact or not model:
        return exact
    # The name `songe models` shows (Qwen3.5-4B for .../Qwen3.5-4B-MLX-4bit).
    from .library import model_name

    named = [row for row in rows if model_name(row["id"]).lower() == str(model).lower()]
    # An ambiguous name matches nothing here; materialize explains it.
    return named[0] if len(named) == 1 else None


def model_config(config, store, model):
    return model_config_with_keys(config, store, model)[0]


def model_config_with_keys(config, store, model):
    """Resolve the model config and report which saved keys were used or ignored."""
    row = find(store, model)
    if row is None:
        raise ValueError(f"Unknown model: {model}")
    saved = row["config"]
    overrides = {key: value for key, value in saved.items() if key in MODEL_FIELDS}
    return (
        replace(config, **overrides, model_key=row["id"]),
        frozenset(overrides),
        saved.keys() - MODEL_FIELDS,
    )


def save_config(config, store):
    values = {k: v for k, v in asdict(config).items() if k in MODEL_FIELDS}
    with store.connect() as db:
        db.execute(
            "UPDATE models SET config=? WHERE id=?",
            (json.dumps(values), config.model_key),
        )


def support(store, model, status, reason=""):
    if status not in {"pending", "trainable", "unsupported"}:
        raise ValueError("Invalid trainability status")
    with store.connect() as db:
        row = db.execute("SELECT support FROM models WHERE id=?", (model,)).fetchone()
        db.execute(
            "UPDATE models SET support=?,reason=? WHERE id=?", (status, reason, model)
        )
    if row and row[0] == "pending" and status != "pending":
        # Measured for the first time, on whichever path: the page says how
        # it ended, once (live: a restart mid-measuring skipped the note).
        import time

        store.scoped(model).set("machine_measured", {"at": time.time(), "support": status, "reason": reason})


def mark_chosen(store, model):
    """An explicit user choice; first use no longer switches learning on."""
    row = find(store, model)
    followed = store.get("followed_models", [])
    if row and row["id"] not in followed:
        store.set("followed_models", [*followed, row["id"]])


def enable(store, model, enabled):
    """Compatibility for the previous boolean local control contract."""
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be true or false")
    return set_learning(store, model, "new" if enabled else "off")


def set_learning(store, model, level):
    if level not in LEARNING_LEVELS:
        raise ValueError("Learning level must be off, new, or all")
    if row := find(store, model):
        model = row["id"]
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT * FROM models WHERE id=?", (model,)).fetchone()
        if not row:
            raise ValueError(f"Unknown model: {model}")
        if row["learning_level"] == level:
            if level != "off" and row["support"] == "unsupported":
                db.execute(
                    "UPDATE models SET support='pending',reason=? WHERE id=?",
                    ("Waiting to recheck measured trainability", model),
                )
                return True
            return False
        cutoff = time.time() if level == "new" else row["learning_cutoff"]
        db.execute(
            "UPDATE models SET enabled=?,learning_level=?,learning_cutoff=? WHERE id=?",
            (level != "off", level, cutoff, model),
        )
        scoped = store.scoped(model)
        scoped._invalidate_pending(db)
        # Start a fresh selection window. Versions and undo remain intact, but
        # replay/scoring cursors must not hide older data when choosing 'all'.
        db.execute("DELETE FROM pool WHERE model_id=?", (scoped.model_id,))
        db.execute("DELETE FROM scored_revisions WHERE model_id=?", (scoped.model_id,))
        for key in (
            "experience_watermark",
            "practice_revisions",
            "rejected_practice",
            "score_watermark",
            "unfit_samples",
            "teacher_gain",
        ):
            db.execute("DELETE FROM state WHERE key=?", (scoped.state_key(key),))
        if level != "off" and row["support"] == "unsupported":
            db.execute(
                "UPDATE models SET support='pending',reason=? WHERE id=?",
                ("Waiting to recheck measured trainability", model),
            )
    from ..sleep.lessons import prune_lessons

    prune_lessons(scoped)
    if level != "off":
        bring_to_front(store, model)
    return True


def auto_dream(store, row):
    """Whether Dreams start on their own for this model (in idle gaps).

    Off: nothing starts by itself, but Dream still runs one now and a Dream
    already started finishes in later idle gaps. A model that is not
    learning at all (level off, e.g. only found in the library) is off;
    otherwise on unless the user turned it off."""
    if row["learning_level"] == "off":
        return False
    value = store.scoped(row["id"]).get("auto_dream")
    return True if value is None else bool(value)


def set_auto_dream(store, model, on):
    """Turn auto-dream on or off. Turning it on makes a library model learn
    from new conversations (what it learns from is Songe's decision: new
    practice, with older practice replayed for retention)."""
    row = find(store, model)
    if not row:
        raise ValueError(f"Unknown model: {model}")
    if on and row["learning_level"] == "off":
        set_learning(store, row["id"], "new")
    store.scoped(row["id"]).set("auto_dream", bool(on))
    mark_chosen(store, row["id"])
    return find(store, row["id"])


ORDER_KEY = "model_order"


def bring_to_front(store, model):
    """Turning learning on lists a model first; turning it off never moves it."""
    order = [m for m in store.get(ORDER_KEY, []) if m != model]
    store.set(ORDER_KEY, [model, *order])


def migrate(config, store):
    """Keep legacy ids, pointers, adapters, watermarks and undo in place."""
    for row in entries(store):
        from . import ModelRef
        from .library import is_embedding

        ref = row["config"].get("model_ref", {})
        if ref and is_embedding(
            ModelRef(
                config.upstream,
                row["id"],
                ref.get("format", "unknown"),
                Path(ref.get("path") or row["config"].get("model") or "/missing"),
                ref.get("architecture", ""),
            )
        ):
            if row["support"] != "unsupported" or row["reason"] != "embedding model":
                support(store, row["id"], "unsupported", "embedding model")
            continue
        # Earlier builds marked a memory-stopped calibration as unsupported;
        # that never meant the model cannot train, so measure it again.
        if row.get("support") == "unsupported" and str(
            row.get("reason", "")
        ).startswith(
            (
                "Calibration stopped at measured memory pressure",
                "Guarded calibration failed",
                "RuntimeError: Model fusion failed",
                "RuntimeError: Calibration child failed",
                "ModuleNotFoundError",
            )
        ):
            support(
                store,
                row["id"],
                "pending",
                "calibration paused · memory · retry when idle",
            )
    if not config.model or entries(store):
        return
    model = (
        config.model_ref.get("serving_id")
        or config.base_identifier
        or config.public_model
    )
    register(config, store, model, legacy=True)


def register(config, store, model, ref=None, *, legacy=False):
    from .library import is_embedding

    if not isinstance(model, str) or not model.strip():
        raise ValueError("A nonempty model id is required")
    row = find(store, model)
    if row:
        if ref and not row["config"].get("model_ref"):
            selected = model_config(config, store, model)
            selected.model = str(ref.path)
            selected.model_ref = asdict(ref) | {"path": str(ref.path)}
            selected.save(store.root)
            support(store, model, "pending")
            ensure_base(store, model, ref)
        if ref and is_embedding(ref):
            support(store, model, "unsupported", "embedding model")
        return find(store, model)
    values = {k: v for k, v in asdict(config).items() if k in MODEL_FIELDS}
    if not legacy:
        alias = public_name(model)
        if any(row["config"].get("public_model") == alias for row in entries(store)):
            alias += f"-{digest(model)[:8]}"
        values.update(
            model=str(ref.path) if ref else "",
            model_ref=asdict(ref) | {"path": str(ref.path)} if ref else {},
            base_identifier=model,
            public_model=alias,
            artifact_kind="",
            calibration_available=False,
            calibrated_peak_gb=0,
            calibrated_host_gb=0,
            calibrated_device_gb=0,
            calibrated_step_seconds=0,
            calibrated_score_seconds=0,
        )
    state = "trainable" if legacy and config.calibration_available else "pending"
    explanation = "" if state == "trainable" else "Waiting for measured trainability"
    if not legacy and ref is None:
        state, explanation = "pending", "Waiting for local weights discovery"
    if ref and is_embedding(ref):
        state, explanation = "unsupported", "embedding model"
    with store.connect() as db:
        db.execute(
            "INSERT OR IGNORE INTO models(id,config,support,reason,learning_level,enabled,learning_cutoff) VALUES (?,?,?,?,?,?,?)",
            (
                model,
                json.dumps(values),
                state,
                explanation,
                "new" if legacy else "off",  # library models start off
                1 if legacy else 0,
                time.time(),
            ),
        )
        if legacy:
            db.execute(
                "INSERT OR IGNORE INTO state VALUES ('primary_model',?)",
                (json.dumps(model),),
            )
    scoped = store.scoped(model)
    if legacy:
        scoped.invalidate_pending()
    if not scoped.get("candidate_generation"):
        import uuid

        scoped.set("candidate_generation", str(uuid.uuid4()))
    if not legacy and ref:
        ensure_base(store, model, ref)
    return find(store, model)


def ensure_base(store, model, ref):
    scoped = store.scoped(model)
    if scoped.version():
        return
    base_id = f"base-{digest(model)[:16]}"
    scoped.add_version(
        {
            "id": base_id,
            "identifier": model,
            "artifact": str(ref.path),
            "adapter": None,
            "kind": "base"
            if ref.upstream == "vllm"
            else "gguf-full"
            if ref.format == "gguf"
            else "mlx-fused",
            "created": time.time(),
            "night": 0,
        }
    )
    scoped.swap(base_id, None)


def observe(config, store, model, refs=()):
    """A model used through Songe starts learning once (auto-follow).

    Library models start off; a model the user later switches off stays off.
    """
    migrate(config, store)
    ref = next((r for r in refs if r.serving_id == model), None)
    row = register(config, store, model, ref)
    followed = store.get("followed_models", [])
    if row and row["id"] not in followed:
        store.set("followed_models", [*followed, row["id"]])
        if row["learning_level"] == "off" and row["support"] != "unsupported":
            set_learning(store, row["id"], "new")
            row = find(store, row["id"])
    return row


def discover(config, upstream=None):
    from . import discovery

    if upstream and hasattr(upstream, "discover"):
        return upstream.discover()
    if config.upstream == "lmstudio":
        return discovery.loaded_lmstudio(
            discovery.lmstudio(folder=config.models_dir or None),
            config.lms,
            config.models_dir or discovery.lmstudio_downloads(),
        )
    if config.upstream == "ollama":
        return discovery.ollama(
            folder=config.ollama_models_dir or None, url=config.upstream_url
        )
    if config.upstream == "llamacpp":
        # As LlamaCPP.discover: the owned server may not be running.
        return discovery.llamacpp(config.server_source_url or config.upstream_url)
    return getattr(discovery, config.upstream)(config.upstream_url)


def resolve(store, model=None):
    models = entries(store)
    if model:
        row = find(store, model)
        if not row:
            raise ValueError(f"Unknown model: {model}")
        return row["id"]
    if len(models) == 1:
        return models[0]["id"]
    if models:
        raise ValueError("Choose a model: songe undo <model>")
    return None


def pending_work(config, scoped):
    from ..practice.pool import pending

    return pending(config, scoped)


def summaries(config, store):
    result = []
    for row in entries(store):
        scoped = store.scoped(row["id"])
        nights = scoped.nights()
        latest = next(
            (n for n in reversed(nights) if n["status"] in {"accepted", "rejected"}),
            None,
        )
        result.append(
            {
                "id": row["id"],
                "enabled": row["enabled"],
                "auto_dream": auto_dream(store, row),
                "learning_level": row["learning_level"],
                "learning_cutoff": row["learning_cutoff"],
                "support": row["support"],
                "status": "unsupported"
                if row["support"] == "unsupported"
                else "off"
                if not row["enabled"]
                else "learning",
                "reason": scoped.get("disk_wait")
                or scoped.get("child_error")
                or row["reason"],
                "pending": pending_work(model_config(config, store, row["id"]), scoped),
                "version": scoped.version(),
                "versions": scoped.versions(),
                "can_undo": bool(scoped.get("previous")),
                "last_result": (latest["record"].get("reason") or latest["status"])
                if latest
                else None,
                "last_trained": scoped.get("last_completed", 0),
                "last_update": scoped.get("last_completed") or latest["created"]
                if latest
                else None,
                "last_status": latest["status"] if latest else None,
            }
        )
    return result


def prepare(config, store, platform):
    """Called only inside the guarded calibration child, in an idle gap."""
    from ..backends import resolve_kind
    from ..platform import Rooted
    from . import ModelRef, trainable_base

    if not config.model_ref:
        if config.model and Path(config.model).is_dir():
            return  # Legacy installs already persist their native training base.
        raise ValueError("Local weights are not discoverable")
    from ..artifacts.storage import artifact_root

    ref = ModelRef(**(config.model_ref | {"path": Path(config.model_ref["path"])}))
    config.model = str(
        trainable_base(
            ref,
            resolve_kind(config, platform),
            Rooted(platform, store.root),
            artifact_directory=artifact_root(config, store),
        )
    )
    config.save(store.root)
