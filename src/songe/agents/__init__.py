"""Reviewable, reversible agent configuration edits."""

import difflib
import json
import os
from pathlib import Path

from ..storage import atomic_write, write_json


class NotPlainJSON(ValueError):
    pass


def read_json(path):
    # Never rewrite JSONC as JSON or silently discard an unparseable config.
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise NotPlainJSON(
            f"{path} has comments or trailing commas · remove them, "
            "or add the Songe provider by hand (songe init --show-diff)"
        ) from None


def targets(home=None):
    home = home or Path.home()
    pi = Path(os.environ.get("PI_CODING_AGENT_DIR", str(home / ".pi/agent")))
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", str(home / ".config")))
    paths = []
    if (config_home / "opencode").exists():
        jsonc = config_home / "opencode/opencode.jsonc"
        plain = config_home / "opencode/opencode.json"
        # A JSONC file is read (and then skipped with a note), never shadowed.
        paths.append(
            ("opencode", jsonc if jsonc.exists() and not plain.exists() else plain)
        )
    if pi.exists():
        paths.append(("pi", pi / "models.json"))
    return paths


KNOWN = ("opencode", "pi")


def routing(home=None):
    """Each installed agent, and whether its default model is a Songe one.

    Init switches opencode's local default; pi's default (settings.json) and
    a cloud default are the user's, so those still need a songe/ model picked.
    """
    home = home or Path.home()
    found = {}
    for kind, path in targets(home):
        if kind == "pi":
            path = path.parent / "settings.json"
        try:
            data = read_json(path)
        except NotPlainJSON:
            data = {}
        if kind == "opencode":
            model = data.get("model")
            found[kind] = isinstance(model, str) and model.startswith("songe/")
        else:
            found[kind] = data.get("defaultProvider") == "songe"
    return found


def proposed(config, home=None, default=None, opencode_v2=False, root=None):
    from .opencode import configure as oc
    from .pi import configure as pi

    changes = []
    for kind, path in targets(home):
        original = path.read_bytes() if path.exists() else None
        try:
            if path.suffix == ".jsonc":
                raise NotPlainJSON(path)
            data = read_json(path)
        except NotPlainJSON:
            # One commented config must not block the other agents.
            skip_note(kind, path, config, default, opencode_v2)
            continue
        if root:
            data = remove_legacy_models(data, path, root, config)
        new = (
            oc(data, config, default=default, v2=opencode_v2)
            if kind == "opencode"
            else pi(data, config)
        )
        result = (json.dumps(new, indent=2, ensure_ascii=False) + "\n").encode()
        if original != result:
            diff = "".join(
                difflib.unified_diff(
                    (original or b"").decode().splitlines(True),
                    result.decode().splitlines(True),
                    fromfile=str(path),
                    tofile=str(path),
                )
            )
            changes.append((path, original, result, diff))
    return changes


def skip_note(kind, path, config, default, opencode_v2):
    import sys

    from .opencode import configure as oc
    from .pi import configure as pi

    key = "providers" if kind == "opencode" and opencode_v2 else "provider"
    entry = (
        oc({}, config, default=default, v2=opencode_v2)
        if kind == "opencode"
        else pi({}, config)
    )
    entry = {key: entry[key]} if key in entry else entry
    print(
        f"Skipped {path} · it has comments · add this by hand:\n"
        + json.dumps(entry, indent=2, ensure_ascii=False),
        file=sys.stderr,
    )


def remove_legacy_models(data, path, root, config, aliases=None):
    """Remove only unchanged model entries recorded in our backup manifest."""
    from ..models.names import catalog

    record = read_json(root / "agent-backups.json").get(str(path))
    if not record:
        return data
    written = json.loads(record["written"])
    aliases = aliases or catalog(config)
    for key in ("provider", "providers"):
        current = data.get(key, {}).get("songe", {}).get("models")
        prior = written.get(key, {}).get("songe", {}).get("models")
        if isinstance(current, dict) and isinstance(prior, dict):
            for raw, alias in aliases.items():
                if raw != alias and raw in prior and current.get(raw) == prior[raw]:
                    current.pop(raw, None)
        elif isinstance(current, list) and isinstance(prior, list):
            prior = {m["id"]: m for m in prior}
            current[:] = [
                m
                for m in current
                if not (
                    m.get("id") in aliases
                    and aliases[m["id"]] != m["id"]
                    and m == prior.get(m["id"])
                )
            ]
    return data


def migrate(config, root, *, update_agents=True):
    """Refresh only previously installed, unchanged Songe agent fields."""
    from ..log import Store
    from ..models import registry
    from ..models.names import catalog, public_name
    from ..steps import digest

    store = Store(root)
    registry.migrate(config, store)
    for model in catalog(config):
        if not registry.find(store, model):
            registry.register(config, store, model)
    for row in registry.entries(store):
        if row["config"].get("public_model") == f"songe/{digest(row['id'])[:16]}":
            selected = registry.model_config(config, store, row["id"])
            selected.public_model = public_name(row["id"])
            selected.save(root)
    aliases = catalog(config, store)
    changes = []
    manifest = read_json(root / "agent-backups.json") if update_agents else {}
    for name, record in manifest.items():
        path = Path(name)
        if not path.exists():
            continue
        original = path.read_bytes()
        try:
            data = json.loads(original)
        except json.JSONDecodeError:
            # The user's own edit (e.g. comments) is never rewritten; the
            # service keeps running and leaves this file as it is.
            continue
        written = json.loads(record["written"])
        # A changed endpoint/provider belongs to the user now.
        for key in ("provider", "providers"):
            current = data.get(key, {}).get("songe")
            previous = written.get(key, {}).get("songe")
            if not current or not previous:
                continue
            current_settings = {k: v for k, v in current.items() if k != "models"}
            previous_settings = {k: v for k, v in previous.items() if k != "models"}
            if current_settings != previous_settings:
                continue
            data = remove_legacy_models(data, path, root, config, aliases)
            from dataclasses import replace

            proposal = replace(config, exposed_models=list(aliases.values()))
            if isinstance(current.get("models"), dict):
                from .opencode import configure

                generated = configure({}, proposal, v2=key == "providers")[key][
                    "songe"
                ]["models"]
                for model, entry in generated.items():
                    current["models"].setdefault(model, entry)
            else:
                from .pi import configure

                generated = configure({}, proposal)["providers"]["songe"]["models"]
                present = {m["id"] for m in current["models"]}
                current["models"].extend(m for m in generated if m["id"] not in present)
        if data.get("model") == written.get("model"):
            for raw, alias in aliases.items():
                if data.get("model") == "songe/" + raw:
                    data["model"] = "songe/" + alias
        result = (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode()
        if data != json.loads(original):
            changes.append((path, original, result, ""))
    if changes:
        apply(changes, root)
    exposed = list(dict.fromkeys(aliases.values()))
    if config.exposed_models != exposed:
        config.exposed_models = exposed
        config.save(root)


def edits(before, after, prefix=()):
    """Track only changed JSON fields; sibling user settings stay unowned."""
    result = []
    for key in before.keys() | after.keys():
        old, new = before.get(key), after.get(key)
        if key in before and key in after and old == new:
            continue
        if isinstance(old, dict) and isinstance(new, dict):
            result.extend(edits(old, new, prefix + (key,)))
        elif key not in before and isinstance(new, dict) and key not in {"songe"}:
            result.extend(edits({}, new, prefix + (key,)))
        else:
            result.append(
                {
                    "path": list(prefix + (key,)),
                    "had": key in before,
                    "old": old,
                    "has": key in after,
                    "new": new,
                }
            )
    return result


def apply(changes, root):
    # Validate the entire reviewed proposal before touching any agent file.
    for path, original, *_ in changes:
        if (path.read_bytes() if path.exists() else None) != original:
            raise RuntimeError(f"Agent configuration changed since review: {path}")
    manifest_path = root / "agent-backups.json"
    manifest = read_json(manifest_path)
    for path, original, result, _ in changes:
        old = json.loads(original) if original else {}
        new = json.loads(result)
        entry = manifest.get(str(path))
        if entry is None:
            backup = root / "backups" / f"{len(manifest)}.json"
            if original is not None:
                atomic_write(backup, original)
            entry = {
                "backup": str(backup) if original is not None else None,
                "written": result.decode(),
                "edits": [],
            }
        if "edits" not in entry:
            # Migrate the original manifest using its first backup, not the
            # newly reviewed document (which may include later user changes).
            first = read_json(Path(entry["backup"])) if entry["backup"] else {}
            entry["edits"] = edits(first, json.loads(entry["written"]))
        owned = {tuple(e["path"]): e for e in entry["edits"]}
        for edit in edits(old, new):
            key = tuple(edit["path"])
            ancestor = next((p for p in owned if key[: len(p)] == p), None)
            if ancestor is not None:
                value = new
                for part in ancestor:
                    value = value.get(part) if isinstance(value, dict) else None
                owned[ancestor]["new"] = value
                continue
            descendants = [p for p in owned if p[: len(key)] == key]
            if descendants:
                first = read_json(Path(entry["backup"])) if entry["backup"] else {}
                for part in key[:-1]:
                    first = first.get(part, {}) if isinstance(first, dict) else {}
                edit = edit | {"had": key[-1] in first, "old": first.get(key[-1])}
                for child in descendants:
                    del owned[child]
            if key in owned:
                edit = edit | {"had": owned[key]["had"], "old": owned[key]["old"]}
            owned[key] = edit
        entry.update(written=result.decode(), edits=list(owned.values()))
        manifest[str(path)] = entry
        write_json(manifest_path, manifest)
        if (path.read_bytes() if path.exists() else None) != original:
            raise RuntimeError(f"Agent configuration changed since review: {path}")
        atomic_write(path, result)


def restore(root):
    manifest_path = root / "agent-backups.json"
    manifest = read_json(manifest_path)
    conflicts = []
    for name, entry in list(manifest.items()):
        path = Path(name)
        try:
            current = read_json(path)
        except NotPlainJSON:
            conflicts.append(name)
            continue
        original = read_json(Path(entry["backup"])) if entry["backup"] else {}
        owned = entry.get("edits", edits(original, json.loads(entry["written"])))
        unresolved = []
        for edit in owned:
            parent = current
            for key in edit["path"][:-1]:
                parent = parent.get(key) if isinstance(parent, dict) else None
            key = edit["path"][-1]
            if not isinstance(parent, dict):
                continue
            if (key in parent) != edit["has"] or parent.get(key) != edit["new"]:
                value = parent.get(key)
                if not (isinstance(value, str) and value.startswith("songe/")):
                    # Changed by the user since init: their value stays, and
                    # that is not a conflict to keep data around for.
                    continue
                # Still points at Songe (another of its models was picked):
                # undo it, or the agent is left with a model that is gone.
            if edit["had"]:
                parent[key] = edit["old"]
            else:
                parent.pop(key, None)
            # Remove empty containers created by Songe, preserving originals.
            for length in range(len(edit["path"]) - 1, 0, -1):
                cursor, previous = current, original
                for part in edit["path"][: length - 1]:
                    cursor = cursor.get(part, {})
                    previous = (
                        previous.get(part, {}) if isinstance(previous, dict) else {}
                    )
                part = edit["path"][length - 1]
                if cursor.get(part) == {} and part not in previous:
                    cursor.pop(part)
        if current == original and not unresolved:
            if entry["backup"]:
                atomic_write(path, Path(entry["backup"]).read_bytes())
            else:
                path.unlink(missing_ok=True)
        else:
            write_json(path, current)
        if unresolved:
            conflicts.append(name)
            entry["edits"] = unresolved
        else:
            del manifest[name]
        write_json(manifest_path, manifest)
    return conflicts


def refresh_limits(config, home=None, root=None):
    """Rewrite only Songe's own entries when served contexts change.

    Agents without a Songe provider are left alone; defaults never change.
    With `root`, the rewrite is recorded as Songe's own edit, so uninstall
    still restores the file instead of preserving it as a user conflict."""
    from .opencode import configure as oc
    from .pi import configure as pi

    changed = []
    for kind, path in targets(home):
        if not path.exists() or path.suffix == ".jsonc":
            continue
        try:
            data = read_json(path)
        except NotPlainJSON:
            continue
        providers = data.get("providers") or data.get("provider") or {}
        if "songe" not in providers:
            continue
        new = (
            oc(data, config, default=False, v2="providers" in data)
            if kind == "opencode"
            else pi(data, config)
        )
        if new != data:
            result = (json.dumps(new, indent=2, ensure_ascii=False) + "\n").encode()
            if root is None:
                atomic_write(path, result)
            else:
                apply([(path, path.read_bytes(), result, "")], root)
            changed.append(path)
    return changed
