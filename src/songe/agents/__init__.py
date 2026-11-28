"""Reviewable, reversible agent configuration edits.

Each harness is an adapter (agents/base.py): its files, the codec that reads
them, and what Songe adds. Everything here works through the adapters, so a
new harness is one module and its edits are recorded and undone like the rest.
"""

import difflib
import json
from pathlib import Path

from ..storage import atomic_write, write_json
from .aider import Aider
from .base import JSON, NotEditable, codec_named
from .continue_ import Continue
from .opencode import OpenCode
from .pi import Pi

ADAPTERS = (OpenCode(), Pi(), Aider(), Continue())

# A file Songe will not rewrite (the older name, still used by callers).
NotPlainJSON = NotEditable


def adapter_for(path):
    return next((a for a in ADAPTERS if a.handles(Path(path))), None)


def codec_for(path, entry=None):
    """How to read an agent file: as recorded in its backup manifest entry
    when there is one, so it is undone the way it was written, else by its
    adapter."""
    if entry and entry.get("codec"):
        return codec_named(entry["codec"])
    adapter = adapter_for(path)
    return adapter.codec(path) if adapter else JSON


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


def load(path, raw=None):
    """An agent file as a tree, through its adapter's codec."""
    raw = path.read_bytes() if raw is None and path.exists() else raw
    try:
        return codec_for(path).load(raw)
    except NotEditable as exc:
        raise NotEditable(
            f"{path}: {exc} · remove them, or add Songe's entries by hand "
            "(songe init --show-diff)"
        ) from None


def targets(home=None):
    home = home or Path.home()
    return [(a.name, path) for a in ADAPTERS for path, _ in a.files(home)]


# Harnesses init names in its plan when they are not installed.
KNOWN = tuple(a.name for a in ADAPTERS if a.suggest_when_missing)


def routing(home=None):
    """Each installed agent, and whether its default model is a Songe one.

    Init switches opencode's local default; pi's default (settings.json) and
    a cloud default are the user's, so those still need a songe/ model picked.
    """
    home = home or Path.home()
    return {a.name: a.on_songe(home) for a in ADAPTERS if a.files(home)}


def proposed(config, home=None, default=None, opencode_v2=False, root=None):
    home = home or Path.home()
    changes = []
    for adapter in ADAPTERS:
        loaded, skipped = [], None
        reason = adapter.skip_reason(home)
        if reason:
            for path, _ in adapter.files(home):
                skip_note(adapter, path, config, default, opencode_v2, reason)
            continue
        for path, codec in adapter.files(home):
            original = path.read_bytes() if path.exists() else None
            try:
                if path.suffix == ".jsonc":
                    raise NotEditable(path)
                loaded.append((path, codec, original, codec.load(original)))
            except NotEditable as exc:
                # JSON files keep their usual note; other codecs say why.
                why = str(exc) if codec is not JSON or "UTF-8" in str(exc) else "it has comments"
                skipped = (path, why)
                break
        if skipped:
            # One commented config must not block the other agents, and an
            # agent is set up in all its files or none: Aider with model
            # metadata but no api_base sends a songe/ model to OpenAI.
            for path, _ in adapter.files(home):
                why = skipped[1] if path == skipped[0] else f"{skipped[0].name} is skipped"
                skip_note(adapter, path, config, default, opencode_v2, why)
            continue
        for path, codec, original, data in loaded:
            if root and adapter.legacy_models:
                data = remove_legacy_models(data, path, root, config)
            elif root:
                data = drop_stale(adapter, path, data, root, config)
            new = adapter.configure(path, data, config, default=default, v2=opencode_v2)
            result = codec.dump(new)
            changes.extend(diff_change(path, original, result))
    return changes


def diff_change(path, original, result):
    """[(path, original, result, diff)] when the file would change, else []."""
    if original == result:
        return []
    diff = "".join(
        difflib.unified_diff(
            (original or b"").decode().splitlines(True),
            result.decode().splitlines(True),
            fromfile=str(path),
            tofile=str(path),
        )
    )
    return [(path, original, result, diff)]


def skip_note(adapter, path, config, default, opencode_v2, why="it has comments"):
    import sys

    entry = adapter.snippet(path, config, default=default, v2=opencode_v2)
    print(
        f"Skipped {path} · {why} · add this by hand:\n"
        + adapter.codec(path).dump(entry).decode().rstrip("\n"),
        file=sys.stderr,
    )


def drop_stale(adapter, path, data, root, config):
    """Without Songe's entries for models it no longer serves (a renamed
    model), when they are unchanged since Songe wrote them."""
    record = read_json(root / "agent-backups.json").get(str(path))
    if not record:
        return data
    try:
        codec = codec_for(path, record)
        written = codec.load(record["written"].encode())
    except NotEditable:
        return data
    # Entries are top-level keys, or those of one section (Continue's models).
    section = getattr(codec, "section", None)
    current = data.get(section) if section else data
    written = written.get(section) if section else written
    if not isinstance(current, dict) or not isinstance(written, dict):
        return data
    kept = {
        key: value
        for key, value in current.items()
        if not (adapter.stale(path, key, config) and written.get(key) == value)
    }
    return {**data, section: kept} if section else kept


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
        adapter = adapter_for(path)
        # Renames and new models are about opencode's and pi's provider
        # entries; other harnesses are kept current by refresh_limits.
        if not path.exists() or (adapter and not adapter.legacy_models):
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


def new_backup(root, manifest):
    """A backup file no manifest entry uses and nothing occupies. Numbering by
    the manifest's size reused a kept entry's backup after a restore with a
    conflict, overwriting that agent's original configuration."""
    used = {Path(e["backup"]).name for e in manifest.values() if e.get("backup")}
    number = len(manifest)
    while (
        f"{number}.json" in used or (root / "backups" / f"{number}.json").exists()
    ):
        number += 1
    return root / "backups" / f"{number}.json"


def apply(changes, root):
    # Validate the entire reviewed proposal before touching any agent file.
    for path, original, *_ in changes:
        if (path.read_bytes() if path.exists() else None) != original:
            raise RuntimeError(f"Agent configuration changed since review: {path}")
    manifest_path = root / "agent-backups.json"
    manifest = read_json(manifest_path)
    for path, original, result, _ in changes:
        codec = codec_for(path)
        old = codec.load(original) if original else {}
        new = codec.load(result)
        entry = manifest.get(str(path))
        if entry is None:
            backup = new_backup(root, manifest)
            if original is not None:
                atomic_write(backup, original)
            entry = {
                "backup": str(backup) if original is not None else None,
                "written": result.decode(),
                "edits": [],
            }
        entry["codec"] = codec.name
        if "edits" not in entry:
            # Migrate the original manifest using its first backup, not the
            # newly reviewed document (which may include later user changes).
            first = read_backup(entry, codec)
            entry["edits"] = edits(first, codec.load(entry["written"].encode()))
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
                first = read_backup(entry, codec)
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


def read_backup(entry, codec):
    """The file as it was before Songe; {} when there was none, or when its
    backup is gone (uninstall still undoes Songe's edits in every file)."""
    backup = Path(entry["backup"]) if entry["backup"] else None
    if backup is None or not backup.exists():
        return {}
    return codec.load(backup.read_bytes())


def restore(root):
    manifest_path = root / "agent-backups.json"
    manifest = read_json(manifest_path)
    conflicts = []
    for name, entry in list(manifest.items()):
        path = Path(name)
        raw = path.read_bytes() if path.exists() else None
        if raw is None or not raw.strip():
            # Deleted or emptied since: nothing of Songe's is left in it, and
            # writing {} would be worse than either (Continue writes its
            # default only for a missing or blank file).
            del manifest[name]
            write_json(manifest_path, manifest)
            continue
        try:
            codec = codec_for(path, entry)
            current = codec.load(raw)
        except NotEditable:
            conflicts.append(name)
            continue
        original = read_backup(entry, codec)
        owned = entry.get("edits", edits(original, codec.load(entry["written"].encode())))
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
        adapter = adapter_for(path)
        if adapter:
            current = adapter.restored(path, current, original)
        backup = Path(entry["backup"]) if entry["backup"] else None
        if current == original and not unresolved and not (backup and not backup.exists()):
            if backup:
                atomic_write(path, backup.read_bytes())
            else:
                path.unlink(missing_ok=True)
        elif codec is JSON:
            write_json(path, current)
        else:
            atomic_write(path, codec.dump(current))
        if unresolved:
            conflicts.append(name)
            entry["edits"] = unresolved
        else:
            del manifest[name]
        write_json(manifest_path, manifest)
    return conflicts


def refresh_limits(config, home=None, root=None):
    """Rewrite only Songe's own entries when served contexts change.

    Agents without Songe's entries are left alone; defaults never change.
    With `root`, the rewrite is recorded as Songe's own edit, so uninstall
    still restores the file instead of preserving it as a user conflict."""
    home = home or Path.home()
    changed = []
    for adapter in ADAPTERS:
        loaded = []
        for path, codec in adapter.files(home):
            if not path.exists() or path.suffix == ".jsonc":
                break
            try:
                data = codec.load(path.read_bytes())
            except NotEditable:
                break
            if not adapter.owns(path, data):
                break
            loaded.append((path, codec, data))
        if len(loaded) != len(adapter.files(home)):
            # All of an agent's files or none, as in proposed(): a file the
            # user deleted, commented or took Songe out of leaves the others
            # alone too (Aider's metadata for a model the settings file has
            # no api_base for would send it to OpenAI).
            continue
        for path, codec, data in loaded:
            new = adapter.configure(path, data, config, default=False)
            if root and not adapter.legacy_models:
                new = drop_stale(adapter, path, new, root, config)
            if new != data:
                result = codec.dump(new)
                if root is None:
                    atomic_write(path, result)
                else:
                    apply([(path, path.read_bytes(), result, "")], root)
                changed.append(path)
    return changed
