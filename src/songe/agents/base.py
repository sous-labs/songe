"""What every agent adapter shares: file codecs and the adapter interface.

A harness Songe sets up is one adapter: which files it reads, what Songe adds
to them, and how to tell whether they point at Songe. Reviewing, applying,
recording and undoing the edits stay in agents/__init__.py for all of them,
and work on any file a codec can read as a tree.
"""

import json
import re


class NotEditable(ValueError):
    """A file Songe must not rewrite (comments, JSONC, unparseable)."""


class JSONCodec:
    name = "json"

    def load(self, raw):
        """`raw` is None for a file that does not exist yet."""
        if raw is None:
            return {}
        # Never rewrite JSONC as JSON or silently discard an unparseable
        # config; an empty file or one with a BOM is left alone too.
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            raise NotEditable("it has comments or trailing commas") from None

    def dump(self, data):
        return (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode()


class YAMLListCodec:
    """A YAML list of entries, read as a map keyed by one field, so field-level
    edit tracking applies (Aider's model settings, keyed by `name`)."""

    def __init__(self, key):
        self.key = key
        self.name = f"yaml-list:{key}"

    def load(self, raw):
        import yaml

        text = raw.decode("utf-8") if isinstance(raw, bytes) else (raw or "")
        # A comment is the user's own writing; dumping would drop it.
        if re.search(r"(^|\s)#", text):
            raise NotEditable("it has comments")
        try:
            items = yaml.safe_load(text) or []
        except yaml.YAMLError:
            raise NotEditable("it is not valid YAML") from None
        if not isinstance(items, list) or not all(
            isinstance(item, dict) and isinstance(item.get(self.key), str)
            for item in items
        ):
            raise NotEditable(f"it is not a list of entries with a {self.key}")
        return {item[self.key]: item for item in items}

    def dump(self, data):
        import yaml

        items = [{self.key: key, **{k: v for k, v in entry.items() if k != self.key}}
                 for key, entry in data.items()]
        return yaml.safe_dump(items, sort_keys=False, allow_unicode=True).encode()


JSON = JSONCodec()


def codec_named(name):
    """The codec a backup manifest entry was written with ("json",
    "yaml-list:name")."""
    if name == JSON.name:
        return JSON
    if name.startswith("yaml-list:"):
        return YAMLListCodec(name.split(":", 1)[1])
    raise ValueError(f"Unknown agent file codec: {name}")


class Agent:
    """One harness. Subclasses set `name` and implement `files` and
    `configure`; the rest has defaults that fit a harness with no default
    model of its own."""

    name = ""
    # Shown as "<name> not found" in init's plan when it is not installed.
    suggest_when_missing = False

    def files(self, home):
        """[(path, codec)] Songe edits, or [] when the harness is not installed."""
        raise NotImplementedError

    def codec(self, path):
        """The codec for one of this adapter's files, by file name."""
        return JSON

    def configure(self, path, data, config, *, default=None, v2=False):
        """`data` with Songe's entries added or refreshed."""
        raise NotImplementedError

    def owns(self, path, data):
        """The file already holds Songe's entries (limits are kept current)."""
        raise NotImplementedError

    def status(self, path, data, config):
        """For doctor: "configured", "missing", "differs", or None to skip."""
        return None

    def on_songe(self, home):
        """Its default model is a songe/ one."""
        return False

    # How the user picks a Songe model there ("choose one with <pick>").
    pick = "a Songe model"
    # Older Songe versions wrote raw model names into this file; they are
    # removed when unchanged (opencode, pi).
    legacy_models = False
    # The file names this adapter edits, to find its codec and its name.
    file_names = ()

    def handles(self, path):
        return path.name in self.file_names

    def snippet(self, path, config, *, default=None, v2=False):
        """What a user adds by hand when the file cannot be edited."""
        return self.configure(path, {}, config, default=default, v2=v2)

    def binding(self, home):
        """(model, base URL) it uses now, for init's preference; ("", "")."""
        return "", ""
