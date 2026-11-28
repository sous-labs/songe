"""What every agent adapter shares: file codecs and the adapter interface.

A harness Songe sets up is one adapter: which files it reads, what Songe adds
to them, and how to tell whether they point at Songe. Reviewing, applying,
recording and undoing the edits stay in agents/__init__.py for all of them,
and work on any file a codec can read as a tree.
"""

import functools
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
        except UnicodeDecodeError:
            raise NotEditable("it is not UTF-8") from None
        except json.JSONDecodeError:
            raise NotEditable("it has comments or trailing commas") from None

    def dump(self, data):
        return (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode()


@functools.cache
def yaml12():
    """(loader, dumper) for YAML 1.2's core schema, which Continue reads
    (PyYAML reads 1.1: there, yes/on/off are booleans, 0755 is 493, 1:20 is
    80 and << merges mappings; in 1.2 they are strings and a plain key)."""
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    class Dumper(yaml.SafeDumper):
        pass

    Loader.yaml_implicit_resolvers = {}
    core = [
        ("tag:yaml.org,2002:null", r"^(?:~|null|Null|NULL|)$", "~nN"),
        ("tag:yaml.org,2002:bool", r"^(?:true|True|TRUE|false|False|FALSE)$", "tTfF"),
        ("tag:yaml.org,2002:int", r"^(?:[-+]?[0-9]+|0o[0-7]+|0x[0-9a-fA-F]+)$", "-+0123456789"),
        (
            "tag:yaml.org,2002:float",
            r"^(?:[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?"
            r"|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$",
            "-+0123456789.",
        ),
    ]
    for tag, pattern, first in core:
        Loader.add_implicit_resolver(tag, re.compile(pattern), list(first) + ([""] if tag.endswith("null") else []))
        # A string either version reads as something else is quoted.
        Dumper.add_implicit_resolver(tag, re.compile(pattern), list(first))

    def integer(loader, node):
        value = loader.construct_scalar(node)
        return int(value, 0) if value[:2] in ("0o", "0x") else int(value, 10)

    Loader.add_constructor("tag:yaml.org,2002:int", integer)
    Loader.add_constructor("tag:yaml.org,2002:bool", lambda loader, node: loader.construct_scalar(node).lower() == "true")
    # Without the merge resolver, << is a plain key; it must not be flattened.
    Loader.flatten_mapping = lambda self, node: None
    return Loader, Dumper


def read_yaml(raw, core12=False):
    """A YAML document Songe may rewrite: UTF-8, valid, and without comments,
    which dumping would drop. core12 reads it as YAML 1.2 does."""
    import yaml

    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else (raw or "")
    except UnicodeDecodeError:
        raise NotEditable("it is not UTF-8") from None
    # A comment is the user's own writing; dumping would drop it.
    if re.search(r"(^|\s)#", text):
        raise NotEditable("it has comments")
    try:
        return yaml.load(text, Loader=yaml12()[0]) if core12 else yaml.safe_load(text)
    except yaml.YAMLError:
        raise NotEditable("it is not valid YAML") from None


def keyed(items, key, what="it"):
    """A YAML list of entries as a map keyed by one field."""
    # An empty file is an empty list; `{}`, `0` or `false` is not one.
    items = [] if items is None else items
    if not isinstance(items, list) or not all(
        isinstance(item, dict) and isinstance(item.get(key), str) for item in items
    ):
        raise NotEditable(f"{what} is not a list of entries with a {key}")
    names = [item[key] for item in items]
    repeated = next((n for n in names if names.count(n) > 1), None)
    if repeated is not None:
        # Keyed by name, one of them would be lost for good.
        raise NotEditable(f"it has two entries named {repeated}")
    return {item[key]: item for item in items}


def listed(entries, key):
    return [{key: name, **{k: v for k, v in entry.items() if k != key}}
            for name, entry in entries.items()]


def dump_yaml(data, core12=False):
    import yaml

    class Dumper(yaml12()[1] if core12 else yaml.SafeDumper):
        pass

    def text(dumper, value):
        # A multi-line string stays a block (|), as people write them.
        style = "|" if "\n" in value else None
        return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)

    Dumper.add_representer(str, text)
    return yaml.dump(data, Dumper=Dumper, sort_keys=False, allow_unicode=True).encode()


class YAMLListCodec:
    """A YAML list of entries, read as a map keyed by one field, so field-level
    edit tracking applies (Aider's model settings, keyed by `name`)."""

    def __init__(self, key):
        self.key = key
        self.name = f"yaml-list:{key}"

    def load(self, raw):
        return keyed(read_yaml(raw), self.key)

    def dump(self, data):
        return dump_yaml(listed(data, self.key))


class YAMLSectionCodec:
    """A YAML 1.2 mapping whose `section` is a list of entries keyed by one
    field (Continue's config.yaml: `models`, keyed by `name`). The rest of
    the document is kept as it is."""

    def __init__(self, section, key):
        self.section, self.key = section, key
        self.name = f"yaml-section:{section}:{key}"

    def load(self, raw):
        document = read_yaml(raw, core12=True)
        document = {} if document is None else document
        if not isinstance(document, dict):
            raise NotEditable("it is not a YAML mapping")
        if self.section not in document:
            return document
        return {
            **document,
            self.section: keyed(document[self.section], self.key, f"its {self.section}"),
        }

    def dump(self, data):
        data = dict(data)
        if isinstance(data.get(self.section), dict):
            data[self.section] = listed(data[self.section], self.key)
        return dump_yaml(data, core12=True)


JSON = JSONCodec()


def codec_named(name):
    """The codec a backup manifest entry was written with ("json",
    "yaml-list:name", "yaml-section:models:name")."""
    if name == JSON.name:
        return JSON
    if name.startswith("yaml-list:"):
        return YAMLListCodec(name.split(":", 1)[1])
    if name.startswith("yaml-section:") and name.count(":") == 2:
        return YAMLSectionCodec(*name.split(":")[1:])
    # Written by a later Songe: kept as a conflict, never guessed at.
    raise NotEditable(f"it was written with a codec this Songe does not know ({name})")


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

    def stale(self, path, key, config):
        """A top-level entry Songe wrote for a model it no longer serves
        (for files keyed by model name; opencode and pi use legacy_models)."""
        return False

    def after_uninstall(self, home):
        """A note for what uninstall cannot undo, or None."""
        return None

    def skip_reason(self, home):
        """Why init must leave all of this harness's files alone, or None."""
        return None

    def restored(self, path, current, original):
        """`current` once uninstall has undone Songe's edits, made right for
        the harness where undoing them field by field is not (`original`
        is the file before Songe, {} when there was none)."""
        return current

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
