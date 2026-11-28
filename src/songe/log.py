"""Versioned SQLite store. Never stores transport headers or bearer tokens."""

import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from . import parts
from .steps import digest
from .storage import private_dir

# 3: repeated parts of recorded requests and steps stored once (parts.py).
SCHEMA_VERSION = 3
# Steps of turns brought in by `songe import` are read for what the user said
# (notes, practice) but are never test moments or training contexts: another
# model answered them, with another agent's tools (imports.py).
NOT_IMPORTED = (
    "NOT EXISTS (SELECT 1 FROM exchanges e WHERE e.id=json_extract(s.payload,'$.source_id')"
    " AND e.binding LIKE 'import:%')"
)

# Tries of one recording while another writer holds the database, each
# waiting the connection timeout (5 s): a minute covers giving back the space
# of a store at its size limit.
RECORD_ATTEMPTS = 12
# How long a request whose history holds turns Songe never recorded waits for
# them (a parent recorded late links it to its session) before it is read as
# a conversation of its own (#263).
UNLINKED_AFTER = 600

_CONNECTIONS = threading.local()
_CACHED_PER_THREAD = 4


def _open(path):
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    db = sqlite3.connect(path, timeout=5)
    db.row_factory = sqlite3.Row
    if new:
        # A new store gives freed space back in small steps
        # (recordings.compact); SQLite allows choosing this only before the
        # write-ahead log and the first table.
        db.execute("PRAGMA auto_vacuum=INCREMENTAL")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    return db


def _identity(path):
    try:
        s = os.stat(path)
    except FileNotFoundError:
        return None
    return s.st_dev, s.st_ino


def _forget(key):
    cache = _CONNECTIONS.__dict__.get("cache", {})
    entry = cache.pop(key, None)
    if entry:
        try:
            entry[0].close()
        except sqlite3.Error:
            pass


def close_connections():
    """Close this thread's cached connections (Windows cannot delete or
    rename a database file that is open; uninstall --purge removes it)."""
    for key in list(_CONNECTIONS.__dict__.get("cache", {})):
        _forget(key)
    for key in list(_CONNECTIONS.__dict__.get("watchers", {})):
        _unwatch(key)


def _unwatch(key):
    entry = _CONNECTIONS.__dict__.get("watchers", {}).pop(key, None)
    if entry:
        try:
            entry[0].close()
        except sqlite3.Error:
            pass


def change_mark(path):
    """A value that differs after every commit to `path`, from any
    connection or process; None when there is no database yet.

    File times cannot do this: Linux stamps them from a clock that ticks
    every few milliseconds, so several commits can leave them unchanged.
    SQLite's data_version changes whenever another connection commits, and
    this connection never writes. The token tells a reopened connection's
    count apart from an earlier one's."""
    identity = _identity(path)
    if identity is None:
        return None
    watchers = _CONNECTIONS.__dict__.setdefault("watchers", {})
    key = str(path)
    if key in watchers and watchers[key][1] != identity:
        _unwatch(key)  # The file was replaced or removed since.
    entry = watchers.pop(key, None)
    if entry is None:
        entry = (sqlite3.connect(path, timeout=5), identity, uuid.uuid4().hex)
    watchers[key] = entry  # most recently used last
    for old in list(watchers)[: max(0, len(watchers) - _CACHED_PER_THREAD)]:
        _unwatch(old)
    return entry[2], entry[0].execute("PRAGMA data_version").fetchone()[0]


def _cached(path):
    """This thread's open connection to `path`, reopened if the file was
    replaced or removed since. Least recently used ones beyond a few close."""
    cache = _CONNECTIONS.__dict__.setdefault("cache", {})
    key = str(path)
    entry = cache.pop(key, None)
    if entry and entry[1] != _identity(path):
        try:
            entry[0].close()
        except sqlite3.Error:
            pass
        entry = None
    if entry is None:
        db = _open(path)
        entry = (db, _identity(path))
    cache[key] = entry  # most recently used last
    held = _CONNECTIONS.__dict__.get("held", set())
    idle = [k for k in cache if k != key and k not in held]
    for old in idle[: max(0, len(cache) - _CACHED_PER_THREAD)]:
        _forget(old)
    return entry[0]

SCHEMA = """
CREATE TABLE IF NOT EXISTS exchanges (
 id INTEGER PRIMARY KEY, created REAL NOT NULL, path TEXT NOT NULL,
 request TEXT NOT NULL, response TEXT, status INTEGER NOT NULL, complete INTEGER NOT NULL,
 binding TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS versions (
 id TEXT PRIMARY KEY, identifier TEXT NOT NULL, artifact TEXT NOT NULL,
 adapter TEXT, created REAL NOT NULL, night INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS nights (
 id INTEGER PRIMARY KEY, created REAL NOT NULL, status TEXT NOT NULL,
 phase TEXT NOT NULL, progress REAL NOT NULL DEFAULT 0, record TEXT NOT NULL DEFAULT '{}');
CREATE TABLE IF NOT EXISTS pool (step_id TEXT PRIMARY KEY, score REAL NOT NULL, step TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS exchanges_created ON exchanges(created);
"""


SCHEMA_V2 = """
CREATE TABLE IF NOT EXISTS steps (
 id TEXT PRIMARY KEY, conversation TEXT NOT NULL, split TEXT NOT NULL,
 signal TEXT, strength REAL NOT NULL DEFAULT 0, evidence TEXT NOT NULL DEFAULT '[]',
 answer_hint TEXT, env_fingerprint TEXT NOT NULL DEFAULT '', retired_at REAL,
 retired_reason TEXT, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS scoreboard (
 day TEXT NOT NULL, model_version TEXT NOT NULL, repeat_error_rate REAL,
 steps_per_task REAL, tool_call_valid_rate REAL, general_drift REAL,
 PRIMARY KEY(day,model_version));
CREATE TABLE IF NOT EXISTS slices (
 id INTEGER PRIMARY KEY, started REAL NOT NULL, ended REAL, phase TEXT NOT NULL,
 units_done INTEGER NOT NULL DEFAULT 0, interrupted_by TEXT);
CREATE TABLE IF NOT EXISTS prefixes (hash TEXT PRIMARY KEY, step_id TEXT NOT NULL, root TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS reconstructed (exchange_id INTEGER PRIMARY KEY, resolved INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS metric_dirty (step_id TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS scored_revisions (step_id TEXT PRIMARY KEY, revision INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS reviews (
 key TEXT PRIMARY KEY, kind TEXT, payload TEXT NOT NULL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS lessons (
 id TEXT NOT NULL, model_id TEXT NOT NULL, group_key TEXT NOT NULL,
 payload TEXT NOT NULL, retired_at REAL, retired_reason TEXT,
 PRIMARY KEY(id,model_id));
CREATE TABLE IF NOT EXISTS step_signals (step_id TEXT PRIMARY KEY, signals TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events (
 id INTEGER PRIMARY KEY, created REAL NOT NULL, model_id TEXT NOT NULL DEFAULT '',
 stage TEXT NOT NULL, text TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS read_sessions (
 session TEXT NOT NULL, model_id TEXT NOT NULL, revision INTEGER NOT NULL,
 read_at REAL NOT NULL, PRIMARY KEY(session,model_id));
CREATE TABLE IF NOT EXISTS signals (
 id TEXT NOT NULL, model_id TEXT NOT NULL, session TEXT NOT NULL,
 revision INTEGER NOT NULL, payload TEXT NOT NULL, retired_at REAL,
 PRIMARY KEY(id,model_id));
CREATE TABLE IF NOT EXISTS reading_cache (
 key TEXT PRIMARY KEY, name TEXT NOT NULL, answer TEXT NOT NULL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS practice (
 id TEXT NOT NULL, model_id TEXT NOT NULL, signal_id TEXT NOT NULL,
 payload TEXT NOT NULL, kept INTEGER NOT NULL, "drop" TEXT, created REAL NOT NULL,
 PRIMARY KEY(id,model_id));
-- Sessions whose recordings were deleted for the size limit after they were
-- learned from (recordings.keep_under); their signals and practice stay.
CREATE TABLE IF NOT EXISTS pruned_sessions (session TEXT PRIMARY KEY, at REAL NOT NULL);
-- Turns brought in from an agent's own logs (songe import), by content, so
-- importing the same logs again adds only what is new.
CREATE TABLE IF NOT EXISTS imports (
 key TEXT PRIMARY KEY, exchange_id INTEGER NOT NULL, source TEXT NOT NULL, at REAL NOT NULL);
"""


# These observations and controls belong to the installation, not a learner.
SHARED_STATE = {
    "installation_id",
    "restoring",
    "primary_model",
    "paused",
    "requested_sleep",
    "requested_model",
    "parked_models",
    "environment_fingerprint",
    "experience_revision",
    "forgotten_at",
    "interface_settings",
    "memory_pressure_history",
    "cloud",
}


def strip_example(example):
    """A before/after example without anything quoted from a conversation."""
    example = dict(example)
    example["context"] = dict(example.get("context") or {}, request=None)
    example["feedback"] = None
    example["lesson"] = None
    for side in ("before", "after"):
        if isinstance(example.get(side), dict):
            example[side] = dict(example[side], text="", tool_calls=[])
    return example



# A step's hindsight is truthy (as Python's `if step.get("hindsight")`),
# judged by SQLite without decoding the payload.
HINDSIGHT = (
    "CASE json_type(s.payload,'$.hindsight')"
    " WHEN 'array' THEN json_array_length(s.payload,'$.hindsight') > 0"
    " WHEN 'object' THEN json_extract(s.payload,'$.hindsight') != '{}'"
    " WHEN 'text' THEN length(json_extract(s.payload,'$.hindsight')) > 0"
    " WHEN 'true' THEN 1"
    " WHEN 'integer' THEN json_extract(s.payload,'$.hindsight') != 0"
    " WHEN 'real' THEN json_extract(s.payload,'$.hindsight') != 0"
    " ELSE 0 END"
)

class Store:
    model_id = ""
    model_key = None

    def scoped(self, model):
        view = object.__new__(type(self))
        view.root, view.path = self.root, self.path
        view.model_key = model
        view.model_id = "" if model == self.get("primary_model") else model
        return view

    def state_key(self, key):
        return (
            f"model:{self.model_id}:{key}"
            if self.model_id and key not in SHARED_STATE
            else key
        )

    def versions(self):
        with self.connect() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM versions WHERE model_id=?", (self.model_id,)
                )
            ]

    def drop_versions(self, keep):
        """Forget this model's versions other than `keep`; their adapters are
        then collected with their work folders."""
        keep = [k for k in keep if k]
        with self.connect() as db:
            db.execute(
                "DELETE FROM versions WHERE model_id=? AND id NOT IN (%s)"
                % ",".join("?" * len(keep)),
                (self.model_id, *keep),
            )

    def __init__(self, root: Path):
        self.root = private_dir(root)
        self.path = root / "experience.sqlite3"
        with self.connect() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise RuntimeError("Experience database was created by a newer Songe")
            db.executescript(SCHEMA)
            db.executescript(SCHEMA_V2)
            db.execute("BEGIN IMMEDIATE")
            # v3: repeated parts of recordings are stored once (parts.py).
            # Older rows stay plain JSON, which every reader still reads, until
            # recordings.dedupe rewrites them in small resumable batches.
            parts.install(db)
            columns = {r[1] for r in db.execute("PRAGMA table_info(exchanges)")}
            if "recorded_at" not in columns:
                # Keep request-start timestamps for traffic history. Older
                # records use that timestamp when recording time is unknown.
                db.execute("ALTER TABLE exchanges ADD COLUMN recorded_at REAL")
            columns = {r[1] for r in db.execute("PRAGMA table_info(versions)")}
            for name in ("kind", "base", "training_adapter"):
                if name not in columns:
                    db.execute(f"ALTER TABLE versions ADD COLUMN {name} TEXT")
            db.execute(
                "CREATE TABLE IF NOT EXISTS models (id TEXT PRIMARY KEY, config TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1, support TEXT NOT NULL DEFAULT 'pending', reason TEXT NOT NULL DEFAULT 'Waiting for measured trainability')"
            )
            columns = {r[1] for r in db.execute("PRAGMA table_info(models)")}
            migrate_learning = "learning_level" not in columns
            if migrate_learning:
                db.execute(
                    "ALTER TABLE models ADD COLUMN learning_level TEXT NOT NULL DEFAULT 'new'"
                )
                db.execute("ALTER TABLE models ADD COLUMN learning_cutoff REAL")
                db.execute(
                    "UPDATE models SET learning_level=CASE WHEN enabled THEN 'new' ELSE 'off' END, learning_cutoff=?",
                    (time.time(),),
                )
            for table in ("versions", "nights", "slices"):
                columns = {r[1] for r in db.execute(f"PRAGMA table_info({table})")}
                if "model_id" not in columns:
                    db.execute(
                        f"ALTER TABLE {table} ADD COLUMN model_id TEXT NOT NULL DEFAULT ''"
                    )
            for table, definition, names in (
                (
                    "pool",
                    "step_id TEXT, score REAL NOT NULL, step TEXT NOT NULL",
                    "step_id,score,step",
                ),
                (
                    "scored_revisions",
                    "step_id TEXT, revision INTEGER NOT NULL",
                    "step_id,revision",
                ),
            ):
                columns = {r[1] for r in db.execute(f"PRAGMA table_info({table})")}
                if "model_id" not in columns:
                    db.execute(f"ALTER TABLE {table} RENAME TO legacy_{table}")
                    db.execute(
                        f"CREATE TABLE {table} ({definition}, model_id TEXT NOT NULL DEFAULT '', PRIMARY KEY(step_id,model_id))"
                    )
                    db.execute(
                        f"INSERT INTO {table} ({names}) SELECT {names} FROM legacy_{table}"
                    )
                    db.execute(f"DROP TABLE legacy_{table}")
            columns = {r[1] for r in db.execute("PRAGMA table_info(slices)")}
            if "record" not in columns:
                db.execute(
                    "ALTER TABLE slices ADD COLUMN record TEXT NOT NULL DEFAULT '{}'"
                )
            # A v1 pool already contains serialized steps. Preserve its scores
            # and split while adding metadata; no source log is rewritten.
            if version < 2:
                for row in db.execute("SELECT step FROM pool").fetchall():
                    item = json.loads(row[0])
                    item.setdefault("env_fingerprint", digest(item.get("tools", [])))
                    self._put_step(db, item)
            # Stream the additive summary migration; the serving API never
            # materializes conversation payloads to display pipeline counts.
            from .sleep.lessons import signal_counts

            for row in db.execute(
                "SELECT * FROM steps WHERE id NOT IN (SELECT step_id FROM step_signals)"
            ):
                db.execute(
                    "INSERT INTO step_signals VALUES (?,?)",
                    (
                        row["id"],
                        json.dumps(signal_counts([parts.payload(db, row)])),
                    ),
                )
            db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            if not db.execute(
                "SELECT 1 FROM state WHERE key='grouped_by_instruction'"
            ).fetchone():
                # Steps recorded when a whole session was one conversation join
                # the part of it they belong to (see steps.extract). Settled
                # splits keep the grouping they were trained and tested under.
                part = "json_extract(payload,'$.task')"
                db.execute(
                    f"UPDATE steps SET conversation={part},"
                    f"payload=json_set(payload,'$.conversation',{part}) "
                    f"WHERE split='pending' AND COALESCE({part},'')!='' "
                    f"AND conversation!={part}"
                )
                db.execute(
                    "INSERT OR REPLACE INTO state VALUES ('grouped_by_instruction','true')"
                )
                db.execute(
                    "INSERT OR REPLACE INTO state VALUES ('experience_revision',?)",
                    (json.dumps(str(uuid.uuid4())),),
                )
            db.execute(
                "INSERT OR IGNORE INTO state VALUES ('installation_id',?)",
                (json.dumps(str(uuid.uuid4())),),
            )
            db.execute(
                "INSERT OR IGNORE INTO state VALUES ('candidate_generation',?)",
                (json.dumps(str(uuid.uuid4())),),
            )
            # The newest recording when the version that keeps a streamed
            # answer's reasoning under either field name first opened this
            # store. Streamed answers up to it may have lost theirs
            # (sleep.prompts.served_switches does not read them as direct).
            db.execute(
                "INSERT OR IGNORE INTO state VALUES ('streamed_reasoning_from',"
                " (SELECT CAST(COALESCE(MAX(id),0) AS TEXT) FROM exchanges))"
            )
            if migrate_learning:
                # Old candidates may contain history outside the new window.
                for row in db.execute("SELECT id FROM models").fetchall():
                    self.scoped(row[0])._invalidate_pending(db)
            from .sleep.recovery import recover_undersized_scoring

            recover_undersized_scoring(db, root)
        self.load_reviews()
        self.path.chmod(0o600)

    @contextmanager
    def connect(self):
        # Each thread keeps one open connection per database and reuses it.
        # Opening and closing a connection for every call hung the serve
        # process twice: two threads stuck for good on SQLite's per-file lock
        # inside open (findReusableFd) and close (sqlite3WalClose). A nested
        # call on the same thread gets its own short-lived connection, as
        # before, so it never commits its caller's transaction.
        held = _CONNECTIONS.__dict__.setdefault("held", set())
        key = str(self.path)
        if key in held:
            db = _open(self.path)
            try:
                with db:
                    yield db
            finally:
                db.close()
            return
        db = _cached(self.path)
        held.add(key)
        try:
            with db:
                yield db
        except sqlite3.ProgrammingError:
            _forget(key)
            raise
        finally:
            held.discard(key)
            try:
                if db.in_transaction:
                    db.rollback()
            except sqlite3.Error:
                _forget(key)

    def get(self, key, default=None):
        with self.connect() as db:
            row = db.execute(
                "SELECT value FROM state WHERE key=?", (self.state_key(key),)
            ).fetchone()
            return json.loads(row[0]) if row else default

    def set(self, key, value):
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO state VALUES (?,?)",
                (self.state_key(key), json.dumps(value, allow_nan=False)),
            )

    def lessons(self):
        from .sleep.lessons import active_lessons

        return active_lessons(self)

    def pipeline(self):
        from .pipeline import metrics

        return metrics(self)

    def record(
        self,
        path,
        request,
        response,
        status=200,
        complete=True,
        binding="",
        started=None,
        import_key=None,
    ):
        # Giving space back (recordings.compact) can hold the write lock
        # longer than one connection timeout; a recording waits for it (in
        # the recording thread, never on the request path) instead of
        # being lost.
        for attempt in range(RECORD_ATTEMPTS):
            try:
                return self._record(
                    path, request, response, status, complete, binding, started, import_key
                )
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc) or attempt == RECORD_ATTEMPTS - 1:
                    raise

    def _record(self, path, request, response, status, complete, binding, started, import_key=None):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if import_key is not None and db.execute(
                "SELECT 1 FROM imports WHERE key=?", (import_key,)
            ).fetchone():
                return None
            cutoff = db.execute(
                "SELECT value FROM state WHERE key='forgotten_at'"
            ).fetchone()
            if started is not None and cutoff and started < json.loads(cutoff[0]):
                return None
            # Hashing and sharing happen here, in the recording thread, never
            # on the request path.
            skeleton, refs = parts.pack(db, request, parts.REQUEST_LISTS)
            # An exchange id is never given out twice: watermarks and read
            # signals refer to ids, and deleting the newest recordings (size
            # limit, forget) would otherwise let SQLite hand them out again.
            high = db.execute(
                "SELECT value FROM state WHERE key='exchange_id_high'"
            ).fetchone()
            last = db.execute("SELECT MAX(id) FROM exchanges").fetchone()[0] or 0
            cur = db.execute(
                "INSERT INTO exchanges(id,created,path,request,response,status,complete,binding,recorded_at,refs) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    max(last, json.loads(high[0]) if high else 0) + 1,
                    started if started is not None else time.time(),
                    path,
                    skeleton,
                    json.dumps(response),
                    status,
                    complete,
                    binding,
                    time.time(),
                    refs,
                ),
            )
            if import_key is not None:
                # One summary line for a whole import, not one per turn.
                db.execute(
                    "INSERT INTO imports VALUES (?,?,?,?)",
                    (import_key, cur.lastrowid, binding, time.time()),
                )
                return cur.lastrowid
            messages = request.get("messages") if isinstance(request, dict) else None
            db.execute(
                "INSERT INTO events(created,model_id,stage,text) VALUES (?,?,?,?)",
                (
                    time.time(),
                    "",
                    "request",
                    " · ".join(
                        filter(
                            None,
                            [
                                "agent request",
                                f"{len(messages)} messages"
                                if isinstance(messages, list)
                                else path,
                                "ok"
                                if status == 200 and complete
                                else f"status {status}",
                            ],
                        )
                    ),
                ),
            )
            return cur.lastrowid

    def exchanges(self):
        with self.connect() as db, parts.snapshot(db):
            cache = {}
            return [
                {k: v for k, v in dict(r).items() if k != "refs"}
                | {
                    "request": parts.request(db, r, cache),
                    "response": json.loads(r["response"]),
                }
                for r in db.execute(
                    "SELECT * FROM exchanges ORDER BY created,id"
                ).fetchall()
            ]

    def version(self, key=None):
        key = key or self.get("current")
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM versions WHERE id=?"
                + (" AND model_id=?" if self.model_key else ""),
                (key, self.model_id) if self.model_key else (key,),
            ).fetchone()
            return dict(row) if row else None

    def add_version(self, version):
        with self.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO versions(id,identifier,artifact,adapter,created,night,kind,base,training_adapter,model_id) VALUES (:id,:identifier,:artifact,:adapter,:created,:night,:kind,:base,:training_adapter,:model_id)",
                {"kind": None, "base": None, "training_adapter": None}
                | version
                | {"model_id": self.model_id},
            )

    def swap(self, target, expected, generation=None):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT value FROM state WHERE key=?", (self.state_key("current"),)
            ).fetchone()
            current = json.loads(row[0]) if row else None
            if current != expected:
                raise RuntimeError(
                    "Served version changed during check; candidate was not installed"
                )
            if generation is not None:
                saved = db.execute(
                    "SELECT value FROM state WHERE key=?",
                    (self.state_key("candidate_generation"),),
                ).fetchone()
                if not saved or json.loads(saved[0]) != generation:
                    raise RuntimeError(
                        "Candidate generation changed; candidate was not installed"
                    )
            if not db.execute(
                "SELECT 1 FROM versions WHERE id=? AND model_id=?",
                (target, self.model_id),
            ).fetchone():
                raise ValueError("Unknown version")
            for key, value in (("previous", current), ("current", target)):
                db.execute(
                    "INSERT OR REPLACE INTO state VALUES (?,?)",
                    (self.state_key(key), json.dumps(value)),
                )
            db.execute(
                "INSERT INTO events(created,model_id,stage,text) VALUES (?,?,?,?)",
                (
                    time.time(),
                    self.model_key or "",
                    "swap",
                    "original model in use"
                    if str(target).startswith("base")
                    else f"now serving {target}",
                ),
            )
            row = db.execute(
                "SELECT value FROM state WHERE key=?", (self.state_key("wake_last"),)
            ).fetchone()
            if row:
                wake = json.loads(row[0])
                if wake.get("candidate") == target:
                    wake["swapped"] = True
                    db.execute(
                        "UPDATE state SET value=? WHERE key=?",
                        (json.dumps(wake), self.state_key("wake_last")),
                    )

    def event(self, stage, text, *, model_id=None):
        """One line of the dashboard's Flow: what just happened, newest last.

        Best effort from any process; a busy store never fails the work."""
        # Flow names the model of every line, the primary model's too.
        model_id = (self.model_key or "") if model_id is None else model_id
        text = str(text)[:300]
        try:
            with self.connect() as db:
                if stage == "state":
                    # A state seen again lately is the same line: refresh its
                    # time instead of writing it once more.
                    shape = re.sub(r"[\d.,]+", "#", text)
                    same = next(
                        (
                            row[0]
                            for row in db.execute(
                                "SELECT id,text FROM events WHERE model_id=? "
                                "AND stage=? ORDER BY id DESC LIMIT 20",
                                (model_id, stage),
                            )
                            if re.sub(r"[\d.,]+", "#", row[1]) == shape
                        ),
                        None,
                    )
                    if same:
                        # Same state with new numbers: it moves to the newest
                        # line. A new row, not a new time: Windows' clock
                        # (Python 3.12 and older) ticks every ~15.6 ms, so
                        # lines within one tick share a time and only the id
                        # orders them.
                        db.execute("DELETE FROM events WHERE id=?", (same,))
                row = db.execute(
                    "INSERT INTO events(created,model_id,stage,text) VALUES (?,?,?,?)",
                    (time.time(), model_id, stage, text),
                ).lastrowid
                if row % 200 == 0:
                    db.execute("DELETE FROM events WHERE id <= ?", (row - 2000,))
        except sqlite3.Error:
            pass

    def events(self, limit=60, *, every_model=False):
        """Newest first: this model's events plus installation-wide ones, or
        every model's (each with its model_id) for the dashboard's one Flow."""
        with self.connect() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT created,model_id,stage,text FROM events "
                    + ("" if every_model else "WHERE model_id IN (:model, :key, '') ")
                    + "ORDER BY created DESC, id DESC LIMIT :limit",
                    {
                        "model": self.model_id,
                        "key": self.model_key or "",
                        "limit": limit,
                    },
                )
            ]

    def night(self, night_id, **updates):
        with self.connect() as db:
            if night_id is None:
                cur = db.execute(
                    "INSERT INTO nights(created,status,phase,record,model_id) VALUES (?,'running','select','{}',?)",
                    (time.time(), self.model_id),
                )
                night_id = cur.lastrowid
            if updates:
                allowed = {"status", "phase", "progress", "record"}
                if set(updates) - allowed:
                    raise ValueError("Invalid night field")
                updates = {
                    k: json.dumps(v, allow_nan=False) if k == "record" else v
                    for k, v in updates.items()
                }
                current = db.execute(
                    "SELECT phase,status FROM nights WHERE id=?", (night_id,)
                ).fetchone()
                moved = [
                    str(updates[k])
                    for k in ("phase", "status")
                    if k in updates and (current is None or current[k] != updates[k])
                ]
                if moved:
                    change = " · ".join(moved)
                    db.execute(
                        "INSERT INTO events(created,model_id,stage,text) VALUES (?,?,?,?)",
                        (
                            time.time(),
                            self.model_key or "",
                            "cycle",
                            f"#{night_id} {change}",
                        ),
                    )
                db.execute(
                    f"UPDATE nights SET {','.join(k + '=?' for k in updates)} WHERE id=?"
                    + (" AND model_id=?" if self.model_key else ""),
                    (*updates.values(), night_id, self.model_id)
                    if self.model_key
                    else (*updates.values(), night_id),
                )
            return night_id

    def nights(self):
        with self.connect() as db:
            return [
                dict(r) | {"record": json.loads(r["record"])}
                for r in db.execute(
                    "SELECT * FROM nights"
                    + (" WHERE model_id=?" if self.model_key else "")
                    + " ORDER BY id",
                    (self.model_id,) if self.model_key else (),
                )
            ]

    def pool(self):
        with self.connect() as db:
            eligible, params = self._learning_filter(db)
            return [
                json.loads(r[0])
                for r in db.execute(
                    "SELECT p.step FROM pool p LEFT JOIN steps s ON s.id=p.step_id WHERE s.retired_at IS NULL AND p.model_id=? AND "
                    + eligible
                    + " ORDER BY p.score DESC,p.step_id",
                    (self.model_id, *params),
                )
            ]

    def add_pool(self, steps, *, scored=()):
        from .steps import revision_id

        if any(s.get("split", "train") != "train" for s in steps):
            raise ValueError("Only training conversations can enter the pool")
        with self.connect() as db:
            # Rescoring replaces an older selection, including when the revised
            # evidence is no longer learnable. Preserve revisions that arrived
            # after this cycle's frozen scoring snapshot.
            for item in scored:
                row = db.execute(
                    "SELECT step FROM pool WHERE step_id=? AND model_id=?",
                    (item["id"], self.model_id),
                ).fetchone()
                if row and revision_id(json.loads(row[0])) <= revision_id(item):
                    db.execute(
                        "DELETE FROM pool WHERE step_id=? AND model_id=?",
                        (item["id"], self.model_id),
                    )
            current = []
            for item in steps:
                row = db.execute(
                    "SELECT payload FROM steps WHERE id=?", (item["id"],)
                ).fetchone()
                if row and revision_id(json.loads(row[0])) > revision_id(item):
                    continue
                self._put_step(db, item)
                current.append(item)
            db.executemany(
                "INSERT OR REPLACE INTO pool VALUES (?,?,?,?)",
                [
                    (s["id"], s.get("surprise", 0.0), json.dumps(s), self.model_id)
                    for s in current
                ],
            )

    def forget(self):
        with self.connect() as db:
            self._invalidate_pending(db)
            db.execute("PRAGMA secure_delete=ON")
            db.execute(
                "INSERT OR REPLACE INTO state VALUES ('forgotten_at',?)",
                (json.dumps(time.time()),),
            )
            # Later recordings take ids above every one deleted here.
            db.execute(
                "INSERT INTO state VALUES ('exchange_id_high',"
                " (SELECT json(MAX(COALESCE(MAX(id),0), COALESCE((SELECT CAST(value AS INTEGER) FROM state WHERE key='exchange_id_high'),0))) FROM exchanges))"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value"
            )
            db.execute("DELETE FROM exchanges")
            db.execute("DELETE FROM pool")
            db.execute("DELETE FROM steps")
            db.execute("DELETE FROM parts")
            db.execute("DELETE FROM part_uses")
            db.execute("DELETE FROM scoreboard")
            db.execute("DELETE FROM prefixes")
            db.execute("DELETE FROM reconstructed")
            db.execute("DELETE FROM metric_dirty")
            db.execute("DELETE FROM scored_revisions")
            db.execute("DELETE FROM step_signals")
            db.execute("DROP TABLE IF EXISTS dreams")
            db.execute("DELETE FROM lessons")
            db.execute("DELETE FROM reviews")
            # Read sessions, their signals, the model's answers and practice
            # scenes quote the conversations: the same forget applies.
            db.execute("DELETE FROM read_sessions")
            db.execute("DELETE FROM signals")
            db.execute("DELETE FROM reading_cache")
            db.execute("DELETE FROM practice")
            db.execute("DELETE FROM pruned_sessions")
            db.execute("DELETE FROM imports")
            # Cycle records keep their counts; example text quotes the
            # conversations and goes (docs/CONTRACTS.md, Integration decisions).
            for night_id, raw in db.execute("SELECT id, record FROM nights").fetchall():
                record = json.loads(raw)
                if record.get("examples"):
                    record["examples"] = [strip_example(e) for e in record["examples"]]
                    db.execute(
                        "UPDATE nights SET record=? WHERE id=?",
                        (json.dumps(record), night_id),
                    )
            db.execute("DELETE FROM events")
            db.execute("DROP TABLE IF EXISTS observations")
            # What cloud calls sent quotes the conversations; their costs stay
            # so tonight's budget still holds.
            from .cloud.ledger import forget_previews

            forget_previews(db)
            for row in db.execute("SELECT id FROM models").fetchall():
                db.execute(
                    "INSERT OR REPLACE INTO state VALUES (?, 'true')",
                    (self.scoped(row[0]).state_key("rebuild_required"),),
                )
            db.execute(
                "INSERT OR REPLACE INTO state VALUES ('rebuild_required','true')"
            )
            db.execute(
                "DELETE FROM state WHERE key IN ('experience_watermark','score_watermark','teacher_gain','unfit_samples','practice_revisions','rejected_practice') OR key LIKE 'model:%:practice_revisions' OR key LIKE 'model:%:rejected_practice' OR key LIKE 'model:%:experience_watermark' OR key LIKE 'model:%:score_watermark' OR key LIKE 'model:%:teacher_gain' OR key LIKE 'model:%:unfit_samples'"
            )
            db.execute(
                "DELETE FROM state WHERE key IN ('wake_last','dream_volume') OR key LIKE 'model:%:wake_last' OR key LIKE 'model:%:dream_volume'"
            )
            # Quick Dream lessons are written from the conversations.
            db.execute(
                "DELETE FROM state WHERE key IN ('prompt_lessons','absorbed_signals','trained_signals','quick_seen','quick_last','quick_history','seen_dreams','quick_checked','dream_stopped','dream_pending_wait','dream_preparing','machine_measured','memory_wait','dream_memory_waits','dream_capacity_edges','fit_wait') OR key LIKE 'model:%:fit_wait' OR key LIKE 'model:%:dream_capacity_edges' OR key LIKE 'model:%:memory_wait' OR key LIKE 'model:%:dream_memory_waits' OR key LIKE 'model:%:dream_pending_wait' OR key LIKE 'model:%:dream_preparing' OR key LIKE 'model:%:machine_measured' OR key LIKE 'model:%:quick_checked' OR key LIKE 'model:%:dream_stopped' OR key LIKE 'model:%:quick_history' OR key LIKE 'model:%:seen_dreams' OR key LIKE 'model:%:prompt_lessons' OR key LIKE 'model:%:absorbed_signals' OR key LIKE 'model:%:trained_signals' OR key LIKE 'model:%:quick_seen' OR key LIKE 'model:%:quick_last'"
            )
        from . import steps

        steps.REVIEWED.clear()
        steps.REVIEW_NOTES.clear()
        with self.connect() as db:
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            db.execute("VACUUM")
        # Summaries are derived conversation data, subject to the same forget.
        cache = self.root / "context-summaries"
        if cache.is_dir() and not cache.is_symlink():
            for path in cache.glob("*.json"):
                if not path.is_symlink():
                    path.unlink()

    def _invalidate_pending(self, db):
        db.execute(
            "UPDATE nights SET status='discarded' WHERE status IN ('running','interrupted','ready','waiting')"
            + (" AND model_id=?" if self.model_key else ""),
            (self.model_id,) if self.model_key else (),
        )
        db.execute(
            "INSERT OR REPLACE INTO state VALUES (?,?)",
            (self.state_key("candidate_generation"), json.dumps(str(uuid.uuid4()))),
        )

        if not self.model_key:
            db.execute(
                "UPDATE state SET value=? WHERE key LIKE 'model:%:candidate_generation'",
                (json.dumps(str(uuid.uuid4())),),
            )

    def invalidate_pending(self):
        with self.connect() as db:
            self._invalidate_pending(db)

    @staticmethod
    def _put_step(db, item):
        from .sleep.lessons import signal_counts

        settled = db.execute(
            "SELECT conversation,split FROM steps WHERE id=? AND split!='pending'",
            (item["id"],),
        ).fetchone()
        if settled:
            # A step already trained on or tested keeps the group it was
            # settled in, however its session is cut now.
            item = item | {"conversation": settled[0], "split": settled[1]}
        elif item.get("split", "pending") == "pending":
            # Later steps join their conversation's settled split.
            row = db.execute(
                "SELECT split FROM steps WHERE conversation=? AND split!='pending' LIMIT 1",
                (item.get("conversation", item["id"]),),
            ).fetchone()
            item = item | {"split": row[0] if row else "pending"}

        db.execute("INSERT OR IGNORE INTO metric_dirty VALUES (?)", (item["id"],))
        db.execute(
            "INSERT OR REPLACE INTO step_signals VALUES (?,?)",
            (item["id"], json.dumps(signal_counts([item]))),
        )
        old = db.execute("SELECT refs FROM steps WHERE id=?", (item["id"],)).fetchone()
        payload, refs = parts.pack(db, item, parts.STEP_LISTS)
        db.execute(
            "INSERT INTO steps(id,conversation,split,signal,strength,evidence,answer_hint,env_fingerprint,retired_at,retired_reason,payload,refs) VALUES (?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET conversation=excluded.conversation,signal=excluded.signal,strength=excluded.strength,evidence=excluded.evidence,answer_hint=excluded.answer_hint,payload=excluded.payload,refs=excluded.refs",
            (
                item["id"],
                item.get("conversation", item["id"]),
                item.get("split", "train"),
                item.get("signal"),
                item.get("strength", 0),
                # The evidence messages are the payload's "evidence", stored
                # once (parts.py); this column stays empty from schema v3. No
                # reader used it, and a second copy of a looping session's
                # evidence grew as its square (issue #73).
                "[]",
                json.dumps(item.get("answer_hint")),
                item.get("env_fingerprint", ""),
                item.get("retired_at"),
                item.get("retired_reason"),
                payload,
                refs,
            ),
        )
        if old and old[0]:
            # Parts only the replaced payload used (a hindsight that grew
            # keeps them all; a changed one frees some).
            parts.collect(db, parts.ids_of(old[0]))

    def sync_steps(self, steps):
        """Retire evidence only when a tool it actually used changes schema.

        Unrelated tools appearing/disappearing in another conversation do not
        invalidate experience. The latest observed schema for each name wins.
        """
        from .steps import digest

        items = [s.dict() if hasattr(s, "dict") else s for s in steps]
        latest = {}
        conversation_env = {}
        for item in items:
            conversation_env[item.get("conversation", item["id"])] = digest(
                item.get("tools", [])
            )
            for tool in item.get("tools", []):
                latest[tool.get("function", {}).get("name")] = digest(tool)
        retired = 0
        with self.connect() as db:
            for item in items:
                self._put_step(db, item)
                existing = db.execute(
                    "SELECT score FROM pool WHERE step_id=? AND model_id=?",
                    (item["id"], self.model_id),
                ).fetchone()
                if existing:
                    db.execute(
                        "UPDATE pool SET step=? WHERE step_id=? AND model_id=?",
                        (
                            json.dumps(item | {"surprise": existing[0]}),
                            item["id"],
                            self.model_id,
                        ),
                    )
            cache = {}
            for row in db.execute(
                "SELECT * FROM steps WHERE retired_at IS NULL"
            ).fetchall():
                item = parts.payload(db, row, cache)
                if (
                    item.get("conversation") in conversation_env
                    and conversation_env[item["conversation"]]
                    != digest(item.get("tools", []))
                ) or any(
                    latest.get(t.get("function", {}).get("name"), digest(t))
                    != digest(t)
                    for t in item.get("tools", [])
                ):
                    db.execute(
                        "UPDATE steps SET retired_at=?,retired_reason='tool schema changed' WHERE id=?",
                        (time.time(), row["id"]),
                    )
                    retired += 1
            db.execute(
                "INSERT OR REPLACE INTO state VALUES ('environment_fingerprint',?)",
                (json.dumps(digest(latest)),),
            )
            db.execute(
                "INSERT OR REPLACE INTO state VALUES ('experience_revision',?)",
                (json.dumps(str(uuid.uuid4())),),
            )
        from .sleep.lessons import prune_lessons

        prune_lessons(self, all_models=True)
        return retired

    def learning_policy(self, db):
        if not self.model_key:
            return "all", None  # The unscoped view reconstructs shared history.
        row = db.execute(
            "SELECT * FROM models WHERE id=?", (self.model_key,)
        ).fetchone()
        if not row:
            return "off", None
        if "learning_level" not in row.keys():
            return ("new" if row["enabled"] else "off"), None
        return row["learning_level"], row["learning_cutoff"]

    def _learning_filter(self, db):
        level, cutoff = self.learning_policy(db)
        if level == "all":
            return NOT_IMPORTED, ()
        if level == "off" or cutoff is None:
            return "0", ()
        # Use the original recording time, never a later hindsight revision.
        # A model's first request starts before registration but is recorded
        # afterward, so request-start time would incorrectly exclude it.
        return (
            "EXISTS (SELECT 1 FROM exchanges e WHERE e.id=json_extract(s.payload,'$.source_id') AND COALESCE(e.recorded_at,e.created)>?)"
            " AND " + NOT_IMPORTED,
            (cutoff,),
        )

    def test_moment_conversations(self):
        """The conversation of each active, eligible step with feedback
        (hindsight): what heldout_units counts. SQLite reads the two fields;
        no step payload (a whole agent history each) becomes Python objects
        (live: 1,495 steps of 128k-token sessions decoded on every dashboard
        refresh starved the API)."""
        with self.connect() as db:
            eligible, params = self._learning_filter(db)
            rows = db.execute(
                "SELECT json_extract(s.payload,'$.conversation') FROM steps s"
                " WHERE s.retired_at IS NULL AND " + eligible + " AND " + HINDSIGHT,
                params,
            )
            return [row[0] for row in rows]

    def sample_revisions(self):
        """{step id: revision} of calibration_state.sample_pool (active steps
        with feedback, and the pool's newer revisions), read by SQLite: no
        step payload becomes Python objects. Measuring writes progress every
        few seconds, so cached answers rarely last; decoding every step for
        each status check starved the service (#71: `songe status` timed out
        and the dashboard took 25 s to connect while measuring)."""
        revision = (
            "MAX(COALESCE(json_extract({0},'$.source_id'),0),"
            " COALESCE(json_extract({0},'$.observed_id'),0))"
        )
        with self.connect() as db:
            eligible, params = self._learning_filter(db)
            found = dict(
                db.execute(
                    "SELECT json_extract(s.payload,'$.id'), "
                    + revision.format("s.payload")
                    + " FROM steps s WHERE s.retired_at IS NULL AND "
                    + eligible
                    + " AND "
                    + HINDSIGHT,
                    params,
                ).fetchall()
            )
            for step_id, value in db.execute(
                "SELECT json_extract(p.step,'$.id'), "
                + revision.format("p.step")
                + " FROM pool p LEFT JOIN steps s ON s.id=p.step_id"
                " WHERE s.retired_at IS NULL AND p.model_id=? AND " + eligible,
                (self.model_id, *params),
            ):
                found[step_id] = max(value, found.get(step_id, value))
            return found

    def active_steps(self, split=None):
        with self.connect() as db, parts.snapshot(db):
            eligible, params = self._learning_filter(db)
            rows = db.execute(
                "SELECT s.* FROM steps s WHERE s.retired_at IS NULL AND "
                + eligible
                + (" AND s.split=?" if split else ""),
                (*params, split) if split else params,
            ).fetchall()
            cache = {}
            return [parts.payload(db, row, cache) for row in rows]

    def reconstruct(self, public_model=None, batch=32):
        """Persist prefix progress one bounded request at a time, in a guarded child.

        Unresolved continuations remain retryable after a late parent commits,
        for UNLINKED_AFTER seconds; then they start a conversation of their own.
        A restart repeats only unresolved/new rows, never the complete log.
        A recording that cannot be read (a request the model server refused,
        with messages that are not a list of messages; text that cannot be
        encoded) is passed over for good, so the ones after it go on.
        """
        from .steps import Step, digest, extract, prefix_keys, unsummarised

        changed = 0
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute("SELECT 1 FROM state WHERE key='prefix_keys_v2'").fetchone():
                # Once: every recorded turn also under the normalised key, so
                # turns echoed back differently link up (live: LM Studio's
                # "\n\n" before a tool call, echoed by pi as null; 22 turns
                # stuck unresolved). Unresolved rows are retried below.
                for step_id, root in db.execute("SELECT DISTINCT step_id, root FROM prefixes").fetchall():
                    saved = db.execute("SELECT * FROM steps WHERE id=?", (step_id,)).fetchone()
                    if not saved:
                        continue
                    step = parts.payload(db, saved)
                    if step.get("context") is None or step.get("action") is None:
                        continue
                    db.execute("INSERT OR IGNORE INTO prefixes VALUES (?,?,?)",
                               (digest(step["context"] + [step["action"]]), step_id, root))
                db.execute("INSERT OR REPLACE INTO state VALUES ('prefix_keys_v2', 'true')")
            rows = db.execute(
                "SELECT e.* FROM exchanges e LEFT JOIN reconstructed r ON r.exchange_id=e.id WHERE r.exchange_id IS NULL OR r.resolved=0 ORDER BY r.exchange_id IS NOT NULL,e.created,e.id LIMIT ?",
                (batch,),
            ).fetchall()
            cache = {}
            for raw in rows:
                turned = False
                db.execute("SAVEPOINT recording")
                try:
                    row = dict(raw) | {
                        "request": parts.request(db, raw, cache),
                        "response": json.loads(raw["response"]),
                    }
                    messages = (
                        row["request"].get("messages", [])
                        if isinstance(row["request"], dict)
                        else []
                    )
                    prefixes, parents = {}, {}
                    bare = unsummarised(messages) or []
                    for key in [k for n in range(1, len(messages) + 1) for k in prefix_keys(messages[:n])] + [
                        k for n in range(1, len(bare) + 1) for k in prefix_keys(bare[:n])
                    ]:
                        link = db.execute(
                            "SELECT step_id,root FROM prefixes WHERE hash=?", (key,)
                        ).fetchone()
                        if link:
                            saved = db.execute(
                                "SELECT * FROM steps WHERE id=?", (link[0],)
                            ).fetchone()
                            if saved:
                                prefixes[key] = (link[0], link[1])
                                parents[link[0]] = Step.load(parts.payload(db, saved, cache))
                    # Waited long enough for a parent that was never
                    # recorded: read as a conversation of its own (#263).
                    waited = time.time() - (raw["recorded_at"] or raw["created"])
                    items = extract([row], public_model, prefixes=prefixes, steps=parents,
                                    unlinked=waited > UNLINKED_AFTER)
                    resolved = bool(items) or not any(
                        m.get("role") == "assistant" for m in messages
                    )
                    if resolved:
                        for item in items:
                            self._put_step(db, item.dict())
                        db.executemany(
                            "INSERT OR REPLACE INTO prefixes VALUES (?,?,?)",
                            [(key, *link) for key, link in prefixes.items()],
                        )
                        turned = True
                    db.execute(
                        "INSERT OR REPLACE INTO reconstructed VALUES (?,?)",
                        (row["id"], int(resolved)),
                    )
                except (TypeError, AttributeError, ValueError, KeyError, IndexError):
                    # Left as recorded and never tried again: one recording
                    # must not hold back every one made after it.
                    db.execute("ROLLBACK TO recording")
                    logging.getLogger("songe").warning(
                        "reconstruct_unreadable %s", raw["id"], exc_info=True
                    )
                    db.execute(
                        "INSERT OR REPLACE INTO reconstructed VALUES (?,1)", (raw["id"],)
                    )
                    turned = False
                db.execute("RELEASE recording")
                changed += turned
            if changed:
                db.execute(
                    "INSERT OR REPLACE INTO state VALUES ('experience_revision',?)",
                    (json.dumps(str(uuid.uuid4())),),
                )
        if changed:
            self.event(
                "record",
                f"{changed} request{'s' if changed != 1 else ''} turned into training steps",
                model_id="",
            )
        return changed, len(rows)

    def unscored(self, steps):
        from .steps import revision_id

        with self.connect() as db:
            level, cutoff = self.learning_policy(db)
            return [
                s
                for s in steps
                if level != "off"
                and (
                    level == "all"
                    or cutoff is not None
                    and db.execute(
                        "SELECT 1 FROM exchanges WHERE id=? AND COALESCE(recorded_at,created)>?",
                        (s.get("source_id"), cutoff),
                    ).fetchone()
                )
                and (
                    (
                        row := db.execute(
                            "SELECT revision FROM scored_revisions WHERE step_id=? AND model_id=?",
                            (s["id"], self.model_id),
                        ).fetchone()
                    )
                    is None
                    or row[0] < revision_id(s)
                )
            ]

    def mark_scored(self, steps):
        from .steps import revision_id

        with self.connect() as db:
            db.executemany(
                "INSERT INTO scored_revisions VALUES (?,?,?) ON CONFLICT(step_id,model_id) DO UPDATE SET revision=MAX(revision,excluded.revision)",
                [(s["id"], revision_id(s), self.model_id) for s in steps],
            )
            db.execute(
                "INSERT OR REPLACE INTO state VALUES ('experience_revision',?)",
                (json.dumps(str(uuid.uuid4())),),
            )

    def release_unfit(self, keep, released):
        """Return revisions whose capacity verdict no longer holds to scoring."""
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO state VALUES (?,?)",
                (self.state_key("unfit_samples"), json.dumps(keep, allow_nan=False)),
            )
            db.executemany(
                "DELETE FROM scored_revisions WHERE step_id=? AND revision=? AND model_id=?",
                [(v["id"], v["revision"], self.model_id) for v in released],
            )
            db.execute(
                "INSERT OR REPLACE INTO state VALUES ('experience_revision',?)",
                (json.dumps(str(uuid.uuid4())),),
            )

    def feedback_counts(self):
        """Examples with an observed outcome, and how many conversations hold them."""
        with self.connect() as db:
            eligible, params = self._learning_filter(db)
            row = db.execute(
                "SELECT COUNT(*),COUNT(DISTINCT s.conversation) FROM steps s "
                "WHERE s.retired_at IS NULL AND json_array_length(json_extract(s.payload,'$.hindsight'))>0 AND "
                + eligible,
                params,
            ).fetchone()
            return row[0] or 0, row[1] or 0

    def assign_splits(self, *, fraction=0.2, minimum=2):
        """Settle waiting conversations: about `fraction` of all conversations
        with feedback become tests (at least `minimum`), drawn across the kinds
        of feedback present so tests resemble training. Settled splits never
        change, so tested conversations are never trained on."""
        from collections import Counter

        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute(
                "SELECT conversation,split,signal FROM steps WHERE retired_at IS NULL "
                "AND json_array_length(json_extract(payload,'$.hindsight'))>0"
            ).fetchall()
            splits, kinds = {}, {}
            for conversation, split, signal in rows:
                if splits.get(conversation, "pending") == "pending":
                    splits[conversation] = split
                kinds.setdefault(conversation, Counter())[signal or "none"] += 1
            waiting = sorted(c for c, s in splits.items() if s == "pending")
            if not waiting:
                return 0
            held = sum(s == "heldout" for s in splits.values())
            target = max(minimum, round(fraction * len(splits)))
            room = len(splits) - held - 1  # at least one conversation trains
            need = max(0, min(target - held, room))
            groups = {}
            for conversation in waiting:
                kind = kinds[conversation].most_common(1)[0][0]
                groups.setdefault(kind, []).append(conversation)
            order = []
            ranked = sorted(groups.values(), key=len, reverse=True)
            while any(ranked):
                for group in ranked:
                    if group:
                        order.append(group.pop(0))
            # A one-step part (a greeting) tests little: longer ones go first.
            order.sort(key=lambda c: sum(kinds[c].values()) < 2)
            tests = set(order[:need])
            for conversation in waiting:
                split = "heldout" if conversation in tests else "train"
                db.execute(
                    "UPDATE steps SET split=?,payload=json_set(payload,'$.split',?) "
                    "WHERE conversation=? AND split='pending'",
                    (split, split, conversation),
                )
            db.execute(
                "INSERT OR REPLACE INTO state VALUES ('experience_revision',?)",
                (json.dumps(str(uuid.uuid4())),),
            )
            return len(waiting)

    def load_reviews(self):
        """Make the local model's reading of user replies the signal source."""
        from . import steps

        with self.connect() as db:
            for key, kind, payload in db.execute(
                "SELECT key,kind,payload FROM reviews"
            ):
                try:
                    note = json.loads(payload or "{}")
                except ValueError:
                    note = {}
                # Reply kind follows the words; issue and lesson stay with
                # the one reply event they were read from.
                steps.REVIEWED[note.get("text_key") or key] = kind
                steps.REVIEW_NOTES[key] = note

    def reviewed(self, keys):
        with self.connect() as db:
            return {
                r[0]
                for r in db.execute(
                    f"SELECT key FROM reviews WHERE key IN ({','.join('?' * len(keys))})",
                    list(keys),
                )
            }

    def add_review(self, key, kind, payload, step_ids, *, since=None):
        """Record a reviewed reply and recompute the signals of steps it ends.
        A review begun before `forget` is dropped, not written afterwards."""
        from . import steps
        from .steps import Step, signal_for

        with self.connect() as db:
            if since is not None:
                row = db.execute(
                    "SELECT value FROM state WHERE key='forgotten_at'"
                ).fetchone()
                if row and json.loads(row[0]) >= since:
                    return
            steps.REVIEWED[payload.get("text_key") or key] = kind
            steps.REVIEW_NOTES[key] = payload
            db.execute(
                "INSERT OR REPLACE INTO reviews VALUES (?,?,?,?)",
                (key, kind, json.dumps(payload, ensure_ascii=False), time.time()),
            )
            note = " · ".join(
                filter(
                    None,
                    [
                        str(payload.get("reply") or kind or "reviewed").replace(
                            "_", " "
                        ),
                        payload.get("issue"),
                        payload.get("lesson") and "lesson: " + payload["lesson"],
                    ],
                )
            )
            db.execute(
                "INSERT INTO events(created,model_id,stage,text) VALUES (?,?,?,?)",
                (time.time(), "", "review", "your reply · " + note),
            )
            for step_id in step_ids:
                row = db.execute(
                    "SELECT * FROM steps WHERE id=?", (step_id,)
                ).fetchone()
                if not row:
                    continue
                step = Step.load(parts.payload(db, row))
                step.signal, step.strength, step.evidence, step.answer_hint = (
                    signal_for(step)
                )
                self._put_step(db, step.dict())

    def mark_unfit(self, key, entry, step):
        """Commit the verdict and consume this revision in one transaction."""
        from .steps import revision_id

        with self.connect() as db:
            row = db.execute(
                "SELECT value FROM state WHERE key=?",
                (self.state_key("unfit_samples"),),
            ).fetchone()
            unfit = json.loads(row[0]) if row else {}
            unfit[key] = entry
            db.execute(
                "INSERT OR REPLACE INTO state VALUES (?,?)",
                (self.state_key("unfit_samples"), json.dumps(unfit, allow_nan=False)),
            )
            if step.get("split") == "train":
                db.execute(
                    "INSERT INTO scored_revisions VALUES (?,?,?) ON CONFLICT(step_id,model_id) DO UPDATE SET revision=MAX(revision,excluded.revision)",
                    (step["id"], revision_id(step), self.model_id),
                )
                row = db.execute(
                    "SELECT step FROM pool WHERE step_id=? AND model_id=?",
                    (step["id"], self.model_id),
                ).fetchone()
                if row and revision_id(json.loads(row[0])) <= revision_id(step):
                    db.execute(
                        "DELETE FROM pool WHERE step_id=? AND model_id=?",
                        (step["id"], self.model_id),
                    )
                db.execute(
                    "INSERT OR REPLACE INTO state VALUES ('experience_revision',?)",
                    (json.dumps(str(uuid.uuid4())),),
                )

    def experience_counts(self):
        with self.connect() as db:
            eligible, params = self._learning_filter(db)
            row = db.execute(
                """SELECT
                SUM(split='train'),
                COUNT(DISTINCT CASE WHEN split='heldout' THEN conversation END),
                SUM(split='train' AND MAX(COALESCE(json_extract(s.payload,'$.source_id'),0),COALESCE(json_extract(s.payload,'$.observed_id'),0))>COALESCE(scored_revisions.revision,-1))
                FROM steps s LEFT JOIN scored_revisions ON scored_revisions.step_id=s.id AND scored_revisions.model_id=? WHERE retired_at IS NULL AND json_array_length(json_extract(s.payload,'$.hindsight'))>0 AND """
                + eligible,
                (self.model_id, *params),
            ).fetchone()
            return tuple(v or 0 for v in row)

    def pending_learnable(self):
        """Count evidence still to score or selected but not yet trained.

        Steps not yet split count too: splits are settled only once a model
        is chosen to learn, so leaving them out kept a first cycle from ever
        being chosen."""
        with self.connect() as db:
            eligible, params = self._learning_filter(db)
            return db.execute(
                """SELECT COUNT(*) FROM steps s
                LEFT JOIN scored_revisions r ON r.step_id=s.id AND r.model_id=?
                LEFT JOIN pool p ON p.step_id=s.id AND p.model_id=?
                WHERE s.retired_at IS NULL AND s.split IN ('train','pending')
                AND json_array_length(json_extract(s.payload,'$.hindsight'))>0
                AND (MAX(COALESCE(json_extract(s.payload,'$.source_id'),0),COALESCE(json_extract(s.payload,'$.observed_id'),0))>COALESCE(r.revision,-1)
                  OR (MAX(COALESCE(json_extract(p.step,'$.source_id'),0),COALESCE(json_extract(p.step,'$.observed_id'),0))>?
                    AND MAX(COALESCE(json_extract(p.step,'$.source_id'),0),COALESCE(json_extract(p.step,'$.observed_id'),0))>COALESCE((SELECT value FROM json_each(?) WHERE key=s.id),-1))) AND """
                + eligible,
                (
                    self.model_id,
                    self.model_id,
                    self.get("experience_watermark", 0),
                    json.dumps(self.get("practice_revisions", {})),
                    *params,
                ),
            ).fetchone()[0]

    def retirement_fraction(self):
        with self.connect() as db:
            active, retired = db.execute(
                "SELECT SUM(retired_at IS NULL),SUM(retired_at > ?) FROM steps",
                (self.get("rebuilt_at", 0),),
            ).fetchone()
        total = (active or 0) + (retired or 0)
        return (retired or 0) / total if total else 0.0

    def slice(
        self,
        slice_id=None,
        *,
        phase="record",
        units_done=0,
        interrupted_by=None,
        ended=None,
        record=None,
    ):
        with self.connect() as db:
            if slice_id is None:
                return db.execute(
                    "INSERT INTO slices(started,phase,model_id) VALUES (?,?,?)",
                    (time.time(), phase, self.model_id),
                ).lastrowid
            db.execute(
                "UPDATE slices SET ended=?,phase=?,units_done=?,interrupted_by=? WHERE id=?",
                (
                    time.time() if ended is None else ended,
                    phase,
                    units_done,
                    interrupted_by,
                    slice_id,
                ),
            )
            if record is not None:
                db.execute(
                    "UPDATE slices SET record=? WHERE id=?",
                    (json.dumps(record, allow_nan=False), slice_id),
                )
            return slice_id

    def last_slice(self):
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM slices"
                + (" WHERE model_id=?" if self.model_key else "")
                + " ORDER BY id DESC LIMIT 1",
                (self.model_id,) if self.model_key else (),
            ).fetchone()
        return dict(row) if row else None

    def slices(self):
        with self.connect() as db:
            return [
                dict(r) | {"record": json.loads(r["record"])}
                for r in db.execute(
                    "SELECT * FROM slices"
                    + (" WHERE model_id=?" if self.model_key else "")
                    + " ORDER BY id",
                    (self.model_id,) if self.model_key else (),
                )
            ]

    def traffic_times(self, limit=512):
        with self.connect() as db:
            rows = db.execute(
                "SELECT created FROM exchanges ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return sorted(r[0] for r in rows)

    def scoreboard(self):
        # Compatibility entry; production reconstruction updates these rows in
        # bounded, guarded units rather than materializing exchanges here.
        self.reconstruct()
        return self.scoreboard_from_steps()

    def scoreboard_from_steps(self):
        from datetime import datetime, timezone

        from .steps import tool_outcome, valid_action

        errors = {
            "tool_error",
            "exec_failure",
            "static_invalid",
            "loop",
            "user_correction",
        }
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("""CREATE TABLE IF NOT EXISTS observations (
                step_id TEXT PRIMARY KEY, day TEXT, binding TEXT, task TEXT,
                error_key TEXT, call_valid INTEGER, drift REAL)""")
            # Stream payloads: even a one-time upgrade never builds an entire
            # exchange log in the serving process. This method runs in a child.
            cache = {}
            for saved in db.execute(
                "SELECT steps.* FROM steps JOIN metric_dirty ON step_id=steps.id WHERE retired_at IS NULL"
            ):
                step = parts.payload(db, saved, cache)
                source = db.execute(
                    "SELECT created,binding,response FROM exchanges WHERE id=?",
                    (step["source_id"],),
                ).fetchone()
                if source is None:
                    continue
                failed = step.get("signal") in errors or any(
                    tool_outcome(m) in errors
                    for m in step["hindsight"]
                    if m.get("role") == "tool"
                )
                action = step["action"]
                key = (
                    digest(
                        [c.get("function", c) for c in action.get("tool_calls", [])]
                        or action
                    )
                    if failed
                    else None
                )
                response = json.loads(source["response"]) or {}
                drift = (response.get("metrics") or {}).get("general_drift")
                day = (
                    datetime.fromtimestamp(source["created"], timezone.utc)
                    .date()
                    .isoformat()
                )
                db.execute(
                    "INSERT OR REPLACE INTO observations VALUES (?,?,?,?,?,?,?)",
                    (
                        step["id"],
                        day,
                        source["binding"],
                        step.get("task") or step["conversation"],
                        key,
                        int(valid_action(action, step["tools"]))
                        if action.get("tool_calls")
                        else None,
                        drift if isinstance(drift, (float, int)) else None,
                    ),
                )
            db.execute("DELETE FROM metric_dirty")
            db.execute("DELETE FROM scoreboard")
            db.execute("""INSERT INTO scoreboard SELECT day,binding,
                CASE WHEN COUNT(error_key)>0 THEN 1.0-COUNT(DISTINCT error_key)*1.0/COUNT(error_key) ELSE 0.0 END,
                COUNT(*)*1.0/COUNT(DISTINCT task),AVG(call_valid),AVG(drift)
                FROM observations GROUP BY day,binding""")
            return [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM scoreboard ORDER BY day,model_version"
                )
            ]
