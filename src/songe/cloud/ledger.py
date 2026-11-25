"""Every cloud call, in Songe's store: what was sent, what it cost.

One row per call (or per call refused before sending), so a Dream record
can list what left the machine and the night's spending can be capped. A
night runs from noon to noon in local time, so a Dream that crosses
midnight stays in one night.
"""

from __future__ import annotations

import sqlite3
import time

SCHEMA = """CREATE TABLE IF NOT EXISTS cloud_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at REAL NOT NULL,
    night TEXT NOT NULL,
    dream TEXT,
    purpose TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    sent INTEGER NOT NULL,
    ok INTEGER NOT NULL,
    error TEXT,
    sent_chars INTEGER NOT NULL DEFAULT 0,
    preview TEXT NOT NULL DEFAULT '',
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    cost_usd REAL,
    cost_source TEXT NOT NULL DEFAULT 'unknown'
)"""

PREVIEW_CHARS = 240


def night_key(at: float | None = None) -> str:
    """Noon to noon, local time: the night a call belongs to."""
    at = time.time() if at is None else at
    # Not below 0: Windows' localtime refuses negative times (a test clock).
    return time.strftime("%Y-%m-%d", time.localtime(max(at - 12 * 3600, 0)))


def _exists(db) -> bool:
    return bool(
        db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='cloud_calls'"
        ).fetchone()
    )


def record(store, **row) -> int:
    row.setdefault("at", time.time())
    row.setdefault("night", night_key(row["at"]))
    row["preview"] = (row.get("preview") or "")[:PREVIEW_CHARS]
    columns = ",".join(row)
    with store.connect() as db:
        db.execute(SCHEMA)
        cursor = db.execute(
            f"INSERT INTO cloud_calls ({columns}) VALUES ({','.join('?' * len(row))})",
            tuple(row.values()),
        )
        return cursor.lastrowid


def calls(store, *, night=None, dream=None, since=None, until=None, limit=200):
    """Calls, oldest first, filtered by night, Dream id or time window."""
    where, args = [], []
    if night is not None:
        where.append("night=?")
        args.append(night)
    if dream is not None and since is not None:
        # A Dream's own calls, and untagged ones made while it ran.
        where.append("(dream=? OR (dream IS NULL AND at>=? AND at<=?))")
        args += [dream, since, until if until is not None else time.time()]
    elif dream is not None:
        where.append("dream=?")
        args.append(dream)
    elif since is not None:
        where.append("at>=? AND at<=?")
        args += [since, until if until is not None else time.time()]
    query = "SELECT * FROM cloud_calls"
    if where:
        query += " WHERE " + " AND ".join(where)
    query += " ORDER BY id DESC LIMIT ?"
    try:
        with store.connect() as db:
            if not _exists(db):
                return []
            found = [dict(r) for r in db.execute(query, (*args, limit))]
    except sqlite3.Error:
        return []
    for row in found:
        row["sent"], row["ok"] = bool(row["sent"]), bool(row["ok"])
    return list(reversed(found))


def spent(store, night=None) -> dict:
    """The night's calls and cost. cost_known is False when some sent call's
    cost is unknown (a custom endpoint that reports none)."""
    night = night or night_key()
    sent = [c for c in calls(store, night=night, limit=100000) if c["sent"]]
    return {
        "night": night,
        "calls": len(sent),
        "usd": round(sum(c["cost_usd"] or 0.0 for c in sent), 6),
        "cost_known": all(c["cost_usd"] is not None for c in sent if c["ok"]),
        "tokens": sum((c["prompt_tokens"] or 0) + (c["completion_tokens"] or 0) for c in sent),
    }


def latest_night(store):
    """The most recent night with any call, or None."""
    try:
        with store.connect() as db:
            if not _exists(db):
                return None
            row = db.execute("SELECT night FROM cloud_calls ORDER BY id DESC LIMIT 1").fetchone()
    except sqlite3.Error:
        return None
    return row[0] if row else None


def forget_previews(db) -> None:
    """`songe forget`: what was sent goes; the night's spending stays."""
    if _exists(db):
        db.execute("UPDATE cloud_calls SET preview=''")
