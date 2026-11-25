"""The cloud API key: in the OS keychain, never in Songe's config or API.

macOS Keychain, Windows Credential Manager or the Linux Secret Service, via
`keyring`. Where no keychain is available (a headless Linux box without a
Secret Service) the key goes to a file only this user can read. The
environment variable SONGE_CLOUD_API_KEY, when set, supplies the key for
that process; it never turns cloud learning on by itself, and no other
variable (OPENAI_API_KEY, ...) is ever read.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

from ..storage import atomic_write, private_acl

ENV = "SONGE_CLOUD_API_KEY"
SERVICE = "songe"
FILE = "cloud-key"

logger = logging.getLogger("songe")

# Tests replace this with an in-memory keychain (or False for "none").
BACKEND = None


def _account(root: Path) -> str:
    # One key per installation: a test or second data folder never reads
    # or overwrites the user's real key.
    digest = hashlib.sha256(str(Path(root).resolve()).encode()).hexdigest()[:12]
    return f"cloud-api-key:{digest}"


def keychain():
    """The OS keychain, or None when this machine has none usable."""
    if BACKEND is not None:
        return BACKEND or None
    try:
        import keyring
        from keyring.backends import fail, null

        found = keyring.get_keyring()
        if isinstance(found, (fail.Keyring, null.Keyring)):
            return None
        if getattr(found, "backends", None) == []:  # a chainer with nothing in it
            return None
        return found
    except Exception as exc:  # noqa: BLE001 - any import/backend failure: no keychain
        logger.info("cloud_keychain_unavailable %s", exc)
        return None


def _file(root: Path) -> Path:
    return Path(root) / FILE


def save(root: Path, key: str) -> str:
    """Store the key; returns where it went: "keychain" or "file"."""
    key = (key or "").strip()
    if not key:
        raise ValueError("The API key is empty")
    chain = keychain()
    if chain is not None:
        try:
            chain.set_password(SERVICE, _account(root), key)
            _file(root).unlink(missing_ok=True)
            return "keychain"
        except Exception as exc:  # noqa: BLE001 - a locked or refusing keychain: file
            logger.info("cloud_keychain_refused %s", exc)
    path = _file(root)
    atomic_write(path, key.encode())  # mkstemp: created readable by this user only
    path.chmod(0o600)
    if os.name == "nt":
        private_acl(path)
    return "file"


def stored(root: Path) -> tuple[str, str]:
    """(key, "keychain" | "file") from storage only, or ("", "")."""
    chain = keychain()
    if chain is not None:
        try:
            key = chain.get_password(SERVICE, _account(root))
        except Exception as exc:  # noqa: BLE001 - unreadable keychain: try the file
            logger.info("cloud_keychain_unreadable %s", exc)
            key = None
        if key:
            return key, "keychain"
    path = _file(root)
    try:
        key = path.read_text(encoding="utf-8").strip()
    except OSError:
        key = ""
    return (key, "file") if key else ("", "")


def load(root: Path) -> tuple[str, str]:
    """(key, source): the environment variable first, then storage."""
    env = os.environ.get(ENV, "").strip()
    if env:
        return env, "environment"
    return stored(root)


def delete(root: Path) -> None:
    chain = keychain()
    if chain is not None:
        try:
            chain.delete_password(SERVICE, _account(root))
        except Exception:  # noqa: BLE001 - nothing stored there
            pass
    _file(root).unlink(missing_ok=True)


def masked(key: str) -> str:
    """What the dashboard shows: the last four characters only."""
    key = (key or "").strip()
    return f"…{key[-4:]}" if len(key) >= 8 else ("…" if key else "")
