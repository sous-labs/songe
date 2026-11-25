"""Redaction before anything leaves the machine. Always on, never optional.

Secrets that look like keys or tokens are replaced by [REDACTED], the value
of a KEY=VALUE line whose name says it is secret is replaced (the name
stays, so the teacher still sees what the line was), and the user's home
folder becomes ~. Redaction errs on the side of removing too much: a lost
token costs the teacher a little context, a leaked one costs the user.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

MARK = "[REDACTED]"

# Whole tokens recognisable by their shape. Order matters only for overlap:
# PEM blocks first (they contain base64 that other patterns could split).
TOKENS = [
    # PEM private keys (RSA, EC, OPENSSH, PGP, ...), whole block.
    re.compile(
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----.*?-----END [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----",
        re.S,
    ),
    # A PEM header whose end was cut off: everything after it.
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----.*\Z", re.S),
    # OpenAI, Anthropic, OpenRouter and most "sk-" style keys.
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    # Stripe-style live/test keys.
    re.compile(r"\b(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{16,}"),
    # GitHub tokens (classic and fine-grained).
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    # GitLab personal access tokens.
    re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}"),
    # AWS access key ids.
    re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|ANPA)[0-9A-Z]{16}\b"),
    # Slack tokens and webhooks.
    re.compile(r"\bxox[abposr]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"https://hooks\.slack\.com/services/[A-Za-z0-9/_\-]+"),
    # Google API keys.
    re.compile(r"\bAIza[0-9A-Za-z_\-]{35}"),
    # Hugging Face tokens.
    re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    # JSON Web Tokens: three base64url parts, the first two JSON objects.
    re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),
    # Credentials inside URLs: scheme://user:password@host
    re.compile(r"(?<=://)[^/\s:@]+:[^/\s@]+(?=@)"),
]

# "Authorization: Bearer abc1..." anywhere: keep the word. A credential has
# a digit; "Bearer authentication" in prose is left alone.
BEARER = re.compile(
    r"(?i)\b(bearer|basic|authorization:\s*token)(\s+)(?=[A-Za-z._~+/\-]*[0-9])([A-Za-z0-9._~+/\-]{12,}=*)"
)

# A name that says its value is secret, followed by = or : and a value.
SECRET_NAME = (
    r"[A-Za-z0-9_.\-]*(?:api[_\-]?key|secret|token|passw(?:or)?d|passwd|pwd|"
    r"credential|private[_\-]?key|access[_\-]?key|auth|dsn|session[_\-]?id|cookie)"
    r"[A-Za-z0-9_.\-]*"
)
ASSIGNMENT = re.compile(
    r"(?i)(?P<name>[\"']?\b" + SECRET_NAME + r"[\"']?)"
    r"(?P<sep>\s*(?:=|:)\s*)"
    r"(?P<quote>[\"'`]?)"
    r"(?P<value>[^\s\"'`,;#)}\]]+)"
)

# A value that reads as code, not a secret: a dotted identifier with no
# digits (self.token, os.environ, None, true), a number, or a call.
CODE_LIKE = re.compile(r"(?:[A-Za-z_][A-Za-z_]*)(?:\.[A-Za-z_][A-Za-z_]*)*\(?")
NUMBER = re.compile(r"[0-9]+(?:\.[0-9]+)?")


def _secret_value(value: str) -> bool:
    if value == MARK or len(value) < 6:
        return False
    if NUMBER.fullmatch(value) or CODE_LIKE.fullmatch(value):
        return False
    # Environment lookups and template placeholders name a secret, not hold it.
    if value.startswith(("$", "{{", "os.", "process.env", "<")):
        return False
    return True


def _assignment(match: re.Match) -> str:
    value = match.group("value")
    if not _secret_value(value):
        return match.group(0)
    return f"{match.group('name')}{match.group('sep')}{match.group('quote')}{MARK}"


def home_paths(home: str | None = None) -> list[str]:
    """The spellings of the home folder to hide, longest first."""
    home = home if home is not None else str(Path.home())
    if not home or home in {"/", "\\"}:
        return []
    found = {home.rstrip("/\\")}
    found.add(home.replace("\\", "/").rstrip("/"))
    found.add(home.replace("/", "\\").rstrip("\\"))
    try:
        found.add(os.path.realpath(home).rstrip("/\\"))
    except OSError:
        pass
    return sorted((h for h in found if len(h) > 1), key=len, reverse=True)


def redact(text: str, home: str | None = None) -> str:
    """The text with secrets replaced by [REDACTED] and the home folder by ~."""
    if not isinstance(text, str) or not text:
        return text
    for pattern in TOKENS:
        text = pattern.sub(MARK, text)
    text = BEARER.sub(lambda m: f"{m.group(1)}{m.group(2)}{MARK}", text)
    text = ASSIGNMENT.sub(_assignment, text)
    for path in home_paths(home):
        # Only whole path components: /Users/ann, not /Users/anna.
        text = re.sub(re.escape(path) + r"(?=[/\\\s\"'`:;,)\]}]|$)", "~", text)
    return text


def redact_value(value, home: str | None = None):
    """Every string inside messages, tool calls and their arguments."""
    if isinstance(value, str):
        return redact(value, home)
    if isinstance(value, list):
        return [redact_value(v, home) for v in value]
    if isinstance(value, dict):
        return {k: redact_value(v, home) for k, v in value.items()}
    return value
