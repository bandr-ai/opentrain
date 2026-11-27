"""Remove credentials from trace text before it leaves the machine.

:func:`scrub` replaces well-known token formats (OpenAI/Anthropic keys, GitHub, Hugging Face, AWS,
Slack, Google, Stripe tokens, JWTs, PEM private keys, credentials inside URLs) and the values of
assignments whose name says key/token/secret/password with ``[REDACTED]``. It is pattern-based:
it catches the common shapes, not every secret a person could paste.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from . import ToolUse, Trace

REDACTED = "[REDACTED]"

_TOKENS = [
    re.compile(
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----.*?(?:-----END [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----|\Z)",
        re.S,
    ),
    re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"\bsk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA)[A-Z0-9]{16}\b"),
    re.compile(r"\bxox[abposr]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{35}"),
    re.compile(r"\b(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{16,}"),
    re.compile(r"\b(?:npm_[A-Za-z0-9]{30,}|pypi-[A-Za-z0-9_\-]{40,})"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),
]

_NAME = (
    r"[A-Za-z0-9_.\-]*(?:api[_\-]?key|apikey|secret(?:[_\-]?(?:access[_\-]?)?key)?|token|passw(?:or)?d|passwd|pwd"
    r"|access[_\-]?key(?:[_\-]?id)?|private[_\-]?key|credentials?)(?![A-Za-z0-9_])"
)
# "name": "value"  /  'name': 'value'
_JSON_PAIR = re.compile(rf"""(["']{_NAME}["']\s*:\s*)(["'])((?:\\.|(?!\2).)+?)\2""", re.I)
# NAME=value, NAME: value, --name value (value up to whitespace / quote / end)
_ASSIGN = re.compile(rf"""(\b{_NAME}\s*(?:=|:(?!//))\s*)(["']?)([^\s"'`,;&|)}}\]]{{6,}})""", re.I)
_FLAG = re.compile(rf"""(--{_NAME}[ =])(["']?)([^\s"']{{6,}})""", re.I)
_URL_CRED = re.compile(r"(\b[a-z][a-z0-9+.\-]*://[^/\s:@]+:)([^@\s/]{3,})(@)", re.I)
_BEARER = re.compile(r"(\b(?:Bearer|Basic|token)\s+)([A-Za-z0-9_\-.=+/]{16,})")

_NOT_SECRET = re.compile(
    r"^(?:\[REDACTED\]|\$\{?[A-Za-z_]\w*\}?|<[^>]*>|\*+|x+|null|none|true|false|undefined|os\.environ.*|process\.env.*)$",
    re.I,
)


def _keep_name(m: re.Match) -> str:
    value = m.group(3)
    if _NOT_SECRET.match(value) or REDACTED in value:
        return m.group(0)
    return f"{m.group(1)}{m.group(2)}{REDACTED}" + (m.group(2) if m.re is _JSON_PAIR else "")


def scrub(text: str) -> str:
    if not text:
        return text
    for pat in _TOKENS:
        text = pat.sub(REDACTED, text)
    text = _URL_CRED.sub(lambda m: f"{m.group(1)}{REDACTED}{m.group(3)}", text)
    text = _BEARER.sub(lambda m: m.group(0) if REDACTED in m.group(2) else f"{m.group(1)}{REDACTED}", text)
    text = _JSON_PAIR.sub(_keep_name, text)
    text = _ASSIGN.sub(_keep_name, text)
    text = _FLAG.sub(_keep_name, text)
    return text


def _scrub_value(v: Any) -> Any:
    if isinstance(v, str):
        return scrub(v)
    if isinstance(v, list):
        return [_scrub_value(x) for x in v]
    if isinstance(v, tuple):
        return tuple(_scrub_value(x) for x in v)
    if isinstance(v, dict):
        return {k: _scrub_value(x) for k, x in v.items()}
    return v


def scrub_tool(t: ToolUse) -> ToolUse:
    return replace(t, args=_scrub_value(t.args), output=scrub(t.output) if t.output is not None else None)


def scrub_trace(trace: Trace) -> Trace:
    """A copy of ``trace`` with every text, tool argument, output and extra value scrubbed."""
    steps = [replace(s, text=scrub(s.text), tools=[scrub_tool(t) for t in s.tools]) for s in trace.steps]
    return replace(trace, steps=steps, extra=_scrub_value(trace.extra))


__all__ = ["REDACTED", "scrub", "scrub_tool", "scrub_trace"]
