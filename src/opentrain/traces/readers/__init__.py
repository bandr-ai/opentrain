"""Every coding agent's local session logs, read into :class:`~opentrain.traces.Trace`.

``SOURCES`` lists one :class:`Source` per agent. ``scan()`` says what exists without parsing it;
``read_all()`` parses it. A reader never stops the scan: a log that cannot be read is reported
as ``(location, reason)``.

Locations are file paths, or ``<db path>#<session id>`` for agents that keep many sessions in one
SQLite file (opencode, Cursor).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from .. import Trace
from . import aider, claude_code, cline, codex, copilot, cursor, gemini_cli, opencode, pi
from ._util import Source, Unreadable

SOURCES: list[Source] = [
    claude_code.SOURCE,
    codex.SOURCE,
    opencode.SOURCE,
    gemini_cli.SOURCE,
    cursor.SOURCE,
    cline.SOURCE,
    aider.SOURCE,
    pi.SOURCE,
    copilot.SOURCE,
]


def _select(names: list[str] | None) -> list[Source]:
    if names is None:
        return list(SOURCES)
    known = {s.name: s for s in SOURCES}
    missing = [n for n in names if n not in known]
    if missing:
        raise KeyError(f"unknown trace source(s): {', '.join(missing)}; known: {', '.join(known)}")
    return [known[n] for n in names]


def _safe(fn, default):
    try:
        return fn(), None
    except Exception as exc:  # noqa: BLE001 - discovery must never break the scan
        return default, f"{type(exc).__name__}: {exc}"


def scan(sources: list[str] | None = None) -> list[dict]:
    """Per source: ``name``, ``label``, ``roots`` (that exist), ``count`` of logs, and ``error``
    when discovery itself failed. Nothing is parsed."""
    out = []
    for s in _select(sources):
        roots, err1 = _safe(s.roots, [])
        locs, err2 = _safe(s.discover, [])
        out.append(
            {
                "name": s.name,
                "label": s.label,
                "roots": [str(r) for r in roots],
                "count": len(locs),
                "error": err1 or err2,
            }
        )
    return out


def _file_of(loc: str) -> str:
    return loc if os.path.exists(loc) else loc.rpartition("#")[0] or loc


def read_one(source: Source, loc: str) -> tuple[Trace | None, str | None]:
    """(trace, None) or (None, reason)."""
    try:
        t = source.read(loc)
    except Unreadable as exc:
        return None, str(exc)
    except OSError as exc:
        return None, f"cannot open: {exc.strerror or exc}"
    except Exception as exc:  # noqa: BLE001 - one bad log must not stop the rest
        return None, f"{type(exc).__name__}: {exc}"
    if t is None:
        return None, "no conversation in log"
    return t, None


def read_all(
    source_names: list[str] | None = None, since: datetime | None = None
) -> tuple[list[Trace], list[tuple[str, str]]]:
    """Every readable trace from the chosen sources (all by default), plus ``(location, reason)``
    for each log that could not be read. ``since`` keeps traces active at or after that time
    (files last modified before it are skipped unread)."""
    if since is not None and since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)
    traces: list[Trace] = []
    bad: list[tuple[str, str]] = []
    for s in _select(source_names):
        locs, err = _safe(s.discover, [])
        if err:
            bad.append((s.name, f"discovery failed: {err}"))
        for loc in locs:
            if since is not None and "#" not in loc:
                try:
                    if os.path.getmtime(_file_of(loc)) < since.timestamp():
                        continue
                except OSError:
                    pass
            t, reason = read_one(s, loc)
            if t is None:
                bad.append((loc, reason or "unreadable"))
                continue
            last = t.ended_at or t.started_at
            if since is not None and last is not None and last < since:
                continue
            traces.append(t)
    return traces, bad


__all__ = ["SOURCES", "Source", "Unreadable", "read_all", "read_one", "scan"]
