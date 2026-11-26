"""Prefixed, time-ordered ids: ``ses_``, ``msg_``, ``prt_``, ``evt_``, ``job_``, ``per_``, ...

26 characters after the prefix: 10 of millisecond time and a per-process counter in base62, then
random base62. Ids of one kind sort by creation time, so listing newest-first needs no extra column.
"""

from __future__ import annotations

import secrets
import threading
import time

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
PREFIXES = {"session": "ses", "message": "msg", "part": "prt", "event": "evt", "job": "job",
            "permission": "per", "inbox": "inb", "run": "run", "question": "qst", "credential": "crd",
            "proposal": "prp"}

_lock = threading.Lock()
_last = 0
_counter = 0


def _b62(n: int, width: int) -> str:
    out = []
    for _ in range(width):
        n, r = divmod(n, 62)
        out.append(ALPHABET[r])
    return "".join(reversed(out))


def new(kind: str) -> str:
    """A new id for ``kind`` (a key of PREFIXES, or a raw prefix)."""
    global _last, _counter
    prefix = PREFIXES.get(kind, kind)
    with _lock:
        now = int(time.time() * 1000)
        if now == _last:
            _counter += 1
        else:
            _last, _counter = now, 0
        stamp = now * 4096 + (_counter % 4096)
    rand = "".join(secrets.choice(ALPHABET) for _ in range(14))
    return f"{prefix}_{_b62(stamp, 12)}{rand}"


def kind_of(ident: str) -> str | None:
    prefix = ident.split("_", 1)[0]
    return next((k for k, p in PREFIXES.items() if p == prefix), None)
