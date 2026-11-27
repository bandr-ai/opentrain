"""The user's stored credentials: ``~/.local/share/opentrain/auth.json`` (mode 0600).

``/connect`` writes here; providers and trainer backends read here before the environment.
One entry per integration id (``anthropic``, ``openrouter``, ``tinker``, ...).
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any


def path() -> Path:
    from ..server.service import data_home

    return Path(os.environ.get("OPENTRAIN_AUTH_FILE") or data_home() / "auth.json")


def _load() -> dict[str, Any]:
    try:
        return json.loads(path().read_text())
    except (OSError, ValueError):
        return {}


_LOCK = threading.RLock()


@contextlib.contextmanager
def _locked():
    """One writer at a time: threads in this process, and other processes (an flock on a side file)."""
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, open(p.with_name(p.name + ".lock"), "a") as fh:
        try:
            import fcntl

            fcntl.flock(fh, fcntl.LOCK_EX)
        except (ImportError, OSError):  # no flock here: the thread lock still holds
            pass
        yield


def _save(data: dict[str, Any]) -> None:
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, p)


def get_or_create(integration: str, make) -> str:
    """The stored key for ``integration``, or a new one from ``make()``, saved once even when many
    threads or processes ask at the same moment."""
    with _locked():
        have = (_load().get(integration) or {}).get("key")
        if have:
            return have
        key = make()
        data = _load()
        data[integration] = {"type": "key", "key": key, "time": time.time()}
        _save(data)
        return key


def get(integration: str) -> dict[str, Any] | None:
    return _load().get(integration)


def set_key(integration: str, key: str, **extra: Any) -> None:
    with _locked():
        data = _load()
        data[integration] = {"type": "key", "key": key, "time": time.time(), **extra}
        _save(data)


def remove(integration: str) -> bool:
    with _locked():
        data = _load()
        if integration not in data:
            return False
        del data[integration]
        _save(data)
        return True


def listing() -> dict[str, dict[str, Any]]:
    """Every stored credential without its secret."""
    return {k: {"type": v.get("type"), "time": v.get("time"), "hint": (v.get("key") or "")[-4:]}
            for k, v in _load().items()}


def env_for(integration: str, var: str) -> str | None:
    """The key for ``integration``, from the store or ``$var``."""
    stored = get(integration)
    return (stored or {}).get("key") or os.environ.get(var)
