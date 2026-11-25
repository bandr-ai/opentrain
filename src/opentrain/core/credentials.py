"""The user's stored credentials: ``~/.local/share/opentrain/auth.json`` (mode 0600).

``/connect`` writes here; providers and trainer backends read here before the environment.
One entry per integration id (``anthropic``, ``openrouter``, ``tinker``, ...).
"""

from __future__ import annotations

import json
import os
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


def _save(data: dict[str, Any]) -> None:
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, p)


def get(integration: str) -> dict[str, Any] | None:
    return _load().get(integration)


def set_key(integration: str, key: str, **extra: Any) -> None:
    data = _load()
    data[integration] = {"type": "key", "key": key, "time": time.time(), **extra}
    _save(data)


def remove(integration: str) -> bool:
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
