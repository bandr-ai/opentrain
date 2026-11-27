"""Harness configuration: ``opentrain.jsonc``, merged global then project (opencode's order).

Locations, lowest to highest precedence:

1. ``~/.config/opentrain/opentrain.json[c]``
2. ``<project>/opentrain.json[c]`` and ``<project>/.opentrain/opentrain.json[c]``
3. ``$OPENTRAIN_CONFIG`` (a file) and ``$OPENTRAIN_CONFIG_CONTENT`` (inline JSON)

Keys: ``model`` (``provider/model``), ``default_agent``, ``agents``, ``permissions``, ``providers``,
``mcp``, ``skills`` (extra dirs), ``instructions`` (extra files), ``steps``. The Trainfile is
separate: it is policy the server enforces; this file is how the harness is set up.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .store import merge

NAMES = ("opentrain.jsonc", "opentrain.json")


def strip_jsonc(text: str) -> str:
    """Remove // and /* */ comments and trailing commas, leaving strings intact."""
    out, i, n, in_str = [], 0, len(text), False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_str = False
        elif c == '"':
            in_str = True
            out.append(c)
        elif text.startswith("//", i):
            while i < n and text[i] != "\n":
                i += 1
            continue
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end < 0 else end + 2
            continue
        else:
            out.append(c)
        i += 1
    return re.sub(r",(\s*[}\]])", r"\1", "".join(out))


def read(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(strip_jsonc(path.read_text()))
    except FileNotFoundError:
        return {}
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from None
    if not isinstance(data, dict):
        raise ValueError(f"{path}: must be a JSON object")
    return _expand(data)


def _expand(obj: Any) -> Any:
    """``{env:NAME}`` and ``{file:path}`` inside strings."""
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    if isinstance(obj, str):
        obj = re.sub(r"\{env:([A-Za-z_][A-Za-z0-9_]*)\}", lambda m: os.environ.get(m.group(1), ""), obj)
        obj = re.sub(r"\{file:([^}]+)\}",
                     lambda m: Path(m.group(1)).expanduser().read_text().strip()
                     if Path(m.group(1)).expanduser().exists() else "", obj)
    return obj


def global_dir() -> Path:
    from ..server.service import config_home

    return config_home()


def load_global() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in NAMES:
        out = merge(out, read(global_dir() / name))
    return out


def load(directory: Path | None = None) -> dict[str, Any]:
    out = load_global()
    if directory:
        for base in (directory, directory / ".opentrain"):
            for name in NAMES:
                out = merge(out, read(base / name))
    if os.environ.get("OPENTRAIN_CONFIG"):
        out = merge(out, read(Path(os.environ["OPENTRAIN_CONFIG"])))
    if os.environ.get("OPENTRAIN_CONFIG_CONTENT"):
        out = merge(out, _expand(json.loads(strip_jsonc(os.environ["OPENTRAIN_CONFIG_CONTENT"]))))
    return out


def save_global(patch: dict[str, Any]) -> Path:
    """Merge ``patch`` into the global config (used by /models and /connect choices)."""
    d = global_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / "opentrain.json"
    cur = read(path) if path.exists() else {}
    path.write_text(json.dumps(merge(cur, patch), indent=2) + "\n")
    return path
