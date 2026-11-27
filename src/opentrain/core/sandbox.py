"""An OS-level wall around every command an agent or a candidate model runs.

Tool lists and path checks are not a wall against a process with a shell: ``python -c`` can read
any file the user can. So every shell the harness starts — the engine's ``shell`` tool, the
teacher sampling a task, a candidate model being scored — runs inside a sandbox that cannot read
or write:

- any project's held-out tasks (``.opentrain/heldout``) and evaluator scratch space,
- any project's event log (``.opentrain/opentrain.db``),
- the user's stored OpenTrain credentials.

macOS uses ``sandbox-exec`` (Seatbelt) with a regex profile that covers every project at once.
Linux uses ``bwrap`` when installed, hiding the sealed folders of known projects under empty
tmpfs mounts. Where neither exists the command runs unsandboxed and ``status`` says so; the
path checks in the tools still apply.

The evaluator itself (OpenTrain's own code, in its own process) is never sandboxed: it is the
one reader of held-out tasks.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

_KNOWN: set[str] = set()  # state dirs of projects this process has opened (for bwrap)


def register(state: Path) -> None:
    _KNOWN.add(str(Path(state).resolve()))


def disabled() -> bool:
    return os.environ.get("OPENTRAIN_SANDBOX", "").lower() in ("0", "off", "false", "none")


def kind() -> str:
    if disabled():
        return "off"
    if sys.platform == "darwin" and shutil.which("sandbox-exec"):
        return "seatbelt"
    if sys.platform.startswith("linux") and shutil.which("bwrap"):
        return "bwrap"
    return "none"


def _auth_file() -> str:
    from . import credentials

    return str(credentials.path().resolve())


def seatbelt_profile() -> str:
    auth = _auth_file().replace("\\", "\\\\").replace('"', '\\"')
    sealed = r'"/\.opentrain/(heldout|evaluator)(/|$)"'
    db = r'"/\.opentrain/opentrain\.db(-wal|-shm)?$"'
    return ("(version 1)\n(allow default)\n"
            f"(deny file-read* file-write* (regex #{sealed}))\n"
            f"(deny file-read* file-write* (regex #{db}))\n"
            f'(deny file-read* file-write* (literal "{auth}"))\n')


def wrap(argv: list[str]) -> list[str]:
    """``argv`` run inside the sandbox (unchanged when there is none)."""
    k = kind()
    if k == "seatbelt":
        return ["sandbox-exec", "-p", seatbelt_profile(), *argv]
    if k == "bwrap":
        out = ["bwrap", "--dev-bind", "/", "/", "--die-with-parent"]
        for state in sorted(_KNOWN):
            for sub in ("heldout", "evaluator"):
                p = Path(state) / sub
                if p.exists():
                    out += ["--tmpfs", str(p)]
            for name in ("opentrain.db", "opentrain.db-wal", "opentrain.db-shm"):
                p = Path(state) / name
                if p.exists():
                    out += ["--ro-bind", "/dev/null", str(p)]
        auth = Path(_auth_file())
        if auth.exists():
            out += ["--ro-bind", "/dev/null", str(auth)]
        return out + argv
    return argv
