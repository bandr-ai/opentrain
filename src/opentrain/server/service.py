"""Finding, starting and registering the one OpenTrain server per user.

The running server writes ``$XDG_STATE_HOME/opentrain/service.json`` (mode 0600) with its url, pid,
version and password. Clients read it, probe ``/api/health`` with the password, and start a server
in the background when there is none or its version differs. This is opencode's service model.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .. import __version__

DEFAULT_PORT = 7477  # "OT" on a phone pad, near opencode's 0xC0DE neighbourhood but not on it


def state_home() -> Path:
    base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    return base / "opentrain"


def data_home() -> Path:
    base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "opentrain"


def config_home() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "opentrain"


def registration_path() -> Path:
    return Path(os.environ.get("OPENTRAIN_SERVICE_FILE") or state_home() / "service.json")


@dataclass
class Registration:
    url: str
    pid: int
    version: str
    password: str

    def auth_header(self) -> str:
        return "Basic " + base64.b64encode(f"opentrain:{self.password}".encode()).decode()


def new_password() -> str:
    return secrets.token_urlsafe(24)


def write_registration(reg: Registration) -> Path:
    path = registration_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(reg.__dict__, f)
    os.replace(tmp, path)
    return path


def clear_registration(pid: int) -> None:
    reg = read_registration()
    if reg and reg.pid == pid:
        registration_path().unlink(missing_ok=True)


def read_registration() -> Registration | None:
    try:
        d = json.loads(registration_path().read_text())
        return Registration(url=d["url"], pid=int(d["pid"]), version=d["version"], password=d["password"])
    except (OSError, ValueError, KeyError):
        return None


def probe(reg: Registration, timeout: float = 1.5) -> dict | None:
    req = urllib.request.Request(reg.url.rstrip("/") + "/api/health", headers={"Authorization": reg.auth_header()})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, OSError, ValueError):
        return None


def discover() -> Registration | None:
    """The running server, if its registration answers and its version matches ours."""
    reg = read_registration()
    if reg is None:
        return None
    health = probe(reg)
    if not health or health.get("version") != __version__:
        return None
    return reg


def stop(reg: Registration | None = None) -> bool:
    reg = reg or read_registration()
    if reg is None:
        return False
    try:
        os.kill(reg.pid, 15)
    except OSError:
        pass
    registration_path().unlink(missing_ok=True)
    return True


def ensure(timeout: float = 15.0) -> Registration:
    """Return the running server, starting one in the background when needed."""
    reg = discover()
    if reg:
        return reg
    old = read_registration()
    if old and probe(old):  # a server of another version: replace it
        stop(old)
    log = state_home() / "server.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as out:
        subprocess.Popen([sys.executable, "-m", "opentrain.server", "--service"], stdout=out, stderr=out,
                         stdin=subprocess.DEVNULL, start_new_session=True, cwd=str(Path.home()))
    deadline = time.time() + timeout
    while time.time() < deadline:
        reg = discover()
        if reg:
            return reg
        time.sleep(0.15)
    raise RuntimeError(f"the OpenTrain server did not start within {timeout:.0f}s; see {log}")
