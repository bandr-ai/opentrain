from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

T0 = datetime(2026, 11, 1, 10, 0, 0, tzinfo=timezone.utc)


def at(seconds: float) -> str:
    return (T0 + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


class SessionWriter:
    """Builds a Claude Code session log one event at a time, with exact timestamps."""

    def __init__(self, cwd: str, branch: str = "main", session_id: str = "s1"):
        self.cwd, self.branch, self.sid = cwd, branch, session_id
        self.records: list[dict] = []
        self._n = 0

    def _base(self, t: float, kind: str, **extra) -> dict:
        self._n += 1
        return {"type": kind, "timestamp": at(t), "cwd": self.cwd, "gitBranch": self.branch,
                "sessionId": self.sid, "uuid": f"u{self._n}", **extra}

    def prompt(self, t: float, text: str, **extra):
        self.records.append(self._base(t, "user", message={"role": "user", "content": text}, **extra))
        return self

    def say(self, t: float, text: str, model: str = "claude-opus-5-5", **extra):
        msg = {"role": "assistant", "model": model, "content": [{"type": "text", "text": text}]}
        self.records.append(self._base(t, "assistant", message=msg, **extra))
        return self

    def call(self, t: float, call_id: str, name: str = "Bash", **inp):
        msg = {"role": "assistant", "model": "claude-opus-5-5",
               "content": [{"type": "tool_use", "id": call_id, "name": name, "input": inp}]}
        self.records.append(self._base(t, "assistant", message=msg))
        return self

    def result(self, t: float, call_id: str, text: str = "ok", is_error: bool = False):
        block = {"type": "tool_result", "tool_use_id": call_id, "content": text, "is_error": is_error}
        self.records.append(self._base(t, "user", message={"role": "user", "content": [block]}))
        return self

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r) for r in self.records) + "\n")
        return path


@pytest.fixture
def writer():
    return SessionWriter


def git(repo: Path, *args: str, when: datetime | None = None) -> str:
    env = None
    if when is not None:
        import os

        env = {**os.environ, "GIT_AUTHOR_DATE": when.isoformat(), "GIT_COMMITTER_DATE": when.isoformat()}
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True, env=env)
    return out.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repo with tests, committed an hour before T0."""
    r = tmp_path / "repos" / "widget"
    r.mkdir(parents=True)
    git(r, "init", "-q", "-b", "main")
    git(r, "config", "user.email", "t@example.com")
    git(r, "config", "user.name", "t")
    (r / "app.py").write_text("def f():\n    return 1\n")
    (r / "tests").mkdir()
    (r / "tests" / "test_app.py").write_text("from app import f\n\ndef test_f():\n    assert f() == 1\n")
    git(r, "add", ".")
    git(r, "commit", "-q", "-m", "base", when=T0 - timedelta(hours=1))
    return r


@pytest.fixture
def live(tmp_path, monkeypatch):
    """A running server whose default project is ``tmp_path / 'proj'``, with its own config, data
    and state homes so nothing of the real user's leaks in or out."""
    import sys

    for var in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME"):
        monkeypatch.setenv(var, str(tmp_path / var.lower()))
    for var in ("ANTHROPIC_API_KEY", "TINKER_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    sys.path.insert(0, str(Path(__file__).parent))
    from harness import LiveServer

    proj = tmp_path / "proj"
    proj.mkdir()
    srv = LiveServer(proj)
    srv.proj = proj
    yield srv
    srv.stop()
