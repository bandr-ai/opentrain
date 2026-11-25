"""The engine end to end on real repos and real session logs; only the models are stand-ins.

The stand-in runner edits real files through the direct runtime's tool format, and the real
verifier judges every attempt, so mining, the split, sampling, the SFT set, the gate and the
registry all run for real.
"""

from __future__ import annotations

import io
import json
import sys
import time
from datetime import timedelta
from pathlib import Path

import pytest
from conftest import T0, git

from opentrain import engine, trainfile
from opentrain.mcp import TOOLS_BY_ROLE, Server
from opentrain.plugs import Attempt
from opentrain.registry import Registry
from opentrain.state import ensure
from opentrain.tasks import HeldoutStore, Store

FIXES = [("add", "a + b"), ("sub", "a - b"), ("mul", "a * b"), ("div", "a / b"), ("mod", "a % b")]


@pytest.fixture
def project(tmp_path, writer):
    """A repo where five sessions each added a function with its test, a day apart."""
    repo = tmp_path / "repos" / "calc"
    repo.mkdir(parents=True)
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "user.name", "t")
    (repo / "calc.py").write_text("")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_base.py").write_text("def test_ok():\n    assert True\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base", when=T0 - timedelta(hours=1))
    logs = tmp_path / "logs" / "proj"
    for i, (name, expr) in enumerate(FIXES):
        day = i * 86400
        src = (repo / "calc.py").read_text() + f"def {name}(a, b):\n    return {expr}\n"
        test = f"from calc import {name}\n\ndef test_{name}():\n    assert {name}(6, 3) == {eval(expr, {'a': 6, 'b': 3})}\n"
        w = (
            writer(str(repo), session_id=f"sess{i}")
            .prompt(day, f"add a {name} function to calc.py")
            .call(day + 5, "e", name="Edit", file_path=f"{repo}/calc.py", old_string="", new_string=src)
            .result(day + 6, "e")
            .call(day + 7, "t", command=f"{sys.executable} -m pytest -q -p no:cacheprovider")
            .result(day + 12, "t", "2 passed")
            .say(day + 14, "done")
        )
        w.write(logs / f"sess{i}.jsonl")
        (repo / "calc.py").write_text(src)
        (repo / "tests" / f"test_{name}.py").write_text(test)
        git(repo, "add", ".")
        git(repo, "commit", "-qm", f"add {name}", when=T0 + timedelta(seconds=day + 600))
    return repo, tmp_path / "logs"


class Solver:
    """A stand-in model: reads the instruction, writes the function, in the direct runtime's format."""

    def __init__(self, name: str, delay: float, extra_steps: int = 0):
        self.name, self.delay, self.extra = name, delay, extra_steps

    def attempt(self, instruction, workspace: Path, *, task_id, timeout):
        start = time.monotonic()
        time.sleep(self.delay)
        name = instruction.split("add a ")[1].split()[0]
        expr = dict(FIXES)[name]
        calc = workspace / "calc.py"
        old = calc.read_text()
        calc.write_text(old + f"def {name}(a, b):\n    return {expr}\n")
        msgs = [{"role": "system", "content": "x"}, {"role": "user", "content": instruction}]
        for k in range(self.extra):
            msgs += [{"role": "assistant", "content": "", "tool_calls": [{"id": f"r{k}", "type": "function",
                      "function": {"name": "read_file", "arguments": json.dumps({"path": "calc.py"})}}]},
                     {"role": "tool", "tool_call_id": f"r{k}", "content": old}]
        msgs += [{"role": "assistant", "content": "", "tool_calls": [{"id": "w", "type": "function",
                  "function": {"name": "write_file", "arguments": json.dumps({"path": "calc.py",
                                                                              "content": calc.read_text()})}}]},
                 {"role": "tool", "tool_call_id": "w", "content": "ok"},
                 {"role": "assistant", "content": "done"}]
        return Attempt(task_id=task_id, model=self.name, passed=None, seconds=time.monotonic() - start,
                       tool_calls=1 + self.extra, messages=msgs)


def test_train_end_to_end_promotes_a_faster_version(project, tmp_path, monkeypatch):
    repo, logs = project
    monkeypatch.setenv("OPENTRAIN_FAKE_BACKEND", "1")
    state = ensure(tmp_path / "work")
    tf = trainfile.parse("goal  speed\nbudget  $5 a week\ntrainer  fake\nbase model  tiny\nheldout  40%\nruntime  direct")
    slow, fast = Solver("frontier", 0.4, extra_steps=2), Solver("specialist", 0.0)
    monkeypatch.setattr(engine, "_baseline_runner", lambda tf: slow)
    monkeypatch.setattr(engine, "specialist_runner", lambda v, max_steps=60: fast)
    monkeypatch.setattr("opentrain.certify.lint", lambda state, timeout=900, only_heldout=False: {"broken": []})

    lines: list[str] = []
    res = engine.train(state, tf, sessions_root=logs, say=lines.append, ask=lambda q: True,
                       teacher=slow, n_samples=2, workers=1, poll=0.01)
    out = "\n".join(lines)
    assert res.promoted, out
    assert len(Store(state).ids()) == 3 and len(HeldoutStore(state).ids()) == 2
    assert "passed the gate" in out and "Median task" in out
    reg = Registry(state)
    assert reg.current().version == "v1" and reg.current().scores["pass_rate"] == 1.0

    rows = [json.loads(x) for x in (state / "data" / "train.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    held = set(HeldoutStore(state).ids())
    manifest = json.loads((state / "data" / "manifest.json").read_text())
    assert not held & {t["task_id"] for t in manifest["tasks"]}

    run = next(iter(__import__("opentrain.record", fromlist=["RunRecord"]).RunRecord(state).runs()))
    kinds = [e["kind"] for e in run.events()]
    assert run.verify() and kinds[0] == "opened" and kinds[-1] == "closed"
    assert {"data", "recipe", "spend", "job", "checkpoint", "eval", "gate"} <= set(kinds)


def test_train_stops_when_declined_and_records_why(project, tmp_path, monkeypatch):
    _, logs = project
    monkeypatch.setenv("OPENTRAIN_FAKE_BACKEND", "1")
    state = ensure(tmp_path / "work")
    tf = trainfile.parse("trainer  fake\nheldout  40%\nruntime  direct")
    s = Solver("x", 0.0)
    monkeypatch.setattr(engine, "_baseline_runner", lambda tf: s)
    res = engine.train(state, tf, sessions_root=logs, say=lambda _: None, ask=lambda q: False, teacher=s, n_samples=1)
    assert not res.promoted and res.version is None and "declined" in res.reasons[0]
    assert Registry(state).current() is None


def test_unattended_upload_needs_prior_approval(project, tmp_path, monkeypatch):
    _, logs = project
    monkeypatch.setenv("OPENTRAIN_FAKE_BACKEND", "1")
    state = ensure(tmp_path / "work")
    tf = trainfile.parse("trainer  fake\nheldout  40%\nruntime  direct\ndata  sessions, local until I approve")
    s = Solver("x", 0.0)
    monkeypatch.setattr(engine, "_baseline_runner", lambda tf: s)
    res = engine.train(state, tf, sessions_root=logs, say=lambda _: None, ask=None, teacher=s, n_samples=1)
    assert "until I approve" in res.reasons[0]


def test_provider_parsing():
    assert engine.parse_provider("https://openrouter.ai/api/v1 key OPENROUTER_API_KEY") == \
        ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY")
    assert engine.parse_provider("http://localhost:11434/v1/") == ("http://localhost:11434/v1", None)
    with pytest.raises(ValueError):
        engine.parse_provider("sk-live-123")


def _rpc(server: Server, *msgs) -> list[dict]:
    out = io.StringIO()
    server.serve(io.StringIO("\n".join(json.dumps(m) for m in msgs) + "\n"), out)
    return [json.loads(x) for x in out.getvalue().splitlines()]


def test_mcp_roles_and_refusals(tmp_path):
    state = ensure(tmp_path)
    tf = trainfile.parse("budget  $3 a week\ntrainer  fake")
    trainer = Server(state, "trainer", None, tf)
    replies = _rpc(
        trainer,
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "budget_status", "arguments": {}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "heldout_lint", "arguments": {}}},
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "train_launch", "arguments": {}}},
        {"jsonrpc": "2.0", "id": 6, "method": "nope"},
    )
    assert [r["id"] for r in replies] == [1, 2, 3, 4, 5, 6]
    names = {t["name"] for t in replies[1]["result"]["tools"]}
    assert names == set(TOOLS_BY_ROLE["trainer"]) and "heldout_lint" not in names
    assert json.loads(replies[2]["result"]["content"][0]["text"])["remaining_usd"] == 3
    assert replies[3]["result"]["isError"] and "not available to the trainer" in replies[3]["result"]["content"][0]["text"]
    assert replies[4]["result"]["isError"] and "no training set" in replies[4]["result"]["content"][0]["text"]
    assert replies[5]["error"]["code"] == -32601
    evaluator = Server(state, "evaluator", trainer.run.run_id, tf)
    assert "train_launch" not in evaluator.handlers and "heldout_lint" in evaluator.handlers


def test_cli_surface_parses(tmp_path, capsys):
    from opentrain.cli import main

    assert main(["debug", "init", "--state", str(tmp_path)]) == 0
    assert (tmp_path / "Trainfile").exists()
    assert main(["debug", "status", "--state", str(tmp_path)]) == 0
    assert "goal/speed" in capsys.readouterr().out
    assert main(["debug", "log", "--state", str(tmp_path)]) == 0
    assert main(["debug", "skill", "ls", "--state", str(tmp_path)]) == 0
    assert "cl/mine" in capsys.readouterr().out
