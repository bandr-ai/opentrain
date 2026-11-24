"""The execution path: runtimes, sampling, SFT data, the evaluator, race. No real model APIs are called."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from conftest import T0, git

from opentrain import workspace
from opentrain.budget import BudgetExceeded, Ledger
from opentrain.data.sft import HeldoutLeak, build_sft
from opentrain.evaluator import EvalCapReached, Evaluator
from opentrain.gate import Score
from opentrain.plugs import AgentRunner, Attempt, Endpoint
from opentrain.race import race
from opentrain.record import RunRecord
from opentrain.runtime import SYSTEM_PROMPT, TOOLS, DirectRunner, HeadlessRunner, headless_command
from opentrain.runtime.direct import ToolError, resolve
from opentrain.runtime.headless import parse_events, transcript
from opentrain.sample import filter_tasks, load_attempts, sample, shortest
from opentrain.tasks import Controls, HeldoutStore, Store, Task, task_toml
from opentrain.trainfile import Budget

# ---------------------------------------------------------------- fixtures


@pytest.fixture
def task(repo: Path) -> Task:
    """'make f return 2': the reference changes app.py and the test; the verifier runs pytest."""
    base = git(repo, "rev-parse", "HEAD")
    (repo / "app.py").write_text("def f():\n    return 2\n")
    (repo / "tests" / "test_app.py").write_text("from app import f\n\ndef test_f():\n    assert f() == 2\n")
    git(repo, "commit", "-qam", "fix", when=T0 + timedelta(minutes=5))
    ref = git(repo, "rev-parse", "HEAD")
    return Task("t1", "make f return 2", str(repo), base, ref, f"{sys.executable} -m pytest -q -p no:cacheprovider",
                ("tests/test_app.py",), created_at=T0.isoformat())


def add_to(store: Store, task: Task, tmp_path: Path) -> None:
    folder = tmp_path / "build" / task.task_id
    folder.mkdir(parents=True)
    (folder / "task.toml").write_text(task_toml(task))
    (folder / "instruction.md").write_text(task.instruction)
    store.add(folder, Controls(reference=True, noop=False))


def call(name: str, **args) -> dict:
    return {"id": f"call_{name}_{len(json.dumps(args))}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


class FakeOpenAI:
    """A scripted OpenAI-compatible /chat/completions server. Each entry is a reply (or an HTTP status)."""

    def __init__(self, script: list):
        self.script, self.requests = list(script), []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append({"path": self.path, "body": body, "auth": self.headers.get("Authorization")})
                step = owner.script.pop(0)
                if isinstance(step, int):
                    self.send_response(step)
                    self.end_headers()
                    self.wfile.write(b'{"error": "busy"}')
                    return
                reply = {"choices": [{"message": {"role": "assistant", "content": step.get("content"),
                                                  "tool_calls": step.get("calls")}}],
                         "usage": {"prompt_tokens": 100, "completion_tokens": 10}}
                data = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


class ScriptedRunner:
    """An AgentRunner that edits the workspace itself, with a chosen number of steps per attempt."""

    def __init__(self, plan: list[tuple[int, bool]], name: str = "scripted"):
        self.plan, self.name, self.lock = list(plan), name, threading.Lock()

    def attempt(self, instruction, workspace_dir, *, task_id, timeout) -> Attempt:
        with self.lock:
            steps, fix = self.plan.pop(0)
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": instruction}]
        for i in range(steps - 1):
            c = call("bash", command=f"echo {i}")
            msgs += [{"role": "assistant", "content": "", "tool_calls": [c]},
                     {"role": "tool", "tool_call_id": c["id"], "content": "exit code 0\n"}]
        if fix:
            (workspace_dir / "app.py").write_text("def f():\n    return 2\n")
        fin = call("finish", summary="done")
        msgs += [{"role": "assistant", "content": "", "tool_calls": [fin]},
                 {"role": "tool", "tool_call_id": fin["id"], "content": "finished"}]
        return Attempt(task_id, self.name, None, seconds=float(steps), tool_calls=steps, tokens_in=5, tokens_out=5,
                       cost_usd=0.1, messages=msgs, patch=workspace.diff(workspace_dir))


# ---------------------------------------------------------------- direct runtime


def test_direct_runner_patch_passes_verification(task, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "sk-test")
    srv = FakeOpenAI([
        503,  # retried
        {"calls": [call("read_file", path="app.py")]},
        {"calls": [call("edit_file", path="app.py", old_string="return 1", new_string="return 2")]},
        {"content": "done", "calls": [call("finish", summary="f returns 2")]},
    ])
    try:
        runner = DirectRunner(Endpoint(srv.url, "fake-model", "FAKE_KEY"), backoff=0.01)
        assert isinstance(runner, AgentRunner)
        with workspace.checkout(task) as ws:
            att = runner.attempt(task.instruction, ws, task_id=task.task_id, timeout=60)
    finally:
        srv.close()
    assert att.error is None
    assert att.passed is None  # the runner never grades itself
    assert att.steps == 3 and att.tool_calls == 3
    assert (att.tokens_in, att.tokens_out) == (300, 30)
    assert "app.py" in att.patch and workspace.verify(task, att.patch).passed
    assert att.messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert att.messages[1] == {"role": "user", "content": task.instruction}
    read_result = att.messages[3]
    assert read_result["role"] == "tool" and "     1\tdef f():" in read_result["content"]
    assert all("seconds" in m for m in att.messages[2:])
    assert 0 < att.model_seconds <= att.seconds
    sent = srv.requests[-1]
    assert sent["path"] == "/v1/chat/completions" and sent["body"]["tools"] == TOOLS
    assert sent["auth"] == "Bearer sk-test"
    assert not any("seconds" in m for m in sent["body"]["messages"])


def test_paths_are_confined_to_the_workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (tmp_path / "secret.txt").write_text("s")
    (ws / "link").symlink_to(tmp_path / "secret.txt")
    for bad in ("../secret.txt", "/etc/passwd", "link", "a/../../secret.txt", ".git/config", ""):
        with pytest.raises(ToolError):
            resolve(ws, bad)
    assert resolve(ws, "sub/new.py") == (ws / "sub" / "new.py").resolve()

    runner = DirectRunner(Endpoint("http://unused", "m"))
    with pytest.raises(ToolError):
        runner.run_tool(ws, "write_file", {"path": "../evil.txt", "content": "x"}, 10)
    assert not (tmp_path / "evil.txt").exists()


def test_direct_runner_reports_escape_to_model_and_stops_at_max_steps(task):
    srv = FakeOpenAI([{"calls": [call("write_file", path="../../evil.txt", content="x")]},
                      {"calls": [call("bash", command="true")]}])
    try:
        with workspace.checkout(task) as ws:
            att = DirectRunner(Endpoint(srv.url, "m"), max_steps=2).attempt("x", ws, task_id="t1", timeout=60)
            assert not (ws.parent.parent / "evil.txt").exists()
    finally:
        srv.close()
    assert att.messages[3]["content"].startswith("error:") and "outside" in att.messages[3]["content"]
    assert "stopped after 2 steps" in att.error and att.patch == ""


def test_direct_runner_gives_up_on_client_errors(task):
    srv = FakeOpenAI([400])
    try:
        with workspace.checkout(task) as ws:
            att = DirectRunner(Endpoint(srv.url, "m"), backoff=0.01).attempt("x", ws, task_id="t1", timeout=60)
    finally:
        srv.close()
    assert "HTTP 400" in att.error and len(srv.requests) == 1


# ---------------------------------------------------------------- sampling and data


def test_sample_verifies_records_and_shortest_picks_fewest_steps(task, tmp_path):
    state = tmp_path / ".opentrain"
    add_to(Store(state), task, tmp_path)
    run = RunRecord(state).open("test", {})
    ledger = Ledger(state, Budget(10.0, 7, "$10 a week"))
    runner = ScriptedRunner([(4, True), (2, True), (1, False)])
    atts = sample([task], runner, 3, state, ledger, run, est_usd_per_attempt=0.5, workers=2)

    assert sorted(a.passed for a in atts) == [False, True, True]  # judged by the verifier, not the runner
    assert len(load_attempts(state)) == 3
    assert sorted(p.name for p in (state / "attempts" / "t1").iterdir()) == ["0.json", "1.json", "2.json"]
    spends = [e for e in run.events() if e["kind"] == "spend"]
    assert len(spends) == 3 and run.verify()
    assert ledger.committed(run.run_id) == pytest.approx(0.3)  # reservations settled with the real cost

    best = shortest(atts)
    assert best["t1"].steps == 2 and best["t1"].passed
    rep = filter_tasks(atts)
    assert rep.mixed == ["t1"] and rep.rates["t1"] == (2, 3)


def test_sample_stops_when_the_budget_refuses(task, tmp_path):
    state = tmp_path / ".opentrain"
    ledger = Ledger(state, Budget(1.0, 7, "$1 a week"))
    run = RunRecord(state).open("test", {})
    with pytest.raises(BudgetExceeded):
        sample([task], ScriptedRunner([(1, True)] * 3), 3, state, ledger, run, est_usd_per_attempt=0.95)
    assert len(load_attempts(state)) == 1  # $0.95 reserved, $0.10 settled; the next $0.95 does not fit
    assert "stopped by the budget" in run.last("note")["data"]["text"]


def test_sample_refuses_heldout_tasks(task, tmp_path):
    state = tmp_path / ".opentrain"
    add_to(HeldoutStore(state), task, tmp_path)
    with pytest.raises(ValueError, match="held-out"):
        sample([task], ScriptedRunner([(1, True)]), 1, state, Ledger(state, Budget(5, None, "$5")))


def test_build_sft_writes_chat_jsonl_and_refuses_heldout(task, tmp_path):
    state = tmp_path / ".opentrain"
    add_to(Store(state), task, tmp_path)
    run = RunRecord(state).open("test", {})
    good = ScriptedRunner([(2, True)]).attempt("ignored", Path(task.repo), task_id="t1", timeout=1)
    good.passed = True
    good.messages[2]["seconds"] = 1.5
    manifest = build_sft({"t1": good}, state, run=run)

    lines = (state / "data" / "train.jsonl").read_text().splitlines()
    assert manifest["examples"] == len(lines) == 1
    ex = json.loads(lines[0])
    assert ex["tools"] == TOOLS
    assert ex["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert ex["messages"][1] == {"role": "user", "content": task.instruction}  # from the store
    assert ex["messages"][-1]["role"] == "assistant"
    assert not any("seconds" in m for m in ex["messages"])
    assert manifest["tasks"][0]["task_digest"] == task.digest()
    assert run.last("data")["data"]["sha256"] == manifest["sha256"]

    off = Attempt("t1", "claude", True, 1.0, messages=[{"role": "user", "content": "x"}, {
        "role": "assistant", "content": "", "tool_calls": [call("Glob", pattern="*")]}])
    assert build_sft({"t1": off}, state)["skipped"][0]["reason"].startswith("uses tool 'Glob'")

    Store(state).move_to("t1", HeldoutStore(state))
    with pytest.raises(HeldoutLeak):
        build_sft({"t1": good}, state)


# ---------------------------------------------------------------- evaluator and race


def test_evaluator_returns_scores_only_and_enforces_the_cap(task, tmp_path):
    state = tmp_path / ".opentrain"
    add_to(HeldoutStore(state), task, tmp_path)
    run = RunRecord(state).open("test", {})
    ev = Evaluator(state, cap=1)
    scores = ev.request(ScriptedRunner([(2, True)]), "v1", run)
    assert scores == [Score("t1", True, 2.0, 10)]
    event = run.last("eval")["data"]
    assert event["request"] == 1 and event["pass_rate"] == 1.0
    assert task.instruction not in json.dumps(run.events())  # no task content in the record
    with pytest.raises(EvalCapReached):
        ev.request(ScriptedRunner([(1, True)]), "v1", run)

    base = ev.baseline(ScriptedRunner([(3, False)]), "base:Qwen/Qwen3-8B")
    assert base == [Score("t1", False, 3.0, 10)]
    assert ev.scores_for("base:Qwen/Qwen3-8B") == base and ev.scores_for("nope") is None
    assert (state / "heldout" / "scores").is_dir()
    assert len(run.events()) == 2  # baselines do not use the run's cap


def test_race_runs_lanes_and_renders(task):
    res = race(task, {"frontier": ScriptedRunner([(5, True)], "claude"),
                      "specialist": ScriptedRunner([(2, True)], "v2")})
    assert res.lanes["frontier"].passed and res.lanes["specialist"].passed
    assert res.winner == "specialist"
    out = res.render()
    assert "frontier" in out and "specialist" in out and "winner: specialist, 2.5x faster than frontier" in out


# ---------------------------------------------------------------- headless runtime


def test_headless_command_claude_argv(tmp_path):
    argv = headless_command("claude", "train the model", system_prompt_file=tmp_path / "role.md",
                            mcp_config=tmp_path / "mcp.json", allowed_tools=["mcp__opentrain__eval", "Read"],
                            cwd=tmp_path, model="opus")
    assert argv[:3] == ["claude", "-p", "train the model"]  # prompt before the variadic flags
    for pair in (["--output-format", "stream-json"], ["--permission-mode", "acceptEdits"],
                 ["--allowedTools", "mcp__opentrain__eval,Read"],
                 ["--append-system-prompt-file", str(tmp_path / "role.md")],
                 ["--mcp-config", str(tmp_path / "mcp.json")], ["--model", "opus"]):
        i = argv.index(pair[0])
        assert argv[i:i + 2] == pair
    assert {"--verbose", "--no-session-persistence", "--strict-mcp-config"} <= set(argv)
    assert headless_command("claude", "-x")[2] == " -x"


def test_headless_command_codex_argv(tmp_path):
    (tmp_path / "role.md").write_text("You are the trainer.")
    (tmp_path / "mcp.json").write_text(json.dumps({"mcpServers": {"opentrain": {
        "command": "opentrain", "args": ["mcp", "serve"], "env": {"OT_RUN": "r1"}}}}))
    argv = headless_command("codex", "go", system_prompt_file=tmp_path / "role.md", mcp_config=tmp_path / "mcp.json",
                            cwd=tmp_path, model="gpt-5")
    assert argv[:3] == ["codex", "exec", "--json"] and argv[-1] == "You are the trainer.\n\ngo"
    assert ["-C", str(tmp_path)] == argv[argv.index("-C"):argv.index("-C") + 2]
    assert 'mcp_servers.opentrain.args=["mcp", "serve"]' in argv
    assert 'mcp_servers.opentrain.env={OT_RUN = "r1"}' in argv


CLAUDE_STREAM = [
    {"type": "system", "subtype": "init", "apiKeySource": "none", "model": "claude-x"},
    {"type": "assistant", "message": {"id": "m1", "content": [{"type": "text", "text": "Looking."}]}},
    {"type": "assistant", "message": {"id": "m1", "content": [
        {"type": "tool_use", "id": "tu1", "name": "Edit",
         "input": {"file_path": "/W/app.py", "old_string": "1", "new_string": "2"}}]}},
    {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "ok"}]}},
    {"type": "assistant", "message": {"id": "m2", "content": [{"type": "text", "text": "Done."}]}},
    {"type": "result", "subtype": "success", "is_error": False, "result": "Done.", "num_turns": 2,
     "duration_api_ms": 2123, "total_cost_usd": 0.0032, "credential_expires_at": "2026-12-23",
     "usage": {"input_tokens": 4, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 50,
               "output_tokens": 81}},
]


def test_parse_claude_stream_json_under_a_subscription(tmp_path):
    lines = [json.dumps(e).replace("/W", str(tmp_path)) for e in CLAUDE_STREAM]
    res = parse_events(["not json", *lines])
    assert res.billing == "subscription" and res.cost_usd is None and res.notional_usd == 0.0032
    assert (res.tokens_in, res.tokens_out, res.tokens, res.turns) == (154, 81, 235, 2)
    assert res.model_seconds == 2.123 and res.text == "Done." and res.error is None
    assert not any("credential" in k or "apiKey" in k for e in res.events for k in e)

    api = parse_events([json.dumps({**CLAUDE_STREAM[0], "apiKeySource": "ANTHROPIC_API_KEY"}), lines[-1]])
    assert api.cost_usd == 0.0032 and api.billing == "api"

    msgs = transcript(res.events, "make f return 2", tmp_path)
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "assistant"]
    fn = msgs[1]["tool_calls"][0]["function"]
    assert msgs[1]["content"] == "Looking."
    assert fn["name"] == "edit_file" and json.loads(fn["arguments"])["path"] == "app.py"


def test_headless_runner_reports_missing_cli(task, monkeypatch):
    from opentrain.runtime import headless

    monkeypatch.setattr(headless, "headless_command", lambda *a, **k: ["opentrain-no-such-cli", "-p", "x"])
    with workspace.checkout(task) as ws:
        att = HeadlessRunner("claude").attempt("x", ws, task_id="t1", timeout=5)
    assert "not installed" in att.error and att.passed is None


# ---------------------------------------------------------------- commands


def test_register_adds_subcommands():
    from opentrain.commands.run import register

    p = argparse.ArgumentParser()
    register(p.add_subparsers(dest="cmd", required=True))
    a = p.parse_args(["sample", "--n", "2", "--model", "qwen", "--endpoint", "http://x/v1"])
    assert a.n == 2 and a.fn.__name__ == "cmd_sample"
    assert p.parse_args(["data", "build"]).fn.__name__ == "cmd_data_build"
    assert p.parse_args(["race", "--model", "v2"]).frontier == "claude"
    a = p.parse_args(["bench-models", "--headless", "claude", "--direct", "http://x/v1", "qwen"])
    assert a.direct == [["http://x/v1", "qwen"]]
