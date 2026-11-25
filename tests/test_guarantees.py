"""The guarantees OpenTrain makes, each pinned by the failure an adversarial review found."""

from __future__ import annotations

import json
import sys
from datetime import timedelta

import pytest
from conftest import T0, git

from opentrain import gate, trainfile, workspace
from opentrain.budget import BudgetExceeded, Ledger
from opentrain.evaluator import EvalBroken, Evaluator
from opentrain.plugs import Attempt
from opentrain.registry import Registry
from opentrain.tasks import Controls, HeldoutStore, Store, Task, split_store, task_toml

S = gate.Score


# ------------------------------------------------------------------ gate


def test_gate_rejects_a_baseline_that_never_saw_the_new_heldout_tasks():
    cur = [S("a", True, 100), S("b", True, 100)]
    cand = [S("a", True, 10), S("b", True, 10)] + [S(f"n{i}", False, 5) for i in range(8)]
    d = gate.decide(cur, cand, "speed")
    assert not d.promote and "re-score" in d.reasons[0]


def test_gate_medians_use_only_tasks_measured_on_both_sides():
    cur = [S("a", True, 10), S("b", True, 100)]
    cand = [S("a", True, 9), S("b", True, None)]  # the slow one went unmeasured
    d = gate.decide(cur, cand, "speed")
    assert d.summary["median_seconds"] == 9 and d.summary["current_median_seconds"] == 10


def test_gate_treats_zero_tokens_as_unreported():
    cur = [S("a", True, tokens=5000)]
    cand = [S("a", True, tokens=0)]
    assert not gate.decide(cur, cand, "cost").promote


def test_gate_fix_counts_an_unscored_flagged_task_as_failing():
    assert not gate.decide([S("a", True)], [S("a", True)], "fix", flagged=["gone"]).promote


# ------------------------------------------------------------------ registry and serving pin


def test_regating_a_promoted_version_keeps_its_history(tmp_path):
    reg = Registry(tmp_path)
    v1 = reg.add(run_id="r", base_model="m", backend="fake", checkpoint="c")
    reg.promote(v1.version, "first")
    reg.record_gate(v1.version, "rejected: whatever", {"pass_rate": 0})
    assert reg.get("v1").decision == "promoted" and reg.get("v1").scores == {"pass_rate": 0}


def test_promote_and_revert_clear_a_serving_pin(tmp_path):
    from opentrain.serving import live_version, use_version

    reg = Registry(tmp_path)
    for _ in range(3):
        reg.add(run_id="r", base_model="m", backend="fake", checkpoint="c")
    reg.promote("v1", "x")
    use_version(tmp_path, "v1")
    reg.promote("v2", "better")
    assert live_version(tmp_path) == "v2"
    use_version(tmp_path, "v3")
    reg.revert("live drop")
    assert live_version(tmp_path) == "v1"


# ------------------------------------------------------------------ held-out isolation


def _folder(tmp_path, tid, created, session=None):
    t = Task(tid, "x", "/r", "a" * 40, "b" * 40, "true", (), created_at=created, source_session=session)
    f = tmp_path / "b" / tid
    f.mkdir(parents=True)
    (f / "task.toml").write_text(task_toml(t))
    (f / "instruction.md").write_text("x")
    return f


def test_a_task_a_model_has_seen_is_never_moved_into_heldout(tmp_path):
    state = tmp_path / "s"
    store = Store(state)
    for i in range(10):
        store.add(_folder(tmp_path, f"t{i}", f"2026-11-{i + 1:02d}T00:00:00+00:00"), Controls(True, False))
    (state / "attempts" / "t6").mkdir(parents=True)  # t6 was sampled
    r = split_store(state, 0.4)
    assert "t6" not in HeldoutStore(state).ids() and r["kept_in_training"] == 1


def test_mcp_refuses_to_train_on_a_stale_training_set(tmp_path):
    from opentrain.mcp import Server
    from opentrain.state import ensure

    state = ensure(tmp_path)
    HeldoutStore(state).add(_folder(tmp_path, "h1", "2026-11-01T00:00:00+00:00"), Controls(True, False))
    (state / "data").mkdir(exist_ok=True)
    (state / "data" / "train.jsonl").write_text('{"messages": []}\n')
    (state / "data" / "manifest.json").write_text(json.dumps({"tasks": [{"task_id": "h1"}]}))
    srv = Server(state, "trainer", None, trainfile.parse("trainer  fake"))
    out = srv.call("train_launch", {})
    assert out["isError"] and "stale" in out["content"][0]["text"]


def test_agent_mode_gives_the_trainer_no_shell_and_no_view_of_state(tmp_path, monkeypatch):
    from opentrain import engine
    from opentrain.state import ensure

    state = ensure(tmp_path)
    seen = {}
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
    monkeypatch.setattr("opentrain.runtime.headless.run_headless",
                        lambda argv, cwd, timeout: seen.setdefault("argv", argv) and _Res())
    engine.agent(state, trainfile.parse("runtime  claude -p"), "go", say=lambda _: None)
    argv = seen["argv"]
    tools = argv[argv.index("--tools") + 1]
    denied = argv[argv.index("--disallowedTools") + 1]
    assert "Bash" not in tools and "Write" not in tools
    assert "Bash" in denied and f"Read({state}/**)" in denied


class _Res:
    text, cost_usd, tokens, turns, error = "done", None, 0, 1, None


# ------------------------------------------------------------------ budget


def test_sampling_cannot_spend_past_the_budget(tmp_path):
    from opentrain.sample import sample

    class Pricey:
        name = "pricey"

        def attempt(self, instruction, ws, *, task_id, timeout):
            return Attempt(task_id=task_id, model="pricey", passed=None, seconds=0.0, cost_usd=4.0)

    repo = tmp_path / "r"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "t@e.com")
    git(repo, "config", "user.name", "t")
    (repo / "a").write_text("x")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base", when=T0)
    base = git(repo, "rev-parse", "HEAD")
    task = Task("t", "x", str(repo), base, base, "true")
    led = Ledger(tmp_path / "state", trainfile.Budget(10, 7, "$10 a week"))
    with pytest.raises(BudgetExceeded):
        sample([task], Pricey(), 5, tmp_path / "state", led, workers=1)
    assert led.committed() <= 10


# ------------------------------------------------------------------ evaluator


def test_a_baseline_where_the_model_never_ran_is_not_kept(tmp_path):
    state = tmp_path / "s"
    repo = tmp_path / "r"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "t@e.com")
    git(repo, "config", "user.name", "t")
    (repo / "a").write_text("x")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base", when=T0)
    base = git(repo, "rev-parse", "HEAD")
    t = Task("h", "x", str(repo), base, base, "false", created_at="2026-11-01T00:00:00+00:00")
    f = tmp_path / "b" / "h"
    f.mkdir(parents=True)
    (f / "task.toml").write_text(task_toml(t))
    (f / "instruction.md").write_text("x")
    HeldoutStore(state).add(f, Controls(True, False))

    class LoggedOut:
        name = "claude"

        def attempt(self, instruction, ws, *, task_id, timeout):
            return Attempt(task_id=task_id, model="claude", passed=None, seconds=0.1, error="not logged in")

    ev = Evaluator(state, 3)
    with pytest.raises(EvalBroken):
        ev.baseline(LoggedOut(), "frontier:claude")
    assert ev.scores_for("frontier:claude") is None


# ------------------------------------------------------------------ verifier


def test_an_attempt_cannot_pass_by_rewriting_test_config(tmp_path, repo):
    base = git(repo, "rev-parse", "HEAD")
    (repo / "app.py").write_text("def f():\n    return 2\n")
    (repo / "tests" / "test_app.py").write_text("from app import f\n\ndef test_f():\n    assert f() == 2\n")
    git(repo, "commit", "-qam", "fix", when=T0 + timedelta(minutes=5))
    ref = git(repo, "rev-parse", "HEAD")
    t = Task("t", "x", str(repo), base, ref, f"{sys.executable} -m pytest -q -p no:cacheprovider", ("tests/test_app.py",))
    with workspace.checkout(t) as ws:
        (ws / "conftest.py").write_text(
            "import pytest\n\ndef pytest_collection_modifyitems(items):\n"
            "    for i in items:\n        i.add_marker(pytest.mark.skip)\n")
        (ws / "tests" / "test_other.py").write_text("def test_x():\n    assert True\n")
        v = workspace.run_tests(t, ws)
        assert not v.passed, v.output
        assert not (ws / "conftest.py").exists()


def test_an_attempt_cannot_delete_the_users_dependencies(tmp_path, repo, monkeypatch):
    monkeypatch.setattr(workspace, "ENV_CACHE", tmp_path / "cache")
    (repo / "node_modules" / "left-pad").mkdir(parents=True)
    (repo / "node_modules" / "left-pad" / "index.js").write_text("module.exports = 1\n")
    base = git(repo, "rev-parse", "HEAD")
    t = Task("t", "x", str(repo), base, base, "true")
    with workspace.checkout(t) as ws:
        assert (ws / "node_modules" / "left-pad" / "index.js").exists()
        import shutil

        shutil.rmtree(ws / "node_modules" / "left-pad")
    assert (repo / "node_modules" / "left-pad" / "index.js").exists()


def test_the_same_session_is_never_mined_twice(tmp_path):
    from opentrain import mine as mine_mod

    state = tmp_path / "s"
    HeldoutStore(state).add(_folder(tmp_path, "old-id", "2026-11-01T00:00:00+00:00", session="sess-1"),
                            Controls(True, False))

    class V:
        class session:
            session_id = "sess-1"

    monkey_id = "new-id"
    orig = mine_mod.task_id
    mine_mod.task_id = lambda v: monkey_id
    try:
        out = mine_mod.mine_one(V(), state)
    finally:
        mine_mod.task_id = orig
    assert out == {"outcome": "kept", "task_id": "old-id", "reason": "session already mined"}
    assert Store(state).ids() == []


