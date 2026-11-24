from __future__ import annotations

import json

import pytest

from opentrain import gate, trainfile
from opentrain.budget import BudgetExceeded, Ledger
from opentrain.record import RunRecord
from opentrain.registry import Registry
from opentrain.tasks import Controls, HeldoutStore, Store, Task, TaskRejected, split_store, task_toml

PLAN_TRAINFILE = """\
# Trainfile
goal         speed
budget       $40 a week, model tokens and compute together
promote      beats current on held-out, no task regresses
runtime      claude -p
trainer      tinker
base model   qwen-27b
data         my agent sessions, redacted, local until I approve
watch        when 30 new sessions arrive, or nightly
"""


def test_trainfile_from_the_plan_parses():
    tf = trainfile.parse(PLAN_TRAINFILE)
    assert tf.goal == "speed"
    assert tf.budget.dollars == 40 and tf.budget.period_days == 7
    assert tf.runtime == "claude"
    assert tf.base_model == "qwen-27b"
    assert tf.watch.new_sessions == 30 and tf.watch.nightly
    assert tf.uploads_need_approval and not tf.allow_regressions


def test_default_trainfile_parses():
    assert trainfile.parse(trainfile.DEFAULT).heldout_fraction == pytest.approx(0.3)


@pytest.mark.parametrize("bad", ["goal  vibes", "budget  lots", "colour  blue", "goal  speed\ngoal  cost",
                                 "runtime  emacs", "heldout  150%"])
def test_trainfile_rejects(bad):
    with pytest.raises(trainfile.TrainfileError):
        trainfile.parse(bad)


def test_budget_periods():
    assert trainfile.parse("budget  $5 per run").budget.period_days is None
    assert trainfile.parse("budget  $5 a day").budget.period_days == 1
    assert trainfile.parse("budget  $100 monthly").budget.period_days == 30


def test_run_record_is_append_only_and_chained(tmp_path):
    rr = RunRecord(tmp_path)
    run = rr.open("first run", {"goal": "speed"})
    run.add("spend", usd=1.25, what="sampling")
    run.add("gate", promote=False, why="regressed")
    run.add("closed")
    assert run.verify() and run.spent() == 1.25
    with pytest.raises(Exception):
        run.add("note", text="after close")
    lines = run.events_path.read_text().splitlines()
    tampered = json.loads(lines[1])
    tampered["data"]["usd"] = 0.01
    lines[1] = json.dumps(tampered)
    run.events_path.write_text("\n".join(lines) + "\n")
    assert not run.verify()
    assert rr.get(run.run_id[:15]).run_id == run.run_id


def test_ledger_refuses_overspend(tmp_path):
    led = Ledger(tmp_path, trainfile.Budget(10, 7, "$10 a week"))
    r1 = led.reserve(6, "train")
    with pytest.raises(BudgetExceeded):
        led.reserve(5, "sample")
    led.settle(r1, 3.5)
    assert led.remaining() == 6.5
    r2 = led.reserve(6, "sample")
    led.release(r2)
    assert led.remaining() == 6.5


def test_per_run_budget_is_scoped_to_run(tmp_path):
    led = Ledger(tmp_path, trainfile.Budget(5, None, "$5 per run"))
    led.settle(led.reserve(5, "x", run_id="a"), 5)
    assert led.remaining("b") == 5
    with pytest.raises(BudgetExceeded):
        led.reserve(1, "y", run_id="a")


S = gate.Score


def test_gate_promotes_faster_with_no_regressions():
    cur = [S("a", True, 100), S("b", True, 200), S("c", False, 50)]
    cand = [S("a", True, 20), S("b", True, 40), S("c", True, 30)]
    d = gate.decide(cur, cand, "speed")
    assert d.promote and d.fixed == ["c"] and d.summary["gain"] == pytest.approx(0.8)


def test_gate_rejects_any_regression():
    cur = [S("a", True, 100), S("b", True, 200)]
    cand = [S("a", True, 10), S("b", False, 10)]
    d = gate.decide(cur, cand, "speed")
    assert not d.promote and d.regressions == ["b"]


def test_gate_rejects_no_gain_and_missing_scores_and_broken_verifiers():
    cur = [S("a", True, 100), S("b", True, 100)]
    assert not gate.decide(cur, [S("a", True, 99), S("b", True, 100)], "speed").promote
    assert not gate.decide(cur, [S("a", True, 10)], "speed").promote
    assert not gate.decide(cur, [S("a", True, 1), S("b", True, 1)], "speed", broken_verifiers=["a"]).promote


def test_gate_fix_goal():
    cur = [S("a", True), S("f", False)]
    assert gate.decide(cur, [S("a", True), S("f", True)], "fix", flagged=["f"]).promote
    assert not gate.decide(cur, [S("a", True), S("f", False)], "fix", flagged=["f"]).promote


def test_gate_first_version():
    assert gate.decide(None, [S("a", True, 5), S("b", False)], "speed").promote
    assert not gate.decide(None, [S("a", False)], "speed").promote


def test_registry_promote_and_revert(tmp_path):
    reg = Registry(tmp_path)
    v1 = reg.add(run_id="r1", base_model="m", backend="dry", checkpoint="c1")
    v2 = reg.add(run_id="r2", base_model="m", backend="dry", checkpoint="c2")
    reg.promote(v1.version, "first")
    reg.promote(v2.version, "faster")
    assert reg.current().version == "v2"
    assert reg.revert().version == "v1"
    assert reg.get("v2").decision == "reverted"
    assert reg.revert() is None and reg.current() is None


def _task_folder(tmp_path, tid="t1", created="2026-11-01T00:00:00+00:00"):
    t = Task(tid, "do it", "/repo", "a" * 40, "b" * 40, "pytest -q", ("tests/test_x.py",), created_at=created)
    f = tmp_path / "build" / tid
    f.mkdir(parents=True)
    (f / "task.toml").write_text(task_toml(t))
    (f / "instruction.md").write_text(t.instruction)
    return f, t


def test_store_requires_valid_controls(tmp_path):
    store = Store(tmp_path / "state")
    f, t = _task_folder(tmp_path)
    with pytest.raises(TaskRejected, match="doing nothing passes"):
        store.add(f, Controls(reference=True, noop=True))
    with pytest.raises(TaskRejected, match="reference"):
        store.add(f, Controls(reference=False, noop=False))
    got = store.add(f, Controls(reference=True, noop=False))
    assert got == t and store.get("t1").test_files == ("tests/test_x.py",)
    assert store.controls("t1").valid


def test_split_store_moves_latest_and_never_moves_back(tmp_path):
    state = tmp_path / "state"
    store = Store(state)
    for i in range(10):
        f, _ = _task_folder(tmp_path, f"t{i}", f"2026-11-{i + 1:02d}T00:00:00+00:00")
        store.add(f, Controls(True, False))
    r = split_store(state, 0.3)
    assert r == {"train": 7, "heldout": 3, "moved": 3, "cutoff": r["cutoff"]}
    held = HeldoutStore(state)
    assert held.ids() == ["t7", "t8", "t9"]
    r2 = split_store(state, 0.1)
    assert r2["heldout"] == 3 and "t8" in held.ids()


def test_verify_reference_passes_and_noop_fails(tmp_path, repo):
    import sys
    from datetime import timedelta

    from conftest import T0, git

    from opentrain import workspace

    base = git(repo, "rev-parse", "HEAD")
    (repo / "app.py").write_text("def f():\n    return 2\n")
    (repo / "tests" / "test_app.py").write_text("from app import f\n\ndef test_f():\n    assert f() == 2\n")
    git(repo, "commit", "-qam", "fix", when=T0 + timedelta(minutes=5))
    ref = git(repo, "rev-parse", "HEAD")
    t = Task("t", "make f return 2", str(repo), base, ref, f"{sys.executable} -m pytest -q -p no:cacheprovider",
             ("tests/test_app.py",))
    patch = workspace.reference_patch(t)
    assert "app.py" in patch and "test_app" not in patch
    assert workspace.verify(t, patch).passed
    assert not workspace.verify(t, "").passed
    assert not workspace.verify(t, "garbage").passed
    assert git(repo, "worktree", "list").count("\n") == 0  # cleaned up


def test_per_run_budget_counts_settled_amount(tmp_path):
    led = Ledger(tmp_path, trainfile.Budget(5, None, "$5 per run"))
    led.settle(led.reserve(5, "x", run_id="a"), 1)
    assert led.remaining("a") == 4


def test_heldout_id_can_never_be_added_to_training(tmp_path):
    state = tmp_path / "state"
    f, _ = _task_folder(tmp_path, "t1")
    HeldoutStore(state).add(f, Controls(True, False))
    with pytest.raises(TaskRejected, match="held out"):
        Store(state).add(f, Controls(True, False))


def test_inconclusive_noop_is_not_valid():
    c = Controls(reference=True, noop=None, detail="environment: python missing")
    assert not c.valid and "inconclusive" in c.reason()
