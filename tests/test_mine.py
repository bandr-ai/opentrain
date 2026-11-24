"""The data path, from a session to an accepted task: mine, certify, export.

Every test builds a real git repo and runs real verifiers (``python -m pytest`` in a worktree),
so the controls are exercised for what they are: running tests against known answers.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import timedelta

import pytest
from conftest import T0, git

from opentrain import certify, mine
from opentrain.bench import yield_
from opentrain.commands import data as data_cmd
from opentrain.export import export
from opentrain.sessions.claude_code import load_session
from opentrain.tasks import Controls, HeldoutStore, Store, read_task, split_store
from opentrain.workspace import Verdict

PY = shlex.quote(sys.executable)
PYTEST = f"{PY} -m pytest -q"


def _session(writer, tmp_path, repo, *, sid="s1", command=PYTEST, start=0.0, text="make f return 2",
             files=("app.py",)):
    w = writer(str(repo), session_id=sid).prompt(start, text)
    for i, f in enumerate(files):
        w.call(start + 5 + i, f"e{i}", name="Edit", file_path=f"{repo}/{f}", old_string="1", new_string="2")
        w.result(start + 5.5 + i, f"e{i}")
    w.call(start + 20, "t", command=command).result(start + 25, "t", "3 passed").say(start + 30, "done")
    return load_session(w.write(tmp_path / "logs" / "proj" / f"{sid}.jsonl"))


def _commit(repo, files: dict[str, str], minutes: float = 5, msg: str = "work"):
    for rel, body in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    git(repo, "add", ".")
    git(repo, "commit", "-qm", msg, when=T0 + timedelta(minutes=minutes))


def _state(tmp_path):
    d = tmp_path / "state" / ".opentrain"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _bugfix(repo):
    """The reference fixes f, adds g in a new module, and updates the tests to check both."""
    _commit(repo, {
        "app.py": "def f():\n    return 2\n",
        "util.py": "def g():\n    return 'g'\n",
        "tests/test_app.py": "from app import f\nfrom util import g\n\n"
                             "def test_f():\n    assert f() == 2\n\ndef test_g():\n    assert g() == 'g'\n",
    })


def test_bugfix_session_becomes_an_accepted_task(tmp_path, writer, repo):
    _bugfix(repo)
    v = yield_.judge(_session(writer, tmp_path, repo, command=f"cd {repo} && CI=1 {PYTEST} 2>&1 | tail -5",
                              files=("app.py", "util.py")))
    assert v.verified, v.row()
    state = _state(tmp_path)
    report = mine.mine_all([v], state)
    row = report["rows"][0]
    assert row["outcome"] == "accepted", row

    store = Store(state)
    task = store.get(row["task_id"])
    assert task.test_command == PYTEST
    assert task.test_files == ("tests/test_app.py",)
    assert task.instruction == "make f return 2"
    assert task.created_at == T0.isoformat()
    c = store.controls(task.task_id)
    assert c.reference and not c.noop and c.wrong is False, c
    assert "withheld" in c.detail

    folder = store.folder(task.task_id)
    for rel in ("task.toml", "instruction.md", "environment/Dockerfile", "tests/test.sh",
                "tests/files/tests/test_app.py", "solution/solution.patch", "solution/solve.sh"):
        assert (folder / rel).is_file(), rel
    patch = (folder / "solution" / "solution.patch").read_text()
    assert "util.py" in patch and "tests/test_app.py" not in patch
    assert task.task_id == mine.task_id(v)  # stable


def test_noop_passing_command_is_narrowed_to_the_changed_tests(tmp_path, writer, repo):
    _commit(repo, {"tests/test_other.py": "def test_other():\n    assert True\n"}, minutes=-30)
    _commit(repo, {"app.py": "def f():\n    return 2\n",
                   "tests/test_app.py": "from app import f\n\ndef test_f():\n    assert f() == 2\n"})
    v = yield_.judge(_session(writer, tmp_path, repo, command=f"{PYTEST} tests/test_other.py"))
    report = mine.mine_all([v], _state(tmp_path))
    row = report["rows"][0]
    assert row["outcome"] == "accepted", row
    assert row["narrowed_from"] == f"{PYTEST} tests/test_other.py"
    task = Store(_state(tmp_path)).get(row["task_id"])
    assert task.test_command == f"{PY} -m pytest -q tests/test_app.py"
    assert json.loads((Store(_state(tmp_path)).folder(task.task_id) / "meta.json").read_text())[
        "test_command_source"] == "narrowed"


def test_change_no_test_detects_is_rejected_with_reason(tmp_path, writer, repo):
    _commit(repo, {"app.py": "def f():\n    # one, as before\n    return 1\n"})
    v = yield_.judge(_session(writer, tmp_path, repo))
    report = mine.mine_all([v], _state(tmp_path))
    row = report["rows"][0]
    assert row["outcome"] == "rejected"
    assert row["reason"] == "controls failed: noop passes"
    assert Store(_state(tmp_path)).ids() == []


def test_broken_environment_is_rejected_as_environment(tmp_path, writer, repo):
    _bugfix(repo)
    v = yield_.judge(_session(writer, tmp_path, repo, command="./.no-such-venv/bin/pytest -q"))
    assert v.verified
    row = mine.mine_all([v], _state(tmp_path))["rows"][0]
    assert row["outcome"] == "rejected"
    assert row["reason"] == "environment", row
    assert Store(_state(tmp_path)).ids() == []


def test_missing_third_party_module_is_environment_but_missing_repo_module_is_not():
    names = {"app", "util", "tests", "test_app"}
    third = Verdict(False, 2, 0.1, "E   ModuleNotFoundError: No module named 'numpy'\n1 error")
    ours = Verdict(False, 2, 0.1, "E   ModuleNotFoundError: No module named 'util'\n1 error")
    cannot = Verdict(False, 2, 0.1, "E   ImportError: cannot import name 'g' from 'app'")
    assert certify.classify(third, names).kind == "environment"
    assert certify.classify(ours, names).kind == "tests"
    assert certify.classify(cannot, names).kind == "tests"
    assert certify.classify(Verdict(False, None, 900, "", timed_out=True), names).kind == "environment"
    assert certify.classify(Verdict(False, 127, 0, "sh: pytest: command not found"), names).kind == "environment"
    assert certify.classify(Verdict(False, 1, 0, "FileNotFoundError: No such file or directory: 'x'"),
                            names).kind == "tests"


def test_env_failure_on_noop_is_not_a_valid_noop_failure(tmp_path, writer, repo):
    # The test command is a script the reference adds. At base the shell cannot find it: every
    # answer would fail that way, so the noop "failure" proves nothing about the verifier.
    (repo / "check.sh").write_text(f"#!/bin/sh\nexec {PYTEST}\n")
    (repo / "check.sh").chmod(0o755)
    _commit(repo, {"app.py": "def f():\n    return 2\n",
                   "tests/test_app.py": "from app import f\n\ndef test_f():\n    assert f() == 2\n"})
    v = yield_.judge(_session(writer, tmp_path, repo, command="./check.sh # runs pytest"))
    c = certify.certify(mine.mine(v, tmp_path / "built"))
    assert c.reference and not c.valid
    assert c.detail.startswith("environment: noop"), c.detail


def test_known_wrong_rule():
    two_files = (
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n"
        "diff --git a/b.py b/b.py\n--- a/b.py\n+++ b/b.py\n@@ -1 +1,3 @@\n-x\n+y\n+z\n+w\n"
    )
    patch, what = certify.known_wrong(two_files)
    assert what == "withheld b.py" and "b.py" not in patch and "a.py" in patch
    one_hunk = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n"
    assert certify.known_wrong(one_hunk) is None
    doc_and_code = one_hunk + "diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-a\n+b\n"
    assert certify.known_wrong(doc_and_code)[1] == "withheld a.py"


@pytest.mark.parametrize("raw,cwd,expected", [
    ("cd /r/w && source .venv/bin/activate && CI=1 PYTHONPATH=src timeout 60 pytest -q 2>&1 | tail", "/x",
     "PYTHONPATH=src pytest -q"),
    ("cd /r/w/web && npm test", "/x", "cd web && npm test"),
    ("pytest /r/w/tests/test_a.py; echo done", "/r/w/pkg", "cd pkg && pytest ../tests/test_a.py"),
    ("/r/w/.venv/bin/pytest -x", "/r/w", "/r/w/.venv/bin/pytest -x"),
    ("go test ./... || true", "/r/w/svc", "cd svc && go test ./..."),
    ("ls -la", "/r/w", None),
])
def test_normalize_command(raw, cwd, expected):
    assert mine.normalize_command(raw, cwd, ["/r/w"]) == expected


def test_narrow_command():
    assert mine.narrow_command("cd sub && python -m pytest -q tests/x.py", ("sub/tests/test_n.py",)) == \
        "cd sub && python -m pytest -q tests/test_n.py"
    assert mine.narrow_command("npm test", ("src/a.test.ts",)) == "npm test -- src/a.test.ts"
    assert mine.narrow_command("go test ./...", ("pkg/a_test.go",)) == "go test ./pkg"
    assert mine.narrow_command("make test", ("tests/test_a.py",)) is None


def test_infer_command_for_sessions_that_never_ran_tests(repo):
    head = git(repo, "rev-parse", "HEAD")
    assert mine.infer_command(str(repo), head) == "python3 -m pytest -q"


def test_mine_all_survives_bad_sessions(tmp_path, writer, repo, monkeypatch):
    _bugfix(repo)
    good = yield_.judge(_session(writer, tmp_path, repo, sid="good", files=("app.py", "util.py")))
    bad_ref = yield_.judge(_session(writer, tmp_path, repo, sid="badref"))
    bad_ref.repo = type(bad_ref.repo)(bad_ref.repo.root, bad_ref.repo.base_commit, True, "0" * 40)
    boom = yield_.judge(_session(writer, tmp_path, repo, sid="boom"))
    chat = yield_.judge(load_session(writer(str(repo), session_id="chat").prompt(0, "hi").say(1, "hello")
                                     .write(tmp_path / "chat.jsonl")))
    real = mine.mine_one

    def flaky(v, state, timeout=900):
        if v.session.session_id == "boom":
            raise RuntimeError("disk on fire")
        return real(v, state, timeout)

    monkeypatch.setattr(mine, "mine_one", flaky)
    report = mine.mine_all([bad_ref, boom, chat, good], _state(tmp_path))
    by = {r["session_id"]: r for r in report["rows"]}
    assert by["good"]["outcome"] == "accepted"
    assert by["badref"]["outcome"] == "rejected" and by["badref"]["reason"] == "git error"
    assert by["boom"]["reason"] == "error" and "disk on fire" in by["boom"]["detail"]
    assert by["chat"]["outcome"] == "skipped" and by["chat"]["reason"] == "no code changed"
    assert report["accepted"] == 1 and report["sessions"] == 4


def test_export_leaves_heldout_out_by_default(tmp_path, writer, repo):
    state = _state(tmp_path)
    _commit(repo, {"app.py": "def f():\n    return 2\n"})
    first = mine.mine(yield_.judge(_session(writer, tmp_path, repo, sid="early")), tmp_path / "b")
    day = 24 * 3600
    _commit(repo, {"app.py": "def f():\n    return 3\n"}, minutes=24 * 60 + 5)
    late = _session(writer, tmp_path, repo, sid="late", start=day, text="SECRET held-out ask: return 3")
    second = mine.mine(yield_.judge(late), tmp_path / "b")
    ok = Controls(True, False)
    for f in (first, second):
        Store(state).add(f, ok)
    split_store(state, 0.5)
    held_id = read_task(second).task_id
    assert HeldoutStore(state).ids() == [held_id]
    (state / "data").mkdir()
    (state / "data" / "train.jsonl").write_text('{"messages": []}\n{"messages": []}\n')

    out = tmp_path / "export"
    m = export(state, out)
    assert m["counts"] == {"tasks": 1, "heldout_withheld": 1, "heldout_exported": 0, "train_records": 2}
    assert json.loads((out / "manifest.json").read_text())["opentrain"]
    assert not (out / "evals.jsonl").exists() and not (out / "heldout").exists()
    assert all("SECRET" not in p.read_text(errors="ignore") for p in out.rglob("*") if p.is_file())
    assert m["files"]["train.jsonl"] and m["tasks"][0]["digest"]

    m2 = export(state, out, include_heldout=True)  # re-export over an earlier export is allowed
    evals = [json.loads(line) for line in (out / "evals.jsonl").read_text().splitlines()]
    assert evals[0]["task_id"] == held_id and "SECRET" in evals[0]["instruction"]
    assert m2["counts"]["heldout_exported"] == 1 and (out / "heldout" / held_id / "task.toml").is_file()

    stranger = tmp_path / "not-empty"
    stranger.mkdir()
    (stranger / "keep.txt").write_text("mine")
    with pytest.raises(ValueError):
        export(state, stranger)


def test_cli_mine_then_task_commands(tmp_path, writer, repo, capsys):
    _bugfix(repo)
    _session(writer, tmp_path, repo, files=("app.py", "util.py"))
    parser = argparse.ArgumentParser(prog="opentrain")
    data_cmd.register(parser.add_subparsers(dest="cmd", required=True))
    state = str(tmp_path / "state")

    def run(*argv):
        args = parser.parse_args(list(argv))
        return args.fn(args)

    assert run("mine", "--sessions", str(tmp_path / "logs"), "--state", state, "--repos", str(repo.parent)) == 0
    out = capsys.readouterr().out
    assert "1 verified tasks, 1 held out for the evaluator" in out  # one task: the split holds it out
    assert run("task", "ls", "--state", state) == 0
    assert "1 held out" in capsys.readouterr().out
    held = HeldoutStore(tmp_path / "state" / ".opentrain").ids()[0]
    assert run("task", "show", held, "--state", state) == 1
    assert "held out" in capsys.readouterr().err
    assert run("lint", "--state", state) == 0
    assert run("export", str(tmp_path / "out"), "--state", state) == 0
