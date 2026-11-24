from __future__ import annotations

import json
from datetime import timedelta

from conftest import T0, git

from opentrain.bench import split, timing, yield_
from opentrain.bench.gitstate import find_repos, resolve_vanished
from opentrain.cli import main
from opentrain.sessions.claude_code import load_session


def _coding_session(writer, tmp_path, cwd, *, passing=True, sid="s1"):
    w = (
        writer(str(cwd), session_id=sid)
        .prompt(0, "make f return 2")
        .call(5, "e", name="Edit", file_path=f"{cwd}/app.py", old_string="1", new_string="2")
        .result(6, "e")
        .call(7, "t", command="pytest -q")
        .result(12, "t", "1 passed" if passing else "1 failed", is_error=not passing)
        .say(14, "done")
    )
    return load_session(w.write(tmp_path / "logs" / "proj" / f"{sid}.jsonl"))


def _commit_fix(repo):
    (repo / "app.py").write_text("def f():\n    return 2\n")
    git(repo, "commit", "-qam", "fix", when=T0 + timedelta(minutes=5))


def test_verified_candidate(tmp_path, writer, repo):
    _commit_fix(repo)
    v = yield_.judge(_coding_session(writer, tmp_path, repo))
    assert v.verified, v.row()
    assert v.repo.base_commit and v.repo.reference_commit and v.repo.tests_in_repo


def test_failed_checks_are_repairable(tmp_path, writer, repo):
    _commit_fix(repo)
    v = yield_.judge(_coding_session(writer, tmp_path, repo, passing=False))
    assert v.skip == "checks not run or failed" and v.repairable


def test_uncommitted_work_has_no_reference(tmp_path, writer, repo):
    v = yield_.judge(_coding_session(writer, tmp_path, repo))
    assert v.skip == "no committed reference"


def test_question_only_session(tmp_path, writer, repo):
    s = load_session(writer(str(repo)).prompt(0, "what does f do?").say(3, "returns 1").write(tmp_path / "q.jsonl"))
    assert yield_.judge(s).skip == "no code changed"


def test_vanished_worktree_resolves_to_main_clone(tmp_path, writer, repo):
    _commit_fix(repo)
    gone = tmp_path / "workspaces" / "widget" / "charlotte"
    s = _coding_session(writer, tmp_path, gone)
    repos = find_repos([repo.parent])
    assert resolve_vanished(str(gone), repos) == str(repo)
    v = yield_.judge(s, repos)
    assert v.verified and v.repo.resolved_from == str(gone)


def test_session_outside_repo_uses_edited_files(tmp_path, writer, repo):
    _commit_fix(repo)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    w = (
        writer(str(scratch)).prompt(0, "edit it")
        .call(5, "e", name="Edit", file_path=f"{repo}/app.py", old_string="1", new_string="2").result(6, "e")
        .call(7, "t", command=f"cd {repo} && pytest").result(9, "t", "1 passed").say(10, "done")
    )
    v = yield_.judge(load_session(w.write(tmp_path / "x.jsonl")))
    assert v.verified and v.repo.root == str(repo)


def test_resolve_prefers_longest_name():
    assert resolve_vanished("/w/acme-platform/astana", ["/r/acme", "/r/acme-platform"]) == "/r/acme-platform"
    assert resolve_vanished("/w/other/x", ["/r/acme"]) is None


def test_funnel_accounts_for_everyone(tmp_path, writer, repo):
    _commit_fix(repo)
    a = yield_.judge(_coding_session(writer, tmp_path, repo, sid="a"))
    b = yield_.judge(load_session(writer(str(repo), session_id="b").prompt(0, "hi").say(1, "x").write(tmp_path / "b.jsonl")))
    summary = yield_.summarise([a, b])
    assert summary["verified_candidates"] == 1
    assert summary["funnel"][-1]["remaining"] == 1
    assert sum(f["dropped"] for f in summary["funnel"]) == 1


def test_split_by_date_holds_out_latest():
    items = [(f"t{i}", T0 + timedelta(days=i)) for i in range(10)]
    sp = split.split_by_date(items, heldout_fraction=0.3)
    assert sp.heldout == ["t7", "t8", "t9"]
    assert set(sp.train).isdisjoint(sp.heldout)
    assert max(T0 + timedelta(days=int(i[1:])) for i in sp.train) < sp.cutoff


def test_split_ties_stay_together():
    items = [("a", T0), ("b", T0 + timedelta(days=1)), ("c", T0 + timedelta(days=1))]
    sp = split.split_by_date(items, heldout_fraction=0.34)
    assert sp.heldout == ["b", "c"]


def test_amdahl_ceiling():
    assert round(timing.ceiling_speedup(0.5, 10), 2) == 1.82
    assert timing.command_head("cd /x && FOO=1 uv run pytest") == "uv"
    assert timing.command_head("cd /x") == "cd"


def test_cli_end_to_end(tmp_path, writer, repo, capsys):
    _commit_fix(repo)
    _coding_session(writer, tmp_path, repo)
    state = tmp_path / "state"
    rc = main(["bench", "--sessions", str(tmp_path / "logs"), "--state", str(state), "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["yield"]["verified_candidates"] == 1
    assert out["split"]["heldout"] == 1
    held = json.loads((state / ".opentrain" / "heldout" / "ids.json").read_text())
    assert held["ids"] == ["s1"]
    assert oct((state / ".opentrain" / "heldout").stat().st_mode)[-3:] == "700"
