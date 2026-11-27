"""Coding work as data: traces to training examples, moments, history tasks, the model catalog, logins
found on disk, and keeping held-out work out of training."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from opentrain import models
from opentrain.history import candidates, strip_leaks
from opentrain.traces import Step, ToolUse, Trace
from opentrain.traces.sft import Skip, examples, patch_calls, translate
from opentrain.traces.signals import moments

T0 = datetime(2026, 11, 20, 10, 0, tzinfo=timezone.utc)


def tu(kind: str, name: str, output: str = "", **args) -> ToolUse:
    return ToolUse(id=f"t{abs(hash((kind, name, str(args)))) % 10**6}", name=name, kind=kind, args=args, output=output)


def trace(*steps: Step, cwd: str = "/repo") -> Trace:
    return Trace(trace_id="tr1", agent="claude-code", path="/x.jsonl", cwd=cwd, started_at=T0, ended_at=T0,
                 steps=list(steps))


# ---------------------------------------------------------------- traces as examples


def test_a_request_becomes_one_example_in_the_runtime_tools():
    t = trace(
        Step("user", "fix the pager"),
        Step("agent", "", [tu("read", "Read", "     1\tdef page(n): return range(n+1)", path="/repo/pager.py")]),
        Step("agent", "", [tu("edit", "Edit", "ok", path="/repo/pager.py", old="n+1", new="n")]),
        Step("agent", "", [tu("shell", "Bash", "1 passed", command="pytest -q")]),
        Step("agent", "Fixed the off-by-one."),
    )
    good, skipped = examples(t)
    assert not skipped and len(good) == 1
    msgs = good[0]["messages"]
    assert msgs[1] == {"role": "user", "content": "fix the pager"}
    calls = [m["tool_calls"][0]["function"] for m in msgs if m["role"] == "assistant"]
    assert [c["name"] for c in calls] == ["read_file", "edit_file", "bash", "finish"]
    assert json.loads(calls[0]["arguments"]) == {"path": "pager.py"}  # relative to the repo
    assert json.loads(calls[1]["arguments"]) == {"path": "pager.py", "old_string": "n+1", "new_string": "n"}
    assert json.loads(calls[3]["arguments"])["summary"] == "Fixed the off-by-one."
    tool_out = [m["content"] for m in msgs if m["role"] == "tool"]
    assert tool_out[2].startswith("exit code 0\n")
    assert good[0]["meta"]["trace"] == "tr1"


def test_requests_without_code_changes_or_with_tools_the_model_lacks_are_left_out_with_a_reason():
    t = trace(
        Step("user", "what does this repo do?"),
        Step("agent", "", [tu("read", "Read", "x", path="/repo/README.md")]),
        Step("user", "now look it up online and patch it"),
        Step("agent", "", [tu("web", "WebSearch", "results", query="x")]),
        Step("agent", "", [tu("edit", "Edit", "ok", path="/repo/a.py", old="a", new="b")]),
    )
    good, skipped = examples(t)
    assert good == []
    reasons = {s["reason"] for s in skipped}
    assert "changed no code" in reasons
    assert any("WebSearch" in r for r in reasons)


def test_files_outside_the_repo_cannot_be_learned_from():
    with pytest.raises(Skip):
        translate(tu("write", "Write", "", path="/etc/hosts", content="x"), "/repo")


def test_apply_patch_becomes_the_edits_it_makes():
    patch = """*** Begin Patch
*** Update File: src/a.py
@@ def f():
 def f():
-    return 1
+    return 2
*** Add File: src/b.py
+print('hi')
*** Delete File: old.py
*** End Patch"""
    calls = patch_calls(patch)
    assert calls[0] == ("edit_file", {"path": "src/a.py", "old_string": "def f():\n    return 1",
                                      "new_string": "def f():\n    return 2"})
    assert calls[1] == ("write_file", {"path": "src/b.py", "content": "print('hi')\n"})
    assert calls[2][0] == "bash" and "rm old.py" in calls[2][1]["command"]


def test_searches_become_the_rg_command_that_finds_the_same_thing():
    (name, args, _), = translate(tu("search", "Grep", "a.py:1:x", pattern="def page", path="src"), "/repo")
    assert name == "bash" and args["command"] == "rg -n 'def page' src"
    (name, args, _), = translate(tu("search", "Glob", "a.py", pattern="**/*.py"), "/repo")
    assert args["command"] == "rg --files . -g '**/*.py'"


# ---------------------------------------------------------------- moments


def test_moments_point_at_corrections_interruptions_and_failing_checks():
    fail = ToolUse(id="f", name="Bash", kind="shell", args={"command": "pytest -q"}, output="1 failed", exit_code=1)
    t = trace(
        Step("user", "add retries to the client"),
        Step("agent", "", [fail]), Step("agent", "", [fail]), Step("agent", "", [fail]),
        Step("user", "no, use the backoff helper we already have instead"),
        Step("agent", "", [tu("edit", "Edit", "ok", path="/repo/c.py", old="a", new="b")]),
        Step("user", "[Request interrupted by user]"),
    )
    kinds = {(m.kind, m.request) for m in moments(t)}
    assert ("failing_check", 0) in kinds
    assert ("pushback", 0) in kinds
    assert ("interrupted", 1) in kinds


# ---------------------------------------------------------------- tasks from history


def _git(repo: Path, *a: str) -> None:
    subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)


def test_history_candidates_are_commits_that_change_code_and_tests(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "dev@example.com")
    _git(repo, "config", "user.name", "Dev")
    (repo / "a.py").write_text("x = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "start")
    (repo / "a.py").write_text("x = 2\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_a.py").write_text("def test(): pass\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "Make x two so the totals add up")
    (repo / "a.py").write_text("x = 3\n")
    _git(repo, "commit", "-qam", "bump x without any tests")
    found, skipped = candidates(str(repo), author="dev@example.com")
    assert [c["message"] for c in found] == ["Make x two so the totals add up"]
    assert skipped["no test change"] == 1 and skipped["first commit"] == 1


def test_commit_text_loses_what_would_leak_the_answer():
    text = "Fix pager off-by-one (fixes #12)\n\nSee https://example.com/x 0123456789abcdef0123\nCo-Authored-By: a <b>"
    assert strip_leaks(text) == "Fix pager off-by-one\n\nSee"


# ---------------------------------------------------------------- the model catalog


def test_serving_flags_turn_tool_calls_on_for_known_families():
    assert models.vllm_tool_args("Qwen/Qwen3.5-4B") == ["--enable-auto-tool-choice", "--tool-call-parser",
                                                        "qwen3_coder", "--reasoning-parser", "qwen3"]
    assert models.vllm_tool_args("Qwen/Qwen2.5-7B-Instruct")[:3] == ["--enable-auto-tool-choice",
                                                                    "--tool-call-parser", "hermes"]
    assert models.vllm_tool_args("some/unknown-model") == []
    assert models.trainers_for("thinkingmachines/Inkling-2-Mini") == ("tinker",)
    assert all(m.source for m in models.CATALOG)


def test_tinker_prices_cover_every_tinker_model_in_the_catalog():
    from opentrain.backends.tinker import TRAIN_USD_PER_MTOK

    missing = [m.id for m in models.CATALOG if "tinker" in m.trainers and m.id not in TRAIN_USD_PER_MTOK]
    assert missing == []


# ---------------------------------------------------------------- logins other tools keep


def test_a_hugging_face_login_is_found_where_the_cli_keeps_it(tmp_path, monkeypatch):
    from opentrain.core import integrations

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN_PATH", raising=False)
    (tmp_path / "token").write_text("hf_abc\n")
    where, key = integrations.found_elsewhere("huggingface")
    assert key == "hf_abc" and str(tmp_path) in where


# ---------------------------------------------------------------- held-out work stays out of training


def test_trace_examples_on_or_after_the_heldout_cutoff_are_left_out(tmp_path, monkeypatch):
    from opentrain.core.tools import coding

    state = tmp_path / ".opentrain"
    (state / "traces").mkdir(parents=True)
    rows = [{"messages": [], "meta": {"trace": "old", "request": "0", "at": "2026-11-01T00:00:00+00:00"}},
            {"messages": [], "meta": {"trace": "new", "request": "0", "at": "2026-11-25T00:00:00+00:00"}},
            {"messages": [], "meta": {"trace": "nodate", "request": "0", "at": ""}}]
    (state / "traces" / "examples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(coding, "heldout_cutoff", lambda s: datetime(2026, 11, 20, tzinfo=timezone.utc))
    keep, dropped = coding.trace_examples(state)
    assert len(keep) == 1 and "meta" not in keep[0]
    assert dropped == {"on or after the held-out cutoff": 2}


# ---------------------------------------------------------------- public trajectories


def test_public_trajectories_in_both_shapes_become_examples():
    from opentrain.traces.public import rows_to_traces

    tool_shape = [{"role": "system", "content": "sys"},
                  {"role": "user", "content": [{"type": "text", "text": "<uploaded_files>\n/testbed\n</uploaded_files> fix it"}]},
                  {"role": "assistant", "content": "look", "tool_calls": [{"id": "a", "function": {
                      "name": "str_replace_editor", "arguments": json.dumps({"command": "view", "path": "/testbed/m.py"})}}]},
                  {"role": "tool", "tool_call_ids": ["a"], "content": [{"type": "text", "text": "OBSERVATION:\n1 x = 1"}]},
                  {"role": "assistant", "content": "", "tool_calls": [{"id": "b", "function": {
                      "name": "str_replace_editor", "arguments": json.dumps(
                          {"command": "str_replace", "path": "/testbed/m.py", "old_str": "x = 1", "new_str": "x = 2"})}}]},
                  {"role": "tool", "tool_call_ids": ["b"], "content": "OBSERVATION:\nedited"},
                  {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "function": {"name": "submit",
                                                                                                "arguments": "{}"}}]}]
    text_shape = [{"role": "user", "content": "Consider the following github issue: in /testbed ..."},
                  {"role": "assistant", "content": "Run it.\n<function=execute_bash>\n<parameter=command>cd /testbed && "
                                                   "python r.py</parameter>\n</function>"},
                  {"role": "user", "content": "Execution output of [execute_bash]:\nTypeError in /testbed/r.py"},
                  {"role": "assistant", "content": "<function=file_editor>\n<parameter=command>str_replace</parameter>\n"
                                                   "<parameter=path>/testbed/r.py</parameter>\n<parameter=old_str>a</parameter>\n"
                                                   "<parameter=new_str>b</parameter>\n</function>"},
                  {"role": "user", "content": "Execution output of [file_editor]:\nok"},
                  {"role": "assistant", "content": "<function=finish>\n<parameter=command>submit</parameter>\n</function>"}]
    one = rows_to_traces("SWE-bench/SWE-smith-trajectories", [{"messages": json.dumps(tool_shape), "resolved": True},
                                                              {"messages": json.dumps(tool_shape), "resolved": False}])
    two = rows_to_traces("R2E-Gym/R2EGym-SFT-Trajectories", [{"messages": text_shape}])
    assert len(one) == 1  # unresolved trajectories are left out
    (a,), _ = examples(one[0])
    (b,), _ = examples(two[0])
    names = lambda ex: [c["function"]["name"] for m in ex["messages"] if m["role"] == "assistant" for c in m["tool_calls"]]  # noqa: E731
    assert names(a) == ["read_file", "edit_file", "finish"]
    assert names(b) == ["bash", "edit_file", "finish"]
    bash = json.loads(b["messages"][2]["tool_calls"][0]["function"]["arguments"])["command"]
    assert bash == "cd . && python r.py"  # the dataset's checkout folder is the repo root
    assert "TypeError in r.py" in b["messages"][3]["content"]


def test_a_trace_started_outside_any_repo_works_in_the_repo_it_edited(tmp_path):
    from opentrain.traces.sft import work_root

    repo = tmp_path / "proj"
    (repo / "src").mkdir(parents=True)
    _git(repo, "init", "-q")
    (repo / "src" / "a.py").write_text("x = 1\n")
    t = trace(Step("user", "change x"),
              Step("agent", "", [tu("edit", "Edit", "ok", path=str(repo / "src" / "a.py"), old="1", new="2")]),
              cwd=str(tmp_path / "scratch"))
    assert Path(work_root(t)).resolve() == repo.resolve()
    (ex,), _ = examples(t)
    edit = json.loads(ex["messages"][2]["tool_calls"][0]["function"]["arguments"])
    assert edit["path"] == "src/a.py"


def test_harness_bookkeeping_calls_are_left_out_not_the_whole_request():
    t = trace(Step("user", "fix it"),
              Step("agent", "", [tu("other", "ToolSearch", "schemas", query="x")]),
              Step("agent", "", [tu("edit", "Edit", "ok", path="/repo/a.py", old="a", new="b")]))
    (ex,), skipped = examples(t)
    names = [c["function"]["name"] for m in ex["messages"] if m["role"] == "assistant" for c in m["tool_calls"]]
    assert names == ["edit_file", "finish"] and not skipped


def test_the_report_holds_counts_not_names(tmp_path, monkeypatch):
    from opentrain import report

    monkeypatch.setattr("opentrain.traces.readers.scan", lambda: [{"label": "Claude Code", "count": 3}])
    d = tmp_path / ".opentrain" / "traces"
    d.mkdir(parents=True)
    (d / "index.jsonl").write_text(json.dumps({"agent": "claude-code", "repo": "/work/acme-secret-repo", "edits": 2,
                                               "examples": 1, "moments": {"corrected": 1}}) + "\n")
    (d / "themes.json").write_text(json.dumps([{"name": "uses the acme billing API wrong", "chosen": True}]))
    text = report.render(report.collect(tmp_path))
    assert "acme" not in text
    assert '"changed_code": 1' in text and '"chosen": 1' in text
    assert "acme billing" in report.render(report.collect(tmp_path, names=True))


def test_a_card_lines_up_in_unicode_and_in_ascii():
    from opentrain.card import Row, Section, render

    for fancy in (True, False):
        out = render("t", ["sub"], [Section("a", [Row("x", "1", 0.5), Row("y", "2")]), Section("b", [Row("z", "3")])],
                     ["next: do it"], fancy=fancy)
        widths = {len(line) for line in out.splitlines()}
        assert widths == {72}, out


def test_one_shot_runs_are_not_work_and_shell_edits_are():
    one = trace(Step("user", "You are grading one answer."), Step("agent", "8/10"))
    assert one.one_shot and not one.changed_code
    shell = trace(Step("user", "bump the version"),
                  Step("agent", "", [ToolUse("s", "Bash", "shell", {"command": "sed -i '' 's/0.1/0.2/' pyproject.toml"},
                                             output="", exit_code=0)]))
    assert not shell.one_shot and shell.changed_code
    (ex,), _ = examples(shell)
    assert ex["meta"]["outcome"]["changed_code"] is True


def test_a_request_ends_well_only_without_pushback_stops_or_a_failing_final_check():
    from opentrain.traces.signals import outcomes

    fail = ToolUse(id="f", name="Bash", kind="shell", args={"command": "pytest -q"}, output="1 failed", exit_code=1)
    ok = ToolUse(id="o", name="Bash", kind="shell", args={"command": "pytest -q"}, output="ok", exit_code=0)
    edit = tu("edit", "Edit", "ok", path="/repo/a.py", old="a", new="b")
    t = trace(Step("user", "fix a"), Step("agent", "", [edit]), Step("agent", "", [fail]), Step("agent", "", [ok]),
              Step("user", "great, now fix b"), Step("agent", "", [edit]), Step("agent", "", [fail]),
              Step("user", "no, that broke c"), Step("agent", "", [edit]))
    a, b, c = outcomes(t)
    assert a["ended_well"] and a["checks"] == "passed"
    assert not b["ended_well"] and b["checks"] == "failed" and b["pushback"]
    assert c["ended_well"]


def test_a_judged_task_fails_an_empty_change_and_asks_the_judge_otherwise(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from opentrain import judge

    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "d@example.com")
    _git(repo, "config", "user.name", "D")
    (repo / "a.py").write_text("x = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "start")
    base = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    task = SimpleNamespace(base_commit=base, instruction="make x two", meta={}, test_command=judge.JUDGE)
    monkeypatch.setattr("opentrain.workspace.reference_patch", lambda t: "-x = 1\n+x = 2\n")
    seen = []
    monkeypatch.setattr("opentrain.core.complete.complete",
                        lambda model, msgs, **k: seen.append(msgs) or SimpleNamespace(text="PASS: sets x to two"))
    assert not judge.verdict(task, repo).passed and not seen  # nothing changed: no judge needed
    (repo / "a.py").write_text("x = 2\n")
    v = judge.verdict(task, repo)
    assert v.passed and "+x = 2" in seen[0][1]["content"]
    task.meta = {"checks": ["false"]}
    assert not judge.verdict(task, repo).passed  # a check the work must not break failed
