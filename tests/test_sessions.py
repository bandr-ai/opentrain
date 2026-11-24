from __future__ import annotations

from opentrain.sessions.claude_code import classify_test_result, clean_prompt, load_session


def test_time_split_model_tool_human(tmp_path, writer):
    w = (
        writer("/nowhere")
        .prompt(0, "fix the bug")
        .call(10, "c1", command="pytest -q")       # model 0..10
        .result(40, "c1", "1 passed")              # tool 10..40
        .say(55, "done")                           # model 40..55
        .prompt(355, "thanks, now the docs")       # human 55..355
        .say(365, "ok")                            # model 355..365
    )
    s = load_session(w.write(tmp_path / "p" / "s1.jsonl"))
    assert s.seconds("model") == 35
    assert s.seconds("tool") == 30
    assert s.seconds("human") == 300
    assert s.human_turns == 2
    assert s.instruction == "fix the bug"


def test_parallel_tools_count_once(tmp_path, writer):
    w = (
        writer("/nowhere")
        .prompt(0, "go")
        .call(5, "a", command="ls")
        .call(5, "b", command="ls")
        .result(20, "a")
        .result(30, "b")
        .say(32, "done")
    )
    s = load_session(w.write(tmp_path / "p" / "s.jsonl"))
    assert s.seconds("tool") == 25          # 5..30 wall clock, not 15 + 25
    assert s.seconds("model") == 7


def test_long_gap_is_idle_not_work(tmp_path, writer):
    w = writer("/nowhere").prompt(0, "go").call(5, "a", command="sleep").result(5 + 3 * 3600, "a").say(5 + 3 * 3600 + 4, "x")
    s = load_session(w.write(tmp_path / "p" / "s.jsonl"))
    assert s.seconds("idle") == 3 * 3600
    assert s.seconds("tool") == 0


def test_sidechain_and_meta_are_not_prompts(tmp_path, writer):
    w = (
        writer("/nowhere")
        .prompt(0, "<system-reminder>noise</system-reminder>", isMeta=True)
        .prompt(1, "<command-name>/clear</command-name>")
        .prompt(2, "real ask")
        .prompt(3, "subagent brief", isSidechain=True)
        .say(4, "ok")
    )
    s = load_session(w.write(tmp_path / "p" / "s.jsonl"))
    assert s.instruction == "real ask"
    assert s.human_turns == 1


def test_clean_prompt_keeps_pasted_content():
    text = "<pasted_content id='x'>stack trace</pasted_content>"
    assert clean_prompt(text) == text


def test_tests_and_edits_detected(tmp_path, writer):
    w = (
        writer("/nowhere")
        .prompt(0, "go")
        .call(1, "e", name="Edit", file_path="/r/app.py", old_string="a", new_string="b")
        .result(2, "e")
        .call(3, "t1", command="cd /r && uv run pytest -q")
        .result(9, "t1", "2 failed, 3 passed", is_error=True)
        .call(10, "t2", command="npm test")
        .result(15, "t2", "Tests: 5 passed")
        .call(16, "x", command="ls tests")
        .result(17, "x")
        .say(18, "fixed")
    )
    s = load_session(w.write(tmp_path / "p" / "s.jsonl"))
    assert s.files_edited == ["/r/app.py"]
    assert [t.outcome for t in s.test_runs] == ["fail", "pass"]


def test_classify_test_result_rules():
    assert classify_test_result("===== 12 passed, 0 failed =====", False) == "pass"
    assert classify_test_result("FAILED tests/test_a.py::test_x", False) == "fail"
    assert classify_test_result("", True) == "fail"
    assert classify_test_result("--- FAIL: TestX", False) == "fail"


def test_cost_state_is_read(tmp_path, writer):
    w = writer("/nowhere").prompt(0, "go").say(3, "hi")
    w.records.append({"type": "cost-state", "totalCostUSD": 1.5, "totalAPIDuration": 3000, "totalToolDuration": 0})
    s = load_session(w.write(tmp_path / "p" / "s.jsonl"))
    assert s.cost_usd == 1.5 and s.reported_api_seconds == 3.0


def test_empty_log_is_none(tmp_path):
    p = tmp_path / "p" / "s.jsonl"
    p.parent.mkdir()
    p.write_text('{"type":"mode","mode":"x"}\nnot json\n')
    assert load_session(p) is None
