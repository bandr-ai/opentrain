"""Readers for every coding agent's local logs, on small synthetic fixtures (no real session data)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from opentrain.traces import Trace
from opentrain.traces.readers import (
    SOURCES,
    aider,
    claude_code,
    cline,
    codex,
    copilot,
    cursor,
    gemini_cli,
    opencode,
    pi,
    read_all,
    scan,
)
from opentrain.traces.readers.codex import js_tool_calls
from opentrain.traces.scrub import REDACTED, scrub, scrub_trace

ENV = [
    "CLAUDE_CONFIG_DIR", "CODEX_HOME", "GEMINI_CLI_HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME", "OPENCODE_DIR",
    "PI_CODING_AGENT_DIR", "PI_CODING_AGENT_SESSION_DIR", "CLINE_DIR", "COPILOT_HOME", "CURSOR_CONFIG_DIR",
    "AIDER_CHAT_HISTORY_FILE",
]  # fmt: skip


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    for var in ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(h / ".config"))
    return h


def jsonl(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def tools_of(t: Trace, main_only: bool = True):
    return t.tool_uses(main_only=main_only)


# ---------------------------------------------------------------- Claude Code


def _cc(kind, ts, **kw):
    return {"type": kind, "timestamp": f"2026-11-01T10:00:{ts:02d}Z", "cwd": "/repo", "gitBranch": "main",
            "sessionId": "s1", **kw}  # fmt: skip


def test_claude_code_reader(home):
    log = home / ".claude" / "projects" / "-repo" / "s1.jsonl"
    edits = [{"old_string": "a", "new_string": "b"}, {"old_string": "c", "new_string": "d"}]
    jsonl(log, [
        {"type": "custom-title", "customTitle": "Fix it"},
        _cc("user", 0, message={"role": "user", "content": "fix the bug<system-reminder>be nice</system-reminder>"}),
        _cc("assistant", 1, message={"id": "m1", "model": "claude-x", "content": [{"type": "text", "text": "Looking."}]}),
        _cc("assistant", 2, message={"id": "m1", "model": "claude-x", "content": [
            {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "pytest -q"}},
            {"type": "tool_use", "id": "t2", "name": "Edit",
             "input": {"file_path": "/repo/a.py", "old_string": "x", "new_string": "y"}},
        ]}),
        _cc("user", 3, message={"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "Exit code 1\n1 failed", "is_error": True},
            {"type": "tool_result", "tool_use_id": "t2", "content": [{"type": "text", "text": "ok"}]},
        ]}),
        _cc("assistant", 4, message={"id": "m2", "model": "claude-x", "content": [
            {"type": "tool_use", "id": "t3", "name": "MultiEdit", "input": {"file_path": "/repo/b.py", "edits": edits}},
            {"type": "tool_use", "id": "t4", "name": "Write", "input": {"file_path": "/repo/c.py", "content": "z"}},
            {"type": "tool_use", "id": "t5", "name": "Read", "input": {"file_path": "/repo/d.py"}},
            {"type": "tool_use", "id": "t6", "name": "Grep", "input": {"pattern": "foo"}},
            {"type": "tool_use", "id": "t7", "name": "mcp__srv__do", "input": {}},
            {"type": "tool_use", "id": "t8", "name": "Task", "input": {"prompt": "look around"}},
            {"type": "tool_use", "id": "t9", "name": "WebFetch", "input": {"url": "https://x.dev"}},
        ]}),
        _cc("user", 5, message={"role": "user", "content": "<task-notification>done</task-notification>"}),
        _cc("user", 6, isSidechain=True, message={"role": "user", "content": "subagent prompt"}),
        _cc("assistant", 7, isSidechain=True, message={"id": "m3", "content": [
            {"type": "tool_use", "id": "s1", "name": "Bash", "input": {"command": "ls"}}]}),
        _cc("user", 8, message={"role": "user", "content": "thanks"}),
    ])  # fmt: skip
    jsonl(log.with_suffix("") / "subagents" / "agent-a.jsonl", [
        _cc("assistant", 9, isSidechain=True, message={"id": "m4", "content": [{"type": "text", "text": "sub says"}]}),
    ])  # fmt: skip

    assert claude_code.discover() == [str(log)]
    t = claude_code.read(str(log))
    assert t.trace_id == "s1" and t.cwd == "/repo" and t.git_branch == "main" and t.models == ["claude-x"]
    assert t.extra["title"] == "Fix it"
    assert t.prompts == ["fix the bug", "thanks"]
    assert any(s.source == "system" and "task-notification" in s.text for s in t.steps)
    # message m1's two records are one step
    first_agent = next(s for s in t.steps if s.source == "agent")
    assert first_agent.text == "Looking." and [x.name for x in first_agent.tools] == ["Bash", "Edit"]
    by = {x.id: x for x in tools_of(t)}
    assert by["t1"].kind == "shell" and by["t1"].args["command"] == "pytest -q"
    assert by["t1"].is_error and by["t1"].exit_code == 1 and "1 failed" in by["t1"].output
    assert by["t2"].args["path"] == "/repo/a.py" and (by["t2"].args["old"], by["t2"].args["new"]) == ("x", "y")
    assert by["t2"].output == "ok" and not by["t2"].is_error
    assert [(by[f"t3#{i}"].args["old"], by[f"t3#{i}"].args["new"]) for i in (0, 1)] == [("a", "b"), ("c", "d")]
    assert by["t4"].kind == "write" and by["t4"].args["content"] == "z"
    assert [by[k].kind for k in ("t5", "t6", "t7", "t8", "t9")] == ["read", "search", "other", "task", "web"]
    assert by["t5"].output is None
    assert t.edited_files == ["/repo/a.py", "/repo/b.py", "/repo/c.py"]
    side = [s for s in t.steps if s.sidechain]
    assert {s.source for s in side} == {"system", "agent"}
    assert any(s.text == "sub says" for s in side)
    assert "s1" not in {x.id for x in tools_of(t)} and "s1" in {x.id for x in tools_of(t, main_only=False)}


# ---------------------------------------------------------------- Codex


def test_js_tool_call_decoding():
    code = (
        'text(await tools.exec_command({cmd:"echo \\"hi\\" \\u00e9\\nls", workdir:\'/w\', max_output_tokens:100}));\n'
        "await tools.apply_patch(`*** Begin Patch\n*** Add File: x.py\n+print(1)\n*** End Patch`);\n"
        "tools.exec_command({cmd:`stats ${name} --json`});\n"
        "rs.map(cmd => tools.exec_command({cmd, yield_time_ms: 1}));\n"
        'tools.view_image({path: "/a/" + file});\n'
        'tools.update_plan({plan:[{step:"a", status:"done"}], "quoted key": null, ok: true})'
    )
    calls = js_tool_calls(code)
    assert [c[0] for c in calls] == ["exec_command", "apply_patch", "exec_command", "exec_command", "view_image",
                                     "update_plan"]  # fmt: skip
    assert calls[0][1] == {"cmd": 'echo "hi" é\nls', "workdir": "/w", "max_output_tokens": 100}
    assert calls[1][1].startswith("*** Begin Patch\n*** Add File: x.py")
    assert calls[2][1]["cmd"] == "stats ${name} --json"
    assert calls[3][1]["cmd"] == "cmd" and type(calls[3][1]["cmd"]).__name__ == "Raw"
    assert calls[4][1]["path"] == "/a/${file}"
    assert calls[5][1] == {"plan": [{"step": "a", "status": "done"}], "quoted key": None, "ok": True}


def _rollout(home: Path, name: str, records: list[dict]) -> Path:
    return jsonl(home / ".codex" / "sessions" / "2026" / "11" / "01" / name, records)


def _ri(ts: int, payload: dict) -> dict:
    return {"timestamp": f"2026-11-01T10:00:{ts:02d}Z", "type": "response_item", "payload": payload}


def test_codex_reader_current_format(home):
    exec_js = (
        'text(await tools.exec_command({cmd:"ls -la",workdir:"/repo"}));\n'
        'text(await tools.apply_patch("*** Begin Patch\\n*** Update File: /repo/a.py\\n@@\\n-x\\n+y\\n*** End Patch"));\n'
        "text(await tools.exec_command({cmd:'false'}));"
    )
    out_chunks = [
        {"type": "input_text", "text": "Script completed\nWall time 0.1 seconds\nOutput:\n"},
        {"type": "input_text", "text": json.dumps({"chunk_id": "a", "exit_code": 0, "output": "total 0\n"})},
        {"type": "input_text", "text": "Success. Updated the following files:\nM /repo/a.py\n"},
        {"type": "input_text", "text": json.dumps({"chunk_id": "b", "exit_code": 1, "output": ""})},
    ]
    path = _rollout(home, "rollout-2026-11-01T10-00-00-abc.jsonl", [
        {"timestamp": "2026-11-01T10:00:00Z", "type": "session_meta",
         "payload": {"id": "abc", "cwd": "/repo", "git": {"branch": "dev"}, "cli_version": "1.0"}},
        {"timestamp": "2026-11-01T10:00:00Z", "type": "turn_context", "payload": {"cwd": "/repo", "model": "gpt-x"}},
        _ri(1, {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "You are Codex"}]}),
        _ri(1, {"type": "message", "role": "user",
                "content": [{"type": "input_text", "text": "<environment_context>\n<cwd>/repo</cwd>\n</environment_context>"}]}),
        {"timestamp": "2026-11-01T10:00:02Z", "type": "event_msg", "payload": {"type": "item_completed",
         "item": {"type": "UserMessage", "content": [{"type": "text", "text": "list files"}]}}},
        _ri(2, {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "list files"}]}),
        {"timestamp": "2026-11-01T10:00:02Z", "type": "event_msg", "payload": {"type": "user_message", "message": "list files"}},
        _ri(3, {"type": "reasoning", "summary": []}),
        _ri(3, {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "On it."}]}),
        _ri(4, {"type": "function_call", "name": "exec_command", "call_id": "c1",
                "arguments": json.dumps({"cmd": "make test"})}),
        _ri(5, {"type": "function_call_output", "call_id": "c1",
                "output": "Chunk ID: 1\nWall time: 0.1 seconds\nProcess exited with code 2\nOutput:\nboom"}),
        _ri(6, {"type": "function_call", "name": "shell", "call_id": "c2",
                "arguments": json.dumps({"command": ["bash", "-lc", "echo hi"]})}),
        _ri(7, {"type": "function_call_output", "call_id": "c2",
                "output": json.dumps({"output": "hi\n", "metadata": {"exit_code": 0}})}),
        _ri(8, {"type": "custom_tool_call", "name": "exec", "call_id": "c3", "input": exec_js}),
        _ri(9, {"type": "custom_tool_call_output", "call_id": "c3", "output": out_chunks}),
        _ri(10, {"type": "function_call", "name": "link_pull_request", "namespace": "mcp__gh", "call_id": "c4",
                 "arguments": "{}"}),
        _ri(11, {"type": "custom_tool_call", "name": "exec", "call_id": "c5",
                 "input": "for (const cmd of cmds) await tools.exec_command({cmd});"}),
        _ri(12, {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "thanks"}]}),
    ])  # fmt: skip
    assert codex.discover() == [str(path)]
    t = codex.read(str(path))
    assert t.trace_id == "abc" and t.cwd == "/repo" and t.git_branch == "dev" and t.models == ["gpt-x"]
    assert t.prompts == ["list files", "thanks"]
    assert [s.source for s in t.steps][:2] == ["system", "user"]
    assert not any("You are Codex" in s.text for s in t.steps)
    tools = tools_of(t)
    by = {x.id: x for x in tools}
    assert by["c1"].kind == "shell" and by["c1"].args["command"] == "make test"
    assert by["c1"].exit_code == 2 and by["c1"].is_error
    assert by["c2"].args["command"] == "echo hi" and by["c2"].output == "hi\n" and by["c2"].exit_code == 0
    assert (
        by["c3#0"].kind == "shell" and by["c3#0"].args["command"] == "ls -la" and by["c3#0"].args["workdir"] == "/repo"
    )
    assert by["c3#0"].output == "total 0\n" and by["c3#0"].exit_code == 0
    assert by["c3#1"].kind == "patch" and "*** Update File: /repo/a.py" in by["c3#1"].args["patch"]
    assert by["c3#2"].exit_code == 1 and by["c3#2"].is_error
    assert by["c4"].name == "mcp__gh__link_pull_request" and by["c4"].kind == "other"
    assert by["c5#0"].kind == "other" and "tools.exec_command({cmd})" in by["c5#0"].args["code"]
    assert t.edited_files == ["/repo/a.py"]


def test_codex_reader_legacy_format(home):
    path = _rollout(home, "rollout-old.jsonl", [
        {"id": "old1", "timestamp": "2025-05-01T10:00:00Z", "instructions": None},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]},
        {"record_type": "state"},
        {"type": "local_shell_call", "call_id": "l1", "status": "completed",
         "action": {"type": "exec", "command": ["bash", "-lc", "pwd"], "working_directory": "/w"}},
        {"type": "local_shell_call_output", "call_id": "l1", "output": json.dumps({"output": "/w", "metadata": {"exit_code": 0}})},
        {"type": "function_call", "name": "apply_patch", "call_id": "p1",
         "arguments": json.dumps({"input": "*** Begin Patch\n*** Add File: n.txt\n+x\n*** End Patch"})},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]},
    ])  # fmt: skip
    t = codex.read(str(path))
    assert t.trace_id == "old1" and t.prompts == ["hello"]
    by = {x.id: x for x in tools_of(t)}
    assert by["l1"].kind == "shell" and by["l1"].args["command"] == "pwd" and by["l1"].output == "/w"
    assert by["p1"].kind == "patch" and t.edited_files == ["n.txt"]
    assert t.steps[-1].text == "done"


def test_codex_event_only_prompts(home):
    path = _rollout(home, "rollout-ev.jsonl", [
        {"timestamp": "2026-11-01T10:00:00Z", "type": "session_meta", "payload": {"id": "ev"}},
        {"timestamp": "2026-11-01T10:00:01Z", "type": "event_msg", "payload": {"type": "user_message", "message": "hi"}},
        {"timestamp": "2026-11-01T10:00:01Z", "type": "event_msg", "payload": {"type": "user_message", "message": "hi"}},
        {"timestamp": "2026-11-01T10:00:02Z", "type": "event_msg", "payload": {"type": "agent_message", "message": "yo"}},
    ])  # fmt: skip
    t = codex.read(str(path))
    assert t.prompts == ["hi"] and t.steps[-1].text == "yo"


# ---------------------------------------------------------------- opencode


def test_opencode_sqlite_and_storage(home):
    root = home / ".local" / "share" / "opencode"
    root.mkdir(parents=True)
    db = root / "opencode.db"
    con = sqlite3.connect(db)
    con.executescript("""
        create table project (id text primary key, worktree text);
        create table session (id text primary key, project_id text, title text, time_created int, time_updated int);
        create table message (id text primary key, session_id text, time_created int, time_updated int, data text);
        create table part (id text primary key, message_id text, session_id text, time_created int, data text);
    """)
    con.execute("insert into project values ('p1', '/proj')")
    con.execute("insert into session values ('ses1', 'p1', 'Title', 1000, 2000)")
    msgs = [
        ("m1", 1_762_000_000_000, {"role": "user", "time": {"created": 1_762_000_000_000}}),
        ("m2", 1_762_000_001_000, {"role": "assistant", "modelID": "glm-x", "time": {"created": 1_762_000_001_000}}),
    ]
    for mid, tc, data in msgs:
        con.execute("insert into message values (?, 'ses1', ?, ?, ?)", (mid, tc, tc, json.dumps(data)))
    parts = [
        ("a", "m1", {"type": "text", "text": "run tests"}),
        ("b", "m1", {"type": "text", "text": "injected", "synthetic": True}),
        ("c", "m2", {"type": "reasoning", "text": "hmm"}),
        ("d", "m2", {"type": "text", "text": "Running."}),
        (
            "e",
            "m2",
            {
                "type": "tool",
                "tool": "bash",
                "callID": "call1",
                "state": {
                    "status": "completed",
                    "input": {"command": "pytest"},
                    "output": "1 failed",
                    "metadata": {"exit": 1},
                    "time": {"start": 1_762_000_001_500, "end": 1_762_000_002_000},
                },
            },
        ),
        (
            "f",
            "m2",
            {
                "type": "tool",
                "tool": "edit",
                "callID": "call2",
                "state": {
                    "status": "completed",
                    "input": {"filePath": "/proj/x.py", "oldString": "a", "newString": "b"},
                    "output": "",
                },
            },
        ),
    ]
    for i, (pid, mid, data) in enumerate(parts):
        con.execute("insert into part values (?, ?, 'ses1', ?, ?)", (pid, mid, i, json.dumps(data)))
    con.commit()
    con.close()

    st = root / "storage"
    (st / "session" / "p2").mkdir(parents=True)
    (st / "session" / "p2" / "ses2.json").write_text(json.dumps({"id": "ses2", "directory": "/old", "title": "t"}))
    (st / "message" / "ses2").mkdir(parents=True)
    (st / "message" / "ses2" / "msg1.json").write_text(
        json.dumps({"id": "msg1", "role": "user", "time": {"created": 1}})
    )
    (st / "part" / "msg1").mkdir(parents=True)
    (st / "part" / "msg1" / "prt1.json").write_text(json.dumps({"id": "prt1", "type": "text", "text": "legacy hi"}))

    locs = opencode.discover()
    assert locs == [f"{db}#ses1", str(st / "session" / "p2" / "ses2.json")]
    t = opencode.read(locs[0])
    assert t.cwd == "/proj" and t.extra["title"] == "Title" and t.models == ["glm-x"]
    assert t.prompts == ["run tests"] and [s.source for s in t.steps] == ["user", "system", "agent"]
    bash, edit = tools_of(t)
    assert bash.kind == "shell" and bash.args["command"] == "pytest" and bash.exit_code == 1 and bash.is_error
    assert edit.kind == "edit" and (edit.args["path"], edit.args["old"], edit.args["new"]) == ("/proj/x.py", "a", "b")
    assert t.steps[-1].text == "Running."
    legacy = opencode.read(locs[1])
    assert legacy.cwd == "/old" and legacy.prompts == ["legacy hi"]


# ---------------------------------------------------------------- Gemini CLI


def test_gemini_cli_reader(home):
    chats = home / ".gemini" / "tmp" / "abc123" / "chats"
    chats.mkdir(parents=True)
    doc = {
        "sessionId": "g1", "projectHash": "abc123", "startTime": "2026-11-01T10:00:00Z",
        "messages": [
            {"id": "1", "type": "user", "content": "add a test", "timestamp": "2026-11-01T10:00:01Z"},
            {"id": "2", "type": "gemini", "content": "Sure.", "model": "gemini-x", "timestamp": "2026-11-01T10:00:02Z",
             "toolCalls": [
                 {"id": "tc1", "name": "run_shell_command", "args": {"command": "ls"}, "status": "success",
                  "result": [{"functionResponse": {"id": "tc1", "name": "run_shell_command",
                                                   "response": {"output": "a.py\nExit Code: 0"}}}]},
                 {"id": "tc2", "name": "replace", "args": {"file_path": "/p/a.py", "old_string": "1", "new_string": "2"},
                  "status": "error", "result": []},
                 {"id": "tc3", "name": "write_file", "args": {"file_path": "/p/t.py", "content": "x"}, "status": "success"},
             ]},
            {"id": "3", "type": "info", "content": "Request cancelled."},
        ],
    }  # fmt: skip
    (chats / "session-1.json").write_text(json.dumps(doc))
    jsonl(chats / "session-2.jsonl", [
        {"sessionId": "g2", "projectHash": "abc123", "startTime": "2026-11-02T10:00:00Z", "kind": "main"},
        {"id": "u1", "type": "user", "content": [{"text": "draft"}]},
        {"id": "u1", "type": "user", "content": [{"text": "final words"}]},
        {"$set": {"summary": "S"}},
    ])  # fmt: skip
    (chats.parent.parent / "only-logs").mkdir()
    (chats.parent.parent / "only-logs" / "logs.json").write_text(
        json.dumps([{"sessionId": "g3", "messageId": 0, "type": "user", "message": "old prompt"}])
    )
    locs = gemini_cli.discover()
    assert len(locs) == 3
    t = gemini_cli.read(str(chats / "session-1.json"))
    assert t.trace_id == "g1" and t.prompts == ["add a test"] and t.models == ["gemini-x"]
    sh, rep, wr = tools_of(t)
    assert sh.kind == "shell" and sh.args["command"] == "ls" and sh.output.startswith("a.py") and sh.exit_code == 0
    assert rep.kind == "edit" and rep.is_error and rep.args["old"] == "1" and rep.args["new"] == "2"
    assert wr.kind == "write" and wr.args["content"] == "x"
    assert t.edited_files == ["/p/a.py", "/p/t.py"]
    assert t.steps[-1].source == "system"
    t2 = gemini_cli.read(str(chats / "session-2.jsonl"))
    assert t2.trace_id == "g2" and t2.prompts == ["final words"] and t2.extra["title"] == "S"
    assert gemini_cli.read(str(chats.parent.parent / "only-logs" / "logs.json")).prompts == ["old prompt"]


# ---------------------------------------------------------------- Cursor


def test_cursor_cli_transcript_and_ide(home):
    tr = home / ".cursor" / "projects" / "Users-me-repo" / "agent-transcripts" / "t1.jsonl"
    jsonl(tr, [
        {"role": "user", "message": {"content": [{"type": "text",
         "text": "<timestamp>Mon</timestamp>\n<user_query>\nrename foo\n</user_query>"}]}},
        {"role": "assistant", "message": {"content": [
            {"type": "text", "text": "Renaming."},
            {"type": "tool_use", "id": "x1", "name": "Edit",
             "input": {"file_path": "/r/a.py", "old_string": "foo", "new_string": "bar"}},
            {"type": "tool_use", "id": "x2", "name": "Shell", "input": {"command": "ls"}},
        ]}},
        {"role": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "x2", "content": "a.py"}]}},
    ])  # fmt: skip
    txt = tr.parent / "t2.txt"
    txt.write_text(
        "user:\n<user_query>\nshow files\n</user_query>\n\nassistant:\nChecking.\n"
        "[Tool call] Shell\n  command=ls -a\n[Tool result]\n  .git\n"
    )

    user = home / ".config" / "Cursor" / "User"
    (user / "globalStorage").mkdir(parents=True)
    db = user / "globalStorage" / "state.vscdb"
    con = sqlite3.connect(db)
    con.execute("create table cursorDiskKV (key text unique on conflict replace, value blob)")
    comp = {"composerId": "c1", "name": "Chat", "createdAt": 1_762_000_000_000,
            "fullConversationHeadersOnly": [{"bubbleId": "b1", "type": 1}, {"bubbleId": "b2", "type": 2}],
            "workspaceIdentifier": {"uri": {"fsPath": "/ws"}}}  # fmt: skip
    bubbles = {
        "b1": {"type": 1, "text": "make it faster", "createdAt": "2026-11-01T10:00:00Z"},
        "b2": {
            "type": 2,
            "text": "Done.",
            "createdAt": "2026-11-01T10:00:05Z",
            "toolFormerData": {
                "toolCallId": "tc",
                "name": "run_terminal_cmd",
                "rawArgs": json.dumps({"command": "make"}),
                "result": json.dumps({"output": "ok", "exitCode": 0}),
                "status": "completed",
            },
        },
    }
    con.execute("insert into cursorDiskKV values (?, ?)", ("composerData:c1", json.dumps(comp)))
    for bid, b in bubbles.items():
        con.execute("insert into cursorDiskKV values (?, ?)", (f"bubbleId:c1:{bid}", json.dumps(b)))
    con.commit()
    con.close()

    locs = cursor.discover()
    assert str(tr) in locs and str(txt) in locs and f"{db}#c1" in locs
    t = cursor.read(str(tr))
    assert t.prompts == ["rename foo"] and t.edited_files == ["/r/a.py"]
    e, s = tools_of(t)
    assert (e.kind, e.args["old"], e.args["new"]) == ("edit", "foo", "bar")
    assert s.kind == "shell" and s.output == "a.py"
    t2 = cursor.read(str(txt))
    assert t2.prompts == ["show files"]
    (sh,) = tools_of(t2)
    assert sh.kind == "shell" and sh.args["command"] == "ls -a" and sh.output == ".git"
    ide = cursor.read(f"{db}#c1")
    assert ide.cwd == "/ws" and ide.extra["title"] == "Chat" and ide.prompts == ["make it faster"]
    (run,) = tools_of(ide)
    assert run.kind == "shell" and run.args["command"] == "make" and run.output == "ok" and run.exit_code == 0


# ---------------------------------------------------------------- Cline / Roo


def test_cline_xml_history_and_roo_native(home):
    gs = home / ".config" / "Code" / "User" / "globalStorage"
    task = gs / "saoudrizwan.claude-dev" / "tasks" / "111"
    task.mkdir(parents=True)
    history = [
        {"role": "user", "content": [
            {"type": "text", "text": "<task>\nfix the import\n</task>"},
            {"type": "text", "text": "<environment_details>\n# Current Working Directory (/proj) Files\na.py\n</environment_details>"},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text":
            "<thinking>Look first.</thinking>\n<execute_command>\n<command>cat a.py</command>\n"
            "<requires_approval>false</requires_approval>\n</execute_command>"}]},
        {"role": "user", "content": [
            {"type": "text", "text": "[execute_command for 'cat a.py'] Result:\nimport sys"},
            {"type": "text", "text": "<environment_details>\nstuff\n</environment_details>"},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text":
            "<replace_in_file>\n<path>a.py</path>\n<diff>\n------- SEARCH\nimport sys\n=======\nimport os\n+++++++ REPLACE\n</diff>\n</replace_in_file>"}]},
        {"role": "user", "content": [{"type": "text", "text":
            "[replace_in_file for 'a.py'] Result:\nThe user denied this operation.\n<feedback>\nuse pathlib\n</feedback>"}]},
    ]  # fmt: skip
    (task / "api_conversation_history.json").write_text(json.dumps(history))
    (task / "ui_messages.json").write_text(json.dumps([
        {"ts": 1_762_000_000_000, "type": "say", "say": "text", "text": "fix the import"},
        {"ts": 1_762_000_000_500, "type": "say", "say": "api_req_started", "text": "{}"},
        {"ts": 1_762_000_009_000, "type": "say", "say": "completion_result", "text": "ok"},
    ]))  # fmt: skip
    roo = gs / "rooveterinaryinc.roo-cline" / "tasks" / "222"
    roo.mkdir(parents=True)
    (roo / "api_conversation_history.json").write_text(json.dumps([
        {"role": "user", "content": [{"type": "text", "text": "<task>write hello</task>"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "w1", "name": "write_to_file",
                                           "input": {"path": "h.txt", "content": "hello"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "w1", "content": "written"}]},
    ]))  # fmt: skip
    ui_only = gs / "kilocode.kilo-code" / "tasks" / "333"
    ui_only.mkdir(parents=True)
    (ui_only / "ui_messages.json").write_text(json.dumps([
        {"ts": 1, "type": "say", "say": "text", "text": "list things"},
        {"ts": 2, "type": "ask", "ask": "command", "text": "ls"},
        {"ts": 3, "type": "say", "say": "command_output", "text": "a\nb"},
        {"ts": 4, "type": "ask", "ask": "tool", "text": json.dumps({"tool": "newFileCreated", "path": "n.py", "content": "x"})},
    ]))  # fmt: skip

    locs = cline.discover()
    assert set(locs) == {str(task), str(roo), str(ui_only)}
    t = cline.read(str(task))
    assert t.cwd == "/proj" and t.extra["app"] == "cline"
    assert t.prompts == ["fix the import", "use pathlib"]
    cat, rep = tools_of(t)
    assert cat.kind == "shell" and cat.args["command"] == "cat a.py" and cat.output == "import sys"
    assert (
        rep.kind == "edit"
        and rep.args["path"] == "a.py"
        and (rep.args["old"], rep.args["new"]) == ("import sys", "import os")
    )
    assert rep.is_error is False and "denied" in rep.output
    assert t.steps[1].text == "Look first."
    r = cline.read(str(roo))
    assert r.extra["app"] == "roo-code" and r.prompts == ["write hello"] and r.edited_files == ["h.txt"]
    assert tools_of(r)[0].output == "written"
    k = cline.read(str(ui_only))
    assert k.prompts == ["list things"]
    sh, new = tools_of(k)
    assert sh.args["command"] == "ls" and sh.output == "a\nb" and new.kind == "write" and new.args["path"] == "n.py"


# ---------------------------------------------------------------- aider


def test_aider_history(home):
    repo = home / "projects" / "demo"
    repo.mkdir(parents=True)
    (repo / ".aider.chat.history.md").write_text(
        "\n# aider chat started at 2026-11-01 10:00:00\n\n> Aider v0.80.0\n> Main model: gpt-x with diff edit format\n\n"
        "#### make greet say hi\n\nHere is the change:\n\ngreet.py\n```python\n<<<<<<< SEARCH\nprint('hello')\n=======\n"
        "print('hi')\n>>>>>>> REPLACE\n```\n\n> Applied edit to greet.py\n\n"
        "# aider chat started at 2026-11-02 09:00:00\n\n#### second session\n\nok\n"
    )
    locs = aider.discover()
    assert locs == [f"{repo / '.aider.chat.history.md'}#0", f"{repo / '.aider.chat.history.md'}#1"]
    t = aider.read(locs[0])
    assert t.cwd == str(repo) and t.prompts == ["make greet say hi"] and t.models == ["gpt-x"]
    (e,) = tools_of(t)
    assert e.kind == "edit" and e.args["path"] == "greet.py" and e.args["old"] == "print('hello')"
    assert e.args["new"] == "print('hi')" and e.output == "Applied edit to greet.py"
    assert aider.read(locs[1]).prompts == ["second session"]


# ---------------------------------------------------------------- pi


def test_pi_reader(home):
    path = jsonl(home / ".pi" / "agent" / "sessions" / "--repo--" / "2026-11-01_s.jsonl", [
        {"type": "session", "version": 3, "id": "pi1", "timestamp": "2026-11-01T10:00:00Z", "cwd": "/repo"},
        {"type": "model_change", "id": "a", "provider": "x", "modelId": "m-1"},
        {"type": "message", "id": "b", "timestamp": "2026-11-01T10:00:01Z",
         "message": {"role": "user", "content": [{"type": "text", "text": "edit it"}]}},
        {"type": "message", "id": "c", "timestamp": "2026-11-01T10:00:02Z", "message": {
            "role": "assistant", "model": "m-1", "content": [
                {"type": "thinking", "thinking": "hm"}, {"type": "text", "text": "Editing."},
                {"type": "toolCall", "id": "k1", "name": "edit", "arguments": {"path": "a.ts", "oldText": "1", "newText": "2"}},
                {"type": "toolCall", "id": "k2", "name": "bash", "arguments": {"command": "npm test"}},
            ]}},
        {"type": "message", "id": "d", "message": {"role": "toolResult", "toolCallId": "k1", "toolName": "edit",
                                                    "content": [{"type": "text", "text": "ok"}], "isError": False}},
        {"type": "message", "id": "e", "message": {"role": "toolResult", "toolCallId": "k2", "toolName": "bash",
                                                    "content": [{"type": "text", "text": "fail\nexit code 1"}], "isError": True}},
    ])  # fmt: skip
    assert pi.discover() == [str(path)]
    t = pi.read(str(path))
    assert t.trace_id == "pi1" and t.cwd == "/repo" and t.prompts == ["edit it"] and t.models == ["m-1"]
    e, b = tools_of(t)
    assert (e.kind, e.args["path"], e.args["old"], e.args["new"], e.output) == ("edit", "a.ts", "1", "2", "ok")
    assert b.kind == "shell" and b.is_error and b.exit_code == 1


# ---------------------------------------------------------------- Copilot


def test_copilot_cli_and_vscode(home):
    ev = jsonl(home / ".copilot" / "session-state" / "s-1" / "events.jsonl", [
        {"type": "session.start", "timestamp": "2026-11-01T10:00:00Z",
         "data": {"sessionId": "cp1", "context": {"cwd": "/c", "branch": "b"}}},
        {"type": "user.message", "timestamp": "2026-11-01T10:00:01Z", "data": {"content": "what is here"}},
        {"type": "assistant.message", "timestamp": "2026-11-01T10:00:02Z", "data": {"content": "", "model": "gpt-c",
         "toolRequests": [{"toolCallId": "q1", "name": "bash", "arguments": json.dumps({"command": "ls"})}]}},
        {"type": "tool.execution_complete", "timestamp": "2026-11-01T10:00:03Z",
         "data": {"toolCallId": "q1", "success": True, "result": {"content": "x.txt"}}},
    ])  # fmt: skip
    ws = home / ".config" / "Code" / "User" / "workspaceStorage" / "h1"
    (ws / "chatSessions").mkdir(parents=True)
    (ws / "workspace.json").write_text(json.dumps({"folder": "file:///proj"}))
    jsonl(ws / "chatSessions" / "v1.jsonl", [
        {"kind": 0, "v": {"sessionId": "vs1", "creationDate": 1_762_000_000_000, "requests": []}},
        {"kind": 2, "k": ["requests"], "v": [{"requestId": "r1", "timestamp": 1_762_000_000_000,
                                              "message": {"text": "rename x"}, "response": []}]},
        {"kind": 2, "k": ["requests", 0, "response"], "v": [
            {"value": "Renaming "}, {"value": "now."},
            {"kind": "toolInvocationSerialized", "toolId": "copilot_replaceString", "toolCallId": "e1",
             "pastTenseMessage": {"value": "Edited a.py"}},
            {"kind": "textEditGroup", "uri": {"fsPath": "/proj/a.py"}, "edits": [[{"text": "y = 1"}]]},
            {"kind": "toolInvocationSerialized", "toolId": "copilot_runInTerminal", "toolCallId": "e2",
             "toolSpecificData": {"kind": "terminal", "commandLine": {"original": "pytest"}}},
        ]},
        {"kind": 1, "k": ["customTitle"], "v": "Rename"},
    ])  # fmt: skip
    locs = copilot.discover()
    assert set(locs) == {str(ev), str(ws / "chatSessions" / "v1.jsonl")}
    t = copilot.read(str(ev))
    assert t.trace_id == "cp1" and t.cwd == "/c" and t.git_branch == "b" and t.prompts == ["what is here"]
    (b,) = tools_of(t)
    assert b.kind == "shell" and b.args["command"] == "ls" and b.output == "x.txt"
    v = copilot.read(str(ws / "chatSessions" / "v1.jsonl"))
    assert v.cwd == "/proj" and v.extra["title"] == "Rename" and v.prompts == ["rename x"]
    assert v.steps[-1].text == "Renaming now."
    e, sh = tools_of(v)
    assert e.kind == "edit" and e.args["path"] == "/proj/a.py" and sh.kind == "shell" and sh.args["command"] == "pytest"
    assert v.edited_files == ["/proj/a.py"]


# ---------------------------------------------------------------- registry


def test_scan_reports_absent_sources_as_zero(home):
    rows = scan()
    assert [r["name"] for r in rows] == [s.name for s in SOURCES]
    assert all(r["count"] == 0 and r["error"] is None for r in rows)
    assert read_all() == ([], [])


def test_read_all_reports_unreadable_and_filters_since(home):
    proj = home / ".claude" / "projects" / "-p"
    jsonl(proj / "good.jsonl", [
        {"type": "user", "timestamp": "2026-11-01T10:00:00Z", "sessionId": "g", "message": {"role": "user", "content": "hi"}},
    ])  # fmt: skip
    jsonl(proj / "empty.jsonl", [{"type": "mode", "mode": "x"}])
    chats = home / ".gemini" / "tmp" / "h" / "chats"
    chats.mkdir(parents=True)
    (chats / "session-bad.json").write_text("{not json")
    traces, bad = read_all()
    assert [t.trace_id for t in traces] == ["g"]
    reasons = dict(bad)
    assert reasons[str(proj / "empty.jsonl")] == "no conversation in log"
    assert reasons[str(chats / "session-bad.json")].startswith("invalid JSON")
    traces, _ = read_all(["claude-code"], since=datetime(2030, 1, 1, tzinfo=timezone.utc))
    assert traces == []
    with pytest.raises(KeyError):
        read_all(["nope"])


def test_registry_survives_reader_crash(home, monkeypatch):
    jsonl(home / ".pi" / "agent" / "sessions" / "x" / "s.jsonl", [{"type": "session", "id": "1"}])

    def boom(loc):
        raise RuntimeError("bad shape")

    monkeypatch.setattr(pi.SOURCE, "read", boom)
    traces, bad = read_all(["pi"])
    assert traces == [] and bad[0][1] == "RuntimeError: bad shape"


# ---------------------------------------------------------------- scrub


@pytest.mark.parametrize(
    "secret",
    [
        "sk-ant-api03-" + "a" * 40,
        "sk-proj-" + "b" * 40,
        "ghp_" + "c" * 36,
        "github_pat_" + "d" * 40,
        "hf_" + "e" * 34,
        "AKIA" + "ABCDEFGHIJKLMNOP",
        "xoxb-1234567890-abcdefghij",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
    ],
)
def test_scrub_token_formats(secret):
    out = scrub(f"use {secret} now")
    assert secret not in out and REDACTED in out and out.startswith("use ") and out.endswith(" now")


def test_scrub_assignments_and_keys():
    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----"
    text = (
        'OPENAI_API_KEY=abc123def456\nexport GITHUB_TOKEN="tok_9876543210"\n'
        '{"api_key": "plainsecret99", "password": "hunter22", "name": "ok"}\n'
        f"{pem}\ncurl https://user:pa55word@host/x\nAuthorization: Bearer abcdefghijklmnop1234\n"
        "max_tokens=4096 tokenizer=Qwen/Qwen3-8B author=someone API_KEY=$OPENAI_API_KEY"
    )
    out = scrub(text)
    for s in (
        "abc123def456",
        "tok_9876543210",
        "plainsecret99",
        "hunter22",
        "MIIEow",
        "pa55word",
        "abcdefghijklmnop1234",
    ):
        assert s not in out
    assert '"name": "ok"' in out
    assert "max_tokens=4096" in out and "tokenizer=Qwen/Qwen3-8B" in out and "author=someone" in out
    assert "API_KEY=$OPENAI_API_KEY" in out


def test_scrub_trace_covers_text_args_outputs(home):
    from opentrain.traces import Step, ToolUse

    key = "sk-ant-" + "z" * 30
    t = Trace("t", "x", "p", None, None, None, steps=[
        Step("user", f"key {key}"),
        Step("agent", "ok", tools=[ToolUse("1", "Bash", "shell", {"command": f"export K={key}", "env": [key]},
                                           output=f"echo {key}")]),
    ], extra={"note": key})  # fmt: skip
    s = scrub_trace(t)
    blob = json.dumps([s.steps[0].text, s.steps[1].tools[0].args, s.steps[1].tools[0].output, s.extra])
    assert key not in blob and REDACTED in blob
    assert key in t.steps[0].text  # the original is untouched
