import threading
import time

import pytest

from opentrain.core.providers.scripted import ScriptedProvider
from opentrain.core.tool import Tool, ToolError, schema


def _tools(live, proj):
    """Register test tools on the project's registry (opening the project if needed)."""
    p = live.hub.project(proj)
    calls = []

    def echo(ctx, text):
        calls.append(text)
        return f"echo: {text}"

    def boom(ctx):
        raise ToolError("nope")

    def slow(ctx, seconds: float):
        time.sleep(seconds)
        return "slept"

    p.services["tools"].add(
        Tool("echo", "Echo text.", schema({"text": {"type": "string"}}, ["text"]), echo),
        Tool("boom", "Always fails.", schema(), boom),
        Tool("slow", "Sleeps.", schema({"seconds": {"type": "number"}}, ["seconds"]), slow),
    )
    return p, calls


def _run(c, sid, text, **kw):
    c.post(f"/api/session/{sid}/prompt", {"text": text, **kw})
    return c.post(f"/api/session/{sid}/wait", {"timeout": 20})


def test_a_turn_with_tool_calls(live):
    c = live.client(live.proj)
    p, calls = _tools(live, live.proj)
    ScriptedProvider.register("t1", [
        ["Let me echo.", {"tool": "echo", "input": {"text": "hi"}}, {"tool": "boom", "input": {}}],
        ["All done."],
    ])
    s = c.post("/api/session", {"model": "scripted/t1"})
    out = _run(c, s["id"], "please echo hi")
    assert out["status"] == "idle" and out["title"] == "please echo hi"
    msgs = c.get(f"/api/session/{s['id']}/message")
    assert [m["role"] for m in msgs] == ["user", "assistant", "assistant"]
    parts = msgs[1]["parts"]
    assert parts[0]["type"] == "text" and parts[0]["text"].strip() == "Let me echo."
    tools = {pt["tool"]["name"]: pt["tool"] for pt in parts if pt["type"] == "tool"}
    assert tools["echo"]["state"] == "completed" and tools["echo"]["output"] == "echo: hi"
    assert tools["boom"]["state"] == "error" and tools["boom"]["error"] == "nope"
    assert msgs[2]["parts"][0]["text"].strip() == "All done."
    assert calls == ["hi"]
    # The second request carried the tool results back, in neutral form.
    req = ScriptedProvider.named("t1").requests[1]
    assert req.messages[-1]["role"] == "tool"
    assert {r["output"] for r in req.messages[-1]["content"]} == {"echo: hi", "Error: nope"}
    assert "echo" in {t.name for t in req.tools}
    assert c.get("/api/event/log")["verified"]


def test_bad_arguments_are_returned_to_the_model(live):
    c = live.client(live.proj)
    _tools(live, live.proj)
    ScriptedProvider.register("t2", [[{"tool": "echo", "input": {"wrong": 1}}], ["ok"]])
    s = c.post("/api/session", {"model": "scripted/t2"})
    _run(c, s["id"], "go")
    t = c.get(f"/api/session/{s['id']}/message")[1]["parts"][0]["tool"]
    assert t["state"] == "error" and "missing required argument(s): text" in t["error"]


def test_steering_arrives_between_steps(live):
    c = live.client(live.proj)
    _tools(live, live.proj)
    prov = ScriptedProvider.register("t3", [[{"tool": "slow", "input": {"seconds": 0.6}}], ["noted"]])
    s = c.post("/api/session", {"model": "scripted/t3"})
    c.post(f"/api/session/{s['id']}/prompt", {"text": "start"})
    time.sleep(0.2)
    c.post(f"/api/session/{s['id']}/prompt", {"text": "also do X"})
    c.post(f"/api/session/{s['id']}/wait", {"timeout": 20})
    second = prov.requests[1].messages
    assert second[-1] == {"role": "user", "content": [{"type": "text", "text": "also do X"}]}


def test_interrupt_stops_a_running_tool(live):
    c = live.client(live.proj)
    _tools(live, live.proj)
    ScriptedProvider.register("t4", [[{"tool": "slow", "input": {"seconds": 3}}]])
    s = c.post("/api/session", {"model": "scripted/t4"})
    c.post(f"/api/session/{s['id']}/prompt", {"text": "go"})
    time.sleep(0.3)
    assert c.post(f"/api/session/{s['id']}/interrupt")["interrupted"] is True
    msgs = c.get(f"/api/session/{s['id']}/message")
    assert msgs[1]["finish"] == "interrupted"
    assert msgs[1]["parts"][0]["tool"]["state"] == "error"
    assert c.get(f"/api/session/{s['id']}")["status"] == "idle"


def test_doom_loop_is_stopped(live):
    c = live.client(live.proj)
    _tools(live, live.proj)
    same = [{"tool": "echo", "input": {"text": "x"}}]
    ScriptedProvider.register("t5", [same] * 10)
    s = c.post("/api/session", {"model": "scripted/t5"})
    _run(c, s["id"], "go")
    msgs = c.get(f"/api/session/{s['id']}/message")
    assert len([m for m in msgs if m["role"] == "assistant"]) == 4  # 3 identical steps, then the stop note
    assert "same tool call" in msgs[-1]["error"]


def test_step_limit_forces_an_answer(live):
    c = live.client(live.proj)
    p, _ = _tools(live, live.proj)
    p.services["agents"].get("train").steps = 2
    try:
        ScriptedProvider.register("t6", [[{"tool": "echo", "input": {"text": str(i)}}] for i in range(5)])
        s = c.post("/api/session", {"model": "scripted/t6"})
        _run(c, s["id"], "go")
        req = ScriptedProvider.named("t6").requests
        assert len(req) == 2 and req[1].tool_choice == "none" and "step limit" in req[1].system
    finally:
        p.services["agents"].get("train").steps = 200


def test_compaction_keeps_recent_messages(live):
    from opentrain.core import runner as runner_mod

    c = live.client(live.proj)
    p, _ = _tools(live, live.proj)
    s = c.post("/api/session", {"model": "scripted/t7"})
    ScriptedProvider.register("t7", [["a"], ["b"], ["c"], ["d"], ["SUMMARY"], ["e"]])
    for i in range(4):
        _run(c, s["id"], f"msg {i}")
    r = p.services["runner"]
    old = runner_mod.COMPACT_AT
    runner_mod.COMPACT_AT = 0.0000001
    try:
        _run(c, s["id"], "after")
    finally:
        runner_mod.COMPACT_AT = old
    hist = r.history(s["id"])
    assert hist[0]["content"][0]["text"].startswith("Summary of the conversation so far")
    assert "SUMMARY" in hist[0]["content"][0]["text"]
    assert hist[-1]["content"][0]["text"].strip() == "e"
    assert len(hist) <= runner_mod.KEEP_RECENT + 2
