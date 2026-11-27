import json
import subprocess
import sys
import textwrap
import time

from opentrain.core import config
from opentrain.core.commands import Command
from opentrain.core.providers.scripted import ScriptedProvider


def test_jsonc_and_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    g = tmp_path / "xdg" / "opentrain"
    g.mkdir(parents=True)
    (g / "opentrain.jsonc").write_text('{"model": "a/b", /* c */ "steps": 3, "x": {"y": 1, "z": 2},}')
    proj = tmp_path / "p"
    (proj / ".opentrain").mkdir(parents=True)
    (proj / "opentrain.json").write_text('{"x": {"y": 9}, "url": "http://a//b"}')
    monkeypatch.setenv("TOKEN_X", "tok")
    (proj / ".opentrain" / "opentrain.jsonc").write_text('{"model": "c/d", "key": "{env:TOKEN_X}"}')
    cfg = config.load(proj)
    assert cfg["model"] == "c/d" and cfg["steps"] == 3 and cfg["x"] == {"y": 9, "z": 2}
    assert cfg["url"] == "http://a//b" and cfg["key"] == "tok"


def test_command_expansion():
    c = Command("eval", "", "Score $1 against $2. All: $ARGUMENTS")
    assert c.expand("v3 v2") == "Score v3 against v2. All: v3 v2"
    assert Command("x", "", "Do it.").expand("now please") == "Do it.\n\nnow please"


def test_commands_run_in_a_session(live):
    c = live.client(live.proj)
    names = {x["name"] for x in c.get("/api/command")}
    assert {"train", "status", "data", "mine", "fix", "eval", "export"} <= names and "plan" not in names
    d = live.proj / ".opentrain" / "commands" / "team"
    d.mkdir(parents=True)
    (d / "review.md").write_text("---\ndescription: review\nagent: monitor\n---\nReview $ARGUMENTS carefully.\n")
    ScriptedProvider.register("cmd", [["reviewed"]])
    s = c.post("/api/session", {"model": "scripted/cmd"})
    c.post(f"/api/session/{s['id']}/command", {"name": "/team/review", "arguments": "the data"})
    c.post(f"/api/session/{s['id']}/wait", {"timeout": 20})
    msgs = c.get(f"/api/session/{s['id']}/message")
    assert msgs[0]["parts"][0]["text"] == "Review the data carefully."
    assert c.get(f"/api/session/{s['id']}")["agent"] == "monitor"


def test_system_prompt_has_skills_and_instructions(live):
    (live.proj / "AGENTS.md").write_text("Always use uv.")
    (live.proj / "notes.md").write_text("Prefer Qwen.")
    (live.proj / "opentrain.json").write_text(json.dumps({"instructions": ["notes.md"]}))
    c = live.client(live.proj)
    prov = ScriptedProvider.register("sys", [["ok"]])
    s = c.post("/api/session", {"model": "scripted/sys"})
    c.post(f"/api/session/{s['id']}/prompt", {"text": "hi"})
    c.post(f"/api/session/{s['id']}/wait", {"timeout": 20})
    sysp = prov.requests[0].system
    assert "Always use uv." in sysp and "Prefer Qwen." in sysp and "goal/speed" in sysp
    assert "You are OpenTrain's train agent" in sysp


def test_first_prompt_writes_a_default_trainfile(live):
    subprocess.run(["git", "init", "-q", str(live.proj)], check=True)
    c = live.client(live.proj)
    ScriptedProvider.register("tf", [["ok"]])
    s = c.post("/api/session", {"model": "scripted/tf"})
    assert not (live.proj / "Trainfile").exists()
    c.post(f"/api/session/{s['id']}/prompt", {"text": "hi"})
    c.post(f"/api/session/{s['id']}/wait", {"timeout": 20})
    assert (live.proj / "Trainfile").read_text().startswith("# Trainfile")


def test_an_mcp_server_adds_tools(live, tmp_path):
    srv = tmp_path / "mcp_srv.py"
    srv.write_text(textwrap.dedent('''
        import json, sys
        for line in sys.stdin:
            m = json.loads(line)
            if "id" not in m: continue
            if m["method"] == "initialize": r = {"protocolVersion": "2025-06-18", "capabilities": {}}
            elif m["method"] == "tools/list": r = {"tools": [{"name": "shout", "description": "upper-case",
                "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}}]}
            elif m["method"] == "tools/call": r = {"content": [{"type": "text", "text": m["params"]["arguments"]["text"].upper()}]}
            else: r = {}
            print(json.dumps({"jsonrpc": "2.0", "id": m["id"], "result": r}), flush=True)
    '''))
    (live.proj / "opentrain.json").write_text(json.dumps({"mcp": {"demo": {"type": "local",
                                                                            "command": [sys.executable, str(srv)]}}}))
    c = live.client(live.proj)
    for _ in range(100):
        if "connected" in (c.get("/api/mcp").get("demo") or ""):
            break
        time.sleep(0.05)
    assert c.get("/api/mcp")["demo"] == "connected (1 tools)"
    ScriptedProvider.register("mcp", [[{"tool": "demo_shout", "input": {"text": "hi"}}], ["ok"]])
    s = c.post("/api/session", {"model": "scripted/mcp"})
    c.post(f"/api/session/{s['id']}/prompt", {"text": "go"})
    c.post(f"/api/session/{s['id']}/wait", {"timeout": 20})
    t = [p["tool"] for m in c.get(f"/api/session/{s['id']}/message") for p in m["parts"] if p["type"] == "tool"][0]
    assert t["output"] == "HI"
