import pytest

from opentrain.core.agent import AgentError, Agents, parse_markdown, rules_from


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


class P:
    def __init__(self, d):
        self.directory = d


def test_builtins_load_with_their_rules(tmp_path):
    a = Agents(P(tmp_path))
    assert [x.name for x in a.primaries()] == ["train", "plan"]
    assert [x.name for x in a.subagents()] == ["data", "monitor"]
    ev = a.find("evaluator")
    assert ev.hidden and ev.subagent
    train = a.get(None)
    assert train.name == "train" and "status" in train.prompt
    assert {"action": "heldout.read", "resource": "*", "effect": "deny"} in train.permissions
    assert {"action": "read", "resource": ".opentrain/heldout/**", "effect": "deny"} in train.permissions
    plan = a.get("plan")
    assert plan.permissions[0] == {"action": "*", "resource": "*", "effect": "deny"}
    assert a.problems == []


def test_front_matter_parsing():
    meta, body = parse_markdown('---\ndescription: "a: b"\nsteps: 7\nhidden: true\npermissions:\n  "edit src/**": deny\n  shell: ask\n---\nHello\n')
    assert meta["description"] == "a: b" and meta["steps"] == 7 and meta["hidden"] is True
    assert rules_from(meta["permissions"]) == [{"action": "edit", "resource": "src/**", "effect": "deny"},
                                               {"action": "shell", "resource": "*", "effect": "ask"}]
    assert body == "Hello"
    with pytest.raises(AgentError):
        rules_from([("shell", "maybe")])


def test_project_agents_override_and_add(tmp_path):
    d = tmp_path / ".opentrain" / "agents"
    d.mkdir(parents=True)
    (d / "train.md").write_text("---\nsteps: 9\npermissions:\n  shell: allow\n---\n")
    (d / "reviewer.md").write_text("---\ndescription: Reviews data\nmode: subagent\n---\nYou review.\n")
    (d / "monitor.md").write_text("---\ndisabled: true\n---\n")
    (tmp_path / "opentrain.jsonc").write_text('{ // comment\n "default_agent": "plan", "agents": {"plan": {"steps": 3}},}')
    a = Agents(P(tmp_path))
    t = a.get("train")
    assert t.steps == 9 and "status" in t.prompt  # body kept from the built-in
    assert t.permissions[-1] == {"action": "shell", "resource": "*", "effect": "allow"}
    assert a.find("reviewer").subagent and a.find("monitor") is None
    assert a.default() == "plan" and a.get("plan").steps == 3


def test_agents_reload_when_files_change(tmp_path):
    a = Agents(P(tmp_path))
    assert a.find("custom") is None
    d = tmp_path / ".opentrain" / "agents"
    d.mkdir(parents=True)
    (d / "custom.md").write_text("---\ndescription: c\n---\nYou are custom.\n")
    assert a.find("custom").prompt == "You are custom."


def test_api_lists_agents(live):
    c = live.client(live.proj)
    names = [a["name"] for a in c.get("/api/agent")]
    assert names[:2] == ["train", "plan"]
    assert c.post("/api/session", {})["agent"] == "train"
