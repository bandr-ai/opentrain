import asyncio

import pytest

from opentrain.core.providers.scripted import ScriptedProvider
from opentrain.tui.app import OpenTrainApp, PermissionPrompt, Prompt
from opentrain.tui.dialogs import KeyScreen, PickerScreen, QuestionScreen
from opentrain.tui.views import JobCard, MessageView, TextPartView, ToolView


async def settle(pilot, until, timeout=10.0):
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while loop.time() < end:
        await pilot.pause(0.05)
        if until():
            return True
    raise AssertionError("condition not reached")


def _app(live, **kw):
    return OpenTrainApp(live.proj, live.url, live.password, **kw)


@pytest.mark.asyncio
async def test_home_then_a_streamed_turn_with_a_tool(live):
    ScriptedProvider.register("tui1", [["Checking the status.", {"tool": "status", "input": {}}],
                                       ["All **good**."]], delay=0.01)
    live.hub.project(live.proj).services  # open the project
    app = _app(live)
    async with app.run_test(size=(140, 45)) as pilot:
        await settle(pilot, lambda: app.query("#home"))
        app.model = "scripted/tui1"
        prompt = app.query_one("#prompt", Prompt)
        prompt.text = "how are we doing"
        await pilot.press("enter")
        await settle(pilot, lambda: len(app.query(MessageView)) >= 3 and app.status_text == "idle")
        texts = [w._text for w in app.query(TextPartView)]
        assert texts[0] == "how are we doing" and "All **good**." in texts[-1]
        tools = list(app.query(ToolView))
        assert tools and tools[0].tool["name"] == "status" and tools[0].tool["state"] == "completed"
        assert "✓ status" in str(tools[0].title)
        assert app.session["title"] == "how are we doing"


@pytest.mark.asyncio
async def test_permission_prompt_answers_with_a_key(live):
    ScriptedProvider.register("tui2", [[{"tool": "shell", "input": {"command": "echo hello"}}], ["ran it"]])
    app = _app(live)
    async with app.run_test(size=(140, 45)) as pilot:
        app.model = "scripted/tui2"
        app.query_one("#prompt", Prompt).text = "run something"
        await pilot.press("enter")
        pp = app.query_one(PermissionPrompt)
        await settle(pilot, lambda: pp.request is not None)
        assert "echo hello" in str(pp.query_one("#perm-text").render())
        await pilot.press("1")
        await settle(pilot, lambda: app.status_text == "idle" and any(
            t.tool.get("state") == "completed" for t in app.query(ToolView)))
        assert pp.request is None
        assert "hello" in [t for t in app.query(ToolView)][0].tool["output"]


@pytest.mark.asyncio
async def test_question_modal(live):
    ScriptedProvider.register("tui3", [[{"tool": "question", "input": {"questions": [
        {"question": "Which trainer?", "options": ["tinker", "local"]}]}}], ["ok"]])
    app = _app(live)
    async with app.run_test(size=(140, 45)) as pilot:
        app.model = "scripted/tui3"
        app.query_one("#prompt", Prompt).text = "ask me"
        await pilot.press("enter")
        await settle(pilot, lambda: isinstance(app.screen, QuestionScreen))
        await pilot.press("down", "enter")
        await settle(pilot, lambda: app.status_text == "idle" and any(
            t.tool.get("state") == "completed" for t in app.query(ToolView)))
        assert "A: local" in list(app.query(ToolView))[0].tool["output"]


@pytest.mark.asyncio
async def test_one_agent_and_models_picker(live):
    app = _app(live)
    async with app.run_test(size=(140, 45)) as pilot:
        await settle(pilot, lambda: app.agents)
        assert app.agent == "train"
        await pilot.press("ctrl+o")
        await settle(pilot, lambda: isinstance(app.screen, PickerScreen))
        await pilot.press(*"haiku 5.5")
        await pilot.press("enter")
        await settle(pilot, lambda: app.model == "anthropic/claude-haiku-5-5")
        assert live.client(live.proj).get("/api/model")["default"] == "anthropic/claude-haiku-5-5"


@pytest.mark.asyncio
async def test_connect_flow_stores_a_key(live):
    app = _app(live)
    async with app.run_test(size=(140, 45)) as pilot:
        app.query_one("#prompt", Prompt).text = "/connect tinker"
        await pilot.press("enter")
        await settle(pilot, lambda: isinstance(app.screen, KeyScreen))
        await pilot.press(*"tk-abc-9876", "enter")
        await settle(pilot, lambda: next(r for r in live.client(live.proj).get("/api/integration")
                                         if r["id"] == "tinker")["connected"])


@pytest.mark.asyncio
async def test_job_card_follows_the_job(live):
    import threading

    p = live.hub.project(live.proj)
    gate = threading.Event()
    ScriptedProvider.register("tui4", [[{"tool": "tasks_mine", "input": {}}], ["started"], ["seen"]])
    from opentrain.core.tools import train

    def fake_mine(ctx, include_repairable=False):
        job = p.services["jobs"].start_local(ctx.session_id, "mine", "mine sessions", lambda j: gate.wait(5) and {"n": 1},
                                             summarize=lambda r: "1 task")
        return train._started(job, "mining")

    p.services["tools"].get("tasks_mine").execute = fake_mine
    app = _app(live)
    async with app.run_test(size=(140, 45)) as pilot:
        app.model = "scripted/tui4"
        app.query_one("#prompt", Prompt).text = "mine"
        await pilot.press("enter")
        await settle(pilot, lambda: app.query(JobCard))
        card = app.query_one(JobCard)
        assert card.job["state"] == "running"
        gate.set()
        await settle(pilot, lambda: card.job.get("state") == "succeeded")
        await settle(pilot, lambda: any(m.msg.get("role") == "synthetic" for m in app.query(MessageView)))
