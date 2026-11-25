"""The OpenTrain terminal app.

A client of the server: everything it shows comes from the API and the event stream, and every
action is a request. Close it and jobs keep running; open it again and it picks up where the
session is.

Keys: enter sends · ctrl+j new line · tab switches agent (train ⇄ plan) · esc interrupts ·
ctrl+p commands · ctrl+o models · ctrl+n new session · ctrl+l sessions · ctrl+b sidebar ·
ctrl+c quits.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Static, TextArea

from .. import __version__
from ..client import AsyncClient, ClientError
from ..core.store import merge
from .dialogs import Item, KeyScreen, PickerScreen, QuestionScreen, TextScreen
from .views import JobCard, MessageView, ToolView, bar, dur, money, part_widget

WORDMARK = r"""
  ___                _____         _
 / _ \ _ __  ___ _ _|_   _| _ __ _(_)_ _
| (_) | '_ \/ -_) ' \ | || '_/ _` | | ' \
 \___/| .__/\___|_||_||_||_| \__,_|_|_||_|
      |_|
"""

LOCAL_COMMANDS = {
    "models": "choose the model that powers the engine",
    "connect": "add a key for a provider or a trainer",
    "trainer": "choose the trainer and base model (edits the Trainfile)",
    "agents": "switch agent",
    "sessions": "open another session",
    "new": "start a new session",
    "jobs": "running and past jobs",
    "versions": "model versions and what is live",
    "sidebar": "show or hide the sidebar",
    "help": "keys and commands",
    "quit": "leave (jobs keep running on the server)",
}

HELP = """\
# OpenTrain

You talk to the **train** agent; it turns your agent sessions into verified tasks, trains a model on
them, and ships a version only when it beats the current one on held-out work. **plan** thinks
without spending anything (tab switches).

| Key | Does |
| --- | --- |
| enter / ctrl+j | send / new line |
| tab | switch agent (train ⇄ plan) |
| esc | interrupt the running turn |
| ctrl+p | commands |
| ctrl+o | models |
| ctrl+n / ctrl+l | new session / sessions |
| ctrl+b | sidebar |
| ctrl+c | quit (jobs keep running) |

When the agent wants to spend money, upload data, launch training, promote a version or run a
command, it asks: **1** allow once, **2** always, **3** reject.

Commands: `/train` `/status` `/mine` `/fix <what is wrong>` `/eval <version>` `/plan` `/export`,
and `/models` `/connect` `/trainer` `/agents` `/sessions` `/new` `/jobs` `/versions` `/help`.
"""


class Prompt(TextArea):
    """Enter sends; ctrl+j (or shift+enter where the terminal reports it) adds a line."""

    class Submitted(Message):
        def __init__(self, text: str):
            super().__init__()
            self.text = text

    async def _on_key(self, event: Any) -> None:
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            text = self.text.strip()
            if text:
                self.post_message(self.Submitted(text))
            return
        if event.key in ("ctrl+j", "shift+enter"):
            event.prevent_default()
            event.stop()
            self.insert("\n")
            return
        await super()._on_key(event)


class PermissionPrompt(Vertical):
    BINDINGS = [Binding("1", "reply('once')", "once"), Binding("2", "reply('always')", "always"),
                Binding("3", "reply('reject')", "reject"), Binding("y", "reply('once')", show=False),
                Binding("n", "reply('reject')", show=False)]

    class Replied(Message):
        def __init__(self, pid: str, reply: str):
            super().__init__()
            self.pid, self.reply = pid, reply

    def __init__(self):
        super().__init__(id="permission")
        self.request: dict[str, Any] | None = None
        self.can_focus = True

    def compose(self) -> ComposeResult:
        yield Static("", id="perm-text")
        with Horizontal():
            yield Button("1 Allow once", id="once", variant="primary")
            yield Button("2 Always", id="always")
            yield Button("3 Reject", id="reject", variant="error")

    def show(self, req: dict[str, Any]) -> None:
        self.request = req
        self.query_one("#perm-text", Static).update(describe_permission(req))
        self.add_class("shown")
        self.focus()

    def hide(self) -> None:
        self.request = None
        self.remove_class("shown")

    def action_reply(self, reply: str) -> None:
        if self.request:
            self.post_message(self.Replied(self.request["id"], reply))

    @on(Button.Pressed)
    def _pressed(self, event: Button.Pressed) -> None:
        self.action_reply(event.button.id or "reject")


def describe_permission(req: dict[str, Any]) -> Text:
    a, res, meta = req.get("action"), req.get("resources") or [], req.get("metadata") or {}
    t = Text()
    t.append(f"{req.get('agent', 'agent')} wants to ", style="bold")
    if a == "spend":
        t.append(f"spend {res[0] if res else ''}", style="bold yellow")
        if meta.get("budget_remaining_usd") is not None:
            t.append(f"   (budget left {money(meta['budget_remaining_usd'])})", style="dim")
    elif a == "upload":
        t.append(f"upload {meta.get('what', 'data')} to {res[0] if res else 'the trainer'}", style="bold yellow")
        t.append("\nNothing leaves this machine without this approval. Redaction has run.", style="dim")
    elif a == "job.start":
        t.append(f"launch training on {res[0] if res else ''}", style="bold yellow")
        if meta.get("base_model"):
            t.append(f"  ({meta['base_model']})", style="dim")
    elif a == "promote":
        t.append(f"promote {', '.join(res)} (the gate still decides)", style="bold yellow")
    elif a == "serve":
        t.append(f"change what is live to {', '.join(res)}", style="bold yellow")
    elif a == "shell":
        t.append("run\n", style="bold")
        t.append("  $ " + "\n  ".join(res), style="cyan")
    elif a in ("edit", "write"):
        t.append(f"{a} {', '.join(res)}", style="bold yellow")
    else:
        t.append(f"{a} {', '.join(res)}", style="bold yellow")
    return t


class OpenTrainApp(App):
    CSS_PATH = "app.tcss"
    TITLE = "OpenTrain"
    BINDINGS = [
        Binding("ctrl+c", "quit", "quit", priority=True),
        Binding("escape", "interrupt", "interrupt"),
        Binding("tab", "cycle_agent", "agent", priority=True),
        Binding("ctrl+p", "palette", "commands", priority=True),
        Binding("ctrl+o", "models", "models", priority=True),
        Binding("ctrl+n", "new_session", "new", priority=True),
        Binding("ctrl+l", "sessions", "sessions", priority=True),
        Binding("ctrl+b", "toggle_sidebar", "sidebar", priority=True),
    ]

    def __init__(self, directory: Path, url: str, password: str, *, session_id: str | None = None,
                 initial_prompt: str | None = None):
        super().__init__()
        self.directory = Path(directory).resolve()
        self.url, self.password = url, password
        self.session_id = session_id
        self.initial_prompt = initial_prompt
        self.session: dict[str, Any] | None = None
        self.agent = "train"
        self.model: str | None = None
        self.agents: list[dict[str, Any]] = []
        self.messages: dict[str, dict[str, Any]] = {}
        self.parts: dict[str, dict[str, Any]] = {}
        self.msg_views: dict[str, MessageView] = {}
        self.part_views: dict[str, Any] = {}
        self.job_cards: dict[str, JobCard] = {}
        self.jobs: dict[str, dict[str, Any]] = {}
        self.status_text = "idle"
        self.client: AsyncClient | None = None
        self.server_commands: dict[str, str] = {}
        self.todos: list[dict[str, Any]] = []
        self.overview: dict[str, Any] = {}

    # ------------------------------------------------------------------ layout

    def compose(self) -> ComposeResult:
        yield Static("", id="topbar")
        with Horizontal(id="body"):
            with Vertical(id="main"):
                yield VerticalScroll(id="log")
                yield PermissionPrompt()
                yield Prompt(id="prompt", soft_wrap=True, show_line_numbers=False, tab_behavior="focus")
                yield Static("", id="hints")
            with VerticalScroll(id="sidebar"):
                yield Static("", id="side")

    async def on_mount(self) -> None:
        self.client = AsyncClient(self.url, self.password, self.directory)
        self.query_one("#prompt", Prompt).focus()
        self._hints()
        try:
            self.agents = await self.client.get("/api/agent")
            self.server_commands = {c["name"]: c["description"] for c in await self.client.get("/api/command")}
        except ClientError as exc:
            self.notify(f"cannot reach the OpenTrain server: {exc.message}", severity="error", timeout=10)
        if self.session_id:
            await self.open_session(self.session_id)
        else:
            await self.show_home()
        self.listen()
        self.set_interval(4.0, self.refresh_overview)
        self.set_interval(1.0, self._tick)
        await self.refresh_overview()
        if self.initial_prompt:
            await self.submit(self.initial_prompt)

    async def on_unmount(self) -> None:
        if self.client:
            await self.client.close()

    # ------------------------------------------------------------------ top bar, hints, sidebar

    def _agent_color(self, name: str) -> str:
        a = next((x for x in self.agents if x["name"] == name), None)
        return (a or {}).get("color") or "green"

    def _topbar(self) -> None:
        t = Text()
        t.append(" OpenTrain ", style="bold reverse")
        t.append(f" {self.directory.name} ", style="bold")
        t.append(" │ ", style="dim")
        t.append(f" {self.agent} ", style=f"bold {self._agent_color(self.agent)} reverse")
        model = self.model or (self.session or {}).get("model") or self.overview.get("default_model") or "default model"
        t.append(f"  {model}", style="dim")
        t.append(" │ ", style="dim")
        st = self.status_text
        t.append(st, style={"busy": "yellow", "waiting": "bold magenta", "retry": "red", "compacting": "cyan"}
                 .get(st.split()[0], "green"))
        if self.session:
            tok = self.session.get("tokens") or {}
            t.append(" │ ", style="dim")
            t.append(f"{tok.get('input', 0) + tok.get('output', 0):,} tok · {money(self.session.get('cost'))}",
                     style="dim")
        b = self.overview.get("budget") or {}
        if b:
            t.append(" │ ", style="dim")
            t.append(f"budget {money(b.get('committed_usd'))} used · {money(b.get('remaining_usd'))} left",
                     style="dim")
        self.query_one("#topbar", Static).update(t)

    def _hints(self) -> None:
        busy = self.status_text != "idle"
        t = Text("enter send · ctrl+j newline · tab agent · ctrl+p commands · ctrl+o models · ", style="dim")
        t.append("esc interrupt" if busy else "ctrl+l sessions", style="bold" if busy else "dim")
        t.append(f" · v{__version__}", style="dim")
        self.query_one("#hints", Static).update(t)

    async def refresh_overview(self) -> None:
        if not self.client:
            return
        try:
            st = await self.client.get("/api/status")
            models = await self.client.get("/api/model")
            st["default_model"] = models.get("default")
            self.overview = st
            if self.session:
                self.todos = await self.client.get(f"/api/session/{self.session['id']}/todo")
        except ClientError:
            return
        self._sidebar()
        self._topbar()

    def _sidebar(self) -> None:
        o = self.overview
        t = Text()
        if self.session:
            t.append(f"{self.session.get('title', '')}\n", style="bold")
            t.append(f"{self.session['id']}\n", style="dim")
        pol = o.get("policy") or {}
        t.append("\nPOLICY\n", style="bold dim")
        if pol:
            t.append(f"goal {pol.get('goal')} · {pol.get('budget')}\n")
            t.append(f"{pol.get('trainer')} · {pol.get('base_model')}\n", style="dim")
            t.append(f"held-out {int((pol.get('heldout') or 0) * 100)}% · eval cap {pol.get('eval_cap')}\n", style="dim")
        tf = o.get("trainfile") or ""
        t.append("Trainfile ✓\n" if tf.startswith("/") else "no Trainfile yet (defaults apply)\n", style="dim")
        b = o.get("budget") or {}
        if b:
            t.append("\nBUDGET\n", style="bold dim")
            used, left = b.get("committed_usd") or 0, b.get("remaining_usd") or 0
            total = used + left
            t.append(bar(used / total if total else 0, 24) + "\n", style="yellow" if left else "red")
            t.append(f"{money(used)} used · {money(left)} left\n", style="dim")
        t.append("\nDATA\n", style="bold dim")
        dsets = o.get("datasets") or []
        for d in dsets[:6]:
            c = d.get("counts") or {}
            t.append(f"{d['name']} ", style="bold")
            t.append(f"{c.get('train', 0)} train · {c.get('heldout', 0)} held out\n", style="dim")
        tasks = o.get("coding_tasks") or {}
        if tasks.get("training") or tasks.get("heldout"):
            t.append(f"coding tasks {tasks.get('training', 0)} · {tasks.get('heldout', 0)} held out\n", style="dim")
        elif not dsets:
            t.append("no datasets yet\n", style="dim")
        live = o.get("live") or {}
        t.append("\nLIVE\n", style="bold dim")
        t.append(f"{live.get('version') or 'nothing trained yet'}", style="green" if live.get("version") else "dim")
        if live.get("base_model"):
            t.append(f"\n{live['base_model']}", style="dim")
        t.append("\n")
        jobs = o.get("jobs_running") or []
        t.append("\nJOBS\n", style="bold dim")
        if not jobs:
            t.append("none running\n", style="dim")
        for j in jobs:
            t.append(f"{j['kind']} ", style="bold")
            t.append(f"{j['title'][:26]}\n")
            t.append(f"{bar(j.get('progress'), 16)} {j.get('detail') or ''}"[:34] + "\n", style="yellow")
        if self.todos:
            t.append("\nTODO\n", style="bold dim")
            for td in self.todos:
                mark = {"completed": "✓", "in_progress": "▸", "cancelled": "✗"}.get(td["status"], "·")
                t.append(f"{mark} {td['content'][:32]}\n", style="dim" if td["status"] == "completed" else "")
        t.append(f"\nsandbox {o.get('sandbox', '?')}\n", style="dim")
        self.query_one("#side", Static).update(t)

    def _tick(self) -> None:
        for card in self.job_cards.values():
            if card.job.get("state") == "running":
                card.refresh_from(card.job)

    # ------------------------------------------------------------------ home and sessions

    async def show_home(self) -> None:
        log = self.query_one("#log", VerticalScroll)
        await log.remove_children()
        self.session, self.session_id = None, None
        self.messages.clear(), self.parts.clear(), self.msg_views.clear(), self.part_views.clear()
        self.job_cards.clear()
        t = Text()
        t.append(WORDMARK, style="bold cyan")
        t.append(f"\n  The open harness for training models · v{__version__}\n\n", style="dim")
        t.append(f"  {self.directory}\n\n", style="bold")
        t.append("  Try\n", style="bold dim")
        for line in ("make me a model that does my routine work faster",
                     "/status    what is live, what it cost, how the last run went",
                     "/mine      turn my agent sessions into verified tasks",
                     "/train     run the goal end to end (asks before spending)",
                     "/plan      think it through without spending anything"):
            t.append(f"    {line}\n")
        t.append("\n  tab switches train ⇄ plan · ctrl+o model · /connect keys · /trainer base model\n", style="dim")
        try:
            sessions = await self.client.get("/api/session") if self.client else []
        except ClientError:
            sessions = []
        if sessions:
            t.append("\n  Recent sessions (ctrl+l)\n", style="dim")
            for s in sessions[:5]:
                t.append(f"    {s.get('title', '')[:60]}", style="")
                t.append(f"  {s.get('agent')} · {money(s.get('cost'))}\n", style="dim")
        await log.mount(Static(t, id="home"))
        self._topbar()

    async def open_session(self, sid: str) -> None:
        assert self.client
        try:
            s = await self.client.get(f"/api/session/{sid}")
            msgs = await self.client.get(f"/api/session/{sid}/message")
            jobs = await self.client.get("/api/job", session=sid)
        except ClientError as exc:
            self.notify(exc.message, severity="error")
            return
        log = self.query_one("#log", VerticalScroll)
        await log.remove_children()
        self.messages.clear(), self.parts.clear(), self.msg_views.clear(), self.part_views.clear()
        self.job_cards.clear()
        self.session, self.session_id = s, sid
        self.agent = s.get("agent") or "train"
        self.model = s.get("model") if isinstance(s.get("model"), str) else None
        self.jobs = {j["id"]: j for j in jobs}
        for m in msgs:
            await self._add_message(m)
            for p in m.get("parts", []):
                await self._add_part(p)
        self.status_text = s.get("status") or "idle"
        log.scroll_end(animate=False)
        self._topbar()
        self._hints()
        for req in await self.client.get("/api/permission", session=sid):
            self.query_one(PermissionPrompt).show(req)
        for q in await self.client.get("/api/question", session=sid):
            self._ask_question(q)

    async def ensure_session(self) -> str:
        if self.session_id:
            return self.session_id
        assert self.client
        body: dict[str, Any] = {"agent": self.agent}
        if self.model:
            body["model"] = self.model
        s = await self.client.post("/api/session", body)
        await self.open_session(s["id"])
        return s["id"]

    # ------------------------------------------------------------------ rendering

    async def _add_message(self, m: dict[str, Any]) -> MessageView:
        m = {k: v for k, v in m.items() if k != "parts"}
        self.messages[m["id"]] = m
        if m["id"] in self.msg_views:
            return self.msg_views[m["id"]]
        view = MessageView(m)
        self.msg_views[m["id"]] = view
        await self.query_one("#log", VerticalScroll).mount(view)
        return view

    async def _add_part(self, p: dict[str, Any]) -> None:
        self.parts[p["id"]] = p
        if p["id"] in self.part_views:
            return
        view = self.msg_views.get(p.get("message_id"))
        if view is None:
            return
        w = part_widget(p)
        self.part_views[p["id"]] = w
        await view.mount(w)
        await self._job_card_for(p)

    async def _job_card_for(self, p: dict[str, Any]) -> None:
        if p.get("type") != "tool":
            return
        job_id = ((p.get("tool") or {}).get("metadata") or {}).get("job_id")
        if not job_id or job_id in self.job_cards or not self.client:
            return
        job = self.jobs.get(job_id)
        if job is None:
            try:
                job = await self.client.get(f"/api/job/{job_id}")
            except ClientError:
                return
        card = JobCard(job)
        self.job_cards[job_id] = card
        view = self.msg_views.get(p.get("message_id"))
        if view:
            await view.mount(card)

    def _follow(self) -> None:
        log = self.query_one("#log", VerticalScroll)
        if log.max_scroll_y - log.scroll_y < 12:
            log.scroll_end(animate=False)

    # ------------------------------------------------------------------ events

    @work(exclusive=True, group="events")
    async def listen(self) -> None:
        delay = 0.5
        while True:
            try:
                assert self.client
                async for ev in self.client.events():
                    delay = 0.5
                    try:
                        await self.on_server_event(ev)
                    except Exception as exc:  # one bad event never kills the stream
                        self.log.error(f"event {ev.get('type')}: {exc}")
            except (ClientError, OSError, asyncio.TimeoutError) as exc:
                self.status_text = "reconnecting"
                self._topbar()
                self.log.warning(f"event stream: {exc}")
            except Exception as exc:
                self.log.error(f"event stream: {exc}")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 10)
            if self.session_id:
                await self.open_session(self.session_id)

    async def on_server_event(self, ev: dict[str, Any]) -> None:
        typ, data = ev.get("type", ""), ev.get("data") or {}
        sid = data.get("session_id")
        mine = self.session_id is not None and sid == self.session_id
        if typ == "server.connected":
            return
        if typ == "part.delta" and mine:
            w = self.part_views.get(data.get("part_id"))
            if w is not None and hasattr(w, "add"):
                await w.add(data.get("delta", ""))
                self._follow()
            return
        if typ in ("message.created", "message.updated") and mine:
            m = merge(self.messages.get(data["id"], {}), data)
            if typ == "message.created" or data["id"] not in self.msg_views:
                await self._add_message(m)
                self._follow()
            else:
                self.messages[data["id"]] = m
                self.msg_views[data["id"]].refresh_meta(m)
            return
        if typ in ("part.created", "part.updated") and mine:
            p = merge(self.parts.get(data["id"], {}), data)
            if data["id"] not in self.part_views:
                await self._add_part(p)
                self._follow()
                return
            self.parts[data["id"]] = p
            w = self.part_views[data["id"]]
            if isinstance(w, ToolView):
                w.refresh_from(p)
                await self._job_card_for(p)
            elif hasattr(w, "set_text") and "text" in data:
                await w.set_text(p.get("text") or "")
            return
        if typ == "session.status" and mine:
            st = data.get("status", "idle")
            if st == "retry":
                st = f"retry {data.get('attempt')} in {dur(data.get('wait'))}"
            self.status_text = st
            self._topbar()
            self._hints()
            return
        if typ in ("session.updated", "session.created") and data.get("id") == self.session_id:
            self.session = merge(self.session or {}, data)
            self._topbar()
            return
        if typ in ("job.created", "job.updated"):
            j = merge(self.jobs.get(data["id"], {}), data)
            self.jobs[data["id"]] = j
            if data["id"] in self.job_cards:
                self.job_cards[data["id"]].refresh_from(j)
            if typ == "job.updated" and j.get("state") in ("succeeded", "failed", "cancelled"):
                self.notify(f"{j.get('kind')} job {j.get('state')}: {j.get('title')}",
                            severity="information" if j["state"] == "succeeded" else "warning")
                await self.refresh_overview()
            return
        if typ == "permission.created" and mine and data.get("status") == "pending":
            self.query_one(PermissionPrompt).show(data)
            self.bell()
            return
        if typ == "permission.updated":
            pp = self.query_one(PermissionPrompt)
            if pp.request and pp.request["id"] == data.get("id") and data.get("status") != "pending":
                pp.hide()
                self.query_one("#prompt", Prompt).focus()
            return
        if typ == "question.created" and mine:
            self._ask_question(data)
            return
        if typ in ("todo.created", "todo.updated") and mine:
            self.todos = data.get("todos") or []
            self._sidebar()

    def _ask_question(self, q: dict[str, Any]) -> None:
        async def answered(answers: list[list[str]] | None) -> None:
            if not self.client or not self.session_id:
                return
            body = {"answers": answers} if answers is not None else {"reject": True}
            try:
                await self.client.post(f"/api/session/{self.session_id}/question/{q['id']}/reply", body)
            except ClientError as exc:
                self.notify(exc.message, severity="warning")

        self.bell()
        self.push_screen(QuestionScreen(q["questions"]), answered)

    @on(PermissionPrompt.Replied)
    async def _permission_replied(self, event: PermissionPrompt.Replied) -> None:
        if not self.client or not self.session_id:
            return
        try:
            await self.client.post(f"/api/session/{self.session_id}/permission/{event.pid}/reply",
                                   {"reply": event.reply})
        except ClientError as exc:
            self.notify(exc.message, severity="warning")
        self.query_one(PermissionPrompt).hide()
        self.query_one("#prompt", Prompt).focus()

    # ------------------------------------------------------------------ input

    @on(Prompt.Submitted)
    async def _submitted(self, event: Prompt.Submitted) -> None:
        self.query_one("#prompt", Prompt).text = ""
        await self.submit(event.text)

    @on(TextArea.Changed, "#prompt")
    def _prompt_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.text == "/":
            event.text_area.text = ""
            self.action_palette()

    async def submit(self, text: str) -> None:
        if not self.client:
            return
        if text.startswith("/"):
            name, _, args = text[1:].partition(" ")
            if name in LOCAL_COMMANDS:
                await self.run_local(name, args.strip())
                return
            if name in self.server_commands:
                sid = await self.ensure_session()
                try:
                    await self.client.post(f"/api/session/{sid}/command", {"name": name, "arguments": args.strip()})
                except ClientError as exc:
                    self.notify(exc.message, severity="error")
                return
            self.notify(f"no command /{name}", severity="warning")
            return
        sid = await self.ensure_session()
        try:
            await self.client.post(f"/api/session/{sid}/prompt", {"text": text, "agent": self.agent})
        except ClientError as exc:
            self.notify(exc.message, severity="error")
        self.query_one("#log", VerticalScroll).scroll_end(animate=False)

    async def run_local(self, name: str, args: str = "") -> None:
        if name == "models":
            self.action_models()
        elif name == "connect":
            self.connect_flow(args)
        elif name == "trainer":
            self.trainer_flow()
        elif name == "agents":
            self.agent_picker()
        elif name == "sessions":
            self.action_sessions()
        elif name == "new":
            await self.action_new_session()
        elif name == "jobs":
            self.jobs_view()
        elif name == "versions":
            self.versions_view()
        elif name == "sidebar":
            self.action_toggle_sidebar()
        elif name == "help":
            self.push_screen(TextScreen("Help", HELP))
        elif name == "quit":
            self.exit()

    # ------------------------------------------------------------------ actions

    async def action_interrupt(self) -> None:
        if self.screen is not self.screen_stack[0]:
            return
        if self.session_id and self.status_text != "idle" and self.client:
            await self.client.post(f"/api/session/{self.session_id}/interrupt")
            self.notify("interrupted")

    async def action_cycle_agent(self) -> None:
        if self.screen is not self.screen_stack[0]:
            return
        primaries = [a["name"] for a in self.agents if a["mode"] in ("primary", "all") and not a["hidden"]]
        if not primaries:
            return
        i = primaries.index(self.agent) if self.agent in primaries else -1
        await self.set_agent(primaries[(i + 1) % len(primaries)])

    async def set_agent(self, name: str) -> None:
        self.agent = name
        if self.session_id and self.client:
            self.session = await self.client.patch(f"/api/session/{self.session_id}", {"agent": name})
        self._topbar()

    def agent_picker(self) -> None:
        items = [Item(a["name"], a["name"], a["description"], "agents")
                 for a in self.agents if a["mode"] in ("primary", "all") and not a["hidden"]]

        async def chosen(name: str | None) -> None:
            if name:
                await self.set_agent(name)

        self.push_screen(PickerScreen("Agent", items, current=self.agent), chosen)

    def action_palette(self) -> None:
        items = [Item("/" + n, "/" + n, d, "OpenTrain") for n, d in LOCAL_COMMANDS.items()]
        items += [Item("/" + n, "/" + n, d, "Commands") for n, d in sorted(self.server_commands.items())]

        async def chosen(cmd: str | None) -> None:
            if not cmd:
                return
            name = cmd[1:]
            if name in LOCAL_COMMANDS:
                await self.run_local(name)
            else:
                p = self.query_one("#prompt", Prompt)
                p.text = cmd + " "
                p.move_cursor(p.document.end)
                p.focus()

        self.push_screen(PickerScreen("Commands", items, help="enter runs · esc closes"), chosen)

    @work(group="dialogs")
    async def action_models(self) -> None:
        assert self.client
        data = await self.client.get("/api/model", session=self.session_id)
        current = data.get("session") or self.model or data.get("default")
        rows = sorted(data["models"], key=lambda m: (not m["connected"], m["provider_name"], m["name"]))
        items = []
        for m in rows:
            price = f"${m['input_price']:g}/${m['output_price']:g} per M" if m["input_price"] else ""
            items.append(Item(m["ref"], m["name"], f"{m['ref']}  {price}".strip(),
                              m["provider_name"] + ("" if m["connected"] else "  (not connected: /connect)"),
                              disabled=False))
        ref = await self.push_screen_wait(PickerScreen("Model for the engine", items, current=current,
                                                       help="Any provider/model also works: type it and press enter "
                                                            "if it is not listed."))
        if not ref:
            return
        self.model = ref
        if self.session_id:
            self.session = await self.client.patch(f"/api/session/{self.session_id}", {"model": ref})
        await self.client.post("/api/model/default", {"ref": ref})
        self.notify(f"model: {ref}")
        self._topbar()

    @work(group="dialogs")
    async def connect_flow(self, iid: str = "") -> None:
        assert self.client
        rows = await self.client.get("/api/integration")
        if not iid:
            items = [Item(r["id"], r["name"], ("connected " + (r.get("via") or "") if r["connected"] else r.get("url") or ""),
                          {"provider": "Model providers (the engine)", "trainer": "Trainers"}[r["kind"]])
                     for r in rows if r["id"] != "claude-code"]
            iid = await self._pick("Connect", items, "Keys are stored in ~/.local/share/opentrain/auth.json (0600) "
                                                     "and never leave this machine except to that service.")
            if not iid:
                return
        row = next((r for r in rows if r["id"] == iid), None)
        if row is None:
            self.notify(f"unknown integration {iid}", severity="warning")
            return
        key = await self._ask_key(f"{row['name']} key", f"Get one at {row.get('url')}" if row.get("url") else "")
        if not key:
            return
        try:
            out = await self.client.post(f"/api/integration/{iid}/connect", {"key": key})
            self.notify(f"{out['name']} connected (…{out.get('hint')})")
        except ClientError as exc:
            self.notify(exc.message, severity="error")
        await self.refresh_overview()

    @work(group="dialogs")
    async def trainer_flow(self) -> None:
        assert self.client
        data = await self.client.get("/api/trainer")
        pol = data.get("policy") or {}
        items = [Item(t["name"], t["name"], ("ready" if t["available"] else t["detail"])[:70], "Trainers")
                 for t in data["trainers"]]
        name = await self._pick("Trainer", items, current=pol.get("trainer"))
        if not name:
            return
        tr = next(t for t in data["trainers"] if t["name"] == name)
        base = None
        if tr["base_models"]:
            bitems = [Item(b["id"], b["id"], f"{b['params']}  {b['note']}", "Base models") for b in tr["base_models"]]
            base = await self._pick(f"Base model on {name}", bitems, current=pol.get("base_model"))
        try:
            out = await self.client.post("/api/trainer", {"trainer": name, **({"base_model": base} if base else {})})
            p = out.get("policy") or {}
            self.notify(f"Trainfile: trainer {p.get('trainer')}, base model {p.get('base_model')}")
        except ClientError as exc:
            self.notify(exc.message, severity="error")
        if not tr["available"] and tr.get("credential"):
            self.notify(f"{name} needs a key: /connect {tr['credential']}", severity="warning", timeout=8)
        await self.refresh_overview()

    async def _pick(self, title: str, items: list[Item], help: str = "", current: str | None = None) -> str | None:
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self.push_screen(PickerScreen(title, items, help=help, current=current),
                         lambda r: fut.set_result(r) if not fut.done() else None)
        return await fut

    async def _ask_key(self, title: str, help: str) -> str | None:
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self.push_screen(KeyScreen(title, help), lambda r: fut.set_result(r) if not fut.done() else None)
        return await fut

    @work(group="dialogs")
    async def action_sessions(self) -> None:
        assert self.client
        rows = await self.client.get("/api/session")
        items = [Item(s["id"], s.get("title") or s["id"],
                      f"{s.get('agent')} · {money(s.get('cost'))} · {s.get('status')}", "Sessions") for s in rows]
        items.insert(0, Item("__new__", "New session", "ctrl+n", "Sessions"))
        sid = await self.push_screen_wait(PickerScreen("Sessions", items, current=self.session_id))
        if sid == "__new__":
            await self.action_new_session()
        elif sid:
            await self.open_session(sid)

    async def action_new_session(self) -> None:
        await self.show_home()
        self.query_one("#prompt", Prompt).focus()

    def action_toggle_sidebar(self) -> None:
        self.query_one("#sidebar").toggle_class("hidden")

    @work(group="dialogs")
    async def jobs_view(self) -> None:
        assert self.client
        rows = await self.client.get("/api/job")
        if not rows:
            self.notify("no jobs yet")
            return
        items = [Item(j["id"], f"{j['kind']}  {j['title']}", f"{j['state']} {money(j.get('cost_usd'))}", "Jobs")
                 for j in rows]
        jid = await self._pick("Jobs", items, help="enter shows the log")
        if jid:
            lines = (await self.client.get(f"/api/job/{jid}/logs", tail=300))["lines"]
            j = await self.client.get(f"/api/job/{jid}")
            body = f"**{j['kind']}** {j['title']} — {j['state']}\n\n```\n" + "\n".join(lines[-300:]) + "\n```"
            self.push_screen(TextScreen(f"Job {jid}", body))

    @work(group="dialogs")
    async def versions_view(self) -> None:
        assert self.client
        st = await self.client.get("/api/status")
        vs = st.get("versions") or []
        if not vs:
            self.notify("no versions yet: the train agent registers one when a training job succeeds")
            return
        live = (st.get("live") or {}).get("version")
        items = [Item(v["version"], v["version"], f"{(v.get('decision') or 'not proposed')[:60]}", "Versions")
                 for v in reversed(vs)]
        v = await self._pick("Versions", items, current=live, help="enter asks the agent to put it live")
        if v and v != live:
            p = self.query_one("#prompt", Prompt)
            p.text = f"Put {v} live for serving (it passed the gate?) and tell me what changes."
            p.focus()
