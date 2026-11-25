"""Widgets for a conversation: messages and their parts, tool calls, live job cards."""

from __future__ import annotations

import json
import time
from typing import Any

from rich.text import Text
from textual.containers import Vertical
from textual.widgets import Collapsible, Markdown, Static

TOOL_ICONS = {"running": ("◌", "yellow"), "pending": ("◌", "yellow"), "completed": ("✓", "green"),
              "error": ("✗", "red")}
JOB_STATES = {"running": "yellow", "succeeded": "green", "failed": "red", "cancelled": "bright_black"}


def money(x: float | None) -> str:
    if not x:
        return "$0"
    return f"${x:.2f}" if x >= 0.01 else f"${x:.4f}"


def dur(seconds: float | None) -> str:
    if seconds is None:
        return ""
    if seconds < 1:
        return f"{int(seconds * 1000)}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m < 60 else f"{m // 60}h{m % 60:02d}m"


def bar(fraction: float | None, width: int = 20) -> str:
    if fraction is None:
        return "▱" * width
    n = int(round(fraction * width))
    return "▰" * n + "▱" * (width - n)


class TextPartView(Markdown):
    def __init__(self, part: dict[str, Any]):
        super().__init__(part.get("text") or "")
        self.part_id = part["id"]
        self._text = part.get("text") or ""

    async def add(self, delta: str) -> None:
        self._text += delta
        await self.append(delta)

    async def set_text(self, text: str) -> None:
        if text != self._text:
            self._text = text
            await self.update(text)


class ReasoningView(Static):
    def __init__(self, part: dict[str, Any]):
        self.part_id = part["id"]
        self._text = part.get("text") or ""
        super().__init__(self._view())

    def _view(self) -> Text:
        t = self._text.strip().replace("\n", " ")
        return Text("thinking  " + (t[-300:] if len(t) > 300 else t), style="italic dim")

    async def add(self, delta: str) -> None:
        self._text += delta
        self.update(self._view())

    async def set_text(self, text: str) -> None:
        self._text = text
        self.update(self._view())


def _tool_summary(tool: dict[str, Any]) -> str:
    title = tool.get("title")
    if title:
        return str(title)
    args = tool.get("input") or {}
    for k in ("command", "path", "pattern", "url", "version", "job_id", "task_id", "name", "agent"):
        if args.get(k):
            return str(args[k])
    return json.dumps(args)[:80] if args else ""


class ToolView(Collapsible):
    def __init__(self, part: dict[str, Any]):
        self.part_id = part["id"]
        self.tool = part.get("tool") or {}
        self.body = Static("", classes="output")
        super().__init__(self.body, title=self._heading(), collapsed=True)

    def _heading(self) -> str:
        t = self.tool
        state = t.get("state", "running")
        icon, _ = TOOL_ICONS.get(state, ("·", "white"))
        times = t.get("time") or {}
        took = (times.get("end") - times["start"]) if times.get("end") and times.get("start") else None
        name = t.get("name", "?")
        summary = _tool_summary(t).replace("\n", " ")
        if len(summary) > 90:
            summary = summary[:89] + "…"
        tail = f"  {dur(took)}" if took is not None else ""
        if state == "error":
            tail += f"  {str(t.get('error') or '')[:80]}"
        return f"{icon} {name}  {summary}{tail}"

    def _output(self) -> Text:
        t = self.tool
        if t.get("state") == "error":
            return Text(str(t.get("error") or "error"), style="red")
        if t.get("state") in ("running", "pending"):
            live = (t.get("metadata") or {}).get("output")
            return Text(live[-3000:] if live else "running…", style="dim")
        out = t.get("output") or ""
        return Text(out[-6000:] if len(out) > 6000 else out)

    def on_mount(self) -> None:
        self.title = self._heading()  # an update may have arrived while mounting
        self.body.update(self._output())

    def refresh_from(self, part: dict[str, Any]) -> None:
        self.tool = part.get("tool") or {}
        if self.is_mounted:
            self.title = self._heading()
            self.body.update(self._output())


class JobCard(Static):
    def __init__(self, job: dict[str, Any]):
        self.job = job
        super().__init__(self._view())

    def _view(self) -> Text:
        j = self.job
        state = j.get("state", "running")
        t = Text()
        t.append(f"{j.get('kind', 'job')}  ", style="bold")
        t.append(f"{j.get('title', '')}\n")
        t.append(bar(j.get("progress")) + "  ", style=JOB_STATES.get(state, "white"))
        t.append(state, style=JOB_STATES.get(state, "white"))
        if j.get("cost_usd"):
            t.append(f"  {money(j['cost_usd'])}")
        if j.get("estimate_usd"):
            t.append(f" of ~{money(j['estimate_usd'])}", style="dim")
        started = (j.get("time") or {}).get("started")
        ended = (j.get("time") or {}).get("ended")
        if started:
            t.append(f"  {dur((ended or time.time()) - started)}", style="dim")
        if j.get("detail"):
            t.append(f"\n{j['detail']}", style="dim")
        if j.get("version"):
            t.append(f"\n→ {j['version']}", style="green")
        if j.get("error") and state != "succeeded":
            t.append(f"\n{j['error']}", style="red")
        t.append(f"   {j.get('id', '')}", style="dim")
        return t

    def refresh_from(self, job: dict[str, Any]) -> None:
        self.job = job
        self.update(self._view())


class MessageView(Vertical):
    def __init__(self, msg: dict[str, Any]):
        role = msg.get("role", "assistant")
        super().__init__(classes=role if role in ("user", "synthetic", "assistant") else "assistant")
        self.msg = msg
        self.meta = Static("", classes="meta")

    def compose(self):
        yield self.meta

    def on_mount(self) -> None:
        self.refresh_meta()

    def refresh_meta(self, msg: dict[str, Any] | None = None) -> None:
        if msg:
            self.msg = msg
        m = self.msg
        role = m.get("role")
        if role == "user":
            self.meta.display = False
            return
        if role == "synthetic":
            self.meta.update(Text("system reminder", style="dim italic"))
            return
        if role == "compaction":
            self.meta.update(Text("— earlier messages compacted —", style="dim"))
            return
        t = Text()
        t.append(m.get("agent") or "", style="bold")
        if m.get("model"):
            t.append(f" · {m['model']}", style="dim")
        times = m.get("time") or {}
        if times.get("completed") and times.get("created"):
            t.append(f" · {dur(times['completed'] - times['created'])}", style="dim")
        tok = m.get("tokens") or {}
        if tok.get("output"):
            t.append(f" · {tok.get('input', 0) + tok.get('cache_read', 0):,} in / {tok['output']:,} out", style="dim")
        if m.get("cost"):
            t.append(f" · {money(m['cost'])}" + (" (plan)" if m.get("billing") == "subscription" else ""), style="dim")
        if m.get("finish") == "interrupted":
            t.append("  interrupted", style="yellow")
        if m.get("error") and m.get("finish") != "interrupted":
            t.append(f"\n{m['error']}", style="red")
        self.meta.update(t)


def part_widget(part: dict[str, Any]):
    kind = part.get("type")
    if kind == "text":
        return TextPartView(part)
    if kind == "reasoning":
        return ReasoningView(part)
    if kind == "tool":
        return ToolView(part)
    return Static(Text(json.dumps(part)[:200], style="dim"))
