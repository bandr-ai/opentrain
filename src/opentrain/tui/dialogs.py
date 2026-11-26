"""Modal dialogs: a filterable picker, a masked key entry, questions from the engine, plain text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, Markdown, OptionList, SelectionList, Static
from textual.widgets.option_list import Option


@dataclass
class Item:
    id: str
    label: str
    detail: str = ""
    group: str = ""
    disabled: bool = False


class PickerScreen(ModalScreen[str | None]):
    """Pick one item; type to filter. Returns the item id, or None on escape."""

    BINDINGS = [Binding("escape", "cancel", "close")]

    def __init__(self, title: str, items: list[Item], help: str = "", initial: str = "", current: str | None = None):
        super().__init__()
        self.title_text, self.items, self.help_text, self.initial, self.current = title, items, help, initial, current

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(self.title_text, classes="title")
            yield Input(value=self.initial, placeholder="type to filter", id="filter")
            yield OptionList(id="options")
            if self.help_text:
                yield Static(self.help_text, classes="help")

    def on_mount(self) -> None:
        self._fill(self.initial)
        self.query_one("#filter", Input).focus()

    def _fill(self, needle: str) -> None:
        ol = self.query_one("#options", OptionList)
        ol.clear_options()
        needle = needle.lower().strip()
        group = None
        first = None
        for it in self.items:
            hay = f"{it.label} {it.detail} {it.id} {it.group}".lower()
            if needle and not all(w in hay for w in needle.split()):
                continue
            if it.group and it.group != group:
                group = it.group
                ol.add_option(Option(Text(it.group, style="bold dim"), disabled=True))
            mark = "● " if it.id == self.current else "  "
            label = Text(mark + it.label, style="dim" if it.disabled else "")
            if it.detail:
                label.append("  " + it.detail, style="dim")
            ol.add_option(Option(label, id=it.id, disabled=it.disabled))
            if first is None and not it.disabled:
                first = ol.option_count - 1
        if first is not None:
            ol.highlighted = first

    @on(Input.Changed, "#filter")
    def _changed(self, event: Input.Changed) -> None:
        self._fill(event.value)

    @on(Input.Submitted, "#filter")
    def _submitted(self) -> None:
        ol = self.query_one("#options", OptionList)
        if ol.highlighted is not None:
            opt = ol.get_option_at_index(ol.highlighted)
            if opt.id and not opt.disabled:
                self.dismiss(opt.id)

    @on(OptionList.OptionSelected, "#options")
    def _selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id:
            self.dismiss(event.option.id)

    def on_key(self, event: Any) -> None:
        if event.key in ("down", "up") and self.query_one("#filter", Input).has_focus:
            ol = self.query_one("#options", OptionList)
            ol.action_cursor_down() if event.key == "down" else ol.action_cursor_up()
            event.prevent_default()
            event.stop()

    def action_cancel(self) -> None:
        self.dismiss(None)


class KeyScreen(ModalScreen[str | None]):
    """Enter a secret (masked)."""

    BINDINGS = [Binding("escape", "cancel", "close")]

    def __init__(self, title: str, help: str = ""):
        super().__init__()
        self.title_text, self.help_text = title, help

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(self.title_text, classes="title")
            yield Input(password=True, placeholder="paste the key and press enter", id="key")
            if self.help_text:
                yield Static(self.help_text, classes="help")

    def on_mount(self) -> None:
        self.query_one("#key", Input).focus()

    @on(Input.Submitted, "#key")
    def _submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class QuestionScreen(ModalScreen[list[list[str]] | None]):
    """The engine's ``question`` tool: one list per question; escape dismisses (the model is told)."""

    BINDINGS = [Binding("escape", "cancel", "dismiss")]

    def __init__(self, questions: list[dict[str, Any]]):
        super().__init__()
        self.questions = questions
        self.index = 0
        self.answers: list[list[str]] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static("", classes="title", id="qtitle")
            yield OptionList(id="single")
            yield SelectionList(id="multi")
            yield Input(placeholder="or type your own answer and press enter", id="other")
            yield Static("enter to answer · space to tick (several allowed) · tab to type · esc to dismiss",
                         classes="help")

    def on_mount(self) -> None:
        self._show()

    def _show(self) -> None:
        q = self.questions[self.index]
        n = len(self.questions)
        self.query_one("#qtitle", Static).update(f"{q['question']}" + (f"  ({self.index + 1}/{n})" if n > 1 else ""))
        single, multi = self.query_one("#single", OptionList), self.query_one("#multi", SelectionList)
        single.clear_options()
        multi.clear_options()
        if q.get("multiple"):
            single.display, multi.display = False, True
            multi.add_options([(o, o) for o in q["options"]])
            multi.focus()
        else:
            single.display, multi.display = True, False
            single.add_options([Option(o, id=str(i)) for i, o in enumerate(q["options"])])
            single.highlighted = 0
            single.focus()

    def _next(self, answer: list[str]) -> None:
        self.answers.append(answer)
        self.index += 1
        if self.index >= len(self.questions):
            self.dismiss(self.answers)
        else:
            self._show()

    @on(OptionList.OptionSelected, "#single")
    def _picked(self, event: OptionList.OptionSelected) -> None:
        self._next([self.questions[self.index]["options"][int(event.option.id)]])

    @on(Input.Submitted, "#other")
    def _typed(self, event: Input.Submitted) -> None:
        if event.value.strip():
            event.input.value = ""
            self._next([event.value.strip()])

    def on_key(self, event: Any) -> None:
        if event.key == "enter" and self.query_one("#multi", SelectionList).has_focus:
            self._next([str(v) for v in self.query_one("#multi", SelectionList).selected])
            event.stop()

    def action_cancel(self) -> None:
        self.dismiss(None)


class TextScreen(ModalScreen[None]):
    """A read-only page (help, a job's log)."""

    BINDINGS = [Binding("escape", "close", "close"), Binding("q", "close", "close")]

    def __init__(self, title: str, markdown: str):
        super().__init__()
        self.title_text, self.markdown = title, markdown

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(self.title_text, classes="title")
            with VerticalScroll():
                yield Markdown(self.markdown)
            yield Static("esc to close", classes="help")

    def action_close(self) -> None:
        self.dismiss(None)
