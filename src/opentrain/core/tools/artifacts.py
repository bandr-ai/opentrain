"""Agent tools for artifacts: the pages that lead post-training (see ``opentrain.artifacts``)."""

from __future__ import annotations

import time
from typing import Any

from ... import artifacts as art
from ..tool import Tool, ToolError, schema


def _state(ctx):
    return ctx.project.state


def _out(a: dict[str, Any]) -> dict[str, Any]:
    d = a.get("decision") or {}
    return {"id": a["id"], "kind": a["kind"], "version": a["version"], "title": a["title"], "summary": a["summary"],
            "open": f"#/a/{a['id']}", "decision": d.get("question"), "options": d.get("options"),
            "chosen": d.get("chosen"), "default": d.get("default")}


def _guard(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except art.ArtifactError as exc:
        raise ToolError(str(exc)) from None


def data_card(ctx) -> Any:
    return _out(_guard(art.data_card, _state(ctx)))


def eval_comparison(ctx, after: str, before: str = "base", links: list[str] | None = None) -> Any:
    return _out(_guard(art.eval_comparison, _state(ctx), before=before, after=after, links=links))


def _take_default_later(ctx, a: dict[str, Any]) -> None:
    """If nobody decides within AUTO_SECONDS, take the default and tell the agent, so the work goes on."""
    d = a.get("decision") or {}
    if not d.get("auto_at") or ctx.loop is None:
        return
    state, sid, aid, version = _state(ctx), ctx.session_id, a["id"], a["version"]
    since = time.time()

    def fire() -> None:
        runner = ctx.project.services.get("runner")
        if runner is None or runner.stepped_in(sid, since):  # they interrupted or wrote: theirs to decide
            return
        try:
            done = art.decide_by_default(state, aid, version)
        except art.ArtifactError:
            return
        if done is not None:
            runner.prompt(sid, f"{d['question']} {d['default']}. (Taken by default: nobody answered within "
                               f"{art.AUTO_SECONDS} seconds; on {a['title']}, #/a/{aid}.) Go on with it; the person "
                               "can still change course by writing.", synthetic=True)

    ctx.loop.call_soon_threadsafe(ctx.loop.call_later, art.AUTO_SECONDS, fire)


def artifact_make(ctx, kind: str, title: str, blocks: list[dict[str, Any]], summary: str = "",
                  decision: dict[str, Any] | None = None, links: list[str] | None = None,
                  id: str | None = None) -> Any:
    if kind in art.BUILTIN:
        raise ToolError(f"{kind} is built from the project's own data: call the {kind} tool")
    if decision and getattr(ctx.agent, "name", "") == "outside":  # no OpenTrain turn to hand a default to
        decision = {k: v for k, v in decision.items() if k != "default"}
    a = _guard(art.save, _state(ctx), kind=kind, title=title, blocks=blocks, summary=summary,
               decision=decision, links=links, id=id)
    _take_default_later(ctx, a)
    return _out(a)


def artifacts_list(ctx) -> Any:
    return {"artifacts": [{**a, "open": f"#/a/{a['id']}"} for a in art.list_all(_state(ctx))]}


def artifact_show(ctx, id: str, version: int | None = None) -> Any:
    a = _guard(art.get, _state(ctx), id, version)
    return {**_out(a), "links": a["links"], "decided": a.get("decision"), "blocks": a["blocks"]}


BLOCK = {"type": "object", "description": "One block: {type: text|stats|bars|chart|table|samples|list|callout|html, "
         "title?, ...}. text: {text}; stats: {items:[{label,value,note?,tone?}]}; bars: {items:[{label,value,max?,"
         "note?,tone?}]}; chart: {kind: line|bar, series:[{name, points:[[x,y],..]}], unit?, x_label?, y_label?, "
         "note?, max?, zero?}; table: {columns, rows}; samples: {items:[{title?,left_label,left,right_label,right}]}; "
         "list: {items:[..]}; callout: {tone: good|warn|bad|info, title?, text}; html: {html} (static HTML and CSS, "
         "no scripts or outside files). tone is good|warn|bad|info."}


def tools() -> list[Tool]:
    S = schema
    T = lambda *a, **k: Tool(*a, group="train", **k)  # noqa: E731
    strs = {"type": "array", "items": {"type": "string"}}
    return [
        T("data_card", "Make (or refresh) the data card: the artifact that shows what the model will learn from "
          "(imported traces: sources, repos, requests that ended well and are trained, what was left out and why, "
          "tool mix, moments, one example) and asks the person whether to train on it. Make it after traces_import "
          "and before data_build. It opens for the person with its question as buttons: reply in two lines at most "
          "and ask nothing yourself.", S(), data_card),
        T("eval_comparison", "Make the eval comparison: the artifact that shows whether a version got better than "
          "`before` (default the untrained base) on the held-out tasks, task by task, regressions first, and asks "
          "ship / improve the data / roll back. Make it after eval_request for both; link the data card and the "
          "recipe plan it came from.",
          S({"after": {"type": "string"}, "before": {"type": "string"}, "links": strs}, ["after"]), eval_comparison),
        T("artifact_make", "Make an artifact of any other kind, or a new version of one (pass its id): a page of "
          "blocks that leads one step and asks the person one decision. Use it at each milestone: recipe_plan "
          "(base model, method stages, settings, trainer, cost and time, which tests gate it; before spending), "
          "run_page (curves as steps and loss, cost so far), release (scores, limits, download and serving), "
          "cost_ledger, or a new kind when the work needs one. Load the artifact/design skill first: it says how a "
          "page reads well (summary, one callout, a few numbers, then evidence as charts, tables or side-by-sides). "
          "Show evidence, not claims. Never put held-out task content or answers in an artifact.",
          S({"kind": {"type": "string", "description": "snake_case: recipe_plan, run_page, release, cost_ledger, ..."},
             "title": {"type": "string"}, "summary": {"type": "string"},
             "blocks": {"type": "array", "items": BLOCK},
             "decision": {"type": "object", "description": "{question, options: [2-5 short labels], default?}. "
                          "default (one of options, your recommendation) is taken after 30 s without an "
                          "answer: give one only when the choice is free and easy to undo, never for spending, "
                          "accounts, sending data out, deleting or what goes live."},
             "links": strs, "id": {"type": "string", "description": "an existing artifact's id: save a new version"}},
            ["kind", "title", "blocks"]), artifact_make),
        T("artifacts", "The project's artifacts, newest first, with the decision each one asks and what the person "
          "chose.", S(), artifacts_list),
        T("artifact_show", "One artifact in full (a version, default the latest), including the person's decision "
          "and note.", S({"id": {"type": "string"}, "version": {"type": "integer"}}, ["id"]), artifact_show),
    ]
