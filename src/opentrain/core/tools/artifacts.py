"""Agent tools for artifacts: the pages that lead post-training (see ``opentrain.artifacts``)."""

from __future__ import annotations

from typing import Any

from ... import artifacts as art
from ..tool import Tool, ToolError, schema


def _state(ctx):
    return ctx.project.state


def _out(a: dict[str, Any]) -> dict[str, Any]:
    d = a.get("decision") or {}
    return {"id": a["id"], "kind": a["kind"], "version": a["version"], "title": a["title"], "summary": a["summary"],
            "open": f"#/a/{a['id']}", "decision": d.get("question"), "options": d.get("options"),
            "chosen": d.get("chosen")}


def _guard(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except art.ArtifactError as exc:
        raise ToolError(str(exc)) from None


def data_card(ctx) -> Any:
    return _out(_guard(art.data_card, _state(ctx)))


def eval_comparison(ctx, after: str, before: str = "base", links: list[str] | None = None) -> Any:
    return _out(_guard(art.eval_comparison, _state(ctx), before=before, after=after, links=links))


def artifact_make(ctx, kind: str, title: str, blocks: list[dict[str, Any]], summary: str = "",
                  decision: dict[str, Any] | None = None, links: list[str] | None = None,
                  id: str | None = None) -> Any:
    if kind in art.BUILTIN:
        raise ToolError(f"{kind} is built from the project's own data: call the {kind} tool")
    return _out(_guard(art.save, _state(ctx), kind=kind, title=title, blocks=blocks, summary=summary,
                       decision=decision, links=links, id=id))


def artifacts_list(ctx) -> Any:
    return {"artifacts": [{**a, "open": f"#/a/{a['id']}"} for a in art.list_all(_state(ctx))]}


def artifact_show(ctx, id: str, version: int | None = None) -> Any:
    a = _guard(art.get, _state(ctx), id, version)
    return {**_out(a), "links": a["links"], "decided": a.get("decision"), "blocks": a["blocks"]}


BLOCK = {"type": "object", "description": "One block: {type: text|stats|bars|table|samples|list, ...}. text: {text}; "
         "stats: {items:[{label,value,note?}]}; bars: {title?, items:[{label,value,max?,note?}]}; table: {columns, rows}; "
         "samples: {items:[{title?,left_label,left,right_label,right}]}; list: {title?, items:[..]}"}


def tools() -> list[Tool]:
    S = schema
    T = lambda *a, **k: Tool(*a, group="train", **k)  # noqa: E731
    strs = {"type": "array", "items": {"type": "string"}}
    return [
        T("data_card", "Make (or refresh) the data card: the artifact that shows what the model will learn from "
          "(imported traces: sources, repos, requests that ended well and are trained, what was left out and why, "
          "tool mix, moments, one example) and asks the person whether to train on it. Make it after traces_import "
          "and before data_build; point the person to it.", S(), data_card),
        T("eval_comparison", "Make the eval comparison: the artifact that shows whether a version got better than "
          "`before` (default the untrained base) on the held-out tasks, task by task, regressions first, and asks "
          "ship / improve the data / roll back. Make it after eval_request for both; link the data card and the "
          "recipe plan it came from.",
          S({"after": {"type": "string"}, "before": {"type": "string"}, "links": strs}, ["after"]), eval_comparison),
        T("artifact_make", "Make an artifact of any other kind, or a new version of one (pass its id): a page of "
          "blocks that leads one step and asks the person one decision. Use it at each milestone: recipe_plan "
          "(base model, method stages, settings, trainer, cost and time, which tests gate it; before spending), "
          "run_page (curves as steps and loss, cost so far), release (scores, limits, download and serving), "
          "cost_ledger, or a new kind when the work needs one. Show evidence (numbers, samples), not claims. "
          "Never put held-out task content or answers in an artifact.",
          S({"kind": {"type": "string", "description": "snake_case: recipe_plan, run_page, release, cost_ledger, ..."},
             "title": {"type": "string"}, "summary": {"type": "string"},
             "blocks": {"type": "array", "items": BLOCK},
             "decision": {"type": "object", "description": "{question, options: [2-5 short labels]}"},
             "links": strs, "id": {"type": "string", "description": "an existing artifact's id: save a new version"}},
            ["kind", "title", "blocks"]), artifact_make),
        T("artifacts", "The project's artifacts, newest first, with the decision each one asks and what the person "
          "chose.", S(), artifacts_list),
        T("artifact_show", "One artifact in full (a version, default the latest), including the person's decision "
          "and note.", S({"id": {"type": "string"}, "version": {"type": "integer"}}, ["id"]), artifact_show),
    ]
