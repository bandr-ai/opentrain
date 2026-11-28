"""Artifacts: the pages that lead the work. Each stage of post-training ends in one, and each one asks
the person for one decision.

An artifact is a versioned page of blocks, kept under ``<state>/artifacts/<id>/v<n>.json``. Two kinds
are built from what OpenTrain already knows (``data_card``: what the model will learn from;
``eval_comparison``: whether it got better). Any other kind (a recipe plan, a run page, a release, or
something no one planned for) is written by the agent from the same blocks, so a new artifact never
needs new code. Every save is a new version; a decision is recorded on the version it was made on.

Blocks (``type`` plus fields):

* ``text``     ``text``: paragraphs; ``**bold**`` and ``code`` render.
* ``stats``    ``items: [{label, value, note?}]``: the numbers that matter, large.
* ``bars``     ``title?``, ``items: [{label, value, max?, note?}]``: one bar per row.
* ``table``    ``columns: [..]``, ``rows: [[..]]``.
* ``samples``  ``items: [{title?, left_label, left, right_label, right}]``: two answers side by side.
* ``list``     ``title?``, ``items: [..]``.

Held-out tasks stay sealed here as everywhere: an artifact may name them and show pass or fail, never
their content or a model's answer to them.
"""

from __future__ import annotations

import html
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .state import write_json

ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
BLOCKS = {
    "text": ("text",),
    "stats": ("items",),
    "bars": ("items",),
    "table": ("columns", "rows"),
    "samples": ("items",),
    "list": ("items",),
}
BUILTIN = ("data_card", "eval_comparison")
MAX_BYTES = 2_000_000


class ArtifactError(ValueError):
    pass


def _root(state: Path) -> Path:
    return state / "artifacts"


AUTO_SECONDS = 30  # a decision with a default takes it when nobody answers in time


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def check_blocks(blocks: Any) -> list[dict[str, Any]]:
    if not isinstance(blocks, list) or not blocks:
        raise ArtifactError("blocks must be a non-empty list")
    for i, b in enumerate(blocks):
        if not isinstance(b, dict) or b.get("type") not in BLOCKS:
            raise ArtifactError(f"block {i}: type must be one of {', '.join(BLOCKS)}")
        missing = [k for k in BLOCKS[b["type"]] if k not in b]
        if missing:
            raise ArtifactError(f"block {i} ({b['type']}): needs {', '.join(missing)}")
        if b["type"] != "text" and not isinstance(b.get("items", b.get("rows")), list):
            raise ArtifactError(f"block {i} ({b['type']}): items/rows must be a list")
    return blocks


def _decision(d: Any) -> dict[str, Any] | None:
    if d is None:
        return None
    if not isinstance(d, dict) or not d.get("question") or not isinstance(d.get("options"), list) \
            or not 2 <= len(d["options"]) <= 5 or not all(isinstance(o, str) and o for o in d["options"]):
        raise ArtifactError("decision needs a question and 2-5 option labels")
    out = {"question": str(d["question"]), "options": list(d["options"])}
    if d.get("default") is not None:
        if d["default"] not in out["options"]:
            raise ArtifactError(f"default {d['default']!r} is not one of the options")
        out.update(default=d["default"], auto_at=time.time() + AUTO_SECONDS)
    return out


def save(state: Path, *, kind: str, title: str, blocks: list[dict[str, Any]], summary: str = "",
         decision: dict[str, Any] | None = None, links: list[str] | None = None, id: str | None = None,
         source: str = "agent") -> dict[str, Any]:
    """Save a new artifact, or a new version of ``id``. Returns it."""
    kind = (kind or "").strip()
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,40}", kind):
        raise ArtifactError("kind is a short snake_case name (data_card, recipe_plan, run_page, ...)")
    if not title or not str(title).strip():
        raise ArtifactError("an artifact needs a title")
    aid = id or _new_id(state, kind)
    if not ID.match(aid):
        raise ArtifactError(f"bad artifact id {aid!r}")
    prev = latest(state, aid)
    if prev and prev["kind"] != kind:
        raise ArtifactError(f"{aid} is a {prev['kind']}, not a {kind}")
    known = {a["id"] for a in list_all(state)}
    links = [x for x in (links or []) if x != aid]
    unknown = [x for x in links if x not in known]
    if unknown:
        raise ArtifactError(f"no artifact {', '.join(unknown)} to link to")
    art = {"id": aid, "kind": kind, "title": str(title).strip()[:200], "summary": str(summary)[:2000],
           "version": (prev["version"] + 1) if prev else 1, "created": _now(), "source": source,
           "blocks": check_blocks(blocks), "decision": _decision(decision), "links": links}
    if len(json.dumps(art, default=str)) > MAX_BYTES:
        raise ArtifactError("artifact too large (2 MB): clip samples or tables")
    write_json(_root(state) / aid / f"v{art['version']}.json", art)
    return art


def _new_id(state: Path, kind: str) -> str:
    base = kind.replace("_", "-")
    n = 1
    while (_root(state) / f"{base}-{n}").exists():
        n += 1
    return f"{base}-{n}"


def _versions(state: Path, aid: str) -> list[Path]:
    d = _root(state) / aid
    if not ID.match(aid) or not d.is_dir():
        return []
    return sorted(d.glob("v*.json"), key=lambda p: int(p.stem[1:]) if p.stem[1:].isdigit() else 0)


def latest(state: Path, aid: str) -> dict[str, Any] | None:
    vs = _versions(state, aid)
    return json.loads(vs[-1].read_text()) if vs else None


def get(state: Path, aid: str, version: int | None = None) -> dict[str, Any]:
    vs = _versions(state, aid)
    if not vs:
        raise ArtifactError(f"no artifact {aid!r}")
    if version is None:
        return json.loads(vs[-1].read_text())
    p = _root(state) / aid / f"v{int(version)}.json"
    if not p.exists():
        raise ArtifactError(f"{aid} has no version {version} (latest {len(vs)})")
    return json.loads(p.read_text())


def list_all(state: Path) -> list[dict[str, Any]]:
    """Every artifact's latest version, newest first, without its blocks."""
    out = []
    root = _root(state)
    for d in root.iterdir() if root.is_dir() else []:
        a = latest(state, d.name)
        if a:
            out.append({k: a.get(k) for k in ("id", "kind", "title", "summary", "version", "created", "source",
                                               "links")} | {"decision": a.get("decision")})
    return sorted(out, key=lambda a: a["created"] or "", reverse=True)


def decide(state: Path, aid: str, option: str, note: str = "", by: str = "person") -> dict[str, Any]:
    """Record the decision on the latest version (in place: it is part of that version's record)."""
    vs = _versions(state, aid)
    if not vs:
        raise ArtifactError(f"no artifact {aid!r}")
    art = json.loads(vs[-1].read_text())
    d = art.get("decision")
    if not d:
        raise ArtifactError(f"{aid} asks for no decision")
    if option not in d["options"]:
        raise ArtifactError(f"choose one of: {', '.join(d['options'])}")
    d.update(chosen=option, note=str(note)[:2000], by=by, at=_now(), auto_at=None)
    write_json(vs[-1], art)
    return art


def hold(state: Path, aid: str) -> dict[str, Any]:
    """The person is looking at it: stop the countdown to the default and wait for them."""
    vs = _versions(state, aid)
    if not vs:
        raise ArtifactError(f"no artifact {aid!r}")
    art = json.loads(vs[-1].read_text())
    if (art.get("decision") or {}).get("auto_at"):
        art["decision"]["auto_at"] = None
        write_json(vs[-1], art)
    return art


def decide_by_default(state: Path, aid: str, version: int) -> dict[str, Any] | None:
    """Take the default if this version's decision is still open and nobody stopped the countdown."""
    a = latest(state, aid)
    d = (a or {}).get("decision") or {}
    if not a or a["version"] != version or d.get("chosen") or not d.get("auto_at") or not d.get("default"):
        return None
    return decide(state, aid, d["default"], by="default")


# ---------------------------------------------------------------------------- built-in artifacts


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    except (OSError, ValueError):
        return []


AGENTS = {"claude-code": "Claude Code", "codex": "Codex", "opencode": "opencode", "gemini-cli": "Gemini CLI",
          "cursor": "Cursor", "cline": "Cline", "aider": "aider", "pi": "pi", "copilot": "GitHub Copilot"}
MOMENTS = {"pushback": "pushback, to review", "interrupted": "stopped by you", "failing_check": "a check kept failing",
           "churn": "same file, again", "tool_errors": "tool errors"}


def data_card(state: Path, *, id: str = "data-card") -> dict[str, Any]:
    """What the model will learn from: the imported traces, how each request ended, what is trained."""
    from collections import Counter

    d = state / "traces"
    index, examples, moments = _jsonl(d / "index.jsonl"), _jsonl(d / "examples.jsonl"), _jsonl(d / "moments.jsonl")
    if not index:
        raise ArtifactError("no traces imported yet: run traces_import (or `opentrain traces --import`) first")
    coding = [r for r in index if r.get("edits")]
    ends = [(e.get("meta") or {}).get("outcome") or {} for e in examples]
    well = sum(1 for o in ends if o.get("ended_well", True))
    lengths = sorted(len(json.dumps(e.get("messages", []))) // 4 for e in examples)
    tool_names = Counter(c["function"]["name"] for e in examples for m in e.get("messages", [])
                         if m.get("tool_calls") for c in m["tool_calls"])
    by_agent = Counter(r["agent"] for r in index)
    by_repo = Counter(r.get("repo") or "(no repo)" for r in coding)
    why = Counter(k for o in ends for k, bad in (("pushback after", o.get("pushback")),
                                                 ("stopped", o.get("interrupted")),
                                                 ("check failing at the end", o.get("checks") is False)) if bad)
    mom = Counter(m.get("kind") for m in moments)
    pct = lambda q: lengths[min(len(lengths) - 1, int(len(lengths) * q))] if lengths else 0  # noqa: E731
    top = max(by_agent.values()) if by_agent else 1
    blocks: list[dict[str, Any]] = [
        {"type": "stats", "items": [
            {"label": "work sessions", "value": len(index), "note": f"from {len(by_agent)} agent(s)"},
            {"label": "changed code", "value": len(coding), "note": f"in {len([r for r in by_repo if r != '(no repo)'])} repos"},
            {"label": "requests to train on", "value": well, "note": f"of {len(examples)}: the ones that ended well"},
            {"label": "tokens per request", "value": f"{pct(0.5):,}", "note": f"median; 90% under {pct(0.9):,}"}]},
        {"type": "bars", "title": "Where the sessions came from",
         "items": [{"label": AGENTS.get(a, a), "value": n, "max": top} for a, n in by_agent.most_common()]},
        {"type": "bars", "title": "Repos with code changes (sessions)",
         "items": [{"label": Path(r).name if r != "(no repo)" else r, "value": n, "max": max(by_repo.values())}
                   for r, n in by_repo.most_common(12)]},
        {"type": "bars", "title": "Requests left out, and why",
         "items": [{"label": k, "value": n, "max": max(len(examples), 1)} for k, n in why.most_common()]
         or [{"label": "none left out", "value": 0, "max": 1}]},
        {"type": "bars", "title": "What the model learns to do (tool calls)",
         "items": [{"label": k, "value": n, "max": max(tool_names.values())} for k, n in tool_names.most_common()]},
        {"type": "bars", "title": "Worth a closer look (moments)",
         "items": [{"label": MOMENTS.get(k, k), "value": n, "max": max(mom.values())} for k, n in mom.most_common()]
         or [{"label": "none found", "value": 0, "max": 1}]},
        {"type": "text", "text": "**How it is cleaned.** Credentials are scrubbed. Commands are written from the repo "
                                 "root and your home folder as `~`, so the model does not learn this machine's "
                                 "folders. Only the agent's turns are trained; your prompts and tool output are "
                                 "context. A request is trained when it changed code and ended well: no pushback "
                                 "after it, not stopped, no check failing at the end."},
    ]
    themes = []
    try:
        themes = json.loads((d / "themes.json").read_text())
    except (OSError, ValueError):
        pass
    if themes:
        blocks.append({"type": "table", "title": "What went wrong, read from the moments",
                       "columns": ["theme", "sessions", "training fixes it?", "the fix", "evidence"],
                       "rows": [[t.get("name", ""), t.get("sessions", t.get("count", "")),
                                 "yes" + (" (chosen)" if t.get("chosen") else "") if t.get("trainable", True)
                                 else "no: a rule", str(t.get("fix", ""))[:200],
                                 ", ".join(map(str, (t.get("evidence") or [])[:6]))
                                 + (" ..." if len(t.get("evidence") or []) > 6 else "")] for t in themes]})
    sample = next((e for e in examples if ((e.get("meta") or {}).get("outcome") or {}).get("ended_well", True)), None)
    if sample:
        msgs = sample["messages"]
        user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        calls = [c["function"] for m in msgs if m.get("tool_calls") for c in m["tool_calls"]]
        steps = "\n".join(f"{c['name']} {_short(c['arguments'])}" for c in calls[:12])
        blocks.append({"type": "samples", "items": [{"title": "One training example", "left_label": "your request",
                                                     "left": user[:1500], "right_label": "what the model learns to do",
                                                     "right": steps[:3000]}]})
    return save(state, kind="data_card", id=id, source="opentrain",
                title=f"Your coding data: {well} requests to train on",
                summary=f"{len(index)} sessions from {', '.join(AGENTS.get(a, a) for a in by_agent)}; "
                        f"{len(coding)} changed code; {well} of {len(examples)} requests ended well and are trained.",
                blocks=blocks,
                decision={"question": "Train on this data?",
                          "options": ["Train on it", "Leave some out first", "Add public coding data too"]})


def _short(args: str, n: int = 140) -> str:
    try:
        a = json.loads(args)
    except ValueError:
        return args[:n]
    v = a.get("command") or a.get("path") or a.get("summary") or json.dumps(a)
    v = str(v).replace("\n", " ")
    return v if len(v) <= n else v[:n] + "..."


def eval_comparison(state: Path, *, before: str = "base", after: str, id: str | None = None,
                    links: list[str] | None = None) -> dict[str, Any]:
    """Did it get better: two score sets on the same held-out tasks, task by task."""
    from .web.api import version_scores

    a, b = version_scores(state, before), version_scores(state, after)
    if not b:
        raise ArtifactError(f"no held-out scores for {after!r} yet: run eval_request first")
    if not a:
        raise ArtifactError(f"no held-out scores for {before!r} yet: score it first")
    tasks = sorted(set(a) | set(b))
    both = [t for t in tasks if t in a and t in b]
    pa = sum(1 for t in both if a[t]["passed"])
    pb = sum(1 for t in both if b[t]["passed"])
    fixed = [t for t in both if b[t]["passed"] and not a[t]["passed"]]
    broke = [t for t in both if a[t]["passed"] and not b[t]["passed"]]
    n = max(len(both), 1)

    def secs(s: dict, t: str) -> str:
        v = s.get(t, {}).get("seconds")
        return f"{v:.0f}s" if isinstance(v, (int, float)) else "-"

    mark = lambda s, t: "pass" if s.get(t, {}).get("passed") else ("fail" if t in s else "-")  # noqa: E731
    order = broke + fixed + [t for t in both if t not in fixed and t not in broke]
    verdict = ("better" if pb > pa and not broke else "worse" if pb < pa else
               "mixed" if broke and fixed else "the same")
    blocks = [
        {"type": "stats", "items": [
            {"label": before, "value": f"{pa}/{len(both)}", "note": f"{pa / n:.0%} of held-out tasks"},
            {"label": after, "value": f"{pb}/{len(both)}", "note": f"{pb / n:.0%} of held-out tasks"},
            {"label": "fixed", "value": len(fixed), "note": "failed before, pass now"},
            {"label": "regressed", "value": len(broke), "note": "passed before, fail now"}]},
        {"type": "bars", "title": "Held-out pass rate",
         "items": [{"label": before, "value": round(pa / n * 100), "max": 100, "note": "%"},
                   {"label": after, "value": round(pb / n * 100), "max": 100, "note": "%"}]},
        {"type": "table", "columns": ["task", before, after, f"{before} time", f"{after} time"],
         "rows": [[t, mark(a, t), mark(b, t), secs(a, t), secs(b, t)] for t in order[:200]]},
        {"type": "text", "text": "Regressions are listed first. The held-out tasks stay sealed: their content and the "
                                 "models' answers never reach the agent that writes training data, so these scores "
                                 "cannot be trained on. Only pass, fail and time leave the evaluator."},
    ]
    if len(both) < len(tasks):
        blocks.insert(2, {"type": "text", "text": f"{len(tasks) - len(both)} task(s) were scored for only one of the "
                                                  "two and are not counted."})
    return save(state, kind="eval_comparison", id=id or f"eval-{_slug(before)}-vs-{_slug(after)}", source="opentrain",
                title=f"{after} vs {before}: {verdict}",
                summary=f"{after} passes {pb}/{len(both)} held-out tasks, {before} {pa}/{len(both)}; "
                        f"{len(fixed)} fixed, {len(broke)} regressed.",
                blocks=blocks, links=links,
                decision={"question": f"What next for {after}?",
                          "options": ["Ship it", "Improve the data and train again", "Roll back"]})


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9.-]+", "-", s.lower()).strip("-")[:24] or "x"


# ---------------------------------------------------------------------------- a page to share


CSS = """:root{--bg:#fbfaf8;--fg:#1d1c1a;--mute:#6b6862;--line:#e6e2db;--bar:#c96442;--card:#fff;--good:#2f7d4f;--bad:#b4443a}
@media (prefers-color-scheme:dark){:root{--bg:#1c1b19;--fg:#ecebe8;--mute:#a19e97;--line:#36342f;--bar:#e07a55;--card:#24231f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 ui-sans-serif,system-ui,sans-serif}
main{max-width:880px;margin:0 auto;padding:32px 16px 64px}h1{font-size:26px;margin:0 0 6px}h2{font-size:15px;margin:28px 0 10px}
.meta,.note{color:var(--mute);font-size:13px}.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:18px 0}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}.stat b{display:block;font-size:24px}
.bar{display:grid;grid-template-columns:minmax(90px,30%) 1fr 64px;gap:10px;align-items:center;margin:5px 0;font-size:14px}
.track{height:10px;background:var(--line);border-radius:5px;overflow:hidden}.fill{height:100%;background:var(--bar)}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left}
.pass{color:var(--good)}.fail{color:var(--bad)}.pair{display:grid;grid-template-columns:1fr 1fr;gap:10px}
@media (max-width:640px){.pair{grid-template-columns:1fr}}pre{white-space:pre-wrap;background:var(--card);border:1px solid var(--line);
border-radius:8px;padding:10px;font-size:12.5px;margin:4px 0}code{font-size:.92em}.decision{margin-top:28px;padding:14px;border:1px solid var(--line);border-radius:10px;background:var(--card)}"""


def _inline(text: str) -> str:
    t = html.escape(str(text))
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", t)


def _block_html(b: dict[str, Any]) -> str:
    t = b["type"]
    head = f"<h2>{html.escape(str(b['title']))}</h2>" if b.get("title") else ""
    if t == "text":
        return "".join(f"<p>{_inline(p)}</p>" for p in str(b["text"]).split("\n\n"))
    if t == "stats":
        return '<div class="stats">' + "".join(
            f'<div class="stat"><span class="note">{html.escape(str(i.get("label", "")))}</span>'
            f'<b>{html.escape(str(i.get("value", "")))}</b><span class="note">{html.escape(str(i.get("note", "")))}</span></div>'
            for i in b["items"]) + "</div>"
    if t == "bars":
        rows = []
        for i in b["items"]:
            v, m = float(i.get("value") or 0), float(i.get("max") or 0) or max(float(x.get("value") or 0) for x in b["items"]) or 1
            rows.append(f'<div class="bar"><span>{html.escape(str(i.get("label", "")))}</span><div class="track">'
                        f'<div class="fill" style="width:{min(100, v / m * 100):.1f}%"></div></div>'
                        f'<span class="note">{html.escape(str(i.get("value", "")))}{html.escape(str(i.get("note", "")))}</span></div>')
        return head + "".join(rows)
    if t == "table":
        cell = lambda c: f'<td class="{c}">{html.escape(str(c))}</td>' if c in ("pass", "fail") else f"<td>{html.escape(str(c))}</td>"  # noqa: E731
        return head + "<table><tr>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in b["columns"]) + "</tr>" + "".join(
            "<tr>" + "".join(cell(c) for c in r) + "</tr>" for r in b["rows"]) + "</table>"
    if t == "samples":
        return head + "".join(
            (f"<h2>{html.escape(str(i['title']))}</h2>" if i.get("title") else "")
            + f'<div class="pair"><div><span class="note">{html.escape(str(i.get("left_label", "")))}</span>'
            f'<pre>{html.escape(str(i.get("left", "")))}</pre></div><div><span class="note">'
            f'{html.escape(str(i.get("right_label", "")))}</span><pre>{html.escape(str(i.get("right", "")))}</pre></div></div>'
            for i in b["items"])
    if t == "list":
        return head + "<ul>" + "".join(f"<li>{_inline(i)}</li>" for i in b["items"]) + "</ul>"
    return ""


def to_html(art: dict[str, Any]) -> str:
    """The artifact as one self-contained page (no scripts, no outside files), to share or keep."""
    d = art.get("decision") or {}
    dec = ""
    if d:
        chosen = f"<p><b>Decided: {html.escape(d['chosen'])}</b> {html.escape(d.get('note') or '')}</p>" if d.get("chosen") \
            else "<p class=\"note\">Open in OpenTrain to decide.</p>"
        dec = (f'<div class="decision"><b>{html.escape(d["question"])}</b><p class="note">'
               f'{" · ".join(html.escape(o) for o in d["options"])}</p>{chosen}</div>')
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,"
            f"initial-scale=1'><title>{html.escape(art['title'])}</title><style>{CSS}</style></head><body><main>"
            f"<div class=meta>OpenTrain · {html.escape(art['kind'].replace('_', ' '))} · v{art['version']} · "
            f"{html.escape(art['created'])}</div><h1>{html.escape(art['title'])}</h1>"
            f"<p class=note>{html.escape(art.get('summary') or '')}</p>"
            + "".join(_block_html(b) for b in art["blocks"]) + dec + "</main></body></html>")
