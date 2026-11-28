"""Artifacts: the pages that lead the work. Each stage of post-training ends in one. A page asks the
person something only when it is theirs to decide (spending on a plan, what goes live); everything
else the agent decides and the page just shows.

An artifact is a versioned page of blocks, kept under ``<state>/artifacts/<id>/v<n>.json``. Two kinds
are built from what OpenTrain already knows (``data_card``: what the model will learn from;
``eval_comparison``: whether it got better). Any other kind (a recipe plan, a run page, a release, or
something no one planned for) is written by the agent from the same blocks, so a new artifact never
needs new code. Every save is a new version; a decision is recorded on the version it was made on.

A decision may carry ``approves: {needs, cost_usd}``: choosing its first option is the person's
go-ahead for that whole plan, so its gated actions and spending (up to about ``cost_usd``) are granted
to the session that made the page and nothing asks again for them.

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

import functools
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .state import write_json

ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
BLOCKS = {
    "text": ("text",),
    "stats": ("items",),
    "bars": ("items",),
    "chart": ("series",),
    "table": ("columns", "rows"),
    "samples": ("items",),
    "list": ("items",),
    "callout": ("text",),
    "html": ("html",),
}
LISTS = {"stats": "items", "bars": "items", "table": "rows", "samples": "items", "list": "items", "chart": "series"}
MAX_HTML = 200_000
BUILTIN = ("data_card", "eval_comparison")
MAX_BYTES = 2_000_000


class ArtifactError(ValueError):
    pass


def _root(state: Path) -> Path:
    return state / "artifacts"


_LOCK = threading.RLock()  # one writer at a time: a save and a click never overwrite each other


def _locked(fn):
    @functools.wraps(fn)
    def run(*a: Any, **kw: Any) -> Any:
        with _LOCK:
            return fn(*a, **kw)
    return run


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
        key = LISTS.get(b["type"])
        if key and not isinstance(b.get(key), list):
            raise ArtifactError(f"block {i} ({b['type']}): {key} must be a list")
        if b["type"] == "chart":
            if b.get("kind", "line") not in ("line", "bar"):
                raise ArtifactError(f"block {i} (chart): kind is line or bar")
            for s in b["series"]:
                pts = s.get("points") if isinstance(s, dict) else None
                if not isinstance(pts, list) or not pts or not all(isinstance(p, list) and len(p) == 2 for p in pts):
                    raise ArtifactError(f"block {i} (chart): each series needs points: [[x, y], ...]")
        if b["type"] == "html":
            if not isinstance(b["html"], str) or len(b["html"]) > MAX_HTML:
                raise ArtifactError(f"block {i} (html): a string of static HTML and CSS, at most 200 KB")
            from .artifact_page import clean_html
            b["html"] = clean_html(b["html"])
        if b["type"] == "callout" and b.get("tone", "info") not in ("good", "warn", "bad", "info"):
            raise ArtifactError(f"block {i} (callout): tone is good, warn, bad or info")
    return blocks


def _decision(d: Any) -> dict[str, Any] | None:
    if d is None:
        return None
    if not isinstance(d, dict) or not d.get("question") or not isinstance(d.get("options"), list) \
            or not 2 <= len(d["options"]) <= 5 or not all(isinstance(o, str) and o for o in d["options"]):
        raise ArtifactError("decision needs a question and 2-5 option labels")
    if d.get("default") is not None:
        raise ArtifactError("a page asks only what the person must decide (spending, what goes live). If you "
                            "would pick a default, decide it yourself, leave out the decision and say what you chose")
    out = {"question": str(d["question"]), "options": list(d["options"])}
    if d.get("approves") is not None:
        from .core.proposals import GRANTABLE

        ap = d["approves"]
        needs = ap.get("needs") if isinstance(ap, dict) else None
        cost = ap.get("cost_usd", 0) if isinstance(ap, dict) else None
        if not isinstance(needs, list) or not needs or any(n not in GRANTABLE for n in needs) \
                or not isinstance(cost, (int, float)) or cost < 0:
            raise ArtifactError(f"approves is {{needs: [{', '.join(GRANTABLE)}], cost_usd}}: what choosing the "
                                "first option lets the plan do without asking again")
        out["approves"] = {"needs": list(dict.fromkeys(needs)), "cost_usd": round(float(cost), 4)}
    return out


@_locked
def save(state: Path, *, kind: str, title: str, blocks: list[dict[str, Any]], summary: str = "",
         decision: dict[str, Any] | None = None, links: list[str] | None = None, id: str | None = None,
         source: str = "agent", session_id: str | None = None) -> dict[str, Any]:
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
    if session_id:
        art["session_id"] = session_id
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


@_locked
def decide(state: Path, aid: str, option: str, note: str = "", by: str = "person",
           version: int | None = None) -> dict[str, Any]:
    """Record the decision on the latest version (in place: it is part of that version's record).
    With ``version``, only if that is still the latest."""
    vs = _versions(state, aid)
    if not vs:
        raise ArtifactError(f"no artifact {aid!r}")
    art = json.loads(vs[-1].read_text())
    if version is not None and art["version"] != version:
        raise ArtifactError(f"{aid} has a newer version (v{art['version']})")
    d = art.get("decision")
    if not d:
        raise ArtifactError(f"{aid} asks for no decision")
    if option not in d["options"]:
        raise ArtifactError(f"choose one of: {', '.join(d['options'])}")
    if d.get("chosen"):
        raise ArtifactError(f"{aid} was already decided: {d['chosen']}")
    d.update(chosen=option, note=str(note)[:2000], by=by, at=_now())
    write_json(vs[-1], art)
    return art


def approval(art: dict[str, Any]) -> dict[str, Any] | None:
    """What the decision just made approves: its ``approves`` when the first option (the go-ahead) was chosen."""
    d = art.get("decision") or {}
    if d.get("approves") and d.get("chosen") == d["options"][0]:
        return d["approves"]
    return None


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
    # one row per repo by name: several checkouts or worktrees of one repo are the same repo
    by_repo = Counter(Path(r["repo"]).name if r.get("repo") else "(no repo)" for r in coding)
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
            {"label": "requests to train on", "value": well, "tone": "good",
             "note": f"of {len(examples)}: the ones that ended well"},
            {"label": "tokens per request", "value": f"{pct(0.5):,}", "note": f"median; 90% under {pct(0.9):,}"}]},
        *([{"type": "callout", "tone": "warn", "title": f"{by_repo['(no repo)']} of {len(coding)} sessions that "
                                                         "changed code have no repo",
            "text": "Their changes cannot be checked against a commit, so they can teach the model but cannot "
                    "become test tasks."}] if by_repo.get("(no repo)", 0) * 3 >= max(len(coding), 1) else []),
        {"type": "bars", "title": "Where the sessions came from",
         "items": [{"label": AGENTS.get(a, a), "value": n, "max": top} for a, n in by_agent.most_common()]},
        {"type": "bars", "title": "Repos with code changes (sessions)",
         "items": [{"label": r, "value": n, "max": max(by_repo.values())}
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
                       "columns": ["theme", "sessions", "training fixes it?", "the fix", "moments cited"],
                       "rows": [[t.get("name", ""), t.get("sessions", t.get("count", "")),
                                 "yes" + (" (chosen)" if t.get("chosen") else "") if t.get("trainable", True)
                                 else "no: a rule", str(t.get("fix", ""))[:200],
                                 len(t.get("evidence") or [])] for t in themes]})
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
                blocks=blocks)


def _short(args: str, n: int = 140) -> str:
    try:
        a = json.loads(args)
    except ValueError:
        return args[:n]
    v = a.get("command") or a.get("path") or a.get("summary") or json.dumps(a)
    v = str(v).replace("\n", " ")
    return v if len(v) <= n else v[:n] + "..."


def eval_comparison(state: Path, *, before: str = "base", after: str, id: str | None = None,
                    links: list[str] | None = None, session_id: str | None = None) -> dict[str, Any]:
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
        {"type": "chart", "kind": "bar", "title": "Held-out pass rate", "unit": "%", "max": 100,
         "series": [{"name": "pass rate", "points": [[before, round(pa / n * 100)], [after, round(pb / n * 100)]]}]},
        {"type": "table", "columns": ["task", before, after, f"{before} time", f"{after} time"],
         "rows": [[t, mark(a, t), mark(b, t), secs(a, t), secs(b, t)] for t in order[:200]]},
        {"type": "text", "text": "Regressions are listed first. The held-out tasks stay sealed: their content and the "
                                 "models' answers never reach the agent that writes training data, so these scores "
                                 "cannot be trained on. Only pass, fail and time leave the evaluator."},
    ]
    tone = {"better": "good", "worse": "bad"}.get(verdict, "warn")
    say = {"better": f"{after} passes {len(fixed)} more held-out task(s) and breaks none.",
           "worse": f"{after} passes fewer held-out tasks than {before}.",
           "mixed": f"{after} fixes {len(fixed)} task(s) but breaks {len(broke)}.",
           "the same": f"{after} passes the same held-out tasks as {before}."}[verdict]
    if len(both) < 10:
        say += (f"\n\nWith {len(both)} held-out tasks, one task is {100 / n:.0f} points: a difference of one or two "
                "tasks can be chance.")
    blocks.insert(0, {"type": "callout", "tone": tone, "title": f"{after} is {verdict}", "text": say})
    if len(both) < len(tasks):
        blocks.insert(2, {"type": "text", "text": f"{len(tasks) - len(both)} task(s) were scored for only one of the "
                                                  "two and are not counted."})
    return save(state, kind="eval_comparison", id=id or f"eval-{_slug(before)}-vs-{_slug(after)}", source="opentrain",
                title=f"{after} vs {before}: {verdict}",
                summary=f"{after} passes {pb}/{len(both)} held-out tasks, {before} {pa}/{len(both)}; "
                        f"{len(fixed)} fixed, {len(broke)} regressed.",
                blocks=blocks, links=links, session_id=session_id,
                decision={"question": f"Ship {after}?",
                          "options": ["Ship it", "Not yet: keep improving"],
                          "approves": {"needs": ["promote"], "cost_usd": 0}})


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9.-]+", "-", s.lower()).strip("-")[:24] or "x"


# ---------------------------------------------------------------------------- a page to share


def to_html(art: dict[str, Any], *, embed: bool = False, theme: str | None = None) -> str:
    """The artifact as one self-contained page (no scripts, no outside files): shown in the app and shared."""
    from .artifact_page import page

    return page(art, embed=embed, theme=theme)
