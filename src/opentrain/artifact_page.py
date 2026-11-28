"""An artifact drawn as one finished page: the same page in the app (in a locked frame) and when shared.

One renderer, so the in-app view and the page someone is sent look the same. The page is static:
no scripts and nothing loaded from outside (the route serves it with a CSP that forbids both), so a
page the agent wrote can show anything but can send nothing anywhere. Charts are drawn here as SVG,
to scale, in the page's own colour tokens, so they read in light and dark.
"""

from __future__ import annotations

import html
import math
import re
from pathlib import Path
from typing import Any

APP_ASSETS = Path(__file__).resolve().parent / "web" / "app" / "assets"

# Reading column; the summary sits before the detail. Colours are tokens, defined for light, for a
# dark system setting, and for an explicit choice either way (the app passes ?theme=).
CSS = """
:root{
  --bg:#fbfbf8;--ink:#1c1c1a;--muted:#5f5f58;--faint:#8d8d84;--line:#e4e4dd;--soft:#f3f3ee;--raised:#ffffff;
  --good:#187a3c;--bad:#c42b2b;--warn:#9a5208;--info:#1f5fc4;
  --good-bg:#e9f4ec;--bad-bg:#fbeceb;--warn-bg:#f9f0e4;--info-bg:#eaf0fa;
  --c1:#2b64b0;--c2:#d0782a;--c3:#3a8a57;--c4:#8a58b4;--c5:#b8475d;
  --display:"Science Gothic",ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
  --text:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --mono:"JetBrains Mono","SF Mono",ui-monospace,Menlo,Consolas,monospace;
  color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#101112;--ink:#e8e8e3;--muted:#a6a69f;--faint:#77776f;--line:#2a2b2e;--soft:#17181a;--raised:#17181a;
  --good:#7fd88f;--bad:#ff8a80;--warn:#f0b45a;--info:#8ab4ff;
  --good-bg:#14261a;--bad-bg:#2c1716;--warn-bg:#2a2114;--info-bg:#152238;
  --c1:#7aa7ec;--c2:#f0a35e;--c3:#79c993;--c4:#bd95e6;--c5:#ee8aa0;color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#101112;--ink:#e8e8e3;--muted:#a6a69f;--faint:#77776f;--line:#2a2b2e;--soft:#17181a;--raised:#17181a;
  --good:#7fd88f;--bad:#ff8a80;--warn:#f0b45a;--info:#8ab4ff;
  --good-bg:#14261a;--bad-bg:#2c1716;--warn-bg:#2a2114;--info-bg:#152238;
  --c1:#7aa7ec;--c2:#f0a35e;--c3:#79c993;--c4:#bd95e6;--c5:#ee8aa0;color-scheme:dark}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 var(--text)}
main{max-width:860px;margin:0 auto;padding-block:28px 56px;
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));
  display:flex;flex-direction:column;gap:28px}
.embed main{padding-block:8px 24px}
header{display:flex;flex-direction:column;gap:8px}
.eyebrow{font:600 11.5px/1 var(--text);letter-spacing:.08em;text-transform:uppercase;color:var(--faint)}
h1{font:650 30px/1.15 var(--display);margin:0;text-wrap:balance;letter-spacing:-.01em}
.lead{margin:0;color:var(--muted);font-size:16px;max-width:65ch}
section{display:flex;flex-direction:column;gap:12px;min-width:0}
h2{font:600 15px/1.3 var(--display);margin:0;padding-top:20px;border-top:1px solid var(--line);text-wrap:balance}
p{margin:0;max-width:68ch}
.prose{display:flex;flex-direction:column;gap:10px}
code{font:.88em var(--mono);background:var(--soft);padding:.1em .35em;border-radius:4px}
a{color:var(--info)}
.num{font-variant-numeric:tabular-nums}
/* figures that lead the page */
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:1px;background:var(--line);
  border:1px solid var(--line);border-radius:10px;overflow:hidden}
.stat{background:var(--raised);padding:14px 16px;display:flex;flex-direction:column;gap:4px;min-width:0}
.stat .k{font:600 11.5px/1.2 var(--text);letter-spacing:.06em;text-transform:uppercase;color:var(--faint)}
.stat .v{font:650 28px/1.1 var(--display);font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.stat .n{font-size:13px;color:var(--muted);line-height:1.4}
.t-good{color:var(--good)}.t-bad{color:var(--bad)}.t-warn{color:var(--warn)}.t-info{color:var(--info)}
/* horizontal bars: one scale per group */
.bars{display:grid;grid-template-columns:minmax(110px,32%) 1fr auto;gap:8px 14px;align-items:center;font-size:14px}
.bars .l{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.track{height:8px;background:var(--soft);border-radius:4px;overflow:hidden}
.fill{height:100%;background:var(--c1);border-radius:4px}
.bars .v{color:var(--muted);font-variant-numeric:tabular-nums;text-align:right;font-size:13px}
/* charts */
figure{margin:0;display:flex;flex-direction:column;gap:8px;min-width:0}
figure svg{width:100%;height:auto;display:block;overflow:visible}
figcaption{font-size:13px;color:var(--muted)}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:13px;color:var(--muted)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.axis{fill:var(--faint);font:11px var(--text);font-variant-numeric:tabular-nums}
.grid{stroke:var(--line);stroke-width:1}
.base{stroke:var(--faint);stroke-width:1}
.vlabel{fill:var(--ink);font:600 11.5px var(--text);font-variant-numeric:tabular-nums}
/* tables */
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:10px}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th{font:600 11.5px/1.2 var(--text);letter-spacing:.05em;text-transform:uppercase;color:var(--faint);text-align:left;
  padding:10px 12px;background:var(--soft);border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:9px 12px;border-bottom:1px solid var(--line);vertical-align:top}
tr:last-child td{border-bottom:0}
td.r,th.r{text-align:right;font-variant-numeric:tabular-nums}
.chip{display:inline-block;font:600 11.5px/1 var(--text);padding:4px 8px;border-radius:999px;white-space:nowrap}
.chip.good{color:var(--good);background:var(--good-bg)}.chip.bad{color:var(--bad);background:var(--bad-bg)}
.chip.warn{color:var(--warn);background:var(--warn-bg)}
/* side by side */
.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.pane{display:flex;flex-direction:column;gap:6px;min-width:0}
.pane .k{font:600 11.5px/1.2 var(--text);letter-spacing:.06em;text-transform:uppercase;color:var(--faint)}
pre{margin:0;white-space:pre-wrap;word-break:break-word;font:12.5px/1.55 var(--mono);background:var(--soft);
  border:1px solid var(--line);border-radius:8px;padding:12px;max-height:420px;overflow:auto}
/* callouts: the one thing that needs attention */
.callout{border-radius:10px;padding:14px 16px;display:flex;flex-direction:column;gap:4px;background:var(--info-bg)}
.callout b{font:600 14.5px/1.35 var(--display)}
.callout.good{background:var(--good-bg)}.callout.good b{color:var(--good)}
.callout.warn{background:var(--warn-bg)}.callout.warn b{color:var(--warn)}
.callout.bad{background:var(--bad-bg)}.callout.bad b{color:var(--bad)}
.callout.info b{color:var(--info)}
ul{margin:0;padding-left:20px;display:flex;flex-direction:column;gap:6px;max-width:68ch}
.decision{border:1px solid var(--line);border-radius:10px;padding:16px;display:flex;flex-direction:column;gap:6px;background:var(--raised)}
.decision .opts{display:flex;flex-wrap:wrap;gap:8px}
.decision .opt{border:1px solid var(--line);border-radius:999px;padding:4px 12px;font-size:13.5px}
.decision .opt.on{border-color:var(--ink);font-weight:600}
.custom{min-width:0}
footer{font-size:12.5px;color:var(--faint)}
@media (max-width:640px){h1{font-size:25px}.pair{grid-template-columns:1fr}.bars{grid-template-columns:minmax(90px,38%) 1fr auto}}
@media print{.scroll{overflow:visible}}
"""

SERIES = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)", "var(--c5)"]
TONES = ("good", "warn", "bad", "info")


def esc(v: Any) -> str:
    return html.escape("" if v is None else str(v))


def inline(text: Any) -> str:
    """A small, safe subset of markdown: **bold**, `code`, [links](https://...)."""
    t = esc(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    return re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<a href="\2" target="_blank" rel="noreferrer">\1</a>', t)


# ---------------------------------------------------------------------------- numbers and scales

def _num(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v) if math.isfinite(v) else None
    m = re.fullmatch(r"\s*[$€£]?\s*(-?[\d,]*\.?\d+)\s*(%|s|ms|x|k|M)?\s*", str(v))
    return float(m.group(1).replace(",", "")) if m else None


def _fmt(v: float, step: float = 1.0) -> str:
    if abs(v) >= 1000 and step >= 1:
        return f"{v:,.0f}"
    if step >= 1 or float(v).is_integer():
        return f"{v:.0f}"
    places = min(4, max(1, -int(math.floor(math.log10(step)))))
    return f"{v:.{places}f}"


def ticks(lo: float, hi: float, n: int = 5) -> tuple[list[float], float]:
    """Round tick values covering [lo, hi], about n of them, and the step between them."""
    if hi == lo:
        hi = lo + (abs(lo) or 1)
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.floor(lo / step) * step
    out, v = [], start
    while v < hi - 1e-9 * step:
        out.append(round(v, 10))
        v += step
    out.append(round(v, 10))
    return out, step


# ---------------------------------------------------------------------------- charts

W, H = 680, 260
ML, MR, MT, MB = 52, 20, 18, 40


def _legend(series: list[dict[str, Any]]) -> str:
    if len(series) < 2:
        return ""
    return '<div class="legend">' + "".join(
        f'<span><i style="background:{SERIES[i % len(SERIES)]}"></i>{esc(s.get("name", ""))}</span>'
        for i, s in enumerate(series)) + "</div>"


def chart_svg(b: dict[str, Any]) -> str:
    series = [s for s in b.get("series") or [] if s.get("points")]
    if not series:
        return ""
    unit = esc(b.get("unit") or "")
    title = esc(b.get("title") or b.get("y_label") or "chart")
    if b.get("kind") == "bar":
        return _bar_svg(series, unit, title, b)
    return _line_svg(series, unit, title, b)


def _axes(y_ticks: list[float], step: float, y, unit: str, b: dict[str, Any]) -> list[str]:
    out = []
    for t in y_ticks:
        yy = y(t)
        out.append(f'<line class="grid" x1="{ML}" x2="{W - MR}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
        out.append(f'<text class="axis" x="{ML - 8}" y="{yy + 3.5:.1f}" text-anchor="end">{_fmt(t, step)}{unit}</text>')
    if b.get("y_label"):
        out.append(f'<text class="axis" x="{ML}" y="{MT - 6}" text-anchor="start">{esc(b["y_label"])}</text>')
    if b.get("x_label"):
        out.append(f'<text class="axis" x="{W - MR}" y="{H - 4}" text-anchor="end">{esc(b["x_label"])}</text>')
    return out


def _line_svg(series: list[dict[str, Any]], unit: str, title: str, b: dict[str, Any]) -> str:
    pts = [[(_num(p[0]), _num(p[1])) for p in s["points"] if isinstance(p, (list, tuple)) and len(p) >= 2]
           for s in series]
    pts = [[(x, yv) for x, yv in s if x is not None and yv is not None] for s in pts]
    xs = [x for s in pts for x, _ in s]
    ys = [yv for s in pts for _, yv in s]
    if not xs:
        return ""
    lo = min(ys) if min(ys) < 0 or b.get("zero") is False else 0.0
    if b.get("zero") is False:
        pad = (max(ys) - min(ys)) * 0.08 or 1
        lo = min(ys) - pad
    y_ticks, step = ticks(lo, max(ys))
    x0, x1 = min(xs), max(xs)  # the data's own range: no empty tail after the last step
    x_ticks, xstep = ticks(x0, x1, 6)
    x_ticks = [t for t in x_ticks if x0 - 1e-9 <= t <= x1 + 1e-9]
    y0, y1 = y_ticks[0], y_ticks[-1]
    x = lambda v: ML + (v - x0) / ((x1 - x0) or 1) * (W - ML - MR)  # noqa: E731
    y = lambda v: MT + (1 - (v - y0) / ((y1 - y0) or 1)) * (H - MT - MB)  # noqa: E731
    out = _axes(y_ticks, step, y, unit, b)
    for t in x_ticks:
        out.append(f'<text class="axis" x="{x(t):.1f}" y="{H - MB + 16}" text-anchor="middle">{_fmt(t, xstep)}</text>')
    out.append(f'<line class="base" x1="{ML}" x2="{W - MR}" y1="{H - MB}" y2="{H - MB}"/>')
    for i, s in enumerate(pts):
        if not s:
            continue
        s = sorted(s)
        color = SERIES[i % len(SERIES)]
        line = " ".join(f"{x(a):.1f},{y(v):.1f}" for a, v in s)
        if len(pts) == 1 and len(s) > 1:
            area = f"{x(s[0][0]):.1f},{y(y0):.1f} {line} {x(s[-1][0]):.1f},{y(y0):.1f}"
            out.append(f'<polygon points="{area}" fill="{color}" fill-opacity="0.10" stroke="none"/>')
        out.append(f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2" '
                   'stroke-linejoin="round" stroke-linecap="round"/>')
        ex, ev = s[-1]
        out.append(f'<circle cx="{x(ex):.1f}" cy="{y(ev):.1f}" r="3.5" fill="{color}"/>')
        out.append(f'<text class="vlabel" x="{min(x(ex), W - MR):.1f}" y="{y(ev) - 9:.1f}" text-anchor="end">'
                   f'{_fmt(ev, step / 10)}{unit}</text>')
    return _figure(out, title, series, b)


def _bar_svg(series: list[dict[str, Any]], unit: str, title: str, b: dict[str, Any]) -> str:
    cats: list[str] = []
    for s in series:
        for p in s["points"]:
            if isinstance(p, (list, tuple)) and len(p) >= 2 and str(p[0]) not in cats:
                cats.append(str(p[0]))
    vals = {(i, str(p[0])): _num(p[1]) for i, s in enumerate(series) for p in s["points"]
            if isinstance(p, (list, tuple)) and len(p) >= 2}
    ys = [v for v in vals.values() if v is not None] or [0.0]
    y_ticks, step = ticks(min(0.0, min(ys)), max(ys) if max(ys) > 0 else 1)
    if b.get("max") is not None and _num(b["max"]) is not None:
        y_ticks, step = ticks(min(0.0, min(ys)), float(_num(b["max"])))
    y0, y1 = y_ticks[0], y_ticks[-1]
    y = lambda v: MT + (1 - (v - y0) / ((y1 - y0) or 1)) * (H - MT - MB)  # noqa: E731
    out = _axes(y_ticks, step, y, unit, b)
    band = (W - ML - MR) / max(len(cats), 1)
    gap = band * 0.28
    bw = (band - gap) / max(len(series), 1)
    for ci, c in enumerate(cats):
        for si in range(len(series)):
            v = vals.get((si, c))
            if v is None:
                continue
            bx = ML + ci * band + gap / 2 + si * bw
            top, base = y(max(v, 0)), y(min(v, 0))
            out.append(f'<rect x="{bx:.1f}" y="{top:.1f}" width="{max(bw - 2, 1):.1f}" height="{max(base - top, 0.5):.1f}" '
                       f'rx="2" fill="{SERIES[si % len(SERIES)]}"/>')
            out.append(f'<text class="vlabel" x="{bx + (bw - 2) / 2:.1f}" y="{top - 6:.1f}" text-anchor="middle">'
                       f'{_fmt(v, step / 10)}{unit}</text>')
        label = c if len(c) <= 16 else c[:15] + "…"
        out.append(f'<text class="axis" x="{ML + ci * band + band / 2:.1f}" y="{H - MB + 16}" text-anchor="middle">'
                   f"{esc(label)}<title>{esc(c)}</title></text>")
    out.append(f'<line class="base" x1="{ML}" x2="{W - MR}" y1="{y(0):.1f}" y2="{y(0):.1f}"/>')
    return _figure(out, title, series, b)


def _figure(parts: list[str], title: str, series: list[dict[str, Any]], b: dict[str, Any]) -> str:
    svg = (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{title}" preserveAspectRatio="xMidYMid meet">'
           + "".join(parts) + "</svg>")
    cap = f"<figcaption>{inline(b['note'])}</figcaption>" if b.get("note") else ""
    return f"<figure>{_legend(series)}{svg}{cap}</figure>"


# ---------------------------------------------------------------------------- agent-written HTML

_DROP = re.compile(r"<(script|iframe|object|embed|frame|frameset|applet|noscript|template)\b[^>]*>.*?</\1\s*>",
                   re.I | re.S)
_VOID = re.compile(r"<\/?(script|iframe|object|embed|frame|frameset|applet|link|meta|base|form|input|button|"
                   r"textarea|select|noscript|template)\b[^>]*>", re.I)
_ON = re.compile(r"""\s+on[a-z]+\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)""", re.I)
_JS = re.compile(r"""(href|src|xlink:href|action|formaction)\s*=\s*(["']?)\s*(javascript|vbscript|data:text/html)""", re.I)


def clean_html(src: str) -> str:
    """Static markup and CSS only. The page's CSP already refuses scripts and outside loads; this
    keeps the stored page honest about it too."""
    s = _DROP.sub("", src)
    s = _VOID.sub("", s)
    s = _ON.sub("", s)
    return _JS.sub(r"\1=\2#", s)


# ---------------------------------------------------------------------------- blocks

def _cell(c: Any) -> tuple[str, str]:
    """(html, class) for a table cell: pass/fail as chips, numbers right-aligned."""
    s = "" if c is None else str(c)
    low = s.strip().lower()
    if low in ("pass", "passed", "yes", "fixed", "better"):
        return f'<span class="chip good">{esc(s)}</span>', ""
    if low in ("fail", "failed", "no", "regressed", "worse"):
        return f'<span class="chip bad">{esc(s)}</span>', ""
    if low.startswith("no:") or low in ("mixed", "skipped"):
        return f'<span class="chip warn">{esc(s)}</span>', ""
    return inline(s), ("r" if _num(c) is not None else "")


def block(b: dict[str, Any]) -> str:
    t = b.get("type")
    head = f"<h2>{esc(b['title'])}</h2>" if b.get("title") and t not in ("callout",) else ""
    if t == "text":
        body = "".join(f"<p>{inline(p)}</p>" for p in str(b.get("text", "")).split("\n\n") if p.strip())
        return f'<section>{head}<div class="prose">{body}</div></section>'
    if t == "stats":
        tiles = "".join(
            f'<div class="stat"><span class="k">{esc(i.get("label"))}</span>'
            f'<span class="v{" t-" + i["tone"] if i.get("tone") in TONES else ""}">{esc(i.get("value"))}</span>'
            + (f'<span class="n">{inline(i["note"])}</span>' if i.get("note") else "") + "</div>"
            for i in b.get("items") or [])
        return f'<section>{head}<div class="stats">{tiles}</div></section>'
    if t == "bars":
        items = b.get("items") or []
        top = max([_num(i.get("max")) or 0 for i in items] + [_num(i.get("value")) or 0 for i in items] + [1])
        rows = []
        for i in items:
            v = _num(i.get("value")) or 0
            m = _num(i.get("max")) or top
            tone = f'background:var(--{i["tone"]});' if i.get("tone") in TONES else ""
            rows.append(f'<span class="l" title="{esc(i.get("label"))}">{esc(i.get("label"))}</span>'
                        f'<div class="track"><div class="fill" style="{tone}width:{min(100, v / (m or 1) * 100):.1f}%"></div></div>'
                        f'<span class="v">{esc(i.get("value"))}{esc(i.get("note") or "")}</span>')
        return f'<section>{head}<div class="bars">{"".join(rows)}</div></section>'
    if t == "chart":
        return f"<section>{head}{chart_svg(b)}</section>"
    if t == "table":
        cols = b.get("columns") or []
        rows = b.get("rows") or []
        numeric = [bool(rows) and all(_num(r[j]) is not None for r in rows if j < len(r) and str(r[j]).strip())
                   for j in range(len(cols))]
        th = "".join(f'<th class="{"r" if numeric[j] else ""}">{esc(c)}</th>' for j, c in enumerate(cols))
        body = []
        for r in rows:
            tds = []
            for c in r:
                inner, cls = _cell(c)
                tds.append(f'<td class="{cls}">{inner}</td>')
            body.append("<tr>" + "".join(tds) + "</tr>")
        return f'<section>{head}<div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table></div></section>'
    if t == "samples":
        out = []
        for i in b.get("items") or []:
            sub = f"<h2>{esc(i['title'])}</h2>" if i.get("title") else ""
            out.append(f'<section>{sub}<div class="pair">'
                       f'<div class="pane"><span class="k">{esc(i.get("left_label"))}</span><pre>{esc(i.get("left"))}</pre></div>'
                       f'<div class="pane"><span class="k">{esc(i.get("right_label"))}</span><pre>{esc(i.get("right"))}</pre></div>'
                       "</div></section>")
        return (f"<section>{head}</section>" if head else "") + "".join(out)
    if t == "list":
        return f'<section>{head}<ul>{"".join(f"<li>{inline(i)}</li>" for i in b.get("items") or [])}</ul></section>'
    if t == "callout":
        tone = b.get("tone") if b.get("tone") in TONES else "info"
        title = f"<b>{esc(b['title'])}</b>" if b.get("title") else ""
        body = "".join(f"<p>{inline(p)}</p>" for p in str(b.get("text", "")).split("\n\n") if p.strip())
        return f'<div class="callout {tone}">{title}{body}</div>'
    if t == "html":
        return f'<section>{head}<div class="custom">{clean_html(str(b.get("html", "")))}</div></section>'
    return ""


def _fonts() -> str:
    """Science Gothic from the app's own files, when the page is shown inside the app."""
    faces = []
    for f in sorted(APP_ASSETS.glob("science-gothic-latin*.woff2")):
        faces.append(f'@font-face{{font-family:"Science Gothic";font-weight:100 900;font-display:swap;'
                     f'src:url("/assets/{f.name}") format("woff2")}}')
    return "".join(faces)


def page(art: dict[str, Any], *, embed: bool = False, theme: str | None = None) -> str:
    """The whole artifact as one page. ``embed``: shown inside the app (the app draws the decision)."""
    d = art.get("decision") or {}
    dec = ""
    if d and not embed:
        opts = "".join(f'<span class="opt{" on" if d.get("chosen") == o else ""}">{esc(o)}</span>' for o in d["options"])
        state = (f"<p>Decided: <b>{esc(d['chosen'])}</b>{(' · ' + esc(d['note'])) if d.get('note') else ''}</p>"
                 if d.get("chosen") else '<p class="lead" style="font-size:13.5px">Open it in OpenTrain to decide.</p>')
        dec = f'<div class="decision"><b>{esc(d["question"])}</b><div class="opts">{opts}</div>{state}</div>'
    when = esc(str(art.get("created", ""))[:16].replace("T", " "))
    head = (f'<header><span class="eyebrow">{esc(art["kind"].replace("_", " "))} · v{art["version"]} · {when}</span>'
            f'<h1>{esc(art["title"])}</h1>' + (f'<p class="lead">{inline(art["summary"])}</p>' if art.get("summary") else "")
            + "</header>")
    links = ""
    if art.get("links") and not embed:
        links = f'<footer>Built on: {", ".join(esc(x) for x in art["links"])}</footer>'
    attr = f' data-theme="{theme}"' if theme in ("light", "dark") else ""
    return (f'<!doctype html><html lang="en"{attr}><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
            f'<base target="_blank"><title>{esc(art["title"])}</title>'
            f"<style>{_fonts() if embed else ''}{CSS}</style></head>"
            f'<body class="{"embed" if embed else ""}"><main>{head}'
            + "".join(block(b) for b in art.get("blocks") or []) + dec + links
            + ("" if embed else "<footer>Made with OpenTrain</footer>") + "</main></body></html>")
