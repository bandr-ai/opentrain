"""One answer on one screen: a boxed card with bars, for the terminal (``opentrain traces``, ``report``).

A card is a title, then sections of rows: a label, a value and an optional bar. It draws with box
characters when the terminal can show them and plain ASCII when it cannot, so it can be pasted
anywhere.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field


@dataclass
class Row:
    label: str
    value: str
    frac: float | None = None
    """0..1 for a bar under the value; None for no bar."""


@dataclass
class Section:
    title: str
    rows: list[Row] = field(default_factory=list)


def _unicode() -> bool:
    enc = (getattr(sys.stdout, "encoding", "") or "").lower()
    return "utf" in enc


def _bar(frac: float, width: int, fancy: bool) -> str:
    frac = max(0.0, min(1.0, frac))
    if not fancy:
        n = round(frac * width)
        return "#" * n + "." * (width - n)
    eighths = " ▏▎▍▌▋▊▉█"
    full, rem = divmod(round(frac * width * 8), 8)
    out = "█" * full + (eighths[rem] if rem and full < width else "")
    if frac > 0 and not out.strip():
        out = "▏"
    return out + "░" * (width - len(out))


def render(title: str, subtitle: list[str], sections: list[Section], footer: list[str] | None = None,
           *, width: int = 72, fancy: bool | None = None) -> str:
    fancy = _unicode() if fancy is None else fancy
    h, v, tl, tr, bl, br, lj, rj, cross, tj, bj = (
        ("─", "│", "┌", "┐", "└", "┘", "├", "┤", "┼", "┬", "┴") if fancy else
        ("-", "|", "+", "+", "+", "+", "+", "+", "+", "+", "+"))
    left = 22
    inner = width - 2
    right = inner - left - 1

    def line(text: str) -> str:
        return f"{v}{text.center(inner)[:inner]}{v}"

    def cells(a: str, b: str) -> str:
        return f"{v} {a[:left - 2].ljust(left - 2)} {v} {b[:right - 2].ljust(right - 2)} {v}"

    out = [f"{tl}{h * inner}{tr}", line(title.upper())]
    out += [line(t) for t in subtitle]
    closed = False  # the last line drawn already separates what comes next
    for sec in sections:
        if not closed:
            out.append(f"{lj}{h * inner}{rj}")
        out.append(line(sec.title.upper()))
        out.append(f"{lj}{h * left}{tj}{h * right}{rj}")
        for i, r in enumerate(sec.rows):
            if i:
                out.append(f"{lj}{h * left}{cross}{h * right}{rj}")
            out.append(cells(r.label, r.value))
            if r.frac is not None:
                out.append(cells("", _bar(r.frac, right - 2, fancy)))
        last = sec is sections[-1]
        out.append(f"{bl}{h * left}{bj}{h * right}{br}" if last and not footer else f"{lj}{h * left}{bj}{h * right}{rj}")
        closed = True
    for f in footer or []:
        out.append(f"{v} {f[:inner - 2].ljust(inner - 2)} {v}")
    if footer:
        out.append(f"{bl}{h * inner}{br}")
    return "\n".join(out)
