"""The Trainfile: the one file a user edits.

It sits at the repo root, in the family of Makefile and Dockerfile. The server enforces it; the
engine reads it and can never change it mid-run. One setting per line, ``key  value``, ``#`` for
comments. Keys may contain a space (``base model``); the value is everything after the first run
of two or more spaces, or after the key when the key is known.

    # Trainfile
    goal         speed
    budget       $40 a week, model tokens and compute together
    promote      beats current on held-out, no task regresses
    runtime      claude -p
    trainer      tinker
    base model   qwen-27b
    data         my agent sessions, redacted, local until I approve
    watch        when 30 new sessions arrive, or nightly
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

GOALS = ("speed", "fix", "cost")
"""Goals with a built-in recipe for coding-agent work. Any other goal is written in plain words
(``goal  draft support replies in our tone``) and is measured on the project's datasets."""
RUNTIMES = {"claude -p": "claude", "claude": "claude", "codex exec": "codex", "codex": "codex", "direct": "direct"}
PERIODS = {"run": None, "day": 1, "week": 7, "month": 30}

KEYS = ("goal", "budget", "promote", "runtime", "trainer", "base model", "data", "watch", "serve", "heldout",
        "eval cap", "provider")

DEFAULT = """\
# Trainfile: what OpenTrain may do on your behalf. The server enforces every line.
# goal: say in plain words what the model should get better at.
goal         better than the current model on my held-out examples
budget       $20 a week, model tokens and compute together
promote      beats current on held-out, no task regresses
trainer      tinker
base model   Qwen/Qwen3-8B
data         local until I approve
heldout      30%
eval cap     8 per run
"""


class TrainfileError(ValueError):
    pass


@dataclass(frozen=True)
class Budget:
    dollars: float
    period_days: int | None
    """None: the budget is per run."""
    text: str

    def describe(self) -> str:
        per = "run" if self.period_days is None else {1: "day", 7: "week", 30: "month"}[self.period_days]
        return f"${self.dollars:g} a {per}"


@dataclass(frozen=True)
class Watch:
    new_sessions: int | None = None
    nightly: bool = False
    flagged: bool = True


@dataclass(frozen=True)
class Trainfile:
    goal: str = "custom"
    """A built-in recipe (``speed``, ``fix``, ``cost``) or ``custom``: see ``objective``."""
    objective: str = ""
    """The goal in the user's words."""
    budget: Budget = field(default_factory=lambda: Budget(20.0, 7, "$20 a week"))
    promote: str = "beats current on held-out, no task regresses"
    runtime: str = "claude"
    trainer: str = "tinker"
    base_model: str = "Qwen/Qwen3-8B"
    data: str = "local until I approve"
    heldout_fraction: float = 0.3
    eval_cap: int = 8
    watch: Watch | None = None
    serve: str | None = None
    provider: str | None = None
    """For the direct runtime: an OpenAI-compatible base URL or a named provider."""
    path: str | None = None
    raw: dict[str, str] = field(default_factory=dict)

    @property
    def uploads_need_approval(self) -> bool:
        return "until i approve" in self.data.lower() or "local" in self.data.lower()

    @property
    def allow_regressions(self) -> bool:
        return "no task regresses" not in self.promote.lower() and "no regress" not in self.promote.lower()


def _budget(text: str) -> Budget:
    m = re.search(r"\$\s*(\d+(?:\.\d+)?)", text)
    if not m:
        raise TrainfileError(f"budget needs a dollar amount, like '$40 a week': {text!r}")
    period = None
    low = text.lower()
    for word, days in PERIODS.items():
        if re.search(rf"\b(a|per|each|/)\s*{word}\b|\b{word}ly\b", low) or (word == "day" and "daily" in low):
            period = days
            break
    return Budget(float(m.group(1)), period, text.strip())


def _watch(text: str) -> Watch:
    low = text.lower()
    m = re.search(r"(\d+)\s+new sessions?", low)
    return Watch(
        new_sessions=int(m.group(1)) if m else None,
        nightly="nightly" in low or "every night" in low,
        flagged="no flags" not in low,
    )


def _split(line: str) -> tuple[str, str]:
    low = line.lower()
    for key in sorted(KEYS, key=len, reverse=True):
        if low.startswith(key) and (len(line) == len(key) or line[len(key)].isspace()):
            return key, line[len(key):].strip()
    parts = re.split(r"\s{2,}|\t", line, maxsplit=1)
    if len(parts) != 2:
        raise TrainfileError(f"cannot read line: {line!r} (expected 'key  value')")
    return parts[0].strip().lower(), parts[1].strip()


def parse(text: str, path: str | None = None) -> Trainfile:
    raw: dict[str, str] = {}
    for n, line in enumerate(text.splitlines(), 1):
        line = line.split(" #", 1)[0].strip() if not line.lstrip().startswith("#") else ""
        if not line:
            continue
        key, value = _split(line)
        if key not in KEYS:
            raise TrainfileError(f"line {n}: unknown setting {key!r}. Known: {', '.join(KEYS)}")
        if key in raw:
            raise TrainfileError(f"line {n}: {key!r} is set twice")
        raw[key] = value

    kw: dict = {"raw": raw, "path": path}
    if "goal" in raw:
        words = raw["goal"].split()
        if not words:
            raise TrainfileError("goal needs words: what should the model get better at?")
        first = words[0].lower().removeprefix("goal/") if words else ""
        kw["goal"] = first if first in GOALS and len(words) == 1 else "custom"
        kw["objective"] = raw["goal"].strip()
    if "budget" in raw:
        kw["budget"] = _budget(raw["budget"])
    if "promote" in raw:
        kw["promote"] = raw["promote"]
    if "runtime" in raw:
        rt = RUNTIMES.get(raw["runtime"].lower())
        if rt is None:
            raise TrainfileError(f"runtime must be one of {', '.join(RUNTIMES)}, not {raw['runtime']!r}")
        kw["runtime"] = rt
    for key, attr in (("trainer", "trainer"), ("base model", "base_model"), ("data", "data"),
                      ("serve", "serve"), ("provider", "provider")):
        if key in raw:
            kw[attr] = raw[key]
    if "heldout" in raw:
        m = re.match(r"(\d+(?:\.\d+)?)\s*%?", raw["heldout"])
        if not m:
            raise TrainfileError("heldout must be a percentage, like 30%")
        frac = float(m.group(1)) / (100 if float(m.group(1)) > 1 else 1)
        if not 0 < frac < 1:
            raise TrainfileError("heldout must be between 0% and 100%")
        kw["heldout_fraction"] = frac
    if "eval cap" in raw:
        m = re.match(r"(\d+)", raw["eval cap"])
        if not m or int(m.group(1)) < 1:
            raise TrainfileError("eval cap must be a positive number of requests per run")
        kw["eval_cap"] = int(m.group(1))
    if "watch" in raw:
        kw["watch"] = _watch(raw["watch"])
    return Trainfile(**kw)


def find(start: Path | None = None) -> Path | None:
    p = (start or Path.cwd()).resolve()
    for d in (p, *p.parents):
        if (d / "Trainfile").is_file():
            return d / "Trainfile"
    return None


def for_project(directory: Path) -> Trainfile:
    """The Trainfile governing ``directory``, or the defaults. Never looks at the process's cwd."""
    path = find(Path(directory))
    return parse(path.read_text(), str(path)) if path else parse(DEFAULT)


def load(path: Path | None = None) -> Trainfile:
    path = path or find()
    if path is None:
        return parse(DEFAULT)
    return parse(path.read_text(), str(path))


def set_value(directory: Path, key: str, value: str) -> Path:
    """Set one line of the project's Trainfile (writing the defaults first if there is none).

    The other lines, comments and alignment are kept; the result must still parse.
    """
    if key not in KEYS:
        raise TrainfileError(f"unknown setting {key!r}")
    path = find(Path(directory)) or Path(directory) / "Trainfile"
    text = path.read_text() if path.exists() else DEFAULT
    lines, done = text.splitlines(), False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            k, _ = _split(stripped.split(" #", 1)[0])
            if k == key:
                lines[i] = f"{key:<12} {value}"
                done = True
    if not done:
        lines.append(f"{key:<12} {value}")
    new = "\n".join(lines) + "\n"
    parse(new, str(path))
    path.write_text(new)
    return path
