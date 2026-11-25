"""Recipes as skills: folders holding a SKILL.md (name, description, the playbook).

Searched in order, later wins on a name clash: the bundled recipes, ``~/.config/opentrain/skills``,
``<project>/.opentrain/skills``. Skill names are ``family/verb`` (``goal/speed``, ``data/shortest``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .agent import parse_markdown
from .tool import ToolError


def bundled_dir() -> Path:
    here = Path(__file__).resolve().parent.parent
    for cand in (here / "skills", here.parents[1] / "skills"):  # installed wheel, then a source checkout
        if cand.is_dir():
            return cand
    return here / "skills"


@dataclass
class Skill:
    name: str
    description: str
    body: str
    path: str
    source: str

    def public(self) -> dict:
        return {"name": self.name, "description": self.description, "path": self.path, "source": self.source}


class Skills:
    def __init__(self, directory: Path | None = None):
        from . import config

        self.dirs = [(bundled_dir(), "bundled"), (config.global_dir() / "skills", "global")]
        if directory:
            self.dirs.append((directory / ".opentrain" / "skills", "project"))
        extra = config.load(directory).get("skills") if directory else None
        for d in extra or []:
            self.dirs.append((Path(d).expanduser(), "config"))

    def all(self) -> dict[str, Skill]:
        out: dict[str, Skill] = {}
        for d, source in self.dirs:
            if not d.is_dir():
                continue
            for f in sorted(d.glob("*/SKILL.md")) + sorted(d.glob("*/*/SKILL.md")):
                try:
                    meta, body = parse_markdown(f.read_text(), str(f))
                except Exception:
                    continue
                name = meta.get("name") or f.parent.name.replace("-", "/", 1)
                out[name] = Skill(name, meta.get("description") or "", body, str(f), source)
        return out

    def get(self, name: str) -> Skill:
        sk = self.all()
        key = name.strip().replace("-", "/", 1) if name not in sk else name
        if key not in sk:
            raise ToolError(f"no skill {name!r}; available: {', '.join(sorted(sk)) or 'none'}")
        return sk[key]

    def index(self) -> str:
        return "\n".join(f"- {s.name}: {s.description}" for s in sorted(self.all().values(), key=lambda s: s.name))
