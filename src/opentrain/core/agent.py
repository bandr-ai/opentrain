"""Agents: who the engine is in a session — instructions, model, step limit, permissions.

Built-ins ship as Markdown in ``opentrain/agents/`` (train, plan, data, monitor, evaluator), and
users add or override agents the same way, in ``~/.config/opentrain/agents/*.md`` and
``<project>/.opentrain/agents/*.md``, or under ``agents`` in ``opentrain.jsonc``. Later sources
override earlier ones field by field; ``disabled: true`` removes an agent.

An agent file::

    ---
    description: one line, shown in pickers and to the model choosing a subagent
    mode: primary | subagent | all
    model: provider/model        # optional; the session or config model otherwise
    steps: 200
    color: green
    hidden: false
    permissions:                 # "<action> [resource]": allow | ask | deny, in order, last match wins
      "*": allow
      shell: ask
      "read .opentrain/heldout/**": deny
    ---
    The system prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BUILTIN_DIR = Path(__file__).resolve().parent.parent / "agents"
MODES = ("primary", "subagent", "all")
EFFECTS = ("allow", "ask", "deny")


@dataclass
class Agent:
    name: str
    description: str
    prompt: str
    mode: str = "primary"
    model: str | None = None
    steps: int = 200
    permissions: list[dict[str, Any]] = field(default_factory=list)
    hidden: bool = False
    color: str | None = None
    source: str = "built-in"

    def public(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "mode": self.mode, "model": self.model,
                "steps": self.steps, "hidden": self.hidden, "color": self.color, "source": self.source,
                "permissions": self.permissions}

    @property
    def primary(self) -> bool:
        return self.mode in ("primary", "all")

    @property
    def subagent(self) -> bool:
        return self.mode in ("subagent", "all")


class AgentError(ValueError):
    pass


def _scalar(v: str) -> Any:
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1]
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    if v.lstrip("-").isdigit():
        return int(v)
    if v in ("", "null", "~"):
        return None
    return v


def _key(k: str) -> str:
    k = k.strip()
    return k[1:-1] if len(k) >= 2 and k[0] == k[-1] and k[0] in "\"'" else k


def parse_markdown(text: str, source: str = "") -> tuple[dict[str, Any], str]:
    """Front matter (a small YAML subset: scalars and one level of mapping) and the body."""
    if not text.startswith("---"):
        return {}, text.strip()
    end = text.find("\n---", 3)
    if end < 0:
        raise AgentError(f"{source}: front matter is not closed with ---")
    head, body = text[3:end], text[end + 4:]
    meta: dict[str, Any] = {}
    current: str | None = None
    for raw in head.splitlines():
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        indented = raw[:1] in (" ", "\t")
        k, sep, v = _split_key(raw.strip())
        if not sep:
            raise AgentError(f"{source}: cannot read front matter line {raw!r}")
        if indented and current is not None:
            meta[current].append((_key(k), _scalar(v)))
        elif v.strip() == "":
            current = _key(k)
            meta[current] = []
        else:
            current = None
            meta[_key(k)] = _scalar(v)
    return meta, body.strip()


def _split_key(line: str) -> tuple[str, str, str]:
    """Split ``key: value`` where the key may be quoted and contain colons."""
    if line[:1] in "\"'":
        q = line[0]
        end = line.find(q, 1)
        if end > 0 and line[end + 1:end + 2] == ":":
            return line[:end + 1], ":", line[end + 2:]
    k, sep, v = line.partition(":")
    return k, sep, v


def rules_from(pairs: Any, source: str = "") -> list[dict[str, Any]]:
    """``[("shell", "ask"), ("read .x/**", "deny")]`` or a dict -> permission rules."""
    if isinstance(pairs, dict):
        pairs = list(pairs.items())
    out = []
    for key, effect in pairs or []:
        if effect not in EFFECTS:
            raise AgentError(f"{source}: permission {key!r} must be allow, ask or deny, not {effect!r}")
        action, _, resource = str(key).strip().partition(" ")
        out.append({"action": action, "resource": resource.strip() or "*", "effect": effect})
    return out


def from_markdown(name: str, text: str, source: str) -> Agent:
    meta, body = parse_markdown(text, source)
    mode = meta.get("mode", "primary")
    if mode not in MODES:
        raise AgentError(f"{source}: mode must be one of {', '.join(MODES)}")
    return Agent(name=name, description=meta.get("description") or name, prompt=body, mode=mode,
                 model=meta.get("model"), steps=int(meta.get("steps") or 200),
                 permissions=rules_from(meta.get("permissions"), source), hidden=bool(meta.get("hidden")),
                 color=meta.get("color"), source=source)


def _override(base: Agent, meta: dict[str, Any], body: str | None, source: str) -> Agent:
    out = Agent(**{**base.__dict__})
    for k in ("description", "mode", "model", "color"):
        if meta.get(k) is not None:
            setattr(out, k, meta[k])
    if meta.get("steps") is not None:
        out.steps = int(meta["steps"])
    if meta.get("hidden") is not None:
        out.hidden = bool(meta["hidden"])
    if meta.get("permissions") is not None:
        out.permissions = base.permissions + rules_from(meta["permissions"], source)
    if meta.get("prompt"):
        out.prompt = meta["prompt"]
    if body:
        out.prompt = body
    out.source = source
    return out


def load(directory: Path | None = None) -> tuple[dict[str, Agent], list[str]]:
    """All agents for a project, and any problems found loading them."""
    from . import config

    agents: dict[str, Agent] = {}
    problems: list[str] = []
    dirs = [(BUILTIN_DIR, "built-in"), (config.global_dir() / "agents", "global")]
    if directory:
        dirs.append((directory / ".opentrain" / "agents", "project"))
    for d, kind in dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.md")):
            name = f.stem
            try:
                text = f.read_text()
                if name in agents:
                    meta, body = parse_markdown(text, str(f))
                    if meta.get("disabled"):
                        agents.pop(name)
                        continue
                    agents[name] = _override(agents[name], meta, body, str(f) if kind != "built-in" else "built-in")
                else:
                    meta, _ = parse_markdown(text, str(f))
                    if meta.get("disabled"):
                        continue
                    agents[name] = from_markdown(name, text, "built-in" if kind == "built-in" else str(f))
            except AgentError as exc:
                problems.append(str(exc))
    try:
        cfg = config.load(directory)
    except ValueError as exc:
        problems.append(str(exc))
        cfg = {}
    for name, spec in (cfg.get("agents") or {}).items():
        if not isinstance(spec, dict):
            problems.append(f"opentrain.jsonc: agents.{name} must be an object")
            continue
        try:
            if spec.get("disabled"):
                agents.pop(name, None)
            elif name in agents:
                agents[name] = _override(agents[name], spec, None, "opentrain.jsonc")
            else:
                agents[name] = Agent(name=name, description=spec.get("description") or name,
                                     prompt=spec.get("prompt") or f"You are the {name} agent.",
                                     mode=spec.get("mode", "primary"), model=spec.get("model"),
                                     steps=int(spec.get("steps") or 200),
                                     permissions=rules_from(spec.get("permissions"), "opentrain.jsonc"),
                                     hidden=bool(spec.get("hidden")), color=spec.get("color"), source="opentrain.jsonc")
        except (AgentError, ValueError) as exc:
            problems.append(str(exc))
    return agents, problems


class Agents:
    """The agents of one project, reloaded when their files change."""

    def __init__(self, project: Any = None):
        self.project = project
        self.directory = getattr(project, "directory", None)
        self.problems: list[str] = []
        self._stamp: tuple = ()
        self.agents: dict[str, Agent] = {}
        self.reload()

    def _sources(self) -> tuple:
        from . import config

        paths = [BUILTIN_DIR, config.global_dir() / "agents", config.global_dir() / "opentrain.json",
                 config.global_dir() / "opentrain.jsonc"]
        if self.directory:
            paths += [self.directory / ".opentrain" / "agents", self.directory / "opentrain.json",
                      self.directory / "opentrain.jsonc"]
        stamp = []
        for p in paths:
            if p.is_dir():
                stamp += [(str(f), f.stat().st_mtime) for f in sorted(p.glob("*.md"))]
            elif p.exists():
                stamp.append((str(p), p.stat().st_mtime))
        return tuple(stamp)

    def reload(self) -> None:
        self.agents, self.problems = load(self.directory)
        self._stamp = self._sources()

    def _fresh(self) -> None:
        if self._sources() != self._stamp:
            self.reload()

    def get(self, name: str | None) -> Agent:
        self._fresh()
        if name and name in self.agents:
            return self.agents[name]
        return self.agents[self.default()]

    def find(self, name: str) -> Agent | None:
        self._fresh()
        return self.agents.get(name)

    def list(self) -> list[Agent]:
        self._fresh()
        order = {"train": 0, "plan": 1, "data": 2, "monitor": 3, "evaluator": 9}
        return sorted(self.agents.values(), key=lambda a: (order.get(a.name, 5), a.name))

    def primaries(self) -> list[Agent]:
        return [a for a in self.list() if a.primary and not a.hidden]

    def subagents(self) -> list[Agent]:
        return [a for a in self.list() if a.subagent and not a.hidden]

    def default(self) -> str:
        from . import config

        name = config.load(self.directory).get("default_agent") if self.directory else None
        if name and name in self.agents and self.agents[name].primary:
            return name
        return "train" if "train" in self.agents else next(iter(self.agents))
