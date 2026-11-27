"""The model registry: every version, what it was trained on, and which one is live.

    .opentrain/models/registry.json
      versions: [{version, run_id, base_model, backend, checkpoint, scores, decision, created_at}]
      current:  "v3" or null
      history:  every change of ``current``, so ``revert`` always has somewhere to go

Only the gate calls ``promote``. ``revert`` is the one-command rollback.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class Version:
    version: str
    run_id: str | None
    base_model: str
    backend: str
    checkpoint: str
    """A backend-specific pointer: a tinker:// path, a local adapter folder, a hosted model id."""
    serving: dict[str, Any]
    scores: dict[str, Any]
    decision: str
    created_at: str
    dataset: str | None = None
    """The dataset it was trained on, when it was trained on one (else it came from mined tasks)."""

    @property
    def model_name(self) -> str:
        return f"opentrain/{self.version}"


class Registry:
    def __init__(self, state: Path):
        self.path = state / "models" / "registry.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"versions": [], "current": None, "history": []}
        return json.loads(self.path.read_text())

    def _save(self, data: dict[str, Any]) -> None:
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.path)

    def versions(self) -> list[Version]:
        return [Version(**v) for v in self._load()["versions"]]

    def get(self, version: str) -> Version:
        for v in self.versions():
            if v.version == version:
                return v
        raise RegistryError(f"no version {version!r}")

    def current(self) -> Version | None:
        cur = self._load()["current"]
        return self.get(cur) if cur else None

    def add(self, *, run_id: str | None, base_model: str, backend: str, checkpoint: str,
            serving: dict[str, Any] | None = None, scores: dict[str, Any] | None = None,
            decision: str = "proposed") -> Version:
        data = self._load()
        v = Version(
            version=f"v{len(data['versions']) + 1}",
            run_id=run_id,
            base_model=base_model,
            backend=backend,
            checkpoint=checkpoint,
            serving=serving or {},
            scores=scores or {},
            decision=decision,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        data["versions"].append(v.__dict__)
        self._save(data)
        return v

    def _set_current(self, version: str | None, why: str) -> None:
        data = self._load()
        data["history"].append({"from": data["current"], "to": version, "why": why,
                                "at": datetime.now(timezone.utc).isoformat()})
        data["current"] = version
        self._save(data)

    def update(self, version: str, **fields: Any) -> Version:
        data = self._load()
        for v in data["versions"]:
            if v["version"] == version:
                v.update(fields)
                self._save(data)
                return Version(**v)
        raise RegistryError(f"no version {version!r}")

    def record_gate(self, version: str, decision: str, scores: dict[str, Any]) -> Version:
        """Store a gate result. A version already promoted or reverted keeps that state: re-gating it
        must not erase the history ``revert`` walks."""
        v = self.get(version)
        fields: dict[str, Any] = {"scores": scores}
        if v.decision not in ("promoted", "reverted"):
            fields["decision"] = decision
        return self.update(version, **fields)

    def _clear_pin(self) -> None:
        # A promotion or a rollback is a decision about what is live; a manual serving pin from
        # before it must not keep serving something else.
        (self.path.parent.parent / "serving" / "live.json").unlink(missing_ok=True)

    def promote(self, version: str, why: str) -> Version:
        self.get(version)
        self.update(version, decision="promoted")
        self._set_current(version, why)
        self._clear_pin()
        return self.get(version)

    def revert(self, why: str = "manual revert") -> Version | None:
        """Put back the version that was live before the current one."""
        data = self._load()
        cur = data["current"]
        if cur is None:
            raise RegistryError("nothing is live")
        live = {v["version"]: v["decision"] for v in data["versions"]}
        previous = None
        for h in reversed(data["history"]):
            if h["to"] and h["to"] != cur and live.get(h["to"]) == "promoted":
                previous = h["to"]
                break
        self.update(cur, decision="reverted")
        self._set_current(previous, why)
        self._clear_pin()
        return self.get(previous) if previous else None
