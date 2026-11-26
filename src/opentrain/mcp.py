"""OpenTrain's train tools, served over MCP (stdio) to the engine's model: ``python -m opentrain.mcp``.

The engine's reasoning runs in the user's own ``claude -p`` or ``codex exec``. That process gets a
shell and files for ordinary work, and these tools for the actions that cost money, touch the
evals or change what is live (plan: "Train tools"). The server side of every tool enforces the
Trainfile: a tool cannot spend past the budget, ask the evaluator more often than the cap, read
held-out content, or promote without the gate.

Tools are scoped by role. The trainer never gets a tool that reads held-out tasks; the evaluator
never gets one that trains. In agent mode the trainer's CLI also gets no shell and no file writes,
and reads under ``.opentrain`` are denied (``engine.agent``). Against a process running as the same
user with a shell, only a separate OS account or container for the held-out store is a true wall.

Protocol: JSON-RPC 2.0, one message per line, MCP methods ``initialize``, ``tools/list``,
``tools/call`` and ``ping``. No dependencies.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from . import __version__
from . import gate as gate_mod
from .budget import BudgetExceeded, Ledger
from .evaluator import EvalCapReached, Evaluator, summarise
from .plugs import TrainSpec
from .record import RunRecord
from .registry import Registry
from .tasks import HeldoutStore, Store
from .trainfile import Trainfile, find, for_project, load  # noqa: F401

PROTOCOL = "2025-06-18"

TOOLS_BY_ROLE = {
    "trainer": ["record_query", "budget_status", "tasks_list", "tasks_mine", "sample_run", "data_build",
                "train_launch", "train_status", "train_cancel", "eval_request", "version_propose"],
    "evaluator": ["record_query", "budget_status", "heldout_lint", "eval_request"],
}


class ToolError(Exception):
    """A refusal the model should read and adapt to, not a crash."""


def _schema(props: dict[str, Any] | None = None, required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": props or {}, "required": required or [], "additionalProperties": False}


SPECS: dict[str, tuple[str, dict[str, Any]]] = {
    "record_query": ("Past and current runs: triggers, data, recipes, spend, scores, gate decisions. "
                     "Memory across sessions; read it before choosing what to try.",
                     _schema({"run_id": {"type": "string"}, "last": {"type": "integer", "default": 5}})),
    "budget_status": ("Dollars committed and left in this budget period, per the Trainfile.", _schema()),
    "tasks_list": ("Training tasks: id, family, instruction preview, verifier command. Held-out tasks "
                   "are not listed and cannot be read.", _schema({"family": {"type": "string"}})),
    "tasks_mine": ("Build verified tasks from the user's sessions (cl/mine + bandits/certify), then "
                   "re-split held-out by date.", _schema({"include_repairable": {"type": "boolean"}})),
    "sample_run": ("Run a model n times on training tasks; each attempt is verified by the task's "
                   "tests. Metered against the budget. teacher: 'frontier' or 'base'.",
                   _schema({"task_ids": {"type": "array", "items": {"type": "string"}},
                            "n": {"type": "integer", "default": 4},
                            "teacher": {"type": "string", "enum": ["frontier", "base"]}})),
    "data_build": ("Assemble the training set: the shortest verified pass per task (data/shortest), "
                   "as chat JSONL. Records exactly which tasks went in.",
                   _schema({"task_ids": {"type": "array", "items": {"type": "string"}}})),
    "train_launch": ("Launch fine-tuning on the Trainfile's backend and base model. The estimate is "
                     "reserved against the budget first; refused if it does not fit.",
                     _schema({"epochs": {"type": "integer"}, "lora_rank": {"type": "integer"},
                              "learning_rate": {"type": "number"}})),
    "train_status": ("A training job's state, progress and cost. When it succeeds the checkpoint is "
                     "registered as a new version (not live).", _schema({"job_id": {"type": "string"}}, ["job_id"])),
    "train_cancel": ("Stop a training job.", _schema({"job_id": {"type": "string"}}, ["job_id"])),
    "eval_request": ("Ask the evaluator to score a version on held-out tasks. Returns pass/fail and "
                     "time per task id only, never task content. Capped per run.",
                     _schema({"version": {"type": "string"}}, ["version"])),
    "version_propose": ("Submit a version to the promotion gate. The gate, not you, decides and deploys.",
                        _schema({"version": {"type": "string"}}, ["version"])),
    "heldout_lint": ("Re-run verifier controls on held-out tasks; ids and reasons only.", _schema()),
}


class Server:
    def __init__(self, state: Path, role: str, run_id: str | None, tf: Trainfile | None = None):
        if role not in TOOLS_BY_ROLE:
            raise ValueError(f"unknown role {role!r}")
        self.state, self.role = state, role
        self.tf = tf or for_project(state.parent)
        self.record = RunRecord(state)
        self.run = self.record.get(run_id) if run_id else self.record.open(f"mcp:{role}", {"goal": self.tf.goal})
        self.ledger = Ledger(state, self.tf.budget)
        self.handlers: dict[str, Callable[..., Any]] = {n: getattr(self, f"t_{n}") for n in TOOLS_BY_ROLE[role]}

    # ------------------------------------------------------------------ tools

    def t_record_query(self, run_id: str | None = None, last: int = 5) -> Any:
        if run_id:
            return self.record.get(run_id).events()
        out = []
        for i, r in enumerate(self.record.runs()):
            if i >= last:
                break
            ev = r.events()
            gate = next((e["data"] for e in reversed(ev) if e["kind"] == "gate"), None)
            out.append({"run_id": r.run_id, "trigger": ev[0]["data"].get("trigger") if ev else None,
                        "spent": r.spent(), "gate": gate, "closed": r.closed,
                        "notes": [e["data"].get("text") for e in ev if e["kind"] == "note"][-5:]})
        return out

    def t_budget_status(self) -> Any:
        return {"budget": self.tf.budget.describe(), "committed_usd": self.ledger.committed(self.run.run_id),
                "remaining_usd": self.ledger.remaining(self.run.run_id)}

    def t_tasks_list(self, family: str | None = None) -> Any:
        rows = []
        for t in Store(self.state):
            if family and t.family != family:
                continue
            rows.append({"task_id": t.task_id, "family": t.family, "created_at": t.created_at,
                         "instruction": t.instruction[:300], "test_command": t.test_command})
        return {"tasks": rows, "heldout_count": len(HeldoutStore(self.state).ids())}

    def t_tasks_mine(self, include_repairable: bool = False) -> Any:
        from .engine import mine_step
        from .sessions.claude_code import DEFAULT_ROOT

        lines: list[str] = []
        out = mine_step(self.state, DEFAULT_ROOT, self.tf, self.run, lines.append, include_repairable)
        return {"log": lines, "train": len(out["train"]), "heldout": len(out["heldout"]),
                "reasons": out["report"]["reasons"]}

    def t_sample_run(self, task_ids: list[str] | None = None, n: int = 4, teacher: str = "frontier") -> Any:
        from .engine import parse_provider, teacher_runner
        from .sample import filter_tasks, sample

        store = Store(self.state)
        ids = task_ids or store.ids()
        missing = [i for i in ids if i not in store]
        if missing:
            raise ToolError(f"not training tasks: {', '.join(missing[:5])}")
        if teacher == "base" and parse_provider(self.tf.provider) is None:
            raise ToolError("no provider line in the Trainfile, so the base model cannot be sampled; use frontier")
        tf = self.tf if teacher == "base" else dataclasses.replace(self.tf, provider=None)
        runner, who = teacher_runner(tf)
        try:
            atts = sample([store.get(i) for i in ids], runner, n, self.state, self.ledger, self.run, workers=2)
        except BudgetExceeded as exc:
            raise ToolError(str(exc)) from None
        fr = filter_tasks(atts)
        return {"teacher": who, "attempts": len(atts), "passed": sum(a.passed is True for a in atts),
                "never_passed": fr.none_passed, "always_passed": fr.all_passed, "rates": fr.rates}

    def t_data_build(self, task_ids: list[str] | None = None) -> Any:
        from .data.sft import build_sft
        from .sample import load_attempts, shortest

        chosen = shortest(load_attempts(self.state, task_ids))
        if not chosen:
            raise ToolError("no verified passes yet; run sample_run first")
        m = build_sft(chosen, self.state, run=self.run)
        return {k: m[k] for k in ("file", "examples", "sha256", "skipped")}

    def t_train_launch(self, epochs: int | None = None, lora_rank: int | None = None,
                       learning_rate: float | None = None) -> Any:
        from .backends import get_backend

        data = self.state / "data" / "train.jsonl"
        if not data.exists() or not data.read_text().strip():
            raise ToolError("no training set; run data_build first")
        manifest = self.state / "data" / "manifest.json"
        used = {t["task_id"] for t in json.loads(manifest.read_text()).get("tasks", [])} if manifest.exists() else set()
        leaked = used & set(HeldoutStore(self.state).ids())
        if leaked or not manifest.exists():
            raise ToolError("the training set is stale (built before the held-out split changed); run data_build again")
        backend = get_backend(self.tf.trainer, self.state)
        ok, why = backend.available()
        if not ok:
            raise ToolError(f"backend {self.tf.trainer} unavailable: {why}")
        kw = {k: v for k, v in {"epochs": epochs, "lora_rank": lora_rank, "learning_rate": learning_rate}.items()
              if v is not None}
        spec = TrainSpec(base_model=self.tf.base_model, data=data, name=f"opentrain-{self.run.run_id}", **kw)
        est = backend.estimate(spec)
        try:
            rid = self.ledger.reserve(est, f"training on {self.tf.trainer}", self.run.run_id)
        except BudgetExceeded as exc:
            raise ToolError(str(exc)) from None
        job_id = backend.launch(spec)
        self.run.add("job", job_id=job_id, backend=self.tf.trainer, state="launched", estimate_usd=est,
                     reservation=rid, spec={k: str(v) for k, v in asdict(spec).items()})
        return {"job_id": job_id, "estimate_usd": est}

    def _job_event(self, job_id: str) -> dict[str, Any]:
        for e in self.run.events():
            if e["kind"] == "job" and e["data"].get("job_id") == job_id and e["data"].get("state") == "launched":
                return e["data"]
        raise ToolError(f"job {job_id} was not launched in this run")

    def t_train_status(self, job_id: str) -> Any:
        from .backends import get_backend

        launched = self._job_event(job_id)
        st = get_backend(launched["backend"], self.state).status(job_id)
        out: dict[str, Any] = {"state": st.state, "progress": st.progress, "cost_usd": st.cost_usd,
                               "detail": st.detail, "logs": st.logs[-10:], "loss": st.loss}
        if not st.done and st.cost_usd > launched["estimate_usd"] + self.ledger.remaining(self.run.run_id):
            get_backend(launched["backend"], self.state).cancel(job_id)
            out["cancelled"] = "over budget"
        if st.done and not any(e["kind"] == "job" and e["data"].get("job_id") == job_id
                               and e["data"].get("state") in ("succeeded", "failed", "cancelled")
                               for e in self.run.events()):
            self.ledger.settle(launched["reservation"], st.cost_usd)
            self.run.add("spend", usd=st.cost_usd, what=f"training job {job_id}")
            self.run.add("job", job_id=job_id, backend=launched["backend"], state=st.state)
            if st.state == "succeeded" and st.checkpoint:
                rank = int((launched.get("spec") or {}).get("lora_rank") or 32)
                v = Registry(self.state).add(run_id=self.run.run_id, base_model=self.tf.base_model,
                                             backend=launched["backend"], checkpoint=st.checkpoint,
                                             serving={"lora_rank": rank})
                self.run.add("checkpoint", version=v.version, checkpoint=st.checkpoint)
                out["version"] = v.version
        return out

    def t_train_cancel(self, job_id: str) -> Any:
        from .backends import get_backend

        get_backend(self._job_event(job_id)["backend"], self.state).cancel(job_id)
        return {"cancelling": job_id}

    def _scores(self, version: str) -> list[gate_mod.Score]:
        from .engine import baseline_name, specialist_runner
        from .evaluator import EvalBroken

        reg = Registry(self.state)
        if version == baseline_name(self.tf, reg):
            raise ToolError(f"{version} is the live version; its scores are the baseline and are not re-requested here")
        ev = Evaluator(self.state, self.tf.eval_cap)
        try:
            scores = ev.request(specialist_runner(reg.get(version)), f"opentrain/{version}", self.run)
        except (EvalCapReached, EvalBroken) as exc:
            raise ToolError(str(exc)) from None
        ev.save_scores(version, scores)
        return scores

    def t_eval_request(self, version: str) -> Any:
        scores = self._scores(version)
        return {**summarise(scores), "per_task": [{"task_id": s.task_id, "passed": s.passed, "seconds": s.seconds}
                                                  for s in scores]}

    def t_version_propose(self, version: str) -> Any:
        from .certify import lint

        reg = Registry(self.state)
        if reg.get(version).decision in ("promoted", "reverted"):
            raise ToolError(f"{version} was already {reg.get(version).decision}; propose a new version")
        ev = Evaluator(self.state, self.tf.eval_cap)
        cand = ev.scores_for(version)
        if cand is None:
            raise ToolError(f"{version} has no held-out scores; call eval_request first")
        from .engine import baseline_name

        name = baseline_name(self.tf, reg)
        cur_scores = ev.scores_for(name)
        if cur_scores is None or {s.task_id for s in cur_scores} != set(HeldoutStore(self.state).ids()):
            # The bar is re-measured here, server-side and outside the trainer's eval cap, whenever
            # the held-out set has moved since it was scored.
            from .engine import _baseline_runner, specialist_runner
            from .evaluator import EvalBroken

            cur = reg.current()
            try:
                cur_scores = ev.baseline(specialist_runner(cur) if cur else _baseline_runner(self.tf), name)
            except EvalBroken as exc:
                raise ToolError(f"could not score the baseline {name}: {exc}") from None
        broken = [r["task_id"] for r in lint(self.state, only_heldout=True)["broken"]]
        d = gate_mod.decide(cur_scores, cand, self.tf.goal, allow_regressions=self.tf.allow_regressions,
                            broken_verifiers=broken)
        reg.record_gate(version, d.explain(), {**summarise(cand), "per_task": [asdict(s) for s in cand]})
        self.run.add("gate", promote=d.promote, version=version, reasons=d.reasons,
                     regressions=d.regressions, fixed=d.fixed, summary=d.summary)
        if d.promote:
            reg.promote(version, d.explain())
        return {"promoted": d.promote, "reasons": d.reasons, "regressed": d.regressions, "fixed": d.fixed}

    def t_heldout_lint(self) -> Any:
        from .certify import lint

        return lint(self.state, only_heldout=True)

    # ------------------------------------------------------------------ protocol

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name not in self.handlers:
            return {"content": [{"type": "text", "text": f"tool {name!r} is not available to the {self.role} role"}],
                    "isError": True}
        try:
            result = self.handlers[name](**(args or {}))
            return {"content": [{"type": "text", "text": json.dumps(result, indent=1, default=str)}]}
        except (ToolError, TypeError, ValueError, KeyError) as exc:
            return {"content": [{"type": "text", "text": f"refused: {exc}"}], "isError": True}
        except Exception as exc:  # report to the model; the server stays up
            return {"content": [{"type": "text", "text": f"error: {type(exc).__name__}: {exc}"}], "isError": True}

    def handle(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        method, mid = msg.get("method"), msg.get("id")
        if mid is None:  # a notification
            return None
        if method == "initialize":
            result: Any = {"protocolVersion": msg.get("params", {}).get("protocolVersion", PROTOCOL),
                           "capabilities": {"tools": {"listChanged": False}},
                           "serverInfo": {"name": "opentrain", "version": __version__}}
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": [{"name": n, "description": SPECS[n][0], "inputSchema": SPECS[n][1]}
                                for n in TOOLS_BY_ROLE[self.role]]}
        elif method == "tools/call":
            p = msg.get("params", {})
            result = self.call(p.get("name", ""), p.get("arguments") or {})
        else:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"no method {method}"}}
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def serve(self, stdin=sys.stdin, stdout=sys.stdout) -> None:
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                stdout.write(json.dumps({"jsonrpc": "2.0", "id": None,
                                         "error": {"code": -32700, "message": "parse error"}}) + "\n")
                stdout.flush()
                continue
            try:
                reply = self.handle(msg)
            except Exception:
                traceback.print_exc(file=sys.stderr)
                reply = {"jsonrpc": "2.0", "id": msg.get("id"), "error": {"code": -32603, "message": "internal error"}}
            if reply is not None:
                stdout.write(json.dumps(reply, default=str) + "\n")
                stdout.flush()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="opentrain mcp", description="Serve OpenTrain's train tools over MCP stdio.")
    p.add_argument("--state", required=True, help="the .opentrain folder")
    p.add_argument("--role", choices=sorted(TOOLS_BY_ROLE), default="trainer")
    p.add_argument("--run", help="run id the tools record into (default: open a new run)")
    a = p.parse_args(argv)
    Server(Path(a.state), a.role, a.run).serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
