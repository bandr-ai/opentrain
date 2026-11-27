"""SFT data: the chosen verified passes as OpenAI chat fine-tuning examples, with tools.

Each example is what the specialist will see when it is served by the direct runtime: the fixed
``SYSTEM_PROMPT``, the task's instruction (from the task store, not from the transcript) as the user
turn, then the recorded assistant tool calls and tool results, ending on the last assistant turn.
``tools`` is the direct runtime's ``TOOLS``. A transcript that called a tool outside ``TOOLS`` (a
frontier run using Glob, say) is left out, not translated, so the model is never trained on tools
it will not have.

The manifest (task ids, task digests, example hashes, file hash) is the run record's data snapshot.
Held-out task ids are refused outright, as defense in depth behind ``sample``'s own check.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..plugs import Attempt
from ..record import Run
from ..runtime.direct import SYSTEM_PROMPT, TOOL_NAMES, TOOLS, strip_timing
from ..state import write_json
from ..tasks import HeldoutStore, Store


class HeldoutLeak(ValueError):
    """An attempt on a held-out task reached the data builder. Nothing is written."""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def example(attempt: Attempt, instruction: str) -> tuple[dict[str, Any] | None, str | None]:
    """The training example for one attempt, or None and the reason it cannot be one."""
    msgs = attempt.messages
    first_user = next((i for i, m in enumerate(msgs) if m.get("role") == "user"), None)
    if first_user is None:
        return None, "transcript has no user turn"
    rest = [strip_timing(m) for m in msgs[first_user + 1:]]
    while rest and rest[-1].get("role") != "assistant":
        rest.pop()
    if not rest:
        return None, "transcript has no assistant turn"
    open_calls: set[str] = set()
    for m in rest:
        if m.get("role") == "assistant":
            for call in m.get("tool_calls") or []:
                name = call.get("function", {}).get("name")
                if name not in TOOL_NAMES:
                    return None, f"uses tool {name!r}, which the direct runtime does not have"
                open_calls.add(call.get("id"))
        elif m.get("role") == "tool":
            if m.get("tool_call_id") not in open_calls:
                return None, "a tool result answers no tool call"
        else:
            return None, f"unexpected {m.get('role')!r} turn after the instruction"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": instruction}, *rest]
    return {"messages": messages, "tools": TOOLS}, None


def build_sft(chosen: dict[str, Attempt], state: Path, out: Path | None = None, *,
              run: Run | None = None, extra: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Write ``train.jsonl`` (default ``<state>/data/train.jsonl``) and ``manifest.json`` beside it.
    ``extra``: more examples already in the runtime's format (imported traces), appended as they are."""
    held = HeldoutStore(state)
    leaked = sorted(t for t in {a.task_id for a in chosen.values()} | set(chosen) if t in held)
    if leaked:
        raise HeldoutLeak(f"held-out tasks must never become training data: {', '.join(leaked)}")

    store = Store(state)
    out = out or state / "data" / "train.jsonl"
    rows: list[str] = []
    included: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for task_id in sorted(chosen):
        att = chosen[task_id]
        if att.task_id != task_id:
            skipped.append({"task_id": task_id, "reason": f"attempt is for {att.task_id}"})
            continue
        if att.passed is not True:
            skipped.append({"task_id": task_id, "reason": "not a verified pass"})
            continue
        if task_id not in store:
            skipped.append({"task_id": task_id, "reason": "not in the training task store"})
            continue
        task = store.get(task_id)
        ex, why = example(att, task.instruction)
        if ex is None:
            skipped.append({"task_id": task_id, "reason": why or "unusable"})
            continue
        line = json.dumps(ex, sort_keys=True)
        rows.append(line)
        included.append({"task_id": task_id, "task_digest": task.digest(), "example_sha256": _sha(line)[:16],
                         "model": att.model, "steps": att.steps, "tool_calls": att.tool_calls,
                         "seconds": att.seconds})

    for ex in extra or []:
        rows.append(json.dumps(ex, sort_keys=True))
    out.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(r + "\n" for r in rows)
    out.write_text(body)
    manifest = {"format": "openai-chat-tools", "file": str(out), "sha256": _sha(body), "examples": len(rows),
                "system_prompt_sha256": _sha(SYSTEM_PROMPT)[:16], "tasks": included, "skipped": skipped,
                "from_traces": len(extra or [])}
    write_json(out.with_name("manifest.json"), manifest)
    if run:
        run.add("data", **manifest)
    return manifest
