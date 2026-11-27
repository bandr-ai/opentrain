"""A judge as a task's verifier, for repos without tests (the last rung of the verifier ladder).

The ladder, strongest first: the repo's own tests; tests written for the task (accepted only when
their controls hold, see ``task_add``); and this judge. A judged task has ``test_command`` set to
``JUDGE``. Its verdict: the checks the work must not break (``meta["checks"]``: a build, a type check,
a lint the trace ran) must pass, the change must not be empty, and a model reading the request, the
reference change and the candidate change must say the candidate does what was asked.

A judge is a weaker proof than tests, so a judged task is certified like any other before it counts
(the reference passes, doing nothing fails, a partial reference fails), and its kind shows next to
its scores. It calls ``claude-code/sonnet`` by default: free on the person's Claude login.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

JUDGE = "opentrain-judge"
MODEL = os.environ.get("OPENTRAIN_JUDGE_MODEL", "claude-code/sonnet")

PROMPT = """You check whether a code change does what a developer asked for. You get the request, the \
change the developer actually made (the reference), and a candidate change. Pass the candidate when it \
does what the request asks, as fully as the reference does: same behaviour, possibly different code. \
Fail it when it is missing part of the work, does something else, or would break what the reference \
keeps working. Style and naming differences do not matter.

Answer with one line: PASS or FAIL, a colon, and the reason in under 25 words."""


def is_judged(test_command: str) -> bool:
    return test_command.strip().startswith(JUDGE)


def candidate_diff(workspace: Path, base_commit: str) -> str:
    """What the attempt changed, against the base commit (the linked dependency folders left out)."""
    from .workspace import ENV_DIRS

    skip = [f":(exclude,glob)**/{d}" for d in ENV_DIRS] + [f":(exclude){d}" for d in ENV_DIRS]
    subprocess.run(["git", "-C", str(workspace), "add", "-A", "--", ".", *skip], capture_output=True)
    r = subprocess.run(["git", "-C", str(workspace), "diff", "--cached", "--no-color", base_commit, "--", ".", *skip],
                       capture_output=True, text=True)
    return r.stdout


def verdict(task, workspace: Path, timeout: float = 900):
    """``workspace.Verdict`` for a judged task."""
    from .core.complete import complete
    from .workspace import Verdict, reference_patch

    start = time.monotonic()
    for cmd in (task.meta or {}).get("checks") or []:
        p = subprocess.run(cmd, shell=True, cwd=workspace, capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            return Verdict(False, p.returncode, round(time.monotonic() - start, 2),
                           f"check failed: {cmd}\n{(p.stdout + p.stderr)[-2000:]}")
    cand = candidate_diff(workspace, task.base_commit)
    if not cand.strip():
        return Verdict(False, 1, round(time.monotonic() - start, 2), "no change was made")
    ref = reference_patch(task)
    body = (f"Request:\n{task.instruction[:6000]}\n\nReference change:\n{ref[:24000]}\n\n"
            f"Candidate change:\n{cand[:24000]}")
    try:
        out = complete(MODEL, [{"role": "system", "content": PROMPT}, {"role": "user", "content": body}],
                       max_tokens=200, timeout=min(timeout, 300)).text.strip()
    except Exception as exc:  # noqa: BLE001 - an unreachable judge is an environment failure, not a fail
        # exit 127 reads as "could not run" to certify.classify: inconclusive, never a pass or a fail
        return Verdict(False, 127, round(time.monotonic() - start, 2), f"judge unavailable: {exc}")
    m = re.search(r"\b(PASS|FAIL)\b", out.upper())
    ok = bool(m and m.group(1) == "PASS")
    return Verdict(ok, 0 if ok else 1, round(time.monotonic() - start, 2), f"judge ({MODEL}): {out[:400]}")
