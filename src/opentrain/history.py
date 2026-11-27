"""Tasks from a repo's own history (recipe ``data/history``): the commits that changed code and tests.

Sessions are the best tasks (the person's own words), but few sessions end in a committed change
with tests, so a laptop yields a handful. The repos those sessions worked in hold many more such
changes. A commit that touched both code and its tests becomes a task the same way a session does:
the repo at the commit's parent is the environment, the commit's message is the instruction, its
test files are the verifier and the rest of its diff is the reference answer. Each one is built and
certified by :mod:`opentrain.mine` (reference passes, doing nothing fails), so nothing is accepted on
trust, and the held-out split keeps the newest of them away from training.

A commit message is a weaker instruction than a person's prompt ("fix the thing" says little, and
"Use bisect in pager.page" gives the answer away). So by default a model rewrites it, from the
message and the diff, into what a person would have asked for: the behaviour wanted, in their words,
without naming the fix (Repo2RLEnv rewrites commits the same way). File and function names a person
would mention stay; the change itself does not. Leaks that survive (commit hashes, links, "fixes #12")
are stripped. Short messages, merges and very large changes are skipped with their reason.
"""

from __future__ import annotations

import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .bench.gitstate import TEST_FILE, RepoState
from .bench.yield_ import Verdict
from .mine import MineError, mine_one
from .sessions.model import Session

MIN_WORDS = 4
MAX_LINES = 600
MAX_FILES = 25


def _git(repo: str, *args: str) -> str:
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def default_author(repo: str) -> str | None:
    return _git(repo, "config", "user.email").strip() or None


def candidates(repo: str, *, author: str | None = None, since: str | None = None,
               limit: int = 40) -> tuple[list[dict[str, Any]], Counter]:
    """Commits that could become tasks, newest first, and why the others could not."""
    args = ["log", "--no-merges", "--format=%x1e%H%x1f%P%x1f%aI%x1f%B%x1f", "--numstat"]
    if author:
        args.append(f"--author={author}")
    if since:
        args.append(f"--since={since}")
    out: list[dict[str, Any]] = []
    skipped: Counter = Counter()
    for chunk in _git(repo, *args).split("\x1e"):
        if not chunk.strip():
            continue
        sha, parents, when, message, stat = (chunk.split("\x1f") + [""] * 5)[:5]
        if not parents.strip():
            skipped["first commit"] += 1
            continue
        files, lines = [], 0
        for row in stat.strip().splitlines():
            parts = row.split("\t")
            if len(parts) == 3:
                files.append(parts[2])
                lines += sum(int(x) for x in parts[:2] if x.isdigit())
        tests = [f for f in files if TEST_FILE.search(f)]
        code = [f for f in files if f not in tests]
        msg = message.strip()
        if not tests:
            skipped["no test change"] += 1
        elif not code:
            skipped["tests only"] += 1
        elif len(msg.split()) < MIN_WORDS:
            skipped["message too short to be an instruction"] += 1
        elif lines > MAX_LINES or len(files) > MAX_FILES:
            skipped["change too large"] += 1
        else:
            out.append({"repo": repo, "commit": sha, "parent": parents.split()[0], "at": when, "message": msg,
                        "files": len(files), "lines": lines})
            if len(out) >= limit:
                break
    return out, skipped


REWRITE = """You turn a git commit into the request a developer would have typed to a coding agent \
before the work was done. Write it from the commit message and the diff below.

Rules:
- Describe the behaviour wanted or the problem to fix, the way the person would ask for it.
- Do not reveal the solution: no code from the diff, no description of how it was implemented, no new \
function or variable names the diff introduces. Existing files, functions and commands a person would \
mention are fine.
- Do not mention tests, the commit, or that this was rewritten.
- One short paragraph, at most 120 words. Output only the request."""

_LEAKS = re.compile(r"\b[0-9a-f]{12,40}\b|https?://\S+|\b(fix(es|ed)?|close[sd]?|resolve[sd]?) #\d+\b|"
                    r"^(Signed-off-by|Co-Authored-By|Change-Id):.*$", re.IGNORECASE | re.MULTILINE)


def strip_leaks(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]*\(\s*\)", "", _LEAKS.sub("", text))).strip()


def rewrite(c: dict[str, Any], model: str = "claude-code/sonnet") -> str:
    """The commit as a person's request, by ``model`` (free on the person's Claude login by default)."""
    from .core.complete import complete

    diff = _git(c["repo"], "show", "--format=", "--stat", "--patch", "--no-color", c["commit"])
    tests = "\n".join(f for f in _git(c["repo"], "show", "--format=", "--name-only", c["commit"]).splitlines()
                      if TEST_FILE.search(f))
    body = f"Commit message:\n{c['message']}\n\nTest files (hidden from the agent): {tests}\n\nDiff:\n{diff[:20000]}"
    out = complete(model, [{"role": "system", "content": REWRITE}, {"role": "user", "content": body}],
                   max_tokens=400, timeout=240).text
    return strip_leaks(out)


def verdict_for(c: dict[str, Any]) -> Verdict:
    """The commit dressed as a judged session, so the session miner builds and certifies it."""
    when = datetime.fromisoformat(c["at"])
    when = when if when.tzinfo else when.replace(tzinfo=timezone.utc)
    instruction = c.get("instruction") or strip_leaks(c["message"])
    s = Session(session_id=f"commit-{c['commit'][:12]}", agent="git", path=f"{c['repo']}@{c['commit']}",
                cwd=c["repo"], git_branch=None, started_at=when, ended_at=when, instruction=instruction,
                human_turns=1, prompts=[instruction])
    rs = RepoState(root=c["repo"], base_commit=c["parent"], tests_in_repo=True, reference_commit=c["commit"])
    return Verdict(session=s, repo=rs, passed=["instruction", "edits", "repo state", "tests in repo"], skip=None)


def mine_history(state: Path, repos: list[str], *, author: str | None = None, since: str | None = None,
                 limit: int = 40, timeout: float = 600, rewrite_with: str | None = "claude-code/sonnet",
                 say=None) -> dict[str, Any]:
    """Build and certify tasks from each repo's commits; one row per commit tried. ``rewrite_with``
    names the model that turns each commit into a request (None: the stripped message as is)."""
    rows: list[dict[str, Any]] = []
    skipped: Counter = Counter()
    per_repo = max(1, limit // max(1, len(repos)))
    for repo in repos:
        if not Path(repo, ".git").exists():
            rows.append({"repo": repo, "outcome": "rejected", "reason": "not a git repo"})
            continue
        who = author if author is not None else default_author(repo)
        found, why = candidates(repo, author=who or None, since=since, limit=per_repo)
        skipped.update(why)
        if say:
            say(f"{Path(repo).name}: {len(found)} commits with code and tests to certify")
        for c in found:
            base = {"repo": repo, "commit": c["commit"][:12], "message": c["message"].splitlines()[0][:100]}
            try:
                if rewrite_with:
                    try:
                        c["instruction"] = rewrite(c, rewrite_with)
                    except Exception as exc:  # noqa: BLE001 - fall back to the message, and say so
                        base["rewrite"] = f"kept the commit message ({type(exc).__name__}: {str(exc)[:120]})"
                rows.append({**base, **mine_one(verdict_for(c), state, timeout)})
            except MineError as exc:
                rows.append({**base, "outcome": "rejected", "reason": exc.reason, "detail": exc.detail})
            except Exception as exc:  # one odd commit must not cost the batch
                rows.append({**base, "outcome": "rejected", "reason": "error",
                             "detail": f"{type(exc).__name__}: {exc}"[:400]})
            if say:
                r = rows[-1]
                say(f"  {base['commit']} {r['outcome']}: {r.get('reason', '')}")
    outcomes = Counter(r["outcome"] for r in rows)
    return {"tried": len(rows), "accepted": outcomes["accepted"], "kept": outcomes["kept"],
            "rejected": outcomes["rejected"],
            "reasons": dict(Counter(r["reason"] for r in rows if r["outcome"] == "rejected").most_common()),
            "not_tried": dict(skipped.most_common()), "rows": rows}
