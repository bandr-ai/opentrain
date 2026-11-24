"""Read Claude Code session logs (``~/.claude/projects/<project>/<session>.jsonl``).

Time is split by walking the main chain's events in order and attributing each gap to what
the session was waiting on when the gap began:

- the next event is a person's prompt                   -> human
- a tool call is still waiting for its result           -> tool
- otherwise (a prompt or a result was just handed over) -> model

Parallel tool calls are counted once, as wall-clock time. A subagent's events are left out of
the timeline: from the main loop's view a subagent is one long tool call, and that is where its
time lands. A model or tool gap longer than ``IDLE_CAP`` is a laptop lid closing, not work, and
is counted as idle.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from .model import Interval, Session, TestRun, ToolCall

IDLE_CAP = 30 * 60.0

DEFAULT_ROOT = Path.home() / ".claude" / "projects"

_SHELL_TOOLS = {"Bash", "bash", "shell", "exec_command"}
_EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
_WRAPPERS = re.compile(
    r"<(system-reminder|command-name|command-message|command-args|local-command-stdout|"
    r"local-command-stderr|local-command-caveat|artifact-view-context)\b[^>]*>.*?</\1>",
    re.DOTALL,
)

TEST_COMMAND = re.compile(
    r"(?<![\w-])("
    r"pytest|py\.test|python3? -m (pytest|unittest)|tox|nox"
    r"|(npm|pnpm|yarn|bun)( run)? test\b|vitest|jest|mocha|playwright test"
    r"|go test|cargo (test|nextest)|make (test|check)\b|ctest"
    r"|rspec|bundle exec rake( test)?|phpunit|mvn( -\S+)* test|gradlew? test|dotnet test|mix test|swift test"
    r")"
)
_FAIL = re.compile(
    r"(\b\d+ (failed|failing|errors?)\b|^FAILED\b|^FAIL\b|Tests?:\s+\d+ failed|test result: FAILED"
    r"|^--- FAIL|AssertionError|Traceback \(most recent call last\)|ERRORS? collecting)",
    re.MULTILINE,
)
_ZERO_FAIL = re.compile(r"\b0 (failed|failing|errors?)\b")


def discover(root: Path = DEFAULT_ROOT) -> Iterator[Path]:
    """Every main-session log under ``root``. Subagent logs nested in session folders are skipped."""
    if not root.is_dir():
        return
    for project in sorted(p for p in root.iterdir() if p.is_dir()):
        yield from sorted(project.glob("*.jsonl"))


def _ts(record: dict) -> datetime | None:
    raw = record.get("timestamp")
    if not isinstance(raw, str):
        return None
    try:
        ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _content(record: dict) -> list[dict]:
    message = record.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def _result_text(block: dict) -> str:
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict))
    return ""


def clean_prompt(text: str) -> str:
    """The words a person typed, without the harness's wrappers and reminders."""
    text = _WRAPPERS.sub("", text)
    return text.strip()


def _human_prompt(record: dict) -> str | None:
    """The prompt text when this record is a person speaking, else None."""
    if record.get("type") != "user" or record.get("isSidechain") or record.get("isMeta"):
        return None
    if record.get("isCompactSummary") or record.get("isVisibleInTranscriptOnly"):
        return None
    blocks = _content(record)
    if not blocks or any(b.get("type") == "tool_result" for b in blocks):
        return None
    text = clean_prompt("\n".join(b.get("text", "") for b in blocks if b.get("type") == "text"))
    return text or None


def classify_test_result(text: str, is_error: bool) -> str:
    if _FAIL.search(text) and not (_ZERO_FAIL.search(text) and not re.search(r"\b[1-9]\d* failed", text)):
        return "fail"
    return "fail" if is_error else "pass"


def load_session(path: Path, idle_cap: float = IDLE_CAP) -> Session | None:
    """Read one session log. Returns None for a log with no timestamped conversation in it."""
    records: list[dict] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)

    results: dict[str, tuple[dict, datetime | None]] = {}
    for r in records:
        if r.get("type") == "user":
            for b in _content(r):
                if b.get("type") == "tool_result" and isinstance(b.get("tool_use_id"), str):
                    results[b["tool_use_id"]] = (b, _ts(r))

    cwd = branch = session_id = instruction = None
    human_turns = 0
    models: list[str] = []
    tool_calls: list[ToolCall] = []
    tests: list[TestRun] = []
    edited: list[str] = []
    cost = api_s = tool_s = None
    # Main-chain timeline: (time, kind, payload)
    timeline: list[tuple[datetime, str, object]] = []

    for r in records:
        rtype = r.get("type")
        if rtype == "cost-state":
            cost = r.get("totalCostUSD", cost)
            if isinstance(r.get("totalAPIDuration"), (int, float)):
                api_s = r["totalAPIDuration"] / 1000
            if isinstance(r.get("totalToolDuration"), (int, float)):
                tool_s = r["totalToolDuration"] / 1000
            continue
        if rtype not in ("user", "assistant"):
            continue
        cwd = cwd or r.get("cwd")
        branch = branch or r.get("gitBranch")
        session_id = session_id or r.get("sessionId")
        ts = _ts(r)
        side = bool(r.get("isSidechain"))

        prompt = _human_prompt(r)
        if prompt is not None:
            if not prompt.startswith("[Request interrupted"):
                human_turns += 1
                instruction = instruction or prompt
            if ts:
                timeline.append((ts, "prompt", None))
            continue

        if rtype == "user":
            if ts and not side:
                ids = [b["tool_use_id"] for b in _content(r) if b.get("type") == "tool_result"]
                if ids:
                    timeline.append((ts, "result", ids))
            continue

        model = (r.get("message") or {}).get("model")
        if model and model != "<synthetic>" and model not in models:
            models.append(model)
        calls = []
        for b in _content(r):
            if b.get("type") != "tool_use" or not isinstance(b.get("id"), str) or ts is None:
                continue
            name = b.get("name") or "tool"
            args = b.get("input") if isinstance(b.get("input"), dict) else {}
            res, res_ts = results.get(b["id"], (None, None))
            is_error = bool(res and res.get("is_error"))
            command = args.get("command") if name in _SHELL_TOOLS else None
            if isinstance(command, list):
                command = " ".join(map(str, command))
            tool_calls.append(ToolCall(b["id"], name, ts, res_ts, is_error, command, side))
            calls.append(b["id"])
            if name in _EDIT_TOOLS:
                fp = args.get("file_path") or args.get("notebook_path")
                if isinstance(fp, str) and fp not in edited:
                    edited.append(fp)
            if isinstance(command, str) and TEST_COMMAND.search(command):
                outcome = "unknown" if res is None else classify_test_result(_result_text(res), is_error)
                tests.append(TestRun(b["id"], command[:500], ts, outcome))
        if ts and not side:
            timeline.append((ts, "assistant", calls))

    if not timeline:
        return None
    timeline.sort(key=lambda e: e[0])

    intervals: list[Interval] = []
    pending: set[str] = set()
    untimed = 0
    for i, (ts, kind, payload) in enumerate(timeline):
        if kind == "assistant":
            pending.update(payload)  # type: ignore[arg-type]
        elif kind == "result":
            pending.difference_update(payload)  # type: ignore[arg-type]
        elif kind == "prompt":
            pending.clear()
        if i + 1 == len(timeline):
            break
        nxt_ts, nxt_kind, _ = timeline[i + 1]
        if nxt_ts <= ts:
            continue
        if nxt_kind == "prompt":
            gap = "human"
        elif pending:
            gap = "tool"
        else:
            gap = "model"
        if gap != "human" and (nxt_ts - ts).total_seconds() > idle_cap:
            gap = "idle"
        intervals.append(Interval(gap, ts, nxt_ts))  # type: ignore[arg-type]
    untimed = sum(1 for c in tool_calls if c.ended_at is None and not c.sidechain)

    return Session(
        session_id=session_id or path.stem,
        agent="claude-code",
        path=str(path),
        cwd=cwd,
        git_branch=branch,
        started_at=timeline[0][0],
        ended_at=timeline[-1][0],
        instruction=instruction,
        human_turns=human_turns,
        tool_calls=tool_calls,
        intervals=intervals,
        test_runs=tests,
        files_edited=edited,
        models=models,
        cost_usd=cost,
        reported_api_seconds=api_s,
        reported_tool_seconds=tool_s,
        untimed=untimed,
    )


def load_all(root: Path = DEFAULT_ROOT) -> tuple[list[Session], list[tuple[str, str]]]:
    """Every readable session under ``root``, plus (path, reason) for each one that was not."""
    sessions: list[Session] = []
    unreadable: list[tuple[str, str]] = []
    for path in discover(root):
        try:
            s = load_session(path)
        except OSError as exc:
            unreadable.append((str(path), f"unreadable: {exc}"))
            continue
        if s is None:
            unreadable.append((str(path), "no timestamped conversation"))
        else:
            sessions.append(s)
    return sessions, unreadable
