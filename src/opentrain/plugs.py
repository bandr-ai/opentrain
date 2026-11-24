"""What each plug must do (plan: "What each plug must do").

Every axis is pluggable: the trainer backend, the base model, the serving, and the agent that
attempts tasks. The core talks only to these protocols; a new backend is one module that
implements one of them and registers itself in ``BACKENDS`` / ``RUNNERS``.

Data shapes are plain dataclasses so they serialise straight into the run record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

JobState = Literal["queued", "running", "succeeded", "failed", "cancelled"]


# ---------------------------------------------------------------- training


@dataclass(frozen=True)
class TrainSpec:
    """One training job. ``data`` is a JSONL file of chat examples (see ``opentrain.data.sft``)."""

    base_model: str
    data: Path
    method: Literal["sft", "rft"] = "sft"
    lora_rank: int = 32
    epochs: int = 2
    learning_rate: float = 1e-4
    batch_size: int = 8
    max_tokens: int = 16384
    name: str = "opentrain"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JobStatus:
    job_id: str
    state: JobState
    cost_usd: float = 0.0
    """Cost accrued so far. Backends that cannot report it return an estimate and say so in ``detail``."""
    checkpoint: str | None = None
    """Set once ``state == 'succeeded'``: what ``Registry.add(checkpoint=...)`` stores."""
    progress: float | None = None
    detail: str = ""
    logs: list[str] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return self.state in ("succeeded", "failed", "cancelled")


@runtime_checkable
class TrainerBackend(Protocol):
    """Launch a job, stream logs, cancel, report cost as it accrues."""

    name: str

    def available(self) -> tuple[bool, str]:
        """Whether this backend can run here (keys, packages, hardware), and if not, what is missing."""
        ...

    def estimate(self, spec: TrainSpec) -> float:
        """Dollars this job is expected to cost. Reserved against the budget before launch."""
        ...

    def launch(self, spec: TrainSpec) -> str:
        ...

    def status(self, job_id: str) -> JobStatus:
        ...

    def cancel(self, job_id: str) -> None:
        ...


# ---------------------------------------------------------------- serving


@dataclass(frozen=True)
class Endpoint:
    """An OpenAI-compatible chat endpoint for a model version."""

    base_url: str
    model: str
    api_key_env: str | None = None
    """Name of the environment variable holding the key; never the key itself."""


@runtime_checkable
class Serving(Protocol):
    """Deploy a version, report speed, switch versions without downtime."""

    name: str

    def deploy(self, checkpoint: str, base_model: str) -> Endpoint:
        ...


# ---------------------------------------------------------------- attempting tasks


@dataclass
class Step:
    """One model turn in an attempt: what the model said or called, and how long it took."""

    role: Literal["assistant", "tool"]
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = None
    seconds: float = 0.0


@dataclass
class Attempt:
    task_id: str
    model: str
    passed: bool | None
    """None until verified."""
    seconds: float
    """End-to-end wall clock, retries and tool time included."""
    model_seconds: float = 0.0
    tool_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    patch: str = ""
    """The diff the attempt left in the workspace."""
    messages: list[dict[str, Any]] = field(default_factory=list)
    """The full transcript in OpenAI chat format, for training on (never for held-out tasks)."""
    error: str | None = None

    @property
    def steps(self) -> int:
        return sum(1 for m in self.messages if m.get("role") == "assistant")


@runtime_checkable
class AgentRunner(Protocol):
    """Runs one model on one task in a prepared workspace and returns what it did.

    The runner does not verify; the caller runs the task's verifier on the workspace afterwards,
    so no runner (and no model) can grade itself.
    """

    name: str

    def attempt(self, instruction: str, workspace: Path, *, task_id: str, timeout: float) -> Attempt:
        ...
