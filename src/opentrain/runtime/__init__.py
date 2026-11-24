"""Runtimes: the two ways a model attempts a task.

``direct``    OpenTrain's own small agent loop against any OpenAI-compatible endpoint. This is how
              the trained specialist works, so its system prompt and tool schemas are the ones the
              specialist is trained under.
``headless``  the user's own ``claude -p`` / ``codex exec``, under their own login. It runs both
              frontier attempts and the engine's roles (trainer, evaluator).

Neither runtime verifies anything: callers run the task's verifier on the workspace afterwards.
"""

from .direct import SYSTEM_PROMPT, TOOLS, DirectRunner
from .headless import HeadlessResult, HeadlessRunner, headless_command, run_headless

__all__ = ["SYSTEM_PROMPT", "TOOLS", "DirectRunner", "HeadlessResult", "HeadlessRunner", "headless_command",
           "run_headless"]
