# Changelog

All notable changes to opentrain-cli. Versions follow [semantic versioning](https://semver.org); before
1.0 a minor version may change the command line, the Trainfile or the project's state format, and
says so here.

## [Unreleased]

## [0.2.0] - 2026-11-27

The first release on PyPI, for beta users.

### A small model that does your coding
- `opentrain traces` finds every coding agent's logs on the machine (Claude Code, Codex, Cursor,
  opencode, Gemini CLI, Cline / Roo / Kilo, aider, pi, Copilot); `--import` reads them locally with
  credentials scrubbed. The web app's start screen shows the same.
- Where traces went wrong (corrections, interruptions, failing checks, churn) becomes themes you pick
  to fix (`data/diagnose`).
- Verified coding tasks from sessions and from the repos' own commits (`tasks_from_history`), each
  commit rewritten into the request a person would type and certified by its tests.
- Training data from teacher passes, your own traces (nothing from the held-out period) and public
  trajectories (SWE-smith, SWE-Gym, R2E-Gym), in the trained model's five tools; loss on every turn.
- Student models with their trainers and serving flags: Inkling 2 Mini and Inkling-Small (Tinker),
  Qwen3.5 4B/9B, Qwen3.8-27B, Qwen3.6-35B-A3B (Modal, RunPod, Tinker). Served models call tools.

### Any other use case
- Bring your data or name a use case: datasets from files, Hugging Face and Kaggle; graders (exact,
  contains, regex, json, number, judge, command, python `reward()`); sealed held-out rows.
- Four training methods: SFT, rejection sampling, on-policy distillation, RL (GRPO), on Modal,
  RunPod, Fireworks or Tinker, priced before they run and held to the Trainfile's budget.
- The gate ships a version only when it beats the baseline with nothing regressing.

### Ways to use it
- `opentrain web`, the terminal app, `opentrain run`, and `opentrain mcp` for Claude Code.
- `opentrain report`: counts only, to send with feedback.
- Logins other tools keep (Modal, Hugging Face, Kaggle, RunPod, Fireworks) are used as they are.

[Unreleased]: https://github.com/bandr-ai/opentrain/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/bandr-ai/opentrain/releases/tag/v0.2.0
