---
name: goal/coding
description: Make a small model that does the person's own coding work, from every coding agent's traces on their machine and the repos they worked in. The playbook for "train on my coding", "make me a small coding model", "fix what my agent keeps getting wrong".
---

# goal/coding

The person's own work is the data: their coding-agent traces (Claude Code, Codex, Cursor, opencode,
Gemini CLI, Cline, Aider, pi, Copilot) and the repos those traces touched. The result is a small
model, served on their own account, that does that work about as well as what they use today, for
far less, with proof on tasks it never saw. Adapt every step to what you find; the order below is
the usual one, not a script.

## 1. Show them what they have (free, local)

`traces_scan`, then tell them in two lines what was found per agent ("587 Claude Code sessions,
26 Codex"). `traces_import` reads and indexes them locally (credentials scrubbed; nothing is
uploaded). Report: how many traces changed code, in which repos, how many training examples, how
many moments where something went wrong. If almost nothing changed code, say so plainly: this
machine has little coding history, and the repos' commit history (step 3) carries the work.

## 2. Ask what to fix (skill data/diagnose)

Read `moments` in batches and name the recurring themes in their words. `themes_set`, then one
`question` with the themes as options (multi-select) plus "just do my routine work, cheaper".
`themes_choose` records the answer. Chosen themes decide which traces matter most and which
held-out tasks the report breaks out.

## 3. Tasks the tests decide (skills cl/mine, data/history)

- `tasks_mine` (`include_repairable=true`): tasks from traces that ended in a commit with tests.
- `tasks_from_history`: tasks from the repos' own commits that changed code and tests, each
  rewritten into the request a person would type (free on their Claude login).
- Check the split (`status`, `tasks_list`): the newest tasks are held out. Under ~10 held-out
  tasks a pass-rate difference of one or two tasks is noise; mine more before claiming anything.

## 4. Pick the student (`students`)

Offer the tiers in one `question`, cheapest first, with the reason for each:
- small: Inkling 2 Mini on Tinker (strongest coding per active parameter; Tinker key needed),
  Qwen3.5-4B / 9B on Modal or RunPod (cheapest self-hosted; works with their GPU account today);
- mid: Qwen3.8-27B (Tinker or one H200 on Modal), Qwen3.6-35B-A3B (3B active);
- large: Inkling-Small on Tinker, when the small ones fall short.
`trainer_set` with their choice. Offer what `students` shows as `ready_on`; for a trainer that is
not connected, say what to connect (Settings, `opentrain login runpod`, or a Tinker key).

## 5. Baselines (proof starts here)

`eval_request` on the held-out tasks for: what they use today (the frontier baseline the gate
measures; free on their Claude login) and `version="base"` (the untrained student). Report pass
rate, median time and cost per task for each, and stop if the small base is already good enough.

## 6. Data and training

- `sample_run` (teacher frontier, n 2-4): the teacher's verified passes on training tasks.
- `data_build` (traces auto, or themes): shortest verified teacher pass per task plus the imported
  trace examples (left out automatically: anything on or after the held-out cutoff).
- `train_estimate`, then `propose` the SFT run with its cost; `job_start` once approved.

## 7. Prove, ship, repeat

`eval_request` the new version on the same held-out tasks; `version_propose`; report fixed and
regressed tasks, and the chosen themes' tasks separately. Then the next step, one change per run:
more tasks (newer sessions, more commits), more teacher passes per task, the themes-only data, or
a bigger student. Rejection sampling on the student's own coding passes is not wired yet: do not
promise it.
