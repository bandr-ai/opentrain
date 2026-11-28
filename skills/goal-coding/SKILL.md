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

## 2. What went wrong, as evidence (skill data/diagnose)

The goal is their routine work, cheaper; do not ask them to pick what to fix from a menu. Read
`moments` in batches, name the recurring themes in their words and sort each one: **trainable**
(how the model does the work: does more than asked, wrong approach in their stack, long or vague
reports) or **a rule, not training** (commit trailers, spending without asking, a setting, the
environment: say the one-line rule or setting that fixes it; OpenTrain's approvals already gate
spending). `themes_set`, then `data_card`: the themes show there with their evidence.
Ask nothing about them. A trainable theme seen in 5+ sessions you read as real goes in:
`themes_choose` it and say so in a line ("Also training against <theme>, seen in 8 sessions").

## 3. Tasks, and what checks them (skills cl/mine, data/history)

The verifier is chosen per task, strongest first: the repo's tests; tests you write for the task
(`task_add` accepts them only when the reference passes and doing nothing fails); a judge comparing
the change with the real one, after the checks the work must not break (a build, a type check).
- `tasks_mine` (`include_repairable=true`): tasks from traces that ended in a commit with tests.
- `tasks_from_history` (verifier auto): tasks from the repos' own commits, each rewritten into the
  request a person would type (free on their Claude login); the commit's tests check it when it
  changed tests, a judge otherwise (pass `checks` such as the repo's type check or build).
- Say how many tasks each kind of verifier checks; a judge is weaker proof than tests.
- Check the split (`status`, `tasks_list`): the newest tasks are held out. Under ~10 held-out
  tasks a pass-rate difference of one or two tasks is noise; mine more before claiming anything.

## 4. Pick the student (`students`)

Pick only models `students` lists, by their exact ids; never a size or family it does not list.

If they named a model, use it (one line if it is not on the list or you expect it to fall short, then
the closest listed one). Otherwise pick yourself and say why in a line: the smallest one that is
`ready_on` a trainer they have connected. Ask only when no trainer is connected, and then only what to
connect. The tiers, cheapest first:
- small: Inkling 2 Mini on Tinker (strongest coding per active parameter; Tinker key needed; a free
  tier at 64K context as `thinkingmachines/Inkling-2-Mini:peft:65536:free`, good for a first run),
  Qwen3.5-4B / 9B on Modal or RunPod (cheapest self-hosted; works with their GPU account today);
- mid: Qwen3.8-27B (Tinker or one H200 on Modal), Qwen3.6-35B-A3B (3B active);
- large: Inkling-Small on Tinker, when the small ones fall short.
`trainer_set` with the pick. For a trainer that is not connected, say what to connect (Settings,
`opentrain login runpod`, or a Tinker key).

## 5. One plan, one go-ahead

Before anything is paid for, one `recipe_plan` page covers the whole run: the baselines below, the
teacher passes, training and the re-score, with the total cost and time, and its `approves` (spend,
upload, job.start; cost_usd the total). After "Run it", do steps 5 to 7 without asking again.

Baselines (proof starts here): `eval_request` on the held-out tasks for: what they use today (the frontier baseline the gate
measures; free on their Claude login) and `version="base"` (the untrained student). Report pass
rate, median time and cost per task for each, and stop if the small base is already good enough.

## 6. Data and training

- `sample_run` (teacher frontier, n 2-4): the teacher's verified passes on training tasks.
- `data_build` (traces auto, or themes): shortest verified teacher pass per task plus the imported
  trace examples (left out automatically: anything on or after the held-out cutoff).
- `train_estimate` (its cost goes in the plan's total); `job_start` once the plan is approved.

## 7. Prove, ship, repeat

`eval_request` the new version on the same held-out tasks; `version_propose`; report fixed and
regressed tasks, and the chosen themes' tasks separately. Then the next step, one change per run:
more tasks (newer sessions, more commits), more teacher passes per task, the themes-only data, or
a bigger student. Rejection sampling on the student's own coding passes is not wired yet: do not
promise it.
