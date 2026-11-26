---
name: cl/rollout
description: Run a model several times on each training task in fresh worktrees, then verify each attempt with the task's own tests. Use to collect demonstrations (goal/speed, goal/cost) or to measure how hard each task is for a model.
---

# cl/rollout

Attempts are the raw material of every training recipe. Each one runs in a fresh `git worktree` at the
task's base commit; the runner never grades itself; the task's verifier runs afterwards on the same
worktree and its exit code alone decides pass or fail.

## Steps

1. Pick the teacher. With a `provider` line in the Trainfile, the base model itself (training then
   sharpens what it can already do). Without one, the user's frontier CLI, restricted to Bash, Read,
   Write and Edit so every pass maps one-to-one onto the specialist's tools.
2. `opentrain sample --n 4` (or the `sample_run` tool). Each attempt reserves its estimated cost
   against the budget first; a refused reservation stops sampling, and finished attempts are kept.
3. Read the result: attempts, passes, tasks that never passed, tasks that always passed.

## Measure

Verified passes per task. A task with none contributes nothing to training.

## Failure modes

- Every attempt fails on one task: check `tasks_lint` before blaming the model; the verifier may
  have broken (a dependency moved).
- Attempts are slow because tools dominate: the time breakdown in `sessions_scan` (CLI: `opentrain debug bench`) says so; a
  faster model cannot fix tool time.
- Never sample held-out tasks. The tools refuse; do not try to work around it.
