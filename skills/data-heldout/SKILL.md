---
name: data/heldout
description: Keep a held-out set of tasks that the evaluator alone reads, split by date so the test matches real use. Use after mining, before any evaluation, and before exporting or sharing data.
---

# data/heldout

The gate is only trustworthy if the model is graded on work it has never seen. Hold out the latest
tasks, keep them away from every trainer, and keep the set fresh.

## Rules

- **Split by date, never at random.** The latest share of tasks (by when the work happened) is held
  out. A random split puts next week's refactor in training and grades the model on its twin.
- **Once held out, always held out.** A task the evaluator has seen is never moved to training,
  even if the cutoff moves.
- **The trainer never reads `.opentrain/heldout/`.** Not the folders, not the instructions, not the
  test output. Only ids and pass/fail counts leave the evaluator.
- **Exports leave it out by default.** Exported data goes to other tools that may train on it.

## Steps

1. Set the share in the Trainfile: `heldout  30%` (the default).
2. `tasks_mine` refreshes the split after every mine: new tasks after the cutoff go to
   `.opentrain/heldout/`, owner-only (0700).
3. `opentrain task ls` shows training tasks and only a count of held-out ones; `opentrain task show`
   refuses a held-out id.
4. `tasks_lint` checks held-out verifiers too, reporting only id and reason.
5. `opentrain export <dir>` writes training tasks, `train.jsonl` and a manifest. Add
   `--include-heldout` (writes `evals.jsonl` and the held-out folders) only for an evaluator you
   control.

## Measure of success

- Held-out set is non-empty and newer than every training task.
- No held-out id appears in `.opentrain/tasks`, `train.jsonl` or a default export.
- New sessions keep adding held-out tasks, so the test does not go stale.

## Failure modes

- Too few tasks: with under ~10 accepted tasks the held-out set is too small to gate on; mine more
  (`cl/mine`) before trusting a promotion.
- Leakage through training data: if a held-out session's text is in `train.jsonl`, the split is
  void; rebuild the training data from training sessions only.
- Same-user trainer: the 0700 folder is not a wall against a process running as you; a separate
  account or container for the evaluator is the real one.
