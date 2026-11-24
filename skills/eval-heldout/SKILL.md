---
name: eval/heldout
description: Score a version on held-out tasks through the evaluator: pass/fail and time per task id, never task content. The only evidence the gate accepts.
---

# eval/heldout

## Steps

1. `opentrain eval --model v3` (or the `eval_request` tool). The evaluator runs the version on every
   held-out task in fresh worktrees and verifies each attempt.
2. You get the pass rate, the median time on passes and pass/fail per task id. Nothing else leaves
   the evaluator: no instructions, no test output, no transcripts.
3. Requests are capped per run (`eval cap` in the Trainfile). Use eval to confirm, not to search.
4. Before any promotion the held-out verifiers' controls are rerun; a broken verifier blocks the gate.

## Measure

Pass rate not lower than the live version's, and the goal's objective better (time for speed,
tokens for cost, the flagged tasks for fix).
