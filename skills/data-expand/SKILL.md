---
name: data/expand
description: From one failure, build related and harder training tasks so a fix generalises instead of memorising. The data half of goal/fix. Experimental.
---

# data/expand

Fixing a whole family of failures from one example is unproven (plan: v0.3 watch). Ship it only as
far as its tests pass.

## Steps

1. Reproduce: replay the flagged session's task and confirm the current model fails it.
2. Look for the pattern: `tasks_list` filtered by family, and earlier sessions with similar edits.
   Count how many earlier tasks share the failure.
3. Build related tasks the same way cl/mine builds tasks: real repo states, real tests. A variant you
   write by hand must still pass bandits/certify (reference passes, doing nothing fails).
4. Keep the flagged task itself held out. Training on the very task the gate checks is a leak.

## Measure

The flagged task passes after training, and no other held-out task regresses.
