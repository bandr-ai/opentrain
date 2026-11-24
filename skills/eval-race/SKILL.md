---
name: eval/race
description: Replay one real task against a frontier model and the specialist side by side, each verified by the task's tests. The launch demo, and a sanity check after a promotion.
---

# eval/race

## Steps

1. `opentrain race` takes the newest held-out task by default (`--task <id>` for another).
2. Both lanes run at once in separate worktrees: the frontier through the user's own CLI, the
   specialist through `opentrain serve`'s endpoint.
3. The result shows time, steps, cost and pass/fail per lane, and the winner.

## Rules

Show both lanes honestly, including a loss. Mark any sped-up footage as sped up.
