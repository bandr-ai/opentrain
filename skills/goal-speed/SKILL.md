---
name: goal/speed
description: Make the user's routine coding tasks much faster at the same pass rate: data/shortest paired with train/sft, proved by median time per task on held-out work.
---

# goal/speed

## The recipe

| Part | Skill |
| --- | --- |
| data | cl/mine -> data/heldout -> cl/rollout -> data/shortest |
| training | train/sft |
| proof | eval/heldout, then the gate: median time on shared passes falls at least 5% and no task regresses |

## Run it

`opentrain train` runs the whole thing with an approval step. `opentrain watch` runs it unattended
on new sessions.

## Before promising a speedup

`opentrain bench time` shows the model's share of working time. If tools take half the time, even
an instant model gives at most 2x end to end. Say so rather than promise 5x.
