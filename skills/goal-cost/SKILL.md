---
name: goal/cost
description: Same results with fewer tokens: keep the verified pass with the fewest tokens and train on it, proved by tokens per task at the same pass rate.
---

# goal/cost

## The recipe

| Part | Skill |
| --- | --- |
| data | cl/rollout, then keep the fewest-token verified pass per task |
| training | train/sft |
| proof | median tokens on shared passes fall; pass rate not lower; nothing regresses |

Set `goal  cost` in the Trainfile. The selection today reuses data/shortest (fewest steps), which
tracks tokens closely; a token-exact selector is the first improvement to make here.
