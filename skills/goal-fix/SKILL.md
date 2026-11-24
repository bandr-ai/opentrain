---
name: goal/fix
description: File a bug against the model: reproduce a flagged failure, build related tasks, train, and promote only if the flagged task now passes and nothing regresses.
---

# goal/fix

## The recipe

| Part | Skill |
| --- | --- |
| data | the flagged session (held out) + data/expand related tasks (training) |
| training | train/sft |
| proof | the flagged tasks pass; no other held-out task regresses |

## Run it

`opentrain fix --session latest -m "it keeps using our old auth API"` flags the session and starts
the engine on this recipe. `--flag-only` records the flag for the next run.

## Honesty

Fixing a family of failures from one example is unproven. Report how many related tasks were found
and how many now pass; do not claim the family is fixed beyond that.
