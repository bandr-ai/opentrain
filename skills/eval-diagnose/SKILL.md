---
name: eval/diagnose
description: Find which held-out tasks flipped between two versions, so a regression is understood before the next run.
---

# eval/diagnose

## Steps

1. `opentrain diff v3 v4` lists tasks that went from pass to fail (REGRESSED) and fail to pass (fixed),
   with times.
2. For a regressed id, the trainer cannot read the task. Look at the training tasks of the same
   family instead (`tasks_list --family`) and at what changed in the data manifest between the runs
   (`opentrain log <run>`).
3. Write the suspected cause in a `note` so the next run can test it.
