---
name: eval/diagnose
description: Find which held-out tasks flipped between two versions, so a regression is understood before the next run.
---

# eval/diagnose

## Steps

1. Compare the two versions' evals: `eval_request` results (coding tasks) or `evals eval_id=` for
   each (datasets) list pass/fail per row id; `dataset_promote` reports `fixed` and `regressed`
   ids. Find the ids that went from pass to fail.
2. For a regressed id, you cannot read the held-out task. Look at the training tasks of the same
   family instead (`tasks_list family=...`) and at what changed in the data between the runs
   (`record_query`).
3. State the suspected cause in your report so the next run can test it.
