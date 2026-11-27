---
name: cl/filter
description: Sort tasks by how often a model passes them (never, sometimes, always) to decide which tasks a recipe can learn from. Use after cl/rollout, before choosing a training recipe.
---

# cl/filter

## What the buckets mean

| Bucket | For goal/speed (SFT on passes) | For RL or rejection sampling |
| --- | --- | --- |
| never passed | nothing to imitate; needs a stronger teacher or a hint | no signal; drop |
| sometimes | the best data: imitate the shortest pass | the best data |
| always passed | still useful: the shortest pass teaches a shorter path | no signal; drop |

## Steps

1. The `sample_run` tool returns the buckets (CLI: `opentrain debug sample`).
2. For never-passed tasks, try the frontier teacher before giving up; if it also fails, the task may
   need context the instruction lacks (see cl/mine: multi-prompt sessions).
3. Say which bucket each task fell in when you report, so the next run starts from it (the run
   record keeps the sample results: `record_query`).
