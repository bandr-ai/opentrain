---
name: data/diagnose
description: Name what goes wrong in the person's coding-agent traces, in their words, so they can choose what the model should get better at. A looped prompt over the moments the import flagged.
---

# data/diagnose

The import flags moments: the person corrected the agent (`corrected`), stopped it
(`interrupted`), a check kept failing (`failing_check`), a file was edited over and over
(`churn`), tool calls errored (`tool_errors`). They are evidence, not verdicts. You turn them into
a few themes the person recognises.

## Loop

1. `moments` (limit 40), newest first. For each batch, write down candidate themes: what the agent
   did wrong and in what situation, in the person's vocabulary (their repo's names, their words
   from the correction). "Uses the old auth API instead of `auth/v2`" beats "API misuse".
2. Open a few moments with `trace_show trace=… request=…` when the line is ambiguous: read what
   the agent did right before the correction.
3. Next batch (`offset`). Merge, split or drop themes as evidence arrives. Stop when a full batch
   changes nothing, or after about 200 moments.
4. Keep themes that recur (3+ moments, or 2 with an explicit correction). Leave out one-offs and
   moments about the environment (network down, missing credentials): no training fixes those.
5. `themes_set`: for each, name, one-sentence description, count, evidence (`trace:request` ids)
   and the fix (what data teaches it: the corrected continuation as the target, repair tasks from
   the failing checks, held-out tasks from the same repo).
6. One `question`, multi-select: the themes (with counts) plus "none: just do my routine work,
   cheaper". `themes_choose` with the answer.

## Honesty

Counts are of moments, not of distinct failures; say "seen in 7 sessions", not "fails 7 times".
A theme fixed is one whose held-out tasks now pass; say how many there are.
