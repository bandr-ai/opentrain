---
name: data/diagnose
description: Name what goes wrong in the person's coding-agent traces, in their words, and sort what training can fix from what a rule fixes. A looped prompt over the moments the import flagged.
---

# data/diagnose

The import flags moments: the next prompt reads like disagreement (`pushback`: a keyword match,
often just an answer to a question, so read it before counting it), the person stopped the agent
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
4. Keep themes that recur (3+ moments you read as real, or 2 with an explicit correction). Leave out one-offs and
   moments about the environment (network down, missing credentials): no training fixes those.
   A keyword "pushback" is often just how the person talks (casual swearing, "nah" to a question):
   judge it against their usual tone and count only real disagreement.
5. Sort each theme. **Trainable**: how the model does the work (scope beyond the request, the
   wrong approach for their stack, long or vague final reports, not running the checks). **A
   rule**: anything a setting or one instruction fixes for good (commit trailers, spending without
   asking, which branch to push to): training a model to remember a rule is slower and less sure
   than the rule; name the rule instead.
6. `themes_set`: for each, name, one-sentence description, count (moments you read as real),
   evidence (`trace:request` ids; the sessions count is the distinct traces in it), `trainable` (true/false) and the fix: for a trainable one, what
   data teaches it (the corrected continuation as the target, repair tasks from the failing
   checks, held-out tasks from the same repo); for a rule, the rule.
7. Do not ask the person about the themes. `data_card` shows them as evidence. A trainable theme
   seen in 5+ sessions goes in (`themes_choose`), said in a line. The default goal stays their
   routine work, cheaper.

## Honesty

Sessions are distinct traces in the evidence; five moments in one session is one session. Say "seen in 7
sessions", not "fails 7 times".
A theme fixed is one whose held-out tasks now pass; say how many there are.
