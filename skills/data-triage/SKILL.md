---
name: data/triage
description: Decide how good a dataset or a pile of traces is before training on it, by reading rows in a loop until the picture is stable, then turn the verdict into a plan (train as is, filter, relabel, or keep only the prompts).
---

# data/triage

Traces and imported data are often worse than they look: wrong or truncated answers, near-copies,
off-task chatter, private data. Training on them teaches exactly that. Read before you train.

Keep a running tally in `todo` (rows read; ok, fixable, bad; the bad rows' ids by reason) so the
loop survives a long conversation.

## The loop

Repeat, one batch at a time:

1. **Read a batch** with `dataset_show`, spread across the whole set (shift the offset each batch),
   never only the first rows. About 25 rows shows the patterns; take fewer when rows are long.
2. **Label every row**: `ok`, `fixable` (right idea, bad reference: wrong, incomplete, wrong format)
   or `bad`, with a reason: wrong answer, incomplete, off-task, format, duplicate, trivial, unsafe or
   private data, needs context the prompt does not give.
3. **Update the tally.**
4. **Stop** when another batch would not change your verdict (the shares and the kinds of problems
   have settled), or you have read every row. Small sets: read them all.

When the set is too big to read enough of it, add a scaled check: `dataset_eval` a strong free
model (`claude-code/sonnet`) on `split=train` with a `limit` (it samples across the set) and the
dataset's grader, then `evals eval_id=...` shows each answer next to its score. Rows where it
disagrees with the reference are suspects; decide for a sample of them whether the reference or the
model is wrong. The share of wrong references is a second estimate; report both.

## The verdict

Report in a few lines: rows read, the share that is ok, the main problems with example ids, and a
plan. Weigh the kind of problem, not just the share: wrong answers poison training far more than
formatting does.

- **Mostly ok, problems cosmetic:** train on it (`sft` first).
- **A minority of wrong or broken rows:** remove the ones you found (`dataset_edit remove_ids`);
  if they are common enough that you cannot find them all, relabel with the best baseline as
  teacher (`dataset_label`, proposed with its cost).
- **Answers mostly unreliable:** do not train on the references. Keep the prompts: distil new
  answers from a strong teacher, or, when answers can be checked, use the prompts for `rejection`
  or `rl` (see train/env).
- Near-copies across training and held-out rows make the eval lie: remove them first.
- Private data (names, emails, keys) in rows that would be uploaded: tell the user before anything
  leaves the machine.

Never "fix" held-out rows by reading them; only training rows are yours to read.
