---
name: data/triage
description: Decide how good a dataset or a pile of traces is before training on it, in a loop that reads rows until the estimate is stable, then turn the verdict into a plan (train as is, filter, relabel, or prompts only).
---

# data/triage

Traces and imported data are often worse than they look: wrong or truncated answers, near-copies,
off-task chatter, private data. Training on them teaches exactly that. Read before you train.

Keep a running tally in `todo` (rows read, ok, fixable, bad, and the bad rows' ids by reason) so the
loop survives a long conversation.

## The loop

Repeat, one batch at a time:

1. **Read a batch.** 25 rows with `dataset_show`, spread across the whole set (offsets
   `k * total / 25` plus a shift each batch), never only the first rows.
2. **Label every row** as `ok`, `fixable` (right idea, bad reference: wrong, incomplete, wrong
   format) or `bad`, with one reason from: wrong answer, incomplete, off-task, format, duplicate,
   trivial, unsafe or private data, needs context the prompt does not give.
3. **Update the tally** and the share of `ok` rows.
4. **Stop** when two batches in a row agree within 10 points and you have read at least 75 rows,
   or you have read every row, or after 8 batches. Otherwise go back to 1.

On a big set (over ~2,000 rows) add one scaled check after the loop: `dataset_eval` a strong free
model (`claude-code/sonnet`) on `split=train` with `limit` 200 and the dataset's grader. Rows where
it disagrees with the reference are suspects: open 20 of them (`evals eval_id=...`) and decide per
row whether the reference or the model is wrong. The share of wrong references is your second
estimate; say both.

## The verdict

Report in a few lines: rows read, quality (share `ok`), the top three problems with example ids,
and one plan:

- **≥ 90% ok:** train on it as it is (`sft` first).
- **70-90% ok:** remove the `bad` ids you found (`dataset_edit remove_ids`), then relabel the rest
  with the best baseline as teacher (`dataset_label`, proposed with its cost) or keep it if the
  problems are cosmetic.
- **< 70% ok:** do not train on the references. Keep the prompts: distil new answers from a strong
  teacher (`dataset_label`), or, when answers can be checked, use the prompts for `rejection` or
  `rl` (see train/env).
- Duplicates or near-copies across training and held-out rows make the eval lie: remove them first.
- Private data (names, emails, keys) in rows that would be uploaded: tell the user before anything
  leaves the machine.

Then go on with the general path (grader, baseline). Never "fix" held-out rows by reading them;
only training rows are yours to read.
