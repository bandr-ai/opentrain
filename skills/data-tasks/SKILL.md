---
name: data/tasks
description: The user brings their own data (traces, Q&A, documents, tables, tickets, logs). Decide what tasks it should become (SFT traces, QA, MCQ, extraction, labels, RL tasks) and build them in checked batches.
---

# data/tasks

## 1. What did they bring?

Look first (`dataset_import` a slice, or read the file), then choose:

- **Prompts and answers already** (chat logs, traces, Q&A): import, run data/triage, train on the
  good rows as SFT traces.
- **Prompts without answers**: distil answers from a strong teacher (`dataset_label`, proposed with
  its cost), or, when answers can be checked, keep them as RL prompts (train/env).
- **Raw material** (docs, manuals, policies, tables, tickets, code): not training data yet. Turn it
  into tasks (step 2).

## 2. Choose the task format

From what the model must do after training, not from what is easy to generate:

| Format | Grader | Trains well with | When |
|---|---|---|---|
| Open QA | judge + rubric | sft, opd | explanations, replies, summaries |
| Short answer / number | exact, number | sft, rl, rejection | facts, calculations |
| Extraction | json | sft, rl, rejection | fields out of text |
| Label | exact | sft, rl | routing, classification |
| Multiple choice | regex or exact on the letter | sft, rl, rejection | cheap checks; chance is 1 in the number of options, so baseline it |
| Reasoning traces | number / judge | sft, then rl | multi-step problems: a teacher writes the working |

A checkable format unlocks `rl` and `rejection` and cannot be fooled by a fluent wrong answer;
prefer it when the use case allows. Ask one `question` if two formats fit equally.

## 3. Evaluation rows first

Set aside part of the source (a section, a time range, some documents) and write the evaluation
rows from it first, into their own dataset with `heldout=1` or as the training dataset's first
rows. Do not write training rows from that part. Rows you write after seeing the eval rows are
paraphrases of them, and the overlap check cannot catch a paraphrase.

## 4. The build loop

Keep a tally in `todo` (rows made, kept, dropped by reason, which parts of the source are covered).
Repeat until the set covers the source and the task's variety (a few hundred rows is a good start
for a narrow task; open-ended tasks need fewer, better rows):

1. Take the next chunk of the source.
2. Write a batch of rows from it (`dataset_add`; a script that emits JSONL and `dataset_import
   path=` for big, regular conversions). Each prompt must be answerable from what the model will
   see at run time, so put the needed context in the prompt. One right answer, varied wording and
   difficulty, no copies.
3. Check the batch: read it back (`dataset_show`) and have a strong free model answer exactly
   those rows (`dataset_eval model=claude-code/sonnet split=train ids=[...]`, then
   `evals eval_id=...` for its answers). A failure is a reason to look at the row, not to drop it:
   drop or fix it only when the reference is wrong or the prompt is ambiguous. Hard rows that are
   right are the valuable ones.
4. Update the tally; next chunk.

## 5. Hand over

Report rows by format, what was dropped and why, the grader, and the plan from `dataset_recipe`.
