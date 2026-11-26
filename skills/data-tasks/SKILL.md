---
name: data/tasks
description: The user brings their own data (traces, Q&A, documents, tables, tickets, logs). Decide what tasks it should become (SFT traces, QA, MCQ, extraction, labels, RL tasks) and build them in checked batches.
---

# data/tasks

## 1. What did they bring?

Look first (`dataset_import` a slice or read the file), then pick the path:

- **Prompt and answer already** (chat logs, traces, Q&A): import, run data/triage, then train on
  the good rows as SFT traces.
- **Prompts without answers**: distil answers from a strong teacher (`dataset_label`, proposed with
  its cost), or, when answers can be checked, keep them as RL prompts (train/env).
- **Raw material** (docs, manuals, policies, tables, tickets, code): it is not training data yet.
  Turn it into tasks (step 2).

## 2. Choose the task format

From what the model must do after training, not from what is easy to generate:

| Format | Grader | Trains with | When |
|---|---|---|---|
| Open QA | judge + rubric | sft, opd | explanations, replies, summaries |
| Short answer / number | exact, number | sft, rl, rejection | facts, calculations |
| Extraction | json | sft, rl, rejection | fields out of text |
| Label | exact | sft, rl | routing, classification |
| MCQ (A-D) | exact on the letter | rl, rejection | cheap checks; 25% is chance, so baseline it |
| Reasoning traces | number / judge | sft, then rl | multi-step problems: teacher writes the working |

Prefer a checkable format when the use case allows it: it unlocks `rl` and `rejection`, and the
grader cannot be fooled. Ask the user one `question` if two formats fit equally.

## 3. The build loop

Keep a tally in `todo` (rows made, kept, dropped by reason, which parts of the source are covered).
Repeat until the target count (200-500 rows to start), covering the whole source, not its first
pages:

1. Take the next chunk of the source.
2. Write 20 rows from it (`dataset_add`): the prompt must be answerable from what the model will
   see at run time (put needed context in the prompt), one right answer, varied wording and
   difficulty, no copies of earlier rows. For big conversions write a script that emits JSONL and
   `dataset_import path=`.
3. Check the batch: read it back (`dataset_show`); a strong free model answers it blind
   (`dataset_eval model=claude-code/sonnet split=train limit=20`). Rows it fails: wrong reference
   or ambiguous prompt, fix or drop; rows that every model gets without context: too easy, keep few.
4. Update the tally; next chunk.

Held-out rows are split off automatically on import; never write eval rows by copying training ones.

## 4. Hand over

Report rows by format, what was dropped and why, the grader, and the plan from `dataset_recipe`
(sft first; rl or rejection when the grader runs in the loop; opd when a teacher exists).
