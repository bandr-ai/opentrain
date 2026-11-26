---
name: data/find
description: The user names a model and a use case but has no data. Search Hugging Face and Kaggle, shortlist, peek, import, triage, and settle on training and evaluation data.
---

# data/find

Start: "make Qwen3.5-4B better at X" with no data. The job: data that teaches X and data that
measures X, honestly.

## 1. Pin the use case

What the model receives, what a good answer looks like, how you would check it (one right answer,
a number, a label, JSON, or open text a judge grades). If you cannot say these, ask one `question`.
Set the goal with `goal_set`.

## 2. Search until you have real candidates

`dataset_search` with several phrasings: the task ("math word problems"), the domain ("grade school
math"), known benchmark names ("gsm8k"), data-type words ("instruct", "sft", "reasoning traces",
"mcq", "qa pairs"), other languages when the task is not English. Shortlist a few by:

- **fit**: the same kind of input and output as the use case;
- **license**: flag non-commercial (`nc`) and "other" to the user;
- **size**: enough rows for the method (hundreds for SFT on a narrow task, more for broad ones;
  RL needs a few hundred prompts);
- **answers**: present, or checkable;
- **health**: downloads, likes or Kaggle usability; not gated.

`dataset_peek` each Hugging Face candidate: which config and split, which columns hold the prompt
and the answer, whether they need reshaping. Nothing fits: rephrase (narrower, the benchmark's name,
another language) and search again; after a few rounds with nothing, say so and offer to build data
with the user instead (`data/tasks`).

## 3. Import and look

Public datasets are free to import; say the license. `dataset_import` with the right `config`,
`split`, columns and a `system` prompt that asks for the format the grader checks. Reshape with
templates rather than scripts: `input_template="{question}\n\n{options_lettered}"`,
`output_template="{answer_idx}"` (or `{cop_letter}` for an index column). Start with a slice
(a few hundred rows) to look at, then take what you need with `limit`.

Run data/triage on it and set the grader (`dataset_edit`; the import suggests one when it can).
A poor fit: drop it and try the next candidate. Several sources can feed one dataset name.

## 4. Measure on something you did not train on

Each dataset seals part of its rows for the gate. A separate public benchmark of the same skill is
a stronger check: import it under its own name with `heldout=1` (every row sealed), never into the
training dataset, and report its score next to the gate's. The gate itself decides on the training
dataset's own held-out rows.

## 5. Report

The data you chose and why (fit, size, license), what you dropped and why, the grader, and the next
step: baselines, then `dataset_recipe`.
