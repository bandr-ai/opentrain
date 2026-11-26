---
name: data/find
description: The user names a model and a use case but has no data. Search Hugging Face and Kaggle in a loop, shortlist, peek, import a slice, triage it, and settle on the training and eval data.
---

# data/find

Start: "make Qwen3.5-4B better at X" with no data. The job is to find data that teaches X and data
that measures X, honestly.

## 1. Pin the use case

One sentence each: what the model receives, what a good answer looks like, how you would check it
(one right answer, a number, a label, JSON, or open text a judge grades). If you cannot write these,
ask one `question` first. Set it with `goal_set`.

## 2. The search loop

Repeat at most 3 rounds:

1. `dataset_search` with at least four phrasings: the task ("math word problems"), the domain
   ("grade school math"), known benchmark names ("gsm8k"), and data-type words ("instruct",
   "sft", "reasoning traces", "mcq", "qa pairs"). Both sources.
2. **Shortlist up to 5** by: fit (same kind of input and output as the use case), license (flag
   `nc` / non-commercial and "other" to the user), size (300+ rows usable, 2,000+ good), answers
   present or checkable, popularity (downloads, likes, Kaggle usability), not gated.
3. `dataset_peek` each Hugging Face candidate: which config and split, which columns are the prompt
   and the answer, whether answers need converting (choice index to text, a final-answer line).
4. If nothing fits, rephrase (narrower domain, the benchmark's name, another language) and loop.

## 3. Try the best one

1. Propose the import (`needs: download`), then `dataset_import` a slice (`limit` 300, the right
   `config`, `split`, `input_key`, `output_key`, and a `system` prompt that asks for the answer
   format the grader checks).
2. Run data/triage with one or two batches. Set the grader (`dataset_edit`; the import suggests
   one).
3. Bad fit or poor quality: drop it (`dataset_edit` or a new name) and try the next candidate.
   Columns that need reshaping (choices lists, nested fields): write a small converter script to
   JSONL and `dataset_import path=`.
4. Good: import more (a few thousand rows is plenty for SFT; RL needs a few hundred prompts).
   Several sources can feed one dataset name: `dataset_import` with the same name extends it.

## 4. Measure on something you did not train on

A well-known benchmark is for measuring. If you train on one source, prefer a different, public
benchmark of the same skill as a second dataset to evaluate on, and never import the benchmark's
test split into training. Say which is which.

## 5. Report

The data you chose and why (fit, size, license), what you dropped and why, the grader, and the next
step: baselines (`dataset_eval` the base model and a frontier model), then `dataset_recipe`.
