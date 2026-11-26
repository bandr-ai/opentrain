---
name: train/env
description: RL and rejection sampling need a reward that runs in the training loop. When the dataset has none, ask the user for their environment, wrap it as reward.py, prove it on controls, or fall back to a checkable grader or to SFT/OPD.
---

# train/env

`rl` and `rejection` learn from a reward computed on the model's own samples, on the GPU, thousands
of times. `opd` needs a teacher model, not an environment; `sft` needs neither.

## 1. Is there a reward already?

If the dataset's grader runs in the loop (exact, contains, regex, json, number, python), there is:
check it on controls (step 3) and go on. Otherwise ask with `question`: "RL needs something that
scores an answer. Do you have an environment or checker for this (tests, a validator, a simulator,
an API, a scoring script)?"

- **They have one:** step 2.
- **They do not, but answers have one right form** (a number, a label, JSON, a pattern): set that
  grader (`dataset_edit`).
- **Neither:** RL is not the tool here. Say so plainly and go with `sft`, then `opd` if the recipe
  names a teacher. A model judge does not run inside the loop.

## 2. Wrap their environment

Write `reward.py` in the project (with their approval to write files):

```python
def reward(prompt: str, answer: str, reference: str | None) -> float:
    """0..1. Also allowed: a bool, or (score, reason)."""
    ...
```

Deterministic, fast (well under a second), standard library plus what the GPU image has (torch,
transformers, numpy); no network unless the user accepts calls from the GPU. Partial credit helps
RL. Set it: `dataset_edit grader={"type": "python", "path": "reward.py"}`; its code is sent with
each training job.

## 3. Prove it before spending

1. Controls with `dataset_grade` on a few training rows: the reference scores high; an empty
   answer, an off-task answer and an answer that stuffs in many candidates (numbers, labels) do
   not. A reward that a stuffed answer passes will be gamed by RL.
2. The base model must get some rows right but not all: evaluate it (`modal/<base model>`). Near
   zero, RL has nothing to reinforce: `sft` or `opd` first. Near perfect, there is little to learn.

## 4. Then train

`dataset_recipe method=rl` (or `rejection`) gives the settings and the price; propose it.
Single-turn only: multi-turn agent environments (tool calls, games over several steps) do not run
in the loop yet. Say so if that is what they have.
