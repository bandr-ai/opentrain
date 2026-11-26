---
name: train/env
description: Before RL or rejection sampling, get a reward that runs in the training loop. Ask the user for their environment, wrap it as reward.py, prove it on controls, or fall back to a checkable grader or to SFT/OPD.
---

# train/env

`rl` and `rejection` learn from a reward computed on the model's own samples, on the GPU, thousands
of times. `opd` needs a teacher model, not an environment. `sft` needs neither.

## 1. Ask before proposing RL

Ask with `question`: "RL needs something that scores an answer. Do you have an environment or
checker for this (tests, a validator, a simulator, an API, a scoring script)?"

- **They have one:** step 2.
- **They do not, but answers have one right form** (a number, a label, JSON, a pattern): set that
  grader (`dataset_edit`); it runs in the loop as is.
- **Neither:** say plainly that RL is not the right tool here; go with `sft`, then `opd` if the
  recipe names a teacher. A model judge does not run inside the loop.

## 2. Wrap their environment

Write `reward.py` in the project (with their approval to write files):

```python
def reward(prompt: str, answer: str, reference: str | None) -> float:
    """0..1. Also allowed: a bool, or (score, reason)."""
    ...
```

Rules: deterministic, fast (well under a second), standard library plus what the GPU image has
(torch, transformers, numpy), no network unless the user accepts it running from the GPU. Partial
credit is fine and helps RL.

Set it: `dataset_edit grader={"type": "python", "path": "reward.py"}`. Its code is sent with each
training job.

## 3. Prove it before spending

1. Controls: the reference answer scores high, an empty answer and an off-task answer score 0, and a
   long answer that stuffs in many candidates (numbers, labels) does not pass. Run them with
   `model_try` outputs or a small `dataset_eval split=train limit=10` of a strong free model.
2. The base model must get some rows right: `dataset_eval` it (`modal/<base model>`); between ~10%
   and ~90% is where RL learns. Near 0%: SFT or OPD first. Near 100%: nothing to learn.

## 4. Then train

`dataset_recipe method=rl` (or `rejection`) gives settings and cost; propose it. Single-turn only:
multi-turn agent environments (tool calls, games over several steps) do not run in the loop yet. Say
so if that is what they have.
