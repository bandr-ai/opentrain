# Recipes: what they are and what they measured

A recipe is the settings for one training run, picked from what OpenTrain already knows about the
dataset (`src/opentrain/recipes.py`):

| Task kind | Recognised by | LoRA rank | Learning rate | Notes |
| --- | --- | --- | --- | --- |
| label | exact / contains grader, a small set of short answers | 16 | 2e-4 | classification, routing, yes/no |
| extract | json / number / regex grader, or long exact answers | 32 | 2e-4 | structured output |
| answer | judge or command grader | 32 | 1e-4 | free-form; distil from the best baseline first |

Epochs fall with the data: 4 under 100 rows, 3 under 500, 2 under 5,000, then 1. Models over 10B
halve the learning rate. Before training, rows without an answer are labelled by a strong teacher
(distillation). If a run does not beat the baseline: more and more varied rows first, then a larger
base model, then one learning-rate sweep (half and double), one change per run.

These are common LoRA starting points, not tuned optima. Every run is judged on its dataset's
sealed held-out rows, so a recipe that does not fit a task shows up as a score, not a surprise.

## Measured

Each row is a real run, end to end, through OpenTrain (data, recipe, training, serving, eval, gate).

### Support tickets → four labels (2026-11-26)

78 synthetic support tickets labelled billing, bug, account or feature: 54 to train on, 24 sealed.
Recipe `label`: rank 16, lr 2e-4, 4 epochs, batch 8. Trained on Modal, driven from Claude Code over
MCP.

| Model | Held-out (24 rows) | Cost per 1k answers | Median latency |
| --- | --- | --- | --- |
| claude-code/sonnet | 100% | $0 (subscription) | 2.4 s |
| claude-code/haiku | 100% | $0.21 | 2.3 s |
| Qwen2.5-0.5B-Instruct, untrained (Modal, L4) | 42% | $0.08 | 0.9 s |
| **v1: Qwen2.5-0.5B-Instruct + LoRA** (Modal, L4) | **96–100%** (two evals) | $0.07 | 0.9 s |

Training: 28 steps in 30 s on an L40S, loss 0.41 → 0.001, **$0.02** of GPU time. The gate promoted
v1 over the untrained base: 14 rows fixed, none regressed. The two evals of v1 differ by one row
(23/24, then 24/24): with 24 held-out rows each row is about 4 points, so this proves the direction,
not the decimal.

Caveats, honestly: the dataset is small and synthetic, and easy for frontier models, so this shows
the loop works and a 0.5B model can learn a narrow task cheaply, not that it beats a frontier model
on real traffic. The serving cost is the GPU time each answer used at 8 concurrent requests; at real
traffic, batched, it is much lower, and cold starts (about two minutes after idle) are not counted.
