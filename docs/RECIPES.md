# Recipes: what they are and what they measured

A recipe is the settings for one training run, picked from what OpenTrain already knows about the
dataset (`src/opentrain/recipes.py`):

| Task kind | Recognised by | LoRA rank | Learning rate | Plan |
| --- | --- | --- | --- | --- |
| label | exact / contains grader, a small set of short answers | 16 | 2e-4 | sft, rejection |
| extract | json / number / regex grader, or long exact answers | 32 | 2e-4 | sft, rejection |
| reason | number / regex grader on long worked answers (GSM8K-like) | 32 | 2e-4 | sft, opd, rl, rejection |
| answer | judge or command grader | 32 | 1e-4 | opd then sft when a teacher exists, else sft (distil from the best baseline first) |

The plan lists the methods worth trying, in order. `rl` and `rejection` appear only when the grader
runs inside the training loop (exact, contains, regex, json, number, or a `python` reward); `opd`
only when a bigger model of the same family is known as a teacher (Qwen3.5 small → 9B, Qwen3 small →
8B, Qwen3-8B → 14B, Qwen2.5 small → 7B, Llama 3.2 → 3.1-8B, Gemma 3 270m/1b → 4b).

| Method | Settings |
| --- | --- |
| sft | the kind's rank and learning rate; epochs fall with the data: 4 under 100 rows, 3 under 500, 2 under 5,000, then 1 |
| rejection | 2 rounds, 8 samples per prompt, keep up to 2 distinct passing answers, half the SFT learning rate, batch 16 |
| opd | per-token reverse KL to the teacher on the student's own samples; lr 1e-4, 16 prompts × 2 samples per step |
| rl | GRPO with a group-mean baseline (no std), token-mean loss, one on-policy step per batch; lr 4e-5, 16 prompts × 8 samples, steps = rows / 4 (20 to 100) |

Models over 10B halve the learning rate. Before training, rows without an answer are labelled by a
strong teacher (distillation). If a run does not beat the baseline: the next method in the plan,
then more and more varied rows, then a larger base model, then one learning-rate sweep (half and
double), one change per run. `dataset_recipe` also prices the run on your trainer.

These are common starting points, not tuned optima. Every run is judged on its dataset's sealed
held-out rows, so a recipe that does not fit a task shows up as a score, not a surprise.

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

### USMLE-style medical questions, from no data (2026-11-27)

An end-to-end run through the agent (`opentrain run --yes`, a $3 Trainfile budget, trainer Modal,
base model Qwen3.5-4B). Asked only "make this model better at USMLE-style multiple-choice questions",
the agent searched Hugging Face and Kaggle, chose GBaker/MedQA-USMLE-4-options (CC-BY-4.0), imported
400 rows with the options in the prompt and the letter as the answer (270 to train, 130 sealed), and
evaluated on 100 held-out rows.

| Model | Held-out (100 rows) | Training cost |
| --- | --- | --- |
| Qwen3.5-4B, untrained (Modal) | **76%** | — |
| v1: SFT, rank 32, lr 1e-4, 3 epochs | 70% | $0.19 (4.7 min, L40S) |
| v2: SFT, rank 32, lr 5e-5, 1 epoch | 75% (one row lost to a rollout 404, since fixed) | $0.07 (1.9 min) |

Nothing was promoted. Training on letter-only answers teaches the format, which a 4B instruct
model already has, and not the medicine; three epochs overfit and lost six points. The agent said
so, stopped, and priced the next options (on-policy distillation from Qwen3.5-9B, about $2.40, or
training on worked explanations instead of letters). Total spend: $0.32, evals included.

The run also caught its own grader: a first regex accepted any option letter and scored the base
model 100%; the agent noticed the score was implausible, wrote a `reward.py` that compares letters,
and re-ran the baseline before training. (A regex with a capture group now compares what it
captures with the reference.)
