---
name: train/sft
description: Fine-tune a small LoRA adapter on the chosen attempts with the Trainfile's backend. The training half of goal/speed, goal/fix and goal/cost.
---

# train/sft

## Steps

1. Check the trainer: `backends` says which can run here (modal and runpod need their sign-in,
   tinker needs `TINKER_API_KEY`, local needs torch and a GPU or MPS); `trainer_set` switches.
2. `train_estimate` gives the price; it goes in the `recipe_plan` page's go-ahead (or `propose` it when no plan covers it). `job_start` reserves the estimate against the
   budget and refuses if it does not fit.
3. The job runs on the server; you get a reminder when it ends. Do not poll; `job_logs` when the
   user asks how it is going, `job_stop` to cancel.
4. When it succeeds the checkpoint becomes a new version (not live). Next: eval/heldout.

## Settings

The trainer's defaults (LoRA rank 32, 2 epochs, learning rate 1e-4) are a starting point. Change
one at a time and say why in your report; the run record keeps every setting with its scores
(`record_query`).

## Failure modes

- Loss falls but held-out time does not: too few examples, or tool time dominates.
- Pass rate drops: too many epochs on too few tasks. Fewer epochs, or more data.
