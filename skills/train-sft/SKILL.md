---
name: train/sft
description: Fine-tune a small LoRA adapter on the chosen attempts with the Trainfile's backend. The training half of goal/speed, goal/fix and goal/cost.
---

# train/sft

## Steps

1. Check the backend: `opentrain backends` (tinker needs `TINKER_API_KEY` and the `tinker` extra;
   local needs torch and a GPU or MPS).
2. The estimate is reserved against the budget before launch (`train_launch` refuses if it does not
   fit). The approval step shows it with the number of tasks that would be uploaded.
3. Watch with `opentrain job ps` / `job logs <id>`; poll sparingly. Cancel with `job cancel <id>`.
4. When it succeeds the checkpoint becomes a new version (not live). Next: eval/heldout.

## Settings

Defaults: LoRA rank 32, 2 epochs, learning rate 1e-4. Change one at a time and write down why in a
`note`; the run record keeps every setting with its scores.

## Failure modes

- Loss falls but held-out time does not: too few examples, or tool time dominates.
- Pass rate drops: too many epochs on too few tasks. Fewer epochs, or more data.
