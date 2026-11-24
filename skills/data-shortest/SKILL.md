---
name: data/shortest
description: Among the verified passes of each task, keep the one with the fewest steps, then build the SFT set from those. The data half of goal/speed.
---

# data/shortest

The speed goal trains the model to take the short path it can already find some of the time.

## Steps

1. Run cl/rollout first (several attempts per task).
2. `opentrain data build` (or the `data_build` tool) picks, per task, the verified pass with the fewest
   assistant turns; ties go to fewer tool calls, then less time.
3. It writes `.opentrain/data/train.jsonl` (OpenAI chat format with tool calls, the direct runtime's
   system prompt and tools) and `manifest.json` (task ids, digests, example hashes). That manifest is
   the run record's data snapshot.

## Rules

- Held-out task ids are refused at build time, whatever the caller asks.
- A transcript that uses a tool the specialist does not have is skipped with its reason in the manifest.

## Measure

Examples written, and median steps per example compared with the median of all passes.
