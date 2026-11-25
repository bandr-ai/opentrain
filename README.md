# OpenTrain

**The open source post-training harness.** What opencode is for coding, OpenTrain is for training.

Your agent sessions already hold the data. OpenTrain turns them into verified tasks, trains a small
model on them, and ships a new version only if it beats the current one on work it has never seen.
The first feature trains a personal coding model from your Claude Code sessions.

```
$ opentrain train

Engine      claude, your own login
Found       412 Claude Code sessions in 9 folders
Trainer is building tasks...
  137 verified tasks, 61 held out for the evaluator
  275 skipped (140 no repo state, 98 no tests in repo, 37 controls failed)
Evaluator   frontier:claude: 41% pass, 3m40 median
Sampling    frontier (claude, headless): 4 attempts on each of 76 training tasks
  58 of 76 tasks have a verified pass (18 never passed)

Proposed    goal/speed: tune Qwen/Qwen3-8B on shortest verified passes
            est. $14.20 on tinker, uploads 58 redacted tasks; $40.00 of $40 a week left
Proceed? [y/n] y
Training    job tinker-20261124-a1b2c3 on tinker
Evaluator   scoring v1 on 61 held-out tasks...

v1 passed the gate
  Median task   3m40 -> 22s
  Pass rate     41% -> 58%    11 fixed, 0 regressed (opentrain diff)
  Live as opentrain/v1 and opentrain/current (opentrain serve; opentrain connect pi)
```

*(Output shape of a real run; the numbers above are illustrative, not a measurement.)*

## What is real today

| Part | State |
| --- | --- |
| Session reading, task yield, time breakdown, held-out split (v0.1 bench) | Runs on real Claude Code logs |
| Mining sessions into Harbor tasks with certified verifiers | Runs on real repos; first real task certified |
| Direct runtime (any OpenAI-compatible model) and headless runtime (`claude -p`, `codex exec`) | Claude headless checked against real runs; Codex flags from docs, untested |
| Sampling, shortest-pass SFT data, evaluator, gate, registry, run record, budget | Tested end to end on real git repos with stand-in models |
| Tinker backend | Written against the documented SDK (0.40) and cookbook; **no real training run yet** (needs `TINKER_API_KEY`) |
| Local backend (TRL + PEFT LoRA) | Written; untested here (needs torch and a GPU or MPS) |
| `opentrain serve` (OpenAI-compatible proxy, hot switch, frontier handoff) | Tested against local fake upstreams |
| MCP train tools, agent mode (`opentrain run`) | Protocol tested; a full agent-driven training run not yet done |
| Web UI | All 8 screens checked in a browser against synthetic state |

## Install

```bash
uv tool install "opentrain-cli[tinker] @ git+https://github.com/bandr-ai/opentrain"
# or, to train on your own GPU:  opentrain-cli[local]
```

Python 3.11+. The core has no dependencies; each trainer backend brings its own.

## Quickstart

```bash
cd your-repo
opentrain init        # writes a Trainfile: goal, budget, gate
opentrain bench       # how many of your sessions can become tasks, and where your time goes
opentrain mine        # sessions -> verified tasks, split held-out by date
opentrain train       # sample, train, evaluate, gate (asks before spending or uploading)
opentrain serve       # opentrain/current as an OpenAI-compatible endpoint
opentrain connect pi  # config for pi, opencode or Claude Code
opentrain web         # the dashboard
```

Then let it run by itself: `opentrain watch`. When the model gets something wrong:
`opentrain fix --session latest -m "it keeps using our old auth API"`.

## The Trainfile

The one file you edit. The server enforces every line; the engine cannot change it mid-run.

```
goal         speed
budget       $40 a week, model tokens and compute together
promote      beats current on held-out, no task regresses
runtime      claude -p
trainer      tinker
base model   Qwen/Qwen3-8B
data         my agent sessions, redacted, local until I approve
heldout      30%
eval cap     8 per run
watch        when 30 new sessions arrive, or nightly
provider     https://openrouter.ai/api/v1 key OPENROUTER_API_KEY   # optional: sample the base model itself
```

## How it works

```
 sessions ──> cl/mine ──> tasks ──split by date──> training tasks ──> cl/rollout ──> data/shortest ──> train/sft
                 │                       │                                                              │
          bandits/certify          held-out tasks ──────────── evaluator (pass/fail only) <──── new version
     (reference passes,                                                    │
      doing nothing fails)                                       gate: better, nothing regresses
                                                                           │
                                                              promote -> opentrain serve
```

**Why a separate harness.** General coding agents asked to post-train a model reach about half of a
vendor's tuned result, and they cheat: they train on the test set and use keys they find. OpenTrain
keeps the parts that must be trusted out of the training agent's hands:

- **Trainer and evaluator are separate roles.** The trainer never reads held-out tasks. The evaluator
  returns pass/fail and time per task id, never content, and only a capped number of times per run.
- **Every verifier is proven before it counts.** The reference answer must pass, doing nothing must
  fail, a known-wrong answer must fail. An environment failure is not a pass.
- **The gate decides, not the trainer.** Better on the goal, no held-out task regresses, verifiers
  re-checked at every evaluation.
- **The budget is enforced by the server.** Every model call and training job reserves its estimate
  first; a run cannot raise the budget.
- **The run record is append-only and hash-chained.** Any score traces back to its data, recipe and
  checkpoint (`opentrain log <run>`).
- **Nothing leaves your machine until you approve it.** The approval step shows what would be uploaded.

## Commands

| Command | Does |
| --- | --- |
| `opentrain` | Opens with the status; on first use, shows where to start |
| `opentrain run "..."` | One instruction to the engine (your own `claude -p` with OpenTrain's train tools) |
| `opentrain init`, `status` | Write a Trainfile; what is live, what it cost |
| `opentrain session import` | Index your coding-agent sessions |
| `opentrain bench [yield\|time\|split\|models]` | v0.1: task yield, time breakdown, held-out split, model baselines |
| `opentrain mine`, `lint` | Sessions into tasks; re-check every verifier |
| `opentrain sample`, `data build` | Attempts on training tasks; the SFT set from the shortest passes |
| `opentrain train`, `eval` | Run the goal end to end; score a model on held-out tasks |
| `opentrain watch`, `fix`, `approve` | The unattended loop; file a bug against the model; allow unattended uploads |
| `opentrain race` | One of your tasks, frontier vs specialist, side by side |
| `opentrain diff`, `log`, `revert` | What flipped between versions; the run record; put the previous version back |
| `opentrain serve`, `connect`, `model ls\|use` | Serve `opentrain/current`; configure pi, opencode, Claude Code; switch versions |
| `opentrain backends`, `job ps\|logs\|cancel` | Trainer backends and jobs |
| `opentrain export` | Tasks (Harbor), evals and training set in open formats |
| `opentrain task ls\|show`, `skill ls\|add` | The task store; recipes |
| `opentrain web` | Runs, evals, versions, race, data, skills, policy |

`ot` is a short alias. Words that mean the same thing in opencode are spelled the same.

## Every recipe is a skill

A recipe is a folder with a written playbook (`SKILL.md`) the engine reads and you can change. A goal
pairs a data recipe with a training recipe. See `opentrain skill ls` and [`skills/`](skills/).

| Goal | Data recipe | Training recipe | Proved by |
| --- | --- | --- | --- |
| `goal/speed` | `data/shortest` | `train/sft` | Time per task, at the same pass rate |
| `goal/fix` | `data/expand`, from flagged failures | `train/sft` | The flagged tasks pass, nothing else regresses |
| `goal/cost` | fewest tokens among verified passes | `train/sft` | Tokens per task, at the same pass rate |

## What plugs in

| Plug | Contract | Today |
| --- | --- | --- |
| Model provider (the engine's reasoning) | Chat with tool calls, usage reported | Headless Claude Code and Codex under your own login; any OpenAI-compatible API |
| Trainer backend | Launch, stream logs, cancel, report cost as it accrues | Tinker, local (TRL/PEFT); Fireworks pending |
| Base model | Chat and tool-call format, adapter type | Any model your backend supports |
| Data source | Sessions in a known format | Claude Code first |
| Serving | Deploy, switch versions without downtime | Tinker's OpenAI- and Anthropic-compatible endpoints; vLLM locally |

## Your data

Everything lives in `.opentrain/` beside the Trainfile: `sessions/`, `tasks/`, `heldout/` (evaluator
only), `attempts/`, `data/`, `runs/`, `models/`, `jobs/`, `serving/`. Tasks are Harbor folders,
training data is OpenAI chat JSONL, weights are standard LoRA adapters. `opentrain export` writes
all of it in open formats for any other trainer.

## Development

```bash
uv sync --extra dev
.venv/bin/pytest -q      # 142 tests, no network
.venv/bin/ruff check .
```

The plan this implements, with the roadmap from v0.1 bench to v0.7 merge: see
[docs/PLAN.md](docs/PLAN.md). Results from running it on real sessions: [docs/FINDINGS.md](docs/FINDINGS.md).
