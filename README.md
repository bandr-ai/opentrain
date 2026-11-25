# OpenTrain

**The open harness for training models.** What opencode is for coding, OpenTrain is for training.

You talk to an agent; the thing it works on is a model, not a codebase. It turns your own work (your
coding-agent sessions and repos) into verified tasks, trains a model on them, and ships a new version
only when it beats the current one on work it has never seen. It asks before it spends money, uploads
data or changes what is live.

```
$ opentrain
```

```
 OpenTrain  myrepo │ train  claude-code/default │ idle │ budget $0 used · $40 left

 ▌ make me a model that does my routine coding work much faster

 train · claude-code/default · 6.2s
 I'll check where things stand, then mine your sessions into verified tasks.
 ▶ ✓ status
 ▶ ✓ tasks_mine  mining tasks
   ┌ mine  mine sessions into tasks ───────────────────────────┐
   │ ▰▰▰▰▰▰▰▰▰▰▰▰▱▱▱▱▱▱▱▱  running  1m12s                       │
   └───────────────────────────────────────────────────────────┘
 ┌──────────────────────────────────────────────────────────────┐
 │ train wants to spend $14.20 on tinker   (budget left $40.00) │
 │ [1 Allow once]  [2 Always]  [3 Reject]                        │
 └──────────────────────────────────────────────────────────────┘
```

## Install

```bash
uv tool install "opentrain-cli[tinker] @ git+https://github.com/bandr-ai/opentrain"
cd your-repo && opentrain
```

Python 3.11+. The engine runs on whatever you already have: your own **Claude Code** login (no key
needed), or an Anthropic, OpenAI, OpenRouter, Together, Groq or Fireworks key, or a local model.
`/connect` adds keys; `/models` picks one.

## How it is built

The same shape as opencode: one server per user, and every front end is a client of it.

```
  opentrain (terminal app)   opentrain run "..."   opentrain web   SDK / scripts
            └──────────────────────┬──────────────────────┘
                     HTTP /api  +  SSE /api/event
            ┌──────────────────────┴──────────────────────┐
            │  server: projects, sessions, event log      │
            │  engine: the agent loop (train, plan, ...)  │
            │  tools ── permissions ── budget ── jobs     │
            │  evaluator (own process) ── gate ── registry│
            └──────┬──────────────┬──────────────┬────────┘
              model providers   trainers      data sources, serving
```

- **The server** owns everything and keeps running when you close the terminal. It registers
  itself in `~/.local/state/opentrain/service.json`; clients find it or start it.
- **State is event-sourced.** Every change is an event in `.opentrain/opentrain.db`, hash-chained
  so any edit to history is detectable. Sessions, messages, parts, jobs and approvals are
  projections of it.
- **The engine** is its own agent loop: steer a turn while it runs, interrupt it, compaction near
  the context limit, subagents in child sessions, a doom-loop guard, a step limit. With Claude Code
  as the provider, `claude -p` runs with no built-in tools and gets OpenTrain's tools over MCP, so
  every call is checked and recorded by the server either way.
- **Jobs belong to the server.** Mining, sampling, scoring and training return at once; clients see
  progress live, and the agent gets a reminder when a job ends.

## Agents

| Agent | Can |
| --- | --- |
| `train` (default) | everything within the Trainfile, asking before shell commands, edits, spending, uploading, launching, promoting and serving |
| `plan` (tab) | read, scan sessions, estimate cost; never spends, trains or changes files (except plans) |
| `data` (subagent) | mine and repair tasks, draft verifiers |
| `monitor` (subagent) | look at jobs, read-only |
| `evaluator` (hidden) | the only reader of held-out tasks; runs in its own process |

Agents are Markdown files with front matter (description, mode, model, steps, permissions). Add
or override them in `~/.config/opentrain/agents/` or `.opentrain/agents/`.

## Why a separate harness, and what it guarantees

General coding agents asked to post-train a model reach about half of a vendor's tuned result, and
they cheat: they train on the test set, swap in other checkpoints and use keys they find. OpenTrain
keeps the parts that must be trusted out of the training agent's hands:

- **The trainer never sees held-out tasks.** They are sealed from every tool, and every shell the
  harness starts (the agent's, the teacher's, a candidate model's) runs in an OS sandbox that cannot
  read them, the event log or your stored keys (Seatbelt on macOS, bubblewrap on Linux).
- **The evaluator is its own process** and returns pass/fail and time per task id, never content,
  a capped number of times per run.
- **Every verifier is proven before it counts**: the reference passes, doing nothing fails, a
  known-wrong answer fails; re-checked at every evaluation.
- **The gate decides, not the agent**: better on the goal, no held-out task regresses, and a
  provenance check from the checkpoint back (enough held-out tasks, records intact, the Trainfile's
  base model, no held-out task or its text in the training data).
- **The budget is enforced by the server** across engine tokens, sampling and training: reserve
  first, settle to actual.
- **Nothing leaves your machine without an approval** that says what would be uploaded.

## The Trainfile

The policy, enforced by the server. Written with defaults the first time you use OpenTrain in a
repo; edit it any time (or use `/trainer`).

```
goal         speed
budget       $40 a week, model tokens and compute together
promote      beats current on held-out, no task regresses
trainer      tinker
base model   Qwen/Qwen3-8B
heldout      30%
eval cap     8 per run
```

`opentrain.jsonc` (global and per project) sets up the harness itself: `model`, `default_agent`,
`agents`, `permissions`, `providers`, `mcp` servers, `commands`, `instructions`, `skills`.

## Commands

| | |
| --- | --- |
| `opentrain [dir]` | the terminal app |
| `opentrain run "..."` | one instruction, no interface (`--yes` approves asks, `--json` streams events) |
| `opentrain serve` | the server in the foreground (`--host`, `--password` for a remote box) |
| `opentrain web` | the dashboard in a browser (one-time sign-in link) |
| `opentrain attach URL` | the terminal app against a server elsewhere |
| `opentrain export` | tasks and training set in open formats, for any trainer |
| `opentrain model serve` | a trained version as an OpenAI-compatible endpoint |
| `opentrain debug ...` | every pipeline step on its own (bench, mine, sample, eval, ...) |

In the terminal app: **enter** send · **ctrl+j** new line · **tab** train ⇄ plan · **esc** interrupt ·
**ctrl+p** commands · **ctrl+o** models · **ctrl+l** sessions · **ctrl+b** sidebar. Slash commands:
`/train` `/status` `/mine` `/fix <what is wrong>` `/eval <version>` `/plan` `/export`, and
`/models` `/connect` `/trainer` `/agents` `/sessions` `/jobs` `/versions` `/help`. Add your own as
Markdown in `.opentrain/commands/`.

## Recipes are skills

| Goal | Data recipe | Training recipe | Proved by |
| --- | --- | --- | --- |
| `goal/speed` | `data/shortest` | `train/sft` | time per task, at the same pass rate |
| `goal/fix` | `data/expand`, from flagged failures | `train/sft` | the flagged tasks pass, nothing regresses |
| `goal/cost` | fewest tokens among verified passes | `train/sft` | tokens per task, at the same pass rate |

## What is real today

| Part | State |
| --- | --- |
| Server, event log, sessions, SSE, pairing | Tested |
| Engine loop with Claude Code (your login) as the provider | Run live |
| Anthropic and OpenAI-compatible providers | Written against the docs; tested with a scripted provider |
| Tools, permissions, budget, jobs, sandbox, provenance gate | Tested |
| Terminal app, `run`, `web`, `attach` | Tested (Textual pilot and live server) |
| Mining tasks from Claude Code sessions | Runs on real logs; yield on real sessions is low (see docs/FINDINGS.md) |
| Tinker training | Written against SDK 0.40; **no real training run yet** (needs `TINKER_API_KEY`) |
| Local training (TRL + PEFT) | Written; untested here |

## Development

```bash
uv sync --extra dev
.venv/bin/pytest -q
```

Plan and roadmap: [docs/PLAN.md](docs/PLAN.md). Results on real sessions: [docs/FINDINGS.md](docs/FINDINGS.md).
