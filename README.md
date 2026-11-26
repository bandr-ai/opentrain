# OpenTrain

**The open harness for training models.** What opencode is for coding, OpenTrain is for training.

You tell an agent what a model should get better at, in your own words. It gets the data, decides
how answers are graded, measures a baseline, trains, and ships a new version only when it beats the
current one on examples it has never seen. There are no modes to pick: one agent works through the
loop, and anything that costs money, sends data out, trains or changes what is live comes to you as
one proposed step, with its price, to approve, change or skip.

```
$ opentrain web        # the web app (or: opentrain, for the terminal app)
```

- **Any data.** A file you have (JSONL, JSON, CSV), a public Hugging Face dataset, examples the
  agent writes with you from a description, a teacher model's answers (distillation), or your
  coding-agent sessions when you ask for that.
- **Any way of checking an answer.** Exact, contains, a pattern, JSON, a number, a model judge with
  your rubric, or your own script.
- **Any model to compare.** Evaluate Claude, GPT, an open model or your own trained versions on the
  same held-out rows; the leaderboard tells you whether training is even worth it.
- **Proof before shipping.** Part of every dataset is sealed from training. A version goes live only
  when it beats the baseline on those rows with nothing regressing.

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
            │  engine: the agent loop and its tools       │
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

## The web app

`opentrain web` opens it in your browser (a one-time sign-in link; it talks to your local server).
It is one screen per project, built around the loop:

```
 goal → data → grade → baseline → train → prove → ship          budget  $0.42 / $20 a week
┌───────────────────────────────────────────────────────┬──────────────────────────────┐
│ the stage you are on: its evidence and its actions    │ the agent                    │
│  · baseline: the scoreboard, every model on the same  │  what it did, in a few lines │
│    held-out rows (score, cost per 1k, latency)        │ ┌──────────────────────────┐ │
│  · train: runs with their loss curve and cost         │ │ next step · prove        │ │
│  · prove: the new version against the baseline,       │ │ Evaluate v2 on the 24    │ │
│    rows fixed and rows broken                         │ │ held-out rows  ~$0.01    │ │
│  · ship: what is live, the gate, a playground         │ │ [Approve] [Change] [Skip]│ │
│                                                       │ └──────────────────────────┘ │
└───────────────────────────────────────────────────────┴──────────────────────────────┘
```

- **The strip is the navigation.** Each stage shows its state (done, running, next) in one line;
  keys 1–7 jump to them. A red dot marks the stage a proposed step belongs to.
- **The middle is the evidence.** Every number opens the rows behind it: `every row →` is the grid
  of held-out rows against every model, with each answer and the grader's reason. Held-out rows
  are sealed from the agent and from training; you can read them, the agent never can.
- **The agent works beside it.** It proposes one step at a time with its cost and time; approving
  covers exactly the gated actions that step needs, so you answer once, not once per permission.
- **Drop a file anywhere** to make a dataset.

The app is built from `web/` (Vite + React) into the package, so using it needs no Node:
`cd web && pnpm install && pnpm build` after changing it, `pnpm dev` to work on it against a
running server (`OPENTRAIN_URL` picks which).

## Agents

| Agent | Can |
| --- | --- |
| `train` (the one you talk to) | starts from what you ask; data, grading, evals, training and the gate. Proposes each costly or risky step (`propose`) and asks before shell commands and edits |
| `data` (subagent) | import, write and clean dataset rows, propose graders; mine and repair coding tasks |
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
goal         draft our support replies as well as the current model, much cheaper
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

In the terminal app: **enter** send · **ctrl+j** new line · **esc** interrupt ·
**ctrl+p** commands · **ctrl+o** models · **ctrl+l** sessions · **ctrl+b** sidebar. Slash commands:
`/train` `/status` `/data` `/mine` `/fix <what is wrong>` `/eval <version>` `/export`, and
`/models` `/connect` `/trainer` `/agents` `/sessions` `/jobs` `/versions` `/help`. Add your own as
Markdown in `.opentrain/commands/`.

`goal` is plain words. `speed`, `fix` and `cost` are built-in recipes for coding-agent work.

## Recipes are skills (coding-agent work)

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
| Datasets (file, Hugging Face, agent-written), graders, evals, the dataset gate | Tested; evals run live with Claude Code |
| Web app | Driven end to end in a browser: upload, grading, a live eval, chat with approvals |
| Terminal app, `run`, `attach` | Tested (Textual pilot and live server) |
| Mining tasks from Claude Code sessions | Runs on real logs; yield on real sessions is low (see docs/FINDINGS.md) |
| Tinker training | Written against SDK 0.40; **no real training run yet** (needs `TINKER_API_KEY`) |
| Local training (TRL + PEFT) | Written; untested here |

## Development

```bash
uv sync --extra dev
.venv/bin/pytest -q
```

Plan and roadmap: [docs/PLAN.md](docs/PLAN.md). Results on real sessions: [docs/FINDINGS.md](docs/FINDINGS.md).
