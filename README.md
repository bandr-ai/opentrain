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

- **A small model that does your coding.** OpenTrain finds every coding agent's logs on your machine
  (Claude Code, Codex, Cursor, opencode, Gemini CLI, Cline, Aider, pi, Copilot), imports them locally,
  shows you where they went wrong so you pick what to fix, builds tasks your repos' tests decide, and
  trains a small model (Inkling 2 Mini, Qwen3.5, Qwen3.8-27B, ...) that has to prove itself on tasks
  it never saw.
- **Or any other use case.** Bring your data (traces, Q&A, documents, a CSV) and the agent checks how
  good it is and turns it into the right tasks; or name a model and a use case and it finds the best
  public data on Hugging Face and Kaggle. Either way you get a sealed eval set and a plan.
- **Any way of checking an answer.** Exact, contains, a pattern, JSON, a number, a model judge with
  your rubric, your own script, or your environment as a Python `reward()`.
- **Four ways to train.** SFT, rejection sampling, on-policy distillation from a bigger model of
  the same family, and RL (GRPO). The recipe says which suit the task, with settings and a price.
- **Any model to compare.** Evaluate Claude, GPT, an open model or your own trained versions on the
  same held-out rows; the scorecard tells you whether training is even worth it.
- **Your compute.** Train and serve on your own Modal account or RunPod pods (any GPU, billed by
  the second, stops when idle), on Fireworks' serverless training, on Tinker, or on your own GPU.
  `modal/Qwen/Qwen3.5-4B` is a model you can evaluate before training it.
- **Proof before shipping.** Part of every dataset is sealed from training. A version goes live only
  when it beats the baseline on those rows with nothing regressing.
- **Use it from Claude Code.** `opentrain mcp --install` gives your Claude Code the same tools,
  under the same budget and the same sealed rows; you watch the scorecard in the web app.

## Install

```bash
uv tool install "opentrain-cli[modal] @ git+https://github.com/bandr-ai/opentrain"
modal setup                     # once: signs in to your Modal account
cd your-project && opentrain web
```

Python 3.11+. The engine runs on whatever you already have: your own **Claude Code** login (no key
needed), or an Anthropic, OpenAI, OpenRouter, Together, Groq or Fireworks key, or a local model.
`/connect` adds keys; `/models` picks one.

## Ways in

**"Make me a small model that does my coding."** (`goal/coding`)

1. *What you have.* `traces_scan` finds every coding agent's logs on this machine and counts them;
   `traces_import` reads them here (credentials scrubbed, nothing uploaded): which repos, which
   requests changed code, training examples in the trained model's tools, and the moments where
   something went wrong. The start screen of the web app shows the same, with one import button.
2. *What to fix.* The agent reads those moments in batches (`data/diagnose`): you corrected it, you
   stopped it, a check kept failing, a file was edited over and over. It names the themes in your
   words ("uses the old auth API") and you tick the ones the model should get right.
3. *Tasks your tests decide.* Sessions that ended in a commit with tests (`tasks_mine`), and the
   repos' own commits that changed code and tests (`tasks_from_history`), each rewritten by Claude
   into the request a person would have typed, without the answer. Every task is certified
   (reference passes, doing nothing fails) and the newest are held out.
4. *Data.* Claude's shortest verified run on each training task, your own traces (minus anything
   from the held-out period), and public trajectories when yours are few
   (`coding_data_import`: SWE-smith, SWE-Gym, R2E-Gym). One format: the trained model's five tools.
5. *A student.* `opentrain.models`: Inkling 2 Mini on Tinker; Qwen3.5 4B/9B on your Modal or RunPod;
   Qwen3.8-27B or Qwen3.6-35B-A3B in the middle; Inkling-Small on Tinker when small falls short.
   Each comes with its trainer, serving flags and price.
6. *Proof.* Held-out tasks for what you use today, the untrained student (`eval_request base`) and
   each trained version: pass rate, time and cost per task. The gate ships a version only when it
   wins with nothing regressing.

**"Here is my data."** Drop a file or name one. The agent reads it in batches until it can say how
good it is (`data/triage`: what share of rows are right, the top problems with examples) and
decides: train on it, filter it, relabel it with a strong teacher, or keep only the prompts. Raw
material (documents, tables, tickets) is turned into tasks that fit what the model must do
(`data/tasks`): open QA, short answers, extraction, labels, multiple choice, reasoning traces, or
RL tasks with a checkable answer, built 20 rows at a time and checked by a strong model each batch.

**"Make this model better at X."** No data needed. The agent searches Hugging Face and Kaggle with
several phrasings, shortlists by fit, size and license, looks at the columns and example rows,
imports a slice of the best fit, triages it, and keeps a separate public benchmark to measure on
(`data/find`).

These are skills: playbooks in `skills/` that the agent loads and follows as a loop with a tally
and a stop rule. Add your own in `.opentrain/skills/`.

## Training methods

| Method | Learns from | Needs | Runs on |
| --- | --- | --- | --- |
| `sft` | the reference answers | answers | Modal, RunPod, Fireworks, Tinker, local |
| `rejection` | the model's own answers that pass the grader | a grader that runs in the loop | Modal, RunPod, Fireworks |
| `opd` | a bigger same-family teacher scoring each of the model's tokens | a teacher (picked for you) | Modal, RunPod |
| `rl` | GRPO on the grader's score of the model's own samples | a grader that runs in the loop | Modal, RunPod, Fireworks |

Graders that run in the loop: exact, contains, regex, json, number, and `python`, your own
environment:

```python
# reward.py, in the project; grader {"type": "python", "path": "reward.py"}
def reward(prompt: str, answer: str, reference: str | None) -> float:   # 0..1
    ...
```

Its code goes to the GPU with every job. Before proposing RL the agent asks whether you have an
environment, wraps it, and proves it on controls (the reference scores high, an empty or stuffed
answer does not). Multi-turn environments are not supported in the loop yet.

`dataset_recipe` returns the plan, the settings for each method and its cost, GPU and minutes on
your trainer, so every proposal carries a real price.

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
One screen: the conversation, with a scorecard pinned above it.

```
 opentrain  ~/support-bot · tickets                              $0.03 spent  jobs  settings
 ───────────────────────────────────────────────────────────────────────────────────────────
                                                 quality   $/1k   speed
   cc/sonnet                     best              100%    free    2.4s
   v1                            live              100%   $0.07    0.9s
   Qwen/Qwen2.5-0.5B-Instruct    base, untrained    42%   $0.08    0.9s
   ▬▬▬▬▬▬▬  ✓ v1 live                                               details
 ───────────────────────────────────────────────────────────────────────────────────────────
   the conversation: what the agent did, in a few lines; jobs with their loss curve
   ┌ next step ─────────────────────────────────────────────┐
   │ Train Qwen2.5-0.5B on 54 rows   ~$0.17 · 2 min          │
   │ [Approve] [Change] [Skip]                               │
   └─────────────────────────────────────────────────────────┘
 > reply…
```

- **The scorecard is the point.** The best model, the cheapest one at about the same quality, what
  is live, what was trained and the untrained base: quality on the sealed rows, cost per 1,000
  answers, speed. Click a row for every held-out row against every model, with each answer and
  the grader's reason (you can read held-out rows; the agent never can).
- **One step at a time.** Anything that costs money, uploads data, trains or changes what is live
  comes as a card with its cost and time. Approving covers exactly what that step needs.
- **Details** opens the step behind the progress bar (data, grading, training runs, the gate).
- **Drop a file anywhere** to make a dataset.

The app is built from `web/` (Vite + React) into the package, so using it needs no Node:
`cd web && pnpm install && pnpm build` after changing it, `pnpm dev` to work on it against a
running server (`OPENTRAIN_URL` picks which).

## From Claude Code

```bash
opentrain mcp --install      # claude mcp add opentrain -- opentrain mcp (project scope)
```

Then ask Claude Code, for example, "use opentrain to find the cheapest model that handles my
tickets dataset, train one if it's worth it". It gets OpenTrain's tools (datasets, graders, evals,
recipes, training, the gate) and none of its own shell over them: every call runs on the OpenTrain
server, which enforces the Trainfile's budget, keeps held-out rows sealed and refuses a promotion
the gate rejects. Claude Code's own permission prompts are the approval. Its calls appear live in
the web app as the "Claude Code (attached)" conversation, and the scorecard fills in as it works.

## Training on RunPod

`trainer runpod`. Sign in once in the browser with `opentrain login runpod` (the same flow as
RunPod's own CLI; a key from `runpodctl` or `RUNPOD_API_KEY` also works). OpenTrain starts a pod on
the vLLM image, trains there, keeps a copy of the adapter, and serves it from the same pod. Community
cloud first, then secure. The pod deletes itself after 15 idle minutes (and after 12 hours in any
case), so nothing bills on after OpenTrain is gone.

## Training on Modal

`trainer modal` in the Trainfile. OpenTrain ships its trainer (TRL + PEFT LoRA) to a Modal GPU,
streams the loss back, and saves the adapter on a Modal volume. Serving is vLLM on Modal, one app
per base model with every adapter trained on it, behind a key OpenTrain generates; it scales to
zero, so the first call after idle waits for a cold start (about two minutes). GPUs are picked by
model size with fallbacks (an L4 or A10 to serve a small model, an L40S or A100 to train it).
Training runs as a detached Modal app: if your laptop loses its connection mid-run, the job goes on
and OpenTrain picks its progress up again from a log kept on Modal. Costs are measured GPU seconds
at Modal's list price. A serving cost per answer is its share of
the GPU while it ran; at real traffic, with many requests batched, it is far lower.

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
trainer      modal
base model   Qwen/Qwen3.5-4B
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
| `opentrain web` | the web app in a browser (one-time sign-in link) |
| `opentrain mcp` | OpenTrain's tools for Claude Code or any MCP client (`--install` adds it to Claude Code) |
| `opentrain login runpod` | sign in to RunPod in the browser |
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

## Recipes

For datasets, `dataset_recipe` picks the settings from the task kind (labels, structured output,
free-form answers, read from the grader and the answers), the number of rows and the base model's
size, says what to do first (distil missing answers with a strong teacher) and what to try if the
run does not beat the baseline. `dataset_train` applies it unless told otherwise. Measured results
are in [docs/RECIPES.md](docs/RECIPES.md).

### Coding-agent work: recipes are skills

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
| Tasks from repo history | Run on this repo: 3 of 4 candidate commits certified; Claude's rewrite of a commit into a request read well |
| Trace readers (Claude Code, Codex, opencode, Gemini CLI, Cursor, Cline, Aider, pi, Copilot) | Claude Code and Codex read on real logs; the others tested on synthetic logs written from their documented formats |
| Public coding trajectories (SWE-smith, SWE-Gym, R2E-Gym) | Converted live from Hugging Face samples: 73 examples from 90 rows |
| A trained coding model, served with tool calls, scored on held-out tasks | Written; **no live run yet** |
| Modal training and serving | **Run live**: Qwen2.5-0.5B on 54 rows, 28 steps on an L40S in 30 s for $0.02; held-out 42% → 100% (docs/RECIPES.md) |
| Claude Code over MCP (`opentrain mcp`) | **Run live**: Claude Code evaluated, trained, waited and promoted through the tools |
| Rejection sampling, on-policy distillation, RL on Modal | **Run live** on GSM8K with Qwen2.5-0.5B: each a few steps on an L40S for about $0.10 (smoke tests, not results); RL with a `python` reward too |
| The agent end to end, from a use case with no data | **Run live** (`opentrain run --yes`, $3 cap): found MedQA on Hugging Face, built the grader (and caught its own broken one), baselined Qwen3.5-4B on Modal, trained twice, refused to ship either; $0.32 (docs/RECIPES.md) |
| Claude Code bringing its own data over MCP | **Run live**: turned a policy document into sealed eval rows and training rows (data/tasks), judged, baselined Haiku and Sonnet; $0 on the Claude login |
| RunPod training and serving | Tested end to end against a fake RunPod API with the real agent and trainer; **no live pod run yet** |
| Fireworks training (SFT, rejection, RL) | Tested against a fake SDK; **no live run yet** |
| Dataset search (Hugging Face, Kaggle), Kaggle import | Search run live; Kaggle import tested against a recorded zip |
| Tinker training | Written against SDK 0.40; **no real training run yet** (needs `TINKER_API_KEY`) |
| Local training (TRL + PEFT) | Written; untested here |

## Development

```bash
uv sync --extra dev
.venv/bin/pytest -q
```

Plan and roadmap: [docs/PLAN.md](docs/PLAN.md). Results on real sessions: [docs/FINDINGS.md](docs/FINDINGS.md).
