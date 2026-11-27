# OpenTrain beta: getting started

OpenTrain makes a small model that does your work, and proves it on tasks it never saw before it
ships. This page is for beta users: what you need, what to try, where your data goes, and how to
send feedback.

## What you need

- macOS or Linux, Python 3.11+ and [uv](https://docs.astral.sh/uv/).
- **Claude Code**, signed in. OpenTrain's agent runs on your own Claude login (no API key), and Claude
  is the teacher whose runs the small model learns from.
- **Somewhere to train**, on your own account:
  - [Modal](https://modal.com): `modal setup` once. Trains Qwen3.5 / Qwen3.8 and serves them.
  - or [Tinker](https://thinkingmachines.ai/tinker): paste your key in Settings. Trains Inkling 2 Mini
    and Inkling-Small.
  - or RunPod: `opentrain login runpod`.
- For coding: repos with tests that run on your machine. The tests decide every task.

## Install

```bash
uv tool install "opentrain-cli[modal]"      # add ,tinker for Tinker
```

## Try it (about 15 minutes, free until you approve a paid step)

```bash
cd a-repo-you-work-in
opentrain traces              # what your coding agents left on this machine
opentrain traces --import     # read them here; credentials are scrubbed; nothing is uploaded
opentrain web                 # opens the app
```

In the app, click **a small model that does my coding**. The agent:

1. shows what it found, and where your agent went wrong; you tick what to fix;
2. builds tasks from your sessions and your repos' commits, certified by your tests, and holds the
   newest out for the eval;
3. scores what you use today and the untrained small model on the held-out tasks;
4. proposes training, with its price; nothing paid runs until you approve it;
5. scores the trained model and ships it only if it wins with nothing regressing.

A first training run on Modal with a 4B model costs about $1–3. Set the ceiling in the `Trainfile`
(`budget  $5 a week`); OpenTrain will not spend past it.

Not coding? Ask for anything else in your own words ("a small model that writes our support replies
from these tickets", "is Haiku good enough on these questions?"), or drop a CSV on the app.

## Where your data goes

| What | Where it goes |
| --- | --- |
| Your coding agents' logs | Read on your machine only. |
| Commits turned into tasks | Each diff is sent to Claude, on your own Claude login, to write the request. |
| Teacher runs | Claude Code runs inside a temporary checkout of your repo, on your login. |
| Training examples | Uploaded, with credentials scrubbed, to **your** Modal / Tinker / RunPod account when you approve a training run. |
| Anything to us | Nothing. OpenTrain has no telemetry. |

The `Trainfile` in your project says what OpenTrain may do (budget, what may leave the machine,
which tasks are held out); the server enforces every line.

## Sending feedback

```bash
opentrain report
```

prints what happened in the project as counts only: agents found, sessions that changed code, tasks
built and certified, scores, spend, and the first line of any error. No code, prompts, file or repo
names. Paste it with a line on what you expected and what happened. `--names` adds the names of the
themes you were shown, if you are happy to share them.

## Known limits in this beta

- Most laptops hold few sessions that end in a tested commit; your repos' commit history usually
  supplies most tasks.
- Readers for Cursor, opencode, Gemini CLI, Cline, aider, pi and Copilot follow those agents'
  documented formats; tell us if one misses your logs (`opentrain report` shows what was found).
- The trained coding model's first live runs are happening now; expect rough edges there first.
