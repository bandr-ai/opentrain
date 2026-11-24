---
name: cl/mine
description: Turn the person's own coding-agent sessions into verified Harbor tasks (instruction, repo at the start, tests from the finished work, reference answer). Use when the task store needs more or fresher tasks, after new sessions arrive, or when yield is lower than expected.
---

# cl/mine

Build tasks from sessions. A task is only as good as its verifier: never accept one whose controls
did not pass, and never invent an instruction, a test or an answer.

## When to use

- New sessions since the last mine (the Trainfile `watch` line fired), or the store is empty.
- `opentrain bench yield` shows verified candidates that are not yet in the store.
- Lint broke tasks and you want them rebuilt from source.

## Steps

1. **Judge the sessions.** `opentrain bench yield` shows the funnel: instruction, code changed, repo
   state, tests in repo, checks green, committed reference. Read the top skip reasons before mining;
   a skip is information, not a bug to route around.
2. **Mine.** `opentrain mine` (add `--include-repairable` to also take sessions whose tests did not
   end green or never ran: the miner runs the tests itself). For each session it:
   - takes the first prompt verbatim as `instruction.md`;
   - uses the repo at the last commit before the session as the environment, and the first commit
     after it that touches an edited file as the reference;
   - makes the test files changed between them the verifier and the rest the solution patch;
   - takes the test command from the session's last passing run, cleaned to run from the repo root
     (`cd sub &&` kept, pipes, redirections and display env vars dropped), or infers one from the
     repo (pytest, `npm test`, `go test ./...`, `cargo test`);
   - certifies it (see `bandits/certify`); if doing nothing passes, retargets the command at the
     changed test files once and certifies again;
   - adds it to `.opentrain/tasks` only if the controls hold, then refreshes the held-out split
     (see `data/heldout`).
3. **Read the report.** The summary line counts every session; the full per-session report is
   `.opentrain/data/mine-<stamp>.json`. Group rejections by reason and fix the biggest group first.
4. **Inspect a sample.** `opentrain task ls`, then `opentrain task show <id>` on two or three new
   tasks: does the instruction ask for what the patch does? Is the test command sane?

## Measure of success

Accepted tasks per 100 sessions (yield), with every accepted task's controls valid, and zero
held-out ids ever present in the training store.

## Failure modes and what to do

| Reason in the report | Meaning | Do |
| --- | --- | --- |
| `no instruction` / `no code changed` | a chat, not work | nothing |
| `no repo state` | folder gone, not git, or no commit before the session | add `--repos <folder of clones>` |
| `no reference` | the work was never committed | nothing; uncommitted work cannot be replayed |
| `patch empty` | the change touched only tests | nothing |
| `no test command` | none run, none inferable | nothing, or note the repo's command for later |
| `environment` | the command cannot run here (missing tool, module, venv, timeout) | fix the machine (install deps), then re-mine; never accept |
| `controls failed: noop passes` | the tests do not detect the change, even narrowed | nothing; the session has no verifier |
| `controls failed: reference fails` | the recorded answer fails its own tests | check for flaky tests; do not loosen the verifier |
| `controls failed: wrong passes` | a partial answer passes | the tests check too little; leave it out |
| `error` | a bug in the miner on this session | report it with the session id |

Re-mining is cheap for known sessions: a task already in either store is kept, not rebuilt.
