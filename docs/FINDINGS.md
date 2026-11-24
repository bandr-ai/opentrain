# Findings from real sessions

First runs of v0.1 bench and `opentrain mine` on one founder's laptop, 24–25 November 2026: 39 Claude
Code sessions, 9 of 12 coding repos still on disk. One machine and one person, so these are a first
reading, not the v0.1 answer. The plan's v0.1 needs both founders' sessions and the cross-person check.

## 1. The 5x speed target is out of reach on these sessions

| Measure | Value |
| --- | --- |
| Model share of working time (OpenTrain's timeline split) | 40.7% |
| Model share per Claude Code's own `cost-state` totals | 42.3% (38 sessions) |
| Median session's model share | 68.4% |
| End-to-end speedup if the model got 10x faster | 1.58x |
| End-to-end speedup if the model took no time at all | 1.69x |

The two independent measurements agree within 2 points. On this sample, tools (shell commands,
builds, CI waits, subagents) take more working time than the model. A faster model cannot fix tool
time, so the plan's done-when for v0.2 ("median task time drops at least 5x") would not be met even
by an instant model.

The plan already names this risk ("Speed gain falls short… under 5x means fix serving or change the
headline"). Before committing to the speed headline:

- Measure on both founders' main machines, where routine coding sessions dominate. The median
  session here is 68% model time; the total is pulled down by a few long sessions waiting on CI.
- Count the time per task, not per session: a task's tool time includes deploys and PR waits that a
  mined task does not replay.
- Consider leading with a headline that does not need 5x end to end, such as cost per solved task.

## 2. Yield: 25% of coding sessions are verified candidates

| Step | Dropped | Left |
| --- | --- | --- |
| Sessions | | 39 |
| No code changed (questions, planning, ops) | 27 | 12 |
| No repo state (folder was not a git repo, or no commit before) | 5 | 7 |
| No tests in repo | 1 | 6 |
| Checks not run or failed | 2 | 4 |
| No committed reference | 1 | 3 |

3 verified candidates, plus 3 repairable, from the 12 coding sessions. After mining and certifying
the verifiers, **1 task was accepted**, from a private web and API monorepo: the reference passes in 8.2s, doing
nothing fails. The other candidates were rejected for good reasons: a reference that changes only
tests, a verifier that already passes before the change, edits never committed.

This laptop is mostly planning and ops sessions, so its yield understates a coding machine's.

## 3. Bugs real data found, now fixed

Each was invisible in synthetic tests:

1. **Deleted worktrees.** Workspace tools (Conductor, T3 worktrees) delete a session's folder after
   merge. Sessions now resolve to the main clone by repo name.
2. **Sessions outside any repo.** A session started in a scratch folder that edits a repo elsewhere
   now resolves to that repo from its edited files.
3. **Tests run in another clone.** A session edited one clone and ran tests in another. Clones are
   now matched by root commit.
4. **The shell remembers `cd`.** Claude Code's shell keeps its folder between commands, so a bare
   `pnpm test` ran in `web/`. Each test run now records the shell's folder.
5. **Subshells.** `(cd web && npx vitest run)` hid the `cd`.
6. **Monorepo dependencies.** `web/node_modules` and `api/.venv` are now linked into the worktree,
   with the worktree's own source first on `PYTHONPATH` so an editable install cannot test the
   user's copy instead.
7. **Unrelated failing tests.** A suite with database tests failed at the reference for reasons
   unrelated to the change. The miner now narrows to the changed test files when the reference fails.
8. **First prompts are openings, not specs.** "Look into this, the data pipeline seems broken"
   cannot be solved against hidden tests. The instruction is now every prompt the person typed, in
   order, with harness notices (task notifications, PR-watch updates, compaction summaries) removed.
9. **Waits counted as model time.** A wait for a background task or CI ended by a harness notice was
   counted as model time. It is now tool time.

## 4. Long sessions make weak tasks

The accepted task's session had 10 prompts across investigation, a fix, a deploy and testing. Its
reference is the first commit touching the edited files, which may be only part of that work. Claude
Haiku, run headless on it, failed in 2m48s. The next data recipe should split a long session into one
task per commit (or per prompt that led to a commit), so each task asks for what its reference did.
