---
name: data/history
description: Verified coding tasks from the repos' own commit history, when sessions alone yield too few. Each commit that changed code and its tests becomes a task, rewritten into the request a person would have typed.
---

# data/history

Few sessions end in a committed change with tests, so a laptop yields a handful of session tasks.
The repos those sessions touched hold many more such changes.

## Steps

1. `tasks_from_history` (default repos: those the imported traces worked in; or `repos=[...]`).
   `author` defaults to the repo's configured git email, so the tasks are the person's own work;
   pass `author=""` for everyone's. `since` keeps it recent.
2. Each commit: environment = the repo at its parent; verifier = its test files at the commit, or,
   for a commit without test changes (verifier auto or judge), a judge comparing the attempt with the
   commit's change after `checks` pass;
   instruction = the commit rewritten by claude-code/sonnet into the request a person would type,
   without the solution (`rewrite=false` keeps the stripped message). It is certified like a
   session task (reference passes, doing nothing fails, a partial answer fails) before it counts.
3. Read the result: accepted, rejected with reasons. `environment` rejections mean the test command
   cannot run on this machine (missing deps): say which repo and what to install.
4. The held-out split refreshes by date, so the newest commits test the model.

## Limits

A commit message may hide what the person really asked; the rewrite works from the diff too, and
the tests decide. Very large commits, merges and test-only commits are skipped with the reason.
Certifying runs each repo's tests three times per commit: expect minutes per task.
