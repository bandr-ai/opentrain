---
name: bandits/certify
description: Prove a task's verifier before it grades or trains anything - the reference passes, doing nothing fails, a known-wrong answer fails, and environment breakage is never mistaken for a failing test. Use on every new task, and before every evaluation.
---

# bandits/certify

A verifier that passes an empty answer, or fails the right one, teaches the wrong thing and grades
wrong. Certify it against answers whose outcome you know in advance.

## When to use

- Every task, before it enters the store (`opentrain mine` does this for you).
- Before every evaluation: `opentrain lint` re-runs the controls on every stored task, because
  environments drift (a dependency upgrade, a deleted clone, a new interpreter).
- After any change to a task folder by hand.

## The controls

Each runs in a fresh git worktree at the base commit, with the test files overlaid as they stood at
the reference commit; the verdict is the test command's exit code alone.

| Control | Answer | Must |
| --- | --- | --- |
| reference | `solution/solution.patch` | pass |
| noop | no change | fail because tests fail |
| wrong | the reference with its largest code file withheld (or, for one file, its largest hunk) | fail because tests fail |

When the reference is one hunk there is no wrong answer distinct from the noop; `wrong` is left
empty. Docs and lockfiles are never the withheld part.

## Environment is not a failure

A run that cannot start fails for every answer, so it proves nothing. These are `environment`, never
a valid noop or wrong failure: timeout, exit 126/127 or "command not found", a missing module the
repo does not provide, a missing npm script or Go module, pytest usage errors, "no tests ran". A
test importing a name the reference adds is a real failure. If the reference fails for an
environment reason, the controls' detail starts with `environment:`.

## Steps

1. `opentrain lint` (all stored tasks) or let `opentrain mine` certify new ones.
2. Read every broken row. For `environment`, fix the machine and lint again. For a verifier that
   now passes the noop or fails the reference, drop or rebuild the task; never edit the tests to
   make the controls pass.
3. Held-out rows show only id and reason; do not open held-out folders to debug them unless you
   are the evaluator.

## Measure of success

`opentrain lint` exits 0: every stored verifier holds, on this machine, today.

## Failure modes

- Flaky tests: the reference passes once and fails once. Lint twice; a task that flips is broken.
- Slow suites: controls run three times; narrow the test command rather than raising the timeout.
- Local runs execute the repo's code as you. Harbor/Docker runs of the same folder are the
  isolated path.
