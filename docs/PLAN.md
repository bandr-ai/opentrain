# Plan and status

The product plan lives in the team doc *OpenTrain: product plan and roadmap*
(an internal document), with tabs for the plan, each stage in detail,
and the harness design. This file tracks where the code stands against it.

| Milestone | Plan | Code |
| --- | --- | --- |
| v0.1 bench | Task yield, time breakdown, baselines, one specialist by hand, cross-person check | `opentrain bench`, `bench models`, `mine` run on real sessions. The specialist-by-hand and the cross-person check need training compute and both founders' sessions |
| v0.2 train | Server + engine, trainer/evaluator, Trainfile, headless runtime, one backend, goal/speed, export | All present. Tinker training is written to the documented SDK but has not run against a real key |
| v0.3 watch | Triggers, unattended runs, promote-if-better, rollback on live drop, fix goal, weekly note | `watch` (sessions/flag/nightly triggers, rollback from live error and handoff rates), `fix`, `revert`. Weekly note: not yet |
| v0.4 mount | Direct runtime, more backends and models, more session sources, web UI, recipe kit, portable update | Direct runtime, web UI, `skill add`, `export`. Codex/pi/opencode session import and the portable update package: not yet |
| v0.5 plan | Capability map, engine chooses recipes, cost goal, RFT, "no training needed" | Agent mode (`opentrain run`) with the train tools is the seed; the rest not yet |
| v0.6 pipe | Non-coding traces, bring-your-own evals, graders | Not yet |
| v0.7 merge | Teams, pooled loops, commons | Not yet |

## Decisions still open (from the plan)

- Licence for the open project (none chosen yet, so the repo is not yet publishable as open source).
- Launch base model, and the trainer and serving behind it.
- Vendor terms for unattended `claude -p` under a plan; whether `watch` defaults to an API key.
- The yield number that counts as a pass for v0.1.
