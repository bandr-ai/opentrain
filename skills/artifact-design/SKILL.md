---
name: artifact/design
description: How to make an artifact page that someone can decide from in a minute. Read it before artifact_make, every time you make a new kind or a page with charts or custom HTML.
---

# artifact/design

An artifact is a page the person decides from. They open it, read the top, glance at the
evidence and click one option. Make every page for that minute.

## Order of the page

1. **Title**: what it is, in their words. "Qwen3.5-0.8B on your 6 held-out tasks", not
   "Evaluation report".
2. **Summary** (one or two sentences): the answer to the question the page exists for.
3. **One callout** when something needs their attention before they decide: a risk, a cost, a
   reason the numbers are weak. Tone `warn` or `bad` for that, `good` when the news is the
   point. One per page; two at most. Never decorate with callouts.
4. **Stats**: three or four numbers that carry the decision, each with a short note saying what
   it is out of ("12 of 14", "median; 90% under 4,100").
5. **Evidence**: a chart, a table or a side-by-side, most important first (regressions before
   wins, failures before passes).
6. **How it was made**: one short `text` block at the end, only if it changes how the numbers
   should be read.

The decision goes in `decision`, not in the page: the app shows it as buttons under the page.

## Blocks

- `text`: short paragraphs. `**bold**`, `` `code` `` and `[links](https://...)` work.
- `stats`: `items: [{label, value, note?, tone?}]`. Labels are a few words. Values are the
  number itself ("31%", "$0.42", "1,206"), not a sentence.
- `bars`: `items: [{label, value, max?, note?, tone?}]` for a ranked count (repos, tool calls,
  reasons). Sort largest first. One scale per group: give the same `max` when it matters.
- `chart`: `kind: line | bar`, `series: [{name, points: [[x, y], ...]}]`, plus `title`, `unit`
  ("%", "s", "$"), `x_label`, `y_label`, `note`, `max`. Use `line` for anything over steps or
  time (loss, reward, cost so far), `bar` to compare a few named things (base vs trained, model
  vs model). Up to five series. The y axis starts at zero unless you pass `zero: false`, which
  you do only for loss-like curves where the shape is the point.
- `table`: `columns`, `rows`. Cells reading pass/fail, yes/no, fixed/regressed become coloured
  chips; numbers right-align by themselves. Keep cells short; put long text in `samples`.
- `samples`: side by side, `left_label/left` and `right_label/right`: a request and what the
  model did, base answer vs trained answer, before vs after.
- `list`: a few short items. If order matters, say so in the title.
- `callout`: `tone: good | warn | bad | info`, `title`, `text`.
- `html`: your own static HTML and CSS, when none of the above shows it well (a timeline, a
  diagram, an annotated diff). See below.

## Making it read well

- Real numbers from tools only. Never estimate a number that a tool can tell you, and never
  round in a way that changes the story (3 of 6 is 50%, not "about half").
- Say what each number is out of. "5 passed" means nothing; "5 of 6 held-out tasks" decides.
- Small samples: when fewer than about 10 tasks decide something, say that one task is a large
  share and a difference of one or two can be chance.
- Name things the way the person knows them: their repo names, their words for the work, the
  model they picked. Never internal ids unless they asked for them.
- Short labels, plain verbs, no hype. "Fails on CSS-only changes", not "Struggles with styling
  tasks".
- Write the page for both light and dark: let the page's own colours do it (below).

## Custom HTML

Use an `html` block only when the standard blocks cannot show the thing. It is drawn inside
the page, which has no scripts and loads nothing from outside: no `<script>`, no external images,
fonts or stylesheets, no forms, no iframes. Inline SVG and a `<style>` element are fine.

- Use the page's colour variables so it reads in light and dark: `var(--ink)` text,
  `var(--muted)` and `var(--faint)` secondary text, `var(--line)` hairlines, `var(--soft)` and
  `var(--raised)` surfaces, `var(--c1)` to `var(--c5)` for series, `var(--good)`, `var(--warn)`,
  `var(--bad)`, `var(--info)` for meaning. Never a fixed colour that only works on white.
- Fonts: `var(--display)` for headings, `var(--text)` for text, `var(--mono)` for code.
- Prefix your class names (`.tl-`, `.diff-`) so your CSS cannot restyle the rest of the page.
- Lay out with flex or grid and `gap`. It must fit a phone: nothing wider than the screen except
  a table or diagram inside a container with `overflow-x: auto`.
- In SVG, use a `viewBox`, leave room for the outermost labels, place every mark with one scale,
  and give every shape an explicit `fill`.
- Everything visible at rest: no animation needed to see the content.

## Never

- Held-out task content, or a model's answers to it, in any artifact.
- A credential, a token or an absolute path from the person's machine.
- Emoji as decoration, "01 / 02 / 03" numbering on things that are not a sequence, or a card
  around every block.
- A second question in chat about what the page asks: the page asks it.
