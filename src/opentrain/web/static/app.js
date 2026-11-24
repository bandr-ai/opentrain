/* OpenTrain web UI. Vanilla JS, no dependencies, no build step. */
"use strict";

const TOKEN = document.querySelector('meta[name="opentrain-token"]').content;
const $main = () => document.getElementById("main");

/* ---------------------------------------------------------------- helpers */

const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const isNum = (x) => typeof x === "number" && isFinite(x);
const dash = '<span class="faint">—</span>';

function usd(x, digits) {
  if (!isNum(x)) return dash;
  const d = digits ?? (Math.abs(x) >= 100 ? 0 : 2);
  return "$" + x.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
}
function pct(x, digits = 1) {
  if (!isNum(x)) return dash;
  return (x * 100).toFixed(digits).replace(/\.0$/, "") + "%";
}
function secs(s) {
  if (!isNum(s)) return dash;
  const a = Math.abs(s);
  if (a < 10) return s.toFixed(1) + " s";
  if (a < 90) return Math.round(s) + " s";
  if (a < 3600) return Math.floor(a / 60) * Math.sign(s) + "m " + Math.round(a % 60) + "s";
  const h = Math.floor(a / 3600), m = Math.round((a % 3600) / 60);
  return (s < 0 ? "-" : "") + h + "h " + m + "m";
}
function signedSecs(s) {
  if (!isNum(s)) return dash;
  return (s > 0 ? "+" : s < 0 ? "−" : "±") + secs(Math.abs(s)).replace(/<[^>]+>/g, "");
}
function parseAt(iso) {
  if (!iso) return null;
  let s = String(iso);
  const m = s.match(/^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z$/);
  if (m) s = `${m[1]}-${m[2]}-${m[3]}T${m[4]}:${m[5]}:${m[6]}Z`;
  const d = new Date(s);
  return isNaN(d) ? null : d;
}
function ago(iso) {
  const d = parseAt(iso);
  if (!d) return dash;
  const s = (Date.now() - d.getTime()) / 1000;
  const t = d.toLocaleString();
  let txt;
  if (s < 45) txt = "just now";
  else if (s < 3600) txt = Math.round(s / 60) + " min ago";
  else if (s < 86400) txt = Math.round(s / 3600) + " h ago";
  else if (s < 86400 * 30) txt = Math.round(s / 86400) + " d ago";
  else txt = d.toLocaleDateString();
  return `<span title="${esc(t)}">${txt}</span>`;
}
function clock(iso) {
  const d = parseAt(iso);
  return d ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "";
}
const short = (s, n = 12) => (s && String(s).length > n ? String(s).slice(0, n) + "…" : String(s ?? ""));
const plural = (n, word, many) => `${n} ${n === 1 ? word : many || word + "s"}`;
const empty = (title, body, cmd) =>
  `<div class="empty"><h3>${esc(title)}</h3><div>${body}</div>${cmd ? `<div class="cmd">${esc(cmd)}</div>` : ""}</div>`;

async function api(path, opts) {
  const r = await fetch("/api/" + path, opts);
  let body = null;
  try { body = await r.json(); } catch (_) { /* not JSON */ }
  if (!r.ok) {
    const e = new Error((body && body.error) || r.status + " " + r.statusText);
    e.status = r.status;
    e.body = body;
    throw e;
  }
  return body;
}
const post = (path, data) =>
  api(path, { method: "POST", headers: { "Content-Type": "application/json", "X-OpenTrain-Token": TOKEN }, body: JSON.stringify(data || {}) });

let toastTimer;
function toast(msg, bad) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.className = "toast show" + (bad ? " bad" : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.className = "toast" + (bad ? " bad" : "")), 3200);
}

/* ---------------------------------------------------------------- icons + nav */

const I = (d) => `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${d}</svg>`;
const ICONS = {
  home: I('<path d="M2.5 7L8 2.5 13.5 7v6.5h-11z"/><path d="M6.5 13.5V10h3v3.5"/>'),
  runs: I('<circle cx="8" cy="8" r="5.5"/><path d="M8 5v3l2 1.5"/>'),
  evals: I('<path d="M3 8.5l3 3 7-7"/>'),
  versions: I('<circle cx="4" cy="4" r="1.5"/><circle cx="4" cy="12" r="1.5"/><circle cx="12" cy="8" r="1.5"/><path d="M4 5.5v5M5.5 4.5c3 0 5 1 5.5 2"/>'),
  race: I('<path d="M3 13.5V2.5M3 3h8l-1.5 2.5L11 8H3"/>'),
  data: I('<ellipse cx="8" cy="4" rx="5" ry="1.8"/><path d="M3 4v8c0 1 2.2 1.8 5 1.8s5-.8 5-1.8V4M3 8c0 1 2.2 1.8 5 1.8S13 9 13 8"/>'),
  skills: I('<path d="M3 2.5h7.5L13 5v8.5H3z"/><path d="M5.5 7h5M5.5 9.5h5M5.5 12h3"/>'),
  policy: I('<path d="M8 1.8l5 2v4c0 3-2.2 5.2-5 6.4-2.8-1.2-5-3.4-5-6.4v-4z"/>'),
};
const SCREENS = [
  ["home", "Home"], ["runs", "Runs"], ["evals", "Evals"], ["versions", "Versions"],
  ["race", "Race"], ["data", "Data"], ["skills", "Skills"], ["policy", "Policy"],
];

function buildNav() {
  const nav = document.getElementById("nav");
  nav.insertAdjacentHTML("beforeend",
    SCREENS.map(([k, label]) => `<a class="item" data-k="${k}" href="#/${k}">${ICONS[k]}<span>${label}</span><span class="count" id="count-${k}"></span></a>`).join("") +
    `<div class="foot"><div>State</div><div class="mono" id="state-path"></div></div>`);
}
function setActive(k) {
  document.querySelectorAll(".nav a.item").forEach((a) => a.classList.toggle("active", a.dataset.k === k));
}
function setCount(k, n) {
  const el = document.getElementById("count-" + k);
  if (el) el.textContent = n ? String(n) : "";
}

/* ---------------------------------------------------------------- charts */

/* One series, one axis. points: [{x, y, cur}]; fmt formats a y value with its unit. */
function lineChart(points, { fmt, title, yMin0 = true, maxY } = {}) {
  const pts = points.filter((p) => isNum(p.y));
  if (!pts.length) return `<div class="chart-empty">No ${esc(title || "data")} yet</div>`;
  const W = 520, H = 170, L = 52, R = 18, T = 14, B = 26;
  let lo = yMin0 ? 0 : Math.min(...pts.map((p) => p.y));
  let hi = maxY ?? Math.max(...pts.map((p) => p.y));
  if (hi === lo) hi = lo + (hi || 1);
  const ticks = niceTicks(lo, hi, 4);
  lo = Math.min(lo, ticks[0]); hi = Math.max(hi, ticks[ticks.length - 1]);
  const n = points.length;
  const x = (i) => (n === 1 ? L + (W - L - R) / 2 : L + (i * (W - L - R)) / (n - 1));
  const y = (v) => T + (H - T - B) * (1 - (v - lo) / (hi - lo));
  const grid = ticks.map((t) => `<line class="gridline" x1="${L}" x2="${W - R}" y1="${y(t)}" y2="${y(t)}"/><text class="axis" x="${L - 8}" y="${y(t) + 3.5}" text-anchor="end">${esc(fmt(t, true))}</text>`).join("");
  const step = Math.ceil(n / 10);
  const xl = points.map((p, i) => (i % step === 0 || i === n - 1 ? `<text class="axis" x="${x(i)}" y="${H - 6}" text-anchor="middle">${esc(p.x)}</text>` : "")).join("");
  const idx = points.map((p, i) => [p, i]).filter(([p]) => isNum(p.y));
  const d = idx.map(([p, i], k) => `${k ? "L" : "M"}${x(i).toFixed(1)},${y(p.y).toFixed(1)}`).join("");
  const area = idx.length > 1 ? `<path class="area" d="${d}L${x(idx[idx.length - 1][1])},${y(lo)}L${x(idx[0][1])},${y(lo)}Z"/>` : "";
  const dots = idx.map(([p, i]) => `<g><circle class="hit" cx="${x(i)}" cy="${y(p.y)}" r="12"><title>${esc(p.x)}: ${esc(fmt(p.y))}</title></circle><circle class="pt${p.cur ? " cur" : ""}" cx="${x(i)}" cy="${y(p.y)}" r="4"/></g>`).join("");
  const [lp, li] = idx[idx.length - 1];
  const lbl = `<text class="lbl" x="${Math.min(x(li), W - R)}" y="${y(lp.y) - 10}" text-anchor="${li === n - 1 && n > 1 ? "end" : "middle"}">${esc(fmt(lp.y))}</text>`;
  return `<div class="chart"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title || "chart")}">${grid}${area}<path class="line" d="${d}"/>${dots}${lbl}${xl}</svg></div>`;
}
function niceTicks(lo, hi, count) {
  const span = hi - lo || 1;
  const raw = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count) || raw;
  const out = [];
  for (let v = Math.floor(lo / step) * step; v <= hi + step * 0.001; v += step) out.push(+v.toFixed(10));
  if (out[out.length - 1] < hi) out.push(out[out.length - 1] + step);
  return out;
}
/* Horizontal bars. rows: [{label, why, value, max, text, drop, soft}] */
function bars(rows) {
  const max = Math.max(1, ...rows.map((r) => r.max ?? r.value ?? 0));
  return `<div class="bars">${rows.map((r) => {
    const w = Math.max(0, Math.min(100, (100 * (r.value || 0)) / max));
    return `<div class="bl">${esc(r.label)}${r.why ? `<span class="why">${esc(r.why)}</span>` : ""}</div>
      <div class="track" title="${esc(r.label)}: ${esc(r.text ?? r.value)}"><div class="fill${r.soft ? " soft" : ""}" style="width:${w.toFixed(2)}%"></div></div>
      <div class="bv">${r.text ?? esc(r.value)}${r.drop ? `<span class="drop">−${esc(r.drop)}</span>` : ""}</div>`;
  }).join("")}</div>`;
}
const pctFmt = (v, axis) => (axis ? Math.round(v * 100) + "%" : (v * 100).toFixed(1).replace(/\.0$/, "") + "%");
const secFmt = (v, axis) => (axis ? (v >= 600 && v % 60 === 0 ? v / 60 + "m" : Math.round(v) + "s") : secs(v));

/* ---------------------------------------------------------------- badges */

function statusBadge(s) {
  const map = { live: ["accent", "live", true], done: ["good", "done"], failed: ["bad", "failed"], stalled: ["warn", "stalled"], empty: ["", "empty"] };
  const [cls, label, pulse] = map[s] || ["", s || "?"];
  return `<span class="badge ${cls}">${pulse || cls ? `<i class="dot${pulse ? " pulse" : ""}"></i>` : ""}${esc(label)}</span>`;
}
function decisionBadge(d) {
  const cls = { promoted: "good", rejected: "bad", reverted: "warn", proposed: "" }[d] ?? "";
  return `<span class="badge ${cls}">${esc(d || "—")}</span>`;
}
const passMark = (p) => (p === true ? '<span class="ok">✓ pass</span>' : p === false ? '<span class="no">✗ fail</span>' : dash);

/* ---------------------------------------------------------------- router */

let cleanup = [];
function onLeave(fn) { cleanup.push(fn); }
function every(ms, fn) {
  const id = setInterval(() => { if (!document.hidden) fn(); }, ms);
  onLeave(() => clearInterval(id));
}

function route() {
  cleanup.forEach((f) => { try { f(); } catch (_) {} });
  cleanup = [];
  const raw = location.hash.replace(/^#\/?/, "") || "home";
  const [pathPart, query = ""] = raw.split("?");
  const parts = pathPart.split("/").map(decodeURIComponent);
  const q = Object.fromEntries(new URLSearchParams(query));
  const screen = SCREENS.some(([k]) => k === parts[0]) ? parts[0] : "home";
  setActive(screen);
  document.title = (SCREENS.find(([k]) => k === screen) || [, "Home"])[1] + " · OpenTrain";
  const view = VIEWS[screen];
  const token = Symbol();
  route.current = token;
  const main = $main();
  main.innerHTML = '<div class="loading">Loading…</div>';
  Promise.resolve(view(parts.slice(1), q, () => route.current === token)).catch((e) => {
    if (route.current !== token) return;
    main.innerHTML = `<div class="alert bad">Could not load: ${esc(e.message)}</div>`;
  });
}

/* Render into main only if this view is still the one on screen. */
function paint(alive, html) {
  if (!alive()) return false;
  $main().innerHTML = html;
  return true;
}

/* ---------------------------------------------------------------- Home */

async function viewHome(_p, _q, alive) {
  const load = async () => {
    const h = await api("home");
    document.getElementById("state-path").textContent = h.state;
    setCount("versions", h.versions);
    const sh = h.share || {};
    const b = h.budget || {};
    const used = isNum(b.committed_usd) && b.dollars ? b.committed_usd / b.dollars : null;
    const cur = h.current;
    const lr = h.latest_run;
    const per = b.period_days == null ? "this run" : { 1: "today", 7: "this week", 30: "this month" }[b.period_days] || `last ${b.period_days} days`;

    const tiles = `
      <div class="grid stats">
        <div class="card stat">
          <div class="label">Work done by your model</div>
          <div class="value">${sh.share == null ? dash : pct(sh.share)}</div>
          <div class="foot">${sh.requests ? `${sh.ours.toLocaleString()} of ${plural(sh.requests, "request")}${sh.window_days ? `, last ${sh.window_days} days` : ", all time"}` : "No requests served yet"}</div>
        </div>
        <div class="card stat">
          <div class="label">Live version</div>
          <div class="value">${cur ? esc(cur.version) : dash}</div>
          <div class="foot">${cur ? `${isNum(cur.pass_rate) ? pct(cur.pass_rate) + " held-out pass" : "not scored"}${isNum(cur.median_seconds) ? " · " + secs(cur.median_seconds) + " median" : ""}` : "Nothing promoted yet"}</div>
        </div>
        <div class="card stat">
          <div class="label">Spend ${esc(per)}</div>
          <div class="value">${usd(b.committed_usd)}<span class="unit">of ${usd(b.dollars, 0)}</span></div>
          <div class="meter ${used > 0.9 ? "bad" : used > 0.7 ? "warn" : ""}"><i style="width:${Math.min(100, (used || 0) * 100).toFixed(1)}%"></i></div>
          <div class="foot">${isNum(b.remaining_usd) ? usd(b.remaining_usd) + " left · " : ""}budget ${esc(b.describe || "")}</div>
        </div>
        <div class="card stat">
          <div class="label">Tasks</div>
          <div class="value">${(h.tasks?.training ?? 0).toLocaleString()}<span class="unit">training</span></div>
          <div class="foot">${(h.tasks?.heldout ?? 0).toLocaleString()} held out for the evaluator · goal <b>${esc(h.goal)}</b></div>
        </div>
      </div>`;

    const steps = [
      [!!h.bench, "Measure how many sessions can become tasks", "opentrain bench"],
      [(h.tasks?.training || 0) > 0, "Build verified tasks from your sessions", "opentrain tasks build"],
      [!!lr, "Train a version on them", "opentrain train"],
      [!!cur, "Promote one past the held-out gate", "opentrain train"],
      [!!sh.requests, "Serve it to your agent", "opentrain serve"],
    ];
    const started = steps.every(([d]) => d);
    const checklist = started ? "" : `
      <div class="card"><div class="card-h"><h3>Getting started</h3><span class="right">${steps.filter(([d]) => d).length} of ${steps.length} done</span></div>
      <div class="card-b"><ul class="steps">${steps.map(([d, what, cmd]) => `<li class="${d ? "done" : ""}"><span class="tick ${d ? "ok" : "faint"}">${d ? "✓" : "○"}</span><span class="what">${esc(what)}</span><code>${esc(cmd)}</code></li>`).join("")}</ul></div></div>`;

    const trend = h.trend || [];
    const charts = trend.length ? `
      <div class="grid cols-2">
        <div class="card"><div class="card-h"><h3>Held-out pass rate</h3><span class="right">% of held-out tasks, per version</span></div><div class="card-b">${lineChart(trend.map((t) => ({ x: t.version, y: t.pass_rate, cur: cur && t.version === cur.version })), { fmt: pctFmt, title: "pass rate", maxY: 1 })}</div></div>
        <div class="card"><div class="card-h"><h3>Median time to pass</h3><span class="right">seconds per task, per version</span></div><div class="card-b">${lineChart(trend.map((t) => ({ x: t.version, y: t.median_seconds, cur: cur && t.version === cur.version })), { fmt: secFmt, title: "median time" })}</div></div>
      </div>` : "";

    const note = h.last_note;
    const lastRun = lr ? `
      <div class="card"><div class="card-h"><h3>Last run</h3><a class="mono" href="#/runs/${esc(lr.run_id)}">${esc(lr.run_id)}</a> ${statusBadge(lr.status)}<span class="right">${ago(lr.last_at)}</span></div>
      <div class="card-b">
        <dl class="kv">
          <dt>Trigger</dt><dd>${esc(lr.trigger || "—")}</dd>
          <dt>Spend</dt><dd class="num">${usd(lr.spend_usd)}</dd>
          ${lr.gate ? `<dt>Gate</dt><dd>${decisionBadge(lr.gate.decision)} <span class="dim">${esc((lr.gate.reasons || []).join("; "))}</span></dd>` : ""}
        </dl>
        ${note ? `<div style="margin-top:10px"><div class="dim" style="font-size:12px;margin-bottom:4px">Note for the next run${note.run_id !== lr.run_id ? ` (from <a class="mono" href="#/runs/${esc(note.run_id)}">${esc(short(note.run_id, 22))}</a>)` : ""}</div><blockquote class="note">${esc(note.text)}</blockquote></div>` : ""}
      </div></div>` : "";

    const warn = h.trainfile_error ? `<div class="alert bad">The Trainfile does not parse: ${esc(h.trainfile_error)}. <a href="#/policy">Fix it in Policy</a>.</div>` : "";
    paint(alive, `
      <div class="head"><div><h1>Home</h1><div class="sub">Your model, what it costs, and what it learned last.</div></div></div>
      ${warn}${tiles}${checklist}${charts}${lastRun}`);
  };
  await load();
  every(10000, () => load().catch(() => {}));
}

/* ---------------------------------------------------------------- Runs */

async function viewRuns(parts, q, alive) {
  if (parts[0]) return viewRun(parts[0], alive);
  const load = async () => {
    const d = await api("runs");
    setCount("runs", d.live || "");
    if (!d.runs.length) {
      return paint(alive, `<div class="head"><div><h1>Runs</h1><div class="sub">Every training run, live and past.</div></div></div>` +
        empty("No runs yet", "A run builds data, trains a candidate, scores it on held-out tasks and lets the gate decide.", "opentrain train"));
    }
    const total = d.runs.reduce((a, r) => a + (r.spend_usd || 0), 0);
    paint(alive, `
      <div class="head"><div><h1>Runs</h1><div class="sub">${plural(d.runs.length, "run")} · ${usd(total)} spent in total${d.live ? ` · <b>${d.live} live</b>` : ""}</div></div></div>
      <div class="card table-wrap"><table class="t">
        <thead><tr><th>Status</th><th>Run</th><th>Trigger</th><th>Started</th><th class="r">Duration</th><th class="r">Events</th><th class="r">Spend</th><th>Gate</th><th>Version</th></tr></thead>
        <tbody>${d.runs.map((r) => `
          <tr class="click" data-href="#/runs/${esc(r.run_id)}">
            <td>${statusBadge(r.status)}</td>
            <td><a class="mono" href="#/runs/${esc(r.run_id)}">${esc(r.run_id)}</a>${r.job && r.job.state ? ` <span class="dim">· job ${esc(r.job.state)}${isNum(r.job.progress) ? " " + pct(r.job.progress, 0) : ""}</span>` : ""}</td>
            <td class="dim">${esc(r.trigger || "—")}</td>
            <td class="dim">${ago(r.opened_at)}</td>
            <td class="r">${secs(r.seconds)}</td>
            <td class="r">${r.events}</td>
            <td class="r">${usd(r.spend_usd)}</td>
            <td>${r.gate ? decisionBadge(r.gate.decision) : dash}</td>
            <td class="mono">${r.versions.length ? r.versions.map((v) => esc(v)).join(", ") : dash}</td>
          </tr>`).join("")}</tbody></table></div>`);
    $main().querySelectorAll("tr[data-href]").forEach((tr) => tr.addEventListener("click", (e) => { if (!e.target.closest("a")) location.hash = tr.dataset.href; }));
  };
  await load();
  every(5000, () => load().catch(() => {}));
}

function eventLine(e) {
  const d = e.data || {};
  const k = e.kind;
  const pick = (...ks) => ks.map((x) => d[x]).find((v) => v != null);
  switch (k) {
    case "opened": return `Opened by <b>${esc(d.trigger || "?")}</b>${d.policy && d.policy.goal ? ` · goal ${esc(d.policy.goal)}` : ""}`;
    case "data": {
      const ids = pick("task_ids", "ids", "tasks");
      const n = Array.isArray(ids) ? ids.length : ids && typeof ids === "object" ? Object.keys(ids).length : pick("count");
      const hash = pick("sha256", "hash", "digest");
      return `Data snapshot: <b class="num">${esc(n ?? "?")}</b> tasks${isNum(d.examples) ? ` · <span class="num">${d.examples}</span> examples` : ""}${Array.isArray(d.skipped) && d.skipped.length ? ` · ${d.skipped.length} skipped` : ""}${hash ? ` · <span class="mono dim">${esc(short(hash, 12))}</span>` : ""}`;
    }
    case "recipe": return "Recipe: " + Object.entries(d).filter(([, v]) => ["string", "number", "boolean"].includes(typeof v)).slice(0, 6).map(([a, v]) => `<span class="dim">${esc(a)}</span> <span class="mono">${esc(v)}</span>`).join(" · ");
    case "spend": {
      const tok = (Number(d.tokens) || 0) + (Number(d.tokens_in) || 0) + (Number(d.tokens_out) || 0);
      return `Spent <b class="num">${usd(Number(d.usd) || 0)}</b>${d.what ? ` on ${esc(d.what)}` : ""}${d.task_id ? ` <span class="mono dim">${esc(d.task_id)}</span>` : ""}${d.passed === true || d.passed === false ? " " + passMark(d.passed) : ""}${tok ? ` · <span class="num">${tok.toLocaleString()}</span> tokens` : ""}`;
    }
    case "job": return `Job <span class="mono">${esc(pick("job_id", "id") || "?")}</span>${d.backend ? ` on ${esc(d.backend)}` : ""}${d.state ? ` · ${esc(d.state)}` : ""}`;
    case "checkpoint": return `Checkpoint <span class="mono">${esc(short(pick("checkpoint", "path", "uri") || "?", 64))}</span>`;
    case "eval": {
      if (d.task_id) return `<span class="mono">${esc(d.task_id)}</span> ${passMark(d.passed)}${isNum(d.seconds) ? ` in <span class="num">${secs(d.seconds)}</span>` : ""}${d.version ? ` · ${esc(d.version)}` : ""}`;
      const sc = d.scores || d.tasks;
      const rows = Array.isArray(sc) ? sc : sc && typeof sc === "object" ? Object.values(sc) : [];
      const passed = rows.filter((r) => r && (r === true || r.passed === true)).length;
      const who = d.version || d.model;
      return rows.length ? `Scored <b class="num">${passed}/${rows.length}</b> held-out tasks${who ? ` · <span class="mono">${esc(who)}</span>` : ""}${isNum(d.median_seconds) ? ` · median ${secs(d.median_seconds)}` : ""}${isNum(d.request) ? ` <span class="dim">(eval ${d.request} of ${esc(d.cap ?? "?")})</span>` : ""}` : "Evaluation";
    }
    case "gate": {
      const dec = d.decision || (d.promote === true ? "promoted" : d.promote === false ? "rejected" : "?");
      const rs = Array.isArray(d.reasons) ? d.reasons.join("; ") : d.reasons || "";
      return `${decisionBadge(dec)} ${d.version ? `<span class="mono">${esc(d.version)}</span> ` : ""}<span class="dim">${esc(rs)}</span>`;
    }
    case "note": return `<blockquote class="note">${esc(pick("text", "note", "message", "summary") || JSON.stringify(d))}</blockquote>`;
    case "closed": return `Closed${d.status ? ` · ${esc(d.status)}` : ""}${d.error ? ` · <span class="no">${esc(d.error)}</span>` : ""}`;
    default: return esc(k);
  }
}

function eventItem(e, t0, fresh) {
  const off = t0 && parseAt(e.at) ? (parseAt(e.at) - t0) / 1000 : null;
  const hasData = e.data && Object.keys(e.data).length;
  return `<li class="${fresh ? "fresh" : ""}" data-seq="${esc(e.seq)}">
    <span class="when"><b>${esc(clock(e.at))}</b>${isNum(off) ? "+" + secs(off).replace(/<[^>]+>/g, "") : ""}</span>
    <span class="kind k-${esc(e.kind)}">${esc(e.kind)}</span>
    <div class="what">${eventLine(e)}${hasData ? `<details><summary>details · #${esc(e.seq)} · ${esc(e.hash)}</summary><pre class="json">${esc(JSON.stringify(e.data, null, 2))}</pre></details>` : ""}</div>
  </li>`;
}

async function viewRun(id, alive) {
  const r = await api("runs/" + encodeURIComponent(id));
  const t0 = parseAt(r.opened_at);
  const job = r.job;
  const head = () => `
    <div class="crumb"><a href="#/runs">Runs</a> /</div>
    <div class="head"><div><h1 class="mono">${esc(r.run_id)}</h1>
      <div class="sub">${statusBadge(r.status)} ${r.verified ? '<span class="badge good" title="Every event carries the hash of the one before it">✓ record intact</span>' : '<span class="badge bad" title="An event was edited or removed">✗ record hash chain broken</span>'} · opened by ${esc(r.trigger || "?")} ${ago(r.opened_at)}</div></div></div>
    <div class="grid stats">
      <div class="card stat"><div class="label">Spend</div><div class="value" id="rs-spend">${usd(r.spend_usd)}</div><div class="foot">${r.tokens ? r.tokens.toLocaleString() + " tokens" : "model tokens + compute"}</div></div>
      <div class="card stat"><div class="label">Duration</div><div class="value">${secs(r.seconds)}</div><div class="foot" id="rs-events">${plural(r.events, "event")}</div></div>
      <div class="card stat"><div class="label">Training job</div><div class="value">${job ? esc(job.state || "?") : dash}</div>
        ${job && isNum(job.progress) ? `<div class="meter"><i style="width:${(job.progress * 100).toFixed(1)}%"></i></div>` : ""}
        <div class="foot">${job ? `<span class="mono">${esc(short(job.job_id, 18))}</span>${isNum(job.cost_usd) ? " · " + usd(job.cost_usd) + " so far" : ""}` : "no job launched"}</div></div>
      <div class="card stat"><div class="label">Gate</div><div class="value">${r.gate ? decisionBadge(r.gate.decision) : dash}</div><div class="foot">${r.versions.length ? "made " + r.versions.map((v) => `<a class="mono" href="#/versions">${esc(v)}</a>`).join(", ") : "no version yet"}</div></div>
    </div>`;
  const body = `${head()}
    <div class="card"><div class="card-h"><h3>Timeline</h3><span class="right" id="live-label">${r.status === "live" ? '<span class="badge accent"><i class="dot pulse"></i>streaming</span>' : "append-only, hash-chained"}</span></div>
      <ul class="timeline" id="timeline">${r.timeline.map((e) => eventItem(e, t0)).join("")}</ul></div>
    ${job && job.logs && job.logs.length ? `<div class="card"><div class="card-h"><h3>Job logs</h3><span class="right">last ${job.logs.length} lines</span></div><div class="card-b"><pre class="logs" id="logs">${esc(job.logs.join("\n"))}</pre></div></div>` : ""}`;
  if (!paint(alive, body)) return;
  const logs = document.getElementById("logs");
  if (logs) logs.scrollTop = logs.scrollHeight;
  if (r.closed || r.status === "done" || r.status === "failed") return;

  const last = r.timeline.length ? r.timeline[r.timeline.length - 1].seq : -1;
  const es = new EventSource(`/api/runs/${encodeURIComponent(r.run_id)}/events?after=${last}`);
  onLeave(() => es.close());
  es.addEventListener("run", (m) => {
    const e = JSON.parse(m.data);
    const ul = document.getElementById("timeline");
    if (!ul || ul.querySelector(`li[data-seq="${CSS.escape(String(e.seq))}"]`)) return;
    ul.insertAdjacentHTML("beforeend", eventItem(e, t0, true));
    if (e.kind === "spend") { r.spend_usd = (r.spend_usd || 0) + (Number(e.data.usd) || 0); const s = document.getElementById("rs-spend"); if (s) s.innerHTML = usd(r.spend_usd); }
    r.events += 1;
    const ev = document.getElementById("rs-events"); if (ev) ev.textContent = plural(r.events, "event");
  });
  es.addEventListener("end", () => {
    es.close();
    const l = document.getElementById("live-label"); if (l) l.textContent = "run closed";
    setTimeout(() => alive() && route(), 600);
  });
  // job state and logs live outside the record; refresh them quietly
  every(5000, async () => {
    const n = await api("runs/" + encodeURIComponent(r.run_id)).catch(() => null);
    if (!n || !alive() || !n.job) return;
    const lg = document.getElementById("logs");
    if (lg && n.job.logs) { const atEnd = lg.scrollTop + lg.clientHeight >= lg.scrollHeight - 8; lg.textContent = n.job.logs.join("\n"); if (atEnd) lg.scrollTop = lg.scrollHeight; }
  });
}

/* ---------------------------------------------------------------- Evals */

async function viewEvals(_p, q, alive) {
  const qs = new URLSearchParams();
  if (q.a) qs.set("a", q.a);
  if (q.b) qs.set("b", q.b);
  const d = await api("evals?" + qs);
  const head = `<div class="head"><div><h1>Evals</h1><div class="sub">Held-out tasks that flipped between two versions. ${plural(d.heldout_tasks, "held-out task")}; the trainer never sees them.</div></div>`;
  if (!d.choices.length) {
    return paint(alive, head + "</div>" + empty("No held-out scores yet",
      "Scores appear once a run evaluates a candidate on the held-out tasks; the gate uses them to decide.", "opentrain train"));
  }
  const opt = (sel) => d.choices.map((c) => `<option value="${esc(c)}"${c === sel ? " selected" : ""}>${esc(c)}</option>`).join("");
  const A = d.a_summary || {}, B = d.b_summary || {};
  const m = d.median_seconds_both_pass || {};
  const dRate = isNum(A.pass_rate) && isNum(B.pass_rate) ? B.pass_rate - A.pass_rate : null;
  const dMed = isNum(m.a) && isNum(m.b) && m.a ? (m.b - m.a) / m.a : null;
  const c = d.counts;
  const filter = q.show || (c.regressed + c.fixed ? "flipped" : "all");
  const filters = [["flipped", "Flipped", c.regressed + c.fixed], ["regressed", "Pass → fail", c.regressed], ["fixed", "Fail → pass", c.fixed], ["all", "All", d.rows.length]];
  const rows = d.rows.filter((r) => filter === "all" || (filter === "flipped" ? r.change === "regressed" || r.change === "fixed" : r.change === filter));
  const changeBadge = (ch) => ({ regressed: '<span class="badge bad">↓ pass → fail</span>', fixed: '<span class="badge good">↑ fail → pass</span>', new: '<span class="badge">new</span>', dropped: '<span class="badge warn">not scored</span>', pass: '<span class="dim">both pass</span>', fail: '<span class="faint">both fail</span>' })[ch];
  const html = `${head}
    <div class="actions"><select id="sel-a" aria-label="from version">${d.a ? "" : '<option value="">—</option>'}${opt(d.a)}</select><span class="dim">→</span><select id="sel-b" aria-label="to version">${opt(d.b)}</select></div></div>
    <div class="grid stats">
      <div class="card stat"><div class="label">Pass rate ${esc(d.a || "")} → ${esc(d.b || "")}</div><div class="value">${pct(B.pass_rate)}<span class="unit">${isNum(dRate) ? (dRate >= 0 ? "+" : "−") + (Math.abs(dRate) * 100).toFixed(1) + " pts" : ""}</span></div><div class="foot">${esc(d.a || "—")}: ${pct(A.pass_rate)} of ${A.tasks ?? 0} · ${esc(d.b)}: ${B.passed ?? 0} of ${B.tasks ?? 0} tasks</div></div>
      <div class="card stat"><div class="label">Median time, tasks both pass</div><div class="value ${isNum(dMed) ? (dMed < 0 ? "good" : dMed > 0 ? "bad" : "") : ""}">${isNum(dMed) ? (dMed <= 0 ? "−" : "+") + Math.abs(dMed * 100).toFixed(1) + "%" : dash}</div><div class="foot">${secs(m.a)} → ${secs(m.b)} over ${plural(m.tasks || 0, "task")}</div></div>
      <div class="card stat"><div class="label">Pass → fail</div><div class="value ${c.regressed ? "bad" : ""}">${c.regressed}<span class="unit">tasks</span></div><div class="foot">${c.regressed ? "regressions block promotion by default" : "no regressions"}</div></div>
      <div class="card stat"><div class="label">Fail → pass</div><div class="value ${c.fixed ? "good" : ""}">${c.fixed}<span class="unit">tasks</span></div><div class="foot">${c.pass} both pass · ${c.fail} both fail</div></div>
    </div>
    <div class="card"><div class="card-h"><div class="chips">${filters.map(([k, l, n]) => `<button class="chip${k === filter ? " on" : ""}" data-show="${k}">${l}<span class="n">${n}</span></button>`).join("")}</div><span class="right">held-out ids only; instructions stay with the evaluator</span></div>
    ${rows.length ? `<div class="table-wrap"><table class="t"><thead><tr><th>Held-out task</th><th>${esc(d.a || "—")}</th><th>${esc(d.b)}</th><th>Change</th><th class="r">Time ${esc(d.a || "")}</th><th class="r">Time ${esc(d.b)}</th><th class="r">Δ time</th></tr></thead><tbody>
      ${rows.map((r) => `<tr><td class="mono">${esc(r.task_id)}</td><td>${passMark(r.a_passed)}</td><td>${passMark(r.b_passed)}</td><td>${changeBadge(r.change)}</td>
        <td class="r">${secs(r.a_seconds)}</td><td class="r">${secs(r.b_seconds)}</td>
        <td class="r ${isNum(r.delta_seconds) ? (r.delta_seconds < 0 ? "ok" : r.delta_seconds > 0 ? "no" : "") : ""}">${isNum(r.delta_seconds) ? signedSecs(r.delta_seconds) + (isNum(r.delta_pct) ? ` <span class="dim">(${r.delta_pct > 0 ? "+" : ""}${r.delta_pct}%)</span>` : "") : dash}</td></tr>`).join("")}
    </tbody></table></div>` : `<div class="card-b dim">Nothing in this filter.</div>`}</div>`;
  if (!paint(alive, html)) return;
  const go = (patch) => {
    const n = new URLSearchParams({ a: d.a || "", b: d.b || "", show: filter, ...patch });
    location.hash = "#/evals?" + n;
  };
  document.getElementById("sel-a").onchange = (e) => go({ a: e.target.value });
  document.getElementById("sel-b").onchange = (e) => go({ b: e.target.value, a: d.a === e.target.value ? "" : d.a || "" });
  $main().querySelectorAll("[data-show]").forEach((b) => (b.onclick = () => go({ show: b.dataset.show })));
}

/* ---------------------------------------------------------------- Versions */

async function viewVersions(_p, _q, alive) {
  const d = await api("versions");
  setCount("versions", d.versions.length);
  const head = `<div class="head"><div><h1>Versions</h1><div class="sub">Every version, what it was trained on, and what the gate decided.</div></div>`;
  if (!d.versions.length) {
    return paint(alive, head + "</div>" + empty("No versions yet", "A run that finishes training registers a version; the gate promotes it if it beats the live one on held-out tasks.", "opentrain train"));
  }
  const cur = d.current;
  const action = d.can_revert
    ? `<button class="btn danger" id="revert">Roll back ${esc(cur)}${d.revert_to ? ` → ${esc(d.revert_to)}` : ""}</button>`
    : "";
  const pts = (key) => d.versions.map((v) => ({ x: v.version, y: v[key], cur: v.current }));
  const html = `${head}<div class="actions">${cur ? `<span class="dim">Live:</span> <span class="badge accent mono">${esc(cur)}</span>` : '<span class="dim">Nothing live</span>'}${action}</div></div>
    <div class="grid cols-2">
      <div class="card"><div class="card-h"><h3>Held-out pass rate</h3><span class="right">% of held-out tasks</span></div><div class="card-b">${lineChart(pts("pass_rate"), { fmt: pctFmt, title: "pass rate", maxY: 1 })}</div></div>
      <div class="card"><div class="card-h"><h3>Median time to pass</h3><span class="right">seconds per passed task</span></div><div class="card-b">${lineChart(pts("median_seconds"), { fmt: secFmt, title: "median time" })}</div></div>
    </div>
    <div class="card table-wrap"><table class="t"><thead><tr><th>Version</th><th>Gate</th><th>Trained on</th><th class="r">Pass rate</th><th class="r">Median time</th><th>Base model</th><th>Run</th><th>Created</th></tr></thead><tbody>
      ${d.versions.slice().reverse().map((v) => {
        const t = v.trained_on || {};
        const recipe = t.recipe ? Object.entries(t.recipe).slice(0, 3).map(([k, x]) => `${esc(k)} ${esc(x)}`).join(" · ") : "";
        return `<tr><td><span class="mono"><b>${esc(v.version)}</b></span> ${v.current ? '<span class="badge accent"><i class="dot"></i>live</span>' : ""}</td>
          <td>${decisionBadge(v.decision)}${t.gate && t.gate.reasons.length ? `<div class="dim" style="font-size:12px;max-width:280px">${esc(t.gate.reasons.join("; "))}</div>` : ""}</td>
          <td>${isNum(t.tasks) ? `<span class="num">${t.tasks}</span> tasks` : dash}${recipe ? `<div class="dim" style="font-size:12px">${recipe}</div>` : ""}${isNum(t.spend_usd) ? `<div class="faint" style="font-size:12px">${usd(t.spend_usd)} spent</div>` : ""}</td>
          <td class="r">${pct(v.pass_rate)}${v.heldout_scored ? `<div class="faint">${v.heldout_scored} tasks</div>` : ""}</td>
          <td class="r">${secs(v.median_seconds)}</td>
          <td><span class="mono">${esc(v.base_model)}</span><div class="faint mono ellip" title="${esc(v.checkpoint)}">${esc(v.backend)} · ${esc(short(v.checkpoint, 28))}</div></td>
          <td>${v.run_id ? `<a class="mono" href="#/runs/${esc(v.run_id)}">${esc(short(v.run_id, 15))}</a>` : dash}</td>
          <td class="dim">${ago(v.created_at)}</td></tr>`;
      }).join("")}
    </tbody></table></div>
    ${d.history.length ? `<div class="card"><div class="card-h"><h3>What was live, and why</h3></div><div class="table-wrap"><table class="t"><tbody>
      ${d.history.slice().reverse().map((h) => `<tr><td class="mono nowrap">${esc(h.from || "none")} → <b>${esc(h.to || "none")}</b></td><td>${esc(h.why || "")}</td><td class="dim">${ago(h.at)}</td></tr>`).join("")}
    </tbody></table></div></div>` : ""}`;
  if (!paint(alive, html)) return;
  const btn = document.getElementById("revert");
  if (btn) btn.onclick = async () => {
    const target = d.revert_to ? `put ${d.revert_to} back live` : "leave no version live (nothing earlier was promoted)";
    if (!confirm(`Roll back ${cur}? This will mark ${cur} reverted and ${target}.`)) return;
    btn.disabled = true;
    try {
      const r = await post("versions/revert", { why: "rolled back from the web UI" });
      toast(`Rolled back ${r.reverted}${r.current ? "; " + r.current + " is live" : "; nothing is live"}`);
      route();
    } catch (e) {
      toast(e.message, true);
      btn.disabled = false;
    }
  };
}

/* ---------------------------------------------------------------- Race */

function contenders(race) {
  for (const k of ["contenders", "entries", "models", "results", "runs", "racers", "lanes"]) {
    const v = race[k];
    if (Array.isArray(v) && v.length && v.every((x) => x && typeof x === "object")) return [k, v];
    if (v && typeof v === "object" && !Array.isArray(v) && Object.values(v).every((x) => x && typeof x === "object" && !Array.isArray(x)))
      return [k, Object.entries(v).map(([name, x]) => ({ name, ...x }))];
  }
  return [null, null];
}
const nameOf = (c, i) => String(c.name ?? c.model ?? c.version ?? c.label ?? c.id ?? `#${i + 1}`);
function perTask(c) {
  const src = c.tasks ?? c.scores ?? c.per_task;
  const out = {};
  if (Array.isArray(src)) src.forEach((r) => r && r.task_id && (out[r.task_id] = r));
  else if (src && typeof src === "object") Object.entries(src).forEach(([k, r]) => (out[k] = typeof r === "boolean" ? { passed: r } : r));
  return out;
}
function fmtVal(k, v) {
  if (typeof v === "boolean") return v ? '<span class="ok">yes</span>' : '<span class="no">no</span>';
  if (isNum(v)) {
    if (/usd|cost|dollar/i.test(k)) return usd(v);
    if (/rate|share|pct_?$/i.test(k) && v <= 1) return pct(v);
    if (/sec|time|duration|latency/i.test(k)) return secs(v);
    return `<span class="num">${v.toLocaleString()}</span>`;
  }
  if (typeof v === "string") return esc(short(v, 80));
  return `<span class="faint mono">${esc(short(JSON.stringify(v), 60))}</span>`;
}

async function viewRace(_p, _q, alive) {
  const d = await api("race");
  const head = `<div class="head"><div><h1>Race</h1><div class="sub">The latest side-by-side of versions or models on the same tasks.</div></div>`;
  if (!d.race) return paint(alive, head + "</div>" + empty("No race yet", "A race runs two or more models on the same tasks and writes the result to <code>.opentrain/races/</code>.", "opentrain race"));
  const race = d.race;
  const [, cs] = typeof race === "object" && !Array.isArray(race) ? contenders(race) : [null, Array.isArray(race) ? race : null];
  const meta = typeof race === "object" && !Array.isArray(race) ? Object.entries(race).filter(([, v]) => v != null && typeof v !== "object") : [];
  let body;
  if (cs) {
    const names = cs.map(nameOf);
    const passing = cs.map((c, i) => [c, i]).filter(([c]) => c.passed === true && isNum(c.seconds)).sort((a, b) => a[0].seconds - b[0].seconds);
    const winner = race.winner ?? race.best ?? (passing.length ? names[passing[0][1]] : null);
    const scalarKeys = [...new Set(cs.flatMap((c) => Object.keys(c).filter((k) => c[k] == null || typeof c[k] !== "object")))].filter((k) => !["name", "model", "version", "label", "id"].includes(k));
    const cards = cs.map((c, i) => {
      const win = c.winner === true || (winner != null && String(winner) === names[i]);
      return `<div class="card${win ? " win" : ""}"><div class="card-h"><h3 class="mono">${esc(names[i])}</h3>${win ? '<span class="badge good">winner</span>' : ""}</div>
        <div class="card-b"><dl class="kv">${scalarKeys.filter((k) => c[k] != null && c[k] !== "").map((k) => `<dt>${esc(k.replace(/_/g, " "))}</dt><dd>${fmtVal(k, c[k])}</dd>`).join("")}</dl></div></div>`;
    }).join("");
    const pt = cs.map(perTask);
    const ids = [...new Set(pt.flatMap((x) => Object.keys(x)))].sort();
    const table = ids.length ? `<div class="card table-wrap"><div class="card-h"><h3>Per task</h3><span class="right">${plural(ids.length, "task")}</span></div><table class="t"><thead><tr><th>Task</th>${names.map((n) => `<th>${esc(n)}</th>`).join("")}</tr></thead><tbody>
      ${ids.map((id) => {
        const times = pt.map((x) => (x[id] && x[id].passed && isNum(x[id].seconds) ? x[id].seconds : null)).filter(isNum);
        const best = times.length ? Math.min(...times) : null;
        return `<tr><td class="mono">${esc(id)}</td>${pt.map((x) => { const r = x[id]; if (!r) return `<td>${dash}</td>`; return `<td>${passMark(r.passed)}${isNum(r.seconds) ? ` <span class="num ${r.seconds === best && times.length > 1 ? "ok" : "dim"}">${secs(r.seconds)}</span>` : ""}</td>`; }).join("")}</tr>`;
      }).join("")}</tbody></table></div>` : "";
    body = `<div class="race">${cards}</div>${table}`;
  } else {
    body = `<div class="card"><div class="card-b"><pre class="json">${esc(JSON.stringify(race, null, 2))}</pre></div></div>`;
  }
  paint(alive, `${head}<div class="actions"><span class="mono dim">${esc(d.file)}</span>${d.count > 1 ? `<span class="faint">latest of ${d.count}</span>` : ""}</div></div>
    ${meta.length ? `<div class="card"><div class="card-b"><dl class="kv">${meta.map(([k, v]) => `<dt>${esc(k.replace(/_/g, " "))}</dt><dd>${fmtVal(k, v)}</dd>`).join("")}</dl></div></div>` : ""}${body}`);
}

/* ---------------------------------------------------------------- Data */

const STEP_LABELS = {
  instruction: "Has an instruction", edits: "Changed code", repo_state: "Repo state recoverable",
  tests_in_repo: "Tests in the repo", checks_passed: "Tests ran green", reference: "Reference committed",
};

async function viewData(parts, q, alive) {
  if (parts[0]) return viewTask(parts[0], alive);
  const d = await api("data");
  setCount("data", d.built.training);
  const b = d.bench;
  const head = `<div class="head"><div><h1>Data</h1><div class="sub">How your sessions became tasks, and what was skipped and why.</div></div>`;
  if (!b && !d.tasks.length) {
    return paint(alive, head + "</div>" + empty("No bench yet", "The bench reads your coding-agent sessions and measures how many can become verified tasks.", "opentrain bench"));
  }
  let funnel = "";
  if (b) {
    const y = b.yield || {};
    const rows = [{ label: "Sessions read", value: y.sessions, text: `<b>${(y.sessions ?? 0).toLocaleString()}</b>` }];
    (b.funnel || []).forEach((f) => rows.push({ label: STEP_LABELS[f.step] || f.step, why: f.dropped ? `skipped: ${f.label}` : "", value: f.remaining, max: y.sessions, text: f.remaining.toLocaleString(), drop: f.dropped || 0 }));
    const built = d.built.training + d.built.heldout;
    rows.push({ label: "Built and verified", why: "reference passes, doing nothing fails", value: built, max: y.sessions, text: built.toLocaleString(), soft: !built });
    const det = Object.entries(b.skip_details || {});
    const t = b.time;
    funnel = `
      <div class="grid stats">
        <div class="card stat"><div class="label">Sessions read</div><div class="value">${(y.sessions ?? 0).toLocaleString()}</div><div class="foot">${(y.coding_sessions ?? 0).toLocaleString()} changed code</div></div>
        <div class="card stat"><div class="label">Verified candidates</div><div class="value">${(y.verified_candidates ?? 0).toLocaleString()}</div><div class="foot">${isNum(y.yield_per_100) ? `<b class="num">${y.yield_per_100}</b> per 100 sessions · ` : ""}+${y.repairable ?? 0} repairable</div></div>
        <div class="card stat"><div class="label">Training tasks</div><div class="value">${d.built.training.toLocaleString()}</div><div class="foot">built and in the store</div></div>
        <div class="card stat"><div class="label">Held-out tasks</div><div class="value">${d.built.heldout.toLocaleString()}</div><div class="foot">count only; only the evaluator reads them</div></div>
      </div>
      <div class="grid cols-2">
        <div class="card"><div class="card-h"><h3>Task yield funnel</h3><span class="right">sessions left after each check</span></div><div class="card-b">${bars(rows)}</div></div>
        <div class="card"><div class="card-h"><h3>Why sessions were skipped</h3><span class="right">${plural(det.reduce((a, [, n]) => a + n, 0), "session")}</span></div>
          ${det.length ? `<div class="table-wrap"><table class="t"><tbody>${det.slice(0, 14).map(([k, n]) => { const [s, ...rest] = k.split(": "); return `<tr><td>${esc(s)}<div class="dim" style="font-size:12px">${esc(rest.join(": "))}</div></td><td class="r">${n}</td></tr>`; }).join("")}</tbody></table></div>` : '<div class="card-b dim">No skip details recorded.</div>'}
        </div>
      </div>
      ${t && t.hours ? `<div class="card"><div class="card-h"><h3>Where working time goes</h3><span class="right">${t.sessions} sessions · hours</span></div><div class="card-b">
        ${bars([["model", "Model"], ["tool", "Tools"], ["human", "Waiting on you"], ["idle", "Idle"]].map(([k, l]) => ({ label: l, value: t.hours[k] || 0, text: (t.hours[k] ?? 0).toFixed(1) + " h", soft: k === "human" || k === "idle" })))}
        <div class="dim" style="margin-top:10px;font-size:12.5px">Model share of working time <b class="num">${isNum(t.model_share_of_working_time_pct) ? t.model_share_of_working_time_pct + "%" : "—"}</b>. A 10× faster model gives at most <b class="num">${t.ceiling_speedup_if_model_10x ?? "—"}×</b> end to end; an instant one <b class="num">${t.ceiling_speedup_if_model_infinite ?? "—"}×</b>.</div>
      </div></div>` : ""}
      <div class="faint" style="font-size:12px;margin:8px 2px 14px">From <span class="mono">${esc(b.file)}</span> · ${ago(b.at)}${b.source ? ` · ${esc(b.source)}` : ""}</div>`;
  }
  const taskList = d.tasks.length ? `
    <div class="card"><div class="card-h"><h3>Training tasks</h3><span class="dim">${d.tasks.length}</span><span class="right"><input class="search" id="filter" placeholder="Filter by id, repo or text" aria-label="Filter tasks"></span></div>
    <ul class="tasks" id="task-list">${d.tasks.map((t) => {
      const ctl = t.controls;
      const ok = ctl && ctl.reference && !ctl.noop && ctl.wrong !== true;
      const at = t.attempts;
      return `<li data-text="${esc((t.task_id + " " + (t.repo || "") + " " + (t.family || "") + " " + (t.preview || "")).toLowerCase())}">
        <div class="top"><a class="mono" href="#/data/${encodeURIComponent(t.task_id)}">${esc(t.task_id)}</a>
          ${t.family ? `<span class="badge">${esc(t.family)}</span>` : ""}
          ${ctl ? (ok ? '<span class="badge good" title="reference passes, doing nothing fails">✓ verifier</span>' : '<span class="badge bad">verifier failed controls</span>') : ""}
          <span class="meta">${at ? `<span title="attempts that passed / attempts">${at.passed}/${at.count} attempts pass${isNum(at.best_seconds) ? " · best " + secs(at.best_seconds) : ""}</span>` : ""}${t.repo ? `<span class="mono">${esc(String(t.repo).split("/").pop())}</span>` : ""}<span>${ago(t.created_at)}</span></span></div>
        <div class="preview">${esc(t.preview)}</div></li>`;
    }).join("")}</ul></div>` : `<div class="card"><div class="card-h"><h3>Training tasks</h3></div><div class="card-b">${empty("No training tasks built yet", "Building turns verified candidates into tasks whose verifier passed its controls.", "opentrain tasks build")}</div></div>`;
  if (!paint(alive, head + "</div>" + funnel + taskList)) return;
  const f = document.getElementById("filter");
  if (f) f.oninput = () => {
    const s = f.value.trim().toLowerCase();
    document.querySelectorAll("#task-list li").forEach((li) => (li.hidden = !!s && !li.dataset.text.includes(s)));
  };
}

async function viewTask(id, alive) {
  const t = await api("tasks/" + encodeURIComponent(id));
  const ctl = t.controls || {};
  paint(alive, `<div class="crumb"><a href="#/data">Data</a> / training task</div>
    <div class="head"><div><h1 class="mono">${esc(t.task_id)}</h1><div class="sub">${t.family ? esc(t.family) + " · " : ""}${esc(t.repo || "")}</div></div></div>
    <div class="grid cols-2">
      <div class="card"><div class="card-h"><h3>Instruction</h3><span class="right">what the person asked</span></div><div class="card-b"><div class="md" style="white-space:pre-wrap">${esc(t.instruction)}</div></div></div>
      <div class="card"><div class="card-h"><h3>Task</h3></div><div class="card-b"><dl class="kv">
        <dt>Repo</dt><dd class="mono">${esc(t.repo || "—")}</dd>
        <dt>Base commit</dt><dd class="mono">${esc(t.base_commit || "—")}</dd>
        <dt>Test command</dt><dd><code>${esc(t.test_command || "—")}</code></dd>
        <dt>Work happened</dt><dd>${ago(t.created_at)}</dd>
        <dt>Reference passes</dt><dd>${passMark(ctl.reference)}</dd>
        <dt>Doing nothing passes</dt><dd>${ctl.noop == null ? dash : ctl.noop ? '<span class="no">yes (invalid)</span>' : '<span class="ok">no</span>'}</dd>
        <dt>Known-wrong passes</dt><dd>${ctl.wrong == null ? '<span class="faint">not tried</span>' : ctl.wrong ? '<span class="no">yes (invalid)</span>' : '<span class="ok">no</span>'}</dd>
        <dt>Verifier time</dt><dd class="num">${secs(ctl.seconds)}</dd>
        ${t.attempts ? `<dt>Attempts</dt><dd class="num">${t.attempts.passed} of ${t.attempts.count} pass · median ${secs(t.attempts.median_seconds)}</dd>` : ""}
      </dl></div></div>
    </div>`);
}

/* ---------------------------------------------------------------- Skills */

function markdown(src) {
  const lines = String(src).replace(/\r/g, "").split("\n");
  const out = [];
  let i = 0;
  const inline = (s) => esc(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<i>$2</i>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  while (i < lines.length) {
    const l = lines[i];
    if (/^```/.test(l)) {
      const buf = [];
      i++;
      while (i < lines.length && !/^```/.test(lines[i])) buf.push(lines[i++]);
      i++;
      out.push(`<pre><code>${esc(buf.join("\n"))}</code></pre>`);
    } else if (/^#{1,4}\s/.test(l)) {
      const n = Math.min(3, l.match(/^#+/)[0].length);
      out.push(`<h${n}>${inline(l.replace(/^#+\s*/, ""))}</h${n}>`);
      i++;
    } else if (/^\s*([-*]|\d+\.)\s/.test(l)) {
      const ordered = /^\s*\d+\./.test(l);
      const items = [];
      while (i < lines.length && /^\s*([-*]|\d+\.)\s/.test(lines[i])) items.push(`<li>${inline(lines[i++].replace(/^\s*([-*]|\d+\.)\s/, ""))}</li>`);
      out.push(`<${ordered ? "ol" : "ul"}>${items.join("")}</${ordered ? "ol" : "ul"}>`);
    } else if (!l.trim()) {
      i++;
    } else {
      const buf = [];
      while (i < lines.length && lines[i].trim() && !/^(```|#{1,4}\s|\s*([-*]|\d+\.)\s)/.test(lines[i])) buf.push(lines[i++]);
      out.push(`<p>${inline(buf.join(" "))}</p>`);
    }
  }
  return out.join("");
}

async function viewSkills(parts, _q, alive) {
  const d = await api("skills");
  setCount("skills", d.skills.length);
  const head = `<div class="head"><div><h1>Skills</h1><div class="sub">Instructions your agent and OpenTrain can load, from the repo and from <span class="mono">.opentrain/skills/</span>.</div></div></div>`;
  if (!d.skills.length) {
    return paint(alive, head + empty("No skills installed", `OpenTrain looks in ${d.roots.map((r) => `<span class="mono">${esc(r)}</span>`).join(", ")}. Add one as a folder with a SKILL.md:`, "mkdir -p skills/my-skill && $EDITOR skills/my-skill/SKILL.md"));
  }
  const sel = Math.max(0, d.skills.findIndex((s) => s.name === parts[0]));
  const s = d.skills[sel];
  paint(alive, `${head}<div class="skills">
    <div class="card"><ul class="skill-list">${d.skills.map((k, i) => `<li><a class="${i === sel ? "active" : ""}" href="#/skills/${encodeURIComponent(k.name)}"><div><b>${esc(k.name)}</b> <span class="badge">${esc(k.source)}</span></div><div class="d">${esc(k.description)}</div></a></li>`).join("")}</ul></div>
    <div class="card"><div class="card-h"><h3>${esc(s.name)}</h3><span class="right mono" title="${esc(s.path)}">${esc(short(s.path.split("/").slice(-3).join("/"), 48))}</span></div>
      <div class="card-b">${s.description ? `<p class="dim" style="margin-top:0">${esc(s.description)}</p>` : ""}<div class="md">${markdown(s.body)}</div></div></div>
  </div>`);
}

/* ---------------------------------------------------------------- Policy */

const KEY_HELP = {
  goal: "speed, fix or cost: what “better” means at the gate",
  budget: "dollars per run, day, week or month; no run can raise it",
  promote: "the gate rule for putting a version live",
  runtime: "how tasks are attempted: claude -p, codex exec or direct",
  trainer: "training backend, e.g. tinker",
  "base model": "the model to fine-tune",
  data: "what may be used, and whether uploads need approval",
  watch: "when to start a run on its own",
  serve: "where the live version is served",
  heldout: "share of tasks kept for the evaluator, e.g. 30%",
  "eval cap": "max held-out evaluations per run",
  provider: "OpenAI-compatible base URL for the direct runtime",
};

async function viewPolicy(_p, _q, alive) {
  const d = await api("policy");
  const parsedView = (p) => p ? `<dl class="kv">${[
    ["Goal", p.goal], ["Budget", p.budget], ["Promote when", p.promote], ["Regressions allowed", p.allow_regressions ? "yes" : "no"],
    ["Runtime", p.runtime], ["Trainer", p.trainer], ["Base model", p.base_model], ["Data", p.data],
    ["Uploads need approval", p.uploads_need_approval ? "yes" : "no"], ["Held out", p.heldout], ["Eval cap", p.eval_cap + " per run"],
    ["Watch", p.watch ? [p.watch.new_sessions ? `${p.watch.new_sessions} new sessions` : "", p.watch.nightly ? "nightly" : ""].filter(Boolean).join(" or ") || "flagged only" : "manual"],
    ["Serve", p.serve || "—"], ["Provider", p.provider || "—"],
  ].map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}</dl>` : '<div class="dim">Fix the error to see what it means.</div>';
  const html = `<div class="head"><div><h1>Policy</h1><div class="sub">The Trainfile: what OpenTrain may do on your behalf. The server enforces every line.</div></div>
      <div class="actions"><span class="mono dim">${esc(d.path)}</span></div></div>
    ${d.exists ? "" : `<div class="alert warn">No Trainfile yet; these are the defaults. Saving creates <span class="mono">${esc(d.path)}</span>.</div>`}
    <div id="policy-msg">${d.error ? `<div class="alert bad">The saved Trainfile does not parse: ${esc(d.error)}</div>` : ""}</div>
    <div class="grid cols-2">
      <div class="card"><div class="card-h"><h3>Trainfile</h3><span class="right"><span id="dirty" class="faint"></span></span>
        <button class="btn" id="reset">Revert</button><button class="btn primary" id="save">Save</button></div>
        <div class="editor-wrap"><textarea class="editor" id="editor" spellcheck="false" autocapitalize="off" autocomplete="off" aria-label="Trainfile">${esc(d.text)}</textarea></div></div>
      <div>
        <div class="card"><div class="card-h"><h3>What it means</h3><span class="right">as the engine reads it</span></div><div class="card-b" id="parsed">${parsedView(d.parsed)}</div></div>
        <div class="card"><div class="card-h"><h3>Settings</h3><span class="right">one per line: key, two spaces, value</span></div><div class="card-b"><dl class="kv">${d.keys.map((k) => `<dt class="mono">${esc(k)}</dt><dd class="dim">${esc(KEY_HELP[k] || "")}</dd>`).join("")}</dl></div></div>
      </div>
    </div>`;
  if (!paint(alive, html)) return;
  const ed = document.getElementById("editor");
  let saved = d.text;
  const dirty = () => (document.getElementById("dirty").textContent = ed.value !== saved ? "unsaved changes" : "");
  ed.addEventListener("input", dirty);
  ed.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "s") { e.preventDefault(); save(); }
    if (e.key === "Tab" && !e.shiftKey) { e.preventDefault(); ed.setRangeText("  ", ed.selectionStart, ed.selectionEnd, "end"); dirty(); }
  });
  document.getElementById("reset").onclick = () => { ed.value = saved; dirty(); document.getElementById("policy-msg").innerHTML = ""; };
  const save = async () => {
    const msg = document.getElementById("policy-msg");
    const btn = document.getElementById("save");
    btn.disabled = true;
    try {
      const r = await post("policy", { text: ed.value });
      saved = ed.value.endsWith("\n") ? ed.value : ed.value + "\n";
      ed.value = saved;
      msg.innerHTML = "";
      document.getElementById("parsed").innerHTML = parsedView(r.parsed);
      toast("Trainfile saved");
    } catch (e) {
      msg.innerHTML = `<div class="alert bad"><b>Not saved.</b> ${esc(e.message)}</div>`;
    } finally {
      btn.disabled = false;
      dirty();
    }
  };
  document.getElementById("save").onclick = save;
  const guard = (e) => { if (ed.value !== saved) { e.preventDefault(); e.returnValue = ""; } };
  window.addEventListener("beforeunload", guard);
  onLeave(() => window.removeEventListener("beforeunload", guard));
}

/* ---------------------------------------------------------------- boot */

const VIEWS = { home: viewHome, runs: viewRuns, evals: viewEvals, versions: viewVersions, race: viewRace, data: viewData, skills: viewSkills, policy: viewPolicy };

document.addEventListener("DOMContentLoaded", () => {
  buildNav();
  window.addEventListener("hashchange", route);
  route();
  // nav counts and state path, once
  api("home").then((h) => {
    document.getElementById("state-path").textContent = h.state;
    document.getElementById("ver").textContent = h.opentrain || "";
    setCount("versions", h.versions);
    setCount("data", h.tasks && h.tasks.training);
    if (h.latest_run && h.latest_run.status === "live") setCount("runs", "●");
  }).catch(() => {});
});
