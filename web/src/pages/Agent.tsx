import { KeyboardEvent, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { api, events, Job, Message, Part, Permission, Proposal, Question, Session } from "../api";
import { go, useApp } from "../App";
import { ago, AutoCountdown, dur, Icon, Markdown, money, pct, Progress, toastError } from "../ui";
import { Curve } from "./Board";
import { Scorecard } from "./Scorecard";
import { ArtifactInline, artifactIdOf } from "./Artifacts";

// The agent sits beside the board: it does the work, proposes one step at a time, and you steer it.

export function merge<T extends Record<string, any>>(base: T, patch: Record<string, any>): T {
  const out: Record<string, any> = { ...base };
  for (const [k, v] of Object.entries(patch)) {
    if (v === null) delete out[k];
    else if (v && typeof v === "object" && !Array.isArray(v) && out[k] && typeof out[k] === "object" && !Array.isArray(out[k]))
      out[k] = merge(out[k], v);
    else out[k] = v;
  }
  return out as T;
}

type Command = { name: string; description: string };

export function useCommands() {
  const [cmds, setCmds] = useState<Command[]>([]);
  useEffect(() => {
    api.get<Command[]>("/api/command").then(setCmds).catch(() => {});
  }, []);
  return cmds;
}

export async function send(sid: string, text: string, commands: Command[]) {
  const m = text.match(/^\/(\S+)\s*([\s\S]*)$/);
  if (m && commands.some((c) => c.name === m[1])) {
    await api.post(`/api/session/${sid}/command`, { name: m[1], arguments: m[2] });
  } else {
    await api.post(`/api/session/${sid}/prompt`, { text });
  }
}

// ------------------------------------------------------------------ composer fill (starters, buttons)

type Fill = (text: string, submit?: boolean) => void;
let fillTarget: Fill | null = null;
/** Put text in the agent's composer (and optionally send it). */
export function fillComposer(text: string, submit = false) {
  fillTarget?.(text, submit);
}

// ------------------------------------------------------------------ the panel

export function AgentPanel({ convo = true }: { convo?: boolean }) {
  const { sid, setSid, sessions, busy, jobs, reloadSessions, board } = useApp();
  const [session, setSession] = useState<Session | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [perms, setPerms] = useState<Permission[]>([]);
  const [questions, setQuestions] = useState<Question[]>([]);
  const [proposals, setProposals] = useState<Proposal[]>([]);
  const [model, setModel] = useState("");
  const [picking, setPicking] = useState(false);
  const commands = useCommands();
  const scroller = useRef<HTMLDivElement>(null);
  // artifacts already in the conversation when it loaded; any made after that open beside the chat
  const opened = useRef<Set<string> | null>(null);
  const stick = useRef(true);

  const loadAsks = useCallback(() => {
    if (!sid) {
      setPerms([]); setQuestions([]); setProposals([]);
      return;
    }
    api.get<Permission[]>(`/api/permission?session=${sid}`).then(setPerms).catch(() => {});
    api.get<Question[]>(`/api/question?session=${sid}`).then(setQuestions).catch(() => {});
    api.get<Proposal[]>(`/api/proposal?session=${sid}`).then(setProposals).catch(() => {});
  }, [sid]);

  useEffect(() => {
    setSession(null);
    setMessages([]);
    stick.current = true;
    if (!sid) return;
    api.get<Session>(`/api/session/${sid}`).then((s) => {
      setSession(s);
      setModel(typeof s.model === "string" ? s.model : "");
    }).catch(() => setSid(null));
    opened.current = null;
    api.get<Message[]>(`/api/session/${sid}/message`).then((ms) => {
      opened.current = new Set(ms.flatMap((m) => m.parts.map((p) => (p.tool ? artifactIdOf(p.tool) && p.id : null))
        .filter((x): x is string => !!x)));
      setMessages(ms);
    }).catch(() => {});
    loadAsks();
  }, [sid, loadAsks, setSid]);

  useEffect(() => {
    if (!opened.current) return; // still loading the history
    for (const m of messages) for (const p of m.parts) {
      const aid = p.tool ? artifactIdOf(p.tool) : null;
      if (aid && !opened.current.has(p.id)) { opened.current.add(p.id); go("a", aid); }
    }
  }, [messages]);

  useEffect(() => events.on((e) => {
    const d = e.data || {};
    if (!sid) return;
    if (d.session_id !== sid && !(e.type.startsWith("session.") && d.id === sid)) return;
    switch (e.type) {
      case "session.updated":
        setSession((s) => (s ? merge(s, d) : s));
        break;
      case "message.created":
        setMessages((ms) => (ms.some((m) => m.id === d.id) ? ms : [...ms, { ...d, parts: [] }]));
        break;
      case "message.updated":
        setMessages((ms) => ms.map((m) => (m.id === d.id ? { ...merge(m, d), parts: m.parts } : m)));
        break;
      case "part.created":
        setMessages((ms) => ms.map((m) => (m.id === d.message_id ? { ...m, parts: [...m.parts.filter((p) => p.id !== d.id), d] } : m)));
        break;
      case "part.updated":
        setMessages((ms) => ms.map((m) => (m.id === d.message_id || m.parts.some((p) => p.id === d.id)
          ? { ...m, parts: m.parts.map((p) => (p.id === d.id ? merge(p, d) : p)) } : m)));
        break;
      case "part.deleted":
        setMessages((ms) => ms.map((m) => ({ ...m, parts: m.parts.filter((p) => p.id !== d.id) })));
        break;
      case "part.delta":
        setMessages((ms) => ms.map((m) => (m.id === d.message_id
          ? { ...m, parts: m.parts.map((p) => (p.id === d.part_id ? { ...p, [d.field]: ((p as any)[d.field] || "") + d.delta } : p)) }
          : m)));
        break;
    }
    if (e.type.startsWith("permission.") || e.type.startsWith("question.") || e.type.startsWith("proposal.")) loadAsks();
  }), [sid, loadAsks]);

  useLayoutEffect(() => {
    const el = scroller.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  });

  const state = (sid && busy[sid]) || (session as any)?.status || "idle";
  const isBusy = !!sid && state !== "idle";
  const jobById = useMemo(() => Object.fromEntries(jobs.map((j) => [j.id, j])), [jobs]);
  // jobs a tool call in the conversation already shows inline; the rest get a card at the end
  const shownJobs = new Set<string>(messages.flatMap((m) => m.parts.map((p) => p.tool?.metadata?.job_id)
    .filter((j): j is string => typeof j === "string" && !!jobById[j])));
  const sessionJobs = jobs.filter((j) => j.session_id === sid && j.state === "running");

  const onSend = async (text: string) => {
    let id = sid;
    if (!id) {
      const s = await api.post<Session>("/api/session", model ? { model } : {});
      id = s.id;
      setSid(id);
    } else if (model && model !== session?.model) {
      await api.patch(`/api/session/${id}`, { model });
    }
    stick.current = true;
    await send(id, text, commands);
    reloadSessions();
  };

  const top = sessions.filter((s) => !s.parent_id);

  return (
    <aside className={`agent ${convo ? "" : "foot-only"}`}>
      {convo && <div className="agent-head">
        <button className="agent-title" onClick={() => setPicking(!picking)} title="Switch conversation">
          <span className={`dot ${isBusy ? (state === "waiting" ? "waiting" : "busy") : ""}`} />
          <span className="t">{session?.title || (sid ? "…" : "new conversation")}</span>
          <Icon name="down" />
        </button>
        {isBusy && <button className="btn sm ghost" onClick={() => api.post(`/api/session/${sid}/interrupt`).catch(toastError)}>stop</button>}
        <button className="btn sm ghost" title="New conversation" onClick={() => { setSid(null); setPicking(false); }}>+ new</button>
      </div>}
      {convo && picking && (
        <div className="agent-sessions">
          {top.length === 0 && <div className="faint small" style={{ padding: "6px 10px" }}>no conversations yet</div>}
          {top.map((s) => (
            <button key={s.id} className={s.id === sid ? "on" : ""} onClick={() => { setSid(s.id); setPicking(false); }}>
              <span className={`dot ${busy[s.id] ? (busy[s.id] === "waiting" ? "waiting" : "busy") : ""}`} />
              <span className="t">{s.title || "new conversation"}</span>
              <span className="faint small">{ago(s.time?.updated ?? s.time?.created)}</span>
            </button>
          ))}
        </div>
      )}
      {convo && board && (board.datasets.length > 0 || board.scoreboard.length > 0) && <Scorecard board={board} />}
      {convo && <div className="agent-scroll" ref={scroller} onScroll={(e) => {
        const el = e.currentTarget;
        stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
      }}>
        <div className="agent-convo">
          {!sid && <AgentHello />}
          {messages.map((m) => <MessageView key={m.id} m={m} jobById={jobById} />)}
          {sessionJobs.filter((j) => !shownJobs.has(j.id)).map((j) => <JobCard key={j.id} job={j} />)}
          {isBusy && state !== "waiting" && !messages.some((m) => m.role === "assistant" && !m.finish && m.parts.length) && (
            <div className="row faint small"><span className="dot busy" /> thinking…</div>
          )}
        </div>
      </div>}
      <div className="agent-foot">
        {proposals.map((p) => <ProposalCard key={p.id} p={p} onDone={loadAsks} />)}
        {perms.map((p) => <PermissionAsk key={p.id} p={p} onDone={loadAsks} />)}
        {questions.map((q) => <QuestionAsk key={q.id} q={q} onDone={loadAsks} />)}
        <Composer onSend={onSend} model={model} setModel={setModel} busy={isBusy} commands={commands}
          placeholder={isBusy ? "steer while it works…" : sid ? "reply…" : "what should the model get better at?"} />
      </div>
    </aside>
  );
}

const STARTERS: [string, string][] = [
  ["a small model that does my coding", "Make me a small model that does my coding work, from my coding-agent traces on this machine and the repos they touched. Show me what you find first."],
  ["fix what my coding agent keeps getting wrong", "Look through my coding-agent traces, tell me what keeps going wrong, and train a small model that gets it right."],
  ["improve a model for my use case", "Find the best public data on Hugging Face and Kaggle and make this model better at it. Model: Qwen/Qwen3.5-4B. Use case: "],
  ["train on my data", "I have data for this (I'll drop the file in or point to it). Check how good it is, turn it into the right tasks, then train: "],
  ["find the cheapest model that does my job", "Compare a few models on my examples, including a small cheap one, and tell me which is good enough and what each costs: "],
  ["build the data with me", "Help me build a dataset for this from scratch: "],
];

type TraceSource = { name: string; label: string; count: number; imported: number };
type TraceView = { sources: TraceSource[]; imported: number; coding: number; examples: number;
  repos: { repo: string; traces: number }[]; moments: Record<string, number>;
  themes: { name: string; description: string; count: number; chosen: boolean }[] };

const MOMENT: Record<string, string> = { pushback: "pushback to review", corrected: "pushback to review", interrupted: "you stopped it",
  failing_check: "a check kept failing", churn: "same file edited over and over", tool_errors: "tool errors" };

// Like Paxel: show what coding-agent traces this machine has, import them locally, then what went wrong.
export function TracesPanel() {
  const [v, setV] = useState<TraceView | null>(null);
  const [busy, setBusy] = useState(false);
  const load = useCallback(() => { api.get<TraceView>("/api/traces").then(setV).catch(() => {}); }, []);
  useEffect(() => { load(); }, [load]);
  useEffect(() => events.on((e: any) => { if (e?.type?.startsWith?.("job")) load(); }), [load]);
  if (!v) return null;
  const found = v.sources.filter((s) => s.count > 0);
  if (!found.length) return null;
  const doImport = async () => {
    setBusy(true);
    try { await api.post("/api/traces/import", {}); } catch (e) { toastError(e); } finally { setBusy(false); }
  };
  const choose = async (name: string) => {
    const chosen = v.themes.filter((t) => (t.name === name ? !t.chosen : t.chosen)).map((t) => t.name);
    try { await api.post("/api/traces/themes", { chosen }); load(); } catch (e) { toastError(e); }
  };
  const moments = Object.entries(v.moments || {}).filter(([, n]) => n > 0);
  return (
    <div className="traces">
      <div className="traces-k">your coding traces · on this machine</div>
      <div className="traces-found">
        {found.map((s) => (
          <span key={s.name}><b>{s.count.toLocaleString()}</b> {s.label}{s.imported ? <span className="faint"> · {s.imported} imported</span> : null}</span>
        ))}
      </div>
      {v.imported ? (
        <div className="traces-sum">
          <span><b>{v.coding}</b> changed code</span>
          <span>in <b>{v.repos.length}</b> repos</span>
          <span><b>{v.examples}</b> training examples</span>
          {moments.map(([k, n]) => <span key={k} className="faint">{n} × {MOMENT[k] || k}</span>)}
        </div>
      ) : (
        <button className="btn" disabled={busy} onClick={doImport}>{busy ? "importing…" : "import (stays on this machine)"}</button>
      )}
      {v.themes.length > 0 && (
        <div className="traces-themes">
          <div className="faint small">what keeps going wrong · pick what to fix</div>
          {v.themes.map((t) => (
            <label key={t.name} className="theme" title={t.description}>
              <input type="checkbox" checked={t.chosen} onChange={() => choose(t.name)} /> {t.name} <span className="faint">· seen {t.count}×</span>
            </label>
          ))}
        </div>
      )}
    </div>
  );
}

function AgentHello() {
  return (
    <div className="hello">
      <pre className="logo">{"opentrain"}</pre>
      <p>What should your model get better at?</p>
      <TracesPanel />
      <div className="starters">
        {STARTERS.map(([t, text], i) => (
          <button key={t} className="starter" onClick={() => fillComposer(text)}>
            <span className="faint">{i + 1}.</span> {t}
          </button>
        ))}
      </div>
      <p className="faint small">or drop a CSV / JSONL file here</p>
    </div>
  );
}

// ------------------------------------------------------------------ the next step

export function ProposalCard({ p, onDone, compact }: { p: Proposal; onDone?: () => void; compact?: boolean }) {
  const [busy, setBusy] = useState(false);
  const [changing, setChanging] = useState(false);
  const [text, setText] = useState("");
  const reply = async (reply: "approve" | "change" | "skip", message?: string) => {
    setBusy(true);
    try {
      await api.post(`/api/session/${p.session_id}/proposal/${p.id}/reply`, { reply, ...(message ? { message } : {}) });
      onDone?.();
    } catch (e) {
      toastError(e);
    } finally {
      setBusy(false);
    }
  };
  const hold = () => api.post(`/api/session/${p.session_id}/proposal/${p.id}/reply`, { reply: "hold" })
    .then(() => onDone?.()).catch(() => {});
  const cost = p.cost_usd ? `~${money(p.cost_usd)}` : "free";
  return (
    <div className={`next ${compact ? "compact" : ""}`}>
      <div className="next-k">next step{p.stage ? ` · ${p.stage}` : ""}</div>
      <div className="next-t">{p.title}</div>
      {p.why && <div className="next-why">{p.why}</div>}
      <div className="next-meta">
        <span>{cost}</span>
        {p.minutes ? <span>~{p.minutes < 1 ? "<1" : Math.round(p.minutes)} min</span> : null}
        {p.needs.map((n) => <span key={n} className="need">{NEEDS[n] || n}</span>)}
      </div>
      <AutoCountdown at={p.auto_at} what="this step" onHold={hold} />
      {changing ? (
        <div className="next-change">
          <input className="input" autoFocus placeholder="what should be different?" value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && text.trim()) reply("change", text.trim()); if (e.key === "Escape") setChanging(false); }} />
          <button className="btn sm primary" disabled={!text.trim() || busy} onClick={() => reply("change", text.trim())}>send</button>
        </div>
      ) : (
        <div className="next-btns">
          <button className="btn primary" disabled={busy} onClick={() => reply("approve")}>Approve{p.cost_usd ? ` · ${cost}` : ""}</button>
          <button className="btn" disabled={busy} onClick={() => { setChanging(true); if (p.auto_at) hold(); }}>Change</button>
          <button className="btn ghost" disabled={busy} onClick={() => reply("skip")}>Skip</button>
        </div>
      )}
    </div>
  );
}

const NEEDS: Record<string, string> = {
  spend: "spends", upload: "uploads data", "job.start": "starts training", promote: "changes what's live",
  serve: "serves a model", webfetch: "fetches from the web",
};

// ------------------------------------------------------------------ messages

function MessageView({ m, jobById }: { m: Message; jobById: Record<string, Job> }) {
  const text = (ps: Part[]) => ps.filter((p) => p.type === "text").map((p) => p.text || "").join("\n");
  if (m.role === "user") return <div className="msg-user">{text(m.parts)}</div>;
  if (m.role === "synthetic") {
    const t = text(m.parts);
    const [first, ...rest] = t.split("\n");
    return <div className="msg-synth"><b>{first}</b>{rest.length ? "\n" + rest.join("\n") : ""}</div>;
  }
  if (m.role === "compaction") return <div className="msg-synth">Earlier messages were summarised to save context.</div>;
  const took = m.time?.completed && m.time?.created ? m.time.completed - m.time.created : null;
  return (
    <div className="msg-ai">
      {m.parts.map((p) => {
        if (p.type === "text") return p.text ? <Markdown key={p.id} text={p.text} /> : null;
        if (p.type === "reasoning") return p.text ? <Reasoning key={p.id} text={p.text} /> : null;
        if (p.type === "tool" && p.tool) {
          if (p.tool.name === "propose") return <ProposalLine key={p.id} tool={p.tool} />;
          const aid = artifactIdOf(p.tool);
          if (aid) return <ArtifactInline key={p.id} id={aid} />;
          const jid = p.tool.metadata?.job_id;
          const job = jid ? jobById[jid] : undefined;
          return (
            <div key={p.id} style={{ display: "grid", gap: 6 }}>
              <ToolView tool={p.tool} />
              {job && <JobCard job={job} />}
            </div>
          );
        }
        return null;
      })}
      {(m.finish || m.error) && (
        <div className="msg-meta">
          {took !== null && <span>{dur(took)}</span>}
          {m.cost ? <span>{money(m.cost)}{m.billing === "subscription" ? " (plan)" : ""}</span> : null}
          {m.finish === "interrupted" && <span className="badge warn">interrupted</span>}
          {m.error && m.finish !== "interrupted" && <span className="err">{m.error}</span>}
        </div>
      )}
    </div>
  );
}

function ProposalLine({ tool }: { tool: NonNullable<Part["tool"]> }) {
  const i = tool.input || {};
  const out = tool.state === "error" ? "skipped" : tool.state === "completed"
    ? (tool.output || "").startsWith("Approved") ? "approved" : "changed" : "waiting for you";
  const cls = out === "approved" ? "ok" : out === "skipped" ? "err" : out === "changed" ? "" : "warn";
  return (
    <div className="prop-line">
      <span className="faint">proposed</span> <b>{i.title}</b>
      {i.cost_usd ? <span className="faint"> · ~{money(i.cost_usd)}</span> : null}
      <span className={`badge ${cls === "ok" ? "good" : cls === "err" ? "bad" : cls === "warn" ? "warn" : ""}`}>{out}</span>
    </div>
  );
}

function Reasoning({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  return <div className={`reasoning ${open ? "open" : ""}`} onClick={() => setOpen(!open)}>{text.trim()}</div>;
}

const TOOL_LABELS: Record<string, string> = {
  dataset_import: "import data", dataset_add: "add rows", dataset_show: "look at data", dataset_edit: "edit dataset",
  dataset_label: "label with a teacher", dataset_eval: "evaluate", dataset_train: "train", dataset_promote: "gate",
  model_try: "try a model", datasets: "datasets", evals: "evals", status: "status", question: "ask you",
  train_estimate: "estimate cost", goal_set: "set goal", shell: "shell", read: "read", write: "write", edit: "edit",
};

function summary(t: NonNullable<Part["tool"]>) {
  if (t.title) return t.title;
  const a = t.input || {};
  for (const k of ["name", "model", "command", "path", "hf", "pattern", "url", "version", "job_id", "prompt", "goal"]) {
    if (a[k]) return String(a[k]);
  }
  return "";
}

function ToolView({ tool }: { tool: NonNullable<Part["tool"]> }) {
  const [open, setOpen] = useState(false);
  const took = tool.time?.end && tool.time?.start ? tool.time.end - tool.time.start : null;
  const running = tool.state === "running" || tool.state === "pending";
  let out = tool.state === "error" ? tool.error || "error" : tool.output || (tool.metadata?.output as string) || "";
  try {
    if (out && (out.startsWith("{") || out.startsWith("["))) out = JSON.stringify(JSON.parse(out), null, 2);
  } catch { /* not JSON */ }
  return (
    <div className="tool">
      <div className="tool-head" onClick={() => setOpen(!open)}>
        <span className={`tool-icon ${tool.state === "completed" ? "ok" : tool.state === "error" ? "bad" : ""}`}>
          {running ? <span className="spinner" /> : <Icon name={tool.state === "error" ? "x" : "check"} />}
        </span>
        <span className="name">{TOOL_LABELS[tool.name] || tool.name.replace(/_/g, " ")}</span>
        <span className="sum">{summary(tool)}</span>
        {took !== null && <span className="t">{dur(took)}</span>}
      </div>
      {open && (
        <div className="tool-body">
          <div className="label">input</div>
          <pre>{JSON.stringify(tool.input, null, 2)}</pre>
          {(out || running) && <><div className="label">{tool.state === "error" ? "error" : "output"}</div>
            <pre className={tool.state === "error" ? "err" : ""}>{out || "running…"}</pre></>}
        </div>
      )}
      {!open && tool.state === "error" && tool.error && (
        <div className="tool-body"><pre className="err">{tool.error.slice(0, 300)}</pre></div>
      )}
    </div>
  );
}

export function JobCard({ job }: { job: Job }) {
  const [now, setNow] = useState(Date.now() / 1000);
  useEffect(() => {
    if (job.state !== "running") return;
    const t = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => clearInterval(t);
  }, [job.state]);
  const started = job.time?.started;
  const ended = job.time?.ended;
  const badge = { running: "accent", succeeded: "good", failed: "bad", cancelled: "" }[job.state];
  return (
    <div className="jobcard">
      <div className="top">
        <span className={`badge ${badge}`}>{job.kind}</span>
        <b className="grow">{job.title}</b>
        {started && <span className="faint small">{dur((ended || now) - started)}</span>}
        {job.state === "running" && (
          <button className="btn ghost sm" onClick={() => api.post(`/api/job/${job.id}/cancel`).catch(toastError)}>cancel</button>
        )}
        <button className="btn ghost sm" onClick={() => go("jobs", job.id)}>log</button>
      </div>
      {job.state === "running" && <Progress value={job.progress} />}
      {job.kind === "train" && (job.reward_curve?.length ?? 0) > 1 ? (
        <div className="curve-l"><span className="faint small">reward {job.reward != null ? pct(job.reward) : ""}</span>
          <Curve run={{ ...(job as any), curve: job.reward_curve }} height={56} /></div>
      ) : job.kind === "train" && (job.curve?.length ?? 0) > 1 && <Curve run={{ ...(job as any), curve: job.curve }} height={56} />}
      <div className="small muted">
        {job.state !== "running" && <span className={job.state === "succeeded" ? "ok" : "err"}>{job.state}. </span>}
        {job.detail}
        {job.cost_usd ? ` · ${money(job.cost_usd)}` : ""}
        {job.estimate_usd ? ` of ~${money(job.estimate_usd)}` : ""}
        {job.error && job.state !== "succeeded" && <span className="err"> {job.error}</span>}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ asks

function describe(p: Permission) {
  const r = p.resources.join(", ");
  const md = p.metadata || {};
  switch (p.action) {
    case "spend": return <>spend <b>{r}</b>{md.budget_remaining_usd !== undefined && <> (budget left {money(md.budget_remaining_usd)})</>}</>;
    case "upload": return <>upload <b>{md.what || "data"}</b> to <b>{r}</b></>;
    case "job.start": return <>start training on <b>{r}</b>{md.base_model && <> with {md.base_model}</>}</>;
    case "shell": return <>run <code>{md.command || r}</code></>;
    case "edit": case "write": return <>{p.action} <code>{r}</code></>;
    case "promote": return <>make <b>{r}</b> the live model</>;
    case "serve": return <>serve <b>{r}</b></>;
    case "webfetch": return <>fetch <b>{r}</b></>;
    default: return <>{p.action} <b>{r}</b></>;
  }
}

function PermissionAsk({ p, onDone }: { p: Permission; onDone: () => void }) {
  const [busy, setBusy] = useState(false);
  const reply = async (r: "once" | "always" | "reject") => {
    setBusy(true);
    try {
      await api.post(`/api/session/${p.session_id}/permission/${p.id}/reply`, { reply: r });
    } catch (e) {
      toastError(e);
    }
    onDone();
  };
  return (
    <div className="ask">
      <h3>Wants to {describe(p)}</h3>
      {p.metadata?.command && p.action === "shell" && p.metadata.description && <div className="what muted">{p.metadata.description}</div>}
      <div className="row">
        <button className="btn primary sm" disabled={busy} onClick={() => reply("once")}>Allow</button>
        {p.action !== "spend" && <button className="btn sm" disabled={busy} onClick={() => reply("always")}>Always</button>}
        <button className="btn sm ghost" disabled={busy} onClick={() => reply("reject")}>Reject</button>
      </div>
    </div>
  );
}

function QuestionAsk({ q, onDone }: { q: Question; onDone: () => void }) {
  const [answers, setAnswers] = useState<string[][]>(q.questions.map(() => []));
  const [other, setOther] = useState<string[]>(q.questions.map(() => ""));
  const submit = async (reject = false) => {
    try {
      const final = answers.map((a, i) => (other[i].trim() ? [...a, other[i].trim()] : a));
      await api.post(`/api/session/${q.session_id}/question/${q.id}/reply`, reject ? { reject: true } : { answers: final });
    } catch (e) {
      toastError(e);
    }
    onDone();
  };
  const ready = answers.every((a, i) => a.length || other[i].trim());
  const held = useRef(false);
  const hold = () => {
    if (!q.auto_at || held.current) return;
    held.current = true;
    api.post(`/api/session/${q.session_id}/question/${q.id}/reply`, { hold: true }).then(onDone).catch(() => {});
  };
  const defaults = q.questions.map((x) => x.default).filter(Boolean).join(", ");
  return (
    <div className="ask">
      <AutoCountdown at={q.auto_at} what={defaults} onHold={hold} />
      {q.questions.map((qq, i) => (
        <div key={i} style={{ display: "grid", gap: 8 }}>
          <h3>{qq.question}</h3>
          <div className="opts">
            {qq.options.map((o) => {
              const on = answers[i].includes(o);
              return (
                <button key={o} className={`opt ${on ? "on" : ""}`} onClick={() => { hold(); setAnswers((a) => a.map((x, j) =>
                  j !== i ? x : qq.multiple ? (on ? x.filter((y) => y !== o) : [...x, o]) : [o])); }}>
                  {o}{qq.default === o ? <span className="faint"> · suggested</span> : null}</button>
              );
            })}
          </div>
          <input className="input" placeholder="or type your own answer" value={other[i]}
            onChange={(e) => { hold(); setOther((a) => a.map((x, j) => (j === i ? e.target.value : x))); }}
            onKeyDown={(e) => { if (e.key === "Enter" && ready) submit(); }} />
        </div>
      ))}
      <div className="row">
        <button className="btn primary sm" disabled={!ready} onClick={() => submit()}>Answer</button>
        <button className="btn ghost sm" onClick={() => submit(true)}>Dismiss</button>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ composer

function Composer({ onSend, model, setModel, busy, placeholder, commands }: {
  onSend: (t: string) => Promise<void> | void; model: string; setModel: (m: string) => void; busy?: boolean;
  placeholder?: string; commands: Command[];
}) {
  const { models, defaultModel } = useApp();
  const [text, setText] = useState("");
  const [sending, setSending] = useState(false);
  const [pick, setPick] = useState(0);
  const ta = useRef<HTMLTextAreaElement>(null);

  useLayoutEffect(() => {
    const el = ta.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(220, el.scrollHeight) + "px";
  }, [text]);

  const submitText = async (raw: string) => {
    const t = raw.trim();
    if (!t || sending) return;
    setSending(true);
    try {
      await onSend(t);
      setText("");
    } catch (e) {
      toastError(e);
    } finally {
      setSending(false);
    }
  };

  const latest = useRef(submitText);
  latest.current = submitText;
  useEffect(() => {
    fillTarget = (t, submit) => {
      if (submit) latest.current(t);
      else {
        setText(t);
        requestAnimationFrame(() => {
          ta.current?.focus();
          ta.current?.setSelectionRange(t.length, t.length);
        });
      }
    };
    return () => { fillTarget = null; };
  }, []);

  const slash = text.startsWith("/") && !text.includes(" ") ? commands.filter((c) => c.name.startsWith(text.slice(1))) : [];

  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (slash.length) {
      if (e.key === "ArrowDown") { setPick((p) => (p + 1) % slash.length); e.preventDefault(); return; }
      if (e.key === "ArrowUp") { setPick((p) => (p - 1 + slash.length) % slash.length); e.preventDefault(); return; }
      if (e.key === "Tab" || (e.key === "Enter" && !e.shiftKey)) {
        setText(`/${slash[Math.min(pick, slash.length - 1)].name} `);
        e.preventDefault();
        return;
      }
    }
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submitText(text);
    }
  };

  const groups = useMemo(() => {
    const g: Record<string, typeof models> = {};
    for (const m of models) if (m.connected || m.ref === model) (g[m.provider_name] ||= []).push(m);
    return g;
  }, [models, model]);

  return (
    <div style={{ position: "relative" }}>
      {slash.length > 0 && (
        <div className="slash">
          {slash.map((c, i) => (
            <div key={c.name} onMouseDown={(e) => { e.preventDefault(); setText(`/${c.name} `); ta.current?.focus(); }}
              className={i === pick ? "on" : ""}>
              <span className="mono">/{c.name}</span> <span className="muted small">{c.description}</span>
            </div>
          ))}
        </div>
      )}
      <div className="composer">
        <span className="prompt-mark">&gt;</span>
        <textarea id="composer" ref={ta} value={text} rows={1} placeholder={placeholder}
          onChange={(e) => { setText(e.target.value); setPick(0); }} onKeyDown={onKey} />
        <div className="composer-bar">
          <select className="select" value={model} onChange={(e) => setModel(e.target.value)} title="The model the agent runs on">
            <option value="">{defaultModel ? `${defaultModel}` : "default model"}</option>
            {Object.entries(groups).map(([g, ms]) => (
              <optgroup key={g} label={g}>
                {ms.map((m) => <option key={m.ref} value={m.ref}>{m.name}</option>)}
              </optgroup>
            ))}
          </select>
          <span className="grow" />
          <button className="btn primary sm" onClick={() => submitText(text)} disabled={!text.trim() || sending}
            title={busy ? "Send now (steers the running turn)" : "Send"}>
            {sending ? <span className="spinner" /> : null} {busy ? "steer" : "send ↵"}
          </button>
        </div>
      </div>
    </div>
  );
}
