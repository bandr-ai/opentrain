import { KeyboardEvent, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { api, events, Job, Message, Part, Permission, Question, Session } from "../api";
import { go, useApp } from "../App";
import { dur, Icon, Markdown, money, Progress, toastError } from "../ui";

// ------------------------------------------------------------------ helpers

function merge<T extends Record<string, any>>(base: T, patch: Record<string, any>): T {
  const out: Record<string, any> = { ...base };
  for (const [k, v] of Object.entries(patch)) {
    if (v === null) delete out[k];
    else if (v && typeof v === "object" && !Array.isArray(v) && out[k] && typeof out[k] === "object" && !Array.isArray(out[k]))
      out[k] = merge(out[k], v);
    else out[k] = v;
  }
  return out as T;
}

const STARTERS = [
  { title: "Fine-tune on my data", text: "I have a dataset of examples (I'll give you the file). Fine-tune a small open model on it and show me if it beats what I use today.", hint: "JSONL, CSV or JSON: prompts and answers" },
  { title: "Distil a big model into a small one", text: "Distil a frontier model into a small open model for this task: ", hint: "a teacher writes the answers, a small model learns them" },
  { title: "Compare models on my examples", text: "Compare a few models on my examples and tell me which is best and what it costs: ", hint: "a baseline eval; nothing is trained" },
  { title: "Start from a public dataset", text: "Use the Hugging Face dataset openai/gsm8k: set it up with a number grader, get a baseline from a frontier model, and plan a fine-tune.", hint: "import, grade, baseline, plan" },
  { title: "Build a dataset from a description", text: "Help me build a dataset for this from scratch, then train on it: ", hint: "we write and check the examples together" },
  { title: "Train on my coding sessions", text: "Train a model on my Claude Code sessions in this repo so it does my routine coding work faster.", hint: "mines your agent sessions into verified tasks" },
];

type Command = { name: string; description: string };

// ------------------------------------------------------------------ the page

export default function Chat({ sid }: { sid?: string }) {
  const isNew = !sid || sid === "new";
  return isNew ? <Welcome /> : <Conversation key={sid} sid={sid!} />;
}

function useCommands() {
  const [cmds, setCmds] = useState<Command[]>([]);
  useEffect(() => {
    api.get<Command[]>("/api/command").then(setCmds).catch(() => {});
  }, []);
  return cmds;
}

function Welcome() {
  const { reloadSessions, status } = useApp();
  const [agent, setAgent] = useState("train");
  const [model, setModel] = useState("");
  const commands = useCommands();

  const start = async (text: string) => {
    const s = await api.post<Session>("/api/session", { agent, ...(model ? { model } : {}) });
    await send(s.id, text, commands);
    reloadSessions();
    go("chat", s.id);
  };

  return (
    <div className="chat">
      <div className="scroll">
        <div className="welcome">
          <h1>What should your model get better at?</h1>
          <p>
            Describe it in your own words. OpenTrain gets the data, decides how to grade answers, measures a baseline,
            trains, and only ships a version that wins on examples it has never seen.
            {status?.policy && <> Budget: <b>{status.policy.budget}</b>.</>}
          </p>
          <Composer onSend={start} agent={agent} setAgent={setAgent} model={model} setModel={setModel}
            autoFocus commands={commands} placeholder="e.g. Make a small model that drafts our support replies, from tickets.csv" />
          <div className="starters">
            {STARTERS.map((s) => (
              <button key={s.title} className="starter" onClick={() => {
                const ta = document.querySelector<HTMLTextAreaElement>(".composer textarea");
                if (ta) {
                  const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")!.set!;
                  setter.call(ta, s.text);
                  ta.dispatchEvent(new Event("input", { bubbles: true }));
                  ta.focus();
                  ta.setSelectionRange(s.text.length, s.text.length);
                }
              }}>
                <b>{s.title}</b>
                <span>{s.hint}</span>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

async function send(sid: string, text: string, commands: Command[]) {
  const m = text.match(/^\/(\S+)\s*([\s\S]*)$/);
  if (m && commands.some((c) => c.name === m[1])) {
    await api.post(`/api/session/${sid}/command`, { name: m[1], arguments: m[2] });
  } else {
    await api.post(`/api/session/${sid}/prompt`, { text });
  }
}

function Conversation({ sid }: { sid: string }) {
  const { busy, jobs, reloadSessions } = useApp();
  const [session, setSession] = useState<Session | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [perms, setPerms] = useState<Permission[]>([]);
  const [questions, setQuestions] = useState<Question[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [agent, setAgent] = useState("train");
  const [model, setModel] = useState("");
  const commands = useCommands();
  const scroller = useRef<HTMLDivElement>(null);
  const stick = useRef(true);

  const loadAsks = useCallback(() => {
    api.get<Permission[]>(`/api/permission?session=${sid}`).then(setPerms).catch(() => {});
    api.get<Question[]>(`/api/question?session=${sid}`).then(setQuestions).catch(() => {});
  }, [sid]);

  useEffect(() => {
    api.get<Session>(`/api/session/${sid}`).then((s) => {
      setSession(s);
      setAgent(s.agent || "train");
      setModel(typeof s.model === "string" ? s.model : "");
    }).catch((e) => setLoadError(e.message));
    api.get<Message[]>(`/api/session/${sid}/message`).then(setMessages).catch(() => {});
    loadAsks();
  }, [sid, loadAsks]);

  useEffect(() => events.on((e) => {
    const d = e.data || {};
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
    if (e.type.startsWith("permission.") || e.type.startsWith("question.")) loadAsks();
  }), [sid, loadAsks]);

  useLayoutEffect(() => {
    const el = scroller.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  });

  const state = busy[sid] || (session as any)?.status || "idle";
  const isBusy = state !== "idle";
  const sessionJobs = useMemo(() => jobs.filter((j) => j.session_id === sid), [jobs, sid]);
  const jobById = useMemo(() => Object.fromEntries(jobs.map((j) => [j.id, j])), [jobs]);
  const shownJobs = new Set<string>();

  const onSend = async (text: string) => {
    try {
      if (agent !== session?.agent || (model && model !== session?.model)) {
        await api.patch(`/api/session/${sid}`, { agent, ...(model ? { model } : {}) });
      }
      stick.current = true;
      await send(sid, text, commands);
      reloadSessions();
    } catch (e) {
      toastError(e);
    }
  };

  if (loadError) return <div className="empty"><h2>Conversation not found</h2><p>{loadError}</p></div>;

  return (
    <div className="chat">
      <div className="chat-head">
        <h2>{session?.title || "New conversation"}</h2>
        <span className={`status-pill ${isBusy ? (state === "waiting" ? "waiting" : "busy") : ""}`}>
          <i /> {state === "waiting" ? "waiting for you" : state === "idle" ? "idle" : state}
        </span>
        {isBusy && (
          <button className="btn sm" onClick={() => api.post(`/api/session/${sid}/interrupt`).catch(toastError)}>
            <Icon name="stop" /> Stop
          </button>
        )}
      </div>
      <div className="scroll" ref={scroller} onScroll={(e) => {
        const el = e.currentTarget;
        stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
      }}>
        <div className="convo">
          {messages.map((m) => (
            <MessageView key={m.id} m={m} jobById={jobById} shownJobs={shownJobs} />
          ))}
          {sessionJobs.filter((j) => j.state === "running" && !shownJobs.has(j.id)).map((j) => <JobCard key={j.id} job={j} />)}
          {perms.map((p) => <PermissionAsk key={p.id} p={p} onDone={loadAsks} />)}
          {questions.map((q) => <QuestionAsk key={q.id} q={q} onDone={loadAsks} />)}
          {isBusy && state !== "waiting" && !messages.some((m) => m.role === "assistant" && !m.finish && m.parts.length) && (
            <div className="row muted small"><span className="spinner" /> thinking…</div>
          )}
        </div>
      </div>
      <div className="composer-wrap">
        <Composer onSend={onSend} agent={agent} setAgent={setAgent} model={model} setModel={setModel}
          busy={isBusy} commands={commands} placeholder={isBusy ? "Steer the agent while it works…" : "Reply…"} autoFocus />
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ messages

function MessageView({ m, jobById, shownJobs }: { m: Message; jobById: Record<string, Job>; shownJobs: Set<string> }) {
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
          const jid = p.tool.metadata?.job_id;
          const job = jid ? jobById[jid] : undefined;
          if (job) shownJobs.add(job.id);
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
          <span className="agent">{m.agent}</span>
          {m.model && <span>{m.model}</span>}
          {took !== null && <span>{dur(took)}</span>}
          {m.cost ? <span>{money(m.cost)}{m.billing === "subscription" ? " (plan)" : ""}</span> : null}
          {m.finish === "interrupted" && <span className="badge warn">interrupted</span>}
          {m.error && m.finish !== "interrupted" && <span className="err">{m.error}</span>}
        </div>
      )}
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
};

function summary(t: NonNullable<Part["tool"]>) {
  if (t.title) return t.title;
  const a = t.input || {};
  for (const k of ["name", "model", "command", "path", "hf", "pattern", "url", "version", "job_id", "prompt"]) {
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
        <span className="name">{TOOL_LABELS[tool.name] || tool.name}</span>
        <span className="sum">{summary(tool)}</span>
        {took !== null && <span className="t">{dur(took)}</span>}
        <Icon name={open ? "down" : "chevron"} className="faint" />
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
        <div className="tool-body"><pre className="err">{tool.error.slice(0, 400)}</pre></div>
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
          <button className="btn ghost sm" onClick={() => api.post(`/api/job/${job.id}/cancel`).catch(toastError)}>Cancel</button>
        )}
        <button className="btn ghost sm" onClick={() => go("jobs", job.id)}>Log</button>
      </div>
      {job.state === "running" && <Progress value={job.progress} />}
      <div className="small muted">
        {job.state !== "running" && <span className={job.state === "succeeded" ? "ok" : "err"}>{job.state}. </span>}
        {job.detail}
        {job.cost_usd ? ` · ${money(job.cost_usd)}` : ""}
        {job.estimate_usd ? ` of ~${money(job.estimate_usd)}` : ""}
        {job.version && <> · <a href="#/models">{job.version}</a></>}
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
    case "job.start": return <>start a training job on <b>{r}</b>{md.base_model && <> with {md.base_model}</>}</>;
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
      <h3>The {p.agent} agent wants to {describe(p)}</h3>
      {p.metadata?.command && p.action === "shell" && p.metadata.description && <div className="what muted">{p.metadata.description}</div>}
      <div className="row">
        <button className="btn primary" disabled={busy} onClick={() => reply("once")}>Allow once</button>
        {p.action !== "spend" && <button className="btn" disabled={busy} onClick={() => reply("always")}>Always allow</button>}
        <button className="btn danger" disabled={busy} onClick={() => reply("reject")}>Reject</button>
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
  return (
    <div className="ask">
      {q.questions.map((qq, i) => (
        <div key={i} style={{ display: "grid", gap: 8 }}>
          <h3>{qq.question}</h3>
          <div className="opts">
            {qq.options.map((o) => {
              const on = answers[i].includes(o);
              return (
                <button key={o} className={`opt ${on ? "on" : ""}`} onClick={() => setAnswers((a) => a.map((x, j) =>
                  j !== i ? x : qq.multiple ? (on ? x.filter((y) => y !== o) : [...x, o]) : [o]))}>{o}</button>
              );
            })}
          </div>
          <input className="input" placeholder="Or type your own answer" value={other[i]}
            onChange={(e) => setOther((a) => a.map((x, j) => (j === i ? e.target.value : x)))} />
        </div>
      ))}
      <div className="row">
        <button className="btn primary" disabled={!ready} onClick={() => submit()}>Answer</button>
        <button className="btn ghost" onClick={() => submit(true)}>Dismiss</button>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ composer

function Composer({ onSend, agent, setAgent, model, setModel, busy, autoFocus, placeholder, commands }: {
  onSend: (t: string) => Promise<void> | void; agent: string; setAgent: (a: string) => void; model: string;
  setModel: (m: string) => void; busy?: boolean; autoFocus?: boolean; placeholder?: string; commands: Command[];
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
    el.style.height = Math.min(260, el.scrollHeight) + "px";
  }, [text]);

  const slash = text.startsWith("/") && !text.includes(" ") ? commands.filter((c) => c.name.startsWith(text.slice(1))) : [];

  const submit = async () => {
    const t = text.trim();
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
      submit();
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
        <div className="card" style={{ position: "absolute", bottom: "100%", left: 0, right: 0, marginBottom: 6, padding: 4, boxShadow: "var(--shadow)", zIndex: 5, maxWidth: 820, marginInline: "auto" }}>
          {slash.map((c, i) => (
            <div key={c.name} onMouseDown={(e) => { e.preventDefault(); setText(`/${c.name} `); ta.current?.focus(); }}
              style={{ padding: "6px 10px", borderRadius: 6, cursor: "pointer", background: i === pick ? "var(--sunken)" : undefined }}>
              <span className="mono">/{c.name}</span> <span className="muted small">{c.description}</span>
            </div>
          ))}
        </div>
      )}
      <div className="composer">
        <textarea ref={ta} value={text} autoFocus={autoFocus} rows={1} placeholder={placeholder}
          onChange={(e) => { setText(e.target.value); setPick(0); }} onKeyDown={onKey} />
        <div className="composer-bar">
          <div className="seg" title="train acts (asking before it spends); plan only reads and estimates">
            {["train", "plan"].map((a) => (
              <button key={a} className={agent === a ? "on" : ""} onClick={() => setAgent(a)}>{a}</button>
            ))}
          </div>
          <select className="select" value={model} onChange={(e) => setModel(e.target.value)} title="The model the agent runs on">
            <option value="">{defaultModel ? `default (${defaultModel})` : "default model"}</option>
            {Object.entries(groups).map(([g, ms]) => (
              <optgroup key={g} label={g}>
                {ms.map((m) => <option key={m.ref} value={m.ref}>{m.name}</option>)}
              </optgroup>
            ))}
          </select>
          <span className="grow" />
          <button className="btn primary sm" onClick={submit} disabled={!text.trim() || sending} title={busy ? "Send now (steers the running turn)" : "Send"}>
            {sending ? <span className="spinner" /> : <Icon name="send" />} {busy ? "Steer" : "Send"}
          </button>
        </div>
      </div>
      <div className="composer-hint">Enter to send · Shift+Enter for a new line · / for commands · nothing is spent or uploaded without asking you</div>
    </div>
  );
}
