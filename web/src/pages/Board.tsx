import { useEffect, useState } from "react";
import { api, Board as BoardT, ScoreRow, Stage, StageKey, TrainRun } from "../api";
import { go, useApp } from "../App";
import { ago, dur, Empty, Icon, money, pct, toast, toastError } from "../ui";
import { fillComposer } from "./Agent";
import { DatasetView, GraderModal, NewDataset } from "./Data";
import { RunEvalModal } from "./Evals";
import { Playground, PromoteModal } from "./Models";

// ------------------------------------------------------------------ the strip

export function StageStrip({ board, active }: { board: BoardT; active: StageKey }) {
  const asks = new Set(board.proposals.map((p) => p.stage).filter(Boolean));
  return (
    <nav className="strip">
      {board.stages.map((s, i) => (
        <button key={s.key} className={`st ${s.state} ${active === s.key ? "on" : ""} ${s.current ? "cur" : ""}`}
          onClick={() => go("s", s.key)} title={s.summary}>
          <span className="k">
            <span className="n">{i + 1}</span> {s.label}
            {asks.has(s.key) && <span className="dot waiting" title="a step waits for your OK" />}
          </span>
          <span className="v">{glyph(s)} {s.summary}</span>
        </button>
      ))}
    </nav>
  );
}

function glyph(s: Stage) {
  if (s.state === "done") return <span className="ok">✓</span>;
  if (s.state === "running") return <span className="dot busy" />;
  return s.current ? <span className="cur-g">→</span> : <span className="faint">·</span>;
}

// ------------------------------------------------------------------ the page

export default function BoardPage({ stage }: { stage?: string }) {
  const { board } = useApp();
  if (!board) return <div className="main-pad row muted"><span className="spinner" /> loading…</div>;
  // no stage picked: show where the action is (a step waiting for you, then a running stage, then the next one)
  const waiting = board.proposals.find((p) => p.stage && board.stages.some((s) => s.key === p.stage))?.stage;
  const key = (stage && board.stages.some((s) => s.key === stage) ? stage : waiting || board.current) as StageKey;
  return (
    <div className="stage-view">
      {key === "goal" && <GoalStage board={board} />}
      {key === "data" && <DataStage board={board} />}
      {key === "grade" && <GradeStage board={board} />}
      {key === "baseline" && <BaselineStage board={board} />}
      {key === "train" && <TrainStage board={board} />}
      {key === "prove" && <ProveStage board={board} />}
      {key === "ship" && <ShipStage board={board} />}
    </div>
  );
}

function Head({ title, children, actions }: { title: string; children?: React.ReactNode; actions?: React.ReactNode }) {
  return (
    <div className="stage-head">
      <div className="grow">
        <h1>{title}</h1>
        {children && <p>{children}</p>}
      </div>
      {actions && <div className="actions">{actions}</div>}
    </div>
  );
}

function Ask({ text, children, primary }: { text: string; children: React.ReactNode; primary?: boolean }) {
  return <button className={`btn ${primary ? "primary" : ""}`} onClick={() => fillComposer(text, true)}>{children}</button>;
}

// ------------------------------------------------------------------ goal

const STARTERS = [
  { t: "Fine-tune on my data", s: "a file of prompts and answers", text: "I have a file of examples (I'll drop it in). Fine-tune a small open model on it and show me whether it beats what I use today." },
  { t: "Distil a big model into a small one", s: "a teacher writes the answers", text: "Distil a frontier model into a small open model for this task: " },
  { t: "Which model is best for this?", s: "a baseline, nothing trained", text: "Compare a few models on my examples and tell me which is best and what it costs: " },
  { t: "Start from a public dataset", s: "import, grade, baseline", text: "Use the Hugging Face dataset openai/gsm8k: grade by the final number, get a baseline, then plan a fine-tune." },
  { t: "Build the data with me", s: "from a description", text: "Help me build a dataset for this from scratch, then train on it: " },
  { t: "Train on my coding sessions", s: "Claude Code sessions → verified tasks", text: "Train a model on my Claude Code sessions in this repo so it does my routine coding work faster." },
];

function GoalStage({ board }: { board: BoardT }) {
  const goalSet = board.stages[0].state === "done";
  const [editing, setEditing] = useState(false);
  const [text, setText] = useState(board.goal);
  const save = async () => {
    try {
      const pol = await api.get<{ text: string }>("/api/policy");
      const lines = pol.text.split("\n");
      const i = lines.findIndex((l) => /^goal\s/.test(l.trim()) || l.trim() === "goal");
      const line = `goal         ${text.trim().replace(/\s+/g, " ")}`;
      if (i >= 0) lines[i] = line; else lines.unshift(line);
      const r = await fetch("/api/policy", { method: "POST", credentials: "same-origin", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: lines.join("\n") }) });
      if (!r.ok) throw new Error((await r.json()).error?.message || "could not save");
      setEditing(false);
      toast("goal saved");
    } catch (e) { toastError(e); }
  };
  const { sessions } = useApp();
  const fresh = !board.datasets.length && !sessions.length;
  if (!goalSet && !editing && fresh) {
    return (
      <div className="hero">
        <h1>What should your model get better at?</h1>
        <p>Say it in your own words, in the box on the right. Or start from one of these:</p>
        <div className="starters">
          {STARTERS.map((s) => (
            <button key={s.t} className="starter" onClick={() => fillComposer(s.text)}>
              <b>{s.t}</b><span>{s.s}</span>
            </button>
          ))}
        </div>
        <p className="faint small" style={{ marginTop: 18 }}>Or drop a file of examples anywhere on this page.</p>
      </div>
    );
  }
  return (
    <>
      <Head title="Goal" actions={!editing && goalSet && <button className="btn" onClick={() => { setText(board.goal); setEditing(true); }}><Icon name="pen" /> edit</button>}>
        What the model should get better at. Every score on this board is measured against it.
      </Head>
      {!goalSet && !editing && (
        <div className="card card-pad" style={{ display: "grid", gap: 10 }}>
          <div className="muted">No goal written down yet. One sentence is enough: what should the model do better, and for whom?</div>
          <div className="row">
            <button className="btn primary" onClick={() => { setText(""); setEditing(true); }}><Icon name="pen" /> write it</button>
            <Ask text="Write my goal into the Trainfile from what we've discussed, in my words. One sentence.">let the agent write it from our conversation</Ask>
          </div>
        </div>
      )}
      {editing ? (
        <div className="card card-pad" style={{ display: "grid", gap: 10 }}>
          <textarea className="textarea" autoFocus value={text} onChange={(e) => setText(e.target.value)} placeholder="e.g. sort support tickets into billing, bug, account and feature" />
          <div className="row"><span className="grow" /><button className="btn ghost" onClick={() => setEditing(false)}>cancel</button><button className="btn primary" disabled={!text.trim()} onClick={save}>save</button></div>
        </div>
      ) : goalSet ? <div className="goal-text">“{board.goal}”</div> : null}
      <div className="facts">
        <div><span>budget</span><b>{board.budget.text}</b><i>{money(board.budget.spent_usd)} spent</i></div>
        <div><span>base model</span><b className="mono">{board.policy.base_model}</b><i>what training starts from</i></div>
        <div><span>trainer</span><b>{board.policy.trainer}</b><i><a href="#/settings">change</a></i></div>
        <div><span>ships when</span><b>it beats the baseline</b><i>on held-out rows, no regressions</i></div>
      </div>
      {!board.dataset && (
        <div className="hint-row">
          <span>Next: examples of the behaviour.</span>
          <button className="btn" onClick={() => go("s", "data")}>go to data →</button>
        </div>
      )}
    </>
  );
}

// ------------------------------------------------------------------ data

function DataStage({ board }: { board: BoardT }) {
  const { setDataset } = useApp();
  const [adding, setAdding] = useState<"file" | "hf" | null>(null);
  if (!board.datasets.length) {
    return (
      <>
        <Head title="Data">Examples of what the model should do. A share of every dataset is held out and sealed from training, so scores mean something.</Head>
        <div className="ways">
          <button className="way" onClick={() => setAdding("file")}><Icon name="upload" /><b>Upload a file</b><span>JSONL, CSV or JSON; prompts and answers, or chat messages</span></button>
          <button className="way" onClick={() => setAdding("hf")}><Icon name="hf" /><b>Hugging Face</b><span>any public dataset by name</span></button>
          <button className="way" onClick={() => fillComposer("Help me build a dataset for my goal: write varied, realistic examples with reference answers, about 40 to start. Ask me anything you need first.")}>
            <Icon name="pen" /><b>Write it with the agent</b><span>from a description, documents or a few examples</span></button>
        </div>
        {adding && <NewDataset tab={adding} onClose={() => setAdding(null)} onCreated={(n) => { setAdding(null); setDataset(n); }} />}
      </>
    );
  }
  return (
    <>
      {board.datasets.length > 1 && (
        <div className="chips">
          {board.datasets.map((d) => (
            <button key={d.name} className={`chip ${d.name === board.dataset ? "on" : ""}`} onClick={() => setDataset(d.name)}>{d.name}</button>
          ))}
          <button className="chip" onClick={() => setAdding("file")}>+ new</button>
        </div>
      )}
      {board.dataset && <DatasetView name={board.dataset} onNew={() => setAdding("file")} />}
      {adding && <NewDataset tab={adding} onClose={() => setAdding(null)} onCreated={(n) => { setAdding(null); setDataset(n); }} />}
    </>
  );
}

// ------------------------------------------------------------------ grade

const GRADER_HELP: [string, string][] = [
  ["exact", "one right answer, word for word (labels, ids)"],
  ["contains / regex", "the answer must include or match something"],
  ["number", "maths and counts: the final number is compared"],
  ["json", "structured output: parses and matches the fields"],
  ["judge", "open answers: a model scores against a rubric, 0 to 10"],
  ["command", "a script you trust decides (tests, validators)"],
];

function GradeStage({ board }: { board: BoardT }) {
  const [open, setOpen] = useState(false);
  const [detail, setDetail] = useState<any>(null);
  useEffect(() => {
    if (board.dataset) api.get<any>(`/api/dataset/${board.dataset}?limit=3`).then(setDetail).catch(() => {});
  }, [board.dataset]);
  if (!board.dataset) return <><Head title="Grade" /><Empty title="No data yet">Grading needs examples first.</Empty></>;
  return (
    <>
      <Head title="Grade" actions={<>
        <Ask text={`Look at the dataset ${board.dataset} and tell me if its grader is the right one; propose a better one if not.`}>ask the agent</Ask>
        <button className="btn primary" onClick={() => setOpen(true)}>change grader</button></>}>
        How an answer is judged right or wrong. Every model on the board is scored the same way.
      </Head>
      <div className="facts">
        <div><span>grader</span><b>{board.focus?.grader}</b><i>on <span className="mono">{board.dataset}</span></i></div>
        <div style={{ gridColumn: "span 3" }}><span>system prompt</span><b className="sys">{detail?.system || <span className="faint">none</span>}</b><i>sent before every prompt, in evals and training</i></div>
      </div>
      {detail?.rows?.length > 0 && (
        <div className="card" style={{ marginTop: 16 }}>
          <div className="card-head"><h3>what a right answer looks like</h3></div>
          <table className="table"><tbody>
            {detail.rows.map((r: any) => {
              const user = r.messages.filter((m: any) => m.role === "user").map((m: any) => m.content).join(" ⏎ ");
              const last = r.messages[r.messages.length - 1];
              return <tr key={r.id}><td className="clip" style={{ width: "65%" }}>{user}</td><td className="ok clip">{last?.role === "assistant" ? last.content : "—"}</td></tr>;
            })}
          </tbody></table>
        </div>
      )}
      <div className="grader-help">
        {GRADER_HELP.map(([k, v]) => <div key={k}><b>{k}</b><span>{v}</span></div>)}
      </div>
      {open && detail && <GraderModal name={board.dataset} spec={detail.grader_spec} system={detail.system} description={detail.description || ""}
        onClose={() => setOpen(false)} onSaved={() => setOpen(false)} />}
    </>
  );
}

// ------------------------------------------------------------------ scoreboard

export function Scoreboard({ rows, dataset, highlight }: { rows: ScoreRow[]; dataset: string; highlight?: string | null }) {
  const n = rows[0]?.rows ?? 0;
  return (
    <div className="card">
      <div className="card-head">
        <h3>scoreboard · {n} held-out rows no model trained on</h3>
        <a className="small" href={`#/grid/${encodeURIComponent(dataset)}`}>every row →</a>
      </div>
      <div className="sb">
        <div className="sb-h"><span>model</span><span /><span className="num">score</span><span className="num">passed</span><span className="num">per 1k</span><span className="num">latency</span></div>
        {rows.map((r) => (
          <div key={r.target} className={`sb-r ${r.target === highlight ? "hl" : ""}`} onClick={() => go("evals", r.eval)} title="open this eval">
            <span className="mono name">{r.target}{r.live && <span className="badge good">live</span>}{r.kind === "version" && !r.live && <span className="badge">trained</span>}</span>
            <span className="track"><i className={r.live ? "live" : r.kind === "version" ? "ver" : ""} style={{ width: `${r.score * 100}%` }} /></span>
            <span className="num"><b>{pct(r.score)}</b></span>
            <span className="num faint">{r.passed}/{r.rows}</span>
            <span className="num faint">{r.cost_per_1k ? money(r.cost_per_1k) : "$0"}</span>
            <span className="num faint">{r.median_seconds ? dur(r.median_seconds) : "—"}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function BaselineStage({ board }: { board: BoardT }) {
  const [run, setRun] = useState(false);
  if (!board.dataset) return <><Head title="Baseline" /><Empty title="No data yet">A baseline needs held-out examples.</Empty></>;
  const base = board.policy.base_model;
  return (
    <>
      <Head title="Baseline" actions={<>
        <Ask text={`Measure a baseline on ${board.dataset}: the base model ${base} and one strong frontier model, on the held-out rows.`}>ask the agent</Ask>
        <button className="btn primary" onClick={() => setRun(true)}><Icon name="play" /> evaluate a model</button></>}>
        The number to beat. Models scored on held-out rows that no model ever trains on.
      </Head>
      {board.scoreboard.length ? <Scoreboard rows={board.scoreboard} dataset={board.dataset} /> : (
        <Empty title="No scores yet" action={<Ask primary text={`Measure a baseline on ${board.dataset}: the base model ${base} and one strong frontier model, on the held-out rows.`}>measure a baseline</Ask>}>
          Score the model you use today, a frontier model and the base model ({base}). Sometimes the baseline alone answers the question.
        </Empty>
      )}
      <RunningFor board={board} kind="eval" />
      {run && <RunEvalModal dataset={board.dataset} onClose={() => setRun(false)} />}
    </>
  );
}

function RunningFor({ board, kind }: { board: BoardT; kind: string }) {
  const rs = board.running.filter((j) => j.kind === kind);
  if (!rs.length) return null;
  return (
    <div className="running">
      {rs.map((j) => (
        <div key={j.id} className="run-row" onClick={() => go("jobs", j.id)}>
          <span className="dot busy" /><span className="grow">{j.title}</span>
          <span className="faint">{j.detail}</span>
          {j.progress != null && <span className="mono">{Math.round(j.progress * 100)}%</span>}
        </div>
      ))}
    </div>
  );
}

// ------------------------------------------------------------------ train

export function Curve({ run, height = 110 }: { run: TrainRun; height?: number }) {
  const pts = run.curve || [];
  if (pts.length < 2) {
    return <div className="curve-empty">{run.state === "running" ? "the loss curve appears once the trainer reports steps" : "no loss reported"}</div>;
  }
  const W = 600, H = height, pad = 4;
  const ys = pts.map((p) => p[1]);
  const lo = Math.min(...ys), hi = Math.max(...ys);
  const x = (v: number) => pad + v * (W - 2 * pad);
  const y = (v: number) => pad + (hi === lo ? 0.5 : (hi - v) / (hi - lo)) * (H - 2 * pad);
  const d = pts.map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join(" ");
  return (
    <svg className="curve" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ height }}>
      <line x1="0" x2={W} y1={H - 1} y2={H - 1} className="axis" />
      <path d={d} className={run.state === "running" ? "run" : ""} />
    </svg>
  );
}

function TrainStage({ board }: { board: BoardT }) {
  if (!board.dataset) return <><Head title="Train" /><Empty title="No data yet">Training needs a dataset with answers.</Empty></>;
  const ds = board.dataset;
  return (
    <>
      <Head title="Train" actions={<Ask primary text={`Train ${board.policy.base_model} on ${ds}. Propose it with the cost first.`}><Icon name="spark" /> train on {ds}</Ask>}>
        Fine-tune <span className="mono">{board.policy.base_model}</span> on <span className="mono">{ds}</span> with {board.policy.trainer}. The agent proposes each run with its cost; nothing uploads or spends until you approve.
      </Head>
      {board.training.length === 0 ? (
        <Empty title="Nothing trained yet">
          {board.scoreboard.length ? <>The best baseline is <b>{pct(board.scoreboard[0].score)}</b> ({board.scoreboard[0].target}). Training is worth it when the base model is cheaper or faster and can learn to match it.</> :
            <>Measure a baseline first, so you know the number training has to beat.</>}
        </Empty>
      ) : board.training.map((r) => (
        <div key={r.id} className="card run-card">
          <div className="card-head">
            <span className={`dot ${r.state === "running" ? "busy" : r.state === "succeeded" ? "good" : r.state === "failed" ? "bad" : ""}`} />
            <h3 style={{ textTransform: "none" }}>{r.title}</h3>
            <span className="small faint">{r.state}{r.version ? ` → ${r.version}` : ""}</span>
            <button className="btn sm ghost" onClick={() => go("jobs", r.id)}>log</button>
          </div>
          <div className="card-pad">
            <Curve run={r} />
            <div className="run-facts">
              {r.progress != null && <span>{Math.round(r.progress * 100)}%</span>}
              {r.loss != null && <span>loss {r.loss.toFixed(3)}</span>}
              <span>{money(r.cost_usd || 0)}{r.estimate_usd ? ` of ~${money(r.estimate_usd)}` : ""}</span>
              {r.time?.started && <span>{ago(r.time.started)}</span>}
              {r.detail && <span className="grow clip-text">{r.detail}</span>}
            </div>
            {r.error && r.state !== "succeeded" && <div className="err small" style={{ marginTop: 6 }}>{r.error}</div>}
          </div>
        </div>
      ))}
      {board.versions.length > 0 && (
        <div className="small faint" style={{ marginTop: 12 }}>versions trained on {ds}: {board.versions.map((v) => v.version).join(", ")} · <a href="#/s/prove">prove them →</a></div>
      )}
    </>
  );
}

// ------------------------------------------------------------------ prove

function ProveStage({ board }: { board: BoardT }) {
  const [evalFor, setEvalFor] = useState<string | null>(null);
  if (!board.dataset) return <><Head title="Prove" /><Empty title="No data yet" /></>;
  const versions = board.versions;
  const scored = new Set(board.scoreboard.map((r) => r.target));
  const g = board.gate;
  return (
    <>
      <Head title="Prove" actions={versions.length > 0 && !scored.has(versions[versions.length - 1].version) &&
        <Ask primary text={`Evaluate ${versions[versions.length - 1].version} on ${board.dataset}'s held-out rows and compare it with the baseline row by row.`}>prove {versions[versions.length - 1].version}</Ask>}>
        A trained version against the baseline, on the same held-out rows: what it fixed, what it broke.
      </Head>
      {!versions.length ? <Empty title="Nothing to prove yet">Train a version first.</Empty> : (
        <>
          {g && g.rows !== undefined && (
            <div className={`gate ${g.promote ? "pass" : "fail"}`}>
              <div className="gate-h">
                <b className="mono">{g.candidate}</b> <span className="faint">vs</span> <b className="mono">{g.baseline}</b>
                <span className="grow" />
                <span className={`badge ${g.promote ? "good" : "bad"}`}>{g.promote ? "beats the baseline" : "does not pass the gate"}</span>
              </div>
              <div className="gate-nums">
                <div><b>{pct(g.candidate_score)}</b><span>{g.candidate}</span></div>
                <div><b>{pct(g.baseline_score)}</b><span>{g.baseline}</span></div>
                <div><b className="ok">+{g.fixed?.length ?? 0}</b><span>rows fixed</span></div>
                <div><b className={g.regressed?.length ? "err" : ""}>−{g.regressed?.length ?? 0}</b><span>rows broken</span></div>
                <div><b>{g.rows}</b><span>rows compared</span></div>
              </div>
              {!g.promote && g.reasons.length > 0 && <ul className="reasons">{g.reasons.map((r) => <li key={r}>{r}</li>)}</ul>}
              <div className="row" style={{ marginTop: 10 }}>
                <a className="btn sm" href={`#/grid/${encodeURIComponent(board.dataset)}`}>see the rows →</a>
                {g.promote && <button className="btn sm primary" onClick={() => go("s", "ship")}>ship it →</button>}
              </div>
            </div>
          )}
          <div className="card" style={{ marginTop: 14 }}>
            <div className="card-head"><h3>versions on {board.dataset}</h3></div>
            <table className="table">
              <thead><tr><th>version</th><th>base</th><th className="num">held-out</th><th>gate</th><th>trained</th><th /></tr></thead>
              <tbody>
                {[...versions].reverse().map((v) => {
                  const s = board.scoreboard.find((r) => r.target === v.version);
                  return (
                    <tr key={v.version}>
                      <td className="mono"><b>{v.version}</b>{v.live && <span className="badge good" style={{ marginLeft: 6 }}>live</span>}</td>
                      <td className="mono small faint">{v.base_model}</td>
                      <td className="num">{s ? <b>{pct(s.score)}</b> : <span className="faint">not scored</span>}</td>
                      <td className="small">{v.decision || "—"}</td>
                      <td className="faint small">{ago(v.created_at)}</td>
                      <td className="num">{!s && <button className="btn sm" onClick={() => setEvalFor(v.version)}>evaluate</button>}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {board.scoreboard.length > 0 && <div style={{ marginTop: 14 }}><Scoreboard rows={board.scoreboard} dataset={board.dataset} highlight={versions[versions.length - 1]?.version} /></div>}
        </>
      )}
      <RunningFor board={board} kind="eval" />
      {evalFor && <RunEvalModal dataset={board.dataset} model={evalFor} onClose={() => setEvalFor(null)} />}
    </>
  );
}

// ------------------------------------------------------------------ ship

function ShipStage({ board }: { board: BoardT }) {
  const { reloadBoard } = useApp();
  const [promote, setPromote] = useState<any>(null);
  const live = board.live.version;
  const candidates = board.versions.filter((v) => !v.live);
  return (
    <>
      <Head title="Ship">What answers when you call your model. A version goes live only through the gate.</Head>
      <div className="facts">
        <div><span>live</span><b className="mono">{live || "nothing yet"}</b><i>{live ? `trained on ${board.live.dataset || "coding tasks"}` : "train and prove a version first"}</i></div>
        <div style={{ gridColumn: "span 3" }}><span>use it</span><b className="mono sys">opentrain model serve</b><i>an OpenAI-compatible endpoint on this machine; or <span className="mono">opentrain model connect</span> for a coding agent</i></div>
      </div>
      {candidates.length > 0 && (
        <div className="card" style={{ marginTop: 14 }}>
          <div className="card-head"><h3>ready to go live?</h3></div>
          <table className="table"><tbody>
            {[...candidates].reverse().map((v) => (
              <tr key={v.version}>
                <td className="mono"><b>{v.version}</b></td>
                <td className="small faint">{v.base_model}</td>
                <td className="small">{v.decision || "not gated"}</td>
                <td className="num"><button className="btn sm primary" onClick={() => setPromote(v)}>run the gate</button></td>
              </tr>
            ))}
          </tbody></table>
        </div>
      )}
      <div style={{ marginTop: 14 }}><Playground versions={board.versions as any} /></div>
      {promote && <PromoteModal v={{ ...promote, dataset: board.dataset } as any} current={live} onClose={() => setPromote(null)} onDone={() => { setPromote(null); reloadBoard(); }} />}
    </>
  );
}

// ------------------------------------------------------------------ drop a file anywhere

export function DropZone() {
  const { setDataset } = useApp();
  const [over, setOver] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  useEffect(() => {
    let depth = 0;
    const hasFile = (e: DragEvent) => !!e.dataTransfer && [...e.dataTransfer.types].includes("Files");
    const enter = (e: DragEvent) => { if (hasFile(e)) { depth++; setOver(true); } };
    const leave = (e: DragEvent) => { if (hasFile(e)) { depth = Math.max(0, depth - 1); if (!depth) setOver(false); } };
    const overF = (e: DragEvent) => { if (hasFile(e)) e.preventDefault(); };
    const drop = (e: DragEvent) => {
      if (!hasFile(e)) return;
      e.preventDefault();
      depth = 0;
      setOver(false);
      if (document.querySelector(".modal")) return; // a dialog handles its own drops
      const f = e.dataTransfer?.files?.[0];
      if (f) setFile(f);
    };
    window.addEventListener("dragenter", enter);
    window.addEventListener("dragleave", leave);
    window.addEventListener("dragover", overF);
    window.addEventListener("drop", drop);
    return () => {
      window.removeEventListener("dragenter", enter);
      window.removeEventListener("dragleave", leave);
      window.removeEventListener("dragover", overF);
      window.removeEventListener("drop", drop);
    };
  }, []);
  return (
    <>
      {over && !document.querySelector(".modal") && <div className="dropping"><div>drop to make a dataset</div></div>}
      {file && <NewDataset file={file} onClose={() => setFile(null)} onCreated={(n) => { setFile(null); setDataset(n); go("s", "data"); }} />}
    </>
  );
}
