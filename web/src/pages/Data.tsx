import { fillComposer } from "./Agent";
import { useEffect, useState } from "react";
import { api, DatasetSummary, EvalSummary, Row } from "../api";
import { go, useApp } from "../App";
import { ago, Empty, Icon, Modal, pct, toast, toastError, useAsync } from "../ui";
import { RunEvalModal } from "./Evals";

const NAME_RE = /^[a-z0-9][a-z0-9._-]{0,63}$/;

function slug(s: string) {
  return s.toLowerCase().replace(/\.[a-z]+$/, "").replace(/[^a-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 60);
}

export function NewDataset({ onClose, onCreated, initialName, tab: initialTab, file: initialFile }: {
  onClose: () => void; onCreated: (n: string) => void; initialName?: string; tab?: "file" | "hf" | "agent"; file?: File | null;
}) {
  const [tab, setTab] = useState<"file" | "hf" | "agent">(initialTab || "file");
  const [name, setName] = useState(initialName || (initialFile ? slug(initialFile.name) : ""));
  const [description, setDescription] = useState("");
  const [file, setFile] = useState<File | null>(initialFile || null);
  const [over, setOver] = useState(false);
  const [hf, setHf] = useState("");
  const [split, setSplit] = useState("");
  const [config, setConfig] = useState("");
  const [limit, setLimit] = useState("2000");
  const [inputKey, setInputKey] = useState("");
  const [outputKey, setOutputKey] = useState("");
  const [goal, setGoal] = useState("");
  const [busy, setBusy] = useState(false);
  const { reloadStatus, reloadBoard } = useApp();
  const extending = !!initialName;

  const pickFile = (f: File | null) => {
    setFile(f);
    if (f && !name) setName(slug(f.name));
  };

  const valid = NAME_RE.test(name) && (tab === "file" ? !!file : tab === "hf" ? !!hf.trim() : !!goal.trim());

  const submit = async () => {
    setBusy(true);
    try {
      if (tab === "agent") {
        fillComposer(`Build a dataset named "${name}" for this, writing varied, realistic examples with reference answers (start with about 40, ask me if anything is unclear), then propose how answers should be graded and set it:\n\n${goal}`, true);
        onClose();
        return;
      }
      let r: any;
      const keys = { input_key: inputKey || undefined, output_key: outputKey || undefined, description: description || undefined };
      if (tab === "file" && file) {
        if (file.size > 30 * 1024 * 1024) throw new Error("files up to 30 MB in the browser; for bigger ones ask the agent to import the path");
        r = await api.post(`/api/dataset/${name}/upload`, { filename: file.name, content: await file.text(), ...keys });
      } else {
        r = await api.post(`/api/dataset/${name}/hf`, { hf: hf.trim(), split: split || undefined, config: config || undefined, limit: Number(limit) || 2000, ...keys });
      }
      const notes = [r.duplicates ? `${r.duplicates} duplicates skipped` : "", r.skipped ? `${r.skipped} unreadable rows skipped (${r.errors?.[0] ?? ""})` : ""].filter(Boolean).join("; ");
      toast(`${r.added_train} training and ${r.added_heldout} held-out rows added${notes ? `. ${notes}` : ""}`);
      reloadStatus();
      reloadBoard();
      onCreated(name);
    } catch (e) {
      toastError(e);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal title={extending ? `Add rows to ${initialName}` : "New dataset"} onClose={onClose} foot={<>
      <button className="btn" onClick={onClose}>Cancel</button>
      <button className="btn primary" disabled={!valid || busy} onClick={submit}>
        {busy && <span className="spinner" />} {tab === "agent" ? "Start in chat" : extending ? "Add rows" : "Create"}
      </button>
    </>}>
      <div className="tabs">
        <button className={tab === "file" ? "on" : ""} onClick={() => setTab("file")}>Upload a file</button>
        <button className={tab === "hf" ? "on" : ""} onClick={() => setTab("hf")}>Hugging Face</button>
        <button className={tab === "agent" ? "on" : ""} onClick={() => setTab("agent")}>Write it with the agent</button>
      </div>
      {!extending && (
        <label className="field">
          <span>Name</span>
          <input className="input mono" value={name} onChange={(e) => setName(e.target.value.toLowerCase())} placeholder="support-replies" />
          {name && !NAME_RE.test(name) && <small className="err">lowercase letters, digits, dot, dash, underscore</small>}
        </label>
      )}
      {tab === "file" && (
        <>
          <label className={`drop ${over ? "over" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
            onDrop={(e) => { e.preventDefault(); setOver(false); pickFile(e.dataTransfer.files[0] || null); }}>
            <input type="file" accept=".jsonl,.json,.csv,.tsv,.ndjson" hidden onChange={(e) => pickFile(e.target.files?.[0] || null)} />
            <Icon name="upload" className="faint" />
            <div style={{ marginTop: 6 }}>{file ? <b>{file.name}</b> : "Drop a file here or click to choose"}</div>
            <div className="small faint">JSONL, JSON or CSV. Rows with chat <code>messages</code>, or columns like prompt/completion, question/answer, input/output.</div>
          </label>
          <div style={{ height: 14 }} />
        </>
      )}
      {tab === "hf" && (
        <>
          <label className="field">
            <span>Dataset</span>
            <input className="input mono" value={hf} onChange={(e) => { setHf(e.target.value); if (!name) setName(slug(e.target.value.split("/").pop() || "")); }} placeholder="openai/gsm8k" />
            <small>Public datasets the Hugging Face viewer can read. Nothing is uploaded.</small>
          </label>
          <div className="grid2">
            <label className="field"><span>Config</span><input className="input" value={config} onChange={(e) => setConfig(e.target.value)} placeholder="auto" /></label>
            <label className="field"><span>Split</span><input className="input" value={split} onChange={(e) => setSplit(e.target.value)} placeholder="train" /></label>
          </div>
          <label className="field"><span>Rows</span><input className="input" type="number" value={limit} onChange={(e) => setLimit(e.target.value)} /></label>
        </>
      )}
      {tab === "agent" && (
        <label className="field">
          <span>What should the examples teach?</span>
          <textarea className="textarea" value={goal} onChange={(e) => setGoal(e.target.value)}
            placeholder="Replies to customer emails about refunds, in our tone: short, warm, always with the next step. Our policy is…" />
          <small>Opens a conversation where the agent drafts examples, shows them to you and sets up grading.</small>
        </label>
      )}
      {tab !== "agent" && (
        <details>
          <summary className="small muted" style={{ cursor: "pointer", marginBottom: 10 }}>Columns and description</summary>
          <div className="grid2">
            <label className="field"><span>Prompt column</span><input className="input mono" value={inputKey} onChange={(e) => setInputKey(e.target.value)} placeholder="auto" /></label>
            <label className="field"><span>Answer column</span><input className="input mono" value={outputKey} onChange={(e) => setOutputKey(e.target.value)} placeholder="auto" /></label>
          </div>
          {!extending && <label className="field"><span>Description</span><input className="input" value={description} onChange={(e) => setDescription(e.target.value)} /></label>}
        </details>
      )}
    </Modal>
  );
}

// ------------------------------------------------------------------ one dataset

type Detail = DatasetSummary & { grader_spec: any; system: string; total: number; offset: number; rows: Row[]; evals: EvalSummary[] };

const GRADERS = [
  { type: "judge", label: "Model judge", help: "A model scores each answer 0-10 against a rubric (and the reference, if there is one). For open-ended answers." },
  { type: "exact", label: "Exact match", help: "Same text as the reference, ignoring case and spacing. For labels and short answers." },
  { type: "contains", label: "Contains", help: "The reference appears somewhere in the answer." },
  { type: "number", label: "Number", help: "The last number in the answer equals the reference's. For math." },
  { type: "json", label: "JSON", help: "The answer parses as JSON equal to the reference's. For extraction." },
  { type: "regex", label: "Pattern", help: "The answer matches a regular expression." },
  { type: "command", label: "Script", help: "Your script reads {prompt, answer, reference} as JSON on stdin; exit 0 passes or print a 0-1 score." },
  { type: "python", label: "Reward function", help: "Your environment: a .py file with reward(prompt, answer, reference) returning 0-1. Also runs inside RL and rejection sampling on the GPU." },
];

export function DatasetView({ name, onNew }: { name: string; onNew?: () => void }) {
  const [q, setQ] = useState("");
  const [offset, setOffset] = useState(0);
  const PAGE = 50;
  const { data, error, reload } = useAsync(() => api.get<Detail>(`/api/dataset/${name}?offset=${offset}&limit=${PAGE}&q=${encodeURIComponent(q)}`), [name, offset, q]);
  const [open, setOpen] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [grading, setGrading] = useState(false);
  const [evalOpen, setEvalOpen] = useState(false);
  const [adding, setAdding] = useState(false);
  const { reloadStatus, reloadBoard } = useApp();

  const train = () => fillComposer(`Train a model on the dataset ${name}. Check the data and the baseline first, then propose the training run with its cost.`, true);

  if (error) return <Empty title="Dataset not found">{error}</Empty>;
  if (!data) return <div className="row muted"><span className="spinner" /> loading…</div>;

  const remove = async () => {
    if (!confirm(`Remove ${selected.size} rows from ${name}?`)) return;
    try {
      await api.post(`/api/dataset/${name}/rows/delete`, { ids: [...selected] });
      setSelected(new Set());
      reload();
    } catch (e) { toastError(e); }
  };
  const del = async () => {
    if (!confirm(`Delete the dataset ${name}, its held-out rows included? Evals stay.`)) return;
    try {
      await api.del(`/api/dataset/${name}`);
      reloadStatus();
      reloadBoard();
      go("s", "data");
    } catch (e) { toastError(e); }
  };
  const scores = Object.entries(data.latest_scores || {}).sort((a, b) => b[1].score - a[1].score);

  return (
    <>
      <div className="stage-head">
        <div className="grow">
          <h1>Data · <span className="mono">{name}</span></h1>
          <p>{data.description || "Examples of what the model should do. A share is held out and sealed from training, so scores mean something."}</p>
        </div>
        <div className="actions">
          <button className="btn" onClick={() => setAdding(true)}><Icon name="plus" /> add rows</button>
          {onNew && <button className="btn" onClick={onNew}>new dataset</button>}
          <button className="btn" onClick={() => setEvalOpen(true)}><Icon name="play" /> evaluate</button>
          <button className="btn primary" onClick={train}><Icon name="spark" /> train on this</button>
        </div>
      </div>
      <div className="stats">
        <div className="stat"><b>{data.counts.train}</b><span>training rows</span></div>
        <div className="stat"><b>{data.counts.heldout}</b><span>held out (sealed)</span></div>
        <div className="stat"><b>{data.counts.labelled}</b><span>training rows with answers</span></div>
        <div className="stat" style={{ cursor: "pointer" }} onClick={() => setGrading(true)}>
          <b style={{ fontSize: 15, paddingTop: 5 }}>{data.grader}</b><span>grader · <a>change</a></span>
        </div>
      </div>

      {scores.length > 0 && (
        <div className="card" style={{ marginBottom: 18 }}>
          <div className="card-head"><h3>held-out scores</h3><a href={`#/grid/${encodeURIComponent(name)}`} className="small">every row →</a></div>
          <div className="bars card-pad">
            {scores.map(([target, s]) => (
              <div key={target} className="barrow" style={{ cursor: "pointer" }} onClick={() => go("evals", s.eval)}>
                <span className="lbl" title={target}>{target}</span>
                <div className="track"><div className="fill" style={{ width: `${s.score * 100}%` }} /></div>
                <span className="val">{pct(s.score)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="card">
        <div className="card-head">
          <h3>Training rows</h3>
          {selected.size > 0 && <button className="btn sm danger" onClick={remove}><Icon name="trash" /> Remove {selected.size}</button>}
          <input className="input" style={{ width: 240, height: 30 }} placeholder="Search rows" value={q}
            onChange={(e) => { setQ(e.target.value); setOffset(0); }} />
        </div>
        {data.rows.length === 0 ? <Empty title={q ? "No rows match" : "No training rows"} /> : (
          <table className="table">
            <thead><tr><th style={{ width: 28 }} /><th>Prompt</th><th>Answer</th><th style={{ width: 110 }}>Id</th></tr></thead>
            <tbody>
              {data.rows.map((r) => {
                const user = r.messages.filter((m) => m.role === "user").map((m) => m.content).join(" ⏎ ");
                const last = r.messages[r.messages.length - 1];
                const ans = last?.role === "assistant" ? last.content : null;
                const isOpen = open === r.id;
                return [
                  <tr key={r.id} className="click" onClick={() => setOpen(isOpen ? null : r.id)}>
                    <td onClick={(e) => e.stopPropagation()}>
                      <input type="checkbox" checked={selected.has(r.id)} onChange={(e) => {
                        const n = new Set(selected);
                        if (e.target.checked) n.add(r.id); else n.delete(r.id);
                        setSelected(n);
                      }} />
                    </td>
                    <td className="clip" style={{ width: "50%" }}>{user}</td>
                    <td className="clip">{ans ?? <span className="faint">no answer yet</span>}</td>
                    <td className="mono faint small clip">{r.id}</td>
                  </tr>,
                  isOpen && (
                    <tr key={r.id + "-open"}><td /><td colSpan={3}>
                      <div className="turns">
                        {r.messages.map((m, i) => (
                          <div key={i} className={`turn ${m.role}`}><span className="role">{m.role}</span><div className="content">{m.content}</div></div>
                        ))}
                        {r.meta && <div className="small faint mono">{JSON.stringify(r.meta)}</div>}
                      </div>
                    </td></tr>
                  ),
                ];
              })}
            </tbody>
          </table>
        )}
        {data.total > PAGE && (
          <div className="row" style={{ padding: "10px 16px", borderTop: "1px solid var(--line)" }}>
            <span className="small muted grow">{offset + 1}–{Math.min(offset + PAGE, data.total)} of {data.total}</span>
            <button className="btn sm" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>Previous</button>
            <button className="btn sm" disabled={offset + PAGE >= data.total} onClick={() => setOffset(offset + PAGE)}>Next</button>
          </div>
        )}
      </div>

      <div className="row" style={{ marginTop: 20 }}>
        <div className="small muted grow">
          {data.sources.map((s, i) => <div key={i}>Source: {s.hf ? `Hugging Face ${s.hf} (${s.config}/${s.split})` : s.upload ? `upload ${s.upload}` : s.file ? s.file : s.agent ? `written by the ${s.agent} agent` : s.web || "?"} · {s.rows} rows · {ago(s.time)}</div>)}
        </div>
        <button className="btn sm danger" onClick={del}><Icon name="trash" /> Delete dataset</button>
      </div>

      {grading && <GraderModal name={name} spec={data.grader_spec} system={data.system} description={data.description || ""} onClose={() => setGrading(false)} onSaved={() => { setGrading(false); reload(); }} />}
      {evalOpen && <RunEvalModal dataset={name} onClose={() => setEvalOpen(false)} />}
      {adding && <NewDataset initialName={name} onClose={() => setAdding(false)} onCreated={() => { setAdding(false); reload(); }} />}
    </>
  );
}

export function GraderModal({ name, spec, system, description, onClose, onSaved }: {
  name: string; spec: any; system: string; description: string; onClose: () => void; onSaved: () => void;
}) {
  const [s, setS] = useState<any>(spec || { type: "judge" });
  const [sys, setSys] = useState(system);
  const [desc, setDesc] = useState(description);
  const [busy, setBusy] = useState(false);
  const { models } = useApp();
  const g = GRADERS.find((x) => x.type === s.type) || GRADERS[0];
  useEffect(() => setS(spec || { type: "judge" }), [spec]);
  const save = async () => {
    setBusy(true);
    try {
      const clean = Object.fromEntries(Object.entries(s).filter(([, v]) => v !== "" && v !== undefined));
      await api.patch(`/api/dataset/${name}`, { grader: clean, system: sys, description: desc });
      toast("Saved");
      onSaved();
    } catch (e) { toastError(e); } finally { setBusy(false); }
  };
  return (
    <Modal title="How answers are graded" onClose={onClose} foot={<>
      <button className="btn" onClick={onClose}>Cancel</button>
      <button className="btn primary" disabled={busy} onClick={save}>Save</button>
    </>}>
      <div className="field">
        <span>Grader</span>
        <div className="row" style={{ flexWrap: "wrap", gap: 6 }}>
          {GRADERS.map((x) => (
            <button key={x.type} className={`btn sm ${s.type === x.type ? "primary" : ""}`} onClick={() => setS({ type: x.type })}>{x.label}</button>
          ))}
        </div>
        <small>{g.help}</small>
      </div>
      {s.type === "judge" && (
        <>
          <label className="field"><span>Rubric</span>
            <textarea className="textarea" value={s.rubric || ""} onChange={(e) => setS({ ...s, rubric: e.target.value })}
              placeholder="Score 10 when the reply solves the customer's problem, follows our refund policy and is under 120 words…" />
          </label>
          <label className="field"><span>Judge model</span>
            <select className="select" value={s.model || ""} onChange={(e) => setS({ ...s, model: e.target.value })}>
              <option value="">default engine model</option>
              {models.filter((m) => m.connected).map((m) => <option key={m.ref} value={m.ref}>{m.name.includes(m.provider_name.split(" (")[0]) ? m.name : `${m.provider_name} · ${m.name}`}</option>)}
            </select>
          </label>
        </>
      )}
      {s.type === "regex" && (
        <label className="field"><span>Pattern</span><input className="input mono" value={s.pattern || ""} onChange={(e) => setS({ ...s, pattern: e.target.value })} placeholder="leave empty to match the reference" /></label>
      )}
      {s.type === "number" && (
        <label className="field"><span>Tolerance (relative)</span><input className="input" type="number" step="any" value={s.tolerance ?? ""} onChange={(e) => setS({ ...s, tolerance: e.target.value === "" ? undefined : Number(e.target.value) })} placeholder="0.000001" /></label>
      )}
      {s.type === "command" && (
        <label className="field"><span>Command</span><input className="input mono" value={s.command || ""} onChange={(e) => setS({ ...s, command: e.target.value })} placeholder="python grade.py" />
          <small>Runs in the project folder, in the sandbox.</small></label>
      )}
      {s.type === "python" && (
        <label className="field"><span>Reward file</span><input className="input mono" value={s.path || ""} onChange={(e) => setS({ ...s, path: e.target.value })} placeholder="reward.py" />
          <small>In the project folder; defines reward(prompt, answer, reference). Its code goes to the GPU with RL jobs.</small></label>
      )}
      {(s.type === "judge" || s.type === "command" || s.type === "python") && (
        <label className="field"><span>Pass when the score is at least</span><input className="input" type="number" step="0.05" min="0" max="1" value={s.threshold ?? 0.7} onChange={(e) => setS({ ...s, threshold: Number(e.target.value) })} /></label>
      )}
      <label className="field"><span>System prompt the model gets on every row</span>
        <textarea className="textarea" style={{ minHeight: 60 }} value={sys} onChange={(e) => setSys(e.target.value)} placeholder="optional" /></label>
      <label className="field"><span>Description</span><input className="input" value={desc} onChange={(e) => setDesc(e.target.value)} /></label>
    </Modal>
  );
}
