import { useMemo, useState } from "react";
import { api, DatasetSummary, EvalRow, EvalSummary, Version } from "../api";
import { go, useApp } from "../App";
import { ago, dur, Empty, Icon, Modal, money, pct, toast, toastError, useAsync } from "../ui";

export default function Evals({ id }: { id?: string }) {
  return id ? <EvalDetail key={id} id={id} /> : <EvalList />;
}

function EvalList() {
  const { data: evals, loading } = useAsync(() => api.get<EvalSummary[]>("/api/eval"), []);
  const { status } = useApp();
  const [run, setRun] = useState(false);
  const [filter, setFilter] = useState<string>("");
  const datasets = useMemo(() => [...new Set((evals || []).map((e) => e.dataset))], [evals]);
  const shown = (evals || []).filter((e) => !filter || e.dataset === filter);
  const live = status?.live?.version;

  return (
    <div className="page"><div className="page-inner">
      <div className="page-head">
        <div>
          <h1>Evals</h1>
          <p>Models scored on the same held-out rows.</p>
        </div>
        <div className="actions"><button className="btn primary" onClick={() => setRun(true)}><Icon name="play" /> Run an eval</button></div>
      </div>
      {loading && !evals ? <div className="row muted"><span className="spinner" /> Loading…</div> : !evals?.length ? (
        <div className="card"><Empty title="No evals yet" action={<button className="btn primary" onClick={() => setRun(true)}><Icon name="play" /> Run an eval</button>}>
          Start with a baseline: the model you use today on your held-out rows. Training only makes sense when you know the number to beat.
        </Empty></div>
      ) : (
        <>
          {datasets.length > 1 && (
            <div className="seg" style={{ marginBottom: 16 }}>
              <button className={!filter ? "on" : ""} onClick={() => setFilter("")}>All</button>
              {datasets.map((d) => <button key={d} className={filter === d ? "on" : ""} onClick={() => setFilter(d)}>{d}</button>)}
            </div>
          )}
          {(filter ? [filter] : datasets).map((d) => <Leaderboard key={d} dataset={d} evals={evals.filter((e) => e.dataset === d)} live={live} />)}
          <div className="card">
            <div className="card-head"><h3>History</h3></div>
            <table className="table">
              <thead><tr><th>Model</th><th>Dataset</th><th>Rows</th><th className="num">Score</th><th className="num">Passed</th><th className="num">Median time</th><th className="num">Cost</th><th>When</th></tr></thead>
              <tbody>
                {shown.map((e) => (
                  <tr key={e.id} className="click" onClick={() => go("evals", e.id)}>
                    <td className="mono">{e.target}{e.target === live && <span className="badge good" style={{ marginLeft: 6 }}>live</span>}</td>
                    <td className="mono">{e.dataset}</td>
                    <td>{e.split === "heldout" ? "held out" : <span className="badge">training</span>}</td>
                    <td className="num"><b>{pct(e.summary.mean_score)}</b></td>
                    <td className="num">{e.summary.passed}/{e.summary.rows}{e.summary.errors ? <span className="err"> · {e.summary.errors} errors</span> : null}</td>
                    <td className="num">{dur(e.summary.median_seconds)}</td>
                    <td className="num">{money(e.summary.cost_usd)}</td>
                    <td className="muted">{ago(e.time)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
      {run && <RunEvalModal dataset={filter || undefined} onClose={() => setRun(false)} />}
    </div></div>
  );
}

function Leaderboard({ dataset, evals, live }: { dataset: string; evals: EvalSummary[]; live?: string | null }) {
  const best = new Map<string, EvalSummary>();
  for (const e of evals) if (e.split === "heldout" && !best.has(e.target)) best.set(e.target, e);
  const rows = [...best.values()].sort((a, b) => b.summary.mean_score - a.summary.mean_score);
  if (!rows.length) return null;
  return (
    <div className="card card-pad" style={{ marginBottom: 16 }}>
      <div className="row" style={{ marginBottom: 12 }}>
        <h3 className="grow">Held-out leaderboard · <a href={`#/data/${dataset}`} className="mono">{dataset}</a></h3>
        <span className="small faint">latest eval per model</span>
      </div>
      <div className="bars">
        {rows.map((e) => (
          <div key={e.target} className="barrow" style={{ cursor: "pointer" }} onClick={() => go("evals", e.id)}>
            <span className="lbl" title={e.target}>{e.target}</span>
            <div className="track"><div className={`fill ${e.target === live ? "live" : ""}`} style={{ width: `${e.summary.mean_score * 100}%` }} /></div>
            <span className="val">{pct(e.summary.mean_score)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ run

export function RunEvalModal({ dataset, onClose, model: initialModel }: { dataset?: string; onClose: () => void; model?: string }) {
  const { models, defaultModel } = useApp();
  const { data: datasets } = useAsync(() => api.get<DatasetSummary[]>("/api/dataset"), []);
  const { data: versions } = useAsync(() => api.get<{ versions: Version[] }>("/api/version"), []);
  const [ds, setDs] = useState(dataset || "");
  const [model, setModel] = useState(initialModel || "");
  const [custom, setCustom] = useState("");
  const [split, setSplit] = useState<"heldout" | "train">("heldout");
  const [limit, setLimit] = useState("");
  const [busy, setBusy] = useState(false);
  const chosen = model === "__custom" ? custom.trim() : model || defaultModel;
  const target = (datasets || []).find((d) => d.name === (ds || datasets?.[0]?.name));
  const n = target ? (split === "heldout" ? target.counts.heldout : target.counts.train) : 0;
  const paid = chosen && !chosen.startsWith("claude-code/") && !/^v\d+$/.test(chosen);
  const start = async () => {
    setBusy(true);
    try {
      const name = ds || datasets?.[0]?.name;
      await api.post(`/api/dataset/${name}/eval`, { model: chosen, split, ...(limit ? { limit: Number(limit) } : {}) });
      toast(`evaluating ${chosen} on ${name}; the scoreboard updates when it finishes`);
      onClose();
    } catch (e) { toastError(e); } finally { setBusy(false); }
  };
  return (
    <Modal title="Run an eval" onClose={onClose} foot={<>
      <button className="btn" onClick={onClose}>Cancel</button>
      <button className="btn primary" disabled={busy || !chosen || !n} onClick={start}>{busy && <span className="spinner" />} Run on {Math.min(n, Number(limit) || n)} rows</button>
    </>}>
      <label className="field"><span>Dataset</span>
        <select className="select" value={ds || datasets?.[0]?.name || ""} onChange={(e) => setDs(e.target.value)}>
          {(datasets || []).map((d) => <option key={d.name} value={d.name}>{d.name} · {d.counts.heldout} held out · {d.grader}</option>)}
        </select>
        {datasets && !datasets.length && <small className="err">Create a dataset first.</small>}
      </label>
      <label className="field"><span>Model</span>
        <select className="select" value={model} onChange={(e) => setModel(e.target.value)}>
          <option value="">{defaultModel ? `default (${defaultModel})` : "default"}</option>
          {versions?.versions?.length ? <optgroup label="Your trained versions">
            {versions.versions.map((v) => <option key={v.version} value={v.version}>{v.version} · {v.base_model}{v.dataset ? ` · ${v.dataset}` : ""}</option>)}
          </optgroup> : null}
          {Object.entries(models.filter((m) => m.connected).reduce<Record<string, typeof models>>((g, m) => ((g[m.provider_name] ||= []).push(m), g), {})).map(([g, ms]) => (
            <optgroup key={g} label={g}>{ms.map((m) => <option key={m.ref} value={m.ref}>{m.name}</option>)}</optgroup>
          ))}
          <option value="__custom">Another model…</option>
        </select>
        {model === "__custom" && <input className="input mono" value={custom} onChange={(e) => setCustom(e.target.value)} placeholder="openrouter/qwen/qwen3-8b" style={{ marginTop: 6 }} />}
      </label>
      <div className="grid2">
        <label className="field"><span>Rows</span>
          <div className="seg">
            <button className={split === "heldout" ? "on" : ""} onClick={() => setSplit("heldout")}>Held out</button>
            <button className={split === "train" ? "on" : ""} onClick={() => setSplit("train")}>Training (shows answers)</button>
          </div>
        </label>
        <label className="field"><span>At most</span><input className="input" type="number" value={limit} onChange={(e) => setLimit(e.target.value)} placeholder={`all ${n}`} /></label>
      </div>
      <p className="small muted" style={{ margin: 0 }}>
        {paid ? "This model is billed per token; the cost counts against your Trainfile budget and the eval stops if it would go over." :
          chosen.startsWith("claude-code/") ? "Runs on your Claude Code login: no API spend." : ""}
        {split === "heldout" && " Answers on held-out rows stay sealed: you see per-row scores, not the rows."}
      </p>
    </Modal>
  );
}

// ------------------------------------------------------------------ one eval

type EvalDetailT = EvalSummary & { rows: EvalRow[]; grader: any };

export function EvalDetail({ id }: { id: string }) {
  const { data, error } = useAsync(() => api.get<EvalDetailT>(`/api/eval/${id}`), [id]);
  const [show, setShow] = useState<"all" | "failed" | "passed">("all");
  const [open, setOpen] = useState<string | null>(null);
  if (error) return <div className="page"><div className="page-inner"><Empty title="Eval not found">{error}</Empty></div></div>;
  if (!data) return <div className="page"><div className="page-inner row muted"><span className="spinner" /> Loading…</div></div>;
  const s = data.summary;
  const rows = data.rows.filter((r) => show === "all" || (show === "failed" ? !r.passed : r.passed));
  return (
    <div className="page"><div className="page-inner">
      <div className="page-head">
        <div className="grow">
          <div className="small muted"><a href="#/evals">Evals</a> /</div>
          <h1><span className="mono">{data.target}</span> on <a className="mono" href={`#/data/${data.dataset}`}>{data.dataset}</a></h1>
          <p>{data.split === "heldout" ? "Held-out rows" : "Training rows"} · {ago(data.time)} · <span className="mono small">{data.id}</span></p>
        </div>
      </div>
      <div className="stats">
        <div className="stat"><b>{pct(s.mean_score)}</b><span>mean score</span></div>
        <div className="stat"><b>{s.passed}/{s.rows}</b><span>passed ({pct(s.pass_rate)})</span></div>
        <div className="stat"><b>{dur(s.median_seconds) || "—"}</b><span>median time per answer</span></div>
        <div className="stat"><b>{money(s.cost_usd)}</b><span>cost</span></div>
        {s.errors > 0 && <div className="stat"><b className="err">{s.errors}</b><span>errors (no answer)</span></div>}
      </div>
      <div className="card card-pad" style={{ marginBottom: 16 }}>
        <div className="row" style={{ marginBottom: 10 }}><h3 className="grow">Every row</h3><span className="small faint">green passed · red failed · amber error</span></div>
        <div className="dots">{data.rows.map((r) => <i key={r.id} className={r.error ? "e" : r.passed ? "p" : ""} title={`${r.id}: ${r.score}${r.error ? " " + r.error : ""}`} />)}</div>
      </div>
      <div className="card">
        <div className="card-head">
          <h3>Rows</h3>
          <div className="seg">
            {(["all", "failed", "passed"] as const).map((k) => <button key={k} className={show === k ? "on" : ""} onClick={() => setShow(k)}>{k}</button>)}
          </div>
        </div>
        <table className="table">
          <thead><tr><th>Row</th>{data.split === "train" && <th>Prompt</th>}<th className="num">Score</th><th className="num">Time</th><th>Note</th></tr></thead>
          <tbody>
            {rows.map((r) => [
              <tr key={r.id} className={data.split === "train" ? "click" : ""} onClick={() => data.split === "train" && setOpen(open === r.id ? null : r.id)}>
                <td className="mono small">{r.id}</td>
                {data.split === "train" && <td className="clip" style={{ width: "45%" }}>{r.prompt}</td>}
                <td className="num"><span className={`badge ${r.error ? "warn" : r.passed ? "good" : "bad"}`}>{r.error ? "error" : pct(r.score)}</span></td>
                <td className="num muted">{dur(r.seconds)}</td>
                <td className="small muted clip">{r.error || r.reason || ""}</td>
              </tr>,
              open === r.id && (
                <tr key={r.id + "-o"}><td colSpan={5}>
                  <div className="compare">
                    <div><div className="small muted" style={{ marginBottom: 4 }}>Prompt</div><div className="answer">{r.prompt}</div></div>
                    <div><div className="small muted" style={{ marginBottom: 4 }}>Model's answer</div><div className="answer">{r.answer}</div></div>
                    <div><div className="small muted" style={{ marginBottom: 4 }}>Reference</div><div className="answer ok">{r.reference || "—"}</div></div>
                  </div>
                </td></tr>
              ),
            ])}
          </tbody>
        </table>
        {data.split === "heldout" && <div className="small faint" style={{ padding: "10px 16px", borderTop: "1px solid var(--line)" }}>
          Held-out rows and the answers to them are sealed so they can never leak into training. To read answers, run the same model on training rows.
        </div>}
      </div>
    </div></div>
  );
}
