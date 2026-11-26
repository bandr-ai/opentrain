import { useState } from "react";
import { api, EvalSummary, Version } from "../api";
import { useApp } from "../App";
import { dur, Icon, Modal, money, pct, toast, toastError, useAsync } from "../ui";


export function PromoteModal({ v, current, onClose, onDone }: { v: Version; current: string | null; onClose: () => void; onDone: () => void }) {
  const { data: evals } = useAsync(() => api.get<EvalSummary[]>(`/api/eval?dataset=${v.dataset}`), [v.dataset]);
  const targets = [...new Set((evals || []).filter((e) => e.split === "heldout" && e.target !== v.version).map((e) => e.target))];
  const [base, setBase] = useState(current && current !== v.version ? current : "");
  const [result, setResult] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const chosen = base || targets[0] || "";
  const hasOwn = (evals || []).some((e) => e.target === v.version && e.split === "heldout");
  const run = async () => {
    setBusy(true);
    try {
      const r = await api.post<any>(`/api/version/${v.version}/promote`, { baseline: chosen });
      setResult(r);
      if (r.promote) toast(`${v.version} is live`);
    } catch (e) { toastError(e); } finally { setBusy(false); }
  };
  return (
    <Modal title={`Gate ${v.version}`} onClose={onClose} foot={result ? <button className="btn primary" onClick={onDone}>Done</button> : <>
      <button className="btn" onClick={onClose}>Cancel</button>
      <button className="btn primary" disabled={busy || !chosen || !hasOwn} onClick={run}>{busy && <span className="spinner" />} Run the gate</button>
    </>}>
      {!result ? (
        <>
          <p className="muted" style={{ marginTop: 0 }}>The gate compares {v.version} with a baseline on the same held-out rows of <b className="mono">{v.dataset}</b>. It goes live only if it scores higher and loses no row the baseline passed.</p>
          {!hasOwn && <p className="err">Evaluate {v.version} on {v.dataset} first.</p>}
          <label className="field"><span>Baseline</span>
            <select className="select" value={chosen} onChange={(e) => setBase(e.target.value)}>
              {targets.length === 0 && <option value="">no other model evaluated on this dataset yet</option>}
              {targets.map((t) => <option key={t} value={t}>{t}{t === current ? " (live)" : ""}</option>)}
            </select>
          </label>
        </>
      ) : (
        <div style={{ display: "grid", gap: 12 }}>
          <div className={`ask`} style={{ borderColor: result.promote ? "var(--good)" : "var(--bad)", background: result.promote ? "var(--good-soft)" : "var(--bad-soft)" }}>
            <h3>{result.promote ? `${v.version} passed the gate and is now live` : `${v.version} stays out`}</h3>
            {result.reasons?.length > 0 && <ul style={{ margin: 0, paddingLeft: 18 }}>{result.reasons.map((r: string) => <li key={r}>{r}</li>)}</ul>}
          </div>
          {result.rows !== undefined && (
            <dl className="kv">
              <dt>{v.version}</dt><dd><b>{pct(result.candidate_score)}</b></dd>
              <dt>{result.baseline}</dt><dd>{pct(result.baseline_score)}</dd>
              <dt>Rows compared</dt><dd>{result.rows}</dd>
              <dt>Fixed</dt><dd className="ok">{result.fixed.length}</dd>
              <dt>Regressed</dt><dd className={result.regressed.length ? "err" : ""}>{result.regressed.length}</dd>
            </dl>
          )}
        </div>
      )}
    </Modal>
  );
}

// ------------------------------------------------------------------ playground

type Answer = { model: string; answer?: string; error?: string; seconds?: number; cost_usd?: number; busy?: boolean };

export function Playground({ versions }: { versions: Version[] }) {
  const { models, defaultModel } = useApp();
  const [picks, setPicks] = useState<string[]>(["", ""]);
  const [prompt, setPrompt] = useState("");
  const [system, setSystem] = useState("");
  const [answers, setAnswers] = useState<Answer[]>([]);
  const options = [
    ...versions.map((v) => ({ ref: v.version, label: `${v.version} · ${v.base_model}` })),
    ...models.filter((m) => m.connected).map((m) => ({ ref: m.ref, label: m.name.includes(m.provider_name.split(" (")[0]) ? m.name : `${m.provider_name} · ${m.name}` })),
  ];
  const run = async () => {
    const chosen = picks.map((p) => p || defaultModel).filter(Boolean);
    setAnswers(chosen.map((m) => ({ model: m, busy: true })));
    await Promise.all(chosen.map(async (m, i) => {
      try {
        const r = await api.post<Answer>("/api/try", { model: m, prompt, ...(system ? { system } : {}) });
        setAnswers((a) => a.map((x, j) => (j === i ? { ...r, model: m } : x)));
      } catch (e: any) {
        setAnswers((a) => a.map((x, j) => (j === i ? { model: m, error: e.message } : x)));
      }
    }));
  };
  return (
    <div className="card">
      <div className="card-head"><h3>Playground</h3><span className="small faint">the same prompt to several models, side by side</span></div>
      <div className="card-pad" style={{ display: "grid", gap: 10 }}>
        <div className="row" style={{ flexWrap: "wrap" }}>
          {picks.map((p, i) => (
            <select key={i} className="select" style={{ width: 260 }} value={p} onChange={(e) => setPicks(picks.map((x, j) => (j === i ? e.target.value : x)))}>
              <option value="">{defaultModel ? `default (${defaultModel})` : "default"}</option>
              {options.map((o) => <option key={o.ref} value={o.ref}>{o.label}</option>)}
            </select>
          ))}
          {picks.length < 4 && <button className="btn sm" onClick={() => setPicks([...picks, ""])}><Icon name="plus" /> model</button>}
          {picks.length > 1 && <button className="btn sm ghost" onClick={() => setPicks(picks.slice(0, -1))}>remove</button>}
        </div>
        <input className="input" placeholder="System prompt (optional)" value={system} onChange={(e) => setSystem(e.target.value)} />
        <textarea className="textarea" placeholder="A prompt to try…" value={prompt} onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && prompt.trim()) run(); }} />
        <div className="row"><span className="small faint grow">⌘/Ctrl+Enter to run. Paid models count against your budget.</span>
          <button className="btn primary" disabled={!prompt.trim()} onClick={run}><Icon name="play" /> Run</button></div>
        {answers.length > 0 && (
          <div className="compare">
            {answers.map((a, i) => (
              <div key={i} className="card card-pad">
                <div className="row small" style={{ marginBottom: 6 }}>
                  <b className="mono grow">{a.model}</b>
                  {a.seconds !== undefined && <span className="faint">{dur(a.seconds)}{a.cost_usd ? ` · ${money(a.cost_usd)}` : ""}</span>}
                </div>
                {a.busy ? <span className="spinner" /> : a.error ? <div className="err small">{a.error}</div> : <div className="answer">{a.answer}</div>}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
