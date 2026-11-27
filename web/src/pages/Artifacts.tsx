import { useEffect, useState } from "react";
import { api } from "../api";
import { ago, Empty, Markdown, toast, toastError } from "../ui";
import { go } from "../App";

// Artifacts: the pages that lead the work. Each one shows evidence for one step and asks one decision.
//   #/a           every artifact, newest first
//   #/a/<id>      one artifact (its latest version), its decision, its links

type Block =
  | { type: "text"; text: string }
  | { type: "stats"; items: { label: string; value: string | number; note?: string }[] }
  | { type: "bars"; title?: string; items: { label: string; value: number; max?: number; note?: string }[] }
  | { type: "table"; title?: string; columns: string[]; rows: (string | number)[][] }
  | { type: "samples"; title?: string; items: { title?: string; left_label: string; left: string; right_label: string; right: string }[] }
  | { type: "list"; title?: string; items: string[] };

type Decision = { question: string; options: string[]; chosen?: string; note?: string; at?: string };

export type Artifact = {
  id: string; kind: string; title: string; summary: string; version: number; created: string;
  source: string; links: string[]; decision: Decision | null; blocks?: Block[];
};

const KIND: Record<string, string> = {
  data_card: "data card", eval_comparison: "eval comparison", recipe_plan: "recipe plan",
  run_page: "run", release: "release", cost_ledger: "cost ledger",
};
const kindLabel = (k: string) => KIND[k] || k.replace(/_/g, " ");

export default function Artifacts({ id }: { id?: string }) {
  return id ? <ArtifactView id={id} /> : <ArtifactList />;
}

function ArtifactList() {
  const [list, setList] = useState<Artifact[] | null>(null);
  const [busy, setBusy] = useState(false);
  const load = () => api.get<Artifact[]>("/api/artifacts").then(setList).catch(toastError);
  useEffect(() => { load(); const t = setInterval(load, 5000); return () => clearInterval(t); }, []);

  const makeCard = async () => {
    setBusy(true);
    try { const a = await api.post<Artifact>("/api/artifacts/data-card"); go("a", a.id); }
    catch (e) { toastError(e); } finally { setBusy(false); }
  };

  return (
    <div className="artifacts">
      <div className="row" style={{ marginBottom: 14 }}>
        <h2 style={{ margin: 0 }}>Artifacts</h2>
        <span className="grow" />
        <button className="btn" disabled={busy} onClick={makeCard}>{busy ? "Reading your data…" : "Make the data card"}</button>
      </div>
      {list === null ? <div className="faint">Loading…</div> : list.length === 0 ? (
        <div className="card"><Empty title="No artifacts yet">
          Each step of the work ends in a page here: what the model will learn from, the plan, the run, whether it got
          better, the release. Each one asks you one decision. Import your coding sessions, then make the data card.
        </Empty></div>
      ) : (
        <div className="art-list">
          {list.map((a) => (
            <a key={a.id} className="card art-row" href={`#/a/${encodeURIComponent(a.id)}`}>
              <div className="faint small">{kindLabel(a.kind)} · v{a.version} · {ago(a.created)}</div>
              <div className="art-title">{a.title}</div>
              <div className="small">{a.summary}</div>
              {a.decision && (
                <div className={`small ${a.decision.chosen ? "faint" : "art-ask"}`}>
                  {a.decision.chosen ? `Decided: ${a.decision.chosen}` : `Needs you: ${a.decision.question}`}
                </div>
              )}
            </a>
          ))}
        </div>
      )}
    </div>
  );
}

function ArtifactView({ id }: { id: string }) {
  const [a, setA] = useState<Artifact | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [note, setNote] = useState("");
  const load = () => api.get<Artifact>(`/api/artifacts/${encodeURIComponent(id)}`).then((x) => { setA(x); setErr(null); })
    .catch((e) => setErr(String(e.message || e)));
  useEffect(() => { load(); const t = setInterval(load, 5000); return () => clearInterval(t); }, [id]);

  if (err) return <div className="card"><Empty title="Not found">{err}</Empty></div>;
  if (!a) return <div className="faint">Loading…</div>;

  const decide = async (option: string) => {
    try {
      setA(await api.post<Artifact>(`/api/artifacts/${encodeURIComponent(id)}/decide`, { option, note }));
      toast(`Decided: ${option}. The agent sees it.`);
    } catch (e) { toastError(e); }
  };

  return (
    <article className="artifact">
      <div className="faint small">
        <a href="#/a">artifacts</a> · {kindLabel(a.kind)} · v{a.version} · {ago(a.created)}
        {a.source === "agent" ? " · made by the agent" : ""}
      </div>
      <h1 className="art-h1">{a.title}</h1>
      {a.summary && <p className="faint">{a.summary}</p>}
      {(a.blocks || []).map((b, i) => <BlockView key={i} b={b} />)}
      {a.links.length > 0 && (
        <p className="small faint">Built on: {a.links.map((l, i) => (
          <span key={l}>{i ? ", " : ""}<a href={`#/a/${encodeURIComponent(l)}`}>{l}</a></span>))}</p>
      )}
      {a.decision && (
        <div className="card card-pad art-decision">
          <b>{a.decision.question}</b>
          {a.decision.chosen ? (
            <p>Decided: <b>{a.decision.chosen}</b>{a.decision.note ? ` (${a.decision.note})` : ""} <span className="faint">{ago(a.decision.at)}</span></p>
          ) : (
            <>
              <input className="input" placeholder="a note for the agent (optional)" value={note}
                onChange={(e) => setNote(e.target.value)} style={{ margin: "10px 0" }} />
              <div className="row" style={{ flexWrap: "wrap" }}>
                {a.decision.options.map((o, i) => (
                  <button key={o} className={`btn ${i === 0 ? "primary" : ""}`} onClick={() => decide(o)}>{o}</button>
                ))}
              </div>
            </>
          )}
        </div>
      )}
      <div className="row" style={{ marginTop: 14 }}>
        <a className="btn sm" href={`/api/artifacts/${encodeURIComponent(a.id)}/html`} target="_blank" rel="noreferrer">Open as a page to share</a>
      </div>
    </article>
  );
}

function BlockView({ b }: { b: Block }) {
  const head = "title" in b && b.title ? <h3 className="art-h3">{b.title}</h3> : null;
  switch (b.type) {
    case "text":
      return <div className="art-text"><Markdown text={b.text} /></div>;
    case "stats":
      return <div className="art-stats">{b.items.map((s, i) => (
        <div key={i} className="card art-stat"><span className="faint small">{s.label}</span><b>{s.value}</b>
          {s.note && <span className="faint small">{s.note}</span>}</div>))}</div>;
    case "bars": {
      const top = Math.max(1, ...b.items.map((x) => Number(x.value) || 0));
      return <div>{head}{b.items.map((x, i) => {
        const max = Number(x.max) || top;
        return <div key={i} className="art-bar"><span>{x.label}</span>
          <div className="art-track"><div className="art-fill" style={{ width: `${Math.min(100, ((Number(x.value) || 0) / max) * 100)}%` }} /></div>
          <span className="faint small">{x.value}{x.note || ""}</span></div>;
      })}</div>;
    }
    case "table":
      return <div className="art-table">{head}<table><thead><tr>{b.columns.map((c, i) => <th key={i}>{c}</th>)}</tr></thead>
        <tbody>{b.rows.map((r, i) => <tr key={i}>{r.map((c, j) => (
          <td key={j} className={c === "pass" ? "good" : c === "fail" ? "bad" : ""}>{c}</td>))}</tr>)}</tbody></table></div>;
    case "samples":
      return <div>{head}{b.items.map((s, i) => (
        <div key={i}>{s.title && <h3 className="art-h3">{s.title}</h3>}
          <div className="art-pair">
            <div><span className="faint small">{s.left_label}</span><pre>{s.left}</pre></div>
            <div><span className="faint small">{s.right_label}</span><pre>{s.right}</pre></div>
          </div></div>))}</div>;
    case "list":
      return <div>{head}<ul>{b.items.map((x, i) => <li key={i}><Markdown text={String(x)} /></li>)}</ul></div>;
    default:
      return null;
  }
}
