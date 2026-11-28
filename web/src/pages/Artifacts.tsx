import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { ago, Empty, money, toast, toastError } from "../ui";
import { go, useApp } from "../App";

// Artifacts: the pages that lead the work. Each one shows evidence for one step; a few ask the one
// thing that is the person's to decide (spending on a plan, what goes live).
//   #/a           every artifact, newest first
//   #/a/<id>      one artifact (its latest version), its decision, its links

type Decision = {
  question: string; options: string[]; chosen?: string; note?: string; at?: string; by?: string;
  approves?: { needs: string[]; cost_usd: number } | null;
};

/** The go-ahead button says what it approves: "Run it · ~$0.65". */
const optionLabel = (d: Decision, o: string, i: number) =>
  i === 0 && d.approves?.cost_usd ? `${o} · ~${money(d.approves.cost_usd)}` : o;

export type Artifact = {
  id: string; kind: string; title: string; summary: string; version: number; created: string;
  source: string; links: string[]; decision: Decision | null; blocks?: unknown[];
};

const KIND: Record<string, string> = {
  data_card: "data card", eval_comparison: "eval comparison", recipe_plan: "recipe plan",
  run_page: "run", release: "release", cost_ledger: "cost ledger",
};
const kindLabel = (k: string) => KIND[k] || k.replace(/_/g, " ");

/** The tools whose result is an artifact; the chat shows it as a card and opens it. */
export const ARTIFACT_TOOLS = new Set(["data_card", "eval_comparison", "artifact_make"]);

/** The artifact id in such a tool's result, if it finished. */
export function artifactIdOf(tool: { name: string; state: string; output?: string }): string | null {
  if (!ARTIFACT_TOOLS.has(tool.name) || tool.state !== "completed" || !tool.output) return null;
  try { return JSON.parse(tool.output).id || null; } catch { return (tool.output.match(/"id":\s*"([^"]+)"/) || [])[1] || null; }
}

/** Record a decision and hand it to the agent's conversation, so the work goes on from there. */
export async function decideArtifact(id: string, option: string, note: string, sid: string | null): Promise<Artifact> {
  const a = await api.post<Artifact>(`/api/artifacts/${encodeURIComponent(id)}/decide`, { option, note });
  const d = a.decision;
  const go = d?.approves && option === d.options[0]
    ? ` This approves the whole plan (${d.approves.needs.join(", ")}${d.approves.cost_usd ? `, up to about ${money(d.approves.cost_usd)}` : ""}): do all of it now without asking again.`
    : "";
  if (sid) {
    await api.post(`/api/session/${sid}/prompt`, {
      text: `${d?.question} ${option}.${note ? ` ${note}` : ""}${go} (decided on ${a.title}, #/a/${a.id})`,
    }).catch(() => toast("Decision saved; the agent will see it next time it looks."));
  }
  return a;
}

/** An artifact as a card in the conversation: what it is, its decision as buttons, a link to open it. */
export function ArtifactInline({ id }: { id: string }) {
  const { sid } = useApp();
  const [a, setA] = useState<Artifact | null>(null);
  const [busy, setBusy] = useState(false);
  const loaded = a !== null;
  const open = !!a?.decision && !a.decision.chosen;
  useEffect(() => {
    const load = () => api.get<Artifact>(`/api/artifacts/${encodeURIComponent(id)}`).then(setA).catch(() => {});
    if (!loaded) load();
    if (loaded && !open) return;  // nothing to decide, or decided: nothing more will change
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, [id, loaded, open]);
  if (!a) return null;
  const d = a.decision;
  const pick = async (o: string) => {
    setBusy(true);
    try { setA(await decideArtifact(a.id, o, "", sid)); } catch (e) { toastError(e); } finally { setBusy(false); }
  };
  return (
    <div className="card card-pad art-inline">
      <div className="faint small">{kindLabel(a.kind)} · v{a.version}</div>
      <a className="art-title" href={`#/a/${encodeURIComponent(a.id)}`}>{a.title}</a>
      {a.summary && <div className="small">{a.summary}</div>}
      {d && (d.chosen ? <div className="small faint">Decided: {d.chosen}</div> : (
        <div className="row" style={{ flexWrap: "wrap", marginTop: 8 }}>
          <span className="small">{d.question}</span>
          {d.options.map((o, i) => (
            <button key={o} disabled={busy} className={`btn sm ${i === 0 ? "primary" : ""}`} onClick={() => pick(o)}>{optionLabel(d, o, i)}</button>
          ))}
        </div>
      ))}
      <div style={{ marginTop: 6 }}><a className="small" href={`#/a/${encodeURIComponent(a.id)}`}>Open the full page →</a></div>
    </div>
  );
}

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
  const { sid } = useApp();
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
      setA(await decideArtifact(id, option, note, sid));
      toast(sid ? `Decided: ${option}. Sent to the agent.` : `Decided: ${option}.`);
    } catch (e) { toastError(e); }
  };

  return (
    <article className="artifact">
      <div className="faint small">
        <a href="#/a">artifacts</a>{a.source === "agent" ? " · made by the agent" : ""}
      </div>
      <ArtifactFrame id={a.id} version={a.version} />
      {a.links.length > 0 && (
        <p className="small faint">Built on: {a.links.map((l, i) => (
          <span key={l}>{i ? ", " : ""}<a href={`#/a/${encodeURIComponent(l)}`}>{l}</a></span>))}</p>
      )}
      {a.decision && (
        <div className="card card-pad art-decision">
          <b>{a.decision.question}</b>
          {a.decision.chosen ? (
            <p>Decided: <b>{a.decision.chosen}</b>
              {a.decision.note ? ` (${a.decision.note})` : ""} <span className="faint">{ago(a.decision.at)}</span></p>
          ) : (
            <>
              <input className="input" placeholder="a note for the agent (optional)" value={note}
                onChange={(e) => setNote(e.target.value)} style={{ margin: "10px 0" }} />
              <div className="row" style={{ flexWrap: "wrap" }}>
                {a.decision.options.map((o, i) => (
                  <button key={o} className={`btn ${i === 0 ? "primary" : ""}`} onClick={() => decide(o)}>{optionLabel(a.decision!, o, i)}</button>
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

/** The artifact's page as the server draws it (the same page that is shared), in a frame that can run
 *  nothing and load nothing from outside, sized to its content. */
function ArtifactFrame({ id, version }: { id: string; version: number }) {
  const ref = useRef<HTMLIFrameElement>(null);
  const [height, setHeight] = useState(600);
  useEffect(() => {
    const f = ref.current;
    if (!f) return;
    let ro: ResizeObserver | null = null;
    const fit = () => {
      const doc = f.contentDocument;
      if (!doc?.documentElement) return;
      setHeight(Math.max(200, doc.documentElement.scrollHeight));
      ro?.disconnect();
      ro = new ResizeObserver(() => setHeight(Math.max(200, doc.documentElement.scrollHeight)));
      ro.observe(doc.body);
    };
    f.addEventListener("load", fit);
    return () => { f.removeEventListener("load", fit); ro?.disconnect(); };
  }, [id, version]);
  return (
    <iframe ref={ref} key={`${id}@${version}`} className="art-frame" title={id} style={{ height }}
      sandbox="allow-same-origin allow-popups allow-popups-to-escape-sandbox"
      src={`/api/artifacts/${encodeURIComponent(id)}/html?embed=1&version=${version}`} />
  );
}
