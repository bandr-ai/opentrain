import { useEffect, useMemo, useState } from "react";
import { api, Grid } from "../api";
import { go, useApp } from "../App";
import { Empty, pct } from "../ui";

// Every held-out row against every model's latest held-out eval. The truth behind the scoreboard.

function stored(k: string) {
  try { return localStorage.getItem(k); } catch { return null; }
}

export default function GridPage({ dataset }: { dataset: string }) {
  const { board } = useApp();
  const [reveal, setReveal] = useState(stored("ot.reveal") === "1");
  const [grid, setGrid] = useState<Grid | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<"all" | "disagree" | string>("all");
  const [open, setOpen] = useState<string | null>(null);
  const [q, setQ] = useState("");

  useEffect(() => {
    if (!dataset) return;
    setError(null);
    api.get<Grid>(`/api/board/grid?dataset=${encodeURIComponent(dataset)}${reveal ? "&reveal=1" : ""}`).then(setGrid)
      .catch((e) => setError(e.message));
  }, [dataset, reveal, board?.scoreboard.length]);

  const rows = useMemo(() => {
    if (!grid) return [];
    return grid.rows.filter((r) => {
      if (filter === "disagree" && !r.disagree) return false;
      if (filter.startsWith("fail:") && r.cells[filter.slice(5)]?.passed !== false) return false;
      if (q && !(`${r.prompt || ""} ${r.reference || ""} ${r.id}`.toLowerCase().includes(q.toLowerCase()))) return false;
      return true;
    });
  }, [grid, filter, q]);

  if (!dataset) return <div className="stage-view"><Empty title="No dataset yet" /></div>;
  if (error) return <div className="stage-view"><Empty title="No grid">{error}</Empty></div>;
  if (!grid) return <div className="stage-view row muted"><span className="spinner" /> loading…</div>;
  const cols = grid.columns;
  const toggleReveal = () => {
    const v = !reveal;
    setReveal(v);
    try { localStorage.setItem("ot.reveal", v ? "1" : "0"); } catch { /* private */ }
  };

  return (
    <div className="stage-view wide">
      <div className="stage-head">
        <div className="grow">
          <h1>Every row · <span className="mono">{dataset}</span></h1>
          <p>{grid.rows.length} held-out rows × {cols.length} models. Click a row to see each model's answer and why it passed or failed.</p>
        </div>
        <div className="actions"><button className="btn" onClick={() => go("s", "baseline")}>← scoreboard</button></div>
      </div>
      {!cols.length ? <Empty title="No scores yet">Evaluate a model on this dataset first.</Empty> : (
        <>
          <div className="grid-bar">
            <div className="seg">
              <button className={filter === "all" ? "on" : ""} onClick={() => setFilter("all")}>all</button>
              <button className={filter === "disagree" ? "on" : ""} onClick={() => setFilter("disagree")}>models disagree</button>
            </div>
            <select className="select" style={{ width: "auto" }} value={filter.startsWith("fail:") ? filter : ""}
              onChange={(e) => setFilter(e.target.value || "all")}>
              <option value="">failed by…</option>
              {cols.map((c) => <option key={c.target} value={`fail:${c.target}`}>{c.target}</option>)}
            </select>
            {reveal && <input className="input" style={{ width: 220 }} placeholder="search rows" value={q} onChange={(e) => setQ(e.target.value)} />}
            <span className="grow" />
            <label className="reveal" title="Held-out rows are sealed from the agent and from training. You can read them; the agent never can.">
              <input type="checkbox" checked={reveal} onChange={toggleReveal} /> show the text of held-out rows
            </label>
          </div>
          <div className="grid-wrap">
            <table className="grid">
              <thead>
                <tr>
                  <th className="g-row">{reveal ? "prompt" : "row"}</th>
                  {reveal && <th className="g-ref">expected</th>}
                  {cols.map((c) => (
                    <th key={c.target} className="g-col" title={c.target}>
                      <div className="mono">{c.target.replace(/^claude-code\//, "cc/")}</div>
                      <b>{pct(c.score)}</b>{c.live && <span className="badge good">live</span>}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => [
                  <tr key={r.id} className={`click ${open === r.id ? "sel" : ""}`} onClick={() => setOpen(open === r.id ? null : r.id)}>
                    <td className="g-row">{reveal ? <span className="clip-text">{r.prompt}</span> : <span className="mono faint">{r.id}</span>}</td>
                    {reveal && <td className="g-ref ok"><span className="clip-text">{r.reference ?? "—"}</span></td>}
                    {cols.map((c) => {
                      const cell = r.cells[c.target];
                      return <td key={c.target} className={`g-cell ${!cell ? "" : cell.error ? "e" : cell.passed ? "p" : "f"}`}>{!cell ? "" : cell.error ? "!" : cell.passed ? "✓" : "✗"}</td>;
                    })}
                  </tr>,
                  open === r.id && (
                    <tr key={r.id + "-o"} className="g-open"><td colSpan={cols.length + (reveal ? 2 : 1)}>
                      {reveal ? (
                        <div className="g-detail">
                          <div className="g-prompt"><span className="k">prompt</span><div>{r.prompt}</div><span className="k">expected</span><div className="ok">{r.reference ?? "—"}</div></div>
                          <div className="g-answers">
                            {cols.map((c) => {
                              const cell = r.cells[c.target];
                              if (!cell) return null;
                              return (
                                <div key={c.target} className={`g-ans ${cell.passed ? "p" : "f"}`}>
                                  <div className="row"><b className="mono grow">{c.target}</b><span className={cell.passed ? "ok" : "err"}>{cell.error ? "error" : cell.passed ? "pass" : "fail"} · {pct(cell.score)}</span></div>
                                  <div className="ans">{cell.error || cell.answer || "—"}</div>
                                  {cell.reason && <div className="why">{cell.reason}</div>}
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      ) : (
                        <div className="small muted" style={{ padding: 8 }}>
                          Turn on <b>show the text of held-out rows</b> to read this row and each model's answer. Only you see it; the agent and training never do.
                        </div>
                      )}
                    </td></tr>
                  ),
                ])}
              </tbody>
            </table>
            {rows.length === 0 && <div className="small faint" style={{ padding: 16 }}>no rows match</div>}
          </div>
        </>
      )}
    </div>
  );
}
