import { useEffect, useRef, useState } from "react";
import { api, Job } from "../api";
import { go, useApp } from "../App";
import { ago, dur, Empty, money, toastError } from "../ui";
import { JobCard } from "./Chat";

export default function Jobs({ id }: { id?: string }) {
  const { jobs } = useApp();
  const job = id ? jobs.find((j) => j.id === id) : undefined;
  if (id) return <JobDetail key={id} id={id} job={job} />;
  const running = jobs.filter((j) => j.state === "running");
  const done = jobs.filter((j) => j.state !== "running");
  return (
    <div className="page"><div className="page-inner">
      <div className="page-head"><div><h1>Jobs</h1><p>Imports, labelling, evals and training. They keep running when you close the tab.</p></div></div>
      {running.length > 0 && <div style={{ display: "grid", gap: 10, marginBottom: 24 }}>{running.map((j) => <JobCard key={j.id} job={j} />)}</div>}
      {jobs.length === 0 ? <div className="card"><Empty title="No jobs yet">Evals, labelling and training show up here while they run.</Empty></div> : done.length > 0 && (
        <div className="card">
          <div className="card-head"><h3>Finished</h3></div>
          <table className="table">
            <thead><tr><th>Kind</th><th>What</th><th>Result</th><th className="num">Took</th><th className="num">Cost</th><th>When</th></tr></thead>
            <tbody>
              {done.map((j) => (
                <tr key={j.id} className="click" onClick={() => go("jobs", j.id)}>
                  <td><span className="badge">{j.kind}</span></td>
                  <td>{j.title}</td>
                  <td className="small">{result(j)}</td>
                  <td className="num muted">{j.time?.started && j.time?.ended ? dur(j.time.ended - j.time.started) : ""}</td>
                  <td className="num">{money(j.cost_usd)}</td>
                  <td className="muted">{ago(j.time?.ended || j.time?.created)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div></div>
  );
}

function result(j: Job) {
  if (j.state === "failed") return <span className="err">{(j.error || "failed").slice(0, 120)}</span>;
  if (j.state === "cancelled") return <span className="muted">cancelled</span>;
  const r = j.result || {};
  if (j.kind === "eval" && r.mean_score !== undefined) return <>score <b>{(r.mean_score * 100).toFixed(1)}%</b> · {r.passed}/{r.rows} passed</>;
  if (j.version) return <>→ <a href="#/models">{j.version}</a></>;
  if (r.labelled !== undefined) return <>{r.labelled} labelled</>;
  return <span className="ok">done</span>;
}

function JobDetail({ id, job }: { id: string; job?: Job }) {
  const [lines, setLines] = useState<string[]>([]);
  const [j, setJ] = useState<Job | undefined>(job);
  const pre = useRef<HTMLPreElement>(null);
  useEffect(() => setJ(job), [job]);
  useEffect(() => {
    let alive = true;
    const load = () => {
      api.get<Job>(`/api/job/${id}`).then((x) => alive && setJ(x)).catch(() => {});
      api.get<{ lines: string[] }>(`/api/job/${id}/logs?tail=1000`).then((r) => alive && setLines(r.lines)).catch(() => {});
    };
    load();
    const t = setInterval(() => (j?.state ?? "running") === "running" && load(), 2000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [id, j?.state]);
  useEffect(() => {
    if (pre.current) pre.current.scrollTop = pre.current.scrollHeight;
  }, [lines]);
  if (!j) return <div className="page"><div className="page-inner row muted"><span className="spinner" /> Loading…</div></div>;
  return (
    <div className="page"><div className="page-inner">
      <div className="page-head">
        <div className="grow"><div className="small muted"><a href="#/jobs">Jobs</a> /</div><h1>{j.title}</h1><p className="mono small">{j.id}</p></div>
        <div className="actions">
          {j.session_id && <button className="btn" onClick={() => go("chat", j.session_id)}>Open conversation</button>}
          {j.state === "running" && <button className="btn danger" onClick={() => api.post(`/api/job/${j.id}/cancel`).catch(toastError)}>Cancel</button>}
        </div>
      </div>
      <div style={{ marginBottom: 16 }}><JobCard job={j} /></div>
      {j.result && <div className="card card-pad" style={{ marginBottom: 16 }}>
        <h3 style={{ marginBottom: 8 }}>Result</h3>
        {j.result.eval && <p style={{ marginTop: 0 }}><a href={`#/evals/${j.result.eval}`}>Open the eval</a></p>}
        <pre className="logs" style={{ maxHeight: 240 }}>{JSON.stringify(j.result, null, 2)}</pre>
      </div>}
      <h3 style={{ marginBottom: 8 }}>Log</h3>
      <pre className="logs" ref={pre}>{lines.length ? lines.join("\n") : "No log lines."}</pre>
    </div></div>
  );
}
