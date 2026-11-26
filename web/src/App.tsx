import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { api, events, Job, ModelRow, Session, Status } from "./api";
import { money, throttle, Toasts } from "./ui";
import Chat from "./pages/Chat";
import Data from "./pages/Data";
import Evals from "./pages/Evals";
import Models from "./pages/Models";
import Jobs from "./pages/Jobs";
import Settings from "./pages/Settings";

// ------------------------------------------------------------------ routing (hash)

export type Route = { page: string; arg?: string };

function parse(hash: string): Route {
  const [page, ...rest] = hash.replace(/^#\/?/, "").split("/");
  return { page: page || "chat", arg: rest.length ? decodeURIComponent(rest.join("/")) : undefined };
}

export function go(page: string, arg?: string) {
  window.location.hash = `#/${page}${arg ? "/" + encodeURIComponent(arg) : ""}`;
}

// ------------------------------------------------------------------ shared state

type AppState = {
  status: Status | null;
  sessions: Session[];
  busy: Record<string, string>;
  models: ModelRow[];
  defaultModel: string;
  jobs: Job[];
  reloadStatus: () => void;
  reloadSessions: () => void;
  reloadModels: () => void;
};

const Ctx = createContext<AppState>(null as unknown as AppState);
export const useApp = () => useContext(Ctx);

const NAV = [
  { page: "chat", label: "Chat", icon: "chat" },
  { page: "data", label: "Data", icon: "data" },
  { page: "evals", label: "Evals", icon: "evals" },
  { page: "models", label: "Models", icon: "models" },
  { page: "jobs", label: "Jobs", icon: "jobs" },
  { page: "settings", label: "Settings", icon: "settings" },
];

export default function App() {
  const [route, setRoute] = useState<Route>(parse(window.location.hash));
  const [status, setStatus] = useState<Status | null>(null);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [busy, setBusy] = useState<Record<string, string>>({});
  const [models, setModels] = useState<ModelRow[]>([]);
  const [defaultModel, setDefaultModel] = useState("");
  const [jobs, setJobs] = useState<Job[]>([]);
  const [connected, setConnected] = useState(true);
  const [authError, setAuthError] = useState(false);

  useEffect(() => {
    const f = () => setRoute(parse(window.location.hash));
    window.addEventListener("hashchange", f);
    return () => window.removeEventListener("hashchange", f);
  }, []);

  useEffect(() => {
    const f = (e: globalThis.KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (e.metaKey || e.ctrlKey || e.altKey || t.closest("input, textarea, select, [contenteditable]")) return;
      const n = Number(e.key);
      if (n >= 1 && n <= NAV.length) go(NAV[n - 1].page);
      else if (e.key === "n") go("chat", "new");
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", f);
    return () => window.removeEventListener("keydown", f);
  }, []);

  const reloadStatus = useCallback(() => {
    api.get<Status>("/api/status").then(setStatus).catch((e) => e?.status === 401 && setAuthError(true));
  }, []);
  const reloadSessions = useCallback(() => {
    api.get<Session[]>("/api/session").then((s) =>
      setSessions([...s].sort((a, b) => (b.time?.updated ?? b.time?.created ?? 0) - (a.time?.updated ?? a.time?.created ?? 0))),
    ).catch(() => {});
  }, []);
  const reloadModels = useCallback(() => {
    api.get<{ models: ModelRow[]; default: string }>("/api/model").then((m) => {
      setModels(m.models);
      setDefaultModel(m.default);
    }).catch(() => {});
  }, []);
  const reloadJobs = useCallback(() => {
    api.get<Job[]>("/api/job").then(setJobs).catch(() => {});
  }, []);

  useEffect(() => {
    reloadStatus();
    reloadSessions();
    reloadModels();
    reloadJobs();
    events.start();
    const st = throttle(reloadStatus, 1500);
    const ss = throttle(reloadSessions, 800);
    const js = throttle(reloadJobs, 500);
    const off = events.on((e) => {
      if (e.type === "session.status") {
        setBusy((b) => {
          const n = { ...b };
          if (e.data.status === "idle") delete n[e.data.session_id];
          else n[e.data.session_id] = e.data.status;
          return n;
        });
      }
      if (e.type.startsWith("session.")) ss();
      if (e.type.startsWith("job.")) {
        js();
        st();
      }
      if (e.type === "trainfile.created" || e.type.startsWith("approval.")) st();
    });
    const offc = events.onConnection((c) => {
      setConnected(c);
      if (c) {
        reloadStatus();
        reloadSessions();
        reloadJobs();
      }
    });
    return () => {
      off();
      offc();
    };
  }, [reloadStatus, reloadSessions, reloadModels, reloadJobs]);

  const state = useMemo(
    () => ({ status, sessions, busy, models, defaultModel, jobs, reloadStatus, reloadSessions, reloadModels }),
    [status, sessions, busy, models, defaultModel, jobs, reloadStatus, reloadSessions, reloadModels],
  );

  if (authError) {
    return (
      <div className="empty" style={{ paddingTop: "20vh" }}>
        <h2>Sign in from your terminal</h2>
        <p>This browser is not paired with the OpenTrain server. Run <code>opentrain web</code> in your project; it opens a one-time sign-in link.</p>
      </div>
    );
  }

  const running = jobs.filter((j) => j.state === "running").length;
  const b = status?.budget;
  const total = b ? b.committed_usd + b.remaining_usd : 0;
  const topSessions = sessions.filter((s) => !s.parent_id);

  return (
    <Ctx.Provider value={state}>
      <div className="shell">
        <aside className="rail">
          <div className="brand">opentrain</div>
          <div className="project" title={status?.project?.directory}>{status?.project?.name || "…"}</div>
          <nav className="nav">
            {NAV.map((n, i) => (
              <button key={n.page} className={route.page === n.page ? "on" : ""} onClick={() => go(n.page)}>
                {n.label.toLowerCase()}
                {n.page === "jobs" && running > 0 && <span className="count">{running} running</span>}
                <span className="key">{i + 1}</span>
              </button>
            ))}
          </nav>
          <div className="rail-section">
            sessions
            <button className="btn ghost sm" title="New session (n)" onClick={() => go("chat", "new")}>+ new</button>
          </div>
          <div className="sessions">
            {topSessions.length === 0 && <div className="faint small" style={{ padding: "4px 8px" }}>none yet</div>}
            {topSessions.map((s) => (
              <button key={s.id} className={route.page === "chat" && route.arg === s.id ? "on" : ""}
                onClick={() => go("chat", s.id)} title={s.title}>
                <span className={`dot ${busy[s.id] ? (busy[s.id] === "waiting" ? "waiting" : "busy") : ""}`} />
                <span className="t">{s.title || "new session"}</span>
              </button>
            ))}
          </div>
          <div className="rail-foot">
            {b && (
              <>
                <div className="row"><span className="grow">budget</span><span>{money(b.committed_usd)} / {money(total)}</span></div>
                <div className="budget-bar"><i style={{ width: `${total ? Math.min(100, (b.committed_usd / total) * 100) : 0}%` }} /></div>
              </>
            )}
            <div className="row">
              <span className={`conn ${connected ? "" : "off"}`}><i /> {connected ? "connected" : "reconnecting…"}</span>
              <span className="grow" />
              {status?.live?.version && <span className="badge good">{status.live.version}</span>}
            </div>
          </div>
        </aside>
        <main>
          {route.page === "chat" && <Chat sid={route.arg} />}
          {route.page === "data" && <Data name={route.arg} />}
          {route.page === "evals" && <Evals id={route.arg} />}
          {route.page === "models" && <Models />}
          {route.page === "jobs" && <Jobs id={route.arg} />}
          {route.page === "settings" && <Settings />}
        </main>
      </div>
      <Toasts />
    </Ctx.Provider>
  );
}
