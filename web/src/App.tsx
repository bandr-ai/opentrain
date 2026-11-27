import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { api, Board, events, Job, ModelRow, Session, Status } from "./api";
import { money, throttle, Toasts } from "./ui";
import { AgentPanel } from "./pages/Agent";
import BoardPage, { DropZone } from "./pages/Board";
import GridPage from "./pages/Grid";
import { EvalDetail } from "./pages/Evals";
import Jobs from "./pages/Jobs";
import Artifacts from "./pages/Artifacts";
import Settings from "./pages/Settings";

// ------------------------------------------------------------------ routing (hash)
//
//   #/            the board, at the stage the loop is on
//   #/s/<stage>   one stage: goal data grade baseline train prove ship
//   #/grid/<ds>   every held-out row against every model
//   #/evals/<id>  one eval · #/jobs[/<id>] jobs · #/settings
//   #/a[/<id>]    artifacts: the pages that lead each step (data card, eval comparison, ...)
//   #/data/<ds>   the data stage on that dataset · #/chat/<sid> opens that conversation beside the board

export type Route = { page: string; arg?: string };

function parse(hash: string): Route {
  const [page, ...rest] = hash.replace(/^#\/?/, "").split("/");
  return { page: page || "", arg: rest.length ? decodeURIComponent(rest.join("/")) : undefined };
}

export function go(page: string, arg?: string) {
  window.location.hash = `#/${page}${arg ? "/" + encodeURIComponent(arg) : ""}`;
}

// ------------------------------------------------------------------ shared state

type AppState = {
  status: Status | null;
  board: Board | null;
  sessions: Session[];
  busy: Record<string, string>;
  models: ModelRow[];
  defaultModel: string;
  jobs: Job[];
  sid: string | null;
  setSid: (s: string | null) => void;
  setDataset: (d: string) => void;
  reloadStatus: () => void;
  reloadBoard: () => void;
  reloadSessions: () => void;
  reloadModels: () => void;
};

const Ctx = createContext<AppState>(null as unknown as AppState);
export const useApp = () => useContext(Ctx);

const STAGE_KEYS = ["goal", "data", "grade", "baseline", "train", "prove", "ship"];

function stored(key: string): string | null {
  try { return localStorage.getItem(key); } catch { return null; }
}
function store(key: string, v: string | null) {
  try { if (v) localStorage.setItem(key, v); else localStorage.removeItem(key); } catch { /* private mode */ }
}

export default function App() {
  const [route, setRoute] = useState<Route>(parse(window.location.hash));
  const [status, setStatus] = useState<Status | null>(null);
  const [board, setBoard] = useState<Board | null>(null);
  const [dataset, setDatasetState] = useState<string | null>(stored("ot.dataset"));
  const [sessions, setSessions] = useState<Session[]>([]);
  const [busy, setBusy] = useState<Record<string, string>>({});
  const [models, setModels] = useState<ModelRow[]>([]);
  const [defaultModel, setDefaultModel] = useState("");
  const [jobs, setJobs] = useState<Job[]>([]);
  const [sid, setSidState] = useState<string | null>(stored("ot.session"));
  const [connected, setConnected] = useState(true);
  const [authError, setAuthError] = useState(false);

  const setSid = useCallback((s: string | null) => { setSidState(s); store("ot.session", s); }, []);
  const setDataset = useCallback((d: string) => { setDatasetState(d); store("ot.dataset", d); }, []);

  useEffect(() => {
    const f = () => {
      const r = parse(window.location.hash);
      if (r.page === "chat") { // a link to a conversation opens it
        if (r.arg && r.arg !== "new") setSid(r.arg); else setSid(null);
        window.location.replace("#/");
        return;
      }
      if (r.page === "data" && r.arg) {
        setDataset(r.arg);
        window.location.replace("#/s/data");
        return;
      }
      setRoute(r);
    };
    f();
    window.addEventListener("hashchange", f);
    return () => window.removeEventListener("hashchange", f);
  }, [setSid, setDataset]);

  useEffect(() => {
    const f = (e: globalThis.KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (e.metaKey || e.ctrlKey || e.altKey || t.closest("input, textarea, select, [contenteditable]")) return;
      const n = Number(e.key);
      if (n >= 1 && n <= STAGE_KEYS.length) go("s", STAGE_KEYS[n - 1]);
      else if (e.key === "Escape" || e.key === "0") { if (document.querySelector(".modal")) return; go(""); }
      else if (e.key === "/" || e.key === "c") document.getElementById("composer")?.focus();
      else if (e.key === "g" && board?.dataset) go("grid", board.dataset);
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", f);
    return () => window.removeEventListener("keydown", f);
  }, [board?.dataset]);

  const reloadStatus = useCallback(() => {
    api.get<Status>("/api/status").then(setStatus).catch((e) => e?.status === 401 && setAuthError(true));
  }, []);
  const reloadBoard = useCallback(() => {
    api.get<Board>(`/api/board${dataset ? `?dataset=${encodeURIComponent(dataset)}` : ""}`).then(setBoard)
      .catch((e) => e?.status === 401 && setAuthError(true));
  }, [dataset]);
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

  useEffect(() => { reloadBoard(); }, [reloadBoard]);

  // the newest conversation opens by default (the panel drops a conversation the server does not know)
  const [picked, setPicked] = useState(!!sid);
  useEffect(() => {
    if (picked || !sessions.length) return;
    if (!sid) setSid(sessions.find((s) => !s.parent_id)?.id || null);
    setPicked(true);
  }, [sessions, sid, setSid, picked]);

  useEffect(() => {
    reloadStatus();
    reloadSessions();
    reloadModels();
    reloadJobs();
    events.start();
    const offc = events.onConnection((c) => {
      setConnected(c);
      if (c) {
        reloadStatus();
        reloadSessions();
        reloadJobs();
      }
    });
    return offc;
  }, [reloadStatus, reloadSessions, reloadModels, reloadJobs]);

  useEffect(() => {
    const st = throttle(reloadStatus, 1500);
    const bd = throttle(reloadBoard, 700);
    const ss = throttle(reloadSessions, 800);
    const js = throttle(reloadJobs, 500);
    return events.on((e) => {
      if (e.type === "session.status") {
        setBusy((b) => {
          const n = { ...b };
          if (e.data.status === "idle") delete n[e.data.session_id];
          else n[e.data.session_id] = e.data.status;
          return n;
        });
      }
      if (e.type.startsWith("session.")) ss();
      if (e.type.startsWith("job.")) { js(); st(); bd(); }
      if (e.type.startsWith("proposal.")) bd();
      if (e.type === "part.updated" && e.data?.tool?.state === "completed") bd(); // a tool may have changed data
      if (e.type === "trainfile.created" || e.type.startsWith("approval.")) { st(); bd(); }
    });
  }, [reloadStatus, reloadBoard, reloadSessions, reloadJobs]);

  const state = useMemo(
    () => ({ status, board, sessions, busy, models, defaultModel, jobs, sid, setSid, setDataset, reloadStatus, reloadBoard, reloadSessions, reloadModels }),
    [status, board, sessions, busy, models, defaultModel, jobs, sid, setSid, setDataset, reloadStatus, reloadBoard, reloadSessions, reloadModels],
  );

  if (authError) {
    return (
      <div className="empty" style={{ paddingTop: "20vh" }}>
        <h2>Sign in from your terminal</h2>
        <p>This browser is not paired with the OpenTrain server. Run <code>opentrain web</code> in your project; it opens a one-time sign-in link.</p>
      </div>
    );
  }

  const running = jobs.filter((j) => j.state === "running");
  const b = board?.budget;
  const onChat = route.page === "";
  const waiting = Object.values(busy).some((s) => s === "waiting");

  // The conversation is the app. Every other view opens above the prompt; Esc comes back.
  let main: React.ReactNode = null;
  if (route.page === "s") main = <BoardPage stage={route.arg || undefined} />;
  else if (route.page === "grid") main = <GridPage dataset={route.arg || board?.dataset || ""} />;
  else if (route.page === "evals" && route.arg) main = <EvalDetail id={route.arg} />;
  else if (route.page === "evals") main = <BoardPage stage="baseline" />;
  else if (route.page === "models") main = <BoardPage stage="ship" />;
  else if (route.page === "jobs") main = <Jobs id={route.arg} />;
  else if (route.page === "settings") main = <Settings />;
  else if (route.page === "a") main = <Artifacts id={route.arg} />;

  return (
    <Ctx.Provider value={state}>
      <div className="app">
        <header className="top">
          <a className="brand" href="#/">opentrain</a>
          <span className="proj" title={board?.project.directory}>{board?.project.directory?.replace(/^\/Users\/[^/]+/, "~") || status?.project?.name || "…"}</span>
          {board?.dataset && <span className="ds" title="the dataset in focus">{board.dataset}</span>}
          <span className="grow" />
          {running.length > 0 && <a className="busy-t" href="#/jobs"><span className="dot busy" /> {running.length} running</a>}
          {waiting && <span className="err"><span className="dot waiting" /> needs you</span>}
          {b && <span className="budget" title={`${money(b.remaining_usd)} left of ${b.text}${b.reserved_usd ? `; ${money(b.reserved_usd)} held for running jobs until they report their real cost` : ""}`}>{money(b.spent_usd)} <span className="faint">spent</span>{b.reserved_usd ? <> <span className="faint">· {money(b.reserved_usd)} held</span></> : null}</span>}
          <a className="toplink" href="#/a">artifacts</a>
          <a className="toplink" href="#/jobs">jobs</a>
          <a className="toplink" href="#/settings">settings</a>
          <span className={`conn ${connected ? "" : "off"}`} title={connected ? "connected" : "reconnecting…"}><i /></span>
        </header>
        <div className={`body ${onChat ? "chat" : "view"}`}>
          {!onChat && (
            <main className="main">
              <div className="back"><a href="#/">← chat</a> <span className="faint">esc</span></div>
              {main}
            </main>
          )}
          <AgentPanel convo={onChat} />
        </div>
      </div>
      <DropZone />
      <Toasts />
    </Ctx.Provider>
  );
}
