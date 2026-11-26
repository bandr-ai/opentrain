// The OpenTrain server's HTTP API and its live event stream. The browser is signed in by a
// one-time pairing link (`opentrain web`), which sets a cookie; every call is same-origin.

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method,
    credentials: "same-origin",
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await res.text();
  let data: any = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { error: text };
  }
  if (!res.ok) {
    const msg = (data && (data.error?.message || data.error || data.message)) || `${res.status} ${res.statusText}`;
    throw new ApiError(res.status, typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return data as T;
}

export const api = {
  get: <T,>(p: string) => call<T>("GET", p),
  post: <T,>(p: string, b: unknown = {}) => call<T>("POST", p, b),
  patch: <T,>(p: string, b: unknown) => call<T>("PATCH", p, b),
  del: <T,>(p: string) => call<T>("DELETE", p),
};

// ------------------------------------------------------------------ types

export type Part = {
  id: string;
  type: "text" | "reasoning" | "tool" | string;
  message_id: string;
  session_id: string;
  text?: string;
  tool?: {
    name: string;
    call_id: string;
    state: "pending" | "running" | "completed" | "error";
    input: Record<string, any>;
    output?: string;
    error?: string;
    title?: string;
    metadata?: Record<string, any>;
    time?: { start?: number; end?: number };
  };
};

export type Message = {
  id: string;
  session_id: string;
  role: "user" | "assistant" | "synthetic" | "compaction";
  agent?: string;
  model?: string;
  finish?: string;
  error?: string;
  cost?: number;
  billing?: string;
  tokens?: { input?: number; output?: number; cache_read?: number };
  time?: { created?: number; completed?: number };
  parts: Part[];
};

export type Session = {
  id: string;
  title?: string;
  agent?: string;
  model?: string;
  parent_id?: string;
  time?: { created?: number; updated?: number };
  metadata?: Record<string, any>;
};

export type Job = {
  id: string;
  session_id?: string;
  kind: string;
  title: string;
  state: "running" | "succeeded" | "failed" | "cancelled";
  progress?: number | null;
  detail?: string;
  cost_usd?: number;
  estimate_usd?: number;
  version?: string;
  dataset?: string;
  error?: string;
  result?: any;
  loss?: number | null;
  curve?: [number, number][];
  reward?: number | null;
  reward_curve?: [number, number][];
  method?: string;
  time?: { created?: number; started?: number; ended?: number };
};

export type Permission = {
  id: string;
  session_id: string;
  agent: string;
  action: string;
  resources: string[];
  metadata: Record<string, any>;
  status: string;
};

export type Question = {
  id: string;
  session_id: string;
  questions: { question: string; options: string[]; multiple?: boolean }[];
};

export type DatasetSummary = {
  name: string;
  description?: string;
  counts: { train: number; heldout: number; labelled: number };
  grader: string;
  system?: string;
  sources: Record<string, any>[];
  latest_scores: Record<string, { eval: string; score: number; pass_rate: number; rows: number }>;
  updated?: number;
};

export type Row = { id: string; messages: { role: string; content: string }[]; meta?: Record<string, any> };

export type EvalSummary = {
  id: string;
  dataset: string;
  target: string;
  split: "heldout" | "train";
  time: number;
  summary: {
    rows: number;
    passed: number;
    mean_score: number;
    pass_rate: number;
    errors: number;
    median_seconds: number | null;
    cost_usd: number;
  };
};

export type EvalRow = {
  id: string;
  score: number;
  passed: boolean;
  seconds?: number;
  tokens?: number;
  error?: string;
  prompt?: string;
  reference?: string;
  answer?: string;
  reason?: string;
};

export type Version = {
  version: string;
  base_model: string;
  backend: string;
  checkpoint: string;
  decision: string;
  created_at: string;
  dataset?: string | null;
  scores?: Record<string, any>;
};

export type ModelRow = {
  ref: string;
  name: string;
  provider: string;
  provider_name: string;
  connected: boolean;
  context?: number;
  input_price?: number;
  output_price?: number;
};

export type Status = {
  trainfile: string;
  policy: Record<string, any>;
  budget: { committed_usd: number; remaining_usd: number };
  datasets: DatasetSummary[];
  coding_tasks: { training: number; heldout: number };
  live: { version: string | null; base_model?: string };
  jobs_running: Job[];
  sandbox: string;
  project: { name: string; directory: string };
};

// ------------------------------------------------------------------ live events

export type Proposal = {
  id: string;
  session_id: string;
  title: string;
  why: string;
  cost_usd: number;
  minutes?: number | null;
  needs: string[];
  stage?: string | null;
  dataset?: string | null;
  status: string;
  agent?: string;
};

export type StageKey = "goal" | "data" | "grade" | "baseline" | "train" | "prove" | "ship";
export type Stage = { key: StageKey; label: string; state: "done" | "running" | "todo"; summary: string; current: boolean };

export type ScoreRow = {
  target: string;
  kind: "model" | "version";
  eval: string;
  score: number;
  pass_rate: number;
  passed: number;
  rows: number;
  errors: number;
  median_seconds?: number | null;
  cost_usd: number;
  cost_per_1k: number;
  live: boolean;
  time: number;
};

export type TrainRun = {
  id: string;
  title: string;
  state: string;
  progress?: number | null;
  detail?: string;
  cost_usd?: number;
  estimate_usd?: number | null;
  loss?: number | null;
  curve?: [number, number][] | null;
  version?: string | null;
  error?: string | null;
  backend?: string;
  time?: { created?: number; started?: number; ended?: number };
};

export type Gate = {
  promote: boolean;
  reasons: string[];
  rows?: number;
  candidate?: string;
  baseline?: string;
  candidate_score?: number;
  baseline_score?: number;
  fixed?: string[];
  regressed?: string[];
};

export type Board = {
  project: { name: string; directory: string };
  goal: string;
  trainfile: boolean;
  policy: { trainer: string; base_model: string };
  budget: { spent_usd: number; remaining_usd: number; limit_usd: number; text: string };
  datasets: DatasetSummary[];
  dataset: string | null;
  focus: DatasetSummary | null;
  stages: Stage[];
  current: StageKey;
  scoreboard: ScoreRow[];
  versions: { version: string; base_model: string; backend: string; decision: string; created_at: string; live: boolean }[];
  live: { version: string | null; dataset: string | null };
  training: TrainRun[];
  gate: Gate | null;
  running: { id: string; kind: string; title: string; progress?: number | null; detail?: string; dataset?: string; target?: string }[];
  proposals: Proposal[];
};

export type GridCell = { passed: boolean; score: number; error?: string | null; answer?: string | null; reason?: string | null };
export type GridRow = { id: string; cells: Record<string, GridCell>; disagree: boolean; prompt?: string; reference?: string | null };
export type Grid = { dataset: string; columns: ScoreRow[]; rows: GridRow[]; revealed: boolean };

export type ServerEvent = { type: string; data: any; time: number; seq?: number };
type Listener = (e: ServerEvent) => void;

class EventHub {
  private listeners = new Set<Listener>();
  private source: EventSource | null = null;
  connected = false;
  private connListeners = new Set<(c: boolean) => void>();

  start() {
    if (this.source) return;
    const open = () => {
      const es = new EventSource("/api/event", { withCredentials: true });
      this.source = es;
      es.onopen = () => this.setConnected(true);
      es.onerror = () => {
        this.setConnected(false);
        if (es.readyState === EventSource.CLOSED) {
          this.source = null;
          setTimeout(open, 2000);
        }
      };
      es.onmessage = (m) => this.dispatch(m.data);
      // the server names each event; EventSource needs a listener per name, so catch them all
      const forward = (m: MessageEvent) => this.dispatch(m.data);
      for (const t of KNOWN_TYPES) es.addEventListener(t, forward as EventListener);
    };
    open();
  }

  private setConnected(c: boolean) {
    if (c === this.connected) return;
    this.connected = c;
    this.connListeners.forEach((l) => l(c));
  }

  private dispatch(raw: string) {
    try {
      const e = JSON.parse(raw) as ServerEvent;
      this.listeners.forEach((l) => l(e));
    } catch {
      /* heartbeat or junk */
    }
  }

  on(l: Listener) {
    this.listeners.add(l);
    return () => void this.listeners.delete(l);
  }

  onConnection(l: (c: boolean) => void) {
    this.connListeners.add(l);
    return () => void this.connListeners.delete(l);
  }
}

const KINDS = ["session", "message", "part", "inbox", "job", "todo", "approval", "question", "permission", "proposal"];
const KNOWN_TYPES = [
  "server.connected",
  "part.delta",
  "session.status",
  "trainfile.created",
  "todo.updated",
  ...KINDS.flatMap((k) => [`${k}.created`, `${k}.updated`, `${k}.deleted`]),
];

export const events = new EventHub();
