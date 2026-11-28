import { ReactNode, useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";
import DOMPurify from "dompurify";
import { marked } from "marked";

// ------------------------------------------------------------------ icons (inline, 24-grid strokes)

const paths: Record<string, string> = {
  chat: "M4 5h16v11H8l-4 4z",
  data: "M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3zm0 0v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3",
  evals: "M4 20V10m6 10V4m6 16v-7m4 7H2",
  models: "M12 2l9 5v10l-9 5-9-5V7zm0 0v20M3 7l9 5 9-5",
  jobs: "M12 7v5l3 2m6-2a9 9 0 11-18 0 9 9 0 0118 0z",
  settings: "M12 15a3 3 0 100-6 3 3 0 000 6zm7.4-3a7.4 7.4 0 00-.1-1.2l2-1.6-2-3.4-2.4 1a7.3 7.3 0 00-2-1.2L14.5 2h-4l-.4 2.6a7.3 7.3 0 00-2 1.2l-2.4-1-2 3.4 2 1.6a7.4 7.4 0 000 2.4l-2 1.6 2 3.4 2.4-1a7.3 7.3 0 002 1.2l.4 2.6h4l.4-2.6a7.3 7.3 0 002-1.2l2.4 1 2-3.4-2-1.6c.1-.4.1-.8.1-1.2z",
  plus: "M12 5v14M5 12h14",
  send: "M5 12h14m-6-6l6 6-6 6",
  stop: "M7 7h10v10H7z",
  check: "M5 12l5 5L20 7",
  x: "M6 6l12 12M18 6L6 18",
  chevron: "M9 6l6 6-6 6",
  down: "M6 9l6 6 6-6",
  upload: "M12 16V4m0 0l-5 5m5-5l5 5M4 16v3a1 1 0 001 1h14a1 1 0 001-1v-3",
  trash: "M4 7h16M9 7V4h6v3m-8 0l1 13h8l1-13",
  play: "M7 5l12 7-12 7z",
  refresh: "M20 11a8 8 0 10-2.3 5.7M20 4v7h-7",
  spark: "M12 3v4m0 10v4M3 12h4m10 0h4M6 6l2.5 2.5m7 7L18 18M6 18l2.5-2.5m7-7L18 6",
  logo: "M4 15l4-4 3 3 6-7",
  shield: "M12 3l8 3v6c0 4.5-3.4 8.3-8 9-4.6-.7-8-4.5-8-9V6z",
  hf: "M12 3a9 9 0 100 18 9 9 0 000-18zM8.5 10h.01M15.5 10h.01M8.5 14.5c1.9 1.6 5.1 1.6 7 0",
  pen: "M4 20h4L19 9l-4-4L4 16zM14 6l4 4",
  file: "M6 3h8l4 4v14H6zM14 3v4h4",
  live: "M12 12m-3 0a3 3 0 106 0 3 3 0 10-6 0M5.6 5.6a9 9 0 000 12.8m12.8 0a9 9 0 000-12.8",
};

export function Icon({ name, className }: { name: keyof typeof paths | string; className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round"
      strokeLinejoin="round" className={className} aria-hidden="true">
      <path d={paths[name] || ""} />
    </svg>
  );
}

// ------------------------------------------------------------------ toasts

type Toast = { id: number; text: string; bad?: boolean };
let toasts: Toast[] = [];
const toastSubs = new Set<() => void>();
let toastSeq = 0;

export function toast(text: string, bad = false) {
  const id = ++toastSeq;
  toasts = [...toasts, { id, text, bad }];
  toastSubs.forEach((f) => f());
  setTimeout(() => {
    toasts = toasts.filter((t) => t.id !== id);
    toastSubs.forEach((f) => f());
  }, bad ? 7000 : 3500);
}

export function toastError(e: unknown) {
  toast(e instanceof Error ? e.message : String(e), true);
}

export function Toasts() {
  const list = useSyncExternalStore(
    (f) => {
      toastSubs.add(f);
      return () => void toastSubs.delete(f);
    },
    () => toasts,
  );
  return (
    <div className="toasts" role="status" aria-live="polite">
      {list.map((t) => (
        <div key={t.id} className={`toast ${t.bad ? "bad" : ""}`}>{t.text}</div>
      ))}
    </div>
  );
}

// ------------------------------------------------------------------ modal

export function Modal({ title, onClose, children, foot, wide }: {
  title: ReactNode; onClose: () => void; children: ReactNode; foot?: ReactNode; wide?: boolean;
}) {
  useEffect(() => {
    const k = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [onClose]);
  return (
    <div className="modal-back" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className={`modal ${wide ? "wide" : ""}`} role="dialog" aria-modal="true">
        <div className="modal-head">
          <h2>{title}</h2>
          <button className="btn ghost sm" onClick={onClose} aria-label="Close"><Icon name="x" /></button>
        </div>
        <div className="modal-body">{children}</div>
        {foot && <div className="modal-foot">{foot}</div>}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ small pieces

export function Progress({ value }: { value?: number | null }) {
  const indet = value === null || value === undefined;
  return (
    <div className={`progress ${indet ? "indet" : ""}`}>
      <i style={{ width: `${Math.round((value ?? 0) * 100)}%` }} />
    </div>
  );
}

export function Empty({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="empty">
      <h2>{title}</h2>
      {children && <p>{children}</p>}
      {action}
    </div>
  );
}

export function Markdown({ text }: { text: string }) {
  const html = DOMPurify.sanitize(marked.parse(text || "", { async: false, gfm: true, breaks: false }) as string);
  return <div className="md" dangerouslySetInnerHTML={{ __html: html }} />;
}

// ------------------------------------------------------------------ formatting

export function money(x?: number | null) {
  if (!x) return "$0";
  return x >= 0.01 ? `$${x.toFixed(2)}` : `$${x.toFixed(4)}`;
}

export function dur(s?: number | null) {
  if (s === null || s === undefined) return "";
  if (s < 1) return `${Math.round(s * 1000)}ms`;
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${String(Math.floor(s % 60)).padStart(2, "0")}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, "0")}m`;
}

export function ago(t?: number | string | null) {
  if (!t) return "";
  const ms = typeof t === "string" ? Date.parse(t) : t * 1000;
  const s = (Date.now() - ms) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return new Date(ms).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export function pct(x?: number | null) {
  return x === null || x === undefined ? "—" : `${(x * 100).toFixed(x >= 0.995 || x === 0 ? 0 : 1)}%`;
}

// ------------------------------------------------------------------ data hooks

export function useAsync<T>(fn: () => Promise<T>, deps: unknown[]) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const seq = useRef(0);
  const reload = useCallback(() => {
    const n = ++seq.current;
    setLoading(true);
    fn()
      .then((d) => {
        if (n === seq.current) {
          setData(d);
          setError(null);
        }
      })
      .catch((e) => n === seq.current && setError(e instanceof Error ? e.message : String(e)))
      .finally(() => n === seq.current && setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  useEffect(reload, [reload]);
  return { data, error, loading, reload, setData };
}

/** Re-run `fn` at most every `ms` while events keep arriving. */
export function throttle(fn: () => void, ms: number) {
  let t: ReturnType<typeof setTimeout> | null = null;
  return () => {
    if (t) return;
    t = setTimeout(() => {
      t = null;
      fn();
    }, ms);
  };
}

// ------------------------------------------------------------------ deciding by default

/** "Going ahead with X in 23s · wait": a free, easy-to-undo choice the agent will take by itself
 *  unless the person stops the countdown. Renders nothing once it is held (``at`` is null). */
export function AutoCountdown({ at, what, onHold }: { at?: number | null; what: string; onHold: () => void }) {
  const [now, setNow] = useState(Date.now() / 1000);
  useEffect(() => {
    if (!at) return;
    const t = setInterval(() => setNow(Date.now() / 1000), 500);
    return () => clearInterval(t);
  }, [at]);
  if (!at) return null;
  const left = Math.max(0, Math.ceil(at - now));
  return (
    <div className="auto-count small">
      <span className="auto-bar"><span style={{ width: `${Math.min(100, (left / 30) * 100)}%` }} /></span>
      <span>{left > 0 ? <>Going ahead with <b>{what}</b> in {left}s</> : <>Going ahead with <b>{what}</b>…</>}</span>
      {left > 0 && <button className="btn ghost sm" onClick={onHold}>wait, I'll decide</button>}
    </div>
  );
}
