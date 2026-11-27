import { Board, ScoreRow } from "../api";
import { go } from "../App";
import { pct } from "../ui";
import { fillComposer } from "./Agent";

// The one card that matters: the models worth knowing about, side by side, and the next step.

type Pick = { row: ScoreRow; tag: string };

function picks(rows: ScoreRow[], base?: string): Pick[] {
  if (!rows.length) return [];
  const out: Pick[] = [];
  const add = (row: ScoreRow | undefined, tag: string) => {
    if (!row) return;
    const have = out.find((p) => p.row.target === row.target);
    if (have) have.tag += ` · ${tag}`;
    else out.push({ row, tag });
  };
  const best = rows[0];
  add(best, "best");
  const close = rows.filter((r) => r.score >= best.score - 0.02);
  add([...close].sort((a, b) => a.cost_per_1k - b.cost_per_1k)[0], "cheapest at ~best");
  add(rows.find((r) => r.live), "live");
  add(rows.find((r) => r.kind === "version" && !r.live), "trained");
  if (base) add(rows.find((r) => r.kind === "model" && r.target.endsWith(base)), "base, untrained");
  if (rows.length === 1) {
    // one model measured: "best" and "cheapest" say nothing yet; keep only what it is
    out[0].tag = out[0].tag.split(" · ").filter((t) => t !== "best" && t !== "cheapest at ~best").join(" · ") || "measured";
  }
  return out.slice(0, 5);
}

const per1k = (v: number) => (v <= 0 ? "free" : v < 0.01 ? "<$0.01" : `$${v < 10 ? v.toFixed(2) : Math.round(v)}`);
const secs = (v?: number | null) => (v == null ? "—" : `${v < 10 ? v.toFixed(1) : Math.round(v)}s`);
const short = (t: string) => t.replace(/^claude-code\//, "cc/").replace(/^modal\//, "");

const NEXT: Record<string, [string, string]> = {
  goal: ["say what to improve", ""],
  data: ["add examples", "Help me get examples for this: "],
  grade: ["set how answers are graded", "Set up how answers on this dataset are graded."],
  baseline: ["measure a few models", "Measure a strong model and a small open one on the held-out rows, so we know the number to beat and what it costs."],
  train: ["train a small model", "Train a small open model on this data and score it on the held-out rows."],
  prove: ["score the trained model", "Score the newest trained version on the held-out rows against the best baseline."],
  ship: ["ship it", "Ship the newest version if it passed the gate."],
};

export function Scorecard({ board }: { board: Board }) {
  const rows = picks(board.scoreboard, board.policy.base_model);
  const running = board.stages.find((s) => s.state === "running");
  const waiting = board.proposals[0];
  const [label, text] = NEXT[board.current] || ["", ""];
  const done = board.stages.filter((s) => s.state === "done").length;
  const finished = done === board.stages.length;

  return (
    <section className="score">
      {rows.length > 0 ? (
        <table className="score-t">
          <thead><tr><th /><th>quality</th><th title="cost per 1,000 answers">$/1k</th><th>speed</th></tr></thead>
          <tbody>
            {rows.map(({ row, tag }) => (
              <tr key={row.target} className="click" onClick={() => go("grid", board.dataset || "")} title="see every row">
                <td><span className="mono">{short(row.target)}</span> <span className="tag">{tag}</span></td>
                <td><b>{pct(row.score)}</b></td>
                <td>{per1k(row.cost_per_1k)}</td>
                <td>{secs(row.median_seconds)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <div className="score-empty">{board.dataset ? <>no scores yet on <span className="mono">{board.dataset}</span></> : "no data yet"}</div>
      )}
      <div className="score-next">
        <span className="pips" title={board.stages.map((s) => `${s.label}: ${s.state}`).join("\n")}>
          {board.stages.map((s) => <i key={s.key} className={s.state} />)}
        </span>
        {waiting ? <span className="err">waiting for your OK ↓</span>
          : running ? <span><span className="dot busy" /> {running.summary}</span>
          : finished ? <span className="ok">✓ {board.stages[board.stages.length - 1].summary}</span>
          : <span className="muted">next: <b>{label}</b></span>}
        <span className="grow" />
        {!waiting && !running && !finished && text && (
          <button className="btn sm primary" onClick={() => fillComposer(text, true)}>do it</button>
        )}
        {!waiting && !running && board.current === "goal" && (
          <button className="btn sm" onClick={() => document.getElementById("composer")?.focus()}>write it</button>
        )}
        <button className="btn sm ghost" onClick={() => go("s", board.current)} title={`${done}/7 steps done`}>details</button>
      </div>
    </section>
  );
}
