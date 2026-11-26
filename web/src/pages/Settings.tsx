import { useEffect, useState } from "react";
import { api } from "../api";
import { useApp } from "../App";
import { ago, Icon, Modal, toast, toastError, useAsync } from "../ui";

type Policy = { path: string; exists: boolean; text: string; parsed: any; error: string | null };
type Integration = { id: string; name: string; kind: string; env?: string | null; url?: string; connected: boolean; via?: string | null; hint?: string | null };
type Trainer = { name: string; available: boolean; detail: string; credential?: string | null; base_models: { id: string; params?: string; note?: string }[] };
type Approval = { id: string; action: string; resource: string; time?: { created?: number } };

export default function Settings() {
  return (
    <div className="page"><div className="page-inner">
      <div className="page-head"><div><h1>Settings</h1><p>Policy, trainer and connected accounts.</p></div></div>
      <Trainfile />
      <TrainerCard />
      <Connections />
      <Approvals />
    </div></div>
  );
}

function Trainfile() {
  const { data, reload } = useAsync(() => api.get<Policy>("/api/policy"), []);
  const { reloadStatus } = useApp();
  const [text, setText] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => { if (data) setText(data.text); }, [data]);
  const dirty = data && text !== data.text;
  const save = async () => {
    setBusy(true);
    try {
      const r = await fetch("/api/policy", { method: "POST", credentials: "same-origin", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
      const j = await r.json();
      if (!r.ok || j.error) { setErr(typeof j.error === "string" ? j.error : j.error?.message || "invalid"); return; }
      setErr(null);
      toast("Trainfile saved");
      reload();
      reloadStatus();
    } catch (e) { toastError(e); } finally { setBusy(false); }
  };
  return (
    <div className="card" style={{ marginBottom: 20 }}>
      <div className="card-head">
        <h3>Trainfile</h3>
        <span className="small faint" title={data?.path}>the policy the server enforces, line by line</span>
      </div>
      <div className="card-pad">
        <textarea className="textarea editor" value={text} onChange={(e) => { setText(e.target.value); setErr(null); }} spellCheck={false} />
        <div className="row" style={{ marginTop: 10 }}>
          <span className="grow small">{err ? <span className="err">{err}</span> : !data?.exists ? <span className="faint">Not saved yet: these are the defaults.</span> : null}</span>
          {dirty && <button className="btn" onClick={() => setText(data!.text)}>Revert</button>}
          <button className="btn primary" disabled={busy || (!dirty && data?.exists)} onClick={save}>Save</button>
        </div>
      </div>
    </div>
  );
}

function TrainerCard() {
  const { data, reload } = useAsync(() => api.get<{ trainers: Trainer[]; policy: any }>("/api/trainer"), []);
  const { reloadStatus, models, defaultModel, reloadModels } = useApp();
  const [base, setBase] = useState("");
  useEffect(() => { if (data?.policy) setBase(data.policy.base_model); }, [data]);
  const set = async (body: any) => {
    try {
      await api.post("/api/trainer", body);
      toast("Updated the Trainfile");
      reload();
      reloadStatus();
    } catch (e) { toastError(e); }
  };
  const current = data?.policy?.trainer;
  const t = data?.trainers.find((x) => x.name === current);
  return (
    <div className="card" style={{ marginBottom: 20 }}>
      <div className="card-head"><h3>Training and the engine</h3></div>
      <div className="card-pad">
        <div className="field">
          <span>Trainer</span>
          <div className="row" style={{ flexWrap: "wrap", gap: 8 }}>
            {data?.trainers.filter((x) => x.name !== "fake").map((x) => (
              <button key={x.name} className={`btn ${x.name === current ? "primary" : ""}`} onClick={() => set({ trainer: x.name })} title={x.detail}>
                {x.name} {x.available ? <Icon name="check" /> : <span className="faint small">needs setup</span>}
              </button>
            ))}
          </div>
          {t && <small className={t.available ? "ok" : "err"}>{t.detail}</small>}
        </div>
        <div className="field">
          <span>Base model</span>
          <div className="row">
            <input className="input mono" list="bases" value={base} onChange={(e) => setBase(e.target.value)} />
            <datalist id="bases">{(t?.base_models || []).map((b) => <option key={b.id} value={b.id}>{[b.params, b.note].filter(Boolean).join(" · ")}</option>)}</datalist>
            <button className="btn" disabled={!base || base === data?.policy?.base_model} onClick={() => set({ base_model: base })}>Set</button>
          </div>
          <small>The open model training starts from.</small>
        </div>
        <div className="field" style={{ marginBottom: 0 }}>
          <span>Engine model (what the agent runs on, and the default judge)</span>
          <select className="select" value={defaultModel} onChange={async (e) => {
            try { await api.post("/api/model/default", { ref: e.target.value }); reloadModels(); toast("Default model set"); } catch (er) { toastError(er); }
          }}>
            {!models.some((m) => m.ref === defaultModel) && <option value={defaultModel}>{defaultModel}</option>}
            {Object.entries(models.filter((m) => m.connected).reduce<Record<string, typeof models>>((g, m) => ((g[m.provider_name] ||= []).push(m), g), {})).map(([g, ms]) => (
              <optgroup key={g} label={g}>{ms.map((m) => <option key={m.ref} value={m.ref}>{m.name}</option>)}</optgroup>
            ))}
          </select>
        </div>
      </div>
    </div>
  );
}

function Connections() {
  const { data, reload } = useAsync(() => api.get<Integration[]>("/api/integration"), []);
  const { reloadModels, reloadStatus } = useApp();
  const [connecting, setConnecting] = useState<Integration | null>(null);
  const [key, setKey] = useState("");
  const connect = async () => {
    try {
      await api.post(`/api/integration/${connecting!.id}/connect`, { key });
      toast(`${connecting!.name} connected`);
      setConnecting(null);
      setKey("");
      reload();
      reloadModels();
      reloadStatus();
    } catch (e) { toastError(e); }
  };
  const disconnect = async (i: Integration) => {
    if (!confirm(`Remove the stored key for ${i.name}?`)) return;
    try { await api.del(`/api/integration/${i.id}`); reload(); reloadModels(); } catch (e) { toastError(e); }
  };
  const groups: Record<string, Integration[]> = {};
  for (const i of data || []) (groups[i.kind] ||= []).push(i);
  return (
    <div className="card" style={{ marginBottom: 20 }}>
      <div className="card-head"><h3>Connections</h3><span className="small faint">keys are stored on this machine only (0600)</span></div>
      <table className="table">
        <tbody>
          {Object.entries(groups).map(([kind, items]) => [
            <tr key={kind}><th colSpan={3}>{kind === "provider" ? "Model providers" : kind === "trainer" ? "Trainers" : kind}</th></tr>,
            ...items.map((i) => (
              <tr key={i.id}>
                <td style={{ width: "40%" }}><b>{i.name}</b>{i.url && <> · <a href={i.url} target="_blank" rel="noreferrer" className="small">site</a></>}</td>
                <td className="small">{i.connected ? <span className="badge good">{i.via === "connected" ? `key stored${i.hint ? ` (${i.hint})` : ""}` : i.via?.startsWith("$") ? `from ${i.via}` : "found"}</span> : <span className="faint">{i.env ? `set ${i.env} or add a key` : "not found"}</span>}</td>
                <td className="num">
                  {i.env && (i.via === "connected" ? <button className="btn sm" onClick={() => disconnect(i)}>Remove</button>
                    : <button className="btn sm" onClick={() => setConnecting(i)}>{i.connected ? "Replace key" : "Connect"}</button>)}
                </td>
              </tr>
            )),
          ])}
        </tbody>
      </table>
      {connecting && (
        <Modal title={`Connect ${connecting.name}`} onClose={() => setConnecting(null)} foot={<>
          <button className="btn" onClick={() => setConnecting(null)}>Cancel</button>
          <button className="btn primary" disabled={!key.trim()} onClick={connect}>Save key</button>
        </>}>
          <label className="field"><span>API key</span>
            <input className="input mono" type="password" autoFocus value={key} onChange={(e) => setKey(e.target.value)} onKeyDown={(e) => e.key === "Enter" && key.trim() && connect()} />
            {connecting.env && <small>Or set {connecting.env} in the environment the server runs in.</small>}
          </label>
        </Modal>
      )}
    </div>
  );
}

function Approvals() {
  const { data, reload } = useAsync(() => api.get<Approval[]>("/api/approval"), []);
  return (
    <div className="card">
      <div className="card-head"><h3>Saved approvals</h3><span className="small faint">things you said "always allow" to</span></div>
      {!data?.length ? <div className="card-pad small muted">None. Spending is always asked, every time.</div> : (
        <table className="table">
          <tbody>{data.map((a) => (
            <tr key={a.id}>
              <td className="mono">{a.action}</td><td className="mono small">{a.resource}</td><td className="muted small">{ago(a.time?.created)}</td>
              <td className="num"><button className="btn sm" onClick={() => api.del(`/api/approval/${a.id}`).then(reload).catch(toastError)}>Revoke</button></td>
            </tr>
          ))}</tbody>
        </table>
      )}
    </div>
  );
}
