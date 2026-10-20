"use client";

import { useCallback, useEffect, useState } from "react";
import FlaggedTable from "@/components/FlaggedTable";
import ReportCard from "@/components/ReportCard";
import Uploader from "@/components/Uploader";
import { api, ApiError, DatasetSummary, FlaggedTxn, Investigation } from "@/lib/api";

const MAX_SELECTED = 10;
const STORAGE_KEY = "fraudshield.dataset";

function remember(id: string | null) {
  try {
    if (id) localStorage.setItem(STORAGE_KEY, id);
    else localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* storage blocked: the page still works, it just forgets on reload */
  }
}

export default function Home() {
  const [dataset, setDataset] = useState<DatasetSummary | null>(null);
  const [flagged, setFlagged] = useState<FlaggedTxn[]>([]);
  const [results, setResults] = useState<Investigation[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState(false);
  const [running, setRunning] = useState<{ done: number; total: number; current: string } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const open = useCallback(async (summary: DatasetSummary) => {
    setDataset(summary);
    remember(summary.dataset_id);
    const [rows, res] = await Promise.all([api.flagged(summary.dataset_id), api.results(summary.dataset_id)]);
    setFlagged(rows);
    setResults(res.reverse());
    setSelected(new Set());
  }, []);

  // Re-open the last dataset after a page reload (the API keeps it in memory).
  useEffect(() => {
    let id: string | null = null;
    try {
      id = localStorage.getItem(STORAGE_KEY);
    } catch {
      /* ignore */
    }
    if (id) api.dataset(id).then(open).catch(() => remember(null));
  }, [open]);

  async function load(fn: () => Promise<DatasetSummary>) {
    setBusy(true);
    setError(null);
    try {
      await open(await fn());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function investigate() {
    if (!dataset) return;
    const ids = flagged.map((r) => r.transaction_id).filter((id) => selected.has(id));
    setError(null);
    for (let i = 0; i < ids.length; i++) {
      setRunning({ done: i, total: ids.length, current: ids[i] });
      try {
        const inv = await api.investigate(dataset.dataset_id, ids[i]);
        setResults((prev) => [inv, ...prev.filter((p) => p.transaction_id !== inv.transaction_id)]);
        setFlagged((prev) => prev.map((r) => (r.transaction_id === ids[i] ? { ...r, investigated: true } : r)));
      } catch (e) {
        setError(`${ids[i]}: ${e instanceof ApiError ? e.message : String(e)}`);
        break;
      }
    }
    setRunning(null);
    setSelected(new Set());
  }

  function toggle(id: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else if (next.size < MAX_SELECTED) next.add(id);
      return next;
    });
  }

  return (
    <>
      {error && <div className="error">{error}</div>}
      <Uploader busy={busy} onFile={(f) => load(() => api.upload(f))} onSample={() => load(api.loadSample)} />

      {dataset && (
        <>
          <section className="metrics" style={{ marginBottom: 20 }}>
            <Metric label="File" value={dataset.name} small />
            <Metric label="Transactions" value={dataset.rows.toLocaleString()} />
            <Metric label="Flagged by model" value={dataset.flagged.toLocaleString()} />
            <Metric label={`Sent to agent (threshold ${dataset.threshold})`} value={`${(dataset.flag_rate * 100).toFixed(1)}%`} />
            {dataset.has_labels && (
              <Metric
                label="Model precision / recall"
                value={`${dataset.model_precision?.toFixed(2)} / ${dataset.model_recall?.toFixed(2)}`}
              />
            )}
          </section>

          <section className="card">
            <div className="row" style={{ marginBottom: 12 }}>
              <h2 style={{ margin: 0 }}>Flagged transactions</h2>
              <span className="spacer" />
              <span className="muted">{selected.size}/{MAX_SELECTED} selected</span>
              <button className="btn primary" disabled={selected.size === 0 || running !== null} onClick={investigate}>
                Investigate selected
              </button>
            </div>
            {running && (
              <div style={{ marginBottom: 12 }}>
                <span className="muted">
                  Investigating {running.current} ({running.done + 1} of {running.total})…
                </span>
                <div className="progress"><span style={{ width: `${(running.done / running.total) * 100}%` }} /></div>
              </div>
            )}
            <FlaggedTable rows={flagged} selected={selected} onToggle={toggle} max={MAX_SELECTED} />
          </section>

          {results.length > 0 && <h2>Case reports</h2>}
          {results.map((inv) => <ReportCard key={inv.transaction_id} inv={inv} />)}
        </>
      )}
    </>
  );
}

function Metric({ label, value, small }: { label: string; value: string; small?: boolean }) {
  return (
    <div className="metric">
      <div className="label">{label}</div>
      <div className="value" style={small ? { fontSize: 15, wordBreak: "break-all" } : undefined}>{value}</div>
    </div>
  );
}
