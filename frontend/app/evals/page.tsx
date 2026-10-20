"use client";

import { useEffect, useState } from "react";
import { api, EvalRun } from "@/lib/api";

// metric key -> [label, format]
const METRICS: [string, string, (v: number) => string][] = [
  ["f1", "F1", (v) => v.toFixed(2)],
  ["precision", "Precision", (v) => v.toFixed(2)],
  ["recall", "Recall", (v) => v.toFixed(2)],
  ["pattern_accuracy", "Pattern accuracy", (v) => `${(v * 100).toFixed(0)}%`],
  ["false_block_rate", "False block rate", (v) => `${(v * 100).toFixed(0)}%`],
  ["legit_approved_rate", "Legit cases approved", (v) => `${(v * 100).toFixed(0)}%`],
  ["injection_success_rate", "Injection success rate", (v) => `${(v * 100).toFixed(0)}%`],
  ["pii_leak_rate", "PII leak rate", (v) => `${(v * 100).toFixed(0)}%`],
  ["avg_tool_calls", "Avg tool calls", (v) => v.toFixed(1)],
  ["avg_tokens", "Avg tokens / case", (v) => v.toLocaleString()],
  ["latency_p50_s", "Latency p50", (v) => `${v}s`],
  ["latency_p95_s", "Latency p95", (v) => `${v}s`],
];

export default function Evals() {
  const [run, setRun] = useState<EvalRun | null | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.latestEval().then(setRun).catch((e) => setError(String(e)));
  }, []);

  if (error) return <div className="error">{error}</div>;
  if (run === undefined) return <p className="muted">Loading…</p>;
  if (run === null) {
    return (
      <section className="card">
        <h2>No eval results yet</h2>
        <p>Run the eval (it calls OpenAI once per case), then refresh this page:</p>
        <pre>docker compose run --rm api python -m evals.run_eval --limit 12</pre>
      </section>
    );
  }

  return (
    <>
      <h1>Latest eval</h1>
      <p className="muted">
        {run.meta.run_at} · model <code>{run.meta.model}</code> · knowledge base <code>{run.meta.kb}</code> ·{" "}
        {run.summary.cases} cases
      </p>
      <section className="metrics" style={{ marginBottom: 20 }}>
        {METRICS.filter(([k]) => k in run.summary).map(([k, lbl, fmt]) => (
          <div className="metric" key={k}>
            <div className="label">{lbl}</div>
            <div className="value">{fmt(run.summary[k])}</div>
          </div>
        ))}
      </section>
      <section className="card">
        <h2>Cases</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Transaction</th><th>Scenario</th><th>Expected</th><th>Verdict</th>
                <th>Pattern</th><th>Action</th><th>Tools</th><th>Seconds</th>
              </tr>
            </thead>
            <tbody>
              {run.cases.map((c) => {
                const correct = c.expected_fraud === (c.verdict !== "legitimate");
                return (
                  <tr key={c.transaction_id}>
                    <td>{c.transaction_id}</td>
                    <td>{c.tag.replace(/_/g, " ")}</td>
                    <td>{c.expected_fraud ? "fraud" : "legit"}</td>
                    <td>
                      <span className={`chip ${c.verdict}`}>{c.verdict}</span> {correct ? "✓" : "✗"}
                    </td>
                    <td>{c.pattern.replace(/_/g, " ")}</td>
                    <td><span className={`chip ${c.action}`}>{c.action.replace(/_/g, " ")}</span></td>
                    <td className="num">{c.tool_calls}</td>
                    <td className="num">{c.latency_s}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}
