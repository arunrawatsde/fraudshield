import { Investigation } from "@/lib/api";

const label = (s: string) => s.replace(/_/g, " ");

export default function ReportCard({ inv }: { inv: Investigation }) {
  const r = inv.report;
  return (
    <article className="card report">
      <h3>
        {inv.transaction_id}
        <span className={`chip ${r.verdict}`}>{r.verdict}</span>
        <span className={`chip ${r.recommended_action}`}>action: {label(r.recommended_action)}</span>
        {r.pattern !== "none" && <span className="chip">{label(r.pattern)}</span>}
        <span className="muted" style={{ fontSize: 14, fontWeight: 400 }}>
          confidence {r.confidence.toFixed(2)}
        </span>
      </h3>
      <p>{r.summary}</p>
      {r.evidence.length > 0 && (
        <>
          <strong>Evidence</strong>
          <ul>{r.evidence.map((e, i) => <li key={i}>{e}</li>)}</ul>
        </>
      )}
      {r.playbook_sections.length > 0 && (
        <p className="muted" style={{ fontSize: 13 }}>Playbook: {r.playbook_sections.join(" · ")}</p>
      )}
      <details>
        <summary>Tool calls ({inv.tool_calls.length})</summary>
        <pre>{JSON.stringify(inv.tool_calls, null, 2)}</pre>
      </details>
      <details>
        <summary>Guardrail events ({inv.guard_events.length})</summary>
        <pre>{JSON.stringify(inv.guard_events, null, 2)}</pre>
      </details>
      <p className="muted" style={{ fontSize: 13, marginBottom: 0 }}>
        {inv.latency_s}s · {inv.input_tokens.toLocaleString()} input / {inv.output_tokens.toLocaleString()} output tokens
      </p>
    </article>
  );
}
