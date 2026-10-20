"use client";

import { FlaggedTxn } from "@/lib/api";

type Props = {
  rows: FlaggedTxn[];
  selected: Set<string>;
  onToggle: (id: string) => void;
  max: number;
};

const money = (n: number) => n.toLocaleString("en-US", { style: "currency", currency: "USD" });

export default function FlaggedTable({ rows, selected, onToggle, max }: Props) {
  if (rows.length === 0) return <p className="muted">The model did not flag any transactions.</p>;
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th></th>
            <th>Transaction</th>
            <th>Time (UTC)</th>
            <th>Customer</th>
            <th style={{ textAlign: "right" }}>Amount</th>
            <th>Merchant</th>
            <th>Country</th>
            <th>Card</th>
            <th>Score</th>
            <th>Why flagged</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const isSelected = selected.has(r.transaction_id);
            return (
              <tr key={r.transaction_id} className={isSelected ? "selected" : ""}>
                <td>
                  <input
                    type="checkbox"
                    aria-label={`Select ${r.transaction_id}`}
                    checked={isSelected}
                    disabled={!isSelected && selected.size >= max}
                    onChange={() => onToggle(r.transaction_id)}
                  />
                </td>
                <td>
                  {r.transaction_id}
                  {r.investigated && <span className="chip on" style={{ marginLeft: 6 }}>done</span>}
                </td>
                <td>{r.timestamp.slice(0, 16).replace("T", " ")}</td>
                <td>{r.customer_id}</td>
                <td className="num">{money(r.amount)}</td>
                <td>
                  {r.merchant}
                  <div className="reasons">{r.merchant_category} · {r.channel.replace("_", " ")}</div>
                </td>
                <td>{r.country}</td>
                <td style={{ fontFamily: "monospace" }}>{r.card || "—"}</td>
                <td>
                  <div className="score">
                    <div className="bar"><span style={{ width: `${r.fraud_score * 100}%` }} /></div>
                    {r.fraud_score.toFixed(2)}
                  </div>
                </td>
                <td className="reasons">{r.reasons.join("; ") || "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
