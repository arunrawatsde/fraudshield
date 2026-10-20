"use client";

import { useRef, useState } from "react";

type Props = {
  busy: boolean;
  onFile: (file: File) => void;
  onSample: () => void;
};

export default function Uploader({ busy, onFile, onSample }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);

  return (
    <section className="card">
      <h2>Load transactions</h2>
      <div
        className={`dropzone ${drag ? "drag" : ""}`}
        onClick={() => input.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          const file = e.dataTransfer.files[0];
          if (file) onFile(file);
        }}
      >
        <strong>{busy ? "Scoring…" : "Drop a CSV here or click to choose"}</strong>
        <div className="muted" style={{ fontSize: 13, marginTop: 6 }}>
          Required: transaction_id, timestamp, customer_id, amount, merchant, merchant_category,
          country, device_id, channel
        </div>
        <input
          ref={input}
          type="file"
          accept=".csv,text/csv"
          hidden
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onFile(file);
            e.target.value = "";
          }}
        />
      </div>
      <div className="row" style={{ marginTop: 12 }}>
        <span className="muted">No data handy?</span>
        <button className="btn" disabled={busy} onClick={onSample}>Use sample data</button>
      </div>
    </section>
  );
}
