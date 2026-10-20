"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { api, Status } from "@/lib/api";

export default function Header() {
  const path = usePathname();
  const [status, setStatus] = useState<Status | null>(null);
  const [down, setDown] = useState(false);

  useEffect(() => {
    api.status().then(setStatus).catch(() => setDown(true));
  }, []);

  return (
    <header className="header">
      <Link href="/" className="brand">🛡️ FraudShield</Link>
      <nav className="nav">
        <Link href="/" className={path === "/" ? "active" : ""}>Investigate</Link>
        <Link href="/evals" className={path === "/evals" ? "active" : ""}>Evals</Link>
      </nav>
      <div className="status">
        {down && <span className="chip fraud">API offline</span>}
        {status && (
          <>
            <span className={`chip ${status.openai ? "on" : ""}`} title="Needed to run the agent">
              OpenAI {status.openai ? `· ${status.model}` : "· no key"}
            </span>
            <span className={`chip ${status.knowledge_base === "pinecone" ? "on" : ""}`}>
              {status.knowledge_base === "pinecone" ? "Pinecone" : "Local search"}
            </span>
            <span className={`chip ${status.langfuse ? "on" : ""}`}>
              Langfuse {status.langfuse ? "on" : "off"}
            </span>
          </>
        )}
      </div>
    </header>
  );
}
