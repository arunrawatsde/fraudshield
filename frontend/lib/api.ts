// Typed client for the FastAPI backend (proxied at /api by next.config.mjs).

export type Status = {
  openai: boolean;
  model: string;
  knowledge_base: "pinecone" | "local-tfidf";
  langfuse: boolean;
  threshold: number;
};

export type DatasetSummary = {
  dataset_id: string;
  name: string;
  rows: number;
  flagged: number;
  flag_rate: number;
  threshold: number;
  has_labels: boolean;
  investigated: number;
  model_precision?: number;
  model_recall?: number;
};

export type FlaggedTxn = {
  transaction_id: string;
  timestamp: string;
  customer_id: string;
  amount: number;
  merchant: string;
  merchant_category: string;
  country: string;
  channel: string;
  card: string;
  fraud_score: number;
  reasons: string[];
  investigated: boolean;
};

export type Report = {
  verdict: "fraud" | "suspicious" | "legitimate";
  pattern: string;
  confidence: number;
  summary: string;
  evidence: string[];
  playbook_sections: string[];
  recommended_action: "block" | "manual_review" | "approve";
};

export type GuardEvent = { guard: string; [key: string]: unknown };

export type Investigation = {
  transaction_id: string;
  report: Report;
  guard_events: GuardEvent[];
  tool_calls: { name: string; args: Record<string, unknown> }[];
  input_tokens: number;
  output_tokens: number;
  latency_s: number;
};

export type EvalRun = {
  meta: { run_at: string; model: string; kb: string };
  summary: Record<string, number>;
  cases: {
    transaction_id: string;
    tag: string;
    expected_fraud: boolean;
    verdict: string;
    pattern: string;
    action: string;
    tool_calls: number;
    latency_s: number;
  }[];
};

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, { cache: "no-store", ...init });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json() as Promise<T>;
}

const post = (body?: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: body === undefined ? undefined : JSON.stringify(body),
});

export const api = {
  status: () => call<Status>("/status"),
  loadSample: () => call<DatasetSummary>("/datasets/sample", post()),
  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return call<DatasetSummary>("/datasets", { method: "POST", body: form });
  },
  dataset: (id: string) => call<DatasetSummary>(`/datasets/${id}`),
  flagged: (id: string) => call<FlaggedTxn[]>(`/datasets/${id}/flagged`),
  results: (id: string) => call<Investigation[]>(`/datasets/${id}/results`),
  investigate: (id: string, transaction_id: string) =>
    call<Investigation>(`/datasets/${id}/investigate`, post({ transaction_id })),
  latestEval: () => call<EvalRun | null>("/evals/latest"),
};
