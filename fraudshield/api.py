"""REST API used by the Next.js frontend.

    uvicorn fraudshield.api:app --reload --port 8000      # docs at http://localhost:8000/docs

Uploaded datasets live in memory (keyed by id) for the life of the process. Fine for a demo; a
real deployment would store them in a database or object storage.
"""
import io
import json
import threading
import uuid
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from fraudshield import model
from fraudshield.config import DATA_DIR, ROOT, get_settings
from fraudshield.dataset import TransactionSet
from fraudshield.knowledge import get_knowledge_base

app = FastAPI(title="FraudShield API", version="0.1.0")
MAX_DATASETS = 10


class Session:
    def __init__(self, data: TransactionSet, name: str):
        self.data, self.name = data, name
        self.results: dict[str, dict] = {}
        self._graph = None
        self.lock = threading.Lock()

    @property
    def graph(self):
        if self._graph is None:
            from fraudshield.agent import build_graph
            self._graph = build_graph(self.data, get_knowledge_base())
        return self._graph


SESSIONS: dict[str, Session] = {}


@lru_cache
def bundle() -> dict:
    return model.load()


def get_session(dataset_id: str) -> Session:
    if dataset_id not in SESSIONS:
        raise HTTPException(404, "Dataset not found. The API may have restarted; upload it again.")
    return SESSIONS[dataset_id]


def mask(value: str, keep: int = 4) -> str:
    value = str(value or "")
    return "•" * max(len(value) - keep, 0) + value[-keep:] if value else ""


def summary(dataset_id: str, s: Session) -> dict:
    df = s.data.df
    out = {"dataset_id": dataset_id, "name": s.name, "rows": len(df), "flagged": int(df["flagged"].sum()),
           "flag_rate": round(float(df["flagged"].mean()), 4), "threshold": get_settings().flag_threshold,
           "has_labels": s.data.has_labels, "investigated": len(s.results)}
    if s.data.has_labels:
        tp = int((df["flagged"] & (df["is_fraud"] == 1)).sum())
        out["model_precision"] = round(tp / max(out["flagged"], 1), 3)
        out["model_recall"] = round(tp / max(int(df["is_fraud"].sum()), 1), 3)
    return out


def register(data: TransactionSet, name: str) -> dict:
    data.score(bundle(), get_settings().flag_threshold)
    if len(SESSIONS) >= MAX_DATASETS:
        SESSIONS.pop(next(iter(SESSIONS)))
    dataset_id = uuid.uuid4().hex[:10]
    SESSIONS[dataset_id] = Session(data, name)
    return summary(dataset_id, SESSIONS[dataset_id])


# ------------------------------------------------------------------ endpoints

@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/status")
def status() -> dict:
    s = get_settings()
    return {"openai": bool(s.openai_api_key), "model": s.openai_model,
            "knowledge_base": "pinecone" if s.use_pinecone else "local-tfidf",
            "langfuse": s.use_langfuse, "threshold": s.flag_threshold}


@app.post("/datasets/sample")
def load_sample() -> dict:
    return register(TransactionSet.from_csv(DATA_DIR / "sample_transactions.csv"), "sample_transactions.csv")


@app.post("/datasets")
async def upload(file: UploadFile = File(...)) -> dict:
    raw = await file.read()
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 20 MB).")
    try:
        data = TransactionSet.from_csv(io.BytesIO(raw))
    except Exception as e:  # noqa: BLE001 - surface any parse/validation error to the user
        raise HTTPException(400, f"Could not read CSV: {e}") from e
    return register(data, file.filename or "upload.csv")


@app.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str) -> dict:
    return summary(dataset_id, get_session(dataset_id))


@app.get("/datasets/{dataset_id}/flagged")
def flagged(dataset_id: str) -> list[dict]:
    s = get_session(dataset_id)
    rows = []
    for r in s.data.flagged().to_dict("records"):
        rows.append({
            "transaction_id": r["transaction_id"], "timestamp": r["timestamp"].isoformat(),
            "customer_id": r["customer_id"], "amount": round(float(r["amount"]), 2),
            "merchant": r["merchant"], "merchant_category": r["merchant_category"],
            "country": r["country"], "channel": r["channel"], "card": mask(r["card_number"]),
            "fraud_score": round(float(r["fraud_score"]), 3), "reasons": r["reasons"],
            "investigated": r["transaction_id"] in s.results,
        })
    return rows


class InvestigateRequest(BaseModel):
    transaction_id: str


@app.post("/datasets/{dataset_id}/investigate")
def investigate(dataset_id: str, req: InvestigateRequest) -> dict:
    """One transaction per call, so the frontend can show progress across several."""
    s = get_session(dataset_id)
    if not get_settings().openai_api_key:
        raise HTTPException(400, "OPENAI_API_KEY is not set on the API server.")
    try:
        s.data.position(req.transaction_id)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    from fraudshield.agent import run_investigation
    with s.lock:            # one graph build per dataset
        graph = s.graph
    result = run_investigation(graph, req.transaction_id, session_id=f"web-{dataset_id}")
    s.results[req.transaction_id] = result
    return result


@app.get("/datasets/{dataset_id}/results")
def results(dataset_id: str) -> list[dict]:
    return list(get_session(dataset_id).results.values())


@app.get("/evals/latest")
def latest_eval() -> Optional[dict]:
    path = ROOT / "evals" / "results" / "latest.json"
    return json.loads(path.read_text()) if path.exists() else None
