"""API endpoints the frontend uses. The investigate endpoint is covered by test_agent_graph."""
from fastapi.testclient import TestClient

from fraudshield.api import app
from fraudshield.config import DATA_DIR

client = TestClient(app)


def test_status():
    body = client.get("/status").json()
    assert body["knowledge_base"] in ("pinecone", "local-tfidf")


def test_sample_dataset_and_flagged_rows_are_masked():
    summary = client.post("/datasets/sample").json()
    assert summary["rows"] > 1000 and 0 < summary["flagged"] < summary["rows"] * 0.05
    rows = client.get(f"/datasets/{summary['dataset_id']}/flagged").json()
    assert len(rows) == summary["flagged"]
    assert all(r["card"].startswith("•") for r in rows)
    assert "email" not in rows[0] and "customer_name" not in rows[0]


def test_upload_csv():
    with open(DATA_DIR / "sample_transactions.csv", "rb") as f:
        res = client.post("/datasets", files={"file": ("mine.csv", f, "text/csv")})
    assert res.status_code == 200 and res.json()["name"] == "mine.csv"


def test_bad_csv_is_a_400():
    res = client.post("/datasets", files={"file": ("bad.csv", b"transaction_id,amount\nT1,5\n", "text/csv")})
    assert res.status_code == 400 and "missing required columns" in res.json()["detail"]


def test_unknown_dataset_is_a_404():
    assert client.get("/datasets/nope/flagged").status_code == 404
