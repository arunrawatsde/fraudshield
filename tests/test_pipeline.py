"""Data, features, model, knowledge base and agent tools. No API keys needed."""
import json

import pytest

from fraudshield import data_gen, model
from fraudshield.config import DATA_DIR
from fraudshield.dataset import TransactionSet
from fraudshield.knowledge import LocalIndex, load_playbook


@pytest.fixture(scope="module")
def scored():
    bundle = model.load()
    data = TransactionSet.from_csv(DATA_DIR / "sample_transactions.csv")
    data.score(bundle, 0.2)
    return data


def test_generator_is_deterministic():
    a, _ = data_gen.generate(seed=1, n_customers=20)
    b, _ = data_gen.generate(seed=1, n_customers=20)
    assert [r["amount"] for r in a] == [r["amount"] for r in b]


def test_rejects_csv_without_required_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("transaction_id,amount\nT1,10\n")
    with pytest.raises(ValueError, match="missing required columns"):
        TransactionSet.from_csv(bad)


def test_model_catches_every_pattern_and_flags_few_rows(scored):
    m = model.evaluate(model.load(), 0.2)
    assert m["recall"] >= 0.7 and m["flag_rate"] < 0.05
    assert all(v >= 0.5 for v in m["recall_by_pattern"].values()), m


def test_flagged_rows_have_reasons(scored):
    assert all(scored.flagged()["reasons"].map(len) > 0)


def test_playbook_search_finds_relevant_section():
    hits = LocalIndex(load_playbook()).search("burst of tiny payments under $5", k=2)
    assert "Card testing" in hits[0]["title"]


def test_agent_tools_never_expose_pii_or_labels(scored):
    from fraudshield.agent import make_tools
    tools = {t.name: t for t in make_tools(scored, type("KB", (), {"search": lambda self, q, k: []})())}
    case = json.loads((DATA_DIR / "eval_cases.json").read_text())[0]["transaction_id"]
    out = tools["customer_profile"].invoke({"transaction_id": case})
    out += tools["activity_around"].invoke({"transaction_id": case, "minutes": 600})
    row = scored.row(case)
    for secret in (row["email"], row["card_number"], row["customer_name"], "is_fraud", "fraud_pattern"):
        assert secret not in out
