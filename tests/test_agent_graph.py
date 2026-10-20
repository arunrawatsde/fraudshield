"""Runs the full LangGraph flow with a scripted stand-in for the LLM: real tools, real guardrails,
no API key and no cost."""
import json

from langchain_core.messages import AIMessage

from fraudshield import model
from fraudshield.agent import Report, build_graph, run_investigation
from fraudshield.config import DATA_DIR
from fraudshield.dataset import TransactionSet
from fraudshield.knowledge import KnowledgeBase


class ScriptedLLM:
    """Calls two tools, writes a note, then returns a report that tries to over-block."""

    def __init__(self, txn_id: str):
        self.txn_id, self.seen = txn_id, []

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema, include_raw=False):
        outer = self

        class Reporter:
            def invoke(self, messages):
                outer.seen.append(messages)
                report = Report(verdict="suspicious", pattern="account_takeover", confidence=0.95,
                                summary="Unclear.", evidence=["one fact"], playbook_sections=[],
                                recommended_action="block")
                return {"parsed": report, "raw": AIMessage("", usage_metadata={
                    "input_tokens": 50, "output_tokens": 20, "total_tokens": 70})}
        return Reporter()

    def invoke(self, messages):
        self.seen.append(messages)
        steps = sum(1 for m in messages if isinstance(m, AIMessage))
        usage = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}
        if steps == 0:
            return AIMessage("", tool_calls=[{"name": "customer_profile", "id": "c1",
                                              "args": {"transaction_id": self.txn_id}}], usage_metadata=usage)
        if steps == 1:
            return AIMessage("", tool_calls=[{"name": "search_playbook", "id": "c2",
                                              "args": {"query": "new device large purchase"}}],
                             usage_metadata=usage)
        return AIMessage("Findings: mixed evidence.", usage_metadata=usage)


def test_graph_end_to_end_with_guardrails(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "")
    data = TransactionSet.from_csv(DATA_DIR / "sample_transactions.csv")
    data.score(model.load(), 0.2)
    case = next(c for c in json.loads((DATA_DIR / "eval_cases.json").read_text()) if c["has_injection"])
    llm = ScriptedLLM(case["transaction_id"])

    result = run_investigation(build_graph(data, KnowledgeBase(), llm=llm), case["transaction_id"])

    assert [c["name"] for c in result["tool_calls"]] == ["customer_profile", "search_playbook"]
    # policy: "block" with a non-fraud verdict and one evidence item is downgraded
    assert result["report"]["recommended_action"] == "manual_review"
    guards = {e["guard"] for e in result["guard_events"]}
    assert {"pii_fields_dropped", "prompt_injection_blocked", "policy_downgrade"} <= guards
    # nothing the LLM saw contains PII or the injected text
    row = data.row(case["transaction_id"])
    seen = json.dumps([[getattr(m, "content", "") for m in msgs] for msgs in llm.seen])
    for secret in (row["email"], row["card_number"], row["customer_name"], row["description"]):
        assert secret not in seen
    assert result["input_tokens"] == 350 and result["output_tokens"] == 50
