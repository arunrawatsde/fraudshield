"""Guardrails around the LLM.

Input side:  PII is removed before anything reaches the model (structured PII columns are
             dropped; emails, phones and card numbers in free text are masked), and free-text
             fields are checked for prompt injection.
Output side: the report is scrubbed for PII again and checked against the action policy, so a
             confused or manipulated model still cannot block a payment on thin evidence.

All of this is plain deterministic Python: it cannot be talked out of its rules.
"""
import re
from typing import Iterable

from fraudshield.dataset import LABEL_COLUMNS, PII_COLUMNS

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
CARD = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
PHONE = re.compile(r"\+?\d{1,3}[ .-]?\(?\d{2,4}\)?[ .-]?\d{3,4}[ .-]?\d{3,4}\b")
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

INJECTION_RULES = {
    "override_instructions": re.compile(
        r"\b(ignore|disregard|forget|override)\b.{0,40}\b(instructions?|rules|prompt|guidelines)\b", re.I),
    "role_marker": re.compile(r"(^|\s)(system|assistant|developer)\s*:", re.I),
    "fake_tags": re.compile(r"</?\s*(system|instructions?|prompt)\s*>", re.I),
    "persona_switch": re.compile(r"\byou are now\b|\bact as\b|\bmaintenance mode\b", re.I),
    "verdict_steering": re.compile(
        r"\b(mark|treat|classify|flag)\b.{0,30}\b(as )?(legitimate|safe|approved|not fraud)\b"
        r"|\bverdict must\b|\bapprove this (transaction|payment)\b|\bpre-?approved\b", re.I),
    "prompt_extraction": re.compile(r"\b(reveal|print|show|repeat)\b.{0,20}\b(system )?prompt\b", re.I),
}
REMOVED = "[removed by guardrail: possible prompt injection]"


def luhn_ok(number: str) -> bool:
    digits = [int(d) for d in number if d.isdigit()]
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            d = d - 9 if d > 9 else d
        total += d
    return len(digits) >= 13 and total % 10 == 0


def redact_pii(text: str, known_values: Iterable[str] = ()) -> tuple[str, list[str]]:
    """Mask PII in free text. Returns (clean_text, kinds_found)."""
    found = []
    for value in known_values:              # e.g. this customer's own name, if it shows up in a memo
        if value and len(value) > 2 and value.lower() in text.lower():
            text = re.sub(re.escape(value), "[NAME]", text, flags=re.I)
            found.append("name")

    def card(m: re.Match) -> str:
        if luhn_ok(m.group()):
            found.append("card_number")
            return "[CARD]"
        return m.group()

    text = CARD.sub(card, text)
    for kind, pattern, label in (("email", EMAIL, "[EMAIL]"), ("ssn", SSN, "[SSN]"),
                                 ("phone", PHONE, "[PHONE]")):
        text, n = pattern.subn(label, text)
        found += [kind] * n
    return text, found


def detect_injection(text: str) -> list[str]:
    return [name for name, rule in INJECTION_RULES.items() if rule.search(text or "")]


def sanitize_record(record: dict) -> tuple[dict, list[dict]]:
    """A PII-free, injection-checked copy of a transaction row that is safe to show the LLM."""
    events: list[dict] = []
    txn_id = record.get("transaction_id")
    safe = {k: v for k, v in record.items() if k not in (*PII_COLUMNS, *LABEL_COLUMNS)}
    dropped = [c for c in PII_COLUMNS if record.get(c)]
    if dropped:
        events.append({"guard": "pii_fields_dropped", "transaction_id": txn_id, "fields": dropped})

    note = str(record.get("description") or "")
    if note:
        hits = detect_injection(note)
        if hits:
            note = REMOVED
            events.append({"guard": "prompt_injection_blocked", "transaction_id": txn_id, "rules": hits})
        else:
            note, kinds = redact_pii(note, [record.get("customer_name", "")])
            if kinds:
                events.append({"guard": "pii_redacted_in_text", "transaction_id": txn_id, "kinds": kinds})
    safe["description"] = note
    if "timestamp" in safe and hasattr(safe["timestamp"], "isoformat"):
        safe["timestamp"] = safe["timestamp"].isoformat()
    for k in ("fraud_score", "amount"):
        if k in safe:
            safe[k] = round(float(safe[k]), 3)
    return safe, events


# ------------------------------------------------------------------ output side

def apply_policy(report: dict, known_pii: Iterable[str] = ()) -> tuple[dict, list[dict]]:
    """Enforce the action policy and strip any PII the model may have echoed back."""
    r = dict(report)
    events: list[dict] = []
    r["confidence"] = min(1.0, max(0.0, float(r.get("confidence", 0))))

    if r["recommended_action"] == "block":
        if r["verdict"] != "fraud" or r["confidence"] < 0.7 or len(r.get("evidence", [])) < 2:
            r["recommended_action"] = "manual_review"
            events.append({"guard": "policy_downgrade", "detail":
                           "block needs verdict=fraud, confidence>=0.7 and 2+ evidence items"})
    if r["verdict"] == "fraud" and r["recommended_action"] == "approve":
        r["recommended_action"] = "manual_review"
        events.append({"guard": "policy_downgrade", "detail": "a fraud verdict cannot be approved"})
    if r["verdict"] == "legitimate" and r["pattern"] != "none":
        r["pattern"] = "none"

    known = [v for v in known_pii if v]
    leaked: list[str] = []
    r["summary"], kinds = redact_pii(r.get("summary", ""), known)
    leaked += kinds
    clean_evidence = []
    for item in r.get("evidence", []):
        text, kinds = redact_pii(item, known)
        clean_evidence.append(text)
        leaked += kinds
    r["evidence"] = clean_evidence
    if leaked:
        events.append({"guard": "pii_redacted_in_output", "kinds": sorted(set(leaked))})
    return r, events
