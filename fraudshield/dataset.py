"""Loads an uploaded CSV, validates it, builds features and scores it with the model."""
from typing import IO, Union

import pandas as pd

from fraudshield.features import build_features, explain

REQUIRED = ["transaction_id", "timestamp", "customer_id", "amount", "merchant",
            "merchant_category", "country", "device_id", "channel"]
PII_COLUMNS = ["customer_name", "email", "phone", "card_number"]
LABEL_COLUMNS = ["is_fraud", "fraud_pattern"]      # used for evaluation only, never shown to the agent


class TransactionSet:
    def __init__(self, df: pd.DataFrame):
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["amount"] = pd.to_numeric(df["amount"], errors="raise")
        df["transaction_id"] = df["transaction_id"].astype(str)
        for col in ["description", *PII_COLUMNS]:
            df[col] = df[col].fillna("").astype(str) if col in df.columns else ""
        df = df.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)
        if df["transaction_id"].duplicated().any():
            raise ValueError("transaction_id values must be unique")

        self.df = df
        self.features = build_features(df)
        self._pos = {t: i for i, t in enumerate(df["transaction_id"])}

    @classmethod
    def from_csv(cls, source: Union[str, IO]) -> "TransactionSet":
        return cls(pd.read_csv(source, dtype={"customer_id": str, "device_id": str}))

    @property
    def has_labels(self) -> bool:
        return "is_fraud" in self.df.columns

    def score(self, bundle: dict, threshold: float) -> None:
        self.df["fraud_score"] = bundle["model"].predict_proba(self.features[bundle["features"]])[:, 1]
        self.df["flagged"] = self.df["fraud_score"] >= threshold
        self.df["reasons"] = [explain(f) for f in self.features.itertuples(index=False)]

    def flagged(self) -> pd.DataFrame:
        return self.df[self.df["flagged"]].sort_values("fraud_score", ascending=False)

    def position(self, txn_id: str) -> int:
        if txn_id not in self._pos:
            raise KeyError(f"unknown transaction_id {txn_id}")
        return self._pos[txn_id]

    def row(self, txn_id: str) -> dict:
        """Full row, including PII. Never send this to the LLM: use guardrails.sanitize_record."""
        return self.df.iloc[self.position(txn_id)].to_dict()
