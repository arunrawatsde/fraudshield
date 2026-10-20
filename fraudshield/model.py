"""The screening model: a small gradient-boosted tree classifier over behavioural features.

It runs on every row in milliseconds; only rows above the threshold go to the (slow, paid) agent.

    python -m fraudshield.model      # trains on data/train_transactions.csv,
                                     # tests on data/sample_transactions.csv (a different seed)
"""
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score

from fraudshield.config import DATA_DIR, MODEL_PATH, get_settings
from fraudshield.dataset import TransactionSet
from fraudshield.features import FEATURES


def train() -> dict:
    train_set = TransactionSet.from_csv(DATA_DIR / "train_transactions.csv")
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=4,
                                           class_weight="balanced", random_state=0)
    model.fit(train_set.features[FEATURES], train_set.df["is_fraud"])
    bundle = {"model": model, "features": FEATURES}
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    return bundle


def load() -> dict:
    if not MODEL_PATH.exists():
        return train()
    return joblib.load(MODEL_PATH)


def evaluate(bundle: dict, threshold: float) -> dict:
    test = TransactionSet.from_csv(DATA_DIR / "sample_transactions.csv")
    test.score(bundle, threshold)
    y, pred = test.df["is_fraud"], test.df["flagged"]
    by_pattern = (test.df[test.df["is_fraud"] == 1].groupby("fraud_pattern")["flagged"].mean()
                  .round(2).to_dict())
    return {
        "precision": round(precision_score(y, pred), 3),
        "recall": round(recall_score(y, pred), 3),
        "pr_auc": round(average_precision_score(y, test.df["fraud_score"]), 3),
        "flag_rate": round(float(pred.mean()), 4),
        "recall_by_pattern": by_pattern,
    }


if __name__ == "__main__":
    b = train()
    print(f"saved {MODEL_PATH}")
    print("held-out metrics at each threshold (the agent clears false positives, so favour recall):")
    for t in (0.1, 0.2, 0.3, 0.5):
        m = evaluate(b, t)
        print(f"  {t:.1f}  precision {m['precision']:.3f}  recall {m['recall']:.3f}  "
              f"sent to agent {m['flag_rate']:.2%}  by pattern {m['recall_by_pattern']}")
    print(f"configured FLAG_THRESHOLD = {get_settings().flag_threshold}")
