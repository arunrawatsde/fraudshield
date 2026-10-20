"""Behavioural features: each payment is compared with the same customer's *earlier* payments.

Only past rows are used for each feature (no peeking at the future), so the same code works on
a training file and on a freshly uploaded CSV.
"""
import math
from collections import Counter, deque
from datetime import timedelta

import numpy as np
import pandas as pd

FEATURES = [
    "amount_log", "amount_vs_median", "txns_last_1h", "small_txns_last_1h", "new_device",
    "foreign_country", "fast_country_change", "night", "new_category", "online",
]
MIN_HISTORY = 3          # below this, "new device" / "foreign" etc. are not meaningful yet
SMALL_AMOUNT = 5.0
WINDOW = timedelta(hours=1)
FAST_COUNTRY_CHANGE = timedelta(hours=3)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df must be sorted by timestamp with a 0..n-1 index."""
    out = np.zeros((len(df), len(FEATURES)))
    ts = df["timestamp"].to_list()
    amounts = df["amount"].to_numpy(dtype=float)
    devices, countries = df["device_id"].to_list(), df["country"].to_list()
    categories, channels = df["merchant_category"].to_list(), df["channel"].to_list()

    for positions in df.groupby("customer_id", sort=False).indices.values():
        past_amounts: list[float] = []
        seen_devices, seen_categories = set(), set()
        country_counts: Counter = Counter()
        recent: deque = deque()                     # (timestamp, amount) within the last hour
        last_ts = last_country = None

        for i in positions:
            now, amt = ts[i], amounts[i]
            while recent and now - recent[0][0] > WINDOW:
                recent.popleft()
            enough = len(past_amounts) >= MIN_HISTORY
            median = float(np.median(past_amounts)) if enough else amt
            home = country_counts.most_common(1)[0][0] if country_counts else countries[i]

            out[i] = [
                math.log1p(amt),
                amt / max(median, 1.0) if enough else 1.0,
                len(recent),
                sum(a < SMALL_AMOUNT for _, a in recent) + (amt < SMALL_AMOUNT),
                float(enough and devices[i] not in seen_devices),
                float(enough and countries[i] != home),
                float(last_country is not None and countries[i] != last_country
                      and now - last_ts < FAST_COUNTRY_CHANGE),
                float(now.hour < 6),
                float(enough and categories[i] not in seen_categories),
                float(channels[i] == "online"),
            ]
            past_amounts.append(amt)
            seen_devices.add(devices[i])
            seen_categories.add(categories[i])
            country_counts[countries[i]] += 1
            recent.append((now, amt))
            last_ts, last_country = now, countries[i]

    return pd.DataFrame(out, columns=FEATURES, index=df.index)


def explain(f: pd.Series) -> list[str]:
    """Human-readable reasons, shown to the analyst and given to the agent as starting leads."""
    reasons = []
    if f.small_txns_last_1h >= 3:
        reasons.append(f"{int(f.small_txns_last_1h)} payments under ${SMALL_AMOUNT:.0f} within an hour")
    if f.fast_country_change:
        reasons.append("country changed less than 3 hours after the previous payment")
    if f.new_device:
        reasons.append("first payment from this device")
    if f.amount_vs_median >= 4:
        reasons.append(f"amount is {f.amount_vs_median:.1f}x the customer's median")
    if f.foreign_country:
        reasons.append("outside the customer's usual country")
    if f.txns_last_1h >= 5:
        reasons.append(f"{int(f.txns_last_1h)} other payments in the last hour")
    if f.night and f.amount_vs_median >= 3:
        reasons.append("large payment between midnight and 6am")
    if f.new_category and f.amount_vs_median >= 3:
        reasons.append("large payment in a category this customer never used")
    return reasons
