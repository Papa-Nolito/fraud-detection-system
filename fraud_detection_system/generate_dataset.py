"""Creates a realistic *synthetic* labelled transaction dataset (your proposal allows
"a suitable labelled dataset or simulated transaction records").

Fraud is rare (~3 %), like in real life. ~15 % of fraud cases are made to look normal
('stealth' fraud) so the model cannot reach an unrealistic 100 % score.

Usage:  python generate_dataset.py            (writes data/transactions_synthetic.csv)
"""
import os

import numpy as np
import pandas as pd

import ml_engine as ml

BASE = os.path.dirname(os.path.abspath(__file__))


def _hour_probs_legit():
    p = np.full(24, 0.0)
    p[0:6] = 0.005
    p[6] = 0.02
    p[7:22] = 0.0613
    p[22:24] = 0.015
    return p / p.sum()


def generate(n=20000, fraud_rate=0.03, seed=42):
    rng = np.random.default_rng(seed)
    is_fraud = rng.random(n) < fraud_rate
    stealth = is_fraud & (rng.random(n) < 0.15)       # fraud that looks legitimate

    # ---- legitimate behaviour ----
    L = {
        "ratio": rng.lognormal(0.0, 0.45, n),
        "hour": rng.choice(24, n, p=_hour_probs_legit()),
        "new_device": rng.random(n) < 0.06,
        "count": 1 + rng.poisson(0.8, n),
        "distance": np.where(rng.random(n) < 0.05, rng.exponential(200, n), rng.exponential(12, n)),
        "country": np.where(rng.random(n) < 0.94, "KE", rng.choice(ml.COUNTRIES[1:], n)),
        "channel": rng.choice(ml.CHANNELS, n, p=[0.35, 0.15, 0.25, 0.25]),
        "category": rng.choice(ml.MERCHANT_CATEGORIES, n,
                               p=[0.22, 0.12, 0.15, 0.10, 0.06, 0.05, 0.02, 0.01, 0.17, 0.10]),
    }
    # ---- fraudulent behaviour ----
    F = {
        "ratio": rng.lognormal(1.3, 0.8, n),
        "hour": np.where(rng.random(n) < 0.55, rng.integers(0, 6, n), rng.integers(0, 24, n)),
        "new_device": rng.random(n) < 0.65,
        "count": 1 + rng.poisson(3.5, n),
        "distance": rng.exponential(500, n),
        "country": np.where(rng.random(n) < 0.5, "KE", rng.choice(ml.COUNTRIES[1:], n)),
        "channel": rng.choice(ml.CHANNELS, n, p=[0.15, 0.15, 0.40, 0.30]),
        "category": rng.choice(ml.MERCHANT_CATEGORIES, n,
                               p=[0.04, 0.02, 0.03, 0.02, 0.22, 0.12, 0.18, 0.15, 0.15, 0.07]),
    }
    use_fraud = is_fraud & ~stealth
    pick = lambda k: np.where(use_fraud, F[k], L[k])

    acct_idx = rng.integers(0, 500, n)
    base_avg = rng.lognormal(8.0, 0.5, 500)           # each account's typical spend (KES)
    ratio = np.clip(pick("ratio"), 0.05, 50)
    amount = np.clip(np.round(base_avg[acct_idx] * ratio, 2), 10, 5_000_000)

    return pd.DataFrame({
        "account_id": [f"ACC{1000 + i}" for i in acct_idx],
        "amount": amount,
        "merchant_category": pick("category"),
        "channel": pick("channel"),
        "country": pick("country"),
        "distance_from_home_km": np.round(pick("distance"), 2),
        "new_device": pick("new_device").astype(int),
        "hour": pick("hour").astype(int),
        "day_of_week": rng.integers(0, 7, n),
        "txn_count_24h": pick("count").astype(int),
        "amount_ratio": np.round(ratio, 3),
        "is_fraud": is_fraud.astype(int),
    })


if __name__ == "__main__":
    os.makedirs(os.path.join(BASE, "data"), exist_ok=True)
    df = generate()
    df.to_csv(os.path.join(BASE, "data", "transactions_synthetic.csv"), index=False)
    print(f"Saved {len(df)} rows | fraud cases: {df.is_fraud.sum()} ({df.is_fraud.mean():.2%})")

    # A small CSV you can upload through the web app to test batch scanning
    cols = ["account_id", "amount", "merchant_category", "channel", "country",
            "distance_from_home_km", "new_device"]
    sample = pd.concat([df[df.is_fraud == 0].head(8), df[df.is_fraud == 1].head(4)])
    sample[cols].sample(frac=1, random_state=1).to_csv(
        os.path.join(BASE, "data", "sample_upload.csv"), index=False)
    print("Saved data/sample_upload.csv (12 rows for the upload test)")
