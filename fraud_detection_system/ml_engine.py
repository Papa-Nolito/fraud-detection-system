"""Machine-learning core shared by the training script and the web app.

Maps to the 'Process' stage of the IPOS framework in your proposal:
    Data cleaning -> Feature engineering -> ML model -> Fraud prediction
"""
import os
import re
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TIME_FMT = "%Y-%m-%d %H:%M:%S"

MERCHANT_CATEGORIES = ["groceries", "fuel", "restaurants", "utilities", "electronics",
                       "travel", "gift_cards", "crypto", "transfers", "clothing"]
CHANNELS = ["pos", "atm", "web", "mobile"]
COUNTRIES = ["KE", "UG", "TZ", "RW", "NG", "ZA", "GB", "US", "AE", "CN"]
HOME_COUNTRY = "KE"

# Columns the raw dataset / database rows contain
RAW_COLUMNS = ["amount", "hour", "day_of_week", "distance_from_home_km", "txn_count_24h",
               "amount_ratio", "new_device", "merchant_category", "channel", "country"]
# Columns the model actually sees after feature engineering
NUMERIC_FEATURES = ["log_amount", "hour", "day_of_week", "distance_from_home_km",
                    "txn_count_24h", "amount_ratio", "new_device", "is_night"]
CATEGORICAL_FEATURES = ["merchant_category", "channel", "country"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

_TRUE = {"1", "true", "yes", "y", "on", "t"}


# --------------------------------------------------------------------------
# 1. DATA CLEANING / VALIDATION  (functional requirement: validate before processing)
# --------------------------------------------------------------------------
def clean_transaction(raw):
    """Validate and normalise one raw transaction (form field dict or CSV row).
    Returns (clean_dict, errors). If errors is non-empty the transaction is rejected."""
    errors, out = [], {}
    get = lambda k: (raw.get(k) if raw.get(k) is not None else "")

    account = str(get("account_id")).strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{3,30}", account):
        out["account_id"] = account
    else:
        errors.append("Account ID must be 3-30 letters, digits, '-' or '_'.")

    try:
        amount = float(str(get("amount")).replace(",", "").strip())
        if not (0 < amount <= 10_000_000):
            raise ValueError
        out["amount"] = round(amount, 2)
    except ValueError:
        errors.append("Amount must be a number greater than 0 and at most 10,000,000.")

    for field, allowed, transform in (("merchant_category", MERCHANT_CATEGORIES, str.lower),
                                      ("channel", CHANNELS, str.lower),
                                      ("country", COUNTRIES, str.upper)):
        value = transform(str(get(field)).strip())
        if value in allowed:
            out[field] = value
        else:
            errors.append(f"{field.replace('_', ' ').title()} must be one of: {', '.join(allowed)}.")

    dist_raw = str(get("distance_from_home_km")).strip() or "0"
    try:
        dist = float(dist_raw)
        if not (0 <= dist <= 20000):
            raise ValueError
        out["distance_from_home_km"] = round(dist, 2)
    except ValueError:
        errors.append("Distance from home must be between 0 and 20,000 km.")

    out["new_device"] = 1 if str(get("new_device")).strip().lower() in _TRUE else 0

    time_raw = str(get("txn_time")).strip()
    if time_raw:
        parsed = None
        for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S", TIME_FMT, "%Y-%m-%d %H:%M"):
            try:
                parsed = datetime.strptime(time_raw, fmt)
                break
            except ValueError:
                continue
        if parsed is None:
            errors.append("Transaction time format is invalid.")
        else:
            out["txn_time"] = parsed
    else:
        out["txn_time"] = datetime.now().replace(microsecond=0)

    if "txn_time" in out:
        out["hour"] = out["txn_time"].hour
        out["day_of_week"] = out["txn_time"].weekday()
    return out, errors


# --------------------------------------------------------------------------
# 2. FEATURE ENGINEERING + NORMALISATION
# --------------------------------------------------------------------------
def prepare_frame(df):
    """Create derived features. (Scaling/encoding happens inside the pipeline so the
    exact same transformation is applied in training and in production.)"""
    df = df.copy()
    df["log_amount"] = np.log1p(df["amount"].astype(float))
    df["is_night"] = (df["hour"].astype(int) < 6).astype(int)
    df["new_device"] = df["new_device"].astype(int)
    return df[FEATURES]


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(categories=[MERCHANT_CATEGORIES, CHANNELS, COUNTRIES],
                              handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
    ])


def build_pipeline(estimator):
    return Pipeline([("prep", build_preprocessor()), ("model", estimator)])


# --------------------------------------------------------------------------
# 3. PREDICTION
# --------------------------------------------------------------------------
class FraudModel:
    """Loads the trained pipeline from disk and scores single transactions."""

    def __init__(self, path):
        self.path = path
        self.bundle = None

    def load(self):
        if os.path.exists(self.path):
            self.bundle = joblib.load(self.path)
        return self.ready

    @property
    def ready(self):
        return self.bundle is not None

    @property
    def threshold(self):
        return self.bundle["threshold"]

    def predict(self, row):
        """row: dict containing RAW_COLUMNS. Returns probability, label and risk level."""
        frame = prepare_frame(pd.DataFrame([row]))
        prob = float(self.bundle["pipeline"].predict_proba(frame)[0, 1])
        thr = self.threshold
        is_fraud = prob >= thr
        if is_fraud and prob >= max(thr, 0.85):
            risk = "high"
        elif is_fraud:
            risk = "medium"
        else:
            risk = "low"
        return {"fraud_probability": prob,
                "prediction": "fraud" if is_fraud else "legitimate",
                "risk_level": risk}


def explain(row):
    """Plain-language 'why was this flagged' indicators for analysts."""
    reasons = []
    if row["amount_ratio"] >= 3:
        reasons.append(f"Amount is {row['amount_ratio']:.1f}x this account's usual spend")
    if int(row["hour"]) < 6:
        reasons.append("Made during night hours (00:00-05:59)")
    if int(row["new_device"]):
        reasons.append("Made from a device not seen before on this account")
    if row["distance_from_home_km"] >= 200:
        reasons.append(f"{row['distance_from_home_km']:.0f} km away from the account holder's home")
    if int(row["txn_count_24h"]) >= 5:
        reasons.append(f"{int(row['txn_count_24h'])} transactions on this account in 24 hours")
    if row["country"] != HOME_COUNTRY:
        reasons.append(f"Foreign transaction ({row['country']})")
    if row["merchant_category"] in ("crypto", "gift_cards"):
        reasons.append(f"High-risk merchant category ({row['merchant_category'].replace('_', ' ')})")
    return reasons
