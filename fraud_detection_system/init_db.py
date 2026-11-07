"""Creates the tables, the first users, and (optionally) demo data.

    python init_db.py            -> tables + admin and analyst accounts
    python init_db.py --demo     -> ... plus 300 scored demo transactions for the dashboard

Works for SQLite (default) or MySQL:  DB_BACKEND=mysql python init_db.py
"""
import argparse
import os
import random
from datetime import datetime, timedelta

import pandas as pd
from werkzeug.security import generate_password_hash

import db
import ml_engine as ml
from app import create_app, now_str

BASE = os.path.dirname(os.path.abspath(__file__))
ADMIN_PW = os.environ.get("ADMIN_PASSWORD", "Admin@12345")
ANALYST_PW = os.environ.get("ANALYST_PASSWORD", "Analyst@12345")


def ensure_user(name, email, pw, role):
    if not db.query_one("SELECT id FROM users WHERE email=?", (email,)):
        db.execute("INSERT INTO users (full_name,email,password_hash,role,is_active,created_at) "
                   "VALUES (?,?,?,?,1,?)", (name, email, generate_password_hash(pw), role, now_str()))
        print(f"  created {role}: {email}  /  {pw}")


def seed_demo(app, n_legit=240, n_fraud=60):
    data_path = os.path.join(BASE, "data", "transactions_synthetic.csv")
    if not app.extensions["fraud_model"].ready or not os.path.exists(data_path):
        print("  skipped demo data: run generate_dataset.py and train_model.py first")
        return
    df = pd.read_csv(data_path)
    sample = pd.concat([df[df.is_fraud == 0].sample(n_legit, random_state=1),
                        df[df.is_fraud == 1].sample(n_fraud, random_state=1)]).sample(frac=1, random_state=2)
    rnd = random.Random(7)
    admin = db.query_one("SELECT id FROM users WHERE role='admin' ORDER BY id")["id"]
    now = datetime.now()
    for r in sample.itertuples():
        when = (now - timedelta(days=rnd.randint(0, 13))).replace(
            hour=int(r.hour), minute=rnd.randint(0, 59), second=rnd.randint(0, 59), microsecond=0)
        clean = {"account_id": r.account_id, "amount": float(r.amount),
                 "merchant_category": r.merchant_category, "channel": r.channel, "country": r.country,
                 "distance_from_home_km": float(r.distance_from_home_km), "new_device": int(r.new_device),
                 "txn_time": when, "hour": when.hour, "day_of_week": when.weekday()}
        app.score_and_store(clean, admin, int(r.txn_count_24h), float(r.amount_ratio),
                            created_at=when.strftime(ml.TIME_FMT))
    print(f"  inserted {len(sample)} demo transactions")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    app = create_app()
    with app.app_context():
        db.init_db()
        print("Database tables ready.")
        ensure_user("System Administrator", "admin@frauddetect.local", ADMIN_PW, "admin")
        ensure_user("Fraud Analyst", "analyst@frauddetect.local", ANALYST_PW, "analyst")
        if args.demo:
            seed_demo(app)
    print("Done. Start the app with:  python app.py")
