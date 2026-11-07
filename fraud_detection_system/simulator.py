"""Live transaction simulator - pretends to be a bank's payment system.

It sends a new random transaction to the running web app every few seconds through the
REST API, exactly like a real transaction processing system (TPS) would.
Watch the dashboard update by itself, and pop-up alerts appear when fraud is detected.

Usage (the web app must already be running in another terminal):
    python simulator.py                      # endless, one transaction every ~3 s
    python simulator.py --interval 1 --fraud-rate 0.25
    python simulator.py --count 30           # stop after 30 transactions
Stop with Ctrl + C.
"""
import argparse
import json
import os
import random
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))


def post(url, key, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "X-API-Key": key})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {"error": str(e)}
    except urllib.error.URLError as e:
        return 0, {"error": f"Cannot reach the server ({e.reason}). Is 'python app.py' running?"}


def realistic_time(hour, rnd, offset_s=0):
    """Today's date with the transaction's hour (yesterday if that hour is still in the future)."""
    now = datetime.now().replace(microsecond=0)
    dt = now.replace(hour=int(hour), minute=rnd.randint(0, 59), second=rnd.randint(0, 59))
    if dt > now:
        dt -= timedelta(days=1)
    return dt + timedelta(seconds=offset_s)


def make_payload(row, when, amount=None):
    return {"account_id": row["account_id"], "amount": round(float(amount or row["amount"]), 2),
            "merchant_category": row["merchant_category"], "channel": row["channel"],
            "country": row["country"], "distance_from_home_km": float(row["distance_from_home_km"]),
            "new_device": int(row["new_device"]), "txn_time": when.strftime("%Y-%m-%d %H:%M:%S")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:5000/api/transactions")
    ap.add_argument("--api-key", default=os.environ.get("FRAUD_API_KEY", "dev-api-key-change-me"))
    ap.add_argument("--interval", type=float, default=3.0, help="seconds between transactions")
    ap.add_argument("--fraud-rate", type=float, default=0.15, help="share of fraud attempts (0-1)")
    ap.add_argument("--count", type=int, default=0, help="stop after N transactions (0 = forever)")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    df = pd.read_csv(os.path.join(BASE, "data", "transactions_synthetic.csv"))
    legit, fraud = df[df.is_fraud == 0], df[df.is_fraud == 1]
    sent = flagged = 0
    print(f"Simulator started -> {args.url}   (Ctrl+C to stop)\n")
    try:
        while args.count == 0 or sent < args.count:
            if rnd.random() < args.fraud_rate:
                # A fraud attempt: several rapid transactions on one account (card testing / drain)
                row = fraud.sample(1, random_state=rnd.randint(0, 10**9)).iloc[0].to_dict()
                batch = [(row, realistic_time(row["hour"], rnd, i * 20), row["amount"] * rnd.uniform(0.8, 1.3))
                         for i in range(rnd.randint(2, 4))]
            else:
                row = legit.sample(1, random_state=rnd.randint(0, 10**9)).iloc[0].to_dict()
                batch = [(row, realistic_time(row["hour"], rnd), row["amount"])]
            for r, when, amount in batch:
                status, body = post(args.url, args.api_key, make_payload(r, when, amount))
                sent += 1
                if status == 201:
                    mark = "FRAUD " if body["prediction"] == "fraud" else "ok    "
                    flagged += body["prediction"] == "fraud"
                    print(f"[{sent:>4}] {mark} {r['account_id']:<8} KES {amount:>12,.2f}  "
                          f"p={body['fraud_probability']:.2f}  {body['risk_level']:<6}  "
                          f"{body['processing_ms']} ms")
                else:
                    print(f"[{sent:>4}] ERROR {status}: {body.get('error')} {body.get('details', '')}")
                    if status in (0, 401):
                        return
                time.sleep(0.3 if len(batch) > 1 else 0)
            time.sleep(max(args.interval + rnd.uniform(-0.3, 0.3), 0.05))
    except KeyboardInterrupt:
        pass
    print(f"\nStopped. Sent {sent} transactions, {flagged} flagged as fraud.")


if __name__ == "__main__":
    main()
