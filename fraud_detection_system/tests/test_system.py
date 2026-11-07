"""Automated tests mapped to the test cases in Chapter 5 (Tables 5.1 and 5.2).

Run with:   python -m unittest discover -s tests -v
(Uses a throw-away SQLite database and a quickly-trained model; your real data is untouched.)
"""
import io
import os
import shutil
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db
import ml_engine as ml
from app import create_app
from generate_dataset import generate
from train_model import train_and_save
from werkzeug.security import generate_password_hash

FRAUDY = {"account_id": "ACC9001", "amount": "950000", "merchant_category": "crypto",
          "channel": "web", "country": "NG", "distance_from_home_km": "3000", "new_device": "1",
          "txn_time": "2026-06-01T03:15"}
NORMAL = {"account_id": "ACC9002", "amount": "1500", "merchant_category": "groceries",
          "channel": "pos", "country": "KE", "distance_from_home_km": "2", "new_device": "",
          "txn_time": "2026-06-01T13:05"}


class SystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        train_and_save(generate(n=6000, seed=5), cls.tmp, quick=True)
        cls.app = create_app({"TESTING": True, "SQLITE_PATH": os.path.join(cls.tmp, "t.db"),
                              "MODEL_PATH": os.path.join(cls.tmp, "fraud_model.joblib"),
                              "METRICS_PATH": os.path.join(cls.tmp, "model_metrics.json")})
        with cls.app.app_context():
            db.init_db()
            for name, email, role in (("Admin", "admin@t.com", "admin"), ("Analyst", "an@t.com", "analyst")):
                db.execute("INSERT INTO users (full_name,email,password_hash,role,is_active,created_at) "
                           "VALUES (?,?,?,?,1,'2026-01-01 00:00:00')",
                           (name, email, generate_password_hash("Passw0rd!"), role))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # ---- helpers -----------------------------------------------------
    def client(self, email=None):
        c = self.app.test_client()
        with c.session_transaction() as s:
            s["csrf"] = "tok"
        if email:
            r = c.post("/login", data={"email": email, "password": "Passw0rd!", "csrf_token": "tok"})
            self.assertEqual(r.status_code, 302)
            with c.session_transaction() as s:
                s["csrf"] = "tok"
        return c

    def submit(self, c, data):
        return c.post("/transactions/new", data={**data, "csrf_token": "tok"})

    def last_txn(self):
        with self.app.app_context():
            return db.query_one("SELECT * FROM transactions ORDER BY id DESC LIMIT 1")

    # ---- Table 5.1 functional test cases ------------------------------
    def test_TC01_login_valid(self):
        c = self.app.test_client()
        with c.session_transaction() as s:
            s["csrf"] = "tok"
        r = c.post("/login", data={"email": "an@t.com", "password": "Passw0rd!", "csrf_token": "tok"})
        self.assertEqual(r.status_code, 302)
        self.assertEqual(c.get("/dashboard").status_code, 200)

    def test_TC02_login_invalid(self):
        c = self.app.test_client()
        with c.session_transaction() as s:
            s["csrf"] = "tok"
        r = c.post("/login", data={"email": "an@t.com", "password": "wrong", "csrf_token": "tok"})
        self.assertEqual(r.status_code, 401)
        self.assertEqual(c.get("/dashboard").status_code, 302)   # still not logged in

    def test_TC03_transaction_submission(self):
        c = self.client("an@t.com")
        r = self.submit(c, NORMAL)
        self.assertEqual(r.status_code, 302)
        self.assertEqual(self.last_txn()["account_id"], "ACC9002")

    def test_TC04_validation_rejects_bad_input(self):
        c = self.client("an@t.com")
        bad = {**NORMAL, "amount": "-5", "country": "XX", "account_id": "a"}
        r = self.submit(c, bad)
        self.assertEqual(r.status_code, 400)
        self.assertIn(b"Amount must be", r.data)

    def test_TC05_preprocessing(self):
        clean, errors = ml.clean_transaction(NORMAL)
        self.assertEqual(errors, [])
        frame = ml.prepare_frame(__import__("pandas").DataFrame([{**clean, "txn_count_24h": 1, "amount_ratio": 1.0}]))
        self.assertEqual(list(frame.columns), ml.FEATURES)
        self.assertEqual(int(frame["is_night"][0]), 0)

    def test_TC06_fraud_prediction(self):
        c = self.client("an@t.com")
        self.submit(c, FRAUDY)
        t = self.last_txn()
        self.assertIn(t["prediction"], ("fraud", "legitimate"))
        self.assertTrue(0 <= t["fraud_probability"] <= 1)

    def test_TC07_alert_generated_for_fraud(self):
        c = self.client("an@t.com")
        self.submit(c, FRAUDY)
        t = self.last_txn()
        self.assertEqual(t["prediction"], "fraud")
        with self.app.app_context():
            self.assertIsNotNone(db.query_one("SELECT id FROM alerts WHERE transaction_id=?", (t["id"],)))

    def test_TC08_legitimate_no_alert(self):
        c = self.client("an@t.com")
        self.submit(c, NORMAL)
        t = self.last_txn()
        self.assertEqual(t["prediction"], "legitimate")
        with self.app.app_context():
            self.assertIsNone(db.query_one("SELECT id FROM alerts WHERE transaction_id=?", (t["id"],)))

    def test_TC09_database_storage(self):
        c = self.client("an@t.com")
        self.submit(c, {**NORMAL, "account_id": "ACC9100", "amount": "1234.56"})
        t = self.last_txn()
        self.assertEqual((t["account_id"], t["amount"], t["submitted_by"] is not None), ("ACC9100", 1234.56, True))

    def test_TC10_dashboard(self):
        c = self.client("an@t.com")
        self.submit(c, FRAUDY)
        r = c.get("/dashboard")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"Transactions analysed", r.data)

    def test_TC11_alert_review(self):
        c = self.client("an@t.com")
        self.submit(c, {**FRAUDY, "account_id": "ACC9200"})
        with self.app.app_context():
            a = db.query_one("SELECT id FROM alerts ORDER BY id DESC LIMIT 1")
        self.assertEqual(c.get(f"/alerts/{a['id']}").status_code, 200)
        r = c.post(f"/alerts/{a['id']}", data={"decision": "confirmed_fraud", "notes": "ok", "csrf_token": "tok"})
        self.assertEqual(r.status_code, 302)
        with self.app.app_context():
            self.assertEqual(db.query_one("SELECT status FROM alerts WHERE id=?", (a["id"],))["status"], "confirmed_fraud")

    def test_TC12_reports_and_export(self):
        c = self.client("an@t.com")
        self.submit(c, NORMAL)
        self.assertEqual(c.get("/reports").status_code, 200)
        csv_resp = c.get("/reports/export.csv")
        self.assertEqual(csv_resp.status_code, 200)
        self.assertIn(b"account_id", csv_resp.data)

    # ---- Table 5.2 non-functional test cases ---------------------------
    def test_NFT01_performance(self):
        c = self.client("an@t.com")
        t0 = time.perf_counter()
        self.submit(c, NORMAL)
        self.assertLess(time.perf_counter() - t0, 2.0)
        self.assertLess(self.last_txn()["processing_ms"], 500)

    def test_NFT02_security_protected_pages(self):
        c = self.client()
        for url in ("/dashboard", "/transactions", "/alerts", "/reports", "/transactions/new"):
            self.assertEqual(c.get(url).status_code, 302, url)
        analyst = self.client("an@t.com")
        self.assertEqual(analyst.get("/admin/users").status_code, 403)   # role-based access
        self.assertEqual(self.client("admin@t.com").get("/admin/users").status_code, 200)

    def test_NFT02b_csrf_enforced(self):
        c = self.client("an@t.com")
        r = c.post("/transactions/new", data=NORMAL)                      # no token
        self.assertEqual(r.status_code, 400)

    def test_NFT03_usability_pages_load(self):
        c = self.client("admin@t.com")
        for url in ("/dashboard", "/transactions/new", "/transactions", "/alerts", "/reports", "/admin/users"):
            self.assertEqual(c.get(url).status_code, 200, url)

    def test_NFT04_reliability_batch(self):
        c = self.client("an@t.com")
        for i in range(10):
            self.assertEqual(self.submit(c, {**NORMAL, "account_id": f"ACC95{i:02d}"}).status_code, 302)

    def test_NFT05_data_integrity(self):
        c = self.client("an@t.com")
        self.submit(c, {**NORMAL, "account_id": "ACC9300", "amount": "777.77", "channel": "atm"})
        r = c.get(f"/transactions/{self.last_txn()['id']}")
        self.assertIn(b"777.77", r.data)
        self.assertIn(b"ACC9300", r.data)

    def test_NFT06_error_handling(self):
        c = self.client("an@t.com")
        self.assertEqual(c.get("/transactions/999999").status_code, 404)
        self.assertEqual(self.submit(c, {"account_id": ""}).status_code, 400)

    # ---- REST API + live updates ---------------------------------------
    API_FRAUD = {"account_id": "ACC9700", "amount": 950000, "merchant_category": "crypto",
                 "channel": "web", "country": "NG", "distance_from_home_km": 3000,
                 "new_device": True, "txn_time": "2026-06-01 03:15:00"}

    def test_API01_requires_key(self):
        c = self.app.test_client()
        self.assertEqual(c.post("/api/transactions", json=self.API_FRAUD).status_code, 401)
        r = c.post("/api/transactions", json=self.API_FRAUD, headers={"X-API-Key": "wrong"})
        self.assertEqual(r.status_code, 401)

    def test_API02_scores_and_raises_alert(self):
        c = self.app.test_client()
        r = c.post("/api/transactions", json=self.API_FRAUD,
                   headers={"X-API-Key": self.app.config["API_KEY"]})
        self.assertEqual(r.status_code, 201)
        body = r.get_json()
        self.assertEqual(body["prediction"], "fraud")
        self.assertIsNotNone(body["alert_id"])

    def test_API03_validation_error(self):
        c = self.app.test_client()
        r = c.post("/api/transactions", json={"account_id": "x"},
                   headers={"X-API-Key": self.app.config["API_KEY"]})
        self.assertEqual(r.status_code, 400)
        self.assertTrue(r.get_json()["details"])

    def test_LIVE01_endpoint_needs_login_and_reports_counts(self):
        self.assertEqual(self.app.test_client().get("/api/live").status_code, 401)
        c = self.client("an@t.com")
        before = c.get("/api/live").get_json()
        self.app.test_client().post("/api/transactions", json={**self.API_FRAUD, "account_id": "ACC9701"},
                                    headers={"X-API-Key": self.app.config["API_KEY"]})
        after = c.get("/api/live").get_json()
        self.assertEqual(after["total"], before["total"] + 1)
        self.assertGreater(after["latest_alert_id"], before["latest_alert_id"])

    # ---- extras -------------------------------------------------------
    def test_csv_upload(self):
        c = self.client("an@t.com")
        csv_text = ("account_id,amount,merchant_category,channel,country,distance_from_home_km,new_device\n"
                    "ACC9400,2000,groceries,pos,KE,3,0\nACC9401,bad,fuel,pos,KE,1,0\n")
        r = c.post("/transactions/upload", data={"csrf_token": "tok",
                   "file": (io.BytesIO(csv_text.encode()), "x.csv")}, content_type="multipart/form-data",
                   follow_redirects=True)
        self.assertIn(b"1 scored", r.data)
        self.assertIn(b"1 rejected", r.data)

    def test_behavioural_features_from_history(self):
        c = self.client("an@t.com")
        for _ in range(3):
            self.submit(c, {**NORMAL, "account_id": "ACC9500", "txn_time": ""})
        self.submit(c, {**NORMAL, "account_id": "ACC9500", "txn_time": ""})
        self.assertEqual(self.last_txn()["txn_count_24h"], 4)


if __name__ == "__main__":
    unittest.main()
