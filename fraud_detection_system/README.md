# AI-Based Fraud Detection System
**Hamza Ombaso · SCT121-C004-0192/2025 · DIT 0306: ICT Project · JKUAT**

A web-based real-time fraud detection system built exactly to your proposal:
**Python + Flask** (backend) · **Scikit-learn + Pandas + NumPy** (ML) · **HTML/CSS/JavaScript** (frontend) ·
**MySQL** (database; SQLite also supported for zero-setup development) · **VS Code**.

---------------------------------------------------------------------

## 1. Quick start (Windows, VS Code) - 5 minutes

Open the project folder in VS Code, open a terminal (`Ctrl + ~`) and run:

```
python -m venv venv
venv\Scripts\activate            # PowerShell:  venv\Scripts\Activate.ps1
pip install -r requirements.txt
python generate_dataset.py       # 1. create the labelled training dataset
python train_model.py            # 2. train + compare models, save the best one
python init_db.py --demo         # 3. create tables, users and 300 demo transactions
python app.py                    # 4. start the web app
```

Open **http://127.0.0.1:5000** in Chrome.

| Role | Email | Password |
|---|---|---|
| Administrator | admin@frauddetect.local | Admin@12345 |
| Fraud Analyst | analyst@frauddetect.local | Analyst@12345 |

(Change these before any real use - set `ADMIN_PASSWORD` / `ANALYST_PASSWORD` environment variables before `init_db.py`.)

Run the automated tests any time: `python -m unittest discover -s tests -v`

---------------------------------------------------------------------

## 2. Project structure - what each file does

```
fraud_detection_system/
├── app.py                 Flask app: routes, security, fraud scoring workflow, alerts, reports
├── ml_engine.py           Data cleaning, feature engineering, model loading & prediction, explanations
├── generate_dataset.py    Builds the synthetic labelled dataset (20,000 rows, ~3% fraud)
├── train_model.py         Trains/compares models, tunes threshold, evaluates, saves model + charts
├── db.py                  Database layer (SQLite or MySQL, parameterised queries)
├── simulator.py           Live transaction simulator (sends transactions to the REST API)
├── init_db.py             Creates tables, default users, optional demo data
├── config.py              All settings (DB, secrets, SMTP) via environment variables
├── sql/schema_*.sql       Table definitions (users, transactions, alerts)
├── templates/  static/    Frontend (HTML, CSS, Chart.js dashboard)
├── tests/test_system.py   25 automated tests = your TC01-TC12 and NFT01-NFT06
├── models/                fraud_model.joblib + model_metrics.json (created by train_model.py)
├── data/                  transactions_synthetic.csv + sample_upload.csv
└── docs/figures/          confusion_matrix.png, roc_curves.png, model_comparison.png
```

---------------------------------------------------------------------

## 3. How the system works (IPOS framework from Section 2.5)

**INPUT** - an analyst submits a transaction (form) or uploads a CSV (`/transactions/new`).

**PROCESS** (`app.py → process_transaction → ml_engine`)
1. **Data cleaning/validation** - `clean_transaction()` rejects bad amounts, unknown categories, malformed IDs (functional requirement "validate before processing").
2. **Feature engineering** - hour, day of week, `is_night`, `log_amount` come from the transaction; **behavioural features come from the database**: `txn_count_24h` (how many transactions this account made in the last 24 h) and `amount_ratio` (this amount ÷ the account's usual average).
3. **Normalisation** - `StandardScaler` for numbers, `OneHotEncoder` for categories (inside one Scikit-learn `Pipeline`, so training and live use are identical).
4. **ML prediction** - Random Forest outputs a fraud probability; if it is ≥ the tuned threshold the transaction is **fraud**. Risk level: *low / medium / high*.

**STORAGE** - the transaction + prediction go to `transactions`; fraud creates a row in `alerts`.

**OUTPUT** - alert on the Alerts page (+ log line, + optional e-mail), dashboard, reports, CSV export.

Analysts open an alert, see **why it was flagged** (plain-language reasons), then click *Confirm fraud* or *False positive*. That feedback feeds the **live precision** figure on the Reports page - your feedback mechanism.

### Database design (for your ERD in Chapter 4)
```
users (id PK, full_name, email UNIQUE, password_hash, role, is_active, created_at)
   │1                                   │1
   │submits                             │reviews
   ▼N                                   ▼N
transactions (id PK, account_id, amount, merchant_category, channel, country,
              distance_from_home_km, new_device, txn_time, hour, day_of_week,
              txn_count_24h, amount_ratio, fraud_probability, prediction,
              risk_level, processing_ms, submitted_by FK→users, created_at)
   │1
   │raises
   ▼0..1
alerts (id PK, transaction_id FK UNIQUE, status, reviewed_by FK→users,
        reviewed_at, notes, created_at)
```
Draw this in Draw.io as your ERD. Entities: **users, transactions, alerts**.

---------------------------------------------------------------------

## 3b. Real-time features (simulator, REST API, live dashboard)

| Feature | File | What it does |
|---|---|---|
| REST API | `app.py` -> `POST /api/transactions` | Another system sends a transaction as JSON + header `X-API-Key`; gets the fraud decision back instantly |
| Simulator | `simulator.py` | Acts like a bank's payment system: sends a new transaction every few seconds, including fraud bursts |
| Auto-refresh + pop-up alerts | `static/live.js`, `GET /api/live` | Browser checks every 5 s; dashboard/alerts/transactions pages reload on new data; red badge on "Alerts"; pop-up + beep on new fraud |

**Run the live demo (needs two terminals):**
```
# Terminal 1
python app.py
# Terminal 2 (activate venv first)
python simulator.py                          # Ctrl+C to stop
python simulator.py --interval 1 --fraud-rate 0.3   # faster, more fraud
```
Log in to http://127.0.0.1:5000 and keep the Dashboard open.

**Try the API by hand (PowerShell):**
```
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/api/transactions `
  -Headers @{"X-API-Key"="dev-api-key-change-me"} -ContentType "application/json" `
  -Body '{"account_id":"ACC5555","amount":900000,"merchant_category":"crypto","channel":"web","country":"NG","distance_from_home_km":3000,"new_device":1}'
```
Change the key before real use: set `FRAUD_API_KEY` (and pass the same value to the simulator with `--api-key`).

---------------------------------------------------------------------

## 4. Using MySQL (as named in your proposal)

1. Install MySQL Server (or XAMPP) and start it.
2. Create the empty database: `CREATE DATABASE fraud_detection;`
3. Set environment variables, then initialise (PowerShell shown):
```
$env:DB_BACKEND="mysql"
$env:MYSQL_USER="root"
$env:MYSQL_PASSWORD="your_password"
$env:MYSQL_DB="fraud_detection"
python init_db.py --demo
python app.py
```
(Command Prompt: use `set DB_BACKEND=mysql` etc.) The same code runs on both databases; only `db.py` knows the difference.

> Note: the automated tests and the live walkthrough were run on SQLite. The MySQL schema and driver code follow the same logic, but test it on your machine and tell me if you hit any error.

---------------------------------------------------------------------

## 5. Model results (from the last training run)

See `models/model_metrics.json` and the **Reports** page. Typical results on the 4,000-row unseen test set:
Random Forest ≈ precision 0.87, recall 0.76, F1 0.81; Logistic Regression and Gradient Boosting are close; Isolation Forest (unsupervised anomaly detection) is lower. Figures for Chapter 5 are in `docs/figures/`.

Important to explain honestly in your report:
* **The dataset is synthetic.** It is realistic (rare fraud, overlapping behaviour, 15% "stealth" fraud) but it is not real bank data. Your proposal's Scope & Limitations already says results depend on dataset quality. If you can obtain a public dataset (e.g. the Kaggle "Credit Card Fraud Detection" dataset), you can retrain by pointing `--data` at it after renaming columns to match `ml_engine.RAW_COLUMNS` plus `is_fraud`.
* **Accuracy alone is misleading** with 3% fraud: a model that says "legitimate" every time scores 97%. That is why you report precision, recall and F1 (your section 5.5.1) and the false-positive rate (objective 5).
* **The threshold is tuned** on cross-validated training predictions (not the test set) to maximise F1.

---------------------------------------------------------------------

## 6. Try these demos (good for your presentation)

| Scenario | Fill in | Expected |
|---|---|---|
| Normal | ACC1001, 1500, groceries, pos, KE, 2 km, no new device | Legitimate |
| Suspicious | ACC1001, 950000, crypto, web, NG, 3000 km, new device ✔, time 03:00 | **Fraud**, alert raised |
| Batch | Upload `data/sample_upload.csv` | Mix of legit + flagged |

---------------------------------------------------------------------

## 7. Mapping to your report chapters

| Your chapter/section | Where it lives in the project |
|---|---|
| 3.3.3 Development | `app.py`, `ml_engine.py`, `templates/` |
| 4.2.3 Functional requirements | Login, submit/upload, validation, pre-processing, ML, alerts, storage, dashboard, review, reports - all implemented |
| 4.3.1 Architecture components | UI = templates; Backend = `app.py`; Pre-processing = `ml_engine.py`; ML = `ml_engine.FraudModel`; DB = `db.py`+`sql/`; Alerts/Reporting = `/alerts`, `/reports` |
| 5.2.1-5.2.4 Testing | `tests/test_system.py` (unit/integration/system); UAT = you + a classmate using the app |
| 5.4 Test cases (Tables 5.1/5.2) | each test is named `test_TC01…`, `test_NFT01…` - run with `-v` and paste results into the "Actual Result" column |
| 5.5 Model evaluation | `train_model.py` output, `model_metrics.json`, `docs/figures/` |

---------------------------------------------------------------------

## 8. Security features included
Password hashing (Werkzeug/PBKDF2), session login, role-based access (admin-only user management), CSRF protection on every form, parameterised SQL (no SQL injection), HTML auto-escaping (no XSS), input validation, upload size/type limits, safe redirect handling.

## 9. Before real deployment (limitations / future work - good for Chapter 6)
Use a strong `SECRET_KEY`; serve with Gunicorn/Waitress behind HTTPS instead of the Flask dev server; add login rate-limiting; retrain periodically with analyst-confirmed labels; integrate with a real transaction processing system via an API; add model drift monitoring.
