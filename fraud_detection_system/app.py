"""Fraud Detection System - Flask web application.

Modules (as listed in the project deliverables):
  * Authentication & roles ........ /login, /admin/users
  * Transaction Monitoring ........ /transactions, /transactions/new, /transactions/upload
  * Fraud Detection ............... score_and_store()  (uses ml_engine.FraudModel)
  * Fraud Alert Notification ...... /alerts, notify_alert()
  * Reporting & Dashboard ......... /dashboard, /reports, /reports/export.csv
"""
import csv
import io
import json
import logging
import os
import secrets
import smtplib
import time
from datetime import datetime, timedelta
from email.message import EmailMessage
from functools import wraps

from flask import (Flask, Response, abort, flash, g, jsonify, redirect, render_template,
                   request, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

import db
import ml_engine as ml
from config import Config

log = logging.getLogger("fraud")
PER_PAGE = 15


def now_str():
    return datetime.now().strftime(ml.TIME_FMT)


def num(x, default=0.0):
    return float(x) if x is not None else default


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)
    db.init_app(app)

    model = ml.FraudModel(app.config["MODEL_PATH"])
    model.load()
    app.extensions["fraud_model"] = model

    # ------------------------------------------------------------------ security
    @app.before_request
    def load_user_and_check_csrf():
        if request.endpoint == "static":
            return
        g.user = None
        uid = session.get("user_id")
        if uid:
            user = db.query_one("SELECT id, full_name, email, role, is_active FROM users WHERE id=?", (uid,))
            if user and user["is_active"]:
                g.user = user
            else:
                session.clear()
        if "csrf" not in session:
            session["csrf"] = secrets.token_hex(16)
        # /api/* uses an API key instead of a browser CSRF token
        if request.method == "POST" and not request.path.startswith("/api/"):
            token = request.form.get("csrf_token", "")
            if not token or not secrets.compare_digest(token, session["csrf"]):
                abort(400, "Invalid or missing security token. Reload the page and try again.")

    @app.context_processor
    def inject_globals():
        return {"csrf_token": session.get("csrf", ""), "current_user": getattr(g, "user", None)}

    def login_required(view):
        @wraps(view)
        def wrapped(*a, **kw):
            if g.user is None:
                return redirect(url_for("login", next=request.path))
            return view(*a, **kw)
        return wrapped

    def admin_required(view):
        @wraps(view)
        @login_required
        def wrapped(*a, **kw):
            if g.user["role"] != "admin":
                abort(403)
            return view(*a, **kw)
        return wrapped

    @app.template_filter("kes")
    def kes(v):
        return f"KES {float(v):,.2f}"

    @app.template_filter("pct")
    def pct(v):
        return f"{float(v) * 100:.1f}%"

    # ------------------------------------------------------------- core detection
    def notify_alert(txn):
        """Alert notification: always logged; e-mailed if SMTP is configured."""
        msg = (f"FRAUD ALERT - txn #{txn['id']} account {txn['account_id']} "
               f"{txn['amount']:,.2f} risk={txn['risk_level']} p={txn['fraud_probability']:.2f}")
        log.warning(msg)
        cfg = app.config
        if cfg["SMTP_HOST"] and cfg["ALERT_EMAIL_TO"]:
            try:
                email = EmailMessage()
                email["Subject"], email["From"], email["To"] = msg, cfg["SMTP_USER"], cfg["ALERT_EMAIL_TO"]
                email.set_content(msg + "\nLog in to the Fraud Detection System to review it.")
                with smtplib.SMTP(cfg["SMTP_HOST"], cfg["SMTP_PORT"], timeout=10) as s:
                    s.starttls()
                    s.login(cfg["SMTP_USER"], cfg["SMTP_PASSWORD"])
                    s.send_message(email)
            except Exception as exc:                     # never let e-mail failure block scoring
                log.error("Alert e-mail failed: %s", exc)

    def score_and_store(clean, user_id, txn_count_24h, amount_ratio, created_at=None):
        """Predict -> store transaction + prediction -> raise alert if fraudulent."""
        t0 = time.perf_counter()
        row = {**clean, "txn_count_24h": txn_count_24h, "amount_ratio": amount_ratio}
        result = model.predict(row)
        ms = (time.perf_counter() - t0) * 1000
        created = created_at or now_str()
        txn_id = db.execute(
            """INSERT INTO transactions (account_id, amount, merchant_category, channel, country,
               distance_from_home_km, new_device, txn_time, hour, day_of_week, txn_count_24h,
               amount_ratio, fraud_probability, prediction, risk_level, processing_ms,
               submitted_by, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (clean["account_id"], clean["amount"], clean["merchant_category"], clean["channel"],
             clean["country"], clean["distance_from_home_km"], clean["new_device"],
             clean["txn_time"].strftime(ml.TIME_FMT), clean["hour"], clean["day_of_week"],
             txn_count_24h, round(amount_ratio, 4), result["fraud_probability"],
             result["prediction"], result["risk_level"], ms, user_id, created))
        if result["prediction"] == "fraud":
            db.execute("INSERT INTO alerts (transaction_id, status, created_at) VALUES (?, 'open', ?)",
                       (txn_id, created))
            notify_alert({"id": txn_id, "account_id": clean["account_id"], "amount": clean["amount"],
                          **result})
        return {"id": txn_id, "processing_ms": ms, **result}

    def process_transaction(clean, user_id):
        """Derives the *behavioural* features from the account's own history, then scores."""
        ts = clean["txn_time"]
        n24 = db.query_one(
            "SELECT COUNT(*) AS n FROM transactions WHERE account_id=? AND txn_time>=? AND txn_time<=?",
            (clean["account_id"], (ts - timedelta(hours=24)).strftime(ml.TIME_FMT),
             ts.strftime(ml.TIME_FMT)))["n"]
        hist = db.query_one(
            """SELECT COUNT(*) AS n, AVG(amount) AS avg_amount FROM transactions
               WHERE account_id=? AND prediction='legitimate' AND txn_time<?""",
            (clean["account_id"], ts.strftime(ml.TIME_FMT)))
        avg = num(hist["avg_amount"]) if hist["n"] >= 3 else app.config["DEFAULT_AVG_AMOUNT"]
        ratio = min(clean["amount"] / avg, 50.0)
        return score_and_store(clean, user_id, int(n24) + 1, ratio)

    app.score_and_store = score_and_store      # exposed for init_db.py (demo data) and tests

    # --------------------------------------------------------------------- auth
    @app.route("/")
    def index():
        return redirect(url_for("dashboard" if g.user else "login"))

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            user = db.query_one("SELECT * FROM users WHERE email=?", (email,))
            if user and user["is_active"] and check_password_hash(user["password_hash"],
                                                                  request.form.get("password", "")):
                session.clear()
                session["user_id"] = user["id"]
                session["csrf"] = secrets.token_hex(16)
                nxt = request.args.get("next", "")
                return redirect(nxt if nxt.startswith("/") and not nxt.startswith("//")
                                else url_for("dashboard"))
            flash("Invalid email or password.", "error")
            return render_template("login.html"), 401
        return render_template("login.html")

    @app.post("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

    # --------------------------------------------------------------- dashboard
    @app.route("/dashboard")
    @login_required
    def dashboard():
        t = db.query_one("""SELECT COUNT(*) AS total,
                SUM(CASE WHEN prediction='fraud' THEN 1 ELSE 0 END) AS flagged,
                SUM(CASE WHEN prediction='fraud' THEN amount ELSE 0 END) AS flagged_amount,
                AVG(processing_ms) AS avg_ms FROM transactions""")
        total, flagged = int(num(t["total"])), int(num(t["flagged"]))
        stats = {"total": total, "flagged": flagged,
                 "rate": flagged / total if total else 0,
                 "flagged_amount": num(t["flagged_amount"]), "avg_ms": num(t["avg_ms"]),
                 "open_alerts": db.query_one("SELECT COUNT(*) AS n FROM alerts WHERE status='open'")["n"]}
        trend = db.query_all("""SELECT SUBSTR(created_at,1,10) AS d, COUNT(*) AS total,
                SUM(CASE WHEN prediction='fraud' THEN 1 ELSE 0 END) AS fraud
                FROM transactions GROUP BY d ORDER BY d DESC LIMIT 14""")[::-1]
        trend = [{"d": r["d"], "total": int(r["total"]), "fraud": int(num(r["fraud"]))} for r in trend]
        risk = {r["risk_level"]: int(r["n"]) for r in db.query_all(
            "SELECT risk_level, COUNT(*) AS n FROM transactions GROUP BY risk_level")}
        recent = db.query_all("""SELECT a.id AS alert_id, a.status, t.account_id, t.amount,
                t.risk_level, t.fraud_probability, t.created_at FROM alerts a
                JOIN transactions t ON t.id=a.transaction_id ORDER BY a.id DESC LIMIT 6""")
        return render_template("dashboard.html", stats=stats, trend=trend, risk=risk, recent=recent)

    # ------------------------------------------------------------ transactions
    def form_defaults():
        return {"txn_time": "", "country": "KE", "channel": "pos", "merchant_category": "groceries",
                "distance_from_home_km": "0"}

    @app.route("/transactions/new", methods=["GET", "POST"])
    @login_required
    def new_transaction():
        if request.method == "POST":
            if not model.ready:
                flash("The ML model is not trained yet. Run: python train_model.py", "error")
                return redirect(url_for("new_transaction"))
            clean, errors = ml.clean_transaction(request.form)
            if errors:
                for e in errors:
                    flash(e, "error")
                return render_template("new_transaction.html", cats=ml.MERCHANT_CATEGORIES,
                                       channels=ml.CHANNELS, countries=ml.COUNTRIES,
                                       v=request.form), 400
            res = process_transaction(clean, g.user["id"])
            flash(f"Transaction scored in {res['processing_ms']:.0f} ms: "
                  f"{res['prediction'].upper()} ({res['fraud_probability'] * 100:.1f}% fraud probability).",
                  "error" if res["prediction"] == "fraud" else "success")
            return redirect(url_for("transaction_detail", txn_id=res["id"]))
        return render_template("new_transaction.html", cats=ml.MERCHANT_CATEGORIES,
                               channels=ml.CHANNELS, countries=ml.COUNTRIES, v=form_defaults())

    @app.post("/transactions/upload")
    @login_required
    def upload_transactions():
        if not model.ready:
            flash("The ML model is not trained yet. Run: python train_model.py", "error")
            return redirect(url_for("new_transaction"))
        f = request.files.get("file")
        if not f or not f.filename.lower().endswith(".csv"):
            flash("Please choose a .csv file.", "error")
            return redirect(url_for("new_transaction"))
        try:
            reader = csv.DictReader(io.StringIO(f.read().decode("utf-8-sig")))
        except UnicodeDecodeError:
            flash("The file is not valid UTF-8 text.", "error")
            return redirect(url_for("new_transaction"))
        needed = {"account_id", "amount", "merchant_category", "channel", "country"}
        if not reader.fieldnames or not needed.issubset(set(reader.fieldnames)):
            flash("CSV must contain the columns: " + ", ".join(sorted(needed)), "error")
            return redirect(url_for("new_transaction"))
        ok = flagged = 0
        bad = []
        for i, raw in enumerate(reader, start=2):          # row 1 is the header
            if i - 1 > app.config["MAX_UPLOAD_ROWS"]:
                bad.append(f"Stopped after {app.config['MAX_UPLOAD_ROWS']} rows.")
                break
            clean, errors = ml.clean_transaction(raw)
            if errors:
                bad.append(f"Row {i}: {errors[0]}")
                continue
            res = process_transaction(clean, g.user["id"])
            ok += 1
            flagged += res["prediction"] == "fraud"
        flash(f"Batch complete: {ok} scored, {flagged} flagged as fraud, {len(bad)} rejected.",
              "success" if ok else "error")
        for msg in bad[:5]:
            flash(msg, "error")
        return redirect(url_for("transactions"))

    @app.route("/transactions")
    @login_required
    def transactions():
        where, params = ["1=1"], []
        pred = request.args.get("prediction", "")
        q = request.args.get("q", "").strip()
        if pred in ("fraud", "legitimate"):
            where.append("t.prediction=?")
            params.append(pred)
        if q:
            where.append("t.account_id LIKE ?")
            params.append(f"%{q}%")
        clause = " AND ".join(where)
        total = db.query_one(f"SELECT COUNT(*) AS n FROM transactions t WHERE {clause}", params)["n"]
        page = max(request.args.get("page", 1, type=int), 1)
        rows = db.query_all(
            f"""SELECT t.*, a.id AS alert_id FROM transactions t
                LEFT JOIN alerts a ON a.transaction_id=t.id WHERE {clause}
                ORDER BY t.id DESC LIMIT ? OFFSET ?""", params + [PER_PAGE, (page - 1) * PER_PAGE])
        pages = max((total + PER_PAGE - 1) // PER_PAGE, 1)
        return render_template("transactions.html", rows=rows, page=page, pages=pages, total=total,
                               pred=pred, q=q)

    @app.route("/transactions/<int:txn_id>")
    @login_required
    def transaction_detail(txn_id):
        t = db.query_one("""SELECT t.*, a.id AS alert_id, a.status AS alert_status
                FROM transactions t LEFT JOIN alerts a ON a.transaction_id=t.id WHERE t.id=?""", (txn_id,))
        if not t:
            abort(404)
        return render_template("transaction_detail.html", t=t, reasons=ml.explain(t))

    # ------------------------------------------------------------------ alerts
    @app.route("/alerts")
    @login_required
    def alerts():
        status = request.args.get("status", "open")
        sql = """SELECT a.id, a.status, a.created_at, t.id AS txn_id, t.account_id, t.amount,
                 t.risk_level, t.fraud_probability FROM alerts a
                 JOIN transactions t ON t.id=a.transaction_id"""
        params = ()
        if status in ("open", "confirmed_fraud", "false_positive"):
            sql += " WHERE a.status=?"
            params = (status,)
        rows = db.query_all(sql + " ORDER BY t.fraud_probability DESC, a.id DESC LIMIT 200", params)
        return render_template("alerts.html", rows=rows, status=status)

    @app.route("/alerts/<int:alert_id>", methods=["GET", "POST"])
    @login_required
    def alert_detail(alert_id):
        a = db.query_one("""SELECT a.*, u.full_name AS reviewer FROM alerts a
                LEFT JOIN users u ON u.id=a.reviewed_by WHERE a.id=?""", (alert_id,))
        if not a:
            abort(404)
        if request.method == "POST":
            decision = request.form.get("decision")
            if decision not in ("confirmed_fraud", "false_positive"):
                abort(400)
            db.execute("UPDATE alerts SET status=?, reviewed_by=?, reviewed_at=?, notes=? WHERE id=?",
                       (decision, g.user["id"], now_str(), request.form.get("notes", "")[:500], alert_id))
            flash("Alert review saved. Thank you - this feedback measures real-world accuracy.", "success")
            return redirect(url_for("alerts"))
        t = db.query_one("SELECT * FROM transactions WHERE id=?", (a["transaction_id"],))
        return render_template("alert_detail.html", a=a, t=t, reasons=ml.explain(t))

    # ----------------------------------------------------------------- reports
    def parse_range():
        out = []
        for key in ("from", "to"):
            try:
                out.append(datetime.strptime(request.args.get(key, ""), "%Y-%m-%d").strftime("%Y-%m-%d"))
            except ValueError:
                out.append("")
        return out

    def range_clause(frm, to, col="created_at"):
        parts, params = [], []
        if frm:
            parts.append(f"{col} >= ?"); params.append(frm + " 00:00:00")
        if to:
            parts.append(f"{col} <= ?"); params.append(to + " 23:59:59")
        return (" WHERE " + " AND ".join(parts)) if parts else "", params

    @app.route("/reports")
    @login_required
    def reports():
        frm, to = parse_range()
        clause, params = range_clause(frm, to)
        s = db.query_one(f"""SELECT COUNT(*) AS total,
                SUM(CASE WHEN prediction='fraud' THEN 1 ELSE 0 END) AS flagged,
                SUM(amount) AS volume,
                SUM(CASE WHEN prediction='fraud' THEN amount ELSE 0 END) AS flagged_amount,
                AVG(processing_ms) AS avg_ms FROM transactions{clause}""", params)
        summary = {"total": int(num(s["total"])), "flagged": int(num(s["flagged"])),
                   "volume": num(s["volume"]), "flagged_amount": num(s["flagged_amount"]),
                   "avg_ms": num(s["avg_ms"])}
        by_cat = db.query_all(f"""SELECT merchant_category, COUNT(*) AS total,
                SUM(CASE WHEN prediction='fraud' THEN 1 ELSE 0 END) AS flagged
                FROM transactions{clause} GROUP BY merchant_category ORDER BY flagged DESC""", params)
        by_cat = [{"cat": r["merchant_category"], "total": int(r["total"]),
                   "flagged": int(num(r["flagged"]))} for r in by_cat]
        a_clause, a_params = range_clause(frm, to, "a.created_at")
        fb = {r["status"]: int(r["n"]) for r in db.query_all(
            f"SELECT a.status, COUNT(*) AS n FROM alerts a{a_clause} GROUP BY a.status", a_params)}
        reviewed = fb.get("confirmed_fraud", 0) + fb.get("false_positive", 0)
        live_precision = fb.get("confirmed_fraud", 0) / reviewed if reviewed else None
        metrics = None
        if os.path.exists(app.config["METRICS_PATH"]):
            with open(app.config["METRICS_PATH"]) as fh:
                metrics = json.load(fh)
        return render_template("reports.html", summary=summary, by_cat=by_cat, fb=fb,
                               live_precision=live_precision, reviewed=reviewed, metrics=metrics,
                               frm=frm, to=to)

    @app.route("/reports/export.csv")
    @login_required
    def export_csv():
        frm, to = parse_range()
        clause, params = range_clause(frm, to)
        rows = db.query_all(f"SELECT * FROM transactions{clause} ORDER BY id", params)
        buf = io.StringIO()
        if rows:
            w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        return Response(buf.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=transactions_report.csv"})

    # ------------------------------------------------------------ administration
    @app.route("/admin/users", methods=["GET", "POST"])
    @admin_required
    def admin_users():
        if request.method == "POST":
            action = request.form.get("action")
            if action == "create":
                name = request.form.get("full_name", "").strip()
                email = request.form.get("email", "").strip().lower()
                pw = request.form.get("password", "")
                role = request.form.get("role")
                if not name or "@" not in email or role not in ("admin", "analyst") or len(pw) < 8:
                    flash("Provide a name, valid email, role and a password of at least 8 characters.", "error")
                elif db.query_one("SELECT id FROM users WHERE email=?", (email,)):
                    flash("That email is already registered.", "error")
                else:
                    db.execute("""INSERT INTO users (full_name,email,password_hash,role,is_active,created_at)
                                  VALUES (?,?,?,?,1,?)""",
                               (name, email, generate_password_hash(pw), role, now_str()))
                    flash(f"User {email} created.", "success")
            elif action == "toggle":
                uid = request.form.get("user_id", type=int)
                if uid == g.user["id"]:
                    flash("You cannot deactivate your own account.", "error")
                else:
                    db.execute("UPDATE users SET is_active = 1 - is_active WHERE id=?", (uid,))
                    flash("User status updated.", "success")
            return redirect(url_for("admin_users"))
        users = db.query_all("SELECT id, full_name, email, role, is_active, created_at FROM users ORDER BY id")
        return render_template("admin_users.html", users=users)

    # ----------------------------------------------------------------- REST API
    def api_key_ok():
        supplied = request.headers.get("X-API-Key", "")
        return bool(supplied) and secrets.compare_digest(supplied, app.config["API_KEY"])

    @app.post("/api/transactions")
    def api_submit_transaction():
        """Real-time ingestion endpoint: another system POSTs a transaction as JSON and gets
        the fraud decision back immediately."""
        if not api_key_ok():
            return jsonify(error="Missing or invalid X-API-Key header."), 401
        if not model.ready:
            return jsonify(error="Model not trained. Run train_model.py."), 503
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Send a JSON object as the request body."), 400
        clean, errors = ml.clean_transaction(payload)
        if errors:
            return jsonify(error="Validation failed.", details=errors), 400
        res = process_transaction(clean, None)
        alert = db.query_one("SELECT id FROM alerts WHERE transaction_id=?", (res["id"],))
        return jsonify(transaction_id=res["id"], prediction=res["prediction"],
                       fraud_probability=round(res["fraud_probability"], 4),
                       risk_level=res["risk_level"], processing_ms=round(res["processing_ms"], 1),
                       alert_id=alert["id"] if alert else None), 201

    @app.get("/api/live")
    def api_live():
        """Polled by static/live.js every few seconds to refresh pages and pop up alerts."""
        if g.user is None:
            return jsonify(error="login required"), 401
        total = db.query_one("SELECT COUNT(*) AS n FROM transactions")["n"]
        open_alerts = db.query_one("SELECT COUNT(*) AS n FROM alerts WHERE status='open'")["n"]
        latest = db.query_one("""SELECT a.id, t.account_id, t.amount, t.risk_level FROM alerts a
                JOIN transactions t ON t.id=a.transaction_id ORDER BY a.id DESC LIMIT 1""")
        return jsonify(total=int(total), open_alerts=int(open_alerts),
                       latest_alert_id=latest["id"] if latest else 0, latest_alert=latest)

    # ------------------------------------------------------------------- errors
    @app.errorhandler(400)
    @app.errorhandler(403)
    @app.errorhandler(404)
    def http_error(e):
        return render_template("error.html", e=e), e.code

    return app


if __name__ == "__main__":
    create_app().run(debug=os.environ.get("FLASK_DEBUG") == "1")
