"""Central configuration. Every value can be overridden with an environment variable,
so you never have to edit code to change the database or secrets."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "change-this-in-production")
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"

    # --- Database -------------------------------------------------------
    # "sqlite" works instantly with zero setup (good for development/testing).
    # "mysql" is the DBMS named in the proposal (see README for setup).
    DB_BACKEND = os.environ.get("DB_BACKEND", "sqlite")
    SQLITE_PATH = os.environ.get("SQLITE_PATH", os.path.join(BASE_DIR, "fraud.db"))
    MYSQL_HOST = os.environ.get("MYSQL_HOST", "localhost")
    MYSQL_PORT = int(os.environ.get("MYSQL_PORT", "3306"))
    MYSQL_USER = os.environ.get("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "")
    MYSQL_DB = os.environ.get("MYSQL_DB", "fraud_detection")

    # --- Machine learning ------------------------------------------------
    MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_model.joblib")
    METRICS_PATH = os.path.join(BASE_DIR, "models", "model_metrics.json")
    DEFAULT_AVG_AMOUNT = 3000.0   # used for accounts with < 3 past transactions (KES)

    # --- Alert e-mail (optional; leave SMTP_HOST empty to disable) --------
    SMTP_HOST = os.environ.get("SMTP_HOST", "")
    SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
    SMTP_USER = os.environ.get("SMTP_USER", "")
    SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
    ALERT_EMAIL_TO = os.environ.get("ALERT_EMAIL_TO", "")

    # --- REST API (for external systems / the simulator) --------------------
    # Any system that sends this key in the X-API-Key header may submit transactions.
    API_KEY = os.environ.get("FRAUD_API_KEY", "dev-api-key-change-me")

    MAX_UPLOAD_ROWS = 5000
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024  # 2 MB uploads
