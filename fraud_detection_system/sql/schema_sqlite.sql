CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name     TEXT NOT NULL,
    email         TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL CHECK (role IN ('admin','analyst')),
    is_active     INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transactions (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id            TEXT NOT NULL,
    amount                REAL NOT NULL,
    merchant_category     TEXT NOT NULL,
    channel               TEXT NOT NULL,
    country               TEXT NOT NULL,
    distance_from_home_km REAL NOT NULL,
    new_device            INTEGER NOT NULL,
    txn_time              TEXT NOT NULL,
    hour                  INTEGER NOT NULL,
    day_of_week           INTEGER NOT NULL,
    txn_count_24h         INTEGER NOT NULL,
    amount_ratio          REAL NOT NULL,
    fraud_probability     REAL NOT NULL,
    prediction            TEXT NOT NULL CHECK (prediction IN ('fraud','legitimate')),
    risk_level            TEXT NOT NULL,
    processing_ms         REAL NOT NULL,
    submitted_by          INTEGER REFERENCES users(id),
    created_at            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_txn_account ON transactions(account_id, txn_time);
CREATE INDEX IF NOT EXISTS idx_txn_pred ON transactions(prediction);

CREATE TABLE IF NOT EXISTS alerts (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id INTEGER NOT NULL UNIQUE REFERENCES transactions(id),
    status         TEXT NOT NULL DEFAULT 'open'
                   CHECK (status IN ('open','confirmed_fraud','false_positive')),
    reviewed_by    INTEGER REFERENCES users(id),
    reviewed_at    TEXT,
    notes          TEXT,
    created_at     TEXT NOT NULL
);
