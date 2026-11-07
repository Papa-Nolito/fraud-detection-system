-- Run automatically by:  DB_BACKEND=mysql python init_db.py
-- (create the empty database first:  CREATE DATABASE fraud_detection;)
CREATE TABLE IF NOT EXISTS users (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    full_name     VARCHAR(120) NOT NULL,
    email         VARCHAR(190) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role          ENUM('admin','analyst') NOT NULL,
    is_active     TINYINT(1) NOT NULL DEFAULT 1,
    created_at    VARCHAR(19) NOT NULL
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS transactions (
    id                    INT AUTO_INCREMENT PRIMARY KEY,
    account_id            VARCHAR(30) NOT NULL,
    amount                DOUBLE NOT NULL,
    merchant_category     VARCHAR(30) NOT NULL,
    channel               VARCHAR(10) NOT NULL,
    country               VARCHAR(2) NOT NULL,
    distance_from_home_km DOUBLE NOT NULL,
    new_device            TINYINT(1) NOT NULL,
    txn_time              VARCHAR(19) NOT NULL,
    hour                  TINYINT NOT NULL,
    day_of_week           TINYINT NOT NULL,
    txn_count_24h         INT NOT NULL,
    amount_ratio          DOUBLE NOT NULL,
    fraud_probability     DOUBLE NOT NULL,
    prediction            ENUM('fraud','legitimate') NOT NULL,
    risk_level            VARCHAR(10) NOT NULL,
    processing_ms         DOUBLE NOT NULL,
    submitted_by          INT NULL,
    created_at            VARCHAR(19) NOT NULL,
    INDEX idx_txn_account (account_id, txn_time),
    INDEX idx_txn_pred (prediction),
    CONSTRAINT fk_txn_user FOREIGN KEY (submitted_by) REFERENCES users(id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS alerts (
    id             INT AUTO_INCREMENT PRIMARY KEY,
    transaction_id INT NOT NULL UNIQUE,
    status         ENUM('open','confirmed_fraud','false_positive') NOT NULL DEFAULT 'open',
    reviewed_by    INT NULL,
    reviewed_at    VARCHAR(19) NULL,
    notes          TEXT NULL,
    created_at     VARCHAR(19) NOT NULL,
    CONSTRAINT fk_alert_txn FOREIGN KEY (transaction_id) REFERENCES transactions(id),
    CONSTRAINT fk_alert_user FOREIGN KEY (reviewed_by) REFERENCES users(id)
) ENGINE=InnoDB;
