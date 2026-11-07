"""Tiny database layer. All SQL in the project is written with '?' placeholders
(parameterised queries = protection against SQL injection). This module converts them
to '%s' for MySQL so the same code runs on SQLite and MySQL."""
import os
import sqlite3

from flask import current_app, g


def _is_mysql():
    return current_app.config["DB_BACKEND"] == "mysql"


def get_db():
    if "db" not in g:
        cfg = current_app.config
        if _is_mysql():
            import mysql.connector  # imported lazily: only needed for MySQL
            g.db = mysql.connector.connect(
                host=cfg["MYSQL_HOST"], port=cfg["MYSQL_PORT"], user=cfg["MYSQL_USER"],
                password=cfg["MYSQL_PASSWORD"], database=cfg["MYSQL_DB"],
            )
        else:
            conn = sqlite3.connect(cfg["SQLITE_PATH"])
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            g.db = conn
    return g.db


def _sql(sql):
    return sql.replace("?", "%s") if _is_mysql() else sql


def query_all(sql, params=()):
    conn = get_db()
    params = tuple(params)
    if _is_mysql():
        cur = conn.cursor(dictionary=True)
        cur.execute(_sql(sql), params)
        rows = cur.fetchall()
        cur.close()
        return rows
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def query_one(sql, params=()):
    rows = query_all(sql, params)
    return rows[0] if rows else None


def execute(sql, params=()):
    """Run INSERT/UPDATE/DELETE, commit, and return the new row id."""
    conn = get_db()
    params = tuple(params)
    if _is_mysql():
        cur = conn.cursor()
        cur.execute(_sql(sql), params)
        conn.commit()
        last_id = cur.lastrowid
        cur.close()
        return last_id
    cur = conn.execute(sql, params)
    conn.commit()
    return cur.lastrowid


def close_db(_exc=None):
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def init_app(app):
    app.teardown_appcontext(close_db)


def init_db():
    """Create all tables (safe to run repeatedly)."""
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sql")
    conn = get_db()
    if _is_mysql():
        with open(os.path.join(base, "schema_mysql.sql"), encoding="utf-8") as f:
            statements = [s.strip() for s in f.read().split(";") if s.strip()]
        cur = conn.cursor()
        for stmt in statements:
            cur.execute(stmt)
        conn.commit()
        cur.close()
    else:
        with open(os.path.join(base, "schema_sqlite.sql"), encoding="utf-8") as f:
            conn.executescript(f.read())
        conn.commit()
