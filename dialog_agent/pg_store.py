"""
Unified database backend for KG federation tables (kg_registry, kg_bridges).

Backend is selected by APP_ENV:

  development / test (default)
      Always uses SQLite (KG_FEDERATION_DB, default data/kg_federation.db).
      No external database required — works out of the box.

  production  (APP_ENV=production)
      Uses PostgreSQL (AWS RDS, credentials from Secrets Manager — see
      pg_secrets.py at the repo root).

Usage
-----
    from dialog_agent import pg_store

    with pg_store.cursor_ctx() as cur:
        cur.ddl("CREATE TABLE IF NOT EXISTS ...")
        cur.execute("INSERT INTO t VALUES (?,?)", (1, "x"))
        rows = cur.fetchall()   # list[dict]
        row  = cur.fetchone()   # dict | None

SQL always uses ? placeholders; they are translated to %s for PostgreSQL.

Environment variables
---------------------
APP_ENV          "production" activates the PostgreSQL backend.
                 Any other value (or unset) uses SQLite + inline KG.
KG_POSTGRES_DSN  Explicit DSN override (local dev only) — see pg_secrets.py.
KG_FEDERATION_DB SQLite path used in dev/test (default: data/kg_federation.db)
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

def is_production() -> bool:
    """True when APP_ENV=production."""
    return os.environ.get("APP_ENV", "").strip().lower() == "production"


def is_postgres() -> bool:
    """
    True when the PostgreSQL backend is active.

    Rules:
      - production env  → PostgreSQL (credentials from Secrets Manager).
      - dev / test env  → always SQLite.
    """
    return is_production()


# Convenience accessor (read at call time so env overrides work after import)
def _sqlite_path() -> str: return os.environ.get("KG_FEDERATION_DB", "data/kg_federation.db")


@contextmanager
def cursor_ctx() -> Iterator[Any]:
    """
    Yield a backend-agnostic cursor wrapper.
    Commits on clean exit; rolls back and re-raises on error.
    """
    if is_postgres():
        import psycopg2.extras
        import pg_secrets
        conn = pg_secrets.connect(cursor_factory=psycopg2.extras.RealDictCursor)
        cur = conn.cursor()
        try:
            yield _PGCursor(conn, cur)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    else:
        import sqlite3
        path = _sqlite_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield _SQLiteCursor(conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


class _PGCursor:
    """Wraps a psycopg2 cursor; translates ? → %s placeholders."""

    __slots__ = ("_conn", "_cur")

    def __init__(self, conn: Any, cur: Any) -> None:
        self._conn = conn
        self._cur  = cur

    # ── DDL helper ────────────────────────────────────────────────────────────
    def ddl(self, *statements: str) -> None:
        """Execute one or more DDL statements (no parameter binding)."""
        for s in statements:
            self._cur.execute(s)

    # ── DML helpers ───────────────────────────────────────────────────────────
    def execute(self, sql: str, params: tuple = ()) -> "_PGCursor":
        self._cur.execute(sql.replace("?", "%s"), params)
        return self

    def fetchall(self) -> List[Dict]:
        return [dict(r) for r in (self._cur.fetchall() or [])]

    def fetchone(self) -> Optional[Dict]:
        r = self._cur.fetchone()
        return dict(r) if r else None

    def insert_returning_id(self, sql: str, params: tuple = ()) -> Optional[int]:
        """Execute an INSERT … RETURNING id statement and return the id."""
        self._cur.execute(sql.replace("?", "%s"), params)
        r = self._cur.fetchone()
        return dict(r)["id"] if r else None


class _SQLiteCursor:
    """Wraps a sqlite3 connection, returning rows as dicts."""

    __slots__ = ("_conn", "_cur")

    def __init__(self, conn: Any) -> None:
        self._conn = conn
        self._cur: Any = None

    # ── DDL helper ────────────────────────────────────────────────────────────
    def ddl(self, *statements: str) -> None:
        for s in statements:
            self._conn.execute(s)

    # ── DML helpers ───────────────────────────────────────────────────────────
    def execute(self, sql: str, params: tuple = ()) -> "_SQLiteCursor":
        self._cur = self._conn.execute(sql, params)
        return self

    def fetchall(self) -> List[Dict]:
        rows = self._cur.fetchall() if self._cur else []
        return [dict(r) for r in rows]

    def fetchone(self) -> Optional[Dict]:
        r = self._cur.fetchone() if self._cur else None
        return dict(r) if r else None

    def insert_returning_id(self, sql: str, params: tuple = ()) -> Optional[int]:
        """Execute INSERT and return lastrowid (no RETURNING clause needed)."""
        self._cur = self._conn.execute(sql, params)
        return self._cur.lastrowid
