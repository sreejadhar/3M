"""
Unified database backend for KG federation tables (kg_registry, kg_bridges).

Set KG_POSTGRES_DSN to a psycopg2-compatible connection string to persist
federation metadata to PostgreSQL.  Falls back to SQLite otherwise.

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
KG_POSTGRES_DSN  psycopg2 DSN, e.g. "host=localhost dbname=datachat user=app password=secret"
KG_FEDERATION_DB Path to SQLite file when PG is not configured (default: data/kg_federation.db)
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

KG_POSTGRES_DSN: str = os.environ.get("KG_POSTGRES_DSN", "")
SQLITE_PATH:     str = os.environ.get("KG_FEDERATION_DB", "data/kg_federation.db")


def is_postgres() -> bool:
    """True when KG_POSTGRES_DSN is set."""
    return bool(KG_POSTGRES_DSN)


@contextmanager
def cursor_ctx() -> Iterator[Any]:
    """
    Yield a backend-agnostic cursor wrapper.
    Commits on clean exit; rolls back and re-raises on error.
    """
    if is_postgres():
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(
            KG_POSTGRES_DSN,
            cursor_factory=psycopg2.extras.RealDictCursor,
        )
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
        os.makedirs(os.path.dirname(SQLITE_PATH) or ".", exist_ok=True)
        conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
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
