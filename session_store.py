"""
Session Store — persists DataChat chat sessions + their message history.

Without this, sessions live only in the in-memory ``_sessions`` dict in
orchestrator_api and are lost on every restart. This module mirrors the
``kg_store`` pattern: SQLite in dev/test, PostgreSQL in production.

In production (APP_ENV=production + SESSION_POSTGRES_DSN, or the shared
KG_POSTGRES_DSN) stores in PostgreSQL. Otherwise stores in SQLite
(SESSION_STORE_DB, default ``$DATA_DIR/session_store.db``).

On server startup call ``load_all()`` to restore every persisted session so
the RECENT list and conversation history survive a restart.

Public API
----------
save(session: dict)        → None    (upsert full session snapshot)
delete(session_id: str)    → None
load_all() -> List[dict]            (restore all persisted sessions)
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List

logger = logging.getLogger(__name__)

# ── Backend selection ──────────────────────────────────────────────────────────

def _is_production() -> bool:
    return os.environ.get("APP_ENV", "").strip().lower() == "production"

def _pg_dsn() -> str:
    # Prefer a session-specific DSN, fall back to the shared KG one.
    return os.environ.get("SESSION_POSTGRES_DSN", "") or os.environ.get("KG_POSTGRES_DSN", "")

def _use_postgres() -> bool:
    if not _is_production():
        return False
    if _pg_dsn():
        return True
    logger.warning("APP_ENV=production but no SESSION/KG_POSTGRES_DSN — session store uses SQLite.")
    return False

def _sqlite_path() -> str:
    explicit = os.environ.get("SESSION_STORE_DB")
    if explicit:
        return explicit
    data_dir = os.environ.get("DATA_DIR", "./data")
    return os.path.join(data_dir, "session_store.db")


@contextmanager
def _cursor_ctx() -> Iterator[Any]:
    if _use_postgres():
        import psycopg2, psycopg2.extras
        conn = psycopg2.connect(_pg_dsn(), cursor_factory=psycopg2.extras.RealDictCursor)
        cur  = conn.cursor()
        try:
            yield _PGCur(conn, cur)
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
            yield _SQLiteCur(conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


class _PGCur:
    def __init__(self, conn, cur): self._conn = conn; self._cur = cur
    def ddl(self, *stmts):
        for s in stmts: self._cur.execute(s)
    def execute(self, sql, params=()):
        self._cur.execute(sql.replace("?", "%s"), params); return self
    def fetchall(self): return [dict(r) for r in (self._cur.fetchall() or [])]
    def fetchone(self):
        r = self._cur.fetchone(); return dict(r) if r else None


class _SQLiteCur:
    def __init__(self, conn): self._conn = conn; self._cur = None
    def ddl(self, *stmts):
        for s in stmts: self._conn.execute(s)
    def execute(self, sql, params=()):
        self._cur = self._conn.execute(sql, params); return self
    def fetchall(self): return [dict(r) for r in (self._cur.fetchall() if self._cur else [])]
    def fetchone(self):
        r = self._cur.fetchone() if self._cur else None; return dict(r) if r else None


# ── DDL ────────────────────────────────────────────────────────────────────────

_DDL_SESSIONS = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id  TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT '',
    persona     TEXT NOT NULL DEFAULT 'business_user',
    source_id   TEXT,
    msg_count   INTEGER NOT NULL DEFAULT 0,
    data_json   TEXT NOT NULL DEFAULT '{}',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
)
"""


def _ensure(cur: Any) -> None:
    cur.ddl(_DDL_SESSIONS)


# Runtime-only keys that must not be persisted:
#   * live job handles / dialog id — recreated fresh on restart;
#   * db_connection — contains DB credentials. For source-backed sessions it is
#     re-hydrated from the (already-persisted) source on restore, so we never
#     duplicate credentials into this store.
_RUNTIME_KEYS = frozenset({
    "extract_job_id", "ontology_job_id", "kg_job_id", "dialog_session_id",
    "db_connection",
})


def _serializable(session: Dict) -> Dict:
    """Drop runtime-only fields and ensure the result is JSON-encodable."""
    return {k: v for k, v in session.items() if k not in _RUNTIME_KEYS}


# ── Public API ─────────────────────────────────────────────────────────────────

def save(session: Dict) -> None:
    """Upsert a chat session (config + full message history) into the store."""
    now = time.time()
    sid = session["id"]
    try:
        data_json = json.dumps(_serializable(session), default=str)
    except (TypeError, ValueError) as exc:
        logger.warning("session_store.save: %s not serializable, skipping (%s)", sid[:8], exc)
        return
    msg_count = len(session.get("messages") or [])

    with _cursor_ctx() as cur:
        _ensure(cur)
        cur.execute("SELECT session_id FROM chat_sessions WHERE session_id=?", (sid,))
        exists = cur.fetchone()
        if exists:
            cur.execute(
                "UPDATE chat_sessions SET title=?, persona=?, source_id=?, "
                "msg_count=?, data_json=?, updated_at=? WHERE session_id=?",
                (
                    session.get("title", ""),
                    session.get("persona", "business_user"),
                    session.get("source_id"),
                    msg_count, data_json, now, sid,
                ),
            )
        else:
            cur.execute(
                "INSERT INTO chat_sessions "
                "(session_id, title, persona, source_id, msg_count, data_json, "
                " created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    sid,
                    session.get("title", ""),
                    session.get("persona", "business_user"),
                    session.get("source_id"),
                    msg_count, data_json,
                    session.get("created_at", now), now,
                ),
            )
    logger.debug("session_store.save: %s (%d messages)", sid[:8], msg_count)


def delete(session_id: str) -> None:
    """Remove a session from the store."""
    with _cursor_ctx() as cur:
        _ensure(cur)
        cur.execute("DELETE FROM chat_sessions WHERE session_id=?", (session_id,))
    logger.info("session_store.delete: %s", session_id[:8])


def load_all() -> List[Dict]:
    """Restore all persisted sessions, newest first. Bad rows are skipped."""
    with _cursor_ctx() as cur:
        _ensure(cur)
        rows = cur.execute(
            "SELECT * FROM chat_sessions ORDER BY created_at DESC"
        ).fetchall()

    result: List[Dict] = []
    for r in rows:
        try:
            data = json.loads(r.get("data_json") or "{}")
        except Exception as exc:
            logger.warning("session_store.load_all: corrupt row %s skipped (%s)",
                           str(r.get("session_id"))[:8], exc)
            continue
        # Restore runtime-only fields to safe defaults.
        data.setdefault("id", r.get("session_id"))
        data["extract_job_id"]   = None
        data["ontology_job_id"]  = None
        data["kg_job_id"]        = None
        data["dialog_session_id"] = None
        result.append(data)
    logger.info("session_store.load_all: restored %d sessions", len(result))
    return result
