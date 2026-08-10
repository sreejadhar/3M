"""
Governed business glossary term registry — DAMA-DMBOK-style term lifecycle
(draft/candidate/approved/deprecated), steward, many-to-many term<->asset
(column/table) links across sources, and an append-only audit log of every
edit/approval.

Standalone module (same SQLite/PostgreSQL backend pattern as
metadata_catalog.py/kpi_store.py, same data/metadata.db file so
biz_glossary_term_assets rows can reference metadata_catalog's metadata_id/attr_id) —
importable from orchestrator_api.py without pulling in dialog_agent.

This is the GOVERNANCE layer only: term storage, lifecycle, links, audit.

Public API
----------
get_term(term_id)                              -> dict | None
list_terms(source_id=None, status=None, domain=None)   -> List[dict]
update_term(term_id, changed_by="", **fields)  -> bool   (writes audit rows)
approve_term(term_id, changed_by="")           -> bool
reject_term(term_id, changed_by="")            -> bool
list_assets_for_term(term_id)                  -> List[dict]
get_asset_link(metadata_id, attr_id="")        -> dict | None
list_audit(term_id)                            -> List[dict]
"""
from __future__ import annotations

import logging
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

VALID_STATUSES = {"draft", "candidate", "approved", "deprecated"}


# ── Environment helpers (mirrors metadata_catalog.py) ───────────────────────────

def _is_production() -> bool:
    return os.environ.get("APP_ENV", "").strip().lower() == "production"


def _is_postgres() -> bool:
    if not _is_production():
        return False
    return bool(os.environ.get("KG_POSTGRES_DSN", ""))


def _pg_dsn() -> str:
    return os.environ.get("KG_POSTGRES_DSN", "")


def _sqlite_path() -> str:
    return os.environ.get("METADATA_DB", "data/metadata.db")


# ── Cursor abstraction (mirrors metadata_catalog.py / kpi_store.py) ────────────

class _PGCur:
    def __init__(self, conn: Any, cur: Any) -> None:
        self._conn, self._cur = conn, cur

    def ddl(self, *stmts: str) -> None:
        for s in stmts:
            self._cur.execute(s)

    def execute(self, sql: str, params: tuple = ()) -> "_PGCur":
        self._cur.execute(sql.replace("?", "%s"), params)
        return self

    def fetchall(self) -> List[Dict]:
        return [dict(r) for r in (self._cur.fetchall() or [])]

    def fetchone(self) -> Optional[Dict]:
        r = self._cur.fetchone()
        return dict(r) if r else None


class _SQLiteCur:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn, self._cur = conn, None

    def ddl(self, *stmts: str) -> None:
        for s in stmts:
            self._conn.execute(s)

    def execute(self, sql: str, params: tuple = ()) -> "_SQLiteCur":
        self._cur = self._conn.execute(sql, params)
        return self

    def fetchall(self) -> List[Dict]:
        rows = self._cur.fetchall() if self._cur else []
        return [dict(r) for r in rows]

    def fetchone(self) -> Optional[Dict]:
        r = self._cur.fetchone() if self._cur else None
        return dict(r) if r else None


@contextmanager
def _cursor_ctx() -> Iterator[Any]:
    if _is_postgres():
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(_pg_dsn(), cursor_factory=psycopg2.extras.RealDictCursor)
        cur = conn.cursor()
        try:
            yield _PGCur(conn, cur)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    else:
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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── DDL ──────────────────────────────────────────────────────────────────────

_DDL_TERMS = """
CREATE TABLE IF NOT EXISTS biz_glossary_terms (
    term_id        TEXT PRIMARY KEY,
    preferred_name TEXT NOT NULL,
    canonical_key  TEXT NOT NULL DEFAULT '',
    definition     TEXT NOT NULL DEFAULT '',
    domain         TEXT NOT NULL DEFAULT '',
    status         TEXT NOT NULL DEFAULT 'draft',
    steward        TEXT NOT NULL DEFAULT '',
    confidence     REAL NOT NULL DEFAULT 0.0,
    match_method   TEXT NOT NULL DEFAULT '',
    version        INTEGER NOT NULL DEFAULT 1,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    approved_by    TEXT NOT NULL DEFAULT '',
    approved_at    TEXT NOT NULL DEFAULT ''
)
"""

_DDL_ASSETS = """
CREATE TABLE IF NOT EXISTS biz_glossary_term_assets (
    link_id      TEXT PRIMARY KEY,
    term_id      TEXT NOT NULL,
    source_id    TEXT NOT NULL,
    metadata_id  TEXT NOT NULL,
    attr_id      TEXT NOT NULL DEFAULT '',
    confidence   REAL NOT NULL DEFAULT 0.0,
    match_method TEXT NOT NULL DEFAULT '',
    linked_at    TEXT NOT NULL,
    UNIQUE(term_id, metadata_id, attr_id)
)
"""

_DDL_AUDIT = """
CREATE TABLE IF NOT EXISTS biz_glossary_term_audit (
    audit_id    TEXT PRIMARY KEY,
    term_id     TEXT NOT NULL,
    changed_by  TEXT NOT NULL DEFAULT '',
    changed_at  TEXT NOT NULL,
    field       TEXT NOT NULL DEFAULT '',
    old_value   TEXT NOT NULL DEFAULT '',
    new_value   TEXT NOT NULL DEFAULT ''
)
"""

_schema_ensured = False


def _ensure(cur: Any) -> None:
    global _schema_ensured
    cur.ddl(_DDL_TERMS, _DDL_ASSETS, _DDL_AUDIT)
    _schema_ensured = True


# ── Term CRUD ────────────────────────────────────────────────────────────────

_ALLOWED_TERM_FIELDS = {
    "preferred_name", "canonical_key", "definition", "domain", "status",
    "steward", "confidence", "match_method",
}


def get_term(term_id: str) -> Optional[Dict]:
    with _cursor_ctx() as cur:
        _ensure(cur)
        row = cur.execute("SELECT * FROM biz_glossary_terms WHERE term_id=?", (term_id,)).fetchone()
    return dict(row) if row else None


def list_terms(source_id: Optional[str] = None, status: Optional[str] = None,
                domain: Optional[str] = None) -> List[Dict]:
    with _cursor_ctx() as cur:
        _ensure(cur)
        if source_id:
            sql = (
                "SELECT DISTINCT t.* FROM biz_glossary_terms t "
                "JOIN biz_glossary_term_assets a ON a.term_id = t.term_id "
                "WHERE a.source_id = ?"
            )
            params: List[Any] = [source_id]
        else:
            sql = "SELECT * FROM biz_glossary_terms t WHERE 1=1"
            params = []
        if status:
            sql += " AND t.status=?"
            params.append(status)
        if domain:
            sql += " AND t.domain=?"
            params.append(domain)
        sql += " ORDER BY t.preferred_name"
        rows = cur.execute(sql, tuple(params)).fetchall()
    return [dict(r) for r in rows]


def _record_audit(cur: Any, term_id: str, changed_by: str, field: str, old_value: Any, new_value: Any) -> None:
    cur.execute(
        "INSERT INTO biz_glossary_term_audit (audit_id, term_id, changed_by, changed_at, field, old_value, new_value) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), term_id, changed_by, _now(), field, str(old_value), str(new_value)),
    )


def update_term(term_id: str, changed_by: str = "", **fields: Any) -> bool:
    """Update a term's fields, writing one audit row per changed field.
    A manual edit (match_method='manual') is the DMBOK 'declared beats
    inferred' trust boundary — callers (the API layer) are responsible for
    setting match_method='manual', status='approved', confidence=1.0 when the
    edit originates from a human, mirroring kg_bridges.promote_to_declared()."""
    updates = {k: v for k, v in fields.items() if k in _ALLOWED_TERM_FIELDS}
    if not updates:
        return False
    if "status" in updates and updates["status"] not in VALID_STATUSES:
        return False
    with _cursor_ctx() as cur:
        _ensure(cur)
        current = cur.execute("SELECT * FROM biz_glossary_terms WHERE term_id=?", (term_id,)).fetchone()
        if not current:
            return False
        for field, new_value in updates.items():
            old_value = current[field]
            if str(old_value) != str(new_value):
                _record_audit(cur, term_id, changed_by, field, old_value, new_value)
        updates["updated_at"] = _now()
        updates["version"] = int(current["version"]) + 1
        set_clause = ", ".join(f"{k}=?" for k in updates)
        cur.execute(
            f"UPDATE biz_glossary_terms SET {set_clause} WHERE term_id=?",
            tuple(updates.values()) + (term_id,),
        )
    return True


def approve_term(term_id: str, changed_by: str = "") -> bool:
    now = _now()
    with _cursor_ctx() as cur:
        _ensure(cur)
        current = cur.execute("SELECT status FROM biz_glossary_terms WHERE term_id=?", (term_id,)).fetchone()
        if not current:
            return False
        _record_audit(cur, term_id, changed_by, "status", current["status"], "approved")
        cur.execute(
            "UPDATE biz_glossary_terms SET status='approved', approved_by=?, approved_at=?, "
            "updated_at=?, version=version+1 WHERE term_id=?",
            (changed_by, now, now, term_id),
        )
    return True


def reject_term(term_id: str, changed_by: str = "") -> bool:
    with _cursor_ctx() as cur:
        _ensure(cur)
        current = cur.execute("SELECT status FROM biz_glossary_terms WHERE term_id=?", (term_id,)).fetchone()
        if not current:
            return False
        _record_audit(cur, term_id, changed_by, "status", current["status"], "deprecated")
        cur.execute(
            "UPDATE biz_glossary_terms SET status='deprecated', updated_at=?, version=version+1 WHERE term_id=?",
            (_now(), term_id),
        )
    return True


# ── Term <-> asset links ─────────────────────────────────────────────────────

def list_assets_for_term(term_id: str) -> List[Dict]:
    with _cursor_ctx() as cur:
        _ensure(cur)
        rows = cur.execute(
            "SELECT * FROM biz_glossary_term_assets WHERE term_id=? ORDER BY linked_at", (term_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_asset_link(metadata_id: str, attr_id: str = "") -> Optional[Dict]:
    """The term (if any) currently linked to a specific entity/attribute —
    used to check whether a column already has a governed term before
    re-running discovery on it."""
    with _cursor_ctx() as cur:
        _ensure(cur)
        row = cur.execute(
            "SELECT a.*, t.preferred_name, t.definition, t.status AS term_status, "
            "t.confidence AS term_confidence, t.match_method AS term_match_method "
            "FROM biz_glossary_term_assets a JOIN biz_glossary_terms t ON t.term_id = a.term_id "
            "WHERE a.metadata_id=? AND a.attr_id=?",
            (metadata_id, attr_id),
        ).fetchone()
    return dict(row) if row else None


def list_audit(term_id: str) -> List[Dict]:
    with _cursor_ctx() as cur:
        _ensure(cur)
        rows = cur.execute(
            "SELECT * FROM biz_glossary_term_audit WHERE term_id=? ORDER BY changed_at", (term_id,)
        ).fetchall()
    return [dict(r) for r in rows]
