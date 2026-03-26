"""
Standalone persistence layer for the metadata catalog — entities (tables)
and attributes (columns).

Intentionally placed at the project root so it can be imported by
orchestrator_api.py WITHOUT triggering dialog_agent/__init__.py
(which pulls in langgraph and other heavy deps).

Backend
-------
Production  (APP_ENV=production + KG_POSTGRES_DSN set) → PostgreSQL
Dev / test  (default)                                   → SQLite  data/metadata.db
                                                          (override: METADATA_DB)

Public API
----------
persist(source_id, source_name, report)   → int  (entities persisted)
list_entities(source_id=None)             → List[dict]
get_entity(metadata_id)                   → dict | None  (includes attributes[])
update_entity(metadata_id, **fields)      → bool
update_attribute(attr_id, **fields)       → bool
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


# ── Environment helpers ────────────────────────────────────────────────────────

def _is_production() -> bool:
    return os.environ.get("APP_ENV", "").strip().lower() == "production"


def _is_postgres() -> bool:
    if not _is_production():
        return False
    if os.environ.get("KG_POSTGRES_DSN", ""):
        return True
    logger.warning(
        "APP_ENV=production but KG_POSTGRES_DSN is not set — "
        "falling back to SQLite for metadata catalog."
    )
    return False


def _pg_dsn() -> str:
    return os.environ.get("KG_POSTGRES_DSN", "")


def _sqlite_path() -> str:
    return os.environ.get("METADATA_DB", "data/metadata.db")


# ── Cursor abstraction ─────────────────────────────────────────────────────────

class _PGCur:
    """Thin psycopg2 cursor wrapper — translates ? → %s placeholders."""
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

    def insert_returning_id(self, sql: str, params: tuple = ()) -> Optional[int]:
        self._cur.execute(sql.replace("?", "%s"), params)
        r = self._cur.fetchone()
        return dict(r)["id"] if r else None


class _SQLiteCur:
    """Thin sqlite3 connection wrapper — returns rows as dicts."""
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

    def insert_returning_id(self, sql: str, params: tuple = ()) -> Optional[int]:
        self._cur = self._conn.execute(sql, params)
        return self._cur.lastrowid


@contextmanager
def _cursor_ctx() -> Iterator[Any]:
    """Yield a backend-agnostic cursor. Commits on success, rolls back on error."""
    if _is_postgres():
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(_pg_dsn(),
                                cursor_factory=psycopg2.extras.RealDictCursor)
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


def _enc_bool(v: Any) -> Any:
    return bool(v) if _is_postgres() else int(bool(v))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── DDL ────────────────────────────────────────────────────────────────────────

_DDL_ENTITIES_SQLITE = """
CREATE TABLE IF NOT EXISTS md_entities (
    metadata_id      TEXT PRIMARY KEY,
    source_id        TEXT NOT NULL,
    source_name      TEXT NOT NULL DEFAULT '',
    schema_name      TEXT NOT NULL DEFAULT '',
    table_name       TEXT NOT NULL,
    description      TEXT NOT NULL DEFAULT '',
    row_count        INTEGER,
    size_bytes       INTEGER,
    primary_keys     TEXT NOT NULL DEFAULT '[]',
    is_golden_record INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    UNIQUE(source_id, schema_name, table_name)
)
"""

_DDL_ENTITIES_PG = """
CREATE TABLE IF NOT EXISTS md_entities (
    metadata_id      TEXT PRIMARY KEY,
    source_id        TEXT NOT NULL,
    source_name      TEXT NOT NULL DEFAULT '',
    schema_name      TEXT NOT NULL DEFAULT '',
    table_name       TEXT NOT NULL,
    description      TEXT NOT NULL DEFAULT '',
    row_count        INTEGER,
    size_bytes       INTEGER,
    primary_keys     TEXT NOT NULL DEFAULT '[]',
    is_golden_record BOOLEAN NOT NULL DEFAULT FALSE,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    UNIQUE(source_id, schema_name, table_name)
)
"""

_DDL_ATTRIBUTES_SQLITE = """
CREATE TABLE IF NOT EXISTS md_attributes (
    attr_id          TEXT PRIMARY KEY,
    metadata_id      TEXT NOT NULL,
    column_name      TEXT NOT NULL,
    data_type        TEXT NOT NULL DEFAULT '',
    domain           TEXT NOT NULL DEFAULT '',
    description      TEXT NOT NULL DEFAULT '',
    is_primary_key   INTEGER NOT NULL DEFAULT 0,
    is_foreign_key   INTEGER NOT NULL DEFAULT 0,
    fk_references    TEXT NOT NULL DEFAULT '',
    nullable         INTEGER NOT NULL DEFAULT 1,
    unique_count     INTEGER,
    null_count       INTEGER,
    row_count        INTEGER,
    min_value        TEXT,
    max_value        TEXT,
    avg_value        REAL,
    stddev_value     REAL,
    pattern_hints    TEXT NOT NULL DEFAULT '[]',
    top_values       TEXT NOT NULL DEFAULT '[]',
    is_golden_record INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    UNIQUE(metadata_id, column_name)
)
"""

_DDL_ATTRIBUTES_PG = """
CREATE TABLE IF NOT EXISTS md_attributes (
    attr_id          TEXT PRIMARY KEY,
    metadata_id      TEXT NOT NULL,
    column_name      TEXT NOT NULL,
    data_type        TEXT NOT NULL DEFAULT '',
    domain           TEXT NOT NULL DEFAULT '',
    description      TEXT NOT NULL DEFAULT '',
    is_primary_key   BOOLEAN NOT NULL DEFAULT FALSE,
    is_foreign_key   BOOLEAN NOT NULL DEFAULT FALSE,
    fk_references    TEXT NOT NULL DEFAULT '',
    nullable         BOOLEAN NOT NULL DEFAULT TRUE,
    unique_count     INTEGER,
    null_count       INTEGER,
    row_count        INTEGER,
    min_value        TEXT,
    max_value        TEXT,
    avg_value        DOUBLE PRECISION,
    stddev_value     DOUBLE PRECISION,
    pattern_hints    TEXT NOT NULL DEFAULT '[]',
    top_values       TEXT NOT NULL DEFAULT '[]',
    is_golden_record BOOLEAN NOT NULL DEFAULT FALSE,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    UNIQUE(metadata_id, column_name)
)
"""


def _ensure(cur: Any) -> None:
    if _is_postgres():
        cur.ddl(_DDL_ENTITIES_PG, _DDL_ATTRIBUTES_PG)
    else:
        cur.ddl(_DDL_ENTITIES_SQLITE, _DDL_ATTRIBUTES_SQLITE)


# ── Public API ─────────────────────────────────────────────────────────────────

def persist(source_id: str, source_name: str, report: Dict) -> int:
    """
    Upsert all tables from a metadata report into md_entities / md_attributes.
    Preserves existing golden-record flags and custom descriptions on re-index.
    Returns the number of entities persisted.
    """
    tables: Dict[str, Any] = report.get("tables") or {}
    now = _now()
    count = 0

    with _cursor_ctx() as cur:
        _ensure(cur)

        for table_name, table_data in tables.items():
            if not isinstance(table_data, dict):
                continue
            schema_name = table_data.get("schema_name") or ""

            existing = cur.execute(
                "SELECT metadata_id, description FROM md_entities "
                "WHERE source_id=? AND schema_name=? AND table_name=?",
                (source_id, schema_name, table_name),
            ).fetchone()

            if existing:
                metadata_id = existing["metadata_id"]
                cur.execute(
                    "UPDATE md_entities SET row_count=?, size_bytes=?, primary_keys=?, "
                    "source_name=?, updated_at=? WHERE metadata_id=?",
                    (
                        table_data.get("row_count"),
                        table_data.get("size_bytes"),
                        json.dumps(table_data.get("primary_keys") or []),
                        source_name, now, metadata_id,
                    ),
                )
                if not existing.get("description"):
                    cur.execute(
                        "UPDATE md_entities SET description=? WHERE metadata_id=?",
                        (table_data.get("description") or "", metadata_id),
                    )
            else:
                metadata_id = str(uuid.uuid4())
                cur.execute(
                    "INSERT INTO md_entities "
                    "(metadata_id, source_id, source_name, schema_name, table_name, "
                    " description, row_count, size_bytes, primary_keys, "
                    " is_golden_record, created_at, updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        metadata_id, source_id, source_name, schema_name, table_name,
                        table_data.get("description") or "",
                        table_data.get("row_count"), table_data.get("size_bytes"),
                        json.dumps(table_data.get("primary_keys") or []),
                        _enc_bool(False), now, now,
                    ),
                )

            columns = table_data.get("columns") or []
            if isinstance(columns, dict):
                columns = [{"name": k, **v} for k, v in columns.items()]

            for col in columns:
                if not isinstance(col, dict):
                    continue
                col_name = col.get("name") or ""
                if not col_name:
                    continue

                existing_attr = cur.execute(
                    "SELECT attr_id, description FROM md_attributes "
                    "WHERE metadata_id=? AND column_name=?",
                    (metadata_id, col_name),
                ).fetchone()

                ph = col.get("pattern_hints") or []
                tv = col.get("top_values") or []

                if existing_attr:
                    attr_id = existing_attr["attr_id"]
                    cur.execute(
                        "UPDATE md_attributes SET "
                        "data_type=?, domain=?, is_primary_key=?, is_foreign_key=?, "
                        "fk_references=?, nullable=?, unique_count=?, null_count=?, "
                        "row_count=?, min_value=?, max_value=?, avg_value=?, stddev_value=?, "
                        "pattern_hints=?, top_values=?, updated_at=? WHERE attr_id=?",
                        (
                            col.get("data_type") or "",
                            col.get("domain") or "",
                            _enc_bool(col.get("is_primary_key", False)),
                            _enc_bool(col.get("is_foreign_key", False)),
                            col.get("fk_references") or "",
                            _enc_bool(col.get("nullable", True)),
                            col.get("unique_count"), col.get("null_count"),
                            col.get("row_count"),
                            str(col["min_value"]) if col.get("min_value") is not None else None,
                            str(col["max_value"]) if col.get("max_value") is not None else None,
                            col.get("avg_value"), col.get("stddev_value"),
                            json.dumps(ph if isinstance(ph, list) else [ph]),
                            json.dumps(tv if isinstance(tv, list) else []),
                            now, attr_id,
                        ),
                    )
                    if not existing_attr.get("description"):
                        cur.execute(
                            "UPDATE md_attributes SET description=? WHERE attr_id=?",
                            (col.get("description") or "", attr_id),
                        )
                else:
                    attr_id = str(uuid.uuid4())
                    cur.execute(
                        "INSERT INTO md_attributes "
                        "(attr_id, metadata_id, column_name, data_type, domain, description, "
                        " is_primary_key, is_foreign_key, fk_references, nullable, "
                        " unique_count, null_count, row_count, min_value, max_value, "
                        " avg_value, stddev_value, pattern_hints, top_values, "
                        " is_golden_record, created_at, updated_at) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            attr_id, metadata_id, col_name,
                            col.get("data_type") or "", col.get("domain") or "",
                            col.get("description") or "",
                            _enc_bool(col.get("is_primary_key", False)),
                            _enc_bool(col.get("is_foreign_key", False)),
                            col.get("fk_references") or "",
                            _enc_bool(col.get("nullable", True)),
                            col.get("unique_count"), col.get("null_count"),
                            col.get("row_count"),
                            str(col["min_value"]) if col.get("min_value") is not None else None,
                            str(col["max_value"]) if col.get("max_value") is not None else None,
                            col.get("avg_value"), col.get("stddev_value"),
                            json.dumps(ph if isinstance(ph, list) else [ph]),
                            json.dumps(tv if isinstance(tv, list) else []),
                            _enc_bool(False), now, now,
                        ),
                    )
            count += 1

    logger.info("persist: %d entities for source %s", count, source_id)
    return count


def list_entities(source_id: Optional[str] = None) -> List[Dict]:
    with _cursor_ctx() as cur:
        _ensure(cur)
        if source_id:
            rows = cur.execute(
                "SELECT * FROM md_entities WHERE source_id=? "
                "ORDER BY schema_name, table_name",
                (source_id,),
            ).fetchall()
        else:
            rows = cur.execute(
                "SELECT * FROM md_entities ORDER BY source_id, schema_name, table_name"
            ).fetchall()
    return [_coerce_entity(r) for r in rows]


def get_entity(metadata_id: str) -> Optional[Dict]:
    with _cursor_ctx() as cur:
        _ensure(cur)
        entity = cur.execute(
            "SELECT * FROM md_entities WHERE metadata_id=?", (metadata_id,)
        ).fetchone()
        if not entity:
            return None
        attrs = cur.execute(
            "SELECT * FROM md_attributes WHERE metadata_id=? ORDER BY column_name",
            (metadata_id,),
        ).fetchall()
    result = _coerce_entity(entity)
    result["attributes"] = [_coerce_attr(a) for a in attrs]
    return result


def get_attribute(attr_id: str) -> Optional[Dict]:
    with _cursor_ctx() as cur:
        _ensure(cur)
        row = cur.execute(
            "SELECT * FROM md_attributes WHERE attr_id=?", (attr_id,)
        ).fetchone()
    return _coerce_attr(row) if row else None


def update_entity(metadata_id: str, **fields: Any) -> bool:
    allowed = {"description", "is_golden_record"}
    updates: Dict[str, Any] = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return False
    if "is_golden_record" in updates:
        updates["is_golden_record"] = _enc_bool(updates["is_golden_record"])
    updates["updated_at"] = _now()
    set_clause = ", ".join(f"{k}=?" for k in updates)
    with _cursor_ctx() as cur:
        _ensure(cur)
        cur.execute(
            f"UPDATE md_entities SET {set_clause} WHERE metadata_id=?",
            tuple(updates.values()) + (metadata_id,),
        )
    return True


def update_attribute(attr_id: str, **fields: Any) -> bool:
    allowed = {"description", "is_golden_record"}
    updates: Dict[str, Any] = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return False
    if "is_golden_record" in updates:
        updates["is_golden_record"] = _enc_bool(updates["is_golden_record"])
    updates["updated_at"] = _now()
    set_clause = ", ".join(f"{k}=?" for k in updates)
    with _cursor_ctx() as cur:
        _ensure(cur)
        cur.execute(
            f"UPDATE md_attributes SET {set_clause} WHERE attr_id=?",
            tuple(updates.values()) + (attr_id,),
        )
    return True


# ── Coercion helpers ───────────────────────────────────────────────────────────

def _coerce_entity(row: Dict) -> Dict:
    r = dict(row)
    r["is_golden_record"] = bool(r.get("is_golden_record", 0))
    try:
        r["primary_keys"] = json.loads(r.get("primary_keys") or "[]")
    except Exception:
        r["primary_keys"] = []
    return r


def _coerce_attr(row: Dict) -> Dict:
    r = dict(row)
    for f in ("is_primary_key", "is_foreign_key", "nullable", "is_golden_record"):
        r[f] = bool(r.get(f, 0))
    for f in ("pattern_hints", "top_values"):
        try:
            r[f] = json.loads(r.get(f) or "[]")
        except Exception:
            r[f] = []
    return r
