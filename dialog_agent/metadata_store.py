"""
Persistence layer for extracted metadata — entities (tables) and attributes (columns).

Uses pg_store backend: SQLite in dev/test, PostgreSQL in production.

Each logical table → one md_entities row with a stable metadata_id (UUID).
Each column       → one md_attributes row with a stable attr_id (UUID).

On re-index the row is updated but metadata_id / attr_id stay the same so that
golden-record flags and custom descriptions are preserved.

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
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

from . import pg_store

logger = logging.getLogger(__name__)


# ── Backend selection ──────────────────────────────────────────────────────────
# Production  (APP_ENV=production) → PostgreSQL (AWS RDS, credentials from
#                                     Secrets Manager — see pg_secrets.py)
# Dev / test  (default)             → SQLite  data/metadata.db
#
# Override the SQLite path with the METADATA_DB env var.

def _sqlite_path() -> str:
    return os.environ.get("METADATA_DB", "data/metadata.db")


@contextmanager
def _cursor_ctx() -> Iterator[Any]:
    """
    Yield a backend-agnostic cursor for the metadata store.

    Production  → PostgreSQL (reuses pg_store connection logic).
    Dev / test  → dedicated SQLite file (METADATA_DB, default data/metadata.db).
                  Separate from the KG federation tables (data/kg_federation.db).
    """
    if pg_store.is_postgres():
        with pg_store.cursor_ctx() as cur:
            yield cur
    else:
        import sqlite3
        path = _sqlite_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        cur = pg_store._SQLiteCursor(conn)
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


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


def _ddl_entities() -> str:
    return _DDL_ENTITIES_PG if pg_store.is_postgres() else _DDL_ENTITIES_SQLITE


def _ddl_attributes() -> str:
    return _DDL_ATTRIBUTES_PG if pg_store.is_postgres() else _DDL_ATTRIBUTES_SQLITE


def _enc_bool(v: Any) -> Any:
    return bool(v) if pg_store.is_postgres() else int(bool(v))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure(cur: Any) -> None:
    cur.ddl(_ddl_entities(), _ddl_attributes())


# ── Public API ─────────────────────────────────────────────────────────────────

def persist(source_id: str, source_name: str, report: Dict) -> int:
    """
    Upsert all tables from a metadata report into md_entities / md_attributes.
    Preserves existing golden-record flags and custom descriptions.
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

            # ── entity upsert ──────────────────────────────────────────────────
            existing = cur.execute(
                "SELECT metadata_id, description FROM md_entities "
                "WHERE source_id=? AND schema_name=? AND table_name=?",
                (source_id, schema_name, table_name),
            ).fetchone()

            if existing:
                metadata_id = existing["metadata_id"]
                # Update stats; preserve description if manager has customised it
                cur.execute(
                    "UPDATE md_entities SET row_count=?, size_bytes=?, primary_keys=?, "
                    "source_name=?, updated_at=? WHERE metadata_id=?",
                    (
                        table_data.get("row_count"),
                        table_data.get("size_bytes"),
                        json.dumps(table_data.get("primary_keys") or []),
                        source_name,
                        now,
                        metadata_id,
                    ),
                )
                # Only overwrite description if it is still empty (never edited)
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
                        metadata_id,
                        source_id,
                        source_name,
                        schema_name,
                        table_name,
                        table_data.get("description") or "",
                        table_data.get("row_count"),
                        table_data.get("size_bytes"),
                        json.dumps(table_data.get("primary_keys") or []),
                        _enc_bool(False),
                        now,
                        now,
                    ),
                )

            # ── attribute upsert ───────────────────────────────────────────────
            # report_node serialises columns as a list of dicts via dataclasses.asdict()
            columns = table_data.get("columns") or []
            if isinstance(columns, dict):
                # safety: handle dict-of-dicts form
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
                        "pattern_hints=?, top_values=?, updated_at=? "
                        "WHERE attr_id=?",
                        (
                            col.get("data_type") or "",
                            col.get("domain") or "",
                            _enc_bool(col.get("is_primary_key", False)),
                            _enc_bool(col.get("is_foreign_key", False)),
                            col.get("fk_references") or "",
                            _enc_bool(col.get("nullable", True)),
                            col.get("unique_count"),
                            col.get("null_count"),
                            col.get("row_count"),
                            str(col["min_value"]) if col.get("min_value") is not None else None,
                            str(col["max_value"]) if col.get("max_value") is not None else None,
                            col.get("avg_value"),
                            col.get("stddev_value"),
                            json.dumps(ph if isinstance(ph, list) else [ph]),
                            json.dumps(tv if isinstance(tv, list) else []),
                            now,
                            attr_id,
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
                            attr_id,
                            metadata_id,
                            col_name,
                            col.get("data_type") or "",
                            col.get("domain") or "",
                            col.get("description") or "",
                            _enc_bool(col.get("is_primary_key", False)),
                            _enc_bool(col.get("is_foreign_key", False)),
                            col.get("fk_references") or "",
                            _enc_bool(col.get("nullable", True)),
                            col.get("unique_count"),
                            col.get("null_count"),
                            col.get("row_count"),
                            str(col["min_value"]) if col.get("min_value") is not None else None,
                            str(col["max_value"]) if col.get("max_value") is not None else None,
                            col.get("avg_value"),
                            col.get("stddev_value"),
                            json.dumps(ph if isinstance(ph, list) else [ph]),
                            json.dumps(tv if isinstance(tv, list) else []),
                            _enc_bool(False),
                            now,
                            now,
                        ),
                    )

            count += 1

    logger.info("persist: %d entities for source %s", count, source_id)
    return count


def list_entities(source_id: Optional[str] = None) -> List[Dict]:
    """Return all entities, optionally filtered by source_id."""
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


def delete_entities_for_source(source_id: str) -> None:
    """Remove every md_entities row (and its md_attributes children) scoped
    to this source — called when the source itself is deleted."""
    with _cursor_ctx() as cur:
        _ensure(cur)
        ids = [r["metadata_id"] for r in cur.execute(
            "SELECT metadata_id FROM md_entities WHERE source_id=?", (source_id,)
        ).fetchall()]
        for metadata_id in ids:
            cur.execute("DELETE FROM md_attributes WHERE metadata_id=?", (metadata_id,))
        cur.execute("DELETE FROM md_entities WHERE source_id=?", (source_id,))


def get_entity(metadata_id: str) -> Optional[Dict]:
    """Return entity dict with attributes list, or None if not found."""
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


def update_entity(metadata_id: str, **fields: Any) -> bool:
    """
    Update allowed fields on an entity: description, is_golden_record.
    Returns True if updated.
    """
    allowed = {"description", "is_golden_record"}
    updates: Dict[str, Any] = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return False
    if "is_golden_record" in updates:
        updates["is_golden_record"] = _enc_bool(updates["is_golden_record"])
    updates["updated_at"] = _now()

    set_clause = ", ".join(f"{k}=?" for k in updates)
    params = tuple(updates.values()) + (metadata_id,)

    with _cursor_ctx() as cur:
        _ensure(cur)
        cur.execute(f"UPDATE md_entities SET {set_clause} WHERE metadata_id=?", params)
    return True


def update_attribute(attr_id: str, **fields: Any) -> bool:
    """
    Update allowed fields on an attribute: description, is_golden_record.
    Returns True if updated.
    """
    allowed = {"description", "is_golden_record"}
    updates: Dict[str, Any] = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return False
    if "is_golden_record" in updates:
        updates["is_golden_record"] = _enc_bool(updates["is_golden_record"])
    updates["updated_at"] = _now()

    set_clause = ", ".join(f"{k}=?" for k in updates)
    params = tuple(updates.values()) + (attr_id,)

    with _cursor_ctx() as cur:
        _ensure(cur)
        cur.execute(f"UPDATE md_attributes SET {set_clause} WHERE attr_id=?", params)
    return True


# ── Internal coercion helpers ──────────────────────────────────────────────────

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
