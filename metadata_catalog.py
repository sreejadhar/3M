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
persist(source_id, source_name, report)        → int  (entities persisted)
list_entities(source_id=None)                  → List[dict]  (includes redundancy_count, deleted_from_source)
get_entity(metadata_id)                        → dict | None  (includes attributes[], deleted_from_source)
get_attribute(attr_id)                         → dict | None
update_entity(metadata_id, **fields)           → bool
update_attribute(attr_id, **fields)            → bool
list_redundancies(source_id=None)              → List[dict]  (Jaccard >= 0.9 pairs)
list_changes(entity_id=None, source_id=None, limit=200)  → List[dict]  (CDC log)
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

# Module-level flag: run ALTER TABLE migrations only once per process
_schema_migrated = False


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
    metadata_id          TEXT PRIMARY KEY,
    source_id            TEXT NOT NULL,
    source_name          TEXT NOT NULL DEFAULT '',
    schema_name          TEXT NOT NULL DEFAULT '',
    table_name           TEXT NOT NULL,
    description          TEXT NOT NULL DEFAULT '',
    row_count            INTEGER,
    size_bytes           INTEGER,
    primary_keys         TEXT NOT NULL DEFAULT '[]',
    is_golden_record     INTEGER NOT NULL DEFAULT 0,
    deleted_from_source  INTEGER NOT NULL DEFAULT 0,
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL,
    UNIQUE(source_id, schema_name, table_name)
)
"""

_DDL_ENTITIES_PG = """
CREATE TABLE IF NOT EXISTS md_entities (
    metadata_id          TEXT PRIMARY KEY,
    source_id            TEXT NOT NULL,
    source_name          TEXT NOT NULL DEFAULT '',
    schema_name          TEXT NOT NULL DEFAULT '',
    table_name           TEXT NOT NULL,
    description          TEXT NOT NULL DEFAULT '',
    row_count            INTEGER,
    size_bytes           INTEGER,
    primary_keys         TEXT NOT NULL DEFAULT '[]',
    is_golden_record     BOOLEAN NOT NULL DEFAULT FALSE,
    deleted_from_source  BOOLEAN NOT NULL DEFAULT FALSE,
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL,
    UNIQUE(source_id, schema_name, table_name)
)
"""

_DDL_ATTRIBUTES_SQLITE = """
CREATE TABLE IF NOT EXISTS md_attributes (
    attr_id              TEXT PRIMARY KEY,
    metadata_id          TEXT NOT NULL,
    column_name          TEXT NOT NULL,
    data_type            TEXT NOT NULL DEFAULT '',
    domain               TEXT NOT NULL DEFAULT '',
    description          TEXT NOT NULL DEFAULT '',
    is_primary_key       INTEGER NOT NULL DEFAULT 0,
    is_foreign_key       INTEGER NOT NULL DEFAULT 0,
    fk_references        TEXT NOT NULL DEFAULT '',
    nullable             INTEGER NOT NULL DEFAULT 1,
    unique_count         INTEGER,
    null_count           INTEGER,
    row_count            INTEGER,
    min_value            TEXT,
    max_value            TEXT,
    avg_value            REAL,
    stddev_value         REAL,
    pattern_hints        TEXT NOT NULL DEFAULT '[]',
    top_values           TEXT NOT NULL DEFAULT '[]',
    is_golden_record     INTEGER NOT NULL DEFAULT 0,
    deleted_from_source  INTEGER NOT NULL DEFAULT 0,
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL,
    UNIQUE(metadata_id, column_name)
)
"""

_DDL_ATTRIBUTES_PG = """
CREATE TABLE IF NOT EXISTS md_attributes (
    attr_id              TEXT PRIMARY KEY,
    metadata_id          TEXT NOT NULL,
    column_name          TEXT NOT NULL,
    data_type            TEXT NOT NULL DEFAULT '',
    domain               TEXT NOT NULL DEFAULT '',
    description          TEXT NOT NULL DEFAULT '',
    is_primary_key       BOOLEAN NOT NULL DEFAULT FALSE,
    is_foreign_key       BOOLEAN NOT NULL DEFAULT FALSE,
    fk_references        TEXT NOT NULL DEFAULT '',
    nullable             BOOLEAN NOT NULL DEFAULT TRUE,
    unique_count         INTEGER,
    null_count           INTEGER,
    row_count            INTEGER,
    min_value            TEXT,
    max_value            TEXT,
    avg_value            DOUBLE PRECISION,
    stddev_value         DOUBLE PRECISION,
    pattern_hints        TEXT NOT NULL DEFAULT '[]',
    top_values           TEXT NOT NULL DEFAULT '[]',
    is_golden_record     BOOLEAN NOT NULL DEFAULT FALSE,
    deleted_from_source  BOOLEAN NOT NULL DEFAULT FALSE,
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL,
    UNIQUE(metadata_id, column_name)
)
"""

_DDL_REDUNDANCIES_SQLITE = """
CREATE TABLE IF NOT EXISTS md_redundancies (
    redundancy_id  TEXT PRIMARY KEY,
    entity_a_id    TEXT NOT NULL,
    entity_b_id    TEXT NOT NULL,
    overlap_pct    REAL NOT NULL,
    shared_columns TEXT NOT NULL DEFAULT '[]',
    detected_at    TEXT NOT NULL,
    UNIQUE(entity_a_id, entity_b_id)
)
"""

_DDL_REDUNDANCIES_PG = """
CREATE TABLE IF NOT EXISTS md_redundancies (
    redundancy_id  TEXT PRIMARY KEY,
    entity_a_id    TEXT NOT NULL,
    entity_b_id    TEXT NOT NULL,
    overlap_pct    REAL NOT NULL,
    shared_columns TEXT NOT NULL DEFAULT '[]',
    detected_at    TEXT NOT NULL,
    UNIQUE(entity_a_id, entity_b_id)
)
"""

_DDL_CHANGES_SQLITE = """
CREATE TABLE IF NOT EXISTS md_changes (
    change_id      TEXT PRIMARY KEY,
    entity_type    TEXT NOT NULL,
    entity_id      TEXT NOT NULL,
    change_type    TEXT NOT NULL,
    changed_fields TEXT NOT NULL DEFAULT '{}',
    source_id      TEXT NOT NULL,
    entity_label   TEXT NOT NULL DEFAULT '',
    detected_at    TEXT NOT NULL
)
"""

_DDL_CHANGES_PG = """
CREATE TABLE IF NOT EXISTS md_changes (
    change_id      TEXT PRIMARY KEY,
    entity_type    TEXT NOT NULL,
    entity_id      TEXT NOT NULL,
    change_type    TEXT NOT NULL,
    changed_fields TEXT NOT NULL DEFAULT '{}',
    source_id      TEXT NOT NULL,
    entity_label   TEXT NOT NULL DEFAULT '',
    detected_at    TEXT NOT NULL
)
"""


def _ensure(cur: Any) -> None:
    global _schema_migrated

    if _is_postgres():
        cur.ddl(
            _DDL_ENTITIES_PG,
            _DDL_ATTRIBUTES_PG,
            _DDL_REDUNDANCIES_PG,
            _DDL_CHANGES_PG,
        )
        if not _schema_migrated:
            cur.execute(
                "ALTER TABLE md_entities ADD COLUMN IF NOT EXISTS "
                "deleted_from_source BOOLEAN NOT NULL DEFAULT FALSE"
            )
            cur.execute(
                "ALTER TABLE md_attributes ADD COLUMN IF NOT EXISTS "
                "deleted_from_source BOOLEAN NOT NULL DEFAULT FALSE"
            )
            _schema_migrated = True
    else:
        cur.ddl(
            _DDL_ENTITIES_SQLITE,
            _DDL_ATTRIBUTES_SQLITE,
            _DDL_REDUNDANCIES_SQLITE,
            _DDL_CHANGES_SQLITE,
        )
        if not _schema_migrated:
            # Add deleted_from_source to md_entities if missing
            cols_e = {r["name"] for r in cur.execute("PRAGMA table_info(md_entities)").fetchall()}
            if "deleted_from_source" not in cols_e:
                cur.execute(
                    "ALTER TABLE md_entities ADD COLUMN deleted_from_source INTEGER NOT NULL DEFAULT 0"
                )
            # Add deleted_from_source to md_attributes if missing
            cols_a = {r["name"] for r in cur.execute("PRAGMA table_info(md_attributes)").fetchall()}
            if "deleted_from_source" not in cols_a:
                cur.execute(
                    "ALTER TABLE md_attributes ADD COLUMN deleted_from_source INTEGER NOT NULL DEFAULT 0"
                )
            _schema_migrated = True


# ── Internal helpers ───────────────────────────────────────────────────────────

def _log_change(
    cur: Any,
    entity_type: str,
    entity_id: str,
    change_type: str,
    changed_fields: Dict,
    source_id: str,
    entity_label: str,
    now: str,
) -> None:
    cur.execute(
        "INSERT INTO md_changes "
        "(change_id,entity_type,entity_id,change_type,changed_fields,source_id,entity_label,detected_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (str(uuid.uuid4()), entity_type, entity_id, change_type,
         json.dumps(changed_fields), source_id, entity_label, now),
    )


def _run_redundancy_check(cur: Any, source_id: str, now: str) -> None:
    """
    For each active entity in source_id, compare against all other active entities.
    Upsert/delete md_redundancies based on Jaccard >= 0.9 threshold.
    Pre-fetches ALL column sets to minimise DB round-trips.
    """
    # 1. Fetch all active entity IDs
    all_active = [
        r["metadata_id"]
        for r in cur.execute(
            "SELECT metadata_id FROM md_entities WHERE deleted_from_source=?",
            (_enc_bool(False),),
        ).fetchall()
    ]

    if len(all_active) < 2:
        return

    # 2. Pre-fetch column name sets for all active entities
    col_sets: Dict[str, set] = {}
    for eid in all_active:
        rows = cur.execute(
            "SELECT column_name FROM md_attributes WHERE metadata_id=? AND deleted_from_source=?",
            (eid, _enc_bool(False)),
        ).fetchall()
        col_sets[eid] = {r["column_name"].lower() for r in rows}

    # 3. Source entities (the ones we just indexed)
    source_active = {
        r["metadata_id"]
        for r in cur.execute(
            "SELECT metadata_id FROM md_entities WHERE source_id=? AND deleted_from_source=?",
            (source_id, _enc_bool(False)),
        ).fetchall()
    }

    # 4. Compare each source entity against all_active
    for a_id in source_active:
        cols_a = col_sets.get(a_id, set())
        if not cols_a:
            continue
        for b_id in all_active:
            if a_id == b_id:
                continue
            cols_b = col_sets.get(b_id, set())
            if not cols_b:
                continue
            # Canonical ordering so UNIQUE constraint is respected
            pair_a, pair_b = (a_id, b_id) if a_id < b_id else (b_id, a_id)
            union = cols_a | cols_b
            jaccard = len(cols_a & cols_b) / len(union)

            existing = cur.execute(
                "SELECT redundancy_id FROM md_redundancies WHERE entity_a_id=? AND entity_b_id=?",
                (pair_a, pair_b),
            ).fetchone()

            if jaccard >= 0.9:
                shared = sorted(cols_a & cols_b)
                if existing:
                    cur.execute(
                        "UPDATE md_redundancies SET overlap_pct=?, shared_columns=?, detected_at=? "
                        "WHERE entity_a_id=? AND entity_b_id=?",
                        (jaccard, json.dumps(shared), now, pair_a, pair_b),
                    )
                else:
                    cur.execute(
                        "INSERT INTO md_redundancies "
                        "(redundancy_id,entity_a_id,entity_b_id,overlap_pct,shared_columns,detected_at) "
                        "VALUES (?,?,?,?,?,?)",
                        (str(uuid.uuid4()), pair_a, pair_b, jaccard, json.dumps(shared), now),
                    )
            elif existing:
                cur.execute(
                    "DELETE FROM md_redundancies WHERE entity_a_id=? AND entity_b_id=?",
                    (pair_a, pair_b),
                )


# ── Public API ─────────────────────────────────────────────────────────────────

def persist(source_id: str, source_name: str, report: Dict) -> int:
    """
    Upsert all tables from a metadata report into md_entities / md_attributes.
    Preserves existing golden-record flags and custom descriptions on re-index.
    Detects and flags tables/columns removed from the source as deleted_from_source=True.
    Detects and restores previously deleted entities/attributes if they reappear.
    Logs all structural changes to md_changes.
    Returns the number of entities persisted.
    """
    tables: Dict[str, Any] = report.get("tables") or {}
    now = _now()
    count = 0

    with _cursor_ctx() as cur:
        _ensure(cur)

        # Pre-fetch all existing entities for this source
        existing_rows = cur.execute(
            "SELECT metadata_id, schema_name, table_name, description, deleted_from_source "
            "FROM md_entities WHERE source_id=?",
            (source_id,),
        ).fetchall()
        existing_by_key: Dict[tuple, Dict] = {
            (r["schema_name"], r["table_name"]): r for r in existing_rows
        }

        seen_entity_ids: set = set()

        for table_name, table_data in tables.items():
            if not isinstance(table_data, dict):
                continue
            schema_name = table_data.get("schema_name") or ""
            key = (schema_name, table_name)
            entity_label = f"{schema_name}.{table_name}" if schema_name else table_name

            existing = existing_by_key.get(key)

            if existing:
                metadata_id = existing["metadata_id"]
                was_deleted = bool(existing.get("deleted_from_source", 0))

                if was_deleted:
                    # Restore previously deleted entity
                    cur.execute(
                        "UPDATE md_entities SET deleted_from_source=?, updated_at=? WHERE metadata_id=?",
                        (_enc_bool(False), now, metadata_id),
                    )
                    _log_change(cur, "entity", metadata_id, "restored", {}, source_id, entity_label, now)

                # Update stats (always); clear deleted_from_source in the same SET
                cur.execute(
                    "UPDATE md_entities SET row_count=?, size_bytes=?, primary_keys=?, "
                    "source_name=?, deleted_from_source=?, updated_at=? WHERE metadata_id=?",
                    (
                        table_data.get("row_count"),
                        table_data.get("size_bytes"),
                        json.dumps(table_data.get("primary_keys") or []),
                        source_name, _enc_bool(False), now, metadata_id,
                    ),
                )
                # Only overwrite description if still empty
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
                    " is_golden_record, deleted_from_source, created_at, updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        metadata_id, source_id, source_name, schema_name, table_name,
                        table_data.get("description") or "",
                        table_data.get("row_count"), table_data.get("size_bytes"),
                        json.dumps(table_data.get("primary_keys") or []),
                        _enc_bool(False), _enc_bool(False), now, now,
                    ),
                )
                _log_change(cur, "entity", metadata_id, "added", {}, source_id, entity_label, now)

            seen_entity_ids.add(metadata_id)

            # ── Attributes ─────────────────────────────────────────────────────
            columns = table_data.get("columns") or []
            if isinstance(columns, dict):
                columns = [{"name": k, **v} for k, v in columns.items()]

            # Pre-fetch existing attributes for this entity
            existing_attr_rows = cur.execute(
                "SELECT attr_id, column_name, data_type, description, deleted_from_source "
                "FROM md_attributes WHERE metadata_id=?",
                (metadata_id,),
            ).fetchall()
            existing_attrs_by_name: Dict[str, Dict] = {
                r["column_name"]: r for r in existing_attr_rows
            }

            seen_col_names: set = set()

            for col in columns:
                if not isinstance(col, dict):
                    continue
                col_name = col.get("name") or ""
                if not col_name:
                    continue

                seen_col_names.add(col_name)
                col_label = f"{entity_label}.{col_name}"

                existing_attr = existing_attrs_by_name.get(col_name)

                ph = col.get("pattern_hints") or []
                tv = col.get("top_values") or []

                if existing_attr:
                    attr_id = existing_attr["attr_id"]
                    attr_was_deleted = bool(existing_attr.get("deleted_from_source", 0))
                    new_data_type = col.get("data_type") or ""
                    old_data_type = existing_attr.get("data_type") or ""

                    if attr_was_deleted:
                        # Restore previously deleted attribute
                        cur.execute(
                            "UPDATE md_attributes SET deleted_from_source=?, updated_at=? WHERE attr_id=?",
                            (_enc_bool(False), now, attr_id),
                        )
                        _log_change(cur, "attribute", attr_id, "restored", {}, source_id, col_label, now)
                    elif old_data_type and new_data_type and old_data_type != new_data_type:
                        _log_change(
                            cur, "attribute", attr_id, "type_changed",
                            {"data_type": {"old": old_data_type, "new": new_data_type}},
                            source_id, col_label, now,
                        )

                    # Normal stat update — preserve description and golden_record
                    cur.execute(
                        "UPDATE md_attributes SET "
                        "data_type=?, domain=?, is_primary_key=?, is_foreign_key=?, "
                        "fk_references=?, nullable=?, unique_count=?, null_count=?, "
                        "row_count=?, min_value=?, max_value=?, avg_value=?, stddev_value=?, "
                        "pattern_hints=?, top_values=?, deleted_from_source=?, updated_at=? "
                        "WHERE attr_id=?",
                        (
                            new_data_type,
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
                            _enc_bool(False), now, attr_id,
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
                        " is_golden_record, deleted_from_source, created_at, updated_at) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
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
                            _enc_bool(False), _enc_bool(False), now, now,
                        ),
                    )
                    _log_change(cur, "attribute", attr_id, "added", {}, source_id, col_label, now)

            # Mark attributes not in this run as deleted (if not already)
            for col_name, attr_row in existing_attrs_by_name.items():
                if col_name not in seen_col_names and not bool(attr_row.get("deleted_from_source", 0)):
                    attr_id = attr_row["attr_id"]
                    col_label = f"{entity_label}.{col_name}"
                    cur.execute(
                        "UPDATE md_attributes SET deleted_from_source=?, updated_at=? WHERE attr_id=?",
                        (_enc_bool(True), now, attr_id),
                    )
                    _log_change(cur, "attribute", attr_id, "deleted", {}, source_id, col_label, now)

            count += 1

        # Mark entities not in this run as deleted (if not already)
        for key, entity_row in existing_by_key.items():
            if entity_row["metadata_id"] not in seen_entity_ids and not bool(entity_row.get("deleted_from_source", 0)):
                mid = entity_row["metadata_id"]
                schema_name, table_name = key
                entity_label = f"{schema_name}.{table_name}" if schema_name else table_name
                cur.execute(
                    "UPDATE md_entities SET deleted_from_source=?, updated_at=? WHERE metadata_id=?",
                    (_enc_bool(True), now, mid),
                )
                _log_change(cur, "entity", mid, "deleted", {}, source_id, entity_label, now)

        # Run redundancy check after all changes are persisted
        _run_redundancy_check(cur, source_id, now)

    logger.info("persist: %d entities for source %s", count, source_id)
    return count


def list_entities(source_id: Optional[str] = None) -> List[Dict]:
    with _cursor_ctx() as cur:
        _ensure(cur)
        base = (
            "SELECT e.*, "
            "(SELECT COUNT(*) FROM md_redundancies r "
            " WHERE r.entity_a_id=e.metadata_id OR r.entity_b_id=e.metadata_id) AS redundancy_count "
            "FROM md_entities e"
        )
        if source_id:
            rows = cur.execute(
                base + " WHERE e.source_id=? ORDER BY e.schema_name, e.table_name",
                (source_id,),
            ).fetchall()
        else:
            rows = cur.execute(
                base + " ORDER BY e.source_id, e.schema_name, e.table_name"
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


def list_redundancies(source_id: Optional[str] = None) -> List[Dict]:
    """Return redundancy pairs, enriched with entity source/table info, optionally filtered by source."""
    with _cursor_ctx() as cur:
        _ensure(cur)
        if source_id:
            rows = cur.execute(
                "SELECT r.redundancy_id, r.entity_a_id, r.entity_b_id, r.overlap_pct, r.shared_columns, r.detected_at, "
                "  ea.source_id as a_source_id, ea.source_name as a_source_name, ea.schema_name as a_schema, ea.table_name as a_table, "
                "  eb.source_id as b_source_id, eb.source_name as b_source_name, eb.schema_name as b_schema, eb.table_name as b_table "
                "FROM md_redundancies r "
                "JOIN md_entities ea ON r.entity_a_id=ea.metadata_id "
                "JOIN md_entities eb ON r.entity_b_id=eb.metadata_id "
                "WHERE ea.source_id=? OR eb.source_id=? "
                "ORDER BY r.overlap_pct DESC",
                (source_id, source_id),
            ).fetchall()
        else:
            rows = cur.execute(
                "SELECT r.redundancy_id, r.entity_a_id, r.entity_b_id, r.overlap_pct, r.shared_columns, r.detected_at, "
                "  ea.source_id as a_source_id, ea.source_name as a_source_name, ea.schema_name as a_schema, ea.table_name as a_table, "
                "  eb.source_id as b_source_id, eb.source_name as b_source_name, eb.schema_name as b_schema, eb.table_name as b_table "
                "FROM md_redundancies r "
                "JOIN md_entities ea ON r.entity_a_id=ea.metadata_id "
                "JOIN md_entities eb ON r.entity_b_id=eb.metadata_id "
                "ORDER BY r.overlap_pct DESC",
            ).fetchall()
    return [_coerce_redundancy(r) for r in rows]


def list_changes(
    entity_id: Optional[str] = None,
    source_id: Optional[str] = None,
    limit: int = 200,
) -> List[Dict]:
    """Return CDC change log entries, newest first."""
    with _cursor_ctx() as cur:
        _ensure(cur)
        if entity_id:
            rows = cur.execute(
                "SELECT * FROM md_changes WHERE entity_id=? ORDER BY detected_at DESC LIMIT ?",
                (entity_id, limit),
            ).fetchall()
        elif source_id:
            rows = cur.execute(
                "SELECT * FROM md_changes WHERE source_id=? ORDER BY detected_at DESC LIMIT ?",
                (source_id, limit),
            ).fetchall()
        else:
            rows = cur.execute(
                "SELECT * FROM md_changes ORDER BY detected_at DESC LIMIT ?", (limit,)
            ).fetchall()
    return [_coerce_change(r) for r in rows]


# ── Coercion helpers ───────────────────────────────────────────────────────────

def _coerce_entity(row: Dict) -> Dict:
    r = dict(row)
    r["is_golden_record"] = bool(r.get("is_golden_record", 0))
    r["deleted_from_source"] = bool(r.get("deleted_from_source", 0))
    r["redundancy_count"] = int(r.get("redundancy_count") or 0)
    try:
        r["primary_keys"] = json.loads(r.get("primary_keys") or "[]")
    except Exception:
        r["primary_keys"] = []
    return r


def _coerce_attr(row: Dict) -> Dict:
    r = dict(row)
    for f in ("is_primary_key", "is_foreign_key", "nullable", "is_golden_record", "deleted_from_source"):
        r[f] = bool(r.get(f, 0))
    for f in ("pattern_hints", "top_values"):
        try:
            r[f] = json.loads(r.get(f) or "[]")
        except Exception:
            r[f] = []
    return r


def _coerce_redundancy(row: Dict) -> Dict:
    r = dict(row)
    try:
        r["shared_columns"] = json.loads(r.get("shared_columns") or "[]")
    except Exception:
        r["shared_columns"] = []
    r["overlap_pct"] = float(r.get("overlap_pct") or 0.0)
    return r


def _coerce_change(row: Dict) -> Dict:
    r = dict(row)
    try:
        r["changed_fields"] = json.loads(r.get("changed_fields") or "{}")
    except Exception:
        r["changed_fields"] = {}
    return r
