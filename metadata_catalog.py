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
list_sources()                                 → List[dict]  (all sources with domain + counts)
update_source(source_id, **fields)             → bool  (update domain / description)
"""
from __future__ import annotations

import json
import logging
import os
import re
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
    canonical_label      TEXT NOT NULL DEFAULT '',
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
    canonical_label      TEXT NOT NULL DEFAULT '',
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
    statistical_type     TEXT NOT NULL DEFAULT '',
    semantic_role        TEXT NOT NULL DEFAULT '',
    taxonomy_tree        TEXT NOT NULL DEFAULT '[]',
    pii_flag             TEXT NOT NULL DEFAULT '',
    pii_type             TEXT NOT NULL DEFAULT '',
    is_golden_record     INTEGER NOT NULL DEFAULT 0,
    deleted_from_source  INTEGER NOT NULL DEFAULT 0,
    canonical_label      TEXT NOT NULL DEFAULT '',
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
    statistical_type     TEXT NOT NULL DEFAULT '',
    semantic_role        TEXT NOT NULL DEFAULT '',
    taxonomy_tree        TEXT NOT NULL DEFAULT '[]',
    pii_flag             TEXT NOT NULL DEFAULT '',
    pii_type             TEXT NOT NULL DEFAULT '',
    is_golden_record     BOOLEAN NOT NULL DEFAULT FALSE,
    deleted_from_source  BOOLEAN NOT NULL DEFAULT FALSE,
    canonical_label      TEXT NOT NULL DEFAULT '',
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

_DDL_SOURCES_SQLITE = """
CREATE TABLE IF NOT EXISTS md_sources (
    source_id    TEXT PRIMARY KEY,
    source_name  TEXT NOT NULL DEFAULT '',
    domain       TEXT NOT NULL DEFAULT '',
    description  TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
)
"""

_DDL_SOURCES_PG = """
CREATE TABLE IF NOT EXISTS md_sources (
    source_id    TEXT PRIMARY KEY,
    source_name  TEXT NOT NULL DEFAULT '',
    domain       TEXT NOT NULL DEFAULT '',
    description  TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
)
"""


_DOMAIN_KEYWORDS: List[tuple] = [
    ("Sales & CRM",     ["sales", "revenue", "order", "crm", "customer", "deal", "opportunity", "quote"]),
    ("HR & Workforce",  ["hr", "employee", "people", "workforce", "payroll", "talent", "headcount", "staff"]),
    ("Finance",         ["finance", "accounting", "ledger", "budget", "cost", "invoice", "payment", "gl", "ap", "ar"]),
    ("Supply Chain",    ["supply", "inventory", "warehouse", "logistics", "shipment", "procurement", "vendor"]),
    ("Product",         ["product", "catalog", "item", "sku", "material", "part", "bom", "spec"]),
    ("Marketing",       ["marketing", "campaign", "email", "social", "lead", "attribution", "channel"]),
    ("IT & Operations", ["it", "infra", "server", "network", "incident", "ticket", "asset", "config"]),
    ("Customer",        ["customer", "client", "account", "contact", "subscriber", "user"]),
]

def _infer_domain(source_name: str) -> str:
    """Infer a business domain from a source name using keyword matching. Returns '' if unclear."""
    low = source_name.lower()
    for domain, keywords in _DOMAIN_KEYWORDS:
        if any(k in low for k in keywords):
            return domain
    return ""


def _ensure(cur: Any) -> None:
    global _schema_migrated

    if _is_postgres():
        cur.ddl(
            _DDL_ENTITIES_PG,
            _DDL_ATTRIBUTES_PG,
            _DDL_REDUNDANCIES_PG,
            _DDL_CHANGES_PG,
            _DDL_SOURCES_PG,
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
            cur.execute(
                "ALTER TABLE md_attributes ADD COLUMN IF NOT EXISTS "
                "statistical_type TEXT NOT NULL DEFAULT ''"
            )
            cur.execute(
                "ALTER TABLE md_attributes ADD COLUMN IF NOT EXISTS "
                "semantic_role TEXT NOT NULL DEFAULT ''"
            )
            cur.execute(
                "ALTER TABLE md_attributes ADD COLUMN IF NOT EXISTS "
                "taxonomy_tree TEXT NOT NULL DEFAULT '[]'"
            )
            cur.execute(
                "ALTER TABLE md_attributes ADD COLUMN IF NOT EXISTS "
                "pii_flag TEXT NOT NULL DEFAULT ''"
            )
            cur.execute(
                "ALTER TABLE md_attributes ADD COLUMN IF NOT EXISTS "
                "pii_type TEXT NOT NULL DEFAULT ''"
            )
            cur.execute(
                "ALTER TABLE md_entities ADD COLUMN IF NOT EXISTS "
                "canonical_label TEXT NOT NULL DEFAULT ''"
            )
            cur.execute(
                "ALTER TABLE md_attributes ADD COLUMN IF NOT EXISTS "
                "canonical_label TEXT NOT NULL DEFAULT ''"
            )
            _schema_migrated = True
    else:
        cur.ddl(
            _DDL_ENTITIES_SQLITE,
            _DDL_ATTRIBUTES_SQLITE,
            _DDL_REDUNDANCIES_SQLITE,
            _DDL_CHANGES_SQLITE,
            _DDL_SOURCES_SQLITE,
        )
        if not _schema_migrated:
            cols_e = {r["name"] for r in cur.execute("PRAGMA table_info(md_entities)").fetchall()}
            if "deleted_from_source" not in cols_e:
                cur.execute(
                    "ALTER TABLE md_entities ADD COLUMN deleted_from_source INTEGER NOT NULL DEFAULT 0"
                )
            cols_a = {r["name"] for r in cur.execute("PRAGMA table_info(md_attributes)").fetchall()}
            if "deleted_from_source" not in cols_a:
                cur.execute(
                    "ALTER TABLE md_attributes ADD COLUMN deleted_from_source INTEGER NOT NULL DEFAULT 0"
                )
            if "statistical_type" not in cols_a:
                cur.execute(
                    "ALTER TABLE md_attributes ADD COLUMN statistical_type TEXT NOT NULL DEFAULT ''"
                )
            if "semantic_role" not in cols_a:
                cur.execute(
                    "ALTER TABLE md_attributes ADD COLUMN semantic_role TEXT NOT NULL DEFAULT ''"
                )
            if "taxonomy_tree" not in cols_a:
                cur.execute(
                    "ALTER TABLE md_attributes ADD COLUMN taxonomy_tree TEXT NOT NULL DEFAULT '[]'"
                )
            if "pii_flag" not in cols_a:
                cur.execute(
                    "ALTER TABLE md_attributes ADD COLUMN pii_flag TEXT NOT NULL DEFAULT ''"
                )
            if "pii_type" not in cols_a:
                cur.execute(
                    "ALTER TABLE md_attributes ADD COLUMN pii_type TEXT NOT NULL DEFAULT ''"
                )
            if "canonical_label" not in cols_e:
                cur.execute(
                    "ALTER TABLE md_entities ADD COLUMN canonical_label TEXT NOT NULL DEFAULT ''"
                )
            if "canonical_label" not in cols_a:
                cur.execute(
                    "ALTER TABLE md_attributes ADD COLUMN canonical_label TEXT NOT NULL DEFAULT ''"
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
    Redundancy check with domain gating:
    - Within-source: always compare (detect duplicate tables inside the same source)
    - Cross-source:  only compare when BOTH sources share the same non-empty domain

    This prevents spurious cross-domain matches (e.g., Sales DB vs HR DB).
    """
    # Resolve domain of the source being indexed
    src_row = cur.execute(
        "SELECT domain FROM md_sources WHERE source_id=?", (source_id,)
    ).fetchone()
    source_domain = (src_row["domain"] if src_row else "") or ""

    # Collect candidate source_ids to compare against:
    # 1. Always include the source itself (within-source)
    # 2. Include other sources only when they share the same non-empty domain
    candidate_source_ids: set = {source_id}
    if source_domain:
        same_domain_rows = cur.execute(
            "SELECT source_id FROM md_sources WHERE domain=? AND source_id != ?",
            (source_domain, source_id),
        ).fetchall()
        candidate_source_ids.update(r["source_id"] for r in same_domain_rows)

    # Fetch all active entities from candidate sources only
    placeholders = ",".join("?" * len(candidate_source_ids))
    candidate_entities = cur.execute(
        f"SELECT metadata_id, source_id FROM md_entities "
        f"WHERE source_id IN ({placeholders}) AND deleted_from_source=?",
        (*candidate_source_ids, _enc_bool(False)),
    ).fetchall()

    if len(candidate_entities) < 2:
        return

    candidate_ids = [r["metadata_id"] for r in candidate_entities]
    entity_source_map = {r["metadata_id"]: r["source_id"] for r in candidate_entities}

    # Pre-fetch column name sets for all candidate entities
    col_sets: Dict[str, set] = {}
    for eid in candidate_ids:
        rows = cur.execute(
            "SELECT column_name FROM md_attributes WHERE metadata_id=? AND deleted_from_source=?",
            (eid, _enc_bool(False)),
        ).fetchall()
        col_sets[eid] = {r["column_name"].lower() for r in rows}

    # Source entities (the ones we just indexed)
    source_active = {eid for eid in candidate_ids if entity_source_map[eid] == source_id}

    # Compare each source entity against all candidate entities
    for a_id in source_active:
        cols_a = col_sets.get(a_id, set())
        if not cols_a:
            continue
        for b_id in candidate_ids:
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

    logger.debug(
        "Redundancy check: source=%s domain=%r candidates=%d",
        source_id, source_domain or "(unset — within-source only)", len(candidate_ids),
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

        # ── Register/update source in md_sources ───────────────────────────────
        existing_src = cur.execute(
            "SELECT source_id, domain FROM md_sources WHERE source_id=?", (source_id,)
        ).fetchone()
        if existing_src:
            cur.execute(
                "UPDATE md_sources SET source_name=?, updated_at=? WHERE source_id=?",
                (source_name, now, source_id),
            )
        else:
            cur.execute(
                "INSERT INTO md_sources (source_id, source_name, domain, description, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?)",
                (source_id, source_name, _infer_domain(source_name), "", now, now),
            )

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
    allowed = {"description", "is_golden_record", "canonical_label"}
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
    allowed = {"description", "is_golden_record", "statistical_type", "semantic_role", "taxonomy_tree", "canonical_label"}
    updates: Dict[str, Any] = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return False
    if "is_golden_record" in updates:
        updates["is_golden_record"] = _enc_bool(updates["is_golden_record"])
    if "taxonomy_tree" in updates:
        v = updates["taxonomy_tree"]
        updates["taxonomy_tree"] = json.dumps(v) if not isinstance(v, str) else v
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


def list_sources() -> List[Dict]:
    """Return all registered metadata sources with their domain assignments."""
    with _cursor_ctx() as cur:
        _ensure(cur)
        rows = cur.execute(
            "SELECT s.*, "
            "  (SELECT COUNT(*) FROM md_entities e WHERE e.source_id=s.source_id AND e.deleted_from_source=0) AS active_entity_count, "
            "  (SELECT COUNT(*) FROM md_redundancies r "
            "   JOIN md_entities ea ON r.entity_a_id=ea.metadata_id "
            "   WHERE ea.source_id=s.source_id) AS redundancy_count "
            "FROM md_sources s ORDER BY s.domain, s.source_name"
        ).fetchall()
    return [dict(r) for r in rows]


def update_source(source_id: str, **fields: Any) -> bool:
    """
    Update allowed fields on a source: domain, description.
    Changing the domain will take effect on the next reindex for redundancy checks.
    Returns True if updated.
    """
    allowed = {"domain", "description"}
    updates: Dict[str, Any] = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return False
    updates["updated_at"] = _now()
    set_clause = ", ".join(f"{k}=?" for k in updates)
    with _cursor_ctx() as cur:
        _ensure(cur)
        cur.execute(
            f"UPDATE md_sources SET {set_clause} WHERE source_id=?",
            tuple(updates.values()) + (source_id,),
        )
    return True


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
    for f in ("pattern_hints", "top_values", "taxonomy_tree"):
        try:
            r[f] = json.loads(r.get(f) or "[]")
        except Exception:
            r[f] = []
    r.setdefault("statistical_type", "")
    r.setdefault("semantic_role", "")
    r.setdefault("pii_flag", "")
    r.setdefault("pii_type", "")
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


# ── PII classification ─────────────────────────────────────────────────────────
#
# Two-stage pipeline:
#   Stage 1 — deterministic: column-name regex + sample-value regex → instant,
#             no LLM cost; covers >90% of obvious cases.
#   Stage 2 — LLM: only for columns that Stage 1 left undecided (flag="") AND
#             that have ≥1 non-null sample value worth inspecting.
#
# pii_flag values: "PII" | "Non-PII"
# pii_type values: a short label like "email" | "phone" | "name" | "ssn" | …
#                  or "" when pii_flag is "Non-PII"

# ── Stage 1a: column-name patterns ───────────────────────────────────────────
# Each entry: (compiled_regex, pii_type)
# Applied to the lowercased column name.  First match wins.
_PII_NAME_RULES: List[tuple] = [
    # Full name / person name
    (re.compile(r'\b(full_?name|person_?name|passenger_?name|employee_?name|customer_?name|client_?name|user_?name(?!_?id))\b', re.I), "name"),
    (re.compile(r'\b(first_?name|given_?name|fname|last_?name|surname|family_?name|lname|middle_?name)\b', re.I), "name"),
    # Email
    (re.compile(r'\b(email|e_?mail|email_?address|contact_?email|work_?email|personal_?email)\b', re.I), "email"),
    # Phone / mobile
    (re.compile(r'\b(phone|mobile|cell|telephone|contact_?no|contact_?number|phone_?no|mobile_?no|tel_?no|msisdn)\b', re.I), "phone"),
    # Date of birth / age (DOB is direct PII; age is quasi-PII)
    (re.compile(r'\b(date_?of_?birth|dob|birth_?date|birthdate|birth_?day)\b', re.I), "date_of_birth"),
    (re.compile(r'\b(age(?!_?group|_?band|_?range|_?bucket))\b', re.I), "age"),
    # National ID / SSN / Tax ID
    (re.compile(r'\b(ssn|social_?security|national_?id|national_?id_?no|nid)\b', re.I), "national_id"),
    (re.compile(r'\b(aadhar|aadhaar|aadhar_?no|aadhaar_?no|aadhar_?number)\b', re.I), "aadhar"),
    (re.compile(r'\b(pan_?no|pan_?number|pan_?card|income_?tax_?id)\b', re.I), "tax_id"),
    (re.compile(r'\b(passport|passport_?no|passport_?number)\b', re.I), "passport"),
    (re.compile(r'\b(driver_?licen[sc]e|driving_?licen[sc]e|dl_?no|dl_?number|license_?no)\b', re.I), "drivers_license"),
    (re.compile(r'\b(voter_?id|voter_?no|election_?id)\b', re.I), "voter_id"),
    # Address components
    (re.compile(r'\b(address|street|addr|street_?address|mailing_?address|residence|home_?address)\b', re.I), "address"),
    (re.compile(r'\b(zip_?code|zipcode|postal_?code|pin_?code|pincode|postcode)\b', re.I), "postal_code"),
    # Financial identifiers
    (re.compile(r'\b(credit_?card|card_?number|card_?no|cc_?no|ccnum)\b', re.I), "credit_card"),
    (re.compile(r'\b(bank_?account|account_?no(?!_?type)|account_?number|iban|bban)\b', re.I), "bank_account"),
    (re.compile(r'\b(cvv|cvc|card_?verification)\b', re.I), "credit_card"),
    # Network / device identifiers
    (re.compile(r'\b(ip_?address|ip_?addr|ipv4|ipv6|client_?ip|source_?ip)\b', re.I), "ip_address"),
    (re.compile(r'\b(mac_?address|mac_?addr|device_?id(?!_?type)|imei|imsi)\b', re.I), "device_id"),
    # Biometric / sensitive
    (re.compile(r'\b(fingerprint|face_?image|biometric|retina|iris_?scan)\b', re.I), "biometric"),
    (re.compile(r'\b(gender|sex(?!_?ratio|_?ual|ual))\b', re.I), "gender"),
    (re.compile(r'\b(race|ethnicity|religion|caste|nationality)\b', re.I), "sensitive_demographic"),
    (re.compile(r'\b(salary|compensation|ctc|annual_?income|gross_?salary|net_?salary|take_?home)\b', re.I), "financial_sensitive"),
    # Location (precise)
    (re.compile(r'\b(latitude|longitude|geo_?lat|geo_?lon|gps_?lat|gps_?lon|lat_?lon)\b', re.I), "geolocation"),
    # Medical
    (re.compile(r'\b(diagnosis|medical_?record|patient_?id(?!_?type)|mrn|health_?id)\b', re.I), "medical"),
]

# ── Stage 1b: sample-value regex patterns ─────────────────────────────────────
# Each entry: (compiled_regex, pii_type)
# Applied to each string sample value.  First matching pattern across all
# sample values wins (we take the most-specific match).
_PII_VALUE_PATTERNS: List[tuple] = [
    (re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'), "email"),
    (re.compile(r'\b(\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}\b'), "phone"),
    (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), "national_id"),       # US SSN
    (re.compile(r'\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b'), "aadhar"),   # Aadhaar (12 digits starting 2-9)
    (re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b'), "tax_id"),           # India PAN
    (re.compile(r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b'), "credit_card"),
    (re.compile(r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b'), "bank_account"),  # IBAN
    (re.compile(r'\b(\d{1,3}\.){3}\d{1,3}\b'), "ip_address"),
    (re.compile(r'\b([0-9A-Fa-f]{2}[:\-]){5}[0-9A-Fa-f]{2}\b'), "device_id"),  # MAC
    (re.compile(r'\b\d{4}-\d{2}-\d{2}\b'), "date_of_birth"),     # ISO date — only treated as PII if col name suggests it
    (re.compile(r'\b-?\d{1,3}\.\d{4,}\b'), "geolocation"),       # lat/lon precision
]

_PII_LLM_SYSTEM = """\
You are a data privacy expert. Given a database table with column names, data types, and sample values,
identify which columns contain Personally Identifiable Information (PII) or sensitive personal data.

PII includes: name, email, phone, date of birth, age, national ID (SSN, Aadhaar, PAN, passport),
address, postal code, credit card number, bank account, IP address, device ID (IMEI/MAC),
geolocation, gender, race/ethnicity/religion, medical/health data, salary/compensation.

For each column return:
  pii_flag: "PII" or "Non-PII"
  pii_type: one of [name, email, phone, date_of_birth, age, national_id, aadhar, tax_id,
            passport, drivers_license, voter_id, address, postal_code, credit_card,
            bank_account, ip_address, device_id, biometric, gender, sensitive_demographic,
            financial_sensitive, medical, geolocation, other_pii] — or "" when Non-PII.

Return ONLY a JSON object, no prose, no markdown fences.

OUTPUT FORMAT:
{
  "columns": [
    {"name": "<column_name>", "pii_flag": "PII"|"Non-PII", "pii_type": "<type_or_empty>"}
  ]
}
"""


def _call_pii_llm(table_name: str, col_specs: List[Dict], model: str) -> List[Dict]:
    """Call LLM to classify PII for a batch of columns."""
    import anthropic as _ant

    lines = []
    for c in col_specs:
        sv = c.get("sample_values") or []
        sv_str = ", ".join(repr(v) for v in sv[:10]) if sv else "(no samples)"
        lines.append(f"  {c['name']} ({c['data_type']}): samples=[{sv_str}]")
    user_msg = (
        f"Table: {table_name}\nColumns:\n" + "\n".join(lines)
        + "\n\nClassify each column for PII."
    )
    try:
        client = _ant.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        msg = client.messages.create(
            model=model, max_tokens=1024, temperature=0.0,
            system=_PII_LLM_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = msg.content[0].text if msg.content else ""
    except Exception as exc:
        logger.warning("classify_pii: LLM call failed for table %r — %s", table_name, exc)
        return []

    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned).get("columns", [])
    except json.JSONDecodeError:
        s, e = cleaned.find("{"), cleaned.rfind("}")
        if s != -1 and e != -1:
            try:
                return json.loads(cleaned[s:e + 1]).get("columns", [])
            except json.JSONDecodeError:
                pass
    logger.warning("classify_pii: could not parse LLM JSON for table %r", table_name)
    return []


def classify_pii(source_id: str, model: Optional[str] = None) -> int:
    """
    Two-stage PII classification for every column of every entity in source_id.

    Stage 1 — deterministic (always runs):
      a) Column-name regex rules (_PII_NAME_RULES)
      b) Sample-value regex patterns (_PII_VALUE_PATTERNS) for columns Stage 1a
         left undecided but that have string-like sample values.

    Stage 2 — LLM (only for columns still undecided after Stage 1 that have
      sample values — avoids LLM cost on obviously non-PII numeric columns):
      Batches undecided columns per table and calls Claude once per table.

    Writes pii_flag ("PII"/"Non-PII") and pii_type to md_attributes.
    Returns the number of attributes updated.
    """
    model = model or os.environ.get("DIALOG_LLM_MODEL", "claude-haiku-4-5")
    now = _now()
    updated = 0

    with _cursor_ctx() as cur:
        _ensure(cur)

        entities = cur.execute(
            "SELECT metadata_id, table_name FROM md_entities "
            "WHERE source_id=? AND deleted_from_source=?",
            (source_id, _enc_bool(False)),
        ).fetchall()

        for ent in entities:
            mid        = ent["metadata_id"]
            table_name = ent["table_name"]

            attrs = cur.execute(
                "SELECT attr_id, column_name, data_type, top_values FROM md_attributes "
                "WHERE metadata_id=? AND deleted_from_source=?",
                (mid, _enc_bool(False)),
            ).fetchall()

            if not attrs:
                continue

            results: Dict[str, Dict] = {}  # attr_id → {pii_flag, pii_type}
            llm_batch: List[Dict]    = []  # columns needing LLM review

            for a in attrs:
                col   = (a["column_name"] or "").lower()
                dtype = (a["data_type"] or "").lower()
                try:
                    top_vals = json.loads(a["top_values"] or "[]")
                except Exception:
                    top_vals = []
                str_samples = [str(v) for v in top_vals if v is not None and str(v).strip()]

                pii_type = ""
                decided  = False

                # ── Stage 1a: column name rules ───────────────────────────
                for pattern, pt in _PII_NAME_RULES:
                    if pattern.search(col):
                        pii_type = pt
                        decided  = True
                        break

                # ── Stage 1b: sample value regex ──────────────────────────
                if not decided and str_samples:
                    for pattern, pt in _PII_VALUE_PATTERNS:
                        if any(pattern.search(sv) for sv in str_samples):
                            # Extra guard: date pattern only counts as PII if
                            # column name also hints at personal dates
                            if pt == "date_of_birth" and not re.search(
                                r'\b(birth|dob|born|age)\b', col, re.I
                            ):
                                continue
                            pii_type = pt
                            decided  = True
                            break

                if decided:
                    results[a["attr_id"]] = {"pii_flag": "PII", "pii_type": pii_type}
                else:
                    # Skip LLM for clearly non-PII numeric/boolean columns with no
                    # string samples — waste of tokens.
                    is_numeric = re.search(
                        r'\b(int|integer|bigint|smallint|numeric|decimal|float|double|real|bool)\b',
                        dtype, re.I,
                    )
                    has_strings = bool(str_samples)
                    if has_strings or not is_numeric:
                        llm_batch.append({
                            "attr_id":      a["attr_id"],
                            "name":         a["column_name"],
                            "data_type":    a["data_type"] or "",
                            "sample_values": str_samples[:10],
                        })
                    else:
                        results[a["attr_id"]] = {"pii_flag": "Non-PII", "pii_type": ""}

            # ── Stage 2: LLM for undecided columns ────────────────────────
            if llm_batch:
                col_specs = [{"name": c["name"], "data_type": c["data_type"],
                              "sample_values": c["sample_values"]} for c in llm_batch]
                annotations = _call_pii_llm(table_name, col_specs, model)
                ann_by_name = {a["name"]: a for a in annotations if "name" in a}

                for c in llm_batch:
                    ann = ann_by_name.get(c["name"])
                    if ann:
                        results[c["attr_id"]] = {
                            "pii_flag": ann.get("pii_flag", "Non-PII"),
                            "pii_type": ann.get("pii_type", "") if ann.get("pii_flag") == "PII" else "",
                        }
                    else:
                        results[c["attr_id"]] = {"pii_flag": "Non-PII", "pii_type": ""}

            # ── Write results ─────────────────────────────────────────────
            for attr_id, res in results.items():
                cur.execute(
                    "UPDATE md_attributes SET pii_flag=?, pii_type=?, updated_at=? "
                    "WHERE attr_id=?",
                    (res["pii_flag"], res["pii_type"], now, attr_id),
                )
                updated += 1

    logger.info("classify_pii: source=%s classified %d attributes", source_id, updated)
    return updated


# ── Taxonomy enrichment ────────────────────────────────────────────────────────

_ENRICH_SYSTEM = """\
You are a data modelling expert. Given a database table with column names, types, and sample values,
classify each column, infer its taxonomy, and write a concise description of what the column
actually stores.  The description is critical — it is used by an AI SQL planner to choose the
correct filter column for natural-language questions.

Return ONLY a JSON object — no prose, no markdown fences.

statistical_type (pick one):
  identifier  — surrogate or natural key (id, code, _id columns)
  nominal     — FK / dimension key linking to another table
  categorical — low-cardinality text with distinct labels (category, status, country)
  ordinal     — ordered set: ranked labels or structured period strings (FY2024, Q1)
  continuous  — numeric measure (revenue, amount, count, rate, %)
  boolean     — yes/no, 0/1, true/false
  free_text   — high-cardinality unstructured text
  date        — actual date/datetime/timestamp column

IMPORTANT classification rules:
  • A column named "maker", "manufacturer", "vendor", "brand", "supplier" is ALWAYS
    categorical — even if the dtype is varchar/nvarchar.  Never classify it as continuous.
  • A column named "year", "month", "quarter", "period", "year_month", "fiscal_year" is
    ALWAYS ordinal — even if the dtype is integer (e.g. year_month=202412).
  • Only use "continuous" for pure numeric measures like revenue, volume, price, quantity.

semantic_role (pick the best match):
  measure | time_period | time_dimension_key | product_category | product_sub_category |
  product_dimension_key | geography | geography_dimension_key | org_unit | org_dimension_key |
  customer_dimension_key | demographic | identifier | boolean_flag | free_text | other

taxonomy_values: for categorical/ordinal columns list all distinct values shown in sample_values.
  For other types return [].

description: A one-sentence plain-English description of what this column stores.
  Be specific about WHAT the values represent, not just the data type.
  Good examples:
    maker      → "Bottler or manufacturer company name (e.g. CCEP, Suntory, Asahi)"
    channel    → "Sales distribution channel (e.g. CVS, Supermarket, Vending)"
    year_month → "Year-month period key in YYYYMM integer format (e.g. 202401 = January 2024)"
    metric     → "Metric type stored in this row: either 'Value' (monetary) or 'Volume' (units)"
  Bad examples (too vague):
    maker      → "descriptive text column"
    year_month → "numeric measure column"

OUTPUT FORMAT:
{
  "columns": [
    {
      "name": "<column_name>",
      "statistical_type": "<type>",
      "semantic_role": "<role>",
      "taxonomy_values": ["<val1>", "<val2>"],
      "description": "<one sentence describing what this column stores>"
    }
  ]
}
"""

# Domain-specific hints injected into the LLM prompt to resolve ambiguous terms.
# Key is the base industry label (first part of "Aviation/FP&A", "Banking/FS", etc.)
_DOMAIN_LLM_HINTS: Dict[str, str] = {
    "Aviation": """\
Domain context: AVIATION / AIRLINE OPERATIONS
Resolve ambiguous terms as follows:
  • station, airport, gate, terminal → geography (airport location), categorical
  • sector, leg, route, segment, rotation, pairing → flight operational dimension, categorical
  • carrier, airline, operator → org_unit, categorical
  • tail, registration → aircraft identifier, identifier
  • class (cabin_class, booking_class, fare_class) → product_category, categorical
  • crew, pilot, captain, officer, purser, attendant → org_unit, categorical
  • delay, delay_code, delay_reason → categorical (IATA delay codes)
  • yield, rask, cask, ask, rpk, load_factor, otp, block_time → continuous measure
  • duty, duty_time, rest_period → continuous measure (crew hours)
  • pax, booked, no_show → continuous measure (passenger counts)
""",
    "Banking": """\
Domain context: BANKING / FINANCIAL SERVICES
Resolve ambiguous terms as follows:
  • sector, segment, class, industry → product_category (loan/customer segment), categorical
  • branch, region, zone, circle → geography, categorical
  • tenor, tenure, maturity, term → ordinal (loan/deposit duration)
  • npa, gnpa, nnpa, dpd → continuous measure (asset quality metrics)
  • account, loan_id, policy_number → identifier
  • product, scheme, facility → product_category, categorical
  • premium (in bancassurance) → continuous measure
""",
    "LS": """\
Domain context: LIFE SCIENCES / PHARMA
Resolve ambiguous terms as follows:
  • batch, lot, lot_number, batch_number → identifier (manufacturing batch)
  • trial, protocol, study → categorical (clinical study identifier/name)
  • arm, cohort, group → categorical (study arm / patient cohort)
  • phase, stage → ordinal (clinical phase I/II/III/IV or disease stage)
  • dose, dosage, strength → continuous measure
  • indication, therapy_area, disease → product_category, categorical
  • site, centre, investigator → geography / org_unit, categorical
  • ae, sae, adverse_event → categorical (safety event type)
  • visit, visit_number → ordinal (study visit sequence)
""",
    "Insurance": """\
Domain context: INSURANCE
Resolve ambiguous terms as follows:
  • policy, policy_number → identifier
  • claim, claim_number, claim_id → identifier
  • premium, sum_assured, coverage, benefit → continuous measure
  • lob, line_of_business, product_line, class → product_category, categorical
  • term, policy_term, tenure → ordinal (policy duration in years)
  • agent, broker, channel, intermediary → org_unit, categorical
  • loss_ratio, combined_ratio, claims_ratio → continuous measure (%)
  • status (policy/claim status) → categorical
""",
    "Manufacturing": """\
Domain context: MANUFACTURING / INDUSTRIALS
Resolve ambiguous terms as follows:
  • lot, batch, work_order, production_order, job_order → identifier
  • shift, plant, line, cell, workcenter → org_unit, categorical
  • machine, equipment, asset → identifier / categorical (asset dimension)
  • operation, process, step, stage → categorical (routing step)
  • yield, oee, throughput, cycle_time, takt_time → continuous measure
  • defect, rejection, rework, scrap → continuous measure (quality counts)
  • downtime, mtbf, mttr → continuous measure (hours/minutes)
  • customer (internal customer / next-station) → customer_dimension_key, categorical
""",
    "Agriculture": """\
Domain context: AGRICULTURE / AGRI-BUSINESS
Resolve ambiguous terms as follows:
  • crop, variety, hybrid, seed_type → product_category, categorical
  • farm, field, plot, parcel, survey_no → geography / identifier
  • season, kharif, rabi, zaid → categorical (crop season)
  • district, taluka, village, block → geography, categorical
  • yield, production, harvest_qty, offtake → continuous measure
  • farmer, farmer_id, fpo → customer_dimension_key, categorical / identifier
  • commodity, produce (wheat, rice, cotton) → product_category, categorical
  • soil_type, irrigation_type → categorical
  • area, acreage, hectare → continuous measure (cultivated area)
""",
    "CPG": """\
Domain context: CONSUMER PACKAGED GOODS (CPG / FMCG)
Resolve ambiguous terms as follows:
  • category, sub_category, segment → product_category, categorical
  • brand, sub_brand, variant → product_dimension_key, categorical
  • channel, trade_channel → org_unit / customer_dimension_key, categorical
  • sku, pack_size, pack_type → product_dimension_key, categorical
  • rsv, gsv, nrv, nsv → continuous measure (revenue metrics)
  • volume, offtake → continuous measure
  • retailer, customer_group, key_account, distributor → customer_dimension_key, categorical
""",
    "Telecom": """\
Domain context: TELECOM / TELCO
Resolve ambiguous terms as follows:
  • subscriber, msisdn, imsi → identifier
  • circle, region, zone → geography, categorical
  • plan, bundle, pack, vas → product_category, categorical
  • arpu, mou, data_usage, revenue_per_user → continuous measure
  • churn, churn_rate, deactivation → continuous measure (%)
  • prepaid, postpaid → categorical (subscription type)
  • operator, network → categorical
""",
    "Retail": """\
Domain context: RETAIL
Resolve ambiguous terms as follows:
  • store, store_id, outlet → geography / identifier
  • format, store_format → categorical (store type)
  • basket, transaction → identifier / continuous measure
  • category, department → product_category, categorical
  • sell_through, stock_turn → continuous measure (%)
  • footfall, traffic → continuous measure (visitor count)
""",
    "Healthcare": """\
Domain context: HEALTHCARE / PROVIDER / PAYER
Resolve ambiguous terms as follows:
  • member, beneficiary → customer_dimension_key, identifier
  • drg, icd, cpt_code → categorical (clinical code)
  • los, length_of_stay → continuous measure (days)
  • admission, discharge → date / categorical (event type)
  • payer, insurer, network → categorical (payer dimension)
  • plan, benefit_plan → product_category, categorical
""",
}


# Domain-specific column-name rules for deterministic inference.
# Applied BEFORE the generic _NAME_RULES when a domain is known.
# Each entry: (compiled_regex, statistical_type, semantic_role)
_DOMAIN_NAME_RULES: Dict[str, List] = {
    "Aviation": [
        (re.compile(r'\b(station|airport|gate|terminal|apron|hub)\b', re.I),     "categorical", "geography"),
        (re.compile(r'\b(carrier|airline|operator|marketing_carrier)\b', re.I),  "categorical", "org_unit"),
        (re.compile(r'\b(sector|leg|route|segment|rotation|pairing)\b', re.I),   "categorical", "other"),
        (re.compile(r'\b(tail|tail_no|tail_number|registration)\b', re.I),       "identifier",  "identifier"),
        (re.compile(r'\b(cabin_class|booking_class|fare_class)\b', re.I),        "categorical", "product_category"),
        (re.compile(r'\b(crew_type|crew_role|rank|crew_base)\b', re.I),          "categorical", "org_unit"),
        (re.compile(r'\b(delay_code|delay_reason|delay_category)\b', re.I),      "categorical", "other"),
        (re.compile(r'\b(rask|cask|ask|rpk|atk|rtk|load_factor|otp)\b', re.I),  "continuous",  "measure"),
        (re.compile(r'\b(duty_time|block_time|flight_time|rest_period)\b', re.I),"continuous",  "measure"),
        (re.compile(r'\b(pax|booked|no_show|seats_available)\b', re.I),          "continuous",  "measure"),
    ],
    "Banking": [
        (re.compile(r'\b(sector|industry_segment|loan_class)\b', re.I),          "categorical", "product_category"),
        (re.compile(r'\b(branch|circle|zone_name)\b', re.I),                     "categorical", "geography"),
        (re.compile(r'\b(tenor|tenure|maturity_bucket|term_bucket)\b', re.I),    "ordinal",     "time_period"),
        (re.compile(r'\b(gnpa|nnpa|npa_amount|dpd_bucket|overdue)\b', re.I),     "continuous",  "measure"),
        (re.compile(r'\b(product_type|facility_type|scheme)\b', re.I),           "categorical", "product_category"),
    ],
    "LS": [
        (re.compile(r'\b(batch|lot|lot_no|batch_no|lot_number|batch_number)\b', re.I), "identifier", "identifier"),
        (re.compile(r'\b(trial_id|protocol_id|study_id)\b', re.I),               "identifier",  "identifier"),
        (re.compile(r'\b(trial|protocol|study_name)\b', re.I),                   "categorical", "other"),
        (re.compile(r'\b(arm|cohort|treatment_group)\b', re.I),                  "categorical", "other"),
        (re.compile(r'\b(trial_phase|clinical_phase)\b', re.I),                  "ordinal",     "time_period"),
        (re.compile(r'\b(indication|therapy_area|disease_area)\b', re.I),        "categorical", "product_category"),
        (re.compile(r'\b(investigator_site|study_site|site_name)\b', re.I),      "categorical", "geography"),
        (re.compile(r'\b(visit_number|visit_seq|visit_name)\b', re.I),           "ordinal",     "time_period"),
    ],
    "Insurance": [
        (re.compile(r'\b(policy_no|policy_number|pol_id)\b', re.I),              "identifier",  "identifier"),
        (re.compile(r'\b(claim_no|claim_number|claim_id)\b', re.I),              "identifier",  "identifier"),
        (re.compile(r'\b(lob|line_of_business|product_line|ins_class)\b', re.I), "categorical", "product_category"),
        (re.compile(r'\b(policy_term|tenure_years|term_years)\b', re.I),         "ordinal",     "time_period"),
        (re.compile(r'\b(agent_code|broker_code|intermediary)\b', re.I),         "categorical", "org_unit"),
        (re.compile(r'\b(loss_ratio|combined_ratio|claims_ratio)\b', re.I),      "continuous",  "measure"),
    ],
    "Manufacturing": [
        (re.compile(r'\b(work_order|wo_number|production_order|job_order)\b', re.I), "identifier", "identifier"),
        (re.compile(r'\b(lot_number|batch_number|lot_no|batch_no)\b', re.I),     "identifier",  "identifier"),
        (re.compile(r'\b(shift_name|shift_type|shift_code)\b', re.I),            "categorical", "org_unit"),
        (re.compile(r'\b(plant_code|plant_name|plant_id)\b', re.I),              "categorical", "org_unit"),
        (re.compile(r'\b(machine_id|equipment_id|asset_id)\b', re.I),            "identifier",  "identifier"),
        (re.compile(r'\b(operation|process_step|routing_step)\b', re.I),         "categorical", "other"),
        (re.compile(r'\b(oee|throughput|takt_time|cycle_time)\b', re.I),         "continuous",  "measure"),
        (re.compile(r'\b(defect_qty|rejection_qty|scrap_qty|rework_qty)\b', re.I),"continuous", "measure"),
    ],
    "Agriculture": [
        (re.compile(r'\b(crop_name|crop_type|variety|hybrid)\b', re.I),          "categorical", "product_category"),
        (re.compile(r'\b(farm_id|field_id|plot_id|parcel_id)\b', re.I),          "identifier",  "identifier"),
        (re.compile(r'\b(farm_name|field_name|survey_no)\b', re.I),              "categorical", "geography"),
        (re.compile(r'\b(crop_season|season_name|kharif|rabi|zaid)\b', re.I),    "categorical", "time_period"),
        (re.compile(r'\b(district|taluka|village|block_name)\b', re.I),          "categorical", "geography"),
        (re.compile(r'\b(yield_per|production_qty|harvest_qty)\b', re.I),        "continuous",  "measure"),
        (re.compile(r'\b(farmer_id|grower_id)\b', re.I),                         "identifier",  "identifier"),
        (re.compile(r'\b(farmer_name|grower_name|fpo_name)\b', re.I),            "categorical", "customer_dimension_key"),
        (re.compile(r'\b(soil_type|irrigation_type|land_type)\b', re.I),         "categorical", "other"),
        (re.compile(r'\b(acreage|cultivated_area|sown_area)\b', re.I),           "continuous",  "measure"),
    ],
    "CPG": [
        (re.compile(r'\b(sub_category|subcategory)\b', re.I),                    "categorical", "product_sub_category"),
        (re.compile(r'\b(category|segment|vertical)\b', re.I),                   "categorical", "product_category"),
        (re.compile(r'\b(trade_channel|channel_type|outlet_type)\b', re.I),      "categorical", "org_unit"),
        (re.compile(r'\b(key_account|distributor|customer_group)\b', re.I),      "categorical", "customer_dimension_key"),
    ],
    "Telecom": [
        (re.compile(r'\b(msisdn|imsi|imei|subscriber_id)\b', re.I),              "identifier",  "identifier"),
        (re.compile(r'\b(circle|telecom_circle|service_area)\b', re.I),          "categorical", "geography"),
        (re.compile(r'\b(plan_name|bundle_name|pack_name|vas_name)\b', re.I),    "categorical", "product_category"),
    ],
}


def _call_enrich_llm(table_name: str, col_specs: List[Dict], model: str,
                     domain: str = "") -> List[Dict]:
    """Call LLM to classify columns and return taxonomy annotations."""
    import anthropic
    import re

    lines = []
    for c in col_specs:
        sv = c.get("sample_values") or []
        sv_str = ", ".join(repr(v) for v in sv[:20]) if sv else "(no samples)"
        lines.append(f"  {c['name']} ({c['data_type']}): samples=[{sv_str}]")

    # Resolve base industry label ("Aviation/FP&A" → "Aviation", "Banking/FS" → "Banking")
    base_domain = (domain or "").split("/")[0].strip()
    domain_hint = _DOMAIN_LLM_HINTS.get(base_domain, "")

    domain_prefix = f"{domain_hint}\n" if domain_hint else ""
    user_msg = (
        f"{domain_prefix}"
        f"Table: {table_name}\nColumns:\n"
        + "\n".join(lines)
        + "\n\nClassify each column."
    )

    try:
        from llm_client import get_client
        client = get_client()
        msg = client.messages.create(
            model=model, max_tokens=2048, temperature=0.0,
            system=_ENRICH_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = msg.content[0].text if msg.content else ""
    except Exception as exc:
        logger.warning("enrich_taxonomy: LLM call failed for table %r — %s", table_name, exc)
        return []

    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    try:
        obj = json.loads(cleaned)
        return obj.get("columns", [])
    except json.JSONDecodeError:
        s, e = cleaned.find("{"), cleaned.rfind("}")
        if s != -1 and e != -1:
            try:
                return json.loads(cleaned[s:e + 1]).get("columns", [])
            except json.JSONDecodeError:
                pass
    logger.warning("enrich_taxonomy: could not parse LLM JSON for table %r", table_name)
    return []


def infer_taxonomy(source_id: str, domain: str = "") -> int:
    """
    Deterministic (no LLM) taxonomy inference using data_type, column name patterns,
    and sample values already stored in top_values.

    Runs immediately after persist() so taxonomy is populated even if the KG pipeline
    or LLM enrichment hasn't run yet.  Does NOT overwrite columns that already have
    a statistical_type set (LLM/KG annotations take precedence).

    domain: industry label (e.g. "Aviation", "Banking/FS") — domain-specific rules
            are applied before the generic rules to resolve ambiguous column names.

    Returns the number of attributes updated.
    """
    # ── Column-name keyword patterns ──────────────────────────────────────────
    # Each entry: (regex_pattern, statistical_type, semantic_role)
    # Checked in order; first match wins.
    _NAME_RULES = [
        # Boolean flags
        (re.compile(r'\b(is|has|flag|active|enabled|status_flag)\b', re.I),
         "boolean", "boolean_flag"),
        # Identifiers / surrogate keys
        (re.compile(r'(^|_)(id|key|code|pk|sk|seq|num|nr|no)($|_)', re.I),
         "identifier", "identifier"),
        # Time — specific periods
        (re.compile(r'\b(fiscal_year|fy|calendar_year|cy|year_period|year_qtr)\b', re.I),
         "ordinal", "time_period"),
        (re.compile(r'\b(year|month|quarter|week|period|qtr|mth|yr)\b', re.I),
         "ordinal", "time_period"),
        # Time — dates
        (re.compile(r'\b(date|datetime|timestamp|time)\b', re.I),
         "date", "time_dimension_key"),
        # Geography
        (re.compile(r'\b(country|nation|region|market|geography|geo|territory|area|zone|city|state|province)\b', re.I),
         "categorical", "geography"),
        # Product hierarchy
        (re.compile(r'\b(sub_category|subcategory|sub_segment|subsegment)\b', re.I),
         "categorical", "product_sub_category"),
        (re.compile(r'\b(category|segment|vertical|product_type|item_type)\b', re.I),
         "categorical", "product_category"),
        # Brand / manufacturer
        (re.compile(r'\b(brand|manufacturer|vendor|supplier|maker|label)\b', re.I),
         "categorical", "product_dimension_key"),
        # Customer / account
        (re.compile(r'\b(customer|account|client|retailer|buyer|shopper)\b', re.I),
         "categorical", "customer_dimension_key"),
        # Org / channel
        (re.compile(r'\b(channel|distribution|outlet|store|shop|depot)\b', re.I),
         "categorical", "org_unit"),
        (re.compile(r'\b(division|business_unit|bu|department|dept|team|org)\b', re.I),
         "categorical", "org_unit"),
    ]

    _NUMERIC_TYPES = re.compile(
        r'\b(int|integer|bigint|smallint|tinyint|number|numeric|decimal|float|double|real|money|currency)\b',
        re.I,
    )
    _DATE_TYPES = re.compile(
        r'\b(date|datetime|timestamp|time)\b', re.I
    )
    _TEXT_TYPES = re.compile(
        r'\b(char|varchar|text|string|nvarchar|clob|blob|object|variant)\b', re.I
    )

    _CATEGORICAL_MAX_UNIQUE = 50

    # Pick domain-specific rules (base label only: "Aviation/FP&A" → "Aviation")
    base_domain = (domain or "").split("/")[0].strip()
    _domain_rules = _DOMAIN_NAME_RULES.get(base_domain, [])

    now = _now()
    updated = 0

    with _cursor_ctx() as cur:
        _ensure(cur)

        entities = cur.execute(
            "SELECT metadata_id, table_name FROM md_entities "
            "WHERE source_id=? AND deleted_from_source=?",
            (source_id, _enc_bool(False)),
        ).fetchall()

        for ent in entities:
            mid = ent["metadata_id"]

            attrs = cur.execute(
                "SELECT attr_id, column_name, data_type, unique_count, "
                "top_values, statistical_type FROM md_attributes "
                "WHERE metadata_id=? AND deleted_from_source=?",
                (mid, _enc_bool(False)),
            ).fetchall()

            for a in attrs:
                # Skip if already annotated (LLM / KG annotations take precedence)
                if (a["statistical_type"] or "").strip():
                    continue

                col    = (a["column_name"] or "").lower()
                dtype  = (a["data_type"] or "").lower()
                unique = a["unique_count"]
                try:
                    top_vals = json.loads(a["top_values"] or "[]")
                except Exception:
                    top_vals = []

                stat_type  = ""
                sem_role   = ""

                # 1. Column-name rules for boolean / identifier always win
                #    (numeric dtype like int/tinyint doesn't mean it's a measure)
                for pattern, st, sr in _NAME_RULES[:2]:   # boolean, identifier
                    if pattern.search(col):
                        stat_type = st
                        sem_role  = sr
                        break

                if not stat_type and _domain_rules:
                    # 2a. Domain-specific rules — applied before generic rules to
                    #     resolve ambiguous column names (e.g. "station" → geography
                    #     in Aviation vs org_unit in other domains; "batch" →
                    #     identifier in LS/Manufacturing vs generic text elsewhere).
                    for pattern, st, sr in _domain_rules:
                        if pattern.search(col):
                            stat_type = st
                            sem_role  = sr
                            break

                if not stat_type:
                    # 2b. Generic column-name rules (time, geography, product, etc.)
                    #     checked BEFORE dtype rules so that period columns stored as
                    #     integers (e.g. year_month INTEGER = 202412) are correctly
                    #     classified as "ordinal/time_period" rather than
                    #     "continuous/measure", ensuring their top_values are surfaced
                    #     as sample hints in the planning prompt.
                    for pattern, st, sr in _NAME_RULES[2:]:
                        if pattern.search(col):
                            stat_type = st
                            sem_role  = sr
                            break

                if not stat_type:
                    # 3. Data type fallback for columns whose name gives no signal
                    if _DATE_TYPES.search(dtype):
                        stat_type = "date"
                        sem_role  = "time_dimension_key"
                    elif _NUMERIC_TYPES.search(dtype):
                        stat_type = "continuous"
                        sem_role  = "measure"
                    else:
                        # 4. Text column with few distinct values → categorical
                        is_text = _TEXT_TYPES.search(dtype) or not dtype
                        few_vals = (unique is not None and unique <= _CATEGORICAL_MAX_UNIQUE) \
                                   or (unique is None and len(top_vals) <= _CATEGORICAL_MAX_UNIQUE)
                        if is_text and few_vals:
                            stat_type = "categorical"
                            sem_role  = "other"

                if not stat_type:
                    continue

                # Build taxonomy_tree from top_values for categorical/ordinal
                if stat_type in ("categorical", "ordinal") and top_vals:
                    tree_json = json.dumps(top_vals)
                else:
                    tree_json = "[]"

                cur.execute(
                    "UPDATE md_attributes SET statistical_type=?, semantic_role=?, "
                    "taxonomy_tree=?, updated_at=? WHERE attr_id=?",
                    (stat_type, sem_role, tree_json, now, a["attr_id"]),
                )
                updated += 1

    logger.info("infer_taxonomy: source=%s inferred %d attributes", source_id, updated)
    return updated


def enrich_taxonomy(source_id: str, model: Optional[str] = None,
                    domain: str = "") -> int:
    """
    LLM-classify every column of every entity in source_id and write:
      statistical_type, semantic_role, taxonomy_tree (JSON array of distinct values)

    Also detects parent-child categorical pairs (category → sub_category) and stores
    the child values grouped by parent in the parent column's taxonomy_tree as a dict.

    domain: industry label (e.g. "Aviation", "Banking/FS") — passed to the LLM
            as a disambiguation hint so ambiguous column names are resolved correctly.

    Returns the number of attributes updated.
    """
    model = model or os.environ.get("DIALOG_LLM_MODEL", "claude-haiku-4-5")
    updated = 0

    with _cursor_ctx() as cur:
        _ensure(cur)

        entities = cur.execute(
            "SELECT metadata_id, table_name FROM md_entities "
            "WHERE source_id=? AND deleted_from_source=?",
            (source_id, _enc_bool(False)),
        ).fetchall()

        now = _now()

        for ent in entities:
            mid = ent["metadata_id"]
            table_name = ent["table_name"]

            attrs = cur.execute(
                "SELECT attr_id, column_name, data_type, top_values FROM md_attributes "
                "WHERE metadata_id=? AND deleted_from_source=?",
                (mid, _enc_bool(False)),
            ).fetchall()

            if not attrs:
                continue

            col_specs = []
            for a in attrs:
                try:
                    sv = json.loads(a["top_values"] or "[]")
                except Exception:
                    sv = []
                col_specs.append({
                    "name": a["column_name"],
                    "data_type": a["data_type"] or "",
                    "sample_values": sv,
                })

            annotations = _call_enrich_llm(table_name, col_specs, model, domain=domain)
            ann_by_name = {a["name"]: a for a in annotations if "name" in a}

            # Build name → attr_id map
            attr_map = {a["column_name"]: a["attr_id"] for a in attrs}

            # Detect parent-child pairs: sub_X → X naming convention
            col_names = {a["column_name"] for a in attrs}
            parent_child: Dict[str, str] = {}  # child_col → parent_col
            for col_name in col_names:
                if col_name.startswith("sub_"):
                    parent_candidate = col_name[4:]
                    if parent_candidate in col_names:
                        parent_child[col_name] = parent_candidate

            # Build hierarchy for parent columns that have children
            parent_hierarchies: Dict[str, Dict] = {}  # parent_col → {parent_val: [child_vals]}
            # We can't run SQL here (no live DB conn), so hierarchy comes from top_values cross-ref
            # We will populate it from the sample values if available

            for col_name, ann in ann_by_name.items():
                attr_id = attr_map.get(col_name)
                if not attr_id:
                    continue

                stat_type   = ann.get("statistical_type") or ""
                sem_role    = ann.get("semantic_role") or ""
                tax_vals    = ann.get("taxonomy_values") or []
                description = ann.get("description") or ""

                # If this is a parent col with a child col, taxonomy_tree stays as flat list
                # for now; the hierarchy enrichment requires live DB access which is in understand_node
                taxonomy_json = json.dumps(tax_vals if isinstance(tax_vals, list) else [])

                # Write description only if the column doesn't already have one —
                # preserves manual golden-record descriptions.
                if description:
                    cur.execute(
                        "UPDATE md_attributes SET statistical_type=?, semantic_role=?, "
                        "taxonomy_tree=?, description=?, updated_at=? WHERE attr_id=? "
                        "AND (description IS NULL OR description='')",
                        (stat_type, sem_role, taxonomy_json, description, now, attr_id),
                    )
                    # Also update stat/role/taxonomy even if description already exists
                    cur.execute(
                        "UPDATE md_attributes SET statistical_type=?, semantic_role=?, "
                        "taxonomy_tree=?, updated_at=? WHERE attr_id=?",
                        (stat_type, sem_role, taxonomy_json, now, attr_id),
                    )
                else:
                    cur.execute(
                        "UPDATE md_attributes SET statistical_type=?, semantic_role=?, "
                        "taxonomy_tree=?, updated_at=? WHERE attr_id=?",
                        (stat_type, sem_role, taxonomy_json, now, attr_id),
                    )
                updated += 1

            logger.debug("enrich_taxonomy: table=%r annotated %d columns", table_name, len(ann_by_name))

    logger.info("enrich_taxonomy: source=%s updated %d attributes", source_id, updated)
    return updated
