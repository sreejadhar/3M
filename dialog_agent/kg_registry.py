"""
KG Registry — catalog of all built Knowledge Graphs.

Persists to PostgreSQL when KG_POSTGRES_DSN is set; falls back to SQLite.
See dialog_agent/pg_store.py for backend configuration.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import List, Optional

from . import pg_store


@dataclass
class KGEntry:
    kg_id:           str
    display_name:    str
    description:     str                   = ""
    domain_keywords: List[str]             = field(default_factory=list)
    entity_types:    List[str]             = field(default_factory=list)
    source_id:       str                   = ""   # orchestrator source id
    source_db_type:  str                   = ""
    source_db_host:  str                   = ""
    source_db_name:  str                   = ""
    source_schema:   str                   = ""
    embedding:       Optional[List[float]] = None
    created_at:      float                 = 0.0
    updated_at:      float                 = 0.0


# DDL is identical for both PG and SQLite (TEXT PK, REAL timestamps).
_DDL = """
CREATE TABLE IF NOT EXISTS kg_registry (
    kg_id             TEXT PRIMARY KEY,
    display_name      TEXT NOT NULL DEFAULT '',
    description       TEXT NOT NULL DEFAULT '',
    keywords_json     TEXT NOT NULL DEFAULT '[]',
    entity_types_json TEXT NOT NULL DEFAULT '[]',
    source_id         TEXT NOT NULL DEFAULT '',
    source_db_type    TEXT NOT NULL DEFAULT '',
    source_db_host    TEXT NOT NULL DEFAULT '',
    source_db_name    TEXT NOT NULL DEFAULT '',
    source_schema     TEXT NOT NULL DEFAULT '',
    embedding_json    TEXT,
    created_at        REAL NOT NULL DEFAULT 0,
    updated_at        REAL NOT NULL DEFAULT 0
)
"""


def _ensure(cur) -> None:
    cur.ddl(_DDL)


# ── Public API ────────────────────────────────────────────────────────────────

def upsert(entry: KGEntry) -> None:
    now = time.time()
    with pg_store.cursor_ctx() as cur:
        _ensure(cur)
        cur.execute("""
        INSERT INTO kg_registry
            (kg_id,display_name,description,keywords_json,entity_types_json,
             source_id,source_db_type,source_db_host,source_db_name,source_schema,
             embedding_json,created_at,updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(kg_id) DO UPDATE SET
            display_name=EXCLUDED.display_name, description=EXCLUDED.description,
            keywords_json=EXCLUDED.keywords_json, entity_types_json=EXCLUDED.entity_types_json,
            source_id=EXCLUDED.source_id, source_db_type=EXCLUDED.source_db_type,
            source_db_host=EXCLUDED.source_db_host, source_db_name=EXCLUDED.source_db_name,
            source_schema=EXCLUDED.source_schema, embedding_json=EXCLUDED.embedding_json,
            updated_at=EXCLUDED.updated_at
        """, (
            entry.kg_id, entry.display_name, entry.description,
            json.dumps(entry.domain_keywords), json.dumps(entry.entity_types),
            entry.source_id, entry.source_db_type, entry.source_db_host,
            entry.source_db_name, entry.source_schema,
            json.dumps(entry.embedding) if entry.embedding else None,
            entry.created_at or now, now,
        ))


def list_all() -> List[KGEntry]:
    with pg_store.cursor_ctx() as cur:
        _ensure(cur)
        rows = cur.execute(
            "SELECT * FROM kg_registry ORDER BY kg_id"
        ).fetchall()
    return [_row(r) for r in rows]


def get(kg_id: str) -> Optional[KGEntry]:
    with pg_store.cursor_ctx() as cur:
        _ensure(cur)
        r = cur.execute(
            "SELECT * FROM kg_registry WHERE kg_id=?", (kg_id,)
        ).fetchone()
    return _row(r) if r else None


def delete(kg_id: str) -> None:
    with pg_store.cursor_ctx() as cur:
        _ensure(cur)
        cur.execute("DELETE FROM kg_registry WHERE kg_id=?", (kg_id,))


# ── Internal ──────────────────────────────────────────────────────────────────

def _row(r: dict) -> KGEntry:
    return KGEntry(
        kg_id=r["kg_id"], display_name=r["display_name"], description=r["description"],
        domain_keywords=json.loads(r["keywords_json"] or "[]"),
        entity_types=json.loads(r["entity_types_json"] or "[]"),
        source_id=r["source_id"], source_db_type=r["source_db_type"],
        source_db_host=r["source_db_host"], source_db_name=r["source_db_name"],
        source_schema=r["source_schema"],
        embedding=json.loads(r["embedding_json"]) if r["embedding_json"] else None,
        created_at=float(r["created_at"]), updated_at=float(r["updated_at"]),
    )
