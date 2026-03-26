"""
KG Registry — SQLite-backed catalog of all built Knowledge Graphs.
Each KG registers here after the build pipeline completes.
Used by the NLQ router to pick the right KG(s) for a question.
"""
from __future__ import annotations
import json, os, sqlite3, time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_DB_PATH = os.environ.get("KG_FEDERATION_DB", "data/kg_federation.db")

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

def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_DB_PATH) or ".", exist_ok=True)
    c = sqlite3.connect(_DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    _init(c)
    return c

def _init(c: sqlite3.Connection) -> None:
    c.executescript("""
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
    );
    """)
    c.commit()

def upsert(entry: KGEntry) -> None:
    now = time.time()
    with _conn() as c:
        c.execute("""
        INSERT INTO kg_registry
            (kg_id,display_name,description,keywords_json,entity_types_json,
             source_id,source_db_type,source_db_host,source_db_name,source_schema,
             embedding_json,created_at,updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(kg_id) DO UPDATE SET
            display_name=excluded.display_name, description=excluded.description,
            keywords_json=excluded.keywords_json, entity_types_json=excluded.entity_types_json,
            source_id=excluded.source_id, source_db_type=excluded.source_db_type,
            source_db_host=excluded.source_db_host, source_db_name=excluded.source_db_name,
            source_schema=excluded.source_schema, embedding_json=excluded.embedding_json,
            updated_at=excluded.updated_at
        """, (
            entry.kg_id, entry.display_name, entry.description,
            json.dumps(entry.domain_keywords), json.dumps(entry.entity_types),
            entry.source_id, entry.source_db_type, entry.source_db_host,
            entry.source_db_name, entry.source_schema,
            json.dumps(entry.embedding) if entry.embedding else None,
            entry.created_at or now, now,
        ))

def list_all() -> List[KGEntry]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM kg_registry ORDER BY kg_id").fetchall()
    return [_row(r) for r in rows]

def get(kg_id: str) -> Optional[KGEntry]:
    with _conn() as c:
        row = c.execute("SELECT * FROM kg_registry WHERE kg_id=?", (kg_id,)).fetchone()
    return _row(row) if row else None

def delete(kg_id: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM kg_registry WHERE kg_id=?", (kg_id,))

def _row(r) -> KGEntry:
    return KGEntry(
        kg_id=r["kg_id"], display_name=r["display_name"], description=r["description"],
        domain_keywords=json.loads(r["keywords_json"] or "[]"),
        entity_types=json.loads(r["entity_types_json"] or "[]"),
        source_id=r["source_id"], source_db_type=r["source_db_type"],
        source_db_host=r["source_db_host"], source_db_name=r["source_db_name"],
        source_schema=r["source_schema"],
        embedding=json.loads(r["embedding_json"]) if r["embedding_json"] else None,
        created_at=r["created_at"], updated_at=r["updated_at"],
    )
