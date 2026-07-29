"""
KG Snapshot Store — persists source registry + KG node/edge snapshots.

In production (APP_ENV=production + KG_POSTGRES_DSN) stores in PostgreSQL.
In dev/test stores in SQLite (KG_STORE_DB, default data/kg_store.db).

On server startup call ``load_all()`` to restore _sources from the last
persisted snapshot so previously-indexed KGs are immediately available.

Public API
----------
save(source: dict)                  → None    (upsert full source snapshot)
delete(source_id: str)              → None
load_all() -> List[dict]            (restore all persisted sources)
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

# ── Backend selection ──────────────────────────────────────────────────────────

def _is_production() -> bool:
    return os.environ.get("APP_ENV", "").strip().lower() == "production"

def _use_postgres() -> bool:
    if not _is_production():
        return False
    dsn = os.environ.get("KG_POSTGRES_DSN", "")
    if dsn:
        return True
    logger.warning("APP_ENV=production but KG_POSTGRES_DSN is not set — KG store uses SQLite.")
    return False

def _sqlite_path() -> str:
    return os.environ.get("KG_STORE_DB", "data/kg_store.db")

def _pg_dsn() -> str:
    return os.environ.get("KG_POSTGRES_DSN", "")


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

_DDL_SOURCES = """
CREATE TABLE IF NOT EXISTS kg_sources (
    source_id       TEXT PRIMARY KEY,
    name            TEXT NOT NULL DEFAULT '',
    description     TEXT NOT NULL DEFAULT '',
    domain          TEXT NOT NULL DEFAULT '',
    icon            TEXT NOT NULL DEFAULT '',
    db_type         TEXT NOT NULL DEFAULT '',
    connection_json TEXT NOT NULL DEFAULT '{}',
    persona_access  TEXT NOT NULL DEFAULT '[]',
    status          TEXT NOT NULL DEFAULT 'ready',
    table_count     INTEGER NOT NULL DEFAULT 0,
    table_names     TEXT NOT NULL DEFAULT '[]',
    report_json     TEXT,
    ontology_text   TEXT,
    indexed_at      REAL,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
)
"""

_DDL_KG_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS kg_snapshots (
    source_id  TEXT PRIMARY KEY,
    nodes_json TEXT NOT NULL DEFAULT '[]',
    edges_json TEXT NOT NULL DEFAULT '[]',
    updated_at REAL NOT NULL
)
"""


def _ensure(cur: Any) -> None:
    cur.ddl(_DDL_SOURCES, _DDL_KG_SNAPSHOTS)


# ── Public API ─────────────────────────────────────────────────────────────────

def save(source: Dict) -> None:
    """Upsert a source (config + KG snapshot) into the store."""
    now = time.time()
    sid = source["id"]
    conn_json = json.dumps(source.get("connection") or {})
    persona   = json.dumps(source.get("persona_access") or [])
    tables    = json.dumps(source.get("table_names") or [])
    report    = json.dumps(source.get("report")) if source.get("report") else None
    ontology  = source.get("ontology_content")
    nodes     = json.dumps(source.get("kg_nodes") or [])
    edges     = json.dumps(source.get("kg_edges") or [])

    with _cursor_ctx() as cur:
        _ensure(cur)
        cur.execute(
            "SELECT source_id FROM kg_sources WHERE source_id=?", (sid,)
        )
        exists = cur.fetchone()
        if exists:
            cur.execute(
                "UPDATE kg_sources SET name=?, description=?, domain=?, icon=?, "
                "db_type=?, connection_json=?, persona_access=?, status=?, "
                "table_count=?, table_names=?, report_json=?, ontology_text=?, "
                "indexed_at=?, updated_at=? WHERE source_id=?",
                (
                    source.get("name", ""),
                    source.get("description", ""),
                    source.get("domain", ""),
                    source.get("icon", ""),
                    source.get("db_type", ""),
                    conn_json, persona,
                    source.get("status", "ready"),
                    source.get("table_count", 0),
                    tables, report, ontology,
                    source.get("indexed_at"),
                    now, sid,
                ),
            )
        else:
            cur.execute(
                "INSERT INTO kg_sources "
                "(source_id, name, description, domain, icon, db_type, "
                " connection_json, persona_access, status, table_count, "
                " table_names, report_json, ontology_text, indexed_at, "
                " created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    sid,
                    source.get("name", ""),
                    source.get("description", ""),
                    source.get("domain", ""),
                    source.get("icon", ""),
                    source.get("db_type", ""),
                    conn_json, persona,
                    source.get("status", "ready"),
                    source.get("table_count", 0),
                    tables, report, ontology,
                    source.get("indexed_at"),
                    source.get("created_at", now),
                    now,
                ),
            )
        # Upsert KG snapshot
        cur.execute("SELECT source_id FROM kg_snapshots WHERE source_id=?", (sid,))
        snap_exists = cur.fetchone()
        if snap_exists:
            cur.execute(
                "UPDATE kg_snapshots SET nodes_json=?, edges_json=?, updated_at=? WHERE source_id=?",
                (nodes, edges, now, sid),
            )
        else:
            cur.execute(
                "INSERT INTO kg_snapshots (source_id, nodes_json, edges_json, updated_at) VALUES (?,?,?,?)",
                (sid, nodes, edges, now),
            )

    logger.debug("kg_store.save: %s (%s nodes, %s edges)", sid[:8], len(source.get("kg_nodes") or []), len(source.get("kg_edges") or []))


def save_snapshot(source_id: str, nodes: List[Dict], edges: List[Dict]) -> None:
    """Upsert just the KG snapshot (nodes/edges) for source_id, leaving any
    kg_sources row untouched. Used by knowledge_graph_agent's execute/embed
    nodes, which only know kg_id — not the full source record that
    orchestrator_api's save() manages."""
    now = time.time()
    nodes_json = json.dumps(nodes or [])
    edges_json = json.dumps(edges or [])
    with _cursor_ctx() as cur:
        _ensure(cur)
        cur.execute("SELECT source_id FROM kg_snapshots WHERE source_id=?", (source_id,))
        if cur.fetchone():
            cur.execute(
                "UPDATE kg_snapshots SET nodes_json=?, edges_json=?, updated_at=? WHERE source_id=?",
                (nodes_json, edges_json, now, source_id),
            )
        else:
            cur.execute(
                "INSERT INTO kg_snapshots (source_id, nodes_json, edges_json, updated_at) VALUES (?,?,?,?)",
                (source_id, nodes_json, edges_json, now),
            )
    logger.debug("kg_store.save_snapshot: %s (%d nodes, %d edges)", source_id[:8], len(nodes or []), len(edges or []))


def load_snapshot(source_id: str) -> "tuple[List[Dict], List[Dict]]":
    """Read back just the KG snapshot (nodes/edges) for source_id.
    Returns ([], []) if no snapshot exists."""
    with _cursor_ctx() as cur:
        _ensure(cur)
        row = cur.execute(
            "SELECT nodes_json, edges_json FROM kg_snapshots WHERE source_id=?", (source_id,)
        ).fetchone()
    if not row:
        return [], []
    try:
        nodes = json.loads(row.get("nodes_json") or "[]")
    except Exception:
        nodes = []
    try:
        edges = json.loads(row.get("edges_json") or "[]")
    except Exception:
        edges = []
    return nodes, edges


def delete(source_id: str) -> None:
    """Remove a source and its KG snapshot from the store."""
    with _cursor_ctx() as cur:
        _ensure(cur)
        cur.execute("DELETE FROM kg_sources    WHERE source_id=?", (source_id,))
        cur.execute("DELETE FROM kg_snapshots  WHERE source_id=?", (source_id,))
    logger.info("kg_store.delete: %s", source_id[:8])


def load_all() -> List[Dict]:
    """
    Restore all persisted sources (with KG snapshots) from the store.
    Returns a list of source dicts compatible with _sources in orchestrator_api.
    """
    with _cursor_ctx() as cur:
        _ensure(cur)
        src_rows  = cur.execute("SELECT * FROM kg_sources").fetchall()
        snap_rows = cur.execute("SELECT * FROM kg_snapshots").fetchall()

    snap_by_id = {r["source_id"]: r for r in snap_rows}
    result = []
    for r in src_rows:
        snap = snap_by_id.get(r["source_id"], {})
        try:
            conn = json.loads(r.get("connection_json") or "{}")
        except Exception:
            conn = {}
        try:
            persona = json.loads(r.get("persona_access") or "[]")
        except Exception:
            persona = ["business_user", "analyst", "admin"]
        try:
            tables = json.loads(r.get("table_names") or "[]")
        except Exception:
            tables = []
        try:
            report = json.loads(r["report_json"]) if r.get("report_json") else None
        except Exception:
            report = None
        try:
            nodes = json.loads(snap.get("nodes_json") or "[]")
        except Exception:
            nodes = []
        try:
            edges = json.loads(snap.get("edges_json") or "[]")
        except Exception:
            edges = []

        result.append({
            "id":               r["source_id"],
            "name":             r.get("name", ""),
            "description":      r.get("description", ""),
            "domain":           r.get("domain", ""),
            "icon":             r.get("icon", ""),
            "db_type":          r.get("db_type", ""),
            "connection":       conn,
            "persona_access":   persona,
            "status":           r.get("status", "ready"),
            "error_message":    None,
            "table_count":      r.get("table_count", 0),
            "table_names":      tables,
            "report":           report,
            "ontology_content": r.get("ontology_text"),
            "kg_nodes":         nodes,
            "kg_edges":         edges,
            "created_at":       r.get("created_at", 0),
            "indexed_at":       r.get("indexed_at"),
            # runtime-only fields
            "extract_job_id":   None,
            "ontology_job_id":  None,
            "kg_job_id":        None,
        })
    logger.info("kg_store.load_all: restored %d sources", len(result))
    return result
