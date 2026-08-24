"""
KG Snapshot Store — persists source registry + KG node/edge snapshots.

Source registry (kg_sources: connection config, status, table list, etc.) is
always relational: PostgreSQL (AWS RDS, credentials from Secrets Manager —
see pg_secrets.py) in production (APP_ENV=production), SQLite otherwise
(KG_STORE_DB, default data/kg_store.db).

The KG graph itself (nodes/edges) is stored in AWS Neptune when
NEPTUNE_WRITER_ENDPOINT is set — see neptune_store.py (IAM/SigV4 auth via the
assumed cross-account role in aws_auth.py, openCypher over the HTTPS Data
API). Falls back to the same SQLite/Postgres store (as a nodes_json/edges_json
blob) when Neptune isn't configured, e.g. local dev.

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
    return _is_production()

def _sqlite_path() -> str:
    return os.environ.get("KG_STORE_DB", "data/kg_store.db")

def _use_neptune() -> bool:
    return bool(os.environ.get("NEPTUNE_WRITER_ENDPOINT", "").strip())


@contextmanager
def _cursor_ctx() -> Iterator[Any]:
    if _use_postgres():
        import psycopg2.extras
        import pg_secrets
        conn = pg_secrets.connect(cursor_factory=psycopg2.extras.RealDictCursor)
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
    created_by      TEXT NOT NULL DEFAULT '',
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
    # Migration: kg_sources predates the created_by column on existing DBs.
    # Postgres supports IF NOT EXISTS directly; SQLite (< 3.35) does not, so
    # fall back to a probe-and-swallow that doesn't poison the transaction.
    if isinstance(cur, _PGCur):
        cur.execute("ALTER TABLE kg_sources ADD COLUMN IF NOT EXISTS created_by TEXT NOT NULL DEFAULT ''")
    else:
        try:
            cur.execute("SELECT created_by FROM kg_sources LIMIT 1")
        except Exception:
            cur.execute("ALTER TABLE kg_sources ADD COLUMN created_by TEXT NOT NULL DEFAULT ''")


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

    with _cursor_ctx() as cur:
        _ensure(cur)
        cur.execute(
            "SELECT source_id FROM kg_sources WHERE source_id=?", (sid,)
        )
        exists = cur.fetchone()
        if exists:
            cur.execute(
                "UPDATE kg_sources SET name=?, description=?, domain=?, icon=?, "
                "db_type=?, connection_json=?, persona_access=?, created_by=?, status=?, "
                "table_count=?, table_names=?, report_json=?, ontology_text=?, "
                "indexed_at=?, updated_at=? WHERE source_id=?",
                (
                    source.get("name", ""),
                    source.get("description", ""),
                    source.get("domain", ""),
                    source.get("icon", ""),
                    source.get("db_type", ""),
                    conn_json, persona,
                    source.get("created_by", ""),
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
                " connection_json, persona_access, created_by, status, table_count, "
                " table_names, report_json, ontology_text, indexed_at, "
                " created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    sid,
                    source.get("name", ""),
                    source.get("description", ""),
                    source.get("domain", ""),
                    source.get("icon", ""),
                    source.get("db_type", ""),
                    conn_json, persona,
                    source.get("created_by", ""),
                    source.get("status", "ready"),
                    source.get("table_count", 0),
                    tables, report, ontology,
                    source.get("indexed_at"),
                    source.get("created_at", now),
                    now,
                ),
            )
    save_snapshot(sid, source.get("kg_nodes") or [], source.get("kg_edges") or [])
    logger.debug("kg_store.save: %s (%s nodes, %s edges)", sid[:8], len(source.get("kg_nodes") or []), len(source.get("kg_edges") or []))


def save_snapshot(source_id: str, nodes: List[Dict], edges: List[Dict]) -> None:
    """Upsert just the KG snapshot (nodes/edges) for source_id, leaving any
    kg_sources row untouched. Used by knowledge_graph_agent's execute/embed
    nodes, which only know kg_id — not the full source record that
    orchestrator_api's save() manages.

    Writes to Neptune (real vertices/edges) when NEPTUNE_WRITER_ENDPOINT is
    set; otherwise falls back to a nodes_json/edges_json blob in the SQLite/
    Postgres store."""
    if _use_neptune():
        import neptune_store
        neptune_store.save_snapshot(source_id, nodes or [], edges or [])
        logger.debug("kg_store.save_snapshot (neptune): %s (%d nodes, %d edges)", source_id[:8], len(nodes or []), len(edges or []))
        return

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
    if _use_neptune():
        import neptune_store
        return neptune_store.load_snapshot(source_id)

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
    if _use_neptune():
        import neptune_store
        neptune_store.delete(source_id)

    with _cursor_ctx() as cur:
        _ensure(cur)
        cur.execute("DELETE FROM kg_sources    WHERE source_id=?", (source_id,))
        if not _use_neptune():
            cur.execute("DELETE FROM kg_snapshots  WHERE source_id=?", (source_id,))
    logger.info("kg_store.delete: %s", source_id[:8])


def load_all() -> List[Dict]:
    """
    Restore all persisted sources (with KG snapshots) from the store.
    Returns a list of source dicts compatible with _sources in orchestrator_api.
    """
    with _cursor_ctx() as cur:
        _ensure(cur)
        src_rows = cur.execute("SELECT * FROM kg_sources").fetchall()

    result = []
    for r in src_rows:
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
        nodes, edges = load_snapshot(r["source_id"])

        result.append({
            "id":               r["source_id"],
            "name":             r.get("name", ""),
            "description":      r.get("description", ""),
            "domain":           r.get("domain", ""),
            "icon":             r.get("icon", ""),
            "db_type":          r.get("db_type", ""),
            "connection":       conn,
            "persona_access":   persona,
            "created_by":       r.get("created_by", ""),
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
