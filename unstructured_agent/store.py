"""
SQLite store for the unstructured data intelligence agent.

Maintains a separate database file (unstructured.db) that is completely
independent of metadata_catalog.db — no joins, no shared connections.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS doc_sources (
    source_id        TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    source_type      TEXT NOT NULL,
    connection_json  TEXT NOT NULL DEFAULT '{}',
    nanite_source_id TEXT,
    domain           TEXT NOT NULL DEFAULT '',
    enabled          INTEGER NOT NULL DEFAULT 1,
    last_indexed_at  TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS unstructured_assets (
    asset_id         TEXT PRIMARY KEY,
    source_id        TEXT NOT NULL REFERENCES doc_sources(source_id),
    file_path        TEXT NOT NULL,
    file_name        TEXT NOT NULL,
    file_type        TEXT NOT NULL,
    checksum         TEXT NOT NULL,
    size_bytes       INTEGER,
    page_count       INTEGER,
    title            TEXT NOT NULL DEFAULT '',
    author           TEXT NOT NULL DEFAULT '',
    created_at_file  TEXT,
    modified_at_file TEXT,
    enriched         INTEGER NOT NULL DEFAULT 0,
    summary          TEXT NOT NULL DEFAULT '',
    domain           TEXT NOT NULL DEFAULT '',
    doc_type         TEXT NOT NULL DEFAULT '',
    language         TEXT NOT NULL DEFAULT 'en',
    topics_json      TEXT NOT NULL DEFAULT '[]',
    entities_json    TEXT NOT NULL DEFAULT '{}',
    time_refs_json   TEXT NOT NULL DEFAULT '[]',
    sensitivity      TEXT NOT NULL DEFAULT 'internal',
    pii_risk         INTEGER NOT NULL DEFAULT 0,
    ocr_used         INTEGER NOT NULL DEFAULT 0,
    ocr_confidence   REAL,
    deleted          INTEGER NOT NULL DEFAULT 0,
    remote_path      TEXT,
    version_num      INTEGER NOT NULL DEFAULT 1,
    indexed_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS doc_asset_versions (
    version_id      TEXT PRIMARY KEY,
    asset_id        TEXT NOT NULL,
    version_num     INTEGER NOT NULL,
    checksum        TEXT NOT NULL,
    size_bytes      INTEGER,
    title           TEXT NOT NULL DEFAULT '',
    summary         TEXT NOT NULL DEFAULT '',
    topics_json     TEXT NOT NULL DEFAULT '[]',
    entities_json   TEXT NOT NULL DEFAULT '{}',
    doc_type        TEXT NOT NULL DEFAULT '',
    change_summary  TEXT NOT NULL DEFAULT '',
    indexed_at      TEXT NOT NULL,
    UNIQUE(asset_id, version_num)
);

CREATE TABLE IF NOT EXISTS doc_relationships (
    rel_id          TEXT PRIMARY KEY,
    from_asset_id   TEXT NOT NULL,
    to_asset_id     TEXT,
    to_nanite_id    TEXT,
    to_nanite_type  TEXT,
    rel_type        TEXT NOT NULL,
    confidence      REAL NOT NULL,
    basis           TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS doc_topic_embeddings (
    asset_id       TEXT PRIMARY KEY REFERENCES unstructured_assets(asset_id),
    embedding_json TEXT NOT NULL,
    model          TEXT NOT NULL,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS index_jobs (
    job_id       TEXT PRIMARY KEY,
    source_id    TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'running',
    total_files  INTEGER NOT NULL DEFAULT 0,
    processed    INTEGER NOT NULL DEFAULT 0,
    enriched     INTEGER NOT NULL DEFAULT 0,
    errors       INTEGER NOT NULL DEFAULT 0,
    error_log    TEXT NOT NULL DEFAULT '[]',
    started_at   TEXT NOT NULL,
    finished_at  TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS doc_search USING fts5(
    asset_id UNINDEXED,
    title,
    summary,
    topics,
    domain UNINDEXED,
    doc_type UNINDEXED
);
"""


class UnstructuredStore:
    def __init__(self, db_path: str) -> None:
        self._path = db_path
        self._local = threading.local()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        if not getattr(self._local, "conn", None):
            conn = sqlite3.connect(self._path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _init_schema(self) -> None:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        try:
            for stmt in _SCHEMA.strip().split(";"):
                s = stmt.strip()
                if s:
                    conn.execute(s)

            # ── Column migrations (idempotent) ────────────────────────────────
            for col_ddl in [
                "ALTER TABLE unstructured_assets ADD COLUMN remote_path TEXT",
                "ALTER TABLE unstructured_assets ADD COLUMN version_num INTEGER NOT NULL DEFAULT 1",
            ]:
                try:
                    conn.execute(col_ddl)
                except Exception:
                    pass

            # ── Deduplicate existing rows ─────────────────────────────────────
            # For cloud files (remote_path IS NOT NULL): keep the row with the
            # latest indexed_at for each (source_id, remote_path) pair.
            conn.execute("""
                DELETE FROM unstructured_assets
                WHERE remote_path IS NOT NULL
                  AND asset_id NOT IN (
                      SELECT asset_id FROM (
                          SELECT asset_id,
                                 ROW_NUMBER() OVER (
                                     PARTITION BY source_id, remote_path
                                     ORDER BY indexed_at DESC
                                 ) AS rn
                          FROM unstructured_assets
                          WHERE remote_path IS NOT NULL
                      ) WHERE rn = 1
                  )
            """)
            # For local files (remote_path IS NULL): keep the latest by
            # (source_id, file_path).
            conn.execute("""
                DELETE FROM unstructured_assets
                WHERE remote_path IS NULL
                  AND asset_id NOT IN (
                      SELECT asset_id FROM (
                          SELECT asset_id,
                                 ROW_NUMBER() OVER (
                                     PARTITION BY source_id, file_path
                                     ORDER BY indexed_at DESC
                                 ) AS rn
                          FROM unstructured_assets
                          WHERE remote_path IS NULL
                      ) WHERE rn = 1
                  )
            """)

            # ── Unique indexes (prevent future duplicates) ────────────────────
            try:
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ux_asset_remote "
                    "ON unstructured_assets(source_id, remote_path) "
                    "WHERE remote_path IS NOT NULL"
                )
            except Exception:
                pass
            try:
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS ux_asset_local "
                    "ON unstructured_assets(source_id, file_path) "
                    "WHERE remote_path IS NULL"
                )
            except Exception:
                pass

            conn.commit()
        finally:
            conn.close()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ── doc_sources ──────────────────────────────────────────────────────────

    def create_source(self, name: str, source_type: str, connection: dict,
                      nanite_source_id: Optional[str], domain: str) -> Dict:
        source_id = str(uuid.uuid4())
        now = self._now()
        self._conn().execute(
            """INSERT INTO doc_sources
               (source_id,name,source_type,connection_json,nanite_source_id,domain,
                created_at,updated_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (source_id, name, source_type, json.dumps(connection),
             nanite_source_id, domain, now, now),
        )
        self._conn().commit()
        return self.get_source(source_id)

    def get_source(self, source_id: str) -> Optional[Dict]:
        row = self._conn().execute(
            "SELECT * FROM doc_sources WHERE source_id=?", (source_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_sources(self) -> List[Dict]:
        rows = self._conn().execute(
            "SELECT * FROM doc_sources WHERE enabled=1 ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def touch_source_indexed(self, source_id: str) -> None:
        self._conn().execute(
            "UPDATE doc_sources SET last_indexed_at=?,updated_at=? WHERE source_id=?",
            (self._now(), self._now(), source_id),
        )
        self._conn().commit()

    # ── unstructured_assets ──────────────────────────────────────────────────

    def upsert_asset(self, source_id: str, file_path: str, file_name: str,
                     file_type: str, checksum: str, structural: Dict,
                     remote_path: Optional[str] = None) -> str:
        """
        Insert or update an asset record.

        Identity key for cloud files (remote_path set): (source_id, remote_path)
        Identity key for local files (no remote_path):  (source_id, file_path)

        Returns asset_id. If the file changed, the old state is snapshotted to
        doc_asset_versions BEFORE updating so callers can compute a diff.
        """
        now = self._now()

        # Resolve canonical identity for this file
        if remote_path:
            existing = self._conn().execute(
                "SELECT * FROM unstructured_assets WHERE source_id=? AND remote_path=?",
                (source_id, remote_path),
            ).fetchone()
        else:
            existing = self._conn().execute(
                "SELECT * FROM unstructured_assets WHERE source_id=? AND file_path=?",
                (source_id, file_path),
            ).fetchone()

        if existing:
            existing = dict(existing)
            if existing["checksum"] == checksum:
                return existing["asset_id"]  # unchanged — skip entirely

            asset_id = existing["asset_id"]
            old_version_num = existing.get("version_num") or 1

            # Snapshot current (old) state before overwriting
            self._conn().execute(
                """INSERT OR IGNORE INTO doc_asset_versions
                   (version_id,asset_id,version_num,checksum,size_bytes,
                    title,summary,topics_json,entities_json,doc_type,indexed_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (str(uuid.uuid4()), asset_id, old_version_num,
                 existing["checksum"], existing.get("size_bytes"),
                 existing.get("title", ""), existing.get("summary", ""),
                 existing.get("topics_json", "[]"), existing.get("entities_json", "{}"),
                 existing.get("doc_type", ""), existing.get("indexed_at", now)),
            )

            new_version_num = old_version_num + 1
            self._conn().execute(
                """UPDATE unstructured_assets SET
                   checksum=?,file_path=?,size_bytes=?,page_count=?,title=?,author=?,
                   created_at_file=?,modified_at_file=?,remote_path=?,
                   enriched=0,summary='',topics_json='[]',entities_json='{}',
                   time_refs_json='[]',version_num=?,updated_at=?,indexed_at=?
                   WHERE asset_id=?""",
                (checksum, file_path,
                 structural.get("size_bytes"), structural.get("page_count"),
                 structural.get("title", ""), structural.get("author", ""),
                 structural.get("created_at_file"), structural.get("modified_at_file"),
                 remote_path, new_version_num, now, now, asset_id),
            )
        else:
            asset_id = str(uuid.uuid4())
            self._conn().execute(
                """INSERT OR IGNORE INTO unstructured_assets
                   (asset_id,source_id,file_path,file_name,file_type,checksum,
                    size_bytes,page_count,title,author,created_at_file,
                    modified_at_file,remote_path,version_num,indexed_at,updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (asset_id, source_id, file_path, file_name, file_type, checksum,
                 structural.get("size_bytes"), structural.get("page_count"),
                 structural.get("title", ""), structural.get("author", ""),
                 structural.get("created_at_file"), structural.get("modified_at_file"),
                 remote_path, 1, now, now),
            )
        self._conn().commit()
        return asset_id

    def save_fingerprint(self, asset_id: str, fp: Dict) -> None:
        now = self._now()
        self._conn().execute(
            """UPDATE unstructured_assets SET
               enriched=1, summary=?, domain=?, doc_type=?, language=?,
               topics_json=?, entities_json=?, time_refs_json=?,
               sensitivity=?, pii_risk=?, updated_at=?
               WHERE asset_id=?""",
            (fp.get("summary", ""), fp.get("domain", ""), fp.get("doc_type", ""),
             fp.get("language", "en"), json.dumps(fp.get("topics", [])),
             json.dumps(fp.get("named_entities", {})),
             json.dumps(fp.get("time_references", [])),
             fp.get("sensitivity", "internal"), int(fp.get("pii_risk", False)),
             now, asset_id),
        )
        self._conn().commit()
        # Update FTS
        self._conn().execute("DELETE FROM doc_search WHERE asset_id=?", (asset_id,))
        self._conn().execute(
            "INSERT INTO doc_search (asset_id,title,summary,topics,domain,doc_type) VALUES (?,?,?,?,?,?)",
            (asset_id, fp.get("title", ""), fp.get("summary", ""),
             " ".join(fp.get("topics", [])), fp.get("domain", ""), fp.get("doc_type", "")),
        )
        self._conn().commit()

    def get_asset(self, asset_id: str) -> Optional[Dict]:
        row = self._conn().execute(
            "SELECT * FROM unstructured_assets WHERE asset_id=?", (asset_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        for field in ("topics_json", "entities_json", "time_refs_json"):
            try:
                d[field.replace("_json", "")] = json.loads(d.pop(field, "[]") or "[]")
            except Exception:
                d[field.replace("_json", "")] = []
        return d

    def list_assets(self, source_id: Optional[str] = None,
                    enriched_only: bool = False, limit: int = 100) -> List[Dict]:
        where = ["deleted=0"]
        params: List = []
        if source_id:
            where.append("source_id=?")
            params.append(source_id)
        if enriched_only:
            where.append("enriched=1")
        rows = self._conn().execute(
            f"SELECT * FROM unstructured_assets WHERE {' AND '.join(where)} "
            f"ORDER BY indexed_at DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        return [self.get_asset(r["asset_id"]) for r in rows]

    def clear_source_assets(self, source_id: str) -> int:
        """Hard-delete all assets (and their version history) for a source."""
        asset_ids = [
            r["asset_id"] for r in
            self._conn().execute(
                "SELECT asset_id FROM unstructured_assets WHERE source_id=?", (source_id,)
            ).fetchall()
        ]
        if asset_ids:
            placeholders = ",".join("?" * len(asset_ids))
            self._conn().execute(
                f"DELETE FROM doc_asset_versions WHERE asset_id IN ({placeholders})",
                asset_ids,
            )
        cur = self._conn().execute(
            "DELETE FROM unstructured_assets WHERE source_id=?", (source_id,)
        )
        self._conn().commit()
        return cur.rowcount

    def soft_delete_asset(self, asset_id: str) -> None:
        self._conn().execute(
            "UPDATE unstructured_assets SET deleted=1,updated_at=? WHERE asset_id=?",
            (self._now(), asset_id),
        )
        self._conn().commit()

    # ── doc_asset_versions ────────────────────────────────────────────────────

    def list_asset_versions(self, asset_id: str) -> List[Dict]:
        """Return all version snapshots for an asset, newest first."""
        rows = self._conn().execute(
            "SELECT * FROM doc_asset_versions WHERE asset_id=? ORDER BY version_num DESC",
            (asset_id,),
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            for field in ("topics_json", "entities_json"):
                try:
                    d[field.replace("_json", "")] = json.loads(d.pop(field, "[]") or "[]")
                except Exception:
                    d[field.replace("_json", "")] = []
            result.append(d)
        return result

    def get_asset_version(self, asset_id: str, version_num: int) -> Optional[Dict]:
        """Return a specific version snapshot (not the live asset row)."""
        row = self._conn().execute(
            "SELECT * FROM doc_asset_versions WHERE asset_id=? AND version_num=?",
            (asset_id, version_num),
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        for field in ("topics_json", "entities_json"):
            try:
                d[field.replace("_json", "")] = json.loads(d.pop(field, "[]") or "[]")
            except Exception:
                d[field.replace("_json", "")] = []
        return d

    def save_version_change_summary(self, asset_id: str, version_num: int,
                                    change_summary: str) -> None:
        """Persist a computed change summary onto an existing version snapshot."""
        self._conn().execute(
            "UPDATE doc_asset_versions SET change_summary=? WHERE asset_id=? AND version_num=?",
            (change_summary, asset_id, version_num),
        )
        self._conn().commit()

    def search_assets(self, query: str, limit: int = 20) -> List[Dict]:
        rows = self._conn().execute(
            """SELECT ua.* FROM doc_search ds
               JOIN unstructured_assets ua ON ua.asset_id=ds.asset_id
               WHERE doc_search MATCH ? AND ua.deleted=0
               ORDER BY rank LIMIT ?""",
            (query, limit),
        ).fetchall()
        return [self.get_asset(r["asset_id"]) for r in rows]

    # ── doc_relationships ────────────────────────────────────────────────────

    def save_relationship(self, from_asset_id: str, rel_type: str,
                          confidence: float, basis: str,
                          to_asset_id: Optional[str] = None,
                          to_nanite_id: Optional[str] = None,
                          to_nanite_type: Optional[str] = None) -> str:
        rel_id = str(uuid.uuid4())
        self._conn().execute(
            """INSERT OR REPLACE INTO doc_relationships
               (rel_id,from_asset_id,to_asset_id,to_nanite_id,to_nanite_type,
                rel_type,confidence,basis,created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (rel_id, from_asset_id, to_asset_id, to_nanite_id, to_nanite_type,
             rel_type, confidence, basis, self._now()),
        )
        self._conn().commit()
        return rel_id

    def get_relationships(self, asset_id: str) -> List[Dict]:
        rows = self._conn().execute(
            "SELECT * FROM doc_relationships WHERE from_asset_id=? ORDER BY confidence DESC",
            (asset_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── index_jobs ────────────────────────────────────────────────────────────

    def create_job(self, source_id: str) -> str:
        job_id = str(uuid.uuid4())
        self._conn().execute(
            "INSERT INTO index_jobs (job_id,source_id,started_at) VALUES (?,?,?)",
            (job_id, source_id, self._now()),
        )
        self._conn().commit()
        return job_id

    def update_job(self, job_id: str, **kwargs) -> None:
        allowed = {"status", "total_files", "processed", "enriched", "errors",
                   "error_log", "finished_at"}
        sets = []
        params = []
        for k, v in kwargs.items():
            if k in allowed:
                sets.append(f"{k}=?")
                params.append(json.dumps(v) if k == "error_log" else v)
        if not sets:
            return
        self._conn().execute(
            f"UPDATE index_jobs SET {','.join(sets)} WHERE job_id=?",
            params + [job_id],
        )
        self._conn().commit()

    def get_job(self, job_id: str) -> Optional[Dict]:
        row = self._conn().execute(
            "SELECT * FROM index_jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["error_log"] = json.loads(d.get("error_log") or "[]")
        except Exception:
            d["error_log"] = []
        return d

    def list_jobs(self, source_id: Optional[str] = None) -> List[Dict]:
        where = ""
        params = []
        if source_id:
            where = "WHERE source_id=?"
            params.append(source_id)
        rows = self._conn().execute(
            f"SELECT * FROM index_jobs {where} ORDER BY started_at DESC LIMIT 50",
            params,
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            try:
                d["error_log"] = json.loads(d.get("error_log") or "[]")
            except Exception:
                d["error_log"] = []
            result.append(d)
        return result
