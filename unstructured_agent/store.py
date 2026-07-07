"""SQLite persistence for Document Intelligence sources, assets and index jobs."""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS doc_sources (
    source_id        TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    connector_type    TEXT NOT NULL,
    config_json       TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'idle',
    error_message     TEXT,
    created_at        TEXT NOT NULL,
    indexed_at        TEXT,
    linked_source_id  TEXT,
    linked_source_ids_json TEXT
);

CREATE TABLE IF NOT EXISTS doc_assets (
    asset_id            TEXT PRIMARY KEY,
    source_id           TEXT NOT NULL,
    remote_id           TEXT NOT NULL,
    file_name           TEXT NOT NULL,
    size_bytes          INTEGER NOT NULL DEFAULT 0,
    mime_type           TEXT,
    checksum            TEXT,
    modified_at         TEXT,
    indexed_at          TEXT NOT NULL,
    local_path          TEXT,
    processing_status   TEXT NOT NULL DEFAULT 'none',
    processing_steps_json TEXT,
    extracted_text      TEXT,
    text_len            INTEGER,
    embedding_model     TEXT,
    embedding_dims      INTEGER,
    topics_json         TEXT,
    entities_json       TEXT,
    pii_findings_json   TEXT,
    xref_links_json     TEXT,
    UNIQUE(source_id, remote_id)
);

CREATE TABLE IF NOT EXISTS index_jobs (
    job_id          TEXT PRIMARY KEY,
    source_id       TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'running',
    total_files     INTEGER NOT NULL DEFAULT 0,
    processed       INTEGER NOT NULL DEFAULT 0,
    errors          INTEGER NOT NULL DEFAULT 0,
    started_at      TEXT NOT NULL,
    finished_at     TEXT
);
"""


_ASSET_MIGRATION_COLUMNS = [
    ("local_path", "TEXT"),
    ("processing_status", "TEXT NOT NULL DEFAULT 'none'"),
    ("processing_steps_json", "TEXT"),
    ("extracted_text", "TEXT"),
    ("text_len", "INTEGER"),
    ("embedding_model", "TEXT"),
    ("embedding_dims", "INTEGER"),
    ("topics_json", "TEXT"),
    ("entities_json", "TEXT"),
    ("pii_findings_json", "TEXT"),
    ("xref_links_json", "TEXT"),
]

_SOURCE_MIGRATION_COLUMNS = [
    ("linked_source_id", "TEXT"),
    ("linked_source_ids_json", "TEXT"),
]


class DocStore:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._local = threading.local()
        self._conn().executescript(_SCHEMA)
        self._migrate()
        self._conn().commit()
        self._sweep_orphaned_running()

    def _sweep_orphaned_running(self) -> None:
        """Marks anything left in 'running' as failed on startup.

        A process just starting up has no background threads yet, so any row
        still marked 'running' from before can only mean the previous process
        was killed (crash, restart, manual stop) while that work was
        in-flight — it will never move again on its own. Left alone, the UI
        would show a spinner forever and a reindex would be permanently
        blocked by the 'indexing already in progress' guard. This runs once,
        synchronously, before the service accepts any requests.
        """
        conn = self._conn()
        note = "interrupted — the service restarted before this finished"

        stuck_assets = conn.execute(
            "SELECT asset_id, processing_steps_json FROM doc_assets WHERE processing_status='running'"
        ).fetchall()
        for row in stuck_assets:
            steps = json.loads(row["processing_steps_json"] or "null") or []
            for step in steps:
                if step.get("status") == "running":
                    step["status"] = "error"
                    step["detail"] = note
            conn.execute(
                "UPDATE doc_assets SET processing_status='error', processing_steps_json=? WHERE asset_id=?",
                (json.dumps(steps), row["asset_id"]),
            )

        stuck_jobs = conn.execute("SELECT job_id FROM index_jobs WHERE status='running'").fetchall()
        for row in stuck_jobs:
            conn.execute(
                "UPDATE index_jobs SET status='error', finished_at=? WHERE job_id=?",
                (self.now(), row["job_id"]),
            )

        stuck_sources = conn.execute("SELECT source_id FROM doc_sources WHERE status='indexing'").fetchall()
        for row in stuck_sources:
            conn.execute(
                "UPDATE doc_sources SET status='error', error_message=? WHERE source_id=?",
                (note, row["source_id"]),
            )

        conn.commit()
        if stuck_assets or stuck_jobs or stuck_sources:
            logger.warning(
                "DocStore startup sweep: recovered %d orphaned asset(s), %d job(s), %d source(s) "
                "left in 'running'/'indexing' by a previous process.",
                len(stuck_assets), len(stuck_jobs), len(stuck_sources),
            )

    def _migrate(self) -> None:
        """Adds columns introduced after these tables already existed (SQLite
        has no `ALTER TABLE ADD COLUMN IF NOT EXISTS`)."""
        conn = self._conn()
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(doc_assets)")}
        for col, decl in _ASSET_MIGRATION_COLUMNS:
            if col not in existing:
                conn.execute(f"ALTER TABLE doc_assets ADD COLUMN {col} {decl}")
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(doc_sources)")}
        for col, decl in _SOURCE_MIGRATION_COLUMNS:
            if col not in existing:
                conn.execute(f"ALTER TABLE doc_sources ADD COLUMN {col} {decl}")

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ── sources ──────────────────────────────────────────────────────────────

    def create_source(self, name: str, connector_type: str, config: dict,
                       linked_source_ids: Optional[list] = None) -> dict:
        source_id = str(uuid.uuid4())
        self._conn().execute(
            "INSERT INTO doc_sources (source_id,name,connector_type,config_json,status,created_at,linked_source_ids_json) "
            "VALUES (?,?,?,?,?,?,?)",
            (source_id, name, connector_type, json.dumps(config), "idle", self.now(),
             json.dumps(linked_source_ids or [])),
        )
        self._conn().commit()
        return self.get_source(source_id)

    def set_linked_sources(self, source_id: str, linked_source_ids: list) -> None:
        self._conn().execute(
            "UPDATE doc_sources SET linked_source_ids_json=? WHERE source_id=?",
            (json.dumps(linked_source_ids or []), source_id),
        )
        self._conn().commit()

    def get_source(self, source_id: str) -> Optional[dict]:
        row = self._conn().execute(
            "SELECT * FROM doc_sources WHERE source_id=?", (source_id,)
        ).fetchone()
        return self._source_public(row) if row else None

    def list_sources(self) -> List[dict]:
        rows = self._conn().execute(
            "SELECT * FROM doc_sources ORDER BY created_at DESC"
        ).fetchall()
        return [self._source_public(r) for r in rows]

    def delete_source(self, source_id: str) -> None:
        conn = self._conn()
        conn.execute("DELETE FROM doc_assets WHERE source_id=?", (source_id,))
        conn.execute("DELETE FROM index_jobs WHERE source_id=?", (source_id,))
        conn.execute("DELETE FROM doc_sources WHERE source_id=?", (source_id,))
        conn.commit()

    def set_source_status(self, source_id: str, status: str, error_message: str = None) -> None:
        conn = self._conn()
        if status == "ready":
            conn.execute(
                "UPDATE doc_sources SET status=?, error_message=?, indexed_at=? WHERE source_id=?",
                (status, error_message, self.now(), source_id),
            )
        else:
            conn.execute(
                "UPDATE doc_sources SET status=?, error_message=? WHERE source_id=?",
                (status, error_message, source_id),
            )
        conn.commit()

    def _source_public(self, row: sqlite3.Row) -> dict:
        d = dict(row)
        d["config"] = json.loads(d.pop("config_json") or "{}")
        d["config"] = _redact(d["config"])
        d["linked_source_ids"] = json.loads(d.pop("linked_source_ids_json") or "null") or []
        # Legacy single-value column — folded into the list above so old rows
        # created before multi-source linking still work; no longer written to.
        legacy = d.pop("linked_source_id", None)
        if legacy and legacy not in d["linked_source_ids"]:
            d["linked_source_ids"].append(legacy)
        d["table_count"] = self._conn().execute(
            "SELECT COUNT(*) FROM doc_assets WHERE source_id=?", (d["source_id"],)
        ).fetchone()[0]
        return d

    # ── assets ───────────────────────────────────────────────────────────────

    def upsert_asset(self, source_id: str, remote_id: str, file_name: str, size_bytes: int,
                      mime_type: str, checksum: Optional[str], modified_at: Optional[str],
                      local_path: Optional[str] = None) -> str:
        conn = self._conn()
        existing = conn.execute(
            "SELECT asset_id FROM doc_assets WHERE source_id=? AND remote_id=?",
            (source_id, remote_id),
        ).fetchone()
        asset_id = existing["asset_id"] if existing else str(uuid.uuid4())
        conn.execute(
            """INSERT INTO doc_assets (asset_id,source_id,remote_id,file_name,size_bytes,mime_type,checksum,modified_at,indexed_at,local_path)
               VALUES (?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(source_id, remote_id) DO UPDATE SET
                 file_name=excluded.file_name, size_bytes=excluded.size_bytes,
                 mime_type=excluded.mime_type, checksum=excluded.checksum,
                 modified_at=excluded.modified_at, indexed_at=excluded.indexed_at,
                 local_path=excluded.local_path""",
            (asset_id, source_id, remote_id, file_name, size_bytes, mime_type,
             checksum, modified_at, self.now(), local_path),
        )
        conn.commit()
        return asset_id

    def get_prior_checksum(self, source_id: str, remote_id: str) -> Optional[tuple]:
        """Returns (checksum, processing_status) for the asset already indexed
        under this (source_id, remote_id), or None if it hasn't been seen
        before. Used to skip reprocessing unchanged files on reindex."""
        row = self._conn().execute(
            "SELECT checksum, processing_status FROM doc_assets WHERE source_id=? AND remote_id=?",
            (source_id, remote_id),
        ).fetchone()
        return (row["checksum"], row["processing_status"]) if row else None

    def list_assets(self, source_id: str, limit: int = 500) -> List[dict]:
        rows = self._conn().execute(
            "SELECT * FROM doc_assets WHERE source_id=? ORDER BY file_name LIMIT ?",
            (source_id, limit),
        ).fetchall()
        return [self._asset_public(r) for r in rows]

    def get_asset(self, asset_id: str) -> Optional[dict]:
        row = self._conn().execute(
            "SELECT * FROM doc_assets WHERE asset_id=?", (asset_id,)
        ).fetchone()
        return self._asset_public(row, include_text=True) if row else None

    @staticmethod
    def _asset_public(row: sqlite3.Row, include_text: bool = False) -> dict:
        d = dict(row)
        d["processing_steps"] = json.loads(d.pop("processing_steps_json") or "null") or []
        d["topics"] = json.loads(d.pop("topics_json") or "null") or []
        d["entities"] = json.loads(d.pop("entities_json") or "null") or []
        d["pii_findings"] = json.loads(d.pop("pii_findings_json") or "null") or []
        d["xref_links"] = json.loads(d.pop("xref_links_json") or "null") or []
        if not include_text:
            d.pop("extracted_text", None)
        return d

    def update_asset_processing(self, asset_id: str, status: Optional[str] = None,
                                 steps: Optional[list] = None, extracted_text: Optional[str] = None,
                                 embedding_model: Optional[str] = None, embedding_dims: Optional[int] = None,
                                 topics: Optional[list] = None, entities: Optional[list] = None,
                                 pii_findings: Optional[list] = None, xref_links: Optional[list] = None) -> None:
        fields = {}
        if status is not None:
            fields["processing_status"] = status
        if steps is not None:
            fields["processing_steps_json"] = json.dumps(steps)
        if extracted_text is not None:
            fields["extracted_text"] = extracted_text
            fields["text_len"] = len(extracted_text)
        if embedding_model is not None:
            fields["embedding_model"] = embedding_model
        if embedding_dims is not None:
            fields["embedding_dims"] = embedding_dims
        if topics is not None:
            fields["topics_json"] = json.dumps(topics)
        if entities is not None:
            fields["entities_json"] = json.dumps(entities)
        if pii_findings is not None:
            fields["pii_findings_json"] = json.dumps(pii_findings)
        if xref_links is not None:
            fields["xref_links_json"] = json.dumps(xref_links)
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        self._conn().execute(
            f"UPDATE doc_assets SET {cols} WHERE asset_id=?",
            (*fields.values(), asset_id),
        )
        self._conn().commit()

    # ── index jobs ───────────────────────────────────────────────────────────

    def create_job(self, source_id: str) -> dict:
        job_id = str(uuid.uuid4())
        self._conn().execute(
            "INSERT INTO index_jobs (job_id,source_id,status,started_at) VALUES (?,?,?,?)",
            (job_id, source_id, "running", self.now()),
        )
        self._conn().commit()
        return self.get_job(job_id)

    def update_job(self, job_id: str, **fields) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        self._conn().execute(
            f"UPDATE index_jobs SET {cols} WHERE job_id=?",
            (*fields.values(), job_id),
        )
        self._conn().commit()

    def get_job(self, job_id: str) -> Optional[dict]:
        row = self._conn().execute(
            "SELECT * FROM index_jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_jobs(self, source_id: str, limit: int = 10) -> List[dict]:
        rows = self._conn().execute(
            "SELECT * FROM index_jobs WHERE source_id=? ORDER BY started_at DESC LIMIT ?",
            (source_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


_SECRET_KEYS = {"aws_secret_access_key", "access_token", "client_secret", "password"}


def _redact(config: dict) -> dict:
    return {k: ("***" if k in _SECRET_KEYS and v else v) for k, v in config.items()}
