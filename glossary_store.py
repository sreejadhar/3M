"""
Business Glossary Store — persistence for domain terms, synonyms, and KPI thresholds.

Follows the same SQLite/PostgreSQL backend pattern as kpi_store.py.

Schema
------
glossary_terms
  term_id      TEXT PRIMARY KEY
  name         TEXT NOT NULL UNIQUE          -- "Gross Margin"
  definition   TEXT NOT NULL DEFAULT ''      -- human-readable definition
  formula      TEXT NOT NULL DEFAULT ''      -- nl formula description
  sql_hint     TEXT NOT NULL DEFAULT ''      -- optional SQL fragment hint for the planner
  domain       TEXT NOT NULL DEFAULT ''      -- Finance | Supply Chain | CPG | …
  owner        TEXT NOT NULL DEFAULT ''
  approved     INTEGER NOT NULL DEFAULT 1
  created_at   TEXT NOT NULL
  updated_at   TEXT NOT NULL

glossary_synonyms
  synonym_id   TEXT PRIMARY KEY
  term_id      TEXT NOT NULL → glossary_terms
  synonym      TEXT NOT NULL               -- "GP%", "gross profit margin"
  domain_scope TEXT NOT NULL DEFAULT ''    -- empty = universal
  created_at   TEXT NOT NULL

glossary_thresholds
  threshold_id     TEXT PRIMARY KEY
  term_id          TEXT NOT NULL → glossary_terms
  threshold_red    REAL                    -- below this value is red
  threshold_amber  REAL                    -- below this value is amber
  benchmark_value  REAL                    -- industry benchmark
  benchmark_source TEXT NOT NULL DEFAULT ''
  direction        TEXT NOT NULL DEFAULT 'higher_is_better'
  unit             TEXT NOT NULL DEFAULT ''   -- '%', '$', 'days', …
  created_at       TEXT NOT NULL
  updated_at       TEXT NOT NULL

Public API
----------
create_term(**fields)                → dict
list_terms(domain, approved)         → List[dict]
get_term(term_id)                    → dict | None
update_term(term_id, **fields)       → bool
delete_term(term_id)                 → bool
search_terms(q, domain, limit)       → List[dict]   # FTS on name + synonyms + definition
add_synonym(term_id, synonym, domain_scope) → dict
remove_synonym(synonym_id)           → bool
upsert_threshold(term_id, **fields)  → dict
get_threshold(term_id)               → dict | None
"""
from __future__ import annotations

import logging
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# ── DB path (shares METADATA_DB so it's co-located with KPI store) ────────────

def _sqlite_path() -> str:
    return os.environ.get(
        "GLOSSARY_DB",
        os.environ.get("METADATA_DB", "data/metadata.db"),
    )


# ── Schema ─────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS glossary_terms (
    term_id      TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    definition   TEXT NOT NULL DEFAULT '',
    formula      TEXT NOT NULL DEFAULT '',
    sql_hint     TEXT NOT NULL DEFAULT '',
    domain       TEXT NOT NULL DEFAULT '',
    owner        TEXT NOT NULL DEFAULT '',
    approved     INTEGER NOT NULL DEFAULT 1,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_glossary_name ON glossary_terms(name);

CREATE TABLE IF NOT EXISTS glossary_synonyms (
    synonym_id   TEXT PRIMARY KEY,
    term_id      TEXT NOT NULL REFERENCES glossary_terms(term_id) ON DELETE CASCADE,
    synonym      TEXT NOT NULL,
    domain_scope TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_syn_term ON glossary_synonyms(term_id);

CREATE TABLE IF NOT EXISTS glossary_thresholds (
    threshold_id     TEXT PRIMARY KEY,
    term_id          TEXT NOT NULL REFERENCES glossary_terms(term_id) ON DELETE CASCADE,
    threshold_red    REAL,
    threshold_amber  REAL,
    benchmark_value  REAL,
    benchmark_source TEXT NOT NULL DEFAULT '',
    direction        TEXT NOT NULL DEFAULT 'higher_is_better',
    unit             TEXT NOT NULL DEFAULT '',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_thresh_term ON glossary_thresholds(term_id);
"""


# ── Connection helper ──────────────────────────────────────────────────────────

@contextmanager
def _db() -> Generator[sqlite3.Connection, None, None]:
    path = _sqlite_path()
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        for stmt in _SCHEMA.strip().split(";"):
            s = stmt.strip()
            if s:
                conn.execute(s)
        conn.commit()
        yield conn
    finally:
        conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row(r: Optional[sqlite3.Row]) -> Optional[Dict]:
    return dict(r) if r else None


# ── Terms ──────────────────────────────────────────────────────────────────────

def create_term(
    name: str,
    definition: str = "",
    formula: str = "",
    sql_hint: str = "",
    domain: str = "",
    owner: str = "",
    approved: bool = True,
) -> Dict:
    term_id = str(uuid.uuid4())
    now = _now()
    with _db() as conn:
        conn.execute(
            """INSERT INTO glossary_terms
               (term_id,name,definition,formula,sql_hint,domain,owner,approved,
                created_at,updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (term_id, name.strip(), definition, formula, sql_hint,
             domain, owner, int(approved), now, now),
        )
        conn.commit()
    return get_term(term_id)


def list_terms(
    domain: str = "",
    approved_only: bool = False,
) -> List[Dict]:
    with _db() as conn:
        clauses, params = [], []
        if domain:
            clauses.append("domain = ?")
            params.append(domain)
        if approved_only:
            clauses.append("approved = 1")
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = conn.execute(
            f"SELECT * FROM glossary_terms {where} ORDER BY domain, name",
            params,
        ).fetchall()
        terms = [dict(r) for r in rows]
    return [_attach_synonyms_and_threshold(t) for t in terms]


def get_term(term_id: str) -> Optional[Dict]:
    with _db() as conn:
        row = conn.execute(
            "SELECT * FROM glossary_terms WHERE term_id=?", (term_id,)
        ).fetchone()
    if not row:
        return None
    return _attach_synonyms_and_threshold(dict(row))


def get_term_by_name(name: str) -> Optional[Dict]:
    with _db() as conn:
        row = conn.execute(
            "SELECT * FROM glossary_terms WHERE LOWER(name)=LOWER(?)", (name,)
        ).fetchone()
    if not row:
        return None
    return _attach_synonyms_and_threshold(dict(row))


def update_term(term_id: str, **fields) -> bool:
    allowed = {"name", "definition", "formula", "sql_hint", "domain", "owner", "approved"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return False
    updates["updated_at"] = _now()
    set_clause = ", ".join(f"{k}=?" for k in updates)
    with _db() as conn:
        cur = conn.execute(
            f"UPDATE glossary_terms SET {set_clause} WHERE term_id=?",
            list(updates.values()) + [term_id],
        )
        conn.commit()
        return cur.rowcount > 0


def delete_term(term_id: str) -> bool:
    with _db() as conn:
        cur = conn.execute(
            "DELETE FROM glossary_terms WHERE term_id=?", (term_id,)
        )
        conn.commit()
        return cur.rowcount > 0


def search_terms(q: str, domain: str = "", limit: int = 20) -> List[Dict]:
    """
    Case-insensitive substring search across term names, definitions, and synonyms.
    Returns terms with attached synonyms and thresholds.
    """
    q_lower = f"%{q.lower()}%"
    with _db() as conn:
        # Match on name or definition
        clauses = ["(LOWER(t.name) LIKE ? OR LOWER(t.definition) LIKE ?)"]
        params: List[Any] = [q_lower, q_lower]
        if domain:
            clauses.append("t.domain = ?")
            params.append(domain)
        where = "WHERE " + " AND ".join(clauses)
        rows = conn.execute(
            f"SELECT DISTINCT t.* FROM glossary_terms t {where} "
            f"UNION "
            f"SELECT DISTINCT t.* FROM glossary_terms t "
            f"JOIN glossary_synonyms s ON s.term_id = t.term_id "
            f"WHERE LOWER(s.synonym) LIKE ? "
            f"{'AND t.domain = ?' if domain else ''} "
            f"ORDER BY t.domain, t.name LIMIT ?",
            params + ([q_lower, domain, limit] if domain else [q_lower, limit]),
        ).fetchall()
        terms = [dict(r) for r in rows]
    return [_attach_synonyms_and_threshold(t) for t in terms]


# ── Synonyms ───────────────────────────────────────────────────────────────────

def add_synonym(term_id: str, synonym: str, domain_scope: str = "") -> Dict:
    sid = str(uuid.uuid4())
    now = _now()
    with _db() as conn:
        conn.execute(
            "INSERT INTO glossary_synonyms (synonym_id,term_id,synonym,domain_scope,created_at) "
            "VALUES (?,?,?,?,?)",
            (sid, term_id, synonym.strip(), domain_scope, now),
        )
        conn.commit()
    return {"synonym_id": sid, "term_id": term_id, "synonym": synonym, "domain_scope": domain_scope}


def remove_synonym(synonym_id: str) -> bool:
    with _db() as conn:
        cur = conn.execute(
            "DELETE FROM glossary_synonyms WHERE synonym_id=?", (synonym_id,)
        )
        conn.commit()
        return cur.rowcount > 0


def list_synonyms(term_id: str) -> List[Dict]:
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM glossary_synonyms WHERE term_id=? ORDER BY synonym",
            (term_id,),
        ).fetchall()
        return [dict(r) for r in rows]


# ── Thresholds ─────────────────────────────────────────────────────────────────

def upsert_threshold(
    term_id: str,
    threshold_red: Optional[float] = None,
    threshold_amber: Optional[float] = None,
    benchmark_value: Optional[float] = None,
    benchmark_source: str = "",
    direction: str = "higher_is_better",
    unit: str = "",
) -> Dict:
    now = _now()
    with _db() as conn:
        existing = conn.execute(
            "SELECT threshold_id FROM glossary_thresholds WHERE term_id=?", (term_id,)
        ).fetchone()
        if existing:
            conn.execute(
                """UPDATE glossary_thresholds SET
                   threshold_red=?, threshold_amber=?, benchmark_value=?,
                   benchmark_source=?, direction=?, unit=?, updated_at=?
                   WHERE term_id=?""",
                (threshold_red, threshold_amber, benchmark_value,
                 benchmark_source, direction, unit, now, term_id),
            )
            tid = existing["threshold_id"]
        else:
            tid = str(uuid.uuid4())
            conn.execute(
                """INSERT INTO glossary_thresholds
                   (threshold_id,term_id,threshold_red,threshold_amber,benchmark_value,
                    benchmark_source,direction,unit,created_at,updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (tid, term_id, threshold_red, threshold_amber, benchmark_value,
                 benchmark_source, direction, unit, now, now),
            )
        conn.commit()
    return get_threshold(term_id)


def get_threshold(term_id: str) -> Optional[Dict]:
    with _db() as conn:
        row = conn.execute(
            "SELECT * FROM glossary_thresholds WHERE term_id=?", (term_id,)
        ).fetchone()
        return _row(row)


# ── Internal helper ────────────────────────────────────────────────────────────

def _attach_synonyms_and_threshold(term: Dict) -> Dict:
    term_id = term["term_id"]
    with _db() as conn:
        syn_rows = conn.execute(
            "SELECT * FROM glossary_synonyms WHERE term_id=? ORDER BY synonym",
            (term_id,),
        ).fetchall()
        thr_row = conn.execute(
            "SELECT * FROM glossary_thresholds WHERE term_id=?", (term_id,)
        ).fetchone()
    term["synonyms"]  = [dict(r) for r in syn_rows]
    term["threshold"] = dict(thr_row) if thr_row else None
    return term
