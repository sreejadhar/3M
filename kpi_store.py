"""
KPI Store — persistence for BI Manager KPI definitions.

Uses the same SQLite/PostgreSQL backend pattern as metadata_catalog.py.

Schema
------
kpis
  kpi_id         TEXT PRIMARY KEY
  name           TEXT NOT NULL
  description    TEXT NOT NULL DEFAULT ''
  category       TEXT NOT NULL DEFAULT ''      -- e.g. "Sales", "Finance", "Supply Chain"
  source_id      TEXT NOT NULL DEFAULT ''      -- which data source this KPI applies to
  nl_formula     TEXT NOT NULL DEFAULT ''      -- natural language formula description
  sql_expression TEXT NOT NULL DEFAULT ''      -- LLM-compiled SQL expression
  unit           TEXT NOT NULL DEFAULT ''      -- '%', '$', 'units', etc.
  direction      TEXT NOT NULL DEFAULT 'up'    -- 'up' | 'down'
  status         TEXT NOT NULL DEFAULT 'draft' -- 'draft' | 'active' | 'deprecated'
  created_at     TEXT NOT NULL
  updated_at     TEXT NOT NULL

Public API
----------
create_kpi(**fields)            → dict
list_kpis(source_id, category, status)  → List[dict]
get_kpi(kpi_id)                 → dict | None
update_kpi(kpi_id, **fields)    → bool
delete_kpi(kpi_id)              → bool
compile_formula(kpi_id, column_context, model)  → str  (SQL expression)
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


# ── Environment helpers ────────────────────────────────────────────────────────

def _is_production() -> bool:
    return os.environ.get("APP_ENV", "").strip().lower() == "production"


def _is_postgres() -> bool:
    if not _is_production():
        return False
    return bool(os.environ.get("KG_POSTGRES_DSN", ""))


def _pg_dsn() -> str:
    return os.environ.get("KG_POSTGRES_DSN", "")


def _sqlite_path() -> str:
    return os.environ.get("KPI_DB", os.environ.get("METADATA_DB", "data/metadata.db"))


# ── Cursor abstraction (mirrors metadata_catalog pattern) ─────────────────────

class _PGCur:
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


class _SQLiteCur:
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


@contextmanager
def _cursor_ctx() -> Iterator[Any]:
    if _is_postgres():
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(_pg_dsn(), cursor_factory=psycopg2.extras.RealDictCursor)
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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── DDL ────────────────────────────────────────────────────────────────────────

_DDL_KPIS_SQLITE = """
CREATE TABLE IF NOT EXISTS kpis (
    kpi_id         TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    description    TEXT NOT NULL DEFAULT '',
    category       TEXT NOT NULL DEFAULT '',
    source_id      TEXT NOT NULL DEFAULT '',
    nl_formula     TEXT NOT NULL DEFAULT '',
    sql_expression TEXT NOT NULL DEFAULT '',
    unit           TEXT NOT NULL DEFAULT '',
    direction      TEXT NOT NULL DEFAULT 'up',
    status         TEXT NOT NULL DEFAULT 'draft',
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
)
"""

_DDL_KPIS_PG = """
CREATE TABLE IF NOT EXISTS kpis (
    kpi_id         TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    description    TEXT NOT NULL DEFAULT '',
    category       TEXT NOT NULL DEFAULT '',
    source_id      TEXT NOT NULL DEFAULT '',
    nl_formula     TEXT NOT NULL DEFAULT '',
    sql_expression TEXT NOT NULL DEFAULT '',
    unit           TEXT NOT NULL DEFAULT '',
    direction      TEXT NOT NULL DEFAULT 'up',
    status         TEXT NOT NULL DEFAULT 'draft',
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
)
"""

_DDL_VERSIONS_SQLITE = """
CREATE TABLE IF NOT EXISTS kpi_versions (
    version_id   TEXT PRIMARY KEY,
    kpi_id       TEXT NOT NULL,
    version_num  INTEGER NOT NULL,
    name           TEXT NOT NULL DEFAULT '',
    description    TEXT NOT NULL DEFAULT '',
    category       TEXT NOT NULL DEFAULT '',
    source_id      TEXT NOT NULL DEFAULT '',
    nl_formula     TEXT NOT NULL DEFAULT '',
    sql_expression TEXT NOT NULL DEFAULT '',
    unit           TEXT NOT NULL DEFAULT '',
    direction      TEXT NOT NULL DEFAULT 'up',
    status         TEXT NOT NULL DEFAULT 'draft',
    changed_by   TEXT NOT NULL DEFAULT '',
    change_note  TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL
)
"""

_DDL_VERSIONS_PG = _DDL_VERSIONS_SQLITE  # same DDL works on PG

_schema_ensured = False


def _ensure(cur: Any) -> None:
    global _schema_ensured
    if _is_postgres():
        cur.ddl(_DDL_KPIS_PG)
        cur.ddl(_DDL_VERSIONS_PG)
    else:
        cur.ddl(_DDL_KPIS_SQLITE)
        cur.ddl(_DDL_VERSIONS_SQLITE)
    # Migration: add approved_by if missing
    try:
        cur.execute("ALTER TABLE kpis ADD COLUMN approved_by TEXT NOT NULL DEFAULT ''")
    except Exception:
        pass
    _schema_ensured = True


# ── Public API ─────────────────────────────────────────────────────────────────

_ALLOWED_CREATE = {"name", "description", "category", "source_id",
                   "nl_formula", "sql_expression", "unit", "direction",
                   "status", "approved_by"}
_ALLOWED_UPDATE = _ALLOWED_CREATE


def create_kpi(**fields: Any) -> Dict:
    """
    Create a new KPI.
    Runs guardrail checks and duplicate detection before persisting.
    Returns {"kpi": {...}, "warnings": [...]}
    """
    guardrail_errors = _guardrail_check(fields)
    if guardrail_errors:
        raise ValueError("; ".join(guardrail_errors))

    warnings = _duplicate_check(fields.get("name", ""), fields.get("nl_formula", ""))

    kpi_id = str(uuid.uuid4())
    now = _now()
    row = {
        "kpi_id":         kpi_id,
        "name":           fields.get("name") or "",
        "description":    fields.get("description") or "",
        "category":       fields.get("category") or "",
        "source_id":      fields.get("source_id") or "",
        "nl_formula":     fields.get("nl_formula") or "",
        "sql_expression": fields.get("sql_expression") or "",
        "unit":           fields.get("unit") or "",
        "direction":      fields.get("direction") or "up",
        "status":         fields.get("status") or "draft",
        "approved_by":    fields.get("approved_by") or "",
        "created_at":     now,
        "updated_at":     now,
    }
    with _cursor_ctx() as cur:
        _ensure(cur)
        cur.execute(
            "INSERT INTO kpis (kpi_id,name,description,category,source_id,"
            "nl_formula,sql_expression,unit,direction,status,approved_by,created_at,updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (row["kpi_id"], row["name"], row["description"], row["category"],
             row["source_id"], row["nl_formula"], row["sql_expression"],
             row["unit"], row["direction"], row["status"], row["approved_by"],
             row["created_at"], row["updated_at"]),
        )
    return {"kpi": row, "warnings": warnings}


def list_kpis(
    source_id: Optional[str] = None,
    category: Optional[str] = None,
    status: Optional[str] = None,
) -> List[Dict]:
    """Return KPIs, optionally filtered."""
    clauses, params = [], []
    if source_id:
        clauses.append("source_id=?")
        params.append(source_id)
    if category:
        clauses.append("LOWER(category)=LOWER(?)")
        params.append(category)
    if status:
        clauses.append("status=?")
        params.append(status)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with _cursor_ctx() as cur:
        _ensure(cur)
        rows = cur.execute(
            f"SELECT * FROM kpis {where} ORDER BY category, name",
            tuple(params),
        ).fetchall()
    return [dict(r) for r in rows]


def get_kpi(kpi_id: str) -> Optional[Dict]:
    with _cursor_ctx() as cur:
        _ensure(cur)
        row = cur.execute("SELECT * FROM kpis WHERE kpi_id=?", (kpi_id,)).fetchone()
    return dict(row) if row else None


def update_kpi(kpi_id: str, changed_by: str = "", change_note: str = "",
               _skip_checks: bool = False, **fields: Any) -> Dict:
    """
    Update a KPI, snapshotting the current state into kpi_versions first.
    Runs guardrail checks and duplicate detection on the new values.
    Returns {"ok": True, "warnings": [...]}; raises ValueError on guardrail failure.
    _skip_checks=True bypasses duplicate detection (used internally for rollback).
    """
    updates = {k: v for k, v in fields.items() if k in _ALLOWED_UPDATE}
    if not updates:
        return {"ok": False, "warnings": []}

    guardrail_errors = _guardrail_check(updates)
    if guardrail_errors:
        raise ValueError("; ".join(guardrail_errors))

    with _cursor_ctx() as cur:
        _ensure(cur)
        existing = cur.execute("SELECT * FROM kpis WHERE kpi_id=?", (kpi_id,)).fetchone()
        if not existing:
            return {"ok": False, "warnings": []}

        # Snapshot current state before overwriting
        version_num = (cur.execute(
            "SELECT COALESCE(MAX(version_num),0) AS max_ver FROM kpi_versions WHERE kpi_id=?",
            (kpi_id,)
        ).fetchone() or {}).get("max_ver", 0) or 0
        version_num += 1
        cur.execute(
            "INSERT INTO kpi_versions "
            "(version_id,kpi_id,version_num,name,description,category,source_id,"
            "nl_formula,sql_expression,unit,direction,status,changed_by,change_note,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), kpi_id, version_num,
             existing.get("name",""), existing.get("description",""),
             existing.get("category",""), existing.get("source_id",""),
             existing.get("nl_formula",""), existing.get("sql_expression",""),
             existing.get("unit",""), existing.get("direction","up"),
             existing.get("status","draft"),
             changed_by, change_note, _now()),
        )

        updates["updated_at"] = _now()
        set_clause = ", ".join(f"{k}=?" for k in updates)
        cur.execute(
            f"UPDATE kpis SET {set_clause} WHERE kpi_id=?",
            tuple(updates.values()) + (kpi_id,),
        )

    warnings: List[Dict] = []
    if not _skip_checks:
        name      = fields.get("name") or (get_kpi(kpi_id) or {}).get("name", "")
        nl        = fields.get("nl_formula") or (get_kpi(kpi_id) or {}).get("nl_formula", "")
        warnings  = _duplicate_check(name, nl, exclude_kpi_id=kpi_id)

    return {"ok": True, "warnings": warnings}


def delete_kpi(kpi_id: str) -> bool:
    with _cursor_ctx() as cur:
        _ensure(cur)
        cur.execute("DELETE FROM kpi_versions WHERE kpi_id=?", (kpi_id,))
        cur.execute("DELETE FROM kpis WHERE kpi_id=?", (kpi_id,))
    return True


# ── Versioning ────────────────────────────────────────────────────────────────

def list_kpi_versions(kpi_id: str) -> List[Dict]:
    """Return all snapshots for a KPI, newest first."""
    with _cursor_ctx() as cur:
        _ensure(cur)
        rows = cur.execute(
            "SELECT * FROM kpi_versions WHERE kpi_id=? ORDER BY version_num DESC",
            (kpi_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def rollback_kpi_version(kpi_id: str, version_num: int,
                         changed_by: str = "") -> Optional[Dict]:
    """
    Restore a KPI to a previous snapshot.
    The current state is itself snapshotted first (so rollback is auditable).
    Returns the restored KPI dict or None if version not found.
    """
    with _cursor_ctx() as cur:
        _ensure(cur)
        snap = cur.execute(
            "SELECT * FROM kpi_versions WHERE kpi_id=? AND version_num=?",
            (kpi_id, version_num)
        ).fetchone()
    if not snap:
        return None
    snap = dict(snap)
    update_kpi(
        kpi_id,
        changed_by=changed_by,
        change_note=f"Rolled back to version {version_num}",
        _skip_checks=True,
        name=snap["name"],
        description=snap["description"],
        category=snap["category"],
        source_id=snap["source_id"],
        nl_formula=snap["nl_formula"],
        sql_expression=snap["sql_expression"],
        unit=snap["unit"],
        direction=snap["direction"],
        status=snap["status"],
    )
    return get_kpi(kpi_id)


# ── Guardrails ────────────────────────────────────────────────────────────────

_SQL_DANGEROUS = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|TRUNCATE|ALTER|CREATE|GRANT|REVOKE|EXEC|EXECUTE)\b",
    re.IGNORECASE
)
_SQL_COMMENT = re.compile(r"(--|/\*|\*/)")


def _guardrail_check(fields: Dict) -> List[str]:
    """
    Returns a list of error strings. Empty list = all checks pass.
    Checks:
      1. SQL expression must not contain DML/DDL keywords or comment tokens
      2. KPI being set to 'active' must have a non-empty sql_expression
      3. direction must be 'up' or 'down'
    """
    errors: List[str] = []
    sql = (fields.get("sql_expression") or "").strip()

    if sql:
        m = _SQL_DANGEROUS.search(sql)
        if m:
            errors.append(
                f"SQL expression contains forbidden keyword '{m.group().upper()}' — "
                "only SELECT expressions are allowed"
            )
        if _SQL_COMMENT.search(sql):
            errors.append("SQL expression must not contain comment tokens (-- or /* */)")

    status = (fields.get("status") or "").strip()
    if status == "active":
        if not sql:
            errors.append(
                "A KPI must have a compiled SQL expression before it can be set to Active"
            )
        direction = (fields.get("direction") or "").strip()
        if direction and direction not in ("up", "down"):
            errors.append("direction must be 'up' or 'down'")

    return errors


def activate_kpi(kpi_id: str, approved_by: str = "") -> Dict:
    """
    Gate: validate the KPI passes all guardrails, then set status → active.
    Returns {"ok": True, "kpi": {...}} or raises ValueError with failure reasons.
    """
    kpi = get_kpi(kpi_id)
    if not kpi:
        raise ValueError(f"KPI {kpi_id!r} not found")

    errors = _guardrail_check({
        "sql_expression": kpi.get("sql_expression", ""),
        "status": "active",
        "direction": kpi.get("direction", "up"),
    })
    if not kpi.get("sql_expression", "").strip():
        errors.append("SQL expression is empty — run Compile first")
    if errors:
        raise ValueError("; ".join(errors))

    update_kpi(
        kpi_id,
        changed_by=approved_by,
        change_note="Activated",
        status="active",
        approved_by=approved_by,
    )
    return get_kpi(kpi_id)


# ── Duplicate Detection ───────────────────────────────────────────────────────

def _levenshtein(a: str, b: str) -> int:
    """Pure-Python Levenshtein distance."""
    a, b = a.lower(), b.lower()
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1,
                            prev[j - 1] + (0 if ca == cb else 1)))
        prev = curr
    return prev[-1]


def _normalise_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _tfidf_similarity(text_a: str, text_b: str) -> float:
    """
    Cosine similarity between two short texts using character n-gram TF-IDF.
    Falls back to 0.0 if scikit-learn is unavailable.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as _cos
        import numpy as np
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        tfidf = vec.fit_transform([text_a, text_b])
        return float(_cos(tfidf[0], tfidf[1])[0][0])
    except Exception:
        return 0.0


def _duplicate_check(name: str, nl_formula: str,
                     exclude_kpi_id: Optional[str] = None) -> List[Dict]:
    """
    Returns list of {type, kpi_id, name, score, message} for suspected duplicates.
    Does NOT block creation — callers surface these as warnings.
    """
    existing = list_kpis()
    warnings: List[Dict] = []
    norm_new = _normalise_name(name)

    for k in existing:
        if k["kpi_id"] == exclude_kpi_id:
            continue

        # Pass 1 — name edit distance
        norm_existing = _normalise_name(k["name"])
        dist = _levenshtein(norm_new, norm_existing)
        if dist <= 3 and norm_new and norm_existing:
            warnings.append({
                "type": "name_similarity",
                "kpi_id": k["kpi_id"],
                "name": k["name"],
                "score": dist,
                "message": (
                    f"'{k['name']}' has a very similar name (edit distance {dist}) — "
                    "check if this is a duplicate"
                ),
            })
            continue  # no need to also check formula if name matches

        # Pass 2 — formula semantic similarity (only if both have formulas)
        if nl_formula and k.get("nl_formula"):
            sim = _tfidf_similarity(nl_formula, k["nl_formula"])
            if sim >= 0.85:
                warnings.append({
                    "type": "formula_similarity",
                    "kpi_id": k["kpi_id"],
                    "name": k["name"],
                    "score": round(sim, 3),
                    "message": (
                        f"'{k['name']}' has a very similar formula "
                        f"(similarity {sim:.0%}) — check if this is a duplicate"
                    ),
                })

    return warnings


# ── LLM Formula Compiler ──────────────────────────────────────────────────────

_COMPILE_SYSTEM = """\
You are a SQL expert. Given a natural language KPI formula description and the available columns
from the data source, convert the formula into a valid SQL expression.

Rules:
- Return ONLY the SQL expression — no SELECT, no FROM, no WHERE, no explanations
- Use the exact column names provided in the available columns list
- For growth/change calculations use: (current_val - prior_val) / NULLIF(prior_val, 0) * 100
- For period-over-period: use a subquery or LAG() window function as appropriate
- If "last period" or "last full period" is mentioned, use: period_col = (SELECT MAX(period_col) FROM table)
- Use standard SQL that works in SQLite and PostgreSQL
- If the formula references columns not in the list, use your best guess based on column names

Return the SQL expression only, nothing else.
"""


def compile_formula(kpi_id: str, column_context: str, model: Optional[str] = None) -> str:
    """
    Call a lightweight LLM (Haiku) to compile the KPI's nl_formula into a SQL expression.
    Saves the result to kpis.sql_expression and returns the compiled expression.
    """
    import anthropic

    model = model or os.environ.get("DIALOG_LLM_MODEL", "claude-haiku-4-5-20251001")
    kpi = get_kpi(kpi_id)
    if not kpi:
        raise ValueError(f"KPI {kpi_id} not found")

    nl = kpi.get("nl_formula") or ""
    if not nl.strip():
        return ""

    user_msg = (
        f"KPI Name: {kpi['name']}\n"
        f"Natural language formula: {nl}\n\n"
        f"Available columns:\n{column_context}\n\n"
        "Convert the formula to a SQL expression."
    )

    try:
        from llm_client import get_client
        client = get_client()
        msg = client.messages.create(
            model=model,
            max_tokens=512,
            temperature=0.0,
            system=_COMPILE_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = msg.content[0].text.strip() if msg.content else ""
    except Exception as exc:
        logger.warning("compile_formula: LLM call failed for kpi %r — %s", kpi_id, exc)
        return ""

    # Strip markdown fences if any
    expr = re.sub(r"```(?:sql)?\s*", "", raw).strip().rstrip("`").strip()
    # Save compiled expression
    update_kpi(kpi_id, sql_expression=expr)
    return expr
