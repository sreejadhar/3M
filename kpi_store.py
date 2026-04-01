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

_schema_ensured = False


def _ensure(cur: Any) -> None:
    global _schema_ensured
    if _is_postgres():
        cur.ddl(_DDL_KPIS_PG)
    else:
        cur.ddl(_DDL_KPIS_SQLITE)
    _schema_ensured = True


# ── Public API ─────────────────────────────────────────────────────────────────

_ALLOWED_CREATE = {"name", "description", "category", "source_id",
                   "nl_formula", "sql_expression", "unit", "direction", "status"}
_ALLOWED_UPDATE = _ALLOWED_CREATE


def create_kpi(**fields: Any) -> Dict:
    """Create a new KPI. Returns the created KPI dict."""
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
        "created_at":     now,
        "updated_at":     now,
    }
    with _cursor_ctx() as cur:
        _ensure(cur)
        cur.execute(
            "INSERT INTO kpis (kpi_id,name,description,category,source_id,"
            "nl_formula,sql_expression,unit,direction,status,created_at,updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (row["kpi_id"], row["name"], row["description"], row["category"],
             row["source_id"], row["nl_formula"], row["sql_expression"],
             row["unit"], row["direction"], row["status"],
             row["created_at"], row["updated_at"]),
        )
    return row


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


def update_kpi(kpi_id: str, **fields: Any) -> bool:
    updates = {k: v for k, v in fields.items() if k in _ALLOWED_UPDATE}
    if not updates:
        return False
    updates["updated_at"] = _now()
    set_clause = ", ".join(f"{k}=?" for k in updates)
    with _cursor_ctx() as cur:
        _ensure(cur)
        cur.execute(
            f"UPDATE kpis SET {set_clause} WHERE kpi_id=?",
            tuple(updates.values()) + (kpi_id,),
        )
    return True


def delete_kpi(kpi_id: str) -> bool:
    with _cursor_ctx() as cur:
        _ensure(cur)
        cur.execute("DELETE FROM kpis WHERE kpi_id=?", (kpi_id,))
    return True


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
        client = anthropic.Anthropic(
            api_key=os.environ.get("CLAUDE_API_KEY", ""),
            base_url=os.environ.get("CLAUDE_BASE_URL", "https://api.anthropic.com"),
        )
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
