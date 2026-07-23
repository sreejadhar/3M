"""
KG Bridge Registry — cross-KG join key definitions.

  declared  — explicitly registered by admin (confidence=1.0)
  inferred  — auto-detected by comparing column names across KGs (confidence<1.0)

Persists to PostgreSQL when KG_POSTGRES_DSN is set; falls back to SQLite.
See dialog_agent/pg_store.py for backend configuration.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from . import pg_store


@dataclass
class Bridge:
    from_kg:     str
    from_column: str
    to_kg:       str
    to_column:   str
    from_entity: str   = ""
    to_entity:   str   = ""
    join_type:   str   = "FK"
    source:      str   = "declared"
    confidence:  float = 1.0
    enabled:     bool  = True
    notes:       str   = ""
    id:          Optional[int] = None
    created_at:  float = 0.0
    updated_at:  float = 0.0

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


# ── DDL — backend-specific because of SERIAL vs AUTOINCREMENT and BOOLEAN ─────

_DDL_PG = """
CREATE TABLE IF NOT EXISTS kg_bridges (
    id          SERIAL PRIMARY KEY,
    from_kg     TEXT NOT NULL,
    from_entity TEXT NOT NULL DEFAULT '',
    from_column TEXT NOT NULL,
    to_kg       TEXT NOT NULL,
    to_entity   TEXT NOT NULL DEFAULT '',
    to_column   TEXT NOT NULL,
    join_type   TEXT NOT NULL DEFAULT 'FK',
    source      TEXT NOT NULL DEFAULT 'declared',
    confidence  REAL NOT NULL DEFAULT 1.0,
    enabled     BOOLEAN NOT NULL DEFAULT TRUE,
    notes       TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL DEFAULT 0,
    updated_at  REAL NOT NULL DEFAULT 0,
    UNIQUE(from_kg, from_column, to_kg, to_column)
)
"""

_DDL_SQLITE = """
CREATE TABLE IF NOT EXISTS kg_bridges (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    from_kg     TEXT NOT NULL,
    from_entity TEXT NOT NULL DEFAULT '',
    from_column TEXT NOT NULL,
    to_kg       TEXT NOT NULL,
    to_entity   TEXT NOT NULL DEFAULT '',
    to_column   TEXT NOT NULL,
    join_type   TEXT NOT NULL DEFAULT 'FK',
    source      TEXT NOT NULL DEFAULT 'declared',
    confidence  REAL NOT NULL DEFAULT 1.0,
    enabled     INTEGER NOT NULL DEFAULT 1,
    notes       TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL DEFAULT 0,
    updated_at  REAL NOT NULL DEFAULT 0,
    UNIQUE(from_kg, from_column, to_kg, to_column)
)
"""


def _ensure(cur) -> None:
    cur.ddl(_DDL_PG if pg_store.is_postgres() else _DDL_SQLITE)


def _enc_bool(v: bool) -> Any:
    """Encode a bool for storage (native bool for PG, int for SQLite)."""
    return bool(v) if pg_store.is_postgres() else int(v)


# ── Public API ────────────────────────────────────────────────────────────────

def upsert(b: Bridge) -> int:
    """Insert or update a bridge. Returns the row id."""
    now = time.time()
    enabled = _enc_bool(b.enabled)
    params = (
        b.from_kg, b.from_entity, b.from_column,
        b.to_kg,   b.to_entity,   b.to_column,
        b.join_type, b.source, b.confidence, enabled, b.notes,
        b.created_at or now, now,
    )

    if pg_store.is_postgres():
        sql = """
        INSERT INTO kg_bridges
            (from_kg,from_entity,from_column,to_kg,to_entity,to_column,
             join_type,source,confidence,enabled,notes,created_at,updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(from_kg,from_column,to_kg,to_column) DO UPDATE SET
            from_entity=EXCLUDED.from_entity, to_entity=EXCLUDED.to_entity,
            join_type=EXCLUDED.join_type,     source=EXCLUDED.source,
            confidence=EXCLUDED.confidence,   enabled=EXCLUDED.enabled,
            notes=EXCLUDED.notes,             updated_at=EXCLUDED.updated_at
        RETURNING id
        """
        with pg_store.cursor_ctx() as cur:
            _ensure(cur)
            return cur.insert_returning_id(sql, params) or -1
    else:
        sql = """
        INSERT INTO kg_bridges
            (from_kg,from_entity,from_column,to_kg,to_entity,to_column,
             join_type,source,confidence,enabled,notes,created_at,updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(from_kg,from_column,to_kg,to_column) DO UPDATE SET
            from_entity=excluded.from_entity, to_entity=excluded.to_entity,
            join_type=excluded.join_type,     source=excluded.source,
            confidence=excluded.confidence,   enabled=excluded.enabled,
            notes=excluded.notes,             updated_at=excluded.updated_at
        """
        with pg_store.cursor_ctx() as cur:
            _ensure(cur)
            cur.execute(sql, params)
            row = cur.execute(
                "SELECT id FROM kg_bridges "
                "WHERE from_kg=? AND from_column=? AND to_kg=? AND to_column=?",
                (b.from_kg, b.from_column, b.to_kg, b.to_column),
            ).fetchone()
            return row["id"] if row else -1


def list_all(enabled_only: bool = False) -> List[Bridge]:
    with pg_store.cursor_ctx() as cur:
        _ensure(cur)
        q = "SELECT * FROM kg_bridges"
        if enabled_only:
            q += " WHERE enabled = " + ("TRUE" if pg_store.is_postgres() else "1")
        q += " ORDER BY source DESC, confidence DESC, id"
        rows = cur.execute(q).fetchall()
    return [_row(r) for r in rows]


def list_for_kgs(kg_ids: List[str], enabled_only: bool = True) -> List[Bridge]:
    s = set(kg_ids)
    return [b for b in list_all(enabled_only=enabled_only)
            if b.from_kg in s and b.to_kg in s]


def get_by_id(bridge_id: int) -> Optional[Bridge]:
    with pg_store.cursor_ctx() as cur:
        _ensure(cur)
        r = cur.execute(
            "SELECT * FROM kg_bridges WHERE id=?", (bridge_id,)
        ).fetchone()
    return _row(r) if r else None


def delete(bridge_id: int) -> None:
    with pg_store.cursor_ctx() as cur:
        _ensure(cur)
        cur.execute("DELETE FROM kg_bridges WHERE id=?", (bridge_id,))


def set_enabled(bridge_id: int, enabled: bool) -> None:
    with pg_store.cursor_ctx() as cur:
        _ensure(cur)
        cur.execute(
            "UPDATE kg_bridges SET enabled=?, updated_at=? WHERE id=?",
            (_enc_bool(enabled), time.time(), bridge_id),
        )


def promote_to_declared(bridge_id: int) -> None:
    with pg_store.cursor_ctx() as cur:
        _ensure(cur)
        cur.execute(
            "UPDATE kg_bridges SET source='declared', confidence=1.0, "
            "updated_at=? WHERE id=?",
            (time.time(), bridge_id),
        )


# ── Internal ──────────────────────────────────────────────────────────────────

def _find_existing(from_kg, from_col, to_kg, to_col) -> Optional[Bridge]:
    with pg_store.cursor_ctx() as cur:
        _ensure(cur)
        r = cur.execute(
            "SELECT * FROM kg_bridges "
            "WHERE from_kg=? AND from_column=? AND to_kg=? AND to_column=?",
            (from_kg, from_col, to_kg, to_col),
        ).fetchone()
    return _row(r) if r else None


def _row(r: dict) -> Bridge:
    return Bridge(
        id=r["id"], from_kg=r["from_kg"], from_entity=r["from_entity"],
        from_column=r["from_column"], to_kg=r["to_kg"], to_entity=r["to_entity"],
        to_column=r["to_column"], join_type=r["join_type"], source=r["source"],
        confidence=float(r["confidence"]), enabled=bool(r["enabled"]),
        notes=r["notes"], created_at=float(r["created_at"]),
        updated_at=float(r["updated_at"]),
    )


# ── Bridge inference ──────────────────────────────────────────────────────────
# The enterprise inference engine lives in kg_inference_engine.py.
# run_inference_and_save() delegates to it; extra kwargs let callers pass
# metadata reports and domain tags for richer signals.

def run_inference_and_save(
    kg_a_id: str,
    kg_a_nodes: List[Dict],
    kg_b_id: str,
    kg_b_nodes: List[Dict],
    kg_a_report: Optional[Dict] = None,
    kg_b_report: Optional[Dict] = None,
    kg_a_domain: str = "",
    kg_b_domain: str = "",
    background_tasks: Any = None,
    sample_fn: Optional[Any] = None,
    main_loop: Any = None,
) -> List[Bridge]:
    """
    Infer bridges between two KGs using the enterprise multi-tier engine.
    Never overwrites declared bridges.  LLM validation runs asynchronously.

    ``sample_fn(kg_id, entity, col) -> set | None``, when provided, powers
    Tier V (data value-overlap matching) — pass the same sampler used
    elsewhere (e.g. orchestrator_api._make_bridge_sample_fn()) so a join key
    whose names don't match (like "personal_no" vs "employee_id") can still
    be found from real data, not just column names. Omit to skip that tier.

    ``main_loop``, when provided, lets LLM validation get scheduled correctly
    even if this call itself is running off the main thread (e.g. via
    asyncio.to_thread) — pass the loop captured on the caller's async thread.
    """
    from .kg_inference_engine import KGContext, run_enterprise_inference_and_save

    ctx_a = KGContext(
        kg_id=kg_a_id, display_name=kg_a_id,
        domain=kg_a_domain, nodes=kg_a_nodes, report=kg_a_report,
    )
    ctx_b = KGContext(
        kg_id=kg_b_id, display_name=kg_b_id,
        domain=kg_b_domain, nodes=kg_b_nodes, report=kg_b_report,
    )
    return run_enterprise_inference_and_save(ctx_a, ctx_b,
                                             background_tasks=background_tasks,
                                             sample_fn=sample_fn,
                                             main_loop=main_loop)
