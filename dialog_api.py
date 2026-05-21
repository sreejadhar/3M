"""
FastAPI service for the Dialog with Data Agent.

Standalone microservice — completely independent of all other agents.
Accepts a natural language query + KG graph data + DB connection config,
decomposes it into SQL, executes against the target DB, and returns insights.

NLQ caching: completed query results are cached in memory by a fingerprint of
(natural_query, db_type, db_host, db_name, db_schema, kg node labels).  A cache
hit returns instantly without re-running the LLM pipeline.

Endpoints
---------
GET  /health
POST /query                  start async dialog job → {job_id}
GET  /jobs/{job_id}          poll status + summary stats
GET  /jobs/{job_id}/results  get full results (queries + rows + insights)
GET  /list                   list all dialog jobs
GET  /cache                  list all cached NLQ entries
DELETE /cache/{cache_key}    invalidate a specific cache entry
DELETE /cache                clear the entire NLQ cache
"""
from __future__ import annotations

import hashlib
import logging
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

from dialog_agent import DialogAgent, DialogConfig
from dialog_agent.state import ConversationTurn
from dialog_agent.nodes.execute_node import (
    _run_sql,
    list_file_dbs,
    purge_all_file_dbs,
    purge_file_db,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Dialog with Data Agent API", version="1.1.0", docs_url="/docs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory job store ────────────────────────────────────────────────────────
_jobs: Dict[str, Dict] = {}
_lock = threading.Lock()

# ── NLQ cache ─────────────────────────────────────────────────────────────────
# Maps cache_key → {cache_key, natural_query, db_fingerprint, kg_fingerprint,
#                   cached_at, job_id, insights, sql_queries, query_results, errors}
_nlq_cache: Dict[str, Dict] = {}
_cache_lock = threading.Lock()

# ── Session store ──────────────────────────────────────────────────────────────
# Maps session_id → list of ConversationTurn (up to MAX_HISTORY_TURNS, oldest first)
_sessions: Dict[str, List[ConversationTurn]] = {}
_sessions_lock = threading.Lock()

MAX_HISTORY_TURNS = 5

DIALOG_NODES = ["understand", "plan", "execute", "synthesize"]


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _cache_key(
    natural_query: str,
    db_type: str,
    db_host: str,
    db_port: int,
    db_name: str,
    db_schema: str,
    kg_nodes: List[Dict],
) -> str:
    """Stable fingerprint for a (query, db, schema, kg) combination."""
    kg_labels = ",".join(sorted(n.get("label", "") for n in kg_nodes if n.get("label")))
    raw = "|".join([
        natural_query.strip().lower(),
        db_type.lower(),
        db_host.lower(),
        str(db_port),
        db_name.lower(),
        db_schema.lower(),
        kg_labels,
    ])
    return hashlib.sha256(raw.encode()).hexdigest()


def _db_fingerprint(db_type: str, db_host: str, db_port: int, db_name: str, db_schema: str) -> str:
    return f"{db_type}://{db_host}:{db_port}/{db_name}/{db_schema}"


def _kg_fingerprint(kg_nodes: List[Dict]) -> str:
    labels = sorted(n.get("label", "") for n in kg_nodes if n.get("label"))
    return f"{len(labels)} tables: {', '.join(labels[:10])}" + ("…" if len(labels) > 10 else "")


# ── Request models ─────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    natural_query: str

    # KG graph data (optional — from the KG Agent's /jobs/{id}/graph endpoint)
    kg_nodes: List[Dict[str, Any]] = []
    kg_edges: List[Dict[str, Any]] = []

    # Target database connection
    db_type:              str = "postgres"
    db_host:              str = ""
    db_port:              int = 5432
    db_name:              str = ""
    db_schema:            str = "public"
    db_user:              str = ""
    db_password:          str = ""
    db_connection_string: str = ""
    db_extra:             Dict[str, Any] = {}
    db_file_path:         Optional[str] = None   # for SQLite / CSV / Excel

    # Behaviour
    max_sql_queries: int = 10
    row_limit:       int = 500
    llm_model:       str = ""                          # synthesize node — empty = use DialogConfig env default
    plan_llm_model:  str = "claude-haiku-4-5" # plan node (SQL generation — cheaper)

    # User context — used to personalise insight framing
    analyst_role: str = ""   # e.g. "Financial Analyst", "Supply Chain Analyst"

    # Source identity (used to load active KPI definitions for this source)
    source_id: Optional[str] = None

    # Cache control
    skip_cache: bool = False   # Set True to force a fresh run even if cached

    # Session / conversation context
    # Pass the session_id returned from a previous /query call to retain context.
    # Omit (or pass null) to start a new independent session.
    session_id: Optional[str] = None


# ── Background runner ──────────────────────────────────────────────────────────

def _run_dialog(
    job_id: str,
    natural_query: str,
    kg_nodes: List[Dict],
    kg_edges: List[Dict],
    cfg: DialogConfig,
    cache_key: str,
    db_fingerprint: str,
    kg_fingerprint: str,
    session_id: str,
    turn_number: int,
) -> None:
    with _lock:
        _jobs[job_id]["status"] = "running"

    try:
        # Fetch current session history before running
        with _sessions_lock:
            history = list(_sessions.get(session_id, []))

        # ── Multi-KG routing ───────────────────────────────────────────────────
        from dialog_agent.kg_router import route as _kg_route
        from dialog_agent.kg_registry import list_all as _list_kgs, get as _get_kg
        from dialog_agent.kg_bridges import list_for_kgs as _list_bridges

        active_kg_ids: List[str] = []
        multi_kg_configs: List[Dict] = []
        kg_bridges_active: List[Dict] = []

        if cfg.multi_kg_enabled and len(_list_kgs()) > 1:
            if cfg.kg_ids:
                active_kg_ids = list(cfg.kg_ids)
            else:
                active_kg_ids = _kg_route(
                    natural_query,
                    threshold=cfg.kg_router_threshold,
                    llm_model=cfg.plan_llm_model,
                    llm_temperature=cfg.llm_temperature,
                )

            if len(active_kg_ids) > 1:
                # Load bridges between active KGs
                bridges = _list_bridges(active_kg_ids)
                kg_bridges_active = [b.to_dict() for b in bridges]
                logger.info(
                    "Multi-KG routing: active_kg_ids=%s, bridges=%d",
                    active_kg_ids, len(kg_bridges_active),
                )
                # Tag kg_nodes with their kg_id (nodes should have kg_id set by orchestrator)
                # multi_kg_configs left empty — orchestrator source configs not available here
                # without ORCHESTRATOR_API integration; execute_node falls back to default cfg
            else:
                # Only one KG selected — behave as single-KG
                active_kg_ids = []

        agent  = DialogAgent(cfg)
        result = agent.run(
            natural_query, kg_nodes, kg_edges,
            conversation_history=history,
            active_kg_ids=active_kg_ids,
            kg_bridges_active=kg_bridges_active,
            multi_kg_configs=multi_kg_configs,
        )

        insights = result.get("insights", "")
        sql_queries = result.get("sql_queries") or []

        with _lock:
            _jobs[job_id].update({
                "status":          "done",
                "completed_nodes": list(DIALOG_NODES),
                "current_node":    "synthesize",
                "query_count":     len(sql_queries),
                "result_count":    len(result.get("query_results") or []),
                "insights":        insights,
                "sql_queries":     sql_queries,
                "query_results":   result.get("query_results") or [],
                "errors":          result.get("errors") or [],
                "session_id":      session_id,
                # Debug fields — available via /jobs/{job_id}/debug
                "_schema_context": result.get("schema_context", ""),
                "_preflight_gaps": [
                    e for e in (result.get("errors") or [])
                    if "pre-flight" in e.lower()
                ],
            })

        # Append this completed turn to the session history (capped at MAX_HISTORY_TURNS)
        tables_queried = list({
            ref
            for q in sql_queries
            for ref in (q.get("table_refs") or [])
        })

        # Build per-query diagnostics so the planner can learn from failures
        # on the NEXT turn rather than starting fresh.
        query_results_list = result.get("query_results") or []
        results_by_id = {qr["query_id"]: qr for qr in query_results_list}
        preflight_errors = [
            e for e in (result.get("errors") or [])
            if "pre-flight" in e.lower()
        ]
        query_diagnostics = []
        for q in sql_queries:
            qid = q.get("query_id", "")
            qr  = results_by_id.get(qid, {})
            query_diagnostics.append({
                "query_id":  qid,
                "sql":       q.get("sql", "")[:800],   # cap to keep history compact
                "row_count": qr.get("row_count", 0),
                "columns":   qr.get("columns") or [],
                "error":     qr.get("error"),
                "preflight_gaps": preflight_errors if qid == sql_queries[0].get("query_id") else [],
            })

        new_turn: ConversationTurn = {
            "turn": turn_number,
            "question": natural_query,
            "insights": insights[:600],   # keep prompt size manageable
            "tables_queried": tables_queried,
            "query_diagnostics": query_diagnostics,
        }
        with _sessions_lock:
            turns = _sessions.setdefault(session_id, [])
            turns.append(new_turn)
            # Keep only the last MAX_HISTORY_TURNS turns
            if len(turns) > MAX_HISTORY_TURNS:
                _sessions[session_id] = turns[-MAX_HISTORY_TURNS:]
        logger.info(
            "Session %s: turn %d added (%d turns in history)",
            session_id[:8], turn_number, len(_sessions[session_id]),
        )

        # Only cache successful results — never cache empty query_results
        # (a failed run would poison the cache and repeat the error forever).
        if result.get("query_results"):
            with _cache_lock:
                _nlq_cache[cache_key] = {
                    "cache_key":      cache_key,
                    "natural_query":  natural_query,
                    "db_fingerprint": db_fingerprint,
                    "kg_fingerprint": kg_fingerprint,
                    "cached_at":      datetime.now(timezone.utc).isoformat(),
                    "job_id":         job_id,
                    "insights":       result.get("insights", ""),
                    "sql_queries":    result.get("sql_queries") or [],
                    "query_results":  result.get("query_results") or [],
                    "errors":         result.get("errors") or [],
                }
                logger.info("NLQ cached: key=%s", cache_key[:12])
        else:
            logger.info(
                "NLQ result NOT cached (empty query_results): key=%s errors=%s",
                cache_key[:12], result.get("errors") or [],
            )

    except Exception as exc:
        logger.exception("Dialog job %s failed", job_id)
        with _lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"]  = str(exc)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", status_code=202)
def start_query(req: QueryRequest, background_tasks: BackgroundTasks):
    if not req.natural_query.strip():
        raise HTTPException(status_code=400, detail="natural_query must not be empty")

    _FILE_BASED = {"sqlite", "csv", "excel"}
    cfg = DialogConfig(
        db_type              = req.db_type,
        db_host              = req.db_host,
        db_port              = req.db_port,
        db_name              = req.db_name,
        # File-based sources have no schema; force empty so planner never qualifies names
        db_schema            = "" if req.db_type.lower() in _FILE_BASED else req.db_schema,
        db_user              = req.db_user,
        db_password          = req.db_password,
        db_connection_string = req.db_connection_string,
        db_extra             = req.db_extra,
        db_file_path         = req.db_file_path or "",
        source_id            = req.source_id or "",
        # Only override llm_model when the caller explicitly picks one;
        # otherwise DialogConfig uses its env-driven default (DIALOG_ENV).
        **({"llm_model": req.llm_model} if req.llm_model else {}),
        plan_llm_model       = req.plan_llm_model,
        max_sql_queries      = req.max_sql_queries,
        row_limit            = req.row_limit,
        analyst_role         = req.analyst_role or "",
    )

    # ── Session resolution ─────────────────────────────────────────────────────
    # Use provided session_id or create a new one
    session_id = req.session_id or str(uuid.uuid4())
    with _sessions_lock:
        turn_number = len(_sessions.get(session_id, [])) + 1

    ck = _cache_key(
        req.natural_query, req.db_type, req.db_host,
        req.db_port, req.db_name, req.db_schema, req.kg_nodes,
    )
    dbfp = _db_fingerprint(req.db_type, req.db_host, req.db_port, req.db_name, req.db_schema)
    kgfp = _kg_fingerprint(req.kg_nodes)

    # ── Cache hit: synthesize a completed job from cached data ─────────────────
    if not req.skip_cache:
        with _cache_lock:
            cached = _nlq_cache.get(ck)
        if cached:
            logger.info("NLQ cache HIT: key=%s", ck[:12])
            job_id = str(uuid.uuid4())
            with _lock:
                _jobs[job_id] = {
                    "id":              job_id,
                    "natural_query":   req.natural_query,
                    "db_type":         req.db_type,
                    "status":          "done",
                    "current_node":    "synthesize",
                    "completed_nodes": list(DIALOG_NODES),
                    "query_count":     len(cached.get("sql_queries") or []),
                    "result_count":    len(cached.get("query_results") or []),
                    "insights":        cached.get("insights", ""),
                    "sql_queries":     cached.get("sql_queries") or [],
                    "query_results":   cached.get("query_results") or [],
                    "errors":          cached.get("errors") or [],
                    "error":           None,
                    "cache_hit":       True,
                    "cache_key":       ck,
                }
            return {"job_id": job_id, "cache_hit": True, "session_id": session_id}

    # ── Cache miss: run the full pipeline ─────────────────────────────────────
    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = {
            "id":              job_id,
            "natural_query":   req.natural_query,
            "db_type":         req.db_type,
            "status":          "queued",
            "current_node":    None,
            "completed_nodes": [],
            "query_count":     0,
            "result_count":    0,
            "insights":        "",
            "sql_queries":     [],
            "query_results":   [],
            "errors":          [],
            "error":           None,
            "cache_hit":       False,
            "cache_key":       ck,
            "session_id":      session_id,
        }

    background_tasks.add_task(
        _run_dialog, job_id,
        req.natural_query, req.kg_nodes, req.kg_edges,
        cfg, ck, dbfp, kgfp,
        session_id, turn_number,
    )
    return {"job_id": job_id, "cache_hit": False, "session_id": session_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {k: v for k, v in job.items()
            if k not in ("sql_queries", "query_results")}


@app.get("/jobs/{job_id}/results")
def get_results(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=202, detail="Results not yet ready")
    return {
        "natural_query":  job["natural_query"],
        "insights":       job["insights"],
        "sql_queries":    job["sql_queries"],
        "query_results":  job["query_results"],
        "errors":         job["errors"],
        "cache_hit":      job.get("cache_hit", False),
    }


@app.get("/jobs/{job_id}/debug")
def get_debug(job_id: str):
    """
    Return diagnostic information about a completed job:
    - The full schema_context the LLM received (after grain annotation)
    - The SQL queries that were generated
    - Row counts and column lists from each query result
    - All errors and pre-flight gaps
    Use this to diagnose why generated SQL is not returning expected data.
    """
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=202, detail="Job not yet complete")

    sql_queries   = job.get("sql_queries") or []
    query_results = job.get("query_results") or []
    results_by_id = {qr["query_id"]: qr for qr in query_results}

    query_debug = []
    for q in sql_queries:
        qid = q.get("query_id", "?")
        qr  = results_by_id.get(qid, {})
        query_debug.append({
            "query_id":    qid,
            "description": q.get("description", ""),
            "sql":         q.get("sql", ""),
            "row_count":   qr.get("row_count", 0),
            "columns":     qr.get("columns") or [],
            "sample_rows": (qr.get("rows") or [])[:3],
            "error":       qr.get("error"),
        })

    return {
        "job_id":          job_id,
        "natural_query":   job.get("natural_query", ""),
        "schema_context":  job.get("_schema_context", ""),
        "preflight_gaps":  job.get("_preflight_gaps", []),
        "errors":          job.get("errors") or [],
        "query_debug":     query_debug,
    }


@app.get("/list")
def list_jobs():
    with _lock:
        jobs = list(_jobs.values())
    return [
        {k: v for k, v in j.items() if k not in ("sql_queries", "query_results")}
        for j in jobs
    ]


# ── NLQ Cache endpoints ────────────────────────────────────────────────────────

@app.get("/cache")
def list_cache():
    """List all cached NLQ entries (without the full result blobs)."""
    with _cache_lock:
        entries = list(_nlq_cache.values())
    return [
        {k: v for k, v in e.items() if k not in ("sql_queries", "query_results")}
        for e in entries
    ]


@app.delete("/cache/{cache_key}", status_code=200)
def delete_cache_entry(cache_key: str):
    """Invalidate a single cached NLQ entry by its cache key."""
    with _cache_lock:
        if cache_key not in _nlq_cache:
            raise HTTPException(status_code=404, detail="Cache entry not found")
        del _nlq_cache[cache_key]
    logger.info("NLQ cache entry deleted: key=%s", cache_key[:12])
    return {"deleted": cache_key}


@app.delete("/cache", status_code=200)
def clear_cache():
    """Clear the entire NLQ cache."""
    with _cache_lock:
        count = len(_nlq_cache)
        _nlq_cache.clear()
    logger.info("NLQ cache cleared: %d entries removed", count)
    return {"deleted_count": count}


# ── File-DB cache endpoints ────────────────────────────────────────────────────
# Call DELETE /file-cache when the user navigates away from the Dialog with Data
# section to release the temp SQLite files that were built for CSV/Excel sources.

# ── Raw SQL execution (tech UI SQL console) ───────────────────────────────────

class ExecuteSQLRequest(BaseModel):
    sql:                  str
    db_type:              str = "postgres"
    db_host:              str = ""
    db_port:              int = 5432
    db_name:              str = ""
    db_schema:            str = "public"
    db_user:              str = ""
    db_password:          str = ""
    db_connection_string: str = ""
    db_extra:             Dict[str, Any] = {}
    db_file_path:         Optional[str] = None
    limit:                int = 500


@app.post("/execute-sql")
def execute_sql_direct(req: ExecuteSQLRequest):
    """
    Execute a raw SQL statement against a database and return rows.
    Used by the tech UI SQL console — no LLM involved.
    """
    if not req.sql.strip():
        raise HTTPException(status_code=400, detail="sql must not be empty")

    _FILE_BASED = {"sqlite", "csv", "excel"}
    cfg = DialogConfig(
        db_type              = req.db_type,
        db_host              = req.db_host,
        db_port              = req.db_port,
        db_name              = req.db_name,
        db_schema            = "" if req.db_type.lower() in _FILE_BASED else req.db_schema,
        db_user              = req.db_user,
        db_password          = req.db_password,
        db_connection_string = req.db_connection_string,
        db_extra             = req.db_extra,
        db_file_path         = req.db_file_path or "",
    )

    import time as _time
    t0 = _time.time()
    result = _run_sql(cfg, req.sql)
    elapsed_ms = int((_time.time() - t0) * 1000)

    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])

    columns = result.get("columns") or []
    raw_rows = (result.get("rows") or [])[:req.limit]
    rows = [dict(zip(columns, r)) for r in raw_rows]

    return {
        "columns":    columns,
        "rows":       rows,
        "row_count":  len(raw_rows),
        "elapsed_ms": elapsed_ms,
    }


@app.get("/file-cache")
def list_file_cache():
    """List all cached temp SQLite DBs (original path → temp file path)."""
    return list_file_dbs()


@app.delete("/file-cache", status_code=200)
def clear_file_cache():
    """
    Delete all cached temp SQLite files built from CSV/Excel sources.
    Call this when the user leaves the Dialog with Data section.
    """
    count = purge_all_file_dbs()
    logger.info("file_db_cache cleared: %d temp DB(s) removed", count)
    return {"deleted_count": count}


# ── Session endpoints ──────────────────────────────────────────────────────────

@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    """Return the conversation history for a session."""
    with _sessions_lock:
        turns = _sessions.get(session_id)
    if turns is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "turns": turns, "turn_count": len(turns)}


@app.delete("/sessions/{session_id}", status_code=200)
def clear_session(session_id: str):
    """Clear the conversation history for a session (start fresh)."""
    with _sessions_lock:
        existed = session_id in _sessions
        _sessions.pop(session_id, None)
    if not existed:
        raise HTTPException(status_code=404, detail="Session not found")
    logger.info("Session %s cleared", session_id[:8])
    return {"deleted": session_id}


# ── KG Registry endpoints ──────────────────────────────────────────────────────

@app.get("/kg-registry")
def list_kg_registry():
    """List all registered Knowledge Graphs."""
    from dialog_agent.kg_registry import list_all as _list_kgs
    return [
        {
            "kg_id":           e.kg_id,
            "display_name":    e.display_name,
            "description":     e.description,
            "domain_keywords": e.domain_keywords,
            "entity_types":    e.entity_types,
            "source_id":       e.source_id,
            "source_db_type":  e.source_db_type,
            "created_at":      e.created_at,
            "updated_at":      e.updated_at,
        }
        for e in _list_kgs()
    ]


@app.get("/kg-bridges")
def list_kg_bridges(enabled_only: bool = False):
    """List all cross-KG bridges."""
    from dialog_agent.kg_bridges import list_all as _list_bridges
    return [b.to_dict() for b in _list_bridges(enabled_only=enabled_only)]


class BridgeCreateRequest(BaseModel):
    from_kg:     str
    from_column: str
    to_kg:       str
    to_column:   str
    from_entity: str   = ""
    to_entity:   str   = ""
    join_type:   str   = "FK"
    notes:       str   = ""


@app.post("/kg-bridges", status_code=201)
def create_kg_bridge(req: BridgeCreateRequest):
    """Create a declared bridge between two KGs."""
    from dialog_agent.kg_bridges import Bridge, upsert as _upsert_bridge
    b = Bridge(
        from_kg=req.from_kg, from_column=req.from_column,
        to_kg=req.to_kg,     to_column=req.to_column,
        from_entity=req.from_entity, to_entity=req.to_entity,
        join_type=req.join_type, notes=req.notes,
        source="declared", confidence=1.0, enabled=True,
    )
    bridge_id = _upsert_bridge(b)
    return {"bridge_id": bridge_id}


class BridgeUpdateRequest(BaseModel):
    enabled:  Optional[bool] = None
    notes:    Optional[str]  = None
    promote:  bool           = False   # promote inferred → declared


@app.put("/kg-bridges/{bridge_id}")
def update_kg_bridge(bridge_id: int, req: BridgeUpdateRequest):
    """Update a bridge: enable/disable, update notes, or promote to declared."""
    from dialog_agent.kg_bridges import (
        get_by_id as _get_bridge,
        set_enabled as _set_enabled,
        promote_to_declared as _promote,
    )
    import dialog_agent.kg_bridges as _kb
    b = _get_bridge(bridge_id)
    if not b:
        raise HTTPException(status_code=404, detail="Bridge not found")
    if req.enabled is not None:
        _set_enabled(bridge_id, req.enabled)
    if req.notes is not None:
        import sqlite3, time as _time
        with _kb._conn() as c:
            c.execute("UPDATE kg_bridges SET notes=?,updated_at=? WHERE id=?",
                      (req.notes, _time.time(), bridge_id))
    if req.promote:
        _promote(bridge_id)
    updated = _get_bridge(bridge_id)
    return updated.to_dict() if updated else {}


@app.delete("/kg-bridges/{bridge_id}", status_code=200)
def delete_kg_bridge(bridge_id: int):
    """Delete a bridge."""
    from dialog_agent.kg_bridges import get_by_id as _get_bridge, delete as _del_bridge
    b = _get_bridge(bridge_id)
    if not b:
        raise HTTPException(status_code=404, detail="Bridge not found")
    _del_bridge(bridge_id)
    return {"deleted": bridge_id}


# ── File-DB cache endpoints ────────────────────────────────────────────────────

@app.delete("/file-cache/{file_path:path}", status_code=200)
def delete_file_cache_entry(file_path: str):
    """Delete the cached temp SQLite DB for a specific source file path."""
    removed = purge_file_db(file_path)
    if not removed:
        raise HTTPException(status_code=404, detail="No cached DB for that path")
    return {"deleted": file_path}
