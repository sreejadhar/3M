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
    llm_model:       str = "claude-sonnet-4-6"        # synthesize node (insight quality)
    plan_llm_model:  str = "claude-haiku-4-5-20251001" # plan node (SQL generation — cheaper)

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

        agent  = DialogAgent(cfg)
        result = agent.run(natural_query, kg_nodes, kg_edges, conversation_history=history)

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
            })

        # Append this completed turn to the session history (capped at MAX_HISTORY_TURNS)
        tables_queried = list({
            ref
            for q in sql_queries
            for ref in (q.get("table_refs") or [])
        })
        new_turn: ConversationTurn = {
            "turn": turn_number,
            "question": natural_query,
            "insights": insights[:600],   # keep prompt size manageable
            "tables_queried": tables_queried,
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
        llm_model            = req.llm_model,
        plan_llm_model       = req.plan_llm_model,
        max_sql_queries      = req.max_sql_queries,
        row_limit            = req.row_limit,
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


@app.delete("/file-cache/{file_path:path}", status_code=200)
def delete_file_cache_entry(file_path: str):
    """Delete the cached temp SQLite DB for a specific source file path."""
    removed = purge_file_db(file_path)
    if not removed:
        raise HTTPException(status_code=404, detail="No cached DB for that path")
    return {"deleted": file_path}
