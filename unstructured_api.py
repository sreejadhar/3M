"""
FastAPI service for the Unstructured Data Intelligence Agent — port 8008

Indexes enterprise documents (PDF, DOCX, PPTX, XLSX, HTML, MD, TXT),
extracts semantic fingerprints, and builds cross-modal links to the
structured data model already known to DataNanite.

Raw document text is NEVER persisted. Only semantic metadata is stored.

Endpoints
---------
GET  /health
POST /sources                         register a new document source
GET  /sources                         list all sources
GET  /sources/{id}                    get source detail
POST /sources/{id}/index              trigger async indexing run
GET  /sources/{id}/jobs               list indexing jobs for source

GET  /jobs/{job_id}                   get job status
GET  /assets                          list all indexed assets  (?source_id, ?enriched_only)
GET  /assets/{id}                     get asset detail + topics/entities
GET  /assets/{id}/links               get cross-modal relationships for asset
GET  /search                          full-text search over title+summary+topics (?q)

Environment variables
---------------------
UNSTRUCTURED_DB      path to unstructured.db   (default: data/unstructured.db)
METADATA_API_URL     upstream metadata service  (default: http://localhost:8000)
ANTHROPIC_API_KEY    required for LLM extraction
UNSTRUCTURED_PORT    port to bind               (default: 8008)
UNSTRUCTURED_WORKERS parallel indexing threads  (default: 4)
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

from unstructured_agent.store import UnstructuredStore
from unstructured_agent.pipeline import run_index_job
from unstructured_agent.query import query_for_context

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_DB_PATH     = os.environ.get("UNSTRUCTURED_DB", "data/unstructured.db")
_METADATA_API = os.environ.get("METADATA_API_URL", "http://localhost:8000")
_PORT        = int(os.environ.get("UNSTRUCTURED_PORT", "8008"))

store = UnstructuredStore(_DB_PATH)

app = FastAPI(
    title="DataNanite Unstructured Intelligence API",
    version="1.0.0",
    description=(
        "Semantic fingerprinting and cross-modal linking for enterprise documents. "
        "Raw document text is never stored — only metadata."
    ),
    docs_url="/docs",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class CreateSourceRequest(BaseModel):
    name: str
    source_type: str = "local"         # local | s3 | gcs | azure
    connection: Dict[str, Any] = {}    # source_type-specific config
    nanite_source_id: Optional[str] = None
    domain: str = ""


class QueryRequest(BaseModel):
    question: str
    kpi_names: List[str] = []          # KPI names extracted from the structured answer


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "unstructured-agent", "port": _PORT}


# ── Sources ───────────────────────────────────────────────────────────────────

@app.post("/sources", status_code=201)
def create_source(req: CreateSourceRequest):
    source = store.create_source(
        name=req.name,
        source_type=req.source_type,
        connection=req.connection,
        nanite_source_id=req.nanite_source_id,
        domain=req.domain,
    )
    return source


@app.get("/sources")
def list_sources():
    return store.list_sources()


@app.get("/sources/{source_id}")
def get_source(source_id: str):
    src = store.get_source(source_id)
    if not src:
        raise HTTPException(404, f"Source {source_id!r} not found")
    return src


# ── Indexing ──────────────────────────────────────────────────────────────────

@app.post("/sources/{source_id}/index", status_code=202)
def start_index(source_id: str, background_tasks: BackgroundTasks):
    src = store.get_source(source_id)
    if not src:
        raise HTTPException(404, f"Source {source_id!r} not found")

    job_id = store.create_job(source_id)
    background_tasks.add_task(
        run_index_job, job_id, source_id, store, _METADATA_API
    )
    return {"job_id": job_id, "status": "running", "source_id": source_id}


@app.get("/sources/{source_id}/jobs")
def list_source_jobs(source_id: str):
    return store.list_jobs(source_id)


# ── Jobs ──────────────────────────────────────────────────────────────────────

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id!r} not found")
    return job


@app.get("/jobs")
def list_all_jobs():
    return store.list_jobs()


# ── Assets ────────────────────────────────────────────────────────────────────

@app.get("/assets")
def list_assets(
    source_id: Optional[str] = Query(None),
    enriched_only: bool = Query(False),
    limit: int = Query(100, le=500),
):
    return store.list_assets(source_id=source_id, enriched_only=enriched_only, limit=limit)


@app.get("/assets/{asset_id}")
def get_asset(asset_id: str):
    asset = store.get_asset(asset_id)
    if not asset:
        raise HTTPException(404, f"Asset {asset_id!r} not found")
    return asset


@app.get("/assets/{asset_id}/links")
def get_asset_links(asset_id: str):
    asset = store.get_asset(asset_id)
    if not asset:
        raise HTTPException(404, f"Asset {asset_id!r} not found")
    return {
        "asset_id": asset_id,
        "title":    asset.get("title", ""),
        "links":    store.get_relationships(asset_id),
    }


# ── Cross-modal query ─────────────────────────────────────────────────────────

@app.post("/query")
def cross_modal_query(req: QueryRequest):
    """
    Given a natural-language question and optional KPI names from a structured
    answer, return the most relevant document context as a formatted markdown
    block. Used by synthesize_node to enrich structured insights with document
    evidence.

    Returns {"doc_context": "<markdown>", "doc_count": N} or
            {"doc_context": null, "doc_count": 0} when nothing relevant found.
    """
    if not req.question.strip():
        raise HTTPException(400, "question must not be empty")
    try:
        ctx = query_for_context(
            question=req.question,
            kpi_names=req.kpi_names,
            store=store,
        )
    except Exception as exc:
        logger.warning("cross_modal_query failed: %s", exc)
        ctx = None

    # Count how many doc blocks are in the context (each ends with a blank line)
    doc_count = ctx.count("\n\n") + 1 if ctx else 0
    return {"doc_context": ctx, "doc_count": doc_count}


# ── Search ────────────────────────────────────────────────────────────────────

@app.get("/search")
def search(q: str = Query(..., min_length=2), limit: int = Query(20, le=100)):
    try:
        results = store.search_assets(q, limit=limit)
    except Exception as exc:
        logger.warning("FTS search failed: %s", exc)
        results = []
    return {"query": q, "results": results, "count": len(results)}


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("unstructured_api:app", host="0.0.0.0", port=_PORT,
                reload=False, log_level="info")
