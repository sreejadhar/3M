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
POST /query                           cross-modal query (question + kpi_names)

Google Drive auth (OAuth2)
--------------------------
GET  /auth/google/url                 generate OAuth2 consent URL
POST /auth/google/token               exchange auth code → save token to disk

Environment variables
---------------------
UNSTRUCTURED_DB      path to unstructured.db   (default: data/unstructured.db)
METADATA_API_URL     upstream metadata service  (default: http://localhost:8000)
ANTHROPIC_API_KEY    required for LLM extraction
UNSTRUCTURED_PORT    port to bind               (default: 8008)
UNSTRUCTURED_WORKERS parallel indexing threads  (default: 4)
GDRIVE_TOKEN_DIR     directory to store OAuth2 tokens (default: data/gdrive_tokens)
GDRIVE_CLIENT_SECRETS path to OAuth2 client secrets JSON from Google Cloud Console
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

_DB_PATH          = os.environ.get("UNSTRUCTURED_DB",        "data/unstructured.db")
_METADATA_API     = os.environ.get("METADATA_API_URL",       "http://localhost:8000")
_PORT             = int(os.environ.get("UNSTRUCTURED_PORT",  "8008"))
_GDRIVE_TOKEN_DIR = os.environ.get("GDRIVE_TOKEN_DIR",       "data/gdrive_tokens")
_GDRIVE_SECRETS   = os.environ.get("GDRIVE_CLIENT_SECRETS",  "")

Path(_GDRIVE_TOKEN_DIR).mkdir(parents=True, exist_ok=True)

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


# ── Google Drive OAuth2 ───────────────────────────────────────────────────────

_GDRIVE_SCOPES    = ["https://www.googleapis.com/auth/drive.readonly"]
_GDRIVE_AUTH_URI  = "https://accounts.google.com/o/oauth2/auth"
_GDRIVE_TOKEN_URI = "https://oauth2.googleapis.com/token"


class GoogleTokenRequest(BaseModel):
    auth_code:     str
    redirect_uri:  str = "urn:ietf:wg:oauth:2.0:oob"
    token_name:    str = "default"   # logical name — token saved as {token_name}.json


@app.get("/auth/google/url")
def gdrive_auth_url(
    redirect_uri: str = "urn:ietf:wg:oauth:2.0:oob",
    token_name: str = "default",
):
    """
    Generate a Google OAuth2 consent URL.

    Steps:
      1. GET /auth/google/url  → copy the 'url' from the response
      2. Open the URL in a browser, sign in, grant read-only Drive access
      3. Copy the authorisation code Google shows
      4. POST /auth/google/token {auth_code, redirect_uri, token_name}
      5. Register a source with source_type='gdrive',
         connection.token_path = the path returned by step 4
    """
    if not _GDRIVE_SECRETS:
        raise HTTPException(
            400,
            "GDRIVE_CLIENT_SECRETS env var not set. "
            "Download OAuth2 client secrets JSON from Google Cloud Console and "
            "set GDRIVE_CLIENT_SECRETS=/path/to/client_secrets.json"
        )
    try:
        from google_auth_oauthlib.flow import Flow  # type: ignore
    except ImportError:
        raise HTTPException(500, "google-auth-oauthlib not installed")

    flow = Flow.from_client_secrets_file(
        _GDRIVE_SECRETS,
        scopes=_GDRIVE_SCOPES,
        redirect_uri=redirect_uri,
    )
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    token_path = str(Path(_GDRIVE_TOKEN_DIR) / f"{token_name}.json")
    return {
        "url":          auth_url,
        "instructions": (
            "1. Open the URL in a browser and sign in with your Google account. "
            "2. Grant 'View files from Google Drive' access. "
            "3. Copy the authorisation code shown. "
            "4. POST /auth/google/token with {auth_code, redirect_uri, token_name}."
        ),
        "token_will_be_saved_to": token_path,
    }


@app.post("/auth/google/token")
def gdrive_exchange_token(req: GoogleTokenRequest):
    """
    Exchange an OAuth2 authorisation code for a refresh token and save it.
    After this call, register a source with:
      source_type = 'gdrive'
      connection  = { 'folder_id': '...', 'token_path': '<returned path>' }
    """
    if not _GDRIVE_SECRETS:
        raise HTTPException(400, "GDRIVE_CLIENT_SECRETS env var not set")
    try:
        from google_auth_oauthlib.flow import Flow  # type: ignore
    except ImportError:
        raise HTTPException(500, "google-auth-oauthlib not installed")

    try:
        flow = Flow.from_client_secrets_file(
            _GDRIVE_SECRETS,
            scopes=_GDRIVE_SCOPES,
            redirect_uri=req.redirect_uri,
        )
        flow.fetch_token(code=req.auth_code)
        creds = flow.credentials
    except Exception as exc:
        raise HTTPException(400, f"Token exchange failed: {exc}")

    token_path = str(Path(_GDRIVE_TOKEN_DIR) / f"{req.token_name}.json")
    import json
    with open(token_path, "w") as f:
        json.dump({
            "token":         creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri":     creds.token_uri,
            "client_id":     creds.client_id,
            "client_secret": creds.client_secret,
            "scopes":        list(creds.scopes or _GDRIVE_SCOPES),
        }, f)

    return {
        "status":     "token saved",
        "token_path": token_path,
        "next_step":  (
            f"Register a source: POST /sources with "
            f"source_type='gdrive', "
            f"connection={{folder_id:'<your folder id>',token_path:'{token_path}'}}"
        ),
    }


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("unstructured_api:app", host="0.0.0.0", port=_PORT,
                reload=False, log_level="info")
