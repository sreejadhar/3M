"""
Document Intelligence service — connects to document repositories (local
filesystem, S3, Google Drive, SharePoint, OneDrive), enumerates and indexes
their files, and exposes sources/assets over HTTP.

Runs standalone on port 8008. The orchestrator (orchestrator_api.py) proxies
/unstructured/* to this service so the frontend never talks to it directly.
"""
from __future__ import annotations

import logging
import mimetypes
import os
import shutil
import string
import threading
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from unstructured_agent.connectors import CONNECTOR_TYPES, ConnectorError
from unstructured_agent.extractor import SUPPORTED_EXTENSIONS
from unstructured_agent.pii import redact_text
from unstructured_agent.pipeline import process_uploaded_asset, run_index_job
from unstructured_agent.store import DocStore

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "info").upper())
logger = logging.getLogger("unstructured_api")

DATA_DIR = os.environ.get("DATA_DIR", str(Path(__file__).parent / "data"))
DB_PATH = os.environ.get("UNSTRUCTURED_DB", str(Path(DATA_DIR) / "unstructured.db"))
ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_API_URL", "http://localhost:8005")

app = FastAPI(title="DataNanite Document Intelligence", docs_url="/docs", redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

store = DocStore(DB_PATH)
UPLOAD_DIR = Path(DATA_DIR) / "uploads"


@app.get("/health")
async def health():
    return {"status": "ok", "service": "unstructured-api"}


class CreateSourceRequest(BaseModel):
    name: str
    connector_type: str
    config: dict = {}


@app.get("/connectors")
async def list_connectors():
    return {"connectors": list(CONNECTOR_TYPES)}


@app.get("/fs/browse")
async def browse_filesystem(path: str = ""):
    """Lists subdirectories under `path` so the frontend can offer a folder
    picker for the local connector. Browsers can't expose absolute OS paths
    from a native file dialog for security reasons, so this walks the
    server's filesystem instead — safe here since the local connector can
    already read any path the caller supplies; this only aids discovery.
    Empty path lists drive roots (Windows) or '/' (POSIX)."""
    if not path:
        if os.name == "nt":
            roots = [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
            return {"path": "", "parent": None,
                    "entries": [{"name": r, "path": r, "is_dir": True} for r in roots]}
        path = "/"

    p = Path(path)
    if not p.exists() or not p.is_dir():
        raise HTTPException(400, f"not a directory: {path}")

    entries = []
    try:
        for child in sorted(p.iterdir(), key=lambda x: x.name.lower()):
            if child.name.startswith("."):
                continue
            try:
                if child.is_dir():
                    entries.append({"name": child.name, "path": str(child), "is_dir": True})
            except OSError:
                continue
    except PermissionError:
        raise HTTPException(403, f"permission denied: {path}")

    parent = str(p.parent) if p.parent != p else None
    return {"path": str(p), "parent": parent, "entries": entries}


@app.post("/sources")
async def create_source(body: CreateSourceRequest):
    if body.connector_type not in CONNECTOR_TYPES:
        raise HTTPException(400, f"unknown connector_type: {body.connector_type!r}")
    if not body.name.strip():
        raise HTTPException(400, "name is required")
    return store.create_source(body.name.strip(), body.connector_type, body.config)


@app.get("/sources")
async def list_sources():
    return store.list_sources()


@app.get("/sources/{source_id}")
async def get_source(source_id: str):
    source = store.get_source(source_id)
    if not source:
        raise HTTPException(404, "source not found")
    return source


@app.delete("/sources/{source_id}")
async def delete_source(source_id: str):
    if not store.get_source(source_id):
        raise HTTPException(404, "source not found")

    # Best-effort: strip each asset's node/edges from every structured
    # datasource's KG it was ever linked into (via cross-modal linking),
    # so deleting a document source here doesn't leave "ghost" document
    # nodes behind in Graph Explorer / DataChat's document context. Must
    # happen BEFORE store.delete_source() — that call wipes xref_links,
    # which is the only record of which KGs each asset was pushed to.
    for asset in store.list_assets(source_id, limit=10_000):
        kg_targets = {link["source_id"] for link in (asset.get("xref_links") or []) if link.get("source_id")}
        for kg_source_id in kg_targets:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    await client.delete(
                        f"{ORCHESTRATOR_URL}/sources/{kg_source_id}/graph/documents/{asset['asset_id']}"
                    )
            except Exception as exc:
                logger.warning("delete_source: could not unlink asset %s from KG %s — %s",
                                asset["asset_id"], kg_source_id, exc)

    store.delete_source(source_id)

    # Remove any files copied here by the Upload Document endpoint (dest_dir
    # in upload_document() below is always UPLOAD_DIR/source_id). Safe for
    # every connector type: "local" sources index files in place at their
    # own root_path and never copy them here; cloud connectors download to
    # a temp file that's removed immediately after processing (see
    # pipeline._maybe_autoprocess) — only uploaded files ever live here.
    upload_dir = UPLOAD_DIR / source_id
    if upload_dir.exists():
        try:
            shutil.rmtree(upload_dir)
        except Exception as exc:
            logger.warning("delete_source: could not remove upload dir %s — %s", upload_dir, exc)

    return {"deleted": True}


@app.post("/sources/{source_id}/index")
async def start_index(source_id: str):
    source = store.get_source(source_id)
    if not source:
        raise HTTPException(404, "source not found")
    if source["status"] == "indexing":
        raise HTTPException(409, "indexing already in progress")

    job = store.create_job(source_id)
    thread = threading.Thread(target=run_index_job, args=(job["job_id"], source_id, store), daemon=True)
    thread.start()
    return job


@app.get("/sources/{source_id}/jobs")
async def list_jobs(source_id: str):
    if not store.get_source(source_id):
        raise HTTPException(404, "source not found")
    return store.list_jobs(source_id)


@app.get("/sources/{source_id}/assets")
async def list_assets(source_id: str, limit: int = 500):
    if not store.get_source(source_id):
        raise HTTPException(404, "source not found")
    return store.list_assets(source_id, limit=limit)


@app.get("/assets/{asset_id}")
async def get_asset(asset_id: str):
    asset = store.get_asset(asset_id)
    if not asset:
        raise HTTPException(404, "asset not found")
    return asset


@app.get("/assets/{asset_id}/excerpt")
async def get_asset_excerpt(asset_id: str, max_chars: int = 1200):
    """Returns a short, PII-redacted excerpt of this document's extracted
    text, plus its topics and cross-modal links — built for DataChat to
    quote from when answering a question about a linked datasource.
    Redaction re-runs detect_pii's regex patterns directly against the raw
    text (see pii.redact_text) rather than reusing the stored pii_findings,
    which intentionally never retain raw values or positions."""
    asset = store.get_asset(asset_id)
    if not asset:
        raise HTTPException(404, "asset not found")
    text = asset.get("extracted_text") or ""
    excerpt = redact_text(text)[:max_chars]
    return {
        "asset_id": asset_id,
        "file_name": asset.get("file_name"),
        "excerpt": excerpt,
        "topics": asset.get("topics") or [],
        "xref_links": asset.get("xref_links") or [],
    }


def _run_upload_pipeline(asset_id: str, dest_path: str, store: DocStore, source_id: str) -> None:
    """Runs the per-asset processing pipeline, then flips the source's status
    back to 'ready' only once every asset on it has finished — so the source
    list's status doesn't say "Ready" while files are still processing, and
    correctly shows "Indexing…" for the whole time any upload on it is still
    running (including other files uploaded concurrently)."""
    process_uploaded_asset(asset_id, dest_path, store)
    still_running = any(
        a.get("processing_status") == "running" for a in store.list_assets(source_id, limit=10_000)
    )
    if not still_running:
        store.set_source_status(source_id, "ready")


@app.post("/sources/{source_id}/upload")
async def upload_document(source_id: str, file: UploadFile = File(...)):
    """Uploads a document to a source and runs it through the processing
    pipeline: text extraction → topic tagging → named entity recognition →
    PII detection → semantic embeddings → cross-modal linking."""
    source = store.get_source(source_id)
    if not source:
        raise HTTPException(404, "source not found")

    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"unsupported file type: {ext or '(none)'} — "
                                  f"supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

    dest_dir = UPLOAD_DIR / source_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{uuid.uuid4()}_{file.filename}"
    with open(dest_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    size_bytes = dest_path.stat().st_size
    mime_type = file.content_type or mimetypes.guess_type(file.filename)[0] or "application/octet-stream"
    asset_id = store.upsert_asset(
        source_id=source_id, remote_id=str(dest_path), file_name=file.filename,
        size_bytes=size_bytes, mime_type=mime_type, checksum=None, modified_at=None,
        local_path=str(dest_path),
    )

    store.set_source_status(source_id, "indexing")
    thread = threading.Thread(
        target=_run_upload_pipeline, args=(asset_id, str(dest_path), store, source_id), daemon=True,
    )
    thread.start()
    return store.get_asset(asset_id)


@app.exception_handler(ConnectorError)
async def connector_error_handler(request, exc: ConnectorError):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception):
    """Without this, FastAPI's default 500 response body is just
    {"detail": "Internal Server Error"} — the real exception only shows up
    in the pod's stdout, which isn't reachable without cluster access. This
    logs the full traceback (still visible via `kubectl logs` when someone
    has access) *and* returns the actual message to the caller, so the
    toast the user sees is actually actionable."""
    from fastapi.responses import JSONResponse
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})
