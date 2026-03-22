"""
DataChat Orchestrator API — port 8005

Serves the web chat UI and manages the end-to-end pipeline:
  file upload → metadata extraction → ontology → knowledge graph → dialog

Real-time progress is pushed to the browser via Server-Sent Events (SSE).

Endpoints
---------
GET  /                              serve chat UI (index.html)
GET  /health
POST /sessions                      create a new chat session
GET  /sessions                      list all sessions (summary)
DELETE /sessions/{id}               delete session + purge its uploaded files

POST /sessions/{id}/upload          upload CSV / Excel files → start pipeline
GET  /sessions/{id}/events          SSE stream: pipeline progress + chat events

POST /sessions/{id}/chat            send a natural-language question
GET  /sessions/{id}/chat/{msg_id}   poll a chat job (fallback when SSE is unavailable)

Environment variables (all optional)
-------------------------------------
METADATA_API_URL   default http://localhost:8000
ONTOLOGY_API_URL   default http://localhost:8001
KG_API_URL         default http://localhost:8002
DIALOG_API_URL     default http://localhost:8003
DATA_DIR           default ./reports
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Service URLs ───────────────────────────────────────────────────────────────
METADATA_API = os.environ.get("METADATA_API_URL", "http://localhost:8000")
ONTOLOGY_API = os.environ.get("ONTOLOGY_API_URL", "http://localhost:8001")
KG_API       = os.environ.get("KG_API_URL",       "http://localhost:8002")
DIALOG_API   = os.environ.get("DIALOG_API_URL",   "http://localhost:8003")

DATA_DIR  = Path(os.environ.get("DATA_DIR", "./reports"))
UI_DIR    = Path(__file__).parent / "chat_ui"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Session store ──────────────────────────────────────────────────────────────
# session_id → session dict
_sessions: Dict[str, Dict] = {}
# session_id → asyncio.Queue  (events pushed by pipeline, consumed by SSE)
_event_queues: Dict[str, asyncio.Queue] = {}
_lock = asyncio.Lock()


def _new_session(title: str = "New conversation") -> Dict:
    return {
        "id":               str(uuid.uuid4()),
        "title":            title,
        "created_at":       time.time(),
        # pipeline state
        "stage":            "idle",        # idle|uploading|extracting|ontology|kg|ready|error
        "pct":              0,
        "stage_message":    "",
        "error":            None,
        # file / db state
        "files":            [],            # [{name, server_path, db_type}]
        "db_file_path":     None,
        "db_type":          None,
        # agent job IDs
        "extract_job_id":   None,
        "ontology_job_id":  None,
        "kg_job_id":        None,
        # graph data (for dialog KG context)
        "kg_nodes":         [],
        "kg_edges":         [],
        # dialog session
        "dialog_session_id": None,
        # chat history  [{id, role, content, results, sql, ts}]
        "messages":         [],
        # pending chat jobs  {msg_id: job_id}
        "chat_jobs":        {},
    }


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="DataChat Orchestrator", version="1.0.0", docs_url="/api/docs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static UI assets under /ui/
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_session(session_id: str) -> Dict:
    s = _sessions.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    return s


async def _push(session_id: str, event: Dict) -> None:
    """Push an event onto the session SSE queue (creates queue if absent)."""
    q = _event_queues.get(session_id)
    if q is None:
        q = asyncio.Queue()
        _event_queues[session_id] = q
    await q.put(event)


def _sse_line(event: Dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


async def _poll_job(
    base_url: str,
    job_id: str,
    session_id: str,
    stage: str,
    stage_label: str,
    pct_start: int,
    pct_end: int,
    poll_interval: float = 1.5,
    timeout_s: float = 600.0,
) -> Optional[Dict]:
    """
    Poll a microservice job until done/error.
    Emits progress events to the SSE queue.
    Returns final job dict or None on error/timeout.
    """
    deadline = time.time() + timeout_s
    node_labels = {
        # extraction nodes
        "connection":  "Connecting to data source",
        "discovery":   "Discovering tables",
        "extraction":  "Extracting column metadata",
        "analysis":    "Analysing dependencies",
        "report":      "Building metadata report",
        # ontology nodes
        "load":        "Loading metadata",
        "build":       "Building OWL classes",
        "serialize":   "Serialising ontology",
        # kg nodes
        "parse":       "Parsing ontology",
        "translate":   "Generating graph queries",
        "execute":     "Writing to knowledge graph",
    }

    last_node = None
    pct_range = pct_end - pct_start

    async with httpx.AsyncClient(timeout=30.0) as client:
        while time.time() < deadline:
            try:
                resp = await client.get(f"{base_url}/jobs/{job_id}")
                job = resp.json()
            except Exception as exc:
                logger.warning("Poll error for %s/%s: %s", base_url, job_id, exc)
                await asyncio.sleep(poll_interval)
                continue

            status = job.get("status", "")
            current_node = job.get("current_node") or ""
            completed = job.get("completed_nodes") or []

            # Estimate pct within stage from completed nodes
            if completed:
                total_nodes = max(len(completed) + 1, 1)
                stage_frac = len(completed) / total_nodes
                pct = pct_start + int(stage_frac * pct_range)
            else:
                pct = pct_start

            node_msg = node_labels.get(current_node, current_node or stage_label)
            if current_node != last_node:
                last_node = current_node
                _sessions[session_id]["stage"] = stage
                _sessions[session_id]["pct"] = pct
                _sessions[session_id]["stage_message"] = node_msg
                await _push(session_id, {
                    "type":    "progress",
                    "stage":   stage,
                    "message": node_msg,
                    "pct":     pct,
                })

            if status == "done":
                _sessions[session_id]["pct"] = pct_end
                await _push(session_id, {
                    "type":    "progress",
                    "stage":   stage,
                    "message": f"{stage_label} complete",
                    "pct":     pct_end,
                })
                return job

            if status == "error":
                err = job.get("error", "Unknown error")
                await _push(session_id, {
                    "type":    "error",
                    "stage":   stage,
                    "message": f"{stage_label} failed: {err}",
                })
                return None

            await asyncio.sleep(poll_interval)

    await _push(session_id, {
        "type":    "error",
        "stage":   stage,
        "message": f"{stage_label} timed out after {timeout_s:.0f}s",
    })
    return None


# ── Pipeline coroutine ─────────────────────────────────────────────────────────

async def _run_pipeline(session_id: str) -> None:
    """
    Full pipeline: extract → ontology → KG → ready.
    Runs as a background asyncio task.
    Pushes SSE events throughout.
    """
    session = _sessions[session_id]
    db_type      = session["db_type"]
    db_file_path = session["db_file_path"]

    try:
        # ── 1. Metadata Extraction ─────────────────────────────────────────────
        await _push(session_id, {
            "type":    "progress",
            "stage":   "extracting",
            "message": "Starting metadata extraction…",
            "pct":     5,
        })

        extract_payload = {
            "db_config": {
                "db_type":   db_type,
                "file_path": db_file_path,
            },
            "sample_size": 10000,
            "fd_threshold": 1.0,
            "id_threshold": 0.95,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{METADATA_API}/extract", json=extract_payload)
            r.raise_for_status()
            extract_job_id = r.json()["job_id"]

        session["extract_job_id"] = extract_job_id
        logger.info("Session %s: extraction job %s started", session_id[:8], extract_job_id[:8])

        extract_job = await _poll_job(
            METADATA_API, extract_job_id, session_id,
            "extracting", "Metadata extraction",
            pct_start=5, pct_end=45,
        )

        if extract_job is None:
            session["stage"] = "error"
            session["error"] = "Metadata extraction failed"
            return

        # Fetch the report
        async with httpx.AsyncClient(timeout=30.0) as client:
            rr = await client.get(f"{METADATA_API}/jobs/{extract_job_id}/report")
            rr.raise_for_status()
            report = rr.json()

        session["report"] = report

        # Collect table names for the ready message
        table_names = list((report.get("tables") or {}).keys())

        # ── Mark as "ready for dialog" (ontology/KG continue in background) ───
        session["stage"]         = "ready"
        session["pct"]           = 100
        session["stage_message"] = "Ready"
        await _push(session_id, {
            "type":    "ready",
            "message": f"Ready! Found {len(table_names)} table{'s' if len(table_names) != 1 else ''}: {', '.join(table_names[:6])}{'…' if len(table_names) > 6 else ''}",
            "tables":  table_names,
        })

        # ── 2. Ontology (background, non-blocking for dialog) ─────────────────
        await _push(session_id, {
            "type":    "progress",
            "stage":   "ontology",
            "message": "Building ontology in background…",
            "pct":     0,
            "background": True,
        })

        ontology_content = None
        try:
            onto_payload = {
                "report":             report,
                "include_statistics": True,
            }
            async with httpx.AsyncClient(timeout=30.0) as client:
                ro = await client.post(f"{ONTOLOGY_API}/generate", json=onto_payload)
                ro.raise_for_status()
                onto_job_id = ro.json()["job_id"]

            session["ontology_job_id"] = onto_job_id
            onto_job = await _poll_job(
                ONTOLOGY_API, onto_job_id, session_id,
                "ontology", "Ontology generation",
                pct_start=0, pct_end=100,
            )
            if onto_job:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    rc = await client.get(f"{ONTOLOGY_API}/jobs/{onto_job_id}/content")
                    if rc.status_code == 200:
                        ontology_content = rc.json().get("content", "")
                await _push(session_id, {
                    "type":       "ontology_ready",
                    "message":    "Ontology built successfully",
                    "job_id":     onto_job_id,
                    "background": True,
                })
        except Exception as exc:
            logger.warning("Session %s: ontology step skipped: %s", session_id[:8], exc)
            await _push(session_id, {
                "type":       "info",
                "message":    "Ontology generation skipped (service unavailable)",
                "background": True,
            })

        # ── 3. Knowledge Graph (background) ───────────────────────────────────
        if ontology_content:
            await _push(session_id, {
                "type":    "progress",
                "stage":   "kg",
                "message": "Building knowledge graph in background…",
                "pct":     0,
                "background": True,
            })
            try:
                kg_payload = {
                    "ontology_content": ontology_content,
                    "store_type":       "memory",
                }
                async with httpx.AsyncClient(timeout=30.0) as client:
                    rk = await client.post(f"{KG_API}/generate", json=kg_payload)
                    rk.raise_for_status()
                    kg_job_id = rk.json()["job_id"]

                session["kg_job_id"] = kg_job_id
                kg_job = await _poll_job(
                    KG_API, kg_job_id, session_id,
                    "kg", "Knowledge graph",
                    pct_start=0, pct_end=100,
                )
                if kg_job:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        rg = await client.get(f"{KG_API}/jobs/{kg_job_id}/graph")
                        if rg.status_code == 200:
                            graph = rg.json()
                            session["kg_nodes"] = graph.get("nodes") or []
                            session["kg_edges"] = graph.get("edges") or []
                    await _push(session_id, {
                        "type":       "kg_ready",
                        "message":    f"Knowledge graph built ({len(session['kg_nodes'])} nodes)",
                        "background": True,
                    })
            except Exception as exc:
                logger.warning("Session %s: KG step skipped: %s", session_id[:8], exc)

    except Exception as exc:
        logger.exception("Pipeline failed for session %s", session_id[:8])
        session["stage"] = "error"
        session["error"] = str(exc)
        await _push(session_id, {
            "type":    "error",
            "stage":   "pipeline",
            "message": f"Pipeline error: {exc}",
        })


# ── SSE generator ──────────────────────────────────────────────────────────────

async def _sse_generator(session_id: str) -> AsyncGenerator[str, None]:
    """Yields SSE-formatted lines from the session event queue."""
    q = _event_queues.get(session_id)
    if q is None:
        q = asyncio.Queue()
        _event_queues[session_id] = q

    # Replay current stage so freshly connected clients get context
    session = _sessions.get(session_id, {})
    stage = session.get("stage", "idle")
    if stage not in ("idle", "ready", "error"):
        yield _sse_line({
            "type":    "progress",
            "stage":   stage,
            "message": session.get("stage_message", stage),
            "pct":     session.get("pct", 0),
        })
    elif stage == "ready":
        tables = list((session.get("report") or {}).get("tables", {}).keys())
        yield _sse_line({
            "type":    "ready",
            "message": f"Ready! {len(tables)} table{'s' if len(tables) != 1 else ''} available.",
            "tables":  tables,
        })
    elif stage == "error":
        yield _sse_line({"type": "error", "message": session.get("error", "Unknown error")})

    heartbeat_interval = 20  # seconds
    last_heartbeat = time.time()

    while True:
        try:
            event = q.get_nowait()
            yield _sse_line(event)
        except asyncio.QueueEmpty:
            await asyncio.sleep(0.3)
            if time.time() - last_heartbeat >= heartbeat_interval:
                yield _sse_line({"type": "heartbeat"})
                last_heartbeat = time.time()


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "services": {
        "metadata": METADATA_API,
        "ontology": ONTOLOGY_API,
        "kg":       KG_API,
        "dialog":   DIALOG_API,
    }}


@app.get("/")
async def serve_ui():
    index = UI_DIR / "index.html"
    if not index.exists():
        return {"error": "UI not found. Ensure chat_ui/index.html exists."}
    return FileResponse(str(index))


# ── Session endpoints ──────────────────────────────────────────────────────────

class NewSessionRequest(BaseModel):
    title: Optional[str] = "New conversation"


@app.post("/sessions", status_code=201)
async def create_session(req: NewSessionRequest):
    s = _new_session(req.title or "New conversation")
    _sessions[s["id"]] = s
    _event_queues[s["id"]] = asyncio.Queue()
    logger.info("Session created: %s", s["id"][:8])
    return {"session_id": s["id"], "title": s["title"]}


@app.get("/sessions")
async def list_sessions():
    return [
        {
            "session_id": s["id"],
            "title":      s["title"],
            "stage":      s["stage"],
            "files":      [f["name"] for f in s.get("files", [])],
            "created_at": s["created_at"],
            "msg_count":  len(s.get("messages", [])),
        }
        for s in sorted(_sessions.values(), key=lambda x: x["created_at"], reverse=True)
    ]


@app.delete("/sessions/{session_id}", status_code=200)
async def delete_session(session_id: str):
    s = _sessions.pop(session_id, None)
    _event_queues.pop(session_id, None)
    if s is None:
        raise HTTPException(status_code=404, detail="Session not found")
    # Purge uploaded files via metadata API (best-effort)
    for f in s.get("files", []):
        path = f.get("server_path")
        if path:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    await client.delete(f"{DIALOG_API}/file-cache/{path}")
            except Exception:
                pass
    logger.info("Session deleted: %s", session_id[:8])
    return {"deleted": session_id}


# ── File upload + pipeline start ───────────────────────────────────────────────

@app.post("/sessions/{session_id}/upload", status_code=202)
async def upload_files(
    session_id: str,
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None,
):
    session = _get_session(session_id)
    if session["stage"] not in ("idle", "error"):
        raise HTTPException(
            status_code=409,
            detail="Pipeline already running or completed for this session. Create a new session.",
        )

    await _push(session_id, {
        "type":    "progress",
        "stage":   "uploading",
        "message": f"Uploading {len(files)} file{'s' if len(files) > 1 else ''}…",
        "pct":     2,
    })
    session["stage"] = "uploading"

    try:
        if len(files) == 1 and files[0].filename.lower().endswith((".xlsx", ".xls", ".xlsm", ".xlsb")):
            # Single Excel file → upload-file
            file = files[0]
            file_bytes = await file.read()
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(
                    f"{METADATA_API}/upload-file",
                    files={"file": (file.filename, file_bytes, file.content_type or "application/octet-stream")},
                )
                r.raise_for_status()
                upload_info = r.json()

            server_path = upload_info["path"]
            db_type     = upload_info["db_type"]
            session["files"] = [{"name": file.filename, "server_path": server_path, "db_type": db_type}]
            session["db_file_path"] = server_path
            session["db_type"]      = db_type

        else:
            # Multiple CSVs or single CSV → upload-files
            form_files = []
            for f in files:
                content = await f.read()
                form_files.append(("files", (f.filename, content, f.content_type or "text/csv")))

            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(f"{METADATA_API}/upload-files", files=form_files)
                r.raise_for_status()
                upload_info = r.json()

            server_path = upload_info["path"]
            db_type     = upload_info["db_type"]
            session["files"] = [{"name": f.filename, "server_path": server_path, "db_type": "csv"} for f in files]
            session["db_file_path"] = server_path
            session["db_type"]      = db_type

    except httpx.HTTPStatusError as exc:
        session["stage"] = "error"
        session["error"] = f"Upload failed: {exc.response.text}"
        await _push(session_id, {"type": "error", "message": session["error"]})
        raise HTTPException(status_code=502, detail=session["error"])
    except Exception as exc:
        session["stage"] = "error"
        session["error"] = f"Upload failed: {exc}"
        await _push(session_id, {"type": "error", "message": session["error"]})
        raise HTTPException(status_code=500, detail=str(exc))

    await _push(session_id, {
        "type":    "progress",
        "stage":   "uploading",
        "message": f"Upload complete — starting pipeline…",
        "pct":     4,
    })

    # Set session title from first file name
    if not session["title"] or session["title"] == "New conversation":
        session["title"] = files[0].filename

    # Fire pipeline as background task
    asyncio.create_task(_run_pipeline(session_id))

    return {
        "session_id": session_id,
        "files":      [f["name"] for f in session["files"]],
        "db_type":    db_type,
    }


# ── SSE events stream ──────────────────────────────────────────────────────────

@app.get("/sessions/{session_id}/events")
async def session_events(session_id: str):
    _get_session(session_id)   # 404 if missing
    return StreamingResponse(
        _sse_generator(session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )


# ── Chat endpoint ──────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    skip_cache: bool = False


@app.post("/sessions/{session_id}/chat", status_code=202)
async def send_chat(session_id: str, req: ChatRequest):
    session = _get_session(session_id)
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")
    if session["stage"] == "error":
        raise HTTPException(status_code=409, detail="Pipeline failed — please start a new session")
    if session["stage"] not in ("ready",):
        raise HTTPException(status_code=409, detail="Data pipeline not ready yet. Please wait.")

    msg_id = str(uuid.uuid4())
    ts     = time.time()

    # Record user message in session history
    session["messages"].append({
        "id":      msg_id,
        "role":    "user",
        "content": req.message,
        "ts":      ts,
    })

    # Push user message to SSE so all connected clients see it
    await _push(session_id, {
        "type":    "user_message",
        "msg_id":  msg_id,
        "content": req.message,
        "ts":      ts,
    })
    await _push(session_id, {
        "type":    "thinking",
        "msg_id":  msg_id,
        "message": "Thinking…",
    })

    # Start background task to run the dialog
    asyncio.create_task(_run_dialog(session_id, msg_id, req.message, req.skip_cache))

    return {"msg_id": msg_id, "session_id": session_id}


async def _run_dialog(session_id: str, msg_id: str, message: str, skip_cache: bool) -> None:
    session = _sessions.get(session_id)
    if not session:
        return

    dialog_session_id = session.get("dialog_session_id")

    dialog_payload = {
        "natural_query":  message,
        "kg_nodes":       session.get("kg_nodes") or [],
        "kg_edges":       session.get("kg_edges") or [],
        "db_type":        session["db_type"],
        "db_file_path":   session["db_file_path"],
        "skip_cache":     skip_cache,
        "session_id":     dialog_session_id,
        "row_limit":      500,
        "max_sql_queries": 10,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{DIALOG_API}/query", json=dialog_payload)
            r.raise_for_status()
            resp = r.json()

        dialog_job_id = resp["job_id"]
        if not dialog_session_id:
            session["dialog_session_id"] = resp.get("session_id", dialog_session_id)

        session["chat_jobs"][msg_id] = dialog_job_id

        # Poll until done
        deadline = time.time() + 300
        result = None
        async with httpx.AsyncClient(timeout=15.0) as client:
            while time.time() < deadline:
                pr = await client.get(f"{DIALOG_API}/jobs/{dialog_job_id}")
                job = pr.json()
                status = job.get("status", "")
                if status == "done":
                    # Fetch full results
                    fr = await client.get(f"{DIALOG_API}/jobs/{dialog_job_id}/results")
                    result = fr.json()
                    break
                if status == "error":
                    raise RuntimeError(job.get("error", "Dialog agent error"))
                await asyncio.sleep(1.0)

        if result is None:
            raise RuntimeError("Dialog agent timed out")

        insights      = result.get("insights", "")
        query_results = result.get("query_results") or []
        sql_queries   = result.get("sql_queries") or []
        errors        = result.get("errors") or []

        # Record AI message
        ai_ts = time.time()
        session["messages"].append({
            "id":            msg_id + "_ai",
            "role":          "assistant",
            "content":       insights,
            "results":       query_results,
            "sql":           sql_queries,
            "errors":        errors,
            "ts":            ai_ts,
        })

        await _push(session_id, {
            "type":          "chat_response",
            "msg_id":        msg_id,
            "content":       insights,
            "results":       query_results,
            "sql":           sql_queries,
            "errors":        errors,
            "ts":            ai_ts,
            "cache_hit":     result.get("cache_hit", False),
        })

    except Exception as exc:
        logger.exception("Dialog failed for session %s msg %s", session_id[:8], msg_id[:8])
        err_msg = str(exc)
        session["messages"].append({
            "id":      msg_id + "_ai",
            "role":    "assistant",
            "content": f"Sorry, I encountered an error: {err_msg}",
            "error":   True,
            "ts":      time.time(),
        })
        await _push(session_id, {
            "type":    "chat_error",
            "msg_id":  msg_id,
            "message": err_msg,
        })


# ── Session history ────────────────────────────────────────────────────────────

@app.get("/sessions/{session_id}/messages")
async def get_messages(session_id: str):
    session = _get_session(session_id)
    return {
        "session_id": session_id,
        "stage":      session["stage"],
        "messages":   session.get("messages", []),
        "files":      [f["name"] for f in session.get("files", [])],
    }


@app.patch("/sessions/{session_id}/file-cache")
async def purge_session_file_cache(session_id: str):
    """Call when the user navigates away — purges the temp SQLite file cache."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.delete(f"{DIALOG_API}/file-cache")
            return r.json()
    except Exception as exc:
        return {"error": str(exc)}
