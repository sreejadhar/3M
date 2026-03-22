"""
FastAPI service wrapping the Metadata Extraction Agent.

Exposes REST endpoints consumed by the Streamlit UI container.

Endpoints
---------
GET  /health
POST /extract                     start async extraction → {job_id}
GET  /jobs/{job_id}               poll status
GET  /jobs/{job_id}/report        retrieve full JSON report (once done)
GET  /history                     list saved runs
DEL  /history/{run_id}            delete a run + its report file
GET  /history/{run_id}/report     retrieve a saved report by history id
POST /history/{run_id}/ask        LLM Q&A on a saved report
GET  /search                      full-text search across all saved reports

POST /ontology/generate           generate OWL ontology from a saved report → {job_id}
GET  /ontology/jobs/{job_id}      poll ontology job status
GET  /ontology/jobs/{job_id}/download  download the OWL/Turtle file
GET  /ontology/list               list all generated ontologies
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── package on PYTHONPATH (set via ENV in Docker; fallback for local dev) ─────
sys.path.insert(0, str(Path(__file__).parent.parent))

from metadata_agent import AgentConfig, DBConfig, DBType, MetadataExtractionAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Storage ────────────────────────────────────────────────────────────────────
import os
DATA_DIR     = Path(os.environ.get("DATA_DIR", "./reports"))
HISTORY_FILE = DATA_DIR / ".history.json"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_history() -> List[Dict]:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())
        except Exception:
            return []
    return []


def _save_history(history: List[Dict]) -> None:
    HISTORY_FILE.write_text(json.dumps(history, indent=2, default=str))


# ── In-memory job store ────────────────────────────────────────────────────────
_jobs: Dict[str, Dict] = {}
_lock = threading.Lock()

PIPELINE_NODES = ["connection", "discovery", "extraction", "analysis", "report"]


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="Metadata Agent API", version="1.0.0", docs_url="/docs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────────
class DBConfigIn(BaseModel):
    db_type: str
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    schema_name: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    project: Optional[str] = None
    credentials_path: Optional[str] = None
    spark_master: Optional[str] = None
    catalog: Optional[str] = None
    extra: Dict[str, Any] = {}
    file_path: Optional[str] = None


class ExtractionRequest(BaseModel):
    db_config: DBConfigIn
    target_tables: Optional[List[str]] = None
    sample_size: int = 10_000
    fd_threshold: float = 1.0
    id_threshold: float = 0.95


class AskRequest(BaseModel):
    question: str


# ── Background extraction runner ───────────────────────────────────────────────
def _run_extraction(job_id: str, agent_cfg: AgentConfig, db_type: str, db_info: Dict) -> None:
    with _lock:
        _jobs[job_id]["status"] = "running"

    completed: List[str] = []
    try:
        agent = MetadataExtractionAgent(agent_cfg)

        pipeline_error: Optional[str] = None
        for node_name, state_update in agent.stream_run():
            clean = node_name.strip("_").replace("error_end", "error")

            with _lock:
                if "error" in node_name:
                    # Pull the actual error messages from state; fall back to node name
                    node_errors = state_update.get("errors") if isinstance(state_update, dict) else []
                    if node_errors:
                        pipeline_error = "; ".join(str(e) for e in node_errors)
                    else:
                        pipeline_error = f"Pipeline failed at node: {node_name}"
                    _jobs[job_id]["status"] = "error"
                    _jobs[job_id]["error"]  = pipeline_error
                    logger.error("Extraction job %s failed: %s", job_id[:8], pipeline_error)
                else:
                    real = clean if clean in PIPELINE_NODES else None
                    if real and real not in completed:
                        completed.append(real)
                    _jobs[job_id]["completed_nodes"] = list(completed)
                    _jobs[job_id]["current_node"]    = clean

        if pipeline_error:
            return   # job already marked "error" — do not save empty report

        report   = agent._report or {}
        if not report:
            with _lock:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["error"]  = "Pipeline completed but produced an empty report"
            return

        out_path = str(DATA_DIR / f"{db_type}_{job_id[:8]}.json")
        Path(out_path).write_text(json.dumps(report, indent=2, default=str))

        # Persist to history
        summary = report.get("summary", {})
        entry = {
            "id":          job_id,
            "timestamp":   datetime.now().isoformat(),
            "db_type":     db_type,
            "host":        db_info.get("host", ""),
            "database":    db_info.get("database", ""),
            "schema":      db_info.get("schema", ""),
            "summary":     summary,
            "report_path": out_path,
        }
        history = _load_history()
        history.insert(0, entry)
        _save_history(history)

        with _lock:
            _jobs[job_id].update({
                "status":          "done",
                "completed_nodes": list(PIPELINE_NODES),
                "current_node":    "report",
                "report_path":     out_path,
                "summary":         summary,
            })

    except Exception as exc:
        logger.exception("Extraction job %s failed", job_id)
        with _lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"]  = str(exc)


# ── Helper: load report from disk ──────────────────────────────────────────────
def _load_report_from_path(path: str) -> Dict:
    if not path:
        raise HTTPException(status_code=404, detail="No report path recorded for this run")
    p = Path(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Report file not found on disk")
    try:
        data = json.loads(p.read_text())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not parse report: {e}")
    if not data:
        raise HTTPException(status_code=404, detail="Report file is empty — extraction may have failed")
    return data


# ── Helper: LLM ask ───────────────────────────────────────────────────────────
def _ask_llm(report: Dict, question: str) -> str:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, SystemMessage

    if not report:
        return "No report available — run an extraction first."

    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.0)
    system = SystemMessage(content=(
        "You are a data engineering expert. You have been provided the full "
        "metadata report from a database schema scan. Answer questions about "
        "the schema structure, data quality, and relationships concisely and accurately.\n\n"
        "METADATA REPORT (JSON):\n"
        + json.dumps(report, indent=2, default=str)[:40_000]
    ))
    human = HumanMessage(content=question)
    response = llm.invoke([system, human])
    return response.content


# ── Schema / table discovery helpers ──────────────────────────────────────────

def _list_schemas(connector, db_type: str) -> List[str]:
    """Return a list of user-visible schema names from the connected database."""
    try:
        if db_type in ("postgres", "redshift"):
            rows = connector.execute(
                "SELECT schema_name FROM information_schema.schemata "
                "WHERE schema_name NOT LIKE 'pg_%' "
                "  AND schema_name NOT IN ('information_schema') "
                "ORDER BY schema_name"
            )
            return [r["schema_name"] for r in rows]

        elif db_type == "oracle":
            rows = connector.execute(
                "SELECT DISTINCT owner FROM all_tables ORDER BY owner"
            )
            _skip = {"SYS", "SYSTEM", "OUTLN", "DBSNMP", "WMSYS", "XDB", "APEX_030200",
                     "CTXSYS", "EXFSYS", "MDSYS", "OLAPSYS", "ORDDATA", "ORDSYS",
                     "ORDPLUGINS", "SI_INFORMTN_SCHEMA"}
            key = next((k for k in (rows[0] if rows else {}) if k.upper() == "OWNER"), "owner")
            return [r[key] for r in rows if r.get(key) not in _skip]

        elif db_type == "sqlserver":
            rows = connector.execute(
                "SELECT name FROM sys.schemas "
                "WHERE name NOT IN ('guest','INFORMATION_SCHEMA','sys') "
                "  AND name NOT LIKE 'db_%' "
                "ORDER BY name"
            )
            return [r["name"] for r in rows]

        elif db_type == "teradata":
            rows = connector.execute(
                "SELECT DatabaseName FROM DBC.Databases WHERE DBKind='D' ORDER BY DatabaseName"
            )
            return [r.get("DatabaseName") or r.get("databasename", "") for r in rows]

        elif db_type == "bigquery":
            rows = connector.execute(
                "SELECT schema_name FROM INFORMATION_SCHEMA.SCHEMATA ORDER BY schema_name"
            )
            return [r.get("schema_name", "") for r in rows]

        elif db_type == "delta_lake":
            rows = connector.execute("SHOW DATABASES")
            return [r.get("databaseName") or r.get("namespace", "") for r in rows if r]

        elif db_type == "sqlite":
            # SQLite has a single implicit schema called "main"
            return ["main"]

        elif db_type in ("csv", "excel"):
            # Tables are the CSV files / Excel sheets; single unnamed schema
            return [""]

        else:
            return []
    except Exception as exc:
        logger.warning("_list_schemas failed for %s: %s", db_type, exc)
        return []


def _build_db_config(db: DBConfigIn) -> DBConfig:
    return DBConfig(
        db_type=DBType(db.db_type),
        host=db.host,
        port=db.port,
        database=db.database,
        schema=db.schema_name,
        username=db.username,
        password=db.password,
        project=db.project,
        credentials_path=db.credentials_path,
        spark_master=db.spark_master,
        catalog=db.catalog,
        extra=db.extra,
        file_path=db.file_path,
    )


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/discover")
def discover_db(db: DBConfigIn):
    """
    Connect to the database and return available schemas and their tables.

    If `schema_name` is provided in the request, only tables for that schema
    are returned.  Otherwise all user-visible schemas are listed and their
    tables are fetched (capped at 50 schemas to avoid long waits).
    """
    try:
        db_cfg = _build_db_config(db)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid DB config: {e}")

    conn = None
    try:
        from metadata_agent.connectors import get_connector  # noqa: PLC0415
        conn = get_connector(db_cfg)
        conn.connect()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Could not connect: {e}")

    try:
        if db.schema_name:
            schemas = [db.schema_name]
        else:
            schemas = _list_schemas(conn, db.db_type)

        tables_by_schema: Dict[str, List[str]] = {}
        for schema in schemas[:50]:
            try:
                tbls = conn.list_tables(schema)
                tables_by_schema[schema] = sorted(t[1] for t in tbls)
            except Exception:
                tables_by_schema[schema] = []

        return {"schemas": schemas, "tables": tables_by_schema}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("discover_db failed for %s", db.db_type)
        raise HTTPException(status_code=500, detail=f"Discovery failed: {e}")
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


# ── File upload storage ────────────────────────────────────────────────────────
UPLOAD_TTL_SECONDS    = 2 * 60 * 60   # base TTL: 2 hours
UPLOAD_EXTEND_SECONDS = 15 * 60       # one-time extension: +15 minutes
UPLOAD_PURGE_INTERVAL = 15 * 60       # purge sweep every 15 minutes
_UPLOAD_DIR = DATA_DIR / "uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Registry: path → {uploaded_at, expires_at, extended, db_type, label}
_upload_registry: Dict[str, Dict] = {}
_registry_lock   = threading.Lock()


def _db_type_from_path(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in (".xlsx", ".xls", ".xlsm", ".xlsb"):
        return "excel"
    if ext in (".sqlite", ".db", ".sqlite3"):
        return "sqlite"
    if Path(path).is_dir():
        return "csv"
    return "unknown"


def _ts(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _register_upload(path: str, db_type: str, label: str, permanent: bool = False) -> Optional[float]:
    """Record an upload; return its expiry epoch (None if permanent)."""
    expires = None if permanent else time.time() + UPLOAD_TTL_SECONDS
    with _registry_lock:
        _upload_registry[path] = {
            "uploaded_at": time.time(),
            "expires_at":  expires,
            "extended":    False,
            "db_type":     db_type,
            "label":       label,
            "permanent":   permanent,
        }
    return expires


def _purge_old_uploads() -> None:
    """Remove entries (and files/dirs) whose expiry has passed."""
    now = time.time()
    with _registry_lock:
        expired = [p for p, m in _upload_registry.items()
                   if m["expires_at"] is not None and m["expires_at"] < now]
    for path in expired:
        try:
            p = Path(path)
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists():
                p.unlink()
            logger.info("Purged expired upload: %s", path)
        except Exception as exc:
            logger.debug("Could not purge %s: %s", path, exc)
        with _registry_lock:
            _upload_registry.pop(path, None)


def _upload_purge_loop() -> None:
    while True:
        time.sleep(UPLOAD_PURGE_INTERVAL)
        _purge_old_uploads()


threading.Thread(target=_upload_purge_loop, daemon=True, name="upload-purge").start()


@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    """Upload a single file (SQLite / Excel); returns path, expires_at, db_type."""
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = _UPLOAD_DIR / file.filename
    with dest.open("wb") as buf:
        shutil.copyfileobj(file.file, buf)
    db_type  = _db_type_from_path(str(dest))
    expires  = _register_upload(str(dest), db_type, file.filename)
    return {"path": str(dest), "expires_at": _ts(expires), "db_type": db_type}


@app.post("/upload-files")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload multiple CSV files; returns directory path + expiry."""
    csv_dir = _UPLOAD_DIR / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for file in files:
        dest = csv_dir / file.filename
        with dest.open("wb") as buf:
            shutil.copyfileobj(file.file, buf)
        names.append(file.filename)
    label   = ", ".join(names)
    expires = _register_upload(str(csv_dir), "csv", label)
    return {"path": str(csv_dir), "expires_at": _ts(expires), "db_type": "csv"}


@app.get("/uploads/list")
def list_uploads():
    """Return all non-expired uploads with expiry info."""
    now = time.time()
    with _registry_lock:
        items = [
            {
                "path":       path,
                "label":      meta["label"],
                "db_type":    meta["db_type"],
                "expires_at": _ts(meta["expires_at"]),
                "extended":   meta["extended"],
                "seconds_left": max(0, int(meta["expires_at"] - now)),
            }
            for path, meta in _upload_registry.items()
            if meta["expires_at"] > now
        ]
    return {"uploads": items}


class ExtendRequest(BaseModel):
    path: str


@app.post("/uploads/extend")
def extend_upload(req: ExtendRequest):
    """Grant a one-time 15-minute extension for an uploaded file."""
    with _registry_lock:
        meta = _upload_registry.get(req.path)
        if meta is None:
            raise HTTPException(status_code=404, detail="Upload not found or already expired.")
        if meta["extended"]:
            raise HTTPException(status_code=409, detail="already_extended")
        meta["expires_at"] += UPLOAD_EXTEND_SECONDS
        meta["extended"]    = True
        new_expires = meta["expires_at"]
    logger.info("Extended upload TTL by 15 min: %s (new expiry %s)", req.path, _ts(new_expires))
    return {"expires_at": _ts(new_expires), "extended": True}


@app.post("/upload-permanent")
async def upload_permanent(file: UploadFile = File(...)):
    """Upload a file for a registered source (no TTL — persists until explicitly purged)."""
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = _UPLOAD_DIR / file.filename
    contents = await file.read()
    dest.write_bytes(contents)
    db_type = _db_type_from_path(str(dest))
    _register_upload(str(dest), db_type, file.filename, permanent=True)
    return {"path": str(dest), "db_type": db_type, "filename": file.filename}


class PurgeRequest(BaseModel):
    path: str


@app.delete("/uploads/purge", status_code=200)
def purge_upload(req: PurgeRequest):
    """Explicitly delete a registered upload from disk and the registry."""
    path = req.path
    with _registry_lock:
        removed = _upload_registry.pop(path, None)
    try:
        p = Path(path)
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()
    except Exception as exc:
        logger.warning("Could not delete file %s: %s", path, exc)
    if removed is None:
        raise HTTPException(status_code=404, detail="Upload not found in registry")
    return {"deleted": path}


@app.post("/extract", status_code=202)
def start_extraction(req: ExtractionRequest, background_tasks: BackgroundTasks):
    db = req.db_config
    try:
        db_cfg = _build_db_config(db)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid DB config: {e}")

    job_id   = str(uuid.uuid4())
    out_path = str(DATA_DIR / f"{db.db_type}_{job_id[:8]}.json")

    agent_cfg = AgentConfig(
        db_config=db_cfg,
        target_tables=req.target_tables,
        sample_size=req.sample_size,
        fd_threshold=req.fd_threshold,
        id_threshold=req.id_threshold,
        output_path=out_path,
    )
    db_info = {
        "host":     db.host or db.project or "",
        "database": db.database or db.project or "",
        "schema":   db.schema_name or "",
    }

    with _lock:
        _jobs[job_id] = {
            "id":              job_id,
            "status":          "queued",
            "current_node":    None,
            "completed_nodes": [],
            "report_path":     None,
            "summary":         {},
            "error":           None,
        }

    background_tasks.add_task(_run_extraction, job_id, agent_cfg, db.db_type, db_info)
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {k: v for k, v in job.items()}   # no large report body


@app.get("/jobs/{job_id}/report")
def get_job_report(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=202, detail="Job not yet complete")
    return _load_report_from_path(job["report_path"])


# ── History ────────────────────────────────────────────────────────────────────
@app.get("/history")
def list_history():
    return _load_history()


@app.delete("/history/{run_id}")
def delete_history_entry(run_id: str):
    history = _load_history()
    entry   = next((h for h in history if h["id"] == run_id), None)
    if entry and entry.get("report_path"):
        Path(entry["report_path"]).unlink(missing_ok=True)
    _save_history([h for h in history if h["id"] != run_id])
    return {"ok": True}


@app.get("/history/{run_id}/report")
def get_history_report(run_id: str):
    history = _load_history()
    entry   = next((h for h in history if h["id"] == run_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="History entry not found")
    return _load_report_from_path(entry.get("report_path", ""))


@app.post("/history/{run_id}/ask")
def ask_about_report(run_id: str, req: AskRequest):
    # Check live jobs first (job_id == run_id for fresh extractions)
    with _lock:
        job = _jobs.get(run_id)
    if job and job["status"] == "done" and job.get("report_path"):
        report = _load_report_from_path(job["report_path"])
    else:
        # Fall back to history on disk
        history = _load_history()
        entry   = next((h for h in history if h["id"] == run_id), None)
        if not entry:
            raise HTTPException(status_code=404, detail="Report not found")
        report  = _load_report_from_path(entry.get("report_path", ""))
    try:
        answer = _ask_llm(report, req.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Search ─────────────────────────────────────────────────────────────────────
@app.get("/search")
def search_metadata(q: str, scope: str = "all", db_type: str = "all"):
    q_lower  = q.strip().lower()
    if not q_lower:
        return []

    results: List[Dict] = []
    history  = _load_history()

    for h in history:
        if db_type != "all" and h.get("db_type") != db_type:
            continue
        rpath = h.get("report_path", "")
        if not rpath or not Path(rpath).exists():
            continue
        try:
            report = json.loads(Path(rpath).read_text())
        except Exception:
            continue

        run_label = f'{h.get("database","?")} / {h.get("schema","?")} ({h["timestamp"][:10]})'

        if scope in ("tables", "all"):
            for tname, tmeta in (report.get("tables") or {}).items():
                if q_lower in tname.lower():
                    results.append({
                        "kind":    "table",
                        "match":   tname,
                        "context": run_label,
                        "detail":  f'{tmeta.get("row_count","?") if isinstance(tmeta, dict) else "?"} rows',
                        "db_type": h["db_type"],
                        "run_id":  h["id"],
                    })
                if isinstance(tmeta, dict):
                    for col in tmeta.get("columns", []):
                        if isinstance(col, dict):
                            cname = col.get("name", "")
                            ctype = col.get("data_type", "")
                            if q_lower in cname.lower() or q_lower in ctype.lower():
                                results.append({
                                    "kind":    "column",
                                    "match":   f"{tname}.{cname}",
                                    "context": run_label,
                                    "detail":  ctype,
                                    "db_type": h["db_type"],
                                    "run_id":  h["id"],
                                })

        if scope in ("fds", "all"):
            for fd in (report.get("functional_dependencies") or []):
                det  = fd.get("determinant", [])
                dep  = fd.get("dependent", [])
                tbl  = fd.get("table", "")
                text = " ".join(det + dep + [tbl]).lower()
                if q_lower in text:
                    results.append({
                        "kind":    "fd",
                        "match":   f'[{", ".join(det)}] → [{", ".join(dep)}]',
                        "context": run_label,
                        "detail":  f"table: {tbl}  conf={fd.get('confidence', 0):.2f}",
                        "db_type": h["db_type"],
                        "run_id":  h["id"],
                    })

    return results[:100]
