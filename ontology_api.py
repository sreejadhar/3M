"""
FastAPI service for the Ontology Agent.

Standalone microservice — completely independent of the metadata extraction API.
The UI sends it a metadata report JSON; it returns an OWL/RDF ontology that the
user can view, edit, and download.

Endpoints
---------
GET  /health
POST /generate                    generate ontology from report JSON → {job_id}
GET  /jobs/{job_id}               poll status + stats
GET  /jobs/{job_id}/content       get raw ontology text (editable)
PUT  /jobs/{job_id}/content       save edited ontology text back to disk
GET  /jobs/{job_id}/download      download the OWL/Turtle file
GET  /list                        list all generated ontology jobs
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Package path (set by Docker ENV; fallback for local dev) ──────────────────
sys.path.insert(0, str(Path(__file__).parent))

from ontology_agent import OntologyAgent, OntologyConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Storage ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(os.environ.get("DATA_DIR", "./reports"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── In-memory job store ────────────────────────────────────────────────────────
_jobs: Dict[str, Dict] = {}
_lock = threading.Lock()

ONTO_NODES = ["load", "build", "serialize"]

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Ontology Agent API", version="1.0.0", docs_url="/docs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request models ─────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    report:             Dict[str, Any]
    base_uri:           str  = "http://metadata-agent.io/ontology/"
    ontology_name:      str  = "DatabaseOntology"
    serialize_format:   str  = "turtle"
    include_statistics: bool = True
    # Concept annotation settings (passed through to OntologyConfig)
    annotate_concepts:  bool = True
    source_domain:      str  = ""   # e.g. "telecom", "banking", "RGM/CPG"
    source_name:        str  = ""   # human-readable source name
    source_description: str  = ""   # source description from registration
    db_type:            str  = ""   # e.g. "postgresql", "bigquery", "snowflake"
    llm_model:          str  = "claude-haiku-4-5"


class ContentUpdate(BaseModel):
    content: str


# ── Background runner ──────────────────────────────────────────────────────────
def _run_ontology(job_id: str, report: Dict, cfg: OntologyConfig) -> None:
    with _lock:
        _jobs[job_id]["status"] = "running"

    completed: List[str] = []
    try:
        agent = OntologyAgent(cfg)

        for node_name, _ in agent.stream_run(report):
            clean = node_name.strip("_").replace("error_end", "error")
            with _lock:
                if "error" in node_name:
                    _jobs[job_id]["status"] = "error"
                    _jobs[job_id]["error"]  = f"Pipeline error at node: {node_name}"
                    return
                real = clean if clean in ONTO_NODES else None
                if real and real not in completed:
                    completed.append(real)
                _jobs[job_id]["completed_nodes"] = list(completed)
                _jobs[job_id]["current_node"]    = clean

        # Run synchronously to get the result (stream_run doesn't return values)
        result = agent.run(report)

        with _lock:
            _jobs[job_id].update({
                "status":          "done",
                "completed_nodes": list(ONTO_NODES),
                "current_node":    "serialize",
                "output_path":     result["output_path"],
                "serialize_format": cfg.serialize_format,
                "class_count":     result["class_count"],
                "property_count":  result["property_count"],
                "triple_count":    result["triple_count"],
                "errors":          result["errors"],
            })

    except Exception as exc:
        logger.exception("Ontology job %s failed", job_id)
        with _lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"]  = str(exc)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate", status_code=202)
def generate_ontology(req: GenerateRequest, background_tasks: BackgroundTasks):
    if not req.report:
        raise HTTPException(status_code=400, detail="report must not be empty")

    job_id = str(uuid.uuid4())
    ext    = {"turtle": ".ttl", "xml": ".owl", "n3": ".n3"}.get(req.serialize_format, ".ttl")
    out    = str(DATA_DIR / f"ontology_{job_id[:8]}{ext}")

    # Build a rich domain context string for the concept annotation LLM.
    # Combines all available source metadata so the LLM can contextualise
    # abbreviations correctly (e.g. "arpu" in telecom vs "arpu" in SaaS).
    domain_parts = [p for p in [
        req.source_domain,
        req.db_type,
        req.source_name,
        req.source_description,
    ] if p and p.strip()]
    full_domain_context = " | ".join(domain_parts)

    cfg = OntologyConfig(
        base_uri           = req.base_uri,
        ontology_name      = req.ontology_name,
        output_path        = out,
        serialize_format   = req.serialize_format,
        include_statistics = req.include_statistics,
        annotate_concepts  = req.annotate_concepts,
        source_domain      = full_domain_context,
        llm_model          = req.llm_model,
    )

    with _lock:
        _jobs[job_id] = {
            "id":              job_id,
            "ontology_name":   req.ontology_name,
            "serialize_format": req.serialize_format,
            "status":          "queued",
            "current_node":    None,
            "completed_nodes": [],
            "output_path":     out,
            "class_count":     0,
            "property_count":  0,
            "triple_count":    0,
            "errors":          [],
            "error":           None,
        }

    background_tasks.add_task(_run_ontology, job_id, req.report, cfg)
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {k: v for k, v in job.items() if k != "output_path"}


@app.get("/jobs/{job_id}/content")
def get_content(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=202, detail="Ontology not yet generated")
    p = Path(job["output_path"])
    if not p.exists():
        raise HTTPException(status_code=404, detail="Ontology file not found on disk")
    return {
        "content": p.read_text(encoding="utf-8"),
        "format":  job.get("serialize_format", "turtle"),
        "path":    str(p),
    }


@app.put("/jobs/{job_id}/content")
def update_content(job_id: str, req: ContentUpdate):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail="Job is not complete")
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="Content must not be empty")

    p = Path(job["output_path"])
    p.write_text(req.content, encoding="utf-8")

    # Re-count triples after edit to keep stats accurate
    try:
        from rdflib import Graph
        g = Graph()
        g.parse(data=req.content, format=job.get("serialize_format", "turtle"))
        with _lock:
            _jobs[job_id]["triple_count"] = len(g)
    except Exception as e:
        logger.warning("Could not re-parse edited ontology: %s", e)

    logger.info("Ontology %s updated by user (%d chars)", job_id, len(req.content))
    return {"ok": True}


@app.get("/jobs/{job_id}/download")
def download(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=202, detail="Ontology not yet generated")
    p = Path(job["output_path"])
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")
    mime = {".ttl": "text/turtle", ".owl": "application/rdf+xml", ".n3": "text/n3"}
    return FileResponse(str(p), media_type=mime.get(p.suffix, "text/plain"),
                        filename=p.name)


@app.get("/list")
def list_jobs():
    with _lock:
        jobs = list(_jobs.values())
    return [{k: v for k, v in j.items() if k != "output_path"} for j in jobs]
