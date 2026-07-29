"""
FastAPI service for the Knowledge Graph Agent.

Standalone microservice — completely independent of metadata_agent and ontology_agent.
Accepts a raw ontology string (Turtle/RDF/N3), converts it into a {nodes, edges}
knowledge graph snapshot persisted to the shared KG snapshot store (kg_store.py),
and returns graph data for visualisation in the UI.

Endpoints
---------
GET  /health
POST /generate                    start async KG creation/update job → {job_id}
POST /fetch                       load an existing snapshot by kg_id → {job_id}
GET  /jobs/{job_id}               poll status + stats
GET  /jobs/{job_id}/graph         get {nodes, edges} for UI visualisation
GET  /jobs/{job_id}/queries       get the generated Cypher-style statements
GET  /list                        list all KG jobs
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

from knowledge_graph_agent import KGAgent, KGConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Knowledge Graph Agent API", version="1.1.0", docs_url="/docs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory job store ────────────────────────────────────────────────────────
_jobs: Dict[str, Dict] = {}
_lock = threading.Lock()

KG_NODES       = ["parse", "translate", "execute"]
KG_LOAD_NODES  = ["fetch"]


# ── Request models ─────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    ontology_text:   str
    ontology_format: str = "turtle"     # "turtle" | "xml" | "n3"

    # KG identity — must match KGConfig.kg_id / DialogConfig.graphrag_kg_id.
    # When set, nodes are stamped with this id so multiple KGs can coexist
    # in the shared KG snapshot store. Defaults to "default" when empty.
    kg_id: str = ""

    # Behaviour: "generate" (default) | "update" (incremental, never clears)
    mode:           str  = "generate"
    clear_existing: bool = False


class FetchRequest(BaseModel):
    """Load an existing graph snapshot from the KG snapshot store."""
    kg_id: str = ""


# ── Job record helpers ─────────────────────────────────────────────────────────

def _new_job(job_id: str, ontology_format: str, mode: str) -> Dict:
    return {
        "id":                job_id,
        "ontology_format":   ontology_format,
        "mode":              mode,
        "status":            "queued",
        "current_node":      None,
        "completed_nodes":   [],
        "node_count":        0,
        "edge_count":        0,
        "executed_count":    0,
        "graph_data":        {"nodes": [], "edges": []},
        "queries":           [],
        "execution_results": [],
        "errors":            [],
        "error":             None,
    }


# ── Background runners ─────────────────────────────────────────────────────────

def _run_kg(job_id: str, ontology_text: str, ontology_format: str, cfg: KGConfig) -> None:
    """Background task for generate / update modes."""
    with _lock:
        _jobs[job_id]["status"] = "running"

    pipeline_nodes = KG_NODES
    completed: List[str] = []
    try:
        agent = KGAgent(cfg)

        for node_name, state_update in agent.stream_run(ontology_text, ontology_format):
            clean = node_name.strip("_").replace("error_end", "error")
            with _lock:
                if "error" in node_name:
                    _jobs[job_id]["status"] = "error"
                    _jobs[job_id]["error"]  = f"Pipeline error at node: {node_name}"
                    return
                real = clean if clean in pipeline_nodes else None
                if real and real not in completed:
                    completed.append(real)
                _jobs[job_id]["completed_nodes"] = list(completed)
                _jobs[job_id]["current_node"]    = clean

        # Run synchronously to capture final result
        result = agent.run(ontology_text, ontology_format)

        with _lock:
            _jobs[job_id].update({
                "status":           "done",
                "completed_nodes":  list(pipeline_nodes),
                "current_node":     "execute",
                "node_count":       result["node_count"],
                "edge_count":       result["edge_count"],
                "executed_count":   result["executed_count"],
                "graph_data":       result["graph_data"],
                "queries":          result["queries"],
                "execution_results": result["execution_results"],
                "errors":           result["errors"],
            })

    except Exception as exc:
        logger.exception("KG job %s failed", job_id)
        with _lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"]  = str(exc)


def _run_fetch(job_id: str, cfg: KGConfig) -> None:
    """Background task for load mode — fetch existing graph from DB."""
    with _lock:
        _jobs[job_id]["status"] = "running"

    try:
        agent = KGAgent(cfg)

        for node_name, state_update in agent.stream_load():
            clean = node_name.strip("_").replace("error_end", "error")
            with _lock:
                if "error" in node_name:
                    _jobs[job_id]["status"] = "error"
                    _jobs[job_id]["error"]  = f"Fetch error at node: {node_name}"
                    return
                if clean in KG_LOAD_NODES and clean not in _jobs[job_id]["completed_nodes"]:
                    _jobs[job_id]["completed_nodes"].append(clean)
                _jobs[job_id]["current_node"] = clean

        result = agent.load()

        with _lock:
            _jobs[job_id].update({
                "status":          "done",
                "completed_nodes": list(KG_LOAD_NODES),
                "current_node":    "fetch",
                "node_count":      result["node_count"],
                "edge_count":      result["edge_count"],
                "executed_count":  0,
                "graph_data":      result["graph_data"],
                "queries":         [],
                "errors":          result["errors"],
            })

    except Exception as exc:
        logger.exception("KG fetch job %s failed", job_id)
        with _lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"]  = str(exc)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate", status_code=202)
def generate_kg(req: GenerateRequest, background_tasks: BackgroundTasks):
    if not req.ontology_text.strip():
        raise HTTPException(status_code=400, detail="ontology_text must not be empty")
    if req.mode not in ("generate", "update"):
        raise HTTPException(status_code=400, detail="mode must be 'generate' or 'update'")

    # "update" mode forces clear_existing=False so existing data is preserved
    clear = req.clear_existing if req.mode == "generate" else False

    cfg = KGConfig(
        kg_id          = req.kg_id,
        mode           = req.mode,
        clear_existing = clear,
    )

    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = _new_job(job_id, req.ontology_format, req.mode)

    background_tasks.add_task(
        _run_kg, job_id, req.ontology_text, req.ontology_format, cfg
    )
    return {"job_id": job_id}


@app.post("/fetch", status_code=202)
def fetch_graph(req: FetchRequest, background_tasks: BackgroundTasks):
    """Load an existing knowledge graph snapshot from the KG snapshot store."""
    cfg = KGConfig(kg_id=req.kg_id, mode="load")

    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = _new_job(job_id, "", "load")

    background_tasks.add_task(_run_fetch, job_id, cfg)
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {k: v for k, v in job.items() if k not in ("graph_data", "queries", "execution_results")}


@app.get("/jobs/{job_id}/graph")
def get_graph(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=202, detail="Graph not yet ready")
    return job["graph_data"]


@app.get("/jobs/{job_id}/queries")
def get_queries(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=202, detail="Queries not yet ready")
    return {
        "queries": job["queries"],
        "count":   len(job["queries"]),
    }


@app.get("/list")
def list_jobs():
    with _lock:
        jobs = list(_jobs.values())
    return [
        {k: v for k, v in j.items() if k not in ("graph_data", "queries", "execution_results")}
        for j in jobs
    ]
