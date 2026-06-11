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

NEO4J_URI          bolt://… — when set, every KG build is persisted to Neo4j
NEO4J_USERNAME     default neo4j
NEO4J_PASSWORD     Neo4j password (required when NEO4J_URI is set)
NEO4J_DATABASE     default neo4j

KG_POSTGRES_DSN    psycopg2 DSN — when set, kg_registry, kg_bridges and metadata use PG
KG_FEDERATION_DB   SQLite path for KG federation tables (default data/kg_federation.db)
METADATA_DB        SQLite path for metadata catalog     (default data/metadata.db)
                   In production (APP_ENV=production) both tables use KG_POSTGRES_DSN.
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
import sqlite3

import bcrypt
import jwt as _jwt
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import metadata_catalog as _mc   # standalone catalog module (no dialog_agent dep)
import kpi_store as _kpi              # BI Manager KPI definitions
import kg_store as _kg_store           # KG snapshot persistence
import access_control as _ac           # RBAC / ABAC engine
import ontology_enricher as _enricher  # glossary+KPI → KG/attribute enrichment

# ── GraphRAG helpers (inlined — no dialog_agent import needed) ─────────────────
# These functions mirror dialog_agent/nodes/retrieve_node.py but have no
# LangGraph / LangChain dependencies, so they work in the orchestrator process.

import hashlib
import re as _re
from collections import deque as _deque


# Per-source embedding cache: f"{schema_hash}:{backend}" → _GRCache
_GR_EMBED_CACHE: Dict[str, Any] = {}


class _GRCache:
    __slots__ = ("key", "backend", "node_ids", "texts", "matrix", "_vect")

    def __init__(self, key, backend, node_ids, texts, matrix, vect=None):
        self.key      = key
        self.backend  = backend
        self.node_ids = node_ids
        self.texts    = texts
        self.matrix   = matrix
        self._vect    = vect

    def embed_query(self, query: str):
        import numpy as np
        if self.backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer("all-MiniLM-L6-v2")
            v = m.encode([query], normalize_embeddings=True)[0]
            return v.astype(np.float32)
        if self.backend == "openai":
            from openai import OpenAI
            resp = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "")).embeddings.create(
                model="text-embedding-3-small", input=[query]
            )
            v = np.array(resp.data[0].embedding, dtype=np.float32)
            n = np.linalg.norm(v)
            return v / max(n, 1e-9)
        if self.backend == "tfidf" and self._vect is not None:
            v = self._vect.transform([query]).toarray()[0].astype(np.float32)
            n = np.linalg.norm(v)
            return v / max(n, 1e-9)
        return _gr_keyword_embed([query], self.texts)[0]


def _gr_node_text(node: Dict) -> str:
    label = node.get("label", "")
    title = node.get("title", "")
    body  = _re.sub(r'^Class:\s*\S+\s*', '', title, flags=_re.IGNORECASE).strip()
    return f"{label} {body}".strip()


def _gr_keyword_embed(texts, vocab_texts=None):
    import numpy as np
    tok   = lambda t: _re.findall(r'\w+', t.lower())
    src   = vocab_texts if vocab_texts is not None else texts
    vocab = sorted({w for t in src for w in tok(t)})
    idx   = {w: i for i, w in enumerate(vocab)}
    mat   = np.zeros((len(texts), max(len(vocab), 1)), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in tok(t):
            if w in idx:
                mat[i, idx[w]] = 1.0
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, 1e-9)


def _gr_detect_backend(preferred: str = "auto") -> str:
    if preferred != "auto":
        return preferred
    for candidate, pkg in [("sentence-transformers", "sentence_transformers"),
                            ("tfidf", "sklearn")]:
        try:
            __import__(pkg)
            return candidate
        except ImportError:
            continue
    return "keyword"


def _gr_schema_hash(nodes) -> str:
    fp = json.dumps(sorted((n.get("id", ""), n.get("title", "")) for n in nodes))
    return hashlib.md5(fp.encode()).hexdigest()


def _gr_build_cache(nodes, backend: str, s_hash: str) -> _GRCache:
    import numpy as np
    node_ids = [n.get("id", "") for n in nodes]
    texts    = [_gr_node_text(n) for n in nodes]

    if backend == "sentence-transformers":
        from sentence_transformers import SentenceTransformer
        model  = SentenceTransformer("all-MiniLM-L6-v2")
        matrix = model.encode(texts, normalize_embeddings=True).astype(np.float32)
        return _GRCache(s_hash, backend, node_ids, texts, matrix)

    if backend == "openai":
        from openai import OpenAI
        resp   = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "")).embeddings.create(
            model="text-embedding-3-small", input=texts)
        vecs   = np.array([d.embedding for d in resp.data], dtype=np.float32)
        norms  = np.linalg.norm(vecs, axis=1, keepdims=True)
        matrix = vecs / np.maximum(norms, 1e-9)
        return _GRCache(s_hash, backend, node_ids, texts, matrix)

    if backend == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        vect   = TfidfVectorizer(max_features=3000, sublinear_tf=True)
        raw    = vect.fit_transform(texts).toarray().astype(np.float32)
        norms  = np.linalg.norm(raw, axis=1, keepdims=True)
        matrix = raw / np.maximum(norms, 1e-9)
        return _GRCache(s_hash, backend, node_ids, texts, matrix, vect=vect)

    matrix = _gr_keyword_embed(texts)
    return _GRCache(s_hash, backend, node_ids, texts, matrix)


def _gr_build_adjacency(edges) -> Dict[str, List]:
    adj: Dict[str, List] = {}
    for e in edges:
        src, tgt = e.get("from", ""), e.get("to", "")
        if src and tgt:
            adj.setdefault(src, []).append(tgt)
            adj.setdefault(tgt, []).append(src)
    return adj


def _gr_bfs_expand(seeds, adjacency, hop_depth: int) -> List[str]:
    visited  = set(seeds)
    frontier = _deque((nid, 0) for nid in seeds)
    while frontier:
        nid, depth = frontier.popleft()
        if depth >= hop_depth:
            continue
        for nb in adjacency.get(nid, []):
            if nb not in visited:
                visited.add(nb)
                frontier.append((nb, depth + 1))
    return list(visited)

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
SHACL_API          = os.environ.get("SHACL_API_URL",          "http://localhost:8007")
UNSTRUCTURED_API   = os.environ.get("UNSTRUCTURED_API_URL",   "http://localhost:8008")

# ── Neo4j persistence ──────────────────────────────────────────────────────────
# KG nodes/edges are persisted to Neo4j only in production (APP_ENV=production)
# and only when NEO4J_URI is provided.
# In development / test the KG stays in-process memory (no Neo4j required).
_NEO4J_URI      = os.environ.get("NEO4J_URI",      "")
_NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
_NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
_NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")


def _use_neo4j() -> bool:
    """True only in production when NEO4J_URI is configured."""
    if os.environ.get("APP_ENV", "").strip().lower() != "production":
        return False
    if _NEO4J_URI:
        return True
    logger.warning(
        "APP_ENV=production but NEO4J_URI is not set — "
        "KG will not be persisted to Neo4j."
    )
    return False

DATA_DIR  = Path(os.environ.get("DATA_DIR", "./reports"))
UI_DIR    = Path(__file__).parent / "chat_ui"
# React build (Vite) for the DataChat UI; served at "/" when present, with the
# legacy chat_ui/ kept as a fallback.
REACT_UI_DIR = Path(__file__).parent / "chat_frontend" / "dist"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Auth config ────────────────────────────────────────────────────────────────
_JWT_SECRET  = os.environ.get("JWT_SECRET", "dev-secret-change-in-production")
_JWT_ALG     = "HS256"
_JWT_EXPIRY  = 24 * 3600  # 24 hours in seconds

_AUTH_DB_PATH = Path(os.environ.get("DATA_DIR", "./data")) / "auth.db"

# Seed users: env var name → email
_SEED_USERS = [
    ("SEED_PASSWORD_RIMNA",      "Rimna.Radhakrishnan@cognizant.com"),
    ("SEED_PASSWORD_IYER",       "Iyer.Kasinath@cognizant.com"),
    ("SEED_PASSWORD_SENTHEESH",  "Sentheesh.Lingam@cognizant.com"),
    ("SEED_PASSWORD_HARPREET",   "Harpreet.Sethi@cognizant.com"),
    ("SEED_PASSWORD_SUNIL",      "Sunil.Pinnamaneni@cognizant.com"),
    ("SEED_PASSWORD_SATHISH",    "Sathish.Hegde@cognizant.com"),
    ("SEED_PASSWORD_PALASH",     "Palash.Ghosh@cognizant.com"),
    ("SEED_PASSWORD_SASWATA",    "Saswata.Kundu@cognizant.com"),
]


def _auth_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_AUTH_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _bootstrap_auth() -> None:
    """Create auth_users table and seed users from env vars."""
    _AUTH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _auth_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS auth_users (
                email       TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                role        TEXT NOT NULL DEFAULT 'user'
            )
        """)
        for env_var, email in _SEED_USERS:
            raw_pw = os.environ.get(env_var, "")
            if not raw_pw:
                continue
            pw_hash = bcrypt.hashpw(raw_pw.encode(), bcrypt.gensalt()).decode()
            conn.execute(
                "INSERT INTO auth_users (email, password_hash, role) VALUES (?, ?, 'user') "
                "ON CONFLICT(email) DO UPDATE SET password_hash=excluded.password_hash",
                (email.lower(), pw_hash),
            )
        conn.commit()
    logger.info("auth: bootstrapped auth_users table")

# ── Session store ──────────────────────────────────────────────────────────────
# session_id → session dict
_sessions: Dict[str, Dict] = {}
# session_id → asyncio.Queue  (events pushed by pipeline, consumed by SSE)
_event_queues: Dict[str, asyncio.Queue] = {}
_lock = asyncio.Lock()

# ── Source Registry ────────────────────────────────────────────────────────────
_sources: Dict[str, Dict] = {}
# Per-source SSE queues for indexing step events (source_id → Queue)
_source_event_queues: Dict[str, asyncio.Queue] = {}
# Per-source recent event buffer for replay on new SSE connections (last 100 events)
_source_event_log: Dict[str, List[Dict]] = {}
# Event loop reference — captured at startup so sync threads can schedule onto it
_main_loop: asyncio.AbstractEventLoop | None = None

DOMAIN_ICONS: Dict[str, str] = {
    "Sales":      "💼",
    "Finance":    "💰",
    "Operations": "⚙️",
    "HR":         "👥",
    "Marketing":  "📢",
    "IT":         "🖥️",
    "Other":      "📊",
}


# ── Source models ──────────────────────────────────────────────────────────────

class SourceConnection(BaseModel):
    # populate_by_name=True lets Python code use field name (schema_) OR alias (schema)
    model_config = ConfigDict(populate_by_name=True)

    file_path:         str            = ""
    uploaded:          bool           = False   # True when file was uploaded via /sources/upload-file
    host:              str            = ""
    port:              int            = 0
    database:          str            = ""
    # alias="schema" makes FastAPI accept {"schema": "..."} from JSON;
    # populate_by_name=True also accepts {"schema_": "..."} from Python/internal code
    schema_:           str            = Field(default="public", alias="schema")
    username:          str            = ""
    password:          str            = ""
    connection_string: str            = ""
    extra:             Dict[str, Any] = {}

class CreateSourceRequest(BaseModel):
    name:           str
    description:    str       = ""
    domain:         str       = "Other"
    db_type:        str
    connection:     SourceConnection
    persona_access: List[str] = ["business_user", "analyst", "admin"]
    auto_index:     bool      = True

class TestConnectionRequest(BaseModel):
    db_type:    str
    connection: SourceConnection


def _new_source(name, description, domain, db_type, connection, persona_access) -> Dict:
    return {
        "id":               str(uuid.uuid4()),
        "name":             name,
        "description":      description,
        "domain":           domain,
        "icon":             DOMAIN_ICONS.get(domain, "📊"),
        "db_type":          db_type,
        "connection":       connection,
        "status":           "idle",
        "error_message":    None,
        "persona_access":   persona_access,
        "created_at":       time.time(),
        "indexed_at":       None,
        "table_count":      0,
        "table_names":      [],
        "extract_job_id":   None,
        "ontology_job_id":  None,
        "kg_job_id":        None,
        "report":           None,
        "ontology_content": None,
        "kg_nodes":         [],
        "kg_edges":         [],
    }


def _source_public(s: Dict) -> Dict:
    """Return source dict with credentials stripped — safe to send to browser."""
    return {
        "id":             s["id"],
        "name":           s["name"],
        "description":    s["description"],
        "domain":         s["domain"],
        "icon":           s["icon"],
        "db_type":        s["db_type"],
        "status":         s["status"],
        "error_message":  s.get("error_message"),
        "persona_access": s["persona_access"],
        "created_at":     s["created_at"],
        "indexed_at":     s.get("indexed_at"),
        "table_count":    s["table_count"],
        "table_names":    s["table_names"],
    }


def _new_session(title: str = "New conversation", persona: str = "business_user") -> Dict:
    return {
        "id":               str(uuid.uuid4()),
        "title":            title,
        "persona":          persona,
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

# Serve static UI assets under /ui/ (legacy chat_ui)
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")

# Serve the React build's hashed assets at /assets/ (index.html references them).
if (REACT_UI_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(REACT_UI_DIR / "assets")), name="react_assets")


# ── Startup: restore KG snapshots + bootstrap RBAC ────────────────────────────

@app.on_event("startup")
async def _startup() -> None:
    """
    In production: reload all previously-indexed sources from kg_store so that
    their KG nodes/edges are immediately available without re-indexing.
    Also bootstrap the RBAC store (creates tables, seeds default admin if needed).
    """
    global _main_loop
    _main_loop = asyncio.get_event_loop()

    try:
        _bootstrap_auth()
    except Exception as exc:
        logger.warning("auth bootstrap failed (non-fatal): %s", exc)

    try:
        _ac.bootstrap()
        logger.info("access_control: RBAC store bootstrapped (enforced=%s)", _ac.is_enforced())
    except Exception as exc:
        logger.warning("access_control bootstrap failed (non-fatal): %s", exc)

    try:
        restored = _kg_store.load_all()
        for src in restored:
            if src["id"] not in _sources:
                _sources[src["id"]] = src
        logger.info("kg_store: restored %d sources from persistent store", len(restored))
        # Back-fill taxonomy annotations for all restored sources whose
        # md_attributes have not yet been annotated.
        for src in restored:
            try:
                n = _sync_taxonomy_from_kg_nodes(src["id"], src.get("kg_nodes") or [])
                if n:
                    logger.info("Startup taxonomy sync: %d columns for %s", n, src["id"][:8])
            except Exception as _ts_exc:
                logger.warning("Startup taxonomy sync failed for %s: %s", src["id"][:8], _ts_exc)

        # Startup bridge inference — run across all restored KG pairs so bridges
        # are available immediately after restart without requiring a re-index.
        # Only runs when ≥2 KGs are registered.  Non-blocking: runs in a thread
        # so it does not delay server startup.
        if len(restored) >= 2:
            def _startup_bridge_inference():
                try:
                    from dialog_agent.kg_registry import list_all as _reg_list
                    from dialog_agent.kg_inference_engine import KGContext, run_all_pairs
                    entries = _reg_list()
                    if len(entries) < 2:
                        return
                    contexts = []
                    for entry in entries:
                        src = _sources.get(entry.source_id) or {}
                        contexts.append(KGContext(
                            kg_id=entry.kg_id,
                            display_name=entry.display_name,
                            domain=(entry.domain_keywords or [""])[0],
                            nodes=src.get("kg_nodes") or [],
                            report=src.get("report"),
                        ))
                    result = run_all_pairs(contexts)
                    logger.info(
                        "Startup bridge inference: %d pairs processed, %d bridges saved",
                        result["pairs_processed"], result["bridges_saved"],
                    )
                except Exception as _bi_exc:
                    logger.warning("Startup bridge inference failed (non-fatal): %s", _bi_exc)

            import threading as _threading
            _threading.Thread(target=_startup_bridge_inference, daemon=True, name="startup-bridge-infer").start()
    except Exception as exc:
        logger.warning("kg_store restore failed (non-fatal): %s", exc)


# ── Auth endpoints ─────────────────────────────────────────────────────────────

class _LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/auth/login")
async def auth_login(body: _LoginRequest):
    with _auth_db() as conn:
        row = conn.execute(
            "SELECT email, password_hash, role FROM auth_users WHERE email = ?",
            (body.email.strip().lower(),),
        ).fetchone()
    if row is None or not bcrypt.checkpw(body.password.encode(), row["password_hash"].encode()):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = _jwt.encode(
        {"sub": row["email"], "role": row["role"], "exp": int(time.time()) + _JWT_EXPIRY},
        _JWT_SECRET,
        algorithm=_JWT_ALG,
    )
    return {"access_token": token, "email": row["email"], "role": row["role"]}


@app.get("/auth/validate")
async def auth_validate(request: Request):
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.removeprefix("Bearer ").strip()
    if not token:
        return {"valid": False}
    try:
        _jwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALG])
        return {"valid": True}
    except _jwt.PyJWTError:
        return {"valid": False}


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
                # Don't overwrite "ready" — ontology/KG run in background
                # and must not block the chat endpoint which requires stage=="ready"
                if _sessions[session_id]["stage"] != "ready":
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
                    "ontology_text": ontology_content,
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


# ── Source indexer ─────────────────────────────────────────────────────────────

_FILE_BASED_TYPES = {"sqlite", "csv", "excel"}


# ---------------------------------------------------------------------------
# Domain inference from extracted report
# ---------------------------------------------------------------------------
# Ordered: first match wins.  Each entry:
# ── Two-tier domain inference ────────────────────────────────────────────────
# INDUSTRY signals: broad vertical (e.g. CPG, Life Sciences, Retail, Telecom)
# FUNCTION signals: cross-cutting capability (e.g. FP&A, Supply Chain, RGM)
#
# The inference algorithm scores both tiers independently and combines them
# as "Industry/Function" (e.g. "CPG/FP&A", "LS/Supply Chain").  When only
# one tier fires the single label is returned (e.g. "Telecom", "Supply Chain").

_INDUSTRY_SIGNALS: List[tuple] = [
    # Aviation / Airlines
    ("Aviation", {
        # Aircraft & fleet
        "aircraft", "aircraft_type", "tail_number", "tail_no", "registration",
        "fleet", "fleet_type", "wide_body", "narrow_body", "turboprop",
        # Flight operations
        "flight", "flight_number", "flight_no", "flt", "sector", "flight_leg",
        "leg", "rotation", "block_time", "flight_time", "air_time",
        "departure", "arrival", "origin", "destination",
        "scheduled_departure", "actual_departure", "scheduled_arrival", "actual_arrival",
        "dep_time", "arr_time", "etd", "eta", "atd", "ata",
        # Airport & station
        "airport", "station", "icao", "iata", "faa",
        "gate", "terminal", "runway", "apron", "taxiway", "hangar",
        "slot", "slot_time",
        # Crew
        "crew", "pilot", "copilot", "co_pilot", "captain", "first_officer",
        "cabin_crew", "flight_attendant", "purser", "crew_id",
        "crew_ot", "overtime", "duty_time", "rest_period", "pairing",
        "crew_complement", "deadhead",
        # Passengers & load
        "passenger", "pax", "load_factor", "booked", "seats_available",
        "cabin_class", "business_class", "economy", "first_class",
        "boarding", "manifest", "no_show",
        # Revenue / yield (aviation-specific)
        "yield", "rask", "cask", "ask", "rpk", "atk", "rtk",
        "revenue_per_seat", "fare_class", "booking_class",
        # Cargo & baggage
        "cargo", "freight", "baggage", "checked_bag", "carry_on",
        "cargo_weight", "payload",
        # Delay & OTP
        "delay", "delay_code", "delay_reason", "on_time", "otp",
        "on_time_performance", "late", "cancelled", "diverted",
        # Fuel
        "fuel", "fuel_burn", "fuel_uplift", "fuel_consumption",
        # MRO / maintenance
        "mro", "maintenance", "airworthiness", "aircraft_maintenance",
        "check_a", "check_b", "check_c", "check_d",
        # Airline identifiers
        "airline", "carrier", "codeshare", "alliance", "iata_code",
        "operating_carrier", "marketing_carrier",
        # ATC / routing
        "atc", "ifr", "vfr", "route", "waypoint", "airway",
    }),
    # Consumer Packaged Goods (CPG / FMCG)
    # Broad product-hierarchy vocabulary that appears in ANY CPG database
    # (supply chain, FP&A, sales, RGM) — not just RGM-specific revenue terms.
    ("CPG", {
        # Product hierarchy (present in virtually every CPG system)
        "sku", "brand", "sub_brand", "pack_size", "pack_type", "upc",
        "ean", "product_hierarchy", "sub_category",
        "brand_pack", "flavor", "variant", "ppg", "pog",
        # CPG-specific revenue / trade vocabulary
        "rsv", "gsv", "nrv", "nsv", "tts", "cti",
        "volume_offtake", "gross_rsv", "net_rsv",
        "trade_spend", "promo_spend", "offtake",
        # CPG channel / customer vocabulary
        "retailer", "customer_group", "trade_channel", "modern_trade",
        "general_trade", "gt", "mt", "key_account", "distributor",
        # Generic identifiers
        "consumer_goods", "fmcg", "cpg", "category_mgmt",
        "distribution_points", "weighted_distribution", "numeric_distribution",
    }),
    # Life Sciences / Pharma
    ("LS", {
        # Clinical / trial
        "clinical_trial", "trial", "protocol", "cohort", "arm", "randomised",
        "placebo", "double_blind", "phase", "trial_phase", "enrollment",
        "site", "investigator", "irb", "ethics_committee",
        # Drug / molecule
        "molecule", "drug", "compound", "active_ingredient", "excipient",
        "formulation", "dosage", "dose", "strength", "route_of_administration",
        "ndc", "anda", "nda", "bla", "inn", "atc_code",
        # Regulatory / safety
        "adverse_event", "ae", "sae", "serious_adverse", "pharmacovigilance",
        "signal_detection", "post_market", "recall", "fda", "ema", "cdsco",
        "510k", "pma", "regulatory_submission", "label",
        # Patient / clinical data
        "patient", "subject", "participant", "diagnosis", "indication",
        "icd10", "icd_code", "procedure_code", "comorbidity", "biomarker",
        "genomic", "genotype", "phenotype", "lab_result", "vital_sign",
        # Commercial / market access
        "formulary", "iqvia", "ims", "rx", "otc", "prescription",
        "market_access", "payer_mix", "rebate", "gross_to_net",
        "therapy_area", "physician", "hospital", "hcp", "specialty",
        # Manufacturing / quality (pharma-specific)
        "batch", "lot_number", "expiry", "shelf_life", "gmp", "capa",
        "deviation", "oos", "out_of_spec", "release_testing",
    }),
    # Healthcare (non-pharma / payer / provider)
    ("Healthcare", {
        "length_of_stay", "los", "bed_occupancy", "bed_days", "census",
        "claims_paid", "denial_rate", "readmission", "readmit",
        "cost_per_patient", "cost_per_episode", "drg", "icd", "cpt_code",
        "admission", "discharge", "inpatient", "outpatient", "ed_visit",
        "emergency", "icu", "nicu", "payer", "insurer", "hmo", "ppo",
        "member", "beneficiary", "eligibility", "prior_auth",
        "network", "in_network", "out_of_network", "copay", "deductible",
        "ehr", "emr", "fhir", "hl7", "hipaa",
    }),
    # Telecom / Telco
    ("Telecom", {
        "arpu", "mou", "data_usage", "subscriber", "subscription",
        "churn_rate", "churn", "prepaid", "postpaid", "sim", "msisdn",
        "imei", "imsi", "roaming", "spectrum", "frequency_band",
        "minutes_of_use", "cell_tower", "cell_site", "bts", "enb",
        "bandwidth", "network_quality", "dropped_call", "call_setup",
        "data_plan", "recharge", "top_up", "bundle", "vas",
        "mnp", "number_portability", "operator", "mvno",
        "fiber", "broadband", "dsl", "ftth", "last_mile",
        "revenue_per_user", "blended_arpu", "voice_revenue", "data_revenue",
    }),
    # Banking / Financial Services
    ("Banking/FS", {
        # Balance sheet / P&L
        "nii", "nim", "net_interest_income", "net_interest_margin",
        "casa", "current_account", "savings_account", "fixed_deposit",
        "gnpa", "nnpa", "npa", "non_performing", "provision_coverage",
        "crar", "capital_adequacy", "tier1", "tier2", "slr", "crr",
        "net_interest", "loan_book", "loan_portfolio", "advances",
        "deposit", "borrowing", "liability", "asset_quality",
        "credit_risk", "provisioning", "write_off", "write_back",
        "return_on_assets", "roa", "roe", "cost_to_income",
        # Lending
        "loan", "mortgage", "home_loan", "auto_loan", "personal_loan",
        "emi", "disbursement", "sanction", "outstanding", "overdue",
        "dpd", "days_past_due", "delinquency", "restructured",
        "credit_score", "bureau_score", "cibil", "ltv_ratio",
        # Products / channels
        "account", "account_number", "ifsc", "bic", "swift",
        "transaction", "txn", "debit", "credit", "transfer",
        "branch", "atm", "digital_banking", "mobile_banking",
        "card", "credit_card", "debit_card", "pos", "merchant",
        # Treasury / markets
        "bond", "yield_curve", "duration", "convexity", "alm",
        "forex", "fx_rate", "treasury", "investment_portfolio",
    }),
    # Insurance
    ("Insurance", {
        # Core policy / premium
        "premium", "gross_premium", "net_premium", "earned_premium",
        "written_premium", "single_premium", "renewal_premium",
        "policy", "policy_number", "policy_term", "policy_holder",
        "insured", "beneficiary", "nominee", "sum_assured",
        # Claims
        "claim", "claim_number", "claim_date", "claim_amount",
        "incurred_loss", "paid_loss", "ibnr", "ibner",
        "loss_ratio", "claims_ratio", "combined_ratio",
        "claims_settled", "claims_repudiated", "claims_pending",
        # Underwriting
        "underwriting", "risk_assessment", "risk_class", "risk_score",
        "proposal", "quote", "endorsement", "exclusion", "deductible",
        "sub_limit", "co_insurance", "reinsurance", "cedant",
        "treaty", "facultative", "retrocession",
        # Actuarial
        "actuarial", "mortality", "morbidity", "lapse_rate",
        "persistency", "surrender", "maturity", "annuity",
        "reserve", "liability_reserve", "unearned_premium",
        # Lines of business
        "life", "health", "motor", "fire", "marine", "liability",
        "property", "casualty", "p_and_c", "general_insurance",
        "term", "ulip", "endowment", "whole_life",
        # Channels / distribution
        "agent", "broker", "bancassurance", "direct", "online_sale",
        "agency_code", "pos_agent", "intermediary",
    }),
    # Retail (physical)
    ("Retail", {
        "store_sales", "same_store", "comp_sales", "footfall", "traffic",
        "basket", "basket_size", "shrinkage", "planogram", "assortment",
        "markdown", "sell_through", "stock_turn", "stock_turnover",
        "store", "store_id", "store_format", "hypermarket", "supermarket",
        "pos_transaction", "till", "checkout_lane",
        "loyalty", "loyalty_card", "points_earned", "redemption",
        "private_label", "own_brand", "national_brand",
        "replenishment", "out_of_stock", "on_shelf_availability",
        "promotion", "price_cut", "feature", "display",
    }),
    # E-commerce
    ("E-commerce", {
        "gmv", "aov", "cac", "ltv", "roas", "cart", "cart_abandonment",
        "conversion_rate", "basket_size", "add_to_cart",
        "checkout", "order_value", "return_rate", "refund",
        "marketplace", "seller", "listing", "sku_listing",
        "session", "visit", "bounce_rate", "page_view",
        "search_rank", "sponsored", "ad_spend",
        "fulfillment", "last_mile", "ndr", "rto",
        "review", "rating", "star_rating",
    }),
    # Manufacturing / Industrials
    ("Manufacturing", {
        # Efficiency / quality
        "oee", "oee_availability", "oee_performance", "oee_quality",
        "scrap_rate", "scrap", "rework", "rejection_rate", "defect_rate",
        "yield_pct", "first_pass_yield", "fpy", "quality_inspection",
        "downtime", "planned_downtime", "unplanned_downtime",
        "mtbf", "mttr", "mttf", "reliability",
        # Production
        "cycle_time", "takt_time", "throughput", "production_order",
        "work_order", "work_in_progress", "wip", "finished_goods",
        "production_schedule", "planned_qty", "actual_qty",
        "shift", "shift_output", "line_efficiency",
        # Equipment / assets
        "machine", "machine_id", "equipment", "asset_id",
        "spindle", "press", "mould", "tooling", "fixture",
        "preventive_maintenance", "pm_schedule", "breakdown",
        "plant", "shop_floor", "cell", "workcenter",
        # Materials
        "bom", "bill_of_materials", "raw_material", "component",
        "sub_assembly", "assembly", "routing", "operation_sequence",
        "material_consumption", "standard_cost", "actual_cost",
        # Standards
        "iso", "iatf", "as9100", "spc", "cpk", "ppk", "control_chart",
    }),
    # Agriculture / Agri-business
    ("Agriculture", {
        # Crops & cultivation
        "crop", "crop_type", "crop_variety", "variety", "hybrid",
        "seed", "seed_rate", "germination", "sowing", "planting",
        "harvest", "harvesting", "yield_per_acre", "yield_per_hectare",
        "acreage", "hectare", "farm", "field", "plot", "parcel",
        "kharif", "rabi", "zaid", "season", "growing_season",
        # Soil & agronomy
        "soil", "soil_type", "soil_health", "ph_level", "organic_matter",
        "fertilizer", "urea", "dap", "potash", "micronutrient",
        "pesticide", "herbicide", "fungicide", "insecticide",
        "irrigation", "drip", "sprinkler", "canal", "groundwater",
        # Livestock & dairy
        "livestock", "cattle", "buffalo", "poultry", "sheep", "goat",
        "milk_yield", "milk_production", "fat_content", "snf",
        "animal_id", "tag_id", "breed", "lactation", "calving",
        "feed", "fodder", "dry_matter",
        # Supply chain / market
        "mandi", "apmc", "procurement_centre", "farmer",
        "farmer_id", "fpo", "cooperative", "agri_input",
        "minimum_support_price", "msp", "procurement_price",
        "storage", "cold_storage", "warehouse", "silo",
        "commodity", "agri_commodity", "produce", "grain",
        "wheat", "rice", "paddy", "maize", "soybean", "cotton",
        "sugarcane", "pulses", "oilseed",
        # Weather / geo
        "rainfall", "temperature", "humidity", "evapotranspiration",
        "geo_lat", "geo_lon", "district", "taluka", "village",
        "land_parcel", "cadastral",
        # Finance / schemes
        "kcc", "kisan_credit", "crop_insurance", "pmfby",
        "subsidy", "input_cost", "cost_of_cultivation",
    }),
    # SaaS / Technology
    ("SaaS", {
        "dau", "mau", "mrr", "arr", "expansion_revenue",
        "activation_rate", "feature_adoption", "session_duration",
        "retention_rate", "nrr", "logo_churn", "saas",
        "trial", "freemium", "paid_conversion", "upgrade",
        "seat", "license", "subscription_tier", "plan",
        "api_call", "api_usage", "endpoint", "latency",
        "uptime", "sla", "incident", "p1", "p2",
        "onboarding", "time_to_value", "ttv", "health_score",
    }),
]

_FUNCTION_SIGNALS: List[tuple] = [
    # Revenue Growth Management (RGM)
    # Keep these specific — avoid terms that appear in non-RGM databases
    ("RGM", {
        "rgm", "revenue_growth", "pricing_impact", "price_index",
        "market_share", "mix_contribution", "price_mix", "volume_mix",
        "promo_effectiveness", "trade_rate", "net_revenue_mgmt",
        "pack_price", "price_ladder", "revenue_mgmt",
    }),
    # Financial Planning & Analysis (FP&A)
    # Cover the full vocabulary of FP&A tables/columns: GL, budget, forecast,
    # P&L, cost centres, period structures, etc.
    ("FP&A", {
        # Core planning vocab
        "budget", "forecast", "actuals", "variance", "rolling_forecast",
        "zero_based", "ytd", "qtd", "mtd", "period_budget",
        # P&L / income statement
        "p_and_l", "pnl", "income_statement", "ebitda", "ebit", "ebitda_margin",
        "gross_profit", "operating_profit", "net_income",
        # Cash / balance sheet
        "cash_flow", "capex", "opex", "working_capital", "balance_sheet",
        # Cost structure
        "cost_centre", "cost_center", "cost_element", "gl_account",
        "chart_of_accounts", "profit_centre", "profit_center",
        "business_unit_plan", "entity_plan",
        # FP&A-specific identifiers
        "financial_planning", "fp_and_a", "fpa", "headcount_plan",
        "scenario", "version", "baseline", "reforecast",
    }),
    # Supply Chain & Logistics
    ("Supply Chain", {
        "otif", "fill_rate", "lead_time", "inventory_days",
        "doh", "woh", "perfect_order", "on_time_delivery",
        "supplier_on_time", "stock_out", "stockout", "safety_stock",
        "replenishment", "demand_plan", "s_and_op", "sop",
        "warehouse", "3pl", "distribution",
        # Additional supply chain vocab
        "purchase_order", "po_line", "goods_receipt", "grn",
        "shipment", "delivery", "transit", "backorder", "reorder",
        "supplier", "vendor", "procurement", "inventory_turn",
    }),
    # Sales & Commercial
    ("Sales", {
        "sales_rep", "territory", "quota", "pipeline", "opportunity",
        "win_rate", "deal_size", "crm", "account_manager",
        "sales_force", "coverage_model", "gtm",
    }),
    # Marketing
    ("Marketing", {
        "impressions", "click_through", "cpm", "cpc", "cpa",
        "media_spend", "brand_awareness", "campaign", "reach",
        "frequency", "attribution", "media_mix",
    }),
    # HR / People Analytics
    ("HR/People", {
        "headcount", "attrition", "time_to_hire", "cost_per_hire",
        "offer_acceptance", "engagement_score", "absenteeism",
        "performance_rating", "fte_count", "salary_band",
    }),
    # Manufacturing / Operations (function, not industry)
    ("Operations", {
        "oee", "downtime", "throughput", "cycle_time", "capacity",
        "utilisation", "shift", "plant", "production_schedule",
    }),
    # Customer Experience / Service
    ("CX", {
        "nps", "csat", "net_promoter", "ticket", "resolution_time",
        "first_contact", "escalation", "customer_effort", "churn",
        "voice_of_customer",
    }),
]


# ---------------------------------------------------------------------------
# Sample-value signals: distinct categorical values that uniquely identify a
# domain.  Matched against the actual top_values sampled from each column.
# Scored at 0.5× the weight of name tokens so they supplement but never
# override strong schema-name evidence.
# ---------------------------------------------------------------------------
_INDUSTRY_SAMPLE_SIGNALS: Dict[str, set] = {
    "Aviation": {
        # Aircraft type codes
        "b737", "a320", "b777", "a330", "b787", "a380", "e190", "crj9", "dh8d",
        "b738", "a321", "a319", "b772", "b788", "a220", "a350", "b767", "b757",
        "atr72", "q400", "atr42",
        # Cabin / booking class
        "business class", "economy class", "first class", "premium economy",
        # Flight status values
        "departed", "arrived", "cancelled", "diverted", "airborne",
        # Crew ranks
        "captain", "first officer", "senior first officer", "purser",
        "senior cabin crew", "cabin crew",
        # Delay category descriptions
        "technical", "reactionary", "weather delay", "atc delay",
        # Operations type
        "domestic flight", "international flight",
    },
    "Banking": {
        # NPA / asset quality buckets
        "standard", "sub-standard", "doubtful", "loss asset",
        "non-performing", "restructured",
        # Loan product names
        "home loan", "personal loan", "auto loan", "vehicle loan",
        "business loan", "gold loan", "education loan", "lap", "msme loan",
        # Deposit products
        "savings account", "current account", "fixed deposit",
        "recurring deposit", "nre account", "nro account",
        # DPD / overdue buckets
        "0-30 days", "31-60 days", "61-90 days", "91-180 days", ">180 days",
        # Credit rating values
        "aaa", "aa+", "aa", "a+", "a", "bbb+", "bbb",
        # Account status
        "dormant", "active", "frozen", "closed",
    },
    "LS": {
        # Clinical trial phases
        "phase i", "phase ii", "phase iii", "phase iv",
        "phase 1", "phase 2", "phase 3", "phase 4",
        # Trial / subject status
        "enrolled", "randomised", "randomized", "screen failure",
        "discontinued", "completed", "withdrawn", "lost to follow-up",
        # AE severity
        "mild", "moderate", "severe", "life-threatening", "fatal",
        # Dosage form
        "tablet", "capsule", "injection", "infusion", "oral solution",
        "suspension", "patch", "inhaler",
        # Route of administration
        "oral", "intravenous", "subcutaneous", "intramuscular", "topical",
        # Regulatory submission types
        "nda", "anda", "bla", "ind",
    },
    "Insurance": {
        # Policy / product types
        "term life", "whole life", "endowment", "ulip", "money back",
        "health insurance", "motor insurance", "fire insurance",
        "marine insurance", "group health", "personal accident",
        # Claim status
        "intimated", "under investigation", "settled", "repudiated",
        "partially settled", "closed", "reopened",
        # Premium payment mode
        "annual", "semi-annual", "quarterly", "single premium",
        # Distribution channel
        "bancassurance", "agency", "direct", "online", "broker",
        # Policy status
        "in force", "lapsed", "surrendered", "matured", "free look",
    },
    "Manufacturing": {
        # Defect / rejection types
        "scratch", "dent", "burr", "flash", "porosity", "crack",
        "dimensional defect", "surface defect", "weld defect",
        # Shift identifiers
        "morning shift", "afternoon shift", "evening shift", "night shift",
        "shift a", "shift b", "shift c", "shift i", "shift ii", "shift iii",
        # Machine / equipment status
        "breakdown", "planned maintenance", "unplanned downtime",
        "idle", "changeover", "setup",
        # Production order status
        "released", "in process", "partially completed", "goods receipt",
        # Quality disposition
        "accept", "reject", "rework", "hold", "scrap",
        # Standards / certification
        "iso 9001", "iatf 16949", "as9100",
    },
    "Agriculture": {
        # Crop names
        "wheat", "rice", "paddy", "maize", "cotton", "sugarcane",
        "soybean", "groundnut", "sorghum", "bajra", "jowar",
        "potato", "onion", "tomato", "mustard", "sunflower",
        "chickpea", "lentil", "arhar", "moong",
        # Crop seasons
        "kharif", "rabi", "zaid",
        # Irrigation methods
        "drip irrigation", "sprinkler", "canal irrigation",
        "rainfed", "flood irrigation", "micro irrigation",
        # Soil classification
        "alluvial", "black soil", "red soil", "laterite", "sandy loam",
        "clay loam", "loamy sand",
        # Fertilizer / input types
        "urea", "dap", "mop", "npk", "compost", "vermicompost",
    },
    "CPG": {
        # Pack types
        "pet bottle", "glass bottle", "tetra pack", "sachet", "pouch",
        "can", "tin", "carton", "blister pack",
        # Trade channels
        "modern trade", "general trade", "horeca", "e-commerce",
        "supermarket", "hypermarket", "convenience store", "wholesale",
        # SKU lifecycle
        "active", "new launch", "discontinued", "seasonal", "limited edition",
        # Volume / value flag
        "volume", "value", "tonnage",
    },
    "Telecom": {
        # Technology generation
        "2g", "3g", "4g", "5g", "lte", "volte",
        # Subscription type
        "prepaid", "postpaid",
        # Services
        "voice", "data", "sms", "roaming", "vas", "broadband",
        # Churn reason
        "porting out", "voluntary deactivation", "non-payment",
        # Network event types
        "call drop", "network failure", "planned outage",
    },
    "Healthcare": {
        # Encounter / visit type
        "inpatient", "outpatient", "emergency", "day surgery",
        "observation", "telehealth",
        # Payer type
        "medicare", "medicaid", "commercial", "self-pay", "uninsured",
        # Clinical status
        "acute", "chronic", "preventive", "follow-up",
        # Disposition
        "discharged", "transferred", "expired", "absconded",
    },
    "Retail": {
        # Store format
        "hypermarket", "supermarket", "convenience", "express",
        "flagship", "kiosk", "warehouse store",
        # Loyalty tier
        "gold", "silver", "platinum", "bronze",
        # Markdown type
        "clearance", "seasonal sale", "promo", "everyday low price",
        # Inventory status
        "in stock", "out of stock", "on order", "discontinued",
    },
}


def _infer_domain_from_report(report: Dict) -> str:
    """
    Scan table names, column names, AND sampled column values in the extraction
    report and return a compound domain label "Industry/Function" (e.g.
    "Aviation/FP&A", "Banking/FS") when both tiers fire, or a single label
    when only one tier fires.

    Scoring:
      • name tokens (table + column names split on _): weight 1.0
      • sample value tokens (top_values from each column): weight 0.5
        — supplements schema-name evidence without overriding it.

    Falls back to "" when no signals match so the caller can detect the
    no-match case and preserve the admin-provided domain.
    """
    tables: Dict = report.get("tables") or {}

    # Build name tokens and sample tokens separately
    name_tokens: set = set()
    sample_tokens: set = set()

    for table_name, table_meta in tables.items():
        tl = table_name.lower()
        name_tokens.add(tl)
        name_tokens.update(tl.split("_"))

        if not isinstance(table_meta, dict):
            continue

        for col in table_meta.get("columns") or []:
            if not isinstance(col, dict):
                continue

            cl = col.get("name", "").lower()
            name_tokens.add(cl)
            name_tokens.update(cl.split("_"))

            # Collect sample values — add full lowercased value AND
            # individual word tokens (skip tokens ≤2 chars to reduce noise)
            for val in col.get("top_values") or []:
                if val is None:
                    continue
                vl = str(val).lower().strip()
                if not vl or len(vl) <= 1:
                    continue
                sample_tokens.add(vl)
                for word in vl.split():
                    if len(word) > 2:
                        sample_tokens.add(word)

    def _best(signal_list: List[tuple]) -> Optional[tuple]:
        scores: Dict[str, float] = {}
        for label, name_signals in signal_list:
            name_hits  = len(name_tokens & name_signals)
            # Sample signals only defined for industry tier
            samp_signals = _INDUSTRY_SAMPLE_SIGNALS.get(label, set())
            samp_hits  = len(sample_tokens & samp_signals)
            total = name_hits + 0.5 * samp_hits
            if total > 0:
                scores[label] = total
        if not scores:
            return None
        best_label = max(scores, key=lambda d: scores[d])
        return best_label, scores[best_label], scores

    industry_result = _best(_INDUSTRY_SIGNALS)
    function_result = _best(_FUNCTION_SIGNALS)

    logger.info(
        "_infer_domain_from_report: industry=%s function=%s",
        industry_result, function_result,
    )

    if industry_result and function_result:
        ind_label, ind_score, _ = industry_result
        fn_label,  fn_score,  _ = function_result
        return f"{ind_label}/{fn_label}"

    if industry_result:
        return industry_result[0]

    if function_result:
        return function_result[0]

    return ""


def _build_db_config(db_type: str, conn: Dict) -> Dict:
    if db_type.lower() in _FILE_BASED_TYPES:
        return {"db_type": db_type, "file_path": conn.get("file_path", "")}
    cfg: Dict[str, Any] = {
        "db_type":     db_type,
        "host":        conn.get("host", ""),
        "port":        conn.get("port", 5432),
        "database":    conn.get("database", ""),
        "username":    conn.get("username", ""),
        "password":    conn.get("password", ""),
        "schema_name": conn.get("schema_", "public"),
    }
    if conn.get("extra"):
        cfg["extra"] = conn["extra"]
    return cfg


def _push_index_event(source_id: str, step: str, status: str, message: str, detail: str = "") -> None:
    """Append an indexing step event to the per-source log and live queue."""
    event = {
        "type":    "index_step",
        "step":    step,
        "status":  status,   # running | done | error | warn
        "message": message,
        "detail":  detail,
        "ts":      time.time(),
    }
    # Append to replay buffer (keep last 100)
    buf = _source_event_log.setdefault(source_id, [])
    buf.append(event)
    if len(buf) > 100:
        buf.pop(0)
    # Push to live SSE queue if a client is connected.
    # _push_index_event is called from a background thread, so we must
    # schedule onto the event loop rather than calling put_nowait directly.
    q = _source_event_queues.get(source_id)
    if q and _main_loop is not None:
        try:
            _main_loop.call_soon_threadsafe(q.put_nowait, event)
        except asyncio.QueueFull:
            pass


_TAX_RE = _re.compile(
    r"taxonomy:\s*"
    r"statistical_type=([^\s|]+)"
    r".*?semantic_role=([^\s|]+)"
    r"(?:.*?format_pattern=([^\s|]+))?",
    _re.IGNORECASE,
)

def _sync_taxonomy_from_kg_nodes(source_id: str, kg_nodes: List[Dict]) -> int:
    """
    Parse taxonomy: annotations from KG node titles (written by profile_node)
    and sync them back into md_attributes.

    Each KG node title for a Class node contains lines like:
        Properties:
          fiscal_year: xsd:string  -- taxonomy: statistical_type=ordinal | semantic_role=time_period | ...
          category: xsd:string     -- taxonomy: statistical_type=categorical | semantic_role=product_category | ...

    We parse these and write statistical_type + semantic_role into md_attributes
    by matching on (source_id, column_name). top_values already in the row are
    preserved as taxonomy_tree if the column is categorical/ordinal.
    """
    import json as _json

    # Parse every node title for "col_name: xsd_type  -- taxonomy: ..." lines
    col_tax: Dict[str, Dict[str, str]] = {}  # column_name → {statistical_type, semantic_role, format_pattern}
    _prop_line_re = _re.compile(r"^\s{2}(.+?):\s+\S+.*?--\s*(taxonomy:\s*.+)$", _re.IGNORECASE)

    for node in kg_nodes:
        title = node.get("title") or ""
        for line in title.splitlines():
            m = _prop_line_re.match(line)
            if not m:
                continue
            col_name, tax_str = m.group(1), m.group(2)
            tm = _TAX_RE.search(tax_str)
            if not tm:
                continue
            col_tax[col_name] = {
                "statistical_type": tm.group(1) or "",
                "semantic_role":    tm.group(2) or "",
                "format_pattern":   tm.group(3) or "",
            }

    if not col_tax:
        return 0

    # Build a normalised lookup: lowercase + underscores-replace-spaces
    def _norm(s: str) -> str:
        return s.strip().lower().replace(" ", "_")

    col_tax_norm = {_norm(k): v for k, v in col_tax.items()}

    # Fetch all attributes for this source from md_attributes
    import json as _json

    now = _mc._now()
    updated = 0
    try:
        with _mc._cursor_ctx() as cur:
            _mc._ensure(cur)
            # Get all active attrs for this source with their top_values
            attrs = cur.execute(
                "SELECT a.attr_id, a.column_name, a.top_values "
                "FROM md_attributes a "
                "JOIN md_entities e ON a.metadata_id = e.metadata_id "
                "WHERE e.source_id=? AND a.deleted_from_source=?",
                (source_id, _mc._enc_bool(False)),
            ).fetchall()

            for attr in attrs:
                col_name = attr["column_name"]
                # Try exact match first, then normalised match
                tax = col_tax.get(col_name) or col_tax_norm.get(_norm(col_name))
                if not tax:
                    continue
                stat_type = tax["statistical_type"]
                sem_role  = tax["semantic_role"]

                # For categorical/ordinal columns, reuse top_values as taxonomy_tree
                tree = "[]"
                if stat_type in ("categorical", "ordinal"):
                    try:
                        tv = _json.loads(attr.get("top_values") or "[]")
                        tree = _json.dumps(tv) if tv else "[]"
                    except Exception:
                        pass

                cur.execute(
                    "UPDATE md_attributes SET statistical_type=?, semantic_role=?, "
                    "taxonomy_tree=?, updated_at=? WHERE attr_id=?",
                    (stat_type, sem_role, tree, now, attr["attr_id"]),
                )
                updated += 1

    except Exception as exc:
        logger.warning("_sync_taxonomy_from_kg_nodes failed for %s: %s", source_id[:8], exc)
        return 0

    logger.info("Taxonomy sync: %d columns annotated for source %s", updated, source_id[:8])
    return updated


async def _index_source(source_id: str) -> None:
    """Full pipeline for a registered data source: extract → ontology → KG."""
    src = _sources.get(source_id)
    if not src:
        return
    src["status"]        = "indexing"
    src["error_message"] = None
    _source_event_log[source_id] = []   # clear log for fresh run
    db_config = _build_db_config(src["db_type"], src["connection"])

    try:
        # ── 1. Extraction ──────────────────────────────────────────────────────
        _push_index_event(source_id, "extract", "running", "Extracting metadata from database…")
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{METADATA_API}/extract", json={
                "db_config":    db_config,
                "sample_size":  10000,
                "fd_threshold": 1.0,
                "id_threshold": 0.95,
            })
            r.raise_for_status()
            extract_job_id = r.json()["job_id"]

        src["extract_job_id"] = extract_job_id
        report = None

        # ── Stream fine-grained events in real-time from agent-api SSE ────────
        extraction_error: Optional[str] = None
        _sse_timeout = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=5.0)
        try:
            async with httpx.AsyncClient(timeout=_sse_timeout) as client:
                async with client.stream(
                    "GET", f"{METADATA_API}/jobs/{extract_job_id}/stream"
                ) as r:
                    async for raw_line in r.aiter_lines():
                        if not raw_line.startswith("data:"):
                            continue
                        try:
                            ev = json.loads(raw_line[5:].strip())
                        except Exception:
                            continue

                        ev_type = ev.get("type", "")
                        if ev_type == "heartbeat":
                            continue
                        if ev_type == "done":
                            if ev.get("status") == "error":
                                extraction_error = ev.get("error", "Extraction failed")
                            break

                        # Forward every fine-grained event straight to the browser SSE
                        _push_index_event(
                            source_id,
                            ev.get("step", "extract"),
                            ev.get("status", "running"),
                            ev.get("message", ""),
                            ev.get("detail", ""),
                        )
        except Exception as _stream_exc:
            logger.warning("SSE stream from agent-api failed (%s) — falling back to polling",
                           _stream_exc)
            # Fallback: poll until done
            deadline = time.time() + 600
            async with httpx.AsyncClient(timeout=30.0) as client:
                while time.time() < deadline:
                    pr  = await client.get(f"{METADATA_API}/jobs/{extract_job_id}")
                    job = pr.json()
                    if job.get("status") == "done":
                        break
                    if job.get("status") == "error":
                        extraction_error = job.get("error", "Extraction failed")
                        break
                    await asyncio.sleep(2.0)

        if extraction_error:
            raise RuntimeError(extraction_error)

        # Fetch the completed report
        async with httpx.AsyncClient(timeout=30.0) as client:
            rr = await client.get(f"{METADATA_API}/jobs/{extract_job_id}/report")
            report = rr.json()

        if not report:
            raise RuntimeError("Extraction timed out")

        src["report"]      = report
        src["table_names"] = list((report.get("tables") or {}).keys())
        src["table_count"] = len(src["table_names"])

        # Auto-infer domain from table/column names if admin left it as
        # the default "Other" or blank — so concept annotation always has
        # a domain context even when none was explicitly set.
        admin_domain = (src.get("domain") or "").strip()
        if not admin_domain or admin_domain.lower() == "other":
            inferred = _infer_domain_from_report(report)
            if inferred:
                src["domain"] = inferred
                _push_index_event(source_id, "extract", "running",
                                  f"Domain auto-detected: {inferred}",
                                  "Inferred from table and column name signals in the schema")
        _push_index_event(source_id, "extract", "done",
                          f"Metadata extracted — {src['table_count']} tables found",
                          f"Tables: {', '.join(src['table_names'][:10])}" + ("…" if src['table_count'] > 10 else ""))

        # ── Persist metadata to catalog store ──────────────────────────────────
        try:
            n = _mc.persist(source_id, src.get("name", ""), report)
            logger.info("Persisted %d metadata entities for source %s", n, source_id[:8])
            # Run deterministic taxonomy inference immediately after persist
            # so taxonomy fields are populated even before the KG / LLM steps run.
            try:
                n_inf = _mc.infer_taxonomy(source_id, domain=src.get("domain", ""))
                if n_inf:
                    _push_index_event(source_id, "taxonomy", "done",
                                      f"Taxonomy inferred — {n_inf} columns classified (pattern-based)")
                    logger.info("Pattern taxonomy: %d columns for %s", n_inf, source_id[:8])
            except Exception as _ti:
                logger.warning("Pattern taxonomy inference failed for %s: %s", source_id[:8], _ti)
        except Exception as _me:
            logger.warning("Metadata persistence failed for %s: %s", source_id[:8], _me)

        # ── 2. Ontology ────────────────────────────────────────────────────────
        ontology_content = None
        _push_index_event(source_id, "ontology", "running", "Building OWL ontology…")
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                ro = await client.post(f"{ONTOLOGY_API}/generate",
                                       json={
                                           "report":             report,
                                           "include_statistics": True,
                                           "annotate_concepts":  True,
                                           "source_domain":      src.get("domain", ""),
                                           "source_name":        src.get("name", ""),
                                           "source_description": src.get("description", ""),
                                           "db_type":            src.get("db_type", ""),
                                           "ontology_name":      src.get("name", "DatabaseOntology"),
                                       })
                ro.raise_for_status()
                onto_job_id = ro.json()["job_id"]

            src["ontology_job_id"] = onto_job_id
            # Safety margin: the build runs one LLM concept-annotation call per
            # table (now parallelized in build_node), but transient API 500s +
            # retries can still stretch a large source past a few minutes.
            deadline2 = time.time() + 600
            async with httpx.AsyncClient(timeout=30.0) as client:
                while time.time() < deadline2:
                    pr  = await client.get(f"{ONTOLOGY_API}/jobs/{onto_job_id}")
                    job = pr.json()
                    if job.get("status") == "done":
                        rc = await client.get(f"{ONTOLOGY_API}/jobs/{onto_job_id}/content")
                        if rc.status_code == 200:
                            ontology_content = rc.json().get("content", "")
                        break
                    if job.get("status") == "error":
                        break
                    await asyncio.sleep(2.0)
            if ontology_content:
                src["ontology_content"] = ontology_content
                lines = ontology_content.count("\n")
                _push_index_event(source_id, "ontology", "done",
                                  f"Ontology built — {lines} lines of OWL/Turtle")
            else:
                _push_index_event(source_id, "ontology", "warn",
                                  "Ontology generation incomplete — skipping KG build")
        except Exception as exc:
            logger.warning("Source %s: ontology failed: %s", source_id[:8], exc)
            _push_index_event(source_id, "ontology", "warn",
                              f"Ontology step failed: {exc}")

        # ── 3. Knowledge Graph ─────────────────────────────────────────────────
        if ontology_content:
            _persist_neo4j = _use_neo4j()
            _kg_msg = "Translating ontology to knowledge graph"
            if _persist_neo4j:
                _kg_msg += " and persisting to Neo4j"
            _push_index_event(source_id, "kg", "running", _kg_msg + "…")
            try:
                kg_req: Dict[str, Any] = {
                    "ontology_text": ontology_content,
                    "kg_id":         source_id,
                }
                # Production only: write nodes/edges directly to Neo4j,
                # stamped with kg_id = source_id for multi-KG isolation.
                if _persist_neo4j:
                    kg_req.update({
                        "neo4j_uri":      _NEO4J_URI,
                        "neo4j_username": _NEO4J_USERNAME,
                        "neo4j_password": _NEO4J_PASSWORD,
                        "neo4j_database": _NEO4J_DATABASE,
                    })
                async with httpx.AsyncClient(timeout=30.0) as client:
                    rk = await client.post(f"{KG_API}/generate", json=kg_req)
                    rk.raise_for_status()
                    kg_job_id = rk.json()["job_id"]

                src["kg_job_id"] = kg_job_id
                # Safety margin: KG profile_node runs one LLM taxonomy call per
                # table (now parallelized); match the ontology step's window.
                deadline3 = time.time() + 600
                async with httpx.AsyncClient(timeout=30.0) as client:
                    while time.time() < deadline3:
                        pr  = await client.get(f"{KG_API}/jobs/{kg_job_id}")
                        job = pr.json()
                        if job.get("status") == "done":
                            rg = await client.get(f"{KG_API}/jobs/{kg_job_id}/graph")
                            if rg.status_code == 200:
                                graph = rg.json()
                                src["kg_nodes"] = graph.get("nodes") or []
                                src["kg_edges"] = graph.get("edges") or []
                            break
                        if job.get("status") == "error":
                            break
                        await asyncio.sleep(2.0)
                _push_index_event(source_id, "kg", "done",
                                  f"Knowledge graph built — {len(src['kg_nodes'])} nodes, {len(src['kg_edges'])} edges")

                # Sync profile_node taxonomy annotations → md_attributes
                n_tax = _sync_taxonomy_from_kg_nodes(source_id, src["kg_nodes"])
                if n_tax:
                    _push_index_event(source_id, "taxonomy", "done",
                                      f"Taxonomy annotations synced — {n_tax} columns classified")
                    logger.info("Taxonomy sync: %d columns for %s", n_tax, source_id[:8])
            except Exception as exc:
                logger.warning("Source %s: KG failed: %s", source_id[:8], exc)
                _push_index_event(source_id, "kg", "warn", f"KG build failed: {exc}")

        # ── Register KG in federation registry ────────────────────────────────
        try:
            from dialog_agent.kg_registry import upsert as _reg_upsert, KGEntry, list_all as _reg_list
            from dialog_agent.kg_bridges import run_inference_and_save as _infer_bridges
            from dialog_agent.kg_router import embed_kg_description

            entry = KGEntry(
                kg_id=src["id"],
                display_name=src["name"],
                description=src.get("description", ""),
                domain_keywords=[src.get("domain", "")],
                entity_types=list({n.get("label", "") for n in src.get("kg_nodes", []) if n.get("label")}),
                source_id=src["id"],
                source_db_type=src.get("db_type", ""),
                source_db_host=src.get("connection", {}).get("host", ""),
                source_db_name=src.get("connection", {}).get("database", ""),
                source_schema=src.get("connection", {}).get("schema_", ""),
            )
            entry.embedding = embed_kg_description(entry)
            _reg_upsert(entry)
            logger.info("KG registry: registered %s (%s)", entry.kg_id[:8], entry.display_name)

            # Run enterprise bridge inference against all other registered KGs
            for other in _reg_list():
                if other.kg_id == entry.kg_id:
                    continue
                other_src = _sources.get(other.source_id)
                if not other_src:
                    # Fall back to loading from persistent store — the source
                    # was indexed in a previous server run and lives in the DB.
                    try:
                        from_store = [s for s in _kg_store.load_all()
                                      if s["id"] == other.source_id]
                        if from_store:
                            other_src = from_store[0]
                            _sources[other.source_id] = other_src  # cache it
                            logger.info(
                                "KG bridge inference: loaded %s from store for bridge pairing",
                                other.source_id[:8],
                            )
                        else:
                            logger.warning(
                                "KG bridge inference skipped: KG %s (source %s) not found "
                                "in _sources or kg_store — re-index that source to enable bridges",
                                other.kg_id[:8], other.source_id[:8],
                            )
                    except Exception as _load_exc:
                        logger.warning(
                            "KG bridge inference: kg_store lookup failed for %s: %s",
                            other.source_id[:8], _load_exc,
                        )
                if other_src:
                    saved = _infer_bridges(
                        entry.kg_id,  src.get("kg_nodes", []),
                        other.kg_id,  other_src.get("kg_nodes", []),
                        kg_a_report   = src.get("report"),
                        kg_b_report   = other_src.get("report"),
                        kg_a_domain   = src.get("domain", ""),
                        kg_b_domain   = other_src.get("domain", ""),
                    )
                    if saved:
                        high = sum(1 for b in saved if b.enabled)
                        med  = len(saved) - high
                        logger.info(
                            "KG bridges %s↔%s: %d auto-enabled, %d queued for LLM validation",
                            entry.kg_id[:8], other.kg_id[:8], high, med,
                        )

            # Transitivity pass after all pairs are processed
            from dialog_agent.kg_inference_engine import infer_transitive_bridges
            trans = infer_transitive_bridges()
            if trans:
                logger.info("Transitive bridges: %d new bridges from A→B+B→C chains", len(trans))

        except Exception as _reg_exc:
            logger.warning("KG registry registration failed (non-fatal): %s", _reg_exc)

        src["status"]     = "ready"
        src["indexed_at"] = time.time()

        # ── Persist KG snapshot so it survives restarts ────────────────────────
        try:
            _kg_store.save(src)
            logger.info("kg_store: snapshot saved for %s", source_id[:8])
        except Exception as _ks_exc:
            logger.warning("kg_store.save failed (non-fatal): %s", _ks_exc)

        _push_index_event(source_id, "complete", "done",
                          f"Indexing complete — {src['table_count']} tables ready for querying")
        logger.info("Source %s indexed: %d tables", source_id[:8], src["table_count"])

    except Exception as exc:
        logger.exception("Source %s indexing failed", source_id[:8])
        src["status"]        = "error"
        src["error_message"] = str(exc)
        _push_index_event(source_id, "complete", "error", f"Indexing failed: {exc}")


# ── SSE generator ──────────────────────────────────────────────────────────────

async def _sse_generator(session_id: str) -> AsyncGenerator[str, None]:
    """Yields SSE-formatted lines from the session event queue."""
    q = _event_queues.get(session_id)
    if q is None:
        q = asyncio.Queue()
        _event_queues[session_id] = q

    # Replay current stage so freshly connected clients get context.
    # We do NOT replay "ready" — the frontend reads stage from GET /messages on
    # resume, so replaying here would render a duplicate Ready card on every
    # SSE reconnect (page reload, chat response, network blip, etc.).
    session = _sessions.get(session_id, {})
    stage = session.get("stage", "idle")
    if stage not in ("idle", "ready", "error"):
        yield _sse_line({
            "type":    "progress",
            "stage":   stage,
            "message": session.get("stage_message", stage),
            "pct":     session.get("pct", 0),
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
    # Prefer the React build; fall back to the legacy chat_ui.
    react_index = REACT_UI_DIR / "index.html"
    if react_index.exists():
        return FileResponse(str(react_index), headers={"Cache-Control": "no-store"})
    index = UI_DIR / "index.html"
    if not index.exists():
        return {"error": "UI not found. Build chat_frontend or ensure chat_ui/index.html exists."}
    return FileResponse(str(index), headers={"Cache-Control": "no-store"})


# ── Session endpoints ──────────────────────────────────────────────────────────

class NewSessionRequest(BaseModel):
    title:     Optional[str] = "New conversation"
    source_id: Optional[str] = None
    persona:   str           = "business_user"


@app.post("/sessions", status_code=201)
async def create_session(req: NewSessionRequest):
    s = _new_session(req.title or "New conversation", req.persona)

    if req.source_id:
        src = _sources.get(req.source_id)
        if not src:
            raise HTTPException(status_code=404, detail="Source not found")
        s["source_id"] = req.source_id
        s["title"]     = req.title or src["name"]

        if src["status"] == "ready":
            s["stage"]         = "ready"
            s["pct"]           = 100
            s["stage_message"] = "Ready"
            s["report"]        = src["report"]
            s["db_type"]       = src["db_type"]
            s["kg_nodes"]      = src.get("kg_nodes", [])
            s["kg_edges"]      = src.get("kg_edges", [])
            s["db_connection"] = src["connection"]
            conn = src["connection"]
            if src["db_type"].lower() in _FILE_BASED_TYPES:
                s["db_file_path"] = conn.get("file_path", "")
            else:
                s["db_file_path"] = None
        elif src["status"] == "indexing":
            s["stage"]         = "extracting"
            s["stage_message"] = "Source is being indexed, please wait…"

    _sessions[s["id"]] = s
    _event_queues[s["id"]] = asyncio.Queue()
    logger.info("Session created: %s (source: %s)", s["id"][:8], req.source_id or "none")
    return {
        "session_id": s["id"],
        "title":      s["title"],
        "stage":      s["stage"],
        "source_id":  req.source_id,
    }


@app.get("/sessions")
async def list_sessions(persona: Optional[str] = None):
    all_sessions = sorted(_sessions.values(), key=lambda x: x["created_at"], reverse=True)
    return [
        {
            "session_id": s["id"],
            "title":      s["title"],
            "stage":      s["stage"],
            "persona":    s.get("persona", "business_user"),
            "source_id":  s.get("source_id"),
            "files":      [f["name"] for f in s.get("files", [])],
            "created_at": s["created_at"],
            "msg_count":  len(s.get("messages", [])),
        }
        for s in all_sessions
        if persona is None or s.get("persona", "business_user") == persona
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
        upload_dir = DATA_DIR / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        if len(files) == 1 and files[0].filename.lower().endswith((".xlsx", ".xls", ".xlsm", ".xlsb")):
            # Single Excel — save locally (for dialog-api) and forward to agent-api (for extraction)
            file = files[0]
            file_bytes = await file.read()
            dest = upload_dir / file.filename
            dest.write_bytes(file_bytes)
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    await client.post(
                        f"{METADATA_API}/upload-file",
                        files={"file": (file.filename, file_bytes,
                                        file.content_type or "application/octet-stream")},
                    )
            except Exception as exc:
                logger.warning("Could not forward session upload to agent-api (non-fatal): %s", exc)
            server_path = str(dest)
            db_type     = _db_type_from_ext(file.filename)
            session["files"] = [{"name": file.filename, "server_path": server_path, "db_type": db_type}]
            session["db_file_path"] = server_path
            session["db_type"]      = db_type

        else:
            # Multiple CSVs or single CSV — save locally and forward to agent-api
            csv_dir = upload_dir / "csv"
            csv_dir.mkdir(parents=True, exist_ok=True)
            form_files = []
            for f in files:
                content = await f.read()
                (csv_dir / f.filename).write_bytes(content)
                form_files.append(("files", (f.filename, content, f.content_type or "text/csv")))
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    await client.post(f"{METADATA_API}/upload-files", files=form_files)
            except Exception as exc:
                logger.warning("Could not forward CSV upload to agent-api (non-fatal): %s", exc)
            server_path = str(csv_dir)
            db_type     = "csv"
            session["files"] = [{"name": f.filename, "server_path": str(csv_dir / f.filename), "db_type": "csv"} for f in files]
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
    message:      str
    skip_cache:   bool = False
    analyst_role: str  = ""   # functional role sent from the UI persona selector
    llm_model:    str  = ""   # synthesis model override (haiku/sonnet/opus)


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
    asyncio.create_task(_run_dialog(session_id, msg_id, req.message, req.skip_cache, req.analyst_role or "", req.llm_model or ""))

    return {"msg_id": msg_id, "session_id": session_id}


async def _run_dialog(session_id: str, msg_id: str, message: str, skip_cache: bool, analyst_role: str = "", llm_model: str = "") -> None:
    session = _sessions.get(session_id)
    if not session:
        return

    dialog_session_id = session.get("dialog_session_id")

    db_type = session.get("db_type") or "sqlite"
    conn    = session.get("db_connection") or {}
    dialog_payload: Dict[str, Any] = {
        "natural_query":   message,
        "kg_nodes":        session.get("kg_nodes") or [],
        "kg_edges":        session.get("kg_edges") or [],
        "db_type":         db_type,
        "skip_cache":      skip_cache,
        "session_id":      dialog_session_id,
        "row_limit":       500,
        "max_sql_queries": 10,
        "analyst_role":    analyst_role,
        "source_id":       session.get("source_id") or "",
        **({"llm_model": llm_model} if llm_model else {}),
    }
    if db_type.lower() in _FILE_BASED_TYPES:
        dialog_payload["db_file_path"] = session.get("db_file_path") or ""
    else:
        dialog_payload["db_host"]              = conn.get("host", "")
        dialog_payload["db_port"]              = conn.get("port", 5432)
        dialog_payload["db_name"]              = conn.get("database", "")
        dialog_payload["db_schema"]            = conn.get("schema_", "public")
        dialog_payload["db_user"]              = conn.get("username", "")
        dialog_payload["db_password"]          = conn.get("password", "")
        dialog_payload["db_connection_string"] = conn.get("connection_string", "")
        dialog_payload["db_extra"]             = conn.get("extra", {})

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

        # Internal pipeline errors (plan_node SQL generation failures, schema
        # gaps, etc.) are useful for engineers but confusing to business users.
        # Log all errors, then only expose them to analyst / admin personas.
        persona = session.get("persona", "business_user")
        if errors:
            for err in errors:
                logger.warning(
                    "plan_node error [session=%s msg=%s persona=%s]: %s",
                    session_id[:8], msg_id[:8], persona, err,
                )
        visible_errors = errors if persona in ("analyst", "admin") else []

        # Record AI message
        ai_ts = time.time()
        session["messages"].append({
            "id":            msg_id + "_ai",
            "role":          "assistant",
            "content":       insights,
            "results":       query_results,
            "sql":           sql_queries,
            "errors":        visible_errors,
            "ts":            ai_ts,
        })

        await _push(session_id, {
            "type":          "chat_response",
            "msg_id":        msg_id,
            "content":       insights,
            "results":       query_results,
            "sql":           sql_queries,
            "errors":        visible_errors,
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


# ── Source Registry endpoints ──────────────────────────────────────────────────

@app.get("/sources")
async def list_sources(persona: str = "business_user"):
    """List all data sources visible to a given persona (credentials never returned)."""
    return [
        _source_public(s) for s in _sources.values()
        if persona in s.get("persona_access", ["business_user", "analyst", "admin"])
    ]


@app.post("/sources", status_code=201)
async def create_source(req: CreateSourceRequest):
    """Register a new data source and optionally start background indexing."""
    s = _new_source(
        name=req.name,
        description=req.description,
        domain=req.domain,
        db_type=req.db_type,
        connection=req.connection.dict(),
        persona_access=req.persona_access,
    )
    _sources[s["id"]] = s
    try:
        _kg_store.save(s)
    except Exception as _ks_exc:
        logger.warning("kg_store.save on register failed (non-fatal): %s", _ks_exc)
    if req.auto_index:
        asyncio.create_task(_index_source(s["id"]))
    logger.info("Source registered: %s (%s)", s["id"][:8], req.name)
    return _source_public(s)


@app.get("/sources/{source_id}")
async def get_source(source_id: str):
    """Get a single source by ID (without credentials)."""
    s = _sources.get(source_id)
    if not s:
        raise HTTPException(status_code=404, detail="Source not found")
    return _source_public(s)


class UpdateSourceRequest(BaseModel):
    name:        Optional[str] = None
    description: Optional[str] = None
    domain:      Optional[str] = None


@app.patch("/sources/{source_id}", status_code=200)
async def update_source(source_id: str, req: UpdateSourceRequest):
    """Update editable fields (name, description, domain) on an existing source."""
    s = _sources.get(source_id)
    if not s:
        raise HTTPException(status_code=404, detail="Source not found")
    if req.name is not None:
        s["name"] = req.name.strip()
    if req.description is not None:
        s["description"] = req.description.strip()
    if req.domain is not None:
        s["domain"] = req.domain.strip()
    try:
        _kg_store.save(s)
    except Exception as _ks_exc:
        logger.warning("kg_store.save on update failed (non-fatal): %s", _ks_exc)
    logger.info("Source updated: %s — domain=%s", source_id[:8], s.get("domain"))
    return _source_public(s)


@app.delete("/sources/{source_id}", status_code=200)
async def delete_source(source_id: str):
    """Remove a registered source and purge any associated uploaded file."""
    s = _sources.pop(source_id, None)
    if not s:
        raise HTTPException(status_code=404, detail="Source not found")
    # Purge persistent source file from the metadata API if it was uploaded
    file_path = s.get("connection", {}).get("file_path", "")
    if file_path and s.get("connection", {}).get("uploaded"):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.request("DELETE", f"{METADATA_API}/uploads/purge",
                                     json={"path": file_path})
        except Exception as exc:
            logger.warning("Could not purge source file %s: %s", file_path, exc)
    # Remove from KG snapshot store
    try:
        _kg_store.delete(source_id)
    except Exception as exc:
        logger.warning("kg_store.delete failed (non-fatal): %s", exc)
    # Remove from KG federation registry so stale entries don't block bridge
    # inference — without this, _reg_list() keeps returning the deleted source
    # and _sources.get(other.source_id) silently returns None, causing all
    # bridge inference calls for that KG pair to be skipped forever.
    try:
        from dialog_agent.kg_registry import delete as _reg_delete
        _reg_delete(source_id)
    except Exception as exc:
        logger.warning("kg_registry.delete failed (non-fatal): %s", exc)
    logger.info("Source deleted: %s (%s)", source_id[:8], s["name"])
    return {"deleted": source_id}


def _db_type_from_ext(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in (".xlsx", ".xls", ".xlsm", ".xlsb"):
        return "excel"
    if ext in (".sqlite", ".db", ".sqlite3"):
        return "sqlite"
    if ext == ".csv":
        return "csv"
    return "unknown"


@app.post("/sources/upload-file", status_code=200)
async def upload_source_file(file: UploadFile = File(...)):
    """
    Upload a file for a registered source.
    Saves to the orchestrator's DATA_DIR/uploads/ (for dialog-api chat queries)
    AND forwards to agent-api's /upload-permanent (for extraction).
    Both pods need the file; the shared PVC gives each pod its own mount binding.
    """
    file_bytes = await file.read()
    upload_dir = DATA_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / file.filename
    try:
        dest.write_bytes(file_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Local save failed: {exc}")

    # Forward to agent-api so it has the file for extraction
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            await client.post(
                f"{METADATA_API}/upload-permanent",
                files={"file": (file.filename, file_bytes,
                                file.content_type or "application/octet-stream")},
            )
    except Exception as exc:
        logger.warning("Could not forward upload to agent-api (non-fatal): %s", exc)

    db_type = _db_type_from_ext(file.filename)
    return {"path": str(dest), "db_type": db_type, "filename": file.filename}


@app.post("/sources/{source_id}/reindex", status_code=202)
async def reindex_source(source_id: str):
    """Trigger a fresh indexing run for a registered source."""
    s = _sources.get(source_id)
    if not s:
        raise HTTPException(status_code=404, detail="Source not found")
    if s["status"] == "indexing":
        raise HTTPException(status_code=409, detail="Source is already being indexed")
    s["status"]        = "idle"
    s["error_message"] = None
    asyncio.create_task(_index_source(source_id))
    return {"source_id": source_id, "status": "indexing"}


@app.post("/sources/test-connection")
async def test_source_connection(req: TestConnectionRequest):
    """Test a database connection without registering or saving it."""
    conn      = req.connection.dict()
    db_config = _build_db_config(req.db_type, conn)
    try:
        # 60s: Snowflake key-pair connects can take ~45s on first OCSP validation.
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(f"{METADATA_API}/discover", json=db_config)
            if r.status_code == 422:
                return {"ok": False, "error": str(r.json().get("detail", "Invalid parameters"))}
            if r.status_code == 503:
                return {"ok": False, "error": r.json().get("detail", "Could not connect to database")}
            r.raise_for_status()
        return {"ok": True}
    except httpx.HTTPStatusError as exc:
        return {"ok": False, "error": f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ── Source detail endpoints (KG / Ontology / SSE) ─────────────────────────────

@app.get("/sources/{source_id}/graph")
async def get_source_graph(source_id: str):
    """Return the KG graph data (nodes + edges) for an indexed source."""
    s = _sources.get(source_id)
    if not s:
        raise HTTPException(status_code=404, detail="Source not found")
    return {
        "nodes": s.get("kg_nodes") or [],
        "edges": s.get("kg_edges") or [],
    }


@app.get("/sources/{source_id}/ontology")
async def get_source_ontology(source_id: str):
    """Return the OWL/Turtle ontology text for an indexed source."""
    s = _sources.get(source_id)
    if not s:
        raise HTTPException(status_code=404, detail="Source not found")
    content = s.get("ontology_content") or ""
    return {"content": content}


class SaveOntologyRequest(BaseModel):
    content: str
    rebuild_kg: bool = True


@app.post("/sources/{source_id}/ontology")
async def save_source_ontology(source_id: str, req: SaveOntologyRequest):
    """
    Save an edited ontology for a source and optionally rebuild the KG.
    Triggers a KG rebuild in the background when rebuild_kg=True.
    """
    s = _sources.get(source_id)
    if not s:
        raise HTTPException(status_code=404, detail="Source not found")
    s["ontology_content"] = req.content
    if req.rebuild_kg:
        asyncio.create_task(_rebuild_kg(source_id, req.content))
    return {"saved": True, "rebuild_kg": req.rebuild_kg}


class ValidateOntologyRequest(BaseModel):
    """
    Optional override — if omitted the saved ontology for this source is used.
    Allows the UI to validate unsaved edits before saving.
    """
    ontology_text:      Optional[str] = None
    min_coverage:       float = 0.5
    check_orphan_classes: bool = True
    check_namespace:    bool  = True
    extra_shapes_ttl:   str   = ""


@app.post("/sources/{source_id}/validate-ontology")
async def validate_source_ontology(source_id: str, req: ValidateOntologyRequest):
    """
    Validate the OWL/RDF ontology for a source using the SHACL Validation Service.

    This is a purely read-only, opt-in endpoint — it does not modify any state.
    It delegates to the standalone SHACL API (port 8006) so the existing
    ontology / KG pipeline is completely unaffected.

    Workflow:
      1. Resolves ontology text (from request body or stored source ontology).
      2. POSTs to SHACL API /validate and polls until done (max 60 s).
      3. Returns the full validation report:
           quality          : "PASS" | "WARN" | "FAIL"
           violations        : list of SHACL sh:Violation entries
           warnings          : list of SHACL sh:Warning entries
           semantic_issues   : orphan classes, low-coverage edges, namespace drift, …
           ontology_stats    : {classes, datatype_props, object_props, triples}
           suggestions       : actionable remediation suggestions
    """
    s = _sources.get(source_id)
    if not s:
        raise HTTPException(status_code=404, detail="Source not found")

    ontology_text = req.ontology_text or s.get("ontology_content") or ""
    if not ontology_text:
        raise HTTPException(
            status_code=409,
            detail="No ontology available for this source. Run indexing first.",
        )

    payload = {
        "ontology_text":       ontology_text,
        "ontology_format":     "auto",
        "min_coverage":        req.min_coverage,
        "check_orphan_classes": req.check_orphan_classes,
        "check_namespace":     req.check_namespace,
        "extra_shapes_ttl":    req.extra_shapes_ttl,
    }

    try:
        # Submit
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(f"{SHACL_API}/validate", json=payload)
            r.raise_for_status()
            job_id = r.json()["job_id"]

        # Poll (SHACL validation is fast — typically < 5 s)
        deadline = time.time() + 60
        async with httpx.AsyncClient(timeout=15.0) as client:
            while time.time() < deadline:
                pr  = await client.get(f"{SHACL_API}/jobs/{job_id}/report")
                if pr.status_code == 200:
                    return pr.json()
                if pr.status_code == 409:
                    # Still running — back off briefly
                    await asyncio.sleep(1)
                    continue
                pr.raise_for_status()

        raise HTTPException(status_code=504, detail="SHACL validation timed out.")

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=(
                "SHACL Validation Service is not reachable. "
                "Start it with: python shacl_api.py  (port 8006)"
            ),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("validate_source_ontology: unexpected error")
        raise HTTPException(status_code=500, detail=str(exc))


class KGPreviewRequest(BaseModel):
    ontology_text: str


@app.post("/sources/{source_id}/kg-preview")
async def preview_source_kg(source_id: str, req: KGPreviewRequest):
    """
    Generate a KG preview from the provided ontology text without saving.
    Calls the KG API synchronously and returns {nodes, edges}.
    """
    s = _sources.get(source_id)
    if not s:
        raise HTTPException(status_code=404, detail="Source not found")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            rk = await client.post(f"{KG_API}/generate",
                                   json={"ontology_text": req.ontology_text})
            rk.raise_for_status()
            kg_job_id = rk.json()["job_id"]

        deadline = time.time() + 120
        async with httpx.AsyncClient(timeout=30.0) as client:
            while time.time() < deadline:
                pr  = await client.get(f"{KG_API}/jobs/{kg_job_id}")
                job = pr.json()
                if job.get("status") == "done":
                    rg = await client.get(f"{KG_API}/jobs/{kg_job_id}/graph")
                    if rg.status_code == 200:
                        return rg.json()
                    return {"nodes": [], "edges": []}
                if job.get("status") == "error":
                    raise RuntimeError(job.get("error", "KG generation failed"))
                await asyncio.sleep(1.5)
        raise RuntimeError("KG preview timed out")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/sources/{source_id}/index-events")
async def source_index_events(source_id: str):
    """
    SSE stream of indexing step events for a source.
    Replays the stored event log first, then streams live events.
    """
    s = _sources.get(source_id)
    if not s:
        raise HTTPException(status_code=404, detail="Source not found")

    async def _gen() -> AsyncGenerator[str, None]:
        # Replay buffered events so reconnecting clients catch up
        for ev in list(_source_event_log.get(source_id, [])):
            yield _sse_line(ev)

        # Attach a live queue
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        _source_event_queues[source_id] = q

        heartbeat_interval = 20
        last_hb = time.time()
        try:
            while True:
                try:
                    ev = q.get_nowait()
                    yield _sse_line(ev)
                    # Stop streaming once indexing is complete or errored
                    if ev.get("step") == "complete":
                        break
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.3)
                    if time.time() - last_hb >= heartbeat_interval:
                        yield _sse_line({"type": "heartbeat"})
                        last_hb = time.time()
        finally:
            # Remove queue so _push_index_event stops trying to write to it
            if _source_event_queues.get(source_id) is q:
                del _source_event_queues[source_id]

    return StreamingResponse(_gen(), media_type="text/event-stream")


class ExecuteSQLRequest(BaseModel):
    sql:   str
    limit: int = 500


@app.post("/sources/{source_id}/execute-sql")
async def execute_sql_for_source(source_id: str, req: ExecuteSQLRequest):
    """
    Execute a raw SQL statement against a registered source's database.
    Proxies to the dialog-api /execute-sql endpoint which uses the full
    connector layer (postgres, redshift, oracle, sqlserver, sqlite, csv…).
    Used by the tech UI SQL console — no LLM involved.
    """
    s = _sources.get(source_id)
    if not s:
        raise HTTPException(status_code=404, detail="Source not found")

    conn = s.get("connection") or {}
    payload: Dict[str, Any] = {
        "sql":      req.sql,
        "limit":    req.limit,
        "db_type":  s["db_type"],
    }
    if s["db_type"].lower() in _FILE_BASED_TYPES:
        payload["db_file_path"] = conn.get("file_path", "")
    else:
        payload.update({
            "db_host":     conn.get("host", ""),
            "db_port":     conn.get("port", 5432),
            "db_name":     conn.get("database", ""),
            "db_schema":   conn.get("schema_", "") or conn.get("schema", "public"),
            "db_user":     conn.get("username", ""),
            "db_password": conn.get("password", ""),
        })
        if conn.get("connection_string"):
            payload["db_connection_string"] = conn["connection_string"]

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(f"{DIALOG_API}/execute-sql", json=payload)
            if r.status_code == 400:
                raise HTTPException(status_code=400, detail=r.json().get("detail", "SQL error"))
            r.raise_for_status()
            return r.json()
    except HTTPException:
        raise
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Dialog API error: {exc.response.text[:200]}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class GraphRAGQueryRequest(BaseModel):
    query:      str
    top_k:      int = 8
    hop_depth:  int = 2


@app.post("/sources/{source_id}/graphrag-query")
async def graphrag_query(source_id: str, req: GraphRAGQueryRequest):
    """
    Run in-memory GraphRAG retrieval against the source's KG nodes and return
    seed nodes (with cosine scores) and BFS-expanded node IDs so the UI can
    highlight them in the KG explorer.
    """
    s = _sources.get(source_id)
    if not s:
        raise HTTPException(status_code=404, detail="Source not found")

    nodes: List[Dict] = s.get("kg_nodes") or []
    edges: List[Dict] = s.get("kg_edges") or []

    if not nodes:
        raise HTTPException(status_code=422, detail="Source has no KG nodes — index it first")
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="Query must not be empty")

    def _run() -> Dict:
        import numpy as np

        backend = _gr_detect_backend("auto")
        s_hash  = _gr_schema_hash(nodes)
        key     = f"{s_hash}:{backend}"

        if key not in _GR_EMBED_CACHE:
            _GR_EMBED_CACHE[key] = _gr_build_cache(nodes, backend, s_hash)

        cache  = _GR_EMBED_CACHE[key]
        q_vec  = cache.embed_query(req.query)
        scores = cache.matrix @ q_vec          # cosine similarity [N]

        k           = min(req.top_k, len(nodes))
        top_indices = scores.argsort()[::-1][:k]
        seed_ids    = [cache.node_ids[i] for i in top_indices]

        adjacency    = _gr_build_adjacency(edges)
        expanded_ids = set(_gr_bfs_expand(seed_ids, adjacency, req.hop_depth))

        seed_nodes = [
            {
                "id":    cache.node_ids[i],
                "label": nodes[i].get("label", cache.node_ids[i]),
                "score": float(scores[i]),
            }
            for i in top_indices
        ]

        return {
            "backend":         backend,
            "seed_nodes":      seed_nodes,
            "expanded_ids":    list(expanded_ids - set(seed_ids)),
            "total_nodes":     len(nodes),
            "retrieved_nodes": len(expanded_ids),
        }

    try:
        result = await asyncio.to_thread(_run)
        return result
    except Exception as exc:
        logger.exception("graphrag_query failed for source %s", source_id[:8])
        raise HTTPException(status_code=500, detail=str(exc))


async def _rebuild_kg(source_id: str, ontology_text: str) -> None:
    """Rebuild the KG for a source from a (possibly edited) ontology."""
    src = _sources.get(source_id)
    if not src:
        return
    _push_index_event(source_id, "kg", "running",
                      "Rebuilding knowledge graph from updated ontology…")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            rk = await client.post(f"{KG_API}/generate",
                                   json={"ontology_text": ontology_text})
            rk.raise_for_status()
            kg_job_id = rk.json()["job_id"]

        deadline = time.time() + 300
        async with httpx.AsyncClient(timeout=30.0) as client:
            while time.time() < deadline:
                pr  = await client.get(f"{KG_API}/jobs/{kg_job_id}")
                job = pr.json()
                if job.get("status") == "done":
                    rg = await client.get(f"{KG_API}/jobs/{kg_job_id}/graph")
                    if rg.status_code == 200:
                        graph = rg.json()
                        src["kg_nodes"] = graph.get("nodes") or []
                        src["kg_edges"] = graph.get("edges") or []
                    break
                if job.get("status") == "error":
                    raise RuntimeError(job.get("error", "KG build failed"))
                await asyncio.sleep(2.0)

        _push_index_event(source_id, "kg", "done",
                          f"KG rebuilt — {len(src['kg_nodes'])} nodes, {len(src['kg_edges'])} edges")
        _push_index_event(source_id, "complete", "done", "KG rebuild complete")
    except Exception as exc:
        logger.warning("Source %s: KG rebuild failed: %s", source_id[:8], exc)
        _push_index_event(source_id, "kg", "error", f"KG rebuild failed: {exc}")
        _push_index_event(source_id, "complete", "error", f"KG rebuild failed: {exc}")


# ── KG Federation admin endpoints ──────────────────────────────────────────────

@app.get("/kg-registry")
async def orc_list_kg_registry():
    """List all registered Knowledge Graphs."""
    try:
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
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/kg-bridges")
async def orc_list_kg_bridges(enabled_only: bool = False):
    """List all cross-KG bridges."""
    try:
        from dialog_agent.kg_bridges import list_all as _list_bridges
        return [b.to_dict() for b in _list_bridges(enabled_only=enabled_only)]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class OrcBridgeCreateRequest(BaseModel):
    from_kg:     str
    from_column: str
    to_kg:       str
    to_column:   str
    from_entity: str   = ""
    to_entity:   str   = ""
    join_type:   str   = "FK"
    notes:       str   = ""


@app.post("/kg-bridges", status_code=201)
async def orc_create_kg_bridge(req: OrcBridgeCreateRequest):
    """Create a declared bridge between two KGs."""
    try:
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
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class OrcBridgeUpdateRequest(BaseModel):
    enabled:  Optional[bool] = None
    notes:    Optional[str]  = None
    promote:  bool           = False


@app.put("/kg-bridges/{bridge_id}")
async def orc_update_kg_bridge(bridge_id: int, req: OrcBridgeUpdateRequest):
    """Update a bridge: enable/disable, update notes, or promote to declared."""
    try:
        import dialog_agent.kg_bridges as _kb
        from dialog_agent.kg_bridges import (
            get_by_id as _get_bridge,
            set_enabled as _set_enabled,
            promote_to_declared as _promote,
        )
        b = _get_bridge(bridge_id)
        if not b:
            raise HTTPException(status_code=404, detail="Bridge not found")
        if req.enabled is not None:
            _set_enabled(bridge_id, req.enabled)
        if req.notes is not None:
            with _kb._conn() as c:
                c.execute("UPDATE kg_bridges SET notes=?,updated_at=? WHERE id=?",
                          (req.notes, time.time(), bridge_id))
        if req.promote:
            _promote(bridge_id)
        updated = _get_bridge(bridge_id)
        return updated.to_dict() if updated else {}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/kg-bridges/{bridge_id}", status_code=200)
async def orc_delete_kg_bridge(bridge_id: int):
    """Delete a bridge."""
    try:
        from dialog_agent.kg_bridges import get_by_id as _get_bridge, delete as _del_bridge
        b = _get_bridge(bridge_id)
        if not b:
            raise HTTPException(status_code=404, detail="Bridge not found")
        _del_bridge(bridge_id)
        return {"deleted": bridge_id}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/sources/{source_id}/dialog-config")
async def get_source_dialog_config(source_id: str):
    """
    Return connection details for building a DialogConfig for this source.
    Used by dialog_api multi-KG routing to get per-KG connection info.
    """
    s = _sources.get(source_id)
    if not s:
        raise HTTPException(status_code=404, detail="Source not found")
    conn = s.get("connection") or {}
    return {
        "kg_id":      source_id,
        "db_type":    s.get("db_type", ""),
        "host":       conn.get("host", ""),
        "port":       conn.get("port", 0),
        "db_name":    conn.get("database", ""),
        "schema":     conn.get("schema_", "public"),
        "username":   conn.get("username", ""),
        "password":   conn.get("password", ""),
        "file_path":  conn.get("file_path", ""),
    }


# ── Metadata Catalog endpoints (Data Manager persona) ─────────────────────────

class UpdateEntityRequest(BaseModel):
    description:      Optional[str]  = None
    is_golden_record: Optional[bool] = None


class UpdateAttributeRequest(BaseModel):
    description:      Optional[str]  = None
    is_golden_record: Optional[bool] = None
    statistical_type: Optional[str]  = None
    semantic_role:    Optional[str]  = None
    taxonomy_tree:    Optional[Any]  = None


@app.get("/metadata/entities")
async def list_metadata_entities(source_id: Optional[str] = None):
    """List persisted metadata entities, optionally filtered by source."""
    try:
        return _mc.list_entities(source_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/metadata/entities/{metadata_id}")
async def get_metadata_entity(metadata_id: str):
    """Get a single metadata entity with its attributes."""
    try:
        entity = _mc.get_entity(metadata_id)
        if not entity:
            raise HTTPException(status_code=404, detail="Entity not found")
        return entity
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.patch("/metadata/entities/{metadata_id}")
async def patch_metadata_entity(metadata_id: str, req: UpdateEntityRequest):
    """Update entity description and/or golden-record flag."""
    try:
        kwargs: Dict[str, Any] = {}
        if req.description is not None:
            kwargs["description"] = req.description
        if req.is_golden_record is not None:
            kwargs["is_golden_record"] = req.is_golden_record
        if not kwargs:
            raise HTTPException(status_code=400, detail="No updatable fields provided")
        _mc.update_entity(metadata_id, **kwargs)
        updated = _mc.get_entity(metadata_id)
        if not updated:
            raise HTTPException(status_code=404, detail="Entity not found")
        updated.pop("attributes", None)
        return updated
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.patch("/metadata/attributes/{attr_id}")
async def patch_metadata_attribute(attr_id: str, req: UpdateAttributeRequest):
    """Update attribute fields: description, golden-record, statistical_type, semantic_role, taxonomy_tree."""
    try:
        kwargs: Dict[str, Any] = {}
        if req.description is not None:
            kwargs["description"] = req.description
        if req.is_golden_record is not None:
            kwargs["is_golden_record"] = req.is_golden_record
        if req.statistical_type is not None:
            kwargs["statistical_type"] = req.statistical_type
        if req.semantic_role is not None:
            kwargs["semantic_role"] = req.semantic_role
        if req.taxonomy_tree is not None:
            kwargs["taxonomy_tree"] = req.taxonomy_tree
        if not kwargs:
            raise HTTPException(status_code=400, detail="No updatable fields provided")
        _mc.update_attribute(attr_id, **kwargs)
        row = _mc.get_attribute(attr_id)
        if not row:
            raise HTTPException(status_code=404, detail="Attribute not found")
        return row
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/metadata/sources/{source_id}/enrich-taxonomy", status_code=202)
async def enrich_source_taxonomy(source_id: str, background_tasks: BackgroundTasks):
    """
    Trigger taxonomy enrichment for all columns of a metadata source.
    Runs pattern-based inference first (always works), then LLM enrichment.
    Runs in the background; returns immediately with a status message.
    """
    async def _run_enrichment(sid: str) -> None:
        src_domain = (_sources.get(sid) or {}).get("domain", "")
        # Step 1: deterministic pattern-based inference (no LLM, always runs)
        try:
            n_pat = _mc.infer_taxonomy(sid, domain=src_domain)
            logger.info("Enrich: pattern taxonomy %d columns for %s", n_pat, sid[:8])
        except Exception as exc:
            logger.warning("Enrich: pattern taxonomy failed for %s: %s", sid[:8], exc)
        # Step 2: LLM enrichment — overwrites/improves the pattern classifications
        try:
            n_llm = _mc.enrich_taxonomy(sid, domain=src_domain)
            logger.info("Enrich: LLM taxonomy %d columns for %s", n_llm, sid[:8])
        except Exception as exc:
            logger.warning("Enrich: LLM taxonomy failed for %s: %s", sid[:8], exc)
        # Step 3: PII classification (deterministic + LLM)
        try:
            n_pii = _mc.classify_pii(sid)
            logger.info("Enrich: PII classification %d columns for %s", n_pii, sid[:8])
        except Exception as exc:
            logger.warning("Enrich: PII classification failed for %s: %s", sid[:8], exc)

    background_tasks.add_task(_run_enrichment, source_id)
    return {"status": "accepted", "source_id": source_id,
            "message": "Taxonomy enrichment started in background"}


@app.post("/metadata/sources/{source_id}/classify-pii", status_code=202)
async def classify_source_pii(source_id: str, background_tasks: BackgroundTasks):
    """
    Run PII classification for all columns of a metadata source.
    Stage 1: deterministic regex on column names and sample values.
    Stage 2: LLM review for ambiguous columns that have sample data.
    Runs in background; returns immediately.
    """
    s = _sources.get(source_id)
    if not s:
        raise HTTPException(status_code=404, detail="Source not found")

    async def _run_pii(sid: str) -> None:
        try:
            n = _mc.classify_pii(sid)
            logger.info("PII classify: %d columns for %s", n, sid[:8])
        except Exception as exc:
            logger.warning("PII classify failed for %s: %s", sid[:8], exc)

    background_tasks.add_task(_run_pii, source_id)
    return {"status": "accepted", "source_id": source_id,
            "message": "PII classification started in background"}


@app.get("/metadata/redundancies")
async def list_metadata_redundancies(source_id: Optional[str] = None):
    """List entity pairs with >= 90% column overlap (potential redundant sources)."""
    try:
        return _mc.list_redundancies(source_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/metadata/changes")
async def list_metadata_changes(
    entity_id: Optional[str] = None,
    source_id: Optional[str] = None,
    limit: int = 200,
):
    """Return the CDC change log, newest first."""
    try:
        return _mc.list_changes(entity_id=entity_id, source_id=source_id, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/metadata/sources")
async def list_metadata_sources():
    """List all registered metadata sources with domain assignments and entity counts."""
    try:
        return _mc.list_sources()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class UpdateSourceRequest(BaseModel):
    domain:      Optional[str] = None
    description: Optional[str] = None


@app.patch("/metadata/sources/{source_id}")
async def patch_metadata_source(source_id: str, req: UpdateSourceRequest):
    """Update domain and/or description of a metadata source."""
    try:
        kwargs: Dict[str, Any] = {}
        if req.domain is not None:
            kwargs["domain"] = req.domain.strip()
        if req.description is not None:
            kwargs["description"] = req.description
        if not kwargs:
            raise HTTPException(status_code=400, detail="No updatable fields provided")
        _mc.update_source(source_id, **kwargs)
        sources = _mc.list_sources()
        updated = next((s for s in sources if s["source_id"] == source_id), None)
        if not updated:
            raise HTTPException(status_code=404, detail="Source not found")
        return updated
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── BI Manager — KPI endpoints ────────────────────────────────────────────────

class CreateKpiRequest(BaseModel):
    name:           str
    description:    Optional[str]  = None
    category:       Optional[str]  = None
    source_id:      Optional[str]  = None
    nl_formula:     Optional[str]  = None
    sql_expression: Optional[str]  = None
    unit:           Optional[str]  = None
    direction:      Optional[str]  = None
    status:         Optional[str]  = None


class UpdateKpiRequest(BaseModel):
    name:           Optional[str]  = None
    description:    Optional[str]  = None
    category:       Optional[str]  = None
    source_id:      Optional[str]  = None
    nl_formula:     Optional[str]  = None
    sql_expression: Optional[str]  = None
    unit:           Optional[str]  = None
    direction:      Optional[str]  = None
    status:         Optional[str]  = None
    changed_by:     Optional[str]  = None
    change_note:    Optional[str]  = None


@app.get("/kpis")
async def list_kpis(
    source_id: Optional[str] = None,
    category:  Optional[str] = None,
    status:    Optional[str] = None,
):
    """List KPI definitions, optionally filtered."""
    try:
        return _kpi.list_kpis(source_id=source_id, category=category, status=status)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/kpis", status_code=201)
async def create_kpi(req: CreateKpiRequest):
    """Create a new KPI definition. Returns {kpi, warnings}."""
    try:
        result = _kpi.create_kpi(**req.dict(exclude_none=True))
        return result  # {"kpi": {...}, "warnings": [...]}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/kpis/{kpi_id}")
async def get_kpi(kpi_id: str):
    """Get a single KPI definition."""
    try:
        row = _kpi.get_kpi(kpi_id)
        if not row:
            raise HTTPException(status_code=404, detail="KPI not found")
        return row
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.patch("/kpis/{kpi_id}")
async def update_kpi(kpi_id: str, req: UpdateKpiRequest):
    """Update fields on a KPI definition. Returns {kpi, warnings}."""
    try:
        payload = req.dict(exclude_none=True)
        changed_by  = payload.pop("changed_by", "") or ""
        change_note = payload.pop("change_note", "") or ""
        if not payload:
            raise HTTPException(status_code=400, detail="No updatable fields provided")
        result = _kpi.update_kpi(kpi_id, changed_by=changed_by,
                                  change_note=change_note, **payload)
        if not result.get("ok"):
            raise HTTPException(status_code=404, detail="KPI not found")
        row = _kpi.get_kpi(kpi_id)
        return {"kpi": row, "warnings": result.get("warnings", [])}
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/kpis/{kpi_id}", status_code=204)
async def delete_kpi(kpi_id: str):
    """Delete a KPI definition."""
    try:
        _kpi.delete_kpi(kpi_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class CompileFormulaRequest(BaseModel):
    column_context: str   # e.g. "rsv_value REAL, fiscal_year TEXT, brand TEXT, …"
    model:          Optional[str] = None


@app.post("/kpis/{kpi_id}/compile")
async def compile_kpi_formula(kpi_id: str, req: CompileFormulaRequest):
    """
    Use a lightweight LLM (Haiku) to compile the KPI's natural language formula
    into a SQL expression. Saves and returns the compiled expression.
    """
    try:
        expr = _kpi.compile_formula(kpi_id, req.column_context, req.model)
        return {"kpi_id": kpi_id, "sql_expression": expr}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/kpis/{kpi_id}/versions")
async def list_kpi_versions(kpi_id: str):
    """Return version history for a KPI, newest first."""
    try:
        return _kpi.list_kpi_versions(kpi_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class RollbackRequest(BaseModel):
    changed_by: Optional[str] = None


@app.post("/kpis/{kpi_id}/versions/{version_num}/rollback")
async def rollback_kpi_version(kpi_id: str, version_num: int,
                               req: RollbackRequest = RollbackRequest()):
    """Restore a KPI to a previous snapshot. Returns the restored KPI dict."""
    try:
        restored = _kpi.rollback_kpi_version(kpi_id, version_num,
                                              changed_by=req.changed_by or "")
        if not restored:
            raise HTTPException(status_code=404,
                                detail=f"Version {version_num} not found for KPI {kpi_id}")
        return restored
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class ActivateKpiRequest(BaseModel):
    approved_by: Optional[str] = None


@app.post("/kpis/{kpi_id}/activate")
async def activate_kpi(kpi_id: str, req: ActivateKpiRequest = ActivateKpiRequest()):
    """
    Activation gate: validates guardrails then sets status → active.
    Returns {ok, kpi} on success; 422 with reasons on guardrail failure.
    """
    try:
        kpi = _kpi.activate_kpi(kpi_id, approved_by=req.approved_by or "")
        return {"ok": True, "kpi": kpi}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── KG Bridge Inference endpoints ─────────────────────────────────────────────

class InferRequest(BaseModel):
    kg_ids:               Optional[List[str]] = None  # subset; None = all KGs
    run_tier2_embeddings: bool  = True
    run_tier3_llm:        bool  = True
    auto_enable_threshold: float = 0.80
    min_confidence:        float = 0.55
    cross_domain_penalty:  float = 0.08


@app.post("/kg-bridges/infer", status_code=202)
async def trigger_bridge_inference(req: InferRequest, background_tasks: BackgroundTasks):
    """
    Trigger enterprise bridge inference across all (or a subset of) registered KGs.
    Runs in the background; returns immediately with a summary of the KGs involved.

    Tiers run:
      1 — structural (exact + normalised name, type compatibility, cardinality)
      2 — semantic embedding similarity          (if run_tier2_embeddings=true)
      3 — async LLM validation of medium-conf    (if run_tier3_llm=true)
    followed by a transitivity pass (A→B + B→C → A→C).
    """
    try:
        from dialog_agent.kg_registry import list_all as _reg_list
        from dialog_agent.kg_inference_engine import (
            KGContext, InferenceOptions, run_all_pairs,
        )
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Import error: {exc}")

    all_entries = _reg_list()
    if req.kg_ids:
        wanted = set(req.kg_ids)
        all_entries = [e for e in all_entries if e.kg_id in wanted]

    if len(all_entries) < 2:
        return {
            "status": "skipped",
            "reason": "fewer than 2 KGs registered" + (
                f" matching ids {req.kg_ids}" if req.kg_ids else ""
            ),
            "kg_count": len(all_entries),
        }

    # Build KGContext for each entry
    contexts: List[KGContext] = []
    for entry in all_entries:
        src = _sources.get(entry.source_id) or {}
        contexts.append(KGContext(
            kg_id        = entry.kg_id,
            display_name = entry.display_name,
            domain       = (entry.domain_keywords or [""])[0],
            nodes        = src.get("kg_nodes") or [],
            report       = src.get("report"),
        ))

    opts = InferenceOptions(
        min_confidence        = req.min_confidence,
        auto_enable_threshold = req.auto_enable_threshold,
        cross_domain_penalty  = req.cross_domain_penalty,
        run_tier2_embeddings  = req.run_tier2_embeddings,
        run_tier3_llm         = req.run_tier3_llm,
    )

    def _run():
        result = run_all_pairs(contexts, opts)
        logger.info(
            "Manual inference job complete: %d pairs, %d bridges saved",
            result["pairs_processed"], result["bridges_saved"],
        )

    background_tasks.add_task(_run)

    return {
        "status":    "accepted",
        "kg_count":  len(contexts),
        "kg_ids":    [c.kg_id for c in contexts],
        "pairs":     len(contexts) * (len(contexts) - 1) // 2,
        "options":   {
            "tier2_embeddings":    req.run_tier2_embeddings,
            "tier3_llm":           req.run_tier3_llm,
            "auto_enable_at":      req.auto_enable_threshold,
            "min_confidence":      req.min_confidence,
            "cross_domain_penalty": req.cross_domain_penalty,
        },
    }


@app.post("/kg-bridges/transitivity", status_code=200)
async def run_bridge_transitivity():
    """
    Run the transitivity pass over all existing high-confidence bridges.
    A→B + B→C  →  infer A→C candidate.
    Returns the list of newly created transitive bridges.
    """
    try:
        from dialog_agent.kg_inference_engine import infer_transitive_bridges
        saved = infer_transitive_bridges()
        return {
            "bridges_created": len(saved),
            "bridges": [b.to_dict() for b in saved],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Ontology Enrichment endpoints ────────────────────────────────────────────

@app.post("/ontology/enrich", status_code=202)
async def trigger_ontology_enrichment(background_tasks: BackgroundTasks):
    """
    Run the full ontology enrichment pipeline in the background:
      1. Glossary terms → md_attributes (semantic_role / description annotation)
      2. Active KPI sql_expressions → KPI→attribute link table
      3. Inject GLOSSARY_CONCEPT + KPI_METRIC nodes+edges into all KG snapshots
    Returns immediately with a run_id; poll GET /ontology/enrich/status for progress.
    """
    def _run():
        try:
            _enricher.run_enrichment()
            # Refresh in-memory KG snapshots so the graph UI sees the new nodes
            # without requiring a container restart.
            for src in _kg_store.load_all():
                if src["id"] in _sources:
                    _sources[src["id"]]["kg_nodes"] = src.get("kg_nodes") or []
                    _sources[src["id"]]["kg_edges"] = src.get("kg_edges") or []
        except Exception as exc:
            logger.error("Ontology enrichment background task failed: %s", exc)

    background_tasks.add_task(_run)
    status = _enricher.get_status()
    return {
        "status":   "accepted",
        "message":  "Ontology enrichment started in the background",
        "last_run": status,
    }


@app.get("/ontology/enrich/status")
async def get_ontology_enrichment_status():
    """Return the most recent ontology enrichment run record."""
    status = _enricher.get_status()
    if not status:
        return {"status": "never_run", "message": "No enrichment run recorded yet"}
    return status


@app.get("/ontology/glossary-links")
async def list_glossary_attr_links(term_id: Optional[str] = None):
    """Return glossary→attribute annotation links, optionally filtered by term."""
    return _enricher.get_glossary_attr_links(term_id=term_id)


@app.get("/ontology/kpi-links")
async def list_kpi_attr_links(kpi_id: Optional[str] = None):
    """Return KPI→attribute annotation links, optionally filtered by KPI."""
    return _enricher.get_kpi_attr_links(kpi_id=kpi_id)


# ── Access Control (RBAC / ABAC) endpoints ─────────────────────────────────────
# Enforcement is active only in production (APP_ENV=production).
# In dev/test all checks pass through — use these endpoints to configure users
# in advance and the rules will take effect automatically once deployed.

class AccessUserCreate(BaseModel):
    user_id: Optional[str] = None
    name:    str
    email:   str
    role:    str = "viewer"   # viewer | analyst | manager | admin
    domain:  str = "*"        # "*" = all domains, or e.g. "Finance"

class AccessUserUpdate(BaseModel):
    name:      Optional[str] = None
    email:     Optional[str] = None
    role:      Optional[str] = None
    domain:    Optional[str] = None
    is_active: Optional[bool] = None


@app.get("/access/status")
async def access_control_status():
    """Return whether RBAC enforcement is active."""
    return {
        "enforced": _ac.is_enforced(),
        "mode":     "production" if _ac.is_enforced() else "dev/test (pass-through)",
        "roles":    list(_ac.ROLES),
    }


@app.get("/access/users")
async def list_access_users():
    """List all users in the RBAC store."""
    try:
        return _ac.list_users()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/access/users", status_code=201)
async def create_access_user(req: AccessUserCreate):
    """Create a new RBAC user."""
    try:
        user = _ac.add_user(req.user_id, req.name, req.email, req.role, req.domain)
        return user
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/access/users/{user_id}")
async def get_access_user(user_id: str):
    user = _ac.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.patch("/access/users/{user_id}")
async def update_access_user(user_id: str, req: AccessUserUpdate):
    """Update name, email, role, domain or is_active for a user."""
    updates = {k: v for k, v in req.dict().items() if v is not None}
    try:
        updated = _ac.update_user(user_id, **updates)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not updated:
        raise HTTPException(status_code=404, detail="User not found or no valid fields")
    return _ac.get_user(user_id) or {}


@app.delete("/access/users/{user_id}", status_code=200)
async def delete_access_user(user_id: str):
    _ac.delete_user(user_id)
    return {"deleted": user_id}


@app.get("/access/check")
async def access_check(user_id: str, source_id: Optional[str] = None, action: Optional[str] = None):
    """
    Dry-run access check for a user.
    Pass source_id to check source access, action to check permission.
    In dev/test always returns allowed=True.
    """
    result: Dict[str, Any] = {
        "user_id":  user_id,
        "enforced": _ac.is_enforced(),
        "allowed":  True,
    }
    if source_id:
        src = _sources.get(source_id)
        if not src:
            raise HTTPException(status_code=404, detail="Source not found")
        result["source_access"] = _ac.can_access_source(user_id, src)
        result["allowed"]       = result["source_access"]
    if action:
        result["action"]      = action
        result["can_perform"] = _ac.can_perform(user_id, action)
        result["allowed"]     = result.get("allowed", True) and result["can_perform"]
    return result


# ── Excel export ──────────────────────────────────────────────────────────────

class ExcelExportRequest(BaseModel):
    title:   str            = "DataNanite Insight Report"
    results: List[Dict[str, Any]]


@app.post("/export-excel")
async def export_excel(req: ExcelExportRequest):
    """Generate a multi-tab Excel workbook from one or more query result sets."""
    from excel_export import build_excel_report
    from fastapi.responses import Response as _XlsxResponse
    data     = build_excel_report(req.results, req.title)
    from datetime import datetime as _dt
    filename = f"datananite_{_dt.now():%Y%m%d_%H%M%S}.xlsx"
    return _XlsxResponse(
        content=data,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Glossary proxy routes ─────────────────────────────────────────────────────
# Forward /metadata/glossary/* → agent-api /glossary/*
# The tech UI uses _GLOSSARY = '/metadata/glossary' which maps to this prefix.

@app.api_route("/metadata/glossary/{path:path}",
               methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_glossary(request: Request, path: str):
    """Forward /metadata/glossary/{path} → METADATA_API/glossary/{path}."""
    body = await request.body()
    qs   = f"?{request.url.query}" if request.url.query else ""
    fwd_headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in _HOP_HEADERS and k.lower() != "host"}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.request(
                method=request.method,
                url=f"{METADATA_API}/glossary/{path}{qs}",
                headers=fwd_headers,
                content=body,
            )
        resp_headers = {k: v for k, v in resp.headers.items()
                        if k.lower() not in _HOP_HEADERS}
        from fastapi.responses import Response as _Resp
        return _Resp(
            content=resp.content,
            status_code=resp.status_code,
            headers=resp_headers,
            media_type=resp.headers.get("content-type"),
        )
    except Exception as exc:
        raise HTTPException(502, f"Glossary service unavailable: {exc}")


# ── Unstructured agent proxy routes ──────────────────────────────────────────
# Two additive routes that forward /unstructured/* to the unstructured service.
# All other orchestrator routes and data are untouched.

@app.api_route("/unstructured/{path:path}",
               methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_unstructured(request: Request, path: str):
    """Forward /unstructured/{path} → UNSTRUCTURED_API/{path}."""
    body = await request.body()
    qs   = f"?{request.url.query}" if request.url.query else ""
    fwd_headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in _HOP_HEADERS and k.lower() != "host"}
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.request(
                method=request.method,
                url=f"{UNSTRUCTURED_API}/{path}{qs}",
                headers=fwd_headers,
                content=body,
            )
        resp_headers = {k: v for k, v in resp.headers.items()
                        if k.lower() not in _HOP_HEADERS}
        from fastapi.responses import Response as _Resp
        return _Resp(
            content=resp.content,
            status_code=resp.status_code,
            headers=resp_headers,
            media_type=resp.headers.get("content-type"),
        )
    except Exception as exc:
        raise HTTPException(502, f"Unstructured service unavailable: {exc}")


# ── /api/* compatibility shim ─────────────────────────────────────────────────
# Older versions of tech_ui_server.py forwarded /api/{path} directly to the
# orchestrator (i.e. GET http://chat-ui:8005/api/sources).  The real routes
# live at the root (GET /sources, etc.).  This catch-all rewrites the path so
# both old and new proxy versions work without a container rebuild.

from fastapi.responses import Response as _Response

_HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
}


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def api_compat_shim(request: Request, path: str):
    """Forward /api/{path}?… → /{path}?… within the same ASGI app."""
    body = await request.body()
    qs   = f"?{request.url.query}" if request.url.query else ""
    fwd_headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in _HOP_HEADERS and k.lower() != "host"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://internal"
    ) as client:
        resp = await client.request(
            method=request.method,
            url=f"http://internal/{path}{qs}",
            headers=fwd_headers,
            content=body,
        )
    resp_headers = {k: v for k, v in resp.headers.items()
                    if k.lower() not in _HOP_HEADERS}
    return _Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=resp_headers,
        media_type=resp.headers.get("content-type"),
    )
