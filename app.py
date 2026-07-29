"""
Interactive UI for the Metadata Extraction Agent.

In containerised mode the UI talks to the FastAPI backend (agent-api service)
via HTTP.  Set AGENT_API_URL in your environment or .env file.

Run locally (no Docker):
    export AGENT_API_URL=http://localhost:8000
    streamlit run app.py
"""
from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Metadata Agent",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── API client ────────────────────────────────────────────────────────────────
AGENT_API_URL = os.environ.get("AGENT_API_URL", "http://localhost:8000").rstrip("/")


class APIClient:
    def __init__(self, base: str):
        self._base = base

    def _get(self, path: str, **kwargs) -> Any:
        r = requests.get(f"{self._base}{path}", timeout=30, **kwargs)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, payload: Dict) -> Any:
        r = requests.post(f"{self._base}{path}", json=payload, timeout=120)
        if not r.ok:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise requests.HTTPError(f"HTTP {r.status_code}: {detail}", response=r)
        return r.json()

    def _delete(self, path: str) -> Any:
        r = requests.delete(f"{self._base}{path}", timeout=15)
        r.raise_for_status()
        return r.json()

    def health(self) -> bool:
        try:
            r = requests.get(f"{self._base}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def start_extraction(self, payload: Dict) -> str:
        return self._post("/extract", payload)["job_id"]

    def get_job(self, job_id: str) -> Dict:
        return self._get(f"/jobs/{job_id}")

    def get_job_report(self, job_id: str) -> Dict:
        return self._get(f"/jobs/{job_id}/report")

    def ask(self, run_id: str, question: str) -> str:
        return self._post(f"/history/{run_id}/ask", {"question": question})["answer"]

    def get_history(self) -> List[Dict]:
        try:
            return self._get("/history")
        except Exception:
            return []

    def delete_history(self, run_id: str) -> None:
        self._delete(f"/history/{run_id}")

    def get_history_report(self, run_id: str) -> Dict:
        return self._get(f"/history/{run_id}/report")

    def search(self, q: str, scope: str = "all", db_type: str = "all") -> List[Dict]:
        return self._get("/search", params={"q": q, "scope": scope, "db_type": db_type})

    def upload_file(self, file_bytes: bytes, filename: str) -> Dict:
        """Upload a single file; returns {path, expires_at}."""
        r = requests.post(
            f"{self._base}/upload-file",
            files={"file": (filename, file_bytes)},
            timeout=120,
        )
        if not r.ok:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise requests.HTTPError(f"HTTP {r.status_code}: {detail}", response=r)
        return r.json()

    def upload_files(self, files: list) -> Dict:
        """Upload multiple CSV files; returns {path, expires_at}."""
        r = requests.post(
            f"{self._base}/upload-files",
            files=[("files", (f.name, f.read())) for f in files],
            timeout=120,
        )
        if not r.ok:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise requests.HTTPError(f"HTTP {r.status_code}: {detail}", response=r)
        return r.json()

    def list_uploads(self) -> List[Dict]:
        """Return non-expired uploaded files from the server."""
        try:
            return self._get("/uploads/list").get("uploads", [])
        except Exception:
            return []

    def extend_upload(self, path: str) -> Dict:
        """Request a one-time 15-minute extension; raises HTTPError if already extended."""
        return self._post("/uploads/extend", {"path": path})

    def discover(self, payload: Dict) -> Dict:
        """Return {schemas: [...], tables: {schema: [table, ...]}} for the given DB."""
        return self._post("/discover", payload)

api = APIClient(AGENT_API_URL)


# ── Ontology API client (separate microservice) ────────────────────────────────
ONTOLOGY_API_URL = os.environ.get("ONTOLOGY_API_URL", "http://localhost:8001").rstrip("/")

# ── Knowledge Graph API client (separate microservice) ─────────────────────────
KG_API_URL = os.environ.get("KG_API_URL", "http://localhost:8002").rstrip("/")

# ── Dialog with Data API client (separate microservice) ────────────────────────
DIALOG_API_URL = os.environ.get("DIALOG_API_URL", "http://localhost:8003").rstrip("/")


class OntologyAPIClient:
    def __init__(self, base: str):
        self._base = base

    def _check(self, r: "requests.Response") -> "requests.Response":
        if not r.ok:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise requests.HTTPError(f"HTTP {r.status_code}: {detail}", response=r)
        return r

    def health(self) -> bool:
        try:
            return requests.get(f"{self._base}/health", timeout=5).status_code == 200
        except Exception:
            return False

    def generate(self, payload: Dict) -> str:
        r = self._check(requests.post(f"{self._base}/generate", json=payload, timeout=120))
        return r.json()["job_id"]

    def get_job(self, job_id: str) -> Dict:
        return self._check(requests.get(f"{self._base}/jobs/{job_id}", timeout=30)).json()

    def get_content(self, job_id: str) -> Dict:
        return self._check(requests.get(f"{self._base}/jobs/{job_id}/content", timeout=30)).json()

    def save_content(self, job_id: str, content: str) -> None:
        self._check(requests.put(f"{self._base}/jobs/{job_id}/content",
                                 json={"content": content}, timeout=30))

    def get_bytes(self, job_id: str):
        r = self._check(requests.get(f"{self._base}/jobs/{job_id}/download", timeout=30))
        disposition = r.headers.get("content-disposition", "")
        filename = (disposition.split("filename=")[-1].strip('"')
                    if "filename=" in disposition else "ontology.ttl")
        return r.content, filename

    def list_jobs(self) -> List[Dict]:
        try:
            return requests.get(f"{self._base}/list", timeout=10).json()
        except Exception:
            return []


onto_api = OntologyAPIClient(ONTOLOGY_API_URL)


# ── Knowledge Graph API client ─────────────────────────────────────────────────
class KGAPIClient:
    def __init__(self, base: str):
        self._base = base

    def _check(self, r: "requests.Response") -> "requests.Response":
        if not r.ok:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise requests.HTTPError(f"HTTP {r.status_code}: {detail}", response=r)
        return r

    def health(self) -> bool:
        try:
            return requests.get(f"{self._base}/health", timeout=5).status_code == 200
        except Exception:
            return False

    def generate(self, payload: Dict) -> str:
        r = self._check(requests.post(f"{self._base}/generate", json=payload, timeout=120))
        return r.json()["job_id"]

    def get_job(self, job_id: str) -> Dict:
        return self._check(requests.get(f"{self._base}/jobs/{job_id}", timeout=30)).json()

    def get_graph(self, job_id: str) -> Dict:
        return self._check(requests.get(f"{self._base}/jobs/{job_id}/graph", timeout=30)).json()

    def get_queries(self, job_id: str) -> Dict:
        return self._check(requests.get(f"{self._base}/jobs/{job_id}/queries", timeout=30)).json()

    def fetch(self, payload: Dict) -> str:
        r = self._check(requests.post(f"{self._base}/fetch", json=payload, timeout=120))
        return r.json()["job_id"]

    def list_jobs(self) -> List[Dict]:
        try:
            return requests.get(f"{self._base}/list", timeout=10).json()
        except Exception:
            return []


kg_api = KGAPIClient(KG_API_URL)


# ── Dialog with Data API client ────────────────────────────────────────────────
class DialogAPIClient:
    def __init__(self, base: str):
        self._base = base

    def _check(self, r: "requests.Response") -> "requests.Response":
        if not r.ok:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise requests.HTTPError(f"HTTP {r.status_code}: {detail}", response=r)
        return r

    def health(self) -> bool:
        try:
            return requests.get(f"{self._base}/health", timeout=5).status_code == 200
        except Exception:
            return False

    def query(self, payload: Dict) -> str:
        r = self._check(requests.post(f"{self._base}/query", json=payload, timeout=120))
        return r.json()["job_id"]

    def get_job(self, job_id: str) -> Dict:
        return self._check(requests.get(f"{self._base}/jobs/{job_id}", timeout=30)).json()

    def get_results(self, job_id: str) -> Dict:
        return self._check(requests.get(f"{self._base}/jobs/{job_id}/results", timeout=30)).json()

    def list_jobs(self) -> List[Dict]:
        try:
            return requests.get(f"{self._base}/list", timeout=10).json()
        except Exception:
            return []

    def list_cache(self) -> List[Dict]:
        try:
            return requests.get(f"{self._base}/cache", timeout=10).json()
        except Exception:
            return []

    def delete_cache_entry(self, cache_key: str) -> None:
        requests.delete(f"{self._base}/cache/{cache_key}", timeout=10)

    def clear_cache(self) -> int:
        try:
            r = requests.delete(f"{self._base}/cache", timeout=10)
            return r.json().get("deleted_count", 0)
        except Exception:
            return 0


dialog_api = DialogAPIClient(DIALOG_API_URL)


# ── Conformity API client ──────────────────────────────────────────────────────
CONFORMITY_API_URL = os.environ.get("CONFORMITY_API_URL", "http://localhost:8004").rstrip("/")


class ConformityAPIClient:
    def __init__(self, base: str):
        self._base = base

    def _check(self, r: "requests.Response") -> "requests.Response":
        if not r.ok:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise requests.HTTPError(f"HTTP {r.status_code}: {detail}", response=r)
        return r

    def health(self) -> bool:
        try:
            return requests.get(f"{self._base}/health", timeout=5).status_code == 200
        except Exception:
            return False

    def analyse(self, payload: Dict) -> str:
        r = self._check(requests.post(f"{self._base}/analyse", json=payload, timeout=120))
        return r.json()["job_id"]

    def get_job(self, job_id: str) -> Dict:
        return self._check(requests.get(f"{self._base}/jobs/{job_id}", timeout=30)).json()

    def get_results(self, job_id: str) -> Dict:
        return self._check(requests.get(f"{self._base}/jobs/{job_id}/results", timeout=30)).json()

    def stitch(self, payload: Dict) -> str:
        r = self._check(requests.post(f"{self._base}/stitch", json=payload, timeout=120))
        return r.json()["stitch_id"]

    def get_stitch(self, stitch_id: str) -> Dict:
        return self._check(requests.get(f"{self._base}/stitch/{stitch_id}", timeout=30)).json()

    def get_stitch_graph(self, stitch_id: str) -> Dict:
        return self._check(requests.get(f"{self._base}/stitch/{stitch_id}/graph", timeout=30)).json()

    def list_super_graphs(self) -> List[Dict]:
        try:
            return requests.get(f"{self._base}/super-graphs", timeout=10).json()
        except Exception:
            return []

    def save_super_graph(self, payload: Dict) -> Dict:
        return self._check(requests.post(f"{self._base}/super-graphs", json=payload, timeout=30)).json()

    def get_super_graph(self, name: str) -> Dict:
        return self._check(requests.get(f"{self._base}/super-graphs/{name}", timeout=30)).json()


conformity_api = ConformityAPIClient(CONFORMITY_API_URL)


# ── Constants ─────────────────────────────────────────────────────────────────
DB_META: Dict[str, Dict] = {
    "postgres":   {"icon": "🐘", "label": "PostgreSQL", "color": "#60a5fa"},
    "oracle":     {"icon": "🏛️",  "label": "Oracle",     "color": "#f87171"},
    "sqlserver":  {"icon": "🪟",  "label": "SQL Server", "color": "#34d399"},
    "teradata":   {"icon": "📊",  "label": "Teradata",   "color": "#fbbf24"},
    "redshift":   {"icon": "🔺",  "label": "Redshift",   "color": "#a78bfa"},
    "bigquery":   {"icon": "📈",  "label": "BigQuery",   "color": "#f472b6"},
    "delta_lake": {"icon": "⬡",   "label": "Delta Lake", "color": "#2dd4bf"},
    "sqlite":     {"icon": "🗄️",  "label": "SQLite",     "color": "#94a3b8"},
    "csv":        {"icon": "📄",  "label": "CSV Files",  "color": "#4ade80"},
    "excel":      {"icon": "📊",  "label": "Excel",      "color": "#22d3ee"},
}

_FILE_BASED = {"sqlite", "csv", "excel"}

PIPELINE_NODES = [
    ("connection",  "🔌", "Connecting to database"),
    ("discovery",   "🔍", "Discovering tables"),
    ("extraction",  "📦", "Extracting schema & statistics"),
    ("analysis",    "🧬", "Analyzing dependencies"),
    ("report",      "📋", "Generating report"),
]


# ── CSS ────────────────────────────────────────────────────────────────────────
def _inject_css() -> None:
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0e1a 0%, #0f1623 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

.hero {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a1f35 50%, #0f2744 100%);
    border: 1px solid rgba(99,179,237,0.18); border-radius: 18px;
    padding: 2rem 2.5rem 1.8rem; margin-bottom: 2rem;
    position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(99,179,237,0.12) 0%, transparent 70%);
}
.hero-title {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(90deg, #63b3ed 0%, #b794f4 50%, #68d391 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; letter-spacing: -0.8px;
}
.hero-sub { color: #718096; font-size: 0.95rem; margin-top: 0.4rem; }

.stat-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin-bottom: 1.5rem; }
.stat-card {
    background: linear-gradient(135deg, #131929 0%, #1e2740 100%);
    border: 1px solid rgba(255,255,255,0.07); border-radius: 14px;
    padding: 1.3rem 1.1rem; text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.stat-card:hover { transform: translateY(-3px); box-shadow: 0 10px 30px rgba(0,0,0,0.4); }
.stat-val { font-size: 2rem; font-weight: 700; line-height: 1; }
.stat-lbl { font-size: 0.7rem; color: #4a5568; text-transform: uppercase; letter-spacing: 0.07em; margin-top: 0.35rem; }

.sec-head {
    font-size: 1.2rem; font-weight: 700; color: #e2e8f0;
    display: flex; align-items: center; gap: 0.5rem;
    margin-bottom: 1.2rem; padding-bottom: 0.6rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}

.hcard {
    background: linear-gradient(135deg, #131929 0%, #1a2035 100%);
    border: 1px solid rgba(255,255,255,0.07); border-radius: 14px;
    padding: 1.3rem 1.5rem; margin-bottom: 0.8rem;
    position: relative; overflow: hidden;
    transition: border-color 0.2s, transform 0.15s;
}
.hcard:hover { border-color: rgba(99,179,237,0.3); transform: translateX(2px); }
.hcard-title { font-size: 1rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.2rem; }
.hcard-meta  { font-size: 0.78rem; color: #64748b; }

.dbbadge {
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 0.22rem 0.65rem; border-radius: 20px;
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase;
}

.node-item {
    display: flex; align-items: center; gap: 0.7rem;
    padding: 0.55rem 1rem; border-radius: 9px;
    margin-bottom: 0.35rem; font-size: 0.85rem; font-weight: 500;
    transition: all 0.3s;
}
.node-pending { background: rgba(255,255,255,0.02); color: #4a5568; }
.node-running {
    background: rgba(99,179,237,0.1); color: #63b3ed;
    border: 1px solid rgba(99,179,237,0.2);
    animation: pulse-blue 1.5s ease-in-out infinite;
}
.node-done  { background: rgba(104,211,145,0.1); color: #68d391; border: 1px solid rgba(104,211,145,0.2); }
.node-error { background: rgba(252,129,74,0.1);  color: #fc814a; border: 1px solid rgba(252,129,74,0.2); }
@keyframes pulse-blue {
    0%, 100% { box-shadow: 0 0 0 0 rgba(99,179,237,0.2); }
    50%       { box-shadow: 0 0 0 6px rgba(99,179,237,0); }
}

.s-result {
    background: #111827; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 11px; padding: 1rem 1.2rem; margin-bottom: 0.55rem;
    transition: border-color 0.2s;
}
.s-result:hover { border-color: rgba(99,179,237,0.3); }
.s-result-title { font-size: 0.95rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.2rem; }
.s-result-sub   { font-size: 0.78rem; color: #64748b; }
mark { background: rgba(251,191,36,0.25); color: #fbbf24; padding: 0 2px; border-radius: 3px; }

.empty-state { text-align: center; padding: 4rem 2rem; color: #374151; }
.empty-state-icon { font-size: 3.5rem; margin-bottom: 1rem; }
.empty-state-text { font-size: 1rem; font-weight: 500; }

.pill {
    display: inline-block; background: rgba(99,179,237,0.12);
    color: #63b3ed; padding: 0.15rem 0.6rem; border-radius: 12px;
    font-size: 0.72rem; margin: 2px; font-weight: 500;
}
.banner-ok  {
    background: rgba(104,211,145,0.1); border: 1px solid rgba(104,211,145,0.3);
    border-radius: 10px; padding: 0.9rem 1.2rem; color: #68d391; font-weight: 500;
}
.banner-err {
    background: rgba(252,129,74,0.1); border: 1px solid rgba(252,129,74,0.3);
    border-radius: 10px; padding: 0.9rem 1.2rem; color: #fc814a; font-weight: 500;
}
.api-status-ok  { color: #68d391; font-size: 0.75rem; }
.api-status-err { color: #fc814a; font-size: 0.75rem; }
.sidebar-logo {
    font-size: 1.4rem; font-weight: 800; margin-bottom: 0.2rem;
    background: linear-gradient(90deg, #63b3ed, #b794f4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def db_badge(db_type: str) -> str:
    m = DB_META.get(db_type, {"icon": "🗄️", "label": db_type, "color": "#94a3b8"})
    c = m["color"]
    r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
    return (
        f'<span class="dbbadge" style="background:rgba({r},{g},{b},0.15);'
        f'color:{c};border:1px solid rgba({r},{g},{b},0.3)">'
        f'{m["icon"]} {m["label"]}</span>'
    )


def _highlight(text: str, q: str) -> str:
    if not q:
        return text
    return re.sub(f"({re.escape(q)})", r"<mark>\1</mark>", text, flags=re.IGNORECASE)


def _stat_card(value: Any, label: str, color: str = "#63b3ed") -> str:
    return (
        f'<div class="stat-card">'
        f'<div class="stat-val" style="color:{color}">{value}</div>'
        f'<div class="stat-lbl">{label}</div></div>'
    )


def _node_html(nodes_state: Dict[str, str]) -> str:
    icons  = {n[0]: n[1] for n in PIPELINE_NODES}
    labels = {n[0]: n[2] for n in PIPELINE_NODES}
    items  = []
    for name, _, _ in PIPELINE_NODES:
        state  = nodes_state.get(name, "pending")
        prefix = {"running": "⟳ ", "done": "✓ ", "error": "✗ "}.get(state, "○ ")
        items.append(
            f'<div class="node-item node-{state}">'
            f'{icons[name]} {prefix}{labels[name]}</div>'
        )
    return "<div>" + "".join(items) + "</div>"


def _summary_stats(summary: Dict) -> str:
    cards = [
        _stat_card(summary.get("total_tables", 0),                    "Tables",             "#63b3ed"),
        _stat_card(summary.get("total_functional_dependencies", 0),   "Func. Dependencies", "#b794f4"),
        _stat_card(summary.get("total_inclusion_dependencies", 0),    "Incl. Dependencies", "#68d391"),
        _stat_card(summary.get("total_cardinality_relationships", 0), "Cardinality Links",  "#f6ad55"),
    ]
    return '<div class="stat-row">' + "".join(cards) + "</div>"


def _nodes_from_status(status: Dict) -> Dict[str, str]:
    completed = set(status.get("completed_nodes") or [])
    current   = status.get("current_node") or ""
    job_status= status.get("status", "queued")
    out = {}
    for name, _, _ in PIPELINE_NODES:
        if name in completed:
            out[name] = "done"
        elif name == current and job_status == "running":
            out[name] = "running"
        elif job_status == "error" and name == current:
            out[name] = "error"
        else:
            out[name] = "pending"
    return out


# ── Session state init ─────────────────────────────────────────────────────────
for _k, _v in [("page", "extract"), ("last_report", None),
                ("last_run_id", None), ("last_run_meta", None),
                ("running_job_id", None),
                ("onto_job_id", None), ("onto_result", None),
                ("onto_content", None), ("onto_last_job_id", None),
                ("kg_job_id", None), ("kg_result", None), ("kg_graph_data", None),
                ("kg_mode", "generate"),
                ("dialog_job_id", None), ("dialog_result", None),
                ("dialog_kg_job_id", None),
                # Schema/table discovery results
                ("ext_disco", None),    # {schemas:[...], tables:{schema:[...]}}
                ("ext_uploaded_path",  None),   # server-side path after file upload (extraction)
                ("ext_upload_expires", None),   # ISO expiry timestamp (extraction)
                ("dlg_uploaded_path",  None),   # server-side path after file upload (dialog)
                ("dlg_upload_expires", None),   # ISO expiry timestamp (dialog)
                ("dlg_disco", None),
                # Conformity agent state
                ("conformity_job_id", None),
                ("conformity_results", None),
                ("conformity_stitch_id", None),
                ("conformity_super_graph", None),
                ("conformity_approved", set()),
                ("conformity_kg_ids", []),
                ]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Sidebar ────────────────────────────────────────────────────────────────────
def _sidebar() -> None:
    with st.sidebar:
        st.markdown('<div class="sidebar-logo">⬡ Metadata Agent</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:0.72rem;color:#475569;margin-bottom:0.4rem">Database schema intelligence</div>',
            unsafe_allow_html=True,
        )
        # API status indicators
        for label, checker in [("Agent API", api.health), ("Ontology API", onto_api.health),
                                ("KG API", kg_api.health), ("Dialog API", dialog_api.health),
                                ("Conformity API", conformity_api.health)]:
            ok  = checker()
            dot = "🟢" if ok else "🔴"
            st.markdown(
                f'<div class="{"api-status-ok" if ok else "api-status-err"}">'
                f'{dot} {label}: {"connected" if ok else "unreachable"}</div>',
                unsafe_allow_html=True,
            )
        st.markdown("---")

        for key, icon, label in [("extract",    "⚡",  "New Extraction"),
                                   ("history",    "🗂️",  "History"),
                                   ("search",     "🔍", "Search Metadata"),
                                   ("ontology",   "🦉", "Ontology Generator"),
                                   ("kg",         "🕸️", "Knowledge Graph"),
                                   ("dialog",     "💬", "Dialog with Data"),
                                   ("conformity", "🔗", "KG Conformity")]:
            active = st.session_state.page == key
            if st.button(
                f"{icon}  {label}",
                key=f"nav_{key}",
                use_container_width=True,
                type="primary" if active else "secondary",
            ):
                st.session_state.page = key
                st.rerun()

        history = api.get_history()
        if history:
            st.markdown("---")
            st.markdown(
                '<div style="font-size:0.72rem;color:#475569;text-transform:uppercase;'
                'letter-spacing:0.08em;margin-bottom:0.5rem">Recent runs</div>',
                unsafe_allow_html=True,
            )
            for h in history[:5]:
                m  = DB_META.get(h["db_type"], {"icon": "🗄️"})
                ts = h.get("timestamp", "")[:16].replace("T", " ")
                st.markdown(
                    f'<div style="font-size:0.78rem;color:#94a3b8;margin-bottom:0.4rem;'
                    f'padding:0.4rem 0.6rem;background:rgba(255,255,255,0.03);border-radius:7px">'
                    f'{m["icon"]} <b>{h.get("database","?")}</b><br>'
                    f'<span style="color:#475569">{ts}</span></div>',
                    unsafe_allow_html=True,
                )


# ── Inline result panel (shared between fresh run & history load) ──────────────
def _show_result_panel() -> None:
    report   = st.session_state.last_report
    run_id   = st.session_state.last_run_id
    run_meta = st.session_state.last_run_meta or {}
    if not report:
        return

    summary = report.get("summary", {})
    st.markdown(
        '<div class="banner-ok" style="margin-bottom:1rem">✓ Extraction complete!</div>',
        unsafe_allow_html=True,
    )
    st.markdown(_summary_stats(summary), unsafe_allow_html=True)

    with st.expander("📊 Report details", expanded=False):
        tables: Dict = report.get("tables") or {}
        if tables:
            st.markdown(f"**Tables ({len(tables)})**")
            for tname, tmeta in list(tables.items())[:20]:
                cols = tmeta.get("columns", []) if isinstance(tmeta, dict) else []
                col_names = [c.get("name", str(c)) if isinstance(c, dict) else str(c) for c in cols]
                pills = " ".join(f'<span class="pill">{c}</span>' for c in col_names[:12])
                extra = f'<span class="pill">+{len(col_names)-12} more</span>' if len(col_names) > 12 else ""
                rows  = tmeta.get("row_count", "?") if isinstance(tmeta, dict) else "?"
                st.markdown(
                    f'<div style="margin-bottom:0.6rem">'
                    f'<span style="color:#e2e8f0;font-weight:600">{tname}</span>'
                    f'<span style="color:#475569;font-size:0.8rem;margin-left:0.5rem">{rows} rows</span>'
                    f'<br>{pills}{extra}</div>',
                    unsafe_allow_html=True,
                )

        fds = report.get("functional_dependencies") or []
        if fds:
            st.markdown(f"**Functional Dependencies ({len(fds)})**")
            for fd in fds[:10]:
                det  = ", ".join(fd.get("determinant", []))
                dep  = ", ".join(fd.get("dependent", []))
                conf = fd.get("confidence", 0)
                st.markdown(
                    f'<div style="font-size:0.82rem;margin-bottom:0.3rem;color:#94a3b8">'
                    f'<span style="color:#63b3ed">[{det}]</span> → '
                    f'<span style="color:#b794f4">[{dep}]</span> '
                    f'<span style="color:#475569">conf={conf:.2f}</span></div>',
                    unsafe_allow_html=True,
                )

        st.download_button(
            "⬇  Download full JSON report",
            data=__import__("json").dumps(report, indent=2, default=str),
            file_name=f'metadata_{run_meta.get("db_type","?")}.json',
            mime="application/json",
            use_container_width=True,
        )

    # LLM Q&A
    st.markdown('<div class="sec-head" style="margin-top:1.5rem">💬 Ask the Agent</div>', unsafe_allow_html=True)
    question = st.text_input("Ask a question about the extracted metadata",
                              placeholder="Which tables have the most null values?",
                              key="llm_question")
    if st.button("Ask", key="ask_btn") and question and run_id:
        try:
            with st.spinner("Thinking…"):
                answer = api.ask(run_id, question)
            st.markdown(
                f'<div style="background:rgba(183,148,244,0.08);border:1px solid rgba(183,148,244,0.2);'
                f'border-radius:10px;padding:1rem 1.2rem;color:#e2e8f0;font-size:0.9rem">{answer}</div>',
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"LLM error: {e}")


# ── View: Extract ──────────────────────────────────────────────────────────────
def _extract_view() -> None:
    st.markdown(
        '<div class="hero"><div class="hero-title">New Metadata Extraction</div>'
        '<div class="hero-sub">Connect to any database and extract complete schema intelligence.</div></div>',
        unsafe_allow_html=True,
    )

    col_form, col_right = st.columns([3, 2], gap="large")

    with col_form:
        st.markdown('<div class="sec-head">🔌 Database Connection</div>', unsafe_allow_html=True)
        db_type = st.selectbox(
            "Database type",
            options=list(DB_META.keys()),
            format_func=lambda k: f'{DB_META[k]["icon"]}  {DB_META[k]["label"]}',
        )

        needs_bq    = db_type == "bigquery"
        needs_spark = db_type == "delta_lake"
        needs_file  = db_type in _FILE_BASED

        if needs_file:
            if db_type == "csv":
                uploaded = st.file_uploader(
                    "Upload CSV file(s)", type=["csv"], accept_multiple_files=True, key="ext_file_upload"
                )
                st.caption("Upload one or more CSV files from your computer.")
            elif db_type == "sqlite":
                uploaded = st.file_uploader(
                    "Upload SQLite database", type=["sqlite", "db", "sqlite3"], key="ext_file_upload"
                )
                st.caption("Upload a SQLite database file from your computer.")
            else:  # excel
                uploaded = st.file_uploader(
                    "Upload Excel file", type=["xlsx", "xls", "xlsm", "xlsb"], key="ext_file_upload"
                )
                st.caption("Upload an Excel file from your computer.")

            # Upload button — sends the file(s) to the agent container
            if st.button("⬆ Upload file", key="ext_upload_btn",
                         disabled=not bool(uploaded if db_type == "csv" else uploaded)):
                try:
                    with st.spinner("Uploading…"):
                        if db_type == "csv":
                            result = api.upload_files(uploaded)
                        else:
                            result = api.upload_file(uploaded.read(), uploaded.name)
                    st.session_state.ext_uploaded_path   = result["path"]
                    st.session_state.ext_upload_expires  = result.get("expires_at", "")
                except Exception as e:
                    st.error(f"Upload failed: {e}")
                    st.session_state.ext_uploaded_path  = None
                    st.session_state.ext_upload_expires = None

            file_path   = st.session_state.get("ext_uploaded_path") or ""
            expires_at  = st.session_state.get("ext_upload_expires") or ""
            if file_path:
                # Format expiry time to local-friendly string
                try:
                    from datetime import datetime, timezone
                    exp_dt = datetime.fromisoformat(expires_at)
                    exp_str = exp_dt.astimezone().strftime("%H:%M %Z")
                except Exception:
                    exp_str = expires_at
                st.info(
                    f"File saved on the server. It will be automatically deleted at **{exp_str}** "
                    f"(2 hours after upload). Make sure to complete your extraction before then.",
                    icon="⏳",
                )
            host = port = database = schema = username = password = ""
            bq_project = bq_dataset = bq_creds = spark_master = http_path = odbc_driver = ""
        elif needs_bq:
            c1, c2 = st.columns(2)
            bq_project = c1.text_input("GCP Project", placeholder="my-gcp-project", key="ext_bq_project")
            bq_dataset = c2.text_input("Dataset (schema)", placeholder="my_dataset", key="ext_bq_dataset")
            bq_creds   = st.text_input("Service account JSON path", placeholder="/path/to/sa.json", key="ext_bq_creds")
            host = port = database = schema = username = password = ""
            spark_master = http_path = odbc_driver = file_path = ""
        elif needs_spark:
            c1, c2 = st.columns(2)
            host         = c1.text_input("Databricks host", placeholder="adb-xxx.azuredatabricks.net", key="ext_spark_host")
            database     = c1.text_input("Catalog / Database", placeholder="my_catalog", key="ext_spark_db")
            schema       = c2.text_input("Schema", placeholder="my_schema", key="ext_spark_schema")
            spark_master = c2.text_input("Spark master (local only)", placeholder="local[*]", key="ext_spark_master")
            http_path    = st.text_input("HTTP path", placeholder="/sql/1.0/warehouses/abc123", key="ext_spark_http")
            password     = st.text_input("Databricks token", type="password", key="ext_spark_token")
            port         = 443
            username     = ""
            bq_project = bq_dataset = bq_creds = odbc_driver = file_path = ""
        else:
            c1, c2 = st.columns([3, 1])
            host = c1.text_input("Host", placeholder="localhost", key="ext_host")
            defaults = {"postgres": 5432, "oracle": 1521, "sqlserver": 1433,
                        "teradata": 1025, "redshift": 5439}
            port = c2.number_input("Port", value=defaults.get(db_type, 5432), step=1, key="ext_port")
            c3, c4 = st.columns(2)
            database = c3.text_input("Database / Service name", placeholder="mydb", key="ext_database")
            schema   = c4.text_input("Schema", placeholder="public", key="ext_schema")
            c5, c6  = st.columns(2)
            username = c5.text_input("Username", key="ext_username")
            password = c6.text_input("Password", type="password", key="ext_password")
            odbc_driver = ""
            if db_type == "sqlserver":
                odbc_driver = st.text_input("ODBC Driver", value="ODBC Driver 18 for SQL Server", key="ext_odbc")
            bq_project = bq_dataset = bq_creds = spark_master = http_path = file_path = ""

        # ── Schema & table discovery ───────────────────────────────────────────
        st.markdown('<hr style="border:none;border-top:1px solid rgba(255,255,255,0.06);margin:0.8rem 0">', unsafe_allow_html=True)
        disco_cols = st.columns([2, 1])
        with disco_cols[0]:
            st.markdown('<div class="sec-head">🔍 Discover Schemas & Tables</div>', unsafe_allow_html=True)
        with disco_cols[1]:
            st.markdown("<br>", unsafe_allow_html=True)
            disco_btn = st.button("🔍 Discover", key="ext_disco_btn", use_container_width=True)

        ext_disco = st.session_state.ext_disco

        if disco_btn:
            if needs_file:
                disco_payload = {
                    "db_type": db_type, "file_path": file_path or None,
                }
            elif needs_bq:
                disco_payload = {
                    "db_type": db_type, "project": bq_project or None,
                    "schema_name": bq_dataset or None, "credentials_path": bq_creds or None,
                }
            elif needs_spark:
                disco_payload = {
                    "db_type": db_type, "host": host or None,
                    "database": database or None, "schema_name": schema or None,
                    "password": password or None, "spark_master": spark_master or None,
                    "extra": {"http_path": http_path} if http_path else {},
                }
            else:
                _extra: Dict[str, Any] = {}
                if db_type == "sqlserver" and odbc_driver:
                    _extra["driver"] = odbc_driver
                disco_payload = {
                    "db_type": db_type, "host": host or None, "port": int(port) if port else None,
                    "database": database or None, "schema_name": schema or None,
                    "username": username or None, "password": password or None, "extra": _extra,
                }
            with st.spinner("Connecting and discovering schemas…"):
                try:
                    ext_disco = api.discover(disco_payload)
                    st.session_state.ext_disco = ext_disco
                except Exception as e:
                    st.error(f"Discovery failed: {e}")
                    ext_disco = None

        # Show discovered schemas and tables
        schema_sel  = schema   # fallback to typed value
        target_raw  = ""
        target_tables_sel: List[str] = []

        if ext_disco and ext_disco.get("schemas"):
            all_schemas = ext_disco["schemas"]
            tables_by_schema: Dict[str, List[str]] = ext_disco.get("tables", {})

            st.caption(f"✓ Found {len(all_schemas)} schema(s) — select one to browse its tables")
            schema_sel = st.selectbox(
                "Schema / Dataset", all_schemas,
                index=all_schemas.index(schema) if schema in all_schemas else 0,
                key="ext_disco_schema",
            )
            # Override schema with discovered selection
            if not needs_bq:
                schema = schema_sel
            elif not bq_dataset:
                bq_dataset = schema_sel

            available_tables = tables_by_schema.get(schema_sel, [])
            if available_tables:
                st.caption(f"📋 {len(available_tables)} table(s) in '{schema_sel}'")
                target_tables_sel = st.multiselect(
                    "Select tables to extract (leave empty = all)",
                    available_tables,
                    key="ext_disco_tables",
                )
            else:
                st.caption(f"No tables found in '{schema_sel}'.")

            if st.button("🔄 Re-discover", key="ext_redisco_btn"):
                st.session_state.ext_disco = None
                st.rerun()
        else:
            # No discovery data yet — raw text input fallback
            target_raw = st.text_input(
                "Target tables (blank = all)", placeholder="users, orders", key="ext_target_raw"
            )

        st.markdown('<hr style="border:none;border-top:1px solid rgba(255,255,255,0.06);margin:1rem 0">', unsafe_allow_html=True)
        st.markdown('<div class="sec-head">⚙️ Agent Settings</div>', unsafe_allow_html=True)

        c7, c8 = st.columns(2)
        sample_size  = c7.number_input("Sample size (rows)", value=10_000, min_value=100, step=1000)
        fd_threshold = c7.slider("FD threshold", 0.80, 1.0, 1.0, 0.01,
                                  help="1.0 = exact FDs only; lower = approximate")
        id_threshold = c8.slider("IND threshold", 0.70, 1.0, 0.95, 0.01)

        run_btn = st.button("⚡  Run Extraction", type="primary", use_container_width=True,
                            disabled=bool(st.session_state.running_job_id))

    # ── Right column: pipeline progress only ──────────────────────────────────
    with col_right:
        st.markdown('<div class="sec-head">📡 Pipeline Progress</div>', unsafe_allow_html=True)
        progress_area = st.empty()
        error_area    = st.empty()

        job_id = st.session_state.running_job_id
        if job_id:
            try:
                status = api.get_job(job_id)
            except Exception as e:
                error_area.markdown(
                    f'<div class="banner-err">⚠ API error while polling: {e}</div>',
                    unsafe_allow_html=True,
                )
                st.session_state.running_job_id = None
                status = None

            if status:
                nodes_state = _nodes_from_status(status)
                progress_area.markdown(_node_html(nodes_state), unsafe_allow_html=True)

                if status["status"] == "done":
                    try:
                        report = api.get_job_report(job_id)
                        st.session_state.last_report    = report
                        st.session_state.last_run_id    = job_id
                        st.session_state.last_run_meta  = {"db_type": db_type}
                        st.session_state.running_job_id = None
                        st.balloons()
                        st.rerun()          # re-render so the result panel appears
                    except Exception as e:
                        error_area.markdown(
                            f'<div class="banner-err">⚠ Could not fetch report: {e}</div>',
                            unsafe_allow_html=True,
                        )
                        st.session_state.running_job_id = None

                elif status["status"] == "error":
                    for k, v in nodes_state.items():
                        if v == "running":
                            nodes_state[k] = "error"
                    progress_area.markdown(_node_html(nodes_state), unsafe_allow_html=True)
                    error_area.markdown(
                        f'<div class="banner-err">⚠ Extraction failed: {status.get("error","unknown error")}</div>',
                        unsafe_allow_html=True,
                    )
                    st.session_state.running_job_id = None
                else:
                    # Still running — poll again
                    time.sleep(1.5)
                    st.rerun()
        else:
            idle = {n[0]: "pending" for n in PIPELINE_NODES}
            progress_area.markdown(_node_html(idle), unsafe_allow_html=True)

    # ── Result panel — rendered in main body so widgets work correctly ─────────
    if st.session_state.last_report and not st.session_state.running_job_id:
        st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.06);margin:1.5rem 0'>",
                    unsafe_allow_html=True)
        _show_result_panel()

    # ── Launch extraction ──────────────────────────────────────────────────────
    if run_btn:
        if not api.health():
            st.error(f"Cannot reach the Agent API at {AGENT_API_URL}. Is the agent-api container running?")
            return

        if needs_file:
            db_cfg_payload: Dict[str, Any] = {
                "db_type": db_type, "file_path": file_path or None,
            }
        elif needs_bq:
            db_cfg_payload = {
                "db_type": db_type, "project": bq_project or None,
                "schema_name": bq_dataset or None, "credentials_path": bq_creds or None,
            }
        elif needs_spark:
            extra: Dict = {}
            if http_path:
                extra["http_path"] = http_path
            db_cfg_payload = {
                "db_type": db_type, "host": host or None, "port": int(port) if host else None,
                "database": database or None, "schema_name": schema or None,
                "password": password or None, "spark_master": spark_master or None, "extra": extra,
            }
        else:
            extra = {}
            if db_type == "sqlserver" and odbc_driver:
                extra["driver"] = odbc_driver
            db_cfg_payload = {
                "db_type": db_type, "host": host or None, "port": int(port) if port else None,
                "database": database or None, "schema_name": schema or None,
                "username": username or None, "password": password or None, "extra": extra,
            }

        # Prefer multiselect discovery result; fall back to raw text input
        if target_tables_sel:
            target_tables: Optional[List[str]] = target_tables_sel
        elif target_raw.strip():
            target_tables = [t.strip() for t in target_raw.split(",") if t.strip()]
        else:
            target_tables = None
        payload = {
            "db_config":     db_cfg_payload,
            "target_tables": target_tables,
            "sample_size":   int(sample_size),
            "fd_threshold":  float(fd_threshold),
            "id_threshold":  float(id_threshold),
        }

        try:
            job_id = api.start_extraction(payload)
            st.session_state.running_job_id = job_id
            st.session_state.last_report    = None
            st.session_state.last_run_id    = None
            st.rerun()
        except Exception as e:
            st.error(f"Failed to start extraction: {e}")


# ── View: History ──────────────────────────────────────────────────────────────
def _history_view() -> None:
    st.markdown(
        '<div class="hero"><div class="hero-title">Extraction History</div>'
        '<div class="hero-sub">Browse, inspect, and manage all previous metadata extractions.</div></div>',
        unsafe_allow_html=True,
    )

    history = api.get_history()

    if not history:
        st.markdown(
            '<div class="empty-state"><div class="empty-state-icon">🗂️</div>'
            '<div class="empty-state-text">No extractions yet. Run your first extraction!</div></div>',
            unsafe_allow_html=True,
        )
        return

    total_tables = sum(h.get("summary", {}).get("total_tables", 0) for h in history)
    total_fds    = sum(h.get("summary", {}).get("total_functional_dependencies", 0) for h in history)
    db_types     = len({h["db_type"] for h in history})
    st.markdown(
        '<div class="stat-row">'
        + _stat_card(len(history), "Total Runs",     "#63b3ed")
        + _stat_card(total_tables, "Tables Scanned", "#b794f4")
        + _stat_card(total_fds,    "FDs Found",       "#68d391")
        + _stat_card(db_types,     "DB Types Used",   "#f6ad55")
        + "</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([3, 1])
    filter_text = c1.text_input("Filter by database / host", placeholder="Search history…",
                                 label_visibility="collapsed")
    filter_type = c2.selectbox("DB type", ["All"] + list(DB_META.keys()), label_visibility="collapsed")

    filtered = [
        h for h in history
        if (filter_type == "All" or h.get("db_type") == filter_type)
        and (not filter_text or any(
            filter_text.lower() in str(h.get(k, "")).lower()
            for k in ("host", "database", "schema", "db_type")
        ))
    ]
    st.markdown(
        f'<div style="color:#475569;font-size:0.8rem;margin-bottom:1rem">'
        f'Showing {len(filtered)} of {len(history)} runs</div>',
        unsafe_allow_html=True,
    )

    to_delete = None
    for h in filtered:
        m       = DB_META.get(h["db_type"], {"icon": "🗄️", "color": "#94a3b8"})
        summary = h.get("summary", {})
        ts      = h.get("timestamp", "")[:19].replace("T", " ")
        color   = m["color"]

        cols = st.columns([6, 1, 1, 1])
        with cols[0]:
            st.markdown(
                f'<div class="hcard" style="border-left:3px solid {color}">'
                f'<div class="hcard-title">{db_badge(h["db_type"])} &nbsp; '
                f'{h.get("database","—")} / {h.get("schema","—")}</div>'
                f'<div class="hcard-meta">🕐 {ts} &nbsp;|&nbsp; host: {h.get("host","—")}'
                f' &nbsp;|&nbsp; id: {h["id"][:8]}…</div>'
                f'<div style="margin-top:0.7rem;display:flex;gap:1.2rem;font-size:0.82rem">'
                f'<span style="color:#63b3ed">📋 {summary.get("total_tables",0)} tables</span>'
                f'<span style="color:#b794f4">⟶ {summary.get("total_functional_dependencies",0)} FDs</span>'
                f'<span style="color:#68d391">⊆ {summary.get("total_inclusion_dependencies",0)} INDs</span>'
                f'<span style="color:#f6ad55">⇌ {summary.get("total_cardinality_relationships",0)} cardinalities</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        with cols[1]:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("⬇ JSON", key=f"dl_{h['id']}"):
                try:
                    report_data = api.get_history_report(h["id"])
                    st.download_button(
                        "Save JSON",
                        data=__import__("json").dumps(report_data, indent=2, default=str),
                        file_name=f"metadata_{h['db_type']}_{h['timestamp'][:10]}.json",
                        mime="application/json",
                        key=f"save_{h['id']}",
                    )
                except Exception as e:
                    st.error(str(e))
        with cols[2]:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔎 Load", key=f"load_{h['id']}"):
                try:
                    report = api.get_history_report(h["id"])
                    st.session_state.last_report   = report
                    st.session_state.last_run_id   = h["id"]
                    st.session_state.last_run_meta = h
                    st.session_state.page          = "extract"
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not load report: {e}")
        with cols[3]:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️ Del", key=f"del_{h['id']}"):
                to_delete = h["id"]

    if to_delete:
        try:
            api.delete_history(to_delete)
        except Exception as e:
            st.error(f"Delete failed: {e}")
        st.rerun()


# ── View: Search ───────────────────────────────────────────────────────────────
def _search_view() -> None:
    st.markdown(
        '<div class="hero"><div class="hero-title">Search Metadata</div>'
        '<div class="hero-sub">Full-text search across all saved metadata — tables, columns, dependencies.</div></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([4, 1, 1])
    query     = c1.text_input("Search", placeholder="e.g. customer_id, VARCHAR, orders …",
                               label_visibility="collapsed")
    scope_map = {"Tables & Columns": "tables", "Functional Deps": "fds", "All": "all"}
    scope_lbl = c2.selectbox("Scope", list(scope_map.keys()), label_visibility="collapsed")
    filter_db = c3.selectbox("DB type", ["All"] + list(DB_META.keys()), label_visibility="collapsed")

    if not query.strip():
        st.markdown(
            '<div class="empty-state"><div class="empty-state-icon">🔍</div>'
            '<div class="empty-state-text">Type a keyword to search all saved metadata.</div></div>',
            unsafe_allow_html=True,
        )
        return

    with st.spinner("Searching…"):
        try:
            results = api.search(
                q=query.strip(),
                scope=scope_map[scope_lbl],
                db_type=filter_db.lower() if filter_db != "All" else "all",
            )
        except Exception as e:
            st.error(f"Search failed: {e}")
            return

    if not results:
        st.markdown(
            '<div class="empty-state"><div class="empty-state-icon">😶</div>'
            f'<div class="empty-state-text">No results for "<b>{query}</b>".</div></div>',
            unsafe_allow_html=True,
        )
        return

    kind_icons  = {"table": "📋", "column": "🔲", "fd": "⟶"}
    kind_colors = {"table": "#63b3ed", "column": "#b794f4", "fd": "#68d391"}

    st.markdown(
        f'<div style="color:#475569;font-size:0.82rem;margin-bottom:1rem">'
        f'{len(results)} result{"s" if len(results)!=1 else ""} for '
        f'<b style="color:#e2e8f0">{query}</b></div>',
        unsafe_allow_html=True,
    )

    for r in results:
        icon    = kind_icons.get(r["kind"], "•")
        color   = kind_colors.get(r["kind"], "#94a3b8")
        hi_m    = _highlight(r["match"],   query)
        hi_d    = _highlight(r["detail"],  query)
        hi_c    = _highlight(r["context"], query)
        db_bdg  = db_badge(r.get("db_type", ""))

        col_res, col_load = st.columns([8, 1])
        with col_res:
            st.markdown(
                f'<div class="s-result">'
                f'<div class="s-result-title">'
                f'<span style="color:{color};margin-right:0.4rem">{icon}</span>{hi_m}'
                f'<span style="font-size:0.72rem;color:#475569;margin-left:0.6rem">{r["kind"].upper()}</span>'
                f'&nbsp; {db_bdg}</div>'
                f'<div class="s-result-sub">{hi_d} &nbsp;·&nbsp; {hi_c}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_load:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Load", key=f"sr_load_{r['run_id']}_{r['match'][:20]}"):
                try:
                    report = api.get_history_report(r["run_id"])
                    st.session_state.last_report   = report
                    st.session_state.last_run_id   = r["run_id"]
                    st.session_state.last_run_meta = {"db_type": r.get("db_type", "")}
                    st.session_state.page          = "extract"
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not load: {e}")


# ── View: Ontology ─────────────────────────────────────────────────────────────
_ONTO_NODES = [
    ("load",      "📂", "Loading metadata report"),
    ("build",     "🔨", "Building OWL ontology"),
    ("serialize", "💾", "Serialising to RDF/Turtle"),
]


def _onto_node_html(nodes_state: Dict[str, str]) -> str:
    icons  = {n[0]: n[1] for n in _ONTO_NODES}
    labels = {n[0]: n[2] for n in _ONTO_NODES}
    items  = []
    for name, _, _ in _ONTO_NODES:
        s      = nodes_state.get(name, "pending")
        prefix = {"running": "⟳ ", "done": "✓ ", "error": "✗ "}.get(s, "○ ")
        items.append(
            f'<div class="node-item node-{s}">'
            f'{icons[name]} {prefix}{labels[name]}</div>'
        )
    return "<div>" + "".join(items) + "</div>"


def _ontology_view() -> None:
    st.markdown(
        '<div class="hero"><div class="hero-title">Ontology Generator</div>'
        '<div class="hero-sub">Convert any extracted schema into a formal OWL/RDF ontology '
        '— view, edit, and save it right here before downloading.</div></div>',
        unsafe_allow_html=True,
    )

    if not onto_api.health():
        st.markdown(
            f'<div class="banner-err">⚠ Ontology API is unreachable at {ONTOLOGY_API_URL}. '
            f'Make sure the ontology-api container is running.</div>',
            unsafe_allow_html=True,
        )
        return

    history = api.get_history()
    if not history:
        st.markdown(
            '<div class="empty-state"><div class="empty-state-icon">🦉</div>'
            '<div class="empty-state-text">No extractions yet. Run a metadata extraction first.</div></div>',
            unsafe_allow_html=True,
        )
        return

    # Filter to only runs that have a valid, non-empty report accessible
    valid_history = [h for h in history if h.get("report_path")]
    if not valid_history:
        st.markdown(
            '<div class="empty-state"><div class="empty-state-icon">🦉</div>'
            '<div class="empty-state-text">No completed extractions with valid reports found. '
            'Run a metadata extraction first.</div></div>',
            unsafe_allow_html=True,
        )
        return

    # ── Config + Progress ─────────────────────────────────────────────────────
    col_cfg, col_right = st.columns([3, 2], gap="large")

    with col_cfg:
        st.markdown('<div class="sec-head">⚙️ Ontology Settings</div>', unsafe_allow_html=True)

        run_options = {
            f'{h.get("db_type","?").upper()}  ·  {h.get("database","?")} / '
            f'{h.get("schema","?")}  ({h.get("timestamp","")[:10]})': h["id"]
            for h in valid_history
        }
        selected_label = st.selectbox("Source extraction run", list(run_options.keys()))
        run_id = run_options[selected_label]

        c1, c2 = st.columns(2)
        ontology_name = c1.text_input("Ontology name", value="DatabaseOntology")
        base_uri      = c2.text_input("Base URI", value="http://metadata-agent.io/ontology/")

        fmt_map    = {"Turtle (.ttl)": "turtle", "RDF/XML (.owl)": "xml", "N3 (.n3)": "n3"}
        fmt_lbl    = st.selectbox("Serialisation format", list(fmt_map.keys()))
        incl_stats = st.checkbox("Annotate properties with column statistics", value=True)

        gen_btn = st.button(
            "🦉  Generate Ontology", type="primary", use_container_width=True,
            disabled=bool(st.session_state.onto_job_id),
        )

    with col_right:
        st.markdown('<div class="sec-head">📡 Generation Progress</div>', unsafe_allow_html=True)
        prog_area  = st.empty()
        error_area = st.empty()

        job_id = st.session_state.onto_job_id
        if job_id:
            try:
                status = onto_api.get_job(job_id)
            except Exception as e:
                error_area.markdown(
                    f'<div class="banner-err">⚠ API error: {e}</div>', unsafe_allow_html=True)
                st.session_state.onto_job_id = None
                status = None

            if status:
                completed = set(status.get("completed_nodes") or [])
                current   = status.get("current_node") or ""
                ns_map    = {
                    name: ("done"    if name in completed else
                           "running" if name == current and status["status"] == "running" else
                           "pending")
                    for name, _, _ in _ONTO_NODES
                }
                prog_area.markdown(_onto_node_html(ns_map), unsafe_allow_html=True)

                if status["status"] == "done":
                    st.session_state.onto_result      = status
                    st.session_state.onto_last_job_id = job_id
                    st.session_state.onto_content     = None  # will be fetched below
                    st.session_state.onto_job_id      = None
                    st.balloons()
                    st.rerun()
                elif status["status"] == "error":
                    for k in ns_map:
                        if ns_map[k] == "running":
                            ns_map[k] = "error"
                    prog_area.markdown(_onto_node_html(ns_map), unsafe_allow_html=True)
                    error_area.markdown(
                        f'<div class="banner-err">⚠ Generation failed: '
                        f'{status.get("error","unknown error")}</div>',
                        unsafe_allow_html=True,
                    )
                    st.session_state.onto_job_id = None
                else:
                    time.sleep(1.5)
                    st.rerun()
        else:
            idle = {n[0]: "pending" for n in _ONTO_NODES}
            prog_area.markdown(_onto_node_html(idle), unsafe_allow_html=True)

    # ── Trigger generation ────────────────────────────────────────────────────
    if gen_btn:
        try:
            report = api.get_history_report(run_id)
        except Exception as e:
            st.error(f"Could not load metadata report: {e}")
            return
        if not report:
            st.error(
                "The selected extraction run has an empty report. "
                "The extraction may have failed or produced no data. "
                "Please re-run the extraction and try again."
            )
            return
        try:
            jid = onto_api.generate({
                "report":             report,
                "base_uri":           base_uri.strip() or "http://metadata-agent.io/ontology/",
                "ontology_name":      ontology_name.strip() or "DatabaseOntology",
                "serialize_format":   fmt_map[fmt_lbl],
                "include_statistics": incl_stats,
            })
            st.session_state.onto_job_id      = jid
            st.session_state.onto_last_job_id = jid
            st.session_state.onto_result      = None
            st.session_state.onto_content     = None
            st.rerun()
        except Exception as e:
            st.error(f"Failed to start ontology generation: {e}")
        return

    # ── Ontology result panel ─────────────────────────────────────────────────
    result   = st.session_state.onto_result
    last_jid = st.session_state.onto_last_job_id

    if not result or not last_jid or st.session_state.onto_job_id:
        # Show list of previously generated ontologies from this session
        done_list = [j for j in onto_api.list_jobs() if j.get("status") == "done"]
        if done_list:
            st.markdown(
                "<hr style='border:none;border-top:1px solid rgba(255,255,255,0.06);margin:1.5rem 0'>",
                unsafe_allow_html=True,
            )
            st.markdown('<div class="sec-head">📚 Generated Ontologies</div>',
                        unsafe_allow_html=True)
            for j in done_list:
                cols = st.columns([7, 1, 1])
                with cols[0]:
                    st.markdown(
                        f'<div class="hcard">'
                        f'<div class="hcard-title">🦉 {j.get("ontology_name","Ontology")}</div>'
                        f'<div class="hcard-meta">'
                        f'{j.get("serialize_format","turtle").upper()} &nbsp;·&nbsp; '
                        f'{j.get("class_count",0)} classes &nbsp;·&nbsp; '
                        f'{j.get("property_count",0)} props &nbsp;·&nbsp; '
                        f'{j.get("triple_count",0)} triples</div></div>',
                        unsafe_allow_html=True,
                    )
                with cols[1]:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("🔎 View", key=f"onto_view_{j['id']}"):
                        st.session_state.onto_result      = j
                        st.session_state.onto_last_job_id = j["id"]
                        st.session_state.onto_content     = None
                        st.rerun()
                with cols[2]:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("⬇", key=f"onto_dl_{j['id']}"):
                        try:
                            content_bytes, filename = onto_api.get_bytes(j["id"])
                            st.download_button("Save", data=content_bytes, file_name=filename,
                                               mime="text/turtle", key=f"onto_save_{j['id']}")
                        except Exception as e:
                            st.error(str(e))
        return

    # Stats
    st.markdown(
        "<hr style='border:none;border-top:1px solid rgba(255,255,255,0.06);margin:1.5rem 0'>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="banner-ok" style="margin-bottom:1rem">✓ Ontology ready</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="stat-row">'
        + _stat_card(result.get("class_count",    0), "OWL Classes",    "#63b3ed")
        + _stat_card(result.get("property_count", 0), "OWL Properties", "#b794f4")
        + _stat_card(result.get("triple_count",   0), "RDF Triples",    "#68d391")
        + "</div>",
        unsafe_allow_html=True,
    )

    # Fetch content once and cache in session state
    if st.session_state.onto_content is None:
        try:
            data = onto_api.get_content(last_jid)
            st.session_state.onto_content = data.get("content", "")
        except Exception as e:
            st.error(f"Could not fetch ontology content: {e}")
            return

    content_text = st.session_state.onto_content or ""
    fmt          = result.get("serialize_format", "turtle")

    # ── Read-only rendered view ────────────────────────────────────────────────
    st.markdown('<div class="sec-head" style="margin-top:1.5rem">📄 Ontology Content</div>',
                unsafe_allow_html=True)

    with st.expander("View rendered ontology", expanded=True):
        st.code(content_text, language="turtle" if fmt == "turtle" else "xml")

    # ── Editable view ──────────────────────────────────────────────────────────
    st.markdown('<div class="sec-head" style="margin-top:1rem">✏️ Edit Ontology</div>',
                unsafe_allow_html=True)
    st.caption("Edit the ontology directly. Click Save to persist changes to the server.")

    edited = st.text_area(
        "Ontology source",
        value=content_text,
        height=400,
        key="onto_editor",
        label_visibility="collapsed",
    )

    save_col, dl_col, _ = st.columns([2, 2, 4])
    with save_col:
        if st.button("💾  Save Changes", use_container_width=True, type="primary"):
            if not edited.strip():
                st.error("Cannot save empty content.")
            else:
                try:
                    onto_api.save_content(last_jid, edited)
                    st.session_state.onto_content = edited
                    st.success("Saved successfully.")
                except Exception as e:
                    st.error(f"Save failed: {e}")

    with dl_col:
        try:
            content_bytes, filename = onto_api.get_bytes(last_jid)
            st.download_button(
                "⬇  Download File",
                data=content_bytes,
                file_name=filename,
                mime="text/turtle" if fmt == "turtle" else "application/rdf+xml",
                use_container_width=True,
                key="onto_dl_main",
            )
        except Exception as e:
            st.error(f"Download error: {e}")

    if result.get("errors"):
        with st.expander(f"⚠ {len(result['errors'])} warning(s) during generation"):
            for err in result["errors"]:
                st.markdown(f"- {err}")


# ── Knowledge Graph view ───────────────────────────────────────────────────────
_KG_NODES = [
    ("parse",     "📂", "Parsing OWL ontology"),
    ("translate", "🔄", "Translating to graph nodes/edges"),
    ("execute",   "⚡", "Persisting snapshot"),
]
_KG_LOAD_NODES = [
    ("fetch", "📥", "Loading graph from database"),
]


def _kg_node_html(nodes_state: Dict[str, str], load_mode: bool = False) -> str:
    node_list = _KG_LOAD_NODES if load_mode else _KG_NODES
    icons  = {n[0]: n[1] for n in node_list}
    labels = {n[0]: n[2] for n in node_list}
    items  = []
    for name, _, _ in node_list:
        s      = nodes_state.get(name, "pending")
        prefix = {"running": "⟳ ", "done": "✓ ", "error": "✗ "}.get(s, "○ ")
        items.append(
            f'<div class="node-item node-{s}">'
            f'{icons[name]} {prefix}{labels[name]}</div>'
        )
    return "<div>" + "".join(items) + "</div>"


def _render_kg_graph(graph_data: Dict) -> None:
    """Render the knowledge graph using pyvis embedded in an iframe."""
    import streamlit.components.v1 as stc

    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    if not nodes:
        st.info("No nodes found in graph data.")
        return

    try:
        from pyvis.network import Network
    except ImportError:
        st.warning("pyvis is not installed in the UI image. Rebuild with updated requirements.ui.txt.")
        return

    net = Network(height="620px", width="100%", bgcolor="#0a0e1a",
                  font_color="#e2e8f0", directed=True)
    net.set_options("""{
      "physics": {
        "forceAtlas2Based": {"gravitationalConstant": -30, "centralGravity": 0.003,
                             "springLength": 200, "springConstant": 0.18},
        "maxVelocity": 100, "solver": "forceAtlas2Based",
        "timestep": 0.35, "stabilization": {"iterations": 180}
      },
      "nodes": {"borderWidth": 2, "font": {"size": 13, "color": "#e2e8f0"}},
      "edges": {
        "font": {"size": 11, "color": "#94a3b8", "strokeWidth": 0},
        "smooth": {"type": "dynamic"},
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.8}}
      },
      "interaction": {"hover": true, "tooltipDelay": 150, "zoomView": true, "dragView": true}
    }""")

    for node in nodes:
        net.add_node(
            node["id"],
            label=node["label"],
            title=node.get("title", ""),
            color={"background": "#1e2740", "border": "#63b3ed",
                   "highlight": {"background": "#2a3a5c", "border": "#93c5fd"}},
            size=node.get("size", 20),
            font={"color": "#e2e8f0"},
        )

    for edge in edges:
        net.add_edge(
            edge["from"], edge["to"],
            label=edge.get("label", ""),
            title=edge.get("title", ""),
            color={"color": "#68d391", "highlight": "#a7f3d0"},
        )

    stc.html(net.generate_html(), height=630, scrolling=False)


def _kg_view() -> None:
    st.markdown(
        '<div class="hero"><div class="hero-title">Knowledge Graph</div>'
        '<div class="hero-sub">Generate a new graph from an OWL ontology, incrementally update '
        'an existing graph, or load a graph already stored in your database.</div></div>',
        unsafe_allow_html=True,
    )

    if not kg_api.health():
        st.markdown(
            f'<div class="banner-err">⚠ Knowledge Graph API is unreachable at {KG_API_URL}. '
            f'Make sure the kg-api container is running.</div>',
            unsafe_allow_html=True,
        )
        return

    # ── Mode selector ─────────────────────────────────────────────────────────
    kg_mode = st.radio(
        "Operation",
        ["🆕 Generate New Graph", "🔄 Incremental Update", "📥 Load Existing Graph"],
        horizontal=True,
        key="kg_mode_radio",
    )
    is_load   = kg_mode == "📥 Load Existing Graph"
    is_update = kg_mode == "🔄 Incremental Update"

    onto_jobs = [j for j in onto_api.list_jobs() if j.get("status") == "done"]
    if not is_load and not onto_jobs:
        st.markdown(
            '<div class="empty-state"><div class="empty-state-icon">🕸️</div>'
            '<div class="empty-state-text">No ontologies available. '
            'Generate an ontology in the Ontology Generator first, or choose '
            '"Load Existing Graph" to query a previously built graph.</div></div>',
            unsafe_allow_html=True,
        )
        return

    # ── Config panel ──────────────────────────────────────────────────────────
    col_cfg, col_prog = st.columns([3, 2], gap="large")

    with col_cfg:
        st.markdown('<div class="sec-head">⚙️ Knowledge Graph Settings</div>', unsafe_allow_html=True)

        # Ontology selector — only for generate/update modes
        onto_job_id = None
        if not is_load:
            onto_options = {
                f'{j.get("ontology_name","Ontology")}  ·  '
                f'{j.get("serialize_format","turtle").upper()}  ·  '
                f'{j.get("class_count",0)} classes  ·  '
                f'{j.get("triple_count",0)} triples': j["id"]
                for j in onto_jobs
            }
            selected_onto_label = st.selectbox("Source ontology", list(onto_options.keys()))
            onto_job_id = onto_options[selected_onto_label]

        kg_id = st.text_input(
            "KG id", value="default", key="kg_id_input",
            help="Isolates this graph's nodes/edges in the shared KG snapshot store.",
        )

        st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)

        # Mode-specific options
        if is_load:
            st.info("📥 Loads all nodes and edges already persisted for this KG id.")
            clear_existing = False
        elif is_update:
            st.info("🔄 Merges new nodes and relationships into the existing graph without clearing it.")
            clear_existing = False
        else:
            clear_existing = st.checkbox("Clear existing graph before loading", value=False,
                                          key="kg_clear_existing")

        btn_label = ("📥  Load from Database" if is_load
                     else "🔄  Incremental Update" if is_update
                     else "🕸️  Create Knowledge Graph")
        action_btn = st.button(
            btn_label, type="primary", use_container_width=True,
            disabled=bool(st.session_state.kg_job_id),
        )

    # ── Progress panel ────────────────────────────────────────────────────────
    with col_prog:
        st.markdown('<div class="sec-head">📡 Pipeline Progress</div>', unsafe_allow_html=True)
        prog_area  = st.empty()
        error_area = st.empty()

        job_id        = st.session_state.kg_job_id
        active_mode   = st.session_state.kg_mode
        load_progress = (active_mode == "load")

        if job_id:
            try:
                status = kg_api.get_job(job_id)
            except Exception as e:
                error_area.markdown(
                    f'<div class="banner-err">⚠ API error: {e}</div>', unsafe_allow_html=True)
                st.session_state.kg_job_id = None
                status = None

            if status:
                completed = set(status.get("completed_nodes") or [])
                current   = status.get("current_node") or ""
                node_list = _KG_LOAD_NODES if load_progress else _KG_NODES
                ns_map    = {
                    name: ("done"    if name in completed else
                           "running" if name == current and status["status"] == "running" else
                           "pending")
                    for name, _, _ in node_list
                }
                prog_area.markdown(_kg_node_html(ns_map, load_progress), unsafe_allow_html=True)

                if status["status"] == "done":
                    try:
                        graph_data = kg_api.get_graph(job_id)
                    except Exception:
                        graph_data = {"nodes": [], "edges": []}
                    st.session_state.kg_result     = status
                    st.session_state.kg_graph_data = graph_data
                    st.session_state.kg_job_id     = None
                    st.balloons()
                    st.rerun()
                elif status["status"] == "error":
                    for k in ns_map:
                        if ns_map[k] == "running":
                            ns_map[k] = "error"
                    prog_area.markdown(_kg_node_html(ns_map, load_progress), unsafe_allow_html=True)
                    error_area.markdown(
                        f'<div class="banner-err">⚠ Pipeline failed: '
                        f'{status.get("error","unknown error")}</div>',
                        unsafe_allow_html=True,
                    )
                    st.session_state.kg_job_id = None
                else:
                    time.sleep(1.5)
                    st.rerun()
        else:
            node_list = _KG_LOAD_NODES if is_load else _KG_NODES
            prog_area.markdown(
                _kg_node_html({n[0]: "pending" for n in node_list}, is_load),
                unsafe_allow_html=True,
            )

    # ── Trigger ───────────────────────────────────────────────────────────────
    if action_btn:
        if is_load:
            # Load existing graph snapshot — no ontology needed
            payload = {"kg_id": kg_id.strip() or "default"}
            try:
                jid = kg_api.fetch(payload)
                st.session_state.kg_job_id    = jid
                st.session_state.kg_result    = None
                st.session_state.kg_graph_data = None
                st.session_state.kg_mode      = "load"
                st.rerun()
            except Exception as e:
                st.error(f"Failed to start load job: {e}")
        else:
            # Generate new or incremental update — needs ontology
            try:
                onto_content    = onto_api.get_content(onto_job_id)
                ontology_text   = onto_content.get("content", "")
                ontology_format = onto_content.get("format", "turtle")
            except Exception as e:
                st.error(f"Could not fetch ontology content: {e}")
                return

            payload = {
                "ontology_text":   ontology_text,
                "ontology_format": ontology_format,
                "kg_id":           kg_id.strip() or "default",
                "mode":            "update" if is_update else "generate",
                "clear_existing":  clear_existing,
            }

            try:
                jid = kg_api.generate(payload)
                st.session_state.kg_job_id    = jid
                st.session_state.kg_result    = None
                st.session_state.kg_graph_data = None
                st.session_state.kg_mode      = "update" if is_update else "generate"
                st.rerun()
            except Exception as e:
                st.error(f"Failed to start KG job: {e}")
        return

    # ── Result panel ──────────────────────────────────────────────────────────
    result     = st.session_state.kg_result
    graph_data = st.session_state.kg_graph_data

    if not result or st.session_state.kg_job_id:
        # Show list of past KG jobs
        done_jobs = [j for j in kg_api.list_jobs() if j.get("status") == "done"]
        if done_jobs:
            st.markdown(
                "<hr style='border:none;border-top:1px solid rgba(255,255,255,0.06);margin:1.5rem 0'>",
                unsafe_allow_html=True,
            )
            st.markdown('<div class="sec-head">📚 Previous Knowledge Graphs</div>',
                        unsafe_allow_html=True)
            for j in done_jobs:
                cols = st.columns([7, 2])
                with cols[0]:
                    job_mode = j.get("mode", "generate")
                    mode_tag = {"load": "📥 Loaded", "update": "🔄 Updated"}.get(job_mode, "🆕 Generated")
                    st.markdown(
                        f'<div class="hcard">'
                        f'<div class="hcard-title">🕸️ Knowledge Graph &nbsp; '
                        f'<span style="font-size:0.75rem;opacity:0.7">{mode_tag}</span></div>'
                        f'<div class="hcard-meta">'
                        f'{j.get("node_count",0)} nodes &nbsp;·&nbsp; '
                        f'{j.get("edge_count",0)} edges &nbsp;·&nbsp; '
                        f'{j.get("executed_count",0)} queries executed</div></div>',
                        unsafe_allow_html=True,
                    )
                with cols[1]:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("🔎 View", key=f"kg_view_{j['id']}"):
                        try:
                            gd = kg_api.get_graph(j["id"])
                        except Exception:
                            gd = {"nodes": [], "edges": []}
                        st.session_state.kg_result     = j
                        st.session_state.kg_graph_data = gd
                        st.rerun()
        return

    # ── Stats ─────────────────────────────────────────────────────────────────
    st.markdown(
        "<hr style='border:none;border-top:1px solid rgba(255,255,255,0.06);margin:1.5rem 0'>",
        unsafe_allow_html=True,
    )
    executed = result.get("executed_count", 0)
    job_mode = result.get("mode", "generate")
    mode_tag = {"load": "loaded from database", "update": "incrementally updated"}.get(
        job_mode, f"{executed} queries executed" if executed else "preview mode"
    )
    mode_lbl = mode_tag
    st.markdown(
        f'<div class="banner-ok" style="margin-bottom:1rem">✓ Knowledge graph ready — {mode_lbl}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="stat-row">'
        + _stat_card(result.get("node_count", 0), "Graph Nodes",    "#63b3ed")
        + _stat_card(result.get("edge_count",  0), "Graph Edges",   "#68d391")
        + _stat_card(result.get("executed_count", 0), "Queries Run", "#b794f4")
        + "</div>",
        unsafe_allow_html=True,
    )

    # ── Interactive graph ─────────────────────────────────────────────────────
    st.markdown('<div class="sec-head" style="margin-top:1.5rem">🕸️ Graph Visualisation</div>',
                unsafe_allow_html=True)
    st.caption("Pan: drag · Zoom: scroll · Hover nodes/edges for details.")

    if graph_data:
        _render_kg_graph(graph_data)
    else:
        st.info("No graph data available.")

    # ── Generated queries ─────────────────────────────────────────────────────
    if st.session_state.kg_result:
        job_id_for_queries = (
            st.session_state.kg_result.get("id") or
            next((j["id"] for j in kg_api.list_jobs()
                  if j.get("status") == "done"), None)
        )
        if job_id_for_queries:
            with st.expander("📋 View generated queries", expanded=False):
                try:
                    q_data = kg_api.get_queries(job_id_for_queries)
                    q_list = q_data.get("queries", [])
                    st.caption(f"{len(q_list)} Cypher-style statements")
                    st.code("\n\n".join(q_list), language="cypher")
                    st.download_button(
                        "⬇ Download queries",
                        data="\n\n".join(q_list),
                        file_name="kg_queries.cypher",
                        mime="text/plain",
                        use_container_width=False,
                    )
                except Exception as e:
                    st.warning(f"Could not fetch queries: {e}")

    if result.get("errors"):
        with st.expander(f"⚠ {len(result['errors'])} warning(s)"):
            for err in result["errors"]:
                st.markdown(f"- {err}")


# ── Dialog with Data view ──────────────────────────────────────────────────────
_DIALOG_NODES = [
    ("understand", "🔎", "Analysing schema context"),
    ("plan",       "📝", "Planning SQL queries"),
    ("execute",    "⚡", "Executing queries"),
    ("synthesize", "💡", "Synthesising insights"),
]


def _dialog_node_html(nodes_state: Dict[str, str]) -> str:
    icons  = {n[0]: n[1] for n in _DIALOG_NODES}
    labels = {n[0]: n[2] for n in _DIALOG_NODES}
    items  = []
    for name, _, _ in _DIALOG_NODES:
        s      = nodes_state.get(name, "pending")
        prefix = {"running": "⟳ ", "done": "✓ ", "error": "✗ "}.get(s, "○ ")
        items.append(
            f'<div class="node-item node-{s}">'
            f'{icons[name]} {prefix}{labels[name]}</div>'
        )
    return "<div>" + "".join(items) + "</div>"


def _render_query_results(results: List[Dict]) -> None:
    """Render each SQL query result as an expandable table."""
    import json as _json
    for qr in results:
        qid   = qr.get("query_id", "")
        desc  = qr.get("description", "")
        sql   = qr.get("sql", "")
        cols  = qr.get("columns") or []
        rows  = qr.get("rows") or []
        err   = qr.get("error")
        count = qr.get("row_count", len(rows))

        header = f"{qid}: {desc}" if desc else qid
        with st.expander(header, expanded=False):
            st.code(sql, language="sql")
            if err:
                st.markdown(
                    f'<div class="banner-err">⚠ Query error: {err}</div>',
                    unsafe_allow_html=True,
                )
            elif cols:
                st.caption(f"{count} row(s) returned" + (" (showing first 500)" if count > 500 else ""))
                import pandas as pd
                df = pd.DataFrame(rows, columns=cols)
                st.dataframe(df, use_container_width=True, height=min(300, 40 + len(df) * 35))
            else:
                st.info("No data returned.")


def _dialog_view() -> None:
    st.markdown(
        '<div class="hero"><div class="hero-title">Dialog with Data</div>'
        '<div class="hero-sub">Ask natural language questions about your data. '
        'The agent traverses the knowledge graph, plans SQL queries, executes them, '
        'and derives insights — all in one pipeline.</div></div>',
        unsafe_allow_html=True,
    )

    if not dialog_api.health():
        st.markdown(
            f'<div class="banner-err">⚠ Dialog API is unreachable at {DIALOG_API_URL}. '
            f'Make sure the dialog-api container is running.</div>',
            unsafe_allow_html=True,
        )
        return

    # ── Config panel ──────────────────────────────────────────────────────────
    col_cfg, col_prog = st.columns([3, 2], gap="large")

    with col_cfg:
        st.markdown('<div class="sec-head">💬 Your Question</div>', unsafe_allow_html=True)

        natural_query = st.text_area(
            "Natural language query",
            placeholder="Which customers placed more than 5 orders in the last 90 days? "
                        "What is the total revenue by product category?",
            height=100,
            label_visibility="collapsed",
        )

        st.markdown('<div class="sec-head" style="margin-top:1rem">🕸️ Schema Context (Knowledge Graph)</div>',
                    unsafe_allow_html=True)
        st.caption("Select a completed KG job so the agent understands your schema.")

        kg_jobs = [j for j in kg_api.list_jobs() if j.get("status") == "done"]
        kg_options: Dict[str, Optional[str]] = {"(no KG — schema-less mode)": None}
        for j in kg_jobs:
            label = (f'{j.get("node_count",0)} nodes · '
                     f'{j.get("edge_count",0)} edges · '
                     f'id={j["id"][:8]}')
            kg_options[label] = j["id"]

        selected_kg_label = st.selectbox("Knowledge graph", list(kg_options.keys()))
        selected_kg_id    = kg_options[selected_kg_label]

        st.markdown('<div class="sec-head" style="margin-top:1rem">🔌 Target Database</div>',
                    unsafe_allow_html=True)

        db_type = st.selectbox(
            "Database type",
            options=list(DB_META.keys()),
            format_func=lambda k: f'{DB_META[k]["icon"]}  {DB_META[k]["label"]}',
            key="dialog_db_type",
        )

        needs_bq   = db_type == "bigquery"
        needs_dfile = db_type in _FILE_BASED

        if needs_dfile:
            # ── Fetch live list of non-expired uploads matching this db_type ──
            all_uploads  = api.list_uploads()
            type_uploads = [u for u in all_uploads if u["db_type"] == db_type]

            d_file_path = ""
            dlg_expires = ""
            dlg_extended = False

            if type_uploads:
                # Build display labels: "filename — expires HH:MM"
                def _fmt_upload(u: Dict) -> str:
                    try:
                        from datetime import datetime, timezone
                        exp_dt  = datetime.fromisoformat(u["expires_at"])
                        exp_str = exp_dt.astimezone().strftime("%H:%M %Z")
                    except Exception:
                        exp_str = "?"
                    ext_tag = " ⟳+15m used" if u["extended"] else ""
                    return f'{u["label"]}  —  expires {exp_str}{ext_tag}'

                upload_labels = [_fmt_upload(u) for u in type_uploads]
                upload_labels.append("⬆ Upload a new file…")

                sel_label = st.selectbox(
                    "Select file", upload_labels, key="dlg_file_sel",
                    help="Choose a file uploaded in this session or still within its 2-hour window."
                )
                sel_idx = upload_labels.index(sel_label)

                if sel_idx < len(type_uploads):
                    # Existing upload selected
                    sel_upload   = type_uploads[sel_idx]
                    d_file_path  = sel_upload["path"]
                    dlg_expires  = sel_upload["expires_at"]
                    dlg_extended = sel_upload["extended"]
                else:
                    sel_upload = None  # "upload new" chosen
            else:
                sel_upload = None

            # Show uploader when no valid files or user chose "Upload new"
            if sel_upload is None:
                if db_type == "csv":
                    dlg_uploaded = st.file_uploader(
                        "Upload CSV file(s)", type=["csv"], accept_multiple_files=True, key="dlg_file_upload"
                    )
                elif db_type == "sqlite":
                    dlg_uploaded = st.file_uploader(
                        "Upload SQLite database", type=["sqlite", "db", "sqlite3"], key="dlg_file_upload"
                    )
                else:
                    dlg_uploaded = st.file_uploader(
                        "Upload Excel file", type=["xlsx", "xls", "xlsm", "xlsb"], key="dlg_file_upload"
                    )
                if st.button("⬆ Upload file", key="dlg_upload_btn",
                             disabled=not bool(dlg_uploaded)):
                    try:
                        with st.spinner("Uploading…"):
                            if db_type == "csv":
                                result = api.upload_files(dlg_uploaded)
                            else:
                                result = api.upload_file(dlg_uploaded.read(), dlg_uploaded.name)
                        st.session_state.dlg_uploaded_path   = result["path"]
                        st.session_state.dlg_upload_expires  = result.get("expires_at", "")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Upload failed: {e}")

                d_file_path = st.session_state.get("dlg_uploaded_path") or ""
                dlg_expires = st.session_state.get("dlg_upload_expires") or ""

            # ── Expiry banner + one-time extend button ─────────────────────────
            if d_file_path:
                try:
                    from datetime import datetime, timezone
                    exp_dt  = datetime.fromisoformat(dlg_expires)
                    exp_str = exp_dt.astimezone().strftime("%H:%M %Z")
                except Exception:
                    exp_str = "?"

                info_col, btn_col = st.columns([3, 1])
                info_col.info(
                    f"File will be deleted at **{exp_str}**. "
                    + ("Extension already used — no further extensions available."
                       if dlg_extended
                       else "You may extend once by +15 min."),
                    icon="⏳",
                )
                if not dlg_extended:
                    with btn_col:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("⏱ +15 min", key="dlg_extend_btn",
                                     help="Extend this file's lifetime by 15 minutes (one time only)"):
                            try:
                                new_exp = api.extend_upload(d_file_path)
                                st.success(
                                    f"Extended! File now expires at "
                                    f"{datetime.fromisoformat(new_exp['expires_at']).astimezone().strftime('%H:%M %Z')}."
                                )
                                st.rerun()
                            except requests.HTTPError as e:
                                if "already_extended" in str(e):
                                    st.warning("This file has already been extended once.")
                                else:
                                    st.error(f"Could not extend: {e}")

            d_host = d_port = d_dbname = d_user = d_password = d_schema_name = ""
            d_project = d_schema = d_creds = ""
        elif needs_bq:
            c1, c2 = st.columns(2)
            d_project = c1.text_input("GCP Project", key="d_project")
            d_schema  = c2.text_input("Dataset", key="d_schema")
            d_creds   = st.text_input("Service account JSON path", key="d_creds")
            d_host = d_port = d_dbname = d_user = d_password = d_schema_name = ""
            d_file_path = ""
        else:
            c1, c2 = st.columns([3, 1])
            d_host = c1.text_input("Host", placeholder="localhost", key="d_host")
            defaults = {"postgres": 5432, "oracle": 1521, "sqlserver": 1433,
                        "teradata": 1025, "redshift": 5439}
            d_port = c2.number_input("Port", value=defaults.get(db_type, 5432),
                                     step=1, key="d_port")
            c3, c4 = st.columns(2)
            d_dbname      = c3.text_input("Database", key="d_dbname")
            d_schema_name = c4.text_input("Schema", value="public", key="d_schema_name")
            c5, c6 = st.columns(2)
            d_user     = c5.text_input("Username", key="d_user")
            d_password = c6.text_input("Password", type="password", key="d_password")
            d_project = d_schema = d_creds = d_file_path = ""

        # ── Schema & table discovery ───────────────────────────────────────────
        st.markdown('<hr style="border:none;border-top:1px solid rgba(255,255,255,0.06);margin:0.8rem 0">', unsafe_allow_html=True)
        dlg_disco_cols = st.columns([2, 1])
        with dlg_disco_cols[0]:
            st.markdown('<div class="sec-head">🔍 Available Schemas & Tables</div>', unsafe_allow_html=True)
        with dlg_disco_cols[1]:
            st.markdown("<br>", unsafe_allow_html=True)
            dlg_disco_btn = st.button("🔍 Discover", key="dlg_disco_btn", use_container_width=True)

        dlg_disco = st.session_state.dlg_disco

        if dlg_disco_btn:
            if needs_dfile:
                dlg_disco_payload: Dict[str, Any] = {
                    "db_type": db_type, "file_path": d_file_path or None,
                }
            elif needs_bq:
                dlg_disco_payload = {
                    "db_type": db_type, "project": d_project or None,
                    "schema_name": d_schema or None, "credentials_path": d_creds or None,
                }
            else:
                dlg_disco_payload = {
                    "db_type": db_type, "host": d_host or None,
                    "port": int(d_port) if d_port else None,
                    "database": d_dbname or None,
                    "schema_name": d_schema_name or None,
                    "username": d_user or None, "password": d_password or None,
                }
            with st.spinner("Connecting and discovering schemas…"):
                try:
                    dlg_disco = api.discover(dlg_disco_payload)
                    st.session_state.dlg_disco = dlg_disco
                except Exception as e:
                    st.error(f"Discovery failed: {e}")
                    dlg_disco = None

        if dlg_disco and dlg_disco.get("schemas"):
            dlg_schemas = dlg_disco["schemas"]
            dlg_tables_map: Dict[str, List[str]] = dlg_disco.get("tables", {})

            # Schema selector — overrides the text input
            typed_schema = d_schema_name or d_schema or ""
            dlg_schema_sel = st.selectbox(
                "Schema / Dataset",
                dlg_schemas,
                index=dlg_schemas.index(typed_schema) if typed_schema in dlg_schemas else 0,
                key="dlg_disco_schema",
            )
            # Push selected schema back to the variables used in the payload below
            if needs_bq:
                d_schema = dlg_schema_sel
            else:
                d_schema_name = dlg_schema_sel

            dlg_avail_tables = dlg_tables_map.get(dlg_schema_sel, [])
            if dlg_avail_tables:
                st.caption(f"📋 {len(dlg_avail_tables)} table(s) in '{dlg_schema_sel}' — select to focus the query")
                dlg_selected_tables = st.multiselect(
                    "Focus on specific tables (leave empty = all)",
                    dlg_avail_tables,
                    key="dlg_disco_tables",
                )
            else:
                dlg_selected_tables = []
                st.caption(f"No tables found in '{dlg_schema_sel}'.")

            if st.button("🔄 Re-discover", key="dlg_redisco_btn"):
                st.session_state.dlg_disco = None
                st.rerun()
        else:
            dlg_selected_tables = []
            st.caption("Click **🔍 Discover** to browse schemas and tables from the connected database.")

        st.markdown('<hr style="border:none;border-top:1px solid rgba(255,255,255,0.06);margin:0.8rem 0">', unsafe_allow_html=True)
        c_q, c_r = st.columns(2)
        max_sql = c_q.number_input("Max SQL queries", min_value=1, max_value=20,
                                    value=5, key="d_max_sql")
        row_lim = c_r.number_input("Row limit per query", min_value=10, max_value=5000,
                                    value=500, key="d_row_limit")

        ask_btn = st.button(
            "💬  Ask the Data", type="primary", use_container_width=True,
            disabled=bool(st.session_state.dialog_job_id),
        )

    # ── Progress panel ────────────────────────────────────────────────────────
    with col_prog:
        st.markdown('<div class="sec-head">📡 Pipeline Progress</div>', unsafe_allow_html=True)
        prog_area  = st.empty()
        error_area = st.empty()

        job_id = st.session_state.dialog_job_id
        if job_id:
            try:
                status = dialog_api.get_job(job_id)
            except Exception as e:
                error_area.markdown(
                    f'<div class="banner-err">⚠ API error: {e}</div>', unsafe_allow_html=True)
                st.session_state.dialog_job_id = None
                status = None

            if status:
                completed = set(status.get("completed_nodes") or [])
                current   = status.get("current_node") or ""
                ns_map    = {
                    name: ("done"    if name in completed else
                           "running" if name == current and status["status"] == "running" else
                           "pending")
                    for name, _, _ in _DIALOG_NODES
                }
                prog_area.markdown(_dialog_node_html(ns_map), unsafe_allow_html=True)

                if status["status"] == "done":
                    try:
                        full = dialog_api.get_results(job_id)
                    except Exception:
                        full = {}
                    st.session_state.dialog_result = {**status, **full}
                    st.session_state.dialog_job_id  = None
                    st.balloons()
                    st.rerun()
                elif status["status"] == "error":
                    for k in ns_map:
                        if ns_map[k] == "running":
                            ns_map[k] = "error"
                    prog_area.markdown(_dialog_node_html(ns_map), unsafe_allow_html=True)
                    error_area.markdown(
                        f'<div class="banner-err">⚠ Pipeline failed: '
                        f'{status.get("error","unknown error")}</div>',
                        unsafe_allow_html=True,
                    )
                    st.session_state.dialog_job_id = None
                else:
                    time.sleep(1.5)
                    st.rerun()
        else:
            prog_area.markdown(
                _dialog_node_html({n[0]: "pending" for n in _DIALOG_NODES}),
                unsafe_allow_html=True,
            )

    # ── Trigger ───────────────────────────────────────────────────────────────
    if ask_btn:
        if not natural_query.strip():
            st.error("Please enter a question.")
            return

        # Fetch KG graph data if a KG job was selected
        kg_nodes: List[Dict] = []
        kg_edges: List[Dict] = []
        if selected_kg_id:
            try:
                gd = kg_api.get_graph(selected_kg_id)
                kg_nodes = gd.get("nodes", [])
                kg_edges = gd.get("edges", [])
            except Exception as e:
                st.warning(f"Could not load KG graph data: {e}. Proceeding in schema-less mode.")

        # If the user selected specific tables via discovery, append a hint to
        # the natural query so the LLM knows which tables to focus on.
        effective_query = natural_query.strip()
        if dlg_selected_tables:
            table_hint = ", ".join(dlg_selected_tables)
            effective_query += f"\n\n[Focus on these tables: {table_hint}]"

        if needs_dfile:
            payload: Dict[str, Any] = {
                "natural_query": effective_query,
                "kg_nodes":      kg_nodes,
                "kg_edges":      kg_edges,
                "db_type":       db_type,
                "db_file_path":  d_file_path or None,
                "db_schema":     "",   # file-based sources have no schema prefix
                "max_sql_queries": int(max_sql),
                "row_limit":     int(row_lim),
            }
        elif needs_bq:
            db_extra: Dict = {}
            if d_creds:
                db_extra["credentials_path"] = d_creds
            payload = {
                "natural_query": effective_query,
                "kg_nodes":      kg_nodes,
                "kg_edges":      kg_edges,
                "db_type":       db_type,
                "db_name":       d_project,
                "db_schema":     d_schema,
                "db_extra":      db_extra,
                "max_sql_queries": int(max_sql),
                "row_limit":     int(row_lim),
            }
        else:
            payload = {
                "natural_query": effective_query,
                "kg_nodes":      kg_nodes,
                "kg_edges":      kg_edges,
                "db_type":       db_type,
                "db_host":       d_host,
                "db_port":       int(d_port),
                "db_name":       d_dbname,
                "db_schema":     d_schema_name or "public",
                "db_user":       d_user,
                "db_password":   d_password,
                "max_sql_queries": int(max_sql),
                "row_limit":     int(row_lim),
            }

        try:
            jid = dialog_api.query(payload)
            st.session_state.dialog_job_id = jid
            st.session_state.dialog_result  = None
            st.rerun()
        except Exception as e:
            st.error(f"Failed to start dialog job: {e}")
        return

    # ── Results panel ─────────────────────────────────────────────────────────
    result = st.session_state.dialog_result

    if not result or st.session_state.dialog_job_id:
        hr = "<hr style='border:none;border-top:1px solid rgba(255,255,255,0.06);margin:1.5rem 0'>"

        # ── Cached NLQs ───────────────────────────────────────────────────────
        cached_list = dialog_api.list_cache()
        if cached_list:
            st.markdown(hr, unsafe_allow_html=True)
            cache_head_cols = st.columns([6, 3])
            cache_head_cols[0].markdown(
                '<div class="sec-head">💾 Cached Queries</div>', unsafe_allow_html=True
            )
            with cache_head_cols[1]:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🗑 Clear all cache", key="dialog_clear_cache"):
                    n = dialog_api.clear_cache()
                    st.success(f"Cleared {n} cached entr{'y' if n == 1 else 'ies'}.")
                    st.rerun()

            for entry in cached_list:
                ck   = entry.get("cache_key", "")
                nq   = entry.get("natural_query", "?")
                dbfp = entry.get("db_fingerprint", "")
                kgfp = entry.get("kg_fingerprint", "")
                cat  = entry.get("cached_at", "")[:19].replace("T", " ")
                cols = st.columns([6, 1, 1])
                with cols[0]:
                    st.markdown(
                        f'<div class="hcard">'
                        f'<div class="hcard-title">💾 {nq[:90]}</div>'
                        f'<div class="hcard-meta">'
                        f'{dbfp} &nbsp;·&nbsp; {kgfp} &nbsp;·&nbsp; cached {cat}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with cols[1]:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("⚡ Use", key=f"cache_use_{ck[:12]}"):
                        # Submit a new query that will hit the cache instantly
                        payload_cached: Dict[str, Any] = {
                            "natural_query": nq,
                            "kg_nodes":      [],
                            "kg_edges":      [],
                            "skip_cache":    False,
                        }
                        try:
                            resp = dialog_api.query(payload_cached)
                            jid  = resp if isinstance(resp, str) else resp
                            st.session_state.dialog_job_id = jid
                            st.session_state.dialog_result  = None
                            st.rerun()
                        except Exception as e:
                            st.error(f"Could not load from cache: {e}")
                with cols[2]:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("🗑", key=f"cache_del_{ck[:12]}"):
                        dialog_api.delete_cache_entry(ck)
                        st.rerun()

        # ── Previous dialog sessions ──────────────────────────────────────────
        done_list = [j for j in dialog_api.list_jobs() if j.get("status") == "done"]
        if done_list:
            st.markdown(hr, unsafe_allow_html=True)
            st.markdown('<div class="sec-head">📚 Previous Dialogs</div>', unsafe_allow_html=True)
            for j in done_list:
                cols = st.columns([7, 2])
                with cols[0]:
                    cache_badge = ' &nbsp;<span style="font-size:0.7rem;color:#68d391">⚡ cached</span>' if j.get("cache_hit") else ""
                    st.markdown(
                        f'<div class="hcard">'
                        f'<div class="hcard-title">💬 {j.get("natural_query","?")[:80]}{cache_badge}</div>'
                        f'<div class="hcard-meta">'
                        f'{j.get("db_type","?").upper()} &nbsp;·&nbsp; '
                        f'{j.get("query_count",0)} queries &nbsp;·&nbsp; '
                        f'{j.get("result_count",0)} results</div></div>',
                        unsafe_allow_html=True,
                    )
                with cols[1]:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("🔎 Load", key=f"dialog_load_{j['id']}"):
                        try:
                            full = dialog_api.get_results(j["id"])
                            st.session_state.dialog_result = {**j, **full}
                            st.rerun()
                        except Exception as e:
                            st.error(f"Could not load: {e}")
        return

    # ── Insights ──────────────────────────────────────────────────────────────
    st.markdown(
        "<hr style='border:none;border-top:1px solid rgba(255,255,255,0.06);margin:1.5rem 0'>",
        unsafe_allow_html=True,
    )
    q_count = result.get("query_count", len(result.get("sql_queries") or []))
    r_count = result.get("result_count", len(result.get("query_results") or []))
    cache_hit = result.get("cache_hit", False)
    banner_suffix = " &nbsp;⚡ <em>served from cache</em>" if cache_hit else ""
    st.markdown(
        f'<div class="banner-ok" style="margin-bottom:1rem">'
        f'✓ Analysis complete — {q_count} queries planned, {r_count} executed{banner_suffix}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="stat-row">'
        + _stat_card(q_count, "Queries Planned", "#63b3ed")
        + _stat_card(r_count, "Queries Executed", "#68d391")
        + _stat_card(
            len([x for x in (result.get("query_results") or []) if not x.get("error")]),
            "Succeeded", "#b794f4",
        )
        + _stat_card(
            len([x for x in (result.get("query_results") or []) if x.get("error")]),
            "Failed", "#fc814a",
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    # Insights narrative
    insights = result.get("insights", "")
    if insights:
        st.markdown('<div class="sec-head" style="margin-top:1.5rem">💡 Insights</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div style="background:rgba(183,148,244,0.08);border:1px solid rgba(183,148,244,0.2);'
            f'border-radius:12px;padding:1.2rem 1.5rem;color:#e2e8f0;font-size:0.9rem">'
            f'{insights}</div>',
            unsafe_allow_html=True,
        )

    # Query results breakdown
    query_results = result.get("query_results") or []
    if query_results:
        st.markdown('<div class="sec-head" style="margin-top:1.5rem">📊 Query Results</div>',
                    unsafe_allow_html=True)
        _render_query_results(query_results)

    # SQL queries (collapsed)
    sql_queries = result.get("sql_queries") or []
    if sql_queries:
        with st.expander(f"📋 Generated SQL queries ({len(sql_queries)})", expanded=False):
            for q in sql_queries:
                st.markdown(
                    f'<div style="color:#63b3ed;font-size:0.82rem;margin-bottom:0.2rem;'
                    f'font-weight:600">{q.get("query_id","")}: {q.get("description","")}</div>',
                    unsafe_allow_html=True,
                )
                st.code(q.get("sql", ""), language="sql")

    # Errors (collapsed)
    errors = result.get("errors") or []
    if errors:
        with st.expander(f"⚠ {len(errors)} warning(s)"):
            for err in errors:
                st.markdown(f"- {err}")


# ── Conformity view ────────────────────────────────────────────────────────────
def _conformity_view() -> None:
    import json as _json

    st.markdown(
        '<div class="hero">'
        '<div class="hero-title">🔗 KG Conformity</div>'
        '<div class="hero-sub">Detect conformed nodes across knowledge graphs and stitch them into a super-graph</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Step 1: Select KGs ─────────────────────────────────────────────────────
    st.markdown('<div class="sec-head">1️⃣ Select Knowledge Graphs to Compare</div>', unsafe_allow_html=True)

    done_jobs = [j for j in kg_api.list_jobs() if j.get("status") == "done"]
    if not done_jobs:
        st.info("No completed Knowledge Graph jobs found. Generate at least two KGs first.")
        return

    kg_options = {f"KG {j['id'][:8]}… ({j.get('node_count',0)} nodes, {j.get('edge_count',0)} edges)": j["id"]
                  for j in done_jobs}
    selected_labels = st.multiselect(
        "Choose 2 or more KGs:",
        options=list(kg_options.keys()),
        default=list(kg_options.keys())[:min(2, len(kg_options))],
        key="conformity_kg_select",
    )
    selected_job_ids = [kg_options[lbl] for lbl in selected_labels]

    # ── Analyse ────────────────────────────────────────────────────────────────
    col_left, col_right = st.columns([2, 1])
    with col_left:
        fuzzy_thresh   = st.slider("Fuzzy label threshold", 50, 100, 80, key="conf_fuzzy")
        jaccard_thresh = st.slider("Property Jaccard threshold", 0.0, 1.0, 0.30, step=0.05, key="conf_jaccard")
    with col_right:
        st.markdown("<br>", unsafe_allow_html=True)
        analyse_btn = st.button(
            "🔍 Analyse Conformities",
            disabled=len(selected_job_ids) < 2,
            use_container_width=True,
        )

    if analyse_btn and len(selected_job_ids) >= 2:
        # Build snapshots
        snapshots = []
        for jid in selected_job_ids:
            try:
                gd = kg_api.get_graph(jid)
                snapshots.append({"kg_id": jid[:8], "nodes": gd.get("nodes", []), "edges": gd.get("edges", [])})
            except Exception as e:
                st.error(f"Could not load graph for job {jid[:8]}: {e}")
                return
        try:
            job_id = conformity_api.analyse({
                "kg_snapshots":    snapshots,
                "fuzzy_threshold": fuzzy_thresh,
                "jaccard_threshold": jaccard_thresh,
            })
            st.session_state.conformity_job_id    = job_id
            st.session_state.conformity_results   = None
            st.session_state.conformity_stitch_id = None
            st.session_state.conformity_super_graph = None
            st.session_state.conformity_approved  = set()
            st.session_state.conformity_kg_ids    = [s["kg_id"] for s in snapshots]
            st.rerun()
        except Exception as e:
            st.error(f"Failed to start analysis: {e}")

    # ── Poll analyse job ───────────────────────────────────────────────────────
    job_id = st.session_state.conformity_job_id
    if job_id and not st.session_state.conformity_results:
        status = {}
        try:
            status = conformity_api.get_job(job_id)
        except Exception as e:
            st.error(f"Could not poll job: {e}")

        if status.get("status") == "running":
            with st.spinner("Analysing conformities…"):
                time.sleep(2)
                st.rerun()
        elif status.get("status") == "completed":
            try:
                results = conformity_api.get_results(job_id)
                st.session_state.conformity_results = results
                st.rerun()
            except Exception as e:
                st.error(f"Could not fetch results: {e}")
        elif status.get("status") == "error":
            st.error(f"Analysis failed: {status.get('errors', [])}")
            st.session_state.conformity_job_id = None

    # ── Show results ───────────────────────────────────────────────────────────
    results = st.session_state.conformity_results
    if not results:
        return

    conformities = results.get("conformities", [])
    exact_c   = results.get("exact_count", 0)
    fuzzy_c   = results.get("fuzzy_count", 0)
    jaccard_c = results.get("jaccard_count", 0)

    st.markdown(
        "<hr style='border:none;border-top:1px solid rgba(255,255,255,0.06);margin:1.5rem 0'>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sec-head">2️⃣ Conformity Analysis Results</div>', unsafe_allow_html=True)

    # Stats row
    stat_cards = [
        _stat_card(len(conformities), "Total Candidates", "#63b3ed"),
        _stat_card(exact_c,   "Exact Matches",    "#68d391"),
        _stat_card(fuzzy_c,   "Fuzzy Matches",    "#f6ad55"),
        _stat_card(jaccard_c, "Property Matches", "#b794f4"),
    ]
    st.markdown('<div class="stat-row">' + "".join(stat_cards) + "</div>", unsafe_allow_html=True)

    # Recommendations
    recs = results.get("recommendations", "")
    if recs:
        with st.expander("📋 AI Recommendations", expanded=True):
            st.markdown(recs)

    if not conformities:
        st.info("No conformity candidates found — graphs appear to have entirely distinct schemas.")
        return

    # ── Step 2: Approve conformities ──────────────────────────────────────────
    st.markdown(
        "<hr style='border:none;border-top:1px solid rgba(255,255,255,0.06);margin:1.5rem 0'>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sec-head">3️⃣ Approve Conformities to Stitch</div>', unsafe_allow_html=True)
    st.caption("Check the conformities you want to merge, then click **Approve & Stitch**.")

    approved: set = st.session_state.conformity_approved

    # Select/deselect all
    sel_col1, sel_col2 = st.columns(2)
    with sel_col1:
        if st.button("✅ Select All", use_container_width=True):
            st.session_state.conformity_approved = {c["index"] for c in conformities}
            st.rerun()
    with sel_col2:
        if st.button("⬜ Deselect All", use_container_width=True):
            st.session_state.conformity_approved = set()
            st.rerun()

    type_colors = {"exact": "#68d391", "fuzzy": "#f6ad55", "property_jaccard": "#b794f4"}

    for c in conformities:
        idx   = c["index"]
        mtype = c.get("match_type", "")
        color = type_colors.get(mtype, "#94a3b8")
        checked = idx in approved
        new_checked = st.checkbox(
            f"#{idx} | **{c['node_a_label']}** ({c['kg_a_id']}) ↔ **{c['node_b_label']}** ({c['kg_b_id']}) "
            f"| {mtype} | score={c['score']:.2f} | jaccard={c['jaccard']:.2f}",
            value=checked,
            key=f"conf_check_{idx}",
        )
        if new_checked != checked:
            new_set = set(approved)
            if new_checked:
                new_set.add(idx)
            else:
                new_set.discard(idx)
            st.session_state.conformity_approved = new_set
            approved = new_set

    # ── Stitch button ──────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    stitch_btn = st.button(
        f"🔗 Approve & Stitch ({len(approved)} selected)",
        disabled=(len(approved) == 0),
        use_container_width=True,
        type="primary",
    )

    if stitch_btn and approved:
        try:
            stitch_id = conformity_api.stitch({
                "job_id":           job_id,
                "approved_indices": list(approved),
            })
            st.session_state.conformity_stitch_id  = stitch_id
            st.session_state.conformity_super_graph = None
            st.rerun()
        except Exception as e:
            st.error(f"Failed to start stitch: {e}")

    # ── Poll stitch job ────────────────────────────────────────────────────────
    stitch_id = st.session_state.conformity_stitch_id
    if stitch_id and not st.session_state.conformity_super_graph:
        try:
            st_status = conformity_api.get_stitch(stitch_id)
        except Exception as e:
            st.error(f"Could not poll stitch job: {e}")
            st_status = {}

        if st_status.get("status") == "running":
            with st.spinner("Stitching super-graph…"):
                time.sleep(2)
                st.rerun()
        elif st_status.get("status") == "completed":
            try:
                sg = conformity_api.get_stitch_graph(stitch_id)
                st.session_state.conformity_super_graph = sg
                st.rerun()
            except Exception as e:
                st.error(f"Could not fetch super-graph: {e}")
        elif st_status.get("status") == "error":
            st.error(f"Stitch failed: {st_status.get('errors', [])}")

    # ── Show super-graph ───────────────────────────────────────────────────────
    sg = st.session_state.conformity_super_graph
    if not sg:
        return

    st.markdown(
        "<hr style='border:none;border-top:1px solid rgba(255,255,255,0.06);margin:1.5rem 0'>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sec-head">4️⃣ Super Knowledge Graph</div>', unsafe_allow_html=True)

    sg_nodes = sg.get("nodes", [])
    sg_edges = sg.get("edges", [])
    stat_cards2 = [
        _stat_card(len(sg_nodes), "Nodes",  "#63b3ed"),
        _stat_card(len(sg_edges), "Edges",  "#68d391"),
        _stat_card(len(approved), "Merged", "#f6ad55"),
    ]
    st.markdown('<div class="stat-row">' + "".join(stat_cards2) + "</div>", unsafe_allow_html=True)

    # Stitch log
    stitch_log = []
    try:
        stitch_log = conformity_api.get_stitch(stitch_id).get("stitch_log", [])
    except Exception:
        pass
    if stitch_log:
        with st.expander("🪢 Stitch log", expanded=False):
            for line in stitch_log:
                st.markdown(f"- {line}")

    # Visualise super-graph using pyvis
    try:
        from pyvis.network import Network as _Network  # noqa: PLC0415
        import tempfile, os as _os  # noqa: PLC0415

        net = _Network(height="600px", width="100%", bgcolor="#0d1b2a", font_color="#e2e8f0")
        net.set_options("""{
          "nodes": {"borderWidth": 2, "shadow": true},
          "edges": {"smooth": {"type": "dynamic"}, "shadow": true},
          "physics": {"stabilization": {"iterations": 120}}
        }""")
        for node in sg_nodes:
            color = node.get("color", "#63b3ed")
            if isinstance(color, dict):
                color = color.get("color", "#63b3ed")
            net.add_node(
                node["id"],
                label=node.get("label", node["id"]),
                title=node.get("title", ""),
                color=color,
                size=node.get("size", 20),
            )
        for edge in sg_edges:
            ec = edge.get("color", {})
            if isinstance(ec, dict):
                ec = ec.get("color", "#68d391")
            net.add_edge(
                edge.get("from", ""),
                edge.get("to", ""),
                label=edge.get("label", ""),
                title=edge.get("title", ""),
                color=ec,
            )
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            net.save_graph(f.name)
            html_content = open(f.name).read()
        st.components.v1.html(html_content, height=620, scrolling=False)
    except ImportError:
        st.warning("pyvis not installed — cannot render graph. Install with: pip install pyvis")
    except Exception as e:
        st.error(f"Graph render error: {e}")

    # ── Download super-graph JSON ──────────────────────────────────────────────
    st.download_button(
        "⬇ Download super-graph JSON",
        data=_json.dumps(sg, indent=2, default=str),
        file_name="super_graph.json",
        mime="application/json",
        use_container_width=True,
    )

    # ── Save super-graph ───────────────────────────────────────────────────────
    st.markdown(
        "<hr style='border:none;border-top:1px solid rgba(255,255,255,0.06);margin:1.5rem 0'>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sec-head">5️⃣ Save Super Knowledge Graph</div>', unsafe_allow_html=True)
    sg_name = st.text_input("Name for this super-graph:", value="super_kg_1", key="sg_save_name")
    if st.button("💾 Save Super-Graph", use_container_width=True, disabled=not sg_name.strip()):
        try:
            saved = conformity_api.save_super_graph({
                "stitch_id": stitch_id,
                "name":      sg_name.strip(),
            })
            st.success(f"Saved '{saved['name']}' ({saved['node_count']} nodes, {saved['edge_count']} edges)")
        except Exception as e:
            st.error(f"Save failed: {e}")

    # ── List saved super-graphs ────────────────────────────────────────────────
    saved_sgs = conformity_api.list_super_graphs()
    if saved_sgs:
        with st.expander("📚 Saved Super-Graphs", expanded=False):
            for sg_rec in saved_sgs:
                c1, c2 = st.columns([6, 2])
                with c1:
                    st.markdown(
                        f'<div class="hcard-title">{sg_rec["name"]}</div>'
                        f'<div class="hcard-meta">{sg_rec["node_count"]} nodes · {sg_rec["edge_count"]} edges</div>',
                        unsafe_allow_html=True,
                    )
                with c2:
                    if st.button("⬇ Download", key=f"dl_sg_{sg_rec['name']}"):
                        try:
                            dl_sg = conformity_api.get_super_graph(sg_rec["name"])
                            st.download_button(
                                "Download JSON",
                                data=_json.dumps(dl_sg, indent=2, default=str),
                                file_name=f"{sg_rec['name']}.json",
                                mime="application/json",
                            )
                        except Exception as e:
                            st.error(f"Download failed: {e}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    _inject_css()
    _sidebar()
    page = st.session_state.page
    if page == "extract":
        _extract_view()
    elif page == "history":
        _history_view()
    elif page == "search":
        _search_view()
    elif page == "ontology":
        _ontology_view()
    elif page == "kg":
        _kg_view()
    elif page == "dialog":
        _dialog_view()
    elif page == "conformity":
        _conformity_view()


if __name__ == "__main__":
    main()
