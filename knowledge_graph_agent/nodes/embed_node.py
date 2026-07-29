"""
embed_node — Attach vector embeddings to KG nodes for production GraphRAG.

Runs after execute_node in the KG pipeline (generate / update modes only).
For every node in graph_data:
  1. Builds a rich text representation: label + column names/types + comments.
  2. Embeds all texts in one batch using the configured backend.
  3. Stores the embedding vector directly on the node dict (node["embedding"]).
  4. Serialises join_columns onto edges so retrieval can reconstruct exact
     JOIN ON conditions without extra lookups.
  5. Re-persists the updated snapshot (now including embeddings) to the KG
     snapshot store (kg_store.py) so retrieve_node can read it back without
     recomputing embeddings.

At query time, retrieve_node embeds the NLQ with the same backend and ranks
nodes by numpy cosine similarity against node["embedding"].

Skip conditions (node is a no-op):
  - embed_enabled = False  (default)
  - graph_data has no nodes
  - Backend is tfidf or keyword (variable-dimension — incompatible with a
    fixed-dimension embedding column)

Supported backends:
  sentence-transformers  — all-MiniLM-L6-v2, 384 dims (default)
  openai                 — text-embedding-3-small, 1536 dims
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

import numpy as np

from ..state import KGState

logger = logging.getLogger(__name__)


# ── Text representation ───────────────────────────────────────────────────────

def _node_text(node: Dict) -> str:
    """Rich text representation of a KG node used as the embedding input."""
    label = node.get("label", "")
    title = node.get("title", "")
    # Strip the redundant "Class: <name>" header line
    body = re.sub(r'^Class:\s*\S+\s*', '', title, flags=re.IGNORECASE).strip()
    return f"{label} {body}".strip()


# ── Embedding ─────────────────────────────────────────────────────────────────

def _embed_sentence_transformers(texts: List[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(texts, normalize_embeddings=True).astype(np.float32)


def _embed_openai(texts: List[str]) -> np.ndarray:
    import os
    from openai import OpenAI
    resp = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "")).embeddings.create(
        model="text-embedding-3-small", input=texts
    )
    vecs  = np.array([d.embedding for d in resp.data], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-9)


def _detect_backend(preferred: str) -> str:
    if preferred != "auto":
        return preferred
    for candidate, pkg in [
        ("sentence-transformers", "sentence_transformers"),
        ("openai", "openai"),
    ]:
        try:
            __import__(pkg)
            return candidate
        except ImportError:
            continue
    return "unsupported"


def _embed_corpus(texts: List[str], backend: str) -> np.ndarray:
    if backend == "sentence-transformers":
        return _embed_sentence_transformers(texts)
    if backend == "openai":
        return _embed_openai(texts)
    raise ValueError(
        f"Backend '{backend}' is not supported for embedding. "
        "Use 'sentence-transformers' or 'openai'."
    )


# ── Node ──────────────────────────────────────────────────────────────────────

def embed_node(state: KGState) -> KGState:
    """
    Attach embedding vectors to KG nodes and re-persist the snapshot for
    production GraphRAG retrieval. No-op when embed_enabled=False.
    """
    logger.info("=== embed_node ===")

    config = state["config"]

    # ── Guard clauses ─────────────────────────────────────────────────────────
    if not getattr(config, "embed_enabled", False):
        logger.info("embed_node: embed_enabled=False — skipping")
        return state

    graph_data = state.get("graph_data") or {"nodes": [], "edges": []}
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    if not nodes:
        logger.info("embed_node: no nodes in graph_data — skipping")
        return state

    # ── Detect and validate backend ───────────────────────────────────────────
    pref    = getattr(config, "embed_backend", "auto")
    backend = _detect_backend(pref)

    if backend == "unsupported":
        msg = (
            "embed_node: no supported embedding backend found. "
            "Install sentence-transformers or openai."
        )
        logger.error(msg)
        state["errors"].append(msg)
        return state

    if backend in ("tfidf", "keyword"):
        msg = (
            f"embed_node: backend '{backend}' produces variable-dimension vectors "
            "and is not compatible with the fixed-dimension embedding column. "
            "Use 'sentence-transformers' or 'openai'."
        )
        logger.error(msg)
        state["errors"].append(msg)
        return state

    # ── Embed ─────────────────────────────────────────────────────────────────
    logger.info("embed_node: embedding %d nodes (backend=%s)", len(nodes), backend)
    try:
        texts      = [_node_text(n) for n in nodes]
        embeddings = _embed_corpus(texts, backend)   # [N, D]
    except Exception as exc:
        msg = f"embed_node: embedding failed — {exc}"
        logger.exception(msg)
        state["errors"].append(msg)
        return state

    dimensions = embeddings.shape[1]
    logger.info("embed_node: %d vectors, %d dims", len(embeddings), dimensions)

    # ── Attach embeddings to nodes, serialise edge metadata ───────────────────
    kg_id = getattr(config, "kg_id", "").strip() or "default"
    for node, emb in zip(nodes, embeddings):
        node["embedding"] = emb.tolist()
    for edge in edges:
        edge["join_columns"] = edge.get("join_columns", [])

    try:
        import kg_store
        kg_store.save_snapshot(kg_id, nodes, edges)

        logger.info(
            "embed_node: attached embeddings to %d nodes, persisted snapshot kg_id=%s",
            len(nodes), kg_id,
        )
        state["embeddings_stored"] = True
        state["graph_data"] = {"nodes": nodes, "edges": edges}

    except Exception as exc:
        msg = f"embed_node: snapshot persistence failed — {exc}"
        logger.exception(msg)
        state["errors"].append(msg)

    state["phase"] = "embedded"
    return state
