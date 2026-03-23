"""
embed_node — Persist vector embeddings on Neo4j KG nodes for production GraphRAG.

Runs after execute_node in the KG pipeline (generate / update modes only).
For every node in graph_data:
  1. Builds a rich text representation: label + column names/types + comments.
  2. Embeds all texts in one batch using the configured backend.
  3. Writes the embedding vector + full title back to the existing Neo4j node
     matched by uri  (MATCH (n:KGNode {uri: $uri}) SET n.embedding = $vec, n.title = $title).
  4. Serialises join_columns onto edges so the production retrieval query can
     reconstruct exact JOIN ON conditions without reading graph_data.
  5. Creates an HNSW vector index (IF NOT EXISTS) on KGNode.embedding.

At query time, retrieve_node uses:
  CALL db.index.vector.queryNodes($index, $k, $query_vec) YIELD node, score

Skip conditions (node is a no-op):
  - embed_enabled = False  (default)
  - neo4j_uri is empty (no Neo4j connection)
  - graph_data has no nodes
  - Backend is tfidf or keyword (variable-dimension — incompatible with HNSW)

Supported backends:
  sentence-transformers  — all-MiniLM-L6-v2, 384 dims (default)
  openai                 — text-embedding-3-small, 1536 dims
"""
from __future__ import annotations

import json
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
        f"Backend '{backend}' is not supported for Neo4j production mode. "
        "Use 'sentence-transformers' or 'openai'."
    )


# ── Neo4j helpers ─────────────────────────────────────────────────────────────

def _ensure_vector_index(session: Any, index_name: str, dimensions: int) -> None:
    """Create HNSW cosine vector index on KGNode.embedding (IF NOT EXISTS)."""
    # Neo4j 5.11+ syntax
    try:
        session.run(
            f"CREATE VECTOR INDEX `{index_name}` IF NOT EXISTS "
            f"FOR (n:KGNode) ON (n.embedding) "
            f"OPTIONS {{indexConfig: {{"
            f"  `vector.dimensions`: {dimensions},"
            f"  `vector.similarity_function`: 'cosine'"
            f"}}}}"
        )
        logger.info("embed_node: vector index '%s' ready (%d dims)", index_name, dimensions)
        return
    except Exception as exc:
        logger.debug("embed_node: new index syntax failed (%s) — trying legacy API", exc)

    # Legacy Neo4j (< 5.11) fallback
    try:
        session.run(
            "CALL db.index.vector.createNodeIndex($name, 'KGNode', 'embedding', $dims, 'cosine')",
            name=index_name, dims=dimensions,
        )
        logger.info("embed_node: vector index '%s' created via legacy API", index_name)
    except Exception as exc2:
        # Index may already exist — not fatal
        logger.warning("embed_node: could not create vector index: %s", exc2)


def _write_embeddings(
    session: Any,
    nodes: List[Dict],
    embeddings: np.ndarray,
) -> int:
    """Batch-write embedding vectors + titles onto Neo4j KGNode nodes."""
    written = 0
    for node, emb in zip(nodes, embeddings):
        uri   = node.get("id", "")
        title = node.get("title", "")
        if not uri:
            continue
        result = session.run(
            "MATCH (n:KGNode {uri: $uri}) "
            "SET n.embedding = $embedding, n.title = $title "
            "RETURN count(n) AS updated",
            uri=uri, embedding=emb.tolist(), title=title,
        )
        record = result.single()
        if record and record["updated"]:
            written += 1
        else:
            logger.warning("embed_node: no KGNode found for uri=%s — was execute_node run?", uri)
    return written


def _write_edge_metadata(session: Any, edges: List[Dict]) -> int:
    """
    Write join_columns + title onto Neo4j relationships.
    join_columns is serialised as a JSON string so it survives the graph store.
    """
    written = 0
    for edge in edges:
        src   = edge.get("from", "")
        tgt   = edge.get("to", "")
        jc    = edge.get("join_columns", [])
        label = edge.get("label", "")
        title = edge.get("title", "")
        if not src or not tgt:
            continue
        result = session.run(
            "MATCH (a:KGNode {uri: $src})-[r]->(b:KGNode {uri: $tgt}) "
            "SET r.join_columns = $jc, r.title = $title "
            "RETURN count(r) AS updated",
            src=src, tgt=tgt,
            jc=json.dumps(jc) if jc else "[]",
            title=title,
        )
        record = result.single()
        if record and record["updated"]:
            written += 1
    return written


# ── Node ──────────────────────────────────────────────────────────────────────

def embed_node(state: KGState) -> KGState:
    """
    Embed KG node titles and persist to Neo4j for production GraphRAG retrieval.
    No-op when embed_enabled=False or neo4j_uri is empty.
    """
    logger.info("=== embed_node ===")

    config = state["config"]

    # ── Guard clauses ─────────────────────────────────────────────────────────
    if not getattr(config, "embed_enabled", False):
        logger.info("embed_node: embed_enabled=False — skipping")
        return state

    neo4j_uri = getattr(config, "neo4j_uri", "")
    if not neo4j_uri:
        logger.info("embed_node: no neo4j_uri configured — skipping")
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
            "and is not compatible with Neo4j HNSW vector index. "
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

    # ── Write to Neo4j ────────────────────────────────────────────────────────
    neo4j_user = getattr(config, "neo4j_username", "neo4j")
    neo4j_pass = getattr(config, "neo4j_password", "")
    neo4j_db   = getattr(config, "neo4j_database", "neo4j")
    index_name = getattr(config, "embed_index_name", "kg-node-embeddings")

    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))

        with driver.session(database=neo4j_db) as session:
            _ensure_vector_index(session, index_name, dimensions)
            node_written = _write_embeddings(session, nodes, embeddings)
            edge_written = _write_edge_metadata(session, edges)

        driver.close()

        logger.info(
            "embed_node: wrote embeddings to %d/%d nodes, metadata to %d/%d edges",
            node_written, len(nodes), edge_written, len(edges),
        )
        state["embeddings_stored"] = True

    except Exception as exc:
        msg = f"embed_node: Neo4j write failed — {exc}"
        logger.exception(msg)
        state["errors"].append(msg)

    state["phase"] = "embedded"
    return state
