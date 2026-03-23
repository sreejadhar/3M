"""
retrieve_node — GraphRAG retrieval for schema context.

Two execution paths, selected automatically:

  PRODUCTION (Neo4j):
    When graphrag_neo4j_uri is set on DialogConfig, uses the HNSW vector index
    written by embed_node during the KG build phase.  Queries Neo4j for the
    top-K most similar nodes, then BFS-expands via FK edges inside Neo4j.
    No per-session embedding of the corpus (embeddings live in Neo4j).
    Safe for multi-worker / multi-tenant deployments.

  IN-MEMORY (development / small schemas):
    When graphrag_neo4j_uri is empty or the Neo4j path fails, falls back to
    the in-memory path: embeds all KG node titles once (module-level cache
    keyed by schema hash), computes cosine similarity with numpy, and expands
    via an in-memory adjacency dict.

Both paths produce the same output: state kg_nodes / kg_edges are replaced
with the retrieved subgraph so understand_node sees only relevant tables.

Skip conditions (node is a no-op):
  - graphrag_enabled = False
  - no kg_nodes in state
  - schema size ≤ graphrag_min_tables

Embedding backends for the NLQ vector (and for in-memory corpus):
  sentence-transformers  — best quality, local, no API key
  openai                 — good quality, requires OPENAI_API_KEY
  tfidf                  — sklearn, no API key (in-memory only)
  keyword                — pure-Python fallback (in-memory only)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..state import DialogState

logger = logging.getLogger(__name__)

# ── Module-level embedding cache ──────────────────────────────────────────────
# Keyed by (schema_hash, backend) so cache is invalidated when schema changes.
_EMBED_CACHE: Dict[str, "_Cache"] = {}


class _Cache:
    """Holds pre-computed node embeddings for one schema snapshot."""

    __slots__ = ("key", "backend", "node_ids", "texts", "matrix", "_vect")

    def __init__(
        self,
        key: str,
        backend: str,
        node_ids: List[str],
        texts: List[str],
        matrix: np.ndarray,          # shape [N, D], L2-normalised rows
        vect: Any = None,            # fitted TfidfVectorizer for "tfidf" backend
    ) -> None:
        self.key      = key
        self.backend  = backend
        self.node_ids = node_ids
        self.texts    = texts
        self.matrix   = matrix
        self._vect    = vect

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string into the same space as the corpus."""
        if self.backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            v = model.encode([query], normalize_embeddings=True)[0]
            return v.astype(np.float32)

        if self.backend == "openai":
            import os
            from openai import OpenAI
            resp = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "")).embeddings.create(
                model="text-embedding-3-small", input=[query]
            )
            v = np.array(resp.data[0].embedding, dtype=np.float32)
            return _l2_norm(v)

        if self.backend == "tfidf" and self._vect is not None:
            v = self._vect.transform([query]).toarray()[0].astype(np.float32)
            return _l2_norm(v)

        # keyword fallback — rebuild vocab from cached corpus texts
        return _keyword_embed([query], self.texts)[0]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _l2_norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / max(n, 1e-9)


def _schema_hash(nodes: List[Dict]) -> str:
    """Stable hash over node IDs + titles — used for cache invalidation."""
    fingerprint = json.dumps(
        sorted((n.get("id", ""), n.get("title", "")) for n in nodes)
    )
    return hashlib.md5(fingerprint.encode()).hexdigest()


def _node_text(node: Dict) -> str:
    """
    Build a rich text representation of a KG node for embedding.
    Uses label (table name) + full title (columns, comments, sample values).
    """
    label = node.get("label", "")
    title = node.get("title", "")
    # Strip "Class: X" prefix — redundant with label
    title_body = re.sub(r'^Class:\s*\S+\s*', '', title, flags=re.IGNORECASE).strip()
    return f"{label} {title_body}".strip()


# ── Embedding backends ────────────────────────────────────────────────────────

def _keyword_embed(texts: List[str], vocab_texts: Optional[List[str]] = None) -> np.ndarray:
    """
    Bag-of-words binary vectors.  Vocabulary is built from vocab_texts (corpus);
    if None, vocabulary is built from texts itself.
    """
    tokenize = lambda t: re.findall(r'\w+', t.lower())
    source   = vocab_texts if vocab_texts is not None else texts
    vocab    = sorted({w for t in source for w in tokenize(t)})
    idx      = {w: i for i, w in enumerate(vocab)}
    mat = np.zeros((len(texts), max(len(vocab), 1)), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in tokenize(t):
            if w in idx:
                mat[i, idx[w]] = 1.0
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, 1e-9)


def _detect_backend(preferred: str) -> str:
    if preferred != "auto":
        return preferred
    for candidate, pkg in [
        ("sentence-transformers", "sentence_transformers"),
        ("tfidf", "sklearn"),
    ]:
        try:
            __import__(pkg)
            return candidate
        except ImportError:
            continue
    return "keyword"


def _build_cache(
    nodes: List[Dict],
    backend: str,
    schema_hash: str,
) -> _Cache:
    """Embed all node texts and return a populated _Cache."""
    node_ids = [n.get("id", "") for n in nodes]
    texts    = [_node_text(n) for n in nodes]

    if backend == "sentence-transformers":
        from sentence_transformers import SentenceTransformer
        model  = SentenceTransformer("all-MiniLM-L6-v2")
        matrix = model.encode(texts, normalize_embeddings=True).astype(np.float32)
        return _Cache(schema_hash, backend, node_ids, texts, matrix)

    if backend == "openai":
        import os
        from openai import OpenAI
        resp   = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "")).embeddings.create(
            model="text-embedding-3-small", input=texts
        )
        vecs   = np.array([d.embedding for d in resp.data], dtype=np.float32)
        norms  = np.linalg.norm(vecs, axis=1, keepdims=True)
        matrix = vecs / np.maximum(norms, 1e-9)
        return _Cache(schema_hash, backend, node_ids, texts, matrix)

    if backend == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        vect   = TfidfVectorizer(max_features=3000, sublinear_tf=True)
        raw    = vect.fit_transform(texts).toarray().astype(np.float32)
        norms  = np.linalg.norm(raw, axis=1, keepdims=True)
        matrix = raw / np.maximum(norms, 1e-9)
        return _Cache(schema_hash, backend, node_ids, texts, matrix, vect=vect)

    # keyword fallback
    matrix = _keyword_embed(texts)
    return _Cache(schema_hash, backend, node_ids, texts, matrix)


# ── Graph expansion ───────────────────────────────────────────────────────────

def _build_adjacency(edges: List[Dict]) -> Dict[str, List[str]]:
    """Bidirectional adjacency: node_id → [neighbour_ids]."""
    adj: Dict[str, List[str]] = {}
    for e in edges:
        src, tgt = e.get("from", ""), e.get("to", "")
        if src and tgt:
            adj.setdefault(src, []).append(tgt)
            adj.setdefault(tgt, []).append(src)
    return adj


def _bfs_expand(
    seeds: List[str],
    adjacency: Dict[str, List[str]],
    hop_depth: int,
) -> List[str]:
    """Return seeds + all nodes reachable within hop_depth hops."""
    visited: set = set(seeds)
    frontier: deque = deque((nid, 0) for nid in seeds)
    while frontier:
        nid, depth = frontier.popleft()
        if depth >= hop_depth:
            continue
        for neighbour in adjacency.get(nid, []):
            if neighbour not in visited:
                visited.add(neighbour)
                frontier.append((neighbour, depth + 1))
    return list(visited)


# ── Neo4j production retrieval ────────────────────────────────────────────────

def _embed_query_vec(query: str, backend: str) -> "Optional[np.ndarray]":
    """
    Embed a single query string into a normalised float32 vector.
    Returns None on failure (caller should fall back to in-memory path).
    Only fixed-dimension backends are attempted here (sentence-transformers,
    openai); tfidf and keyword are handled inside _Cache.embed_query.
    """
    try:
        if backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            v = model.encode([query], normalize_embeddings=True)[0]
            return v.astype(np.float32)
        if backend == "openai":
            import os
            from openai import OpenAI
            resp = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "")).embeddings.create(
                model="text-embedding-3-small", input=[query]
            )
            v = np.array(resp.data[0].embedding, dtype=np.float32)
            return _l2_norm(v)
    except Exception as exc:
        logger.warning("retrieve_node: query embedding failed (%s)", exc)
    return None


def _neo4j_retrieve(
    nodes: List[Dict],
    edges: List[Dict],
    query: str,
    config: Any,
) -> "Tuple[Optional[List[Dict]], Optional[List[Dict]]]":
    """
    Use the Neo4j HNSW vector index + graph traversal to retrieve the relevant
    subgraph.  Returns (sub_nodes, sub_edges) filtered from the in-memory
    nodes/edges, or (None, None) to signal the caller to fall back to in-memory.
    """
    neo4j_uri      = getattr(config, "graphrag_neo4j_uri",      "")
    neo4j_user     = getattr(config, "graphrag_neo4j_username",  "neo4j")
    neo4j_pass     = getattr(config, "graphrag_neo4j_password",  "")
    neo4j_db       = getattr(config, "graphrag_neo4j_database",  "neo4j")
    # Resolve KG identity — must match KGConfig.kg_id used at build time
    kg_id          = getattr(config, "graphrag_kg_id", "").strip() or "default"
    # Index name: explicit override → else derive from kg_id (matches embed_node logic)
    index_name     = getattr(config, "graphrag_neo4j_index", "").strip() or f"kg-{kg_id}-embeddings"
    top_k          = getattr(config, "graphrag_top_k",           8)
    hop_depth      = getattr(config, "graphrag_hop_depth",       2)
    pref_backend   = getattr(config, "graphrag_embedding_backend", "auto")

    if not neo4j_uri:
        return None, None

    # For Neo4j path we need fixed-dimension embeddings
    backend = _detect_backend(pref_backend)
    if backend in ("tfidf", "keyword"):
        logger.info(
            "retrieve_node: Neo4j path requires fixed-dimension embeddings; "
            "backend=%s — falling back to in-memory", backend
        )
        return None, None

    # Embed the query
    q_vec = _embed_query_vec(query, backend)
    if q_vec is None:
        return None, None

    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))

        # ── Step 1: vector similarity search → seed URIs ──────────────────────
        with driver.session(database=neo4j_db) as session:
            result = session.run(
                "CALL db.index.vector.queryNodes($idx, $k, $vec) "
                "YIELD node RETURN node.uri AS uri",
                idx=index_name, k=top_k, vec=q_vec.tolist(),
            )
            seed_uris: set = {r["uri"] for r in result if r["uri"]}

        if not seed_uris:
            logger.warning(
                "retrieve_node: Neo4j vector search returned 0 results "
                "(index may not be built yet) — falling back to in-memory"
            )
            driver.close()
            return None, None

        logger.info("retrieve_node (Neo4j): %d seed nodes from vector search", len(seed_uris))

        # ── Step 2: BFS-expand via FK edges (within same KG only) ────────────
        subgraph_uris: set = set(seed_uris)
        frontier: set = set(seed_uris)
        for hop in range(hop_depth):
            if not frontier:
                break
            with driver.session(database=neo4j_db) as session:
                result = session.run(
                    "MATCH (n:KGNode {kg_id: $kg_id})-[]-(m:KGNode {kg_id: $kg_id}) "
                    "WHERE n.uri IN $uris "
                    "RETURN DISTINCT m.uri AS uri",
                    uris=list(frontier), kg_id=kg_id,
                )
                new_uris = {r["uri"] for r in result if r["uri"]} - subgraph_uris
            subgraph_uris.update(new_uris)
            frontier = new_uris

        driver.close()

        logger.info(
            "retrieve_node (Neo4j): subgraph has %d nodes after %d-hop expansion",
            len(subgraph_uris), hop_depth,
        )

        # ── Step 3: filter in-memory graph_data to subgraph ───────────────────
        sub_nodes = [n for n in nodes if n.get("id", "") in subgraph_uris]
        sub_edges = [
            e for e in edges
            if e.get("from", "") in subgraph_uris and e.get("to", "") in subgraph_uris
        ]
        return sub_nodes, sub_edges

    except Exception as exc:
        logger.warning(
            "retrieve_node: Neo4j retrieval failed (%s) — falling back to in-memory", exc
        )
        return None, None


# ── Main node ─────────────────────────────────────────────────────────────────

def retrieve_node(state: DialogState) -> DialogState:
    """
    GraphRAG retrieval: filter kg_nodes / kg_edges to the subgraph most
    relevant to the current NLQ before understand_node builds schema context.
    """
    logger.info("=== retrieve_node ===")

    config        = state["config"]
    nodes: List   = state.get("kg_nodes") or []
    edges: List   = state.get("kg_edges") or []
    query: str    = state.get("natural_query", "").strip()

    top_k       = getattr(config, "graphrag_top_k",       8)
    hop_depth   = getattr(config, "graphrag_hop_depth",   2)
    min_tables  = getattr(config, "graphrag_min_tables",  top_k)
    enabled     = getattr(config, "graphrag_enabled",     True)
    pref_backend = getattr(config, "graphrag_embedding_backend", "auto")

    # ── Early-exit conditions ─────────────────────────────────────────────────
    if not enabled:
        logger.info("retrieve_node: graphrag_enabled=False — skipping")
        return state

    if not nodes or not query:
        logger.info("retrieve_node: no nodes or no query — skipping")
        return state

    if len(nodes) <= min_tables:
        logger.info(
            "retrieve_node: schema has %d tables (≤ min_tables=%d) — skipping retrieval, using full schema",
            len(nodes), min_tables,
        )
        return state

    # ── Try Neo4j production path first ──────────────────────────────────────
    if getattr(config, "graphrag_neo4j_uri", ""):
        sub_nodes, sub_edges = _neo4j_retrieve(nodes, edges, query, config)
        if sub_nodes is not None:
            logger.info(
                "retrieve_node: Neo4j path — %d/%d tables, %d/%d edges",
                len(sub_nodes), len(nodes), len(sub_edges), len(edges),
            )
            state["kg_nodes"] = sub_nodes
            state["kg_edges"] = sub_edges
            return state
        logger.info("retrieve_node: Neo4j path unavailable — using in-memory fallback")

    # ── Get or build in-memory embedding cache ────────────────────────────────
    s_hash  = _schema_hash(nodes)
    backend = _detect_backend(pref_backend)
    key     = f"{s_hash}:{backend}"

    if key not in _EMBED_CACHE:
        logger.info(
            "retrieve_node: building embedding cache for %d nodes (backend=%s)",
            len(nodes), backend,
        )
        try:
            _EMBED_CACHE[key] = _build_cache(nodes, backend, s_hash)
            logger.info("retrieve_node: cache built (%s)", backend)
        except Exception as exc:
            logger.warning(
                "retrieve_node: embedding build failed (%s) — falling back to keyword", exc
            )
            try:
                _EMBED_CACHE[key] = _build_cache(nodes, "keyword", s_hash)
            except Exception as exc2:
                logger.error("retrieve_node: keyword fallback also failed (%s) — using full schema", exc2)
                return state
    else:
        logger.info("retrieve_node: using cached embeddings (backend=%s)", backend)

    cache = _EMBED_CACHE[key]

    # ── Embed query and rank nodes ─────────────────────────────────────────────
    try:
        q_vec  = cache.embed_query(query)                        # [D]
        scores = cache.matrix @ q_vec                            # [N] cosine similarities
    except Exception as exc:
        logger.warning("retrieve_node: query embedding failed (%s) — using full schema", exc)
        return state

    # top-K seed node IDs
    k = min(top_k, len(nodes))
    top_indices = scores.argsort()[::-1][:k]
    seed_ids    = [cache.node_ids[i] for i in top_indices]

    logger.info(
        "retrieve_node: top-%d seeds: %s  (scores: %s)",
        k,
        [nodes[i].get("label", "") for i in top_indices],
        [f"{scores[i]:.3f}" for i in top_indices],
    )

    # ── BFS-expand via edges ──────────────────────────────────────────────────
    adjacency     = _build_adjacency(edges)
    subgraph_ids  = set(_bfs_expand(seed_ids, adjacency, hop_depth))

    # ── Filter nodes and edges to subgraph ───────────────────────────────────
    sub_nodes = [n for n in nodes if n.get("id", "") in subgraph_ids]
    sub_edges = [
        e for e in edges
        if e.get("from", "") in subgraph_ids and e.get("to", "") in subgraph_ids
    ]

    logger.info(
        "retrieve_node: retrieved %d/%d tables, %d/%d edges",
        len(sub_nodes), len(nodes), len(sub_edges), len(edges),
    )

    state["kg_nodes"] = sub_nodes
    state["kg_edges"] = sub_edges
    return state
