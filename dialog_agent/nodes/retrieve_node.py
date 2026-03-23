"""
retrieve_node — In-memory GraphRAG retrieval for schema context.

For each incoming NLQ:
  1. Embed all KG node titles once (cached in a module-level dict keyed by
     schema hash, so re-embedding is skipped on subsequent turns).
  2. Embed the NLQ.
  3. Cosine-similarity search → top-K seed nodes (tables most relevant to
     the question).
  4. BFS-expand the seed set by `hop_depth` hops via KG edges (pulls in JOIN
     partners even if they were not directly retrieved).
  5. Replace kg_nodes / kg_edges in state with the retrieved subgraph so
     understand_node only builds schema context for those tables.

When the schema is small (≤ graphrag_top_k tables) or graphrag_enabled=False,
the node is a no-op and all nodes/edges pass through unchanged.

Embedding backends (auto-detected, in order of preference):
  sentence-transformers  — best quality, local, no API key required
  openai                 — good quality, requires OPENAI_API_KEY
  tfidf                  — sklearn TF-IDF, no API key, always available if
                           scikit-learn is installed
  keyword                — pure-Python bag-of-words fallback, zero extra deps
"""
from __future__ import annotations

import hashlib
import json
import logging
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

    # ── Get or build embedding cache ──────────────────────────────────────────
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
