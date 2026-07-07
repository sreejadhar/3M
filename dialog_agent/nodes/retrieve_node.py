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
# Keyed by (schema_hash, backend, _NODE_TEXT_VERSION) so cache is invalidated
# when the schema changes OR when _node_text() logic is updated.
_EMBED_CACHE: Dict[str, "_Cache"] = {}
_NODE_TEXT_VERSION = "6"   # bump when _node_text() or _expand_table_name() changes, or the tokenizer changes


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

# Patterns that signal a cross-table analytic query: the user is asking about
# the relationship between TWO distinct data domains.  In these cases the NLQ
# embedding will cluster around whichever domain dominates the wording and
# miss the second domain entirely.  We decompose the query into two halves and
# run independent retrieval for each, then union the seed sets.
_CROSS_DOMAIN_PATTERNS = re.compile(
    r'\b(correlat\w*|vs\.?|versus|compared?\s+to|relationship\s+between|impact\s+of'
    r'|effect\s+of|influence\s+of|driven\s+by|against|by\s+\w+\s+condition'
    r'|trend\s+with|associat\w*|co.?relat\w*'
    r'|systematically|trailing\s+\d+|over\s+the\s+\w+\s+\d+\s+period'
    r'|dragging|portfolio\s+toward|consistently\s+\w+\s+over)',
    re.IGNORECASE,
)

# Split-point markers — words that separate the two sides of the comparison
_CROSS_SPLIT_RE = re.compile(
    r'\s+(?:vs\.?|versus|and|with|against|on|by|across|compared?\s+to|'
    r'impact\s+of|effect\s+of|influence\s+of)\s+',
    re.IGNORECASE,
)
# "between X and Y" → normalise to "X and Y" so the splitter can split on " and "
_BETWEEN_RE = re.compile(r'\bbetween\s+', re.IGNORECASE)

# Limit / exceedance questions: "exceed the meal limit", "over budget", "breach
# of policy", "amount exceeded", "exceed %". These are single-domain questions
# (the dominant wording is usually about the fact being measured — meals,
# spending, attendees — not the limit itself), so the embedding clusters
# around that dominant domain and can miss a separate limit/threshold/policy
# table entirely, exactly like the trailing-period case below. Domain-agnostic:
# applies to expense caps, credit limits, rate limits, budget overages, SLA
# breaches, or any other "did X exceed Y" question, not any one specific table.
_EXCEED_LIMIT_PATTERN = re.compile(
    r'\b(exceed\w*|excess\w*'
    r'|over\s+(?:the\s+)?(?:limit|budget|cap|threshold|allowance|allotment)'
    r'|above\s+(?:the\s+)?(?:limit|maximum|cap|threshold|allowance)'
    r'|more\s+than\s+(?:allowed|permitted|authoriz\w*)'
    r'|breach\w*|non.?compliant|out\s+of\s+policy|overspen\w*|overage\w*)\b',
    re.IGNORECASE,
)


def _decompose_cross_query(query: str) -> List[str]:
    """
    If the query is a cross-domain analytic question (correlation, vs, impact of…),
    a multi-table analytical question (systematically X over trailing N periods,
    dragging portfolio, etc.), or a limit/exceedance question ("exceed the meal
    limit", "over budget"), return multiple sub-queries so all relevant tables
    get fair representation in the seed set.
    Otherwise return [query] (single-vector path).

    Examples:
      "temperature vs purchasing behavior"
          → ["temperature", "purchasing behavior"]
      "correlation between weather and sales"
          → ["weather", "sales"]
      "customers showing systematically negative mix contribution over trailing 12 periods"
          → [original query, "customer mix contribution metrics", "period time trailing"]
      "meals with HCPs which exceed the meal limit"
          → [original query, "limit threshold maximum allowance cap policy rule compliance"]
    """
    # ── Limit / exceedance questions ──────────────────────────────────────────
    # Independent of the cross-domain gate below — this is a single-domain
    # question needing one extra sub-query, not a two-way split.
    if _EXCEED_LIMIT_PATTERN.search(query):
        return [
            query,
            "limit threshold maximum allowance cap policy rule compliance",
        ]

    if not _CROSS_DOMAIN_PATTERNS.search(query):
        return [query]

    # ── Trailing-period / systematically-over-time patterns ───────────────────
    # These are single-domain but multi-table: entity (dim) + metric (fact) + time (dim).
    # Don't split them; instead emit additional sub-queries for the dimension tables.
    _TRAILING_PATTERN = re.compile(
        r'\b(systematically|trailing\s+\d+|over\s+the\s+\w+\s+\d+\s+period'
        r'|dragging|portfolio\s+toward|consistently\s+\w+\s+over)\b',
        re.IGNORECASE,
    )
    if _TRAILING_PATTERN.search(query):
        # Sub-queries that pull in metric/fact table and time dimension
        return [
            query,
            "metric contribution measure value fact table",
            "period time date calendar trailing rolling",
        ]

    # ── Two-domain cross-domain split ─────────────────────────────────────────
    # Normalise "between X and Y" → "X and Y" so the splitter finds " and "
    normalised = _BETWEEN_RE.sub("", query).strip()

    # Try to split on the separator keyword
    parts = _CROSS_SPLIT_RE.split(normalised, maxsplit=1)
    if len(parts) == 2 and all(p.strip() for p in parts):
        return [p.strip() for p in parts]

    # Fallback: return the full query so multi-vector path is a no-op
    return [query]


def _l2_norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / max(n, 1e-9)


def _schema_hash(nodes: List[Dict]) -> str:
    """Stable hash over node IDs + titles — used for cache invalidation."""
    fingerprint = json.dumps(
        sorted((n.get("id", ""), n.get("title", "")) for n in nodes)
    )
    return hashlib.md5(fingerprint.encode()).hexdigest()


def _expand_table_name(label: str) -> str:
    """
    Expand a table name into a richer phrase for embedding by splitting on
    underscores/camelCase and appending a granularity hint when the name
    encodes a specific dimension (channel, category, maker, etc.).

    Examples:
      JSR_bottler_channel         → "JSR bottler channel channel-level breakdown"
      JSR_bottler_category        → "JSR bottler category category-level breakdown"
      JSR_maker_category_channel  → "JSR maker category channel combined breakdown"
      vw_fact_month_combined_m    → "vw fact month combined m monthly fact sales"
      retail_All_data             → "retail All data sales retail"
    """
    # Split on underscores and camelCase transitions
    parts = re.sub(r'([a-z])([A-Z])', r'\1 \2', label)
    parts = re.sub(r'[_]+', ' ', parts).strip()

    lower = label.lower()

    hints = []
    if re.search(r'\bfact\b', lower):
        hints.append("fact table sales measures metrics")
    if re.search(r'\bdim\b|^dim_', lower):
        hints.append("dimension lookup reference attributes master")
    if re.search(r'\bchannel\b', lower) and not re.search(r'\bcategory\b', lower):
        hints.append("channel-level breakdown distribution channel")
    if re.search(r'\bcategory\b', lower) and not re.search(r'\bchannel\b', lower):
        hints.append("category-level breakdown product category")
    if re.search(r'\bcategory\b', lower) and re.search(r'\bchannel\b', lower):
        hints.append("combined breakdown maker category channel granular")
    if re.search(r'\bretail\b', lower):
        hints.append("retail sales")
    if re.search(r'\bpromotion\b|\bpromo\b', lower):
        hints.append("promotion promotional activity")
    if re.search(r'\bweather\b|\bclimate\b|\btemperatur\b|\bprecip\b|\brainfall\b|\bhumid\b|\bwindspeed\b|\bforecast\b', lower):
        hints.append("weather climate temperature environmental conditions")
    if re.search(r'\bview\b|^v_|^vw_', lower):
        hints.append("view aggregated precomputed")
    if re.search(r'\bcustomer\b', lower):
        hints.append("customer account client buyer")
    if re.search(r'\bperiod\b|\btime\b|\bdate\b|\bcalendar\b|\bfiscal\b', lower):
        hints.append("period time date calendar fiscal trailing rolling")

    if hints:
        return f"{parts} {' '.join(hints)}"
    return parts


def _extract_col_semantic_hints(label: str, title: str) -> str:
    """
    Extract semantically meaningful tokens from column names in a node title
    and return them as a space-separated hint string to append to the node text.

    This is the generic complement to _expand_table_name: instead of looking
    at the table name, we look at every column name in the title and split each
    one into its constituent words.  Tokens that already appear in the table
    name are deduplicated so we only surface *new* information.

    Example:
      label = "v_env_metrics"
      title contains columns: weighted_temperature, precipitation_mm,
                               humidity_pct, wind_speed_ms, region_code, period_date

      table tokens already in label: {"v", "env", "metrics"}
      column tokens after split:     {"weighted", "temperature", "precipitation",
                                       "mm", "humidity", "pct", "wind", "speed",
                                       "ms", "region", "code", "period", "date"}
      new tokens (not in label):     "weighted temperature precipitation humidity
                                      wind speed region period"

    These tokens are appended to the node text so that a query for "temperature"
    or "humidity" will score high against this node even though neither word
    appears in the table name.

    Business concept annotations ("Business concept: <label>") from the ontology
    annotation step are also extracted and included — these are the richest signal
    since they are human-readable labels like "temperature", "avg-revenue-per-user".
    """
    # Tokens already present in the table label — don't repeat them
    label_tokens: set = set(re.findall(r'[a-z]+', label.lower()))

    col_tokens: List[str] = []
    concept_labels: List[str] = []

    # Structural tokens from the title format itself — not meaningful for retrieval
    _STRUCTURAL = {
        "properties", "class", "columns", "type", "comment", "comments",
        "integer", "varchar", "decimal", "boolean", "date", "timestamp",
        "float", "text", "bigint", "smallint", "numeric", "char",
        "true", "false", "null", "none",
        "min", "max", "avg", "count", "sum", "distinct", "values",
        "sample", "range", "null_rate", "row", "rows",
        "pk", "fk", "id", "key", "ref",
        "mm", "pct", "ms", "km", "kg", "gb", "mb",  # unit suffixes
    }

    # Parse every "  colname: type  -- ..." line from the Properties section.
    # Require at least 2 leading spaces so the "Properties:" header (0 spaces)
    # and top-level comments are not misidentified as column lines.
    col_line_re  = re.compile(r'^ {2,4}(\w+)\s*:', re.MULTILINE)
    concept_re   = re.compile(r'Business concept:\s*([^\n\-\|]+)', re.IGNORECASE)

    for m in col_line_re.finditer(title):
        col_name = m.group(1)
        # Split on underscores and camelCase transitions
        raw = re.sub(r'([a-z])([A-Z])', r'\1 \2', col_name)
        raw = re.sub(r'[_]+', ' ', raw)
        for tok in raw.lower().split():
            if tok not in label_tokens and tok not in _STRUCTURAL and len(tok) > 2:
                col_tokens.append(tok)

    for m in concept_re.finditer(title):
        raw_label = m.group(1).strip().rstrip('.')
        # kebab-case concept labels → space-separated words
        words = re.split(r'[-\s]+', raw_label)
        concept_labels.extend(w.lower() for w in words if len(w) > 1)

    # Deduplicate while preserving first-occurrence order
    seen: set = set()
    unique: List[str] = []
    for tok in concept_labels + col_tokens:   # concepts first — higher signal
        if tok not in seen and tok not in label_tokens:
            seen.add(tok)
            unique.append(tok)

    return " ".join(unique)


def _node_text(node: Dict) -> str:
    """
    Build a rich text representation of a KG node for embedding.

    Combines three layers:
      1. Expanded table label  — _expand_table_name splits underscores/camelCase
                                 and appends pattern-based domain hints.
      2. Title body            — full KG title: class comments, column names,
                                 types, stats, and business concept annotations.
      3. Column semantic hints — _extract_col_semantic_hints splits every column
                                 name into its constituent tokens so that queries
                                 for "temperature" match v_env_metrics even though
                                 the table name contains no such word.

    Layer 3 is the generic fix for opaque table names: the embedding sees the
    column-level vocabulary regardless of what the table is called.
    """
    label = node.get("label", "")
    title = node.get("title", "")
    expanded   = _expand_table_name(label)
    # Strip "Class: X" prefix — redundant with expanded label
    title_body = re.sub(r'^Class:\s*\S+\s*', '', title, flags=re.IGNORECASE).strip()
    col_hints  = _extract_col_semantic_hints(label, title)

    parts = [expanded, title_body]
    if col_hints:
        parts.append(f"columns: {col_hints}")
    return " ".join(p for p in parts if p).strip()


# ── Embedding backends ────────────────────────────────────────────────────────

def _stem_token(tok: str) -> str:
    """
    Minimal suffix-stripping stemmer — no external dependency. Normalises
    simple plural/verb-form variance ("treaties" -> "treaty", "reinsurers" ->
    "reinsurer") so a query and a schema column/table name that differ only
    by grammatical form still land close together in TF-IDF / keyword space.
    This is the fix for retrieval missing REINSURANCE for "Which reinsurers
    carry the most exposure?" and similar term-form mismatches.
    """
    if len(tok) > 4 and tok.endswith('ies'):
        return tok[:-3] + 'y'
    if len(tok) > 4 and tok.endswith('es') and tok[-3] not in 'aeiou':
        return tok[:-2]
    if len(tok) > 3 and tok.endswith('s') and not tok.endswith('ss'):
        return tok[:-1]
    return tok


def _tokenize_stemmed(text: str) -> List[str]:
    return [_stem_token(w) for w in re.findall(r'\w+', text.lower())]


def _keyword_embed(texts: List[str], vocab_texts: Optional[List[str]] = None) -> np.ndarray:
    """
    Bag-of-words binary vectors.  Vocabulary is built from vocab_texts (corpus);
    if None, vocabulary is built from texts itself.
    """
    tokenize = _tokenize_stemmed
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
        # token_pattern=None + custom tokenizer: apply _stem_token so plural/
        # verb-form variance between the NLQ and schema names ("treaties" vs
        # "TREATY_TYPE", "reinsurers" vs "REINSURER_NAME") doesn't cause a
        # zero-overlap retrieval miss. transform() at query time reuses the
        # same fitted tokenizer automatically.
        vect   = TfidfVectorizer(max_features=3000, sublinear_tf=True,
                                  tokenizer=_tokenize_stemmed, token_pattern=None)
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

        # Cap to graphrag_max_tables (seed URIs = highest-scored nodes)
        max_tables_neo = getattr(config, "graphrag_max_tables", 15)
        if len(sub_nodes) > max_tables_neo:
            # Seed URIs are already the highest-similarity nodes; keep those first
            seed_set = set(seed_uris)
            seeded   = [n for n in sub_nodes if n.get("id", "") in seed_set]
            expanded = [n for n in sub_nodes if n.get("id", "") not in seed_set]
            sub_nodes = (seeded + expanded)[:max_tables_neo]
            kept_ids = {n.get("id", "") for n in sub_nodes}
            sub_edges = [
                e for e in sub_edges
                if e.get("from", "") in kept_ids and e.get("to", "") in kept_ids
            ]
            logger.info(
                "retrieve_node (Neo4j): capped subgraph from %d → %d tables (max_tables=%d)",
                len(subgraph_uris), max_tables_neo, max_tables_neo,
            )

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
    hop_depth   = getattr(config, "graphrag_hop_depth",   1)
    min_tables  = getattr(config, "graphrag_min_tables",  top_k)
    max_tables  = getattr(config, "graphrag_max_tables",  15)
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
    key     = f"{s_hash}:{backend}:v{_NODE_TEXT_VERSION}"

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
    # For cross-domain analytic queries (correlation / vs / impact of …) the NLQ
    # embedding clusters around whichever domain dominates the wording.  We
    # decompose into sub-queries and take the element-wise max so both sides
    # of the comparison get fair representation in the seed set.
    sub_queries = _decompose_cross_query(query)
    try:
        scores = np.zeros(len(nodes), dtype=np.float32)
        for sq in sub_queries:
            q_vec   = cache.embed_query(sq)                      # [D]
            sq_scores = cache.matrix @ q_vec                     # [N] cosine similarities
            scores  = np.maximum(scores, sq_scores)              # element-wise max
        if len(sub_queries) > 1:
            logger.info(
                "retrieve_node: cross-domain query decomposed into %d sub-queries: %s",
                len(sub_queries), sub_queries,
            )
    except Exception as exc:
        logger.warning("retrieve_node: query embedding failed (%s) — using full schema", exc)
        return state

    # ── Widen the seed budget for decomposed (multi-concept) queries ─────────
    # A query that _decompose_cross_query split into N sub-queries is asking
    # about N distinct concepts at once (two domains being compared, a metric
    # + a time dimension, or a fact + the limit/threshold it's being checked
    # against). A single fixed top_k budget forces those concepts to compete
    # for the same slots — e.g. several sibling "compliance/policy" tables can
    # all score similarly on a limit-related sub-query and crowd out the one
    # that's actually needed. This widens top_k/max_tables proportionally to
    # the number of concepts detected, generic across ALL decomposition cases
    # (cross-domain, trailing-period, exceedance) — not any one specific
    # table or pattern. A non-decomposed query (the overwhelming majority)
    # takes len(sub_queries) == 1 and this is a complete no-op: k and
    # max_tables are byte-for-byte identical to before this change.
    _extra_concepts = max(len(sub_queries) - 1, 0)
    if _extra_concepts:
        widened_top_k     = top_k + 4 * _extra_concepts
        widened_max_tables = max_tables + 4 * _extra_concepts
        logger.info(
            "retrieve_node: query decomposed into %d concepts — widening "
            "top_k %d→%d, max_tables %d→%d so no single concept crowds "
            "out another",
            len(sub_queries), top_k, widened_top_k, max_tables, widened_max_tables,
        )
        top_k, max_tables = widened_top_k, widened_max_tables

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

    # ── Drop BFS-expanded nodes that are below the relevance floor ────────────
    # Seed nodes (top-K by similarity) are always kept.  Nodes that were pulled
    # in purely via FK expansion and score below the floor are dropped so the
    # LLM does not see unrelated tables and join them unnecessarily.
    expand_score_floor = getattr(config, "graphrag_expand_score_floor", 0.10)
    seed_id_set  = set(seed_ids)
    node_score   = {cache.node_ids[i]: float(scores[i]) for i in range(len(scores))}
    before_floor = len(sub_nodes)
    sub_nodes = [
        n for n in sub_nodes
        if n.get("id", "") in seed_id_set
        or node_score.get(n.get("id", ""), 0.0) >= expand_score_floor
    ]
    if len(sub_nodes) < before_floor:
        pruned_ids = {n.get("id", "") for n in sub_nodes}
        sub_edges  = [
            e for e in sub_edges
            if e.get("from", "") in pruned_ids and e.get("to", "") in pruned_ids
        ]
        logger.info(
            "retrieve_node: pruned %d low-relevance BFS-expanded nodes "
            "(score < %.2f) — %d tables remain",
            before_floor - len(sub_nodes), expand_score_floor, len(sub_nodes),
        )

    # ── Cap to graphrag_max_tables to prevent token bloat from deep FK chains ─
    if len(sub_nodes) > max_tables:
        sub_nodes.sort(
            key=lambda n: node_score.get(n.get("id", ""), 0.0), reverse=True
        )
        sub_nodes = sub_nodes[:max_tables]
        kept_ids = {n.get("id", "") for n in sub_nodes}
        sub_edges = [
            e for e in sub_edges
            if e.get("from", "") in kept_ids and e.get("to", "") in kept_ids
        ]
        logger.info(
            "retrieve_node: capped subgraph from %d → %d tables (graphrag_max_tables=%d)",
            len(subgraph_ids), max_tables, max_tables,
        )

    logger.info(
        "retrieve_node: retrieved %d/%d tables, %d/%d edges",
        len(sub_nodes), len(nodes), len(sub_edges), len(edges),
    )

    state["kg_nodes"] = sub_nodes
    state["kg_edges"] = sub_edges
    return state
