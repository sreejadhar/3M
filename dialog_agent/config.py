"""
Configuration for the Dialog with Data Agent.

LLM model selection is driven by the DIALOG_ENV environment variable:

    DIALOG_ENV=production   → synthesize uses claude-sonnet-4-6  (default)
    DIALOG_ENV=development  → synthesize uses claude-haiku-4-5 (cheap)

SQL planning (plan_node) always uses Haiku regardless of environment.
Set DIALOG_ENV in docker-compose.yml or your .env file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Any, List

# ---------------------------------------------------------------------------
# Environment-aware model defaults
# ---------------------------------------------------------------------------
_ENV = os.environ.get("DIALOG_ENV", "production").strip().lower()
_IS_DEV = _ENV in ("development", "dev", "local")

_DEFAULT_SYNTH_MODEL = (
    "claude-haiku-4-5"   # cheap — fast iteration in dev
    if _IS_DEV else
    "claude-sonnet-4-6"           # quality — what business users read
)


@dataclass
class DialogConfig:
    # ── Source / KG identity ──────────────────────────────────────────────────
    source_id: str = ""                 # orchestrator source id — used to load active KPIs

    # ── Target database (SQL execution) ───────────────────────────────────────
    db_type: str = "postgres"          # "postgres" | "redshift" | "oracle" | "sqlserver" |
                                       # "bigquery" | "teradata" | "delta_lake" | "databricks" |
                                       # "sqlite" | "csv" | "excel"
    db_host: str = ""
    db_port: int = 5432
    db_name: str = ""
    db_schema: str = "public"
    db_user: str = ""
    db_password: str = ""
    db_connection_string: str = ""     # overrides individual fields when set
    db_extra: Dict[str, Any] = field(default_factory=dict)
    db_file_path: str = ""             # for SQLite / CSV / Excel sources

    # ── LLM settings ──────────────────────────────────────────────────────────
    # plan_llm_model: SQL generation — always Haiku (structured JSON output,
    #   10-15× cheaper than Sonnet, no quality difference for SQL tasks).
    plan_llm_model: str = "claude-haiku-4-5"
    # llm_model: insight synthesis — Sonnet in production, Haiku in dev.
    #   Driven by DIALOG_ENV env var; can also be overridden per-request via API.
    llm_model: str = field(default_factory=lambda: _DEFAULT_SYNTH_MODEL)
    llm_temperature: float = 0.0

    # ── Query behaviour ───────────────────────────────────────────────────────
    max_sql_queries: int = 5           # max SQL queries the planner may emit
    row_limit: int = 500               # LIMIT applied to each query
    max_insight_rows: int = 2000       # rows passed to the synthesizer LLM
    # For raw-row queries (no GROUP BY / aggregation), automatically inject a
    # companion COUNT(*) query so the user sees the total matching row count,
    # not just the size of the sampled page.
    raw_row_count_companion: bool = True

    # ── User context ──────────────────────────────────────────────────────────
    analyst_role: str = ""             # e.g. "Financial Analyst" — personalises insights

    # ── GraphRAG retrieval ────────────────────────────────────────────────────
    # Hybrid graph retrieval: embed KG node titles, find top-K tables most
    # relevant to the NLQ via cosine similarity, BFS-expand via FK edges.
    #
    # Two paths selected automatically:
    #   Production (Neo4j): set graphrag_neo4j_uri — uses the HNSW vector
    #     index written by embed_node at KG build time.  Safe for multi-worker
    #     deployments; embeddings are shared across all processes.
    #   In-memory (dev/small schemas): graphrag_neo4j_uri empty — embeds node
    #     titles once per session, caches in process memory with numpy.
    graphrag_enabled: bool = True
    # Number of seed tables returned by vector search before graph expansion.
    graphrag_top_k: int = 8
    # BFS hops from seed nodes when expanding via FK edges.
    graphrag_hop_depth: int = 1
    # Hard cap on tables included in the retrieved subgraph after BFS expansion.
    # Prevents token bloat when FK chains connect many tables. Top-scored tables
    # are kept when the cap triggers.
    graphrag_max_tables: int = 8
    # Only activate retrieval when the schema has more than this many tables.
    # For small schemas the full schema fits easily in the prompt.
    graphrag_min_tables: int = 10
    # Embedding backend: "auto" | "sentence-transformers" | "openai" | "tfidf" | "keyword"
    # "auto" tries sentence-transformers → tfidf → keyword in order.
    # Note: "tfidf" and "keyword" are in-memory only; Neo4j path requires
    # "sentence-transformers" or "openai" (fixed-dimension vectors).
    graphrag_embedding_backend: str = "auto"

    # ── Neo4j connection for production GraphRAG ──────────────────────────────
    # Leave graphrag_neo4j_uri empty to use the in-memory fallback.
    # These should point to the same Neo4j instance used by the KG pipeline.
    graphrag_neo4j_uri:      str = ""        # e.g. "bolt://localhost:7687"
    graphrag_neo4j_username: str = "neo4j"
    graphrag_neo4j_password: str = ""
    graphrag_neo4j_database: str = "neo4j"
    # Which KG to query.  Must match KGConfig.kg_id used when the KG was built.
    # e.g. "sales_prod", "hr_staging".  Defaults to "default" when empty.
    graphrag_kg_id:          str = ""
    # HNSW index name.  When empty, derived automatically as "kg-{graphrag_kg_id}-embeddings"
    # so it matches the index created by embed_node.  Only set this explicitly
    # if you used a custom KGConfig.embed_index_name.
    graphrag_neo4j_index:    str = ""

    # ── Multi-KG federation ───────────────────────────────────────────────────────
    multi_kg_enabled: bool = True          # enable NLQ router
    kg_ids: List[str] = field(default_factory=list)  # explicit KG list (bypasses router)
    kg_router_threshold: float = 0.30      # min cosine similarity for routing
