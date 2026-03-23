"""
Configuration for the Dialog with Data Agent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class DialogConfig:
    # ── Target database (SQL execution) ───────────────────────────────────────
    db_type: str = "postgres"          # "postgres" | "oracle" | "sqlserver" | "sqlite" | "csv" | "excel"
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
    # plan_llm_model: used by plan_node to generate SQL.
    #   Haiku is ~10-15× cheaper than Sonnet and fully capable of structured
    #   JSON output for SQL generation — the biggest per-question cost driver.
    plan_llm_model: str = "claude-haiku-4-5-20251001"
    # llm_model: used by synthesize_node to write the final user-facing insight.
    #   Kept at Sonnet — this is what the business user reads.
    llm_model: str = "claude-sonnet-4-6"
    llm_temperature: float = 0.0

    # ── Query behaviour ───────────────────────────────────────────────────────
    max_sql_queries: int = 10          # max SQL queries the planner may emit
    row_limit: int = 500               # LIMIT applied to each query
    max_insight_rows: int = 2000       # rows passed to the synthesizer LLM

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
    graphrag_hop_depth: int = 2
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
    # Name of the HNSW vector index created by embed_node (must match KGConfig.embed_index_name)
    graphrag_neo4j_index:    str = "kg-node-embeddings"
