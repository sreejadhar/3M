"""
Configuration dataclass for the Knowledge Graph Agent.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KGConfig:
    # ── KG snapshot store ─────────────────────────────────────────────────────
    # SQLite/Postgres-backed snapshot store (see kg_store.py) — the graph is
    # persisted as {nodes, edges} JSON keyed by kg_id, not written to a live
    # graph database. Same store the orchestrator and kg_optimizer read from.
    kg_store_path: str = "data/kg_store.db"

    # ── KG identity ───────────────────────────────────────────────────────────
    # Unique name for this knowledge graph.  Used to isolate nodes/edges in the
    # snapshot store so multiple KGs can coexist without collision.
    # e.g. "sales_prod", "hr_staging", "finance_v2".
    # Defaults to "default" when empty.  Must match DialogConfig.graphrag_kg_id
    # for the dialog agent to retrieve the correct KG at query time.
    kg_id: str = ""

    # ── Behaviour ─────────────────────────────────────────────────────────────
    mode:           str  = "generate"  # "generate" | "update" | "load"
    clear_existing: bool = False       # Drop all existing vertices/edges before loading
    batch_size:     int  = 50          # Queries executed per batch (for progress tracking)

    # ── Column taxonomy profiling ─────────────────────────────────────────────
    # When enabled, profile_node runs after parse_node to annotate each column
    # with statistical_type, semantic_role, and format_pattern via an LLM call.
    # Requires ANTHROPIC_API_KEY.  Disable to skip the LLM enrichment step.
    profile_enabled: bool = True

    # ── GraphRAG embedding (production) ───────────────────────────────────────
    # When enabled, embed_node runs after execute_node to attach embedding
    # vectors to each node in the snapshot, enabling vector-search retrieval
    # (numpy cosine similarity) via retrieve_node at dialog time.
    embed_enabled:    bool = False     # Set True to activate
    # Backend: "auto" | "sentence-transformers" | "openai"
    # "auto" tries sentence-transformers first, then openai.
    # tfidf / keyword are NOT supported (variable-dimension, incompatible with
    # the fixed-dimension vectors stored in the snapshot).
    embed_backend:    str  = "auto"
    # Must match the chosen backend's output dimension:
    #   sentence-transformers all-MiniLM-L6-v2 → 384
    #   openai text-embedding-3-small           → 1536
    embed_dimensions: int  = 384
