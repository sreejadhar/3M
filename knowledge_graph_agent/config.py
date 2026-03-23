"""
Configuration dataclass for the Knowledge Graph Agent.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KGConfig:
    # Target graph database type
    graph_type: str = "neo4j"          # "neo4j" | "gremlin"

    # ── Neo4j settings ────────────────────────────────────────────────────────
    neo4j_uri:      str = ""           # e.g. "bolt://localhost:7687" — empty = skip execution
    neo4j_username: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"

    # ── Gremlin / TinkerPop settings ──────────────────────────────────────────
    gremlin_url:               str = ""    # e.g. "ws://localhost:8182/gremlin" — empty = skip
    gremlin_traversal_source:  str = "g"

    # ── Behaviour ─────────────────────────────────────────────────────────────
    mode:           str  = "generate"  # "generate" | "update" | "load"
    clear_existing: bool = False       # Drop all existing vertices/edges before loading
    batch_size:     int  = 50          # Queries executed per batch (for progress tracking)

    # ── GraphRAG embedding (production) ───────────────────────────────────────
    # When enabled, embed_node runs after execute_node to write embedding
    # vectors onto Neo4j nodes, enabling production vector-search retrieval
    # via retrieve_node at dialog time.
    embed_enabled:    bool = False     # Set True to activate
    # Backend: "auto" | "sentence-transformers" | "openai"
    # "auto" tries sentence-transformers first, then openai.
    # tfidf / keyword are NOT supported (variable-dimension, incompatible with HNSW).
    embed_backend:    str  = "auto"
    # Must match the chosen backend's output dimension:
    #   sentence-transformers all-MiniLM-L6-v2 → 384
    #   openai text-embedding-3-small           → 1536
    embed_dimensions: int  = 384
    # Name of the Neo4j HNSW vector index created on KGNode.embedding
    embed_index_name: str  = "kg-node-embeddings"
