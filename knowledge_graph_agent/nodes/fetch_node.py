"""
Knowledge Graph pipeline node: fetch an existing graph snapshot.

Used in "load" mode — skips parse/translate and reads the {nodes, edges}
snapshot previously written by execute_node to the KG snapshot store
(kg_store.py, table kg_snapshots), keyed by kg_id.
"""
from __future__ import annotations

import logging

from ..state import KGState

logger = logging.getLogger(__name__)


def fetch_node(state: KGState) -> KGState:
    """Retrieve an existing graph snapshot from the KG snapshot store."""
    logger.info("=== fetch_node (load mode) ===")
    config = state["config"]
    kg_id  = getattr(config, "kg_id", "").strip() or "default"

    try:
        import kg_store
        nodes, edges = kg_store.load_snapshot(kg_id)

        state["graph_data"]        = {"nodes": nodes, "edges": edges}
        state["node_count"]        = len(nodes)
        state["edge_count"]        = len(edges)
        state["queries"]           = []
        state["execution_results"] = []
        state["executed_count"]    = 0
        state["phase"]             = "fetched"
        logger.info("Fetched %d nodes, %d edges from snapshot store (kg_id=%s)", len(nodes), len(edges), kg_id)

    except Exception as exc:
        logger.exception("fetch_node: failed to retrieve graph")
        state["errors"].append(f"fetch_node: {exc}")
        state["phase"] = "error"

    return state
