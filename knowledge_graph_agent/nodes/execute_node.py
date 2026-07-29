"""
Knowledge Graph pipeline node: persist the translated graph.

There is no live graph database target — the graph is the {nodes, edges}
structure already built by translate_node. This node's job is just to make
that snapshot durable (so fetch_node / mode="load" and other processes can
read it back later) by writing it to the shared KG snapshot store
(see kg_store.py, table kg_snapshots), keyed by kg_id.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from ..state import KGState

logger = logging.getLogger(__name__)


def execute_node(state: KGState) -> KGState:
    config     = state["config"]
    graph_data = state.get("graph_data") or {"nodes": [], "edges": []}
    nodes: List[Dict[str, Any]] = graph_data.get("nodes", [])
    edges: List[Dict[str, Any]] = graph_data.get("edges", [])
    kg_id = getattr(config, "kg_id", "").strip() or "default"

    try:
        import kg_store
        kg_store.save_snapshot(kg_id, nodes, edges)

        state["execution_results"] = [{
            "query": f"persisted snapshot for kg_id={kg_id}",
            "ok": True,
            "error": None,
        }]
        state["executed_count"] = len(nodes)
        state["phase"]          = "executed"
        logger.info("Persisted KG snapshot kg_id=%s (%d nodes, %d edges).", kg_id, len(nodes), len(edges))

    except Exception as exc:
        logger.exception("Failed to persist KG snapshot")
        state["errors"].append(f"Execution failed: {exc}")
        state["execution_results"] = []
        state["executed_count"]    = 0
        state["phase"]             = "executed"

    return state
