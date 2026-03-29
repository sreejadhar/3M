"""
Knowledge Graph Agent — LangGraph pipeline.

Graph topology (mode = "generate" or "update"):
    START → parse → [error] → error_end → END
                 → profile  (LLM taxonomy enrichment — adds rdfs:comment annotations)
                 → translate → execute → END

Graph topology (mode = "load"):
    START → fetch → END

Completely decoupled from metadata_agent and ontology_agent.
Inputs: raw ontology string + KGConfig (generate/update), or just KGConfig (load).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph

from .config import KGConfig
from .nodes import embed_node, execute_node, fetch_node, parse_node, profile_node, translate_node
from .state import KGState

logger = logging.getLogger(__name__)


# ── Routing ───────────────────────────────────────────────────────────────────

def _route_from_start(state: KGState) -> str:
    """Route to fetch_node for load mode, otherwise run the full pipeline."""
    return "fetch" if state["config"].mode == "load" else "parse"


def _route_after_parse(state: KGState) -> str:
    return "error_end" if state.get("phase") == "error" else "profile"


def _route_after_profile(state: KGState) -> str:
    # profile_node never sets phase=error (failures are non-fatal warnings)
    return "translate"


def _route_after_translate(state: KGState) -> str:
    return "error_end" if state.get("phase") == "error" else "execute"


def _route_after_execute(state: KGState) -> str:
    """Route to embed_node when embedding is enabled, otherwise end."""
    if state.get("phase") == "error":
        return "error_end"
    config = state.get("config")
    if config and getattr(config, "embed_enabled", False) and getattr(config, "neo4j_uri", ""):
        return "embed"
    return END


def _route_after_fetch(state: KGState) -> str:
    return "error_end" if state.get("phase") == "error" else END


def _error_end_node(state: KGState) -> KGState:
    logger.error("KG pipeline terminating: %s", state.get("errors"))
    state["phase"] = "error"
    return state


# ── Graph builder ─────────────────────────────────────────────────────────────

def _build_graph() -> Any:
    graph = StateGraph(KGState)
    graph.add_node("parse",     parse_node)
    graph.add_node("profile",   profile_node)
    graph.add_node("translate", translate_node)
    graph.add_node("execute",   execute_node)
    graph.add_node("embed",     embed_node)
    graph.add_node("fetch",     fetch_node)
    graph.add_node("error_end", _error_end_node)

    graph.add_conditional_edges(
        START,
        _route_from_start,
        {"parse": "parse", "fetch": "fetch"},
    )
    graph.add_conditional_edges(
        "parse",
        _route_after_parse,
        {"profile": "profile", "error_end": "error_end"},
    )
    graph.add_conditional_edges(
        "profile",
        _route_after_profile,
        {"translate": "translate"},
    )
    graph.add_conditional_edges(
        "translate",
        _route_after_translate,
        {"execute": "execute", "error_end": "error_end"},
    )
    graph.add_conditional_edges(
        "execute",
        _route_after_execute,
        {"embed": "embed", END: END, "error_end": "error_end"},
    )
    graph.add_conditional_edges(
        "fetch",
        _route_after_fetch,
        {END: END, "error_end": "error_end"},
    )
    graph.add_edge("embed",     END)
    graph.add_edge("error_end", END)

    return graph.compile()


# ── Public API ────────────────────────────────────────────────────────────────

class KGAgent:
    """
    Convert an OWL/RDF ontology to a Cypher or Gremlin knowledge graph,
    incrementally update an existing graph, or load a graph already stored
    in the database for visualisation.

    Usage (generate / update)::

        cfg    = KGConfig(graph_type="neo4j", neo4j_uri="bolt://localhost:7687",
                          neo4j_username="neo4j", neo4j_password="secret",
                          mode="generate")   # or "update" for incremental
        agent  = KGAgent(cfg)
        result = agent.run(turtle_text)

    Usage (load existing graph)::

        cfg    = KGConfig(graph_type="neo4j", neo4j_uri="bolt://localhost:7687",
                          neo4j_username="neo4j", neo4j_password="secret",
                          mode="load")
        agent  = KGAgent(cfg)
        result = agent.load()
    """

    def __init__(self, config: Optional[KGConfig] = None):
        self._config = config or KGConfig()
        self._graph  = _build_graph()

    def _initial_state(self, ontology_text: str = "", ontology_format: str = "turtle") -> KGState:
        return {
            "config":            self._config,
            "ontology_text":     ontology_text,
            "ontology_format":   ontology_format,
            "ontology_graph":    None,
            "queries":           [],
            "graph_data":        {"nodes": [], "edges": []},
            "execution_results": [],
            "node_count":        0,
            "edge_count":        0,
            "executed_count":    0,
            "errors":            [],
            "phase":             "init",
        }

    def _to_result(self, final: Dict) -> Dict:
        return {
            "queries":           final.get("queries", []),
            "graph_data":        final.get("graph_data", {"nodes": [], "edges": []}),
            "node_count":        final.get("node_count", 0),
            "edge_count":        final.get("edge_count", 0),
            "executed_count":    final.get("executed_count", 0),
            "execution_results": final.get("execution_results", []),
            "errors":            final.get("errors", []),
            "phase":             final.get("phase", ""),
        }

    def run(self, ontology_text: str, ontology_format: str = "turtle") -> Dict:
        """Synchronous generate/update execution — returns a result summary dict."""
        final = self._graph.invoke(self._initial_state(ontology_text, ontology_format))
        for err in final.get("errors", []):
            logger.warning("KG warning: %s", err)
        return self._to_result(final)

    def load(self) -> Dict:
        """Synchronous load execution — fetches existing graph from the DB."""
        final = self._graph.invoke(self._initial_state())
        for err in final.get("errors", []):
            logger.warning("KG load warning: %s", err)
        return self._to_result(final)

    def stream_run(self, ontology_text: str, ontology_format: str = "turtle"):
        """Yield (node_name, state_update) as each pipeline node completes."""
        for event in self._graph.stream(self._initial_state(ontology_text, ontology_format)):
            for node_name, state_update in event.items():
                yield node_name, state_update

    def stream_load(self):
        """Yield (node_name, state_update) for the load pipeline."""
        for event in self._graph.stream(self._initial_state()):
            for node_name, state_update in event.items():
                yield node_name, state_update
