"""
LangGraph agent for the Dialog with Data pipeline.

Pipeline:
    START
      → retrieve     (GraphRAG: embed KG nodes, retrieve relevant subgraph)
      → understand   (build schema context from retrieved subgraph)
      → resolve      (LLM maps user query terms → exact categorical DB values)
      → plan         (LLM decomposes NQL → SQL queries using resolved values)
      → execute      (run SQL against target DB)
      → synthesize   (LLM stitches results → insights)
      → END
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Generator, List, Optional, Tuple

from langgraph.graph import END, START, StateGraph

from .config import DialogConfig
from .nodes import (
    retrieve_node,
    execute_node,
    plan_node,
    resolve_node,
    synthesize_node,
    understand_node,
)
from .state import ConversationTurn, DialogState

logger = logging.getLogger(__name__)

_NODES = ["retrieve", "understand", "resolve", "plan", "execute", "synthesize"]


def _build_graph() -> Any:
    g = StateGraph(DialogState)

    g.add_node("retrieve",   retrieve_node)
    g.add_node("understand", understand_node)
    g.add_node("resolve",    resolve_node)
    g.add_node("plan",       plan_node)
    g.add_node("execute",    execute_node)
    g.add_node("synthesize", synthesize_node)

    g.add_edge(START,        "retrieve")
    g.add_edge("retrieve",   "understand")
    g.add_edge("understand", "resolve")
    g.add_edge("resolve",    "plan")
    g.add_edge("plan",       "execute")
    g.add_edge("execute",    "synthesize")
    g.add_edge("synthesize", END)

    return g.compile()


class DialogAgent:
    def __init__(self, config: DialogConfig):
        self._config = config
        self._graph  = _build_graph()

    def _initial_state(
        self,
        natural_query: str,
        kg_nodes: Optional[List[Dict]] = None,
        kg_edges: Optional[List[Dict]] = None,
        conversation_history: Optional[List[ConversationTurn]] = None,
        active_kg_ids: Optional[List[str]] = None,
        kg_bridges_active: Optional[List[Dict]] = None,
        multi_kg_configs: Optional[List[Any]] = None,
    ) -> DialogState:
        return DialogState(
            config               = self._config,
            natural_query        = natural_query,
            schema_context       = "",
            kg_nodes             = kg_nodes or [],
            kg_edges             = kg_edges or [],
            conversation_history = conversation_history or [],
            sql_queries          = [],
            query_results        = [],
            insights             = "",
            errors               = [],
            phase                = "start",
            # Categorical resolution fields
            categorical_columns  = {},
            column_hierarchy     = {},
            term_resolution      = [],
            # Multi-KG federation fields (default to empty = single-KG mode)
            active_kg_ids        = active_kg_ids or [],
            kg_bridges_active    = kg_bridges_active or [],
            multi_kg_configs     = multi_kg_configs or [],
        )

    def run(
        self,
        natural_query: str,
        kg_nodes: Optional[List[Dict]] = None,
        kg_edges: Optional[List[Dict]] = None,
        conversation_history: Optional[List[ConversationTurn]] = None,
        active_kg_ids: Optional[List[str]] = None,
        kg_bridges_active: Optional[List[Dict]] = None,
        multi_kg_configs: Optional[List[Any]] = None,
    ) -> DialogState:
        """Synchronous end-to-end execution; returns the final state."""
        state = self._initial_state(
            natural_query, kg_nodes, kg_edges, conversation_history,
            active_kg_ids=active_kg_ids,
            kg_bridges_active=kg_bridges_active,
            multi_kg_configs=multi_kg_configs,
        )
        result = self._graph.invoke(state)
        result["phase"] = "done"
        return result

    def stream_run(
        self,
        natural_query: str,
        kg_nodes: Optional[List[Dict]] = None,
        kg_edges: Optional[List[Dict]] = None,
        conversation_history: Optional[List[ConversationTurn]] = None,
        active_kg_ids: Optional[List[str]] = None,
        kg_bridges_active: Optional[List[Dict]] = None,
        multi_kg_configs: Optional[List[Any]] = None,
    ) -> Generator[Tuple[str, DialogState], None, None]:
        """Yield (node_name, state_update) for each completed pipeline node."""
        state = self._initial_state(
            natural_query, kg_nodes, kg_edges, conversation_history,
            active_kg_ids=active_kg_ids,
            kg_bridges_active=kg_bridges_active,
            multi_kg_configs=multi_kg_configs,
        )
        for event in self._graph.stream(state, stream_mode="updates"):
            for node_name, state_update in event.items():
                yield node_name, state_update
