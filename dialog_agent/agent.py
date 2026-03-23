"""
LangGraph agent for the Dialog with Data pipeline.

Pipeline:
    START
      → retrieve     (GraphRAG: embed KG nodes, retrieve relevant subgraph)
      → understand   (build schema context from retrieved subgraph)
      → plan         (LLM decomposes NQL → SQL queries)
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
    synthesize_node,
    understand_node,
)
from .state import ConversationTurn, DialogState

logger = logging.getLogger(__name__)

_NODES = ["retrieve", "understand", "plan", "execute", "synthesize"]


def _build_graph() -> Any:
    g = StateGraph(DialogState)

    g.add_node("retrieve",   retrieve_node)
    g.add_node("understand", understand_node)
    g.add_node("plan",       plan_node)
    g.add_node("execute",    execute_node)
    g.add_node("synthesize", synthesize_node)

    g.add_edge(START,        "retrieve")
    g.add_edge("retrieve",   "understand")
    g.add_edge("understand", "plan")
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
        )

    def run(
        self,
        natural_query: str,
        kg_nodes: Optional[List[Dict]] = None,
        kg_edges: Optional[List[Dict]] = None,
        conversation_history: Optional[List[ConversationTurn]] = None,
    ) -> DialogState:
        """Synchronous end-to-end execution; returns the final state."""
        state = self._initial_state(natural_query, kg_nodes, kg_edges, conversation_history)
        result = self._graph.invoke(state)
        result["phase"] = "done"
        return result

    def stream_run(
        self,
        natural_query: str,
        kg_nodes: Optional[List[Dict]] = None,
        kg_edges: Optional[List[Dict]] = None,
        conversation_history: Optional[List[ConversationTurn]] = None,
    ) -> Generator[Tuple[str, DialogState], None, None]:
        """Yield (node_name, state_update) for each completed pipeline node."""
        state = self._initial_state(natural_query, kg_nodes, kg_edges, conversation_history)
        for event in self._graph.stream(state, stream_mode="updates"):
            for node_name, state_update in event.items():
                yield node_name, state_update
