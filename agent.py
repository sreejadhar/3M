"""
Metadata Extraction Agent — LangGraph orchestration layer.

Graph topology:
                      ┌──────────┐
                      │  START   │
                      └────┬─────┘
                           │
                    ┌──────▼──────┐
                    │  connection  │  (open DB connection)
                    └──────┬──────┘
                    error? │ ok?
                    ┌──────┼──────────────┐
                    │                     │
              ┌─────▼─────┐        ┌──────▼──────┐
              │   error   │        │  discovery  │  (list tables)
              └───────────┘        └──────┬──────┘
                                   error? │ ok?
                                   ┌──────┼──────┐
                                   │             │
                             ┌─────▼─────┐  ┌───▼──────────┐
                             │   error   │  │  extraction  │  (schema + stats)
                             └───────────┘  └──────┬───────┘
                                                   │
                                            ┌──────▼───────┐
                                            │   analysis   │  (FD + IND + card.)
                                            └──────┬───────┘
                                                   │
                                            ┌──────▼───────┐
                                            │    report    │  (aggregate + write)
                                            └──────┬───────┘
                                                   │
                                               ┌───▼───┐
                                               │  END  │
                                               └───────┘

The agent also exposes an LLM-powered "interpret" step (optional) that uses
a LangChain ReAct agent to answer questions about the extracted metadata.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from .config import AgentConfig
from .nodes.analysis_node import analysis_node
from .nodes.connection_node import connection_node
from .nodes.discovery_node import discovery_node
from .nodes.extraction_node import extraction_node
from .nodes.report_node import report_node
from .state import AgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def _route_after_connection(state: AgentState) -> str:
    return "error_end" if state["phase"] == "error" else "discovery"


def _route_after_discovery(state: AgentState) -> str:
    return "error_end" if state["phase"] == "error" else "extraction"


def _error_end_node(state: AgentState) -> AgentState:
    logger.error("Agent terminating due to errors: %s", state["errors"])
    state["phase"] = "error"
    return state


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Construct and compile the LangGraph StateGraph."""
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("connection",  connection_node)
    graph.add_node("discovery",   discovery_node)
    graph.add_node("extraction",  extraction_node)
    graph.add_node("analysis",    analysis_node)
    graph.add_node("report",      report_node)
    graph.add_node("error_end",   _error_end_node)

    # Edges
    graph.add_edge(START, "connection")

    graph.add_conditional_edges(
        "connection",
        _route_after_connection,
        {"discovery": "discovery", "error_end": "error_end"},
    )

    graph.add_conditional_edges(
        "discovery",
        _route_after_discovery,
        {"extraction": "extraction", "error_end": "error_end"},
    )

    graph.add_edge("extraction", "analysis")
    graph.add_edge("analysis",   "report")
    graph.add_edge("report",     END)
    graph.add_edge("error_end",  END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

class MetadataExtractionAgent:
    """
    High-level wrapper around the LangGraph metadata extraction pipeline.

    Usage:
        agent = MetadataExtractionAgent(config)
        report = agent.run()
        # report is a dict with full metadata

    Optional LLM interpretation:
        insights = agent.ask("Which tables have the most null values?")
    """

    def __init__(self, config: AgentConfig):
        self._config = config
        self._graph = build_graph()
        self._report: Optional[Dict[str, Any]] = None

        # Optional LLM for the interpret step — uses AWS Bedrock via llm_client
        from llm_client import get_client
        self._llm_client = get_client()
        self._llm_model  = config.llm_model
        self._llm_temp   = config.llm_temperature

    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        """
        Execute the full extraction pipeline and return the final report dict.
        """
        initial_state: AgentState = {
            "agent_config":   self._config,
            "db_config":      self._config.db_config,
            "connector":      None,
            "phase":          "init",
            "all_tables":     [],
            "tables_done":    set(),
            "table_metadata": {},
            "func_deps":      [],
            "incl_deps":      [],
            "cardinalities":  [],
            "messages":       [],
            "errors":         [],
            "final_report":   {},
        }

        logger.info("Starting metadata extraction pipeline …")
        final_state = self._graph.invoke(initial_state)

        self._report = final_state.get("final_report", {})
        errors = final_state.get("errors", [])

        if errors:
            logger.warning("Pipeline completed with %d error(s):", len(errors))
            for e in errors:
                logger.warning("  • %s", e)

        logger.info(
            "Pipeline done. Tables: %d | FDs: %d | INDs: %d | Cardinalities: %d",
            self._report.get("summary", {}).get("total_tables", 0),
            self._report.get("summary", {}).get("total_functional_dependencies", 0),
            self._report.get("summary", {}).get("total_inclusion_dependencies", 0),
            self._report.get("summary", {}).get("total_cardinality_relationships", 0),
        )
        return self._report

    # ------------------------------------------------------------------
    def ask(self, question: str) -> str:
        """
        Use the LLM to answer a natural-language question about the report.
        Requires run() to have been called first.
        """
        if not self._report:
            return "No report available yet — call run() first."

        system_content = (
            "You are a data engineering expert.  You have been provided the full "
            "metadata report from a database schema scan.  Answer questions about "
            "the schema structure, data quality, and relationships concisely and accurately.\n\n"
            "METADATA REPORT (JSON):\n"
            + json.dumps(self._report, indent=2, default=str)[:40_000]
        )
        msg = self._llm_client.messages.create(
            model=self._llm_model,
            max_tokens=1024,
            temperature=self._llm_temp,
            system=system_content,
            messages=[{"role": "user", "content": question}],
        )
        return msg.content[0].text

    # ------------------------------------------------------------------
    def stream_run(self):
        """
        Yield (node_name, partial_state) tuples as the pipeline executes.
        Useful for progress monitoring in notebooks or UIs.
        """
        initial_state: AgentState = {
            "agent_config":   self._config,
            "db_config":      self._config.db_config,
            "connector":      None,
            "phase":          "init",
            "all_tables":     [],
            "tables_done":    set(),
            "table_metadata": {},
            "func_deps":      [],
            "incl_deps":      [],
            "cardinalities":  [],
            "messages":       [],
            "errors":         [],
            "final_report":   {},
        }
        for event in self._graph.stream(initial_state):
            for node_name, state_update in event.items():
                if node_name == "report" and isinstance(state_update, dict):
                    self._report = state_update.get("final_report") or {}
                yield node_name, state_update
