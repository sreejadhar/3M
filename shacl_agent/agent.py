"""
SHACL Validation Agent — LangGraph pipeline.

Graph topology:
    START → parse → validate → report → END
                ↓ (error)
            error_end → END

Input :  ontology_text (Turtle / XML / N3 string)
Output:  summary dict with quality label, violations, warnings, suggestions
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph

from .config import SHACLConfig
from .nodes  import parse_node, report_node, validate_node
from .state  import SHACLState

logger = logging.getLogger(__name__)


# ── Routing ────────────────────────────────────────────────────────────────────

def _route_after_parse(state: SHACLState) -> str:
    return "error_end" if state.get("phase") == "error" else "validate"


def _error_end_node(state: SHACLState) -> SHACLState:
    logger.error("SHACL pipeline terminating: %s", state.get("errors"))
    state["phase"] = "error"
    return state


# ── Graph ──────────────────────────────────────────────────────────────────────

def _build_graph() -> Any:
    graph = StateGraph(SHACLState)
    graph.add_node("parse",     parse_node)
    graph.add_node("validate",  validate_node)
    graph.add_node("report",    report_node)
    graph.add_node("error_end", _error_end_node)

    graph.add_edge(START, "parse")
    graph.add_conditional_edges(
        "parse",
        _route_after_parse,
        {"validate": "validate", "error_end": "error_end"},
    )
    graph.add_edge("validate", "report")
    graph.add_edge("report",   END)
    graph.add_edge("error_end", END)

    return graph.compile()


_GRAPH = _build_graph()


# ── Public API ─────────────────────────────────────────────────────────────────

class SHACLAgent:
    """
    Validate an OWL/RDF ontology against SHACL shapes.

    Usage::

        agent  = SHACLAgent(SHACLConfig())
        result = agent.run(ontology_text)

        # result["summary"]["quality"]         → "PASS" | "WARN" | "FAIL"
        # result["summary"]["violations"]      → list of violation dicts
        # result["summary"]["warnings"]        → list of warning dicts
        # result["summary"]["suggestions"]     → actionable suggestions
        # result["summary"]["ontology_stats"]  → {classes, datatype_props, …}
    """

    def __init__(self, config: Optional[SHACLConfig] = None):
        self._config = config or SHACLConfig()

    def run(self, ontology_text: str) -> Dict:
        initial: SHACLState = {
            "config":        self._config,
            "ontology_text": ontology_text,
            "errors":        [],
            "phase":         "init",
        }
        final = _GRAPH.invoke(initial)
        return {
            "quality":       final.get("summary", {}).get("quality", "ERROR"),
            "conforms":      final.get("conforms", False),
            "summary":       final.get("summary", {}),
            "ontology_format": final.get("ontology_format", "unknown"),
            "phase":         final.get("phase", "error"),
            "errors":        final.get("errors", []),
        }
