"""
Dialog with Data Agent — natural-language-to-SQL via knowledge graph traversal.

Standalone package; zero imports from metadata_agent, ontology_agent, or
knowledge_graph_agent.
"""
try:
    # agent.py builds the LangGraph pipeline and requires the `langgraph`
    # package, which is NOT installed in every consumer of this package —
    # e.g. orchestrator_api.py only needs lightweight submodules like
    # kg_bridges/kg_registry/pg_store (none of which touch langgraph) and
    # runs in a container whose requirements file omits it. Without this
    # guard, importing ANY dialog_agent submodule crashes those consumers
    # with ModuleNotFoundError, since Python always runs __init__.py first.
    from .agent import DialogAgent
except ImportError:
    DialogAgent = None
from .config import DialogConfig

__all__ = ["DialogAgent", "DialogConfig"]
