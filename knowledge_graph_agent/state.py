"""
LangGraph state definition for the Knowledge Graph Agent.
TypedDict(total=False) so LangGraph 0.2+ can introspect the schema and
merge partial updates returned by each node correctly.
"""
from __future__ import annotations

from typing import Any, Dict, List

from typing_extensions import TypedDict


class KGState(TypedDict, total=False):
    config:             Any        # KGConfig instance
    ontology_text:      str        # Raw ontology string
    ontology_format:    str        # "turtle" | "xml" | "n3"
    ontology_graph:     Any        # rdflib.Graph after parsing
    queries:            List[str]  # Generated Cypher or Gremlin statements
    graph_data:         Dict       # {nodes: [...], edges: [...]} for UI visualisation
    execution_results:  List[Dict] # Per-query execution summaries
    node_count:         int        # OWL classes mapped to graph nodes
    edge_count:         int        # OWL object properties mapped to graph edges
    executed_count:     int        # Queries successfully executed
    errors:             List[str]  # Accumulated non-fatal errors
    phase:              str        # "init"|"parsed"|"translated"|"executed"|"embedded"|"fetched"|"error"
    embeddings_stored:  bool       # True after embed_node successfully persists embeddings
