"""
Knowledge Graph Agent — converts OWL/RDF ontologies into a {nodes, edges}
knowledge graph snapshot, persisted via the KG snapshot store (kg_store.py).

Completely decoupled from metadata_agent and ontology_agent.
The only input is a raw ontology string (Turtle / RDF-XML / N3).
"""
from .agent import KGAgent
from .config import KGConfig

__all__ = ["KGAgent", "KGConfig"]
