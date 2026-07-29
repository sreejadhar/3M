"""
GA + ablation optimization pipeline for ontology/knowledge-graph construction.

Separate from, and read-only with respect to, ontology_agent / knowledge_graph_agent /
dialog_agent — it imports their config dataclasses and agent classes, but does not
modify them. See kg_optimizer/cli.py for the entrypoint.
"""
