"""
LangGraph state for the SHACL Validation Agent.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class SHACLState(TypedDict, total=False):
    """
    config          : SHACLConfig instance
    ontology_text   : raw ontology string (Turtle / XML / N3)
    ontology_format : detected/declared format — "turtle" | "xml" | "n3"
    ontology_graph  : rdflib.Graph of the parsed ontology
    shapes_graph    : rdflib.Graph of the merged SHACL shapes
    conforms        : True when the ontology satisfies all SHACL shapes
    violations      : list of violation dicts (shape, severity, node, message)
    warnings        : list of warning dicts (same schema as violations)
    semantic_issues : Python-detected issues beyond SHACL scope
    summary         : high-level validation summary dict
    errors          : non-fatal pipeline errors
    phase           : current pipeline phase string
    """
    config:          Any
    ontology_text:   str
    ontology_format: str
    ontology_graph:  Any
    shapes_graph:    Any
    conforms:        bool
    violations:      List[Dict]
    warnings:        List[Dict]
    semantic_issues: List[Dict]
    summary:         Dict
    errors:          List[str]
    phase:           str
