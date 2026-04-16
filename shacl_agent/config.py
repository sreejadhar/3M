"""
Configuration for the SHACL Validation Agent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SHACLConfig:
    """
    ontology_format     : Hint for the parser — "turtle" | "xml" | "n3" | "auto".
                          "auto" (default) tries turtle first, falls back to xml.
    extra_shapes_ttl    : Additional SHACL shapes in Turtle syntax supplied by the
                          caller.  Merged with the built-in shapes before validation.
    min_coverage        : Minimum acceptable coverage for ObjectProperty INDs (0–1).
                          ObjectProperties whose rdfs:comment mentions a coverage
                          percentage below this threshold are flagged as warnings.
                          Default 0.5 (50 %).
    check_orphan_classes: Flag classes that are never referenced as domain or range
                          of any property.  Orphan classes suggest missing FK/IND
                          relationships.  Default True.
    check_namespace     : Flag properties / classes whose URI does not share the
                          same base namespace as the owl:Ontology declaration.
                          Helps catch copy-paste URI drift.  Default True.
    abort_on_parse_error: If True the pipeline aborts when the ontology cannot be
                          parsed; if False it returns a partial report.
                          Default True.
    """
    ontology_format:      str   = "auto"
    extra_shapes_ttl:     str   = ""
    min_coverage:         float = 0.5
    check_orphan_classes: bool  = True
    check_namespace:      bool  = True
    abort_on_parse_error: bool  = True
