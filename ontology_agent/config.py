"""
Configuration for the Ontology Agent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class OntologyConfig:
    """
    base_uri          : Base URI for the ontology namespace.
    ontology_name     : Human-readable name embedded in the OWL header.
    output_path       : Where to write the serialised ontology file.
                        If None the file is not written to disk.
    serialize_format  : "turtle" (.ttl) | "xml" (.owl) | "n3"
    include_statistics: Annotate datatype properties with column stats
                        (unique count, null count, min/max) as rdfs:comment.
    annotate_concepts : Use an LLM call at index time to annotate each column
                        with its standard business concept label (e.g. "arpu" →
                        "avg-revenue-per-user", "nii" → "net-interest-income").
                        Stored as rdfs:comment "Business concept: <label>" so
                        downstream nodes can surface the label in schema_context.
                        Default: True — disable only to reduce indexing cost.
    llm_model         : Model used for concept annotation (if annotate_concepts).
                        Defaults to claude-haiku-4-5 (cheapest, sufficient for
                        column-name interpretation).
    source_domain     : Optional domain hint passed to the concept LLM
                        (e.g. "telecom", "banking", "healthcare").  When set the
                        LLM resolves ambiguous abbreviations in domain context.
    enable_name_resolution : Replace raw table/column names in RDFS.label with a
                        GA-chosen canonical label (see kg_optimizer/naming_resolver.py)
                        when a mapping exists in table_labels/column_labels.
                        Default: False — raw names pass through unchanged, identical
                        to pre-existing behavior.
    table_labels      : raw table name -> canonical label. Only used when
                        enable_name_resolution=True; missing entries fall back to
                        the raw name.
    column_labels     : raw column name -> canonical label. Only used when
                        enable_name_resolution=True; missing entries fall back to
                        the raw name.
    """
    base_uri:           str  = "http://metadata-agent.io/ontology/"
    ontology_name:      str  = "DatabaseOntology"
    output_path:        Optional[str] = None
    serialize_format:   str  = "turtle"
    include_statistics: bool = True
    annotate_concepts:  bool = True
    llm_model:          str  = "claude-haiku-4-5"
    source_domain:      str  = ""
    enable_name_resolution: bool = False
    table_labels:       Dict[str, str] = field(default_factory=dict)
    column_labels:      Dict[str, str] = field(default_factory=dict)
