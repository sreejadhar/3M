"""
SHACL pipeline node: build the final structured validation report.

Aggregates SHACL violations/warnings and semantic issues into a single
human-readable + machine-consumable summary dict stored in state["summary"].
"""
from __future__ import annotations

import logging
from typing import Dict, List

import rdflib
from rdflib.namespace import OWL, RDF, RDFS

from ..state import SHACLState

logger = logging.getLogger(__name__)


def _count_owl_types(g: rdflib.Graph) -> Dict[str, int]:
    return {
        "classes":           sum(1 for _ in g.subjects(RDF.type, OWL.Class)
                                 if not isinstance(_, rdflib.BNode)),
        "datatype_props":    sum(1 for _ in g.subjects(RDF.type, OWL.DatatypeProperty)
                                 if not isinstance(_, rdflib.BNode)),
        "object_props":      sum(1 for _ in g.subjects(RDF.type, OWL.ObjectProperty)
                                 if not isinstance(_, rdflib.BNode)),
        "functional_props":  sum(1 for _ in g.subjects(RDF.type, OWL.FunctionalProperty)
                                 if not isinstance(_, rdflib.BNode)),
        "triples":           len(g),
    }


def _severity_label(conforms: bool, n_violations: int, n_semantic: int) -> str:
    """Return a traffic-light quality label."""
    if conforms and n_violations == 0 and n_semantic == 0:
        return "PASS"
    if n_violations == 0:
        return "WARN"
    return "FAIL"


def report_node(state: SHACLState) -> SHACLState:
    onto_graph      = state.get("ontology_graph") or rdflib.Graph()
    conforms        = state.get("conforms", False)
    violations      = state.get("violations", [])
    warnings        = state.get("warnings", [])
    semantic_issues = state.get("semantic_issues", [])

    # ── Ontology stats ────────────────────────────────────────────────────────
    stats = _count_owl_types(onto_graph)

    # ── Breakdown by check / shape ────────────────────────────────────────────
    def _group_by(items: List[Dict], key: str) -> Dict[str, int]:
        groups: Dict[str, int] = {}
        for item in items:
            k = item.get(key) or "unknown"
            k = str(k).split("/")[-1].split("#")[-1]   # strip namespace prefix
            groups[k] = groups.get(k, 0) + 1
        return groups

    shacl_by_shape    = _group_by(violations + warnings, "shape")
    semantic_by_check = _group_by(semantic_issues, "check")

    # ── Quality label ─────────────────────────────────────────────────────────
    quality = _severity_label(
        conforms,
        len(violations),
        sum(1 for i in semantic_issues if i.get("severity") == "Violation"),
    )

    # ── Suggestions ───────────────────────────────────────────────────────────
    suggestions: List[str] = []

    orphans = [i for i in semantic_issues if i.get("check") == "OrphanClass"]
    if orphans:
        suggestions.append(
            f"{len(orphans)} class(es) have no FK or IND edges. "
            "Review the IND detection threshold or add explicit FK constraints."
        )

    low_cov = [i for i in semantic_issues if i.get("check") == "LowCoverage"]
    if low_cov:
        suggestions.append(
            f"{len(low_cov)} relationship(s) have coverage below threshold. "
            "These should be treated as candidate links only — validate before using as JOIN keys."
        )

    ns_drift = [i for i in semantic_issues if i.get("check") == "NamespaceDrift"]
    if ns_drift:
        suggestions.append(
            f"{len(ns_drift)} URI(s) use a different namespace prefix. "
            "Normalise all URIs to the declared owl:Ontology base URI."
        )

    dup_labels = [i for i in semantic_issues if i.get("check") == "DuplicateClassLabel"]
    if dup_labels:
        suggestions.append(
            f"{len(dup_labels)} class label(s) are duplicated. "
            "Use unique rdfs:label values — the NL query planner uses labels to disambiguate tables."
        )

    missing_domain = [v for v in violations if "domain" in v.get("message", "").lower()]
    if missing_domain:
        suggestions.append(
            f"{len(missing_domain)} propert(ies) are missing rdfs:domain. "
            "This breaks OWL-DL reasoning and KG node→edge translation."
        )

    missing_range = [v for v in violations if "range" in v.get("message", "").lower()]
    if missing_range:
        suggestions.append(
            f"{len(missing_range)} propert(ies) are missing rdfs:range. "
            "The KG translator cannot determine column types without rdfs:range."
        )

    # ── Assemble summary ──────────────────────────────────────────────────────
    summary = {
        "quality":          quality,
        "conforms_shacl":   conforms,
        "ontology_stats":   stats,
        "violation_count":  len(violations),
        "warning_count":    len(warnings) + len(
            [i for i in semantic_issues if i.get("severity") == "Warning"]
        ),
        "semantic_issue_count": len(semantic_issues),
        "violations":       violations,
        "warnings":         warnings,
        "semantic_issues":  semantic_issues,
        "shacl_by_shape":   shacl_by_shape,
        "semantic_by_check": semantic_by_check,
        "suggestions":      suggestions,
        "pipeline_errors":  state.get("errors", []),
    }

    state["summary"] = summary
    state["phase"]   = "reported"

    logger.info(
        "report_node: quality=%s | violations=%d | warnings=%d | semantic=%d",
        quality, len(violations), len(warnings), len(semantic_issues),
    )
    return state
