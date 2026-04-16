"""
SHACL pipeline node: run pyshacl validation + semantic checks.

Two independent validation passes:
  1. SHACL structural validation  (pyshacl — shapes graph vs ontology graph)
  2. Semantic checks              (pure Python, beyond SHACL expressivity)
     • orphan classes (no domain/range reference in any property)
     • low-coverage ObjectProperties (coverage < config.min_coverage)
     • namespace drift (URIs outside declared base namespace)
     • duplicate rdfs:label on owl:Class nodes
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List

import rdflib
from rdflib.namespace import OWL, RDF, RDFS

from ..state import SHACLState

logger = logging.getLogger(__name__)

SH   = rdflib.Namespace("http://www.w3.org/ns/shacl#")
MQSH = rdflib.Namespace("http://metadata-agent.io/shacl/")

# Regex to extract coverage percentage from rdfs:comment strings like
# "Coverage: 89.3%" or "coverage 89.3 %"
_COVERAGE_RE = re.compile(r'[Cc]overage[:\s]+(\d+(?:\.\d+)?)\s*%')


def _run_shacl(onto_graph: rdflib.Graph, shapes_graph: rdflib.Graph):
    """
    Run pyshacl and return (conforms, results_graph, results_text).
    Returns (None, None, error_str) when pyshacl is not importable.
    """
    try:
        import pyshacl
    except ImportError:
        return None, None, "pyshacl not installed — install with: pip install pyshacl"

    try:
        conforms, results_graph, results_text = pyshacl.validate(
            data_graph=onto_graph,
            shacl_graph=shapes_graph,
            inference="rdfs",       # run RDFS-level inference so owl:Class instances resolve
            abort_on_first=False,
            allow_warnings=True,
        )
        return conforms, results_graph, results_text
    except Exception as exc:
        return False, None, str(exc)


def _parse_results(results_graph: rdflib.Graph) -> tuple[List[Dict], List[Dict]]:
    """
    Walk the SHACL results graph and split results into violations (sh:Violation)
    and warnings (sh:Warning / sh:Info).
    Returns (violations, warnings).
    """
    violations: List[Dict] = []
    warnings:   List[Dict] = []

    if results_graph is None:
        return violations, warnings

    for result in results_graph.subjects(RDF.type, SH.ValidationResult):
        severity = results_graph.value(result, SH.resultSeverity)
        node     = results_graph.value(result, SH.focusNode)
        path     = results_graph.value(result, SH.resultPath)
        message  = results_graph.value(result, SH.resultMessage)
        shape    = results_graph.value(result, SH.sourceShape)

        entry = {
            "node":     str(node)    if node    else None,
            "path":     str(path)    if path    else None,
            "message":  str(message) if message else "(no message)",
            "shape":    str(shape)   if shape   else None,
            "severity": str(severity).split("#")[-1] if severity else "Violation",
        }

        sev_str = entry["severity"].lower()
        if sev_str in ("violation",):
            violations.append(entry)
        else:
            warnings.append(entry)

    return violations, warnings


# ── Semantic checks ───────────────────────────────────────────────────────────

def _check_orphan_classes(g: rdflib.Graph) -> List[Dict]:
    """Classes that are never a domain or range of any property."""
    all_classes = set(g.subjects(RDF.type, OWL.Class))
    referenced  = set()
    for _, o in g.subject_objects(RDFS.domain):
        referenced.add(o)
    for _, o in g.subject_objects(RDFS.range):
        referenced.add(o)

    issues = []
    for cls in all_classes:
        if isinstance(cls, rdflib.BNode):
            continue
        if cls not in referenced:
            label = g.value(cls, RDFS.label)
            issues.append({
                "check":    "OrphanClass",
                "severity": "Warning",
                "node":     str(cls),
                "label":    str(label) if label else None,
                "message":  (
                    f"Class '{label or cls}' is not referenced as domain or range "
                    "of any property — possible missing FK/IND relationship."
                ),
            })
    return issues


def _check_coverage(g: rdflib.Graph, min_coverage: float) -> List[Dict]:
    """
    ObjectProperties whose rdfs:comment declares a coverage below min_coverage
    are flagged.  The LLM-generated comments contain patterns like:
      "Coverage: 73.2%"
    """
    issues = []
    for prop in g.subjects(RDF.type, OWL.ObjectProperty):
        for comment in g.objects(prop, RDFS.comment):
            m = _COVERAGE_RE.search(str(comment))
            if m:
                cov = float(m.group(1)) / 100.0
                if cov < min_coverage:
                    label = g.value(prop, RDFS.label)
                    issues.append({
                        "check":    "LowCoverage",
                        "severity": "Warning",
                        "node":     str(prop),
                        "label":    str(label) if label else None,
                        "coverage": round(cov, 4),
                        "threshold": min_coverage,
                        "message": (
                            f"ObjectProperty '{label or prop}' has coverage "
                            f"{cov * 100:.1f}% which is below the threshold "
                            f"{min_coverage * 100:.0f}% — treat this relationship "
                            "as a candidate link, not a confirmed FK."
                        ),
                    })
    return issues


def _check_namespace(g: rdflib.Graph) -> List[Dict]:
    """
    All owl:Class and owl:DatatypeProperty / owl:ObjectProperty URIs should
    share the same base namespace as the owl:Ontology declaration.
    """
    issues = []

    # Detect declared base namespace from owl:Ontology subject
    base_ns = None
    for onto in g.subjects(RDF.type, OWL.Ontology):
        if isinstance(onto, rdflib.URIRef):
            base_ns = str(onto).rstrip("/#")
            break

    if base_ns is None:
        return issues  # No owl:Ontology declaration — can't check

    types_to_check = [OWL.Class, OWL.DatatypeProperty, OWL.ObjectProperty]
    for rdf_type in types_to_check:
        for subject in g.subjects(RDF.type, rdf_type):
            if isinstance(subject, rdflib.BNode):
                continue
            uri = str(subject)
            if not uri.startswith(base_ns):
                label = g.value(subject, RDFS.label)
                issues.append({
                    "check":    "NamespaceDrift",
                    "severity": "Warning",
                    "node":     uri,
                    "label":    str(label) if label else None,
                    "expected_ns": base_ns,
                    "message": (
                        f"URI '{uri}' does not share the ontology base namespace "
                        f"'{base_ns}' — ensure all resources use a consistent prefix."
                    ),
                })
    return issues


def _check_duplicate_labels(g: rdflib.Graph) -> List[Dict]:
    """Two owl:Class nodes with the same rdfs:label (case-insensitive)."""
    label_map: Dict[str, List[str]] = {}
    for cls in g.subjects(RDF.type, OWL.Class):
        if isinstance(cls, rdflib.BNode):
            continue
        for lbl in g.objects(cls, RDFS.label):
            key = str(lbl).strip().lower()
            label_map.setdefault(key, []).append(str(cls))

    issues = []
    for lbl, uris in label_map.items():
        if len(uris) > 1:
            issues.append({
                "check":    "DuplicateClassLabel",
                "severity": "Warning",
                "label":    lbl,
                "nodes":    uris,
                "message": (
                    f"Label '{lbl}' is shared by {len(uris)} classes: "
                    + ", ".join(uris[:3])
                    + (" …" if len(uris) > 3 else "")
                    + " — use distinct labels to avoid ambiguity in NL queries."
                ),
            })
    return issues


# ── Node entry point ──────────────────────────────────────────────────────────

def validate_node(state: SHACLState) -> SHACLState:
    config        = state["config"]
    onto_graph    = state.get("ontology_graph")
    shapes_graph  = state.get("shapes_graph")

    if onto_graph is None:
        state["errors"].append("validate_node: ontology_graph is missing — parse_node failed?")
        state["phase"] = "error"
        return state

    # ── Pass 1: SHACL structural validation ──────────────────────────────────
    state["violations"] = []
    state["warnings"]   = []

    if shapes_graph and len(shapes_graph) > 0:
        conforms, results_graph, results_text = _run_shacl(onto_graph, shapes_graph)

        if conforms is None:
            # pyshacl not available — record as a non-fatal warning
            state["errors"].append(str(results_text))
            logger.warning("validate_node: SHACL check skipped — %s", results_text)
            state["conforms"] = True   # treat as conforms so pipeline continues
        else:
            violations, warnings = _parse_results(results_graph)
            state["violations"] = violations
            state["warnings"]   = warnings
            state["conforms"]   = conforms
            logger.info(
                "validate_node: SHACL — conforms=%s, violations=%d, warnings=%d",
                conforms, len(violations), len(warnings),
            )
            if not conforms:
                logger.debug("SHACL results text:\n%s", results_text)
    else:
        logger.warning("validate_node: shapes_graph is empty — SHACL check skipped")
        state["conforms"] = True

    # ── Pass 2: semantic checks ───────────────────────────────────────────────
    semantic: List[Dict] = []

    if config.check_orphan_classes:
        orphans = _check_orphan_classes(onto_graph)
        semantic.extend(orphans)
        if orphans:
            logger.info("validate_node: %d orphan class(es) found", len(orphans))

    coverage_issues = _check_coverage(onto_graph, config.min_coverage)
    semantic.extend(coverage_issues)

    if config.check_namespace:
        ns_issues = _check_namespace(onto_graph)
        semantic.extend(ns_issues)
        if ns_issues:
            logger.info("validate_node: %d namespace drift issue(s)", len(ns_issues))

    dup_issues = _check_duplicate_labels(onto_graph)
    semantic.extend(dup_issues)

    state["semantic_issues"] = semantic
    logger.info("validate_node: %d semantic issue(s)", len(semantic))

    state["phase"] = "validated"
    return state
