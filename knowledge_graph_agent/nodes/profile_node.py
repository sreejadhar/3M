"""
profile_node — Enrich the parsed OWL graph with LLM-inferred data taxonomy.

Runs between parse_node and translate_node.

For every table (owl:Class) in the rdflib Graph, it sends all column names and
XSD types to a lightweight LLM and gets back per-column taxonomy annotations:

  statistical_type  : nominal | categorical | ordinal | continuous | boolean | identifier
  semantic_role     : measure | time_period | product_category | product_sub_category |
                      product_dimension_key | geography_dimension_key | time_dimension_key |
                      org_unit | demographic | free_text | boolean_flag | identifier | other
  format_pattern    : (optional) e.g. "FY{YYYY}", "CY{YYYY}", "{YYYY}-Q{Q}"
  domain            : product | geography | time | finance | org | customer | logistics | other

These annotations are written back as rdfs:comment triples on each DatatypeProperty.
translate_node._build_graph_data already surfaces col_comments[0] in the node title
(as "  -- <annotation>"), so the annotations propagate automatically into:
  • the KG node titles stored in Neo4j
  • the schema context text the dialog agent reads
  • the embedding corpus used for GraphRAG retrieval

Impact on the dialog pipeline
──────────────────────────────
  • plan_node sees "semantic_role: geography_dimension_key" → never joins a product
    FK to a geography FK (fixes segment_id = region_id class of bugs)
  • resolve_node sees "format_pattern: FY{YYYY}" → maps "2024" → "FY2024"
  • understand_node already annotates [categorical] sample values; profile_node adds
    the *structural* taxonomy (nominal/ordinal/etc.) that doesn't require data access
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── LLM prompt ────────────────────────────────────────────────────────────────

_PROFILE_SYSTEM = """\
You are a data modelling expert. Given a database table and its column names + XSD types,
classify each column's statistical measurement type and semantic role.

Return ONLY a JSON object — no prose, no markdown fences.

CLASSIFICATION GUIDE:

statistical_type (pick one):
  identifier  — surrogate or natural key (e.g. id, code, _id columns)
  nominal     — FK / dimension key that links to another table (e.g. segment_id, region_id)
  categorical — low-cardinality text with distinct labels (e.g. category, status, country)
  ordinal     — ordered set: either ranked labels or structured period strings (e.g. FY2024, Q1)
  continuous  — numeric measure (revenue, amount, count, rate, %)
  boolean     — yes/no, 0/1, true/false
  free_text   — high-cardinality unstructured text (description, notes, address)
  date        — actual date/datetime/timestamp column

semantic_role (pick the best match):
  measure                    — numeric KPI, amount, revenue, count, rate
  time_period                — year, quarter, month, week, fiscal period
  time_dimension_key         — FK to a date/period dimension table
  product_category           — top-level product grouping
  product_sub_category       — second-level product grouping
  product_dimension_key      — FK to product/segment/SKU dimension
  geography                  — country, region, city stored directly (not FK)
  geography_dimension_key    — FK to a geography/region dimension table
  org_unit                   — division, department, cost centre
  org_dimension_key          — FK to an organisation dimension
  customer_dimension_key     — FK to a customer dimension
  demographic                — age, gender, segment attribute
  identifier                 — surrogate PK or natural key with no join meaning
  boolean_flag               — binary attribute
  free_text                  — unstructured text attribute
  other                      — does not fit any above category

format_pattern (optional — only for ordinal / time_period columns):
  Describe the stored string format, e.g.:
    "FY{YYYY}"      for FY2024
    "CY{YYYY}"      for CY2023
    "{YYYY}-Q{Q}"   for 2024-Q1
    "{YYYY}"        for plain year strings
    "Q{Q}"          for Q1, Q2, Q3, Q4
    ISO 8601        for date strings like 2024-01-15

domain (pick one):
  product | geography | time | finance | org | customer | logistics | risk | other

OUTPUT FORMAT (return exactly this JSON):
{
  "table": "<table_name>",
  "columns": [
    {
      "name": "<column_name>",
      "statistical_type": "<type>",
      "semantic_role": "<role>",
      "format_pattern": "<pattern or null>",
      "domain": "<domain>"
    }
  ]
}
"""

_PROFILE_USER = """\
Table: {table_name}
Columns:
{column_list}

Classify each column. Return the JSON object.
"""


# ── LLM call ──────────────────────────────────────────────────────────────────

def _call_profile_llm(table_name: str, columns: List[Dict[str, str]], model: str) -> List[Dict]:
    """
    Returns a list of column annotation dicts, one per column.
    Falls back to empty list on any error.
    """
    import anthropic

    col_lines = "\n".join(
        f"  {c['name']}: {c['xsd_type']}" for c in columns
    )
    user_msg = _PROFILE_USER.format(table_name=table_name, column_list=col_lines)

    try:
        from llm_client import get_client
        client = get_client()
        msg = client.messages.create(
            model=model,
            max_tokens=2048,
            temperature=0.0,
            system=_PROFILE_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = msg.content[0].text if msg.content else ""
    except Exception as exc:
        logger.warning("profile_node: LLM call failed for table %r — %s", table_name, exc)
        return []

    # Extract JSON
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    try:
        obj = json.loads(cleaned)
        return obj.get("columns", [])
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1:
            try:
                obj = json.loads(cleaned[start : end + 1])
                return obj.get("columns", [])
            except json.JSONDecodeError:
                pass
    logger.warning("profile_node: could not parse LLM JSON for table %r", table_name)
    return []


# ── rdflib annotation helpers ─────────────────────────────────────────────────

def _local_name(uri: Any) -> str:
    s = str(uri)
    if "#" in s:
        return s.rsplit("#", 1)[-1]
    if "/" in s:
        return s.rsplit("/", 1)[-1]
    return s


def _build_taxonomy_comment(annotation: Dict[str, Any]) -> str:
    """
    Serialise a column annotation dict into the compact comment string that
    translate_node._build_graph_data includes in the node title.

    Format:  taxonomy: statistical_type=ordinal | semantic_role=time_period | format_pattern=FY{YYYY} | domain=time
    """
    parts = [
        f"statistical_type={annotation.get('statistical_type', '')}",
        f"semantic_role={annotation.get('semantic_role', '')}",
        f"domain={annotation.get('domain', '')}",
    ]
    fp = annotation.get("format_pattern")
    if fp:
        parts.append(f"format_pattern={fp}")
    return "taxonomy: " + " | ".join(p for p in parts if "=" in p and p.split("=", 1)[1])


# ── node ──────────────────────────────────────────────────────────────────────

def profile_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Annotate the parsed OWL graph with LLM-inferred column taxonomy.
    Adds rdfs:comment triples on DatatypeProperty nodes; these are picked up
    automatically by translate_node._build_graph_data and appear as column
    annotations in the KG node titles.

    Skipped when:
      - profile_enabled = False in KGConfig
      - ANTHROPIC_API_KEY is not set
      - ontology_graph is None (parse_node failed)
    """
    from rdflib import OWL, RDF, RDFS, BNode, Literal, URIRef

    logger.info("=== profile_node ===")

    config = state.get("config")
    if config and not getattr(config, "profile_enabled", True):
        logger.info("profile_node: disabled by config — skipping")
        return state

    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning("profile_node: ANTHROPIC_API_KEY not set — skipping taxonomy enrichment")
        return state

    g = state.get("ontology_graph")
    if g is None:
        logger.warning("profile_node: ontology_graph is None — skipping")
        return state

    model = os.environ.get("PROFILE_LLM_MODEL", "claude-haiku-4-5-20251001")

    _SKIP_NS = (
        "http://www.w3.org/2001/XMLSchema#",
        "http://www.w3.org/2002/07/owl#",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "http://www.w3.org/2000/01/rdf-schema#",
    )

    # Index: class_uri → {name, props: [{prop_uri, name, xsd_type}]}
    classes: Dict[str, Dict] = {}
    for cls_uri in g.subjects(RDF.type, OWL.Class):
        if isinstance(cls_uri, BNode):
            continue
        uri_str = str(cls_uri)
        if any(uri_str.startswith(ns) for ns in _SKIP_NS):
            continue
        label = next(g.objects(cls_uri, RDFS.label), None)
        name  = str(label) if label else _local_name(cls_uri)
        classes[uri_str] = {"name": name, "props": []}

    # Attach each DatatypeProperty to its domain class
    for prop_uri in g.subjects(RDF.type, OWL.DatatypeProperty):
        if isinstance(prop_uri, BNode):
            continue
        domain = next(g.objects(prop_uri, RDFS.domain), None)
        if domain is None or str(domain) not in classes:
            continue
        range_uri  = next(g.objects(prop_uri, RDFS.range), None)
        prop_label = next(g.objects(prop_uri, RDFS.label), None)
        prop_name  = str(prop_label) if prop_label else _local_name(prop_uri)
        xsd_type   = _local_name(range_uri) if range_uri else "string"
        classes[str(domain)]["props"].append({
            "prop_uri": prop_uri,
            "name":     prop_name,
            "xsd_type": xsd_type,
        })

    total_annotated = 0
    for uri_str, cls in classes.items():
        props = cls["props"]
        if not props:
            continue

        table_name = cls["name"]
        columns    = [{"name": p["name"], "xsd_type": p["xsd_type"]} for p in props]

        logger.info(
            "profile_node: classifying %d columns for table %r",
            len(columns), table_name,
        )
        annotations = _call_profile_llm(table_name, columns, model)

        # Build a name → annotation lookup (case-insensitive)
        ann_by_name = {a.get("name", "").lower(): a for a in annotations}

        for prop in props:
            ann = ann_by_name.get(prop["name"].lower())
            if not ann:
                continue
            comment_str = _build_taxonomy_comment(ann)
            # Add rdfs:comment to the DatatypeProperty — translate_node reads
            # the first comment on each property and includes it in the title.
            g.add((prop["prop_uri"], RDFS.comment, Literal(comment_str)))
            total_annotated += 1
            logger.debug(
                "profile_node: %s.%s → %s",
                table_name, prop["name"], comment_str,
            )

    logger.info(
        "profile_node: annotated %d column(s) across %d table(s)",
        total_annotated, len(classes),
    )
    state["phase"] = "profiled"
    return state
