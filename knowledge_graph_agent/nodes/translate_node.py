"""
Knowledge Graph pipeline node: translate an rdflib OWL graph into
Cypher (Neo4j) or Gremlin (TinkerPop) statements.

Mapping strategy
----------------
owl:Class            -> graph node  (label = class name)
owl:DatatypeProperty -> node property / attribute (stored as node metadata)
owl:ObjectProperty   -> directed edge between two class nodes
owl:FunctionalProperty on ObjectProperty  -> edge annotated as 1:N or 1:1
rdfs:comment         -> tooltip / title annotation in graph_data
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Set, Tuple

from rdflib import OWL, RDF, RDFS, BNode, Graph, URIRef

from ..state import KGState

logger = logging.getLogger(__name__)


# ── URI helpers ───────────────────────────────────────────────────────────────

def _local_name(uri: Any) -> str:
    s = str(uri)
    if "#" in s:
        return s.rsplit("#", 1)[-1]
    if "/" in s:
        return s.rsplit("/", 1)[-1]
    return s


def _get_label(g: Graph, uri: URIRef) -> str:
    label = next(g.objects(uri, RDFS.label), None)
    return str(label) if label else ""


def _safe_label(name: str) -> str:
    """Convert to a valid Neo4j node label / Gremlin vertex label."""
    s = re.sub(r"[^a-zA-Z0-9]", "_", name)
    if s and s[0].isdigit():
        s = "_" + s
    return s or "Unknown"


def _safe_rel(name: str) -> str:
    """Convert to a valid Neo4j relationship type (uppercase, underscores)."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name).upper().strip("_") or "RELATED_TO"


def _escape_cypher(value: str) -> str:
    """Escape a string for safe embedding in a Cypher query."""
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _escape_gremlin(value: str) -> str:
    """Escape a string for safe embedding in a Gremlin query."""
    return value.replace("\\", "\\\\").replace("'", "\\'")


# ── Extract classes and properties from rdflib Graph ─────────────────────────

def _extract_ontology(g: Graph) -> Tuple[Dict, List]:
    """
    Returns:
        classes   — {uri_str: {uri, name, comments, datatype_props}}
        obj_props — [{uri, name, domain, range, is_functional, is_inv_functional}]
    """
    # Namespaces whose URIs should never be treated as domain classes
    _SKIP_NS = (
        "http://www.w3.org/2001/XMLSchema#",
        "http://www.w3.org/2002/07/owl#",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "http://www.w3.org/2000/01/rdf-schema#",
    )

    # --- OWL Classes ---
    classes: Dict[str, Dict] = {}
    for cls_uri in g.subjects(RDF.type, OWL.Class):
        if isinstance(cls_uri, BNode):
            continue
        uri_str = str(cls_uri)
        # Skip XSD built-ins, OWL/RDF/RDFS meta-vocabulary
        if any(uri_str.startswith(ns) for ns in _SKIP_NS):
            continue
        label   = _get_label(g, cls_uri) or _local_name(cls_uri)
        comments = [str(c) for c in g.objects(cls_uri, RDFS.comment)]
        classes[uri_str] = {
            "uri":            uri_str,
            "name":           label,
            "comments":       comments,
            "datatype_props": [],
        }

    # --- OWL DatatypeProperties → attach to domain class ---
    for prop_uri in g.subjects(RDF.type, OWL.DatatypeProperty):
        if isinstance(prop_uri, BNode):
            continue
        domain = next(g.objects(prop_uri, RDFS.domain), None)
        if domain is None or str(domain) not in classes:
            continue
        range_uri = next(g.objects(prop_uri, RDFS.range), None)
        prop_label = _get_label(g, prop_uri) or _local_name(prop_uri)
        xsd_type   = _local_name(range_uri) if range_uri else "string"
        prop_comments = [str(c) for c in g.objects(prop_uri, RDFS.comment)]
        classes[str(domain)]["datatype_props"].append({
            "name":     prop_label,
            "range":    xsd_type,
            "comments": prop_comments,
        })

    # --- OWL ObjectProperties → directed edges ---
    obj_props: List[Dict] = []
    seen_props: Set[str]  = set()
    for prop_uri in g.subjects(RDF.type, OWL.ObjectProperty):
        if isinstance(prop_uri, BNode):
            continue
        uri_str = str(prop_uri)
        if uri_str in seen_props:
            continue
        seen_props.add(uri_str)

        domain = next(g.objects(prop_uri, RDFS.domain), None)
        range_ = next(g.objects(prop_uri, RDFS.range), None)
        if domain is None or range_ is None:
            continue
        if str(domain) not in classes or str(range_) not in classes:
            continue

        prop_label      = _get_label(g, prop_uri) or _local_name(prop_uri)
        is_func         = (prop_uri, RDF.type, OWL.FunctionalProperty) in g
        is_inv_func     = (prop_uri, RDF.type, OWL.InverseFunctionalProperty) in g
        comments        = [str(c) for c in g.objects(prop_uri, RDFS.comment)]

        obj_props.append({
            "uri":              uri_str,
            "name":             prop_label,
            "domain":           str(domain),
            "range":            str(range_),
            "is_functional":    is_func,
            "is_inv_functional": is_inv_func,
            "comments":         comments,
        })

    return classes, obj_props


# ── Cypher generation ─────────────────────────────────────────────────────────

def _generate_cypher(classes: Dict, obj_props: List, config: Any) -> List[str]:
    queries: List[str] = []

    # Resolve the KG identifier — default to "default" when not set.
    # Every node and edge is stamped with this so multiple KGs can coexist
    # in the same Neo4j database without URI collisions.
    kg_id = _escape_cypher(getattr(config, "kg_id", "").strip() or "default")

    if config.clear_existing:
        # Only delete nodes/edges belonging to this KG, not the entire graph.
        queries.append(f"MATCH (n:KGNode {{kg_id: '{kg_id}'}}) DETACH DELETE n")

    # Composite uniqueness constraint: (uri, kg_id) so the same URI can exist
    # in multiple KGs without violating the constraint.
    queries.append(
        "CREATE CONSTRAINT kg_node_uri_kg_id IF NOT EXISTS "
        "FOR (n:KGNode) REQUIRE (n.uri, n.kg_id) IS UNIQUE"
    )

    # Class nodes
    for uri, cls in classes.items():
        safe_uri   = _escape_cypher(uri)
        safe_name  = _escape_cypher(cls["name"])
        node_label = _safe_label(cls["name"])
        comment    = _escape_cypher(cls["comments"][0]) if cls["comments"] else ""

        # Datatype props as node properties (name:xsd_type pairs)
        dt_props = {_escape_cypher(p["name"]): _escape_cypher(p["range"])
                    for p in cls["datatype_props"]}
        dt_str = ", ".join(f"n.`{k}` = '{v}'" for k, v in dt_props.items())

        # MERGE key includes kg_id so nodes from different KGs never collide.
        q = (
            f"MERGE (n:KGNode:{node_label} "
            f"{{uri: '{safe_uri}', kg_id: '{kg_id}'}}) "
            f"ON CREATE SET n.name = '{safe_name}', n.type = 'owl:Class'"
            + (f", n.description = '{comment}'" if comment else "")
        )
        if dt_str:
            q += f"\nON CREATE SET {dt_str}"
        queries.append(q)

    # Object property edges — MATCH filters by kg_id so edges only connect
    # nodes within the same KG; kg_id is stored on the relationship too.
    for op in obj_props:
        rel_type    = _safe_rel(op["name"])
        domain_uri  = _escape_cypher(op["domain"])
        range_uri   = _escape_cypher(op["range"])
        op_name     = _escape_cypher(op["name"])
        cardinality = (
            "1:1" if op["is_functional"] and op["is_inv_functional"]
            else "1:N" if op["is_functional"]
            else "M:N"
        )
        q = (
            f"MATCH (a:KGNode {{uri: '{domain_uri}', kg_id: '{kg_id}'}}), "
            f"(b:KGNode {{uri: '{range_uri}', kg_id: '{kg_id}'}})\n"
            f"MERGE (a)-[r:{rel_type} {{name: '{op_name}', "
            f"type: 'owl:ObjectProperty', cardinality: '{cardinality}', "
            f"kg_id: '{kg_id}'}}]->(b)"
        )
        queries.append(q)

    return queries


# ── Gremlin generation ────────────────────────────────────────────────────────

def _generate_gremlin(classes: Dict, obj_props: List, config: Any) -> List[str]:
    queries: List[str] = []

    kg_id = _escape_gremlin(getattr(config, "kg_id", "").strip() or "default")

    if config.clear_existing:
        # Only drop vertices belonging to this KG.
        queries.append(f"g.V().has('kg_id', '{kg_id}').drop().iterate()")

    for uri, cls in classes.items():
        safe_uri  = _escape_gremlin(uri)
        safe_name = _escape_gremlin(cls["name"])
        comment   = _escape_gremlin(cls["comments"][0]) if cls["comments"] else ""

        # Build property chain — kg_id added so vertices are KG-scoped.
        props = [
            f".property('uri', '{safe_uri}')",
            f".property('kg_id', '{kg_id}')",
            f".property('name', '{safe_name}')",
            ".property('type', 'owl:Class')",
        ]
        if comment:
            props.append(f".property('description', '{comment}')")
        for p in cls["datatype_props"]:
            pn = _escape_gremlin(p["name"])
            pr = _escape_gremlin(p["range"])
            props.append(f".property('{pn}_xsd_type', '{pr}')")

        props_str = "".join(props)
        # Upsert on (uri, kg_id) so the same URI in different KGs creates
        # separate vertices rather than merging into one.
        q = (
            f"g.V().has('uri', '{safe_uri}').has('kg_id', '{kg_id}').fold()"
            f".coalesce(unfold(), addV('{safe_name}'){props_str}).next()"
        )
        queries.append(q)

    for op in obj_props:
        edge_label  = _escape_gremlin(op["name"]).replace(" ", "_")
        domain_uri  = _escape_gremlin(op["domain"])
        range_uri   = _escape_gremlin(op["range"])
        cardinality = (
            "1:1" if op["is_functional"] and op["is_inv_functional"]
            else "1:N" if op["is_functional"]
            else "M:N"
        )
        # Look up vertices by both uri AND kg_id to avoid cross-KG edge creation.
        q = (
            f"g.V().has('uri', '{domain_uri}').has('kg_id', '{kg_id}').as('a')"
            f".V().has('uri', '{range_uri}').has('kg_id', '{kg_id}')"
            f".coalesce("
            f"inE('{edge_label}').where(outV().as('a')), "
            f"addE('{edge_label}').from('a')"
            f".property('type', 'owl:ObjectProperty')"
            f".property('cardinality', '{cardinality}')"
            f".property('kg_id', '{kg_id}')"
            f").next()"
        )
        queries.append(q)

    return queries


# ── Graph data for UI visualisation ──────────────────────────────────────────

def _build_graph_data(classes: Dict, obj_props: List) -> Dict:
    nodes = []
    edges = []

    # Regex to extract join column pairs from ObjectProperty rdfs:comments.
    # Matches: "Join columns: col1 → col2" or "Explicit FK: tbl1.col1 → tbl2.col2"
    _join_re     = re.compile(r'Join\s+columns?\s*:\s*(\w+)\s*[→>-]+\s*(\w+)', re.IGNORECASE)
    _expl_fk_re  = re.compile(r'Explicit\s+FK\s*:\s*\S+?\.(\w+)\s*[→>-]+\s*\S+?\.(\w+)', re.IGNORECASE)

    for uri, cls in classes.items():
        # Include ALL properties — no cap. The dialog agent reads every column
        # from this title to build the schema context; truncating here causes
        # columns to disappear from the LLM's view of the schema.
        dt_line_parts = []
        for p in cls["datatype_props"]:
            col_line = f"  {p['name']}: {p['range']}"
            # Include ALL column-level comments: statistics/description first,
            # then taxonomy annotation (if profile_node ran).
            # Taxonomy comment must come last so _sync_taxonomy_from_kg_nodes
            # regex can reliably find it at the end of the line.
            col_comments = p.get("comments", [])
            if col_comments:
                non_tax = [c for c in col_comments if not str(c).startswith("taxonomy:")]
                tax_com = [c for c in col_comments if str(c).startswith("taxonomy:")]
                # First non-taxonomy comment (statistics / description), then taxonomy
                for c in (non_tax[:1] + tax_com):
                    col_line += f"  -- {c}"
            dt_line_parts.append(col_line)
        dt_lines = "\n".join(dt_line_parts)

        title = f"Class: {cls['name']}"
        # Include ALL class comments (not just first two) — FD hints, row
        # counts, and other metadata all live here and are useful for the LLM.
        for c in cls["comments"]:
            title += f"\n{c}"
        if dt_lines:
            title += f"\n\nProperties:\n{dt_lines}"

        nodes.append({
            "id":    uri,
            "label": cls["name"],
            "title": title,
            "color": "#63b3ed",
            "size":  20 + min(len(cls["datatype_props"]) * 2, 20),
            # Structured property list consumed by the bridge inference engine
            # (_profiles_from_nodes in kg_inference_engine.py). Without this,
            # the engine sees node.get("properties") == None and produces empty
            # column profiles, so no cross-KG bridges are ever inferred.
            "properties": [
                {"name": p["name"], "type": p["range"]}
                for p in cls["datatype_props"]
            ],
        })

    for op in obj_props:
        cardinality = (
            "1:1" if op["is_functional"] and op["is_inv_functional"]
            else "1:N" if op["is_functional"]
            else "M:N"
        )
        # Parse explicit FK join-column pairs from ObjectProperty rdfs:comments.
        # These look like "Join columns: customer_id → id" or
        # "Explicit FK: orders.customer_id → customers.id".
        # Stored as a list of [src_col, tgt_col] so understand_node can emit
        # precise "JOIN ON t1.col = t2.col" lines for the SQL planner.
        join_columns: List[List[str]] = []
        for c in op["comments"]:
            m = _join_re.search(c)
            if m:
                join_columns.append([m.group(1), m.group(2)])
                continue
            m2 = _expl_fk_re.search(c)
            if m2:
                join_columns.append([m2.group(1), m2.group(2)])

        comments = "; ".join(op["comments"])

        # Edge label: short cardinality only (e.g. "1:N") so the graph stays
        # readable. Full property name + cardinality + comments go in the
        # tooltip (title) so engineers can inspect details on hover.
        edges.append({
            "from":         op["domain"],
            "to":           op["range"],
            "label":        cardinality,
            "title":        f"{op['name']} ({cardinality})" + (f"\n{comments}" if comments else ""),
            "join_columns": join_columns,   # [[src_col, tgt_col], ...]
        })

    return {"nodes": nodes, "edges": edges}


# ── Node ──────────────────────────────────────────────────────────────────────

def translate_node(state: KGState) -> KGState:
    config = state["config"]
    g      = state["ontology_graph"]

    try:
        classes, obj_props = _extract_ontology(g)
    except Exception as exc:
        state["errors"].append(f"Ontology extraction failed: {exc}")
        state["phase"] = "error"
        return state

    if not classes:
        state["errors"].append("No OWL classes found in ontology — nothing to translate.")
        state["phase"] = "error"
        return state

    if config.graph_type == "neo4j":
        queries = _generate_cypher(classes, obj_props, config)
    else:
        queries = _generate_gremlin(classes, obj_props, config)

    graph_data = _build_graph_data(classes, obj_props)

    logger.info(
        "Translated ontology: %d classes, %d object properties → %d %s statements",
        len(classes), len(obj_props), len(queries),
        "Cypher" if config.graph_type == "neo4j" else "Gremlin",
    )

    state["queries"]    = queries
    state["graph_data"] = graph_data
    state["node_count"] = len(classes)
    state["edge_count"] = len(obj_props)
    state["phase"]      = "translated"
    return state
