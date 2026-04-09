"""
Knowledge Graph pipeline node: fetch an existing graph from the database.

Used in "load" mode — skips parse/translate and reads the graph that was
previously written to Neo4j or Gremlin, reconstructing the {nodes, edges}
visualisation data from the live database.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from ..state import KGState

logger = logging.getLogger(__name__)


# ── Neo4j fetch ───────────────────────────────────────────────────────────────

def _fetch_neo4j(config: Any) -> Tuple[Dict, int, int]:
    from neo4j import GraphDatabase  # noqa: PLC0415

    kg_id  = getattr(config, "kg_id", "").strip() or "default"
    driver = GraphDatabase.driver(
        config.neo4j_uri,
        auth=(config.neo4j_username, config.neo4j_password),
    )
    nodes: List[Dict] = []
    edges: List[Dict] = []

    # Properties written by embed_node / internal fields — never shown as columns
    _INTERNAL = {"uri", "name", "type", "description", "kg_id", "title", "embedding"}

    try:
        with driver.session(database=config.neo4j_database) as session:
            # Fetch KGNode vertices scoped to this KG only
            for rec in session.run(
                "MATCH (n:KGNode {kg_id: $kg_id}) RETURN n", kg_id=kg_id
            ):
                props = dict(rec["n"].items())
                uri   = props.get("uri", str(rec["n"].id))
                name  = props.get("name", uri)

                # Use the rich title stored by embed_node when available;
                # otherwise reconstruct from stored column properties.
                stored_title = props.get("title", "")
                if stored_title:
                    title = stored_title
                    dt_lines: List[str] = []
                else:
                    desc  = props.get("description", "")
                    title = f"Class: {name}"
                    if desc:
                        title += f"\n{desc}"
                    dt_lines = [
                        f"  {k}: {v}"
                        for k, v in props.items()
                        if k not in _INTERNAL
                    ]
                    if dt_lines:
                        title += "\n\nProperties:\n" + "\n".join(dt_lines)

                # Structured column list for bridge inference engine.
                # Non-internal props are the datatype-property columns stored
                # by translate_node as "ON CREATE SET n.col_name = 'xsd:type'".
                col_props = [
                    {"name": k, "type": v}
                    for k, v in props.items()
                    if k not in _INTERNAL
                ]
                nodes.append({
                    "id":         uri,
                    "label":      name,
                    "title":      title,
                    "color":      "#63b3ed",
                    "size":       20 + min(len(dt_lines) * 2, 20),
                    "properties": col_props,
                })

            # Fetch relationships scoped to this KG only
            for rec in session.run(
                "MATCH (a:KGNode {kg_id: $kg_id})-[r]->(b:KGNode {kg_id: $kg_id}) "
                "RETURN a.uri AS from_uri, b.uri AS to_uri, "
                "type(r) AS rel_type, r.name AS rel_name, r.cardinality AS cardinality, "
                "r.join_columns AS join_columns, r.title AS rel_title",
                kg_id=kg_id,
            ):
                import json as _json
                rel_name = rec["rel_name"] or rec["rel_type"]
                card     = rec["cardinality"] or "M:N"
                jc_raw   = rec["join_columns"] or "[]"
                try:
                    join_columns = _json.loads(jc_raw)
                except Exception:
                    join_columns = []
                edges.append({
                    "from":         rec["from_uri"],
                    "to":           rec["to_uri"],
                    "label":        rel_name,
                    "title":        rec["rel_title"] or f"{rel_name} ({card})",
                    "join_columns": join_columns,
                })

    finally:
        driver.close()

    logger.info("Fetched %d nodes, %d edges from Neo4j (kg_id=%s)", len(nodes), len(edges), kg_id)
    return {"nodes": nodes, "edges": edges}, len(nodes), len(edges)


# ── Gremlin fetch ─────────────────────────────────────────────────────────────

def _fetch_gremlin(config: Any) -> Tuple[Dict, int, int]:
    from gremlin_python.driver import client as gremlin_client  # noqa: PLC0415

    kg_id = getattr(config, "kg_id", "").strip() or "default"
    gc    = gremlin_client.Client(
        config.gremlin_url,
        config.gremlin_traversal_source,
    )
    nodes: List[Dict] = []
    edges: List[Dict] = []

    _INTERNAL = {"uri", "name", "type", "description", "kg_id", "title", "embedding",
                 "T.id", "T.label"}

    try:
        # Fetch only vertices belonging to this KG
        vertices = gc.submit(
            f"g.V().has('kg_id', '{kg_id}').valueMap(true)"
        ).all().result()

        for v in vertices:
            uri   = v.get("uri", [None])[0] or str(v.get("id", ""))
            name  = v.get("name", [uri])[0]

            # Use stored title from embed_node when available
            stored_title_list = v.get("title", [])
            stored_title = stored_title_list[0] if stored_title_list else ""
            if stored_title:
                title    = stored_title
                dt_lines = []
            else:
                desc  = (v.get("description", [None])[0] or "")
                title = f"Class: {name}"
                if desc:
                    title += f"\n{desc}"
                dt_lines = []
                for k, vals in v.items():
                    if str(k) not in _INTERNAL and isinstance(vals, list):
                        for val in vals:
                            dt_lines.append(f"  {k}: {val}")
                if dt_lines:
                    title += "\n\nProperties:\n" + "\n".join(dt_lines)

            # Structured column list for bridge inference engine.
            # Gremlin stores column types as "{col}_xsd_type" vertex properties.
            col_props = [
                {"name": str(k)[:-len("_xsd_type")], "type": vals[0]}
                for k, vals in v.items()
                if str(k).endswith("_xsd_type") and isinstance(vals, list) and vals
            ]
            nodes.append({
                "id":         uri,
                "label":      name,
                "title":      title,
                "color":      "#63b3ed",
                "size":       20 + min(len(dt_lines) * 2, 20),
                "properties": col_props,
            })

        # Fetch edges scoped to this KG
        edge_data = gc.submit(
            f"g.E().has('kg_id', '{kg_id}')"
            ".project('from_uri','to_uri','label','card','join_columns')"
            ".by(outV().values('uri'))"
            ".by(inV().values('uri'))"
            ".by(label())"
            ".by(coalesce(values('cardinality'), constant('M:N')))"
            ".by(coalesce(values('join_columns'), constant('[]')))"
        ).all().result()

        import json as _json
        for e in edge_data:
            try:
                join_columns = _json.loads(e.get("join_columns", "[]"))
            except Exception:
                join_columns = []
            edges.append({
                "from":         e["from_uri"],
                "to":           e["to_uri"],
                "label":        e["label"],
                "title":        f"{e['label']} ({e['card']})",
                "join_columns": join_columns,
            })

    finally:
        gc.close()

    logger.info("Fetched %d nodes, %d edges from Gremlin (kg_id=%s)", len(nodes), len(edges), kg_id)
    return {"nodes": nodes, "edges": edges}, len(nodes), len(edges)


# ── Node ──────────────────────────────────────────────────────────────────────

def fetch_node(state: KGState) -> KGState:
    """Retrieve an existing graph from the connected graph database."""
    logger.info("=== fetch_node (load mode) ===")
    config = state["config"]

    connected = (
        (config.graph_type == "neo4j"   and bool(config.neo4j_uri)) or
        (config.graph_type == "gremlin" and bool(config.gremlin_url))
    )
    if not connected:
        state["errors"].append(
            "fetch_node: no graph database URI provided — cannot load existing graph"
        )
        state["phase"] = "error"
        return state

    try:
        if config.graph_type == "neo4j":
            graph_data, nc, ec = _fetch_neo4j(config)
        else:
            graph_data, nc, ec = _fetch_gremlin(config)

        state["graph_data"]        = graph_data
        state["node_count"]        = nc
        state["edge_count"]        = ec
        state["queries"]           = []
        state["execution_results"] = []
        state["executed_count"]    = 0
        state["phase"]             = "fetched"

    except Exception as exc:
        logger.exception("fetch_node: failed to retrieve graph")
        state["errors"].append(f"fetch_node: {exc}")
        state["phase"] = "error"

    return state
