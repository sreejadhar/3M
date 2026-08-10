"""
Amazon Neptune-backed KG snapshot storage.

Persists the same {nodes, edges} JSON shape that kg_store.py's snapshot
functions have always used — but as an openCypher graph in a managed Neptune
cluster instead of a JSON blob in a relational table — so every downstream
consumer (GraphRAG retrieval, GraphExplorer.jsx, kg_bridges, etc.) is
unaffected by which backend is active. kg_store.py decides whether to route
through here or through SQLite/Postgres; this module only knows Neptune.

Auth: IAM via the same cross-account role used elsewhere in this app
(see aws_auth.py) — SigV4-signed HTTPS requests to Neptune's openCypher HTTP
endpoint, no password. Writes go to the writer endpoint; reads go to the
reader endpoint (falls back to the writer endpoint if no reader is
configured, e.g. single-instance dev clusters).

Config:
  NEPTUNE_WRITER_ENDPOINT — cluster writer endpoint (host only, no scheme/port).
                            Presence of this var is what enables the Neptune
                            backend; empty/unset means "not configured".
  NEPTUNE_READER_ENDPOINT — cluster reader endpoint (default: same as writer)
  NEPTUNE_PORT            — default 8182
  NEPTUNE_REGION          — default us-east-1
  AWS_ROLE_ARN            — cross-account role assumed for Neptune access
                            (see aws_auth.py; shared with other AWS calls)

Public API
----------
enabled()                                    -> bool
save_snapshot(source_id, nodes, edges)       -> None
load_snapshot(source_id)                     -> (nodes, edges)
delete_snapshot(source_id)                   -> None
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

from aws_auth import assume_role

logger = logging.getLogger(__name__)

_WRITER_ENDPOINT = os.environ.get("NEPTUNE_WRITER_ENDPOINT", "").strip()
_READER_ENDPOINT = os.environ.get("NEPTUNE_READER_ENDPOINT", "").strip() or _WRITER_ENDPOINT
_PORT = os.environ.get("NEPTUNE_PORT", "8182")
_REGION = os.environ.get("NEPTUNE_REGION", "us-east-1")

# Neptune's HTTP request-body size limit is generous, but keep batches modest
# so one oversized source doesn't produce one gigantic (and hard to retry)
# request — chunk large node/edge lists across multiple UNWIND queries.
_BATCH_SIZE = 500


def enabled() -> bool:
    return bool(_WRITER_ENDPOINT)


def _chunks(items: List[Any], size: int) -> List[List[Any]]:
    return [items[i:i + size] for i in range(0, len(items), size)] if items else []


def _run_opencypher(endpoint: str, query: str, parameters: Optional[Dict[str, Any]] = None) -> Dict:
    """POST a SigV4-signed openCypher query to a Neptune endpoint."""
    from botocore.awsrequest import AWSRequest
    from botocore.auth import SigV4Auth
    from botocore.credentials import Credentials

    creds = assume_role(session_name="datananite-neptune")
    aws_creds = Credentials(
        access_key=creds["AccessKeyId"],
        secret_key=creds["SecretAccessKey"],
        token=creds["SessionToken"],
    )

    url = f"https://{endpoint}:{_PORT}/openCypher"
    body_fields = {"query": query}
    if parameters is not None:
        body_fields["parameters"] = json.dumps(parameters)
    body = urlencode(body_fields).encode("utf-8")

    signable = AWSRequest(method="POST", url=url, data=body,
                           headers={"Content-Type": "application/x-www-form-urlencoded"})
    SigV4Auth(aws_creds, "neptune-db", _REGION).add_auth(signable)

    req = urllib.request.Request(url, data=body, headers=dict(signable.headers), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        logger.error("neptune_store: openCypher request failed (%s): %s", exc.code, detail)
        raise


# ── Node/edge <-> Neptune property-map conversion ──────────────────────────

def _node_to_props(source_id: str, node: Dict) -> Dict[str, Any]:
    props: Dict[str, Any] = {
        "kg_id": source_id,
        "node_id": str(node.get("id", "")),
        "label": node.get("label", ""),
        "title": node.get("title", ""),
        "color": node.get("color", "") or "",
        "size": float(node.get("size") or 0),
        "properties_json": json.dumps(node.get("properties") or {}),
    }
    embedding = node.get("embedding")
    if embedding:
        props["embedding"] = [float(x) for x in embedding]
    return props


def _props_to_node(props: Dict) -> Dict:
    node: Dict[str, Any] = {
        "id": props.get("node_id", ""),
        "label": props.get("label", ""),
        "title": props.get("title", ""),
    }
    if props.get("color"):
        node["color"] = props["color"]
    if props.get("size"):
        node["size"] = props["size"]
    try:
        node["properties"] = json.loads(props.get("properties_json") or "{}")
    except Exception:
        node["properties"] = {}
    if props.get("embedding"):
        node["embedding"] = props["embedding"]
    return node


def _edge_to_params(source_id: str, edge: Dict) -> Dict[str, Any]:
    return {
        "from_id": str(edge.get("from", "")),
        "to_id": str(edge.get("to", "")),
        "props": {
            "kg_id": source_id,
            "label": edge.get("label", ""),
            "title": edge.get("title", ""),
            "join_columns_json": json.dumps(edge.get("join_columns") or []),
        },
    }


# ── Public API ───────────────────────────────────────────────────────────────

def save_snapshot(source_id: str, nodes: List[Dict], edges: List[Dict]) -> None:
    """Replace this source's whole subgraph in Neptune — detach-delete the
    old one then recreate from scratch, mirroring kg_store.py's
    upsert-the-whole-blob semantics (simplest correct behavior given each
    KG build recomputes nodes/edges from that source's current ontology)."""
    if not enabled():
        return

    _run_opencypher(
        _WRITER_ENDPOINT,
        "MATCH (n:KGNode {kg_id: $kg_id}) DETACH DELETE n",
        {"kg_id": source_id},
    )

    for batch in _chunks([_node_to_props(source_id, n) for n in (nodes or [])], _BATCH_SIZE):
        _run_opencypher(
            _WRITER_ENDPOINT,
            "UNWIND $nodes AS props CREATE (n:KGNode) SET n = props",
            {"nodes": batch},
        )

    for batch in _chunks([_edge_to_params(source_id, e) for e in (edges or [])], _BATCH_SIZE):
        _run_opencypher(
            _WRITER_ENDPOINT,
            "UNWIND $edges AS edge "
            "MATCH (a:KGNode {kg_id: $kg_id, node_id: edge.from_id}), "
            "      (b:KGNode {kg_id: $kg_id, node_id: edge.to_id}) "
            "CREATE (a)-[e:KG_EDGE]->(b) SET e = edge.props",
            {"kg_id": source_id, "edges": batch},
        )

    logger.info("neptune_store.save_snapshot: %s (%d nodes, %d edges)",
                source_id[:8], len(nodes or []), len(edges or []))


def load_snapshot(source_id: str) -> Tuple[List[Dict], List[Dict]]:
    """Read back the KG snapshot (nodes/edges) for source_id from Neptune.
    Returns ([], []) if Neptune is not configured or nothing is stored."""
    if not enabled():
        return [], []

    node_result = _run_opencypher(
        _READER_ENDPOINT,
        "MATCH (n:KGNode {kg_id: $kg_id}) RETURN n",
        {"kg_id": source_id},
    )
    nodes = [_props_to_node(row["n"]) for row in node_result.get("results", [])]

    edge_result = _run_opencypher(
        _READER_ENDPOINT,
        "MATCH (a:KGNode {kg_id: $kg_id})-[e:KG_EDGE]->(b:KGNode {kg_id: $kg_id}) "
        "RETURN a.node_id AS from_id, b.node_id AS to_id, e",
        {"kg_id": source_id},
    )
    edges = []
    for row in edge_result.get("results", []):
        e = row["e"]
        try:
            join_columns = json.loads(e.get("join_columns_json") or "[]")
        except Exception:
            join_columns = []
        edges.append({
            "from": row["from_id"],
            "to": row["to_id"],
            "label": e.get("label", ""),
            "title": e.get("title", ""),
            "join_columns": join_columns,
        })

    return nodes, edges


def delete_snapshot(source_id: str) -> None:
    """Remove a source's whole subgraph from Neptune."""
    if not enabled():
        return
    _run_opencypher(
        _WRITER_ENDPOINT,
        "MATCH (n:KGNode {kg_id: $kg_id}) DETACH DELETE n",
        {"kg_id": source_id},
    )
    logger.info("neptune_store.delete_snapshot: %s", source_id[:8])
