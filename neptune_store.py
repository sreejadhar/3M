"""
AWS Neptune graph backend for the knowledge-graph snapshot store (kg_store.py).

Auth is IAM-based (SigV4), via the same cross-account role assumed for
Bedrock — no username/password (see aws_auth.py). The cluster supports both
Gremlin and openCypher; this module uses openCypher over Neptune's HTTPS Data
API since kg_store's node/edge dict shape maps directly onto Cypher MERGE
statements.

Config:
  NEPTUNE_WRITER_ENDPOINT — openCypher writer endpoint (mutations). Presence
                            of this var is what turns on the Neptune backend
                            in kg_store.py.
  NEPTUNE_READER_ENDPOINT — openCypher reader endpoint (queries); falls back
                            to the writer endpoint if unset.
  NEPTUNE_PORT            — default 8182.
  AWS_REGION              — must match the Neptune cluster's region
                             (default: us-east-1, shared with aws_auth.py).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import requests
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

from aws_auth import AWS_REGION, get_session

_WRITER = os.environ.get(
    "NEPTUNE_WRITER_ENDPOINT",
    "datananite-dev-neptune.cluster-c676y6esoazm.us-east-1.neptune.amazonaws.com",
)
_READER = os.environ.get("NEPTUNE_READER_ENDPOINT", "") or _WRITER
_PORT = int(os.environ.get("NEPTUNE_PORT", "8182"))


def _signed_post(host: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    creds = get_session().get_credentials().get_frozen_credentials()
    url = f"https://{host}:{_PORT}{path}"
    body = json.dumps(payload)
    signer_request = AWSRequest(method="POST", url=url, data=body,
                                 headers={"Content-Type": "application/json"})
    SigV4Auth(creds, "neptune-db", AWS_REGION).add_auth(signer_request)
    resp = requests.post(url, data=body, headers=dict(signer_request.headers), timeout=30)
    resp.raise_for_status()
    return resp.json()


def run_opencypher(query: str, parameters: Dict[str, Any] | None = None, *, write: bool = False) -> List[dict]:
    host = _WRITER if write else _READER
    payload: Dict[str, Any] = {"query": query}
    if parameters:
        payload["parameters"] = json.dumps(parameters)
    result = _signed_post(host, "/openCypher", payload)
    return result.get("results", [])


def save_snapshot(source_id: str, nodes: List[Dict], edges: List[Dict]) -> None:
    """Replace the graph for one source: drop its existing subgraph, then
    write nodes/edges as real Neptune vertices/edges, tagged with source_id
    so multiple sources' snapshots coexist in one shared cluster."""
    run_opencypher(
        "MATCH (n:Node {source_id: $source_id}) DETACH DELETE n",
        {"source_id": source_id}, write=True,
    )
    for n in nodes:
        run_opencypher(
            "MERGE (n:Node {source_id: $source_id, id: $id}) SET n += $props",
            {"source_id": source_id, "id": n.get("id"), "props": n}, write=True,
        )
    for e in edges:
        run_opencypher(
            "MATCH (a:Node {source_id: $source_id, id: $src}), "
            "      (b:Node {source_id: $source_id, id: $dst}) "
            "MERGE (a)-[r:RELATES {source_id: $source_id}]->(b) "
            "SET r += $props",
            {
                "source_id": source_id,
                "src": e.get("source") or e.get("src"),
                "dst": e.get("target") or e.get("dst"),
                "props": e,
            },
            write=True,
        )


def load_snapshot(source_id: str) -> Tuple[List[Dict], List[Dict]]:
    node_rows = run_opencypher(
        "MATCH (n:Node {source_id: $source_id}) RETURN n",
        {"source_id": source_id},
    )
    edge_rows = run_opencypher(
        "MATCH (a:Node {source_id: $source_id})-[r:RELATES {source_id: $source_id}]->(b:Node {source_id: $source_id}) "
        "RETURN r, a.id AS src, b.id AS dst",
        {"source_id": source_id},
    )
    nodes = [row["n"]["~properties"] for row in node_rows]
    edges = [dict(row["r"]["~properties"], source=row["src"], target=row["dst"]) for row in edge_rows]
    return nodes, edges


def delete(source_id: str) -> None:
    run_opencypher(
        "MATCH (n:Node {source_id: $source_id}) DETACH DELETE n",
        {"source_id": source_id}, write=True,
    )
