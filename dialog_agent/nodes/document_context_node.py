"""
document_context_node — pulls in Document Intelligence content linked to
whichever tables retrieve_node selected as relevant to the current query.

Runs immediately after retrieve_node, using state["doc_mention_edges"] (the
KG's document "mentions" edges, set aside by retrieve_node before it
stripped document nodes out of the table subgraph — see retrieve_node.py).

For each selected table, finds documents linked to it, fetches a short
PII-redacted excerpt from the unstructured-api service, and stores the
result in state["document_context"] for synthesize_node to fold into its
answer. Best-effort throughout: a source with no linked documents, or an
unreachable unstructured-api, is a no-op — never blocks or errors out an
otherwise-normal structured query.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List

import httpx

from ..state import DialogState

logger = logging.getLogger(__name__)

UNSTRUCTURED_API_URL = os.environ.get("UNSTRUCTURED_API_URL", "http://localhost:8008")

# How many distinct documents to pull context from per query — kept small so
# a heavily-linked source doesn't bloat the synthesize prompt with excerpts
# that mostly repeat the same subject matter. Ties at the cutoff are all
# included rather than arbitrarily truncated — see the ranking note below.
_MAX_DOCUMENTS = 3
_FETCH_TIMEOUT = 8.0


def document_context_node(state: DialogState) -> DialogState:
    logger.info("=== document_context_node ===")

    mention_edges: List[Dict] = state.get("doc_mention_edges") or []
    selected_ids = {n.get("id") for n in (state.get("kg_nodes") or [])}
    table_scores: Dict[str, float] = state.get("table_relevance_scores") or {}

    if not mention_edges or not selected_ids:
        return state

    # Rank each linked document by the query-relevance of its BEST-matching
    # selected table (falling back to hit count when no scores are
    # available, e.g. retrieve_node took an early-exit path). Ranking by
    # hit count alone is not enough: it's common for several documents to
    # each touch exactly one selected table, tying at 1 — in that case a
    # plain top-N cut is arbitrary (picks whichever happened to iterate
    # first) and can silently drop the single most relevant document for
    # the question. Scoring by the table's own relevance to the NLQ (which
    # retrieve_node already computed) breaks that tie meaningfully instead.
    asset_table_hits: Dict[str, List[str]] = {}
    for edge in mention_edges:
        table_id = edge.get("to")
        if table_id not in selected_ids:
            continue
        from_id = edge.get("from", "")
        asset_id = from_id.split("doc:", 1)[-1] if from_id.startswith("doc:") else ""
        if not asset_id:
            continue
        asset_table_hits.setdefault(asset_id, []).append(table_id)

    if not asset_table_hits:
        logger.info("document_context_node: no documents linked to the selected tables")
        return state

    def rank_key(asset_id: str) -> tuple:
        tables = asset_table_hits[asset_id]
        best_score = max((table_scores.get(t, 0.0) for t in tables), default=0.0)
        return (best_score, len(tables))

    ranked = sorted(asset_table_hits, key=rank_key, reverse=True)

    # Ties at the cutoff are all kept — truncating mid-tie would just
    # reintroduce the same arbitrary-exclusion problem this ranking exists
    # to avoid, only one document later instead of five.
    if len(ranked) > _MAX_DOCUMENTS:
        cutoff_score = rank_key(ranked[_MAX_DOCUMENTS - 1])
        ranked_asset_ids = [aid for aid in ranked if rank_key(aid) >= cutoff_score]
    else:
        ranked_asset_ids = ranked

    document_context = []
    with httpx.Client(timeout=_FETCH_TIMEOUT) as client:
        for asset_id in ranked_asset_ids:
            try:
                resp = client.get(f"{UNSTRUCTURED_API_URL}/assets/{asset_id}/excerpt")
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.warning("document_context_node: could not fetch excerpt for asset %s — %s",
                                asset_id, exc)
                continue
            document_context.append({
                "file_name": data.get("file_name"),
                "excerpt": data.get("excerpt", ""),
                "topics": data.get("topics") or [],
                "matched_tables": sorted(set(asset_table_hits[asset_id])),
            })

    if document_context:
        logger.info("document_context_node: attached %d document(s): %s",
                     len(document_context), [d["file_name"] for d in document_context])
    state["document_context"] = document_context
    return state
