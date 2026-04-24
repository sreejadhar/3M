"""
Cross-modal linker for the unstructured data intelligence agent.

Detects semantic relationships between:
  - DocNode ↔ DocNode    (topic similarity, citation references)
  - DocNode ↔ KPI/metric (DESCRIBES_KPI — via fuzzy KPI name match)
  - DocNode ↔ Table      (REFERENCES_TABLE — via entity/topic match)

Reuses the 3-strategy fuzzy match algorithm from resolve_node.py's
_fuzzy_match_candidates() so link quality inherits from the NLQ resolver.
"""
from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .store import UnstructuredStore

logger = logging.getLogger(__name__)

_METADATA_API = "http://localhost:8000"   # override via METADATA_API_URL env var

# Confidence thresholds
_KG_WRITE_THRESHOLD   = 0.80   # written to KG immediately
_STORE_THRESHOLD      = 0.60   # stored in SQLite for review, not in KG
_DOC_SIM_THRESHOLD    = 0.85   # cosine similarity for SIMILAR_TOPIC edges
_DOC_SIM_WEAK         = 0.70   # stored as WEAKLY_SIMILAR (not surfaced in UI)


# ── Fuzzy match (same 3-strategy algorithm as resolve_node.py) ────────────────

_STOP = frozenset({"the", "a", "an", "of", "in", "for", "and", "or", "to",
                   "with", "at", "by", "as", "is", "it", "on", "be"})


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"\w+", text) if t.lower() not in _STOP]


def _stem(token: str) -> str:
    for suffix in ("ing", "tion", "tions", "ness", "ment", "ity", "ies", "ed", "er", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def _edit_distance(a: str, b: str) -> int:
    if abs(len(a) - len(b)) > 2:
        return 99
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        new_dp = [i + 1]
        for j, cb in enumerate(b):
            new_dp.append(min(dp[j + 1] + 1, new_dp[j] + 1,
                              dp[j] + (0 if ca == cb else 1)))
        dp = new_dp
    return dp[-1]


def _fuzzy_score(candidate: str, query_tokens: List[str]) -> float:
    cand_tokens = _tokens(candidate)
    if not cand_tokens or not query_tokens:
        return 0.0

    # Strategy 1: substring
    cand_str = " ".join(cand_tokens)
    query_str = " ".join(query_tokens)
    if query_str in cand_str or cand_str in query_str:
        return 1.0

    # Strategy 2: stemmed token overlap
    cand_stems = {_stem(t) for t in cand_tokens}
    q_stems    = {_stem(t) for t in query_tokens}
    overlap = len(cand_stems & q_stems)
    if overlap > 0:
        score = overlap / max(len(cand_stems), len(q_stems))
        if score >= 0.5:
            return 0.7 + score * 0.2

    # Strategy 3: edit distance on long tokens
    for qt in query_tokens:
        if len(qt) < 5:
            continue
        for ct in cand_tokens:
            if len(ct) < 5:
                continue
            if _edit_distance(qt, ct) <= 1:
                return 0.6

    return 0.0


# ── KPI link detection ────────────────────────────────────────────────────────

def detect_kpi_links(asset_id: str, kpis_from_doc: List[str],
                     store: UnstructuredStore,
                     metadata_api: str = _METADATA_API) -> List[Dict]:
    """
    Match KPI names extracted from the document against the Nanite KPI store
    and attribute semantic_roles.
    Returns list of relationship records ready for store.save_relationship().
    """
    if not kpis_from_doc:
        return []

    try:
        resp = httpx.get(f"{metadata_api}/kpis", timeout=5)
        nanite_kpis = resp.json() if resp.is_success else []
    except Exception:
        nanite_kpis = []

    links = []
    for doc_kpi in kpis_from_doc:
        if not doc_kpi or not doc_kpi.strip():
            continue
        query_tokens = _tokens(doc_kpi)
        best_score = 0.0
        best_id    = None
        best_name  = None

        for kpi in nanite_kpis:
            kpi_name = kpi.get("kpi_name", "")
            nl_formula = kpi.get("nl_formula", "")
            score = max(
                _fuzzy_score(kpi_name, query_tokens),
                _fuzzy_score(nl_formula, query_tokens),
            )
            # Exact match shortcut
            if doc_kpi.lower() == kpi_name.lower():
                score = 1.0
            if score > best_score:
                best_score = score
                best_id    = str(kpi.get("kpi_id", ""))
                best_name  = kpi_name

        if best_score >= _STORE_THRESHOLD and best_id:
            links.append({
                "from_asset_id":  asset_id,
                "rel_type":       "DESCRIBES_KPI",
                "confidence":     round(min(best_score, 1.0), 3),
                "basis":          f"kpi_fuzzy:{doc_kpi}→{best_name}",
                "to_nanite_id":   best_id,
                "to_nanite_type": "kpi",
            })

    return links


# ── Table reference detection ─────────────────────────────────────────────────

def detect_table_links(asset_id: str, entities: Dict, topics: List[str],
                       domain: str, store: UnstructuredStore,
                       metadata_api: str = _METADATA_API) -> List[Dict]:
    """
    Match document topics and named entities against Nanite table/entity names.
    """
    try:
        resp = httpx.get(f"{metadata_api}/metadata/entities", timeout=5)
        entities_list = resp.json() if resp.is_success else []
    except Exception:
        entities_list = []

    if not entities_list:
        return []

    # Flatten text to match: orgs + products + topics
    search_terms: List[str] = []
    search_terms.extend(entities.get("organizations", []))
    search_terms.extend(entities.get("products", []))
    search_terms.extend(topics)

    links = []
    seen = set()
    for term in search_terms:
        if not term or not term.strip():
            continue
        query_tokens = _tokens(term)
        for entity in entities_list:
            tbl_name = entity.get("table_name", "")
            description = entity.get("description", "")
            entity_id = str(entity.get("metadata_id", entity.get("entity_id", "")))

            if entity_id in seen:
                continue

            score = max(
                _fuzzy_score(tbl_name, query_tokens),
                _fuzzy_score(description, query_tokens) * 0.8,
            )
            # Exact table name mention: full confidence
            if term.lower() == tbl_name.lower():
                score = 1.0

            if score >= _STORE_THRESHOLD:
                seen.add(entity_id)
                links.append({
                    "from_asset_id":  asset_id,
                    "rel_type":       "REFERENCES_TABLE",
                    "confidence":     round(min(score, 1.0), 3),
                    "basis":          f"entity_fuzzy:{term}→{tbl_name}",
                    "to_nanite_id":   entity_id,
                    "to_nanite_type": "entity",
                })

    return sorted(links, key=lambda x: -x["confidence"])


# ── Doc↔Doc similarity (lightweight cosine on topic sets) ────────────────────

def detect_doc_similarity(asset_id: str, topics: List[str],
                           store: UnstructuredStore) -> List[Dict]:
    """
    Compare topics of the current document against all other documents in the
    same store using Jaccard similarity (no embedding needed for topic sets).
    """
    if not topics:
        return []

    topic_set = {t.lower() for t in topics if t}
    all_assets = store.list_assets(enriched_only=True, limit=500)
    links = []

    for asset in all_assets:
        if asset["asset_id"] == asset_id:
            continue
        other_topics = {t.lower() for t in (asset.get("topics") or []) if t}
        if not other_topics:
            continue
        # Jaccard similarity
        intersection = len(topic_set & other_topics)
        union = len(topic_set | other_topics)
        sim = intersection / union if union else 0.0

        if sim >= _DOC_SIM_WEAK:
            links.append({
                "from_asset_id": asset_id,
                "rel_type":      "SIMILAR_TOPIC" if sim >= _DOC_SIM_THRESHOLD else "WEAKLY_SIMILAR",
                "confidence":    round(sim, 3),
                "basis":         "jaccard_topics",
                "to_asset_id":   asset["asset_id"],
            })

    return sorted(links, key=lambda x: -x["confidence"])[:20]


# ── Main entry point ─────────────────────────────────────────────────────────

def run_linking(asset_id: str, fingerprint: Dict, store: UnstructuredStore,
                metadata_api: str = _METADATA_API) -> int:
    """
    Run all linking strategies for a newly indexed document.
    Saves qualifying relationships to store.
    Returns total number of relationships written.
    """
    entities    = fingerprint.get("named_entities", {})
    topics      = fingerprint.get("topics", [])
    kpis        = entities.get("kpis", [])
    domain      = fingerprint.get("domain", "")
    count       = 0

    # KPI links
    for link in detect_kpi_links(asset_id, kpis, store, metadata_api):
        store.save_relationship(**link)
        count += 1

    # Table links
    for link in detect_table_links(asset_id, entities, topics, domain, store, metadata_api):
        store.save_relationship(**link)
        count += 1

    # Doc↔Doc similarity
    for link in detect_doc_similarity(asset_id, topics, store):
        store.save_relationship(**link)
        count += 1

    logger.info("Linking complete for %s: %d relationships written", asset_id, count)
    return count
