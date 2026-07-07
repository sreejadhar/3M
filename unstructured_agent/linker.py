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

from .embedder import cosine_similarity
from .store import UnstructuredStore

logger = logging.getLogger(__name__)

import time

_METADATA_API = "http://localhost:8000"   # override via METADATA_API_URL env var

# Confidence thresholds
_KG_WRITE_THRESHOLD   = 0.80   # written to KG immediately
_STORE_THRESHOLD      = 0.60   # stored in SQLite for review, not in KG
_DOC_SIM_THRESHOLD    = 0.85   # cosine similarity for SIMILAR_TOPIC edges
_DOC_SIM_WEAK         = 0.70   # stored as WEAKLY_SIMILAR (not surfaced in UI)

# Metric semantic roles that qualify as KPI candidates when KPI store is empty
_METRIC_ROLES = frozenset({
    "measure", "metric", "kpi", "measure_calculated",
    "calculated_measure", "fact", "amount",
})

# Module-level cache: avoids N+1 attribute fetches on every document in a run
_attr_cache: Dict[str, Any] = {}   # key: metadata_api url → {ts, kpis}
_CACHE_TTL = 300                   # seconds — one indexing run shares one fetch


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

def _fetch_kpi_candidates(metadata_api: str) -> List[Dict]:
    """
    Build a unified KPI candidate list from two sources:
      1. KPI store  (GET /kpis)              — explicitly defined business KPIs
      2. Metric attributes (GET /metadata/entities + detail) — columns whose
         semantic_role is measure/metric/kpi, inferred by taxonomy enrichment

    Result is cached for _CACHE_TTL seconds so all documents in one indexing
    run share a single round of API calls (avoids N+1 per document).

    Each candidate:
      {"kpi_id": str, "kpi_name": str, "nl_formula": str, "source": "kpi_store"|"attribute"}
    """
    cached = _attr_cache.get(metadata_api)
    if cached and (time.time() - cached["ts"]) < _CACHE_TTL:
        return cached["kpis"]

    candidates: List[Dict] = []

    # Source 1 — KPI store
    try:
        resp = httpx.get(f"{metadata_api}/kpis", timeout=5)
        if resp.is_success:
            for k in (resp.json() or []):
                candidates.append({
                    "kpi_id":    str(k.get("kpi_id", "")),
                    "kpi_name":  k.get("kpi_name", ""),
                    "nl_formula": k.get("nl_formula", "") or k.get("description", ""),
                    "source":    "kpi_store",
                })
    except Exception as exc:
        logger.debug("KPI store fetch failed: %s", exc)

    # Source 2 — metric attributes from catalog (fallback when KPI store empty,
    # or always — both sources are merged and deduplicated)
    try:
        ent_resp = httpx.get(f"{metadata_api}/metadata/entities", timeout=5)
        entities = ent_resp.json() if ent_resp.is_success else []
        for entity in entities:
            eid = entity.get("metadata_id") or entity.get("entity_id", "")
            tbl = entity.get("table_name", "")
            try:
                detail = httpx.get(
                    f"{metadata_api}/metadata/entities/{eid}", timeout=5
                ).json()
                for attr in (detail.get("attributes") or []):
                    role = (attr.get("semantic_role") or "").lower()
                    if role not in _METRIC_ROLES:
                        continue
                    col   = attr.get("column_name", "")
                    desc  = attr.get("description", "")
                    aid   = str(attr.get("attr_id", f"{eid}:{col}"))
                    # Avoid duplicating what KPI store already has
                    if any(c["kpi_name"].lower() == col.lower()
                           for c in candidates):
                        continue
                    candidates.append({
                        "kpi_id":    aid,
                        "kpi_name":  col,
                        "nl_formula": desc or f"{col} from {tbl}",
                        "source":    "attribute",
                    })
            except Exception:
                continue
    except Exception as exc:
        logger.debug("Metric attribute fetch failed: %s", exc)

    _attr_cache[metadata_api] = {"ts": time.time(), "kpis": candidates}
    logger.info("linker: %d KPI candidates loaded (%d from kpi_store, %d from attributes)",
                len(candidates),
                sum(1 for c in candidates if c["source"] == "kpi_store"),
                sum(1 for c in candidates if c["source"] == "attribute"))
    return candidates


def detect_kpi_links(asset_id: str, kpis_from_doc: List[str],
                     store: UnstructuredStore,
                     metadata_api: str = _METADATA_API) -> List[Dict]:
    """
    Match KPI names extracted from the document against:
      - Nanite KPI store (explicitly defined business KPIs)
      - Metric-role attributes from the metadata catalog (always available
        after taxonomy enrichment, even when KPI store is empty)

    Returns list of relationship records ready for store.save_relationship().
    """
    if not kpis_from_doc:
        return []

    nanite_kpis = _fetch_kpi_candidates(metadata_api)
    if not nanite_kpis:
        logger.debug("detect_kpi_links: no KPI candidates available — skipping")
        return []

    links = []
    for doc_kpi in kpis_from_doc:
        if not doc_kpi or not doc_kpi.strip():
            continue
        query_tokens = _tokens(doc_kpi)
        best_score = 0.0
        best_id    = None
        best_name  = None

        for kpi in nanite_kpis:
            kpi_name   = kpi.get("kpi_name", "")
            nl_formula = kpi.get("nl_formula", "")
            score = max(
                _fuzzy_score(kpi_name, query_tokens),
                _fuzzy_score(nl_formula, query_tokens),
            )
            if doc_kpi.lower() == kpi_name.lower():
                score = 1.0
            if score > best_score:
                best_score = score
                best_id    = kpi["kpi_id"]
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


# ── Doc↔Doc similarity (semantic embedding, falls back to topic Jaccard) ─────

def detect_doc_similarity(asset_id: str, topics: List[str],
                           store: UnstructuredStore) -> List[Dict]:
    """
    Compare the current document against all other documents in the same
    store. Prefers cosine similarity over semantic embeddings when both
    documents have one; falls back to Jaccard similarity over topic sets
    for documents with no embedding (e.g. no embedding backend installed).
    """
    own_embedding = store.get_embedding(asset_id)
    topic_set = {t.lower() for t in topics if t}

    if not own_embedding and not topic_set:
        return []

    all_assets = store.list_assets(enriched_only=True, limit=500)
    other_embeddings = {
        e["asset_id"]: e["embedding"] for e in store.list_embeddings(exclude_asset_id=asset_id)
    } if own_embedding else {}

    links = []
    for asset in all_assets:
        if asset["asset_id"] == asset_id:
            continue

        other_embedding = other_embeddings.get(asset["asset_id"])
        if own_embedding and other_embedding:
            sim = cosine_similarity(own_embedding["embedding"], other_embedding)
            basis = f"embedding_cosine:{own_embedding['model']}"
        else:
            other_topics = {t.lower() for t in (asset.get("topics") or []) if t}
            if not topic_set or not other_topics:
                continue
            intersection = len(topic_set & other_topics)
            union = len(topic_set | other_topics)
            sim = intersection / union if union else 0.0
            basis = "jaccard_topics"

        if sim >= _DOC_SIM_WEAK:
            links.append({
                "from_asset_id": asset_id,
                "rel_type":      "SIMILAR_TOPIC" if sim >= _DOC_SIM_THRESHOLD else "WEAKLY_SIMILAR",
                "confidence":    round(sim, 3),
                "basis":         basis,
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
