"""
Cross-modal query engine for the unstructured data intelligence agent.

Given a natural-language question and a list of KPI names from the structured
answer, retrieves the most relevant document fingerprints and formats them as
a "Supporting Context" block ready to be appended to the structured insight.

No LLM call is made here — purely retrieval + formatting.
Cost: zero. Latency: <100ms (SQLite FTS + key-value lookups).
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

from .store import UnstructuredStore

logger = logging.getLogger(__name__)

_MAX_DOCS     = 4    # max documents to surface per answer
_MIN_CONF     = 0.60 # minimum link confidence to include a KPI link


def query_for_context(
    question: str,
    kpi_names: List[str],
    store: UnstructuredStore,
    limit: int = _MAX_DOCS,
    base_url: str = "",
) -> Optional[str]:
    """
    Returns a formatted markdown block of supporting document context,
    or None if no relevant documents are found.

    Two retrieval paths run in parallel and are merged by relevance:
      Path A — FTS search: question text → matching doc fingerprints
      Path B — KPI graph: kpi_names → DESCRIBES_KPI edges → linked docs
    """
    candidates: Dict[str, Dict] = {}   # asset_id → {asset, score, kpi_links}

    # ── Path A: FTS search on the question ───────────────────────────────────
    if question.strip():
        try:
            fts_hits = store.search_assets(_fts_query(question), limit=limit * 2)
            for asset in fts_hits:
                aid = asset["asset_id"]
                candidates[aid] = {"asset": asset, "score": 1.0, "kpi_links": [],
                                   "change_summary": _latest_change_summary(asset, store)}
        except Exception as exc:
            logger.debug("FTS search failed: %s", exc)

    # ── Path B: KPI graph lookup ──────────────────────────────────────────────
    if kpi_names:
        # Get all DESCRIBES_KPI relationships, find those matching our KPIs
        try:
            kpi_lower = {k.lower() for k in kpi_names if k}
            # Search docs that have DESCRIBES_KPI links mentioning these KPIs
            kpi_docs = _find_kpi_linked_docs(kpi_names, store)
            for asset_id, links in kpi_docs.items():
                asset = store.get_asset(asset_id)
                if not asset or asset.get("deleted"):
                    continue
                if asset_id in candidates:
                    candidates[asset_id]["kpi_links"].extend(links)
                    # Boost score for docs found by both paths
                    candidates[asset_id]["score"] = min(
                        candidates[asset_id]["score"] * 1.3, 1.0
                    )
                else:
                    candidates[asset_id] = {
                        "asset": asset,
                        "score": max(l["confidence"] for l in links),
                        "kpi_links": links,
                        "change_summary": _latest_change_summary(asset, store),
                    }
        except Exception as exc:
            logger.debug("KPI graph lookup failed: %s", exc)

    if not candidates:
        return None

    # ── Rank and select top-K ─────────────────────────────────────────────────
    ranked = sorted(candidates.values(), key=lambda x: -x["score"])[:limit]

    # ── Filter: must have a summary to be useful ──────────────────────────────
    ranked = [c for c in ranked if c["asset"].get("summary") or c["asset"].get("topics")]
    if not ranked:
        return None

    return _format_context(ranked, base_url=base_url)


def _fts_query(question: str) -> str:
    """Convert a natural language question to an FTS5 query — pick key terms."""
    import re
    # Strip stop words and question words, take first 6 meaningful tokens
    stop = {"what", "how", "why", "when", "where", "who", "is", "are", "was",
            "the", "a", "an", "of", "in", "to", "and", "or", "for", "do",
            "does", "did", "can", "could", "would", "should", "tell", "me",
            "show", "give", "find", "about", "with", "from", "this", "that"}
    tokens = [t.lower() for t in re.findall(r"\w+", question)
              if t.lower() not in stop and len(t) > 2]
    if not tokens:
        return question
    # Use OR to cast a wider net
    return " OR ".join(tokens[:6])


def _find_kpi_linked_docs(kpi_names: List[str],
                           store: UnstructuredStore) -> Dict[str, List[Dict]]:
    """
    Find documents with DESCRIBES_KPI relationships matching any of kpi_names.
    Returns {asset_id: [relationship_dict, ...]}
    """
    import re
    kpi_lower = [k.lower() for k in kpi_names if k]
    result: Dict[str, List[Dict]] = {}

    # Fetch all assets with enriched fingerprints — check their KPI links
    assets = store.list_assets(enriched_only=True, limit=500)
    for asset in assets:
        asset_id = asset["asset_id"]
        rels = store.get_relationships(asset_id)
        matching = []
        for rel in rels:
            if rel["rel_type"] != "DESCRIBES_KPI":
                continue
            if rel["confidence"] < _MIN_CONF:
                continue
            basis = rel.get("basis", "").lower()
            # basis format: "kpi_fuzzy:doc_kpi→nanite_kpi"
            if any(k in basis for k in kpi_lower):
                matching.append(rel)
        if matching:
            result[asset_id] = matching

    return result


def _latest_change_summary(asset: Dict, store: UnstructuredStore) -> str:
    """
    Return the change_summary from the most recent version snapshot,
    or "" if this is the first version or no summary exists yet.
    """
    version_num = asset.get("version_num") or 1
    if version_num <= 1:
        return ""
    try:
        versions = store.list_asset_versions(asset["asset_id"])
        # versions are newest-first; the most recent snapshot is the previous version
        if versions:
            return versions[0].get("change_summary", "") or ""
    except Exception:
        pass
    return ""


def _format_context(ranked: List[Dict], base_url: str = "") -> str:
    """
    Format the ranked document context as a clean markdown block.
    The output is appended to the structured insight by synthesize_node.
    """
    lines = []

    for item in ranked:
        asset      = item["asset"]
        kpi_links  = item["kpi_links"]

        title    = asset.get("title") or asset.get("file_name", "Untitled")
        doc_type = asset.get("doc_type", "")
        domain   = asset.get("domain", "")
        summary  = asset.get("summary", "").strip()
        topics   = asset.get("topics") or []
        time_refs = asset.get("time_references") or []
        ents     = asset.get("named_entities") or {}
        sensitivity = asset.get("sensitivity", "internal")
        version_num = asset.get("version_num") or 1
        change_summary = item.get("change_summary", "")

        # Meta line — includes version badge when >1
        meta_parts = [p for p in [doc_type, domain] if p]
        if version_num > 1:
            meta_parts.append(f"v{version_num}")
        meta = f" *({', '.join(meta_parts)})*" if meta_parts else ""

        lines.append(f"**{title}**{meta}")

        if summary:
            lines.append(summary)

        # Change callout for documents that have been updated
        if version_num > 1 and change_summary:
            lines.append(f"*Latest update:* {change_summary}")

        # KPIs discussed in this document that match the structured answer
        if kpi_links:
            linked_kpis = []
            seen = set()
            for rel in kpi_links:
                basis = rel.get("basis", "")
                # Extract the nanite-side KPI name from "kpi_fuzzy:doc_kpi→nanite_kpi"
                if "→" in basis:
                    nanite_kpi = basis.split("→")[-1].strip()
                    if nanite_kpi not in seen:
                        linked_kpis.append(nanite_kpi)
                        seen.add(nanite_kpi)
            if linked_kpis:
                lines.append(f"*Discusses:* {', '.join(linked_kpis)}")

        # Time references ground the document temporally
        if time_refs:
            lines.append(f"*Period:* {', '.join(str(t) for t in time_refs[:4])}")

        # Sensitivity watermark — important for governance
        if sensitivity in ("confidential", "restricted"):
            lines.append(f"*⚠ {sensitivity.upper()}*")

        if base_url:
            asset_id = asset.get("asset_id", "")
            if asset_id:
                lines.append(f"[↓ Download]({base_url}/assets/{asset_id}/download)")
                if version_num > 1:
                    lines.append(f"[Version history]({base_url}/assets/{asset_id}/versions)")

        lines.append("")  # blank line between docs

    return "\n".join(lines).strip()
