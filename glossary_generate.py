"""
Orchestrates on-demand, per-source business glossary generation:
NLP normalization -> canonical-term match -> cross-source structural/semantic
match -> LLM fallback -> governed persistence (glossary_registry).

NOT part of the indexing pipeline — only invoked by
POST /metadata/sources/{source_id}/generate-glossary (orchestrator_api.py).

Confidence -> governance status mapping (per the approved plan):
  >= 0.9  -> approved   (canonical match, or a very strong LLM/structural hit)
  0.6-0.9 -> candidate  (needs steward review before being treated as authoritative)
  <  0.6  -> draft      (weak signal, flagged for review)
  manual edit / steward approval -> approved, confidence=1.0 (handled by the API layer,
  not here — this module only ever CREATES candidate/draft/canonical terms)
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional

import metadata_catalog as _mc
import glossary_nlp as _nlp
import glossary_matcher as _matcher
import glossary_registry as _reg

logger = logging.getLogger(__name__)

_GLOSSARY_SYSTEM = """\
You are a business analyst annotating a data catalog. Given a database table name
and its columns (with data types and sample values), propose a business-friendly
GLOSSARY TERM and a one-sentence DEFINITION for the table itself and for each
column — the plain-English name and meaning a business user (not an engineer)
would recognize, not a restatement of the technical name or data type.

You will usually be given a BUSINESS and/or DOMAIN context line describing what
industry/function this data actually belongs to. Ground every interpretation in
that context. Many column/table name fragments and abbreviations are ambiguous
in general English (e.g. "CBI" can mean "Citizenship by Investment" on the open
web, or something entirely different inside a pharma/life-sciences dataset).
When a fragment or abbreviation could plausibly mean something else outside the
given business/domain, DO NOT default to the generic/most-common web meaning —
prefer the interpretation consistent with the stated business/domain, and lower
your confidence score instead of guessing. If no business/domain context is
given, treat any ambiguous abbreviation as low-confidence rather than picking
the first meaning that comes to mind.

Return ONLY a JSON object — no prose, no markdown fences.

For each item also return a "confidence" float between 0.0 and 1.0 reflecting how
confident you are that the term/definition is correct given the available
evidence (column name + sample values + business/domain context). Use lower
confidence when the column name is abbreviated, ambiguous, the sample values
don't clearly confirm your interpretation, or you had to guess without
business/domain context to anchor the meaning.

OUTPUT FORMAT:
{
  "entity": {"term": "<business name for the table>", "definition": "<one sentence>", "confidence": 0.0},
  "columns": [
    {"name": "<column_name>", "term": "<business term>", "definition": "<one sentence>", "confidence": 0.0}
  ]
}
"""


def _call_glossary_llm(table_name: str, col_specs: List[Dict], model: str,
                        domain: str = "", business: str = "") -> Dict[str, Any]:
    """Same call+parse pattern already proven in metadata_catalog.py's
    _call_enrich_llm: from llm_client import get_client, markdown-fence strip,
    brace-extraction JSON fallback, fail-soft on any error."""
    lines = []
    for c in col_specs:
        sv = c.get("sample_values") or []
        sv_str = ", ".join(repr(v) for v in sv[:20]) if sv else "(no samples)"
        lines.append(f"  {c['name']} ({c.get('data_type', '')}): samples=[{sv_str}]")

    context_lines = []
    if business:
        context_lines.append(f"Business: {business}")
    if domain:
        context_lines.append(f"Domain: {domain}")
    context_prefix = ("\n".join(context_lines) + "\n") if context_lines else ""
    user_msg = (
        f"{context_prefix}Table: {table_name}\nColumns:\n"
        + "\n".join(lines)
        + "\n\nPropose a business glossary term and definition for the table and each column."
    )

    try:
        from llm_client import get_client
        client = get_client()
        msg = client.messages.create(
            model=model, max_tokens=2048, temperature=0.0,
            system=_GLOSSARY_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = msg.content[0].text if msg.content else ""
    except Exception as exc:
        logger.warning("generate_glossary: LLM call failed for table %r — %s", table_name, exc)
        return {}

    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        s, e = cleaned.find("{"), cleaned.rfind("}")
        if s != -1 and e != -1:
            try:
                return json.loads(cleaned[s:e + 1])
            except json.JSONDecodeError:
                pass
    logger.warning("generate_glossary: could not parse LLM JSON for table %r", table_name)
    return {}


def _get_embedder():
    """Lazy, best-effort import of the existing embedding backend — degrades
    to name-matching-only (no semantic tier) if dialog_agent/sentence-
    transformers can't load, same graceful-degradation convention already
    used throughout this codebase (e.g. dialog_agent/kg_router.py itself)."""
    try:
        from dialog_agent.kg_router import _embed, _cosine
        return _embed, _cosine
    except Exception as exc:
        logger.info("generate_glossary: embedding backend unavailable (%s) — name-match + LLM only", exc)
        return None, None


def _status_for_confidence(confidence: float) -> str:
    if confidence >= 0.9:
        return "approved"
    if confidence >= 0.6:
        return "candidate"
    return "draft"


def _hydrate_pool(source_id: str) -> List[Dict]:
    """Cross-source candidate pool: every column already linked to a term in
    OTHER sources, augmented with that column's data_type (single lookup per
    distinct attr_id) so type-compatibility scoring has real signal."""
    raw = _reg.list_all_linked_columns(exclude_source_id=source_id)
    pool: List[Dict] = []
    attr_cache: Dict[str, str] = {}
    for cand in raw:
        dt = ""
        attr_id = cand.get("attr_id") or ""
        if attr_id:
            if attr_id not in attr_cache:
                a = _mc.get_attribute(attr_id)
                attr_cache[attr_id] = (a or {}).get("data_type", "")
            dt = attr_cache[attr_id]
        pool.append({**cand, "normalized_phrase": cand.get("canonical_key", ""), "data_type": dt})
    return pool


def generate_glossary_for_source(
    source_id: str, model: Optional[str] = None, domain: str = "", business: str = "",
    progress_cb: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, int]:
    """
    Runs the full discovery pipeline for every entity in source_id.
    progress_cb(stage, message) is called at each stage boundary for SSE
    progress reporting — optional, no-op if omitted.

    Returns {"columns_processed", "terms_created", "terms_linked", "needs_review"}.
    """
    def progress(stage: str, message: str) -> None:
        if progress_cb:
            try:
                progress_cb(stage, message)
            except Exception:
                pass

    model = model or os.environ.get("DIALOG_LLM_MODEL", "claude-haiku-4-5")
    embed_fn, cosine_fn = _get_embedder()
    stats = {"columns_processed": 0, "terms_created": 0, "terms_linked": 0, "needs_review": 0,
              "skipped_existing": 0}

    entities = _mc.list_entities(source_id)
    progress("normalize", f"Normalizing columns for {len(entities)} table(s)…")
    pool = _hydrate_pool(source_id)
    progress("cross_source", f"Comparing against {len(pool)} already-glossaried column(s) from other sources…")

    for ent_summary in entities:
        full = _mc.get_entity(ent_summary["metadata_id"])
        if not full:
            continue
        metadata_id = full["metadata_id"]
        table_name = full["table_name"]
        attrs = full.get("attributes", [])

        items: List[Dict] = [{
            "attr_id": "", "column_name": table_name, "data_type": "",
            "description": full.get("description", ""), "sample_values": [],
        }]
        items += [{
            "attr_id": a["attr_id"], "column_name": a["column_name"],
            "data_type": a.get("data_type", ""), "description": a.get("description", ""),
            "sample_values": a.get("top_values") or a.get("sample_values") or [],
        } for a in attrs]

        need_llm: List[Dict] = []
        for item in items:
            existing_link = _reg.get_asset_link(metadata_id, item["attr_id"])
            if existing_link:
                # Already governed by a prior run (or a manual edit) — re-running discovery
                # must never re-score/relink it. Without this guard, a column that got an
                # llm_generated term on run 1 becomes a match for that same term's
                # canonical_key on run 2, and link_asset() unconditionally overwrites
                # confidence/match_method — silently laundering a low-confidence guess into
                # a 0.95 "canonical" match with no review.
                stats["skipped_existing"] += 1
                continue

            stats["columns_processed"] += 1
            normalized = _nlp.normalize_identifier(item["column_name"])

            canon = _reg.find_term_by_canonical_key(normalized, domain=domain)
            if canon:
                _reg.link_asset(canon["term_id"], source_id, metadata_id, item["attr_id"],
                                 confidence=0.95, match_method="canonical")
                stats["terms_linked"] += 1
                continue

            col_spec = {
                "name": item["column_name"], "data_type": item["data_type"],
                "description": item["description"], "normalized_phrase": normalized,
            }
            match = _matcher.find_best_match(col_spec, pool, embed_fn=embed_fn, cosine_fn=cosine_fn)
            if match:
                _reg.link_asset(match["term_id"], source_id, metadata_id, item["attr_id"],
                                 confidence=match["match_confidence"], match_method="cross_source")
                stats["terms_linked"] += 1
                if match["match_confidence"] < 0.9:
                    stats["needs_review"] += 1
                continue

            need_llm.append({**item, "normalized_phrase": normalized})

        if need_llm:
            progress("llm_generate", f"Generating terms via LLM for {len(need_llm)} item(s) in {table_name}…")
            col_specs_for_llm = [
                {"name": i["column_name"], "data_type": i["data_type"], "sample_values": i["sample_values"]}
                for i in need_llm if i["attr_id"]
            ]
            entity_needs_llm = any(i["attr_id"] == "" for i in need_llm)
            llm_result = _call_glossary_llm(table_name, col_specs_for_llm, model, domain=domain, business=business) \
                if (col_specs_for_llm or entity_needs_llm) else {}

            if entity_needs_llm:
                e_ann = llm_result.get("entity") or {}
                if e_ann.get("term"):
                    conf = min(max(float(e_ann.get("confidence", 0.5) or 0.5), 0.0), 0.9)
                    status = _status_for_confidence(conf)
                    entity_item = next(i for i in need_llm if i["attr_id"] == "")
                    term = _reg.create_term(
                        e_ann["term"], canonical_key=entity_item["normalized_phrase"],
                        definition=e_ann.get("definition", ""), domain=domain,
                        status=status, confidence=conf, match_method="llm_generated",
                    )
                    _reg.link_asset(term["term_id"], source_id, metadata_id, "",
                                     confidence=conf, match_method="llm_generated")
                    stats["terms_created"] += 1
                    if status != "approved":
                        stats["needs_review"] += 1

            col_ann_by_name = {c["name"]: c for c in (llm_result.get("columns") or []) if "name" in c}
            for item in need_llm:
                if not item["attr_id"]:
                    continue
                ann = col_ann_by_name.get(item["column_name"])
                if not ann or not ann.get("term"):
                    continue
                conf = min(max(float(ann.get("confidence", 0.5) or 0.5), 0.0), 0.9)
                status = _status_for_confidence(conf)
                term = _reg.create_term(
                    ann["term"], canonical_key=item["normalized_phrase"],
                    definition=ann.get("definition", ""), domain=domain,
                    status=status, confidence=conf, match_method="llm_generated",
                )
                _reg.link_asset(term["term_id"], source_id, metadata_id, item["attr_id"],
                                 confidence=conf, match_method="llm_generated")
                stats["terms_created"] += 1
                if status != "approved":
                    stats["needs_review"] += 1

    progress("done", f"{stats['terms_created']} term(s) created, {stats['terms_linked']} column(s) linked "
                      f"to existing terms, {stats['needs_review']} need review, "
                      f"{stats['skipped_existing']} already governed (skipped)")
    logger.info("generate_glossary_for_source: source=%s %s", source_id, stats)
    return stats
