"""
Cross-source column matching for business glossary discovery.

Pure scoring functions (no DB access — column specs are dicts passed in by
the orchestration layer) so they're directly unit-testable, per the plan's
verification section. Adapts the reusable signals from
dialog_agent/kg_inference_engine.py (type-family compatibility, tiered
base+boost+cap confidence convention) but deliberately DROPS the PK/FK/
uniqueness boosts that engine bakes in for join-key inference — those boosts
suppress matches between two non-key synonym columns (e.g. two free-text
"customer_name" columns), which is exactly the case a glossary needs to
catch. Embedding similarity reuses dialog_agent.kg_router._embed()/_cosine()
directly (the model-loading/fallback logic there is already correct and
stateful — no reason to duplicate it).

Confidence bands (mirrors the plan's governance mapping):
  normalized-name exact match  : 0.80-0.90 -> "candidate" on creation
  semantic similarity match    : 0.55-0.85 -> "candidate" (needs steward review)
  no match found                : caller falls back to LLM generation
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_TYPE_FAMILIES: Dict[str, set] = {
    "integer": {"int", "integer", "bigint", "smallint", "tinyint", "int2",
                "int4", "int8", "int64", "long", "serial", "bigserial",
                "number", "numeric", "decimal"},
    "float":   {"float", "double", "real", "float4", "float8", "money",
                "currency", "double precision"},
    "text":    {"varchar", "nvarchar", "char", "nchar", "text", "string",
                "clob", "ntext", "bpchar", "character varying"},
    "date":    {"date", "datetime", "timestamp", "timestamptz",
                "timestamp with time zone", "timestamp without time zone"},
    "uuid":    {"uuid", "guid", "uniqueidentifier"},
    "boolean": {"bool", "boolean", "bit"},
}
_NUMERIC = {"integer", "float"}

# Minimum embedding similarity to even consider a semantic-tier candidate —
# below this, two columns are treated as unrelated rather than a weak match.
SEMANTIC_SIM_THRESHOLD = 0.55
# Below this final blended confidence, no match is reported at all (caller
# falls back to LLM generation for that column).
MIN_MATCH_CONFIDENCE = 0.55


def _type_family(dt: str) -> Optional[str]:
    dt = (dt or "").strip().lower()
    for fam, members in _TYPE_FAMILIES.items():
        if dt in members:
            return fam
    return None


def type_compat(ta: str, tb: str) -> float:
    """Compatibility score [0.0-1.0] between two data types — unknown/blank
    types are treated neutrally rather than penalized, since some connectors
    (CSV/Excel) don't always report a precise type."""
    if not ta or not tb:
        return 0.5
    fa, fb = _type_family(ta), _type_family(tb)
    if fa is None or fb is None:
        return 0.5
    if fa == fb:
        return 1.0
    if {fa, fb} <= _NUMERIC:
        return 0.85
    return 0.2


def _embed_text(spec: Dict) -> str:
    parts = [spec.get("normalized_phrase") or spec.get("name", ""),
              spec.get("description", ""), spec.get("data_type", "")]
    return " | ".join(p for p in parts if p)


def score_pair(col_a: Dict, col_b: Dict) -> Tuple[float, str]:
    """Score how likely col_a and col_b represent the same business concept.

    col_a / col_b shape: {name, data_type, description, normalized_phrase}
    (normalized_phrase from glossary_nlp.normalize_identifier).

    Returns (confidence, method) where method is one of
    "normalized_name" | "semantic" | "" (no match, confidence will be 0.0).
    """
    phrase_a = (col_a.get("normalized_phrase") or "").strip().lower()
    phrase_b = (col_b.get("normalized_phrase") or "").strip().lower()

    if phrase_a and phrase_a == phrase_b:
        tc = type_compat(col_a.get("data_type", ""), col_b.get("data_type", ""))
        base = 0.80
        if tc >= 0.85:
            base += 0.06
        elif tc < 0.5:
            base -= 0.10
        return round(min(base, 0.90), 3), "normalized_name"

    # Semantic tier — only attempted if the caller supplies embeddings
    # (kept out of this pure module; see find_best_match below).
    return 0.0, ""


def _semantic_score(sim: float, type_a: str, type_b: str) -> float:
    if sim < SEMANTIC_SIM_THRESHOLD:
        return 0.0
    base = 0.55 + (sim - SEMANTIC_SIM_THRESHOLD) / (1.0 - SEMANTIC_SIM_THRESHOLD) * 0.30
    tc = type_compat(type_a, type_b)
    if tc >= 0.85:
        base += 0.03
    elif tc < 0.5:
        base -= 0.05
    return round(min(base, 0.85), 3)


def find_best_match(
    column_spec: Dict, candidate_pool: List[Dict],
    embed_fn=None, cosine_fn=None,
) -> Optional[Dict]:
    """
    column_spec: {name, data_type, description, normalized_phrase}
    candidate_pool: list of {name/normalized_phrase, data_type, description,
                              term_id, preferred_name, definition, ...}
                    (one entry per already-linked column from OTHER sources —
                    see glossary_registry.list_all_linked_columns)
    embed_fn/cosine_fn: injected (dialog_agent.kg_router._embed/_cosine in
                        production; omit in unit tests to exercise the
                        normalized-name tier without a model dependency).

    Returns the best-scoring candidate dict augmented with
    {"match_confidence": float, "match_method": str}, or None if nothing
    clears MIN_MATCH_CONFIDENCE.
    """
    best: Optional[Dict] = None
    best_score = 0.0
    best_method = ""

    # Tier 1: normalized-name exact match (cheap, no embedding call needed).
    for cand in candidate_pool:
        score, method = score_pair(column_spec, cand)
        if score > best_score:
            best, best_score, best_method = cand, score, method

    # Tier 2: semantic similarity — only for candidates that didn't already
    # hit the (stronger) normalized-name tier, and only if an embedder is
    # available (gracefully skipped when the embedding model can't load,
    # same degrade-to-neutral convention as dialog_agent/kg_router.py).
    if best_score < 0.80 and embed_fn is not None and cosine_fn is not None:
        emb_a = embed_fn(_embed_text(column_spec))
        if emb_a is not None:
            for cand in candidate_pool:
                if score_pair(column_spec, cand)[1] == "normalized_name":
                    continue
                emb_b = embed_fn(_embed_text(cand))
                if emb_b is None:
                    continue
                sim = cosine_fn(emb_a, emb_b)
                score = _semantic_score(sim, column_spec.get("data_type", ""), cand.get("data_type", ""))
                if score > best_score:
                    best, best_score, best_method = cand, score, "semantic"

    if best is None or best_score < MIN_MATCH_CONFIDENCE:
        return None
    result = dict(best)
    result["match_confidence"] = best_score
    result["match_method"] = best_method
    return result
