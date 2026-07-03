"""
domain_classifier.py
LLM-based SUB-DOMAIN classifier for datasources — with an OPEN, self-growing
taxonomy instead of the fixed _FUNCTION_SIGNALS keyword list that
orchestrator_api._infer_domain_from_report used to be the source of truth for.

business_classifier.py is the MAIN classification (e.g. "Footwear Manufacturing
& Retail") — this module classifies the SUB-DOMAIN underneath it (e.g. "Supply
Chain", "FP&A", "Merchandising"): the specific business function/process this
source's data represents *within* that already-classified business. The
business label is always fetched first (via business_classifier.predict) and
passed to the LLM as fixed context, so the two never disagree the way the old
independent keyword scorers could (e.g. business="Footwear Manufacturing &
Retail" but domain="CPG/Supply Chain" — mismatched industries).

Mirrors business_classifier.py's mechanics exactly: an LLM looks at the schema
signals (+ the main business label) and names the sub-domain — reusing an
existing label when it genuinely fits, or coining a new one when it doesn't.
Every decision is persisted in `md_domain_labels`, so the taxonomy is simply
"whatever distinct labels exist in that table" — it grows automatically, with
no code change required.

The keyword scorer (_FUNCTION_SIGNALS, copied from orchestrator_api.py) is
kept purely as a last-resort fallback for when the LLM is unreachable — it is
never the source of truth for the label space.

Standalone / root-level (like metadata_catalog.py and business_classifier.py)
so it can be imported by orchestrator_api.py without pulling in unrelated deps.

Public API
----------
known_labels()          -> List[str]                  the current (dynamic) sub-domain taxonomy
label_source(source_id) -> dict | None                 LLM-classify one source, cache + return it
predict(source_id)      -> dict | None                 {"sub_domain", "confidence", "method"}
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import business_classifier as _bc
import metadata_catalog as _mc

logger = logging.getLogger(__name__)

# Minimum keyword-signal score (name hits + 0.5 * sample hits) for the
# last-resort fallback scorer to consider itself confident.
_MIN_BOOTSTRAP_SCORE = 1.0

# A cached label with a numeric confidence below this is treated as "wrongly
# generated" — reindexing will re-classify it rather than trusting the cache.
# A cached label with no confidence value at all (e.g. rule-fallback) is left
# alone; this threshold only applies to LLM-scored labels. Tunable via env
# for environments where the LLM tends to under/over-report confidence.
_MIN_CONFIDENCE = float(os.environ.get("DOMAIN_MIN_CONFIDENCE", "0.7"))

# ── Last-resort fallback ONLY (used when the LLM can't be reached) ─────────────
# Kept in sync with orchestrator_api.py's _FUNCTION_SIGNALS. Copied rather than
# imported so this module stays free of orchestrator_api's heavy runtime deps,
# matching business_classifier.py's design. NOT the taxonomy — see
# label_source()/known_labels() for the real, open-ended one.
_FUNCTION_SIGNALS: List[Tuple[str, set]] = [
    ("RGM", {
        "rgm", "revenue_growth", "pricing_impact", "price_index",
        "market_share", "mix_contribution", "price_mix", "volume_mix",
        "promo_effectiveness", "trade_rate", "net_revenue_mgmt",
        "pack_price", "price_ladder", "revenue_mgmt",
    }),
    ("FP&A", {
        "budget", "forecast", "actuals", "variance", "rolling_forecast",
        "zero_based", "ytd", "qtd", "mtd", "period_budget",
        "p_and_l", "pnl", "income_statement", "ebitda", "ebit", "ebitda_margin",
        "gross_profit", "operating_profit", "net_income",
        "cash_flow", "capex", "opex", "working_capital", "balance_sheet",
        "cost_centre", "cost_center", "cost_element", "gl_account",
        "chart_of_accounts", "profit_centre", "profit_center",
        "business_unit_plan", "entity_plan",
        "financial_planning", "fp_and_a", "fpa", "headcount_plan",
        "scenario", "version", "baseline", "reforecast",
    }),
    ("Supply Chain", {
        "otif", "fill_rate", "lead_time", "inventory_days",
        "doh", "woh", "perfect_order", "on_time_delivery",
        "supplier_on_time", "stock_out", "stockout", "safety_stock",
        "replenishment", "demand_plan", "s_and_op", "sop",
        "warehouse", "3pl", "distribution",
        "purchase_order", "po_line", "goods_receipt", "grn",
        "shipment", "delivery", "transit", "backorder", "reorder",
        "supplier", "vendor", "procurement", "inventory_turn",
    }),
    ("Sales", {
        "sales_rep", "territory", "quota", "pipeline", "opportunity",
        "win_rate", "deal_size", "crm", "account_manager",
        "sales_force", "coverage_model", "gtm",
    }),
    ("Marketing", {
        "impressions", "click_through", "cpm", "cpc", "cpa",
        "media_spend", "brand_awareness", "campaign", "reach",
        "frequency", "attribution", "media_mix",
    }),
    ("HR/People", {
        "headcount", "attrition", "time_to_hire", "cost_per_hire",
        "offer_acceptance", "engagement_score", "absenteeism",
        "performance_rating", "fte_count", "salary_band",
    }),
    ("Operations", {
        "oee", "downtime", "throughput", "cycle_time", "capacity",
        "utilisation", "shift", "plant", "production_schedule",
    }),
    ("CX", {
        "nps", "csat", "net_promoter", "ticket", "resolution_time",
        "first_contact", "escalation", "customer_effort", "churn",
        "voice_of_customer",
    }),
]


def _score_subdomain(name_tokens: set, sample_tokens: set) -> Optional[Tuple[str, float]]:
    """Fallback only — see module docstring. Same keyword-overlap rule
    orchestrator_api._infer_domain_from_report used for its function tier."""
    best_label, best_score = None, 0.0
    for label, name_signals in _FUNCTION_SIGNALS:
        name_hits = len(name_tokens & name_signals)
        total = name_hits
        if total > best_score:
            best_score, best_label = total, label
    if best_label is None or best_score < _MIN_BOOTSTRAP_SCORE:
        return None
    return best_label, best_score


# ── Open taxonomy: persisted LLM labels ─────────────────────────────────────────

_DDL_LABELS = """
CREATE TABLE IF NOT EXISTS md_domain_labels (
    source_id   TEXT PRIMARY KEY,
    label       TEXT NOT NULL,
    is_new      INTEGER NOT NULL DEFAULT 0,
    confidence  REAL,
    updated_at  TEXT NOT NULL
)
"""


def _ensure_labels_table(cur) -> None:
    cur.ddl(_DDL_LABELS)


def known_labels() -> List[str]:
    """The current sub-domain taxonomy — every distinct label assigned to any source so far. Grows dynamically; never a fixed list."""
    with _mc._cursor_ctx() as cur:
        _ensure_labels_table(cur)
        rows = cur.execute(
            "SELECT DISTINCT label FROM md_domain_labels ORDER BY label"
        ).fetchall()
    return [r["label"] for r in rows]


def _get_cached_label(source_id: str) -> Optional[Dict]:
    with _mc._cursor_ctx() as cur:
        _ensure_labels_table(cur)
        row = cur.execute(
            "SELECT label, is_new, confidence FROM md_domain_labels WHERE source_id=?",
            (source_id,),
        ).fetchone()
    if not row:
        return None
    return {
        "sub_domain": row["label"],
        "confidence": row.get("confidence"),
        "method":     "llm",
        "is_new":     bool(row.get("is_new")),
    }


def get_cached_label(source_id: str) -> Optional[Dict]:
    """Public, LLM-free read of whatever sub-domain indexing already assigned
    this source (or None if it hasn't been indexed/classified yet).
    Classification itself only ever happens from the indexing/reindexing
    pipeline (see orchestrator_api._index_source) — this is for callers
    (e.g. the UI) that just want to display the current label without
    triggering a new LLM call."""
    return _get_cached_label(source_id)


def _store_label(source_id: str, label: str, is_new: bool, confidence: Optional[float]) -> None:
    with _mc._cursor_ctx() as cur:
        _ensure_labels_table(cur)
        cur.execute(
            "INSERT INTO md_domain_labels (source_id, label, is_new, confidence, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(source_id) DO UPDATE SET "
            "label=excluded.label, is_new=excluded.is_new, "
            "confidence=excluded.confidence, updated_at=excluded.updated_at",
            (source_id, label, int(is_new), confidence, _mc._now()),
        )


_CLASSIFY_SYSTEM = (
    "You classify a datasource's SUB-DOMAIN — the specific business function or "
    "process its data represents, one level more specific than the overall "
    "business/industry it already belongs to (which is given to you below, "
    "already decided — do not re-derive or contradict it). Judge purely from "
    "the table names, column names, and sample values given below; there is no "
    "preset list to fit into.\n\n"
    "Give a concise, accurate sub-domain label (1-4 words, Title Case, e.g. "
    "'Supply Chain', 'FP&A', 'Merchandising', 'Store Operations') that names "
    "the function/process this data captures within that business.\n\n"
    "Respond with ONLY a JSON object, no markdown fences: "
    '{"label": "<sub-domain>", "confidence": <0.0-1.0>}'
)


def _llm_classify_subdomain(
    text: str, business_label: Optional[str], model: Optional[str] = None,
) -> Optional[Dict]:
    """Ask the LLM to name this source's sub-domain given its schema signals
    and the already-decided main business label. Returns None on any failure
    (caller falls back)."""
    model = model or os.environ.get("DIALOG_LLM_MODEL", "claude-haiku-4-5")
    signals = " ".join(text.split()[:400])
    business_line = (
        f"This datasource's overall business/industry has already been classified as: {business_label!r}.\n\n"
        if business_label else ""
    )
    user_msg = (
        f"{business_line}"
        f"Schema signals for this datasource (table names, column names, sample values):\n"
        f"{signals}\n\n"
        f"What is this datasource's sub-domain (the specific function/process within that business)?"
    )
    try:
        from llm_client import get_client
        client = get_client()
        msg = client.messages.create(
            model=model, max_tokens=256, temperature=0.0,
            system=_CLASSIFY_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = msg.content[0].text if msg.content else ""
    except Exception as exc:
        logger.warning("domain_classifier: LLM call failed — %s", exc)
        return None

    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        s, e = cleaned.find("{"), cleaned.rfind("}")
        if s == -1 or e == -1:
            logger.warning("domain_classifier: could not parse LLM response: %r", raw[:200])
            return None
        try:
            obj = json.loads(cleaned[s:e + 1])
        except json.JSONDecodeError:
            logger.warning("domain_classifier: could not parse LLM response: %r", raw[:200])
            return None

    label = str(obj.get("label") or "").strip()
    if not label:
        return None
    return {
        "label":      label,
        "confidence": float(obj.get("confidence")) if obj.get("confidence") is not None else None,
    }


def label_source(source_id: str, model: Optional[str] = None, force: bool = False) -> Optional[Dict]:
    """
    Classify one source's sub-domain from its schema signals plus the main
    business label (business_classifier is always resolved first — it is the
    main classification this one is nested under) and persist the result.
    Returns the cached label without an LLM call unless force=True, or unless
    the cached label is missing altogether, or was a low-confidence ("wrongly
    generated") LLM call — either case re-classifies as if force=True.
    Returns None if the source has no metadata indexed, or the LLM is
    unreachable.
    """
    if not force:
        cached = _get_cached_label(source_id)
        if cached and not (
            cached.get("method") == "llm"
            and cached.get("confidence") is not None
            and cached["confidence"] < _MIN_CONFIDENCE
        ):
            return cached

    per_source = _bc._fetch_source_texts()
    bucket = per_source.get(source_id)
    if not bucket or not (bucket["names"] or bucket["samples"]):
        return None

    # Main classification first — business_classifier is the source of truth
    # for "what business is this", so the sub-domain is always judged relative
    # to it rather than independently re-deriving an industry of its own.
    business = _bc.predict(source_id)
    business_label = business.get("business") if business else None

    text = _bc._text_blob(bucket)
    result = _llm_classify_subdomain(text, business_label, model=model)
    if not result:
        return None

    # is_new is a post-hoc, deterministic check against the taxonomy accumulated
    # so far (case-insensitive exact match) — purely informational for the UI/
    # event log, it plays no part in how the LLM arrived at the label above.
    existing = {l.lower() for l in known_labels()}
    is_new = result["label"].lower() not in existing

    _store_label(source_id, result["label"], is_new, result["confidence"])
    return {
        "sub_domain": result["label"],
        "confidence": result["confidence"],
        "method":     "llm",
        "is_new":     is_new,
    }


def predict(source_id: str) -> Optional[Dict]:
    """
    Predict the sub-domain for one already-indexed source, from the OPEN
    taxonomy (see module docstring) — never a fixed category list.

    Order of precedence:
      1. Cached LLM label for this exact source (no LLM call).
      2. A fresh, live LLM classification (label_source()) — the source of
         truth, tried before the keyword fallback on every never-before-seen
         source; caches the result and can coin a brand-new taxonomy entry.
      3. Keyword-scorer fallback — ONLY reached if the LLM is unreachable at
         all (e.g. no ANTHROPIC_API_KEY configured). Degraded offline mode,
         never the source of truth for what categories can exist.

    Returns None if the source has no metadata indexed.
    """
    cached = _get_cached_label(source_id)
    if cached:
        return cached

    live = label_source(source_id)
    if live:
        return live

    per_source = _bc._fetch_source_texts()
    bucket = per_source.get(source_id)
    if not bucket or not (bucket["names"] or bucket["samples"]):
        return None

    scored = _score_subdomain(bucket["names"], bucket["samples"])
    if not scored:
        return None
    label, score = scored
    return {"sub_domain": label, "confidence": None, "method": "rule-fallback", "score": score}
