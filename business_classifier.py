"""
business_classifier.py
ML-based business/industry classifier for datasources — with an OPEN, self-growing
taxonomy instead of a fixed category list.

The label space is NOT hardcoded. For each source, an LLM looks at the actual
table/column names and sampled values and decides the business/industry — reusing
one of the categories already seen across other sources when it genuinely fits,
or coining a new one (e.g. "Human Resources") when none of the existing labels
apply. Every decision is persisted in `md_business_labels`, so the taxonomy is
simply "whatever distinct labels exist in that table" — it grows automatically
as new kinds of datasources get indexed, with no code change required.

A TF-IDF + Logistic Regression model is trained on top of those LLM-assigned
labels so that classification is fast (no LLM round-trip) for sources that
already resemble ones seen before; its class list is whatever labels exist in
the table at training time, so it grows/shrinks with the taxonomy too.

The only hardcoded piece left is a small keyword scorer (_INDUSTRY_SIGNALS),
kept purely as a last-resort fallback for when the LLM is unreachable (e.g. no
ANTHROPIC_API_KEY configured) — it is never the source of truth for the label
space and is not used to decide what categories can exist.

Standalone / root-level (like metadata_catalog.py) so it can be imported by
orchestrator_api.py without pulling in unrelated heavy deps.

Public API
----------
known_labels()          -> List[str]                  the current (dynamic) taxonomy
label_source(source_id) -> dict | None                 LLM-classify one source, cache + return it
build_training_set()    -> List[Tuple[text, label]]    cached LLM-labeled examples
train(min_examples=4)   -> dict                         trains + persists the model, returns stats
is_trained()             -> bool
predict(source_id)       -> dict | None                 {"business", "confidence", "method"}
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import metadata_catalog as _mc

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("BUSINESS_MODEL_PATH", "data/business_classifier.joblib")

# Minimum keyword-signal score (name hits + 0.5 * sample hits) for the
# last-resort fallback scorer to consider itself confident.
_MIN_BOOTSTRAP_SCORE = 1.0

# ── Last-resort fallback ONLY (used when the LLM can't be reached) ─────────────
# Kept in sync with orchestrator_api.py's _INDUSTRY_SIGNALS / _INDUSTRY_SAMPLE_SIGNALS
# (industry tier only — copied rather than imported so this module stays free of
# orchestrator_api's heavy runtime deps, matching metadata_catalog.py's design).
# NOT the taxonomy — see label_source()/known_labels() for the real, open-ended one.
_INDUSTRY_SIGNALS: List[Tuple[str, set]] = [
    ("Aviation", {
        "aircraft", "aircraft_type", "tail_number", "tail_no", "registration",
        "fleet", "fleet_type", "wide_body", "narrow_body", "turboprop",
        "flight", "flight_number", "flight_no", "flt", "sector", "flight_leg",
        "leg", "rotation", "block_time", "flight_time", "air_time",
        "departure", "arrival", "origin", "destination",
        "scheduled_departure", "actual_departure", "scheduled_arrival", "actual_arrival",
        "dep_time", "arr_time", "etd", "eta", "atd", "ata",
        "airport", "station", "icao", "iata", "faa",
        "gate", "terminal", "runway", "apron", "taxiway", "hangar",
        "slot", "slot_time",
        "crew", "pilot", "copilot", "co_pilot", "captain", "first_officer",
        "cabin_crew", "flight_attendant", "purser", "crew_id",
        "crew_ot", "overtime", "duty_time", "rest_period", "pairing",
        "crew_complement", "deadhead",
        "passenger", "pax", "load_factor", "booked", "seats_available",
        "cabin_class", "business_class", "economy", "first_class",
        "boarding", "manifest", "no_show",
        "yield", "rask", "cask", "ask", "rpk", "atk", "rtk",
        "revenue_per_seat", "fare_class", "booking_class",
        "cargo", "freight", "baggage", "checked_bag", "carry_on",
        "cargo_weight", "payload",
        "delay", "delay_code", "delay_reason", "on_time", "otp",
        "on_time_performance", "late", "cancelled", "diverted",
        "fuel", "fuel_burn", "fuel_uplift", "fuel_consumption",
        "mro", "maintenance", "airworthiness", "aircraft_maintenance",
        "check_a", "check_b", "check_c", "check_d",
        "airline", "carrier", "codeshare", "alliance", "iata_code",
        "operating_carrier", "marketing_carrier",
        "atc", "ifr", "vfr", "route", "waypoint", "airway",
    }),
    ("CPG", {
        "sku", "brand", "sub_brand", "pack_size", "pack_type", "upc",
        "ean", "product_hierarchy", "sub_category",
        "brand_pack", "flavor", "variant", "ppg", "pog",
        "rsv", "gsv", "nrv", "nsv", "tts", "cti",
        "volume_offtake", "gross_rsv", "net_rsv",
        "trade_spend", "promo_spend", "offtake",
        "retailer", "customer_group", "trade_channel", "modern_trade",
        "general_trade", "gt", "mt", "key_account", "distributor",
        "consumer_goods", "fmcg", "cpg", "category_mgmt",
        "distribution_points", "weighted_distribution", "numeric_distribution",
    }),
    ("LS", {
        "clinical_trial", "trial", "protocol", "cohort", "arm", "randomised",
        "placebo", "double_blind", "phase", "trial_phase", "enrollment",
        "site", "investigator", "irb", "ethics_committee",
        "molecule", "drug", "compound", "active_ingredient", "excipient",
        "formulation", "dosage", "dose", "strength", "route_of_administration",
        "ndc", "anda", "nda", "bla", "inn", "atc_code",
        "adverse_event", "ae", "sae", "serious_adverse", "pharmacovigilance",
        "signal_detection", "post_market", "recall", "fda", "ema", "cdsco",
        "510k", "pma", "regulatory_submission", "label",
        "patient", "subject", "participant", "diagnosis", "indication",
        "icd10", "icd_code", "procedure_code", "comorbidity", "biomarker",
        "genomic", "genotype", "phenotype", "lab_result", "vital_sign",
        "formulary", "iqvia", "ims", "rx", "otc", "prescription",
        "market_access", "payer_mix", "rebate", "gross_to_net",
        "therapy_area", "physician", "hospital", "hcp", "specialty",
        "batch", "lot_number", "expiry", "shelf_life", "gmp", "capa",
        "deviation", "oos", "out_of_spec", "release_testing",
    }),
    ("Healthcare", {
        "length_of_stay", "los", "bed_occupancy", "bed_days", "census",
        "claims_paid", "denial_rate", "readmission", "readmit",
        "cost_per_patient", "cost_per_episode", "drg", "icd", "cpt_code",
        "admission", "discharge", "inpatient", "outpatient", "ed_visit",
        "emergency", "icu", "nicu", "payer", "insurer", "hmo", "ppo",
        "member", "beneficiary", "eligibility", "prior_auth",
        "network", "in_network", "out_of_network", "copay", "deductible",
        "ehr", "emr", "fhir", "hl7", "hipaa",
    }),
    ("Telecom", {
        "arpu", "mou", "data_usage", "subscriber", "subscription",
        "churn_rate", "churn", "prepaid", "postpaid", "sim", "msisdn",
        "imei", "imsi", "roaming", "spectrum", "frequency_band",
        "minutes_of_use", "cell_tower", "cell_site", "bts", "enb",
        "bandwidth", "network_quality", "dropped_call", "call_setup",
        "data_plan", "recharge", "top_up", "bundle", "vas",
        "mnp", "number_portability", "operator", "mvno",
        "fiber", "broadband", "dsl", "ftth", "last_mile",
        "revenue_per_user", "blended_arpu", "voice_revenue", "data_revenue",
    }),
    ("Banking/FS", {
        "nii", "nim", "net_interest_income", "net_interest_margin",
        "casa", "current_account", "savings_account", "fixed_deposit",
        "gnpa", "nnpa", "npa", "non_performing", "provision_coverage",
        "crar", "capital_adequacy", "tier1", "tier2", "slr", "crr",
        "net_interest", "loan_book", "loan_portfolio", "advances",
        "deposit", "borrowing", "liability", "asset_quality",
        "credit_risk", "provisioning", "write_off", "write_back",
        "return_on_assets", "roa", "roe", "cost_to_income",
        "loan", "mortgage", "home_loan", "auto_loan", "personal_loan",
        "emi", "disbursement", "sanction", "outstanding", "overdue",
        "dpd", "days_past_due", "delinquency", "restructured",
        "credit_score", "bureau_score", "cibil", "ltv_ratio",
        "account", "account_number", "ifsc", "bic", "swift",
        "transaction", "txn", "debit", "credit", "transfer",
        "branch", "atm", "digital_banking", "mobile_banking",
        "card", "credit_card", "debit_card", "pos", "merchant",
        "bond", "yield_curve", "duration", "convexity", "alm",
        "forex", "fx_rate", "treasury", "investment_portfolio",
    }),
    ("Insurance", {
        "premium", "gross_premium", "net_premium", "earned_premium",
        "written_premium", "single_premium", "renewal_premium",
        "policy", "policy_number", "policy_term", "policy_holder",
        "insured", "beneficiary", "nominee", "sum_assured",
        "claim", "claim_number", "claim_date", "claim_amount",
        "incurred_loss", "paid_loss", "ibnr", "ibner",
        "loss_ratio", "claims_ratio", "combined_ratio",
        "claims_settled", "claims_repudiated", "claims_pending",
        "underwriting", "risk_assessment", "risk_class", "risk_score",
        "proposal", "quote", "endorsement", "exclusion", "deductible",
        "sub_limit", "co_insurance", "reinsurance", "cedant",
        "treaty", "facultative", "retrocession",
        "actuarial", "mortality", "morbidity", "lapse_rate",
        "persistency", "surrender", "maturity", "annuity",
        "reserve", "liability_reserve", "unearned_premium",
        "life", "health", "motor", "fire", "marine", "liability",
        "property", "casualty", "p_and_c", "general_insurance",
        "term", "ulip", "endowment", "whole_life",
        "agent", "broker", "bancassurance", "direct", "online_sale",
        "agency_code", "pos_agent", "intermediary",
    }),
    ("Retail", {
        "store_sales", "same_store", "comp_sales", "footfall", "traffic",
        "basket", "basket_size", "shrinkage", "planogram", "assortment",
        "markdown", "sell_through", "stock_turn", "stock_turnover",
        "store", "store_id", "store_format", "hypermarket", "supermarket",
        "pos_transaction", "till", "checkout_lane",
        "loyalty", "loyalty_card", "points_earned", "redemption",
        "private_label", "own_brand", "national_brand",
        "replenishment", "out_of_stock", "on_shelf_availability",
        "promotion", "price_cut", "feature", "display",
    }),
    ("E-commerce", {
        "gmv", "aov", "cac", "ltv", "roas", "cart", "cart_abandonment",
        "conversion_rate", "basket_size", "add_to_cart",
        "checkout", "order_value", "return_rate", "refund",
        "marketplace", "seller", "listing", "sku_listing",
        "session", "visit", "bounce_rate", "page_view",
        "search_rank", "sponsored", "ad_spend",
        "fulfillment", "last_mile", "ndr", "rto",
        "review", "rating", "star_rating",
    }),
    ("Manufacturing", {
        "oee", "oee_availability", "oee_performance", "oee_quality",
        "scrap_rate", "scrap", "rework", "rejection_rate", "defect_rate",
        "yield_pct", "first_pass_yield", "fpy", "quality_inspection",
        "downtime", "planned_downtime", "unplanned_downtime",
        "mtbf", "mttr", "mttf", "reliability",
        "cycle_time", "takt_time", "throughput", "production_order",
        "work_order", "work_in_progress", "wip", "finished_goods",
        "production_schedule", "planned_qty", "actual_qty",
        "shift", "shift_output", "line_efficiency",
        "machine", "machine_id", "equipment", "asset_id",
        "spindle", "press", "mould", "tooling", "fixture",
        "preventive_maintenance", "pm_schedule", "breakdown",
        "plant", "shop_floor", "cell", "workcenter",
        "bom", "bill_of_materials", "raw_material", "component",
        "sub_assembly", "assembly", "routing", "operation_sequence",
        "material_consumption", "standard_cost", "actual_cost",
        "iso", "iatf", "as9100", "spc", "cpk", "ppk", "control_chart",
    }),
    ("Agriculture", {
        "crop", "crop_type", "crop_variety", "variety", "hybrid",
        "seed", "seed_rate", "germination", "sowing", "planting",
        "harvest", "harvesting", "yield_per_acre", "yield_per_hectare",
        "acreage", "hectare", "farm", "field", "plot", "parcel",
        "kharif", "rabi", "zaid", "season", "growing_season",
        "soil", "soil_type", "soil_health", "ph_level", "organic_matter",
        "fertilizer", "urea", "dap", "potash", "micronutrient",
        "pesticide", "herbicide", "fungicide", "insecticide",
        "irrigation", "drip", "sprinkler", "canal", "groundwater",
        "livestock", "cattle", "buffalo", "poultry", "sheep", "goat",
        "milk_yield", "milk_production", "fat_content", "snf",
        "animal_id", "tag_id", "breed", "lactation", "calving",
        "feed", "fodder", "dry_matter",
        "mandi", "apmc", "procurement_centre", "farmer",
        "farmer_id", "fpo", "cooperative", "agri_input",
        "minimum_support_price", "msp", "procurement_price",
        "storage", "cold_storage", "warehouse", "silo",
        "commodity", "agri_commodity", "produce", "grain",
        "wheat", "rice", "paddy", "maize", "soybean", "cotton",
        "sugarcane", "pulses", "oilseed",
        "rainfall", "temperature", "humidity", "evapotranspiration",
        "geo_lat", "geo_lon", "district", "taluka", "village",
        "land_parcel", "cadastral",
        "kcc", "kisan_credit", "crop_insurance", "pmfby",
        "subsidy", "input_cost", "cost_of_cultivation",
    }),
    ("SaaS", {
        "dau", "mau", "mrr", "arr", "expansion_revenue",
        "activation_rate", "feature_adoption", "session_duration",
        "retention_rate", "nrr", "logo_churn", "saas",
        "trial", "freemium", "paid_conversion", "upgrade",
        "seat", "license", "subscription_tier", "plan",
        "api_call", "api_usage", "endpoint", "latency",
        "uptime", "sla", "incident", "p1", "p2",
        "onboarding", "time_to_value", "ttv", "health_score",
    }),
]

_INDUSTRY_SAMPLE_SIGNALS: Dict[str, set] = {
    "Aviation": {
        "b737", "a320", "b777", "a330", "b787", "a380", "e190", "crj9", "dh8d",
        "b738", "a321", "a319", "b772", "b788", "a220", "a350", "b767", "b757",
        "atr72", "q400", "atr42",
        "business class", "economy class", "first class", "premium economy",
        "departed", "arrived", "cancelled", "diverted", "airborne",
        "captain", "first officer", "senior first officer", "purser",
        "senior cabin crew", "cabin crew",
        "technical", "reactionary", "weather delay", "atc delay",
        "domestic flight", "international flight",
    },
    "Banking": {
        "standard", "sub-standard", "doubtful", "loss asset",
        "non-performing", "restructured",
        "home loan", "personal loan", "auto loan", "vehicle loan",
        "business loan", "gold loan", "education loan", "lap", "msme loan",
        "savings account", "current account", "fixed deposit",
        "recurring deposit", "nre account", "nro account",
        "0-30 days", "31-60 days", "61-90 days", "91-180 days", ">180 days",
        "aaa", "aa+", "aa", "a+", "a", "bbb+", "bbb",
        "dormant", "active", "frozen", "closed",
    },
    "LS": {
        "phase i", "phase ii", "phase iii", "phase iv",
        "phase 1", "phase 2", "phase 3", "phase 4",
        "enrolled", "randomised", "randomized", "screen failure",
        "discontinued", "completed", "withdrawn", "lost to follow-up",
        "mild", "moderate", "severe", "life-threatening", "fatal",
        "tablet", "capsule", "injection", "infusion", "oral solution",
        "suspension", "patch", "inhaler",
        "oral", "intravenous", "subcutaneous", "intramuscular", "topical",
        "nda", "anda", "bla", "ind",
    },
    "Insurance": {
        "term life", "whole life", "endowment", "ulip", "money back",
        "health insurance", "motor insurance", "fire insurance",
        "marine insurance", "group health", "personal accident",
        "intimated", "under investigation", "settled", "repudiated",
        "partially settled", "closed", "reopened",
        "annual", "semi-annual", "quarterly", "single premium",
        "bancassurance", "agency", "direct", "online", "broker",
        "in force", "lapsed", "surrendered", "matured", "free look",
    },
    "Manufacturing": {
        "scratch", "dent", "burr", "flash", "porosity", "crack",
        "dimensional defect", "surface defect", "weld defect",
        "morning shift", "afternoon shift", "evening shift", "night shift",
        "shift a", "shift b", "shift c", "shift i", "shift ii", "shift iii",
        "breakdown", "planned maintenance", "unplanned downtime",
        "idle", "changeover", "setup",
        "released", "in process", "partially completed", "goods receipt",
        "accept", "reject", "rework", "hold", "scrap",
        "iso 9001", "iatf 16949", "as9100",
    },
    "Agriculture": {
        "wheat", "rice", "paddy", "maize", "cotton", "sugarcane",
        "soybean", "groundnut", "sorghum", "bajra", "jowar",
        "potato", "onion", "tomato", "mustard", "sunflower",
        "chickpea", "lentil", "arhar", "moong",
        "kharif", "rabi", "zaid",
        "drip irrigation", "sprinkler", "canal irrigation",
        "rainfed", "flood irrigation", "micro irrigation",
        "alluvial", "black soil", "red soil", "laterite", "sandy loam",
        "clay loam", "loamy sand",
        "urea", "dap", "mop", "npk", "compost", "vermicompost",
    },
    "CPG": {
        "pet bottle", "glass bottle", "tetra pack", "sachet", "pouch",
        "can", "tin", "carton", "blister pack",
        "modern trade", "general trade", "horeca", "e-commerce",
        "supermarket", "hypermarket", "convenience store", "wholesale",
        "active", "new launch", "discontinued", "seasonal", "limited edition",
        "volume", "value", "tonnage",
    },
    "Telecom": {
        "2g", "3g", "4g", "5g", "lte", "volte",
        "prepaid", "postpaid",
        "voice", "data", "sms", "roaming", "vas", "broadband",
        "porting out", "voluntary deactivation", "non-payment",
        "call drop", "network failure", "planned outage",
    },
    "Healthcare": {
        "inpatient", "outpatient", "emergency", "day surgery",
        "observation", "telehealth",
        "medicare", "medicaid", "commercial", "self-pay", "uninsured",
        "acute", "chronic", "preventive", "follow-up",
        "discharged", "transferred", "expired", "absconded",
    },
    "Retail": {
        "hypermarket", "supermarket", "convenience", "express",
        "flagship", "kiosk", "warehouse store",
        "gold", "silver", "platinum", "bronze",
        "clearance", "seasonal sale", "promo", "everyday low price",
        "in stock", "out of stock", "on order", "discontinued",
    },
}


def _score_industry(name_tokens: set, sample_tokens: set) -> Optional[Tuple[str, float]]:
    """Fallback only — see module docstring. Same rule as orchestrator_api._infer_domain_from_report's industry tier."""
    best_label, best_score = None, 0.0
    for label, name_signals in _INDUSTRY_SIGNALS:
        name_hits = len(name_tokens & name_signals)
        samp_hits = len(sample_tokens & _INDUSTRY_SAMPLE_SIGNALS.get(label, set()))
        total = name_hits + 0.5 * samp_hits
        if total > best_score:
            best_score, best_label = total, label
    if best_label is None or best_score < _MIN_BOOTSTRAP_SCORE:
        return None
    return best_label, best_score


# ── Open taxonomy: persisted LLM labels ─────────────────────────────────────────

_DDL_LABELS = """
CREATE TABLE IF NOT EXISTS md_business_labels (
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
    """The current taxonomy — every distinct business/industry label assigned to any source so far. Grows dynamically; never a fixed list."""
    with _mc._cursor_ctx() as cur:
        _ensure_labels_table(cur)
        rows = cur.execute(
            "SELECT DISTINCT label FROM md_business_labels ORDER BY label"
        ).fetchall()
    return [r["label"] for r in rows]


def _get_cached_label(source_id: str) -> Optional[Dict]:
    with _mc._cursor_ctx() as cur:
        _ensure_labels_table(cur)
        row = cur.execute(
            "SELECT label, is_new, confidence FROM md_business_labels WHERE source_id=?",
            (source_id,),
        ).fetchone()
    if not row:
        return None
    return {
        "business":   row["label"],
        "confidence": row.get("confidence"),
        "method":     "llm",
        "is_new":     bool(row.get("is_new")),
    }


def _store_label(source_id: str, label: str, is_new: bool, confidence: Optional[float]) -> None:
    with _mc._cursor_ctx() as cur:
        _ensure_labels_table(cur)
        cur.execute(
            "INSERT INTO md_business_labels (source_id, label, is_new, confidence, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(source_id) DO UPDATE SET "
            "label=excluded.label, is_new=excluded.is_new, "
            "confidence=excluded.confidence, updated_at=excluded.updated_at",
            (source_id, label, int(is_new), confidence, _mc._now()),
        )


_CLASSIFY_SYSTEM = (
    "You classify a datasource's business/industry purely from its own schema — the "
    "table names, column names, and sample values given below. Judge ONLY what this "
    "specific data looks like; do not try to match, reuse, or steer toward any category "
    "used for other datasources — there is no preset list to fit into.\n\n"
    "Give a concise, accurate business/industry label (1-4 words, Title Case, e.g. "
    "'Human Resources', 'Aviation Operations', 'Legal Services') that describes what "
    "this data is actually about.\n\n"
    "Respond with ONLY a JSON object, no markdown fences: "
    '{"label": "<category name>", "confidence": <0.0-1.0>}'
)


def _llm_classify_business(text: str, model: Optional[str] = None) -> Optional[Dict]:
    """Ask the LLM to name this source's business/industry from its schema signals alone — no prior taxonomy is shown to it. Returns None on any failure (caller falls back)."""
    model = model or os.environ.get("DIALOG_LLM_MODEL", "claude-haiku-4-5")
    # Cap prompt size — schema signal tokens are already deduped/sorted by _text_blob.
    signals = " ".join(text.split()[:400])
    user_msg = (
        f"Schema signals for this datasource (table names, column names, sample values):\n"
        f"{signals}\n\n"
        f"What business/industry is this datasource?"
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
        logger.warning("business_classifier: LLM call failed — %s", exc)
        return None

    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        s, e = cleaned.find("{"), cleaned.rfind("}")
        if s == -1 or e == -1:
            logger.warning("business_classifier: could not parse LLM response: %r", raw[:200])
            return None
        try:
            obj = json.loads(cleaned[s:e + 1])
        except json.JSONDecodeError:
            logger.warning("business_classifier: could not parse LLM response: %r", raw[:200])
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
    Classify one source purely from its own schema signals (no other source's
    label is shown to the LLM) and persist the result. Returns the cached
    label without an LLM call unless force=True. Returns None if the source
    has no metadata indexed, or the LLM is unreachable.
    """
    if not force:
        cached = _get_cached_label(source_id)
        if cached:
            return cached

    per_source = _fetch_source_texts()
    bucket = per_source.get(source_id)
    if not bucket or not (bucket["names"] or bucket["samples"]):
        return None

    text = _text_blob(bucket)
    result = _llm_classify_business(text, model=model)
    if not result:
        return None

    # is_new is a post-hoc, deterministic check against the taxonomy accumulated
    # so far (case-insensitive exact match) — purely informational for the UI/
    # event log, it plays no part in how the LLM arrived at the label above.
    existing = {l.lower() for l in known_labels()}
    is_new = result["label"].lower() not in existing

    _store_label(source_id, result["label"], is_new, result["confidence"])
    return {
        "business":   result["label"],
        "confidence": result["confidence"],
        "method":     "llm",
        "is_new":     is_new,
    }


def _fetch_source_texts() -> Dict[str, Dict[str, set]]:
    """Group table/column names + sampled values by source_id, straight from the metadata catalog DB."""
    with _mc._cursor_ctx() as cur:
        _mc._ensure(cur)
        rows = cur.execute(
            "SELECT e.source_id, e.table_name, a.column_name, a.top_values "
            "FROM md_entities e JOIN md_attributes a ON a.metadata_id = e.metadata_id "
            "WHERE e.deleted_from_source = 0"
        ).fetchall()

    per_source: Dict[str, Dict[str, set]] = defaultdict(lambda: {"names": set(), "samples": set()})
    for r in rows:
        bucket = per_source[r["source_id"]]
        tbl = (r.get("table_name") or "").lower()
        col = (r.get("column_name") or "").lower()
        bucket["names"].add(tbl)
        bucket["names"].update(tbl.split("_"))
        bucket["names"].add(col)
        bucket["names"].update(col.split("_"))
        try:
            values = json.loads(r.get("top_values") or "[]")
        except (TypeError, ValueError):
            values = []
        for val in values or []:
            if val is None:
                continue
            vl = str(val).lower().strip()
            if not vl or len(vl) <= 1:
                continue
            bucket["samples"].add(vl)
            for word in vl.split():
                if len(word) > 2:
                    bucket["samples"].add(word)
    return per_source


def _text_blob(bucket: Dict[str, set]) -> str:
    return " ".join(sorted(bucket["names"])) + " " + " ".join(sorted(bucket["samples"]))


def build_training_set() -> List[Tuple[str, str]]:
    """
    (text, label) pairs for every source that already has a cached LLM label
    in md_business_labels. Sources never classified yet (label_source() hasn't
    run for them — e.g. via indexing or a /detect-business call) are skipped;
    they'll be picked up once they get a label. The label set here — and
    therefore the model's classes — is exactly the current open taxonomy.
    """
    with _mc._cursor_ctx() as cur:
        _ensure_labels_table(cur)
        rows = cur.execute("SELECT source_id, label FROM md_business_labels").fetchall()
    labels_by_source = {r["source_id"]: r["label"] for r in rows}

    per_source = _fetch_source_texts()
    examples: List[Tuple[str, str]] = []
    for source_id, label in labels_by_source.items():
        bucket = per_source.get(source_id)
        if not bucket or not (bucket["names"] or bucket["samples"]):
            continue
        examples.append((_text_blob(bucket), label))
    return examples


def train(min_examples: int = 4) -> Dict:
    """Train a TF-IDF + Logistic Regression classifier on the bootstrap set and persist it to MODEL_PATH."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    import joblib

    examples = build_training_set()
    if len(examples) < min_examples:
        raise RuntimeError(
            f"Only {len(examples)} bootstrap-labeled datasources available "
            f"(need >= {min_examples}). Index more datasources before training."
        )

    texts, labels = zip(*examples)
    label_counts: Dict[str, int] = defaultdict(int)
    for label in labels:
        label_counts[label] += 1
    if len(label_counts) < 2:
        only = next(iter(label_counts))
        raise RuntimeError(
            f"All bootstrap-labeled datasources fall under a single business "
            f"category ({only!r}) — need at least 2 distinct categories to train a classifier."
        )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    pipeline.fit(texts, labels)

    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    stats = {
        "trained_on":   len(examples),
        "classes":      sorted(label_counts),
        "class_counts": dict(label_counts),
        "model_path":   MODEL_PATH,
    }
    logger.info(
        "business_classifier: trained on %d sources across %d classes -> %s",
        stats["trained_on"], len(stats["classes"]), MODEL_PATH,
    )
    return stats


def is_trained() -> bool:
    return os.path.exists(MODEL_PATH)


def _load_model():
    import joblib
    if not is_trained():
        return None
    return joblib.load(MODEL_PATH)


def predict(source_id: str) -> Optional[Dict]:
    """
    Predict the business/industry for one already-indexed source, from the
    OPEN taxonomy (see module docstring) — never a fixed category list.

    Order of precedence:
      1. Cached LLM label for this exact source (no LLM call — cheap, and
         still fully dynamic since it's whatever label the LLM assigned).
      2. A fresh, live LLM classification (label_source()) — this is the
         source of truth, so it's tried before the ML model on every
         never-before-seen source; caches the result for next time and can
         coin a brand-new taxonomy entry on the spot (e.g. "Human Resources").
      3. Trained ML model — ONLY reached if the LLM call itself failed
         (e.g. transient API error). Its classes are whatever the taxonomy
         currently contains, so it stays roughly in sync, but it's a
         same-request fallback, not the primary decision path.
      4. Keyword-scorer fallback — ONLY reached if the LLM is unreachable at
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

    per_source = _fetch_source_texts()
    bucket = per_source.get(source_id)
    if not bucket or not (bucket["names"] or bucket["samples"]):
        return None
    text = _text_blob(bucket)

    model = _load_model()
    if model is not None:
        proba = model.predict_proba([text])[0]
        classes = model.classes_
        best_idx = proba.argmax()
        return {
            "business":   str(classes[best_idx]),
            "confidence": round(float(proba[best_idx]), 4),
            "method":     "ml-fallback",
        }

    scored = _score_industry(bucket["names"], bucket["samples"])
    if not scored:
        return None
    label, score = scored
    return {"business": label, "confidence": None, "method": "rule-fallback", "score": score}


def backfill() -> Dict:
    """Live-classify (LLM) every indexed source that has no cached label yet. Makes one LLM call per unlabeled source."""
    per_source = _fetch_source_texts()
    labeled, skipped = [], []
    for source_id in per_source:
        result = label_source(source_id)
        if result:
            labeled.append({"source_id": source_id, **result})
        else:
            skipped.append(source_id)
    return {"labeled": len(labeled), "skipped": len(skipped), "results": labeled}


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Business/industry classifier for datasources")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train")
    sub.add_parser("backfill")
    sub.add_parser("labels")
    predict_parser = sub.add_parser("predict")
    predict_parser.add_argument("source_id")
    label_parser = sub.add_parser("label")
    label_parser.add_argument("source_id")
    label_parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    if args.cmd == "train":
        print(json.dumps(train(), indent=2))
    elif args.cmd == "backfill":
        print(json.dumps(backfill(), indent=2))
    elif args.cmd == "labels":
        print(json.dumps(known_labels(), indent=2))
    elif args.cmd == "predict":
        result = predict(args.source_id)
        print(json.dumps(result, indent=2) if result else "null")
    elif args.cmd == "label":
        result = label_source(args.source_id, force=args.force)
        print(json.dumps(result, indent=2) if result else "null")


if __name__ == "__main__":
    _cli()
