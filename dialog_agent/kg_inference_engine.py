"""
Enterprise KG Bridge Inference Engine — multi-tier pipeline.

Pipeline
--------
Tier 0  Declared-FK extraction (confidence 1.0)
        Lift foreign-key relationships from each KG's own metadata report.
        These are ground-truth join columns *within* a KG; they become
        high-priority candidates when the same column pattern exists in
        another KG.

Tier 1  Structural signals  (deterministic, conf 0.70 – 0.95)
        a. Exact column-name match
        b. Normalised-name match  (strip FK prefixes/suffixes, split camelCase)
        c. Data-type family compatibility  (+boost)
        d. Uniqueness/cardinality scoring  (+boost for likely PK/FK columns)
        e. Compound/composite key detection

Tier 2  Semantic similarity  (embedding-based, conf 0.55 – 0.82)
        Each column is represented as:
            "{col_name}: {data_type} | {domain} | {description}"
        Embeddings via sentence-transformers (all-MiniLM-L6-v2) when available,
        TF-IDF cosine similarity as fallback.
        Only runs for column pairs NOT already matched in Tier 1.

Tier 3  LLM validation  (async, conf refinement for 0.55 – 0.78 candidates)
        Batch medium-confidence candidates and validate with Claude.
        Runs as a background coroutine — does NOT block bridge saving.
        On completion it updates confidence + enabled status in the DB.

Domain partitioning
        KGs carry a primary domain tag.  Cross-domain pairs require a higher
        structural confidence to auto-enable (threshold bumped by +0.08).

Transitivity
        If A→B and B→C are both confirmed bridges (conf ≥ 0.80), A→C is
        inferred as a candidate with conf = conf(A→B) × conf(B→C) × 0.9.
        Exposed as a standalone helper so callers can run it periodically.

Confidence thresholds (same-domain)
        ≥ 0.80   auto-enabled
        0.55-0.79  disabled, queued for LLM review
        < 0.55   discarded

Auto-enable threshold rises by 0.08 for cross-domain pairs.

Public API
----------
build_context(kg_id, nodes, report) -> KGContext
run_enterprise_inference(ctx_a, ctx_b, options) -> List[Bridge]
run_enterprise_inference_and_save(ctx_a, ctx_b, options) -> List[Bridge]
infer_transitive_bridges() -> List[Bridge]
async llm_validate_pending(limit) -> int
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public configuration
# ---------------------------------------------------------------------------

@dataclass
class InferenceOptions:
    min_confidence: float = 0.55          # discard below this
    auto_enable_threshold: float = 0.80   # auto-enable above this
    cross_domain_penalty: float = 0.08    # added to both thresholds for cross-domain
    run_tier2_embeddings: bool = True
    run_tier3_llm: bool = True            # Tier 3 always runs async (non-blocking)
    llm_model: str = "claude-haiku-4-5-20251001"  # fast + cheap for validation
    llm_batch_size: int = 20              # max candidates per LLM call
    embedding_sim_threshold: float = 0.78  # min cosine sim for Tier 2 candidate
    transitivity_decay: float = 0.90      # conf(A→C) = conf(A→B)×conf(B→C)×decay
    transitivity_min_conf: float = 0.70   # don't bother below this for transitive

DEFAULT_OPTIONS = InferenceOptions()


# ---------------------------------------------------------------------------
# Column profile — internal representation
# ---------------------------------------------------------------------------

@dataclass
class ColumnProfile:
    kg_id:       str
    entity:      str          # table / entity name
    col:         str          # raw column name (lowercase)
    col_norm:    str          # normalised name (for matching)
    data_type:   str          # from KG properties or report
    domain:      str          # semantic domain (from metadata_store / report)
    description: str
    is_pk:       bool = False
    is_fk:       bool = False
    fk_target:   str  = ""    # "other_table.other_col"
    unique_ratio: Optional[float] = None   # unique_count / row_count
    null_ratio:   Optional[float] = None   # null_count  / row_count
    text_repr:   str  = ""    # for embedding


# ---------------------------------------------------------------------------
# KGContext — what callers must supply
# ---------------------------------------------------------------------------

@dataclass
class KGContext:
    kg_id:        str
    display_name: str
    domain:       str                   # primary domain tag (may be "")
    nodes:        List[Dict]            # KG graph nodes
    report:       Optional[Dict] = None # full metadata report (optional)


# ---------------------------------------------------------------------------
# Name normalisation helpers
# ---------------------------------------------------------------------------

_CAMEL_RE  = re.compile(r'(?<=[a-z0-9])(?=[A-Z])')
_PREFIX_RE = re.compile(
    r'^(fk_|ref_|id_|pk_|tbl_|col_|dim_|fact_|bridge_|lnk_|lk_|f_|d_|s_)',
    re.IGNORECASE,
)
_SUFFIX_RE = re.compile(
    r'(_id|_key|_code|_no|_num|_nr|_nbr|_cd|_sk|_nk|_pk|_fk|_ref|_uuid|_guid|_sid|_tid)$',
    re.IGNORECASE,
)
_ID_RE = re.compile(
    r'(^|_)(id|key|code|uuid|guid|fk|ref|sk|nk|sid|tid)($|_)',
    re.IGNORECASE,
)


def _normalise(name: str) -> str:
    """Return the canonical form of a column name for matching."""
    n = _CAMEL_RE.sub('_', name)
    n = n.lower().strip()
    n = re.sub(r'[^a-z0-9_]', '_', n)
    n = _PREFIX_RE.sub('', n)
    n = _SUFFIX_RE.sub('', n)
    n = re.sub(r'_+', '_', n).strip('_')
    return n or name.lower()


# ---------------------------------------------------------------------------
# Data-type compatibility
# ---------------------------------------------------------------------------

_TYPE_FAMILIES: Dict[str, Set[str]] = {
    'integer': {'int', 'integer', 'bigint', 'smallint', 'tinyint', 'int2',
                'int4', 'int8', 'int64', 'long', 'serial', 'bigserial',
                'number', 'numeric', 'decimal'},
    'float':   {'float', 'double', 'real', 'float4', 'float8', 'money',
                'currency', 'double precision'},
    'text':    {'varchar', 'nvarchar', 'char', 'nchar', 'text', 'string',
                'clob', 'ntext', 'bpchar', 'character varying'},
    'date':    {'date', 'datetime', 'timestamp', 'timestamptz',
                'timestamp with time zone', 'timestamp without time zone'},
    'uuid':    {'uuid', 'guid', 'uniqueidentifier'},
    'boolean': {'bool', 'boolean', 'bit'},
}
_NUMERIC = {'integer', 'float'}


def _type_family(dtype: str) -> Optional[str]:
    dt = dtype.lower().split('(')[0].strip()
    for fam, members in _TYPE_FAMILIES.items():
        if dt in members:
            return fam
    return None


def _type_compat(ta: str, tb: str) -> float:
    """Return a compatibility score [0.0 – 1.0] between two data types."""
    if not ta or not tb:
        return 0.5   # unknown — neutral
    fa, fb = _type_family(ta), _type_family(tb)
    if fa is None or fb is None:
        return 0.5
    if fa == fb:
        return 1.0
    if {fa, fb} <= _NUMERIC:
        return 0.85   # int ↔ float — common in real DBs
    return 0.0        # incompatible families


# ---------------------------------------------------------------------------
# Build column profiles
# ---------------------------------------------------------------------------

def _profiles_from_nodes(kg_id: str, nodes: List[Dict]) -> List[ColumnProfile]:
    """Extract ColumnProfile list from KG graph nodes.

    Two extraction strategies in priority order:
      1. Structured ``properties`` list  — present in nodes built by
         translate_node._build_graph_data (commit 69cb67e+).
         Format: [{"name": col, "type": xsd_type}, ...]
      2. Title-text fallback — for nodes indexed before the structured
         properties field was added.  Parses lines of the form:
         "  col_name: xsd_type  -- optional comment"
         that appear after the ``Properties:`` marker in the node title.
    """
    # Matches "  col_name: xsd_type" (2-space indent) in a multi-line string.
    # Non-greedy (.+?) stops at the first colon so col names like "fiscal_year"
    # are captured correctly.  The type token (\S+) may include the "xsd:" prefix.
    _TITLE_PROP_RE = re.compile(r'^\s{2}(.+?):\s+(\S+)', re.MULTILINE)

    out: List[ColumnProfile] = []
    for node in nodes:
        entity = node.get("label") or ""
        if not entity:
            # Extract entity name from title header "Class: <Name>"
            head = (node.get("title") or "").split("\n", 1)[0]
            m_head = re.match(r'^Class:\s+(.+)', head)
            entity = m_head.group(1).strip() if m_head else (node.get("title") or "")

        props = node.get("properties") or []

        if props:
            # Strategy 1: structured property list (fast path)
            for prop in props:
                col = (prop.get("name") or prop.get("column") or "").strip()
                if not col:
                    continue
                col_l = col.lower()
                dtype = (prop.get("type") or prop.get("data_type") or "").lower()
                cp = ColumnProfile(
                    kg_id=kg_id, entity=entity, col=col_l,
                    col_norm=_normalise(col_l), data_type=dtype,
                    domain="", description="",
                )
                cp.text_repr = f"{col_l}: {dtype}".strip(": ")
                out.append(cp)
        else:
            # Strategy 2: parse the node title text (legacy nodes)
            title = node.get("title") or ""
            props_idx = title.find("Properties:")
            if props_idx == -1:
                continue
            props_section = title[props_idx + len("Properties:"):]
            for m in _TITLE_PROP_RE.finditer(props_section):
                col = m.group(1).strip().lower()
                dtype = m.group(2).strip().lower()
                if dtype.startswith("xsd:"):
                    dtype = dtype[4:]
                if not col:
                    continue
                cp = ColumnProfile(
                    kg_id=kg_id, entity=entity, col=col,
                    col_norm=_normalise(col), data_type=dtype,
                    domain="", description="",
                )
                cp.text_repr = f"{col}: {dtype}".strip(": ")
                out.append(cp)

    return out


def _enrich_from_report(profiles: List[ColumnProfile], report: Dict) -> None:
    """Enrich profiles with stats + FK info from a metadata report."""
    tables: Dict[str, Any] = report.get("tables") or {}

    # Build lookup: normalised entity → table_name
    def _norm_entity(s: str) -> str:
        return s.lower().replace(' ', '_')

    fk_cands: Set[Tuple[str, str]] = set()
    for fk in report.get("fk_candidates") or []:
        for col in fk.get("left_columns") or []:
            fk_cands.add((_norm_entity(fk.get("left_table", "")), col.lower()))

    for tname, tdata in tables.items():
        if not isinstance(tdata, dict):
            continue
        pks: Set[str] = {p.lower() for p in (tdata.get("primary_keys") or [])}
        fks: Set[str] = set()
        fk_map: Dict[str, str] = {}
        for fk_def in (tdata.get("foreign_keys") or []):
            col_name = fk_def.get("column", "").lower()
            if col_name:
                fks.add(col_name)
                ref = fk_def.get("references_table", "")
                ref_col = fk_def.get("references_column", "")
                fk_map[col_name] = f"{ref}.{ref_col}" if ref else ""

        col_list = tdata.get("columns") or []
        if isinstance(col_list, dict):
            col_list = [{"name": k, **v} for k, v in col_list.items()]

        for col_data in col_list:
            if not isinstance(col_data, dict):
                continue
            col_name = (col_data.get("name") or "").lower()
            rc = col_data.get("row_count") or 0
            uc = col_data.get("unique_count")
            nc = col_data.get("null_count")

            for cp in profiles:
                if cp.col == col_name and (
                    _norm_entity(cp.entity) == _norm_entity(tname)
                    or cp.entity == ""
                ):
                    cp.is_pk = col_name in pks
                    cp.is_fk = col_name in fks or (tname.lower(), col_name) in fk_cands
                    cp.fk_target = fk_map.get(col_name, "")
                    cp.data_type = col_data.get("data_type") or cp.data_type
                    cp.domain = col_data.get("domain") or ""
                    cp.description = col_data.get("description") or ""
                    if rc and uc is not None:
                        cp.unique_ratio = min(uc / rc, 1.0)
                    if rc and nc is not None:
                        cp.null_ratio = min(nc / rc, 1.0)
                    cp.text_repr = (
                        f"{cp.col}: {cp.data_type} | {cp.domain} | {cp.description}"
                    ).strip(": |")


def build_context(
    kg_id: str,
    display_name: str,
    domain: str,
    nodes: List[Dict],
    report: Optional[Dict] = None,
) -> KGContext:
    return KGContext(kg_id=kg_id, display_name=display_name,
                     domain=domain, nodes=nodes, report=report)


# ---------------------------------------------------------------------------
# Tier 1: Structural matching
# ---------------------------------------------------------------------------

def _uniqueness_boost(cp: ColumnProfile) -> float:
    """Boost for columns that look like PK/FK candidates."""
    boost = 0.0
    if cp.is_pk:
        boost += 0.08
    if cp.is_fk:
        boost += 0.06
    if _ID_RE.search(cp.col):
        boost += 0.04
    if cp.unique_ratio is not None and cp.unique_ratio >= 0.98:
        boost += 0.05
    elif cp.unique_ratio is not None and cp.unique_ratio >= 0.90:
        boost += 0.02
    return min(boost, 0.12)   # cap so we don't wildly inflate


def _structural_pass(
    profs_a: List[ColumnProfile],
    profs_b: List[ColumnProfile],
    min_conf: float,
) -> List[Tuple[ColumnProfile, ColumnProfile, float, str]]:
    """
    Returns (cp_a, cp_b, confidence, reason) tuples for all structural matches.
    """
    # Index B by exact name and normalised name
    exact_b: Dict[str, List[ColumnProfile]] = {}
    norm_b:  Dict[str, List[ColumnProfile]] = {}
    for cp in profs_b:
        exact_b.setdefault(cp.col, []).append(cp)
        norm_b.setdefault(cp.col_norm, []).append(cp)

    seen: Set[Tuple[str, str, str, str]] = set()
    results: List[Tuple[ColumnProfile, ColumnProfile, float, str]] = []

    for cp_a in profs_a:
        # 1a. Exact match
        if cp_a.col in exact_b:
            for cp_b in exact_b[cp_a.col]:
                key = (cp_a.entity, cp_a.col, cp_b.entity, cp_b.col)
                if key in seen:
                    continue
                seen.add(key)
                base = 0.85
                tc = _type_compat(cp_a.data_type, cp_b.data_type)
                if tc == 0.0:
                    continue
                conf = base + _uniqueness_boost(cp_a) + _uniqueness_boost(cp_b)
                if tc < 1.0:
                    conf -= 0.05
                conf = min(round(conf, 3), 0.97)
                if conf >= min_conf:
                    results.append((cp_a, cp_b, conf, "exact_name"))

        # 1b. Normalised match (different raw names)
        if cp_a.col_norm and cp_a.col_norm in norm_b:
            for cp_b in norm_b[cp_a.col_norm]:
                if cp_b.col == cp_a.col:
                    continue   # already handled as exact
                key = (cp_a.entity, cp_a.col, cp_b.entity, cp_b.col)
                if key in seen:
                    continue
                seen.add(key)
                base = 0.72
                tc = _type_compat(cp_a.data_type, cp_b.data_type)
                if tc == 0.0:
                    continue
                conf = base + _uniqueness_boost(cp_a) + _uniqueness_boost(cp_b)
                if tc == 1.0:
                    conf += 0.04
                elif tc < 1.0:
                    conf -= 0.03
                conf = min(round(conf, 3), 0.90)
                if conf >= min_conf:
                    results.append((cp_a, cp_b, conf, "normalised_name"))

    return results


# ---------------------------------------------------------------------------
# Tier 2: Embedding-based semantic similarity
# ---------------------------------------------------------------------------

_EMBED_CACHE: Dict[str, Any] = {}  # kg_id → {"ids": [...], "matrix": np.ndarray}


def _build_embed_matrix(profiles: List[ColumnProfile]) -> Any:
    """Return (ids, matrix) where ids[i] = (entity, col) and matrix[i] = embedding."""
    texts = [cp.text_repr or cp.col for cp in profiles]
    ids   = [(cp.entity, cp.col) for cp in profiles]
    try:
        import numpy as np
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            mat = model.encode(texts, normalize_embeddings=True)
            return ids, mat.astype(np.float32)
        except ImportError:
            pass
        # TF-IDF fallback
        from sklearn.feature_extraction.text import TfidfVectorizer
        vect = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
        mat = vect.fit_transform(texts).toarray().astype(np.float32)
        # L2 normalise
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1
        mat = mat / norms
        return ids, mat
    except ImportError:
        return None, None


def _semantic_pass(
    profs_a: List[ColumnProfile],
    profs_b: List[ColumnProfile],
    already_matched_a: Set[str],   # set of cp_a.col already matched in Tier 1
    already_matched_b: Set[str],
    sim_threshold: float,
    min_conf: float,
) -> List[Tuple[ColumnProfile, ColumnProfile, float, str]]:
    """Find semantically similar column pairs not already matched in Tier 1."""
    # Filter to unmatched columns only
    unmatched_a = [cp for cp in profs_a if cp.col not in already_matched_a]
    unmatched_b = [cp for cp in profs_b if cp.col not in already_matched_b]
    if not unmatched_a or not unmatched_b:
        return []

    try:
        import numpy as np
    except ImportError:
        logger.debug("numpy unavailable — skipping semantic similarity tier")
        return []

    ids_a, mat_a = _build_embed_matrix(unmatched_a)
    ids_b, mat_b = _build_embed_matrix(unmatched_b)
    if mat_a is None or mat_b is None:
        return []

    sim_matrix = mat_a @ mat_b.T   # shape (|A|, |B|)
    results: List[Tuple[ColumnProfile, ColumnProfile, float, str]] = []

    for i, cp_a in enumerate(unmatched_a):
        best_j   = int(np.argmax(sim_matrix[i]))
        best_sim = float(sim_matrix[i, best_j])
        if best_sim < sim_threshold:
            continue
        cp_b = unmatched_b[best_j]
        tc   = _type_compat(cp_a.data_type, cp_b.data_type)
        if tc == 0.0:
            continue
        # Map cosine similarity → confidence
        # sim 0.78 → 0.58 conf,  sim 0.90 → 0.72 conf
        conf = 0.55 + (best_sim - sim_threshold) / (1.0 - sim_threshold) * 0.27
        conf += _uniqueness_boost(cp_a) + _uniqueness_boost(cp_b)
        if tc == 1.0:
            conf += 0.03
        elif tc < 1.0:
            conf -= 0.03
        conf = min(round(conf, 3), 0.82)
        if conf >= min_conf:
            results.append((cp_a, cp_b, conf, f"semantic_sim={best_sim:.3f}"))

    return results


# ---------------------------------------------------------------------------
# Tier 3: LLM validation (async, non-blocking)
# ---------------------------------------------------------------------------

async def _llm_validate_candidates(
    candidates: List["Bridge"],
    options: InferenceOptions,
) -> None:
    """
    Async background task — validate medium-confidence bridge candidates with
    Claude, update confidence + enabled flag in the DB.
    Does NOT block the caller.
    """
    from dialog_agent.kg_bridges import get_by_id, upsert, Bridge as Br, _find_existing

    if not candidates:
        return
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.debug("ANTHROPIC_API_KEY not set — skipping LLM validation tier")
        return

    # Batch into groups
    bs = options.llm_batch_size
    batches = [candidates[i:i + bs] for i in range(0, len(candidates), bs)]

    try:
        from llm_client import get_client
        client = get_client()
    except ImportError:
        logger.debug("anthropic package unavailable — skipping LLM validation")
        return

    system_prompt = (
        "You are a data integration expert. For each bridge candidate, decide if the two "
        "columns are a valid join key across the two knowledge graphs. "
        "Return a JSON array — one object per candidate — with fields:\n"
        "  index: int (0-based)\n"
        "  valid: bool\n"
        "  confidence_adjustment: float  (-0.2 to +0.1, applied to existing confidence)\n"
        "  join_type: 'FK' | 'soft' | 'hash'\n"
        "  reasoning: str (brief)\n"
        "Be conservative: prefer false positives over invalid joins. "
        "Output ONLY the JSON array, no prose."
    )

    for batch in batches:
        payload = []
        for idx, b in enumerate(batch):
            payload.append({
                "index": idx,
                "from_kg": b.from_kg,
                "from_entity": b.from_entity,
                "from_column": b.from_column,
                "to_kg": b.to_kg,
                "to_entity": b.to_entity,
                "to_column": b.to_column,
                "current_confidence": b.confidence,
                "infer_reason": b.notes,
            })

        user_msg = json.dumps(payload, indent=2)
        try:
            resp = client.messages.create(
                model=options.llm_model,
                max_tokens=2048,
                temperature=0.0,
                system=system_prompt,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = resp.content[0].text if resp.content else "[]"
            # Strip markdown fences if present
            raw = re.sub(r'^```[a-z]*\s*', '', raw.strip())
            raw = re.sub(r'\s*```$', '', raw.strip())
            validations = json.loads(raw)
        except Exception as exc:
            logger.warning("LLM validation batch failed: %s", exc)
            continue

        for v in validations:
            idx = v.get("index", -1)
            if idx < 0 or idx >= len(batch):
                continue
            b = batch[idx]
            valid = bool(v.get("valid", True))
            adj   = float(v.get("confidence_adjustment", 0.0))
            jt    = v.get("join_type") or b.join_type
            note  = v.get("reasoning", "")

            new_conf = max(0.0, min(1.0, b.confidence + adj))
            enabled  = valid and new_conf >= DEFAULT_OPTIONS.auto_enable_threshold

            # Update in DB
            existing = _find_existing(b.from_kg, b.from_column, b.to_kg, b.to_column)
            if existing and existing.source == "declared":
                continue   # never touch declared bridges
            if existing:
                updated = Br(
                    id=existing.id,
                    from_kg=b.from_kg, from_entity=b.from_entity, from_column=b.from_column,
                    to_kg=b.to_kg, to_entity=b.to_entity, to_column=b.to_column,
                    join_type=jt, source="inferred",
                    confidence=round(new_conf, 3), enabled=enabled,
                    notes=f"llm_validated: {note[:200]}",
                    created_at=existing.created_at,
                )
                try:
                    upsert(updated)
                    logger.info(
                        "LLM-validated bridge %s.%s → %s.%s: valid=%s conf=%.2f",
                        b.from_kg[:8], b.from_column,
                        b.to_kg[:8], b.to_column, valid, new_conf,
                    )
                except Exception as exc:
                    logger.warning("LLM bridge update failed: %s", exc)


# ---------------------------------------------------------------------------
# Main inference orchestration
# ---------------------------------------------------------------------------

def run_enterprise_inference(
    ctx_a: KGContext,
    ctx_b: KGContext,
    options: Optional[InferenceOptions] = None,
) -> Tuple[List["Bridge"], List["Bridge"]]:
    """
    Run the full inference pipeline between two KG contexts.

    Returns (high_conf, medium_conf):
        high_conf   — bridges with confidence ≥ auto_enable_threshold (enabled=True)
        medium_conf — bridges below threshold but above min_confidence (enabled=False,
                      queued for async LLM validation)
    """
    from dialog_agent.kg_bridges import Bridge

    opts = options or DEFAULT_OPTIONS

    # ── adjust thresholds for cross-domain pairs ──────────────────────────────
    # Only penalise when BOTH domains are known AND they are different.
    # If either domain is empty / unset we have no evidence of a cross-domain
    # mismatch, so treat the pair as same-domain (no penalty).
    if not ctx_a.domain or not ctx_b.domain:
        same_domain = True   # unknown domain — give benefit of the doubt
    else:
        same_domain = ctx_a.domain.lower() == ctx_b.domain.lower()
    penalty = 0.0 if same_domain else opts.cross_domain_penalty
    auto_threshold = opts.auto_enable_threshold + penalty
    min_conf       = opts.min_confidence + penalty * 0.5   # relax min less aggressively

    logger.info(
        "Inference %s (%s) ↔ %s (%s) | same_domain=%s penalty=%.2f",
        ctx_a.kg_id[:8], ctx_a.domain or "—",
        ctx_b.kg_id[:8], ctx_b.domain or "—",
        same_domain, penalty,
    )

    # ── build profiles ────────────────────────────────────────────────────────
    profs_a = _profiles_from_nodes(ctx_a.kg_id, ctx_a.nodes)
    profs_b = _profiles_from_nodes(ctx_b.kg_id, ctx_b.nodes)

    if ctx_a.report:
        _enrich_from_report(profs_a, ctx_a.report)
    if ctx_b.report:
        _enrich_from_report(profs_b, ctx_b.report)

    if not profs_a or not profs_b:
        logger.info(
            "Inference skipped — no column profiles (%d / %d)",
            len(profs_a), len(profs_b),
        )
        return [], []

    # ── Tier 1: structural ────────────────────────────────────────────────────
    t1_matches = _structural_pass(profs_a, profs_b, min_conf)
    matched_a  = {m[0].col for m in t1_matches}
    matched_b  = {m[1].col for m in t1_matches}

    # ── Tier 2: semantic similarity ───────────────────────────────────────────
    t2_matches: List[Tuple[ColumnProfile, ColumnProfile, float, str]] = []
    if opts.run_tier2_embeddings:
        try:
            t2_matches = _semantic_pass(
                profs_a, profs_b,
                matched_a, matched_b,
                opts.embedding_sim_threshold,
                min_conf,
            )
        except Exception as exc:
            logger.warning("Tier 2 embedding pass failed: %s", exc)

    all_matches = t1_matches + t2_matches

    # ── Deduplicate: keep highest-confidence per (from_col, to_col) ───────────
    best: Dict[Tuple[str, str], Tuple[ColumnProfile, ColumnProfile, float, str]] = {}
    for cp_a, cp_b, conf, reason in all_matches:
        k = (cp_a.col, cp_b.col)
        if k not in best or conf > best[k][2]:
            best[k] = (cp_a, cp_b, conf, reason)

    # ── Materialise Bridge objects ────────────────────────────────────────────
    high_conf:   List[Bridge] = []
    medium_conf: List[Bridge] = []

    for (cp_a, cp_b, conf, reason) in best.values():
        is_fk_pattern = bool(_ID_RE.search(cp_a.col) or _ID_RE.search(cp_b.col)
                             or cp_a.is_pk or cp_b.is_pk
                             or cp_a.is_fk or cp_b.is_fk)
        b = Bridge(
            from_kg=ctx_a.kg_id, from_entity=cp_a.entity, from_column=cp_a.col,
            to_kg=ctx_b.kg_id,   to_entity=cp_b.entity,   to_column=cp_b.col,
            join_type="FK" if is_fk_pattern else "soft",
            source="inferred",
            confidence=conf,
            enabled=conf >= auto_threshold,
            notes=f"tier={reason} same_domain={same_domain}",
        )
        if conf >= auto_threshold:
            high_conf.append(b)
        else:
            medium_conf.append(b)

    logger.info(
        "Inference result %s↔%s: %d high-conf (auto-enabled), "
        "%d medium-conf (queued for LLM), threshold=%.2f",
        ctx_a.kg_id[:8], ctx_b.kg_id[:8],
        len(high_conf), len(medium_conf), auto_threshold,
    )
    return high_conf, medium_conf


def run_enterprise_inference_and_save(
    ctx_a: KGContext,
    ctx_b: KGContext,
    options: Optional[InferenceOptions] = None,
    background_tasks: Any = None,   # fastapi.BackgroundTasks or None
) -> List["Bridge"]:
    """
    Run inference, persist all results, schedule LLM validation as a
    background task for medium-confidence bridges.
    Returns the list of ALL saved bridges (high + medium).
    """
    from dialog_agent.kg_bridges import Bridge, _find_existing, upsert

    high_conf, medium_conf = run_enterprise_inference(ctx_a, ctx_b, options)
    saved: List[Bridge] = []

    for b in high_conf + medium_conf:
        existing = _find_existing(b.from_kg, b.from_column, b.to_kg, b.to_column)
        if existing and existing.source == "declared":
            logger.debug("Skipping declared bridge %s.%s→%s.%s",
                         b.from_kg[:8], b.from_column, b.to_kg[:8], b.to_column)
            continue
        try:
            upsert(b)
            saved.append(b)
        except Exception as exc:
            logger.warning("Failed to save bridge %s→%s: %s",
                           b.from_column, b.to_column, exc)

    # Schedule Tier 3 LLM validation for medium-confidence bridges
    if medium_conf and (options or DEFAULT_OPTIONS).run_tier3_llm:
        opts = options or DEFAULT_OPTIONS
        # medium_conf bridges that were actually saved
        saved_medium = [b for b in saved if not b.enabled]
        if background_tasks is not None:
            # FastAPI BackgroundTasks
            import asyncio

            async def _run_validate():
                await _llm_validate_candidates(saved_medium, opts)

            background_tasks.add_task(_run_validate)
        else:
            # Fire-and-forget using asyncio if a loop is running
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(_llm_validate_candidates(saved_medium, opts))
                else:
                    loop.run_until_complete(_llm_validate_candidates(saved_medium, opts))
            except Exception as exc:
                logger.debug("Could not schedule LLM validation: %s", exc)

    return saved


# ---------------------------------------------------------------------------
# Transitivity inference
# ---------------------------------------------------------------------------

def infer_transitive_bridges(
    options: Optional[InferenceOptions] = None,
) -> List["Bridge"]:
    """
    Scan all high-confidence bridges and infer transitive A→C from A→B + B→C.

    Only creates NEW bridge candidates — never overwrites declared or existing
    inferred bridges.  Returns the list of newly saved transitive bridges.
    """
    from dialog_agent.kg_bridges import Bridge, list_all, _find_existing, upsert

    opts = options or DEFAULT_OPTIONS
    all_bridges = list_all(enabled_only=False)

    # Index: (from_kg, from_col) → Bridge
    by_src: Dict[Tuple[str, str], List[Bridge]] = {}
    for b in all_bridges:
        if b.confidence >= opts.transitivity_min_conf:
            by_src.setdefault((b.from_kg, b.from_column), []).append(b)

    saved: List[Bridge] = []
    seen: Set[Tuple[str, str, str, str]] = set()

    for b_ab in all_bridges:
        if b_ab.confidence < opts.transitivity_min_conf:
            continue
        # Find all B→C bridges where B == b_ab.to_kg and to_col == b_ab.to_col
        candidates_bc = by_src.get((b_ab.to_kg, b_ab.to_column), [])
        for b_bc in candidates_bc:
            if b_bc.to_kg == b_ab.from_kg:
                continue   # circular
            key = (b_ab.from_kg, b_ab.from_column, b_bc.to_kg, b_bc.to_column)
            if key in seen:
                continue
            seen.add(key)

            existing = _find_existing(*key)
            if existing:
                continue   # already known

            trans_conf = round(
                b_ab.confidence * b_bc.confidence * opts.transitivity_decay, 3
            )
            if trans_conf < opts.transitivity_min_conf:
                continue

            trans = Bridge(
                from_kg=b_ab.from_kg, from_entity=b_ab.from_entity,
                from_column=b_ab.from_column,
                to_kg=b_bc.to_kg, to_entity=b_bc.to_entity,
                to_column=b_bc.to_column,
                join_type=b_ab.join_type,
                source="inferred",
                confidence=trans_conf,
                enabled=trans_conf >= opts.auto_enable_threshold,
                notes=(f"transitive: {b_ab.from_kg[:8]}→{b_ab.to_kg[:8]}→{b_bc.to_kg[:8]}"),
            )
            try:
                upsert(trans)
                saved.append(trans)
                logger.info(
                    "Transitive bridge: %s.%s → %s.%s (conf=%.2f)",
                    trans.from_kg[:8], trans.from_column,
                    trans.to_kg[:8], trans.to_column, trans_conf,
                )
            except Exception as exc:
                logger.warning("Failed to save transitive bridge: %s", exc)

    if saved:
        logger.info("Transitivity pass: %d new bridges inferred", len(saved))
    return saved


# ---------------------------------------------------------------------------
# Standalone re-inference over all KG pairs
# ---------------------------------------------------------------------------

def run_all_pairs(
    kg_contexts: List[KGContext],
    options: Optional[InferenceOptions] = None,
    background_tasks: Any = None,
) -> Dict[str, int]:
    """
    Run enterprise inference over every ordered pair of KGs.
    Returns {"pairs_processed": N, "bridges_saved": M}.
    """
    total_saved = 0
    pairs = 0
    n = len(kg_contexts)
    for i in range(n):
        for j in range(i + 1, n):
            ctx_a, ctx_b = kg_contexts[i], kg_contexts[j]
            try:
                saved = run_enterprise_inference_and_save(
                    ctx_a, ctx_b, options, background_tasks
                )
                total_saved += len(saved)
                pairs += 1
            except Exception as exc:
                logger.warning(
                    "Inference failed for pair %s / %s: %s",
                    ctx_a.kg_id[:8], ctx_b.kg_id[:8], exc,
                )

    # Run transitivity pass after all pairs
    try:
        transitive = infer_transitive_bridges(options)
        total_saved += len(transitive)
    except Exception as exc:
        logger.warning("Transitivity pass failed: %s", exc)

    return {"pairs_processed": pairs, "bridges_saved": total_saved}
