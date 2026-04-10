"""
LangGraph node: run FD detection, IND detection, and cardinality analysis.

This is the most compute-intensive node.  It:
  1. Runs FunctionalDependencyTool on each table individually.
  2. Runs InclusionDependencyTool for every ordered pair of tables.
  3. Runs CardinalityAnalyzerTool for every unordered pair of tables.

Results are stored in state['func_deps'], state['incl_deps'],
and state['cardinalities'].
"""
from __future__ import annotations

import itertools
import json
import logging
import re
from typing import Any, Dict, List

from ..state import (
    AgentState,
    CardinalityRelationship,
    FunctionalDependency,
    InclusionDependency,
    TableMeta,
)
from ..tools import (
    CardinalityAnalyzerTool,
    FunctionalDependencyTool,
    InclusionDependencyTool,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ── Fact table name keywords ──────────────────────────────────────────────
# Strong prior: table name contains one of these → likely a fact or aggregate
_FACT_NAME_RE = re.compile(
    r'\b(fact|fct|measure|metric|kpi)\b', re.IGNORECASE
)
# Weaker prior: aggregate / periodic summary table names
_AGG_NAME_RE = re.compile(
    r'\b(agg|aggregat|summary|summ|snapshot|trans|event|'
    r'daily|weekly|monthly|hourly|annual|yearly)\b',
    re.IGNORECASE,
)

# ── Measure column name keywords ──────────────────────────────────────────
# Named measures: keyword in column name + numeric dtype → clear additive measure
_MEASURE_COL_RE = re.compile(
    r'\b(sales|revenue|qty|quantity|amount|volume|price|cost|'
    r'count|total|value|profit|margin|units|spend|budget|forecast|'
    r'actual|transaction|order|balance|inventory|stock|weight|'
    r'rate|ratio|score|index|pct|percent|share|growth|diff|variance|'
    r'dist|distribution|avg|average|sum|fee|charge|tax|discount|'
    r'return|conversion|impression|click|session|visit|call|event|'
    r'duration|supply|demand)\b',
    re.IGNORECASE,
)

# ── FK / dimension-reference column suffixes ──────────────────────────────
_FK_COL_RE = re.compile(
    r'(_id|_key|_sk|_nk|_bk|_fk|_wid|_code|_ref|_num)\s*$',
    re.IGNORECASE,
)

# ── Time / grain column name keywords ─────────────────────────────────────
_TIME_COL_RE = re.compile(
    r'\b(date|month|year|week|quarter|period|fiscal|timestamp|time)\b',
    re.IGNORECASE,
)

# ── SCD2 effective / expiry date column patterns ──────────────────────────
# Presence of BOTH an effective-date col AND an expiry/end-date col is a
# strong signal that this is a slowly-changing dimension, not a fact table.
#
# NOTE: do NOT use \b word boundaries here — underscores are word-chars in
# Python regex, so \beffective\b fails to match "effective_date".
# Plain substring matching is intentional and sufficient.
_SCD_EFF_RE = re.compile(
    r'effective|eff_|valid_from|start_date|start_dt|active_from|row_start',
    re.IGNORECASE,
)
_SCD_EXP_RE = re.compile(
    r'expir|valid_to|end_date|end_dt|active_to|row_end|close_date',
    re.IGNORECASE,
)
# is_current / current_flag is an unambiguous SCD2 marker on its own.
_SCD_CURRENT_RE = re.compile(
    r'is_current|current_flag|current_ind|current_row|is_active_row',
    re.IGNORECASE,
)

# ── Numeric dtypes ────────────────────────────────────────────────────────
_NUMERIC_DTYPE_RE = re.compile(
    r'\b(int|integer|bigint|smallint|tinyint|number|numeric|decimal|'
    r'float|double|real|money)\b',
    re.IGNORECASE,
)

# ── Text / string dtypes ──────────────────────────────────────────────────
_TEXT_DTYPE_RE = re.compile(
    r'\b(varchar|nvarchar|char|nchar|text|ntext|string|clob|character)\b',
    re.IGNORECASE,
)

# ── Boolean / flag column patterns ───────────────────────────────────────
_BOOL_DTYPE_RE = re.compile(r'\b(bool|boolean|bit)\b', re.IGNORECASE)
_BOOL_COL_RE   = re.compile(
    r'^(is_|has_|flag_|ind_|can_)|\b(flag|active|enabled|deleted|'
    r'current|latest|archived|is_current)\b',
    re.IGNORECASE,
)


def _score_fact_table(table_meta: TableMeta) -> int:  # noqa: C901
    """
    Score a table using column-level sampled metadata to determine whether it
    is a fact table (any Kimball variant) or a dimension/lookup.

    Returns an integer; tables scoring >= 2 are treated as fact tables.

    ── POSITIVE signals (fact indicators) ───────────────────────────────────
    +2  table name matches _FACT_NAME_RE  (fact/fct/measure/metric/kpi)
    +1  table name matches _AGG_NAME_RE   (aggregate/snapshot/event name)
    +1  ≥1 NAMED measure column  (_MEASURE_COL_RE keyword + numeric dtype)
    +1  ≥2 NAMED measure columns (strongly additive, clearly a fact)
    +1  ≥2 generic inferred measure columns (numeric, not FK, not time, not
        boolean, uniqueness_ratio < 0.95 when available — catches domain-
        specific abbreviations like `orders`, `9l_vol`, `net_rev`)
    +1  ≥2 FK / dimension-reference columns (_id/_key/_sk/… or is_foreign_key)
    +1  ≥1 time grain column (_TIME_COL_RE)
    +1  ≥3 time/date columns → accumulating snapshot (milestone dates)
    +1  no single non-FK column with uniqueness_ratio ≥ 0.95 → composite
        grain, typical of facts  [only fires when row_count > 0]
    +1  factless / bridge signal: ≥2 FK cols AND zero named measures AND
        zero generic measures → event/coverage/bridge table; fires even
        when sampling data is absent

    ── NEGATIVE signals (dimension / lookup indicators) ─────────────────────
    -1  ≤3 total columns (tiny lookup table)
    -1  text-heavy: >55% of columns have a text/string dtype (dimension attr)
    -1  SCD2 pattern: table has BOTH an effective-date col AND an expiry-date
        col by name (slowly-changing dimension, not a fact)
    -2  strong dimension: table has exactly one declared PK column
        (len(primary_keys) == 1) AND text-heavy (text_ratio > 0.40) AND
        fewer than 2 named measure columns → very likely a standard dimension
    """
    score = 0
    name = table_meta.table_name

    # ── Name priors ───────────────────────────────────────────────────────
    if _FACT_NAME_RE.search(name):
        score += 2
    elif _AGG_NAME_RE.search(name):
        score += 1

    cols = table_meta.columns
    if not cols:
        return score

    n_cols    = len(cols)
    row_count = table_meta.row_count or 0

    named_measure_cols   = 0
    generic_measure_cols = 0
    fk_cols              = 0
    time_cols            = 0
    text_cols            = 0
    scd_eff_found        = False
    scd_exp_found        = False
    scd_current_found    = False
    has_unique_col       = False   # any non-FK col with uniqueness_ratio ≥ 0.95
    # Count how many distinct FK-named columns reference different entity types.
    # Surrogate+natural key pairs for the SAME entity both end in _sk/_nk or
    # _key/_id — we detect this to avoid the factless signal firing on SCD dims.
    fk_root_names: set = set()

    for c in cols:
        col   = c.name
        dtype = c.data_type
        is_fk_named  = bool(_FK_COL_RE.search(col))
        is_numeric   = bool(_NUMERIC_DTYPE_RE.search(dtype))
        is_text      = bool(_TEXT_DTYPE_RE.search(dtype))
        is_time_col  = bool(_TIME_COL_RE.search(col))
        is_bool      = bool(_BOOL_DTYPE_RE.search(dtype)) or bool(_BOOL_COL_RE.search(col))

        # Named measure: keyword match + numeric dtype
        if _MEASURE_COL_RE.search(col) and is_numeric:
            named_measure_cols += 1

        # FK / dimension reference
        if is_fk_named or c.is_foreign_key:
            fk_cols += 1
            # Strip trailing suffix to get the entity root name
            # e.g. customer_sk → customer, customer_nk → customer (same root)
            root = _FK_COL_RE.sub('', col).rstrip('_').lower()
            fk_root_names.add(root)

        # Time grain
        if is_time_col:
            time_cols += 1

        # Text column ratio
        if is_text:
            text_cols += 1

        # SCD2 signals — checked on all columns (not just time/bool)
        # because column names like "effective_date" and "expiry_date"
        # match both time AND SCD patterns simultaneously.
        if _SCD_EFF_RE.search(col):
            scd_eff_found = True
        if _SCD_EXP_RE.search(col):
            scd_exp_found = True
        if _SCD_CURRENT_RE.search(col):
            scd_current_found = True

        # Generic inferred measure: numeric, not FK, not time, not bool,
        # and (when available) uniqueness_ratio < 0.95 (it varies per row)
        if (
            is_numeric
            and not is_fk_named
            and not c.is_foreign_key
            and not c.is_primary_key
            and not is_time_col
            and not is_bool
        ):
            u_ratio = (
                c.unique_count / row_count
                if c.unique_count is not None and row_count > 0
                else None
            )
            # Exclude columns that look like a numeric surrogate/natural key
            # (very high uniqueness with no FK suffix → e.g. a numeric PK in a dim)
            if u_ratio is None or u_ratio < 0.95:
                generic_measure_cols += 1

        # Natural-key detection (for composite-grain signal below):
        # A column has uniqueness ≈ 1.0 and is NOT an FK-named column →
        # this table may have a natural unique key, typical of a dimension.
        # We deliberately include PK-flagged columns here so that a dim's
        # surrogate key (e.g. customer_key, which also ends in _key matching
        # _FK_COL_RE) does NOT blindly suppress this signal.
        if (
            c.is_primary_key
            and c.unique_count is not None
            and row_count > 0
            and c.unique_count / row_count >= 0.95
        ):
            has_unique_col = True
        elif (
            not is_fk_named
            and not c.is_primary_key
            and c.unique_count is not None
            and row_count > 0
            and c.unique_count / row_count >= 0.95
        ):
            has_unique_col = True

    text_ratio = text_cols / n_cols if n_cols else 0.0

    # ── Positive signals ──────────────────────────────────────────────────
    if named_measure_cols >= 1:
        score += 1
    if named_measure_cols >= 2:
        score += 1   # double-signal: clearly measure-heavy
    if generic_measure_cols >= 2:
        score += 1
    if fk_cols >= 2:
        score += 1
    if time_cols >= 1:
        score += 1
    if time_cols >= 3:
        score += 1   # accumulating snapshot (multiple milestone dates)
    if not has_unique_col and row_count > 0:
        score += 1   # composite grain — no single unique identifier

    # Factless / bridge fact: ≥2 FK cols referencing ≥2 DISTINCT entity roots,
    # zero measures of any kind → event / coverage / bridge table.
    # Requiring ≥2 distinct roots prevents SCD dims (customer_sk + customer_nk
    # → same root "customer") from triggering this signal.
    # Does not require row_count.
    distinct_fk_roots = len(fk_root_names)
    if fk_cols >= 2 and distinct_fk_roots >= 2 and named_measure_cols == 0 and generic_measure_cols == 0:
        score += 1

    # ── Negative signals ──────────────────────────────────────────────────
    if n_cols <= 3:
        score -= 1
    if text_ratio > 0.55:
        score -= 1

    # SCD2 slowly-changing dimension:
    # -1 for effective+expiry date pair; -1 more if is_current flag also present
    # (belt-and-suspenders: even one of these alongside a single PK is enough).
    is_scd2 = scd_eff_found and scd_exp_found
    if is_scd2:
        score -= 1
    if scd_current_found:
        score -= 1   # is_current / current_flag is an unambiguous SCD2 marker

    # Strong dimension penalty: single declared PK + text-heavy + few measures
    # → almost certainly a standard or conformed dimension.
    single_pk = len(table_meta.primary_keys) == 1
    if single_pk and text_ratio > 0.40 and named_measure_cols < 2:
        score -= 2

    return score


def _infer_fact_tables(table_metadata: Dict[str, TableMeta]) -> Dict[str, bool]:
    """
    Return a mapping of table_name → is_fact for every table in table_metadata.
    Tables are classified as fact if their score >= 2 (see _score_fact_table).
    Logs the score for each table at DEBUG level for transparency.
    """
    result: Dict[str, bool] = {}
    for name, meta in table_metadata.items():
        score = _score_fact_table(meta)
        is_fact = score >= 2
        result[name] = is_fact
        logger.debug(
            "fact-inference: %s → score=%d is_fact=%s", name, score, is_fact
        )
    return result


def _col_dicts(table_meta: TableMeta) -> List[Dict[str, str]]:
    return [{"name": c.name, "data_type": c.data_type} for c in table_meta.columns]


def _col_names(table_meta: TableMeta) -> List[str]:
    return [c.name for c in table_meta.columns]


def _col_stats(table_meta: TableMeta) -> Dict[str, Any]:
    out = {}
    for c in table_meta.columns:
        row_count = (c.row_count or table_meta.row_count) or 1
        null_count = c.null_count or 0
        unique_count = c.unique_count
        null_rate = null_count / row_count if row_count else 0.0
        uniqueness_ratio = (unique_count / row_count) if unique_count is not None and row_count else None
        out[c.name] = {
            "unique_count": unique_count,
            "null_count": null_count,
            "row_count": row_count,
            "null_rate": null_rate,
            "uniqueness_ratio": uniqueness_ratio,
        }
    return out


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def analysis_node(state: AgentState) -> AgentState:
    config = state["agent_config"]
    connector = state["connector"]
    table_metadata: Dict[str, TableMeta] = state["table_metadata"]
    cb = getattr(config, "progress_callback", None)

    if not table_metadata:
        state["phase"] = "analysed"
        return state

    fd_tool  = FunctionalDependencyTool(connector=connector)
    id_tool  = InclusionDependencyTool(connector=connector)
    car_tool = CardinalityAnalyzerTool(connector=connector)

    table_names = list(table_metadata.keys())
    n_tables = len(table_names)
    n_pairs  = n_tables * (n_tables - 1) // 2

    # Infer fact tables from sampled column metadata (falls back to name heuristic
    # when no sampling data is available).
    _fact_map = _infer_fact_tables(table_metadata)

    def _is_fact(name: str) -> bool:
        return _fact_map.get(name, False)

    # ------------------------------------------------------------------
    # 1. Functional Dependencies (per table)
    # ------------------------------------------------------------------
    logger.info("=== Functional Dependency Detection ===")
    if cb:
        cb("fd", "running",
           f"Functional Dependency detection — scanning {n_tables} table{'s' if n_tables != 1 else ''}",
           f"Algorithm: pairwise column-value comparison · threshold={config.fd_threshold:.0%} · cap={config.max_fd_column_pairs} pairs/table")

    total_fds_found = 0
    for table_name, meta in table_metadata.items():
        logger.info("  FD scan: %s", table_name)
        if cb:
            cb("fd:table", "running",
               f"FD scan: {table_name}",
               f"{len(meta.columns)} columns · testing X→Y for all compatible pairs")
        result_json = fd_tool._run(
            schema_name=meta.schema_name,
            table_name=table_name,
            columns=_col_names(meta),
            primary_keys=meta.primary_keys,
            sample_size=config.sample_size,
            threshold=config.fd_threshold,
            max_pairs=config.max_fd_column_pairs,
            column_stats=_col_stats(meta),
        )
        result = json.loads(result_json)
        if "error" in result:
            state["errors"].append(f"FD error [{table_name}]: {result['error']}")
            if cb:
                cb("fd:table", "error", f"FD scan: {table_name}", result["error"])
            continue

        fds_this = result.get("functional_dependencies", [])
        tested   = result.get("candidates_tested", 0)
        for fd in fds_this:
            state["func_deps"].append(
                FunctionalDependency(
                    table_name=table_name,
                    determinant=fd["determinant"],
                    dependent=fd["dependent"],
                    confidence=fd["confidence"],
                    num_violations=fd.get("violations", 0),
                    fd_type=fd.get("fd_type", "non_key"),
                    description=fd.get("description"),
                )
            )
        total_fds_found += len(fds_this)
        logger.info("    → %d FDs found (%d pairs tested)", len(fds_this), tested)
        if cb:
            if fds_this:
                fd_examples = "; ".join(
                    f"({', '.join(f['determinant'])}) → ({', '.join(f['dependent'])}) [{f['confidence']:.0%}]"
                    for f in fds_this[:3]
                ) + (" …" if len(fds_this) > 3 else "")
            else:
                fd_examples = "no FDs detected"
            cb("fd:table", "done",
               f"FD scan: {table_name} — {len(fds_this)} FD{'s' if len(fds_this) != 1 else ''} ({tested} pairs tested)",
               fd_examples)

    if cb:
        cb("fd", "done",
           f"Functional Dependencies complete — {total_fds_found} FD{'s' if total_fds_found != 1 else ''} across {n_tables} tables",
           "Types: primary_key · candidate_key · partial_key · non_key · transitively_implied")

    # ------------------------------------------------------------------
    # 2. Inclusion Dependencies (ordered table pairs)
    # ------------------------------------------------------------------
    logger.info("=== Inclusion Dependency Detection ===")
    if cb:
        cb("ind", "running",
           f"Inclusion Dependency detection — {n_pairs} table pair{'s' if n_pairs != 1 else ''} to scan",
           f"Algorithm: value-set containment test · threshold={config.id_threshold:.0%} · cap={config.max_id_column_pairs} col-pairs total")

    pair_count = 0
    total_col_pairs_tested = 0

    for left_name, right_name in itertools.combinations(table_names, 2):
        if total_col_pairs_tested >= config.max_id_column_pairs:
            logger.info("  IND scan: column-pair budget (%d) reached, stopping.",
                        config.max_id_column_pairs)
            if cb:
                cb("ind", "warn",
                   f"IND budget reached ({config.max_id_column_pairs} col-pairs) — remaining pairs skipped",
                   "Increase max_id_column_pairs in config to scan more pairs")
            break
        left_meta  = table_metadata[left_name]
        right_meta = table_metadata[right_name]

        remaining         = config.max_id_column_pairs - total_col_pairs_tested
        max_pairs_this_run = min(50, remaining)

        logger.info("  IND scan: %s → %s (budget remaining: %d col pairs)",
                    left_name, right_name, remaining)
        if cb:
            cb("ind:pair", "running",
               f"IND: {left_name} ⊆ {right_name}",
               f"Testing value-set containment across compatible column pairs")
        result_json = id_tool._run(
            schema_name=left_meta.schema_name,
            left_table=left_name,
            right_table=right_name,
            left_columns=_col_dicts(left_meta),
            right_columns=_col_dicts(right_meta),
            left_col_stats=_col_stats(left_meta),
            right_col_stats=_col_stats(right_meta),
            sample_size=config.sample_size,
            threshold=config.id_threshold,
            max_pairs=max_pairs_this_run,
        )
        result = json.loads(result_json)
        if "error" in result:
            state["errors"].append(f"IND error [{left_name}→{right_name}]: {result['error']}")
            if cb:
                cb("ind:pair", "error", f"IND: {left_name} ⊆ {right_name}", result["error"])
            total_col_pairs_tested += max_pairs_this_run
            continue

        existing_ind_keys = {
            (d.left_table, tuple(d.left_columns), d.right_table, tuple(d.right_columns))
            for d in state["incl_deps"]
        }
        new_inds = []
        for ind in result.get("inclusion_dependencies", []):
            lt = ind.get("left_table", left_name)
            rt = ind.get("right_table", right_name)
            lc = ind["left_columns"]
            rc = ind["right_columns"]
            dedup_key = (lt, tuple(lc), rt, tuple(rc))
            if dedup_key in existing_ind_keys:
                continue
            existing_ind_keys.add(dedup_key)
            # Fact↔fact INDs share ID columns but represent no meaningful FK
            # relationship — demote them so build_node never creates an edge.
            is_fk_cand = ind["is_foreign_key_candidate"]
            ind_type   = ind.get("ind_type", "value_subset")
            if is_fk_cand and _is_fact(lt) and _is_fact(rt):
                is_fk_cand = False
                ind_type   = "value_subset"
                logger.debug("IND: demoted fact↔fact FK candidate %s → %s to value_subset", lt, rt)
            state["incl_deps"].append(
                InclusionDependency(
                    left_table=lt,
                    left_columns=lc,
                    right_table=rt,
                    right_columns=rc,
                    coverage=ind["coverage"],
                    is_foreign_key_candidate=is_fk_cand,
                    ind_type=ind_type,
                    description=ind.get("description"),
                )
            )
            new_inds.append(ind)
        total_col_pairs_tested += result.get("pairs_tested", max_pairs_this_run)
        pair_count += 1
        if cb:
            if new_inds:
                ind_examples = "; ".join(
                    f"{i['left_columns']} ⊆ {i['right_columns']} [{i['coverage']:.0%}{'  FK-candidate' if i.get('is_foreign_key_candidate') else ''}]"
                    for i in new_inds[:3]
                ) + (" …" if len(new_inds) > 3 else "")
            else:
                ind_examples = "no INDs detected"
            cb("ind:pair", "done",
               f"IND: {left_name} ⊆ {right_name} — {len(new_inds)} inclusion dep{'s' if len(new_inds) != 1 else ''}",
               ind_examples)

    total_inds = len(state["incl_deps"])
    logger.info("Total INDs found: %d (across %d ordered pairs, %d column pairs tested)",
                total_inds, pair_count, total_col_pairs_tested)
    if cb:
        fk_count = sum(1 for d in state["incl_deps"] if d.is_foreign_key_candidate)
        cb("ind", "done",
           f"Inclusion Dependencies complete — {total_inds} IND{'s' if total_inds != 1 else ''} ({fk_count} FK candidates)",
           f"Types: exact_foreign_key · strong_fk_candidate · partial_inclusion · value_subset")

    # ------------------------------------------------------------------
    # 3. Cardinality Analysis (unordered table pairs, capped)
    # ------------------------------------------------------------------
    logger.info("=== Cardinality Analysis ===")
    cardinality_cap = min(config.max_fd_column_pairs, 200)
    if cb:
        cb("cardinality", "running",
           f"Cardinality analysis — {n_pairs} table pair{'s' if n_pairs != 1 else ''} (cap {cardinality_cap})",
           "Algorithm: join-column uniqueness ratio — determines 1:1, 1:N, N:1, M:N relationships")
    cardinality_pair_count = 0

    for left_name, right_name in itertools.combinations(table_names, 2):
        if cardinality_pair_count >= cardinality_cap:
            logger.info("  Cardinality: pair cap (%d) reached, stopping.", cardinality_cap)
            if cb:
                cb("cardinality", "warn",
                   f"Cardinality cap ({cardinality_cap} pairs) reached — remaining pairs skipped", "")
            break

        # Fact↔fact pairs share ID columns but have no meaningful join path;
        # skip them to avoid spurious cardinality edges in the KG.
        if _is_fact(left_name) and _is_fact(right_name):
            logger.debug("  Cardinality: skipping fact↔fact pair %s ↔ %s", left_name, right_name)
            cardinality_pair_count += 1
            continue

        left_meta  = table_metadata[left_name]
        right_meta = table_metadata[right_name]
        all_fks = left_meta.foreign_keys + right_meta.foreign_keys

        logger.info("  Cardinality: %s ↔ %s", left_name, right_name)
        result_json = car_tool._run(
            schema_name=left_meta.schema_name,
            left_table=left_name,
            right_table=right_name,
            left_columns=_col_names(left_meta),
            right_columns=_col_names(right_meta),
            foreign_keys=all_fks,
        )
        result = json.loads(result_json)
        if "error" in result:
            state["errors"].append(f"Cardinality error [{left_name}↔{right_name}]: {result['error']}")
            cardinality_pair_count += 1
            continue

        new_cards = result.get("cardinality_results", [])
        for cr in new_cards:
            state["cardinalities"].append(
                CardinalityRelationship(
                    left_table=left_name,
                    right_table=right_name,
                    join_columns=cr["join_columns"],
                    relationship_type=cr["relationship_type"],
                    left_unique=cr.get("left_unique_values", 0),
                    right_unique=cr.get("right_unique_values", 0),
                )
            )
        if new_cards and cb:
            rel_str = ", ".join(
                f"{left_name} {c['relationship_type']} {right_name} on ({', '.join(c['join_columns'])})"
                for c in new_cards[:2]
            )
            cb("cardinality:pair", "done",
               f"Cardinality: {left_name} ↔ {right_name}",
               rel_str)
        cardinality_pair_count += 1

    total_cards = len(state["cardinalities"])
    logger.info("Total cardinality relationships: %d (across %d pairs)", total_cards, cardinality_pair_count)
    if cb:
        cb("cardinality", "done",
           f"Cardinality analysis complete — {total_cards} relationship{'s' if total_cards != 1 else ''} mapped",
           "Relationship types: 1:1 · 1:N · N:1 · M:N")

    state["phase"] = "analysed"
    return state
