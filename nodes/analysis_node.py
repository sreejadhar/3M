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
# Strong prior: table name contains one of these → likely a fact or aggregate.
# NOTE: plain anchor match — underscores are word-chars so \bfact\b fails on
# "vw_fact_sales" (no \b between '_' and 'f').
_FACT_NAME_RE = re.compile(
    r'(^|_)(fact|fct|measure|metric|kpi)($|_)', re.IGNORECASE
)
# Weaker prior: aggregate / periodic summary / OLTP transaction event names
# NOTE: plain substring match (no \b) — underscores are word-chars so \bticker\b
# would fail on "support_tickets".
_AGG_NAME_RE = re.compile(
    r'(^|_)(agg|aggregat|summary|summ|snapshot|trans|event|'
    r'daily|weekly|monthly|hourly|annual|yearly|'
    r'tickets?|cases?|incidents?|sessions?|'
    r'claims?|requests?|complaints?|issues?|'
    r'alerts?|bookings?|reservations?)($|_)',
    re.IGNORECASE,
)
# Explicit dimension name: table starts with dim_ or contains _dim_
# → strong prior it is a dimension, not a fact.
# NOTE: plain anchor match for same \b reason.
_DIM_NAME_RE = re.compile(r'(^|_)dim(ension)?(_|$)', re.IGNORECASE)

# ── Measure column name keywords ──────────────────────────────────────────
# Named measures: keyword in column name + numeric dtype → clear additive measure
#
# NOTE: do NOT use \b word boundaries — underscores are word-chars so \btotal\b
# FAILS on "total_amount" (no boundary between 'l' and '_').
# Pattern: keyword appears at start, end, or between underscores.
_MEASURE_COL_RE = re.compile(
    r'(^|_)(sales|revenue|qty|quantity|amount|volume|price|cost|'
    r'count|total|value|profit|margin|units|spend|budget|forecast|'
    r'actual|transaction|order|balance|inventory|stock|weight|'
    r'rate|ratio|score|index|pct|percent|share|growth|diff|variance|'
    r'dist|distribution|avg|average|sum|fee|charge|tax|discount|'
    r'return|conversion|impression|click|session|visit|call|event|'
    r'duration|supply|demand)($|_)',
    re.IGNORECASE,
)

# ── FK / dimension-reference column suffixes ──────────────────────────────
_FK_COL_RE = re.compile(
    r'(_id|_key|_sk|_nk|_bk|_fk|_wid|_code|_ref|_num)\s*$',
    re.IGNORECASE,
)

# ── Time / grain column name keywords ─────────────────────────────────────
# NOTE: do NOT use \b word boundaries — underscores are word-chars in Python
# regex, so \bdate\b FAILS to match "order_date" (the _ before date is a word
# char, so there is no word boundary before 'd').  Use explicit anchor patterns.
#
# Patterns covered:
#   (^|_)keyword($|_)  — keyword at start, end, or between underscores
#   (_at|_ts|_dt)$     — created_at, updated_ts, modified_dt suffix
_TIME_COL_RE = re.compile(
    r'(^|_)(date|month|year|week|quarter|period|fiscal|timestamp|datetime|time)($|_)'
    r'|(_at|_ts|_dt)$',
    re.IGNORECASE,
)

# ── Time / grain data-type keywords ──────────────────────────────────────
# Catches columns whose NAME doesn't signal time but whose DTYPE does:
# e.g. col named "created" with dtype "datetime", or "ts" with dtype "timestamp"
_TIME_DTYPE_RE = re.compile(
    r'\b(date|datetime|timestamp|time)\b',
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

# ── SCD Type 3 column patterns ────────────────────────────────────────────
# SCD3 stores limited history via "current_X" / "previous_X" column pairs.
# Detecting ≥1 "previous/prior/old/original" value column is a strong dim
# indicator — facts never carry prior-state columns.
# NOTE: plain substring match, no \b, to handle underscores correctly.
_SCD3_PREV_RE = re.compile(
    r'(^|_)(prev(ious)?|prior|old|original|last|former|historic)_',
    re.IGNORECASE,
)
# "current_X" alongside a prev-style column confirms SCD3.
# We detect this separately so we only penalise when BOTH patterns co-exist.
_SCD3_CURR_RE = re.compile(
    r'(^|_)current_(?!flag|ind|row)',   # current_segment, current_tier, etc.
    re.IGNORECASE,                       # but NOT current_flag (that is SCD2)
)

# ── SCD Type 4 history-table name patterns ────────────────────────────────
# The history / audit table in a Type-4 design carries _hist/_history/_audit
# in the name and should be treated as a dimension, not a fact.
# Using end-of-string anchor ($) — the prefix underscore is already literal.
_SCD4_HIST_NAME_RE = re.compile(
    r'(_hist|_history|_archive|_audit|_log|_shadow)$',
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


def _score_fact_table(table_meta: TableMeta) -> tuple:  # noqa: C901  → (score: int, has_unique_col: bool)
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
    -1  SCD2: table has BOTH an effective-date col AND an expiry-date col
    -1  SCD2: is_current / current_flag present (unambiguous SCD2 marker)
    -1  SCD3: ≥1 column with previous_*/prior_*/old_*/former_* prefix
        (prior-value column — facts never store prior attribute states)
    -1  SCD3: confirmed current_<attr> + previous_<attr> pair present
    -2  SCD4: table name ends with _hist/_history/_archive/_audit/_log
        (history side-table — always a dimension, never a fact)
    -2  strong dimension: single declared PK + text-heavy (>40%) + few measures
    -1  OLTP entity/master table: single PK + unique natural key + <2 external
        FK references + table name does not contain fact/fct → intrinsic entity
        attributes are not measurements of events (catches compact product/
        employee/account tables with numeric attrs like price, weight, salary)

    Returns (score, has_unique_col) — callers use has_unique_col to determine
    whether this table has composite grain (OLAP fact) vs a natural PK key
    (OLTP transaction header), which changes the fact↔fact suppression logic.
    """
    score = 0
    name = table_meta.table_name

    # ── Name priors ───────────────────────────────────────────────────────
    if _FACT_NAME_RE.search(name):
        score += 2
    elif _AGG_NAME_RE.search(name):
        score += 1
    # Explicit dim_ prefix/infix is a strong dimension indicator regardless of
    # what the columns look like (e.g. dim_date has many time-related columns).
    if _DIM_NAME_RE.search(name):
        score -= 2

    cols = table_meta.columns
    if not cols:
        return score, False  # no column data — cannot determine grain

    n_cols    = len(cols)
    row_count = table_meta.row_count or 0

    named_measure_cols   = 0
    generic_measure_cols = 0
    fk_cols              = 0   # external FK references only — PK excluded
    time_cols            = 0
    text_cols            = 0
    scd_eff_found        = False
    scd_exp_found        = False
    scd_current_found    = False
    scd3_prev_count      = 0     # columns with previous_/prior_/old_ prefix
    scd3_curr_count      = 0     # columns with current_<attr> prefix (not flags)
    has_unique_col       = False   # any single col with uniqueness_ratio ≥ 0.95
    # Distinct FK root names for factless-fact detection.
    # Only counts non-PK FK columns so that surrogate+natural key pairs for the
    # same entity (customer_sk + customer_nk → root "customer") don't masquerade
    # as two separate entity references.
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

        # FK / dimension reference — deliberately exclude PK columns.
        # A PK column that happens to end in _id/_key (e.g. product_id as PK)
        # is the entity's own identifier, not a reference to another table.
        # Counting it inflates fk_cols for entity/master tables and confuses
        # the factless-fact distinct-root logic.
        if (is_fk_named or c.is_foreign_key) and not c.is_primary_key:
            fk_cols += 1
            root = _FK_COL_RE.sub('', col).rstrip('_').lower()
            fk_root_names.add(root)

        # Time grain — name pattern OR dtype, but only when the dtype is not plain
        # text (varchar/nvarchar/char). Text columns named "month_name", "year_name",
        # "quarter_desc" are descriptive dimension attributes, not grain columns.
        if (is_time_col and not is_text) or bool(_TIME_DTYPE_RE.search(dtype)):
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

        # SCD3 signals: previous_*/prior_*/old_* column prefix (prior-value col)
        # and current_<attr> prefix (current-value col).  Facts never carry
        # prior-state columns, so even one is a strong dimension indicator.
        if _SCD3_PREV_RE.search(col):
            scd3_prev_count += 1
        if _SCD3_CURR_RE.search(col):
            scd3_curr_count += 1

        # Generic inferred measure: numeric, not FK, not time, not bool,
        # not an SCD3 prior/current attribute column,
        # and (when available) uniqueness_ratio < 0.95 (it varies per row)
        is_scd3_col = bool(_SCD3_PREV_RE.search(col)) or bool(_SCD3_CURR_RE.search(col))
        if (
            is_numeric
            and not is_fk_named
            and not c.is_foreign_key
            and not c.is_primary_key
            and not is_time_col
            and not is_bool
            and not is_scd3_col
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

        # Natural-key / unique-identifier detection.
        # Signals that this table has a single-column identifier (dim or OLTP
        # transaction header), not a purely composite grain (OLAP fact).
        # Fires on:
        #   • Any declared PK column with uniqueness_ratio ≥ 0.95, OR
        #   • Any non-FK, non-PK column with uniqueness_ratio ≥ 0.95
        #     (natural business keys like customer_no, order_number)
        if c.unique_count is not None and row_count > 0:
            u_ratio = c.unique_count / row_count
            if u_ratio >= 0.95 and (c.is_primary_key or (not is_fk_named and not c.is_foreign_key)):
                has_unique_col = True

    text_ratio = text_cols / n_cols if n_cols else 0.0

    # Self-referencing / hierarchical table: any FK points back to this table.
    # These are entity tables (categories, org_chart, bill_of_materials) — never facts.
    is_self_referencing = any(
        (fk.get("references_table", "") or "").lower() == name.lower()
        for fk in table_meta.foreign_keys
    )

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

    # SCD3 slowly-changing dimension (prior-value columns):
    # Any "previous_*/prior_*/old_*" column is a dead giveaway — facts never
    # store prior attribute values.
    # -1 for ≥1 prior-value column; -1 more if current_<attr> columns also
    # present (confirming the current/previous pair pattern).
    if scd3_prev_count >= 1:
        score -= 1
    if scd3_prev_count >= 1 and scd3_curr_count >= 1:
        score -= 1   # confirmed current+previous pair → SCD3

    # SCD4 history table: the history/audit side-table carries _hist/_history/
    # _archive/_audit in its name. Treat as dimension regardless of columns.
    if _SCD4_HIST_NAME_RE.search(table_meta.table_name):
        score -= 2

    # Self-referencing / hierarchical entity table (categories, org chart, BOM).
    # A table that has a FK back to itself is always an entity, never a fact.
    if is_self_referencing:
        score -= 2

    # Strong dimension penalty: single declared PK + text-heavy + few measures
    # → almost certainly a standard or conformed dimension.
    single_pk = len(table_meta.primary_keys) == 1
    if single_pk and text_ratio > 0.40 and named_measure_cols < 2:
        score -= 2

    # OLTP entity / master table penalty.
    # Key insight: transaction tables ALWAYS have at least one event date/time
    # column (order_date, payment_date, created_at).  Pure entity/attribute
    # tables (products, departments, config) do not.
    # Fires when:
    #   • single unique identifier (has_unique_col) — entity or txn header
    #   • fewer than 2 external FK references — not a junction/resolution table
    #   • NO time/date column (time_cols == 0) — no event timestamp → entity
    #   • not an explicit fact/aggregate name
    # Penalty is -2 (not -1) so it overcomes entity tables that happen to carry
    # two numeric attribute columns (e.g. products.unit_price + products.weight).
    if (
        has_unique_col
        and fk_cols < 2
        and time_cols == 0
        and not _FACT_NAME_RE.search(name)
        and not _AGG_NAME_RE.search(name)
    ):
        score -= 2

    return score, has_unique_col


def _infer_fact_tables(table_metadata: Dict[str, TableMeta]) -> Dict[str, Dict]:
    """
    Return a mapping of table_name → {"is_fact": bool, "has_unique_col": bool}
    for every table in table_metadata.

    is_fact         — True if score >= 2 (any Kimball fact variant)
    has_unique_col  — True if the table has a single-column unique identifier.
                      Used by analysis_node to distinguish OLAP composite-grain
                      facts (where fact↔fact edges are suppressed) from OLTP
                      transaction headers (where header→line FK edges are valid).
    """
    result: Dict[str, Dict] = {}
    for name, meta in table_metadata.items():
        score, has_unique_col = _score_fact_table(meta)
        is_fact = score >= 2
        result[name] = {"is_fact": is_fact, "has_unique_col": has_unique_col}
        logger.debug(
            "fact-inference: %s → score=%d is_fact=%s has_unique_col=%s",
            name, score, is_fact, has_unique_col,
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

    def _is_composite_grain_fact(name: str) -> bool:
        """True only for OLAP-style facts with no single unique identifier.
        OLTP transaction headers also score as facts but have has_unique_col=True
        (the order_id / invoice_id PK), so fact↔fact suppression should NOT fire
        for header→line relationships — only for pure composite-grain OLAP facts.
        """
        info = _fact_map.get(name, {})
        return info.get("is_fact", False) and not info.get("has_unique_col", False)

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
            # Suppress spurious FK edges between two OLAP composite-grain facts
            # (e.g. fact_sales ↔ fact_budget sharing date_id).  Do NOT suppress
            # when either table is an OLTP transaction header (has_unique_col=True)
            # because header→line relationships (orders → order_items) are valid.
            is_fk_cand = ind["is_foreign_key_candidate"]
            ind_type   = ind.get("ind_type", "value_subset")
            if is_fk_cand and _is_composite_grain_fact(lt) and _is_composite_grain_fact(rt):
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

        # Skip cardinality analysis for two OLAP composite-grain facts — they
        # share dimension keys but have no meaningful direct join path.
        # OLTP header↔line pairs (both scored as fact but has_unique_col=True)
        # are NOT skipped because their cardinality relationship is real and
        # needed for query generation (e.g. orders 1:N order_items).
        if _is_composite_grain_fact(left_name) and _is_composite_grain_fact(right_name):
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
