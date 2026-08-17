"""
plan_node — Use LLM to decompose a natural language query into SQL statements.

The LLM receives:
  - The user's natural language query
  - The schema context (qualified table names, columns, relationships)
  - Target DB type (to pick the right SQL dialect)
  - The schema name and row limit

It must return a JSON array of query objects:
    [{"query_id": "q1", "description": "...", "sql": "...", "table_refs": [...]}, ...]
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .. import verified_queries
from ..state import DialogState, SQLQuery
from ..token_guard import guard_plan_prompt
from . import sql_identifier_resolver as _ast_resolver

logger = logging.getLogger(__name__)

# Show this many recent turns verbatim; summarize anything older.
_MAX_VERBATIM_TURNS = 5

# Phase 2 of the AST-based hallucination-resolver rollout: run the new
# sqlglot-backed resolver (sql_identifier_resolver.py) alongside the existing
# regex-based _find_hallucinated_tables/_find_hallucinated_columns checks and
# log any disagreement, without changing which queries get dropped/repaired.
# Flip PLAN_NODE_AST_HALLUCINATION_SHADOW=0 to disable; nothing downstream of
# the shadow-mode block reads its output, so it can never affect a response.
_AST_RESOLVER_SHADOW_MODE = os.getenv("PLAN_NODE_AST_HALLUCINATION_SHADOW", "1") != "0"

# Phase 3 cutover: when enabled, plan_node's table/column hallucination
# checks run ENTIRELY through the AST resolver instead of the regex pair
# (_find_hallucinated_tables / _find_hallucinated_columns / _strip_hallucinated_
# conditions) — those functions become dead code on this path but are left in
# place as the fallback for when this flag is off, and for the other checks
# in _validate_plan_items (_find_cross_table_columns, join validation, etc.)
# that stay on the regex path regardless (out of scope for this resolver).
#
# Default OFF. Evidence gate before flipping per-environment (see
# tools/run_hallucination_shadow_harness.py and hallucination_shadow_report.txt):
# 363 real DataChat-generated SQL statements checked across PNC, LIFESCIENCE
# STESTUSECASE, and Claim Underwriting — 0 cases where the AST resolver would
# have produced a wrong drop/allow outcome vs. the regex path. Still limited to
# Snowflake-dialect sources; treat other dialects as unverified until checked.
_AST_RESOLVER_ENABLED = os.getenv("PLAN_NODE_AST_HALLUCINATION_ENABLED", "0") == "1"

# Map this codebase's db_type strings to sqlglot dialect identifiers.
_SQLGLOT_DIALECT_MAP = {
    "sqlite": "sqlite", "csv": "sqlite", "excel": "sqlite",
    "postgres": "postgres", "postgresql": "postgres",
    "redshift": "redshift",
    "sqlserver": "tsql",
    "oracle": "oracle",
    "snowflake": "snowflake",
    "bigquery": "bigquery",
}


def _ast_dialect(db_type: str) -> str:
    return _SQLGLOT_DIALECT_MAP.get((db_type or "").lower(), "snowflake")


def _build_ast_schema(
    table_labels: List[str],
    table_columns_map: Dict[str, set],
) -> "_ast_resolver.SchemaGraph":
    """
    table_columns_map keys are already bare/uppercased by _extract_table_columns;
    fold in table_labels with empty column sets so pure table-hallucination
    checks (no columns known for a label) still resolve correctly.
    """
    merged_map: Dict[str, set] = {t: set() for t in table_labels}
    merged_map.update(table_columns_map)
    return _ast_resolver.SchemaGraph(merged_map)


def _shadow_check_hallucination(
    sql: str,
    schema: "_ast_resolver.SchemaGraph",
    dialect: str,
    query_id: str,
    old_bad_tables: Optional[List[str]] = None,
    old_bad_cols: Optional[List[str]] = None,
) -> None:
    """
    Phase 2 shadow-mode comparison — log-only, never affects control flow.
    Runs the new AST resolver against the same SQL/schema the regex checks
    just evaluated and logs any disagreement for later triage before Phase 3
    cutover. Any exception here is swallowed: shadow mode must never be able
    to break the real validation path it's observing. No-op when the Phase 3
    cutover is already live for this call (redundant — the live path already
    exercises the AST resolver directly).
    """
    if not _AST_RESOLVER_SHADOW_MODE or _AST_RESOLVER_ENABLED:
        return
    try:
        new_bad = _ast_resolver.find_hallucinated_identifiers(sql, schema, dialect=dialect)
        new_bad_tables = sorted({b.name.lower() for b in new_bad if b.kind == "table"})
        new_bad_cols = sorted({b.name.lower() for b in new_bad if b.kind == "column"})
        old_tables = sorted({t.lower() for t in (old_bad_tables or [])})
        old_cols = sorted({c.lower() for c in (old_bad_cols or [])})
        if new_bad_tables != old_tables or new_bad_cols != old_cols:
            logger.warning(
                "plan_node: AST-resolver shadow-mode disagreement on query %s — "
                "regex found tables=%s cols=%s; AST resolver found tables=%s cols=%s; sql=%s",
                query_id, old_tables, old_cols, new_bad_tables, new_bad_cols, sql,
            )
    except Exception:
        logger.exception(
            "plan_node: AST-resolver shadow-mode check raised on query %s (ignored)",
            query_id,
        )


def _summarize_old_turns_plan(old_turns: list, model: str) -> str:
    """
    Summarize older conversation turns into a compact paragraph for the plan
    prompt, using a cheap Haiku call.  Returns empty string on failure.
    The summary is optimised for SQL planning context: tables used, filters,
    metrics, and references that may appear as pronouns in the new question.
    """
    lines = [
        "Summarize these past Q&A exchanges into 3-5 bullet points. "
        "Be very concise. Focus on: tables/columns queried, filters applied, "
        "metrics computed, and any entity names that may be referenced later:\n"
    ]
    for turn in old_turns:
        lines.append(f"Q{turn['turn']}: {turn['question']}")
        if turn.get("tables_queried"):
            lines.append(f"  Tables: {', '.join(turn['tables_queried'])}")
        if turn.get("insights"):
            lines.append(f"  Answer: {turn['insights'][:200]}")
    try:
        from llm_client import get_client
        client = get_client()
        msg = client.messages.create(
            model=model,
            max_tokens=256,
            temperature=0.0,
            system="You summarize SQL analysis Q&A history into concise bullet points.",
            messages=[{"role": "user", "content": "\n".join(lines)}],
        )
        return msg.content[0].text if msg.content else ""
    except Exception as exc:
        logger.warning("plan_node: history summarization failed — %s", exc)
        return ""


def _build_history_section_plan(history: list, model: str) -> str:
    """
    Build the CONVERSATION HISTORY section for the plan prompt.
    Includes per-query diagnostics (SQL used, row counts, columns returned,
    errors, pre-flight gaps) so the planner can avoid repeating failed approaches.
    If history exceeds MAX_VERBATIM_TURNS, older turns are summarized with Haiku.
    """
    if not history:
        return ""

    lines = ["CONVERSATION HISTORY (previous questions in this session):"]

    if len(history) > _MAX_VERBATIM_TURNS:
        old_turns = history[:-3]
        recent    = history[-3:]
        summary   = _summarize_old_turns_plan(old_turns, model)
        if summary:
            lines.append(
                f"[Summary of {len(old_turns)} earlier turn(s) — use to resolve pronouns "
                f"and implied filters:]"
            )
            lines.append(summary)
            lines.append("")
    else:
        recent = history

    for turn in recent:
        lines.append(f"Q{turn['turn']}: {turn['question']}")
        if turn.get("tables_queried"):
            lines.append(f"  Tables used: {', '.join(turn['tables_queried'])}")
        if turn.get("insights"):
            lines.append(f"  Answer summary: {turn['insights'][:300]}")

        # ── Per-query execution diagnostics ──────────────────────────────
        diags = turn.get("query_diagnostics") or []
        if diags:
            lines.append("  PRIOR QUERY ATTEMPTS (what was run and what it returned):")
            for d in diags:
                qid      = d.get("query_id", "?")
                rows     = d.get("row_count", 0)
                cols     = d.get("columns") or []
                err      = d.get("error")
                gaps     = d.get("preflight_gaps") or []
                sql_snip = (d.get("sql") or "")[:300]

                lines.append(f"    [{qid}] rows_returned={rows}  columns={cols}")
                if sql_snip:
                    lines.append(f"      SQL: {sql_snip}")
                if err:
                    lines.append(f"      ❌ ERROR: {err}")
                if gaps:
                    for g in gaps:
                        lines.append(f"      ⚠️  GAP: {g}")
            lines.append(
                "  INSTRUCTION: Study the prior attempts above. Do NOT repeat the same "
                "SQL patterns that returned 0 rows, produced errors, or had analytical "
                "gaps flagged above. Build on what worked and fix what didn't."
            )

    lines.append(
        "Use this history to resolve pronouns (e.g. 'it', 'that', 'those'), "
        "implied filters (e.g. 'same service line'), or comparisons to previous results."
    )
    return "\n".join(lines) + "\n"

def _build_dialect_rules(db_type: str) -> str:
    """Return database-specific SQL syntax rules injected into the system prompt."""
    db = db_type.lower()

    if db in ("sqlite", "csv", "excel"):
        return """\
ROW LIMITING      : LIMIT N at the end of the query  (e.g. SELECT col FROM t LIMIT 100)
TOP-N QUERIES     : ORDER BY col DESC LIMIT N
CASE-INSENSITIVE  : LOWER(col) LIKE LOWER('%term%')   — ILIKE does NOT exist in SQLite
DATE EXTRACTION   : strftime('%Y', date_col) for year; strftime('%m', date_col) for month
                    strftime('%Y-%m', date_col) for year-month
DATE COMPARISON   : date_col BETWEEN '2024-01-01' AND '2024-12-31'
STRING CONCAT     : col1 || col2   — do NOT use + or CONCAT()
IDENTIFIER QUOTING: double-quotes "col name" only when a name has spaces or is reserved;
                    never use backticks (`)
JOIN SYNTAX       : JOIN t2 ON t1.col = t2.col  — USING (col) is not reliably supported
PERCENTILES       : NOT SUPPORTED — use MIN/MAX/AVG as approximations, or note unavailability
PERCENTAGE CALC   : SUM(COUNT(*)) OVER () is INVALID in SQLite — use a CTE scalar subquery:
                      WITH grp AS (SELECT cat, COUNT(*) AS cnt FROM t GROUP BY cat)
                      SELECT cat, cnt,
                             ROUND(cnt * 100.0 / (SELECT SUM(cnt) FROM grp), 2) AS Pct
                      FROM grp
NULL HANDLING     : COALESCE(col, 0) or IFNULL(col, 0)
BOOLEAN           : use 1 / 0 integers — TRUE/FALSE literals are not reliable in SQLite
TYPE CASTING      : CAST(col AS INTEGER), CAST(col AS REAL), CAST(col AS TEXT)
                    NEVER use :: PostgreSQL-style casting (col::int is a SYNTAX ERROR)
                    ★ CRITICAL ★ SQLite's CAST does NOT raise an error on non-numeric text —
                    CAST('abc' AS INTEGER) silently returns 0. This means a WHERE filter like
                    CAST(col AS INTEGER) < 20 will silently match every dirty/non-numeric row
                    (they all cast to 0, which is < 20) instead of excluding them. When a TEXT
                    column may contain non-numeric values, add an explicit guard so dirty rows
                    are excluded rather than silently coerced to 0:
                      WHERE col GLOB '[0-9]*' AND CAST(col AS INTEGER) < 20   ← correct
                      WHERE CAST(col AS INTEGER) < 20                        ← WRONG if col has junk
WINDOW FUNCTIONS  : ROW_NUMBER(), RANK(), DENSE_RANK(), LAG(), LEAD() supported (SQLite ≥3.25)
                    ALL navigation functions MUST have ORDER BY inside OVER():
                      ROW_NUMBER() OVER (ORDER BY col)          ← correct
                      ROW_NUMBER() OVER ()                      ← ERROR
CURRENT DATE/TIME : DATE('now'), DATETIME('now')  — do NOT use NOW() or GETDATE()
UNSUPPORTED       : FULL OUTER JOIN, PIVOT, PERCENTILE_CONT, PERCENTILE_DISC,
                    GENERATE_SERIES, ANY/ALL subquery operators, ILIKE"""

    if db == "redshift":
        return """\
ROW LIMITING      : LIMIT N at the end  (e.g. SELECT col FROM t LIMIT 100)
TOP-N QUERIES     : ORDER BY col DESC LIMIT N
CASE-INSENSITIVE  : col ILIKE '%term%'   — preferred; or LOWER(col) LIKE LOWER('%term%')
DATE EXTRACTION   : EXTRACT(YEAR FROM date_col), EXTRACT(MONTH FROM date_col)
                    DATE_TRUNC('month', date_col)
DATE COMPARISON   : date_col BETWEEN '2024-01-01' AND '2024-12-31'
                    date_col >= '2024-01-01'::date
STRING CONCAT     : col1 || col2   or   CONCAT(col1, col2)
IDENTIFIER QUOTING: double-quotes "col name" only when needed; never backticks
PERCENTILES       : PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY col) AS median
                    PERCENTILE_DISC(0.25) WITHIN GROUP (ORDER BY col) AS q1
PERCENTAGE CALC   : ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS Pct
                    ROUND(SUM(col) * 100.0 / SUM(SUM(col)) OVER (), 2) AS Pct
NULL HANDLING     : COALESCE(col, 0)
TYPE CASTING      : value::integer, value::numeric, value::text  (:: casting IS supported)
                    ★ CRITICAL ★ Redshift has NO TRY_CAST / TRY_CONVERT. Casting a TEXT column
                    that may hold non-numeric values with :: or CAST() throws a hard error and
                    aborts the ENTIRE query on the first bad value. Guard with a regex check in
                    the WHERE clause (or a CASE expression) before casting:
                      WHERE col ~ '^[0-9]+(\\.[0-9]+)?$' AND col::numeric < 20   ← correct
                      WHERE col::numeric < 20                                    ← WRONG if col has junk
                    Or inline: CASE WHEN col ~ '^[0-9]+(\\.[0-9]+)?$' THEN col::numeric ELSE NULL END
CURRENT DATE/TIME : NOW() or CURRENT_TIMESTAMP
WINDOW FUNCTIONS  : ROW_NUMBER(), RANK(), DENSE_RANK(), LAG(), LEAD() — fully supported
                    ALL navigation/offset functions MUST have ORDER BY inside OVER():
                      LAG(col) OVER (PARTITION BY x ORDER BY period_col)  ← correct
                      LAG(col) OVER (PARTITION BY x)                      ← ERROR
STRING AGGREGATION: LISTAGG(col, ', ') WITHIN GROUP (ORDER BY col)
                    — do NOT use STRING_AGG (PostgreSQL-only, NOT supported in Redshift)
ARRAY AGG         : array_agg(col)  — ORDER BY inside array_agg() is NOT supported in Redshift
                    WRONG: array_agg(col ORDER BY col)
                    RIGHT: array_agg(col)
LATERAL JOINS     : LATERAL keyword is NOT supported in Redshift — use subqueries instead
GENERATE_SERIES   : NOT supported in Redshift — use a numbers table or VALUES list
UNSUPPORTED       : STRING_AGG, LATERAL, GENERATE_SERIES, array_agg(ORDER BY),
                    CREATE TABLE AS SELECT with DISTSTYLE/SORTKEY requires Redshift DDL"""

    if db in ("postgres", "postgresql"):
        return """\
ROW LIMITING      : LIMIT N at the end  (e.g. SELECT col FROM t LIMIT 100)
TOP-N QUERIES     : ORDER BY col DESC LIMIT N
CASE-INSENSITIVE  : col ILIKE '%term%'   — preferred; or LOWER(col) LIKE LOWER('%term%')
DATE EXTRACTION   : EXTRACT(YEAR FROM date_col), EXTRACT(MONTH FROM date_col)
                    DATE_TRUNC('month', date_col)
DATE COMPARISON   : date_col BETWEEN '2024-01-01' AND '2024-12-31'
                    date_col >= '2024-01-01'::date
STRING CONCAT     : col1 || col2   or   CONCAT(col1, col2)
IDENTIFIER QUOTING: double-quotes "col name" only when needed; never backticks
PERCENTILES       : PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY col) AS median
                    PERCENTILE_DISC(0.25) WITHIN GROUP (ORDER BY col) AS q1
PERCENTAGE CALC   : ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS Pct
                    ROUND(SUM(col) * 100.0 / SUM(SUM(col)) OVER (), 2) AS Pct
NULL HANDLING     : COALESCE(col, 0)
TYPE CASTING      : value::integer, value::numeric, value::text
                    ★ CRITICAL ★ PostgreSQL has NO TRY_CAST. Casting a TEXT column that may
                    hold non-numeric values with :: or CAST() throws a hard error and aborts
                    the ENTIRE query on the first bad value. Guard with a regex check before
                    casting:
                      WHERE col ~ '^[0-9]+(\\.[0-9]+)?$' AND col::numeric < 20   ← correct
                      WHERE col::numeric < 20                                    ← WRONG if col has junk
                    Or inline: CASE WHEN col ~ '^[0-9]+(\\.[0-9]+)?$' THEN col::numeric ELSE NULL END
CURRENT DATE/TIME : NOW() or CURRENT_TIMESTAMP
WINDOW FUNCTIONS  : ROW_NUMBER(), RANK(), DENSE_RANK(), LAG(), LEAD() — fully supported
                    ALL navigation/offset functions MUST have ORDER BY inside OVER():
                      LAG(col) OVER (PARTITION BY x ORDER BY period_col)  ← correct
                      LAG(col) OVER (PARTITION BY x)                      ← ERROR
STRING AGGREGATION: STRING_AGG(col, ', ' ORDER BY col)"""

    if db == "sqlserver":
        return """\
ROW LIMITING      : SELECT TOP N col FROM t   — LIMIT does NOT exist in SQL Server
                    For top-N with ordering: SELECT TOP 10 col FROM t ORDER BY col DESC
                    For paging: ORDER BY col OFFSET 0 ROWS FETCH NEXT 100 ROWS ONLY
DISTINCT + TOP    : ALWAYS write SELECT DISTINCT TOP N ... — NEVER SELECT TOP N DISTINCT ...
                    Correct : SELECT DISTINCT TOP 10 [col] FROM t ORDER BY [col]
                    Wrong   : SELECT TOP 10 DISTINCT [col] FROM t   ← SYNTAX ERROR
CASE-INSENSITIVE  : LOWER(col) LIKE LOWER('%term%')  — ILIKE does NOT exist in SQL Server
DATE EXTRACTION   : YEAR(date_col), MONTH(date_col), DAY(date_col)
                    DATEPART(year, date_col), DATEPART(month, date_col)
                    FORMAT(date_col, 'yyyy-MM')
DATE COMPARISON   : date_col BETWEEN '2024-01-01' AND '2024-12-31'
DATE TRUNCATION   : DATEADD(month, DATEDIFF(month, 0, date_col), 0) for month-start
                    DATE_TRUNC does NOT exist in SQL Server
CURRENT DATETIME  : GETDATE()  — do NOT use NOW(), CURRENT_TIMESTAMP is also valid
STRING CONCAT     : col1 + col2   or   CONCAT(col1, col2)   — do NOT use ||
STRING LENGTH     : LEN(col)   — LENGTH() does NOT exist in SQL Server
IDENTIFIER QUOTING: square brackets [col name] when needed — never backticks (`)
JOIN SYNTAX       : JOIN t2 ON t1.col = t2.col  — USING (col) is a SYNTAX ERROR in SQL Server
PERCENTILES       : PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY col) OVER () AS median
                    PERCENTILE_DISC(0.25) WITHIN GROUP (ORDER BY col) OVER () AS q1
PERCENTAGE CALC   : ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS Pct
NULL HANDLING     : ISNULL(col, 0) or COALESCE(col, 0)
TYPE CASTING      : CAST(col AS INT), CAST(col AS DECIMAL(10,2)), CAST(col AS NVARCHAR(100))
                    NEVER use :: PostgreSQL-style casting (col::int is a SYNTAX ERROR)
                    ★ CRITICAL ★ When casting a VARCHAR/NVARCHAR column that is NOT guaranteed
                    to hold clean numeric strings, ALWAYS use TRY_CAST / TRY_CONVERT instead of
                    CAST/CONVERT — plain CAST throws a hard error and aborts the entire query
                    on the first non-numeric value:
                      TRY_CAST(col AS INT)          ← correct — dirty values become NULL
                      CAST(col AS INT)               ← WRONG for VARCHAR columns with any junk
                    Use plain CAST only when the schema context shows the column's native type
                    is already numeric (INT/DECIMAL/FLOAT/NUMERIC).
STRING AGGREGATION: STRING_AGG(col, ', ') WITHIN GROUP (ORDER BY col)
WINDOW FUNCTIONS  : ROW_NUMBER(), RANK(), DENSE_RANK(), LAG(), LEAD() — fully supported
                    ALL navigation/offset functions MUST have ORDER BY inside OVER():
                      LAG(col) OVER (PARTITION BY x ORDER BY period_col)  ← correct
                      LAG(col) OVER (PARTITION BY x)                      ← ERROR
                      LAG(col) OVER ()                                    ← ERROR
                    Aggregate windows (SUM/AVG/COUNT OVER (...)) do NOT need ORDER BY.
                    Window functions cannot be used in WHERE clause — use a CTE or subquery
ORDER BY IN SUBS  : ORDER BY is ILLEGAL inside subqueries, CTEs, derived tables, views,
                    and inline functions UNLESS the subquery also has TOP, OFFSET, or FOR XML.
                    Error 1033 will be raised at runtime.
                    WRONG: SELECT * FROM (SELECT col FROM t ORDER BY col) sub
                    WRONG: WITH cte AS (SELECT col FROM t ORDER BY col) SELECT * FROM cte
                    RIGHT: SELECT * FROM (SELECT TOP 10 col FROM t ORDER BY col) sub
                    RIGHT: Use ORDER BY only at the outermost query level unless paired with TOP
SUBQUERY COLUMNS  : A subquery used as a scalar value (IN, NOT IN, =, <>) MUST return
                    exactly ONE column.
                    WRONG: WHERE id IN (SELECT id, rn FROM ranked WHERE rn <= 12)
                    RIGHT: WHERE id IN (SELECT id FROM ranked WHERE rn <= 12)"""

    if db == "oracle":
        return """\
ROW LIMITING      : FETCH FIRST N ROWS ONLY  (Oracle 12c+)
                    Example: SELECT col FROM t ORDER BY col DESC FETCH FIRST 10 ROWS ONLY
                    Legacy (pre-12c): WHERE ROWNUM <= N in an outer query
                    NEVER use LIMIT — it does NOT exist in Oracle
CASE-INSENSITIVE  : LOWER(col) LIKE LOWER('%term%')   — ILIKE does NOT exist
DATE EXTRACTION   : EXTRACT(YEAR FROM date_col), EXTRACT(MONTH FROM date_col)
                    TO_CHAR(date_col, 'YYYY-MM')
DATE COMPARISON   : date_col BETWEEN DATE '2024-01-01' AND DATE '2024-12-31'
STRING CONCAT     : col1 || col2   — do NOT use +
IDENTIFIER QUOTING: double-quotes "col name" only when needed; never backticks
PERCENTILES       : PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY col) OVER () AS median
PERCENTAGE CALC   : ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS Pct
NULL HANDLING     : NVL(col, 0) or COALESCE(col, 0)
CURRENT DATE/TIME : SYSDATE  or  CURRENT_DATE  — do NOT use NOW()
TYPE CASTING      : CAST(col AS NUMBER), CAST(col AS VARCHAR2(100))
                    NEVER use :: PostgreSQL-style casting (col::int is a SYNTAX ERROR)
                    ★ CRITICAL ★ Oracle has NO TRY_CAST. Casting a VARCHAR2 column that may hold
                    non-numeric values with CAST() throws a hard error (ORA-01722) and aborts the
                    ENTIRE query on the first bad value. Guard with REGEXP_LIKE before casting:
                      WHERE REGEXP_LIKE(col, '^[0-9]+(\\.[0-9]+)?$') AND CAST(col AS NUMBER) < 20
                      WHERE CAST(col AS NUMBER) < 20   ← WRONG if col has any non-numeric value
                    Oracle 12.2+ also has VALIDATE_CONVERSION(col AS NUMBER) which returns 1/0.
STRING AGGREGATION: LISTAGG(col, ', ') WITHIN GROUP (ORDER BY col)
WINDOW FUNCTIONS  : ROW_NUMBER(), RANK(), DENSE_RANK(), LAG(), LEAD() — fully supported
                    ALL navigation/offset functions MUST have ORDER BY inside OVER():
                      LAG(col) OVER (PARTITION BY x ORDER BY period_col)  ← correct
                      LAG(col) OVER (PARTITION BY x)                      ← ERROR
SUBQUERY COLUMNS  : A subquery used as a scalar value (IN, NOT IN, =) MUST return
                    exactly ONE column — same ANSI rule as all other databases."""

    if db == "snowflake":
        return """\
IDENTIFIER CASE   : ★ CRITICAL ★ Snowflake folds UNQUOTED identifiers to UPPER CASE.
                    Table/column names here are stored in their exact (often
                    lower-case) form, so wrap every table and column name in
                    double quotes using the EXACT case shown in the schema
                    context — e.g. "fact_gl", "dim_account"."account_name".
                    Database and schema names (e.g. SYNTHGEN_DB, FPNA) are upper-case
                    and need no quotes. NEVER change the case of an identifier and
                    NEVER leave a table/column name unquoted.
VIEW COLUMNS      : ★ CRITICAL ★ View columns inherit the quoted-lowercase identifiers
                    from their base tables.  If the view was created with quoted
                    lower-case columns (e.g. "hcp_key"), you MUST reference them as
                    "hcp_key" — NOT as "HCP_KEY" or HCP_KEY.  The schema context shows
                    the exact case; copy it verbatim.  Never uppercase a column name
                    that appears lowercase in the schema context.
TABLE ALIASES     : When using a table alias, EVERY column reference MUST be quoted:
                      t."hcp_key"  ← correct
                      t.HCP_KEY    ← WRONG — Snowflake folds to uppercase, fails
                      t.hcp_key    ← WRONG — same folding, fails
                    Copy the quoted column name from the schema context and prefix
                    it with the alias: alias."column_name"
JOIN COLUMNS      : In JOIN ON clauses, BOTH sides must quote column names exactly:
                      t1."join_col" = t2."join_col"  ← correct (exact case, quoted)
                      t1.JOIN_COL = t2.JOIN_COL       ← WRONG (unquoted, folds to upper)
                    Copy the column name verbatim from the schema context for both sides.
COLUMN ALIASES    : Quote every alias / CTE column you introduce with AS in lower
                    case at BOTH its definition AND every reference — e.g.
                    SUM("net_amount") AS "total_net" ... ORDER BY "total_net".
                    An UNQUOTED alias is folded to UPPER CASE, so a later quoted
                    lower-case reference fails with "invalid identifier".
ROW LIMITING      : LIMIT N at the end  (e.g. SELECT "col" FROM "t" LIMIT 100)
TOP-N QUERIES     : ORDER BY "col" DESC LIMIT N
CASE-INSENSITIVE  : col ILIKE '%term%'   — preferred; or LOWER(col) LIKE LOWER('%term%')
DATE EXTRACTION   : EXTRACT(YEAR FROM date_col), EXTRACT(MONTH FROM date_col)
                    DATE_TRUNC('month', date_col)
DATE COMPARISON   : date_col BETWEEN '2024-01-01' AND '2024-12-31'
STRING CONCAT     : col1 || col2   or   CONCAT(col1, col2)
PERCENTILES       : PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY col) AS median
PERCENTAGE CALC   : ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS Pct
                    ROUND(SUM(col) * 100.0 / SUM(SUM(col)) OVER (), 2) AS Pct
NULL HANDLING     : COALESCE(col, 0)
TYPE CASTING      : value::integer, value::numeric, value::text  (:: casting IS supported)
                    ★ CRITICAL ★ When casting a TEXT/VARCHAR column that is NOT
                    guaranteed to hold clean numeric strings (i.e. any column
                    whose declared type in the schema context is TEXT/VARCHAR/STRING
                    rather than a native NUMBER/INTEGER/FLOAT type), ALWAYS use
                    TRY_CAST / TRY_TO_NUMBER / TRY_TO_DECIMAL instead of CAST or ::.
                      TRY_CAST(col AS NUMBER)      ← correct — dirty values become NULL
                      CAST(col AS NUMBER)          ← WRONG for TEXT columns — throws a
                                                       hard error on ANY non-numeric value
                                                       and aborts the entire query
                    Use plain CAST only when the schema context shows the column's
                    native type is already numeric (NUMBER/INTEGER/FLOAT/DECIMAL).
CURRENT DATE/TIME : CURRENT_TIMESTAMP  or  CURRENT_DATE  — do NOT use NOW()/GETDATE()
WINDOW FUNCTIONS  : ROW_NUMBER(), RANK(), DENSE_RANK(), LAG(), LEAD() — fully supported
                    ALL navigation/offset functions MUST have ORDER BY inside OVER():
                      LAG(col) OVER (PARTITION BY x ORDER BY period_col)  ← correct
                      LAG(col) OVER (PARTITION BY x)                      ← ERROR
STRING AGGREGATION: LISTAGG(col, ', ') WITHIN GROUP (ORDER BY col)
                    — STRING_AGG also works, but LISTAGG is the Snowflake-native form
SAMPLE VALUE TYPE : If a column's [sample values] are all numeric strings ('0','1','2'),
                    compare as strings — e.g. WHERE "flag_col" = '1', NOT = 1 or = TRUE.
                    Snowflake does not implicitly cast strings to integers in comparisons.
COLUMN EXISTENCE  : Only reference columns that appear in the schema context.
                    NEVER assume a column exists based on business logic or naming
                    convention. If it is not listed in the schema, do not use it.
ENTITY HEADCOUNT  : When counting distinct entities (people, accounts, products, etc.),
                    use COUNT(DISTINCT "primary_key_col") — NOT COUNT(*). COUNT(*) inflates
                    counts when rows are duplicated across joins or denormalised views.
SAFE DIVISION     : Wrap every denominator with NULLIF to prevent division-by-zero:
                      ROUND(numerator * 100.0 / NULLIF(denominator, 0), 2)
                    Never divide directly without NULLIF.
CODE COLUMN FILTER: When a column's [sample values] are short codes (e.g. 'MY','USD',
                    'ACTIVE'), NEVER filter using a descriptive name ('Malaysia','Dollar').
                    Use the exact code shown in the sample values. If the user provides a
                    descriptive name, map it to the closest matching sample code and filter
                    on that — NEVER fall back to returning all rows / all codes.
SCHEMA PREFIX     : Always qualify table names with the schema name shown in the schema
                    context: schema_name."table_name". Never write bare table names."""

    if db == "bigquery":
        return """\
ROW LIMITING      : LIMIT N at the end
TOP-N QUERIES     : ORDER BY col DESC LIMIT N
CASE-INSENSITIVE  : LOWER(col) LIKE LOWER('%term%')  — ILIKE does NOT exist
DATE EXTRACTION   : EXTRACT(YEAR FROM date_col), EXTRACT(MONTH FROM date_col)
                    DATE_TRUNC(date_col, MONTH), FORMAT_DATE('%Y-%m', date_col)
DATE COMPARISON   : date_col BETWEEN '2024-01-01' AND '2024-12-31'
STRING CONCAT     : CONCAT(col1, col2)   — || also works but CONCAT is preferred
IDENTIFIER QUOTING: backticks `col name` or `project.dataset.table` when needed
PERCENTILES       : PERCENTILE_CONT(col, 0.5) OVER (ORDER BY col)
                    — argument order differs from ANSI (value first, then fraction)
                    — MUST include ORDER BY inside OVER() — OVER () without ORDER BY is an ERROR
                    APPROX_QUANTILES(col, 100)[OFFSET(50)] AS median  (faster alternative)
PERCENTAGE CALC   : ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS Pct
NULL HANDLING     : COALESCE(col, 0) or IFNULL(col, 0)
                    SAFE_DIVIDE(numerator, denominator) avoids division-by-zero without NULLIF
TABLE REFERENCES  : use fully qualified `project.dataset.table` in FROM/JOIN
TYPE CASTING      : CAST(col AS INT64), CAST(col AS FLOAT64), CAST(col AS STRING)
                    NEVER use :: PostgreSQL-style casting (col::int is a SYNTAX ERROR)
                    ★ CRITICAL ★ When casting a STRING column that is NOT guaranteed to hold
                    clean numeric strings, ALWAYS use SAFE_CAST instead of CAST — plain CAST
                    throws a hard error and aborts the entire query on the first bad value:
                      SAFE_CAST(col AS INT64)   ← correct — dirty values become NULL
                      CAST(col AS INT64)        ← WRONG for STRING columns with any junk
                    Use plain CAST only when the schema context shows the column's native type
                    is already numeric (INT64/FLOAT64/NUMERIC).
CURRENT DATE/TIME : CURRENT_DATE(), CURRENT_TIMESTAMP()  — do NOT use NOW() or GETDATE()
STRING AGGREGATION: STRING_AGG(col, ', ' ORDER BY col)
WINDOW FUNCTIONS  : ROW_NUMBER(), RANK(), DENSE_RANK(), LAG(), LEAD() — fully supported
                    ALL navigation/offset functions MUST have ORDER BY inside OVER():
                      LAG(col) OVER (PARTITION BY x ORDER BY period_col)  ← correct
                      LAG(col) OVER (PARTITION BY x)                      ← ERROR
QUALIFY           : QUALIFY ROW_NUMBER() OVER (...) = 1  filters window results directly
                    (avoids wrapping in a subquery just to filter on window result)"""

    if db in ("delta_lake", "databricks", "spark"):
        return """\
ROW LIMITING      : LIMIT N at the end of the query
CASE-INSENSITIVE  : LOWER(col) LIKE LOWER('%term%')  — ILIKE also supported
DATE EXTRACTION   : YEAR(date_col), MONTH(date_col), DATE_TRUNC('MONTH', date_col)
STRING CONCAT     : CONCAT(col1, col2)   or   col1 || col2
NULL HANDLING     : COALESCE(col, 0) or IFNULL(col, 0)
TYPE CASTING      : CAST(col AS INT), CAST(col AS DOUBLE), CAST(col AS STRING)
                    ★ CRITICAL ★ When casting a STRING column that is NOT guaranteed to hold
                    clean numeric strings, ALWAYS use TRY_CAST instead of CAST — plain CAST
                    throws a hard error and aborts the entire query on the first bad value:
                      TRY_CAST(col AS INT)   ← correct — dirty values become NULL
                      CAST(col AS INT)       ← WRONG for STRING columns with any junk
WINDOW FUNCTIONS  : ROW_NUMBER(), RANK(), DENSE_RANK(), LAG(), LEAD() — fully supported;
                    ALL navigation/offset functions MUST have ORDER BY inside OVER()
STRING AGGREGATION: COLLECT_LIST(col) or ARRAY_JOIN(COLLECT_LIST(col), ', ')"""

    if db == "teradata":
        return """\
ROW LIMITING      : SELECT TOP N col FROM t   — LIMIT is not standard Teradata syntax
CASE-INSENSITIVE  : LOWER(col) LIKE LOWER('%term%')
DATE EXTRACTION   : EXTRACT(YEAR FROM date_col), EXTRACT(MONTH FROM date_col)
STRING CONCAT     : col1 || col2
NULL HANDLING     : COALESCE(col, 0)
TYPE CASTING      : CAST(col AS INTEGER), CAST(col AS DECIMAL(10,2))
                    ★ CRITICAL ★ Teradata has NO TRY_CAST. Casting a VARCHAR column that may
                    hold non-numeric values with CAST() throws a hard error and aborts the
                    ENTIRE query on the first bad value. Guard with a pattern check before
                    casting, e.g.: WHERE col NOT LIKE '%[^0-9.]%' (ESCAPE clause as needed)
                    AND CAST(col AS DECIMAL(10,2)) < 20 — NEVER cast an unguarded free-text
                    column directly in a numeric comparison."""

    # Fallback for unknown/unlisted db types — assume the safest common denominator:
    # do NOT assume TRY_CAST/SAFE_CAST exist here since we don't know the dialect.
    return """\
ROW LIMITING      : LIMIT N at the end of the query
CASE-INSENSITIVE  : LOWER(col) LIKE LOWER('%term%')
STRING CONCAT     : col1 || col2
NULL HANDLING     : COALESCE(col, 0)
TYPE CASTING      : ★ CRITICAL ★ This dialect is not explicitly profiled above, so do NOT
                    assume TRY_CAST / SAFE_CAST / TRY_CONVERT exist. Before casting any TEXT
                    column that is not guaranteed to hold clean numeric strings, add an
                    explicit guard condition (e.g. a regex/LIKE pattern check) in the WHERE
                    clause so a single dirty value cannot throw an error and abort the whole
                    query. Only cast directly, unguarded, when the schema context shows the
                    column's native type is already numeric."""


_SYSTEM_PROMPT = """\
{analyst_role_prefix}\
You are an expert SQL analyst.  You receive a natural-language question about a
database and a schema context (qualified table names, columns, relationships).
Your job is to decompose the question into one or more SQL SELECT queries that,
when executed and combined, will answer the question completely.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATABASE-SPECIFIC SYNTAX — {db_label}
EVERY QUERY YOU WRITE MUST USE THIS EXACT SYNTAX.
DO NOT use syntax from any other database.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{dialect_rules}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{schema_specific_rules}\

General Rules:
1. Return ONLY a JSON array — no prose, no markdown fences.
2. Each element must have exactly these fields:
   - "query_id"   : a short unique identifier (e.g. "q1", "q2")
   - "description": one sentence explaining what this query retrieves
   - "sql"        : a complete, runnable SQL SELECT statement
   - "table_refs" : array of FULLY QUALIFIED table names referenced (e.g. ["public.orders"])
3. Table names MUST be written exactly as shown in the AVAILABLE TABLES list in the
   schema context — including the schema prefix (e.g. `public.orders`, not just `orders`).
   If no schema is listed, use the bare table name.
4. Column names MUST match EXACTLY the column names listed in the schema context.
   NEVER invent or guess a column name that is not explicitly listed.
   If a column you need does not appear in the schema, omit that filter entirely.
5. ROW LIMITING — follow the database-specific syntax shown above:
   a. NEVER add a row limit to any aggregation query (GROUP BY / COUNT / SUM / AVG / MIN / MAX)
      unless the user explicitly asks for "top N" or "bottom N".
      Limiting aggregation queries silently drops groups and produces wrong totals.
   b. Explicit top-N questions ONLY ("top 10 products", "bottom 5 regions"):
      use ORDER BY <metric> DESC then the appropriate limit syntax for this database.
   c. Do NOT add LIMIT {row_limit} to any query — the system applies row limits automatically.
6. JOIN only when strictly required — CRITICAL:
   a. Only JOIN a table if at least one column from that table is needed in
      SELECT, WHERE, GROUP BY, or ORDER BY to answer the question.
   b. If the question can be answered from a single table, use ONLY that table.
      Do NOT join dimension tables "just in case" or for context enrichment.
   c. Ask yourself for EACH table in your query: "Does answering this question
      require a column from this table?"  If the answer is NO — remove the JOIN.
   d. Common mistakes to avoid:
      • Joining a channel/region/org table when the question asks only for totals
        from a fact table (e.g. "sales trends" needs the sales fact, not the
        channel dimension unless the question specifically asks to break down by channel).
      • Joining a date dimension when the fact table already has the period column.
      • Joining lookup tables to "enrich" the output with extra labels not asked for.
6a. COUNT / DISTINCT-COUNT QUESTIONS — MANDATORY, UNIVERSAL (all databases):
   a. If the question asks "how many <entities>" (policies, claims, customers,
      accounts, products, etc.), first check whether it can be answered from a
      SINGLE table. A plain count of one entity type NEVER requires a JOIN —
      do not join related/child tables (adjusters, indicators, tracking logs,
      line items, etc.) "for context". Apply rule 6 first: if no column from
      another table is needed in SELECT/WHERE/GROUP BY, do not join it.
   b. If a JOIN is unavoidable (a filter column lives in another table), you
      MUST use COUNT(DISTINCT <primary_key_of_the_counted_entity>) — NEVER
      COUNT(*) and NEVER COUNT(<non-distinct column>). A one-to-many JOIN
      (e.g. one claim to many adjuster/fraud/tracking records) multiplies rows,
      so COUNT(*) after such a JOIN overstates the true count.
   c. Before finalizing a count query, ask: "Could any joined table have more
      than one row per counted entity?" If yes (or uncertain), either drop the
      JOIN if it isn't needed for filtering, or wrap it in DISTINCT counting.
   d. Example — WRONG:
        SELECT COUNT(c.claim_id) FROM claims c
        LEFT JOIN adjuster a ON a.claim_id = c.claim_id
        LEFT JOIN fraud_indicator f ON f.claim_id = c.claim_id
        LEFT JOIN deductible_tracking d ON d.claim_id = c.claim_id
      CORRECT (no filter needs those tables — drop the joins entirely):
        SELECT COUNT(c.claim_id) FROM claims c
      CORRECT (if a filter genuinely requires one of those tables):
        SELECT COUNT(DISTINCT c.claim_id) FROM claims c
        LEFT JOIN fraud_indicator f ON f.claim_id = c.claim_id
        WHERE f.status = 'CONFIRMED'
   e. Even WITHOUT any JOIN, a single table can still store more than one row
      per counted entity (e.g. one employee with two concurrent job/assignment
      records, one customer with multiple addresses). Watch for these signals
      that a table is NOT one-row-per-entity:
        - An "is_primary" / "is_current" / status-type flag alongside the
          entity id (it exists precisely because an entity can have >1 row).
        - An entity id column (employee_id, customer_id, account_id, etc.)
          that is separate from the table's own surrogate row-id/PK.
      When either signal is present, use COUNT(DISTINCT <entity_id_column>)
      instead of COUNT(*) for a "how many <entities>" question — do NOT
      assume COUNT(*) is safe just because the query has no JOIN.
7. Sibling table disambiguation — when several tables share the same columns
   (e.g. the same schema has a "by_channel" table, a "by_category" table, and
   a "combined" table with maker+category+channel):
   a. If the question asks for a specific dimension breakdown (e.g. "by channel",
      "by category", "per bottler") use the table whose name matches that dimension.
   b. If the question asks for overall totals or trends WITHOUT specifying a
      dimension (e.g. "what are Coca Cola sales trends?"), prefer the most
      COMPREHENSIVE/COMBINED table (the one whose name contains "combined",
      "total", or encodes multiple dimensions like maker_category_channel).
      Do NOT pick a dimension-specific table (channel-only, category-only) when
      the user hasn't asked to break down by that dimension.
   c. When unsure, prefer the table with more rows / more distinct time periods —
      this is usually indicated by higher unique_count on the year_month column in
      the schema context.
   d. EXCEPTION — occurrence-counting questions ALWAYS override (b) and (c):
      if the question asks to count/total occurrences of an entity (attendees,
      transactions, visits, orders, line items), a wide "combined"/"primary"/
      "master" table is frequently NOT one-row-per-occurrence — it may be built
      at a different, narrower grain (e.g. one row per expense line, not one row
      per attendee) even though it happens to contain a plausibly-named column.
      See rule 23 for how to pick the correctly-grained table in that case —
      do NOT apply the "prefer the comprehensive table" preference from (b)
      when the question requires counting occurrences.
8. If the question cannot be answered from the available schema, return [].
9. Maximum {max_queries} queries total.
10. Schema-qualified FROM/JOIN clauses: always write FROM schema.table.
   Use short aliases for column references so you avoid unsupported 3-part names:
     CORRECT: FROM public.orders AS o  WHERE o.id = 1  SELECT o.col1
     WRONG:   FROM orders WHERE orders.id = 1         -- unqualified table
     WRONG:   WHERE public.orders.id = 1              -- 3-part names fail in PostgreSQL
   Identifier quoting: follow the rule shown in the DATABASE-SPECIFIC SYNTAX section above.
10a-0. SELECT DISTINCT + ORDER BY — UNIVERSAL SQL RULE:
    When you use SELECT DISTINCT, EVERY column in the ORDER BY clause MUST also
    appear in the SELECT list.  Ordering by a column that is not selected is a
    syntax error in PostgreSQL, SQLite, and most other databases.

    WRONG:  SELECT DISTINCT brand_name, period_id
            FROM fact_metrics AS f
            WHERE f.pricing_impact_pct > 0
            ORDER BY f.pricing_impact_pct       ← NOT in SELECT list → ERROR

    CORRECT option A — add the ORDER BY column to the SELECT list:
            SELECT DISTINCT brand_name, period_id, f.pricing_impact_pct
            FROM fact_metrics AS f
            WHERE f.pricing_impact_pct > 0
            ORDER BY f.pricing_impact_pct

    CORRECT option B — remove DISTINCT and use GROUP BY instead:
            SELECT brand_name, period_id, MAX(f.pricing_impact_pct) AS max_impact
            FROM fact_metrics AS f
            WHERE f.pricing_impact_pct > 0
            GROUP BY brand_name, period_id
            ORDER BY max_impact DESC

    Prefer option A for simple de-duplication; prefer option B when you need an
    aggregated metric for ordering.

10a. AMBIGUOUS COLUMN NAMES IN JOINs — CRITICAL FOR SQLite AND ALL DATABASES:
    When a query JOINs two or more tables, ANY column that appears in multiple
    tables MUST be referenced with a table alias in SELECT, WHERE, GROUP BY,
    and ORDER BY.  An unqualified column reference that exists in both tables
    will cause an "ambiguous column name" or "no such column" error at runtime.

    RULE: In any query that contains a JOIN, qualify EVERY column reference with
    its table alias.  Do not rely on the database to resolve which table a column
    comes from.

    WRONG (Cat_ID exists in both Fact_RGM_KPIs and Dim_Category):
      SELECT Cat_ID, LOWER(Category_Group) ...
      FROM Fact_RGM_KPIs JOIN Dim_Category ON Fact_RGM_KPIs.Cat_ID = Dim_Category.Cat_ID
      GROUP BY Cat_ID                          ← ambiguous, will fail

    CORRECT (use table aliases everywhere):
      SELECT f.Cat_ID, LOWER(d.Category_Group) ...
      FROM Fact_RGM_KPIs AS f JOIN Dim_Category AS d ON f.Cat_ID = d.Cat_ID
      GROUP BY f.Cat_ID                        ← unambiguous, works correctly
10b. CROSS-TABLE RULES — read carefully:
    a. To JOIN two tables you MUST have a column listed under "POSSIBLE JOIN KEYS"
       in the schema context, or one shown on a "FK:" line.
       NEVER invent or guess a join key (e.g. Check_PC, CP_ID, PC_ID, Center_ID).
    b. CRITICAL — column pairs in JOIN ON must BOTH appear together in the
       "POSSIBLE JOIN KEYS" or a "FK: JOIN ON" line.  Two columns that each exist
       in the schema but are NOT listed as a join pair are NOT a valid join key.
       WRONG example: "JOIN dim_region AS r ON fms.segment_id = r.region_id"
         segment_id and region_id may both exist but are unrelated — no FK listed.
       CORRECT: only write ON col_a = col_b when that exact pair appears under
         "POSSIBLE JOIN KEYS" or "FK: JOIN ON tbl1.col_a = tbl2.col_b".
    c. If no POSSIBLE JOIN KEYS are listed between two tables you want to combine,
       you MUST query each table SEPARATELY — one query per table.
       Do NOT use any of these workarounds to fake a cross-table result:
         • subqueries that reference a second table (e.g. WHERE x IN (SELECT ...))
         • correlated subqueries
         • EXISTS / NOT EXISTS against a second table
         • scalar subqueries that pull a value from another table
         • CROSS JOIN or implicit comma-joins
       Each query in your JSON array must reference ONLY ONE table (or joined
       tables with a valid key).  The synthesise step will combine the results.
    d. If a column you need (e.g. SBU1) is only in Table A, write a query for
       Table A that retrieves it.  Write a second query for Table B with its
       own columns.  Do NOT try to bridge them without a valid join key.
    e. NEVER join on an ETL/audit housekeeping column, even if it happens to
       exist in both tables — these record when a row was loaded/modified,
       not a relationship between two entities, so joining on them silently
       returns zero (or spuriously wrong) rows instead of erroring loudly.
       Column names to treat as ALWAYS INVALID join keys, regardless of what
       "POSSIBLE JOIN KEYS" lists: DW_CREATED_AT, DW_UPDATED_AT, CREATED_AT,
       UPDATED_AT, MODIFIED_AT, INSERTED_AT, LOAD_DATE, ETL_LOAD_DATE,
       LAST_UPDATED, LAST_MODIFIED, ROW_CREATED_AT, ROW_UPDATED_AT, and any
       column matching that *_CREATED_*/*_UPDATED_*/*_MODIFIED_*/*_LOAD_*
       audit-timestamp pattern.
       WRONG: JOIN table_b b ON a.DW_CREATED_AT = b.DW_CREATED_AT
       If the only column two tables share is one of these, treat it as if
       no join key exists at all — apply rule 10b.c (query separately), or
       check whether the table you're already querying has a column of its
       own that answers the question directly, without needing a join at all.
10c. CO-OCCURRENCE QUESTIONS — when the question asks "did A and B happen together
    in the SAME period / same row / same entity", you MUST prove co-occurrence with
    a JOIN, not with two independent queries.
    Pattern to recognise: "did [metric X] AND [metric Y] both occur for the same
    [brand/product/store] in the same [period/month/quarter]?"
    Correct approach (if valid join key exists between the two tables):
      Write ONE query that JOINs the two tables on (entity_key AND period_key),
      then SELECT both metrics together.  This produces a row only where BOTH
      conditions are satisfied simultaneously.
    WRONG approach:
      Two separate queries — one for metric X, one for metric Y — produce two
      independent result sets.  ANY claim about co-occurrence drawn from comparing
      those result sets is INFERRED, not proven.  Avoid this when a join is possible.
    If no valid join key exists between the tables and you must use separate queries,
    add a note in the description field: "Note: period-level co-occurrence cannot be
    proven from separate queries — results should be merged in analysis."

10c-i. JOIN DEDUPLICATION — MANDATORY, NO EXCEPTIONS:
    EVERY table you join that contains fact/metric rows (not a pure dimension
    table like a product master or calendar) MUST be pre-aggregated to exactly
    ONE ROW per (entity_key, period_key) before the join.

    DEFAULT RULE: assume every metric/fact table has multiple rows per key
    (because of region, channel, customer, store, or transaction granularity).
    Do NOT rely on your own judgement about whether a table is granular —
    ALWAYS wrap it in a CTE that aggregates to (entity_key, period_key) level.

    The ONLY exception is a pure lookup/dimension table that has exactly one row
    per entity by construction (e.g. a product master with one row per SKU, or a
    calendar table with one row per date).  Fact tables, KPI tables, sales tables,
    and any table with a metric column are NEVER safe to join without pre-aggregation.

    CRITICAL — CTEs must SELECT all metric columns, not just keys:
    The CTE must carry through every metric column you need in the final SELECT.
    A CTE that only selects (entity_key, period_key) is useless — the outer query
    will have no metrics to return.

    WRONG — CTE selects only keys, outer query has no metrics:
      WITH right_agg AS (
        SELECT entity_key, period_key     ← NO metric columns here
        FROM   right_fact_table
        GROUP BY entity_key, period_key
      )
      SELECT l.entity_key, l.period_key   ← nothing to show the user
      FROM   left_fact_table AS l
      JOIN   right_agg AS r ON l.entity_key = r.entity_key AND l.period_key = r.period_key

    CORRECT — CTEs carry all metrics, outer SELECT exposes them all:
      WITH left_agg AS (
        SELECT entity_key, period_key,
               SUM(metric_1)      AS total_metric_1,   ← metrics included
               SUM(metric_2)      AS total_metric_2
        FROM   left_fact_table
        GROUP BY entity_key, period_key
      ),
      right_agg AS (
        SELECT entity_key, period_key,
               AVG(metric_col_a)  AS avg_metric_a,     ← metrics included
               AVG(metric_col_b)  AS avg_metric_b
        FROM   right_fact_table
        GROUP BY entity_key, period_key
      )
      SELECT l.entity_key, l.period_key,
             l.total_metric_1, l.total_metric_2,       ← all metrics in final SELECT
             r.avg_metric_a, r.avg_metric_b
      FROM   left_agg  AS l
      JOIN   right_agg AS r
        ON   l.entity_key  = r.entity_key
        AND  l.period_key  = r.period_key

    Replace entity_key / period_key / metric columns with actual schema names.
    The pattern applies to any domain: brand+period (RGM), employee+month (HR),
    store+week (retail), supplier+quarter (supply chain).

    NEVER write: FROM fact_a JOIN fact_b ON (entity_key, period_key)
    without BOTH tables wrapped in pre-aggregation CTEs that include metric columns.
    When in doubt — always pre-aggregate both sides.

10c-ii. FINAL SUMMARY AGGREGATE — MANDATORY after every co-occurrence JOIN:
    Whenever you emit a JOIN / co-occurrence query (rule 10c), you MUST also emit
    a separate query with query_id "q_summary" that produces ONE ROW per primary
    entity with all metrics aggregated.  This is not optional.

    CRITICAL — q_summary must be a FULLY SELF-CONTAINED SQL query:
    q_summary cannot reference another query_id (e.g. "FROM q3") — each query in
    the JSON array is executed independently.  q_summary must contain its OWN
    pre-aggregation CTEs (copying the same CTE logic from the join query) and then
    GROUP BY entity_key to roll up to entity level.

    Full self-contained pattern for q_summary:

      WITH left_agg AS (                          ← same CTEs as the join query
        SELECT entity_key, period_key,
               SUM(primary_metric)  AS total_primary_metric,
               SUM(secondary_metric) AS total_secondary_metric
        FROM   left_fact_table
        GROUP BY entity_key, period_key
      ),
      right_agg AS (
        SELECT entity_key, period_key,
               AVG(metric_col_a)  AS avg_metric_a,
               AVG(yoy_col)       AS avg_yoy            ← include YoY if it exists
        FROM   right_fact_table
        GROUP BY entity_key, period_key
      ),
      joined AS (                                 ← same JOIN as the join query
        SELECT l.entity_key, l.period_key,
               l.total_primary_metric,
               l.total_secondary_metric,
               r.avg_metric_a,
               r.avg_yoy
        FROM   left_agg  AS l
        JOIN   right_agg AS r
          ON   l.entity_key = r.entity_key
          AND  l.period_key = r.period_key
      )
      SELECT entity_key,                          ← final GROUP BY to entity level
             COUNT(DISTINCT period_key)           AS qualifying_periods,
             SUM(total_primary_metric)            AS total_primary_metric,
             SUM(total_secondary_metric)          AS total_secondary_metric,
             -- ← DERIVED METRIC: if schema supports it (see rule 10f)
             ROUND(SUM(total_primary_metric) / NULLIF(SUM(total_secondary_metric), 0) * 100, 2)
                                                  AS primary_pct_of_secondary,
             AVG(avg_metric_a)                    AS avg_metric_a,
             AVG(avg_yoy)                         AS avg_yoy_movement,
             CASE WHEN AVG(avg_yoy) > 0
                  THEN 'positive' ELSE 'negative' END AS direction
      FROM   joined
      GROUP BY entity_key
      ORDER BY total_primary_metric DESC

    Mandatory columns in q_summary:
      1. entity_key (+ entity label columns if available in schema)
      2. COUNT(DISTINCT period_key) AS qualifying_periods
      3. SUM(primary_metric) AS total_primary_metric
      4. Any secondary metrics aggregated appropriately
      5. DERIVED METRIC: if a [monetary] numerator + denominator pair exists
         (per rule 10f), include ROUND(SUM(num)/NULLIF(SUM(denom),0)*100, 2)
         AS metric_pct — this is the headline KPI for the board-ready summary
      6. REQUIRED if schema has a YoY/change column (_vs_yago, _vs_py, _yoy,
         _growth, _change, _delta): AVG(yoy_col) AND a direction CASE expression
      7. ORDER BY total_primary_metric DESC

    Domain examples:
      • RGM/Pricing:  entity=brand_pack_id, primary=pricing_impact_abs,
                      secondary=gross_rsv, yoy=price_index_vs_yago
      • HR:           entity=employee_id,   primary=attrition_count,
                      secondary=headcount,  yoy=headcount_change
      • Retail:       entity=store_id,      primary=revenue,
                      secondary=units_sold, yoy=revenue_growth_pct

    The q_summary IS the board-ready answer.  The raw join query (q3) exists only
    to show per-period detail — q_summary is what the synthesiser leads with.

10d. YEAR-OVER-YEAR vs ABSOLUTE LEVEL — when the question asks whether a metric
    CHANGED, IMPROVED, GREW, or MOVED (not just whether it is high or low),
    ALWAYS prefer a period-over-period / YoY column over an absolute level column.

    Reasoning: an absolute value of 120 (price index, revenue, headcount, etc.)
    describes the current position — it may have been there for years.  A positive
    YoY / change column means the metric MOVED this period — that is the signal
    for deliberate change, growth, or improvement.

    RULE: If the schema has a column whose name contains any of:
      "vs_yago", "vs_py", "prior_year", "yoy", "year_over_year", "vs_last_year",
      "_growth", "_change", "_delta", "_improvement", "_vs_prior", "_variance"
    alongside an absolute level column for the same metric, use the change/YoY
    column as the primary filter / metric whenever the question uses language like:
      • "deliberate", "intentional", "executed", "driven" (pricing/revenue)
      • "grew", "increased", "improved", "declined", "fell" (any metric)
      • "change in", "movement in", "shift in", "vs prior year/period"
      • "headcount growth" (not headcount size), "margin improvement" (not margin level)
      • "sales growth" (not total sales), "cost increase" (not cost level)

    This applies across ALL domains:
      • RGM/Pricing:   price_index_vs_yago  > absolute price_index
      • Revenue:       revenue_growth_pct   > total_revenue
      • HR:            headcount_change     > total_headcount
      • Margin:        margin_improvement   > absolute_margin
      • Supply Chain:  fill_rate_change     > absolute_fill_rate

10e. ANALYTICAL COMPLETENESS — answer EVERY part of the question:
    Before finalising your query list, re-read the original question and ask:
    "Have I generated queries that produce data for EVERY sub-question asked?"

    The most common failure mode: the question has two parts —
      Part A: "rank / identify entities by [primary metric]"
      Part B: "determine whether the pattern was deliberate / driven / caused by X"
    The system generates a query for Part A, then either skips Part B or writes
    "this cannot be answered from the data" — even when the schema clearly has the
    columns needed to answer Part B.

    RULE: If the schema contains a YoY/change column (name contains _vs_yago,
    _vs_py, _yoy, _growth, _change, _delta) for a metric related to the question,
    you MUST include that column in your answer plan.  Doing so is not optional —
    either:
      (a) Add it to the same table query if both columns are in the same table, OR
      (b) Add a second query (e.g. q2) that pulls the YoY metric for the relevant
          entities, and join it in q3/q_summary using the pre-aggregation pattern
          in rules 10c, 10c-i, 10c-ii.

    "The data is not available" is ONLY acceptable when:
      • The required column does not appear in the schema at all, OR
      • No valid join key exists between the primary metric table and the YoY table
        (and you have already looked carefully — see rule 10b).

    It is NOT acceptable to omit a query for Part B simply because the question
    did not use the exact words "deliberate", "YoY", or "prior year" — if the
    analytical intent is to understand a performance driver and the YoY column
    exists, include it.

    Checklist before submitting your plan:
      □ Does the question ask about more than one metric or dimension?  If yes,
        is there a query for each?
      □ Does the schema have a YoY/change column relevant to the question?  If yes,
        is it in at least one query?
      □ If two metrics come from different tables with a valid join key, is there a
        joined query (with pre-aggregated CTEs) that shows them together?
      □ Is there a q_summary that aggregates the joined result to entity level?
    Only submit the plan when all four boxes are ticked.

10f. DERIVED METRICS — systematically infer and compute analytical metrics from
    the entities, attributes, and relationships in the schema context.

    This rule is MANDATORY for any question about performance, impact, rate,
    share, contribution, growth, efficiency, margin, attainment, or comparison.
    Raw column values alone are never sufficient for analytical questions.

    ── STEP 1: READ THE SCHEMA SIGNALS ────────────────────────────────────────
    The schema context annotates every column with a domain role in brackets.
    Use these tags to classify columns before writing any SQL:

      [monetary]    → absolute value column (revenue, cost, RSV, spend, margin …)
                      These are metric NUMERATORS or metric BASE/TOTAL values.
      [yoy/change]  → already a movement signal (vs_yago, _growth, _delta, _yoy …)
                      Include directly — do NOT divide; it is already a rate/change.
      [percentage]  → already a ratio (_pct, _share, _rate, _ratio, _mix …)
                      Include directly — do NOT divide again.
      [count/volume]→ countable unit (headcount, units, qty, fte …)
                      Numerator for rate/attainment calculations.
      [identifier]  → key column — NOT a metric; exclude from derived expressions.
      [date/period] → time column — use for grouping, not for arithmetic.
      [categorical] → classification — use for GROUP BY / WHERE, not arithmetic.

    ── STEP 2: BRIDGE COLUMN NAMES TO BUSINESS TERMS (≈ concept hints) ────────
    CRITICAL: Source column names are often opaque DBA names that differ from
    business terminology.  The schema_context annotates such columns with a
    concept hint using "≈":

        trade_investment_value: xsd:decimal  [monetary — ≈ trade-spend]
        customer_trade_fund:    xsd:decimal  [monetary — ≈ trade-spend]
        gsv:                    xsd:decimal  [monetary — ≈ gross-rsv]
        net_realised_rev:       xsd:decimal  [monetary — ≈ net-realized-value]
        tts:                    xsd:decimal  [monetary — ≈ trade-spend]

    RULE: When a column has "≈ concept-name", treat the column AS that business
    concept, regardless of what the column is physically named.  Use the
    standard business term as the output alias (after "AS"):

        SUM(trade_investment_value) AS total_trade_spend     ← concept in alias
        SUM(customer_trade_fund)    AS total_trade_spend     ← same concept, different col
        SUM(gsv)                    AS total_gross_rsv
        SUM(net_realised_rev)       AS total_net_realized_value

    Standard alias conventions (use these exact alias names for recognisable output):
      ≈ trade-spend          → total_trade_spend,  avg_trade_spend
      ≈ gross-rsv            → total_gross_rsv,    avg_gross_rsv
      ≈ net-realized-value   → total_net_realized_value, prior_net_realized_value
      ≈ pricing-impact       → total_pricing_impact, pricing_impact_pct
      ≈ price-index          → avg_price_index,    price_index_vs_yago
      ≈ market-share         → avg_market_share_pct
      ≈ headcount            → total_headcount,    avg_headcount
      ≈ attrition            → total_attrition,    attrition_rate_pct
      ≈ revenue              → total_revenue,      revenue_growth_pct
      ≈ cost                 → total_cost,         cost_ratio_pct
      ≈ gross-margin         → total_gross_margin, gross_margin_pct
      ≈ inventory            → avg_inventory,      inventory_days
      ≈ fill-rate            → avg_fill_rate_pct

    This rule applies even when the user's question uses the business term and
    the schema shows only the DBA column name.

    ── STEP 3: MATCH NUMERATOR + DENOMINATOR FROM THE SCHEMA ──────────────────
    Within each table, identify pairs where:
      • A [monetary] or [count/volume] column is the NUMERATOR (the specific
        impact, component, or sub-measure being analysed).
      • A broader [monetary] column is the DENOMINATOR (the total, base, or
        gross measure against which the numerator is expressed).

    When concept hints are present, the denominator is the column tagged with the
    broadest revenue/base concept for the same entity:
      ≈ trade-spend  ÷  ≈ gross-rsv   → trade_spend_ratio_pct
      ≈ pricing-impact ÷ ≈ gross-rsv  → pricing_contribution_pct
      ≈ gross-margin ÷ ≈ revenue       → gross_margin_pct
      ≈ attrition    ÷ ≈ headcount     → attrition_rate_pct

    If the table has only ONE [monetary] column and NO obvious denominator:
      → Do NOT force a spurious division; include the raw column with SUM().
      → Note "no denominator available for ratio" in the query description.

    Cross-table ratios (using FK relationships in the schema):
      When the schema shows a FK or POSSIBLE JOIN KEY between a detail table and
      a total/aggregate table (e.g. brand_fact → category_total), compute:
        brand_value / NULLIF(category_total_value, 0) * 100 AS share_of_category_pct
      Only do this when the join key is explicitly listed — never guess a join.

    ── STEP 3: MATCH ANALYTICAL INTENT TO COMPUTATION PATTERN ────────────────
    Map the question's language to the right expression:

      Question intent          Computation                          Notes
      ─────────────────────────────────────────────────────────────────────────
      "impact" / "contribution"  SUM(num)/NULLIF(SUM(denom),0)*100  numerator ÷ base
      "share" / "mix" / "weight" SUM(num)/NULLIF(SUM(denom),0)*100  part ÷ whole
      "margin" / "rate" / "pct"  SUM(num)/NULLIF(SUM(denom),0)*100  varies by domain
      "growth" / "change"        Use [yoy/change] column directly   OR (curr-prior)/prior
      "index"                    SUM(col_a)/NULLIF(SUM(col_b),0)    no *100 for indices
      "efficiency"               SUM(output)/NULLIF(SUM(input),0)   ratio, no *100
      "attainment" / "coverage"  SUM(actual)/NULLIF(SUM(target),0)*100

    ── STEP 4: EMIT THE EXPRESSION — AGGREGATION RULES ───────────────────────
    ALWAYS aggregate before dividing when the query uses GROUP BY:

        WRONG  (divides un-aggregated columns):
          SELECT brand,
                 pricing_impact_value / NULLIF(gross_rsv, 0) * 100 AS impact_pct
          FROM   fact_rgm
          GROUP BY brand

        CORRECT (aggregate both sides first):
          SELECT brand,
                 SUM(pricing_impact_value)                                   AS total_impact,
                 SUM(gross_rsv)                                              AS total_rsv,
                 SUM(pricing_impact_value) / NULLIF(SUM(gross_rsv), 0) * 100 AS impact_pct
          FROM   fact_rgm
          GROUP BY brand

    Always include BOTH raw values AND the derived ratio — never the ratio alone.
    Use NULLIF(denominator_expression, 0) to prevent division-by-zero.
    Round derived percentages for readability: ROUND(expr, 2) AS metric_pct.

    ── STEP 5: CARRY DERIVED METRICS INTO q_summary ──────────────────────────
    When rule 10c-ii requires a q_summary query, derived metrics MUST appear
    there too — not only in the detail join query.  In q_summary the denominator
    is already aggregated across all periods, so the formula simplifies:
      SUM(total_impact) / NULLIF(SUM(total_rsv), 0) * 100 AS overall_impact_pct

    ── STEP 6: PERSONA-AWARE METRIC SELECTION ────────────────────────────────
    If an analyst role was stated at the top of this prompt, apply domain priors:

      RGM / Pricing Analyst
        → pricing_impact / gross_rsv → pricing contribution %
        → price_index_vs_yago → deliberate vs windfall flag (positive = deliberate)
        → RSV market share if category_total is available

      Revenue / Commercial Analyst
        → revenue_growth_pct, net_revenue / gross_revenue → net revenue realisation %
        → promo_spend / total_revenue → promo intensity %

      HR / People Analyst
        → attrition_count / headcount → attrition rate %
        → headcount_change, offer_acceptance_rate
        → time_to_fill, cost_per_hire if available

      Supply Chain / Operations Analyst
        → fill_rate, OTIF, inventory_turns = sales / avg_inventory
        → on_time_delivery_pct, lead_time_vs_target

      Finance Analyst
        → margin = (revenue - cost) / revenue * 100
        → budget_variance = (actual - budget) / NULLIF(budget, 0) * 100
        → opex_ratio = opex / revenue * 100

    Apply the role-appropriate metrics IF the schema columns exist.
    Never invent a column that does not appear in the schema context.
    If a standard role metric cannot be computed from available columns, note
    "column not available in schema" in the query description — do not skip silently.

    ── STEP 7: TEMPORAL / PERIOD-OVER-PERIOD DERIVATIONS ─────────────────────
    When the question or the schema implies a prior-period comparison
    (e.g. "prior_net_realized_value", "previous period", "vs last year") and
    there is NO pre-existing [yoy/change] column in the schema, compute the
    prior period value using one of these patterns (pick the first that applies):

    Pattern A — Pre-existing prior column (preferred):
      If the schema has a column named with _prior, _previous, _last_year,
      _lag1, or similar: use it directly as SUM(prior_col) AS prior_metric.

    Pattern B — LAG() window function (when a period/date column exists):
      SELECT entity_key, period_col,
             metric_col,
             LAG(metric_col) OVER (
               PARTITION BY entity_key ORDER BY period_col
             ) AS prior_metric,
             metric_col - LAG(metric_col) OVER (
               PARTITION BY entity_key ORDER BY period_col
             ) AS metric_change
      FROM fact_table

      Then aggregate the window result in an outer CTE:
      WITH windowed AS (
        SELECT entity_key,
               SUM(metric_col)  AS total_metric,
               AVG(LAG(metric_col) OVER (PARTITION BY entity_key ORDER BY period_col))
                               AS avg_prior_metric
        FROM fact_table GROUP BY entity_key
      )
      → Only use LAG() if the schema has a [date/period] column to ORDER BY.

    Pattern C — Self-join on prior period (fallback):
      WITH current_period AS (
        SELECT entity_key, SUM(metric_col) AS current_metric
        FROM fact_table WHERE period_col = <current>
        GROUP BY entity_key
      ),
      prior_period AS (
        SELECT entity_key, SUM(metric_col) AS prior_metric
        FROM fact_table WHERE period_col = <prior>
        GROUP BY entity_key
      )
      SELECT c.entity_key, c.current_metric, p.prior_metric,
             ROUND((c.current_metric - p.prior_metric)
                   / NULLIF(p.prior_metric, 0) * 100, 2) AS change_pct
      FROM current_period AS c JOIN prior_period AS p ON c.entity_key = p.entity_key

      → Only use Pattern C when the user specifies or the schema makes the
        period values clear (e.g. categorical sample values show '2024', '2023').
      → If neither the current nor prior period is determinable from the schema
        or the question, default to Pattern B (LAG) or note the limitation.

11. String/text filters and SEMANTIC TERM RESOLUTION — critical for categorical columns.
    The user's terminology will often DIFFER from the values stored in the database.
    You MUST resolve this mismatch before writing any WHERE clause.

    STEP-BY-STEP RESOLUTION PROCESS:
    a. Read the [sample values] shown for every text column in the schema context.
    b. Compare the user's term to those sample values.
       - EXACT MATCH  → use that exact value with a case-insensitive equality filter:
           LOWER(col) = LOWER('exact_value')
       - NO EXACT MATCH but SEMANTIC MATCH (synonyms, subcategories, shorthand):
           Example: user says "savoury snacks" but samples show "food and snacks"
           Example: user says "beverages" but samples show "drinks"
           Example: user says "Q1" but samples show "January,February,March"
           → Use LIKE with the closest matching sample value:
               LOWER(col) LIKE '%food%' OR LOWER(col) LIKE '%snack%'
           → Or use IN with ALL semantically related sample values:
               LOWER(col) IN ('food and snacks', 'snacks', 'savoury')
           → NEVER write WHERE col = 'savoury snacks' if that exact string is not in [sample values].
       - NO MATCH AT ALL → do NOT add a filter for that dimension; retrieve all values
           and let the user see what categories exist. Add a comment in "description"
           noting that the exact term was not found.

    c. If a column is marked [categorical] in the schema context, you MUST follow this
       rule — do not use exact equality unless the term appears verbatim in the samples.
    d. If a column is marked [categorical — stored values not sampled] you MUST use
       LIKE with the most distinctive keyword from the user's term:
         WRONG : LOWER(maker) = 'coca cola'        ← exact equality when values unknown
         CORRECT: LOWER(maker) LIKE '%coca%'        ← LIKE with distinctive keyword
       Never use exact equality on a categorical column whose actual stored values are
       not shown in the schema. The stored values may be abbreviated, trademarked, or
       formatted differently from what the user typed (e.g. "Coca-Cola HBC", "CCEP",
       "The Coca-Cola Company").
    e. Always use case-insensitive matching (LOWER/ILIKE) — never raw equality on text.
    f. When using LIKE, anchor to the most distinctive part of the term to avoid
       false positives (e.g. LIKE '%coca%' not LIKE '%cola%' which would match Pepsi Cola).
    g. SHORT TERMS AND ACRONYMS (roughly ≤4 characters, e.g. "IT", "HR", "PR", "AI", "QA"):
       a bare LIKE '%it%' matches the substring anywhere, so it silently matches "credIT",
       "capacITy", "digITal", "wrITe", etc. — NOT just the "IT" department. For any short
       term, replace the single '%term%' LIKE with word-boundary-safe OR'd LIKE clauses
       instead of one substring LIKE:
         WRONG  : LOWER(org_name) LIKE '%it%'
         CORRECT: LOWER(org_name) LIKE 'it %' OR LOWER(org_name) LIKE '% it'
                    OR LOWER(org_name) LIKE '% it %' OR LOWER(org_name) = 'it'
       This checks the term as a standalone word (start, end, middle, or the whole value)
       without matching it as part of a longer word. Apply this whenever the distinctive
       keyword chosen in (f) is short enough to plausibly appear inside unrelated words —
       longer, more distinctive keywords (e.g. "coca", "snack") don't need this treatment.
12. Date/period filters — always check the [sample values] for the period column before
    writing a year filter.  Period columns often store values with a prefix or suffix:
      WRONG : WHERE fiscal_year = '2024'       ← if samples show 'FY2024'
      CORRECT: WHERE LOWER(fiscal_year) = 'fy2024'
      WRONG : WHERE calendar_year = '2023'     ← if samples show 'CY2023'
      CORRECT: WHERE LOWER(calendar_year) = 'cy2023'
    If the PRE-RESOLVED CATEGORY MAPPINGS section provides a fiscal_year sql_fragment,
    use it verbatim — do not rewrite the year value.
    For true date columns use the date extraction functions shown above.
13. COUNT vs SUM — choose the correct aggregate:
    a. Use COUNT(*) or COUNT(column) when the question asks for:
         headcount, number of people, how many employees/records/rows,
         total count, number of [entities].
       COUNT(*) counts ROWS — one row = one person/record.
    b. Use SUM(column) ONLY when the question asks for:
         total revenue, total amount, total value, sum of [a numeric measure].
       SUM adds up the VALUES stored in a numeric column — use it only when
       each row stores a quantity that should be added together.
    c. NEVER use SUM() to answer a headcount or "how many" question.
       NEVER use COUNT() to answer a "total revenue" or "total amount" question.
    d. If the schema has a dedicated numeric "Headcount" column (e.g. col_Headcount,
       Headcount, FTE) where each row stores a count value (not just 1),
       use SUM(Headcount) — not COUNT(*).  Otherwise use COUNT(*).
    e. When asked for breakdown by category (e.g. onshore vs offshore headcount),
       GROUP BY the category column and apply the correct aggregate per group.
14. Percentage calculations — when the question asks for %, share, proportion,
    or percentage of a total:
    a. Always compute the percentage IN SQL — do not leave it to the reader.
    b. Use the PERCENTAGE CALC syntax shown in the DATABASE-SPECIFIC SYNTAX section above.
    c. Always include both the raw value (count or sum) AND the percentage column
       in the SELECT list so the user sees both.
    d. Label the percentage column clearly, e.g. AS Headcount_Pct or AS Revenue_Pct.
    e. Round percentages to 2 decimal places with ROUND(..., 2).
    f. If asked "what percentage is X of Y" without a GROUP BY, use:
         SELECT
           SUM(CASE WHEN condition THEN 1 ELSE 0 END) AS numerator,
           COUNT(*) AS denominator,
           ROUND(SUM(CASE WHEN condition THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS Pct
         FROM table_name
15. Multi-KG federation — when ACTIVE KG IDS contains multiple entries:
    a. Each query in your JSON array MUST include a "kg_id" field set to one of
       the active KG ids listed in ACTIVE KG IDS.
    b. Use the bridge keys listed under CROSS-KG BRIDGES to plan which queries
       join data across KGs. The bridge key is the shared column that links two KGs.
    c. When a question requires data from multiple KGs, emit one query per KG
       and use the bridge column names so the synthesizer can merge them.
    d. If no bridges are listed, treat each KG independently.
16a. TRAILING N PERIODS — when the question asks about behaviour "over the last/trailing N
    periods", restrict to the N most recent periods available in the data, not a
    hard-coded period value:

      WITH ranked_periods AS (
        SELECT period_id,
               ROW_NUMBER() OVER (ORDER BY period_id DESC) AS rn
        FROM   dim_period                        ← or whatever the period dim is named
      ),
      trailing AS (
        SELECT period_id FROM ranked_periods WHERE rn <= {{N}}   ← N from the question
      )
      SELECT ...
      FROM   fact_table f
      JOIN   trailing t ON f.period_id = t.period_id
      ...

    If there is no dim_period table, use:
      WHERE period_id IN (
        SELECT DISTINCT period_id
        FROM   fact_table
        ORDER BY period_id DESC
        LIMIT {{N}}
      )

16b. SYSTEMATICALLY NEGATIVE / CONSISTENTLY ABOVE / ALWAYS BELOW — when the question
    asks which entities show a metric that is negative (or above/below a threshold)
    in ALL or MOST of the qualifying periods, use a HAVING clause that counts the
    qualifying periods and compares it to the total period count:

    Pattern A — ALL periods must satisfy the condition:
      HAVING SUM(CASE WHEN metric_col < 0 THEN 1 ELSE 0 END) = COUNT(DISTINCT period_id)

    Pattern B — MAJORITY of periods (>50%) must satisfy:
      HAVING SUM(CASE WHEN metric_col < 0 THEN 1 ELSE 0 END) * 1.0
           / NULLIF(COUNT(DISTINCT period_id), 0) > 0.5

    "Systematically" means Pattern A (every period).  "Consistently" or "tend to"
    means Pattern B.  Always include both:
      - negative_periods  = SUM(CASE WHEN metric_col < 0 THEN 1 ELSE 0 END)
      - total_periods     = COUNT(DISTINCT period_id)
      - avg_metric        = AVG(metric_col)
      - direction         = CASE WHEN AVG(metric_col) < 0 THEN 'negative' ELSE 'positive' END
    in the SELECT so the synthesiser can explain the finding.

    Full example — customers with systematically negative mix_contribution over trailing 12:
      WITH trailing AS (
        SELECT period_id FROM (
          SELECT DISTINCT period_id FROM fact_rgm_metrics
          ORDER BY period_id DESC LIMIT 12
        )
      )
      SELECT
        f.customer_id,
        c.customer_name,
        COUNT(DISTINCT f.period_id)                                         AS total_periods,
        SUM(CASE WHEN f.mix_contribution < 0 THEN 1 ELSE 0 END)            AS negative_periods,
        ROUND(AVG(f.mix_contribution), 4)                                   AS avg_mix_contribution,
        ROUND(SUM(f.mix_contribution), 4)                                   AS total_mix_contribution
      FROM fact_rgm_metrics f
      JOIN dim_customer     c ON f.customer_id = c.customer_id
      JOIN trailing         t ON f.period_id   = t.period_id
      GROUP BY f.customer_id, c.customer_name
      HAVING SUM(CASE WHEN f.mix_contribution < 0 THEN 1 ELSE 0 END) = COUNT(DISTINCT f.period_id)
      ORDER BY avg_mix_contribution ASC

18. Pre-defined KPI formulas — when DEFINED KPIs section is present:
    a. If the user's question references a KPI by name (e.g. "RSV Growth", "Market Share"),
       check the DEFINED KPIs section for a matching KPI with a sql_expression.
    b. When a match is found AND sql_expression is non-empty, use that expression
       verbatim as the measure in your SELECT clause — do NOT rewrite it.
    c. For growth / period-over-period KPIs:
       - "Last Full Period" means the latest complete period available in the data.
         Use a subquery: WHERE period_col = (SELECT MAX(period_col) FROM table)
       - Growth formula: (current_val - prior_val) / NULLIF(prior_val, 0) * 100
       - Use LAG() window function when comparing consecutive periods in a single query.
    d. If the KPI has no sql_expression yet, use the nl_formula as a hint and write
       your best SQL equivalent — note in "description" that the formula was inferred.

19. COLUMN SELECTION FOR ENTITY-PRESENCE / FLAG FILTERS — UNIVERSAL, ALL DOMAINS,
    ALL DATABASES. This rule applies regardless of industry or dialect.
    a. When the question implies "does this row represent/involve entity X" (e.g.
       "with customers", "involving employees", "for HCPs", "by supplier"), and
       several columns in the schema relate to X by name, do NOT default to whichever
       column merely contains X as a substring. Check the null-rate / completeness
       annotation shown for each candidate column in the schema context (e.g.
       "(99.0% null)" or "(0.3% null)") and prefer the column that:
         - functions as an identifier/key for that entity (name ends in _key, _id, or
           is marked PK/FK in the schema), AND
         - has the LOWEST null rate among the candidates.
       A descriptive/classification/category column for the same entity (e.g. a
       "*_classification", "*_type", or "*_category" column) is frequently far
       sparser than the entity's own key column and describes a subset of rows, not
       presence of the entity — it is usually the WRONG signal for "does this row
       involve entity X".
    b. Before finalizing any WHERE clause that ANDs 2+ conditions together, check the
       null-rate annotation for every filtered column. If any filtered column is
       highly sparse (roughly >80% null), pause and verify it is genuinely the
       column the question is asking about — a heavily-null column ANDed with other
       filters can silently collapse a syntactically valid query to 0 rows, in any
       domain and any database.
    c. For yes/no, opt-in, eligibility, or participation-style questions (the
       question uses a word like "attended", "opted in", "enrolled", "eligible",
       "flagged", "active"), actively look in the schema context for a column with
       exactly 2 (or very few) distinct values — these boolean/flag-style columns
       are usually the intended filter target, even when their name does not
       literally contain the question's wording, and even when the annotation
       pipeline judged the column name "self-explanatory" and therefore did not
       give it a separate business-concept label. Do not overlook a plainly-named
       flag column just because it has no concept annotation — read the raw column
       name and its distinct-value count directly from the schema context.

20. NEVER FILTER A DIMENSION THE QUESTION DID NOT MENTION — UNIVERSAL, ALL
    DOMAINS, ALL DATABASES. This is distinct from rule 6/6a (unnecessary JOINs)
    and from the CATEGORICAL FILTERS rule (mapping a term the user DID use to
    a real stored value) — this rule covers a different mistake: inventing a
    WHERE filter on a column for a dimension the question never referenced at
    all, using guessed values instead of the schema's real ones.
    a. If the schema context shows a categorical column's cardinality (e.g.
       "unique values=7") but does NOT list the actual stored values for it
       (no [sample values] / top values shown), you do NOT know what those 7
       values are. Do NOT guess them from the table/column name, domain
       conventions, or general world knowledge (e.g. assuming a table whose
       name suggests a region covers a specific, commonly-known list of
       countries for that region).
    b. If the user's question does not name a specific value — or even
       mention that dimension at all — do NOT add a WHERE filter on it "to
       scope the query sensibly". Leave that column unfiltered so the query
       returns whatever values genuinely exist in the data. An unfiltered
       query that surfaces the true, complete set of values is correct; a
       filtered query built on guessed values is wrong even if the guessed
       values happen to look plausible.
    c. This applies to every kind of scoping dimension: geography/market,
       time period, product line, business unit, customer segment, or any
       other categorical column — not just the specific case of a region or
       market name.
    d. Only filter a dimension when either (i) the user's question explicitly
       names a value for it (use the CATEGORICAL FILTERS rule to map it), or
       (ii) a PRE-RESOLVED CATEGORY MAPPING / DEFINED KPI section above
       instructs you to. Never filter a dimension solely because you inferred
       from the table or column name what its "typical" values probably are.

21. "COUNT OF X" / "NUMBER OF X" / "HEADCOUNT" QUESTIONS — STORED VALUE vs
    COMPUTED AGGREGATE. UNIVERSAL, ALL DOMAINS, ALL DATABASES.
    a. Before filtering any numeric column against a threshold implied by a
       counting question ("more than N attendees", "fewer than N people",
       "at least N transactions", "headcount above N"), check that column's
       observed min/max range shown in the schema context (e.g. "min=0.0000,
       max=6.0000"). If the threshold N falls outside — or even close to the
       edge of — that column's real range, the column almost certainly does
       NOT represent the count the question is asking about, even if its name
       sounds like it should (e.g. a column literally named "no_of_people" or
       "num_attendees" can still be the wrong signal if its real range is 0-6
       while the question implies dozens).
    b. A "count of X" question is asking for the number of rows/occurrences of
       X for some entity (an event, an order, a customer, etc.) — that is
       usually NOT a value stored directly on a single row. It has to be
       COMPUTED: GROUP BY the entity's key on the table where X has one row
       per occurrence, then COUNT(*) or COUNT(DISTINCT <id>), and apply the
       question's threshold with HAVING, not WHERE.
       WRONG:  SELECT * FROM expense_lines WHERE attendee_count_field > 20
       RIGHT:  SELECT event_id, COUNT(*) AS attendee_count
               FROM attendance_detail_table
               GROUP BY event_id
               HAVING COUNT(*) > 20
    c. Only trust a stored numeric column as the answer to a "count of X"
       question when its observed range in the schema context is actually
       consistent with what the question implies (e.g. a "total_attendees"
       column on an event-summary table whose max is in the hundreds IS a
       legitimate stored aggregate — use it directly, do not recompute).
    d. If you cannot find any column or table capable of producing the
       requested count (no detail table with one row per occurrence, and no
       stored aggregate column whose range fits), say so in the description
       rather than filtering a column whose range makes the question
       unanswerable — a query that runs and returns 0 rows is worse than an
       honest "not available" because it looks like a real, verified answer.

22. STATUS/VALIDITY COMPANION COLUMNS — AVOID DUPLICATE AMBIGUOUS ROWS.
    UNIVERSAL, ALL DOMAINS, ALL DATABASES.
    a. Some tables store one row per (entity, sub-dimension) combination —
       e.g. one row per (market, category, business-unit) — where the
       sub-dimension is often NOT applicable, so most rows are placeholder
       rows with a zero/blank value, and only the applicable sub-dimension's
       row holds the real number. This is visible in the schema as a
       categorical "status" / "validity" column sitting alongside the
       numeric value column, whose sample values look like an
       applicability flag — e.g. "allowed" / "empty" (or "valid"/"invalid",
       "active"/"inactive", "approved"/"n/a", a "1"/"0" flag, etc. — the
       exact words vary by dataset, read the sample values shown).
    b. If you query the numeric column WITHOUT filtering on that status
       column (or without filtering the numeric column itself to exclude
       the placeholder value, typically 0), you will get MULTIPLE rows per
       entity with CONFLICTING values — some real, some placeholder — and
       nothing in the result distinguishes which is which. This reads as
       inconsistent/wrong data even though every individual row is real.
    c. Before finalising a query that returns "the value of X for entity Y"
       from a table like this, check whether a status/validity column exists
       for that value. If it does, filter on it (e.g. status = 'allowed') or
       filter the numeric column itself (e.g. amount > 0) — same principle as
       the CATEGORICAL FILTERS rule, applied specifically to a
       status-vs-value column pair. Also SELECT the status column in the
       output so the answer's provenance is visible, not just the number.
    d. Do NOT assume this applies universally to every table — many tables
       have exactly one row per entity and no such ambiguity. Only apply
       this rule when the schema context actually shows a categorical column
       whose sample values look like an applicability/validity indicator
       sitting next to the numeric column you are about to query.

23. GRAIN-CORRECT TABLE SELECTION FOR OCCURRENCE-COUNTING QUESTIONS.
    UNIVERSAL, ALL DOMAINS, ALL DATABASES. This is a TABLE-CHOICE rule, not a
    column-choice rule — it applies BEFORE you decide which column to filter.
    a. When the question asks to count/total OCCURRENCES of an entity
       (attendees at an event, transactions on an account, visits to a site,
       line items on an order — anything phrased as "how many X", "more/fewer
       than N X", "total X"), the ONLY tables that can correctly answer it are
       ones with genuinely ONE ROW PER OCCURRENCE of that entity. On such a
       table, COUNT(*) or COUNT(DISTINCT <id>) IS the count.
    b. A wide "combined"/"primary"/"master" table that also contains a
       plausibly-named column (e.g. something that sounds like "headcount" or
       "attendee count") is NOT automatically the right table — it is often
       built at a DIFFERENT, narrower grain (e.g. one row per expense line, per
       order line, or per transaction — not one row per occurrence of the
       entity the question is asking about). A column merely CONTAINING a
       matching word is not evidence it holds the full count; matching the
       question's WORDING is not the same as matching its GRAIN.
    c. Prefer a table whose name and columns describe the entity's own detail/
       transaction/event record (e.g. an "attendance", "visit", "transaction",
       "line_item" style table) for counting occurrences of that entity, even
       when a wider combined table also exists and even when the combined
       table's column name is a closer lexical match to the question. Do NOT
       let rule 7's "prefer the comprehensive table" guidance override this —
       rule 7(d) explicitly does not apply here.
    d. A concrete tell that a candidate column is at the WRONG grain: its
       observed min/max range in the schema context is implausibly small for
       the real-world entity being counted (e.g. a "number of people" column
       whose max is 6 cannot represent total attendance at an event that
       plausibly draws dozens of people — see rule 21 and the numeric-range
       guidance elsewhere in this prompt). When you see this mismatch, do not
       just look for a different column on the SAME table — look for a
       DIFFERENT table in the schema whose grain actually matches one row per
       occurrence of the entity in question.
    e. If the wide combined table and the detail/occurrence table share the
       filter columns the question needs (e.g. both have a status/participation
       flag), that is not a reason to prefer the combined table — it just means
       the correctly-grained table can answer the ENTIRE question on its own,
       with no join required. Check this before assuming you need to combine
       both tables or join anything.

24. HISTORICAL / VERSIONED DATA — USE THE LATEST RECORD PER ENTITY.
    UNIVERSAL, ALL DOMAINS, ALL DATABASES.
    a. Some tables genuinely track how a SINGLE entity's record changed over
       TIME — e.g. a table literally named with "history", "log", "audit",
       "snapshot", "versions", or "_hist" in the schema, or one where the FD/
       join-key evidence shows the same entity key repeats across rows with a
       date/timestamp column distinguishing them (an SCD-style table). For
       these tables, when the question asks about an entity's CURRENT /
       latest / present state (or doesn't specify a point in time at all),
       return only the MOST RECENT row per entity — via
       ROW_NUMBER()/QUALIFY (Snowflake/BigQuery) or a correlated
       MAX(date_col) subquery (other dialects) partitioned/grouped by the
       entity key — not every historical row, and not an arbitrary row.
    b. Only return multiple historical rows per entity when the question
       explicitly asks for a trend, history, timeline, or change over time
       (e.g. "how has X changed", "show the history of Y").
    c. ★ CRITICAL BOUNDARY — this rule does NOT apply to, and must never be
       used to resolve, the pattern in rule 22 (status/validity companion
       columns). Rule 22's tables are NOT time-series data — their duplicate
       rows represent different SUB-DIMENSIONS at essentially the same point
       in time (e.g. one row per business sector for the same market), most
       of which are placeholder/not-applicable, not successive versions of
       the same fact. "Most recent" is NOT a reliable signal for which
       sub-dimension row is real in that pattern — recency and validity are
       independent there. Only apply THIS rule (24) when the table is
       genuinely about the same fact changing over time for one entity, and
       apply rule 22 (status/validity filtering) when it's about disambiguating
       which sub-dimension row is real — do not substitute one fix for the
       other. When unsure which pattern you're looking at, prefer rule 22's
       explicit status/validity filtering signal over guessing based on dates.
"""

_USER_PROMPT = """\
SCHEMA CONTEXT:
{schema_context}

TARGET DATABASE TYPE: {db_type}
{schema_line}
{history_section}{multi_kg_section}{lexicon_section}{glossary_section}{generated_glossary_section}{kpi_section}{resolution_section}{verified_section}NATURAL LANGUAGE QUESTION:
{natural_query}

CRITICAL REMINDERS:
- Use ONLY column names that appear in the DETAILED SCHEMA above. Do NOT invent column names.
- Use ONLY table names from the AVAILABLE TABLES list above.
- ⛔ NO MATCH TERMS: If PRE-RESOLVED CATEGORY MAPPINGS shows a "⛔ NO MATCH" entry for a
  user term, do NOT add any WHERE clause filter for that term. The term does not exist in
  the data. Instead retrieve all rows (no filter) so the user can see what values are available.
  NEVER fabricate a filter like LOWER(col) = 'term' or LOWER(col) LIKE '%term%' for a NO MATCH term.
- CROSS-TABLE: If no POSSIBLE JOIN KEYS exist between two tables, query them SEPARATELY.
  Do NOT use subqueries, IN (...), EXISTS, correlated queries, or any trick to combine
  data from two tables that have no valid join key. One query = one table (or validly joined tables).
- CATEGORICAL FILTERS — PRE-RESOLVED MAPPINGS (see PRE-RESOLVED section above if present):
  If PRE-RESOLVED CATEGORY MAPPINGS are shown above, you MUST copy those sql_fragment
  values verbatim into your WHERE clauses. Do NOT substitute your own terminology.
  If no pre-resolved section is shown, check [categorical] sample values in the schema
  and use LIKE or IN when the user's term does not appear verbatim in the samples.
    user: "savoury snacks"   → data: "Snacks & Foods"    → use LOWER(col) = 'snacks & foods'
    user: "beverages"        → data: "Drinks"             → use LOWER(col) LIKE '%drink%'
    user: "EMEA"             → data: "Europe","Middle East","Africa" → use IN (...)
- COUNT vs SUM: use COUNT(*) for headcount/how-many questions; use SUM(col) only for
  monetary/quantity totals. NEVER use SUM() to count people or rows.
- PERCENTAGES: if the question asks for %, share, or proportion — compute it in SQL
  using the PERCENTAGE CALC syntax from the DATABASE-SPECIFIC SYNTAX section in your
  instructions. Always include both the raw value and the percentage column.
- KPI FORMULAS: if a DEFINED KPIs section is shown above and the question mentions a
  KPI by name, use the pre-defined sql_expression verbatim — do not rewrite it.
- VERIFIED EXAMPLES (see VERIFIED PAST CORRECTIONS section above if present): these are
  SQL queries a human has already confirmed correct for a similar question on THIS EXACT
  data source. They encode source-specific facts (which column is the real signal, which
  one is a red herring) that cannot be derived from column names alone. When the current
  question closely matches one, follow its column choices and filter pattern — do not
  independently re-derive a different answer that contradicts a verified example.

Return the JSON array of SQL queries now.
"""


# ── Dimension column name signals (indicate sub-key grain when present) ───────
# If a table has these column patterns ALONGSIDE a period key, it records data
# at sub-period grain (e.g. one row per brand-pack × period × channel).
_DIMENSION_COL_SIGNALS = (
    "channel", "region", "geography", "territory", "district",
    "customer", "retailer", "outlet", "store", "account",
    "sub_channel", "trade_channel", "route_to_market",
    "segment", "cluster", "zone",
)

# Period key column name signals
_PERIOD_COL_SIGNALS = (
    "period", "month", "quarter", "week", "date", "year",
    "fiscal", "time_period",
)

# Entity key column name signals (the primary join key, not a dimension)
_ENTITY_KEY_SIGNALS = (
    "_id", "_key", "_code", "brand", "product", "sku", "item",
    "employee", "supplier", "store_id", "cost_centre",
)


def _annotate_schema_grain(schema_context: str) -> str:
    """
    Parse the schema context produced by understand_node and inject ⚠️ grain
    annotations for any table that has sub-join-key granularity.

    Detection heuristic:
      A table is flagged as "multi-row-per-key" when it has ALL of:
        1. At least one period-key column (_PERIOD_COL_SIGNALS)
        2. At least one entity-key column (_ENTITY_KEY_SIGNALS or join-key cols)
        3. At least one extra dimension column (_DIMENSION_COL_SIGNALS) beyond
           the entity + period keys — indicating rows repeat per entity+period

    For flagged tables, a warning block is inserted immediately after the
    "Table: ..." line so the LLM sees it before reading the column list.

    This is deterministic Python — no LLM call, no false negatives from
    prompt rules the LLM can ignore.
    """
    lines = schema_context.splitlines()
    result: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        result.append(line)

        # Detect a "Table: ..." header line
        if not re.match(r'\s*Table:\s+\S', line, re.IGNORECASE):
            i += 1
            continue

        # Collect the columns for this table (lines until the next blank or Table:)
        j = i + 1
        col_names: List[str] = []
        while j < len(lines):
            peek = lines[j]
            if re.match(r'\s*Table:\s+', peek, re.IGNORECASE):
                break
            if re.match(r'\s*={3,}', peek):
                break
            # Column lines look like "    col_name: type ..."
            col_m = re.match(r'\s{2,}(\w+)\s*:', peek)
            if col_m:
                col_names.append(col_m.group(1).lower())
            j += 1

        if not col_names:
            i += 1
            continue

        # Check for period key, entity key, and dimension signals
        has_period  = any(any(sig in c for sig in _PERIOD_COL_SIGNALS) for c in col_names)
        has_entity  = any(any(sig in c for sig in _ENTITY_KEY_SIGNALS) for c in col_names)
        dim_cols    = [c for c in col_names
                       if any(sig in c for sig in _DIMENSION_COL_SIGNALS)]

        if has_period and has_entity and dim_cols:
            # Extract table name for a targeted warning
            tbl_match = re.search(r'Table:\s+(\S+)', line, re.IGNORECASE)
            tbl_name  = tbl_match.group(1) if tbl_match else "this table"
            dim_sample = ", ".join(dim_cols[:3])
            result.append(
                f"  ⚠️  GRAIN WARNING: {tbl_name} has MULTIPLE ROWS per (entity_key, period_key)."
            )
            result.append(
                f"     Extra dimension columns detected: {dim_sample}."
            )
            result.append(
                f"     MANDATORY: before joining this table, wrap it in a CTE that"
            )
            result.append(
                f"     aggregates to one row per (entity_key, period_key):"
            )
            result.append(
                f"       WITH tbl_agg AS ("
            )
            result.append(
                f"         SELECT entity_key, period_key,"
            )
            result.append(
                f"                AVG(metric_col_a) AS avg_metric_a,  -- replace with real cols"
            )
            result.append(
                f"         FROM   {tbl_name}"
            )
            result.append(
                f"         GROUP BY entity_key, period_key"
            )
            result.append(
                f"       )"
            )
            result.append(
                f"     NEVER join {tbl_name} directly on (entity_key, period_key)"
            )
            result.append(
                f"     without this pre-aggregation step."
            )
            logger.debug(
                "plan_node: grain warning injected for table %s (dim cols: %s)",
                tbl_name, dim_sample,
            )

        i += 1

    return "\n".join(result)


# ── YoY / change column name patterns ────────────────────────────────────────
_YOY_PATTERNS = (
    "vs_yago", "vs_py", "_yoy", "year_over_year", "prior_year",
    "vs_last_year", "_growth", "_change", "_delta", "_improvement",
    "_vs_prior", "_variance",
)

# Keywords in the natural question that signal change/driver intent
_CHANGE_INTENT_WORDS = (
    "deliberate", "intentional", "windfall", "why", "driver", "cause",
    "grew", "increased", "improved", "declined", "fell", "change",
    "movement", "shift", "execution", "vs prior", "year over year",
    "classify", "classification", "performance", "driven",
)


_SPARSE_NULL_RE = re.compile(r'^\s{2,4}"?(\w+)"?\s*:.*?\((\d+(?:\.\d+)?)%\s*null\)', re.MULTILINE)

# A column this sparse, ANDed with any other filter condition, is high-risk on
# its own: requiring a near-always-empty column to be non-null (or match a
# value) alongside anything else collapses the result set almost regardless of
# the other condition's selectivity. This is the anchor threshold for check 5.
_SPARSE_NULL_THRESHOLD = 85.0


def _extract_column_null_rates(schema_context: str) -> Dict[str, float]:
    """
    Map lowercase column name -> null percentage, parsed from schema_context
    property lines such as:
        "  id_hcp_classification: string  -- TEXT descriptive text column (99.0% null)"
    First occurrence wins if the same column name appears under multiple tables.
    """
    rates: Dict[str, float] = {}
    for m in _SPARSE_NULL_RE.finditer(schema_context or ""):
        col = m.group(1).lower()
        if col not in rates:
            rates[col] = float(m.group(2))
    return rates


# A column's property block spans several lines — the column declaration line,
# then a "Semantic domain: ..." line and a "Statistics: ... min=X, max=Y, ..."
# line of their own. In the ACTUAL text understand_node produces, column names
# AND these landmark labels are double-quoted, e.g.:
#     "no_of_people": string  -- TEXT descriptive text column (36.1% null)
#     "Semantic domain": descriptive text
#     "Statistics": unique values=4, null count=2253, min=0.0000, max=6.0000
# Quotes are optional in the regex below (older/unquoted formats also occur)
# so this matches either. "Statistics" itself would otherwise be misread as a
# column declaration (it matches the same `"?word"?\s*:` shape) — excluded
# explicitly rather than relied upon to fail the pattern.
_COLUMN_DECL_RE = re.compile(r'^\s{2,4}"?(\w+)"?\s*:', re.MULTILINE)
_STATS_LINE_RE = re.compile(r'^\s{2,4}"?Statistics"?\s*:\s*(.*)$', re.MULTILINE | re.IGNORECASE)
_MINMAX_RE = re.compile(r'\bmin=(-?\d+(?:\.\d+)?)\s*,?\s*max=(-?\d+(?:\.\d+)?)')
_NON_COLUMN_LABELS = {"statistics", "semantic", "business"}  # pseudo-entries, never real columns


def _extract_column_numeric_ranges(schema_context: str) -> Dict[str, tuple]:
    """
    Map lowercase column name -> (min, max), parsed from schema_context.
    A column's declaration line and its "Statistics: ... min=X, max=Y" line are
    NOT on the same line, and are not even adjacent — a "Semantic domain" line
    (and sometimes a "Business concept" line) sits between them. This finds
    every column declaration, then associates it with the nearest "Statistics"
    line that appears strictly BEFORE the next column declaration — robust to
    however many landmark lines sit in between. This is the OBSERVED range in
    the actual data — used by preflight Check 6 to catch a WHERE clause
    comparing a column against a threshold the data can never satisfy (e.g.
    "> 20" when the column's real max is 6). First occurrence wins if the same
    column name appears under multiple tables.
    """
    ranges: Dict[str, tuple] = {}
    text = schema_context or ""
    col_matches = [
        m for m in _COLUMN_DECL_RE.finditer(text)
        if m.group(1).lower() not in _NON_COLUMN_LABELS
    ]
    stats_matches = list(_STATS_LINE_RE.finditer(text))
    if not col_matches or not stats_matches:
        return ranges

    for i, cm in enumerate(col_matches):
        col = cm.group(1).lower()
        if col in ranges:
            continue
        span_end = col_matches[i + 1].start() if i + 1 < len(col_matches) else len(text)
        for sm in stats_matches:
            if cm.end() <= sm.start() < span_end:
                mm = _MINMAX_RE.search(sm.group(1))
                if mm:
                    ranges[col] = (float(mm.group(1)), float(mm.group(2)))
                break
    return ranges


def _preflight_check_plan(
    plan_items: List[Dict],
    known_columns: set,
    natural_query: str,
    schema_context: str = "",
) -> List[str]:
    """
    Pre-execution completeness checks on the raw LLM plan (list of dicts).

    Returns a list of gap descriptions (strings).  Empty list = plan passes.
    These gaps drive a targeted correction retry — unlike _validate_plan_items
    which fires when queries are outright invalid, pre-flight fires when the
    plan is structurally valid but analytically incomplete.

    Checks
    ------
    1. JOIN without WITH (pre-aggregation CTEs): any query that joins tables
       but has no CTE pre-aggregating to (entity_key, period_key) level.
    2. CTE with no aggregate functions: a WITH block whose SELECT list contains
       no SUM / AVG / COUNT / MIN / MAX — indicates a key-only CTE that carries
       no metrics into the outer query.
    3. Missing q_summary: a JOIN query exists but no query_id ends in "summary".
    4. YoY column in schema + change-intent question but no YoY column in any
       query SQL.
    5. Sparse-column co-filter risk: WHERE clause ANDs two or more columns that
       are each individually >50% null (per schema_context) — a near-guaranteed
       0-row result when the columns belong to different record subtypes in a
       wide/denormalised table (e.g. an HCP classification code and a meal
       headcount field that are populated for disjoint sets of rows).
    """
    gaps: List[str] = []
    sql_items = [it for it in plan_items if it.get("sql")]
    if not sql_items:
        return gaps

    # ── helpers ──────────────────────────────────────────────────────────────

    def _has_join(sql: str) -> bool:
        return bool(re.search(r'\b(?:INNER\s+|LEFT\s+|RIGHT\s+|FULL\s+)?JOIN\b', sql, re.IGNORECASE))

    def _has_cte(sql: str) -> bool:
        return bool(re.search(r'^\s*WITH\b', sql, re.IGNORECASE))

    def _is_dimension_join_only(sql: str) -> bool:
        """
        Heuristic: if every JOIN clause joins ON a single column equality that
        looks like a surrogate key (ends in _id, _key, _code, _num) and the
        joined table name contains 'dim' or 'dimension', treat it as a
        pure dimension lookup — pre-aggregation not required.
        """
        join_tables = re.findall(r'\bJOIN\s+\S+\s+AS\s+(\w+)', sql, re.IGNORECASE)
        if not join_tables:
            join_tables = re.findall(r'\bJOIN\s+(\w+)\b', sql, re.IGNORECASE)
        for tbl in join_tables:
            # resolve alias back to table label
            alias_match = re.search(
                r'\b(\S+)\s+(?:AS\s+)?' + re.escape(tbl) + r'\b', sql, re.IGNORECASE
            )
            tbl_name = alias_match.group(1).lower() if alias_match else tbl.lower()
            if 'dim' not in tbl_name and 'lookup' not in tbl_name and 'master' not in tbl_name:
                return False
        return bool(join_tables)

    def _cte_bodies(sql: str) -> List[str]:
        """
        Extract the full body of each CTE using paren-depth tracking.
        Returns list of body strings (content between the outer parens).
        """
        results = []
        for m in re.finditer(r'\b\w+\s+AS\s*\(', sql, re.IGNORECASE):
            start = m.end()
            depth = 1
            pos = start
            while pos < len(sql) and depth > 0:
                if sql[pos] == '(':
                    depth += 1
                elif sql[pos] == ')':
                    depth -= 1
                pos += 1
            results.append(sql[start: pos - 1])
        return results

    def _has_aggregates(col_list_text: str) -> bool:
        return bool(re.search(
            r'\b(SUM|AVG|COUNT|MIN|MAX|PERCENTILE|STRING_AGG|LISTAGG)\s*\(',
            col_list_text, re.IGNORECASE,
        ))

    # ── Check 1 + 2: JOIN queries ─────────────────────────────────────────────
    for item in sql_items:
        sql = item["sql"]
        qid = item.get("query_id", "?")

        if not _has_join(sql):
            continue
        if _is_dimension_join_only(sql):
            continue  # pure dim lookup — pre-aggregation not needed

        # Check 1: JOIN without any CTE
        if not _has_cte(sql):
            gaps.append(
                f"Query {qid!r} has a JOIN between metric/fact tables but no pre-aggregation "
                f"CTEs (WITH clause). Both tables must be wrapped in CTEs that GROUP BY "
                f"(entity_key, period_key) before joining to prevent row multiplication. "
                f"See rule 10c-i."
            )
        else:
            # Check 2: CTEs that GROUP BY but contain no aggregate functions
            # (pass-through / join CTEs with no GROUP BY are intentional and OK)
            bad_cte_indices = []
            for i, body in enumerate(_cte_bodies(sql), 1):
                has_group_by = bool(re.search(r'\bGROUP\s+BY\b', body, re.IGNORECASE))
                if has_group_by and not _has_aggregates(body):
                    bad_cte_indices.append(i)
            if bad_cte_indices:
                gaps.append(
                    f"Query {qid!r}: CTE(s) {bad_cte_indices} use GROUP BY but contain "
                    f"no aggregate functions (SUM/AVG/COUNT) — they select only key "
                    f"columns and carry no metrics into the outer query. Each "
                    f"pre-aggregation CTE must include metric columns. See rule 10c-i."
                )

    # ── Check 3: JOIN exists but no q_summary ─────────────────────────────────
    has_fact_join = any(
        _has_join(it["sql"]) and not _is_dimension_join_only(it["sql"])
        for it in sql_items
    )
    has_summary = any(
        (it.get("query_id") or "").lower().endswith("summary")
        for it in sql_items
    )
    if has_fact_join and not has_summary:
        gaps.append(
            "The plan has a JOIN query but no 'q_summary'. Every fact-table JOIN "
            "MUST be followed by a self-contained q_summary query that re-runs the "
            "same CTEs + JOIN and then GROUP BY entity to produce one row per entity "
            "with all metrics aggregated. q_summary cannot reference another query_id "
            "— it must be fully self-contained. See rule 10c-ii."
        )

    # ── Check 4: YoY column in schema + change intent but not in any query ────
    yoy_cols_in_schema = [
        col for col in known_columns
        if any(pat in col for pat in _YOY_PATTERNS)
    ]
    if yoy_cols_in_schema:
        q_lower = natural_query.lower()
        has_change_intent = any(w in q_lower for w in _CHANGE_INTENT_WORDS)
        if has_change_intent:
            all_sql_lower = " ".join(it["sql"] for it in sql_items).lower()
            referenced = any(col in all_sql_lower for col in yoy_cols_in_schema)
            if not referenced:
                sample = yoy_cols_in_schema[:3]
                gaps.append(
                    f"The question has change/classification intent but none of the "
                    f"YoY/change columns in the schema ({sample}…) appear in any query. "
                    f"At least one must be included — in a standalone query or in "
                    f"q_summary — to answer the 'deliberate vs windfall' / direction "
                    f"part of the question. See rules 10d and 10e."
                )

    # ── Check 5: sparse-column co-filter risk ─────────────────────────────────
    null_rates = _extract_column_null_rates(schema_context)
    if null_rates:
        for item in sql_items:
            sql = item["sql"]
            qid = item.get("query_id", "?")

            where_match = re.search(
                r'\bWHERE\b(.*?)(?=\bGROUP\s+BY\b|\bORDER\s+BY\b|\bHAVING\b|\bQUALIFY\b|$)',
                sql, re.IGNORECASE | re.DOTALL,
            )
            if not where_match:
                continue
            where_clause = where_match.group(1)

            # Need at least 2 AND-connected conditions for a co-filter risk to
            # exist at all — a single filter on a sparse column is often exactly
            # what the user asked for (e.g. "show me the flagged records").
            and_condition_count = len(re.findall(r'\bAND\b', where_clause, re.IGNORECASE)) + 1
            if and_condition_count < 2:
                continue

            # Columns referenced in a null-check or equality/inequality comparison
            # inside the WHERE clause (quoted or bare identifier forms).
            cols_in_where = set(re.findall(
                r'"(\w+)"\s*(?:IS\s+(?:NOT\s+)?NULL|!?=)', where_clause, re.IGNORECASE,
            ))
            cols_in_where |= set(re.findall(
                r'\b(\w+)\s+IS\s+(?:NOT\s+)?NULL\b', where_clause, re.IGNORECASE,
            ))

            sparse_cols = sorted(
                {(c, null_rates[c.lower()]) for c in cols_in_where
                 if c.lower() in null_rates and null_rates[c.lower()] >= _SPARSE_NULL_THRESHOLD},
                key=lambda t: -t[1],
            )
            if sparse_cols:
                detail = ", ".join(f"{c!r} ({pct:.0f}% null)" for c, pct in sparse_cols)
                gaps.append(
                    f"Query {qid!r}: WHERE clause ANDs a near-always-empty column with "
                    f"{and_condition_count - 1} other condition(s) — {detail}. Requiring a "
                    f"column this sparse to be non-null (or match a value) at the same time "
                    f"as other filters is very likely to collapse the result to 0 rows, "
                    f"especially in a wide/denormalised table where sparse columns often "
                    f"belong to a different record subtype than the rest of the filter. "
                    f"Re-check whether a more complete identifier column (e.g. a '_key' "
                    f"column, usually populated far more consistently than a descriptive "
                    f"classification column for the same entity) should be used instead."
                )

    # ── Check 6: impossible numeric-threshold filter ──────────────────────────
    # Catches a WHERE clause comparing a column against a threshold the column's
    # OWN observed data range can never satisfy — e.g. "> 20" on a column whose
    # real max is 6. Such a query runs cleanly and returns 0 rows with no error,
    # which reads as "there is no matching data" when the real problem is that
    # the wrong column was used to answer a "count of X" / "number of Y" style
    # question (the true count usually has to be computed via GROUP BY + HAVING
    # COUNT(...) over a detail table, not read off a stored per-row value).
    numeric_ranges = _extract_column_numeric_ranges(schema_context)
    if numeric_ranges:
        for item in sql_items:
            msg = _detect_impossible_numeric_filter(item["sql"], numeric_ranges)
            if msg:
                gaps.append(f"Query {item.get('query_id', '?')!r}: {msg}")

    return gaps


_IMPOSSIBLE_CMP_RE = re.compile(
    r'(?:TRY_CAST|CAST|TRY_TO_NUMBER|TRY_TO_DOUBLE|TRY_TO_DECIMAL|SAFE_CAST)?\s*'
    r'\(?\s*[\w]*\.?"?(\w+)"?\s*(?:AS\s+\w+\))?\s*'
    r'(>=|<=|>|<|=)\s*'
    r'(-?\d+(?:\.\d+)?)',
    re.IGNORECASE,
)


def _detect_impossible_numeric_filter(sql: str, numeric_ranges: Dict[str, tuple]) -> Optional[str]:
    """
    Return a human-readable message if `sql` compares a column against a
    threshold the column's OWN observed data range (from numeric_ranges) can
    never satisfy — e.g. "> 20" when the real max is 6. None if no issue is
    found. Shared by preflight Check 6 (detection/logging, in
    _preflight_check_plan) and the corrective retry in plan_node (which uses
    the identical logic to decide whether a one-shot LLM rewrite is needed,
    and to verify the rewrite actually fixed it) — one detector, two callers,
    so they can never silently drift apart.
    """
    if not numeric_ranges:
        return None
    for m in _IMPOSSIBLE_CMP_RE.finditer(sql):
        col, op, thresh_str = m.group(1).lower(), m.group(2), m.group(3)
        if col not in numeric_ranges:
            continue
        thresh = float(thresh_str)
        lo, hi = numeric_ranges[col]
        if op == ">":
            impossible = thresh >= hi
        elif op == ">=":
            impossible = thresh > hi
        elif op == "<":
            impossible = thresh <= lo
        elif op == "<=":
            impossible = thresh < lo
        else:  # "="
            impossible = thresh < lo or thresh > hi
        if impossible:
            return (
                f"filters '{col}' {op} {thresh_str}, but the schema context shows "
                f"'{col}''s OBSERVED data range is min={lo} max={hi} — this comparison "
                f"can NEVER match a row, so the query will silently return 0 results "
                f"with no error. This usually means '{col}' is the WRONG column for "
                f"what the question is actually asking (e.g. a per-row stored value "
                f"being used where the question actually needs a COUNT/SUM computed "
                f"via GROUP BY + HAVING over a detail/transaction table — a "
                f"'headcount'/'number of X' question about an aggregate is rarely "
                f"answerable by filtering a single stored column directly, and if the "
                f"query already has a GROUP BY, the threshold likely belongs in HAVING "
                f"against the aggregate result, not in WHERE against the raw column). "
                f"Compute the count/sum with GROUP BY + HAVING {op} {thresh_str} instead."
            )
    return None


_COST_PER_M = {
    "claude-haiku-4-5": (0.80, 4.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-5":   (15.00, 75.00),
}


def _log_cost(node: str, model: str, usage) -> None:
    inp, out = usage.input_tokens, usage.output_tokens
    in_price, out_price = _COST_PER_M.get(model, (0.80, 4.00))
    cost = (inp * in_price + out * out_price) / 1_000_000
    logger.info(
        "COST %s [%s] in=%d out=%d  $%.5f",
        node, model, inp, out, cost,
    )


def _call_llm(
    system: str,
    user: str,
    model: str,
    temperature: float,
) -> str:
    """Call Anthropic Claude and return the raw text response."""
    from llm_client import get_client
    client = get_client()
    msg = client.messages.create(
        model=model,
        max_tokens=8192,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    _log_cost("plan_node", model, msg.usage)
    return msg.content[0].text if msg.content else ""


_JSON_RETRY_MAX_ATTEMPTS = 1  # one extra attempt after the initial call


def _call_plan_llm_and_parse(
    system: str,
    user: str,
    model: str,
    temperature: float,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Call the plan LLM and parse its JSON array response, retrying once with
    an augmented prompt if the response is not valid JSON.

    The plan JSON's "sql" field often has to carry a long, verbatim SQL
    expression (e.g. a KPI's pre-defined sql_expression, which the LLM is
    explicitly told to copy without rewriting — see the DEFINED KPIs section
    below). A long/complex expression with several quoted literals is a
    plausible spot for the model to slip on JSON escaping, producing a
    JSONDecodeError. Previously that was fatal with zero retry — an
    otherwise-recoverable one-off formatting mistake took down the whole
    response ("No query results were produced"). One retry, with the
    specific parse error fed back to the model, recovers most of these
    without materially changing latency (LLM formatting mistakes are rare;
    this path is a no-op in the overwhelming majority of requests).
    """
    raw = _call_llm(system, user, model, temperature)
    try:
        return raw, _extract_json(raw)
    except json.JSONDecodeError as exc:
        # `except ... as exc` implicitly deletes `exc` when the block exits, so
        # it must be captured under a different name to survive past here.
        last_exc = exc
        logger.warning(
            "plan_node: LLM returned unparseable JSON (%s) — retrying once with the "
            "parse error fed back",
            exc,
        )

    for attempt in range(_JSON_RETRY_MAX_ATTEMPTS):
        retry_user = (
            f"{user}\n\n---\n"
            "Your previous response was not valid JSON and could not be parsed "
            "(this exact error occurred while parsing it):\n"
            f"{last_exc}\n\n"
            "Return ONLY a valid JSON array in the exact same shape as before. "
            "Pay special attention to correctly escaping any quotes, newlines, or "
            "special characters inside \"sql\" string values — the entire SQL text "
            "must be a single valid JSON string."
        )
        raw = _call_llm(system, retry_user, model, temperature)
        try:
            return raw, _extract_json(raw)
        except json.JSONDecodeError as exc2:
            last_exc = exc2
            logger.warning(
                "plan_node: retry attempt %d/%d also returned unparseable JSON — %s",
                attempt + 1, _JSON_RETRY_MAX_ATTEMPTS, exc2,
            )

    # All attempts exhausted — re-raise the last parse error so the caller's
    # existing except-block handles it exactly as before this fix.
    raise last_exc


_FIX_THRESHOLD_SYSTEM = """\
You are a SQL expert fixing a query that will silently return 0 rows due to \
an impossible filter condition — it runs without error, but the threshold \
can never be satisfied by the column's real data range.

There are TWO possible root causes — check both, in this order:

1. WRONG TABLE (check this first). The column's tiny/implausible range often \
means the table you queried is at the WRONG GRAIN for what's being counted \
— e.g. it has one row per expense line or transaction, not one row per \
occurrence of the entity the question asks about (attendee, visit, order).
   Look at the schema context below for a DIFFERENT table whose name and \
   columns describe the entity's own detail/transaction/attendance record \
   (one row per occurrence). If one exists and has the other columns this \
   query needs (the same filters, keys, etc.), REWRITE THE QUERY AGAINST \
   THAT TABLE INSTEAD — group by the entity key and use COUNT(*) or \
   COUNT(DISTINCT <id>) as the count, then apply the threshold via HAVING.
   Do not assume you need to JOIN both tables — the correctly-grained table
   alone usually already has everything the query needs.

2. WRONG CLAUSE (if the table is genuinely already correct). The question \
needs a COUNT/SUM across multiple rows per entity (e.g. "attendees more \
than N", "orders more than N"), but the threshold was applied with WHERE \
against a raw per-row column instead of HAVING against the aggregated \
result.
   WRONG: SELECT entity_id, SUM(TRY_CAST(col AS NUMERIC)) AS total
          FROM t WHERE TRY_CAST(col AS NUMERIC) > 20 GROUP BY entity_id
   RIGHT: SELECT entity_id, SUM(TRY_CAST(col AS NUMERIC)) AS total
          FROM t GROUP BY entity_id HAVING SUM(TRY_CAST(col AS NUMERIC)) > 20
   If there is no GROUP BY at all, add one grouped by the natural entity key
   for the question, aggregate with SUM or COUNT, and apply the threshold via
   HAVING — never filter the raw column in WHERE for a per-entity total.

Preserve the original question's intent exactly. Only change table/column
choices when case 1 genuinely applies — do not swap tables speculatively if
the original table can be fixed by case 2 alone. Return ONLY the corrected
SQL — no explanation, no markdown fences, no commentary.
"""


def _fix_impossible_threshold_sql(
    sql: str,
    gap_message: str,
    natural_query: str,
    schema_context: str,
    model: str,
    temperature: float,
) -> str:
    """
    One-shot LLM correction for a query flagged by
    _detect_impossible_numeric_filter (Check 6) — a semantic mistake that
    never surfaces as a runtime SQL error (the query runs cleanly and just
    returns 0 rows), so execute_node's error-driven self-heal never gets a
    chance to catch it. This has to happen at plan time, before execution.

    Includes schema_context so the LLM can see whether a different,
    correctly-grained table exists to switch to (see rule 23) rather than
    only being able to patch WHERE/HAVING placement on the original table —
    a small max value is often a sign the table itself is wrong, not just the
    clause. schema_context is trimmed defensively; this call already runs
    only when Check 6 fires, so it's rare enough that the extra token cost is
    fine, but a runaway-huge schema must never crash this correction.

    Returns the corrected SQL, or the original unchanged if the LLM call
    fails, returns nothing usable, or (deliberately) doesn't change it.
    Never raises — a failure here must never block the rest of planning.
    """
    _MAX_SCHEMA_CHARS = 120_000
    trimmed_schema = schema_context[:_MAX_SCHEMA_CHARS]
    user = (
        f"Original question: {natural_query}\n\n"
        f"Problem detected:\n{gap_message}\n\n"
        f"Original SQL:\n{sql}\n\n"
        f"SCHEMA CONTEXT (for checking whether a better-grained table exists):\n"
        f"{trimmed_schema}\n\n"
        "Return the corrected SQL only."
    )
    try:
        from llm_client import get_client
        client = get_client()
        msg = client.messages.create(
            model=model,
            max_tokens=2048,
            temperature=temperature,
            system=_FIX_THRESHOLD_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        _log_cost("plan_node.threshold_fix", model, msg.usage)
        fixed = (msg.content[0].text if msg.content else "").strip()
        fixed = re.sub(r"^```[a-z]*\s*", "", fixed, flags=re.IGNORECASE)
        fixed = re.sub(r"\s*```$", "", fixed).strip()
        return fixed if fixed else sql
    except Exception as exc:
        logger.warning("plan_node: impossible-threshold corrective LLM call failed — %s", exc)
        return sql


def _extract_json(text: str) -> List[Dict[str, Any]]:
    """
    Extract a JSON array from the LLM response.

    Uses bracket-counting to find the exact closing bracket for the first
    top-level '[', so trailing text (notes, explanations, etc.) containing
    ']' characters does not cause a JSONDecodeError.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    cleaned = cleaned.rstrip("`").strip()

    start = cleaned.find("[")
    if start == -1:
        return []

    depth = 0
    in_string = False
    escape = False
    end = -1
    for i, ch in enumerate(cleaned[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        return []
    return json.loads(cleaned[start:end + 1])


def _extract_known_columns(schema_context: str) -> set:
    """
    Parse the schema context text produced by understand_node and return a
    lower-cased set of every column name listed under 'Columns:' sections.
    Used to reject SQL that references hallucinated column names.
    """
    known: set = set()
    in_columns = False
    for line in schema_context.splitlines():
        stripped = line.strip()
        if stripped.lower() == "columns:":
            in_columns = True
            continue
        if in_columns:
            # Column lines look like: "col_name: integer  [sample values: ...]"
            # or just "col_name: integer"
            # Stop at blank lines, table headers, FK lines, or section dividers
            if not stripped or stripped.startswith("Table:") or stripped.startswith("FK:") \
                    or stripped.startswith("--") or stripped.startswith("=") \
                    or stripped.startswith("-"):
                in_columns = False
                continue
            col_name = stripped.split(":")[0].split("[")[0].strip().strip('"')
            if col_name:
                known.add(col_name.lower())
    return known


def _dotted_literal_parts(known_cols: set) -> set:
    """
    Return every individual prefix/tail token of literal dotted column names
    in known_cols (e.g. "evt.event_key" -> {"evt", "event_key"}). These are
    not real standalone columns, but the LLM sometimes mis-splits the single
    literal identifier into alias.token or token.token — see the comment at
    known_columns.update(_dotted_literal_parts(...)) for why callers need this.
    """
    parts: set = set()
    for c in known_cols:
        if "." in c:
            parts.update(p for p in c.split(".") if p)
    return parts


def _extract_table_columns(schema_context: str) -> Dict[str, set]:
    """
    Parse schema_context into {BARE_TABLE_NAME (uppercase): {lowercased column
    names}}. Unlike _extract_known_columns (a flat set across the whole
    schema), this is scoped per table — it lets us catch a column that is a
    genuine column SOMEWHERE in the schema but referenced against a table
    alias that does not actually have it (e.g. filtering EVENT_ATTENDANCE on
    "country_code" when only the HCP table has that column and
    EVENT_ATTENDANCE has "country_key" instead). _find_hallucinated_columns
    alone can't catch this because the column genuinely exists in the schema.
    """
    table_cols: Dict[str, set] = {}
    current_table: Optional[str] = None
    in_columns = False
    for line in schema_context.splitlines():
        stripped = line.strip()
        m = re.match(r'^Table:\s*(.+)$', stripped)
        if m:
            bare = m.group(1).split(".")[-1].strip().strip('"').upper()
            current_table = bare
            table_cols.setdefault(current_table, set())
            in_columns = False
            continue
        if stripped.lower() == "columns:":
            in_columns = True
            continue
        if in_columns and current_table:
            if not stripped or stripped.startswith("Table:") or stripped.startswith("FK:") \
                    or stripped.startswith("--") or stripped.startswith("=") \
                    or stripped.startswith("-"):
                in_columns = False
                continue
            col_name = stripped.split(":")[0].split("[")[0].strip().strip('"')
            if col_name:
                table_cols[current_table].add(col_name.lower())
    return table_cols


def _extract_alias_table_map(sql: str) -> Dict[str, str]:
    """
    Map each table alias used in FROM/JOIN clauses to its bare table name
    (uppercase), e.g. `FROM foo."SEA_IDISCOVER_HCPS" AS h` → {"h": "SEA_IDISCOVER_HCPS"}.
    Skips matches where the "alias" is actually a SQL keyword (happens when a
    table has no real alias and the regex greedily grabs the next clause word).
    """
    mapping: Dict[str, str] = {}
    for m in re.finditer(
        r'\b(?:FROM|JOIN)\s+(?:[\w]+\.)?"?(\w+)"?\s+(?:AS\s+)?"?(\w+)"?',
        sql, re.IGNORECASE,
    ):
        table_tok, alias_tok = m.group(1), m.group(2)
        if alias_tok.lower() in _SQL_KEYWORDS or alias_tok.upper() == table_tok.upper():
            continue
        mapping[alias_tok.lower()] = table_tok.upper()
    return mapping


def _find_cross_table_columns(
    sql: str, table_col_map: Dict[str, set], known_cols: set,
) -> List[str]:
    """
    Find alias.column references where `column` is a genuine column in the
    schema (so _find_hallucinated_columns lets it through) but does NOT
    belong to the specific table that alias refers to. Returns bare column
    names (same shape as _find_hallucinated_columns) so callers can reuse
    _strip_hallucinated_conditions / _has_hallucinated_join unchanged.
    """
    if not table_col_map:
        return []
    alias_table = _extract_alias_table_map(sql)
    if not alias_table:
        return []

    sql_stripped = re.sub(r"'[^']*'", "''", sql)
    bad: List[str] = []
    seen: set = set()
    for m in re.finditer(r'\b(\w+)\."?([A-Za-z_]\w*)"?(?!\s*\()', sql_stripped):
        alias, col = m.group(1).lower(), m.group(2)
        low = col.lower()
        if low in _SQL_KEYWORDS or low in seen:
            continue
        table = alias_table.get(alias)
        if not table:
            continue
        cols_for_table = table_col_map.get(table)
        if cols_for_table is None or low in cols_for_table:
            continue
        # Not on this table — only flag if it's a real column elsewhere,
        # otherwise it's plain hallucination already caught upstream.
        if low in known_cols:
            bad.append(col)
            seen.add(low)
    return bad


# SQL keywords and functions that look like identifiers but are never column names
_SQL_KEYWORDS = {
    "select", "from", "where", "join", "inner", "left", "right", "outer",
    "on", "group", "order", "by", "having", "limit", "offset", "as",
    "and", "or", "not", "null", "true", "false", "case", "when", "then",
    "else", "end", "in", "between", "like", "is", "distinct", "all", "any",
    "exists", "union", "intersect", "except", "with", "values", "set",
    "count", "sum", "avg", "min", "max", "coalesce", "cast", "lower",
    "upper", "trim", "substr", "length", "round", "abs", "ifnull",
    "strftime", "date", "datetime", "asc", "desc", "over", "partition",
    "row_number", "rank", "iif", "replace", "typeof",
    # Window functions — never column names
    "lag", "lead", "first_value", "last_value", "nth_value",
    "dense_rank", "percent_rank", "cume_dist", "ntile",
    "nullif", "greatest", "least", "extract", "floor", "ceil", "ceiling",
    "current_date", "current_timestamp", "now", "concat",
    "left", "right", "ltrim", "rtrim", "lpad", "rpad", "split_part",
    "to_char", "to_date", "to_timestamp", "date_trunc", "date_part",
    "format_date", "parse_date", "timestamp_diff", "timestamp_add",
    "interval", "epoch", "unbounded", "preceding", "following", "rows",
    "range", "groups", "current", "exclude",
    "approx_count_distinct", "approx_quantiles", "percentile_cont", "percentile_disc",
    "safe_divide", "safe_cast", "ifnull", "zeroifnull", "nanvl",
    "regexp_extract", "regexp_replace", "regexp_contains",
    "array_agg", "string_agg", "listagg", "group_concat",
    "json_extract", "json_value", "json_query",
}


def _extract_cte_names(sql: str) -> set:
    """
    Names defined by a WITH clause (CTEs) — these are legitimately referenced
    later via FROM/JOIN like a table, but won't appear in table_labels, so
    _find_hallucinated_tables must not flag them.
    """
    names: set = set()
    m = re.search(r'\bWITH\b(.*)', sql, re.IGNORECASE | re.DOTALL)
    if not m:
        return names
    for cm in re.finditer(r'(\w+)\s+AS\s*\(', m.group(1), re.IGNORECASE):
        names.add(cm.group(1).lower())
    return names


def _find_hallucinated_tables(sql: str, table_labels: List[str]) -> List[str]:
    """
    Find FROM/JOIN table names that don't match any real table in the schema
    (case-insensitive), and aren't a CTE name defined earlier in the same
    query. Catches the LLM inventing a plausible-sounding but nonexistent
    table (e.g. SEA_IDISCOVER_EXPENSES when the real table is
    SEA_IDISCOVER_EVENT_EXPENSE) — a mistake _find_hallucinated_columns can't
    catch since it only looks at column references.
    """
    known = {t.lower() for t in table_labels}
    cte_names = _extract_cte_names(sql)
    bad: List[str] = []
    seen: set = set()
    for m in re.finditer(r'\b(?:FROM|JOIN)\s+(?:[\w]+\.)?"?(\w+)"?', sql, re.IGNORECASE):
        tbl = m.group(1)
        low = tbl.lower()
        if low in known or low in cte_names or low in _SQL_KEYWORDS or low in seen:
            continue
        bad.append(tbl)
        seen.add(low)
    return bad


def _find_hallucinated_columns(sql: str, known_cols: set) -> List[str]:
    """
    Find column references in the form  alias.ColumnName  where ColumnName is
    NOT in the known schema.  This pattern (e.g. md.Check_PC) is the most
    reliable hallucination signal — the LLM uses a table alias and a column it
    invented from domain knowledge.

    We intentionally limit the check to dotted references to avoid false
    positives from table/alias names that are not in known_cols.

    Tolerates the column being double-quoted (alias."Col") — the dominant
    style this pipeline generates for Snowflake and other quote-preserving
    dialects. Without this, the regex would never match real generated SQL
    and this whole safety net would be silently inert for those dialects.
    """
    if not known_cols:
        return []

    # Strip string literals so quoted values don't confuse the regex
    sql_stripped = re.sub(r"'[^']*'", "''", sql)

    hallucinated = []
    seen: set = set()
    # Match   word.Identifier   where Identifier is not followed by '(' (functions)
    for m in re.finditer(r'\b[A-Za-z_]\w*\."?([A-Za-z_]\w*)"?(?!\s*\()', sql_stripped):
        col = m.group(1)
        low = col.lower()
        if low in _SQL_KEYWORDS:
            continue
        if low in known_cols:
            continue
        if low not in seen:
            hallucinated.append(col)
            seen.add(low)
    return hallucinated


def _strip_hallucinated_conditions(sql: str, bad_cols: List[str]) -> str:
    """
    Remove WHERE / AND / OR conditions that reference hallucinated dotted columns
    (e.g. ``AND md.Check_PC = 'X'``).  Returns cleaned SQL so the rest of the
    query can still execute.  Falls back to the original SQL on any error.

    Handles the three most common placements:
      1. WHERE alias.col op value  (sole condition → remove entire WHERE clause)
      2. WHERE alias.col op value AND next_cond  (→ convert next_cond to WHERE)
      3. AND/OR alias.col op value  (→ remove the AND/OR arm)
    """
    try:
        for col in bad_cols:
            # Column may appear bare (alias.col) or quoted (alias."col") in the
            # generated SQL — the schema-quoting dialects (Snowflake, etc.) this
            # pipeline targets always quote columns, so tolerate an optional
            # quote on both sides of the bare name everywhere below.
            cp = re.escape(col) + r'"?'
            # Value token: a quoted string, a number, or a bare word (handles =, LIKE, IN, IS)
            val = r"""(?:'[^']*'|\([^)]*\)|[^\s,)]+)"""
            op  = r"(?:=|!=|<>|>=|<=|>|<|(?:NOT\s+)?LIKE|(?:NOT\s+)?IN|IS(?:\s+NOT)?)"

            # Case 3: AND/OR condition  — simplest, remove the entire arm
            sql = re.sub(
                r"(?i)\s+(?:AND|OR)\s+\w+\.\"?" + cp + r"\s+" + op + r"\s*" + val,
                "",
                sql,
            )

            # Case 2: WHERE col ... AND next → replace with WHERE next
            sql = re.sub(
                r"(?i)\bWHERE\s+\w+\.\"?" + cp + r"\s+" + op + r"\s*" + val + r"\s+AND\s+",
                "WHERE ",
                sql,
            )

            # Case 1: WHERE col ... (nothing follows, or clause keywords follow)
            sql = re.sub(
                r"(?i)\s+WHERE\s+\w+\.\"?" + cp + r"\s+" + op + r"\s*" + val
                + r"(?=\s*(?:GROUP\b|ORDER\b|HAVING\b|LIMIT\b|$))",
                "",
                sql,
            )

        return sql.strip()
    except Exception:
        return sql


def _strip_invalid_join(sql: str, invalid_join_descriptions: List[str]) -> Optional[str]:
    """
    Attempt to salvage a query that contains an invalid JOIN (one whose ON
    condition is not a real, schema-listed FK pair) by removing the entire
    JOIN clause and any SELECT/WHERE/GROUP BY/ORDER BY references to it.

    Returns the salvaged SQL string, or None if the query cannot be salvaged.
    """
    try:
        # Extract aliases from JOIN clauses — "JOIN tbl AS alias" or "JOIN tbl alias"
        join_aliases = re.findall(
            r'\bJOIN\b\s+\S+\s+(?:AS\s+)?(\w+)\b',
            sql, re.IGNORECASE,
        )
        if not join_aliases:
            return None

        # Determine which aliases are involved in the invalid join conditions
        bad_aliases: set = set()
        for desc in invalid_join_descriptions:
            # desc looks like "alias1.col = alias2.col"
            for m in re.finditer(r'\b(\w+)\.\w+', desc):
                bad_aliases.add(m.group(1).lower())

        # Only keep aliases that appear in actual JOIN clauses (not the primary table)
        bad_aliases = {a for a in bad_aliases if a.lower() in [j.lower() for j in join_aliases]}
        if not bad_aliases:
            return None

        return _strip_join_aliases(sql, bad_aliases)
    except Exception:
        return None


def _find_unnecessary_joins(sql: str) -> List[str]:
    """
    Return aliases of joined tables whose columns are never referenced
    outside their own JOIN...ON clause (not in SELECT/WHERE/GROUP BY/HAVING/
    ORDER BY, and not used as a bridge by another JOIN's ON condition).

    Such a JOIN contributes nothing to the result — it can only multiply
    base-table rows (e.g. a one-to-many child table joined "for context")
    and silently inflate COUNT(*)/SUM() results. See the claims/adjuster/
    fraud_indicator/deductible_tracking bug this guards against.
    """
    try:
        join_matches = list(re.finditer(
            r'\b(?:LEFT\s+|RIGHT\s+|INNER\s+|OUTER\s+|FULL\s+)?JOIN\s+\S+\s+(?:AS\s+)?(\w+)\s+ON\s+(.+?)'
            r'(?=\b(?:LEFT|RIGHT|INNER|OUTER|FULL|JOIN|WHERE|GROUP|ORDER|HAVING|LIMIT)\b|$)',
            sql, re.IGNORECASE | re.DOTALL,
        ))
        if not join_matches:
            return []

        own_on_clauses = {m.group(1).lower(): m.group(2) for m in join_matches}

        # Remove each JOIN...ON block so residual "alias." hits reflect real
        # usage (SELECT/WHERE/GROUP BY/etc), not the join's own condition.
        rest = sql
        for m in join_matches:
            rest = rest.replace(m.group(0), ' ', 1)

        unnecessary = []
        for alias, _ in own_on_clauses.items():
            ae = re.escape(alias)
            # \."? tolerates the quoted-column style this pipeline generates
            # for Snowflake etc. (alias."col") — without it, EVERY usage of a
            # quoted column looks unused and the join gets wrongly stripped,
            # orphaning the alias and cascading into self-heal guessing a
            # replacement table name.
            if re.search(r'\b' + ae + r'\."?\w+', rest, re.IGNORECASE):
                continue  # used in SELECT/WHERE/GROUP BY/HAVING/ORDER BY

            # Don't strip if another JOIN's ON clause depends on this alias
            # (it's a bridge table in a multi-hop join, not dead weight).
            bridged = any(
                other_alias != alias and re.search(r'\b' + ae + r'\."?\w+', on_body, re.IGNORECASE)
                for other_alias, on_body in own_on_clauses.items()
            )
            if bridged:
                continue

            unnecessary.append(alias)
        return unnecessary
    except Exception:
        return []


def _strip_unnecessary_join(sql: str, aliases: List[str]) -> Optional[str]:
    """Remove JOIN(s) identified by _find_unnecessary_joins as contributing nothing."""
    return _strip_join_aliases(sql, {a.lower() for a in aliases})


def _strip_join_aliases(sql: str, bad_aliases: set) -> Optional[str]:
    """
    Remove the JOIN clause(s) for the given table aliases, plus any SELECT
    columns / WHERE / GROUP BY / ORDER BY references that use those aliases.

    Shared removal core used by both _strip_invalid_join (aliases joined via
    a non-existent FK pair) and _strip_unnecessary_join (aliases joined but
    never referenced outside their own ON clause).

    Returns the salvaged SQL string, or None if the query cannot be salvaged
    (e.g. all SELECT columns came from the removed table).
    """
    try:
        if not bad_aliases:
            return None

        salvaged = sql
        for alias in bad_aliases:
            ae = re.escape(alias)
            # Remove the JOIN block: JOIN ... ON condition (stop at next JOIN/WHERE/GROUP/ORDER)
            salvaged = re.sub(
                r'(?i)\b(?:LEFT\s+|RIGHT\s+|INNER\s+|OUTER\s+|FULL\s+)?JOIN\s+\S+\s+'
                r'(?:AS\s+)?' + ae +
                r'\b\s+ON\s+.+?(?=\b(?:LEFT|RIGHT|INNER|OUTER|FULL|JOIN|WHERE|GROUP|ORDER|HAVING|LIMIT)\b|$)',
                ' ',
                salvaged,
                flags=re.IGNORECASE | re.DOTALL,
            )
            # Remove SELECT columns referencing the removed alias: "alias.col [AS name],"
            salvaged = re.sub(
                r'(?i),?\s*' + ae + r'\.\w+(?:\s+AS\s+\w+)?(?=\s*[,\n]|\s*FROM)',
                '',
                salvaged,
            )
            # Remove WHERE / AND conditions referencing the removed alias
            val = r"""(?:'[^']*'|\([^)]*\)|[^\s,)]+)"""
            op  = r"(?:=|!=|<>|>=|<=|>|<|(?:NOT\s+)?LIKE|(?:NOT\s+)?IN|IS(?:\s+NOT)?)"
            salvaged = re.sub(
                r'(?i)\s+(?:AND|OR)\s+' + ae + r'\.\w+\s+' + op + r'\s*' + val, '', salvaged
            )
            salvaged = re.sub(
                r'(?i)\bWHERE\s+' + ae + r'\.\w+\s+' + op + r'\s*' + val + r'\s+AND\s+',
                'WHERE ', salvaged,
            )
            salvaged = re.sub(
                r'(?i)\s+WHERE\s+' + ae + r'\.\w+\s+' + op + r'\s*' + val
                + r'(?=\s*(?:GROUP\b|ORDER\b|HAVING\b|LIMIT\b|$))',
                '', salvaged,
            )
            # Remove GROUP BY / ORDER BY references to the removed alias
            salvaged = re.sub(r'(?i),\s*' + ae + r'\.\w+', '', salvaged)

        # Collapse multiple spaces/commas left behind
        salvaged = re.sub(r',\s*,', ',', salvaged)
        salvaged = re.sub(r'SELECT\s*,', 'SELECT ', salvaged, flags=re.IGNORECASE)
        salvaged = re.sub(r',\s*(FROM\b)', r' \1', salvaged, flags=re.IGNORECASE)
        salvaged = salvaged.strip()

        # Sanity check: must still have a FROM clause and at least one SELECT column
        if not re.search(r'\bFROM\b', salvaged, re.IGNORECASE):
            return None
        if not re.search(r'\bSELECT\b.+\bFROM\b', salvaged, re.IGNORECASE | re.DOTALL):
            return None

        return salvaged
    except Exception:
        return None


_AUDIT_JOIN_COL_RE = re.compile(
    r"^(dw_)?(created|updated|modified|inserted|loaded)_?(at|date|ts|time|on)?$"
    r"|^(etl_load|load|dw_load)_(date|ts|time)$"
    r"|^(row|record)_(created|updated|inserted)_?(at|date)?$"
    r"|^last_(updated|modified)_?(at|date)?$",
    re.IGNORECASE,
)


def _find_audit_column_joins(sql: str) -> List[str]:
    """Return audit/ETL housekeeping column names (e.g. DW_CREATED_AT,
    UPDATED_AT) used in a JOIN ON clause. These are row-load/modify
    timestamps, not real foreign keys — joining on them silently returns
    zero or spuriously wrong rows instead of erroring, so a hard reject here
    (rather than trusting the LLM to have followed rule 10b.e) is needed."""
    on_blocks = re.findall(
        r'\bON\b\s+(.+?)(?=\bWHERE\b|\bGROUP\b|\bORDER\b|\bHAVING\b|\bLIMIT\b|\bJOIN\b|$)',
        sql, re.IGNORECASE | re.DOTALL,
    )
    if not on_blocks:
        return []
    on_text = " ".join(on_blocks)
    col_refs = re.findall(r'\b\w+\.\"?([A-Za-z_][A-Za-z0-9_]*)\"?', on_text)
    return sorted({c for c in col_refs if _AUDIT_JOIN_COL_RE.match(c.lower())})


def _has_hallucinated_join(sql: str, bad_cols: List[str]) -> bool:
    """
    Return True if any of bad_cols appear in a context that cannot be salvaged
    by stripping a WHERE condition.  Covers:
      • JOIN ... ON alias.bad_col = ...
      • Subqueries / IN (...) / EXISTS (...) that reference bad_col
    """
    sql_lower = sql.lower()

    # Check JOIN ON blocks
    on_blocks = re.findall(
        r'\bON\b\s+(.+?)(?=\bWHERE\b|\bGROUP\b|\bORDER\b|\bHAVING\b|\bLIMIT\b|\bJOIN\b|$)',
        sql, re.IGNORECASE | re.DOTALL,
    )
    if on_blocks:
        on_text = " ".join(on_blocks).lower()
        if any(col.lower() in on_text for col in bad_cols):
            return True

    # Check inside any subquery parentheses (catches IN (...), EXISTS (...), scalar)
    # A subquery contains SELECT, so look for (... SELECT ... bad_col ...)
    subquery_blocks = re.findall(r'\(([^()]*\bSELECT\b[^()]*)\)', sql, re.IGNORECASE | re.DOTALL)
    for block in subquery_blocks:
        block_lower = block.lower()
        if any(col.lower() in block_lower for col in bad_cols):
            return True

    return False


def _extract_valid_join_pairs(schema_context: str) -> set:
    """
    Parse the schema context text and return a set of frozensets, each
    containing the two lower-cased column names that form a valid join key.

    Recognises two formats produced by _summarise_graph:
      • Shared column  : "  - colname: shared by table1, table2"
      • Explicit FK    : "  - JOIN ON schema.tbl1.col1 = schema.tbl2.col2"
      • Per-table FK   : "  FK (rel): JOIN ON schema.tbl1.col1 = schema.tbl2.col2"

    frozenset({'customer_id'}) — same-name join (both sides use same col)
    frozenset({'order_id', 'fk_order'}) — cross-name FK
    """
    valid: set = set()

    for line in schema_context.splitlines():
        stripped = line.strip()

        # Shared column: "- colname: shared by ..."
        m = re.match(r'^-\s+([\w]+):\s+shared\s+by\s+', stripped, re.IGNORECASE)
        if m:
            col = m.group(1).lower()
            valid.add(frozenset([col]))  # single-element frozenset = same-name join
            continue

        # FK / join-on lines: both "  - JOIN ON ..." and "  FK (...): JOIN ON ..."
        # Each side may be bare ("tbl.col"), schema-qualified ("schema.tbl.col"),
        # or deeper — (?:\w+\.)* greedily eats every leading identifier segment
        # so the capture group always lands on the FINAL segment (the column),
        # regardless of how many dots precede it. A lazy \S+? here would grab
        # the wrong segment (e.g. the table name) on whichever side has no
        # trailing token forcing it to keep backtracking — verified this was
        # silently mis-parsing every schema-qualified cross-name FK line.
        m = re.search(
            r'\bJOIN\s+ON\s+(?:\w+\.)*(\w+)\s*=\s*(?:\w+\.)*(\w+)',
            stripped, re.IGNORECASE,
        )
        if m:
            c1 = m.group(1).lower()
            c2 = m.group(2).lower()
            valid.add(frozenset([c1, c2]))

    return valid


def _find_invalid_join_conditions(sql: str, valid_join_pairs: set) -> List[str]:
    """
    Scan every JOIN ... ON lhs.col1 = rhs.col2 in the SQL.
    Return a list of human-readable strings describing join conditions whose
    column pair does NOT appear in valid_join_pairs.

    If valid_join_pairs is empty the schema had no join key info — return [].
    """
    if not valid_join_pairs:
        return []

    # Strip string literals to avoid false positives inside quoted values
    sql_clean = re.sub(r"'[^']*'", "''", sql)

    invalid: List[str] = []
    # Match:  [JOIN ...] ON alias.col1 = alias.col2
    # Handles multi-line SQL with DOTALL, stops at the next JOIN/WHERE keyword
    for m in re.finditer(
        r'\bON\s+(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)',
        sql_clean, re.IGNORECASE,
    ):
        lhs_alias, lhs_col, rhs_alias, rhs_col = (
            m.group(1), m.group(2), m.group(3), m.group(4),
        )
        col1 = lhs_col.lower()
        col2 = rhs_col.lower()
        pair = frozenset([col1, col2])

        if col1 == col2:
            # Same-name join (e.g. ON a.customer_id = b.customer_id):
            # valid when the column appears as a shared key (single-element frozenset)
            # OR as part of an explicit FK pair.
            if frozenset([col1]) not in valid_join_pairs and pair not in valid_join_pairs:
                invalid.append(f"{lhs_alias}.{lhs_col} = {rhs_alias}.{rhs_col}")
        else:
            # Cross-name join (e.g. ON fms.segment_id = dr.region_id):
            # ONLY valid when this exact (col1, col2) pair appears in an FK line.
            # The fact that col1 is a shared column elsewhere does NOT make this
            # join valid — frozenset fallback must NOT apply here.
            if pair not in valid_join_pairs:
                invalid.append(f"{lhs_alias}.{lhs_col} = {rhs_alias}.{rhs_col}")

    return invalid


def _qualify_sql(sql: str, db_schema: str, known_tables: List[str]) -> str:
    """
    Safety net: if the LLM wrote `FROM orders` but the schema is `public`,
    rewrite bare table references to schema-qualified form `public.orders`.

    Restricted to FROM / JOIN contexts ONLY.  Replacing table names in SELECT
    column lists or WHERE predicates creates invalid 3-part names like
    `public.column_name` which cause syntax errors in PostgreSQL/SQLite.
    """
    if not db_schema or not known_tables:
        return sql

    # Sort longest first to avoid partial replacements (e.g. "order" before "orders")
    for table in sorted(known_tables, key=len, reverse=True):
        qualified = f"{db_schema}.{table}"
        # Skip if already present in the SQL (case-insensitive)
        if re.search(re.escape(qualified), sql, re.IGNORECASE):
            continue
        # Only replace immediately after FROM or JOIN keywords so we never
        # accidentally qualify a column reference or a string literal value.
        sql = re.sub(
            r'(\b(?:FROM|JOIN)\b)(\s+)' + re.escape(table) + r'(?![\.\w])',
            lambda m, q=qualified: m.group(1) + m.group(2) + q,
            sql,
            flags=re.IGNORECASE,
        )

    return sql


_PERCENTAGE_KEYWORDS = re.compile(
    r'\b(percent(?:age)?|%|share|proportion|breakdown|distribution|'
    r'ratio|split|how\s+much\s+(?:is|are)|out\s+of\s+total)\b',
    re.IGNORECASE,
)


def _fix_percentage(sql: str, natural_query: str, db_type: str = "") -> str:
    """
    If the question asks for a percentage and the SQL has a GROUP BY aggregate
    but no percentage column, inject a window-function percentage expression.

    Works for both COUNT(*) and SUM(col) aggregates.
    Leaves the SQL unchanged if it already contains a percentage expression.

    NOTE: SUM(COUNT(*)) OVER () is a nested aggregate inside a window function.
    This is valid in PostgreSQL, Redshift, BigQuery, and SQL Server 2012+, but
    NOT in SQLite (used for CSV/Excel sources).  Skip injection for SQLite so
    we do not introduce syntax errors — the system prompt instructs the LLM to
    handle percentages directly.
    """
    if not _PERCENTAGE_KEYWORDS.search(natural_query):
        return sql

    # Skip for SQLite/file-based sources to avoid nested-aggregate syntax errors
    _SQLITE_BASED = {"sqlite", "csv", "excel"}
    if db_type.lower() in _SQLITE_BASED:
        return sql

    sql_upper = sql.upper()

    # Already has a percentage calculation — leave it alone
    if "100.0" in sql or "100.0" in sql_upper or re.search(r'\bPCT\b|\bPERCENT', sql, re.IGNORECASE):
        return sql

    # Only patch GROUP BY queries (window function needs GROUP BY context)
    if "GROUP BY" not in sql_upper:
        return sql

    # Find the last column in the SELECT list before FROM and inject percentage
    # Pattern: match COUNT(*) or SUM(col) aggregate in SELECT
    agg_match = re.search(
        r'(COUNT\s*\(\s*\*\s*\)|SUM\s*\([^)]+\))\s*(?:AS\s+\w+)?',
        sql, re.IGNORECASE
    )
    if not agg_match:
        return sql

    agg_expr = agg_match.group(1)  # e.g. COUNT(*) or SUM(col_Revenue)

    # Build percentage window expression
    pct_expr = f"ROUND({agg_expr} * 100.0 / SUM({agg_expr}) OVER (), 2) AS Percentage"

    # Inject before FROM
    from_pos = sql_upper.find(" FROM ")
    if from_pos == -1:
        return sql

    patched = sql[:from_pos] + ",\n       " + pct_expr + sql[from_pos:]
    logger.info("plan_node: percentage question detected — injected window percentage column")
    return patched


_HEADCOUNT_KEYWORDS = re.compile(
    r'\b(headcount|head\s*count|head count|fte|people|employees?|'
    r'staff|workforce|workers?|resources?|associates?|members?|'
    r'how\s+many|number\s+of\s+(?:people|employees?|staff|resources?))\b',
    re.IGNORECASE,
)

_HEADCOUNT_COL_NAMES = re.compile(
    r'\b(headcount|head_count|fte|employee_count|emp_count|staff_count|'
    r'resource_count|count_of|no_of|num_of)\b',
    re.IGNORECASE,
)


def _fix_count_vs_sum(sql: str, natural_query: str) -> str:
    """
    If the question is about headcount / how many people, and the SQL uses
    SUM(col) on a column that is NOT a dedicated numeric headcount column,
    replace it with COUNT(*).

    We do NOT touch SUM() on columns that look like genuine numeric measures
    (revenue, amount, cost, salary, etc.).
    """
    if not _HEADCOUNT_KEYWORDS.search(natural_query):
        return sql  # Not a headcount question — leave SQL unchanged

    _MONEY_COL = re.compile(
        r'\b(revenue|amount|cost|salary|wage|budget|expense|'
        r'price|value|margin|profit|loss)\b',
        re.IGNORECASE,
    )

    def _replace_sum(m: re.Match) -> str:
        col = m.group(1)
        # If the column itself looks like a monetary/measure column, leave it
        if _MONEY_COL.search(col):
            return m.group(0)
        # If the column looks like a dedicated headcount column, keep SUM (values >1)
        if _HEADCOUNT_COL_NAMES.search(col):
            return m.group(0)
        # Otherwise replace SUM(col) → COUNT(*)
        return "COUNT(*)"

    fixed = re.sub(r'\bSUM\s*\(\s*([^()]+?)\s*\)', _replace_sum, sql, flags=re.IGNORECASE)
    if fixed != sql:
        logger.info(
            "plan_node: headcount question detected — replaced SUM() with COUNT(*) in SQL"
        )
    return fixed


_COUNT_ARG_PATTERN = re.compile(r'\bCOUNT\s*\(\s*([^()]+?)\s*\)', re.IGNORECASE)


def _fix_count_after_join(sql: str) -> str:
    """
    A one-to-many JOIN duplicates base-table rows once per matching child row
    (e.g. one claim JOINed to 3 adjuster records becomes 3 rows for that
    claim). COUNT(col) or COUNT(*) after such a JOIN silently overcounts.

    If the query contains a JOIN, rewrite any non-DISTINCT COUNT(<simple
    column reference>) to COUNT(DISTINCT <same column>) — this is always
    safe and never changes the result for a query that didn't need the
    JOIN's fan-out in the first place.

    COUNT(*) is left untouched here: there is no column to make DISTINCT, and
    guessing a primary-key name would be unsafe. The caller flags remaining
    COUNT(*)-after-JOIN cases via _has_unguarded_count_star_after_join so a
    human/reviewer sees them instead of the query silently overcounting.
    """
    if not re.search(r'\bJOIN\b', sql, re.IGNORECASE):
        return sql

    def _wrap_distinct(m: re.Match) -> str:
        arg = m.group(1).strip()
        if re.match(r'(?i)^distinct\s', arg):
            return m.group(0)  # already DISTINCT
        if arg == '*':
            return m.group(0)  # no safe column to guess — leave for caller to flag
        # Only rewrite simple column references (col or alias.col) — never
        # touch expressions, CASE, nested calls, etc.
        if re.fullmatch(r'\w+(\.\w+)?', arg):
            return f"COUNT(DISTINCT {arg})"
        return m.group(0)

    fixed = _COUNT_ARG_PATTERN.sub(_wrap_distinct, sql)
    if fixed != sql:
        logger.info(
            "plan_node: JOIN present — rewrote non-distinct COUNT(col) to "
            "COUNT(DISTINCT col) to prevent row-inflation from the JOIN"
        )
    return fixed


def _has_unguarded_count_star_after_join(sql: str) -> bool:
    """True if the query JOINs another table and uses COUNT(*) with no DISTINCT guard."""
    if not re.search(r'\bJOIN\b', sql, re.IGNORECASE):
        return False
    return bool(re.search(r'\bCOUNT\s*\(\s*\*\s*\)', sql, re.IGNORECASE))


_AGG_PATTERN = re.compile(
    r'\b(GROUP\s+BY|COUNT\s*\(|SUM\s*\(|AVG\s*\(|MIN\s*\(|MAX\s*\()',
    re.IGNORECASE,
)

_LIMIT_PATTERN      = re.compile(r'\bLIMIT\s+\d+(\s+OFFSET\s+\d+)?', re.IGNORECASE)
_TOP_PATTERN        = re.compile(r'\bSELECT\s+TOP\s+\d+\b', re.IGNORECASE)
_FETCH_FIRST_PATTERN = re.compile(r'\bFETCH\s+FIRST\s+\d+\s+ROWS?\s+ONLY\b', re.IGNORECASE)
_ROWNUM_PATTERN     = re.compile(r'\bROWNUM\s*<=?\s*\d+\b', re.IGNORECASE)


def _enforce_sql_limits(sql: str, row_limit: int, db_type: str = "") -> str:
    """
    Enforce row-limit rules after the LLM has generated SQL, using the correct
    syntax for the target database.

    - Aggregation queries (GROUP BY / COUNT / SUM / AVG / MIN / MAX):
        Remove any limit clause unless it is a small intentional top-N
        (≤50 rows AND there is an ORDER BY).  Limiting aggregations silently
        drops groups and produces wrong totals.

    - Raw-row queries (no aggregation):
        If no limit is present, add one using db-appropriate syntax:
          PostgreSQL / SQLite / Redshift / BigQuery / CSV / Excel : LIMIT N
          SQL Server                                               : SELECT TOP N …
          Oracle                                                   : … FETCH FIRST N ROWS ONLY
    """
    db        = db_type.lower()
    is_agg    = bool(_AGG_PATTERN.search(sql))
    has_limit = bool(_LIMIT_PATTERN.search(sql))
    has_top   = bool(_TOP_PATTERN.search(sql))
    has_fetch = bool(_FETCH_FIRST_PATTERN.search(sql))
    has_rownum = bool(_ROWNUM_PATTERN.search(sql))
    has_any_limit = has_limit or has_top or has_fetch or has_rownum

    has_order_by = bool(re.search(r'\bORDER\s+BY\b', sql, re.IGNORECASE))

    if is_agg:
        if not has_any_limit:
            return sql  # already clean

        # Determine the limit value for the intentional-top-N check
        limit_val = 0
        if has_limit:
            m = re.search(r'\bLIMIT\s+(\d+)', sql, re.IGNORECASE)
            limit_val = int(m.group(1)) if m else 0
        elif has_top:
            m = re.search(r'\bSELECT\s+TOP\s+(\d+)\b', sql, re.IGNORECASE)
            limit_val = int(m.group(1)) if m else 0
        elif has_fetch:
            m = re.search(r'\bFETCH\s+FIRST\s+(\d+)\s+ROWS?\s+ONLY\b', sql, re.IGNORECASE)
            limit_val = int(m.group(1)) if m else 0

        # Keep intentional small top-N with ORDER BY
        if limit_val <= 50 and has_order_by:
            return sql

        # Strip the limit clause
        cleaned = sql
        if has_limit:
            cleaned = _LIMIT_PATTERN.sub('', cleaned)
        if has_top:
            # Remove TOP N from SELECT clause: "SELECT TOP 100 " → "SELECT "
            cleaned = re.sub(
                r'(\bSELECT\b\s+)TOP\s+\d+\s+', r'\1', cleaned, flags=re.IGNORECASE
            )
        if has_fetch:
            cleaned = _FETCH_FIRST_PATTERN.sub('', cleaned)
        cleaned = cleaned.strip().rstrip(';').strip()
        logger.info("plan_node: stripped limit clause from aggregation query")
        return cleaned

    else:
        # Raw-row query — add db-appropriate limit if none present
        if has_any_limit:
            return sql  # LLM already added a top-N; leave it

        sql_clean = sql.rstrip().rstrip(';').strip()

        if db == "sqlserver":
            # Insert TOP N immediately after SELECT (before DISTINCT / column list)
            patched = re.sub(
                r'(\bSELECT\b)(\s+)',
                lambda m: m.group(1) + m.group(2) + f"TOP {row_limit} ",
                sql_clean, count=1, flags=re.IGNORECASE,
            )
            logger.info("plan_node: added TOP %d to raw-row SQL Server query", row_limit)
            return patched

        elif db == "oracle":
            patched = sql_clean + f" FETCH FIRST {row_limit} ROWS ONLY"
            logger.info("plan_node: added FETCH FIRST %d to raw-row Oracle query", row_limit)
            return patched

        else:
            # PostgreSQL, Redshift, BigQuery, SQLite, CSV, Excel
            patched = sql_clean + f" LIMIT {row_limit}"
            logger.info("plan_node: added LIMIT %d to raw-row query", row_limit)
            return patched


def _fix_sqlserver_subquery_limits(sql: str, db_type: str) -> str:
    """
    SQL Server does not support LIMIT.  After _enforce_sql_limits has handled
    the outermost SELECT, any LIMIT N that remains (inside CTEs, derived tables,
    or IN-subqueries) is still invalid SQL Server syntax.

    This pass converts all remaining  "ORDER BY <terms> LIMIT N"  patterns to
    "ORDER BY <terms> OFFSET 0 ROWS FETCH NEXT N ROWS ONLY", which is the
    SQL Server equivalent and is also legal inside CTEs and derived tables
    (because OFFSET/FETCH counts as a valid row-limiter).

    When LIMIT appears without a preceding ORDER BY (unusual), it is simply
    dropped — the caller should rely on _enforce_sql_limits having added TOP
    to the outermost SELECT.

    Running this BEFORE _fix_subquery_order_by ensures that ORDER BY clauses
    that precede LIMIT are now protected by OFFSET, so the subquery-ORDER-BY
    fixer leaves them in place.
    """
    if db_type.lower() != "sqlserver":
        return sql
    if not re.search(r'\bLIMIT\s+\d+\b', sql, re.IGNORECASE):
        return sql

    # Replace "ORDER BY <anything> LIMIT N" with OFFSET/FETCH equivalent.
    # The non-greedy [^;)]+ stops at subquery boundaries (')' or ';').
    # Captures: (ORDER BY clause)(optional whitespace)(LIMIT N)
    sql = re.sub(
        r'(\bORDER\s+BY\b[^;)]+?)\s+LIMIT\s+(\d+)',
        lambda m: m.group(1) + f" OFFSET 0 ROWS FETCH NEXT {m.group(2)} ROWS ONLY",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Strip any bare LIMIT N that remain (no preceding ORDER BY in the block)
    leftover = re.search(r'\bLIMIT\s+\d+\b', sql, re.IGNORECASE)
    if leftover:
        sql = re.sub(r'\bLIMIT\s+\d+\b', '', sql, flags=re.IGNORECASE)
        logger.info("plan_node: stripped bare LIMIT (no ORDER BY) from SQL Server query")

    logger.info("plan_node: converted LIMIT → OFFSET/FETCH for SQL Server subqueries")
    return sql


def _fix_subquery_order_by(sql: str, db_type: str) -> str:
    """
    SQL Server (error 1033): ORDER BY is illegal inside a subquery, CTE body,
    derived table, view, or inline function unless TOP, OFFSET, or FOR XML is
    also present in that same block.

    This function walks paren depth to locate every ORDER BY at depth > 0,
    determines whether the enclosing paren is a window-function OVER clause
    (in which case ORDER BY is REQUIRED and must never be stripped), checks
    whether the non-OVER enclosing block contains TOP / OFFSET / FOR XML, and
    strips the ORDER BY only when none of those protectors is present.

    Key fix vs the naive approach
    ------------------------------
    The backward-scan finds the NEAREST enclosing '('.  For:
        LAG(x) OVER (PARTITION BY k ORDER BY period_id)
    the nearest '(' is the OVER paren, not the outer CTE/subquery paren.
    Without the OVER check, the old code stripped the ORDER BY from the window
    spec itself, breaking the window function.  We now detect the OVER paren
    and skip it — ORDER BY inside OVER is always valid, even in CTEs.

    All other dialects allow ORDER BY in subqueries, so this is a no-op for them.
    """
    if db_type.lower() != "sqlserver":
        return sql
    if not re.search(r'\bORDER\s+BY\b', sql, re.IGNORECASE):
        return sql

    spans_to_remove: List[tuple] = []

    for m in re.finditer(r'\bORDER\s+BY\b', sql, re.IGNORECASE):
        ob_pos = m.start()

        # Paren depth at this ORDER BY position
        prefix = sql[:ob_pos]
        depth  = prefix.count('(') - prefix.count(')')
        if depth <= 0:
            continue  # outer-level ORDER BY — leave alone

        # Scan backwards to find the nearest enclosing open paren
        block_open: Optional[int] = None
        d = 0
        for i in range(ob_pos - 1, -1, -1):
            c = sql[i]
            if c == ')':
                d += 1
            elif c == '(':
                if d == 0:
                    block_open = i
                    break
                d -= 1

        if block_open is None:
            continue

        # ── Critical guard: skip ORDER BY that is inside an OVER() clause ────
        # SQL Server (and all other dialects) require ORDER BY inside OVER()
        # for window functions.  The restriction on ORDER BY applies only to
        # SET-ordering at the query/CTE level, NOT to window specs.
        # Detection: if the text immediately before the '(' (ignoring spaces)
        # is the keyword OVER, this is a window spec — leave it completely alone.
        text_before_open = sql[:block_open].rstrip()
        if re.search(r'\bOVER$', text_before_open, re.IGNORECASE):
            continue  # ORDER BY is part of a window function spec — do not touch

        # Scan forwards to find the matching close paren
        block_close: Optional[int] = None
        d = 1
        for i in range(block_open + 1, len(sql)):
            c = sql[i]
            if c == '(':
                d += 1
            elif c == ')':
                d -= 1
                if d == 0:
                    block_close = i
                    break

        if block_close is None:
            continue

        # ── Protection checks — must be depth-aware ──────────────────────────
        # BUG FIX: the naive approach of re.search(r'\bOFFSET\b', block_content)
        # is too broad.  If a *nested* subquery inside this CTE/subquery block has
        # an OFFSET clause (e.g. added by _fix_sqlserver_subquery_limits), the outer
        # ORDER BY would be incorrectly left alone.  We must only count TOP / OFFSET /
        # FOR XML that appear at depth-0 relative to this block.
        #
        # "Depth 0 within the block" means: not inside any further nested ( ).
        # • TOP N appears BEFORE the ORDER BY → scan block_open..ob_pos
        # • OFFSET / FOR XML appear AFTER the ORDER BY → scan ob_pos..block_close

        # Helper: does `pattern` appear at paren-depth 0 within `text`?
        def _keyword_at_d0(text: str, pattern: str) -> bool:
            depth = 0
            for tok in re.finditer(r'[()]|' + pattern, text, re.IGNORECASE):
                s = tok.group(0)
                if s == '(':
                    depth += 1
                elif s == ')':
                    depth -= 1
                elif depth == 0:
                    return True
            return False

        before_ob = sql[block_open + 1 : ob_pos]
        after_ob  = sql[ob_pos : block_close]          # includes ORDER BY itself

        # Leave alone if SELECT TOP N precedes ORDER BY at depth 0 in this block
        if _keyword_at_d0(before_ob, r'SELECT\s+(?:DISTINCT\s+)?TOP\s+\d+'):
            continue
        # Leave alone if OFFSET follows ORDER BY at depth 0 (i.e. same ORDER BY clause)
        if _keyword_at_d0(after_ob, r'\bOFFSET\b'):
            continue
        # Leave alone if FOR XML follows ORDER BY at depth 0
        if _keyword_at_d0(after_ob, r'\bFOR\s+XML\b'):
            continue

        # Strip from ORDER BY up to (but not past) the closing paren,
        # also consuming any leading whitespace before ORDER BY
        strip_start = ob_pos
        while strip_start > block_open and sql[strip_start - 1] in (' ', '\t', '\n', '\r'):
            strip_start -= 1

        spans_to_remove.append((strip_start, block_close))

    if not spans_to_remove:
        return sql

    # Apply right-to-left to preserve earlier positions
    result = sql
    for start, end in sorted(spans_to_remove, key=lambda x: -x[0]):
        result = result[:start] + result[end:]
        logger.info("plan_node: stripped bare ORDER BY from subquery/CTE context (SQL Server)")

    return result


def _split_select_list(text: str) -> List[str]:
    """
    Split a SQL SELECT-list string on top-level commas only (depth 0).
    Handles nested parentheses from functions like COALESCE(a, b),
    ROUND(x, 2), CASE WHEN ... END, window functions, etc.
    """
    parts: List[str] = []
    depth = 0
    buf: List[str] = []
    for ch in text:
        if ch == '(':
            depth += 1
            buf.append(ch)
        elif ch == ')':
            depth -= 1
            buf.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append(''.join(buf).strip())
    return [p for p in parts if p]


def _split_order_by_terms(text: str) -> List[str]:
    """
    Split ORDER BY clause on top-level commas and strip direction keywords
    (ASC / DESC / NULLS FIRST / NULLS LAST) from each term.
    Handles expressions like COALESCE(a, b) DESC without splitting on the
    inner comma.
    """
    raw_terms = _split_select_list(text)  # reuse paren-aware splitter
    cleaned: List[str] = []
    for term in raw_terms:
        t = re.sub(
            r'\s*\b(NULLS\s+(?:FIRST|LAST)|ASC|DESC)\b\s*$', '',
            term, flags=re.IGNORECASE,
        ).rstrip(';').strip()
        if t:
            cleaned.append(t)
    return cleaned


def _fix_window_functions(sql: str, db_type: str) -> str:
    """
    Ensure every navigation/offset window function has ORDER BY inside its
    OVER(...) clause.

    Affected functions (all dialects require ORDER BY for these):
        LAG, LEAD, FIRST_VALUE, LAST_VALUE, NTH_VALUE, NTILE,
        ROW_NUMBER, RANK, DENSE_RANK, CUME_DIST, PERCENT_RANK

    SQL Server is strictest — it raises an error at parse time if ORDER BY is
    missing.  PostgreSQL, Oracle, BigQuery also require it.  This fixer runs
    for ALL dialects so the LLM's output is always valid.

    Aggregate window functions (SUM/AVG/COUNT/MIN/MAX ... OVER (...)) do NOT
    require ORDER BY and are left untouched.

    Strategy
    --------
    For each call `<fn>(args) OVER (...)`:
    1. If the OVER clause already contains ORDER BY → leave it alone.
    2. If PARTITION BY exists inside OVER → insert
           ORDER BY <partition_col>
       after the PARTITION BY clause (reuses the same column; valid for all
       dialects and gives stable but deterministic ordering).
    3. Otherwise insert ORDER BY (SELECT NULL) which is the universally
       accepted no-op order for SQL Server, Oracle, and other dialects when
       the calling code truly does not care about ordering (e.g. NTILE with
       only one partition).

    The function is written as a regex + paren-depth scanner so it handles
    nested function calls inside OVER() correctly.
    """
    # Functions that REQUIRE ORDER BY in their OVER clause
    _ORDER_REQUIRED = re.compile(
        r'\b(LAG|LEAD|FIRST_VALUE|LAST_VALUE|NTH_VALUE|NTILE'
        r'|ROW_NUMBER|RANK|DENSE_RANK|CUME_DIST|PERCENT_RANK)\s*\(',
        re.IGNORECASE,
    )
    if not _ORDER_REQUIRED.search(sql):
        return sql

    result = []
    pos = 0

    for fn_match in _ORDER_REQUIRED.finditer(sql):
        fn_name = fn_match.group(1)
        fn_start = fn_match.start()

        # Append everything up to start of this function call
        result.append(sql[pos:fn_start])

        # Find the matching close-paren for the function args  (depth-1 walk)
        args_open = fn_match.end() - 1   # position of '(' opening the args
        depth = 1
        i = args_open + 1
        while i < len(sql) and depth:
            if sql[i] == '(':
                depth += 1
            elif sql[i] == ')':
                depth -= 1
            i += 1
        args_close = i - 1   # position of matching ')'

        # Now look for OVER keyword immediately after the args close
        after_args = sql[args_close + 1:]
        over_m = re.match(r'\s*OVER\s*\(', after_args, re.IGNORECASE)
        if not over_m:
            # No OVER clause at all — leave function untouched
            result.append(sql[fn_start:args_close + 1])
            pos = args_close + 1
            continue

        # Find the matching close-paren of the OVER(...)
        over_open_abs = args_close + 1 + over_m.end() - 1  # abs pos of '('
        over_body_start = over_open_abs + 1
        depth = 1
        i = over_body_start
        while i < len(sql) and depth:
            if sql[i] == '(':
                depth += 1
            elif sql[i] == ')':
                depth -= 1
            i += 1
        over_close_abs = i - 1  # abs pos of matching ')' of OVER(...)

        over_body = sql[over_body_start:over_close_abs]

        # Does it already have ORDER BY?
        if re.search(r'\bORDER\s+BY\b', over_body, re.IGNORECASE):
            # Already correct — emit verbatim
            result.append(sql[fn_start:over_close_abs + 1])
            pos = over_close_abs + 1
            continue

        # Need to inject ORDER BY.  Prefer to reuse the PARTITION BY column.
        pb_match = re.search(
            r'\bPARTITION\s+BY\s+([\w\.\[\]`"]+)',
            over_body, re.IGNORECASE,
        )
        if pb_match:
            order_col = pb_match.group(1)
        else:
            # Safe no-op fallback understood by SQL Server, Oracle, BigQuery
            order_col = "(SELECT NULL)"

        new_over_body = over_body.rstrip() + f" ORDER BY {order_col}"
        fixed_fragment = (
            sql[fn_start:over_open_abs + 1]
            + new_over_body
            + ")"
        )
        result.append(fixed_fragment)
        pos = over_close_abs + 1
        logger.info(
            "plan_node: injected ORDER BY %s into %s() OVER() [dialect=%s]",
            order_col, fn_name, db_type,
        )

    result.append(sql[pos:])
    return "".join(result)


def _fix_multicolumn_subquery(sql: str) -> str:
    """
    Fix: "Only one expression can be specified in the select list when the
    subquery is not introduced with EXISTS."  (SQL Server, and ANSI SQL rule
    for all dialects.)

    Two patterns the LLM generates that trigger this:

    Pattern 1 — multi-column IN/NOT IN/= subquery
    ──────────────────────────────────────────────
    Scalar-context subqueries (IN, NOT IN, =, <>, <, >, ANY, ALL) must return
    exactly one column.  The LLM sometimes writes:

        WHERE period_id IN (SELECT period_id, rn FROM ranked_periods WHERE rn <= 12)
        WHERE col = (SELECT a, b FROM t WHERE ...)

    Fix: keep only the first SELECT expression inside the subquery.

    Pattern 2 — row-value constructor  (a, b) IN (SELECT x, y FROM …)
    ──────────────────────────────────────────────────────────────────
    SQL Server does not support row-value constructors in WHERE clauses.
    PostgreSQL and SQLite do, but BigQuery and Oracle do not support them
    either in all contexts.  The universal rewrite is EXISTS:

        WHERE (a, b) IN (SELECT x, y FROM t WHERE cond)
        →
        WHERE EXISTS (SELECT 1 FROM t WHERE x = a AND y = b AND cond)

    The rewrite maps outer tuple columns to subquery SELECT columns by
    position, then merges any existing WHERE condition with AND.

    This function runs for ALL dialects (same ANSI rule; SQL Server is just
    strictest about the error message).
    """
    if not re.search(r'\bIN\s*\(', sql, re.IGNORECASE):
        return sql

    def _find_matching_close(s: str, open_pos: int) -> int:
        """Return the position of the ')' that matches '(' at open_pos."""
        depth = 1
        i = open_pos + 1
        in_str = False
        esc = False
        while i < len(s) and depth:
            c = s[i]
            if esc:
                esc = False
            elif c == '\\' and in_str:
                esc = True
            elif c == "'":
                in_str = not in_str
            elif not in_str:
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
            i += 1
        return i - 1  # position of matching ')'

    # ── Pattern 2 runs FIRST: row-value constructor ──────────────────────────
    #   (a, b, …) IN (SELECT x, y, … FROM t …)
    # Must run before Pattern 1 below. Pattern 1's regex matches "IN (SELECT"
    # regardless of what precedes the IN, so on a tuple LHS like
    # "(o.customer_id, o.region) IN (SELECT customer_id, region FROM …)" it
    # used to fire first and truncate the subquery to its first column —
    # which then made Pattern 2's `len(sub_cols) != len(outer_cols)` guard
    # fail, so the EXISTS rewrite could never trigger for the exact
    # multi-column case it exists to handle, leaving a tuple compared against
    # a now-mismatched single-column subquery (invalid SQL, worse than the
    # unfixed input). Running the more specific row-value-constructor pattern
    # first consumes those spans as EXISTS(...) before Pattern 1 ever sees an
    # "IN (SELECT" there.
    # Match:  \( col_list \) \s* [NOT\s+]? IN \s* \( SELECT …
    _ROW_CTOR_RE = re.compile(
        r'\(\s*([\w\.\[\]`"]+(?:\s*,\s*[\w\.\[\]`"]+)+)\s*\)\s*(NOT\s+)?IN\s*(\(\s*SELECT\b)',
        re.IGNORECASE,
    )
    pieces = []
    pos = 0
    for m in _ROW_CTOR_RE.finditer(sql):
        outer_cols_str = m.group(1)
        negated = bool(m.group(2))
        paren_open = m.start(3) + m.group(3).index('(')
        paren_close = _find_matching_close(sql, paren_open)
        sub_body = sql[paren_open + 1: paren_close]

        outer_cols = [c.strip() for c in outer_cols_str.split(',')]

        # Parse subquery: SELECT list + everything after SELECT list
        sel_m = re.match(r'\s*SELECT\s+(?:DISTINCT\s+)?', sub_body, re.IGNORECASE)
        if not sel_m:
            continue

        after_select = sub_body[sel_m.end():]
        from_pos = None
        depth = 0
        for i, ch in enumerate(after_select):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif depth == 0 and re.match(r'\bFROM\b', after_select[i:], re.IGNORECASE):
                from_pos = i
                break

        if from_pos is None:
            continue

        select_list_text = after_select[:from_pos]
        sub_cols = [c.strip() for c in _split_select_list(select_list_text)]
        from_onwards = after_select[from_pos:]  # "FROM t [WHERE …] [GROUP BY …] …"

        if len(sub_cols) != len(outer_cols):
            # Column count mismatch — can't safely rewrite; skip
            continue

        # Build the correlated WHERE conditions: sub_col = outer_col
        join_conds = " AND ".join(
            f"{sc} = {oc}" for sc, oc in zip(sub_cols, outer_cols)
        )

        # Merge with existing WHERE inside the subquery (if any)
        where_m = re.match(
            r'(.*?\bFROM\b\s+[\w\.\[\]`"]+(?:\s+(?:AS\s+)?\w+)?)\s+WHERE\b\s*(.*)',
            from_onwards, re.IGNORECASE | re.DOTALL,
        )
        if where_m:
            from_part = where_m.group(1)
            existing_where = where_m.group(2).strip()
            # Anything after the WHERE that is at depth 0 up to GROUP BY / ORDER BY / end
            exists_body = f"SELECT 1 {from_part} WHERE {join_conds} AND ({existing_where})"
        else:
            exists_body = f"SELECT 1 {from_onwards} WHERE {join_conds}"

        exists_expr = ("NOT EXISTS" if negated else "EXISTS") + f" ({exists_body})"

        # Replace the whole  (outer_cols) [NOT] IN (subquery)  span
        span_start = m.start()
        pieces.append(sql[pos:span_start])
        pieces.append(exists_expr)
        pos = paren_close + 1
        logger.info(
            "plan_node: rewrote row-value constructor (%s) %sIN → %s",
            outer_cols_str, "NOT " if negated else "", exists_expr[:80],
        )

    if pieces:
        pieces.append(sql[pos:])
        sql = "".join(pieces)

    # ── Pattern 1 runs SECOND: scalar IN / NOT IN / = subquery with multiple
    # columns ──────────────────────────────────────────────────────────────
    # Match:  [NOT ]IN\s*( SELECT …
    # Also:   = \s*( SELECT …    >= ( SELECT …   etc.
    # By this point Pattern 2 has already consumed every row-value-constructor
    # tuple IN (SELECT ...) as EXISTS(...), so this only ever sees genuine
    # scalar-context subqueries (col IN (SELECT ...), col = (SELECT ...)).
    _SCALAR_SUB_RE = re.compile(
        r'\b((?:NOT\s+)?IN|=|<>|!=|>=|<=|>(?!=)|<(?!=))\s*(\(\s*SELECT\b)',
        re.IGNORECASE,
    )
    pieces = []
    pos = 0
    for m in _SCALAR_SUB_RE.finditer(sql):
        paren_open = m.start(2) + m.group(2).index('(')
        paren_close = _find_matching_close(sql, paren_open)
        sub_body = sql[paren_open + 1: paren_close]  # content inside the (…)

        # Extract the SELECT list of the inner query
        sel_m = re.match(r'\s*SELECT\s+', sub_body, re.IGNORECASE)
        if not sel_m:
            continue

        # Find the FROM keyword at depth 0 to isolate the SELECT list
        after_select = sub_body[sel_m.end():]
        from_pos = None
        depth = 0
        for i, ch in enumerate(after_select):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif depth == 0 and re.match(r'\bFROM\b', after_select[i:], re.IGNORECASE):
                from_pos = i
                break

        if from_pos is None:
            continue  # can't parse — leave alone

        select_list_text = after_select[:from_pos]
        select_items = _split_select_list(select_list_text)

        # Only fix when there are multiple columns
        if len(select_items) <= 1:
            continue

        # Rebuild: keep only the first expression
        first_col = select_items[0].strip()
        rest_of_sub = after_select[from_pos:]  # FROM … onwards
        new_sub = f"(SELECT {first_col} {rest_of_sub})"
        pieces.append(sql[pos: m.start(2)])
        pieces.append(new_sub)
        pos = paren_close + 1
        logger.info(
            "plan_node: stripped extra SELECT columns from scalar subquery "
            "(kept '%s', dropped %d col(s))",
            first_col, len(select_items) - 1,
        )

    if pieces:
        pieces.append(sql[pos:])
        sql = "".join(pieces)

    return sql


# A single identifier token in any of the quoting styles used across the
# dialects this planner targets:
#   bare        col
#   PostgreSQL / Snowflake / Oracle / BigQuery standard-SQL   "col"
#   SQL Server                                                [col]
#   BigQuery legacy / MySQL                                   `col`
_IDENT_SRC = r'"[^"]+"|\[[^\]]+\]|`[^`]+`|\w+'
# No leading \b: a quote/bracket/backtick is a non-word char, so \b would
# fail to match right after whitespace (both sides non-word) and silently
# skip every quoted identifier — exactly the dialects that need this most
# (Postgres/Snowflake "col", SQL Server [col], BigQuery/MySQL `col`).
_QUALIFIED_IDENT_RE = re.compile(rf'({_IDENT_SRC})\.({_IDENT_SRC})')
_ALIAS_CAPTURE_RE = re.compile(rf'\bAS\s+({_IDENT_SRC})\s*$', re.IGNORECASE)


def _strip_ident_quotes(token: str) -> str:
    """Strip whichever quoting style wraps an identifier, for comparison only."""
    if len(token) >= 2 and token[0] == '"' and token[-1] == '"':
        return token[1:-1]
    if len(token) >= 2 and token[0] == '[' and token[-1] == ']':
        return token[1:-1]
    if len(token) >= 2 and token[0] == '`' and token[-1] == '`':
        return token[1:-1]
    return token


def _fix_order_by_alias_qualifier(sql: str) -> str:
    """
    Fix: LLM qualifies a computed SELECT alias with a table prefix inside
    ORDER BY, e.g. "... AS years_in_position ... ORDER BY jc.years_in_position".

    "years_in_position" is not a real column on table jc — it only exists as
    an output alias of an EXTRACT(...)/computed expression, so referencing it
    as "jc.years_in_position" is invalid SQL in every dialect (the table has
    no such column). This is independent of SELECT DISTINCT and must be fixed
    even when DISTINCT is absent.

    Detects each "<table_alias>.<name>" term in the outermost ORDER BY clause
    where <name> (case-insensitively, quoting-insensitively) matches a
    "... AS <name>" alias in the SELECT list, and rewrites the ORDER BY
    reference to the bare alias. Identifier quoting is dialect-agnostic:
    unquoted, "double-quoted" (Postgres/Snowflake/Oracle/BigQuery standard
    SQL), [bracketed] (SQL Server), and `backticked` (BigQuery legacy/MySQL)
    are all recognized on either side of the dot.
    """
    sel_match = re.search(r'\bSELECT\b', sql, re.IGNORECASE)
    if not sel_match:
        return sql

    from_match = None
    for m in re.finditer(r'\bFROM\b', sql, re.IGNORECASE):
        prefix = sql[:m.start()]
        if prefix.count('(') - prefix.count(')') == 0 and m.start() > sel_match.end():
            from_match = m
            break
    if not from_match:
        return sql

    select_list_text = sql[sel_match.end():from_match.start()]
    select_list_text = re.sub(r'^\s*DISTINCT\b', '', select_list_text, flags=re.IGNORECASE)
    select_items = _split_select_list(select_list_text)

    aliases: set = set()
    for item in select_items:
        alias_m = _ALIAS_CAPTURE_RE.search(item)
        if alias_m:
            aliases.add(_strip_ident_quotes(alias_m.group(1)).lower())
    if not aliases:
        return sql

    ob_match = None
    for m in re.finditer(r'\bORDER\s+BY\b', sql, re.IGNORECASE):
        prefix = sql[:m.start()]
        if prefix.count('(') - prefix.count(')') == 0:
            ob_match = m
    if not ob_match:
        return sql

    limiter_match = re.search(
        r'\b(LIMIT\s+\d+|OFFSET\s+\d+|FETCH\s+FIRST\b|ROWNUM\s*<=?\s*\d+)\b',
        sql[ob_match.end():], re.IGNORECASE,
    )
    ob_end = ob_match.end() + limiter_match.start() if limiter_match else len(sql)
    order_clause = sql[ob_match.end():ob_end]

    def _strip_qualifier(m: re.Match) -> str:
        table_tok, col_tok = m.group(1), m.group(2)
        if _strip_ident_quotes(col_tok).lower() in aliases:
            logger.info(
                "plan_node: stripped invalid table qualifier '%s.' from ORDER BY alias '%s'",
                table_tok, col_tok,
            )
            return col_tok
        return m.group(0)

    fixed_clause = _QUALIFIED_IDENT_RE.sub(_strip_qualifier, order_clause)
    if fixed_clause == order_clause:
        return sql

    return sql[:ob_match.end()] + fixed_clause + sql[ob_end:]


def _fix_distinct_order_by(sql: str) -> str:
    """
    Fix: "for SELECT DISTINCT, ORDER BY expressions must appear in select list"

    Universal SQL rule (PostgreSQL, SQLite, SQL Server, Oracle, BigQuery):
    when SELECT DISTINCT is used every column in ORDER BY must also appear in
    the SELECT list.

    The LLM sometimes generates ORDER BY on a column it did not select.
    This function detects that and adds the missing columns to the SELECT list.

    The SELECT list is parsed with a paren-aware splitter so expressions like
    COALESCE(a, b), ROUND(x, 2), CASE WHEN ... END do not confuse the comma
    split.
    """
    if not re.search(r'\bSELECT\s+DISTINCT\b', sql, re.IGNORECASE):
        return sql

    # ── locate outermost ORDER BY ─────────────────────────────────────────
    ob_match = None
    for m in re.finditer(r'\bORDER\s+BY\b', sql, re.IGNORECASE):
        prefix = sql[:m.start()]
        if prefix.count('(') - prefix.count(')') == 0:
            ob_match = m
    if not ob_match:
        return sql

    order_clause = sql[ob_match.end():]

    # Truncate at a trailing row-limiting clause (LIMIT/OFFSET/FETCH FIRST/
    # ROWNUM) so it doesn't get glued onto the last ORDER BY term. Without
    # this, "ORDER BY a DESC, b DESC LIMIT 50" splits its last term as
    # "b DESC LIMIT 50" — the DESC/ASC-stripping regex only matches at the
    # end of the term, so it fails to strip DESC here, the whole garbled
    # fragment is then treated as "missing" from the SELECT list and spliced
    # in verbatim, and a later LIMIT-stripping pass (for COUNT(*) companion
    # queries) removes the embedded "LIMIT 50" but leaves the "DESC" behind.
    limiter_match = re.search(
        r'\b(LIMIT\s+\d+|OFFSET\s+\d+|FETCH\s+FIRST\b|ROWNUM\s*<=?\s*\d+)\b',
        order_clause, re.IGNORECASE,
    )
    if limiter_match:
        order_clause = order_clause[:limiter_match.start()]

    order_terms = _split_order_by_terms(order_clause)
    if not order_terms:
        return sql

    # ── locate SELECT list (between SELECT DISTINCT and first outer FROM) ─
    sel_match = re.search(r'\bSELECT\s+DISTINCT\b', sql, re.IGNORECASE)
    from_match = None
    for m in re.finditer(r'\bFROM\b', sql, re.IGNORECASE):
        prefix = sql[:m.start()]
        if prefix.count('(') - prefix.count(')') == 0 and m.start() > sel_match.end():
            from_match = m
            break
    if not from_match:
        return sql

    select_list_text = sql[sel_match.end():from_match.start()]
    select_items = _split_select_list(select_list_text)

    # Build normalised set: aliases, bare column names, full expressions.
    # Table-qualifier stripping is quoting-agnostic so it works the same for
    # bare, "double-quoted", [bracketed], and `backticked` identifiers across
    # PostgreSQL / Snowflake / SQL Server / Oracle / BigQuery / MySQL.
    selected_exprs: set = set()
    for item in select_items:
        # alias: "expr AS alias"
        alias_m = _ALIAS_CAPTURE_RE.search(item)
        if alias_m:
            selected_exprs.add(_strip_ident_quotes(alias_m.group(1)).lower())
        # bare column (strip table qualifier, if any)
        qualified_m = re.fullmatch(rf'\s*({_IDENT_SRC})\.({_IDENT_SRC})\s*', item)
        bare = _strip_ident_quotes(qualified_m.group(2)).lower() if qualified_m else item.strip().lower()
        selected_exprs.add(bare)
        # full expression as-is
        selected_exprs.add(item.strip().lower())

    # ── find ORDER BY terms absent from SELECT list ───────────────────────
    missing = []
    for term in order_terms:
        t_lower = term.lower()
        qualified_t = re.fullmatch(rf'\s*({_IDENT_SRC})\.({_IDENT_SRC})\s*', term)
        bare_t = (
            _strip_ident_quotes(qualified_t.group(2)).lower() if qualified_t
            else _strip_ident_quotes(term.strip()).lower()
        )
        if t_lower not in selected_exprs and bare_t not in selected_exprs:
            missing.append(term)

    if not missing:
        return sql

    # ── patch: insert missing columns just before FROM ────────────────────
    insert_pos = from_match.start()
    additions = ', '.join(missing)
    patched = sql[:insert_pos] + ', ' + additions + ' ' + sql[insert_pos:]
    logger.info(
        "plan_node: added %d column(s) to SELECT list to satisfy DISTINCT+ORDER BY: %s",
        len(missing), missing,
    )
    return patched


def _fix_dialect_syntax(sql: str, db_type: str) -> str:
    """
    Runtime cross-dialect contamination fixer.

    Even when the LLM is instructed to write dialect-correct SQL it sometimes
    bleeds PostgreSQL/ANSI idioms into SQL Server / Oracle / BigQuery queries.
    This function catches the most common mis-fires and rewrites them to the
    target dialect *before* execution.

    Covered transformations per dialect:
      SQL Server:
        • ILIKE  → LOWER(col) LIKE LOWER(pat)
        • ::type → CAST(col AS type)
        • col || other → col + other  (string concat — very conservative)
        • NOW()  → GETDATE()
        • CURRENT_DATE → CAST(GETDATE() AS DATE)
        • CURRENT_TIMESTAMP → GETDATE()
        • DATE_TRUNC('unit', col) → DATEADD/CAST equivalent
        • LENGTH(col) → LEN(col)
        • JOIN … USING (col) → JOIN … ON t1.col = t2.col  (best-effort)
        • LIMIT N  → already handled by _enforce_sql_limits; skip

      Oracle:
        • ILIKE  → LOWER(col) LIKE LOWER(pat)
        • ::type → CAST(col AS type)
        • NOW()  → SYSDATE
        • CURRENT_TIMESTAMP → SYSTIMESTAMP
        • CURRENT_DATE → SYSDATE
        • LIMIT N → FETCH FIRST N ROWS ONLY  (outer query only)
        • LENGTH(col) — Oracle already has LENGTH; no-op

      BigQuery:
        • ILIKE  → LOWER(col) LIKE LOWER(pat)
        • ::type → CAST(col AS type)
        • NOW()  → CURRENT_TIMESTAMP()
        • CURRENT_DATE → CURRENT_DATE()
        • LIMIT stays (BigQuery supports LIMIT)

      SQLite:
        • ILIKE  → LOWER(col) LIKE LOWER(pat)
        • ::type → CAST(col AS type)
        • NOW()  → DATE('now')
        • CURRENT_TIMESTAMP — SQLite keyword; keep

    Transformations that require deep parse-trees (e.g. rewriting every
    JOIN … USING to ON) are approximated conservatively and logged when they
    fire so developers can audit edge cases.
    """
    if not sql:
        return sql

    dtype = db_type.lower()

    # ── 1. ILIKE → LOWER(col) LIKE LOWER(pat) ────────────────────────────────
    # ILIKE is PostgreSQL-only; everything else must use LOWER().
    # Pattern: <expr> ILIKE <pat>  where pat may be a quoted string or param.
    if dtype not in ("postgresql", "postgres", "redshift"):
        def _ilike_replace(m: re.Match) -> str:
            col = m.group(1).strip()
            pat = m.group(2).strip()
            result = f"LOWER({col}) LIKE LOWER({pat})"
            logger.info("plan_node: rewrote ILIKE → LOWER LIKE for %s", dtype)
            return result

        sql = re.sub(
            r'(\b[\w.]+\b(?:\s*\([^)]*\))?)\s+ILIKE\s+(\S+)',
            _ilike_replace,
            sql,
            flags=re.IGNORECASE,
        )

    # ── 2. PostgreSQL :: casting → CAST(col AS type) ─────────────────────────
    if dtype not in ("postgresql", "postgres", "redshift"):
        def _cast_replace(m: re.Match) -> str:
            expr = m.group(1).strip()
            typ  = m.group(2).strip()
            result = f"CAST({expr} AS {typ})"
            logger.info("plan_node: rewrote ::%s → CAST for %s", typ, dtype)
            return result

        # match: word_or_paren_expr::TYPE  e.g. col::DATE, (expr)::INTEGER
        sql = re.sub(
            r'(\w+|(?:\([^)]+\)))\s*::\s*([A-Z_][A-Z0-9_]*(?:\s*\(\d+(?:,\d+)?\))?)',
            _cast_replace,
            sql,
            flags=re.IGNORECASE,
        )

    # ── 3. Date / time functions ──────────────────────────────────────────────
    if dtype in ("sqlserver", "mssql", "sql server"):
        sql = re.sub(r'\bNOW\s*\(\s*\)', 'GETDATE()', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bCURRENT_TIMESTAMP\b(?!\s*\()', 'GETDATE()', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bCURRENT_DATE\b', 'CAST(GETDATE() AS DATE)', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bCURRENT_TIME\b', 'CAST(GETDATE() AS TIME)', sql, flags=re.IGNORECASE)

    elif dtype == "oracle":
        sql = re.sub(r'\bNOW\s*\(\s*\)', 'SYSDATE', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bCURRENT_TIMESTAMP\b(?!\s*\()', 'SYSTIMESTAMP', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bCURRENT_DATE\b', 'SYSDATE', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bCURRENT_TIME\b', 'SYSDATE', sql, flags=re.IGNORECASE)

    elif dtype in ("bigquery",):
        sql = re.sub(r'\bNOW\s*\(\s*\)', 'CURRENT_TIMESTAMP()', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bCURRENT_TIMESTAMP\b(?!\s*\()', 'CURRENT_TIMESTAMP()', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bCURRENT_DATE\b(?!\s*\()', 'CURRENT_DATE()', sql, flags=re.IGNORECASE)

    elif dtype == "sqlite":
        sql = re.sub(r'\bNOW\s*\(\s*\)', "DATE('now')", sql, flags=re.IGNORECASE)
        sql = re.sub(r'\bCURRENT_DATE\b', "DATE('now')", sql, flags=re.IGNORECASE)

    # ── 4. DATE_TRUNC → dialect equivalent ───────────────────────────────────
    # DATE_TRUNC('unit', col) is PostgreSQL / BigQuery (BigQuery uses DATE_TRUNC(col, unit))
    # SQL Server: convert truncation to DATEADD + DATEDIFF idiom
    if dtype in ("sqlserver", "mssql", "sql server"):
        _DT_MAP = {
            'year':    ('year',  'year'),
            'quarter': ('quarter', 'quarter'),
            'month':   ('month', 'month'),
            'week':    ('week',  'week'),
            'day':     ('day',   'day'),
            'hour':    ('hour',  'hour'),
            'minute':  ('minute','minute'),
        }

        def _datetrunc_sqlserver(m: re.Match) -> str:
            unit = m.group(1).strip().strip("'\"").lower()
            col  = m.group(2).strip()
            dp   = _DT_MAP.get(unit)
            if dp:
                result = f"DATEADD({dp[0]}, DATEDIFF({dp[1]}, 0, {col}), 0)"
                logger.info("plan_node: rewrote DATE_TRUNC('%s', ...) → DATEADD/DATEDIFF for SQL Server", unit)
                return result
            return m.group(0)  # unknown unit — leave as-is

        sql = re.sub(
            r'\bDATE_TRUNC\s*\(\s*([\'"]?\w+[\'"]?)\s*,\s*([^)]+)\)',
            _datetrunc_sqlserver,
            sql,
            flags=re.IGNORECASE,
        )

    elif dtype == "oracle":
        # Oracle uses TRUNC(date, 'unit')
        def _datetrunc_oracle(m: re.Match) -> str:
            unit = m.group(1).strip().strip("'\"").lower()
            col  = m.group(2).strip()
            _ORACLE_UNITS = {
                'year': 'YYYY', 'quarter': 'Q', 'month': 'MM',
                'week': 'IW',   'day': 'DD',
            }
            oracle_fmt = _ORACLE_UNITS.get(unit, unit.upper())
            logger.info("plan_node: rewrote DATE_TRUNC('%s', ...) → TRUNC for Oracle", unit)
            return f"TRUNC({col}, '{oracle_fmt}')"

        sql = re.sub(
            r'\bDATE_TRUNC\s*\(\s*([\'"]?\w+[\'"]?)\s*,\s*([^)]+)\)',
            _datetrunc_oracle,
            sql,
            flags=re.IGNORECASE,
        )

    # ── 5. LENGTH → LEN for SQL Server ───────────────────────────────────────
    if dtype in ("sqlserver", "mssql", "sql server"):
        # Only rewrite standalone LENGTH( — not CHAR_LENGTH or BIT_LENGTH
        sql = re.sub(r'(?<![A-Z_])LENGTH\s*\(', 'LEN(', sql, flags=re.IGNORECASE)

    # ── 5b. Redshift-specific rewrites ───────────────────────────────────────
    if dtype == "redshift":
        # STRING_AGG(col, sep ORDER BY ...) → LISTAGG(col, sep) WITHIN GROUP (ORDER BY ...)
        # Handles: STRING_AGG(col, 'sep' ORDER BY sort_col)
        def _string_agg_to_listagg(m: re.Match) -> str:
            col  = m.group(1).strip()
            sep  = m.group(2).strip()
            rest = m.group(3).strip()  # may be empty or "ORDER BY ..."
            if re.match(r'ORDER\s+BY\b', rest, re.IGNORECASE):
                result = f"LISTAGG({col}, {sep}) WITHIN GROUP ({rest})"
            else:
                order_part = f"ORDER BY {rest}" if rest else "ORDER BY 1"
                result = f"LISTAGG({col}, {sep}) WITHIN GROUP ({order_part})"
            logger.info("plan_node: rewrote STRING_AGG → LISTAGG for Redshift")
            return result

        sql = re.sub(
            r'\bSTRING_AGG\s*\(\s*([^,]+?)\s*,\s*([\'"][^\'\"]*[\'"])\s*(?:,\s*|\s+)?((?:ORDER\s+BY\s+[^)]+)?)\s*\)',
            _string_agg_to_listagg,
            sql,
            flags=re.IGNORECASE,
        )

        # array_agg(col ORDER BY ...) → array_agg(col)  — strip unsupported ORDER BY
        def _array_agg_strip_order(m: re.Match) -> str:
            col = m.group(1).strip()
            logger.info("plan_node: stripped ORDER BY from array_agg() for Redshift")
            return f"array_agg({col})"

        sql = re.sub(
            r'\barray_agg\s*\(\s*([^)]+?)\s+ORDER\s+BY\s+[^)]+\)',
            _array_agg_strip_order,
            sql,
            flags=re.IGNORECASE,
        )

    # ── 6. LIMIT N at top-level for Oracle → FETCH FIRST N ROWS ONLY ─────────
    # _enforce_sql_limits already handles this, but if the LLM re-added a LIMIT
    # after the enforcer ran we catch it here as a safety net.
    if dtype == "oracle":
        # Only replace outermost LIMIT (not inside parens)
        # Simple heuristic: replace LIMIT N that appears at end of statement
        parts = re.split(r'\bLIMIT\s+(\d+)\s*$', sql.rstrip('; '), flags=re.IGNORECASE)
        if len(parts) == 3:
            logger.info("plan_node: rewrote LIMIT %s → FETCH FIRST ROWS ONLY for Oracle", parts[1])
            sql = parts[0] + f"\nFETCH FIRST {parts[1]} ROWS ONLY"

    # ── 7. || string concat → + for SQL Server ───────────────────────────────
    # Very conservative: only replace when both sides are string literals or
    # simple column refs — avoids breaking boolean OR in other contexts.
    # Note: SQL Server uses + for string concat; || is ANSI but not supported.
    if dtype in ("sqlserver", "mssql", "sql server"):
        # Only rewrite string-context || (surrounded by quoted strings or words)
        def _concat_replace(m: re.Match) -> str:
            logger.info("plan_node: rewrote || → + for SQL Server")
            return m.group(1) + ' + ' + m.group(2)

        sql = re.sub(
            r"(['\w)]\s*)\|\|(\s*['\w(])",
            _concat_replace,
            sql,
        )

    return sql


def _is_raw_row_query(sql: str) -> bool:
    """Return True when the SQL has no aggregation (GROUP BY / COUNT / SUM …)."""
    return not bool(_AGG_PATTERN.search(sql))


def _strip_row_limit(sql: str) -> str:
    """
    Remove any LIMIT / TOP N / FETCH FIRST … ROWS ONLY / ROWNUM clause so the
    SQL can be wrapped in a COUNT subquery that reflects the *total* matching
    rows, not just the sampled page.
    """
    cleaned = _LIMIT_PATTERN.sub("", sql)
    cleaned = re.sub(
        r'(\bSELECT\b\s+)TOP\s+\d+\s+', r'\1', cleaned, flags=re.IGNORECASE
    )
    cleaned = _FETCH_FIRST_PATTERN.sub("", cleaned)
    cleaned = _ROWNUM_PATTERN.sub("1=1", cleaned)
    return cleaned.strip().rstrip(";").strip()


def _build_count_companion(
    raw_sql: str,
    original_query_id: str,
    table_refs: list,
    db_type: str,
) -> dict:
    """
    Wrap a raw-row SQL query in a COUNT subquery so the user learns the total
    number of matching rows (not just the size of the sampled page).

    Returns a plain dict ready to become a SQLQuery.

    Works across all supported dialects:
      PostgreSQL / SQLite / Redshift / BigQuery: SELECT COUNT(*) FROM (...) AS _c
      SQL Server  : same (subquery alias required)
      Oracle      : SELECT COUNT(*) FROM (...) _c   (no AS keyword for aliases)
    """
    base_sql = _strip_row_limit(raw_sql)
    alias    = "_total_count"

    if db_type.lower() == "oracle":
        # Oracle does not use AS for table aliases in FROM
        count_sql = f"SELECT COUNT(*) AS total_matching_rows FROM (\n  {base_sql}\n) {alias}"
    else:
        count_sql = f"SELECT COUNT(*) AS total_matching_rows FROM (\n  {base_sql}\n) AS {alias}"

    return {
        "query_id":    f"{original_query_id}_count",
        "description": f"Total number of rows matching the query (full dataset, no row limit)",
        "sql":         count_sql,
        "table_refs":  table_refs,
        "kg_id":       "",
    }


def plan_node(state: DialogState) -> DialogState:
    """Decompose the NQL into SQL queries via LLM."""
    logger.info("=== plan_node ===")

    config         = state["config"]
    natural_query  = state.get("natural_query", "").strip()
    schema_context = state.get("schema_context", "(no schema)")

    if not natural_query:
        state["errors"].append("plan_node: natural_query is empty")
        state["sql_queries"] = []
        state["phase"] = "plan"
        return state

    # ── Ambiguous proper-noun disambiguation ───────────────────────────────
    # resolve_node found 2+ equally-close candidate stored values for a
    # proper-noun term in the question (see resolve_node._detect_ambiguous_terms)
    # — e.g. "Smith" could be "John Smith" or "Jane Smith". Ask the user to
    # pick one instead of guessing and running SQL against a possibly-wrong
    # entity. Reuses the same plan_explanation/no-SQL mechanism as the
    # meta-question path below. Empty (the default) is a complete no-op.
    clarification_needed = state.get("clarification_needed") or []
    if clarification_needed:
        lines = [
            "I found more than one close match and want to make sure I pick the right one:",
            "",
        ]
        for c in clarification_needed:
            candidates_str = ", ".join(f'"{v}"' for v in c["candidates"])
            lines.append(f'- For "{c["term"]}", did you mean one of: {candidates_str}?')
        lines.append("")
        lines.append("Please reply with the one you meant and I'll run the query.")
        logger.info(
            "plan_node: %d ambiguous term(s) — asking user to disambiguate instead of guessing",
            len(clarification_needed),
        )
        state["plan_explanation"] = "\n".join(lines)
        state["sql_queries"] = []
        state["phase"] = "plan"
        return state

    # ── Meta-question detection: discovery / capability questions ─────────────
    # "What insights can I generate?", "What can I analyse?", "What does this
    # data tell me?", "What questions can I ask?" etc. have no SQL answer.
    # Instead of returning [] and a generic error, answer using the schema context.
    _META_PATTERNS = re.compile(
        r"\b(what\s+(all\s+)?(insights?|analysis|analyses|questions?|metrics?|kpis?|"
        r"reports?|stories?|trends?|patterns?|can\s+(?:i|we|be)\s+(?:analyz|generat|explor|ask|answ))"
        r"|what\s+(?:can|could)\s+(?:i|we|this\s+data)\s+(?:tell|show|analyz|generat|explor)"
        r"|how\s+(?:can|could)\s+(?:i|we)\s+(?:use|analyz|explor)\s+this"
        r"|tell\s+me\s+(?:about\s+)?(?:this\s+data(?:set|model)?|what\s+insights?)"
        r"|(?:explore|discover|understand)\s+(?:this\s+)?(?:data(?:set|model)?|schema)"
        r"|what\s+is\s+(?:in|available\s+in)\s+(?:this\s+)?(?:data(?:set|model)?|schema))\b",
        re.IGNORECASE,
    )

    if _META_PATTERNS.search(natural_query):
        logger.info("plan_node: meta/discovery question detected — answering from schema context")
        schema_context_for_meta = state.get("schema_context", "(no schema)")
        domain  = state.get("domain", "") or ""
        role    = getattr(state.get("config"), "analyst_role", "") or ""

        _META_SYSTEM = (
            "You are an expert data analyst. The user is asking what insights or analyses "
            "are possible from a given data model. Based ONLY on the schema context provided "
            "(table names, columns, data types, sample values, and relationships), enumerate "
            "the specific, actionable insights and analyses that can be generated.\n\n"
            "Structure your response as:\n"
            "1. A one-sentence summary of what this dataset is about.\n"
            "2. 5–8 specific insight areas (use bold headers), each with 2–3 example questions "
            "a business user could ask. Ground every example in the actual column/table names "
            "visible in the schema — do not invent columns.\n"
            "3. A brief note on any notable metrics, KPIs, or derived calculations the schema "
            "supports (e.g. growth rates, share, index, ratio).\n\n"
            "Keep the language business-friendly — no SQL, no technical jargon."
        )

        domain_line = f"Domain: {domain}\n" if domain else ""
        role_line   = f"Analyst role context: {role}\n" if role else ""
        user_msg    = (
            f"{domain_line}{role_line}"
            f"Question: {natural_query}\n\n"
            f"Schema context:\n{schema_context_for_meta}"
        )

        try:
            from llm_client import get_client as _get_client
            _client = _get_client()
            _model  = getattr(config, "plan_llm_model", "claude-haiku-4-5")
            _resp   = _client.messages.create(
                model=_model,
                max_tokens=1024,
                system=_META_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            meta_answer = _resp.content[0].text.strip()
        except Exception as _e:
            logger.warning("plan_node: meta-question LLM call failed — %s", _e)
            meta_answer = (
                "Based on the available schema, this dataset supports analyses across "
                "the following areas. Please ask a specific question to get started — "
                "for example: trends over time, top/bottom performers, period-over-period "
                "comparisons, distribution by dimension, or contribution to totals."
            )

        state["plan_explanation"] = meta_answer
        state["sql_queries"]      = []
        state["phase"]            = "plan"
        return state

    # File-based sources (SQLite / CSV / Excel) load into in-memory SQLite with no schema
    # prefix.  Force empty schema so the LLM and the safety-net qualify step both use
    # bare table names.
    _FILE_BASED_TYPES = {"sqlite", "csv", "excel"}
    db_schema = "" if config.db_type.lower() in _FILE_BASED_TYPES else (config.db_schema or "")

    # Extract table labels from KG nodes for the SQL post-processor.
    # Exclude XSD data types and SQL/SPARQL keywords that must never be
    # treated as table names — these can appear as KG node labels when an
    # ontology generator incorrectly marks xsd:string etc. as owl:Class.
    _EXCLUDED_LABELS = {
        "string", "integer", "int", "float", "double", "decimal", "boolean",
        "date", "datetime", "time", "duration", "anyuri", "literal",
        "long", "short", "byte", "binary", "hexbinary", "base64binary",
        "nonnegativeinteger", "positiveinteger", "negativinteger",
        "unsignedlong", "unsignedint", "unsignedshort", "unsignedbyte",
        # SQL reserved words that must not be table-qualified
        "select", "from", "where", "join", "on", "group", "order", "by",
        "having", "limit", "offset", "as", "and", "or", "not", "null",
        "true", "false", "case", "when", "then", "else", "end", "in",
        "between", "like", "is", "distinct", "all", "any", "exists",
        "union", "intersect", "except", "with", "values", "set",
    }
    kg_nodes     = state.get("kg_nodes") or []
    _is_file_based = config.db_type.lower() in _FILE_BASED_TYPES

    def _sql_table(name: str) -> str:
        """Sanitize a table name — must match understand_node._to_sql_table."""
        import re as _re
        s = _re.sub(r"[^A-Za-z0-9_]", "_", str(name))
        return ("t_" + s if s and s[0].isdigit() else s) or "tbl"

    table_labels = [
        (_sql_table(n["label"]) if _is_file_based else n["label"])
        for n in kg_nodes
        if n.get("label") and n.get("label", "").lower() not in _EXCLUDED_LABELS
    ]

    # Inject grain annotations before the LLM sees the schema context.
    # This deterministically flags any table with sub-join-key granularity
    # (e.g. price_index recorded at channel grain, not brand-pack-period grain)
    # so the LLM knows it MUST pre-aggregate that table before joining.
    schema_context = _annotate_schema_grain(schema_context)

    # Build a set of all valid column names from the schema context so we can
    # reject any SQL the LLM generates using hallucinated column names.
    known_columns = _extract_known_columns(schema_context)
    # Add table labels as valid identifiers (they can appear bare in SQL too)
    known_columns.update(t.lower() for t in table_labels)
    # Cross-table check needs the un-augmented set: a dotted-literal prefix
    # like "evt" is not a genuine standalone column on ANY table, so if it
    # were treated as "known", _find_cross_table_columns would misfire on
    # every table alias that doesn't happen to own that exact dotted literal
    # (see the note on the augmented known_columns below for why this
    # exists at all, and why the two checks need different column sets).
    known_columns_base = set(known_columns)
    # Add the individual prefix/tail tokens of every literal dotted column
    # name (e.g. "evt.event_key" -> "evt", "event_key"). The LLM sometimes
    # mis-splits a single literal dotted identifier into alias.token or
    # token.token, which would otherwise look like a hallucinated column and
    # get the whole query dropped here — before it ever reaches execute_node's
    # _snowflake_quote_dotted_columns self-heal, which is designed to repair
    # exactly this mistake. Treating both halves as known lets those queries
    # through so the runtime self-heal gets a chance to fix them. Used ONLY
    # by _find_hallucinated_columns, not by _find_cross_table_columns.
    known_columns.update(_dotted_literal_parts(known_columns))
    logger.debug("plan_node: %d known columns extracted from schema context", len(known_columns))

    # Per-table column map — catches a column that's real SOMEWHERE in the
    # schema but referenced against the wrong table's alias (see rule 6b note
    # and _find_cross_table_columns docstring).
    table_columns_map = _extract_table_columns(schema_context)

    # Built once per plan-validation request — shared by shadow-mode logging
    # and (when PLAN_NODE_AST_HALLUCINATION_ENABLED=1) the live AST-based
    # table/column hallucination check in _validate_plan_items below.
    _ast_schema_graph = _build_ast_schema(table_labels, table_columns_map)
    _ast_dialect_name = _ast_dialect(config.db_type)

    _DB_LABELS = {
        "sqlite": "SQLite", "csv": "SQLite (CSV file)", "excel": "SQLite (Excel file)",
        "postgres": "PostgreSQL", "postgresql": "PostgreSQL",
        "redshift": "Amazon Redshift (PostgreSQL-compatible)",
        "sqlserver": "SQL Server (T-SQL)",
        "oracle": "Oracle SQL",
        "snowflake": "Snowflake",
        "bigquery": "Google BigQuery",
    }
    db_label = _DB_LABELS.get(config.db_type.lower(), config.db_type.upper())

    # Build persona-aware prefix if an analyst_role is configured.
    # analyst_role may be a short name ("Portfolio Manager") or the full role
    # description ("Portfolio Manager — focus on pool-level exposure…").
    # Use only the short name in template text so the prompt stays concise.
    analyst_role = getattr(config, "analyst_role", "").strip()
    if analyst_role:
        role_name = re.split(r"\s*[—\-–]\s*", analyst_role, maxsplit=1)[0].strip()
        analyst_role_prefix = (
            f"You are an expert SQL analyst assisting a **{role_name}**.\n"
            f"Prioritise the metrics, ratios, and analytical patterns most relevant "
            f"to a {role_name}.  Compute derived metrics (rates, shares, growth %) "
            f"wherever the schema supports it — do not return raw values alone.\n\n"
        )
    else:
        analyst_role_prefix = ""

    # claim_underwriting-only rule: policy_id is an identifier, never a name-match
    # target, and "policy evaluation/decision" questions mean FACT_UNDERWRITING_DECISION,
    # not fact_claim/dim_policy (a claim filed against an already-issued policy is a
    # different concept). Gated on db_schema so no other data source's prompt changes.
    if db_schema.lower() == "claim_underwriting":
        schema_specific_rules = (
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "CLAIM_UNDERWRITING SCHEMA — DOMAIN-SPECIFIC RULES\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "- A person's name (applicant, customer, beneficiary, agent, underwriter,\n"
            "  reviewer, policyholder) must be filtered against a *_name column — e.g.\n"
            "  FACT_UNDERWRITING_DECISION.applicant_name. NEVER filter a person's name\n"
            "  against an *_id / *_key / *_code identifier column such as\n"
            "  dim_policy.policy_id, even if no exact *_name column filter was found\n"
            "  in another candidate table.\n"
            "- Questions about a policy \"evaluation\", \"decision\", \"approval\",\n"
            "  \"rejection\", \"underwriting\", or \"risk assessment\" refer to\n"
            "  FACT_UNDERWRITING_DECISION (decision, uw_status, risk_score,\n"
            "  applicant_name, ...) — NOT fact_claim / dim_policy, which represent a\n"
            "  claim filed against an already-issued policy, a different concept.\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        )
    else:
        schema_specific_rules = ""

    system = _SYSTEM_PROMPT.format(
        row_limit=config.row_limit,
        max_queries=config.max_sql_queries,
        db_label=db_label,
        dialect_rules=_build_dialect_rules(config.db_type),
        analyst_role_prefix=analyst_role_prefix,
        schema_specific_rules=schema_specific_rules,
    )
    schema_line = (
        f"TARGET SCHEMA: {db_schema}"
        if db_schema
        else "TARGET SCHEMA: (none — use bare table names WITHOUT any schema prefix)"
    )

    # Build conversation history section (with summarization for long sessions)
    history          = state.get("conversation_history") or []
    haiku_model      = getattr(config, "plan_llm_model", "claude-haiku-4-5")
    history_section  = _build_history_section_plan(history, haiku_model)

    # Build multi-KG section if applicable
    active_kg_ids = state.get("active_kg_ids") or []
    kg_bridges_active = state.get("kg_bridges_active") or []
    if len(active_kg_ids) > 1:
        bridges_text = "\n".join(
            f"  {b.get('from_kg','')}.{b.get('from_column','')}  →  "
            f"{b.get('to_kg','')}.{b.get('to_column','')}  [{b.get('join_type','FK')}]"
            for b in kg_bridges_active
        ) or "  (none detected)"
        multi_kg_section = (
            f"ACTIVE KG IDS: {active_kg_ids}\n"
            f"CROSS-KG BRIDGES (join keys between schemas):\n{bridges_text}\n\n"
        )
    else:
        multi_kg_section = ""

    # Build glossary section from terms loaded by understand_node
    glossary_terms = state.get("glossary_terms") or []
    if glossary_terms:
        gl_lines = [
            "BUSINESS GLOSSARY — TERM DEFINITIONS & SYNONYMS",
            "=" * 60,
            "These are approved business definitions. When the user's question uses any",
            "of these terms (or their synonyms), interpret them using the definition below.",
            "IMPORTANT: a glossary term's Name is business wording, NOT a real table or",
            "column identifier — NEVER emit the Term text itself as a column/table name in",
            "generated SQL. Use the Term only to understand what the user means, then find",
            "the matching real column via the schema/RESOLVED BUSINESS CONCEPTS sections",
            "(the SQL hint below, if present, shows the real expression/column to use).",
            "",
        ]
        for term in glossary_terms:
            gl_lines.append(f"  Term        : {term['name']}")
            if term.get("domain"):
                gl_lines.append(f"  Domain      : {term['domain']}")
            if term.get("definition"):
                gl_lines.append(f"  Definition  : {term['definition']}")
            if term.get("formula"):
                gl_lines.append(f"  Formula     : {term['formula']}")
            if term.get("sql_hint"):
                gl_lines.append(f"  SQL hint    : {term['sql_hint']}")
            syns = [s["synonym"] for s in (term.get("synonyms") or [])]
            if syns:
                gl_lines.append(f"  Synonyms    : {', '.join(syns)}")
            gl_lines.append("")
        gl_lines.append("=" * 60)
        glossary_section = "\n".join(gl_lines) + "\n\n"
    else:
        glossary_section = ""

    # Build the auto-generated, governed glossary section (glossary_registry,
    # source-scoped, loaded by understand_node into generated_glossary_terms).
    # Strictly supplementary: never overrides RESOLVED BUSINESS CONCEPTS above
    # or the curated BUSINESS GLOSSARY above — the LLM is told to defer to both.
    generated_glossary_terms = state.get("generated_glossary_terms") or []
    if generated_glossary_terms:
        gg_lines = [
            "AUTO-GENERATED BUSINESS GLOSSARY — SUPPLEMENTARY CONTEXT (non-authoritative)",
            "=" * 60,
            "These terms were auto-discovered from this data source and approved by a",
            "steward. Use them ONLY to understand terminology in the user's question that",
            "is NOT already covered by RESOLVED BUSINESS CONCEPTS or BUSINESS GLOSSARY",
            "above. Do not let these override a binding or definition given above.",
            "When 'Observed values' is present, those are real values sampled from the",
            "actual column — prefer them verbatim in WHERE/filter clauses over guessing",
            "a spelling/casing from the definition text alone.",
            "IMPORTANT: the Term itself is business wording, NOT a real table or column",
            "identifier — NEVER emit the Term text as a column/table name in generated",
            "SQL. When 'Maps to' is present below, that is the real table.column this",
            "term refers to and MUST be the identifier used in the SQL.",
            "",
        ]
        for term in generated_glossary_terms:
            gg_lines.append(f"  Term        : {term.get('preferred_name', '')}")
            if term.get("domain"):
                gg_lines.append(f"  Domain      : {term['domain']}")
            if term.get("definition"):
                gg_lines.append(f"  Definition  : {term['definition']}")
            mapped_columns = term.get("mapped_columns") or []
            if mapped_columns:
                gg_lines.append(f"  Maps to     : {', '.join(mapped_columns)}")
            observed = term.get("observed_values") or []
            if observed:
                gg_lines.append(f"  Observed values (sampled from actual data): {observed}")
            gg_lines.append("")
        gg_lines.append("=" * 60)
        generated_glossary_section = "\n".join(gg_lines) + "\n\n"
        max_chars = getattr(config, "generated_glossary_section_max_chars", 3000)
        if len(generated_glossary_section) > max_chars:
            generated_glossary_section = generated_glossary_section[:max_chars] + "\n…(truncated)\n\n"
    else:
        generated_glossary_section = ""

    # Build KPI section from active KPI definitions loaded by understand_node
    active_kpis = state.get("active_kpis") or []
    compiled_kpis = [k for k in active_kpis if k.get("sql_expression")]
    if compiled_kpis:
        kpi_lines = [
            "DEFINED KPIs — USE THESE SQL EXPRESSIONS FOR MATCHING METRICS",
            "=" * 60,
        ]
        for kpi in compiled_kpis:
            kpi_lines.append(f"  KPI name    : {kpi['name']}")
            if kpi.get("description"):
                kpi_lines.append(f"  Description : {kpi['description']}")
            if kpi.get("unit"):
                kpi_lines.append(f"  Unit/dir    : {kpi['unit']} ({kpi.get('direction','up')})")
            kpi_lines.append(f"  SQL formula : {kpi['sql_expression']}")
            kpi_lines.append(
                "  ⚡ When the user asks about this metric, embed this SQL expression "
                "directly into your SELECT clause."
            )
            kpi_lines.append("")
        kpi_lines.append("=" * 60)
        kpi_section = "\n".join(kpi_lines) + "\n\n"
    else:
        kpi_section = ""

    # Build resolution section from resolve_node output (if any)
    term_resolution = state.get("term_resolution") or []
    if term_resolution:
        res_lines = [
            "PRE-RESOLVED CATEGORY MAPPINGS — MANDATORY",
            "=" * 60,
            "These mappings were computed BEFORE SQL generation by inspecting the",
            "actual stored values.  You MUST use the sql_fragment shown for each",
            "filter.  Do NOT replace these with the user's original terminology.",
            "",
        ]
        for r in term_resolution:
            user_term      = r.get("user_term", "")
            column         = r.get("column", "")
            matched        = r.get("matched_values") or []
            sql_frag       = r.get("sql_fragment") or ""
            reasoning      = r.get("reasoning", "")
            no_match       = r.get("no_match", False) or not sql_frag
            if no_match:
                res_lines.append(
                    f'  ⛔ NO MATCH: User said "{user_term}" — this term does NOT exist'
                    f' in any categorical column in the schema.'
                )
                if reasoning:
                    res_lines.append(f"    Reason: {reasoning}")
                res_lines.append(
                    f"    ⚡ DO NOT add a WHERE filter for \"{user_term}\"."
                    f" Retrieve all data and let the user see what values exist."
                )
                res_lines.append("")
            elif sql_frag:
                res_lines.append(
                    f'  User said "{user_term}"  →  column "{column}"'
                )
                if reasoning:
                    res_lines.append(f"    Reasoning: {reasoning}")
                res_lines.append(f"    Matched stored values: {matched}")
                res_lines.append(f"    ⚡ USE THIS IN WHERE CLAUSE: {sql_frag}")
                res_lines.append("")
        res_lines.append(
            "⚠ Any WHERE clause on a categorical column MUST use the sql_fragment "
            "above.  NEVER write WHERE col = '<user terminology>' directly."
        )
        res_lines.append("=" * 60)
        resolution_section = "\n".join(res_lines) + "\n\n"
    else:
        resolution_section = ""

    # Build verified-examples section from human-corrected past queries for
    # this data source (see dialog_agent/verified_queries.py).
    # graphrag_kg_id is never populated by dialog_api.py (only used for
    # multi-KG snapshot lookups) — it would resolve to the same "default"
    # bucket for every single-KG production request, silently mixing verified
    # examples across unrelated data sources. source_id IS populated on every
    # /query request (dialog_api.py: DialogConfig(source_id=req.source_id...))
    # and is unique per data source, so it's the correct scoping key here.
    _default_kg_id = (
        getattr(config, "graphrag_kg_id", "").strip()
        or getattr(config, "source_id", "").strip()
        or "default"
    )
    _verified_kg_ids = active_kg_ids or [_default_kg_id]
    verified_examples: List[Dict] = []
    for _kgid in _verified_kg_ids:
        try:
            verified_examples.extend(
                verified_queries.get_similar(_kgid, natural_query, top_k=3)
            )
        except Exception as exc:
            logger.debug(
                "plan_node: verified_queries lookup failed for kg_id=%s — %s", _kgid, exc,
            )
    verified_examples.sort(key=lambda e: -e["similarity"])
    verified_examples = verified_examples[:3]

    if verified_examples:
        ve_lines = [
            "VERIFIED PAST CORRECTIONS — HUMAN-CONFIRMED EXAMPLES FOR THIS DATA SOURCE",
            "=" * 60,
            "A human previously corrected the system's SQL for a similar question on this",
            "exact data source. Treat these as ground truth for this schema — when the",
            "current question closely matches one, follow its column choices and filter",
            "logic rather than independently re-deriving a different answer.",
            "",
        ]
        for ex in verified_examples:
            ve_lines.append(
                f"  Similar question (similarity={ex['similarity']:.2f}): {ex['question']}"
            )
            ve_lines.append(f"  Verified SQL:\n    {ex['sql']}")
            if ex.get("note"):
                ve_lines.append(f"  Note: {ex['note']}")
            ve_lines.append("")
        ve_lines.append("=" * 60)
        verified_section = "\n".join(ve_lines) + "\n\n"
    else:
        verified_section = ""

    # ── RESOLVED BUSINESS CONCEPTS (dissect_node) ───────────────────────────
    # Empty string when the feature is disabled or in shadow mode, or when
    # no concepts were resolved this turn — byte-identical prompt to before
    # this feature existed in that case.
    derived_metrics = state.get("derived_metrics") or []
    if derived_metrics:
        dm_lines = [
            "RESOLVED BUSINESS CONCEPTS — MANDATORY COLUMN BINDINGS",
            "=" * 60,
            "Each concept below has been resolved to REAL columns and validated",
            "against this schema (every column confirmed present; every filter",
            "literal confirmed present in that column's actual data). You MUST",
            "express these concepts using the bindings given. Do NOT reference a",
            "pre-aggregated column for them — none exists.",
            "",
        ]
        for dm in derived_metrics:
            cols = ", ".join(
                f"{b['table']}.{b['column']}" for b in (dm.get("bindings") or [])
            )
            dm_lines.append(f"  Concept: {dm.get('term', '')}")
            dm_lines.append(f"    Columns:     {cols}")
            if dm.get("aggregation"):
                dm_lines.append(f"    Aggregation: {dm['aggregation']}")
            if dm.get("grain"):
                dm_lines.append(f"    Grain:       {dm['grain']}")
            for f in (dm.get("filter_predicates") or []):
                dm_lines.append(
                    f"    Filter:      {f['column']} {f['op']} {f['value']!r}"
                )
            if dm.get("time_window"):
                tw = dm["time_window"]
                dm_lines.append(
                    f"    Window:      {tw.get('column')} within {tw.get('span')}"
                )
            dm_lines.append("")
        dm_lines.append("=" * 60)
        lexicon_section = "\n".join(dm_lines) + "\n\n"
        max_chars = getattr(config, "lexicon_section_max_chars", 4000)
        if len(lexicon_section) > max_chars:
            lexicon_section = lexicon_section[:max_chars] + "\n…(truncated)\n\n"
    else:
        lexicon_section = ""

    user = _USER_PROMPT.format(
        schema_context=schema_context,
        db_type=config.db_type,
        schema_line=schema_line,
        history_section=history_section,
        multi_kg_section=multi_kg_section,
        lexicon_section=lexicon_section,
        glossary_section=glossary_section,
        generated_glossary_section=generated_glossary_section,
        kpi_section=kpi_section,
        resolution_section=resolution_section,
        verified_section=verified_section,
        natural_query=natural_query,
    )

    # Token guard: trim schema_context if total prompt exceeds 180k token budget.
    system, user = guard_plan_prompt(system, user, schema_context, model=config.plan_llm_model)

    try:
        raw, plan = _call_plan_llm_and_parse(system, user, config.plan_llm_model, config.llm_temperature)
        logger.info("LLM plan response (first 500 chars): %s", raw[:500])
    except Exception as exc:
        logger.exception("plan_node LLM call failed")
        state["errors"].append(f"plan_node: LLM error — {exc}")
        state["sql_queries"] = []
        state["phase"] = "plan"
        return state

    # If the LLM returned [] it usually includes a prose explanation of why the
    # question cannot be answered from the schema.  Capture that text so
    # synthesize_node can surface it to the user instead of a generic error.
    if not plan:
        # Detect whether raw looks like a JSON plan that failed to parse
        # (malformed / truncated).  If raw starts with '[' after stripping
        # code fences, it is a broken JSON plan — do not use it as prose.
        raw_stripped = re.sub(r'```(?:json)?', '', raw).strip().rstrip('`').strip()
        # raw_stripped starts with '[{' → broken JSON plan (array of objects)
        # raw_stripped starts with '[]' → empty array, prose may follow → extract it
        # raw_stripped starts with '[' but not '[{' or '[]' → broken partial JSON → skip
        if raw_stripped.startswith('[{') or (
            raw_stripped.startswith('[') and not raw_stripped.startswith('[]')
        ):
            logger.warning(
                "plan_node: LLM returned unparseable JSON plan (%d chars) — "
                "no plan_explanation set to avoid leaking raw JSON to user",
                len(raw),
            )
        else:
            prose = re.sub(r'```(?:json)?\s*\[\s*\]\s*```', '', raw).strip()
            prose = re.sub(r'^\s*\[\s*\]\s*', '', prose).strip()
            if prose:
                state["plan_explanation"] = prose
                logger.info("plan_node: LLM returned [] with explanation (%d chars)", len(prose))

    # ── Pre-flight completeness check (detection only — no LLM correction) ───
    # Runs on the raw plan to detect analytical gaps and log them.
    # NOTE: A correction LLM call was tried but caused regressions — the
    # full-plan rewrite destroyed working queries.  Gap detection is kept
    # for observability; the real fix is grain annotations injected into the
    # schema context BEFORE the first LLM call (see _annotate_schema_grain).
    if plan:
        preflight_gaps = _preflight_check_plan(plan, known_columns, natural_query, schema_context)
        if preflight_gaps:
            gap_text = "\n".join(f"  • {g}" for g in preflight_gaps)
            logger.warning(
                "plan_node: pre-flight check found %d gap(s):\n%s",
                len(preflight_gaps), gap_text,
            )
            state.setdefault("errors", []).append(
                "plan_node: pre-flight gaps (logged, not corrected): "
                + "; ".join(preflight_gaps)
            )

    # Build valid join pair whitelist from schema context (used in join validation below)
    valid_join_pairs = _extract_valid_join_pairs(schema_context)
    logger.debug("plan_node: %d valid join pair(s) extracted from schema context", len(valid_join_pairs))

    # Mandatory-filter enforcement setup — applies to PostgreSQL and every
    # file-based dialect (Excel/CSV/SQLite all load into the same in-memory
    # SQLite engine via _is_file_based), since the underlying bug (an LLM-
    # generated sub-query silently dropping a resolved mandatory filter) is
    # dialect-agnostic. See the drop check inside _validate_plan_items below.
    _enforce_mandatory_filters = (
        config.db_type.lower() in ("postgres", "postgresql") or _is_file_based
    )
    _mandatory_fragments = [
        r.get("sql_fragment", "").strip()
        for r in term_resolution
        if r.get("sql_fragment") and not r.get("no_match")
    ]

    def _normalize_sql_fragment(s: str) -> str:
        return re.sub(r"\s+", " ", s or "").strip().lower()

    def _validate_plan_items(
        plan_items: List[Dict],
        offset: int = 0,
    ) -> tuple:
        """
        Validate, qualify, and post-process a list of LLM-generated plan items.

        Returns:
            valid_queries   : List[SQLQuery] — items that passed all checks
            retry_reasons   : List[str]      — human-readable rejection reasons
                              (only for invalid-join drops, which benefit from retry)
        """
        valid: List[SQLQuery] = []
        reasons: List[str] = []

        for item in plan_items[: config.max_sql_queries]:
            if not item.get("sql"):
                continue

            sql = item["sql"].strip().rstrip(";").strip()
            sql = _qualify_sql(sql, db_schema, table_labels)
            sql = _fix_count_vs_sum(sql, natural_query)
            sql = _fix_percentage(sql, natural_query, config.db_type)
            sql = _enforce_sql_limits(sql, config.row_limit, config.db_type)
            sql = _fix_dialect_syntax(sql, config.db_type)
            sql = _fix_sqlserver_subquery_limits(sql, config.db_type)
            sql = _fix_subquery_order_by(sql, config.db_type)
            sql = _fix_window_functions(sql, config.db_type)
            sql = _fix_multicolumn_subquery(sql)
            sql = _fix_order_by_alias_qualifier(sql)
            sql = _fix_distinct_order_by(sql)

            # Table hallucination check — a FROM/JOIN table name that doesn't
            # exist in the schema at all (e.g. SEA_IDISCOVER_EXPENSES when the
            # real table is SEA_IDISCOVER_EVENT_EXPENSE). Nothing else below
            # can salvage this, so drop immediately.
            if table_labels:
                if _AST_RESOLVER_ENABLED:
                    bad_tables = sorted({
                        b.name for b in _ast_resolver.find_hallucinated_identifiers(
                            sql, _ast_schema_graph, dialect=_ast_dialect_name,
                        )
                        if b.kind == "table"
                    })
                else:
                    bad_tables = _find_hallucinated_tables(sql, table_labels)
                    _shadow_check_hallucination(
                        sql, _ast_schema_graph, _ast_dialect_name,
                        item.get("query_id", "?"), old_bad_tables=bad_tables,
                    )
                if bad_tables:
                    logger.warning(
                        "plan_node: dropping query %s — hallucinated table name(s): %s",
                        item.get("query_id", "?"), bad_tables,
                    )
                    state["errors"].append(
                        f"plan_node: query {item.get('query_id','?')} skipped — "
                        f"hallucinated table name(s) not in schema: {bad_tables}"
                    )
                    reasons.append(
                        f"Query {item.get('query_id','?')} referenced table(s) {bad_tables} "
                        "that do not exist in the schema. Use only the exact table names "
                        "listed under AVAILABLE TABLES."
                    )
                    continue

            # Multi-table / no-join-key check
            sql_no_strings = re.sub(r"'[^']*'", "''", sql)
            tables_in_sql = [
                t for t in table_labels
                if re.search(r'\b' + re.escape(t) + r'\b', sql_no_strings, re.IGNORECASE)
            ]
            if len(tables_in_sql) > 1 and "JOIN KEYS: No columns" in schema_context:
                logger.warning(
                    "plan_node: dropping query %s — cross-table ref with no valid join key: %s",
                    item.get("query_id", "?"), tables_in_sql,
                )
                state["errors"].append(
                    f"plan_node: query {item.get('query_id','?')} skipped — "
                    f"cross-table reference with no valid join key: {tables_in_sql}"
                )
                reasons.append(
                    f"Query {item.get('query_id','?')} referenced {tables_in_sql} but "
                    "there are no valid join keys between those tables."
                )
                continue

            # Audit/ETL housekeeping column used as a join key — e.g.
            # DW_CREATED_AT. Always invalid regardless of what POSSIBLE JOIN
            # KEYS lists, since these are row-load timestamps, not real FKs.
            audit_join_cols = _find_audit_column_joins(sql)
            if audit_join_cols:
                logger.warning(
                    "plan_node: dropping query %s — joined on audit/ETL column(s) %s",
                    item.get("query_id", "?"), audit_join_cols,
                )
                state["errors"].append(
                    f"plan_node: query {item.get('query_id','?')} skipped — "
                    f"joined on audit/ETL housekeeping column(s) {audit_join_cols} "
                    "(row-load timestamps, not a real relationship)"
                )
                reasons.append(
                    f"Query {item.get('query_id','?')} joined on {audit_join_cols}, "
                    "which are ETL/audit load-timestamp columns, not real foreign "
                    "keys — they exist on almost every table but never encode a "
                    "relationship between entities. If no other valid join key "
                    "exists between these tables, query each one separately, or "
                    "check whether the table you're already querying has its own "
                    "column that answers the question without a join at all."
                )
                continue

            # Mandatory entity/category filter enforcement (PostgreSQL + every
            # file-based dialect — Excel/CSV/SQLite) — resolve_node pre-resolves
            # free-text terms (e.g. a customer name) into an exact sql_fragment
            # that the prompt tells the LLM is MANDATORY for every query in this
            # plan. That instruction is only prose, so nothing previously
            # stopped one query in a multi-query plan (e.g. a supplementary
            # aggregate) from silently dropping the fragment and returning an
            # unfiltered, whole-table number that synthesize_node then narrates
            # as if it were entity-scoped — regardless of which entity was
            # actually asked about. Drop any query that omits a mandatory
            # fragment rather than execute it.
            if _enforce_mandatory_filters and _mandatory_fragments:
                missing_fragments = [
                    frag for frag in _mandatory_fragments
                    if _normalize_sql_fragment(frag) not in _normalize_sql_fragment(sql)
                ]
                if missing_fragments:
                    logger.warning(
                        "plan_node: dropping query %s — missing mandatory pre-resolved "
                        "filter fragment(s) required for this question: %s",
                        item.get("query_id", "?"), missing_fragments,
                    )
                    state["errors"].append(
                        f"plan_node: query {item.get('query_id','?')} skipped — "
                        f"omitted mandatory filter fragment(s) {missing_fragments} from "
                        "PRE-RESOLVED CATEGORY MAPPINGS, which would have returned an "
                        "unfiltered/whole-table result mislabeled as entity-scoped."
                    )
                    reasons.append(
                        f"Query {item.get('query_id','?')} did not include the "
                        f"required filter fragment(s) {missing_fragments} from the "
                        "PRE-RESOLVED CATEGORY MAPPINGS section. Every query in this "
                        "plan MUST include every mandatory sql_fragment verbatim, "
                        "unless the query is explicitly an unfiltered benchmark/"
                        "comparison the user asked for by name."
                    )
                    continue

            # Column hallucination check
            if known_columns:
                if _AST_RESOLVER_ENABLED:
                    bad_cols = sorted({
                        b.name for b in _ast_resolver.find_hallucinated_identifiers(
                            sql, _ast_schema_graph, dialect=_ast_dialect_name,
                        )
                        if b.kind == "column"
                    })
                else:
                    bad_cols = _find_hallucinated_columns(sql, known_columns)
                    _shadow_check_hallucination(
                        sql, _ast_schema_graph, _ast_dialect_name,
                        item.get("query_id", "?"), old_bad_cols=bad_cols,
                    )
                if bad_cols:
                    if _has_hallucinated_join(sql, bad_cols):
                        logger.warning(
                            "plan_node: dropping query %s — hallucinated column(s) %s in JOIN",
                            item.get("query_id", "?"), bad_cols,
                        )
                        state["errors"].append(
                            f"plan_node: query {item.get('query_id','?')} skipped — "
                            f"hallucinated JOIN key(s): {bad_cols}"
                        )
                        reasons.append(
                            f"Query {item.get('query_id','?')} used JOIN column(s) "
                            f"{bad_cols} that do not exist in the schema."
                        )
                        continue

                    if _AST_RESOLVER_ENABLED:
                        repaired_sql, changelog = _ast_resolver.repair(
                            sql, _ast_schema_graph, dialect=_ast_dialect_name,
                        )
                        if repaired_sql is None:
                            logger.warning(
                                "plan_node: dropping query %s — unremovable hallucinated "
                                "col(s) %s (AST resolver: %s)",
                                item.get("query_id", "?"), bad_cols, changelog,
                            )
                            state["errors"].append(
                                f"plan_node: query {item.get('query_id','?')} skipped — "
                                f"unremovable hallucinated column(s): {bad_cols}"
                            )
                            reasons.append(
                                f"Query {item.get('query_id','?')} referenced column(s) "
                                f"{bad_cols} that do not exist in the schema. If these "
                                "represent a computed/derived metric (a count, sum, average, "
                                "or similar rollup), do not assume a pre-aggregated column "
                                "exists — compute it with COUNT()/SUM()/GROUP BY over the "
                                "raw table(s) that hold the underlying records instead."
                            )
                            continue
                        sql = repaired_sql
                        state["errors"].append(
                            f"plan_node: query {item.get('query_id','?')} — "
                            f"stripped hallucinated condition(s) for column(s): {bad_cols} "
                            f"(AST resolver: {'; '.join(changelog)})"
                        )
                    else:
                        sql = _strip_hallucinated_conditions(sql, bad_cols)
                        still_bad = _find_hallucinated_columns(sql, known_columns)
                        if still_bad:
                            logger.warning(
                                "plan_node: dropping query %s — unremovable hallucinated cols %s",
                                item.get("query_id", "?"), still_bad,
                            )
                            state["errors"].append(
                                f"plan_node: query {item.get('query_id','?')} skipped — "
                                f"unremovable hallucinated column(s): {still_bad}"
                            )
                            reasons.append(
                                f"Query {item.get('query_id','?')} referenced column(s) "
                                f"{still_bad} that do not exist in the schema. If these "
                                "represent a computed/derived metric (a count, sum, average, "
                                "or similar rollup), do not assume a pre-aggregated column "
                                "exists — compute it with COUNT()/SUM()/GROUP BY over the "
                                "raw table(s) that hold the underlying records instead."
                            )
                            continue
                        else:
                            state["errors"].append(
                                f"plan_node: query {item.get('query_id','?')} — "
                                f"stripped hallucinated condition(s) for column(s): {bad_cols}"
                            )

            # Cross-table column check — a column that's real SOMEWHERE in the
            # schema but referenced against a table alias that doesn't actually
            # have it (e.g. filtering EVENT_ATTENDANCE on "country_code" when
            # only the HCP table has that column). _find_hallucinated_columns
            # above can't catch this since the column genuinely exists.
            if table_columns_map:
                cross_bad = _find_cross_table_columns(sql, table_columns_map, known_columns_base)
                if cross_bad:
                    if _has_hallucinated_join(sql, cross_bad):
                        logger.warning(
                            "plan_node: dropping query %s — cross-table column(s) %s in JOIN",
                            item.get("query_id", "?"), cross_bad,
                        )
                        state["errors"].append(
                            f"plan_node: query {item.get('query_id','?')} skipped — "
                            f"cross-table column reference(s) in JOIN: {cross_bad}"
                        )
                        reasons.append(
                            f"Query {item.get('query_id','?')} referenced column(s) "
                            f"{cross_bad} against a table alias that does not have that "
                            "column — that column belongs to a different table in the schema."
                        )
                        continue

                    sql = _strip_hallucinated_conditions(sql, cross_bad)
                    still_bad = _find_cross_table_columns(sql, table_columns_map, known_columns_base)
                    if still_bad:
                        logger.warning(
                            "plan_node: dropping query %s — unremovable cross-table cols %s",
                            item.get("query_id", "?"), still_bad,
                        )
                        state["errors"].append(
                            f"plan_node: query {item.get('query_id','?')} skipped — "
                            f"unremovable cross-table column(s): {still_bad}"
                        )
                        continue
                    else:
                        state["errors"].append(
                            f"plan_node: query {item.get('query_id','?')} — "
                            f"stripped cross-table column condition(s): {cross_bad}"
                        )

            # Join pair validation
            if valid_join_pairs:
                invalid_joins = _find_invalid_join_conditions(sql, valid_join_pairs)
                if invalid_joins:
                    salvaged_sql = _strip_invalid_join(sql, invalid_joins)
                    if salvaged_sql:
                        logger.info(
                            "plan_node: salvaged query %s by stripping invalid JOIN(s): %s",
                            item.get("query_id", "?"), invalid_joins,
                        )
                        sql = salvaged_sql
                    else:
                        logger.warning(
                            "plan_node: dropping query %s — invalid JOIN pair(s) not salvageable: %s",
                            item.get("query_id", "?"), invalid_joins,
                        )
                        state["errors"].append(
                            f"plan_node: query {item.get('query_id','?')} skipped — "
                            f"invalid JOIN column pair(s) not in schema join keys: {invalid_joins}"
                        )
                        reasons.append(
                            f"Query {item.get('query_id','?')} used JOIN ON condition(s) "
                            f"{invalid_joins} — these column pairs are NOT valid join keys "
                            "in the schema. Check POSSIBLE JOIN KEYS; do not join tables via "
                            "unrelated FK columns from different dimension tables."
                        )
                        continue

            # Unnecessary-join check — drop JOINs whose table contributes no
            # column to SELECT/WHERE/GROUP BY/HAVING/ORDER BY. A join like this
            # can only multiply base-table rows (fan-out) and inflate COUNT/SUM
            # results without adding any data to the answer.
            unnecessary_aliases = _find_unnecessary_joins(sql)
            if unnecessary_aliases:
                salvaged_sql = _strip_unnecessary_join(sql, unnecessary_aliases)
                if salvaged_sql:
                    logger.info(
                        "plan_node: query %s — removed unnecessary JOIN(s) %s "
                        "(no columns from these tables were used)",
                        item.get("query_id", "?"), unnecessary_aliases,
                    )
                    state["errors"].append(
                        f"plan_node: query {item.get('query_id','?')} — removed unnecessary "
                        f"JOIN(s) {unnecessary_aliases}: no columns from these tables were "
                        "referenced in SELECT/WHERE/GROUP BY/ORDER BY, so the JOIN could only "
                        "inflate row counts (fan-out) without contributing data."
                    )
                    sql = salvaged_sql

            # COUNT-after-JOIN guard — a remaining (necessary) JOIN can still be
            # one-to-many, so any non-DISTINCT COUNT(col) must be made DISTINCT
            # to avoid silently overcounting the base entity.
            sql = _fix_count_after_join(sql)
            if _has_unguarded_count_star_after_join(sql):
                logger.warning(
                    "plan_node: query %s uses COUNT(*) after a JOIN with no DISTINCT guard — "
                    "result may be inflated if the joined table has multiple rows per base row",
                    item.get("query_id", "?"),
                )
                state["errors"].append(
                    f"plan_node: query {item.get('query_id','?')} uses COUNT(*) after a JOIN — "
                    "if the joined table can have multiple rows per base-table row this count "
                    "may be inflated. Could not auto-fix (no safe primary-key column identified); "
                    "verify manually or prefer COUNT(DISTINCT <base_table_pk>)."
                )

            valid.append(
                SQLQuery(
                    query_id    = item.get("query_id", f"q{offset + len(valid) + 1}"),
                    description = item.get("description", ""),
                    sql         = sql,
                    table_refs  = item.get("table_refs", []),
                    kg_id       = item.get("kg_id", ""),
                )
            )

        return valid, reasons

    # ── First pass ────────────────────────────────────────────────────────────
    sql_queries, retry_reasons = _validate_plan_items(plan)

    # ── Retry pass (once) when all queries were dropped due to fixable errors ─
    # Feed the specific rejection reasons back to the LLM so it can rewrite
    # without the invalid constructs (invalid joins, hallucinated columns, etc.)
    if not sql_queries and retry_reasons and plan:
        reason_text = "\n".join(f"  • {r}" for r in retry_reasons)
        retry_user = (
            "CORRECTION REQUIRED\n"
            "=" * 60 + "\n"
            "Your previous query attempt was REJECTED for the following reason(s):\n\n"
            f"{reason_text}\n\n"
            "Rules for the corrected query:\n"
            "1. Do NOT use any of the invalid JOIN conditions listed above.\n"
            "2. Only use JOIN ON conditions that appear in POSSIBLE JOIN KEYS or FK lines.\n"
            "3. If filtering by a dimension requires a join that is NOT in the schema "
            "(e.g. no valid path from the fact table to a geography dimension), "
            "OMIT that filter — do not attempt to route through an unrelated dimension "
            "table (e.g. do not use dim_segment to reach dim_region).\n"
            "4. Do NOT reference any column that is not listed verbatim in the schema "
            "below. If the question asks for a count, sum, average, rate, or other "
            "rollup/derived metric and no column with that exact meaning exists, do "
            "NOT assume a pre-aggregated column exists — instead compute it yourself "
            "with COUNT()/SUM()/AVG()/GROUP BY over the raw table(s) that hold the "
            "underlying records for that concept.\n"
            "5. Note in the query 'description' field any dimension you could not filter.\n"
            "6. Return the best query you can from the available schema.\n"
            "=" * 60 + "\n\n"
            + user
        )
        logger.info(
            "plan_node: all queries dropped — retrying with %d rejection reason(s)",
            len(retry_reasons),
        )
        try:
            retry_raw  = _call_llm(system, retry_user, config.plan_llm_model, config.llm_temperature)
            logger.info("plan_node: retry LLM response (first 500 chars): %s", retry_raw[:500])
            retry_plan = _extract_json(retry_raw)
            sql_queries, _ = _validate_plan_items(retry_plan, offset=0)
            if sql_queries:
                logger.info("plan_node: retry produced %d valid query/queries", len(sql_queries))
            else:
                logger.warning("plan_node: retry also produced no valid queries")
        except Exception as exc:
            logger.warning("plan_node: retry LLM call failed — %s", exc)

    if not sql_queries:
        state["errors"] = state.get("errors") or []
        state["errors"].append(
            f"plan_node: LLM returned {len(plan)} item(s) but all were dropped or empty. "
            "Check logs for hallucination/multi-table drops, or the LLM may have returned [] "
            "because the schema context had no tables."
        )
        logger.warning("plan_node: 0 SQL queries produced from LLM plan of %d item(s)", len(plan))
    else:
        logger.info("plan_node: %d SQL queries planned", len(sql_queries))

    # ── Corrective retry: impossible numeric-threshold filter (Check 6) ──────
    # A query filtering a column against a threshold its own data range can
    # never satisfy (e.g. "> 20" when the real max is 6) runs cleanly and
    # just returns 0 rows — no runtime error, so execute_node's error-driven
    # self-heal never gets a chance to fix it. This has to be caught and
    # corrected here, at plan time. Bounded to _MAX_THRESHOLD_FIX_ATTEMPTS
    # corrective LLM calls per affected query (same retry-count convention as
    # execute_node's self-heal, _MAX_RETRIES) — never unbounded. Each rewrite
    # is re-run through the exact same qualify/dialect/hallucination/
    # cross-table safety net as any other plan item before being trusted; if
    # no attempt can be verified fixed, the original query is kept unchanged
    # and the gap is still visible in state["errors"] via Check 6's preflight
    # logging. Queries without this exact issue are never touched, so this
    # cannot affect any other query pattern.
    _MAX_THRESHOLD_FIX_ATTEMPTS = 2
    if sql_queries:
        _threshold_ranges = _extract_column_numeric_ranges(schema_context)
        if _threshold_ranges:
            for q in sql_queries:
                gap_msg = _detect_impossible_numeric_filter(q["sql"], _threshold_ranges)
                if not gap_msg:
                    continue

                current_sql = q["sql"]
                for attempt in range(_MAX_THRESHOLD_FIX_ATTEMPTS):
                    logger.info(
                        "plan_node: query %s has an impossible numeric filter — "
                        "corrective rewrite attempt %d/%d",
                        q["query_id"], attempt + 1, _MAX_THRESHOLD_FIX_ATTEMPTS,
                    )
                    attempt_gap_msg = gap_msg if attempt == 0 else (
                        f"{gap_msg}\n\nNote: a previous corrective attempt for this "
                        f"exact problem ALSO failed to fix it (it kept filtering the "
                        f"same wrong column/table). Do not repeat that attempt — "
                        f"actively look for a DIFFERENT table in the schema context "
                        f"whose grain is one-row-per-occurrence of the entity being "
                        f"counted, even if it has no column with a similar name to "
                        f"the one that failed."
                    )
                    fixed_sql = _fix_impossible_threshold_sql(
                        current_sql, attempt_gap_msg, natural_query, schema_context,
                        config.plan_llm_model, config.llm_temperature,
                    )
                    if fixed_sql.strip().rstrip(";").strip() == current_sql.strip().rstrip(";").strip():
                        logger.info(
                            "plan_node: corrective rewrite attempt %d for %s made "
                            "no change", attempt + 1, q["query_id"],
                        )
                        continue

                    candidate = fixed_sql.strip().rstrip(";").strip()
                    candidate = _qualify_sql(candidate, db_schema, table_labels)
                    candidate = _fix_count_vs_sum(candidate, natural_query)
                    candidate = _fix_percentage(candidate, natural_query, config.db_type)
                    candidate = _enforce_sql_limits(candidate, config.row_limit, config.db_type)
                    candidate = _fix_dialect_syntax(candidate, config.db_type)
                    candidate = _fix_sqlserver_subquery_limits(candidate, config.db_type)
                    candidate = _fix_subquery_order_by(candidate, config.db_type)
                    candidate = _fix_window_functions(candidate, config.db_type)
                    candidate = _fix_multicolumn_subquery(candidate)
                    candidate = _fix_order_by_alias_qualifier(candidate)
                    candidate = _fix_distinct_order_by(candidate)

                    if table_labels and _find_hallucinated_tables(candidate, table_labels):
                        logger.warning(
                            "plan_node: corrective rewrite attempt %d for %s "
                            "introduced a hallucinated table — discarding",
                            attempt + 1, q["query_id"],
                        )
                        continue
                    # A table-switch rewrite (rule 23) is a bigger change than a
                    # WHERE/HAVING tweak — also verify it didn't leave a stale
                    # alias.column reference pointing at the table it just left.
                    if _find_cross_table_columns(candidate, table_columns_map, known_columns_base):
                        logger.warning(
                            "plan_node: corrective rewrite attempt %d for %s "
                            "introduced a cross-table column reference — discarding",
                            attempt + 1, q["query_id"],
                        )
                        continue
                    if _detect_impossible_numeric_filter(candidate, _threshold_ranges):
                        logger.warning(
                            "plan_node: corrective rewrite attempt %d for %s "
                            "still has an impossible filter — %s",
                            attempt + 1, q["query_id"],
                            "trying again" if attempt + 1 < _MAX_THRESHOLD_FIX_ATTEMPTS
                            else "giving up, keeping original query",
                        )
                        current_sql = candidate  # let the next attempt build on this one
                        continue

                    q["sql"] = candidate
                    logger.info(
                        "plan_node: query %s corrected (impossible-threshold fix "
                        "applied on attempt %d)", q["query_id"], attempt + 1,
                    )
                    break

    # ── Inject COUNT companions for raw-row queries ───────────────────────────
    # For every sampled raw-row query (no aggregation), prepend a companion
    # COUNT(*) query so the user and the synthesizer know the total matching
    # row count, not just the size of the page returned.
    if sql_queries and getattr(config, "raw_row_count_companion", True):
        enriched: List[SQLQuery] = []
        companions_added = 0
        for q in sql_queries:
            if _is_raw_row_query(q["sql"]):
                companion_dict = _build_count_companion(
                    q["sql"], q["query_id"], list(q.get("table_refs") or []), config.db_type
                )
                companion = SQLQuery(
                    query_id    = companion_dict["query_id"],
                    description = companion_dict["description"],
                    sql         = companion_dict["sql"],
                    table_refs  = companion_dict["table_refs"],
                    kg_id       = companion_dict["kg_id"],
                )
                enriched.append(companion)
                companions_added += 1
                logger.info(
                    "plan_node: injected COUNT companion %s for raw-row query %s",
                    companion["query_id"], q["query_id"],
                )
            enriched.append(q)

        if companions_added:
            sql_queries = enriched
            logger.info(
                "plan_node: %d COUNT companion(s) injected → %d total queries",
                companions_added, len(sql_queries),
            )

    state["sql_queries"] = sql_queries
    state["phase"] = "plan"
    return state
