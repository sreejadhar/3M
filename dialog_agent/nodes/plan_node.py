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
from typing import Any, Dict, List

from ..state import DialogState, SQLQuery

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert SQL analyst.  You receive a natural-language question about a
database and a schema context (qualified table names, columns, relationships).
Your job is to decompose the question into one or more SQL SELECT queries that,
when executed and combined, will answer the question completely.

Rules:
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
5. LIMIT rules — apply carefully based on query type:
   a. Aggregation queries (any query that uses GROUP BY, or has COUNT/SUM/AVG/MIN/MAX
      without GROUP BY): do NOT add a LIMIT clause.  The database computes the
      aggregate over ALL rows and returns only the summarised result, which is
      already small.  Adding LIMIT here silently drops groups and produces wrong
      totals.
   b. Raw-row queries (SELECT without aggregation, used to show examples or a
      ranked list): add LIMIT {row_limit}.
   c. Ranked / top-N questions ("top 10 products", "bottom 5 regions"): use
      ORDER BY <metric> DESC LIMIT <N> where N is the number explicitly requested
      or a sensible default (10–20).  Do NOT use LIMIT {row_limit} here.
6. Prefer simple queries; only join when necessary.
7. If the question cannot be answered from the available schema, return [].
8. Maximum {max_queries} queries total.
9. Do NOT use table aliases that shadow schema-qualified names — always write the
   full qualified reference in FROM/JOIN clauses.
9b. CROSS-TABLE RULES — read carefully:
    a. To JOIN two tables you MUST have a column listed under "POSSIBLE JOIN KEYS"
       in the schema context, or one shown on a "FK:" line.
       NEVER invent or guess a join key (e.g. Check_PC, CP_ID, PC_ID, Center_ID).
    b. If no POSSIBLE JOIN KEYS are listed between two tables you want to combine,
       you MUST query each table SEPARATELY — one query per table.
       Do NOT use any of these workarounds to fake a cross-table result:
         • subqueries that reference a second table (e.g. WHERE x IN (SELECT ...))
         • correlated subqueries
         • EXISTS / NOT EXISTS against a second table
         • scalar subqueries that pull a value from another table
         • CROSS JOIN or implicit comma-joins
       Each query in your JSON array must reference ONLY ONE table (or joined
       tables with a valid key).  The synthesise step will combine the results.
    c. If a column you need (e.g. SBU1) is only in Table A, write a query for
       Table A that retrieves it.  Write a second query for Table B with its
       own columns.  Do NOT try to bridge them without a valid join key.
10. String/text filters: ALWAYS use case-insensitive matching.
    - If [sample values] are shown for a column, use the exact spelling from the samples.
    - If sample values are NOT shown, use: LOWER(column_name) LIKE LOWER('%search_term%')
    - Never rely on an exact case-sensitive equality match for text unless you copied
      the value directly from a [sample values] list.
11. Date/period filters: if filtering by year/month, check column names in the schema
    carefully — use the correct column (e.g. Year, Month, Period) and match the sample
    value format (e.g. integer 2026 vs string '2026').
12. COUNT vs SUM — choose the correct aggregate:
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
13. Percentage calculations — when the question asks for %, share, proportion,
    or percentage of a total:
    a. Always compute the percentage IN SQL — do not leave it to the reader.
    b. Use a window function to avoid a subquery where possible:
         ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS Percentage
       or for SUM-based measures:
         ROUND(SUM(col) * 100.0 / SUM(SUM(col)) OVER (), 2) AS Percentage
    c. Always include both the raw value (count or sum) AND the percentage column
       in the SELECT list so the user sees both.
    d. Label the percentage column clearly, e.g. AS Headcount_Pct or AS Revenue_Pct.
    e. Round percentages to 2 decimal places with ROUND(..., 2).
    f. If asked "what percentage is X of Y" without a GROUP BY, use a single-row
       scalar expression:
         SELECT ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM table), 2) AS Pct
       — still include the raw numerator and denominator counts alongside it.
"""

_USER_PROMPT = """\
SCHEMA CONTEXT:
{schema_context}

TARGET DATABASE TYPE: {db_type}
{schema_line}
{history_section}
NATURAL LANGUAGE QUESTION:
{natural_query}

CRITICAL REMINDERS:
- Use ONLY column names that appear in the DETAILED SCHEMA above. Do NOT invent column names.
- Use ONLY table names from the AVAILABLE TABLES list above.
- CROSS-TABLE: If no POSSIBLE JOIN KEYS exist between two tables, query them SEPARATELY.
  Do NOT use subqueries, IN (...), EXISTS, correlated queries, or any trick to combine
  data from two tables that have no valid join key. One query = one table (or validly joined tables).
- For any text/string filter, use case-insensitive matching (LOWER() LIKE or exact sample value).
- If [sample values] are shown for a column, pick the matching value verbatim from that list.
- COUNT vs SUM: use COUNT(*) for headcount/how-many questions; use SUM(col) only for
  monetary/quantity totals. NEVER use SUM() to count people or rows.
- PERCENTAGES: if the question asks for %, share, or proportion — compute it in SQL
  using ROUND(value * 100.0 / SUM(value) OVER (), 2). Always include both the raw
  value and the percentage column. Do not leave percentage calculation to the reader.

Return the JSON array of SQL queries now.
"""


def _call_llm(
    system: str,
    user: str,
    model: str,
    temperature: float,
) -> str:
    """Call Anthropic Claude and return the raw text response."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    msg = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text if msg.content else ""


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
            col_name = stripped.split(":")[0].split("[")[0].strip()
            if col_name:
                known.add(col_name.lower())
    return known


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
}


def _find_hallucinated_columns(sql: str, known_cols: set) -> List[str]:
    """
    Find column references in the form  alias.ColumnName  where ColumnName is
    NOT in the known schema.  This pattern (e.g. md.Check_PC) is the most
    reliable hallucination signal — the LLM uses a table alias and a column it
    invented from domain knowledge.

    We intentionally limit the check to dotted references to avoid false
    positives from table/alias names that are not in known_cols.
    """
    if not known_cols:
        return []

    # Strip string literals so quoted values don't confuse the regex
    sql_stripped = re.sub(r"'[^']*'", "''", sql)

    hallucinated = []
    seen: set = set()
    # Match   word.Identifier   where Identifier is not followed by '(' (functions)
    for m in re.finditer(r'\b[A-Za-z_]\w*\.([A-Za-z_]\w*)(?!\s*\()', sql_stripped):
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
            cp = re.escape(col)
            # Value token: a quoted string, a number, or a bare word (handles =, LIKE, IN, IS)
            val = r"""(?:'[^']*'|\([^)]*\)|[^\s,)]+)"""
            op  = r"(?:=|!=|<>|>=|<=|>|<|(?:NOT\s+)?LIKE|(?:NOT\s+)?IN|IS(?:\s+NOT)?)"

            # Case 3: AND/OR condition  — simplest, remove the entire arm
            sql = re.sub(
                r"(?i)\s+(?:AND|OR)\s+\w+\." + cp + r"\s+" + op + r"\s*" + val,
                "",
                sql,
            )

            # Case 2: WHERE col ... AND next → replace with WHERE next
            sql = re.sub(
                r"(?i)\bWHERE\s+\w+\." + cp + r"\s+" + op + r"\s*" + val + r"\s+AND\s+",
                "WHERE ",
                sql,
            )

            # Case 1: WHERE col ... (nothing follows, or clause keywords follow)
            sql = re.sub(
                r"(?i)\s+WHERE\s+\w+\." + cp + r"\s+" + op + r"\s*" + val
                + r"(?=\s*(?:GROUP\b|ORDER\b|HAVING\b|LIMIT\b|$))",
                "",
                sql,
            )

        return sql.strip()
    except Exception:
        return sql


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


def _qualify_sql(sql: str, db_schema: str, known_tables: List[str]) -> str:
    """
    Safety net: if the LLM wrote `FROM orders` but the schema is `public`,
    rewrite bare table references to schema-qualified form `public.orders`.

    Only touches identifiers that exactly match known table labels and are not
    already preceded by a dot (i.e., not already schema-qualified).
    """
    if not db_schema or not known_tables:
        return sql

    # Sort longest first to avoid partial replacements (e.g. "order" before "orders")
    for table in sorted(known_tables, key=len, reverse=True):
        qualified = f"{db_schema}.{table}"
        # Skip if already present in the SQL (case-insensitive)
        if re.search(re.escape(qualified), sql, re.IGNORECASE):
            continue
        # Replace bare table name not preceded by a dot
        # Negative lookbehind for `.` or alphanumeric ensures we don't touch substrings
        pattern = r'(?<![.\w])\b' + re.escape(table) + r'\b(?!\s*\.)'
        if re.search(pattern, sql, re.IGNORECASE):
            sql = re.sub(pattern, qualified, sql, flags=re.IGNORECASE)

    return sql


_PERCENTAGE_KEYWORDS = re.compile(
    r'\b(percent(?:age)?|%|share|proportion|breakdown|distribution|'
    r'ratio|split|how\s+much\s+(?:is|are)|out\s+of\s+total)\b',
    re.IGNORECASE,
)


def _fix_percentage(sql: str, natural_query: str) -> str:
    """
    If the question asks for a percentage and the SQL has a GROUP BY aggregate
    but no percentage column, inject a window-function percentage expression.

    Works for both COUNT(*) and SUM(col) aggregates.
    Leaves the SQL unchanged if it already contains a percentage expression.
    """
    if not _PERCENTAGE_KEYWORDS.search(natural_query):
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


_AGG_PATTERN = re.compile(
    r'\b(GROUP\s+BY|COUNT\s*\(|SUM\s*\(|AVG\s*\(|MIN\s*\(|MAX\s*\()\b',
    re.IGNORECASE,
)

def _remove_limit_from_aggregation(sql: str) -> str:
    """
    Strip LIMIT / OFFSET clauses from aggregation queries so that GROUP BY
    results are never silently truncated.

    Logic:
    - If the query has no aggregation (GROUP BY / COUNT / SUM / AVG / MIN / MAX),
      leave it untouched — the row_limit LIMIT is correct for raw-row queries.
    - If the query has aggregation and a LIMIT of ≤50, it is an intentional
      top-N ("top 10 products") — leave it untouched.
    - If the query has aggregation and a LIMIT of >50 (i.e. the default
      row_limit like 500), remove the LIMIT so all groups are returned.
    """
    if not _AGG_PATTERN.search(sql):
        return sql  # not an aggregation query

    # Find any LIMIT clause
    m = re.search(r'\bLIMIT\s+(\d+)', sql, re.IGNORECASE)
    if not m:
        return sql  # no LIMIT present — nothing to do

    limit_val = int(m.group(1))
    if limit_val <= 50:
        return sql  # intentional small top-N — keep it

    # Large LIMIT (e.g. 500) on an aggregation query — strip it
    cleaned = re.sub(
        r'\s+LIMIT\s+\d+(\s+OFFSET\s+\d+)?\s*$',
        '',
        sql,
        flags=re.IGNORECASE,
    ).strip()

    if cleaned != sql:
        logger.info(
            "plan_node: removed LIMIT %d from aggregation query (would silently drop groups)",
            limit_val,
        )

    return cleaned


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

    # Build a set of all valid column names from the schema context so we can
    # reject any SQL the LLM generates using hallucinated column names.
    known_columns = _extract_known_columns(schema_context)
    # Add table labels as valid identifiers (they can appear bare in SQL too)
    known_columns.update(t.lower() for t in table_labels)
    logger.debug("plan_node: %d known columns extracted from schema context", len(known_columns))

    system = _SYSTEM_PROMPT.format(
        row_limit=config.row_limit,
        max_queries=config.max_sql_queries,
    )
    schema_line = (
        f"TARGET SCHEMA: {db_schema}"
        if db_schema
        else "TARGET SCHEMA: (none — use bare table names WITHOUT any schema prefix)"
    )

    # Build conversation history section (last 5 turns, oldest first)
    history = state.get("conversation_history") or []
    if history:
        lines = ["CONVERSATION HISTORY (previous questions in this session — use for context only):"]
        for turn in history:
            lines.append(f"Q{turn['turn']}: {turn['question']}")
            if turn.get("tables_queried"):
                lines.append(f"  Tables used: {', '.join(turn['tables_queried'])}")
            if turn.get("insights"):
                lines.append(f"  Answer summary: {turn['insights'][:300]}")
        lines.append(
            "Use this history to resolve pronouns (e.g. 'it', 'that', 'those'), "
            "implied filters (e.g. 'same service line'), or comparisons to previous results. "
            "Do NOT reproduce previous SQL — generate fresh SQL for the new question."
        )
        history_section = "\n".join(lines) + "\n"
    else:
        history_section = ""

    user = _USER_PROMPT.format(
        schema_context=schema_context,
        db_type=config.db_type,
        schema_line=schema_line,
        history_section=history_section,
        natural_query=natural_query,
    )

    try:
        raw = _call_llm(system, user, config.plan_llm_model, config.llm_temperature)
        logger.debug("LLM plan response (first 500 chars): %s", raw[:500])
        plan: List[Dict] = _extract_json(raw)
    except Exception as exc:
        logger.exception("plan_node LLM call failed")
        state["errors"].append(f"plan_node: LLM error — {exc}")
        state["sql_queries"] = []
        state["phase"] = "plan"
        return state

    # Validate, qualify, and cap
    sql_queries: List[SQLQuery] = []
    for item in plan[: config.max_sql_queries]:
        if not item.get("sql"):
            continue

        sql = item["sql"].strip()
        # Safety net: ensure table names are schema-qualified even if LLM forgot
        sql = _qualify_sql(sql, db_schema, table_labels)
        # Headcount fix: replace SUM() with COUNT(*) when question is about people
        sql = _fix_count_vs_sum(sql, natural_query)
        # Percentage fix: inject window-function percentage column when % is asked for
        sql = _fix_percentage(sql, natural_query)
        # Aggregation fix: remove LIMIT from queries that aggregate over all rows.
        # The LLM sometimes ignores the prompt rule and adds LIMIT anyway, which
        # silently drops groups and produces incorrect totals.
        sql = _remove_limit_from_aggregation(sql)

        # Multi-table check: reject any query that references more than one table
        # when no valid join key was listed in the schema context.  This catches
        # "cross-reference approach" patterns (subqueries, IN (...), EXISTS, etc.)
        # that the LLM uses to sneak cross-table lookups past the JOIN ON check.
        tables_in_sql = [
            t for t in table_labels
            if re.search(r'\b' + re.escape(t) + r'\b', sql, re.IGNORECASE)
        ]
        if len(tables_in_sql) > 1:
            # Check whether the schema advertises a valid join key for this pair
            possible_join_section = ""
            if "POSSIBLE JOIN KEYS" in schema_context:
                possible_join_section = schema_context
            elif "JOIN KEYS: No columns" in schema_context:
                # Explicit "no join keys" message — drop immediately
                logger.warning(
                    "plan_node: dropping query %s — references %d tables %s but "
                    "schema has no POSSIBLE JOIN KEYS. SQL: %s",
                    item.get("query_id", "?"), len(tables_in_sql), tables_in_sql, sql[:200],
                )
                state["errors"].append(
                    f"plan_node: query {item.get('query_id','?')} skipped — "
                    f"cross-table reference with no valid join key: {tables_in_sql}"
                )
                continue

        # Column hallucination check: strip conditions that reference columns not
        # in the schema context (e.g. md.Check_PC invented by the LLM).
        # If the hallucinated column appears in a JOIN ON clause we cannot salvage
        # the query — drop it entirely.  For WHERE/AND/OR conditions we strip the
        # bad predicate and keep the rest.
        if known_columns:
            bad_cols = _find_hallucinated_columns(sql, known_columns)
            if bad_cols:
                if _has_hallucinated_join(sql, bad_cols):
                    logger.warning(
                        "plan_node: dropping query %s — hallucinated column(s) %s "
                        "used in JOIN ON clause (cannot salvage). SQL: %s",
                        item.get("query_id", "?"), bad_cols, sql[:200],
                    )
                    state["errors"].append(
                        f"plan_node: query {item.get('query_id','?')} skipped — "
                        f"hallucinated JOIN key(s): {bad_cols}"
                    )
                    continue

                logger.warning(
                    "plan_node: query %s references unknown column(s) %s — "
                    "stripping those conditions. SQL: %s",
                    item.get("query_id", "?"), bad_cols, sql[:200],
                )
                sql = _strip_hallucinated_conditions(sql, bad_cols)
                # After stripping, verify no bad columns remain; drop only if still present
                still_bad = _find_hallucinated_columns(sql, known_columns)
                if still_bad:
                    logger.warning(
                        "plan_node: dropping query %s — could not remove all "
                        "hallucinated columns %s", item.get("query_id", "?"), still_bad,
                    )
                    state["errors"].append(
                        f"plan_node: query {item.get('query_id','?')} skipped — "
                        f"unremovable hallucinated column(s): {still_bad}"
                    )
                    continue
                else:
                    state["errors"].append(
                        f"plan_node: query {item.get('query_id','?')} — "
                        f"stripped hallucinated condition(s) for column(s): {bad_cols}"
                    )

        sql_queries.append(
            SQLQuery(
                query_id    = item.get("query_id", f"q{len(sql_queries)+1}"),
                description = item.get("description", ""),
                sql         = sql,
                table_refs  = item.get("table_refs", []),
            )
        )

    logger.info("plan_node: %d SQL queries planned", len(sql_queries))
    state["sql_queries"] = sql_queries
    state["phase"] = "plan"
    return state
