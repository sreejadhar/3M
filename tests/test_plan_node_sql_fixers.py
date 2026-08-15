"""
Golden-query regression tests for the plan_node.py SQL post-generation
repair pipeline (dialog_agent/nodes/plan_node.py).

Purpose
-------
plan_node runs a growing chain of regex-based `_fix_*` functions over
LLM-generated SQL to repair known classes of LLM mistakes (ORDER BY on a
SELECT-DISTINCT alias, missing ORDER BY inside window functions, multi-column
scalar subqueries, dialect drift, etc.) before the query is executed. There is
no guarantee against *novel* LLM mistakes, deeply nested queries, or semantic
correctness — but every mistake pattern that HAS already been diagnosed and
fixed must stay fixed. This file is the safety net for that: one test per
confirmed bug pattern, using the exact (or minimally reduced) SQL shape that
originally triggered the fix.

Run with: pytest tests/test_plan_node_sql_fixers.py -v

When adding a new `_fix_*`/`_find_*` function to plan_node.py because of a
newly observed LLM mistake, add its golden case here in the same commit —
that is what turns "we patched this once" into "this stays patched."
"""
import importlib
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_TMP_DB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_TMP_DB.close()
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("KG_FEDERATION_DB", _TMP_DB.name)

# dialog_agent/nodes/__init__.py does `from .plan_node import plan_node`,
# which rebinds the `plan_node` attribute on the `dialog_agent.nodes` package
# to the FUNCTION, shadowing the submodule (same gotcha documented in
# test_dissect_node.py). Use importlib to get the actual module object so we
# can reach its private `_fix_*` functions.
pn = importlib.import_module("dialog_agent.nodes.plan_node")


# ---------------------------------------------------------------------------
# _fix_order_by_alias_qualifier
# ---------------------------------------------------------------------------
class TestOrderByAliasQualifier:
    """LLM qualifies a computed SELECT alias with a table prefix in ORDER BY
    — invalid in every dialect since the alias is not a real column on that
    table."""

    def test_strips_table_qualifier_from_computed_alias(self):
        sql = (
            'SELECT jc."employee_id", '
            'EXTRACT(YEAR FROM jc."end_date") - EXTRACT(YEAR FROM jc."start_date") '
            'AS years_in_position '
            'FROM job_change jc ORDER BY jc.years_in_position DESC'
        )
        fixed = pn._fix_order_by_alias_qualifier(sql)
        assert "ORDER BY years_in_position DESC" in fixed
        assert "jc.years_in_position" not in fixed

    def test_leaves_real_qualified_column_untouched(self):
        sql = 'SELECT jc."employee_id" AS emp FROM job_change jc ORDER BY jc.employee_id'
        fixed = pn._fix_order_by_alias_qualifier(sql)
        assert fixed == sql

    def test_handles_bracket_and_backtick_quoting(self):
        sql = 'SELECT t.[start_date] AS [yr] FROM job_change t ORDER BY t.[yr]'
        fixed = pn._fix_order_by_alias_qualifier(sql)
        assert "ORDER BY [yr]" in fixed

        sql2 = 'SELECT t.`start_date` AS `yr` FROM job_change t ORDER BY t.`yr`'
        fixed2 = pn._fix_order_by_alias_qualifier(sql2)
        assert "ORDER BY `yr`" in fixed2

    def test_respects_row_limiting_clause_boundary(self):
        sql = (
            'SELECT jc."employee_id", COUNT(*) AS c FROM job_change jc '
            'GROUP BY jc."employee_id" ORDER BY jc.c DESC LIMIT 50'
        )
        fixed = pn._fix_order_by_alias_qualifier(sql)
        assert "ORDER BY c DESC" in fixed
        assert "LIMIT 50" in fixed


# ---------------------------------------------------------------------------
# _fix_distinct_order_by
# ---------------------------------------------------------------------------
class TestDistinctOrderBy:
    """SELECT DISTINCT requires every ORDER BY expression to appear in the
    select list; the LLM often orders by a column it didn't select."""

    def test_splices_missing_order_by_column_into_distinct_select(self):
        sql = (
            'SELECT DISTINCT e."name" FROM employee e ORDER BY e."hire_date" DESC'
        )
        fixed = pn._fix_distinct_order_by(sql)
        assert 'e."hire_date"' in fixed
        assert fixed.index('e."hire_date"') < fixed.index("ORDER BY")

    def test_noop_when_order_by_column_already_selected(self):
        sql = 'SELECT DISTINCT e."name", e."hire_date" FROM employee e ORDER BY e."hire_date"'
        fixed = pn._fix_distinct_order_by(sql)
        assert fixed == sql

    def test_noop_without_distinct(self):
        sql = 'SELECT e."name" FROM employee e ORDER BY e."hire_date" DESC'
        fixed = pn._fix_distinct_order_by(sql)
        assert fixed == sql

    def test_truncates_at_trailing_limit_before_splitting_terms(self):
        # Regression: without truncating at LIMIT first, the last ORDER BY
        # term absorbed "LIMIT 50" as part of its text, DESC failed to strip,
        # and the garbled "hire_date DESC LIMIT 50" fragment got spliced into
        # the SELECT list verbatim (the fixer only ever ADDS to the SELECT
        # list before FROM — it never rewrites the ORDER BY clause itself, so
        # "DESC LIMIT 50" legitimately remains at the tail of the output).
        sql = (
            'SELECT DISTINCT e."dept" FROM employee e '
            'ORDER BY e."dept" DESC, e."hire_date" DESC LIMIT 50'
        )
        fixed = pn._fix_distinct_order_by(sql)
        assert "LIMIT 50" in fixed
        select_list = fixed[: fixed.upper().index(" FROM ")]
        assert 'e."hire_date"' in select_list
        assert "DESC" not in select_list
        assert "LIMIT" not in select_list.upper()


# ---------------------------------------------------------------------------
# _fix_window_functions
# ---------------------------------------------------------------------------
class TestWindowFunctions:
    """Navigation/ranking window functions require ORDER BY inside OVER();
    aggregate window functions (SUM/AVG/... OVER) must NOT be touched."""

    def test_injects_order_by_using_partition_column(self):
        sql = 'SELECT employee_id, LAG(salary) OVER (PARTITION BY dept_id) AS prev FROM t'
        fixed = pn._fix_window_functions(sql, "sqlserver")
        assert "ORDER BY dept_id" in fixed
        assert "PARTITION BY dept_id ORDER BY dept_id" in fixed

    def test_injects_noop_order_by_without_partition(self):
        sql = 'SELECT employee_id, ROW_NUMBER() OVER () AS rn FROM t'
        fixed = pn._fix_window_functions(sql, "oracle")
        assert "ORDER BY (SELECT NULL)" in fixed

    def test_leaves_existing_order_by_untouched(self):
        sql = 'SELECT employee_id, RANK() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS r FROM t'
        fixed = pn._fix_window_functions(sql, "postgres")
        assert fixed == sql

    def test_does_not_touch_aggregate_window_function(self):
        sql = 'SELECT dept_id, SUM(salary) OVER (PARTITION BY dept_id) AS dept_total FROM t'
        fixed = pn._fix_window_functions(sql, "sqlserver")
        assert fixed == sql


# ---------------------------------------------------------------------------
# _fix_subquery_order_by (SQL Server only — ORDER BY illegal in subqueries)
# ---------------------------------------------------------------------------
class TestSubqueryOrderBy:
    def test_strips_bare_order_by_inside_cte_for_sqlserver(self):
        sql = (
            'WITH ranked AS (SELECT id, salary FROM t ORDER BY salary DESC) '
            'SELECT * FROM ranked'
        )
        fixed = pn._fix_subquery_order_by(sql, "sqlserver")
        assert "ORDER BY salary DESC" not in fixed

    def test_keeps_order_by_when_top_present(self):
        sql = (
            'WITH ranked AS (SELECT TOP 10 id, salary FROM t ORDER BY salary DESC) '
            'SELECT * FROM ranked'
        )
        fixed = pn._fix_subquery_order_by(sql, "sqlserver")
        assert "ORDER BY salary DESC" in fixed

    def test_never_strips_order_by_inside_over_clause(self):
        # Regression: naive backward-scan found the OVER() paren instead of
        # the enclosing CTE paren and stripped ORDER BY out of the window
        # spec itself, breaking the window function.
        sql = (
            'WITH ranked AS ('
            'SELECT id, LAG(salary) OVER (PARTITION BY dept_id ORDER BY period_id) AS prev '
            'FROM t) SELECT * FROM ranked'
        )
        fixed = pn._fix_subquery_order_by(sql, "sqlserver")
        assert "OVER (PARTITION BY dept_id ORDER BY period_id)" in fixed

    def test_noop_for_non_sqlserver_dialects(self):
        sql = 'WITH ranked AS (SELECT id FROM t ORDER BY id) SELECT * FROM ranked'
        fixed = pn._fix_subquery_order_by(sql, "postgres")
        assert fixed == sql


# ---------------------------------------------------------------------------
# _fix_multicolumn_subquery
# ---------------------------------------------------------------------------
class TestMulticolumnSubquery:
    def test_keeps_only_first_column_in_scalar_in_subquery(self):
        sql = (
            'SELECT * FROM orders WHERE customer_id IN '
            '(SELECT customer_id, order_count FROM ranked WHERE rn <= 12)'
        )
        fixed = pn._fix_multicolumn_subquery(sql)
        assert "SELECT customer_id FROM ranked" in fixed
        assert "order_count" not in fixed

    def test_noop_for_single_column_subquery(self):
        sql = 'SELECT * FROM orders WHERE customer_id IN (SELECT customer_id FROM active)'
        fixed = pn._fix_multicolumn_subquery(sql)
        assert fixed == sql

    def test_rewrites_row_value_constructor_to_exists(self):
        # Regression: Pattern 1 (scalar IN/=/<>/... subquery) matches
        # "IN (SELECT ..." regardless of whether the LEFT side is a single
        # value or a row-value tuple like (o.customer_id, o.region). It used
        # to run first and truncate the subquery to its first projected
        # column, which made Pattern 2's `len(sub_cols) != len(outer_cols)`
        # guard fail — so the EXISTS rewrite never fired for the exact
        # multi-column case it exists to handle. Fixed by running the
        # row-value-constructor pattern (Pattern 2) before the scalar pattern
        # (Pattern 1) so it consumes the tuple span first.
        sql = (
            "SELECT * FROM orders o WHERE (o.customer_id, o.region) IN "
            "(SELECT customer_id, region FROM top_customers WHERE tier = 'gold')"
        )
        fixed = pn._fix_multicolumn_subquery(sql)
        assert "EXISTS" in fixed
        assert "customer_id = o.customer_id" in fixed
        assert "region = o.region" in fixed
        assert "tier = 'gold'" in fixed

    def test_rewrites_negated_row_value_constructor_to_not_exists(self):
        # Same regression as above, negated form.
        sql = (
            "SELECT * FROM orders o WHERE (o.customer_id, o.region) NOT IN "
            "(SELECT customer_id, region FROM blocked)"
        )
        fixed = pn._fix_multicolumn_subquery(sql)
        assert "NOT EXISTS" in fixed


# ---------------------------------------------------------------------------
# _fix_count_after_join / _has_unguarded_count_star_after_join
# ---------------------------------------------------------------------------
class TestCountAfterJoin:
    def test_wraps_count_column_with_distinct_after_join(self):
        sql = 'SELECT COUNT(o.order_id) FROM customers c JOIN orders o ON o.customer_id = c.id'
        fixed = pn._fix_count_after_join(sql)
        assert "COUNT(DISTINCT o.order_id)" in fixed

    def test_leaves_already_distinct_count_untouched(self):
        sql = 'SELECT COUNT(DISTINCT o.order_id) FROM customers c JOIN orders o ON o.customer_id = c.id'
        fixed = pn._fix_count_after_join(sql)
        assert fixed == sql

    def test_noop_without_join(self):
        sql = 'SELECT COUNT(o.order_id) FROM orders o'
        fixed = pn._fix_count_after_join(sql)
        assert fixed == sql

    def test_leaves_count_star_for_caller_to_flag(self):
        sql = 'SELECT COUNT(*) FROM customers c JOIN orders o ON o.customer_id = c.id'
        fixed = pn._fix_count_after_join(sql)
        assert fixed == sql
        assert pn._has_unguarded_count_star_after_join(fixed) is True


# ---------------------------------------------------------------------------
# _fix_count_vs_sum
# ---------------------------------------------------------------------------
class TestCountVsSum:
    def test_replaces_sum_with_count_for_headcount_question(self):
        sql = 'SELECT SUM(active_flag) FROM employee'
        fixed = pn._fix_count_vs_sum(sql, "how many employees are there")
        assert "COUNT(*)" in fixed
        assert "SUM(" not in fixed

    def test_leaves_monetary_sum_untouched_even_for_headcount_phrasing(self):
        sql = 'SELECT SUM(salary) FROM employee'
        fixed = pn._fix_count_vs_sum(sql, "how many employees are there")
        assert fixed == sql

    def test_noop_for_non_headcount_question(self):
        sql = 'SELECT SUM(revenue) FROM sales'
        fixed = pn._fix_count_vs_sum(sql, "what is total revenue")
        assert fixed == sql


# ---------------------------------------------------------------------------
# _enforce_sql_limits
# ---------------------------------------------------------------------------
class TestEnforceSqlLimits:
    def test_strips_limit_from_aggregation_query_without_order_by(self):
        sql = 'SELECT dept, COUNT(*) FROM employee GROUP BY dept LIMIT 10'
        fixed = pn._enforce_sql_limits(sql, row_limit=100, db_type="postgres")
        assert "LIMIT" not in fixed.upper()

    def test_keeps_small_intentional_top_n_with_order_by(self):
        sql = 'SELECT dept, COUNT(*) AS c FROM employee GROUP BY dept ORDER BY c DESC LIMIT 5'
        fixed = pn._enforce_sql_limits(sql, row_limit=100, db_type="postgres")
        assert "LIMIT 5" in fixed

    def test_adds_limit_to_raw_row_query_for_postgres(self):
        sql = 'SELECT * FROM employee'
        fixed = pn._enforce_sql_limits(sql, row_limit=200, db_type="postgres")
        assert fixed.rstrip().endswith("LIMIT 200")

    def test_adds_top_n_to_raw_row_query_for_sqlserver(self):
        sql = 'SELECT * FROM employee'
        fixed = pn._enforce_sql_limits(sql, row_limit=200, db_type="sqlserver")
        assert "SELECT TOP 200" in fixed

    def test_adds_fetch_first_to_raw_row_query_for_oracle(self):
        sql = 'SELECT * FROM employee'
        fixed = pn._enforce_sql_limits(sql, row_limit=200, db_type="oracle")
        assert "FETCH FIRST 200 ROWS ONLY" in fixed

    def test_does_not_double_limit_when_llm_already_added_top_n(self):
        sql = 'SELECT * FROM employee ORDER BY hire_date LIMIT 20'
        fixed = pn._enforce_sql_limits(sql, row_limit=200, db_type="postgres")
        assert fixed == sql


# ---------------------------------------------------------------------------
# _fix_sqlserver_subquery_limits
# ---------------------------------------------------------------------------
class TestSqlServerSubqueryLimits:
    def test_converts_order_by_limit_to_offset_fetch(self):
        sql = 'WITH r AS (SELECT id FROM t ORDER BY id LIMIT 10) SELECT * FROM r'
        fixed = pn._fix_sqlserver_subquery_limits(sql, "sqlserver")
        assert "OFFSET 0 ROWS FETCH NEXT 10 ROWS ONLY" in fixed
        assert "LIMIT" not in fixed.upper()

    def test_noop_for_non_sqlserver(self):
        sql = 'WITH r AS (SELECT id FROM t ORDER BY id LIMIT 10) SELECT * FROM r'
        fixed = pn._fix_sqlserver_subquery_limits(sql, "postgres")
        assert fixed == sql


# ---------------------------------------------------------------------------
# _find_hallucinated_tables / _find_hallucinated_columns
# ---------------------------------------------------------------------------
class TestHallucinationDetection:
    def test_flags_table_not_in_schema(self):
        sql = 'SELECT * FROM SEA_IDISCOVER_EXPENSES e'
        bad = pn._find_hallucinated_tables(sql, ["SEA_IDISCOVER_EVENT_EXPENSE"])
        assert "SEA_IDISCOVER_EXPENSES" in bad

    def test_does_not_flag_real_table(self):
        sql = 'SELECT * FROM SEA_IDISCOVER_EVENT_EXPENSE e'
        bad = pn._find_hallucinated_tables(sql, ["SEA_IDISCOVER_EVENT_EXPENSE"])
        assert bad == []

    def test_does_not_flag_cte_name_as_hallucinated_table(self):
        sql = (
            'WITH recent_events AS (SELECT * FROM SEA_IDISCOVER_EVENT_MASTER) '
            'SELECT * FROM recent_events'
        )
        bad = pn._find_hallucinated_tables(sql, ["SEA_IDISCOVER_EVENT_MASTER"])
        assert bad == []

    def test_flags_hallucinated_dotted_column(self):
        sql = 'SELECT md.Check_PC FROM master_data md'
        bad = pn._find_hallucinated_columns(sql, {"employee_id", "name"})
        assert "Check_PC" in bad

    def test_does_not_flag_known_column(self):
        sql = 'SELECT e."employee_id" FROM employee e'
        bad = pn._find_hallucinated_columns(sql, {"employee_id"})
        assert bad == []

    def test_does_not_flag_function_call_as_column(self):
        sql = 'SELECT UPPER(e.name) FROM employee e'
        bad = pn._find_hallucinated_columns(sql, {"name"})
        assert bad == []


# ---------------------------------------------------------------------------
# _fix_dialect_syntax
# ---------------------------------------------------------------------------
class TestDialectSyntax:
    def test_rewrites_ilike_for_sqlserver(self):
        sql = "SELECT * FROM t WHERE name ILIKE '%smith%'"
        fixed = pn._fix_dialect_syntax(sql, "sqlserver")
        assert "ILIKE" not in fixed.upper()
        assert "LOWER(name) LIKE LOWER(" in fixed

    def test_leaves_ilike_untouched_for_postgres(self):
        sql = "SELECT * FROM t WHERE name ILIKE '%smith%'"
        fixed = pn._fix_dialect_syntax(sql, "postgres")
        assert "ILIKE" in fixed

    def test_rewrites_postgres_cast_shorthand_for_oracle(self):
        sql = "SELECT amount::numeric FROM t"
        fixed = pn._fix_dialect_syntax(sql, "oracle")
        assert "CAST(amount AS numeric)" in fixed
