"""
Golden fixtures + unit tests for dialog_agent.nodes.sql_identifier_resolver.

Fixtures cover:
  - the confirmed LIFESCIENCESTESTUSECASE regression pattern (dotted-literal /
    aliased column hallucinated in SELECT, ORDER BY, GROUP BY — previously
    caused the whole query to be dropped instead of repaired)
  - the confirmed table-hallucination pattern (LLM invents a plausible table
    name not present in the schema)
  - a dialect x clause x nesting matrix (subquery, CTE, window function)

Run with: pytest tests/test_sql_identifier_resolver.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dialog_agent.nodes.sql_identifier_resolver import (
    SchemaGraph,
    find_hallucinated_identifiers,
    repair,
)


LIFESCIENCE_SCHEMA = SchemaGraph({
    "SEA_IDISCOVER_EVENT_MASTER": {
        "event_key", "name", "event_type_1", "therapeutic_area",
        "product_name", "created_date", "start_date", "end_date",
        "approve_date", "status",
    },
    "SEA_IDISCOVER_EVENT_EXPENSE": {
        "expense_key", "event_key", "actual_amount", "created_date",
    },
    "SEA_IDISCOVER_HCPS": {
        "hcp_key", "hcp_name", "country_code", "speciality_1_type",
    },
})


class TestColumnHallucinationRepair:
    """Reproduces the plan_node.py regression: a hallucinated column that
    appears outside WHERE must be stripped, not cause the whole query to drop."""

    def test_hallucinated_column_in_select_is_removed_not_dropped(self):
        sql = (
            'SELECT e."event_key", e."bogus_col" AS "x", COUNT(*) AS "c" '
            'FROM SEA_IDISCOVER_EVENT_MASTER AS e GROUP BY e."event_key"'
        )
        bad = find_hallucinated_identifiers(sql, LIFESCIENCE_SCHEMA)
        assert any(b.kind == "column" and b.name == "bogus_col" for b in bad)

        fixed, log = repair(sql, LIFESCIENCE_SCHEMA)
        assert fixed is not None, log
        assert "bogus_col" not in fixed
        assert "event_key" in fixed  # real columns survive

    def test_hallucinated_column_in_order_by_is_stripped(self):
        sql = (
            'SELECT e."event_key" FROM SEA_IDISCOVER_EVENT_MASTER AS e '
            'ORDER BY e."bogus_sort_col" DESC'
        )
        bad = find_hallucinated_identifiers(sql, LIFESCIENCE_SCHEMA)
        fixed, log = repair(sql, LIFESCIENCE_SCHEMA)
        assert fixed is not None, log
        assert "bogus_sort_col" not in fixed

    def test_hallucinated_column_in_group_by_is_stripped(self):
        sql = (
            'SELECT e."name", COUNT(*) AS "c" FROM SEA_IDISCOVER_EVENT_MASTER AS e '
            'GROUP BY e."name", e."bogus_group_col"'
        )
        bad = find_hallucinated_identifiers(sql, LIFESCIENCE_SCHEMA)
        fixed, log = repair(sql, LIFESCIENCE_SCHEMA)
        assert fixed is not None, log
        assert "bogus_group_col" not in fixed
        assert '"name"' in fixed

    def test_hallucinated_column_sole_select_item_is_unsalvageable(self):
        sql = 'SELECT e."bogus_col" FROM SEA_IDISCOVER_EVENT_MASTER AS e'
        bad = find_hallucinated_identifiers(sql, LIFESCIENCE_SCHEMA)
        fixed, log = repair(sql, LIFESCIENCE_SCHEMA)
        assert fixed is None
        assert any("empty SELECT" in msg for msg in log)

    def test_real_column_not_flagged(self):
        sql = 'SELECT e."event_key", e."name" FROM SEA_IDISCOVER_EVENT_MASTER AS e'
        bad = find_hallucinated_identifiers(sql, LIFESCIENCE_SCHEMA)
        assert bad == []


class TestTableHallucination:
    def test_invented_table_name_detected(self):
        sql = 'SELECT * FROM SEA_IDISCOVER_EVENTS'  # real table is _EVENT_MASTER
        bad = find_hallucinated_identifiers(sql, LIFESCIENCE_SCHEMA)
        assert any(b.kind == "table" and b.name == "SEA_IDISCOVER_EVENTS" for b in bad)

    def test_hallucinated_join_table_cascades_to_orphaned_columns(self):
        sql = (
            'SELECT e."event_key", x."made_up_col" '
            'FROM SEA_IDISCOVER_EVENT_MASTER AS e '
            'JOIN SEA_IDISCOVER_EXPENSES AS x ON e."event_key" = x."event_key"'
        )
        bad = find_hallucinated_identifiers(sql, LIFESCIENCE_SCHEMA)
        assert any(b.kind == "table" for b in bad)
        fixed, log = repair(sql, LIFESCIENCE_SCHEMA)
        assert fixed is not None, log
        assert "SEA_IDISCOVER_EXPENSES" not in fixed
        assert "made_up_col" not in fixed
        assert "event_key" in fixed

    def test_hallucinated_base_table_is_unsalvageable(self):
        sql = 'SELECT "x" FROM SEA_IDISCOVER_NOPE'
        bad = find_hallucinated_identifiers(sql, LIFESCIENCE_SCHEMA)
        fixed, log = repair(sql, LIFESCIENCE_SCHEMA)
        assert fixed is None

    def test_cte_name_not_flagged_as_hallucinated_table(self):
        sql = (
            'WITH agg AS (SELECT event_key FROM SEA_IDISCOVER_EVENT_MASTER) '
            'SELECT * FROM agg'
        )
        bad = find_hallucinated_identifiers(sql, LIFESCIENCE_SCHEMA)
        assert not any(b.kind == "table" and b.name.lower() == "agg" for b in bad)


class TestNestingAndDialects:
    def test_works_inside_subquery(self):
        sql = (
            'SELECT * FROM (SELECT e."event_key", e."bogus_col" '
            'FROM SEA_IDISCOVER_EVENT_MASTER AS e) AS sub'
        )
        bad = find_hallucinated_identifiers(sql, LIFESCIENCE_SCHEMA)
        assert any(b.name == "bogus_col" for b in bad)

    def test_postgres_dialect_parses(self):
        sql = 'SELECT e.event_key, e.bogus_col FROM sea_idiscover_event_master e'
        bad = find_hallucinated_identifiers(sql, LIFESCIENCE_SCHEMA, dialect="postgres")
        assert any(b.name == "bogus_col" for b in bad)

    def test_unparseable_sql_returns_empty_not_exception(self):
        bad = find_hallucinated_identifiers("SELECT FROM WHERE (((", LIFESCIENCE_SCHEMA)
        assert bad == []
