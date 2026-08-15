"""
Regression tests for the pre-execution EXPLAIN/dry-run gate in
dialog_agent/nodes/execute_node.py (_explain_sql / _sqlite_explain_check).

The gate is a cheap dry-run check that runs before every real execution
attempt in _run_sql_with_retry / _exec_on_conn_with_retry — if the target
dialect's dry-run rejects the plan, that error is fed into the existing
self-heal retry pipeline WITHOUT paying for a real (possibly expensive)
execution. These tests never touch a real database — every DB-facing call is
mocked — and exist to guard two properties:

  1. When the dry-run rejects a query, the real execution function is never
     called for that attempt (the whole point of the gate).
  2. When the dry-run is unsupported for a dialect (oracle/teradata/
     databricks) or passes, behavior is byte-for-byte identical to before
     this gate existed — real execution runs exactly as it always did.

Run with: pytest tests/test_execute_node_explain_gate.py -v
"""
import importlib
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_TMP_DB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_TMP_DB.close()
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("KG_FEDERATION_DB", _TMP_DB.name)

# dialog_agent/nodes/__init__.py does `from .execute_node import execute_node`,
# which rebinds the `execute_node` attribute on the `dialog_agent.nodes`
# package to the FUNCTION, shadowing the submodule (same gotcha as
# test_dissect_node.py / test_plan_node_sql_fixers.py). Use importlib to get
# the actual module object.
en = importlib.import_module("dialog_agent.nodes.execute_node")

from dialog_agent.config import DialogConfig  # noqa: E402


def _cfg(db_type: str, **overrides) -> DialogConfig:
    base = dict(source_id="src1", db_type=db_type)
    base.update(overrides)
    return DialogConfig(**base)


class TestExplainSqlDialectCoverage:
    """_explain_sql: which dialects get a real dry-run check vs. skip (None)."""

    def test_postgres_uses_explain_prefix_via_run_postgres(self):
        with patch.object(en, "_run_postgres") as mock_pg:
            mock_pg.return_value = {"columns": [], "rows": [], "error": None}
            result = en._explain_sql(_cfg("postgres"), "SELECT 1")
            assert result is None
            mock_pg.assert_called_once()
            called_sql = mock_pg.call_args[0][1]
            assert called_sql == "EXPLAIN SELECT 1"

    def test_postgres_dry_run_error_is_surfaced(self):
        with patch.object(en, "_run_postgres") as mock_pg:
            mock_pg.return_value = {"columns": [], "rows": [], "error": "column \"bogus\" does not exist"}
            result = en._explain_sql(_cfg("postgres"), "SELECT bogus FROM t")
            assert result == "column \"bogus\" does not exist"

    def test_redshift_routes_through_same_postgres_explain(self):
        with patch.object(en, "_run_postgres") as mock_pg:
            mock_pg.return_value = {"columns": [], "rows": [], "error": None}
            assert en._explain_sql(_cfg("redshift"), "SELECT 1") is None
            mock_pg.assert_called_once()

    def test_snowflake_uses_explain_using_text(self):
        with patch.object(en, "_run_snowflake") as mock_sf:
            mock_sf.return_value = {"columns": [], "rows": [], "error": None}
            en._explain_sql(_cfg("snowflake"), "SELECT 1")
            called_sql = mock_sf.call_args[0][1]
            assert called_sql == "EXPLAIN USING TEXT SELECT 1"

    def test_bigquery_uses_dry_run_helper(self):
        with patch.object(en, "_bigquery_dry_run_error") as mock_bq:
            mock_bq.return_value = None
            assert en._explain_sql(_cfg("bigquery"), "SELECT 1") is None
            mock_bq.assert_called_once()

    def test_bigquery_dry_run_exception_surfaces_as_error_string(self):
        with patch.object(en, "_bigquery_dry_run_error", side_effect=RuntimeError("bad query")):
            result = en._explain_sql(_cfg("bigquery"), "SELECT bogus")
            assert result == "bad query"

    def test_sqlserver_uses_parseonly_helper(self):
        with patch.object(en, "_sqlserver_parseonly_error") as mock_mssql:
            mock_mssql.return_value = None
            assert en._explain_sql(_cfg("sqlserver"), "SELECT 1") is None
            mock_mssql.assert_called_once()

    def test_oracle_is_skipped_entirely(self):
        with patch.object(en, "_run_oracle") as mock_ora:
            result = en._explain_sql(_cfg("oracle"), "SELECT 1")
            assert result is None
            mock_ora.assert_not_called()

    def test_teradata_is_skipped_entirely(self):
        with patch.object(en, "_run_teradata") as mock_td:
            result = en._explain_sql(_cfg("teradata"), "SELECT 1")
            assert result is None
            mock_td.assert_not_called()

    def test_databricks_is_skipped_entirely(self):
        with patch.object(en, "_run_databricks") as mock_db:
            result = en._explain_sql(_cfg("databricks"), "SELECT 1")
            assert result is None
            mock_db.assert_not_called()

    def test_kill_switch_disables_gate_even_for_covered_dialect(self):
        with patch.dict(os.environ, {"EXECUTE_NODE_EXPLAIN_GATE": "0"}):
            with patch.object(en, "_EXPLAIN_GATE_ENABLED", False):
                with patch.object(en, "_run_postgres") as mock_pg:
                    result = en._explain_sql(_cfg("postgres"), "SELECT 1")
                    assert result is None
                    mock_pg.assert_not_called()

    def test_explain_check_itself_never_raises(self):
        with patch.object(en, "_run_postgres", side_effect=ConnectionError("network down")):
            result = en._explain_sql(_cfg("postgres"), "SELECT 1")
            assert result == "network down"


class TestRunSqlWithRetryGateWiring:
    """_run_sql_with_retry: real execution must be skipped on an explain
    rejection, and must run normally when explain passes/is skipped."""

    def test_real_execution_skipped_when_explain_rejects_and_error_not_retryable(self):
        with patch.object(en, "_explain_sql", return_value="relation \"ghost_table\" does not exist"), \
             patch.object(en, "_run_sql") as mock_run:
            result = en._run_sql_with_retry(_cfg("postgres"), "SELECT * FROM ghost_table")
            mock_run.assert_not_called()
            assert result["error"] == "relation \"ghost_table\" does not exist"

    def test_real_execution_proceeds_normally_when_explain_passes(self):
        with patch.object(en, "_explain_sql", return_value=None), \
             patch.object(en, "_run_sql") as mock_run:
            mock_run.return_value = {"columns": ["c"], "rows": [[1]], "error": None}
            result = en._run_sql_with_retry(_cfg("postgres"), "SELECT 1 AS c")
            mock_run.assert_called_once()
            assert result["error"] is None
            assert result["rows"] == [[1]]

    def test_retryable_explain_rejection_triggers_llm_fix_before_any_real_execution(self):
        # First explain call rejects (retryable-looking), triggers the LLM
        # fix path; second explain call (on the fixed SQL) passes; only then
        # should the real execution ever run.
        explain_calls = {"n": 0}

        def fake_explain(cfg, sql):
            explain_calls["n"] += 1
            if explain_calls["n"] == 1:
                return "syntax error at or near \"FORM\""
            return None

        with patch.object(en, "_explain_sql", side_effect=fake_explain), \
             patch.object(en, "_llm_fix_sql", return_value="SELECT 1 FROM t") as mock_fix, \
             patch.object(en, "_run_sql") as mock_run:
            mock_run.return_value = {"columns": ["c"], "rows": [[1]], "error": None}
            result = en._run_sql_with_retry(_cfg("postgres"), "SELECT 1 FORM t")
            mock_fix.assert_called_once()
            mock_run.assert_called_once_with(
                _cfg("postgres"), "SELECT 1 FROM t", state=None, kg_id=""
            )
            assert result["error"] is None
            assert result["sql"] == "SELECT 1 FROM t"


class TestSqliteExplainCheck:
    def _memconn(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE t (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO t VALUES (1, 'a')")
        conn.commit()
        return conn

    def test_valid_query_returns_none(self):
        conn = self._memconn()
        try:
            assert en._sqlite_explain_check(conn, "SELECT * FROM t") is None
        finally:
            conn.close()

    def test_unknown_table_is_rejected_with_enrichment(self):
        conn = self._memconn()
        try:
            error = en._sqlite_explain_check(conn, "SELECT * FROM ghost")
            assert error is not None
            assert "no such table" in error.lower()
            assert "t" in error  # _exec_on_conn's available-tables enrichment
        finally:
            conn.close()

    def test_kill_switch_disables_sqlite_gate(self):
        conn = self._memconn()
        try:
            with patch.object(en, "_EXPLAIN_GATE_ENABLED", False):
                assert en._sqlite_explain_check(conn, "SELECT * FROM ghost") is None
        finally:
            conn.close()

    def test_exec_on_conn_with_retry_skips_real_exec_on_explain_rejection(self):
        conn = self._memconn()
        try:
            with patch.object(en, "_exec_on_conn") as mock_exec:
                mock_exec.return_value = {"columns": [], "rows": [], "error": "no such table: ghost"}
                result = en._exec_on_conn_with_retry(conn, "SELECT * FROM ghost", _cfg("sqlite"))
                # Every call (explain + any retries) goes through the same
                # mocked _exec_on_conn; the real (non-EXPLAIN) SQL must never
                # be passed to it since the error is non-retryable.
                for call in mock_exec.call_args_list:
                    passed_sql = call[0][1]
                    assert passed_sql.startswith("EXPLAIN QUERY PLAN")
                assert result["error"] == "no such table: ghost"
        finally:
            conn.close()
