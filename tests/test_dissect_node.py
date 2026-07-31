"""
Unit tests for dialog_agent.nodes.dissect_node.

Covers: the master off-switch (state untouched when lexicon_enabled=False),
graceful degradation when internal steps raise, mechanical binding/literal
rejection, and probe-failure handling — without any real LLM or DB calls.

Run with: pytest tests/test_dissect_node.py -v --confcutdir=tests
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_TMP_DB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_TMP_DB.close()
os.environ["APP_ENV"] = "test"
os.environ["KG_FEDERATION_DB"] = _TMP_DB.name

import importlib  # noqa: E402

from dialog_agent.config import DialogConfig  # noqa: E402
from dialog_agent.nodes.sql_identifier_resolver import SchemaGraph  # noqa: E402

# dialog_agent/nodes/__init__.py does `from .dissect_node import dissect_node`,
# which rebinds the `dissect_node` attribute on the `dialog_agent.nodes`
# package to the FUNCTION, shadowing the submodule. Use importlib to get the
# actual module object (needed so patch.object(dn, "_helper", ...) below
# patches the real functions dissect_node() calls internally).
dn = importlib.import_module("dialog_agent.nodes.dissect_node")


def _state(**overrides):
    base = {
        "config": DialogConfig(source_id="src1"),
        "natural_query": "who is eligible for next promotion",
        "schema_context": "Table: job_history\nColumns:\n  event: text\n  employee_id: integer\n",
        "kg_edges": [],
        "categorical_columns": {"job_history": {"event": ["Promotion", "Transfer"]}},
    }
    base.update(overrides)
    return base


def test_disabled_by_default_returns_state_unchanged():
    state = _state()
    before = dict(state)
    out = dn.dissect_node(state)
    assert out is state
    assert out == before
    assert "derived_metrics" not in out


def test_enabled_but_dissect_disabled_only_records_unresolved():
    cfg = DialogConfig(source_id="src1", lexicon_enabled=True, lexicon_shadow_mode=False,
                        dissect_enabled=False)
    state = _state(config=cfg)
    with patch.object(dn, "_identify_concepts", return_value=["promotion count"]):
        out = dn.dissect_node(state)
    assert out["derived_metrics"] == []
    assert out["unresolved_terms"] == ["promotion count"]


def test_shadow_mode_injects_nothing_but_logs():
    cfg = DialogConfig(source_id="src1", lexicon_enabled=True, lexicon_shadow_mode=True,
                        dissect_enabled=True)
    state = _state(config=cfg)
    resolved = {
        "term": "promotion count", "kind": "derived_metric",
        "bindings": [{"table": "job_history", "column": "event"}],
        "aggregation": "COUNT(*)", "grain": "", "filter_predicates": [],
        "time_window": None, "provenance": "llm_dissector", "approved": 0,
    }
    with patch.object(dn, "_identify_concepts", return_value=["promotion count"]), \
         patch("dialog_agent.semantic_lexicon.lookup", return_value=None), \
         patch.object(dn, "_run_evaluation_loop", return_value=resolved):
        out = dn.dissect_node(state)
    # Shadow mode: never injected into derived_metrics.
    assert "derived_metrics" not in out or out["derived_metrics"] == []
    assert any("shadow" in d for d in out["lexicon_diagnostics"])


def test_exception_in_identify_concepts_leaves_pipeline_unbroken():
    cfg = DialogConfig(source_id="src1", lexicon_enabled=True)
    state = _state(config=cfg)
    with patch.object(dn, "_identify_concepts", side_effect=RuntimeError("boom")):
        out = dn.dissect_node(state)
    # No crash; concepts treated as empty, nothing resolved/unresolved.
    assert out.get("derived_metrics", []) == []
    assert out.get("unresolved_terms", []) == []


def test_lexicon_hit_bumps_hit_count_and_injects_binding():
    from dialog_agent import semantic_lexicon as sl
    for e in sl.list_all():
        sl.delete(e.entry_id)
    entry = sl.LexiconEntry(
        source_id="src1", term=sl.normalize_term("promotion count"),
        display_term="promotion count", kind="derived_metric",
        bindings=[{"table": "job_history", "column": "event"}],
        provenance="human", approved=True,
    )
    sl.save(entry)

    cfg = DialogConfig(source_id="src1", lexicon_enabled=True, lexicon_shadow_mode=False)
    state = _state(config=cfg)
    with patch.object(dn, "_identify_concepts", return_value=["promotion count"]):
        out = dn.dissect_node(state)

    assert len(out["derived_metrics"]) == 1
    assert out["derived_metrics"][0]["bindings"] == [{"table": "job_history", "column": "event"}]


def test_validate_proposal_rejects_unknown_column():
    schema = SchemaGraph({"job_history": {"event", "employee_id"}})
    proposal = {"bindings": [{"table": "job_history", "column": "nonexistent_col"}],
                "filter_predicates": []}
    ok, reason = dn._validate_proposal(proposal, schema, {"literal_values": {}})
    assert not ok
    assert "does not exist" in reason


def test_validate_proposal_rejects_invented_literal():
    schema = SchemaGraph({"job_history": {"event", "employee_id"}})
    proposal = {
        "bindings": [{"table": "job_history", "column": "event"}],
        "filter_predicates": [{"column": "event", "value": "MadeUpValue"}],
    }
    evidence = {"literal_values": {"job_history": {"event": ["Promotion", "Transfer"]}}}
    ok, reason = dn._validate_proposal(proposal, schema, evidence)
    assert not ok
    assert "top_values" in reason


def test_validate_proposal_accepts_valid_binding_and_literal():
    schema = SchemaGraph({"job_history": {"event", "employee_id"}})
    proposal = {
        "bindings": [{"table": "job_history", "column": "event"}],
        "filter_predicates": [{"column": "event", "value": "Promotion"}],
    }
    evidence = {"literal_values": {"job_history": {"event": ["Promotion", "Transfer"]}}}
    ok, reason = dn._validate_proposal(proposal, schema, evidence)
    assert ok, reason


def test_validate_proposal_rejects_pk_to_pk_join():
    schema = SchemaGraph({
        "certificates": {"id", "user_id"},
        "job_current": {"id", "employee_id"},
    })
    proposal = {
        "bindings": [{"table": "job_current", "column": "id"}, {"table": "certificates", "column": "id"}],
        "filter_predicates": [],
        "aggregation": "COUNT",
    }
    evidence = {
        "literal_values": {},
        "profiling": {
            "job_current": [
                {"column_name": "id", "is_primary_key": True, "semantic_role": "identifier"},
                {"column_name": "employee_id", "is_primary_key": False, "semantic_role": "identifier"},
            ],
            "certificates": [
                {"column_name": "id", "is_primary_key": True, "semantic_role": "identifier"},
                {"column_name": "user_id", "is_primary_key": False, "semantic_role": "identifier"},
            ],
        },
    }
    ok, reason = dn._validate_proposal(proposal, schema, evidence, "employee with most certificates")
    assert not ok
    assert "primary key" in reason


def test_validate_proposal_rejects_text_join_when_identifier_available():
    schema = SchemaGraph({
        "job_current": {"employee_id", "manager_user_sys_id"},
        "education": {"user_sys_id", "manager_last_name"},
    })
    proposal = {
        "bindings": [
            {"table": "job_current", "column": "employee_id"},
            {"table": "education", "column": "manager_last_name"},
        ],
        "filter_predicates": [],
        "aggregation": "count",
    }
    evidence = {
        "literal_values": {},
        "profiling": {
            "job_current": [
                {"column_name": "employee_id", "is_primary_key": False, "semantic_role": "identifier"},
                {"column_name": "manager_user_sys_id", "is_primary_key": False, "is_foreign_key": True},
            ],
            "education": [
                {"column_name": "user_sys_id", "is_primary_key": False, "semantic_role": "identifier"},
                {"column_name": "manager_last_name", "is_primary_key": False, "statistical_type": "categorical"},
            ],
        },
    }
    ok, reason = dn._validate_proposal(proposal, schema, evidence, "employees reporting to manager")
    assert not ok
    assert "identifier" in reason


def test_validate_proposal_rejects_bare_minmax_for_duration_concept():
    schema = SchemaGraph({"job_current": {"event_date"}})
    proposal = {
        "bindings": [{"table": "job_current", "column": "event_date"}],
        "filter_predicates": [],
        "aggregation": "MAX",
    }
    evidence = {"literal_values": {}, "profiling": {}}
    ok, reason = dn._validate_proposal(proposal, schema, evidence, "longest tenure")
    assert not ok
    assert "duration" in reason


def test_validate_proposal_accepts_duration_expressed_as_difference():
    schema = SchemaGraph({"job_current": {"hire_date"}})
    proposal = {
        "bindings": [{"table": "job_current", "column": "hire_date"}],
        "filter_predicates": [],
        "aggregation": "reference_date - hire_date",
    }
    evidence = {"literal_values": {}, "profiling": {}}
    ok, reason = dn._validate_proposal(proposal, schema, evidence, "longest tenure")
    assert ok, reason


def test_probe_failure_is_reported_not_raised():
    cfg = DialogConfig(source_id="src1", db_type="sqlite", dissect_probe_enabled=True)
    proposal = {"bindings": [{"table": "job_history", "column": "event"}], "filter_predicates": []}
    with patch("dialog_agent.nodes.execute_node._run_sql", return_value={"error": "no such table"}):
        ok, reason = dn._probe(proposal, cfg, {})
    assert not ok
    assert "no such table" in reason


def test_probe_disabled_short_circuits_to_ok():
    cfg = DialogConfig(source_id="src1", dissect_probe_enabled=False)
    proposal = {"bindings": [{"table": "job_history", "column": "event"}], "filter_predicates": []}
    ok, _ = dn._probe(proposal, cfg, {})
    assert ok
