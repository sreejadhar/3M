"""Isolated test for the new mandatory-filter drop check in plan_node.py.

Mocks the LLM call so no network/API access is needed, and exercises the
real plan_node() function end-to-end against a canned 2-query plan: one
query correctly includes the resolved customer filter, the other omits it
(the exact shape of the reported bug). Asserts the offending query is
dropped and the compliant one survives.
"""
import json
import sys
import types
from unittest.mock import patch

# langgraph isn't installed in this sandbox and dialog_agent/__init__.py
# imports it transitively (via agent.py) purely to build the LangGraph
# pipeline graph, which this test never touches — stub it out so we can
# import dialog_agent.nodes.plan_node without pulling in the real package.
if "langgraph" not in sys.modules:
    _lg = types.ModuleType("langgraph")
    _lg_graph = types.ModuleType("langgraph.graph")

    class _FakeStateGraph:
        def __init__(self, *a, **k): pass
        def add_node(self, *a, **k): pass
        def add_edge(self, *a, **k): pass
        def compile(self, *a, **k): return None

    _lg_graph.StateGraph = _FakeStateGraph
    _lg_graph.START = "__start__"
    _lg_graph.END = "__end__"
    _lg.graph = _lg_graph
    sys.modules["langgraph"] = _lg
    sys.modules["langgraph.graph"] = _lg_graph

from dialog_agent.config import DialogConfig
from dialog_agent.state import DialogState

CANNED_PLAN = json.dumps([
    {
        "query_id": "q1",
        "description": "Kevin Taylor's claim count",
        "sql": "SELECT COUNT(*) AS claim_count FROM fact_claim WHERE customer_key = 7",
        "table_refs": ["fact_claim"],
    },
    {
        "query_id": "q2",
        "description": "Total appeals filed (BUG: forgot the customer filter)",
        "sql": "SELECT COUNT(*) AS appeal_count FROM fact_claim_appeal",
        "table_refs": ["fact_claim_appeal"],
    },
])


def run_case(db_type: str):
    # dialog_agent/nodes/__init__.py does `from .plan_node import plan_node`,
    # which overwrites the `plan_node` attribute on the `nodes` package with
    # the FUNCTION (shadowing the submodule) — `import ... as pn` walks
    # attributes and would grab that function, not the module. Go through
    # sys.modules directly instead, which import machinery still populates
    # with the real module regardless of that shadowing.
    import dialog_agent.nodes.plan_node  # ensures sys.modules entry exists
    pn = sys.modules["dialog_agent.nodes.plan_node"]

    config = DialogConfig(db_type=db_type, db_schema="claim_underwriting")
    state = DialogState(
        config=config,
        natural_query="Give me insight on Kevin Taylor's claims and appeals",
        schema_context="TABLE fact_claim (claim_key, customer_key, ...)\nTABLE fact_claim_appeal (appeal_key, claim_key, ...)",
        kg_nodes=[],
        kg_edges=[],
        conversation_history=[],
        sql_queries=[],
        query_results=[],
        insights="",
        errors=[],
        phase="start",
        categorical_columns={},
        column_hierarchy={},
        term_resolution=[{
            "user_term": "Kevin Taylor",
            "column": "customer_key",
            "matched_values": [7],
            "sql_fragment": "customer_key = 7",
            "reasoning": "matched DIM_CUSTOMER.customer_key",
            "no_match": False,
        }],
        active_kg_ids=[],
        kg_bridges_active=[],
        multi_kg_configs=[],
    )

    with patch.object(pn, "_call_llm", return_value=CANNED_PLAN):
        result = pn.plan_node(state)

    surviving_ids = [q["query_id"] for q in result["sql_queries"]]
    print(f"\n--- db_type={db_type} ---")
    print("Surviving query_ids:", surviving_ids)
    print("Errors logged:", [e for e in result["errors"] if "mandatory" in e.lower() or "omitted" in e.lower()])
    return surviving_ids, result["errors"]


if __name__ == "__main__":
    pg_ids, pg_errors = run_case("postgres")
    assert "q1" in pg_ids, "q1 (correctly filtered) should survive"
    assert "q2" not in pg_ids, "q2 (missing mandatory filter) should be DROPPED for postgres"
    assert any("omitted mandatory filter fragment" in e for e in pg_errors), "expected drop reason logged"
    print("\nPASS: postgres — q2 correctly dropped, q1 survives.")

    other_ids, _ = run_case("snowflake")
    assert "q1" in other_ids and "q2" in other_ids, "non-postgres dialects must be unaffected by this fix"
    print("PASS: snowflake — unaffected (both queries survive), confirming the fix is scoped to postgres only.")
