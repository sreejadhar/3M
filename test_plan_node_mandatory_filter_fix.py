"""Isolated test for the mandatory-filter drop check in plan_node.py.

Mocks the LLM call so no network/API access is needed, and exercises the
real plan_node() function end-to-end against a canned 2-query plan: one
query correctly includes a resolved entity filter, the other omits it
(the general shape of the reported bug — not tied to any specific person).
Asserts the offending query is dropped for every dialect the check now
covers (PostgreSQL + every file-based dialect: Excel, CSV, SQLite), and
left alone for dialects outside that scope.
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
        "description": "Resolved entity's claim count",
        "sql": "SELECT COUNT(*) AS claim_count FROM fact_claim WHERE customer_key = 42",
        "table_refs": ["fact_claim"],
    },
    {
        "query_id": "q2",
        "description": "Total appeals filed (BUG: forgot the resolved entity filter)",
        "sql": "SELECT COUNT(*) AS appeal_count FROM fact_claim_appeal",
        "table_refs": ["fact_claim_appeal"],
    },
])

# Dialects the mandatory-filter check now covers (Postgres + all file-based
# sources — Excel/CSV/SQLite all route through the same in-memory SQLite
# engine per _is_file_based in plan_node.py).
COVERED_DIALECTS = ["postgres", "postgresql", "excel", "csv", "sqlite"]
# Dialects it deliberately does NOT touch.
UNCOVERED_DIALECTS = ["snowflake", "redshift", "bigquery", "sqlserver"]


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
        natural_query="Give me insight on this customer's claims and appeals",
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
            "user_term": "<resolved entity>",
            "column": "customer_key",
            "matched_values": [42],
            "sql_fragment": "customer_key = 42",
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
    for dialect in COVERED_DIALECTS:
        ids, errors = run_case(dialect)
        assert "q1" in ids, f"q1 (correctly filtered) should survive for {dialect}"
        assert "q2" not in ids, f"q2 (missing mandatory filter) should be DROPPED for {dialect}"
        assert any("omitted mandatory filter fragment" in e for e in errors), f"expected drop reason logged for {dialect}"
        print(f"PASS: {dialect} — q2 correctly dropped, q1 survives.")

    for dialect in UNCOVERED_DIALECTS:
        ids, _ = run_case(dialect)
        assert "q1" in ids and "q2" in ids, f"{dialect} must be unaffected by this fix"
        print(f"PASS: {dialect} — unaffected (both queries survive), confirming the fix's scope.")

    print("\nALL PASS")
