"""
Run: python -m pytest tests/test_kg_optimizer_quick_tune.py -v
"""
from kg_optimizer.quick_tune import tune_bridge_options

_NODES = [
    {
        "id": "orders", "label": "orders",
        "properties": [{"name": "customer_id", "type": "int"}, {"name": "id", "type": "int"}],
    },
    {
        "id": "customers", "label": "customers",
        "properties": [{"name": "id", "type": "int"}, {"name": "name", "type": "string"}],
    },
]

_REPORT = {
    "tables": {
        "orders": {
            "foreign_keys": [
                {"column": "customer_id", "references_table": "customers", "references_column": "id"},
            ],
        },
    },
}


def test_tune_bridge_options_returns_none_without_report():
    assert tune_bridge_options("kg1", _NODES, None) is None


def test_tune_bridge_options_returns_none_without_gold_bridges():
    assert tune_bridge_options("kg1", _NODES, {"tables": {}}) is None


def test_tune_bridge_options_returns_tuned_options_with_gold_bridges():
    options = tune_bridge_options("kg1", _NODES, _REPORT, population=4, generations=2, seed=0)
    assert options is not None
    assert 0.30 <= options.min_confidence <= 0.70
    assert 0.60 <= options.auto_enable_threshold <= 0.95
    assert 0.50 <= options.embedding_sim_threshold <= 0.95
    # Tuning eval must never touch a live DB or run async LLM validation.
    assert options.run_tier_value_overlap is True   # untouched production default
    assert options.run_tier3_llm is True             # untouched production default
