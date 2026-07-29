"""
Run: python -m pytest tests/test_kg_optimizer_datasets.py -v
Schema round-trip tests for the generic EvalDataset adapter — no live services.
"""
import json
import os
import tempfile

from kg_optimizer.datasets import (
    EvalDataset, EvalQuestion, GoldBridge, gold_bridges_from_metadata_report,
    load_dataset, save_dataset,
)


def test_dataset_round_trips_through_json():
    dataset = EvalDataset(
        source_id="demo_source",
        questions=[
            EvalQuestion(question="How many rows?", gold_answer="42"),
            EvalQuestion(question="No gold answer here"),
        ],
        gold_bridges=[GoldBridge(table_a="a", col_a="id", table_b="b", col_b="a_id")],
        kg_store_source_id="abc-123",
    )
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "dataset.json")
        save_dataset(dataset, path)
        loaded = load_dataset(path)

    assert loaded.source_id == "demo_source"
    assert len(loaded.questions) == 2
    assert loaded.questions[0].gold_answer == "42"
    assert loaded.questions[1].gold_answer is None
    assert loaded.gold_bridges[0].table_a == "a"
    assert loaded.kg_store_source_id == "abc-123"


def test_fixture_lifesciences_sample_matches_schema():
    fixture_path = os.path.join(
        os.path.dirname(__file__), "..", "kg_optimizer", "fixtures", "lifesciences_sample.json"
    )
    dataset = load_dataset(fixture_path)
    assert dataset.source_id == "lifesciences_testusecase"
    assert len(dataset.questions) > 0
    assert all(isinstance(q, EvalQuestion) for q in dataset.questions)
    assert dataset.kg_store_source_id


def test_gold_bridges_from_declared_foreign_keys():
    report = {
        "tables": {
            "orders": {"foreign_keys": [{"column": "customer_id", "references_table": "customers", "references_column": "id"}]},
        },
    }
    gold = gold_bridges_from_metadata_report(report)
    assert len(gold) == 1
    assert gold[0] == GoldBridge(table_a="orders", col_a="customer_id", table_b="customers", col_b="id", expected=True)


def test_gold_bridges_from_fk_candidates():
    report = {
        "fk_candidates": [
            {"left_table": "orders", "left_columns": ["customer_id"], "right_table": "customers", "right_columns": ["id"]},
        ],
    }
    gold = gold_bridges_from_metadata_report(report)
    assert len(gold) == 1
    assert gold[0].table_a == "orders" and gold[0].table_b == "customers"


def test_gold_bridges_from_cardinality_relationships_respects_type_filter():
    report = {
        "cardinality_relationships": [
            {"left_table": "a", "right_table": "b", "join_columns": ["id"], "type": "1:1"},
            {"left_table": "a", "right_table": "c", "join_columns": ["region"], "type": "M:N"},
        ],
    }
    gold_all = gold_bridges_from_metadata_report(report)
    assert len(gold_all) == 2

    gold_filtered = gold_bridges_from_metadata_report(report, include_cardinality_types=["1:1", "1:N", "N:1"])
    assert len(gold_filtered) == 1
    assert gold_filtered[0].table_b == "b"


def test_gold_bridges_from_real_lifesciences_report_fixture():
    fixture_path = os.path.join(
        os.path.dirname(__file__), "..", "kg_optimizer", "fixtures", "lifesciences_metadata_report.json"
    )
    if not os.path.exists(fixture_path):
        return  # large real fixture — optional, skip if not present in this checkout
    with open(fixture_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    gold = gold_bridges_from_metadata_report(report)
    assert len(gold) == 378  # all from cardinality_relationships — no declared FKs/fk_candidates for this source
    assert all(isinstance(g, GoldBridge) for g in gold)
