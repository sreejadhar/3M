"""
Run: python -m pytest tests/test_kg_optimizer_fitness.py -v
Pure fitness-math tests — no live services, no network (all questions use
gold_answer=None so scoring stays on the no-gold heuristic path).
"""
from kg_optimizer.config import FitnessWeights
from kg_optimizer.datasets import GoldBridge
from kg_optimizer.eval_harness import DatasetRunResult, QuestionRunResult
from kg_optimizer.fitness import bridge_precision_recall, compute_fitness, pareto_front


class _FakeBridge:
    def __init__(self, from_entity, from_column, to_entity, to_column):
        self.from_entity = from_entity
        self.from_column = from_column
        self.to_entity = to_entity
        self.to_column = to_column


def test_bridge_precision_recall_perfect_match():
    gold = [GoldBridge(table_a="orders", col_a="customer_id", table_b="customers", col_b="id", expected=True)]
    predicted = [_FakeBridge("orders", "customer_id", "customers", "id")]
    precision, recall, f1 = bridge_precision_recall(predicted, gold)
    assert precision == 1.0
    assert recall == 1.0
    assert f1 == 1.0


def test_bridge_precision_recall_missed_bridge():
    gold = [GoldBridge(table_a="orders", col_a="customer_id", table_b="customers", col_b="id", expected=True)]
    precision, recall, f1 = bridge_precision_recall([], gold)
    assert recall == 0.0
    assert f1 == 0.0


def test_bridge_precision_recall_false_positive_penalized():
    gold = [GoldBridge(table_a="orders", col_a="customer_id", table_b="customers", col_b="id", expected=False)]
    predicted = [_FakeBridge("orders", "customer_id", "customers", "id")]
    precision, recall, f1 = bridge_precision_recall(predicted, gold)
    assert precision == 0.0


def test_bridge_precision_recall_no_gold_returns_neutral_signal():
    precision, recall, f1 = bridge_precision_recall([_FakeBridge("a", "b", "c", "d")], [])
    assert (precision, recall, f1) == (1.0, 1.0, 1.0)


def _make_run_result(errors=0, ok=2):
    results = [QuestionRunResult(question=f"q{i}", answer="some answer", sql_count=1) for i in range(ok)]
    results += [QuestionRunResult(question=f"err{i}", error="boom") for i in range(errors)]
    return DatasetRunResult(question_results=results, total_time_s=10.0)


def test_compute_fitness_no_gold_bridges_excludes_bridge_term():
    run_result = _make_run_result(ok=3, errors=0)
    weights = FitnessWeights(answer_quality=1.0, bridge_f1=0.5, cost=0.0, latency=0.0)
    fitness = compute_fitness(
        run_result=run_result, predicted_bridges=[], gold_bridges=[],
        build_time_s=1.0, infer_time_s=1.0, weights=weights, judge_model="claude-haiku-4-5",
    )
    assert fitness.no_gold_bridges is True
    # composite should equal weighted answer_quality only (bridge term skipped)
    assert abs(fitness.composite - weights.answer_quality * fitness.answer_quality) < 1e-9


def test_compute_fitness_penalizes_errors():
    good = _make_run_result(ok=3, errors=0)
    bad = _make_run_result(ok=0, errors=3)
    weights = FitnessWeights(answer_quality=1.0, bridge_f1=0.0, cost=0.0, latency=0.0)
    fit_good = compute_fitness(good, [], [], 1.0, 1.0, weights, "claude-haiku-4-5")
    fit_bad = compute_fitness(bad, [], [], 1.0, 1.0, weights, "claude-haiku-4-5")
    assert fit_good.composite > fit_bad.composite


def test_pareto_front_keeps_only_nondominated_entries():
    entries = [
        {"quality": 0.9, "cost": 0.8},   # dominated by nothing better on both axes simultaneously
        {"quality": 0.5, "cost": 0.2},   # cheap, lower quality — non-dominated tradeoff
        {"quality": 0.3, "cost": 0.9},   # dominated: entry 0 has higher quality AND lower cost
    ]
    front = pareto_front(entries)
    fronts_as_tuples = {(e["quality"], e["cost"]) for e in front}
    assert (0.3, 0.9) not in fronts_as_tuples
    assert (0.9, 0.8) in fronts_as_tuples
    assert (0.5, 0.2) in fronts_as_tuples
