"""
Composite fitness scoring + Pareto-front extraction.

The GA (ga.py) selects on the scalar `composite` score (simple, robust tournament
selection). Pareto-front extraction (quality vs. cost) is a reporting utility run
over the full population history at the end, via DEAP's NSGA-II ranking —
independent of what drove selection during the run.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from kg_optimizer.config import FitnessWeights
from kg_optimizer.datasets import GoldBridge
from kg_optimizer.eval_harness import DatasetRunResult
from kg_optimizer.judge import score_answer


@dataclass
class FitnessBreakdown:
    composite: float
    answer_quality: float   # mean judge score, normalized to 0-1
    bridge_f1: float         # 0-1, or None-equivalent (1.0, no_signal=True) if no gold bridges
    cost_tokens_est: float
    latency_s: float
    no_gold_bridges: bool = False
    consistency: float = 1.0      # naming resolution: same concept -> same label (1.0 = no signal)
    distinctiveness: float = 1.0  # naming resolution: different concepts -> different labels (1.0 = no signal)


def _bridge_key(table: str, col: str) -> Tuple[str, str]:
    return (table.strip().lower(), col.strip().lower())


def bridge_precision_recall(predicted_bridges: List[Any], gold_bridges: List[GoldBridge]) -> Tuple[float, float, float]:
    """predicted_bridges: list of dialog_agent.kg_bridges.Bridge (or duck-typed
    objects with from_entity/from_column/to_entity/to_column)."""
    if not gold_bridges:
        return 1.0, 1.0, 1.0  # no signal — caller should treat as no_gold_bridges

    positive_gold = {
        (_bridge_key(g.table_a, g.col_a), _bridge_key(g.table_b, g.col_b))
        for g in gold_bridges if g.expected
    }
    negative_gold = {
        (_bridge_key(g.table_a, g.col_a), _bridge_key(g.table_b, g.col_b))
        for g in gold_bridges if not g.expected
    }

    predicted = set()
    for b in predicted_bridges:
        predicted.add((_bridge_key(b.from_entity, b.from_column), _bridge_key(b.to_entity, b.to_column)))
        predicted.add((_bridge_key(b.to_entity, b.to_column), _bridge_key(b.from_entity, b.from_column)))

    tp = len(predicted & positive_gold)
    fp = len(predicted & negative_gold)
    fn = len(positive_gold - predicted)

    precision = tp / (tp + fp) if (tp + fp) else (1.0 if not fp else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _norm_label(label: str) -> str:
    return " ".join(label.strip().lower().split())


def consistency_score(resolved_labels: Dict[str, str], link_pairs: List[Tuple[str, str]]) -> float:
    """Fraction of link_pairs (names known/expected to be the same concept, e.g.
    FK-linked columns) whose resolved labels match after normalization. 1.0 (no
    signal) when there are no link_pairs to check."""
    if not link_pairs:
        return 1.0
    matches = 0
    for a, b in link_pairs:
        if _norm_label(resolved_labels.get(a, a)) == _norm_label(resolved_labels.get(b, b)):
            matches += 1
    return matches / len(link_pairs)


def distinctiveness_score(resolved_labels: Dict[str, str], link_pairs: List[Tuple[str, str]]) -> float:
    """1 - collision rate among name pairs that are NOT in link_pairs but ended up
    with the same resolved label (different concepts collapsing onto one label).
    1.0 (no signal) when there are fewer than 2 names or all pairs are linked."""
    names = list(resolved_labels.keys())
    n = len(names)
    if n < 2:
        return 1.0

    linked_set = {frozenset((a, b)) for a, b in link_pairs}

    groups: Dict[str, List[str]] = {}
    for name, label in resolved_labels.items():
        groups.setdefault(_norm_label(label), []).append(name)

    collisions = 0
    for group in groups.values():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if frozenset((group[i], group[j])) not in linked_set:
                    collisions += 1

    total_unlinked_pairs = n * (n - 1) // 2 - len(linked_set)
    if total_unlinked_pairs <= 0:
        return 1.0
    return 1.0 - (collisions / total_unlinked_pairs)


def answer_quality_score(run_result: DatasetRunResult, judge_model: str) -> float:
    """Mean judge score across questions, normalized from [1,5] to [0,1]."""
    if not run_result.question_results:
        return 0.0
    scores = []
    for qr in run_result.question_results:
        raw = score_answer(
            question=qr.question, answer=qr.answer, gold_answer=qr.gold_answer,
            had_error=qr.error is not None, sql_count=qr.sql_count, model=judge_model,
        )
        scores.append((raw - 1.0) / 4.0)  # [1,5] -> [0,1]
    return sum(scores) / len(scores)


def compute_fitness(run_result: DatasetRunResult, predicted_bridges: List[Any],
                    gold_bridges: List[GoldBridge], build_time_s: float, infer_time_s: float,
                    weights: FitnessWeights, judge_model: str,
                    cost_tokens_est: float = 0.0,
                    resolved_labels: Optional[Dict[str, str]] = None,
                    naming_link_pairs: Optional[List[Tuple[str, str]]] = None) -> FitnessBreakdown:
    quality = answer_quality_score(run_result, judge_model)
    no_gold = not gold_bridges
    _, _, f1 = bridge_precision_recall(predicted_bridges, gold_bridges)

    latency_s = build_time_s + infer_time_s + run_result.total_time_s
    # normalize cost/latency into rough 0-1 penalty bands so weights are comparable
    cost_penalty = min(cost_tokens_est / 200_000.0, 1.0)
    latency_penalty = min(latency_s / 600.0, 1.0)

    if no_gold:
        composite = weights.answer_quality * quality - weights.cost * cost_penalty - weights.latency * latency_penalty
    else:
        composite = (
            weights.answer_quality * quality
            + weights.bridge_f1 * f1
            - weights.cost * cost_penalty
            - weights.latency * latency_penalty
        )

    # Naming-resolution terms are only included in the composite when a resolved
    # label mapping is actually supplied (i.e. enable_name_resolution was on for this
    # trial) — this keeps runs with the feature disabled bit-for-bit identical to the
    # pre-existing composite formula.
    consistency = 1.0
    distinctiveness = 1.0
    if resolved_labels is not None:
        link_pairs = naming_link_pairs or []
        consistency = consistency_score(resolved_labels, link_pairs)
        distinctiveness = distinctiveness_score(resolved_labels, link_pairs)
        composite += weights.consistency * consistency + weights.distinctiveness * distinctiveness

    return FitnessBreakdown(
        composite=composite, answer_quality=quality, bridge_f1=f1,
        cost_tokens_est=cost_tokens_est, latency_s=latency_s, no_gold_bridges=no_gold,
        consistency=consistency, distinctiveness=distinctiveness,
    )


def pareto_front(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """entries: [{"genome": ..., "quality": float, "cost": float, ...}, ...]
    Returns the subset that is Pareto-optimal (maximize quality, minimize cost),
    ranked via DEAP's NSGA-II non-dominated sort."""
    if not entries:
        return []
    from deap import base, creator, tools

    if not hasattr(creator, "_KGOptFitnessParetoReport"):
        creator.create("_KGOptFitnessParetoReport", base.Fitness, weights=(1.0, -1.0))
    if not hasattr(creator, "_KGOptIndividualParetoReport"):
        creator.create("_KGOptIndividualParetoReport", list, fitness=creator._KGOptFitnessParetoReport)

    individuals = []
    for i, e in enumerate(entries):
        ind = creator._KGOptIndividualParetoReport([i])
        ind.fitness.values = (e["quality"], e["cost"])
        individuals.append(ind)

    fronts = tools.sortNondominated(individuals, len(individuals), first_front_only=True)
    first_front_idx = {ind[0] for ind in fronts[0]}
    return [entries[i] for i in sorted(first_front_idx)]
