"""
Run: python -m pytest tests/test_kg_optimizer_importance.py -v
"""
import random
from dataclasses import dataclass

from kg_optimizer.genome import random_genome
from kg_optimizer.importance import gene_importance


@dataclass
class _FakeFitness:
    composite: float


@dataclass
class _FakeTrial:
    genome: dict
    fitness: _FakeFitness


def _make_trials(n: int, rng: random.Random):
    trials = []
    for _ in range(n):
        g = random_genome(rng)
        # graphrag_top_k drives the score deterministically; everything else is noise.
        composite = g["graphrag_top_k"] / 16.0 + rng.uniform(-0.01, 0.01)
        trials.append(_FakeTrial(genome=g, fitness=_FakeFitness(composite=composite)))
    return trials


def test_gene_importance_empty_below_min_trials():
    rng = random.Random(0)
    trials = _make_trials(5, rng)
    assert gene_importance(trials, min_trials=10) == {}


def test_gene_importance_ranks_dominant_gene_highest():
    rng = random.Random(0)
    trials = _make_trials(60, rng)
    importance = gene_importance(trials, min_trials=10)
    assert importance, "expected a non-empty importance mapping with enough trials"
    top_gene = next(iter(importance))
    assert top_gene == "graphrag_top_k"


def test_gene_importance_normalizes_to_one():
    rng = random.Random(0)
    trials = _make_trials(60, rng)
    importance = gene_importance(trials, min_trials=10)
    assert abs(sum(importance.values()) - 1.0) < 1e-6
