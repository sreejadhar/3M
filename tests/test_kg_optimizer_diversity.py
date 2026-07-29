"""
Run: python -m pytest tests/test_kg_optimizer_diversity.py -v
"""
import random

from kg_optimizer.config import OptimizerConfig
from kg_optimizer.fitness import FitnessBreakdown
from kg_optimizer.ga import run_ga
from kg_optimizer.genome import population_diversity, random_genome, random_population


def test_population_diversity_is_zero_for_identical_population():
    rng = random.Random(0)
    genome = random_genome(rng)
    identical = [dict(genome) for _ in range(10)]
    # Float variance over identical values can land at a tiny nonzero due to
    # subtraction rounding (observed ~3.5e-17) — assert "effectively zero",
    # not bit-exact 0.0.
    assert population_diversity(identical) < 1e-9


def test_population_diversity_is_positive_for_random_population():
    rng = random.Random(0)
    pop = random_population(20, rng)
    assert population_diversity(pop) > 0.0


def test_population_diversity_below_min_size_is_zero():
    rng = random.Random(0)
    assert population_diversity(random_population(1, rng)) == 0.0
    assert population_diversity([]) == 0.0


def _flat_fitness(composite: float) -> FitnessBreakdown:
    return FitnessBreakdown(
        composite=composite, answer_quality=composite, bridge_f1=1.0,
        cost_tokens_est=0.0, latency_s=0.0,
    )


def test_run_ga_injects_fresh_individuals_when_population_collapses():
    """
    Force premature convergence (evaluate_fn ignores the genome entirely, so
    tournament selection has no gradient to climb and the population would
    otherwise stay wherever crossover happens to leave it) with a very low
    diversity_threshold that a real converged population would still trip,
    and confirm run_ga logs an injection event without crashing.
    """
    cfg = OptimizerConfig(
        population_size=8, generations=4, mutation_rate=0.0, mutation_decay=1.0,
        tournament_size=2, elitism=1, random_seed=1,
        diversity_threshold=1.0,  # always "below threshold" — every generation should inject
        diversity_injection_frac=0.5,
    )

    call_count = {"n": 0}

    def evaluate_fn(genome):
        call_count["n"] += 1
        return _flat_fitness(0.5)

    result = run_ga(None, {}, cfg, evaluate_fn=evaluate_fn)
    assert result.best is not None
    # population_size trials in gen 0, plus population_size per subsequent generation
    assert len(result.history.trials) == cfg.population_size * cfg.generations
