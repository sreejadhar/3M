"""
Run: python -m pytest tests/test_kg_optimizer_ablation_stats.py -v
"""
from kg_optimizer.ablation import ABLATIONS, MIN_REPEATS_FOR_SIGNIFICANCE, run_ablations
from kg_optimizer.config import OptimizerConfig
from kg_optimizer.fitness import FitnessBreakdown
from kg_optimizer.genome import random_genome
import random


def _fitness(composite: float) -> FitnessBreakdown:
    return FitnessBreakdown(
        composite=composite, answer_quality=composite, bridge_f1=1.0,
        cost_tokens_est=0.0, latency_s=0.0,
    )


def test_min_repeats_for_significance_is_wilcoxons_smallest_viable_n():
    # Exact two-sided Wilcoxon signed-rank floor for n paired samples is
    # 2 * (1/2)**n == 0.5**(n-1). n=5 -> 0.0625 (can never clear 0.05);
    # n=6 -> 0.03125 (can). The constant must be 6, not 5 — this pins that
    # down so nobody "fixes" it back to 5 later.
    def floor(n):
        return 0.5 ** (n - 1)

    assert floor(MIN_REPEATS_FOR_SIGNIFICANCE) < 0.05
    assert floor(MIN_REPEATS_FOR_SIGNIFICANCE - 1) >= 0.05


def test_single_repeat_matches_original_behavior():
    cfg = OptimizerConfig(random_seed=0)
    champion = random_genome(random.Random(0))
    champion_fitness = _fitness(0.80)

    rows = run_ablations(
        champion, champion_fitness, dataset=None, ontology_report={}, cfg=cfg,
        evaluate_fn=lambda g: _fitness(0.60), n_repeats=1,
    )
    assert len(rows) == len(ABLATIONS)
    for row in rows:
        assert row.n_repeats == 1
        assert row.mean_composite == 0.60
        assert row.std_composite == 0.0
        assert row.ci95 == (0.60, 0.60)
        assert row.p_value is None
        assert row.significant is None
        assert abs(row.delta_composite - (0.60 - 0.80)) < 1e-9


def test_multi_repeat_computes_significance_for_real_drop():
    cfg = OptimizerConfig(random_seed=0)
    # Explicit genome (not random) so every ABLATIONS override is guaranteed
    # to actually change the genome — a randomly drawn bool could already
    # match an override's target value and make that variant == champion.
    champion = {
        "annotate_concepts": True, "ontology_llm_model": "claude-haiku-4-5",
        "profile_enabled": True, "profile_llm_model": "claude-haiku-4-5",
        "embed_backend": "sentence-transformers", "enable_name_resolution": False,
        "min_confidence": 0.5, "auto_enable_threshold": 0.8, "cross_domain_penalty": 0.1,
        "embedding_sim_threshold": 0.7, "value_overlap_threshold": 0.6,
        "transitivity_decay": 0.9, "transitivity_min_conf": 0.5,
        "enable_tier3_llm_validation": True, "enable_value_overlap_tier": True,
        "graphrag_top_k": 8,
    }
    champion_fitness = _fitness(0.90)

    # champion re-evals return 0.90 (identity check), every ablation variant
    # (which differs from champion by construction) returns 0.40 — a real,
    # non-noisy drop every repeat.
    def evaluate_fn(genome):
        return _fitness(0.90) if genome == champion else _fitness(0.40)

    rows = run_ablations(
        champion, champion_fitness, dataset=None, ontology_report={}, cfg=cfg,
        evaluate_fn=evaluate_fn,
        n_repeats=MIN_REPEATS_FOR_SIGNIFICANCE,
    )
    assert len(rows) == len(ABLATIONS)
    for row in rows:
        assert row.n_repeats == MIN_REPEATS_FOR_SIGNIFICANCE
        assert abs(row.mean_composite - 0.40) < 1e-9
        assert row.p_value is not None
        assert row.significant is True
        # Tiny float-rounding slack: ci95 bounds are a bootstrap resample mean
        # of identical-valued samples, not bit-identical to mean_composite.
        assert row.ci95[0] - 1e-9 <= row.mean_composite <= row.ci95[1] + 1e-9


def test_ablations_sorted_by_delta_ascending():
    cfg = OptimizerConfig(random_seed=0)
    champion = random_genome(random.Random(0))
    champion_fitness = _fitness(0.80)

    scores_by_name = {name: 0.10 * i for i, name in enumerate(ABLATIONS)}

    def evaluate_fn(genome):
        for name, overrides in ABLATIONS.items():
            if all(genome.get(k) == v for k, v in overrides.items()):
                return _fitness(scores_by_name[name])
        return _fitness(0.80)

    rows = run_ablations(
        champion, champion_fitness, dataset=None, ontology_report={}, cfg=cfg,
        evaluate_fn=evaluate_fn, n_repeats=1,
    )
    deltas = [r.delta_composite for r in rows]
    assert deltas == sorted(deltas)
