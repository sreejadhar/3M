"""
Component ablation on the GA's champion genome — disable one tier at a time,
re-evaluate, report the fitness delta. Ties directly to real tiers in
dialog_agent.kg_inference_engine and the ontology/profiling toggles, not
hypothetical ones.

Statistical rigor (n_repeats > 1): re-evaluates the champion and each ablation
variant multiple times and reports mean +/- a bootstrap 95% CI, plus a paired
Wilcoxon signed-rank test against the champion's samples, so a fitness drop
can be told apart from noise from LLM-judge variance (fitness.answer_quality
comes from an LLM judge — see kg_optimizer.judge — which is not deterministic
across repeats even for the same genome).
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from kg_optimizer.cache import BuildCache
from kg_optimizer.config import OptimizerConfig
from kg_optimizer.datasets import EvalDataset
from kg_optimizer.fitness import FitnessBreakdown
from kg_optimizer.ga import evaluate_genome
from kg_optimizer.genome import Genome

# name -> gene overrides that disable that component
ABLATIONS: Dict[str, Dict[str, Any]] = {
    "no_tier3_llm_validation": {"enable_tier3_llm_validation": False},
    "no_value_overlap_tier":   {"enable_value_overlap_tier": False},
    "no_transitivity":         {"transitivity_min_conf": 1.01},  # effectively disables transitive closure
    "no_profiling":            {"profile_enabled": False},
    "no_concept_annotation":   {"annotate_concepts": False},
}

# The exact two-sided Wilcoxon signed-rank test's smallest achievable
# p-value is 2 * (1/2)**n — for n=5 that floor is 0.0625, which can NEVER
# clear the conventional 0.05 threshold no matter how large or consistent
# the drop is. n=6 (floor 0.03125) is the smallest n where significance is
# even mathematically possible, so that's the minimum we'll compute a
# p-value for at all — anything smaller is reported as None rather than a
# number that can't mean what a normal p-value means.
MIN_REPEATS_FOR_SIGNIFICANCE = 6


@dataclass
class AblationRow:
    name: str
    fitness: FitnessBreakdown           # first repeat's full breakdown, for backward-compat display
    delta_composite: float              # mean_composite - champion's mean composite
    mean_composite: float
    std_composite: float
    ci95: Tuple[float, float]           # bootstrap 95% CI of the mean composite
    n_repeats: int
    p_value: Optional[float] = None     # paired Wilcoxon vs. champion samples; None if not computed
    significant: Optional[bool] = None  # p_value < 0.05; None if p_value wasn't computed


def _bootstrap_ci(samples: List[float], rng: random.Random, n_boot: int = 2000,
                  alpha: float = 0.05) -> Tuple[float, float]:
    n = len(samples)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (samples[0], samples[0])
    means = []
    for _ in range(n_boot):
        resample = [samples[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(int((1 - alpha / 2) * n_boot), n_boot - 1)]
    return (lo, hi)


def run_ablations(champion: Genome, champion_fitness: FitnessBreakdown, dataset: EvalDataset,
                  ontology_report: Dict[str, Any], cfg: OptimizerConfig, source_domain: str = "",
                  evaluate_fn: Optional[Callable[[Genome], FitnessBreakdown]] = None,
                  n_repeats: int = 1, random_seed: Optional[int] = None) -> List[AblationRow]:
    """
    n_repeats=1 (default) reproduces the original single-evaluation behavior
    (mean == that one sample, ci95 collapses to a point, p_value/significant
    stay None). Pass n_repeats>=5 for statistically meaningful ablation
    deltas — each repeat re-runs the real eval_harness/judge pipeline, so
    cost scales linearly with n_repeats; budget LLM-judge calls accordingly.
    """
    cache = BuildCache(cfg.cache_dir)
    if evaluate_fn is None:
        evaluate_fn = lambda g: evaluate_genome(g, dataset, ontology_report, cfg, cache, source_domain)

    rng = random.Random(random_seed if random_seed is not None else cfg.random_seed)

    champion_samples = [champion_fitness.composite]
    champion_samples.extend(evaluate_fn(champion).composite for _ in range(max(n_repeats, 1) - 1))
    champion_mean = statistics.mean(champion_samples)

    rows: List[AblationRow] = []
    for name, overrides in ABLATIONS.items():
        variant = dict(champion)
        variant.update(overrides)

        samples: List[float] = []
        first_fitness: Optional[FitnessBreakdown] = None
        for _ in range(max(n_repeats, 1)):
            fitness = evaluate_fn(variant)
            if first_fitness is None:
                first_fitness = fitness
            samples.append(fitness.composite)

        mean_composite = statistics.mean(samples)
        std_composite = statistics.pstdev(samples) if len(samples) > 1 else 0.0
        ci95 = _bootstrap_ci(samples, rng)

        p_value: Optional[float] = None
        significant: Optional[bool] = None
        if n_repeats >= MIN_REPEATS_FOR_SIGNIFICANCE and len(samples) == len(champion_samples):
            try:
                from scipy.stats import wilcoxon
                _, p_value = wilcoxon(champion_samples, samples)
                significant = bool(p_value < 0.05)
            except (ImportError, ValueError):
                p_value = None
                significant = None

        rows.append(AblationRow(
            name=name, fitness=first_fitness,
            delta_composite=mean_composite - champion_mean,
            mean_composite=mean_composite, std_composite=std_composite, ci95=ci95,
            n_repeats=n_repeats, p_value=p_value, significant=significant,
        ))
    return sorted(rows, key=lambda r: r.delta_composite)
