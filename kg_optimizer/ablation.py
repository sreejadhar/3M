"""
Component ablation on the GA's champion genome — disable one tier at a time,
re-evaluate, report the fitness delta. Ties directly to real tiers in
dialog_agent.kg_inference_engine and the ontology/profiling toggles, not
hypothetical ones.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

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


@dataclass
class AblationRow:
    name: str
    fitness: FitnessBreakdown
    delta_composite: float


def run_ablations(champion: Genome, champion_fitness: FitnessBreakdown, dataset: EvalDataset,
                  ontology_report: Dict[str, Any], cfg: OptimizerConfig, source_domain: str = "",
                  evaluate_fn: Optional[Callable[[Genome], FitnessBreakdown]] = None) -> List[AblationRow]:
    cache = BuildCache(cfg.cache_dir)
    if evaluate_fn is None:
        evaluate_fn = lambda g: evaluate_genome(g, dataset, ontology_report, cfg, cache, source_domain)

    rows = []
    for name, overrides in ABLATIONS.items():
        variant = dict(champion)
        variant.update(overrides)
        fitness = evaluate_fn(variant)
        rows.append(AblationRow(
            name=name, fitness=fitness,
            delta_composite=fitness.composite - champion_fitness.composite,
        ))
    return sorted(rows, key=lambda r: r.delta_composite)
