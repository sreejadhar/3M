"""
GA loop: tournament selection + elitism over the scalar composite fitness
(kg_optimizer.fitness.compute_fitness), mixed-type crossover/mutation from
kg_optimizer.genome. eaMuPlusLambda-style: each generation replaces the
non-elite population with offspring.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from kg_optimizer.build_runner import get_or_build_kg, run_bridge_inference
from kg_optimizer.cache import BuildCache
from kg_optimizer.config import OptimizerConfig
from kg_optimizer.datasets import EvalDataset
from kg_optimizer.eval_harness import run_dataset
from kg_optimizer.fitness import FitnessBreakdown, compute_fitness
from kg_optimizer.genome import (
    Genome, GENOME_SPEC, crossover, mutate, population_diversity, random_population,
)

logger = logging.getLogger(__name__)


@dataclass
class TrialResult:
    genome: Genome
    fitness: FitnessBreakdown
    generation: int
    was_cached_build: bool


@dataclass
class GAHistory:
    trials: List[TrialResult] = field(default_factory=list)

    def as_pareto_entries(self) -> List[Dict[str, Any]]:
        return [
            {
                "genome": t.genome,
                "quality": t.fitness.answer_quality,
                "cost": t.fitness.cost_tokens_est,
                "composite": t.fitness.composite,
            }
            for t in self.trials
        ]


def evaluate_genome(genome: Genome, dataset: EvalDataset, ontology_report: Dict[str, Any],
                   cfg: OptimizerConfig, cache: BuildCache, source_domain: str = "") -> FitnessBreakdown:
    kg_build = get_or_build_kg(genome, ontology_report, cfg, cache, source_domain)
    bridges = run_bridge_inference(genome, kg_build, ontology_report, domain=source_domain)
    run_result = run_dataset(dataset, kg_build, cfg)
    predicted = bridges.high_conf + bridges.medium_conf

    resolved_labels = None
    naming_link_pairs = None
    if genome.get("enable_name_resolution"):
        from kg_optimizer.naming_resolver import resolve_names

        # Cache hit inside resolve_names — build_ontology_and_kg already computed
        # this once for the same (source_id, schema, model) when it built kg_build.
        naming = resolve_names(
            ontology_report, source_id=cfg.source_id, domain=source_domain,
            llm_model=genome["ontology_llm_model"],
        )
        # Table and column names are separate namespaces — only column labels
        # feed the composite score here since link_pairs (and the collisions
        # distinctiveness_score checks for) are column-scoped; table-level
        # naming quality still reaches the ontology output via build_node.py,
        # it's just not double-counted in this outer fitness signal.
        resolved_labels = naming.column_labels
        naming_link_pairs = naming.link_pairs

    return compute_fitness(
        run_result=run_result, predicted_bridges=predicted, gold_bridges=dataset.gold_bridges,
        build_time_s=kg_build.build_time_s, infer_time_s=bridges.infer_time_s,
        weights=cfg.fitness_weights, judge_model=cfg.judge_model,
        resolved_labels=resolved_labels, naming_link_pairs=naming_link_pairs,
    )


def run_ga(dataset: EvalDataset, ontology_report: Dict[str, Any], cfg: OptimizerConfig,
          source_domain: str = "",
          evaluate_fn: Optional[Callable[[Genome], FitnessBreakdown]] = None) -> "GAResult":
    rng = random.Random(cfg.random_seed)
    cache = BuildCache(cfg.cache_dir)
    history = GAHistory()

    if evaluate_fn is None:
        evaluate_fn = lambda g: evaluate_genome(g, dataset, ontology_report, cfg, cache, source_domain)

    population = random_population(cfg.population_size, rng)
    mutation_rate = cfg.mutation_rate

    def score_all(pop: List[Genome], generation: int) -> List[TrialResult]:
        results = []
        for g in pop:
            fitness = evaluate_fn(g)
            results.append(TrialResult(genome=g, fitness=fitness, generation=generation, was_cached_build=False))
        return results

    scored = score_all(population, generation=0)
    history.trials.extend(scored)

    for gen in range(1, cfg.generations):
        scored.sort(key=lambda t: t.fitness.composite, reverse=True)
        elites = [t.genome for t in scored[: cfg.elitism]]

        diversity = population_diversity([t.genome for t in scored])

        offspring: List[Genome] = list(elites)
        while len(offspring) < cfg.population_size:
            p1 = _tournament_select(scored, cfg.tournament_size, rng)
            p2 = _tournament_select(scored, cfg.tournament_size, rng)
            c1, c2 = crossover(p1, p2, rng)
            c1 = mutate(c1, rng, mutation_rate)
            c2 = mutate(c2, rng, mutation_rate)
            offspring.append(c1)
            if len(offspring) < cfg.population_size:
                offspring.append(c2)

        if diversity < cfg.diversity_threshold:
            n_inject = min(
                max(1, int((cfg.population_size - cfg.elitism) * cfg.diversity_injection_frac)),
                len(offspring) - cfg.elitism,
            )
            fresh = random_population(n_inject, rng)
            offspring[cfg.elitism: cfg.elitism + n_inject] = fresh
            logger.info(
                "Generation %d: diversity=%.3f below threshold=%.3f — injected %d fresh individual(s)",
                gen, diversity, cfg.diversity_threshold, n_inject,
            )

        scored = score_all(offspring, generation=gen)
        history.trials.extend(scored)
        mutation_rate *= cfg.mutation_decay
        logger.info(
            "Generation %d: best composite=%.4f diversity=%.3f",
            gen, max(t.fitness.composite for t in scored), diversity,
        )

    best = max(history.trials, key=lambda t: t.fitness.composite)
    return GAResult(best=best, history=history)


def _tournament_select(scored: List[TrialResult], k: int, rng: random.Random) -> Genome:
    contenders = rng.sample(scored, min(k, len(scored)))
    return max(contenders, key=lambda t: t.fitness.composite).genome


@dataclass
class GAResult:
    best: TrialResult
    history: GAHistory
