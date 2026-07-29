"""
Synthetic demo run — exercises the REAL genome/GA/fitness/ablation/Pareto code
(kg_optimizer.genome, .ga, .fitness, .ablation) with a synthetic evaluate_fn
standing in for build_runner/eval_harness/judge, which need a built KG +
dialog API (port 8003) + LLM calls. Not part of the production pipeline —
this exists purely to demonstrate the GA mechanics without the local stack
running. See kg_optimizer/cli.py for the real entrypoint.

Run: python -m kg_optimizer.demo_synthetic_run
"""
from __future__ import annotations

import hashlib
import json
import math
import random

from kg_optimizer.ablation import run_ablations
from kg_optimizer.config import FitnessWeights, OptimizerConfig
from kg_optimizer.fitness import FitnessBreakdown, pareto_front
from kg_optimizer.ga import run_ga
from kg_optimizer.genome import Genome

WEIGHTS = FitnessWeights(answer_quality=1.0, bridge_f1=0.5, cost=0.4, latency=0.15)


def _genome_rng(genome: Genome) -> random.Random:
    # Deterministic per-genome noise (stands in for LLM-judge stochasticity)
    blob = json.dumps(genome, sort_keys=True, default=str).encode()
    seed = int(hashlib.sha256(blob).hexdigest()[:8], 16)
    return random.Random(seed)


def synthetic_quality(genome: Genome, rng: random.Random) -> float:
    # Plausible landscape: a few genes have a sweet spot, others are monotonic
    def bump(x, center, width):
        return math.exp(-((x - center) ** 2) / (2 * width ** 2))

    score = 0.35
    score += 0.20 * bump(genome["embedding_sim_threshold"], 0.75, 0.12)
    score += 0.15 * bump(genome["min_confidence"], 0.50, 0.10)
    score += 0.10 * bump(genome["auto_enable_threshold"], 0.80, 0.08)
    score += 0.08 * bump(genome["graphrag_top_k"], 8, 3)
    score += 0.06 if genome["annotate_concepts"] else 0.0
    score += 0.05 if genome["profile_enabled"] else 0.0
    score += 0.07 if genome["enable_tier3_llm_validation"] else -0.02
    score += 0.03 if genome["enable_value_overlap_tier"] else 0.0
    score += rng.gauss(0, 0.03)
    return max(0.0, min(1.0, score))


def synthetic_cost(genome: Genome) -> float:
    cost = 0.10
    if genome["annotate_concepts"]:
        cost += 0.15
    if genome["profile_enabled"]:
        cost += 0.15
    if genome["enable_tier3_llm_validation"]:
        cost += 0.25
    if genome["embed_backend"] == "openai":
        cost += 0.20
    if genome["enable_value_overlap_tier"]:
        cost += 0.10
    if genome["ontology_llm_model"] == "claude-sonnet-5":
        cost += 0.15
    if genome["profile_llm_model"] == "claude-sonnet-5":
        cost += 0.15
    return cost


def synthetic_evaluate(genome: Genome) -> FitnessBreakdown:
    rng = _genome_rng(genome)
    quality = synthetic_quality(genome, rng)
    cost = synthetic_cost(genome)
    latency = 0.20 + cost * 0.5 + rng.gauss(0, 0.02)
    bridge_f1 = max(0.0, min(1.0, 0.6 + 0.3 * bump_helper(genome) + rng.gauss(0, 0.05)))

    cost_penalty = min(cost, 1.0)
    latency_penalty = min(latency, 1.0)
    composite = (
        WEIGHTS.answer_quality * quality
        + WEIGHTS.bridge_f1 * bridge_f1
        - WEIGHTS.cost * cost_penalty
        - WEIGHTS.latency * latency_penalty
    )
    return FitnessBreakdown(
        composite=composite, answer_quality=quality, bridge_f1=bridge_f1,
        cost_tokens_est=cost * 200_000, latency_s=latency * 300, no_gold_bridges=False,
    )


def bump_helper(genome: Genome) -> float:
    x = genome["transitivity_decay"]
    return math.exp(-((x - 0.85) ** 2) / (2 * 0.08 ** 2))


def main() -> None:
    cfg = OptimizerConfig(
        source_id="synthetic_demo",
        population_size=16,
        generations=12,
        mutation_rate=0.25,
        mutation_decay=0.92,
        tournament_size=3,
        elitism=2,
        random_seed=7,
        fitness_weights=WEIGHTS,
    )

    ga_result = run_ga(dataset=None, ontology_report=None, cfg=cfg, evaluate_fn=synthetic_evaluate)
    champion = ga_result.best

    print(f"Champion composite fitness: {champion.fitness.composite:.4f}")
    print(f"Champion genome: {json.dumps(champion.genome, indent=2, default=str)}")

    ablation_rows = run_ablations(
        champion.genome, champion.fitness, dataset=None, ontology_report=None,
        cfg=cfg, evaluate_fn=synthetic_evaluate,
    )
    print("\nAblation deltas (composite fitness change when disabling each component):")
    for row in ablation_rows:
        print(f"  {row.name:28s} delta={row.delta_composite:+.4f}")

    # ── Per-generation summary for the convergence chart ──────────────────────
    by_gen = {}
    for t in ga_result.history.trials:
        by_gen.setdefault(t.generation, []).append(t.fitness.composite)
    generations = sorted(by_gen)
    best_per_gen = [max(by_gen[g]) for g in generations]
    mean_per_gen = [sum(by_gen[g]) / len(by_gen[g]) for g in generations]

    pareto_entries = ga_result.history.as_pareto_entries()
    front = pareto_front(pareto_entries)
    front_keys = {(round(e["quality"], 6), round(e["cost"], 6)) for e in front}

    with open("kg_optimizer_demo_result.json", "w", encoding="utf-8") as f:
        json.dump({
            "champion_genome": champion.genome,
            "champion_fitness": champion.fitness.__dict__,
            "ablations": [{"name": r.name, "delta_composite": r.delta_composite} for r in ablation_rows],
            "generations": generations,
            "best_per_gen": best_per_gen,
            "mean_per_gen": mean_per_gen,
            "pareto_entries": pareto_entries,
            "front_keys": list(front_keys),
        }, f, indent=2, default=str)
    print("\nWrote kg_optimizer_demo_result.json")


if __name__ == "__main__":
    main()
