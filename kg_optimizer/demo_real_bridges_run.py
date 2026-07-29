"""
REAL demo run — no dialog API. Uses an already-built KG snapshot
read straight from data/kg_store.db (the same store the KG-generation endpoint
persists into) via kg_optimizer.datasets.load_connection_from_kg_store, and
runs the GA over the cheap-tier (bridge-inference) genes using the REAL
dialog_agent.kg_inference_engine.run_enterprise_inference against that REAL
graph. No synthetic data anywhere in the fitness path.

Two things this environment can't do, worked around explicitly below:
  1. No KG rebuild here — build_runner.load_kg_snapshot() wraps the stored
     nodes/edges directly; expensive-tier (ontology/profiling/embedding)
     genes have no effect since nothing is rebuilt.
  2. No internet access to huggingface.co — Tier 2 semantic matching would
     otherwise retry a failed model download for 60-150s per trial. We force
     the existing TF-IDF fallback (see _build_embed_matrix in
     kg_inference_engine.py) by making `sentence_transformers` look
     unimportable in THIS process only — no repo file is touched.

No LLM calls happen anywhere in this run: run_enterprise_inference (the
scoring-only path GA trials use) never invokes Tier-3 LLM validation — that
only runs inside run_enterprise_inference_and_save, on the final persisted
save. So `enable_tier3_llm_validation` has no effect on trial fitness here;
this is a real, reproducible finding, not a demo artifact (see kg_inference_
engine.py:942, where run_tier3_llm is read, versus run_enterprise_inference's
body, which never reads it).

Fitness now uses REAL precision/recall against gold bridges derived from the
real metadata extraction report (kg_optimizer/fixtures/lifesciences_metadata_
report.json, pulled from kg_sources.report_json — see kg_optimizer.datasets.
gold_bridges_from_metadata_report). This source has no declared FKs and no
fk_candidates (both empty in the report), so gold comes entirely from
cardinality_relationships (378 value-overlap-detected joins). We restrict to
the "1:1"/"1:N"/"N:1" types (68 of the 378) as the confident gold set —
"M:N" relationships often reflect a shared categorical dimension (e.g. the
same region_name repeating across many tables) rather than a true join key,
and would otherwise dominate/distort recall.

Run: python -m kg_optimizer.demo_real_bridges_run
"""
from __future__ import annotations

import json
import sys
import time

sys.modules.setdefault("sentence_transformers", None)  # force TF-IDF fallback — no HF Hub access here

from kg_optimizer.ablation import run_ablations
from kg_optimizer.build_runner import load_kg_snapshot, run_bridge_inference
from kg_optimizer.config import FitnessWeights, OptimizerConfig
from kg_optimizer.datasets import gold_bridges_from_metadata_report, load_connection_from_kg_store
from kg_optimizer.fitness import FitnessBreakdown, bridge_precision_recall, pareto_front
from kg_optimizer.ga import run_ga
from kg_optimizer.genome import Genome

LS_SOURCE_ID = "ec94dc92-2c1c-43bd-9f0f-d73b64b2b159"  # LIFESCIENCESTESTUSECASE
REPORT_PATH = "kg_optimizer/fixtures/lifesciences_metadata_report.json"
CONFIDENT_CARDINALITY_TYPES = ["1:1", "1:N", "N:1"]

_cache: dict = {}


def _genome_key(genome: Genome) -> str:
    return json.dumps(genome, sort_keys=True, default=str)


def make_evaluate_fn(kg_build, report, gold_bridges, weights: FitnessWeights):
    def evaluate(genome: Genome) -> FitnessBreakdown:
        key = _genome_key(genome)
        if key in _cache:
            return _cache[key]

        result = run_bridge_inference(genome, kg_build, report=report, domain="healthcare")
        predicted = result.high_conf + result.medium_conf
        precision, recall, f1 = bridge_precision_recall(predicted, gold_bridges)

        latency_penalty = min(result.infer_time_s / 30.0, 1.0)
        composite = weights.bridge_f1 * f1 - weights.latency * latency_penalty

        fitness = FitnessBreakdown(
            composite=composite, answer_quality=0.0, bridge_f1=f1,
            cost_tokens_est=0.0, latency_s=result.infer_time_s, no_gold_bridges=False,
        )
        _cache[key] = fitness
        return fitness

    return evaluate


def main() -> None:
    print(f"Loading real KG snapshot for source_id={LS_SOURCE_ID} from data/kg_store.db ...")
    loaded = load_connection_from_kg_store(LS_SOURCE_ID)
    kg_build = load_kg_snapshot("lifesciences_snapshot", loaded["nodes"], loaded["edges"])
    print(f"Loaded real graph: {len(kg_build.nodes)} nodes, {len(kg_build.edges)} declared edges")

    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)
    all_gold = gold_bridges_from_metadata_report(report)
    gold_bridges = gold_bridges_from_metadata_report(report, include_cardinality_types=CONFIDENT_CARDINALITY_TYPES)
    print(f"Loaded real metadata report: {len(all_gold)} total cardinality-derived relationships, "
          f"{len(gold_bridges)} confident (1:1/1:N/N:1) gold bridges")

    weights = FitnessWeights(answer_quality=0.0, bridge_f1=1.0, cost=0.0, latency=0.3)
    cfg = OptimizerConfig(
        source_id="lifesciences_snapshot",
        population_size=8,
        generations=5,
        mutation_rate=0.25,
        mutation_decay=0.9,
        tournament_size=3,
        elitism=2,
        random_seed=11,
        fitness_weights=weights,
    )
    evaluate_fn = make_evaluate_fn(kg_build, report, gold_bridges, weights)

    t0 = time.time()
    ga_result = run_ga(dataset=None, ontology_report=None, cfg=cfg, evaluate_fn=evaluate_fn)
    champion = ga_result.best
    print(f"\nGA run took {time.time() - t0:.1f}s over {len(ga_result.history.trials)} real bridge-inference trials")
    print(f"Champion composite fitness: {champion.fitness.composite:.4f} (bridge F1 vs. real gold={champion.fitness.bridge_f1:.4f})")
    print(f"Champion genome: {json.dumps(champion.genome, indent=2, default=str)}")

    champion_bridges = run_bridge_inference(champion.genome, kg_build, report=report, domain="healthcare")
    champion_predicted = champion_bridges.high_conf + champion_bridges.medium_conf
    precision, recall, f1 = bridge_precision_recall(champion_predicted, gold_bridges)
    print(f"\nChampion found {len(champion_bridges.high_conf)} high-confidence + "
          f"{len(champion_bridges.medium_conf)} medium-confidence real candidate bridges")
    print(f"Real precision={precision:.3f} recall={recall:.3f} f1={f1:.3f} against {len(gold_bridges)} confident gold bridges")

    print("\nRunning ablations on champion (real re-inference per ablation)...")
    ablation_rows = run_ablations(
        champion.genome, champion.fitness, dataset=None, ontology_report=None,
        cfg=cfg, evaluate_fn=evaluate_fn,
    )
    for row in ablation_rows:
        print(f"  {row.name:28s} delta={row.delta_composite:+.4f}")

    by_gen = {}
    for t in ga_result.history.trials:
        by_gen.setdefault(t.generation, []).append(t.fitness.composite)
    generations = sorted(by_gen)
    best_per_gen = [max(by_gen[g]) for g in generations]
    mean_per_gen = [sum(by_gen[g]) / len(by_gen[g]) for g in generations]

    pareto_entries = [
        {"genome": t.genome, "quality": t.fitness.bridge_f1, "cost": t.fitness.latency_s, "composite": t.fitness.composite}
        for t in ga_result.history.trials
    ]
    front = pareto_front(pareto_entries)
    front_keys = [[round(e["quality"], 6), round(e["cost"], 6)] for e in front]

    with open("kg_optimizer_real_bridges_result.json", "w", encoding="utf-8") as f:
        json.dump({
            "source": "real LifeSciences KG snapshot (data/kg_store.db) + real metadata report gold bridges, no LLM calls",
            "nodes": len(kg_build.nodes),
            "declared_edges": len(kg_build.edges),
            "gold_bridges_total_cardinality": len(all_gold),
            "gold_bridges_confident": len(gold_bridges),
            "champion_genome": champion.genome,
            "champion_fitness": champion.fitness.__dict__,
            "champion_precision": precision,
            "champion_recall": recall,
            "champion_high_conf_bridges": len(champion_bridges.high_conf),
            "champion_medium_conf_bridges": len(champion_bridges.medium_conf),
            "ablations": [{"name": r.name, "delta_composite": r.delta_composite} for r in ablation_rows],
            "generations": generations,
            "best_per_gen": best_per_gen,
            "mean_per_gen": mean_per_gen,
            "pareto_entries": pareto_entries,
            "front_keys": front_keys,
        }, f, indent=2, default=str)
    print("\nWrote kg_optimizer_real_bridges_result.json")


if __name__ == "__main__":
    main()
