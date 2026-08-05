"""
Entrypoint: python -m kg_optimizer.cli run --dataset <path> --config <path> --report <path> --output <path>

`--report` must point to a pre-extracted metadata report JSON (the same shape
OntologyAgent.run() consumes) — produced once via the existing metadata_agent
/extract flow. Extraction genes are out of scope for this optimizer.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys

from kg_optimizer.ablation import run_ablations
from kg_optimizer.build_runner import build_ontology_and_kg, persist_champion_bridges, run_bridge_inference
from kg_optimizer.config import OptimizerConfig, load_config
from kg_optimizer.datasets import (
    from_lifesciences_assets, gold_bridges_from_metadata_report, load_dataset,
)
from kg_optimizer.eval_harness import run_dataset
from kg_optimizer.fitness import bridge_precision_recall, compute_fitness, pareto_front
from kg_optimizer.ga import run_ga
from kg_optimizer.importance import gene_importance

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _json_default(obj):
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    return str(obj)


def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    dataset = load_dataset(args.dataset)
    if not cfg.source_id:
        cfg.source_id = dataset.source_id

    with open(args.report, "r", encoding="utf-8") as f:
        ontology_report = json.load(f)

    logger.info(
        "Starting GA run: population=%d generations=%d questions=%d dataset=%s",
        cfg.population_size, cfg.generations, len(dataset.questions), dataset.source_id,
    )
    ga_result = run_ga(dataset, ontology_report, cfg, source_domain=args.domain)
    champion = ga_result.best

    logger.info("Champion composite fitness: %.4f", champion.fitness.composite)
    logger.info("Running ablations on champion (%d repeat(s) per variant)...", args.ablation_repeats)
    ablation_rows = run_ablations(
        champion.genome, champion.fitness, dataset, ontology_report, cfg, source_domain=args.domain,
        n_repeats=args.ablation_repeats,
    )
    for row in ablation_rows:
        sig_note = (
            f" p={row.p_value:.3f} significant={row.significant}"
            if row.p_value is not None else ""
        )
        logger.info(
            "Ablation %-26s delta=%.4f mean=%.4f ci95=(%.4f, %.4f)%s",
            row.name, row.delta_composite, row.mean_composite, row.ci95[0], row.ci95[1], sig_note,
        )

    logger.info("Mining gene importance from %d GA trials...", len(ga_result.history.trials))
    importance = gene_importance(ga_result.history.trials)
    if importance:
        top = list(importance.items())[:5]
        logger.info("Top genes by importance: %s", ", ".join(f"{g}={v:.3f}" for g, v in top))
    else:
        logger.info("Not enough trials for a gene-importance surrogate model — skipping.")

    from kg_optimizer.cache import BuildCache
    cache = BuildCache(cfg.cache_dir)

    if args.persist_champion:
        logger.info("Persisting champion's bridges to the real kg_bridges store...")
        from kg_optimizer.build_runner import get_or_build_kg
        kg_build = get_or_build_kg(champion.genome, ontology_report, cfg, cache, args.domain)
        persist_champion_bridges(champion.genome, kg_build, ontology_report, domain=args.domain)

    output = {
        "champion_genome": champion.genome,
        "champion_fitness": champion.fitness,
        "pareto_front": pareto_front(ga_result.history.as_pareto_entries()),
        "ablations": ablation_rows,
        "gene_importance": importance,
        "cache_stats": cache.stats(),
        "total_trials": len(ga_result.history.trials),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=_json_default)
    logger.info("Wrote results to %s", args.output)


def apply_champion(args: argparse.Namespace) -> None:
    """Build a real ontology + KG from an already-known champion genome (e.g. the
    output of `run`, or any hand-picked genome), optionally persist its bridges to
    the real kg_bridges store, and optionally check answer quality against a real
    question set. Skips the GA search entirely — for re-applying a genome you
    already trust, not for finding one."""
    with open(args.genome, "r", encoding="utf-8") as f:
        genome_raw = json.load(f)
    genome = genome_raw["champion_genome"] if "champion_genome" in genome_raw else genome_raw

    with open(args.report, "r", encoding="utf-8") as f:
        report = json.load(f)

    cfg = OptimizerConfig(source_id=args.source_id, judge_model=args.judge_model)

    logger.info("Building real ontology + KG (kg_id=%s, source_id=%s) ...", args.kg_id, args.source_id)
    kg_build = build_ontology_and_kg(genome, report, args.kg_id, cfg, source_domain=args.domain)
    logger.info("Built KG: %d nodes, %d edges in %.1fs", len(kg_build.nodes), len(kg_build.edges), kg_build.build_time_s)

    if args.persist_bridges:
        logger.info("Persisting champion's bridges to the real kg_bridges store...")
        predicted = persist_champion_bridges(genome, kg_build, report, domain=args.domain)
    else:
        result = run_bridge_inference(genome, kg_build, report, domain=args.domain)
        predicted = result.high_conf + result.medium_conf

    gold_types = args.gold_cardinality_types.split(",") if args.gold_cardinality_types else None
    gold_bridges = gold_bridges_from_metadata_report(report, include_cardinality_types=gold_types)
    precision, recall, f1 = bridge_precision_recall(predicted, gold_bridges)
    logger.info("Bridges: precision=%.3f recall=%.3f f1=%.3f against %d gold bridges",
               precision, recall, f1, len(gold_bridges))

    fitness = None
    run_result = None
    if args.dataset or args.lifesciences_questions:
        dataset = (load_dataset(args.dataset) if args.dataset
                  else from_lifesciences_assets(limit=args.lifesciences_questions))
        logger.info("Running %d questions against dialog-api (%s) ...", len(dataset.questions), cfg.kg_query_api_base)
        run_result = run_dataset(dataset, kg_build, cfg)
        fitness = compute_fitness(
            run_result=run_result, predicted_bridges=predicted, gold_bridges=gold_bridges,
            build_time_s=kg_build.build_time_s, infer_time_s=0.0,
            weights=cfg.fitness_weights, judge_model=cfg.judge_model,
        )
        logger.info("Answer quality=%.4f composite=%.4f", fitness.answer_quality, fitness.composite)

    output = {
        "genome": genome,
        "kg_id": args.kg_id,
        "nodes": len(kg_build.nodes),
        "edges": len(kg_build.edges),
        "build_time_s": kg_build.build_time_s,
        "persisted_bridge_count": len(predicted) if args.persist_bridges else None,
        "bridge_precision": precision,
        "bridge_recall": recall,
        "bridge_f1": f1,
        "fitness": fitness,
        "question_results": run_result.question_results if run_result else None,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=_json_default)
    logger.info("Wrote results to %s", args.output)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="kg_optimizer")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run the GA + ablation pipeline")
    p_run.add_argument("--dataset", required=True, help="Path to an EvalDataset JSON file")
    p_run.add_argument("--config", required=True, help="Path to an OptimizerConfig JSON file")
    p_run.add_argument("--report", required=True, help="Path to a pre-extracted metadata report JSON")
    p_run.add_argument("--output", default="kg_optimizer_result.json")
    p_run.add_argument("--domain", default="", help="Optional domain hint (e.g. 'healthcare')")
    p_run.add_argument("--persist-champion", action="store_true",
                       help="Save the champion's bridges to the real kg_bridges store")
    p_run.add_argument("--ablation-repeats", type=int, default=1,
                       help="Re-evaluate champion + each ablation variant this many times "
                            "(>=6 enables a paired Wilcoxon significance test — n=5's exact "
                            "p-value floor of 0.0625 can never clear 0.05) to separate a real "
                            "fitness drop from LLM-judge noise. Cost scales linearly.")
    p_run.set_defaults(func=run)

    p_apply = sub.add_parser("apply-champion",
                             help="Build a real ontology + KG from an already-known genome and check answer quality")
    p_apply.add_argument("--genome", required=True,
                         help="Path to a JSON file with a raw genome dict, or a `run` result "
                              "(read from its top-level champion_genome key)")
    p_apply.add_argument("--report", required=True, help="Path to a pre-extracted metadata report JSON")
    p_apply.add_argument("--source-id", required=True, help="Logical source id (cache/kg namespace)")
    p_apply.add_argument("--kg-id", required=True, help="kg_id to build/persist the KG snapshot under")
    p_apply.add_argument("--domain", default="", help="Optional domain hint (e.g. 'healthcare')")
    p_apply.add_argument("--persist-bridges", action="store_true",
                         help="Save the genome's bridges to the real kg_bridges store (else score-only)")
    p_apply.add_argument("--gold-cardinality-types", default="1:1,1:N,N:1",
                         help="Comma-separated cardinality_relationships types to trust as gold bridges "
                              "(empty string = include all types)")
    p_apply.add_argument("--dataset", default=None, help="Path to an EvalDataset JSON to check answer quality against")
    p_apply.add_argument("--lifesciences-questions", type=int, default=None,
                         help="Convenience: use N questions from the built-in LifeSciences question set "
                              "instead of --dataset")
    p_apply.add_argument("--judge-model", default="claude-haiku-4-5")
    p_apply.add_argument("--output", default="kg_optimizer_apply_champion_result.json")
    p_apply.set_defaults(func=apply_champion)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
