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
from kg_optimizer.build_runner import persist_champion_bridges
from kg_optimizer.config import load_config
from kg_optimizer.datasets import load_dataset
from kg_optimizer.fitness import pareto_front
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

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
