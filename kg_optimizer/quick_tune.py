"""
Lightweight cheap-tier GA sweep for reindex-time bridge tuning.

Unlike the full kg_optimizer.cli pipeline (LLM-judge answer quality, a
dataset of NL questions, a full ontology/KG rebuild per trial), this only
tunes bridge-inference thresholds against a single already-built KG,
scored purely by bridge F1 against gold bridges derived from that source's
own metadata report (declared FKs / fk_candidates / cardinality
relationships — see kg_optimizer.datasets.gold_bridges_from_metadata_report).
No dialog-api calls, no LLM judge, no DEAP dependency (only used by
kg_optimizer.fitness.pareto_front for reporting elsewhere) — cheap enough
to run synchronously inside the orchestrator's reindex flow, every time a
source is (re)indexed, so bridge thresholds keep adapting as the schema
changes instead of drifting stale against hardcoded defaults.

Deliberately excludes from the tunable gene set:
  - cross_domain_penalty: a no-op for a self-pass (ctx_a is ctx_b, so
    same_domain is always True in run_enterprise_inference).
  - value_overlap_threshold / enable_value_overlap_tier: Tier V needs a
    live sample_fn hitting the source database — running that
    population x generations times during reindex would defeat the whole
    point of the orchestrator's _bridge_sweep_lock (avoid hammering
    source DBs during a bridge sweep). Left at production defaults.
  - transitivity_*, enable_tier3_llm_validation: not exercised by
    run_enterprise_inference at all — transitivity is a separate pass run
    after bridge inference, and Tier 3 LLM validation runs async
    afterwards, so neither one affects synchronous fitness scoring here.
"""
from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

from kg_optimizer.datasets import gold_bridges_from_metadata_report
from kg_optimizer.fitness import bridge_precision_recall
from kg_optimizer.genome import GENOME_SPEC, Genome, crossover, mutate, random_population

logger = logging.getLogger(__name__)

QUICK_TUNE_SPEC = {
    k: GENOME_SPEC[k] for k in ("min_confidence", "auto_enable_threshold", "embedding_sim_threshold")
}


def _fitness(genome: Genome, kg_id: str, nodes: List[Dict], report: Optional[Dict],
            domain: str, gold_bridges: List[Any]) -> float:
    from dialog_agent.kg_inference_engine import InferenceOptions, KGContext, run_enterprise_inference

    options = InferenceOptions(
        min_confidence=genome["min_confidence"],
        auto_enable_threshold=genome["auto_enable_threshold"],
        embedding_sim_threshold=genome["embedding_sim_threshold"],
        run_tier_value_overlap=False,   # no DB access during tuning
        run_tier3_llm=False,            # scored synchronously here, no async validation
    )
    ctx = KGContext(kg_id=kg_id, display_name=kg_id, domain=domain, nodes=nodes, report=report)
    high_conf, medium_conf = run_enterprise_inference(ctx, ctx, options)
    _, _, f1 = bridge_precision_recall(high_conf + medium_conf, gold_bridges)
    return f1


def _tournament(scored: List[Tuple[Genome, float]], rng: random.Random, k: int = 3) -> Genome:
    contenders = rng.sample(scored, min(k, len(scored)))
    return max(contenders, key=lambda gs: gs[1])[0]


def tune_bridge_options(kg_id: str, nodes: List[Dict], report: Optional[Dict],
                        domain: str = "", population: int = 8, generations: int = 3,
                        seed: Optional[int] = None) -> "Optional[Any]":
    """
    Small cheap-tier GA sweep tuning bridge-inference thresholds against
    this source's OWN gold bridges. Returns a tuned InferenceOptions for the
    caller to persist bridges with (everything besides the three tuned
    fields is left at production defaults — Tier V / Tier 3 are only forced
    off during the tuning evaluation itself, not in the returned options),
    or None if there's nothing to tune against (no report, no nodes, or no
    gold bridges derivable from the report).
    """
    from dialog_agent.kg_inference_engine import InferenceOptions

    if not report or not nodes:
        return None

    gold_bridges = gold_bridges_from_metadata_report(report)
    if not gold_bridges:
        logger.info("quick_tune: no gold bridges derivable from report for %s — skipping GA sweep", kg_id[:8])
        return None

    rng = random.Random(seed)
    pop = random_population(population, rng, QUICK_TUNE_SPEC)
    scored = [(g, _fitness(g, kg_id, nodes, report, domain, gold_bridges)) for g in pop]

    for _gen in range(1, generations):
        scored.sort(key=lambda gs: gs[1], reverse=True)
        elites: List[Genome] = [g for g, _ in scored[:2]]

        offspring: List[Genome] = list(elites)
        while len(offspring) < population:
            p1 = _tournament(scored, rng)
            p2 = _tournament(scored, rng)
            c1, c2 = crossover(p1, p2, rng, QUICK_TUNE_SPEC)
            c1 = mutate(c1, rng, 0.3, QUICK_TUNE_SPEC)
            c2 = mutate(c2, rng, 0.3, QUICK_TUNE_SPEC)
            offspring.append(c1)
            if len(offspring) < population:
                offspring.append(c2)

        scored = [(g, _fitness(g, kg_id, nodes, report, domain, gold_bridges)) for g in offspring]

    champion, champion_f1 = max(scored, key=lambda gs: gs[1])
    logger.info(
        "quick_tune: champion bridge F1=%.3f for %s "
        "(min_confidence=%.2f auto_enable_threshold=%.2f embedding_sim_threshold=%.2f)",
        champion_f1, kg_id[:8],
        champion["min_confidence"], champion["auto_enable_threshold"], champion["embedding_sim_threshold"],
    )
    return InferenceOptions(
        min_confidence=champion["min_confidence"],
        auto_enable_threshold=champion["auto_enable_threshold"],
        embedding_sim_threshold=champion["embedding_sim_threshold"],
    )
