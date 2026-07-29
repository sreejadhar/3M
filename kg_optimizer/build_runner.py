"""
In-process build execution — turns a genome into a real ontology + KG build,
skipping the ontology_api.py / kg_api.py HTTP wrappers entirely (OntologyAgent
and KGAgent are plain importable classes; see ontology_agent/agent.py:68 and
knowledge_graph_agent/agent.py:119).

Bridge inference for GA trials uses `run_enterprise_inference` (no "_and_save"),
which returns candidate bridges without touching the shared kg_bridges store —
only the final champion is persisted via the real save path
(dialog_agent.kg_bridges.run_inference_and_save), done by the caller after the
GA run completes, not by this module.

The KG itself is never written to a live graph database — KGAgent persists
{nodes, edges} snapshots to the shared KG snapshot store (kg_store.py,
SQLite/Postgres-backed), which is what get_or_build_kg / load_existing_kg
read back on a cache hit.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from kg_optimizer.cache import BuildCache
from kg_optimizer.config import OptimizerConfig
from kg_optimizer.genome import Genome

logger = logging.getLogger(__name__)

_EMBED_DIMENSIONS = {
    "sentence-transformers": 384,
    "openai": 1536,
}


@dataclass
class KGBuildResult:
    kg_id: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    was_cached: bool
    build_time_s: float = 0.0


@dataclass
class BridgeResult:
    high_conf: List[Any] = field(default_factory=list)
    medium_conf: List[Any] = field(default_factory=list)
    infer_time_s: float = 0.0


def _ontology_config(genome: Genome, ontology_report: Optional[Dict[str, Any]] = None,
                     source_domain: str = "", source_id: str = ""):
    from ontology_agent.config import OntologyConfig

    cfg = OntologyConfig(
        annotate_concepts=genome["annotate_concepts"],
        llm_model=genome["ontology_llm_model"],
        source_domain=source_domain,
    )

    if genome.get("enable_name_resolution") and ontology_report is not None:
        from kg_optimizer.naming_resolver import resolve_names

        naming = resolve_names(
            ontology_report, source_id=source_id, domain=source_domain,
            llm_model=genome["ontology_llm_model"],
        )
        cfg.enable_name_resolution = True
        cfg.table_labels = naming.table_labels
        cfg.column_labels = naming.column_labels

    return cfg


def _kg_config(genome: Genome, kg_id: str, cfg: OptimizerConfig, mode: str = "generate"):
    from knowledge_graph_agent.config import KGConfig

    backend = genome["embed_backend"]
    return KGConfig(
        kg_store_path=cfg.kg_store_path,
        kg_id=kg_id,
        mode=mode,
        clear_existing=(mode == "generate"),
        profile_enabled=genome["profile_enabled"],
        embed_enabled=True,
        embed_backend=backend,
        embed_dimensions=_EMBED_DIMENSIONS[backend],
    )


def build_ontology_and_kg(genome: Genome, ontology_report: Dict[str, Any], kg_id: str,
                          cfg: OptimizerConfig, source_domain: str = "") -> KGBuildResult:
    from ontology_agent.agent import OntologyAgent
    from knowledge_graph_agent.agent import KGAgent

    t0 = time.time()
    onto_agent = OntologyAgent(_ontology_config(genome, ontology_report, source_domain, cfg.source_id))
    onto_result = onto_agent.run(ontology_report)
    if onto_result.get("phase") == "error":
        raise RuntimeError(f"ontology build failed: {onto_result.get('errors')}")

    kg_agent = KGAgent(_kg_config(genome, kg_id, cfg, mode="generate"))
    kg_result = kg_agent.run(onto_result["ontology_turtle"])
    if kg_result.get("phase") == "error":
        raise RuntimeError(f"KG build failed: {kg_result.get('errors')}")

    graph_data = kg_result.get("graph_data", {"nodes": [], "edges": []})
    return KGBuildResult(
        kg_id=kg_id,
        nodes=graph_data.get("nodes", []),
        edges=graph_data.get("edges", []),
        was_cached=False,
        build_time_s=time.time() - t0,
    )


def load_kg_snapshot(kg_id: str, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> KGBuildResult:
    """Wrap an already-built graph (e.g. read straight from data/kg_store.db via
    datasets.load_connection_from_kg_store) as a KGBuildResult. Bridge inference
    (run_bridge_inference below) only needs `nodes` — it never rebuilds the KG —
    so this is enough to run real bridge-inference GA trials against an existing
    snapshot. Expensive-tier genes (ontology/profiling/embedding) have no effect
    in this mode since nothing is rebuilt; only cheap-tier
    (dialog_agent.kg_inference_engine) genes matter."""
    return KGBuildResult(kg_id=kg_id, nodes=nodes, edges=edges, was_cached=True, build_time_s=0.0)


def load_existing_kg(kg_id: str, genome: Genome, cfg: OptimizerConfig) -> KGBuildResult:
    from knowledge_graph_agent.agent import KGAgent

    t0 = time.time()
    kg_agent = KGAgent(_kg_config(genome, kg_id, cfg, mode="load"))
    kg_result = kg_agent.load()
    graph_data = kg_result.get("graph_data", {"nodes": [], "edges": []})
    return KGBuildResult(
        kg_id=kg_id,
        nodes=graph_data.get("nodes", []),
        edges=graph_data.get("edges", []),
        was_cached=True,
        build_time_s=time.time() - t0,
    )


def get_or_build_kg(genome: Genome, ontology_report: Dict[str, Any], cfg: OptimizerConfig,
                     cache: BuildCache, source_domain: str = "") -> KGBuildResult:
    """Cache-aware entrypoint: reuse a build if its expensive-tier genes are unchanged."""
    entry = cache.get(cfg.source_id, genome)
    if entry is not None:
        try:
            return load_existing_kg(entry.kg_id, genome, cfg)
        except Exception as exc:
            logger.warning("Cache hit for %s but load failed (%s) — rebuilding", entry.kg_id, exc)

    from kg_optimizer.cache import expensive_hash

    kg_id = f"gaopt_{cfg.source_id}_{expensive_hash(cfg.source_id, genome)}"
    result = build_ontology_and_kg(genome, ontology_report, kg_id, cfg, source_domain)
    cache.put(cfg.source_id, genome, kg_id)
    return result


def _inference_options(genome: Genome):
    from dialog_agent.kg_inference_engine import InferenceOptions

    return InferenceOptions(
        min_confidence=genome["min_confidence"],
        auto_enable_threshold=genome["auto_enable_threshold"],
        cross_domain_penalty=genome["cross_domain_penalty"],
        run_tier2_embeddings=True,
        run_tier3_llm=genome["enable_tier3_llm_validation"],
        embedding_sim_threshold=genome["embedding_sim_threshold"],
        run_tier_value_overlap=genome["enable_value_overlap_tier"],
        value_overlap_threshold=genome["value_overlap_threshold"],
        transitivity_decay=genome["transitivity_decay"],
        transitivity_min_conf=genome["transitivity_min_conf"],
    )


def run_bridge_inference(genome: Genome, kg_build: KGBuildResult, report: Optional[Dict[str, Any]] = None,
                         display_name: str = "", domain: str = "") -> BridgeResult:
    """Score candidate bridges WITHOUT persisting them (uses run_enterprise_inference,
    not run_enterprise_inference_and_save) — safe to call hundreds of times per GA run."""
    from dialog_agent.kg_inference_engine import build_context, run_enterprise_inference

    t0 = time.time()
    ctx = build_context(kg_build.kg_id, display_name or kg_build.kg_id, domain, kg_build.nodes, report)
    options = _inference_options(genome)
    high_conf, medium_conf = run_enterprise_inference(ctx, ctx, options)
    return BridgeResult(high_conf=high_conf, medium_conf=medium_conf, infer_time_s=time.time() - t0)


def persist_champion_bridges(genome: Genome, kg_build: KGBuildResult, report: Optional[Dict[str, Any]] = None,
                             display_name: str = "", domain: str = "") -> List[Any]:
    """Real save path — call once, on the GA's final champion only.

    dialog_agent.kg_bridges.run_inference_and_save always uses DEFAULT_OPTIONS
    (it has no options parameter), so persisting the champion's *tuned*
    bridge-inference genes requires calling the lower-level
    run_enterprise_inference_and_save directly with our InferenceOptions.
    """
    from dialog_agent.kg_inference_engine import build_context, run_enterprise_inference_and_save

    options = _inference_options(genome)
    ctx = build_context(kg_build.kg_id, display_name or kg_build.kg_id, domain, kg_build.nodes, report)
    return run_enterprise_inference_and_save(ctx, ctx, options=options)
