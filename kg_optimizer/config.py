"""
OptimizerConfig — everything the GA loop / eval harness needs that isn't a gene.
Loadable from a JSON file (see kg_optimizer/fixtures/pilot_config.json).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class FitnessWeights:
    answer_quality: float = 1.0
    bridge_f1:      float = 0.5
    cost:           float = 0.3
    latency:        float = 0.2
    consistency:    float = 0.3   # same concept -> same resolved label (naming resolution)
    distinctiveness: float = 0.3  # different concepts -> different resolved labels


@dataclass
class OptimizerConfig:
    source_id: str = ""              # logical source id, used to namespace cache/kg_id

    # ── GA loop ──────────────────────────────────────────────────────────────
    population_size: int = 10
    generations:      int = 5
    mutation_rate:    float = 0.2
    mutation_decay:   float = 0.9     # multiplied into mutation_rate each generation
    tournament_size:  int = 3
    elitism:          int = 2
    random_seed:      int = 42

    # ── Fitness ──────────────────────────────────────────────────────────────
    fitness_weights: FitnessWeights = field(default_factory=FitnessWeights)
    judge_model: str = "claude-haiku-4-5"

    # ── KG snapshot store (shared by all trial builds, isolated by kg_id) ────
    kg_store_path: str = field(default_factory=lambda: os.environ.get("KG_STORE_DB", "data/kg_store.db"))

    # ── KG-query API (used by eval_harness to run questions against a built KG) ──
    kg_query_api_base: str = field(default_factory=lambda: os.environ.get("KG_QUERY_API_BASE", "http://127.0.0.1:8003"))
    query_timeout_s:   int = 240

    # ── Misc ─────────────────────────────────────────────────────────────────
    cache_dir: str = ".kg_optimizer_cache"


def load_config(path: str) -> OptimizerConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = json.load(f)
    weights_raw = raw.pop("fitness_weights", None)
    cfg = OptimizerConfig(**raw)
    if weights_raw:
        cfg.fitness_weights = FitnessWeights(**weights_raw)
    return cfg
