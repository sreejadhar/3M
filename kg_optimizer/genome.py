"""
Genome spec + mixed-type GA operators for the ontology/KG optimizer.

A genome is a plain dict mapping gene name -> value. Genes fall into two tiers:

  EXPENSIVE_GENES — feed ontology_agent.OntologyConfig / knowledge_graph_agent.KGConfig.
    Changing one of these requires a full ontology + KG rebuild.

  CHEAP_GENES — feed dialog_agent.kg_inference_engine.InferenceOptions and
    dialog_agent.config.DialogConfig (query-time). These can be re-swept against an
    already-built graph without rebuilding it (see kg_optimizer/cache.py).

GENOME_SPEC entries are ("int", lo, hi) | ("float", lo, hi) | ("cat", [choices]) | ("bool",).
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

GeneSpec = Tuple[Any, ...]

GENOME_SPEC: Dict[str, GeneSpec] = {
    # ── Expensive tier: ontology_agent.OntologyConfig / knowledge_graph_agent.KGConfig ──
    "annotate_concepts":  ("bool",),
    "ontology_llm_model": ("cat", ["claude-haiku-4-5", "claude-sonnet-5"]),
    "profile_enabled":    ("bool",),
    "profile_llm_model":  ("cat", ["claude-haiku-4-5", "claude-sonnet-5"]),
    "embed_backend":      ("cat", ["sentence-transformers", "openai"]),
    "enable_name_resolution": ("bool",),

    # ── Cheap tier: dialog_agent.kg_inference_engine.InferenceOptions ──
    "min_confidence":            ("float", 0.30, 0.70),
    "auto_enable_threshold":     ("float", 0.60, 0.95),
    "cross_domain_penalty":      ("float", 0.00, 0.20),
    "embedding_sim_threshold":   ("float", 0.50, 0.95),
    "value_overlap_threshold":   ("float", 0.40, 0.90),
    "transitivity_decay":        ("float", 0.60, 0.99),
    "transitivity_min_conf":     ("float", 0.40, 0.90),
    "enable_tier3_llm_validation": ("bool",),
    "enable_value_overlap_tier":   ("bool",),

    # ── Cheap tier: dialog_agent.config.DialogConfig (query-time retrieval) ──
    "graphrag_top_k": ("int", 4, 16),
}

EXPENSIVE_GENES = frozenset({
    "annotate_concepts", "ontology_llm_model",
    "profile_enabled", "profile_llm_model",
    "embed_backend", "enable_name_resolution",
})
CHEAP_GENES = frozenset(set(GENOME_SPEC) - EXPENSIVE_GENES)

Genome = Dict[str, Any]


def _random_gene(spec: GeneSpec, rng: random.Random) -> Any:
    kind = spec[0]
    if kind == "int":
        return rng.randint(spec[1], spec[2])
    if kind == "float":
        return rng.uniform(spec[1], spec[2])
    if kind == "cat":
        return rng.choice(spec[1])
    if kind == "bool":
        return rng.random() < 0.5
    raise ValueError(f"unknown gene kind: {kind}")


def random_genome(rng: random.Random, spec: Dict[str, GeneSpec] = GENOME_SPEC) -> Genome:
    return {name: _random_gene(s, rng) for name, s in spec.items()}


def random_population(n: int, rng: random.Random, spec: Dict[str, GeneSpec] = GENOME_SPEC) -> List[Genome]:
    return [random_genome(rng, spec) for _ in range(n)]


def clamp_genome(genome: Genome, spec: Dict[str, GeneSpec] = GENOME_SPEC) -> Genome:
    """Clamp numeric genes into range and coerce ints; leaves cat/bool untouched
    (callers are responsible for only ever assigning valid choices to those)."""
    out = dict(genome)
    for name, s in spec.items():
        kind = s[0]
        if kind == "int":
            out[name] = int(round(min(max(out[name], s[1]), s[2])))
        elif kind == "float":
            out[name] = float(min(max(out[name], s[1]), s[2]))
    return out


def crossover(g1: Genome, g2: Genome, rng: random.Random,
              spec: Dict[str, GeneSpec] = GENOME_SPEC) -> Tuple[Genome, Genome]:
    """Type-aware crossover: blend for numeric genes, uniform pick for cat/bool."""
    c1, c2 = dict(g1), dict(g2)
    for name, s in spec.items():
        kind = s[0]
        a, b = g1[name], g2[name]
        if kind in ("int", "float"):
            t = rng.random()
            v1 = a * t + b * (1 - t)
            v2 = a * (1 - t) + b * t
            c1[name], c2[name] = v1, v2
        else:  # cat / bool — uniform crossover
            if rng.random() < 0.5:
                c1[name], c2[name] = b, a
    return clamp_genome(c1, spec), clamp_genome(c2, spec)


def mutate(genome: Genome, rng: random.Random, rate: float,
           spec: Dict[str, GeneSpec] = GENOME_SPEC) -> Genome:
    """Gaussian mutation for numeric genes, random-reset for cat/bool."""
    out = dict(genome)
    for name, s in spec.items():
        if rng.random() >= rate:
            continue
        kind = s[0]
        if kind == "float":
            lo, hi = s[1], s[2]
            sigma = (hi - lo) * 0.15
            out[name] = out[name] + rng.gauss(0, sigma)
        elif kind == "int":
            lo, hi = s[1], s[2]
            span = max(1, hi - lo)
            out[name] = out[name] + rng.gauss(0, span * 0.15)
        else:  # cat / bool — random reset
            out[name] = _random_gene(s, rng)
    return clamp_genome(out, spec)


def population_diversity(population: List[Genome], spec: Dict[str, GeneSpec] = GENOME_SPEC) -> float:
    """
    0-1 diversity score, averaged across genes: numeric genes contribute their
    population stdev normalized by the gene's (hi - lo) range; cat/bool genes
    contribute (distinct values seen - 1) / (population size - 1). 0 means
    every individual is identical on that gene; 1 means maximally spread.

    Used by the GA loop to detect premature convergence and trigger fresh
    random-individual injection (see kg_optimizer.ga.run_ga).
    """
    n = len(population)
    if n < 2:
        return 0.0

    scores: List[float] = []
    for name, s in spec.items():
        kind = s[0]
        values = [g[name] for g in population]
        if kind in ("int", "float"):
            lo, hi = s[1], s[2]
            span = (hi - lo) or 1.0
            mean = sum(values) / n
            variance = sum((v - mean) ** 2 for v in values) / n
            scores.append(min((variance ** 0.5) / span, 1.0))
        else:  # cat / bool
            distinct = len(set(values))
            scores.append((distinct - 1) / (n - 1))
    return sum(scores) / len(scores) if scores else 0.0


def expensive_key(genome: Genome) -> Dict[str, Any]:
    """The subset of genes that determine whether a build must be redone."""
    return {k: genome[k] for k in sorted(EXPENSIVE_GENES)}


def cheap_key(genome: Genome) -> Dict[str, Any]:
    return {k: genome[k] for k in sorted(CHEAP_GENES)}
