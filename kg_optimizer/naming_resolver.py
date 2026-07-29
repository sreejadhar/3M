"""
Naming resolution — a small, cheap, per-schema GA that picks the best canonical
label for each raw table/column name from a short candidate list, to fix noisy
metadata (e.g. "SEA_IDISCOVER_HCPS", "speciality_1_type_english") before it
becomes an ontology/KG RDFS.label (see ontology_agent/nodes/build_node.py).

This is a SIBLING to the hyperparameter GA in kg_optimizer/ga.py, not an
extension of it: GENOME_SPEC there is a small fixed hyperparameter vector,
while naming resolution needs one gene per raw name (hundreds, varying per
schema). genome.py's crossover()/mutate()/clamp_genome() already accept a
`spec` parameter, so they're reused unmodified against a spec built here.

Unlike the hyperparameter GA, fitness here is pure in-process structure
comparison (consistency_score/distinctiveness_score from fitness.py) — no KG
rebuild, no LLM call in the inner loop — so it can run many generations
cheaply. The only LLM cost is candidate generation, which runs once per table
and is cached in-process per (source signature, model).
"""
from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from kg_optimizer.config import FitnessWeights
from kg_optimizer.fitness import consistency_score, distinctiveness_score
from kg_optimizer.genome import GeneSpec, Genome, clamp_genome, crossover, mutate, random_population

logger = logging.getLogger(__name__)

_SUFFIX_RE = re.compile(r"(_(id|key|code|num|no|ref|fk|pk|flag|type|english))+$", re.IGNORECASE)
_KEY_SUFFIX_RE = re.compile(r"(_key|_id|_code|_ref|_no|_num|_sk|_nk|_bk|_fk)\s*$", re.IGNORECASE)


@dataclass
class NamingResolutionResult:
    table_labels: Dict[str, str] = field(default_factory=dict)
    column_labels: Dict[str, str] = field(default_factory=dict)
    link_pairs: List[Tuple[str, str]] = field(default_factory=list)
    consistency: float = 1.0
    distinctiveness: float = 1.0


def _bare(col_name: str) -> str:
    """Strip alias prefix: 'evt.event_key' -> 'event_key' (same rule as
    ontology_agent/nodes/build_node.py's _bare, duplicated to avoid a
    cross-package import into a node module)."""
    return col_name.rsplit(".", 1)[-1] if "." in col_name else col_name


def _title_case(raw_name: str) -> str:
    name = _bare(raw_name)
    name = _SUFFIX_RE.sub("", name)
    name = re.sub(r"^[a-z0-9]+_", "", name, count=1) if name.count("_") > 2 else name
    words = re.sub(r"[\s_\-]+", " ", name).strip()
    return " ".join(w.capitalize() for w in words.split(" ") if w) or raw_name


def rule_based_candidates(raw_name: str) -> List[str]:
    cleaned = _title_case(raw_name)
    out = [raw_name]
    if cleaned and cleaned.lower() != raw_name.lower():
        out.append(cleaned)
    return out


def glossary_candidates(raw_name: str, domain: str = "") -> List[str]:
    """Best-effort — glossary_store requires its own DB; degrade to no
    candidates rather than fail the whole naming-resolution pass."""
    try:
        import glossary_store

        terms = glossary_store.search_terms(_bare(raw_name).replace("_", " "), domain=domain, limit=5)
    except Exception as exc:
        logger.debug("naming_resolver: glossary lookup skipped for %r (%s)", raw_name, exc)
        return []

    out: List[str] = []
    for term in terms:
        if term.get("name"):
            out.append(term["name"])
        for syn in term.get("synonyms") or []:
            syn_name = syn.get("synonym") if isinstance(syn, dict) else syn
            if syn_name:
                out.append(syn_name)
    return out


_LLM_SYSTEM_PROMPT = (
    "You are a database metadata expert. For each raw column or table name below, "
    "suggest 1-2 short, human-readable business labels a non-technical user would "
    "recognize (e.g. 'speciality_1_type_english' -> 'Specialty', "
    "'SEA_IDISCOVER_HCPS' -> 'Healthcare Providers'). Keep labels 1-4 words, "
    "Title Case. Return ONLY a JSON object mapping each input name to a list of "
    "label strings, no prose, no markdown fences."
)


def llm_candidates_for_table(table_name: str, column_names: List[str], model: str,
                             domain: str = "") -> Dict[str, List[str]]:
    """One batched LLM call per table (table name + its column names) -> extra
    candidate labels per raw name. Returns {} on any failure — candidate
    generation is additive, never load-bearing on its own (identity + rule-based
    candidates always exist)."""
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {}

    from llm_client import get_client

    names = [table_name] + list(column_names)
    user_msg = (
        (f"Domain: {domain}\n" if domain else "")
        + f"Table: {table_name}\nNames to label:\n" + "\n".join(f"- {n}" for n in names)
    )
    try:
        client = get_client()
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=0.0,
            system=_LLM_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = resp.content[0].text if resp.content else "{}"
        raw = re.sub(r"^```[a-z]*\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw.strip())
        parsed = json.loads(raw)
        return {k: list(v) for k, v in parsed.items() if isinstance(v, list)}
    except Exception as exc:
        logger.warning("naming_resolver: LLM candidate generation failed for table %r: %s", table_name, exc)
        return {}


def generate_candidates_for_report(report: Dict[str, Any], domain: str = "",
                                   llm_model: str = "claude-haiku-4-5",
                                   use_llm: bool = True, use_glossary: bool = True,
                                   ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Returns (candidates_by_table, candidates_by_column). Column candidates are
    keyed by raw column name, deduped across tables (matches build_node.py, which
    also assigns one RDFS.label per raw column-name string)."""
    tables: Dict[str, Any] = report.get("tables") or {}

    candidates_by_table: Dict[str, List[str]] = {}
    candidates_by_column: Dict[str, List[str]] = {}

    for table_name, table_meta in tables.items():
        col_names = [
            c.get("name", "") for c in (table_meta.get("columns") or [])
            if isinstance(c, dict) and c.get("name")
        ] if isinstance(table_meta, dict) else []

        candidates_by_table.setdefault(table_name, []).extend(rule_based_candidates(table_name))
        if use_glossary:
            candidates_by_table[table_name].extend(glossary_candidates(table_name, domain))

        for col_name in col_names:
            candidates_by_column.setdefault(col_name, []).extend(rule_based_candidates(col_name))
            if use_glossary:
                candidates_by_column[col_name].extend(glossary_candidates(col_name, domain))

        if use_llm:
            llm_out = llm_candidates_for_table(table_name, col_names, llm_model, domain)
            for name, labels in llm_out.items():
                if name == table_name:
                    candidates_by_table[table_name].extend(labels)
                elif name in candidates_by_column:
                    candidates_by_column[name].extend(labels)

    def _dedup(names_map: Dict[str, List[str]]) -> Dict[str, List[str]]:
        out = {}
        for name, cands in names_map.items():
            seen = []
            for c in cands:
                if c and c not in seen:
                    seen.append(c)
            out[name] = seen or [name]
        return out

    return _dedup(candidates_by_table), _dedup(candidates_by_column)


def build_link_pairs(report: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Names expected to represent the same concept, for consistency scoring:
    columns sharing the same bare key-suffixed name (evt.event_key ~ event_key)
    across two or more tables, using the same matching rule as build_node.py's
    isolated-table fallback edge inference (_KEY_SUFFIX_RE/_bare)."""
    tables: Dict[str, Any] = report.get("tables") or {}
    bare_to_raw: Dict[str, List[str]] = {}

    for table_meta in tables.values():
        if not isinstance(table_meta, dict):
            continue
        for col in table_meta.get("columns") or []:
            if not isinstance(col, dict):
                continue
            raw = col.get("name", "")
            bare = _bare(raw)
            if raw and _KEY_SUFFIX_RE.search(bare):
                bare_to_raw.setdefault(bare.lower(), []).append(raw)

    pairs: List[Tuple[str, str]] = []
    for raws in bare_to_raw.values():
        uniq = sorted(set(raws))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                pairs.append((uniq[i], uniq[j]))
    return pairs


def build_naming_spec(candidates_by_name: Dict[str, List[str]]) -> Dict[str, GeneSpec]:
    return {name: ("cat", cands) for name, cands in candidates_by_name.items()}


def _naming_fitness(genome: Genome, link_pairs: List[Tuple[str, str]], weights: FitnessWeights) -> float:
    c = consistency_score(genome, link_pairs)
    d = distinctiveness_score(genome, link_pairs)
    return weights.consistency * c + weights.distinctiveness * d


def run_naming_ga(candidates_by_name: Dict[str, List[str]], link_pairs: List[Tuple[str, str]],
                  weights: Optional[FitnessWeights] = None, population_size: int = 30,
                  generations: int = 20, mutation_rate: float = 0.3, elitism: int = 4,
                  tournament_size: int = 3, random_seed: int = 42) -> Tuple[Dict[str, str], float, float]:
    """Search the per-name candidate-label assignment space. Returns
    (resolved_labels, consistency, distinctiveness) for the best individual found.
    Mirrors ga.py:run_ga's structure (elitism + tournament selection + type-aware
    crossover/mutation + decaying mutation rate) but with a cheap in-process
    fitness — no KG rebuild per genome."""
    weights = weights or FitnessWeights()
    if not candidates_by_name:
        return {}, 1.0, 1.0

    spec = build_naming_spec(candidates_by_name)
    rng = random.Random(random_seed)

    def score(individual: Genome) -> float:
        return _naming_fitness(individual, link_pairs, weights)

    population = random_population(population_size, rng, spec)
    scored = sorted(population, key=score, reverse=True)
    rate = mutation_rate

    for _ in range(generations):
        elites = scored[:elitism]
        offspring: List[Genome] = list(elites)
        while len(offspring) < population_size:
            p1 = max(rng.sample(scored, min(tournament_size, len(scored))), key=score)
            p2 = max(rng.sample(scored, min(tournament_size, len(scored))), key=score)
            c1, c2 = crossover(p1, p2, rng, spec)
            c1 = mutate(c1, rng, rate, spec)
            c2 = mutate(c2, rng, rate, spec)
            offspring.append(c1)
            if len(offspring) < population_size:
                offspring.append(c2)
        # cat genes have no numeric range — clamp_genome is a no-op for them, but
        # crossover/mutate can still emit a value outside the candidate list only
        # if spec itself changes mid-run, which it doesn't here.
        scored = sorted(offspring, key=score, reverse=True)
        rate *= 0.92

    best = scored[0]
    return best, consistency_score(best, link_pairs), distinctiveness_score(best, link_pairs)


_resolution_cache: Dict[Tuple[Any, ...], NamingResolutionResult] = {}


def resolve_names(report: Dict[str, Any], source_id: str = "", domain: str = "",
                  llm_model: str = "claude-haiku-4-5", weights: Optional[FitnessWeights] = None,
                  use_llm: bool = True) -> NamingResolutionResult:
    """Top-level entrypoint: generate candidates, derive link pairs, run the
    naming GA for tables and columns, and cache the result in-process keyed by
    (source_id, sorted raw names, model) so repeated calls within one GA/CLI run
    (e.g. across hyperparameter-GA trials that share the same schema) don't
    re-pay LLM candidate-generation cost."""
    tables: Dict[str, Any] = report.get("tables") or {}
    table_names = tuple(sorted(tables.keys()))
    cache_key = (source_id, table_names, llm_model, use_llm)
    if cache_key in _resolution_cache:
        return _resolution_cache[cache_key]

    candidates_by_table, candidates_by_column = generate_candidates_for_report(
        report, domain=domain, llm_model=llm_model, use_llm=use_llm,
    )
    link_pairs = build_link_pairs(report)

    table_labels, _, _ = run_naming_ga(candidates_by_table, [], weights)
    column_labels, consistency, distinctiveness = run_naming_ga(candidates_by_column, link_pairs, weights)

    result = NamingResolutionResult(
        table_labels=table_labels, column_labels=column_labels, link_pairs=link_pairs,
        consistency=consistency, distinctiveness=distinctiveness,
    )
    _resolution_cache[cache_key] = result
    return result
