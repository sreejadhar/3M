"""
Run: python -m pytest tests/test_kg_optimizer_genome.py -v
"""
import random

from kg_optimizer.genome import (
    GENOME_SPEC, CHEAP_GENES, EXPENSIVE_GENES,
    clamp_genome, crossover, mutate, random_genome, random_population,
)


def test_genome_spec_partition_is_disjoint_and_covers_all_genes():
    assert EXPENSIVE_GENES | CHEAP_GENES == set(GENOME_SPEC)
    assert EXPENSIVE_GENES & CHEAP_GENES == set()


def test_random_genome_respects_bounds_and_choices():
    rng = random.Random(1)
    g = random_genome(rng)
    for name, spec in GENOME_SPEC.items():
        kind = spec[0]
        v = g[name]
        if kind == "int":
            assert isinstance(v, int) and spec[1] <= v <= spec[2]
        elif kind == "float":
            assert spec[1] <= v <= spec[2]
        elif kind == "cat":
            assert v in spec[1]
        elif kind == "bool":
            assert isinstance(v, bool)


def test_random_population_size():
    rng = random.Random(2)
    pop = random_population(5, rng)
    assert len(pop) == 5
    assert all(set(g) == set(GENOME_SPEC) for g in pop)


def test_clamp_genome_bounds_numeric_genes():
    rng = random.Random(3)
    g = random_genome(rng)
    g["min_confidence"] = 99.0
    g["graphrag_top_k"] = -5
    clamped = clamp_genome(g)
    spec = GENOME_SPEC["min_confidence"]
    assert clamped["min_confidence"] == spec[2]
    spec_k = GENOME_SPEC["graphrag_top_k"]
    assert clamped["graphrag_top_k"] == spec_k[1]


def test_crossover_produces_valid_children():
    rng = random.Random(4)
    g1 = random_genome(rng)
    g2 = random_genome(rng)
    c1, c2 = crossover(g1, g2, rng)
    for child in (c1, c2):
        for name, spec in GENOME_SPEC.items():
            kind = spec[0]
            if kind == "int":
                assert spec[1] <= child[name] <= spec[2]
            elif kind == "float":
                assert spec[1] <= child[name] <= spec[2]
            elif kind == "cat":
                assert child[name] in (g1[name], g2[name])
            elif kind == "bool":
                assert child[name] in (g1[name], g2[name])


def test_mutate_zero_rate_is_identity():
    rng = random.Random(5)
    g = random_genome(rng)
    mutated = mutate(g, rng, rate=0.0)
    assert mutated == g


def test_mutate_full_rate_changes_numeric_genes_within_bounds():
    rng = random.Random(6)
    g = random_genome(rng)
    mutated = mutate(g, rng, rate=1.0)
    for name, spec in GENOME_SPEC.items():
        kind = spec[0]
        if kind in ("int", "float"):
            assert spec[1] <= mutated[name] <= spec[2]
        elif kind == "cat":
            assert mutated[name] in spec[1]
