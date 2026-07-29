"""
Gene-importance analysis — mines the GA's own trial history (genome ->
composite fitness, for every individual ever evaluated, not just the
champion) to answer "which genes did the search actually exploit?"

Approach: one-hot/numeric-encode every trial's genome as a feature vector,
fit a gradient-boosted regression tree against composite fitness, and read
off its impurity-based feature importances (grouping one-hot columns for
cat/bool genes back into a single per-gene score). This is the same family
of technique as SHAP (both explain a fitted surrogate model) but needs no
extra heavy dependency beyond scikit-learn, which the optimizer already
requires.

Genes with near-zero importance are candidates to freeze/drop from the
genome in a follow-up run, shrinking the search space (ablation-guided
dimensionality reduction).
"""
from __future__ import annotations

from typing import Dict, List

from kg_optimizer.genome import GENOME_SPEC, Genome

MIN_TRIALS_FOR_IMPORTANCE = 10


def _encode_genome(genome: Genome, spec: Dict = GENOME_SPEC) -> Dict[str, float]:
    """Flatten one genome into {feature_name: value} — numeric genes pass
    through as-is, cat/bool genes one-hot encode into "{gene}={choice}"."""
    row: Dict[str, float] = {}
    for name, s in spec.items():
        kind = s[0]
        value = genome[name]
        if kind in ("int", "float"):
            row[name] = float(value)
        elif kind == "bool":
            row[f"{name}=True"] = 1.0 if value else 0.0
        elif kind == "cat":
            for choice in s[1]:
                row[f"{name}={choice}"] = 1.0 if value == choice else 0.0
    return row


def _feature_to_gene(feature_name: str) -> str:
    """Map an encoded feature column back to its owning gene name."""
    return feature_name.split("=", 1)[0]


def gene_importance(trials: List, spec: Dict = GENOME_SPEC,
                    min_trials: int = MIN_TRIALS_FOR_IMPORTANCE) -> Dict[str, float]:
    """
    trials: kg_optimizer.ga.GAHistory.trials (or any list of objects with
    .genome and .fitness.composite).

    Returns {gene_name: normalized_importance} summing to ~1.0, sorted
    descending, or {} if there aren't enough trials to fit a meaningful
    surrogate model (min_trials, default 10 — a GA run with fewer
    evaluations than that doesn't have enough signal to separate real gene
    effects from noise).
    """
    if len(trials) < min_trials:
        return {}

    try:
        import numpy as np
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError:
        return {}

    rows = [_encode_genome(t.genome, spec) for t in trials]
    feature_names = sorted({k for row in rows for k in row})
    X = np.array([[row.get(f, 0.0) for f in feature_names] for row in rows])
    y = np.array([t.fitness.composite for t in trials])

    model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0)
    model.fit(X, y)

    per_gene: Dict[str, float] = {}
    for feature_name, importance in zip(feature_names, model.feature_importances_):
        gene = _feature_to_gene(feature_name)
        per_gene[gene] = per_gene.get(gene, 0.0) + float(importance)

    total = sum(per_gene.values())
    if total <= 0:
        return {gene: 0.0 for gene in per_gene}

    normalized = {gene: score / total for gene, score in per_gene.items()}
    return dict(sorted(normalized.items(), key=lambda kv: kv[1], reverse=True))
