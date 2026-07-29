"""
Renders the two charts for the synthetic demo run: GA convergence (best/mean
composite fitness per generation) and the quality-vs-cost Pareto front.
Run after kg_optimizer.demo_synthetic_run. Uses the repo's validated
light-mode chart palette (dataviz skill reference palette).

Run: python -m kg_optimizer.demo_plot
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"     # series slot 1 — "best"
ORANGE = "#eb6834"   # series slot 2 — "mean"
AQUA = "#1baf7a"     # Pareto-front points


def load_result(path: str = "kg_optimizer_demo_result.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_convergence(data: dict, ax) -> None:
    gens = data["generations"]
    best = data["best_per_gen"]
    mean = data["mean_per_gen"]

    ax.set_facecolor(SURFACE)
    ax.plot(gens, best, color=BLUE, linewidth=2, marker="o", markersize=5, label="Best composite fitness")
    ax.plot(gens, mean, color=ORANGE, linewidth=2, marker="o", markersize=5, label="Mean composite fitness")

    ax.set_title("GA convergence (synthetic demo)", color=PRIMARY_INK, fontsize=13, loc="left", pad=12)
    ax.set_xlabel("Generation", color=SECONDARY_INK, fontsize=10)
    ax.set_ylabel("Composite fitness", color=SECONDARY_INK, fontsize=10)
    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xticks(gens)
    legend = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for text in legend.get_texts():
        text.set_color(SECONDARY_INK)


def plot_pareto(data: dict, ax) -> None:
    entries = data["pareto_entries"]
    front_keys = {tuple(k) for k in data["front_keys"]}

    dominated_x, dominated_y = [], []
    front_x, front_y = [], []
    for e in entries:
        key = (round(e["quality"], 6), round(e["cost"], 6))
        if key in front_keys:
            front_x.append(e["cost"])
            front_y.append(e["quality"])
        else:
            dominated_x.append(e["cost"])
            dominated_y.append(e["quality"])

    ax.set_facecolor(SURFACE)
    ax.scatter(dominated_x, dominated_y, color=BLUE, alpha=0.35, s=28, label="Dominated trials")
    ax.scatter(front_x, front_y, color=AQUA, s=48, edgecolors=PRIMARY_INK, linewidths=0.5, label="Pareto front", zorder=3)

    ax.set_title("Quality vs. cost — Pareto front", color=PRIMARY_INK, fontsize=13, loc="left", pad=12)
    ax.set_xlabel("Cost (token estimate, normalized)", color=SECONDARY_INK, fontsize=10)
    ax.set_ylabel("Answer quality (judge score, 0-1)", color=SECONDARY_INK, fontsize=10)
    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    legend = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for text in legend.get_texts():
        text.set_color(SECONDARY_INK)


def main() -> None:
    data = load_result()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=SURFACE)
    plot_convergence(data, ax1)
    plot_pareto(data, ax2)
    fig.tight_layout(pad=2.0)
    out_path = "kg_optimizer_demo_charts.png"
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
