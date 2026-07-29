"""
Renders charts for the REAL bridge-inference demo run (kg_optimizer.demo_real_
bridges_run) — real KG snapshot, real dialog_agent.kg_inference_engine calls,
no synthetic data. Same palette as demo_plot.py (dataviz skill reference).

Run: python -m kg_optimizer.demo_plot_real
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
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"


def load_result(path: str = "kg_optimizer_real_bridges_result.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)


def plot_convergence(data: dict, ax) -> None:
    gens = data["generations"]
    ax.plot(gens, data["best_per_gen"], color=BLUE, linewidth=2, marker="o", markersize=5, label="Best composite fitness")
    ax.plot(gens, data["mean_per_gen"], color=ORANGE, linewidth=2, marker="o", markersize=5, label="Mean composite fitness")
    ax.set_title("GA convergence — real bridge inference (LifeSciences KG)", color=PRIMARY_INK, fontsize=12, loc="left", pad=12)
    ax.set_xlabel("Generation", color=SECONDARY_INK, fontsize=10)
    ax.set_ylabel("Composite fitness (real F1 vs. gold bridges)", color=SECONDARY_INK, fontsize=10)
    ax.set_xticks(gens)
    _style_axes(ax)
    legend = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for text in legend.get_texts():
        text.set_color(SECONDARY_INK)


def plot_pareto(data: dict, ax) -> None:
    entries = data["pareto_entries"]
    front_keys = {tuple(k) for k in data["front_keys"]}
    dominated_x, dominated_y, front_x, front_y = [], [], [], []
    for e in entries:
        key = (round(e["quality"], 6), round(e["cost"], 6))
        (front_x if key in front_keys else dominated_x).append(e["cost"])
        (front_y if key in front_keys else dominated_y).append(e["quality"])

    ax.scatter(dominated_x, dominated_y, color=BLUE, alpha=0.35, s=32, label="Dominated trials")
    ax.scatter(front_x, front_y, color=AQUA, s=52, edgecolors=PRIMARY_INK, linewidths=0.5, label="Pareto front", zorder=3)
    ax.set_title("Bridge quality proxy vs. real latency", color=PRIMARY_INK, fontsize=12, loc="left", pad=12)
    ax.set_xlabel("Inference wall-clock time (s, real)", color=SECONDARY_INK, fontsize=10)
    ax.set_ylabel("Bridge F1 vs. real gold bridges (0-1)", color=SECONDARY_INK, fontsize=10)
    _style_axes(ax)
    legend = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for text in legend.get_texts():
        text.set_color(SECONDARY_INK)


def main() -> None:
    data = load_result()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=SURFACE)
    plot_convergence(data, ax1)
    plot_pareto(data, ax2)
    fig.tight_layout(pad=2.0)
    out_path = "kg_optimizer_real_bridges_charts.png"
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
