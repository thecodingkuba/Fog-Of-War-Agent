"""Generate poster figures from tournament results (learned priors only)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent

# k=10 round-robin (depth 2, opening-plies 4, 100 games/direction)
# All agents except the hand-crafted heuristic. Ordered worst -> best.
# (label, win_rate, color_key)
WIN_RATE_K10 = [
    ("Random",                       6.0,  "Random"),
    ("Dense PPO\n(policy only)",     14.0, "Dense PPO"),
    ("Information Prior\n+ Search",  33.3, "Trained info reward only"),
    ("Policy Prior\n+ Search",       39.7, "Policy Prior + Search"),
    ("Info Prior + Search\n+ Info Bonus", 41.7,
     "Information Prior + Search (Ours)"),
]

# Top-k ablation, all agents except the hand-crafted heuristic.
# Order controls legend/draw order (Ours first so it reads on top).
K_VALUES = [1, 2, 4, 8, 10]
WIN_RATE_BY_K = {
    "Info Prior + Info Bonus": [36.1, 36.3, 35.8, 41.3, 41.7],
    "Policy Prior + Search": [37.3, 35.6, 35.8, 39.6, 39.7],
    "Information Prior + Search": [7.1, 13.2, 20.4, 29.5, 33.3],
    "Dense PPO (policy only)": [36.7, 33.0, 27.7, 17.8, 14.0],
    "Random": [25.2, 16.9, 10.4, 7.0, 6.0],
}
NODES_BY_K = {
    "Info Prior + Info Bonus": [1663, 4448, 11_329, 23_154, 31_150],
    "Policy Prior + Search": [1671, 4419, 10_916, 23_872, 30_655],
    "Information Prior + Search": [2454, 5238, 12_016, 25_778, 33_135],
    "Dense PPO (policy only)": [0, 0, 0, 0, 0],
    "Random": [0, 0, 0, 0, 0],
}

COLORS = {
    # win-rate bar-chart color keys
    "Random": "#D1D5DB",
    "Dense PPO": "#B0792C",
    "Policy Prior + Search": "#4C78A8",
    "Information Prior + Search (Ours)": "#8C1515",
    "Trained info reward only": "#9CA3AF",
    # ablation line color keys
    "Info Prior + Info Bonus": "#8C1515",
    "Information Prior + Search": "#9CA3AF",
    "Dense PPO (policy only)": "#B0792C",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
})


def fig1_win_rate():
    labels = [row[0] for row in WIN_RATE_K10]
    values = [row[1] for row in WIN_RATE_K10]
    colors = [COLORS[row[2]] for row in WIN_RATE_K10]

    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=1.2, width=0.68)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=15,
        )

    ax.set_ylabel("Round-Robin Win Rate")
    ax.set_title("Win Rate Comparison ($k{=}10$, depth ${=}2$)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 52)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(OUT / "win_rate_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "win_rate_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def fig2_ablation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    for name in WIN_RATE_BY_K:
        ax1.plot(
            K_VALUES, WIN_RATE_BY_K[name],
            marker="o", linewidth=2.5, markersize=8,
            label=name, color=COLORS[name],
        )
    ax1.set_xlabel("Candidate Set Size ($k$)")
    ax1.set_ylabel("Win Rate")
    ax1.set_title("Win Rate vs. Top-$k$")
    ax1.set_xticks(K_VALUES)
    ax1.set_ylim(0, 48)
    ax1.grid(alpha=0.25, linestyle="--")
    ax1.legend(loc="lower right", frameon=True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    for name in NODES_BY_K:
        ax2.plot(
            K_VALUES, NODES_BY_K[name],
            marker="s", linewidth=2.5, markersize=8,
            label=name, color=COLORS[name],
        )
    ax2.set_xlabel("Candidate Set Size ($k$)")
    ax2.set_ylabel("Avg Search Nodes / Game")
    ax2.set_title("Search Cost vs. Top-$k$")
    ax2.set_xticks(K_VALUES)
    ax2.grid(alpha=0.25, linestyle="--")
    ax2.legend(loc="upper left", frameon=True, fontsize=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Top-$k$ Ablation (depth ${=}2$)", fontsize=18, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "topk_ablation.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "topk_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_win_rate()
    fig2_ablation()
    print(f"Wrote figures to {OUT}/")
