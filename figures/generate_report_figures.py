"""Report figures: learned/RL agents only (no hand-designed baseline), no 'Ours'.

Distinguishes the two ways information is injected:
  - Information Prior + Search  : information baked into the prior via training reward
  - Information-Bonus Search     : policy prior + inference-time information bonus
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent

# Final round-robin, k=10, depth=2, 100 games/direction, 4 random opening plies.
WIN_RATE_K10 = [
    ("Random",                       6.0,  "#D1D5DB"),
    ("Dense PPO\n(no search)",       14.0, "#B0792C"),
    ("Information Prior\n+ Search",   33.3, "#9CA3AF"),
    ("Policy Prior\n+ Search",        39.7, "#4C78A8"),
    ("Information-Bonus\nSearch",     41.7, "#8C1515"),
]

K_VALUES = [1, 2, 4, 8, 10]
SERIES = {
    "Information-Bonus Search": (
        [36.1, 36.3, 35.8, 41.3, 41.7],
        [1663, 4448, 11_329, 23_154, 31_150],
        "#8C1515",
    ),
    "Policy Prior + Search": (
        [37.3, 35.6, 35.8, 39.6, 39.7],
        [1671, 4419, 10_916, 23_872, 30_655],
        "#4C78A8",
    ),
    "Information Prior + Search": (
        [7.1, 13.2, 20.4, 29.5, 33.3],
        [2454, 5238, 12_016, 25_778, 33_135],
        "#9CA3AF",
    ),
    "Dense PPO (no search)": (
        [36.7, 33.0, 27.7, 17.8, 14.0],
        [0, 0, 0, 0, 0],
        "#B0792C",
    ),
    "Random": (
        [25.2, 16.9, 10.4, 7.0, 6.0],
        [0, 0, 0, 0, 0],
        "#D1D5DB",
    ),
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
})


def fig_win_rate():
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    labels = [r[0] for r in WIN_RATE_K10]
    values = [r[1] for r in WIN_RATE_K10]
    colors = [r[2] for r in WIN_RATE_K10]
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=1.2, width=0.68)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.7,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=13)
    ax.set_ylabel("Round-Robin Win Rate (%)")
    ax.set_title("Win Rate Comparison ($k{=}10$, depth ${=}2$)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 50)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(OUT / "report_win_rate.pdf", bbox_inches="tight")
    fig.savefig(OUT / "report_win_rate.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_ablation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0))
    for name, (wr, _nodes, color) in SERIES.items():
        ax1.plot(K_VALUES, wr, marker="o", linewidth=2.3, markersize=7,
                 label=name, color=color)
    ax1.set_xlabel("Candidate Set Size ($k$)")
    ax1.set_ylabel("Win Rate (%)")
    ax1.set_title("Win Rate vs. Top-$k$")
    ax1.set_xticks(K_VALUES)
    ax1.grid(alpha=0.25, linestyle="--")
    ax1.legend(loc="center right", frameon=True, fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    for name, (_wr, nodes, color) in SERIES.items():
        ax2.plot(K_VALUES, nodes, marker="s", linewidth=2.3, markersize=7,
                 label=name, color=color)
    ax2.set_xlabel("Candidate Set Size ($k$)")
    ax2.set_ylabel("Avg Search Nodes / Game")
    ax2.set_title("Search Cost vs. Top-$k$")
    ax2.set_xticks(K_VALUES)
    ax2.grid(alpha=0.25, linestyle="--")
    ax2.legend(loc="upper left", frameon=True, fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Top-$k$ Ablation (depth ${=}2$)", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "report_topk_ablation.pdf", bbox_inches="tight")
    fig.savefig(OUT / "report_topk_ablation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_win_rate()
    fig_ablation()
    print(f"Wrote report figures to {OUT}/")
