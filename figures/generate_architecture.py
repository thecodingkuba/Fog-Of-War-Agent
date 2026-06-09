"""Architecture / decision-pipeline diagram for the report.

Two rows:
  (1) the CNN-LSTM actor-critic that produces the action prior, and
  (2) the inference pipeline: prior -> top-k (+ info bonus) -> restricted search -> move.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent

CARDINAL = "#8C1515"
BLUE = "#4C78A8"
GOLD = "#B0792C"
GRAY = "#5B6770"
LIGHT = "#EDEFF2"

plt.rcParams.update({"font.family": "sans-serif"})


def box(ax, x, y, w, h, text, face, edge, fontcolor="white", fs=11, bold=True):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.6, edgecolor=edge, facecolor=face, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=fontcolor, fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=3, wrap=True)


def arrow(ax, x0, y0, x1, y1, color=GRAY):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
        linewidth=1.8, color=color, zorder=1))


def main():
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    # ---- Row 1: network (actor-critic) -------------------------------------
    y1 = 3.55
    h = 1.05
    box(ax, 0.15, y1, 1.7, h, "Partial\nObservation\n$o_t$\n($14{\\times}8{\\times}8$)",
        LIGHT, GRAY, fontcolor="#1F2937", fs=10)
    arrow(ax, 1.85, y1 + h / 2, 2.45, y1 + h / 2)
    box(ax, 2.45, y1, 1.55, h, "CNN\nencoder", BLUE, "#2F4D6E", fs=12)
    arrow(ax, 4.0, y1 + h / 2, 4.6, y1 + h / 2)
    box(ax, 4.6, y1, 1.55, h, "LSTM\n(belief\nmemory)", CARDINAL, "#5E0E0E", fs=12)

    arrow(ax, 6.15, y1 + h * 0.72, 6.85, y1 + h * 0.9)
    arrow(ax, 6.15, y1 + h * 0.28, 6.85, y1 + h * 0.1)
    box(ax, 6.85, y1 + 0.6, 2.2, 0.55,
        "Policy head: prior $\\pi(a\\,|\\,o_t)$", GOLD, "#7A5318", fs=10.5)
    box(ax, 6.85, y1 - 0.1, 2.2, 0.55,
        "Value head: $V(o_t)$", "#6B7280", "#374151", fs=10.5)

    ax.text(5.5, y1 + h + 0.32, "CNN--LSTM Actor--Critic (PPO self-play)",
            ha="center", va="bottom", fontsize=12.5, fontweight="bold",
            color="#111827")

    # ---- Row 2: inference pipeline -----------------------------------------
    y2 = 1.35
    hh = 1.0
    # elbow connector: policy head (prior) -> candidate-scores box
    elbow_x = [6.85, 6.4, 6.4, 1.18]
    elbow_y = [y1 + 0.875, y1 + 0.875, 2.95, 2.95]
    ax.plot(elbow_x, elbow_y, color=GOLD, linewidth=1.8, zorder=1,
            solid_capstyle="round")
    arrow(ax, 1.18, 2.95, 1.18, y2 + hh, color=GOLD)
    ax.text(3.7, 3.02, "action prior", ha="center", va="bottom",
            fontsize=9.5, style="italic", color=GOLD)

    box(ax, 0.15, y2, 2.05, hh,
        "Candidate scores\n$s(a)=\\pi(a)+\\lambda I(a)$", GOLD, "#7A5318", fs=10)
    arrow(ax, 2.2, y2 + hh / 2, 2.8, y2 + hh / 2)
    box(ax, 2.8, y2, 1.9, hh, "Top-$k$\ncandidate set\n$A_k$", BLUE, "#2F4D6E", fs=11)
    arrow(ax, 4.7, y2 + hh / 2, 5.3, y2 + hh / 2)
    box(ax, 5.3, y2, 2.45, hh,
        "Depth-2 restricted\nnegamax search\n(over $A_k$)", CARDINAL, "#5E0E0E", fs=11)
    arrow(ax, 7.75, y2 + hh / 2, 8.35, y2 + hh / 2)
    box(ax, 8.35, y2, 1.85, hh,
        "Selected move\n$a_t=\\arg\\max_a Q(o_t,a)$", "#374151", "#111827", fs=10)

    ax.text(5.1, y2 - 0.32, "Inference: prior-guided candidate selection $\\rightarrow$ restricted search",
            ha="center", va="top", fontsize=12.5, fontweight="bold", color="#111827")

    fig.tight_layout()
    fig.savefig(OUT / "architecture.pdf", bbox_inches="tight")
    fig.savefig(OUT / "architecture.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote architecture figure to {OUT}/")


if __name__ == "__main__":
    main()
