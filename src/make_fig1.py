"""
Fig 1 schematic for FlowGAT-Paper / Communications Physics submission.

Two-panel figure:
  (A) 4-variant feature-decomposition matrix — which of the two switched
      input channels (x[3:6] direction, x[8] magnitude) carry target
      information vs. a geometric proxy.
  (B) 46-aorta cohort breakdown — split (37 train / 4 val / 5 test) and
      pathology composition of the 5 test cases.

Outputs:  results/figures/fig1_schema.{pdf,png}

Run (from project root):
  /users/scratch1/dancies/conda_envs/py312/bin/python src/make_fig1.py
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D


mpl.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        9.0,
    "axes.titlesize":   10.0,
    "axes.labelsize":   9.0,
    "axes.linewidth":   0.8,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
})

TARGET = "#c0392b"          # leaked-from-target
GEOM   = "#2980b9"          # geometric proxy
INK    = "#1a1a1a"
SOFT   = "#7f8c8d"
HEALTHY = "#27ae60"
COA_R   = "#8e44ad"
COA_F   = "#e67e22"


def _cell(ax, x, y, w, h, label, color, *, fontcolor="white", fontweight="bold"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.06",
        linewidth=0.0, facecolor=color, edgecolor="none",
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            color=fontcolor, fontsize=8.5, fontweight=fontweight)


def panel_a(ax):
    """Panel A — 4-variant feature decomposition matrix."""
    ax.set_xlim(0, 14); ax.set_ylim(0, 6.4)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("A   Four-variant feature decomposition", loc="left",
                 fontsize=12, fontweight="bold", pad=8)

    variants = ["withleak", "leak_dir_only", "leak_mag_only", "noleak"]
    cells = [  # (dir, mag): T = target leak, G = geometric
        ("T", "T"),
        ("T", "G"),
        ("G", "T"),
        ("G", "G"),
    ]
    notes = [
        "both channels leaked",
        r"direction leak only $\rightarrow$ matches withleak",
        r"magnitude leak only $\rightarrow$ worse than noleak",
        "honest geometry-only baseline",
    ]

    # column x positions (left edges) and widths
    x_var,  w_var  = 0.10,  2.10
    x_dir,  w_dir  = 2.45,  3.40
    x_mag,  w_mag  = 6.00,  3.40
    x_note          = 9.55

    # Header row
    ax.text(x_var + w_var / 2, 5.55, "variant",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color=INK)
    ax.text(x_dir + w_dir / 2, 5.55, r"$\mathbf{x}[3{:}6]$  unit direction",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color=INK)
    ax.text(x_mag + w_mag / 2, 5.55, r"$\mathbf{x}[8]$  scalar magnitude",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color=INK)
    ax.text(x_note, 5.55, "interpretation",
            ha="left", va="center", fontsize=9.5, fontweight="bold", color=INK)

    # underline header
    ax.add_patch(Rectangle((0.05, 5.20), 13.90, 0.015,
                           facecolor=SOFT, edgecolor="none"))

    row_h = 0.78
    y0    = 4.30
    for i, (v, (d, m), note) in enumerate(zip(variants, cells, notes)):
        y = y0 - i * (row_h + 0.18)
        ax.text(x_var + 0.05, y + row_h / 2, v, ha="left", va="center",
                fontsize=9.5, fontfamily="monospace", fontweight="bold", color=INK)
        d_color = TARGET if d == "T" else GEOM
        m_color = TARGET if m == "T" else GEOM
        d_lbl = r"$\hat{\mathbf{u}}_{\mathrm{CFD}}$  (target)" if d == "T" else "PCA tangent  (geom.)"
        m_lbl = r"$\|\mathbf{u}_{\mathrm{CFD}}\|/u_c$  (target)" if m == "T" else r"local radius / $L_c$  (geom.)"
        _cell(ax, x_dir, y, w_dir, row_h, d_lbl, d_color)
        _cell(ax, x_mag, y, w_mag, row_h, m_lbl, m_color)
        ax.text(x_note, y + row_h / 2, note, ha="left", va="center",
                fontsize=8.8, color=INK)

    # Legend
    legend_handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=TARGET,
               markeredgecolor="none", markersize=11, label="target leakage"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=GEOM,
               markeredgecolor="none", markersize=11, label="geometric proxy"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(0.01, -0.02),
              frameon=False, ncol=2, fontsize=9, handletextpad=0.4, columnspacing=1.4)


def panel_b(ax):
    """Panel B — cohort breakdown."""
    ax.set_xlim(0, 14); ax.set_ylim(0, 6.4)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("B   46-aorta VMR cohort  (split seed 22, frozen 2026-04-30)",
                 loc="left", fontsize=12, fontweight="bold", pad=8)

    # Split bar
    total = 46.0
    splits = [("train", 37, "#34495e"), ("val", 4, "#7f8c8d"), ("test", 5, "#16a085")]
    x0, y0, bar_w, bar_h = 0.10, 4.50, 13.80, 0.65
    ax.text(x0, y0 + bar_h + 0.18, "46 patient-specific aortic CFD cases  (25 MR + 21 CT)",
            ha="left", va="bottom", fontsize=9.5, color=INK)
    x = x0
    for name, n, color in splits:
        w = bar_w * n / total
        ax.add_patch(Rectangle((x, y0), w, bar_h, facecolor=color, edgecolor="white", lw=1.5))
        if w > 1.4:
            ax.text(x + w / 2, y0 + bar_h / 2, f"{name}  n={n}",
                    ha="center", va="center", color="white",
                    fontsize=9.5, fontweight="bold")
        else:
            ax.text(x + w / 2, y0 - 0.25, f"{name}  n={n}",
                    ha="center", va="top", color=INK, fontsize=8.5)
        x += w

    # Test cohort breakdown
    ax.text(0.10, 3.45, "5 test cases:", ha="left", va="top",
            fontsize=10, fontweight="bold", color=INK)

    cases = [
        ("0007", "healthy",       HEALTHY),
        ("0017", "CoA rigid",     COA_R),
        ("0020", "CoA rigid",     COA_R),
        ("0225", "CoA FSI",       COA_F),
        ("0226", "CoA FSI",       COA_F),
    ]
    col_w = 2.60; gap = 0.15; n = len(cases)
    total_w = n * col_w + (n - 1) * gap
    x = (14 - total_w) / 2
    y, h = 1.40, 1.55
    for cid, kind, color in cases:
        box = FancyBboxPatch((x, y), col_w, h,
                             boxstyle="round,pad=0.0,rounding_size=0.12",
                             facecolor=color, edgecolor="none", lw=0)
        ax.add_patch(box)
        ax.text(x + col_w / 2, y + h * 0.66, cid, ha="center", va="center",
                fontsize=12, fontweight="bold", color="white", fontfamily="monospace")
        ax.text(x + col_w / 2, y + h * 0.28, kind, ha="center", va="center",
                fontsize=9, color="white")
        x += col_w + gap

    # Pathology legend
    legend_handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=HEALTHY,
               markeredgecolor="none", markersize=11, label="healthy (n=1)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COA_R,
               markeredgecolor="none", markersize=11, label="coarctation, rigid wall (n=2)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COA_F,
               markeredgecolor="none", markersize=11, label="coarctation, FSI (n=2)"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(0.01, -0.02),
              frameon=False, ncol=3, fontsize=9,
              handletextpad=0.4, columnspacing=1.4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/figures",
                    help="output directory for fig1_schema.{pdf,png}")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    outdir = (project_root / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(10.0, 7.4),
                                   gridspec_kw=dict(hspace=0.30))
    panel_a(axA)
    panel_b(axB)
    fig.subplots_adjust(left=0.02, right=0.99, top=0.97, bottom=0.03)

    pdf = outdir / "fig1_schema.pdf"
    png = outdir / "fig1_schema.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=args.dpi)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
