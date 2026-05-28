#!/usr/bin/env python3
"""
Generate the 5 paper figures from results/aggregated/.

Figures:
  Fig 2: Leakage delta bar plot — central Claim A.
  Fig 3: Per-case PP@10 / angle / peak_loc breakdown (heatmap-style).
  Fig 4: Peak_loc benchmark — central Claim B (pure-geom wins).
  Fig 5: Ablation table from historical results (text-only artefact).

Fig 1 (pipeline diagram) and Fig 3 qualitative velocity field are produced by
external tooling (Inkscape / ParaView) and not by this script.

Output: results/figures/{fig2_leakage_delta, fig3_per_case, fig4_peak_loc}.{pdf,png}
"""

from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def fig2_leakage_delta(agg_3seed: pd.DataFrame, out_dir: Path) -> None:
    test = agg_3seed[agg_3seed.split == "test"].set_index("variant")
    if "withleak" not in test.index or "noleak" not in test.index:
        print("[fig2] missing withleak/noleak rows — skip")
        return
    metrics = [("pp10_mean", "pp10_std", "PP@10"),
               ("angle_err_mean", "angle_err_std", "Angle err [°]"),
               ("peak_loc_mm_mean", "peak_loc_mm_std", "Peak loc [mm]"),
               ("ewrmse_he_mean", "ewrmse_he_std", "ewRMSE_HE [m/s]")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    x = np.arange(2)
    for ax, (mc, sc, label) in zip(axes, metrics):
        wl = test.loc["withleak", mc]
        wls = test.loc["withleak", sc] if sc in test.columns else 0
        nl = test.loc["noleak", mc]
        nls = test.loc["noleak", sc] if sc in test.columns else 0
        ax.bar(x, [wl, nl], yerr=[wls, nls], capsize=5,
               color=["#c62828", "#2e7d32"], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(["with-leak", "no-leak"])
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Leakage delta: identical model + identical training, "
                 "swap x[:,3:6] (3 seeds × 5 test cases)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_leakage_delta.pdf")
    fig.savefig(out_dir / "fig2_leakage_delta.png", dpi=150)
    plt.close(fig)
    print(f"[fig2] -> {out_dir / 'fig2_leakage_delta.pdf'}")


def fig3_per_case(per_case: pd.DataFrame, out_dir: Path) -> None:
    test = per_case[per_case.split == "test"].copy()
    if test.empty:
        print("[fig3] no test rows — skip")
        return
    cases = sorted(test.case.unique())
    metrics = [("pp10_mean", "PP@10"),
               ("angle_err_mean", "Angle err [°]"),
               ("peak_loc_mm_mean", "Peak loc [mm]")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    width = 0.35
    x = np.arange(len(cases))
    for ax, (mc, label) in zip(axes, metrics):
        wl = [test[(test.case == c) & (test.variant == "withleak")][mc].mean()
              for c in cases]
        nl = [test[(test.case == c) & (test.variant == "noleak")][mc].mean()
              for c in cases]
        ax.bar(x - width / 2, wl, width, label="with-leak", color="#c62828", alpha=0.8)
        ax.bar(x + width / 2, nl, width, label="no-leak", color="#2e7d32", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in cases], rotation=30, ha="right", fontsize=8)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Per-case test-set metrics (3-seed mean)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_per_case.pdf")
    fig.savefig(out_dir / "fig3_per_case.png", dpi=150)
    plt.close(fig)
    print(f"[fig3] -> {out_dir / 'fig3_per_case.pdf'}")


def fig4_peak_loc(per_case: pd.DataFrame, out_dir: Path) -> None:
    """Show that no-leak (geom-only) matches with-leak on peak_loc — Claim B."""
    test = per_case[per_case.split == "test"].copy()
    if test.empty:
        print("[fig4] no test rows — skip")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    data = []
    labels = []
    for variant, color in [("withleak", "#c62828"), ("noleak", "#2e7d32")]:
        vals = test[test.variant == variant]["peak_loc_mm_mean"].dropna().tolist()
        if vals:
            data.append(vals)
            labels.append(variant)
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], ["#c62828", "#2e7d32"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Peak localization error [mm]")
    ax.set_title("Peak localization (lower is better) — pure geometry matches with-leak")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_peak_loc.pdf")
    fig.savefig(out_dir / "fig4_peak_loc.png", dpi=150)
    plt.close(fig)
    print(f"[fig4] -> {out_dir / 'fig4_peak_loc.pdf'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-dir",
                    default="/users/scratch1/dancies/flowgat_paper")
    args = ap.parse_args()
    paper_dir = Path(args.paper_dir)
    agg_dir = paper_dir / "results" / "aggregated"
    fig_dir = paper_dir / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    per_case = pd.read_csv(agg_dir / "per_case_3seed.csv")
    whole = pd.read_csv(agg_dir / "aggregate_3seed.csv")

    fig2_leakage_delta(whole, fig_dir)
    fig3_per_case(per_case, fig_dir)
    fig4_peak_loc(per_case, fig_dir)


if __name__ == "__main__":
    main()
