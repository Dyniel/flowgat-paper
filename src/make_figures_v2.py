#!/usr/bin/env python3
"""
Publication-grade figures for the controlled leakage ablation paper (CP).

Reads results from:
  results/aggregated/    (per_case_3seed.csv, aggregate_3seed.csv)
  results/stratified/    (stratified_test.csv)
  results/bootstrap/     (bootstrap_test_*.csv)
  results/diagnostics/   (peak_loc_diagnostic.csv, peak_loc_dispersion.csv)
  results/predictions/   (per-case dumps; used for fig 4 map overlay if present)

Produces:
  Fig 2  4-variant headline (PP@10, angle, peak_loc, ewRMSE) with bootstrap CI
  Fig 3  Per-case grid: 5 cases × 4 metrics × N variants
  Fig 4  Peak-loc world map: predicted vs true peak per (variant, case)
  Fig 5  Per-pathology stratification (rigid_all vs CoA_FSI)
  Fig 6  Decomposition: (PP@10 vs angle) × variants — 2D scatter that
         reveals which leaked feature dominates

Robust to missing variants (only renders rows for variants present in CSV).
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VARIANT_COLORS = {
    "withleak":      "#c62828",
    "leak_dir_only": "#ef6c00",
    "leak_mag_only": "#1565c0",
    "noleak":        "#2e7d32",
}

VARIANT_ORDER = ["withleak", "leak_dir_only", "leak_mag_only", "noleak"]
VARIANT_LABELS = {
    "withleak":      "with-leak (dir + mag)",
    "leak_dir_only": "dir-leak only",
    "leak_mag_only": "mag-leak only",
    "noleak":        "no-leak (geometry)",
}


def variants_in(df: pd.DataFrame) -> List[str]:
    seen = set(df["variant"].unique()) if "variant" in df.columns else set()
    return [v for v in VARIANT_ORDER if v in seen]


# ---------------------------------------------------------------------------
# Fig 2 — Headline 4-variant
# ---------------------------------------------------------------------------

def fig2_headline(agg: pd.DataFrame, out_dir: Path) -> None:
    test = agg[agg.split == "test"].set_index("variant")
    metrics = [
        ("pp10_mean", "pp10_std", "PP@10", True),
        ("angle_err_mean", "angle_err_std", "Angle err [°]", False),
        ("peak_loc_mm_mean", "peak_loc_mm_std",
         "Peak loc median [mm]", False),
        ("ewrmse_he_mean", "ewrmse_he_std", "ewRMSE_HE", False),
    ]
    variants = [v for v in VARIANT_ORDER if v in test.index]
    if len(variants) < 2:
        print("[fig2] not enough variants — skip")
        return

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    x = np.arange(len(variants))
    for ax, (mc, sc, label, hib) in zip(axes, metrics):
        means = [float(test.loc[v, mc]) for v in variants]
        stds  = [float(test.loc[v, sc]) if sc in test.columns else 0
                 for v in variants]
        colors = [VARIANT_COLORS.get(v, "#888") for v in variants]
        ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85,
               edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([VARIANT_LABELS.get(v, v) for v in variants],
                           rotation=20, ha="right", fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Leakage decomposition: identical model + training, "
                 "swap x[:,3:6] (direction) and x[:,8] (magnitude)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_headline.pdf")
    fig.savefig(out_dir / "fig2_headline.png", dpi=150)
    plt.close(fig)
    print(f"[fig2] -> {out_dir / 'fig2_headline.pdf'}")


# ---------------------------------------------------------------------------
# Fig 3 — Per-case grid
# ---------------------------------------------------------------------------

def fig3_per_case_grid(per_case: pd.DataFrame, out_dir: Path) -> None:
    test = per_case[per_case.split == "test"].copy()
    if test.empty:
        print("[fig3] no test rows — skip")
        return
    cases = sorted(test.case.unique())
    variants = variants_in(test)
    metrics = [
        ("pp10_mean", "PP@10", True),
        ("angle_err_mean", "Angle err [°]", False),
        ("ewrmse_he_mean", "ewRMSE_HE", False),
        ("peak_loc_mm_mean", "Peak loc [mm]", False),
    ]
    fig, axes = plt.subplots(len(metrics), 1,
                             figsize=(max(8, 1.2 * len(cases)),
                                      2.2 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
    width = 0.85 / max(len(variants), 1)
    x = np.arange(len(cases))
    for ax, (mc, label, hib) in zip(axes, metrics):
        for vi, v in enumerate(variants):
            vals = []
            for c in cases:
                sub = test[(test.case == c) & (test.variant == v)][mc]
                vals.append(sub.mean() if not sub.empty else np.nan)
            offset = (vi - (len(variants) - 1) / 2) * width
            ax.bar(x + offset, vals, width,
                   label=VARIANT_LABELS.get(v, v),
                   color=VARIANT_COLORS.get(v, "#888"),
                   alpha=0.85, edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        labels_ = [str(c).replace("_3D_RIGID", "")
                   .replace("_3D_FSI_REST", "")
                   for c in cases]
        ax.set_xticklabels(labels_, rotation=25, ha="right", fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        if ax is axes[0]:
            ax.legend(loc="upper right", fontsize=7, ncol=len(variants))
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_per_case_grid.pdf")
    fig.savefig(out_dir / "fig3_per_case_grid.png", dpi=150)
    plt.close(fig)
    print(f"[fig3] -> {out_dir / 'fig3_per_case_grid.pdf'}")


# ---------------------------------------------------------------------------
# Fig 5 — Per-pathology stratification
# ---------------------------------------------------------------------------

def fig5_stratified(strat: pd.DataFrame, out_dir: Path) -> None:
    if strat.empty:
        print("[fig5] empty stratification CSV — skip")
        return
    strata = ["healthy", "CoA_rigid", "rigid_all", "CoA_FSI"]
    metrics = [
        ("PP@10", "val/pp_at_0.100"),
        ("angle°", "val/angle_err_he_mean_deg"),
        ("peak_loc_mm", "clin/peak_loc_mm_mean"),
        ("ewRMSE", "val/ewrmse_he"),
    ]
    variants = sorted(strat["variant"].unique())
    variants = [v for v in VARIANT_ORDER if v in variants]
    if len(variants) < 2:
        print("[fig5] not enough variants — skip")
        return
    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(3.2 * len(metrics), 4.5))
    width = 0.85 / max(len(variants), 1)
    x = np.arange(len(strata))
    for ax, (label, metric_key) in zip(axes, metrics):
        for vi, v in enumerate(variants):
            means, stds = [], []
            for st in strata:
                sub = strat[(strat["variant"] == v) &
                             (strat["stratum"] == st) &
                             (strat["metric"] == metric_key)]
                if sub.empty:
                    means.append(np.nan); stds.append(0)
                else:
                    means.append(float(sub["mean"].iloc[0]))
                    stds.append(float(sub["std"].iloc[0]))
            offset = (vi - (len(variants) - 1) / 2) * width
            ax.bar(x + offset, means, width, yerr=stds, capsize=2,
                   label=VARIANT_LABELS.get(v, v),
                   color=VARIANT_COLORS.get(v, "#888"),
                   alpha=0.85, edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(strata, rotation=20, ha="right", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        if ax is axes[0]:
            ax.legend(loc="upper left", fontsize=7)
    fig.suptitle("Per-pathology stratification (3-seed mean ± std)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_stratified.pdf")
    fig.savefig(out_dir / "fig5_stratified.png", dpi=150)
    plt.close(fig)
    print(f"[fig5] -> {out_dir / 'fig5_stratified.pdf'}")


# ---------------------------------------------------------------------------
# Fig 6 — Decomposition scatter (PP@10 vs angle) per variant
# ---------------------------------------------------------------------------

def fig6_decomposition(agg: pd.DataFrame, out_dir: Path) -> None:
    test = agg[agg.split == "test"].set_index("variant")
    variants = [v for v in VARIANT_ORDER if v in test.index]
    if len(variants) < 3:
        print("[fig6] need ≥3 variants for decomposition — skip")
        return
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for v in variants:
        x = float(test.loc[v, "pp10_mean"])
        y = float(test.loc[v, "angle_err_mean"])
        x_err = float(test.loc[v, "pp10_std"]) if "pp10_std" in test.columns else 0
        y_err = float(test.loc[v, "angle_err_std"]) if "angle_err_std" in test.columns else 0
        c = VARIANT_COLORS.get(v, "#888")
        ax.errorbar(x, y, xerr=x_err, yerr=y_err, fmt="o", markersize=14,
                    color=c, ecolor=c, elinewidth=1.5, capsize=4, alpha=0.9)
        ax.annotate(VARIANT_LABELS.get(v, v), (x, y),
                    xytext=(8, 8), textcoords="offset points", fontsize=9)
    ax.set_xlabel("PP@10 (higher = better)")
    ax.set_ylabel("Angle err [°] (lower = better)")
    ax.set_title("Feature-leakage decomposition on test set "
                 "(3-seed mean ± std)")
    ax.grid(alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_dir / "fig6_decomposition.pdf")
    fig.savefig(out_dir / "fig6_decomposition.png", dpi=150)
    plt.close(fig)
    print(f"[fig6] -> {out_dir / 'fig6_decomposition.pdf'}")


# ---------------------------------------------------------------------------
# Fig 4 — Peak-loc world map (uses dumped predictions if available)
# ---------------------------------------------------------------------------

def fig4_peak_world_map(predictions_dir: Path, out_dir: Path,
                         test_cases: Optional[List[str]] = None) -> None:
    if not predictions_dir.exists():
        print(f"[fig4] no predictions dir at {predictions_dir} — skip")
        return
    variants = [d.name for d in sorted(predictions_dir.iterdir())
                if d.is_dir() and d.name in VARIANT_ORDER]
    if not variants:
        print("[fig4] no variant subdirs in predictions — skip")
        return

    # collect: per case, true peak + per-variant predicted peaks (3-seed mean)
    by_case: Dict[str, Dict] = {}
    for v in variants:
        for seed_dir in sorted((predictions_dir / v).iterdir()):
            if not seed_dir.name.startswith("seed_"):
                continue
            for npz in sorted(seed_dir.glob("test_*.npz")):
                d = np.load(npz, allow_pickle=False)
                pred = d["pred"]; y = d["y"]; pos = d["pos"]
                cid = str(d["case_id"])
                if test_cases and cid not in test_cases:
                    continue
                # true peak
                speed_y = np.linalg.norm(y, axis=-1)
                pt = pos[int(np.argmax(speed_y))]
                # pred peak (top1 -> world frame)
                speed_p = np.linalg.norm(pred, axis=-1)
                pp = pos[int(np.argmax(speed_p))]
                slot = by_case.setdefault(cid, {"pos": pos, "true": pt, "preds": {}})
                slot["preds"].setdefault(v, []).append(pp)

    if not by_case:
        print("[fig4] no test predictions found — skip")
        return

    cases = sorted(by_case.keys())
    n_cases = len(cases)
    fig, axes = plt.subplots(1, n_cases, figsize=(4.5 * n_cases, 4.5),
                             squeeze=False)
    axes = axes.ravel()
    for ax, cid in zip(axes, cases):
        slot = by_case[cid]
        pos = slot["pos"]
        # 2D projection: drop z (could also use PCA — z is fine for aortas)
        ax.scatter(pos[:, 0] * 1000, pos[:, 1] * 1000, s=0.05, c="#dddddd",
                   alpha=0.6, marker=".")
        for v, preds_list in slot["preds"].items():
            preds = np.array(preds_list)  # [n_seeds, 3]
            mean_p = preds.mean(axis=0)
            ax.scatter(mean_p[0] * 1000, mean_p[1] * 1000, s=120,
                       color=VARIANT_COLORS.get(v, "#888"),
                       label=VARIANT_LABELS.get(v, v),
                       edgecolors="black", linewidths=0.6, zorder=4)
            # also raw seeds as small dots
            ax.scatter(preds[:, 0] * 1000, preds[:, 1] * 1000, s=20,
                       color=VARIANT_COLORS.get(v, "#888"), alpha=0.4,
                       zorder=3)
        # true peak
        pt = slot["true"]
        ax.scatter(pt[0] * 1000, pt[1] * 1000, s=200, marker="*",
                   color="black", label="true peak", zorder=5)
        ax.set_title(cid.replace("_3D_RIGID", "").replace("_3D_FSI_REST", ""),
                     fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        if ax is axes[0]:
            ax.legend(fontsize=7, loc="best")
    fig.suptitle("Peak-velocity localization on test cases "
                 "(world frame, x-y projection; per-seed dots, "
                 "large filled = mean across 3 seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_peak_world_map.pdf")
    fig.savefig(out_dir / "fig4_peak_world_map.png", dpi=150)
    plt.close(fig)
    print(f"[fig4] -> {out_dir / 'fig4_peak_world_map.pdf'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-dir",
                    default="/users/scratch1/dancies/flowgat_paper")
    args = ap.parse_args()
    paper_dir = Path(args.paper_dir)
    res = paper_dir / "results"
    fig_dir = res / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Aggregated headline
    agg_path = res / "aggregated" / "aggregate_3seed.csv"
    pc_path  = res / "aggregated" / "per_case_3seed.csv"
    if agg_path.exists():
        agg = pd.read_csv(agg_path)
        fig2_headline(agg, fig_dir)
        fig6_decomposition(agg, fig_dir)
    else:
        print(f"[main] missing {agg_path} — skipping fig2/fig6")

    if pc_path.exists():
        pc = pd.read_csv(pc_path)
        fig3_per_case_grid(pc, fig_dir)
    else:
        print(f"[main] missing {pc_path} — skipping fig3")

    # Stratified
    strat_path = res / "stratified" / "stratified_test.csv"
    if strat_path.exists():
        strat = pd.read_csv(strat_path)
        fig5_stratified(strat, fig_dir)
    else:
        print(f"[main] missing {strat_path} — skipping fig5")

    # Peak-loc world map (requires dumped predictions)
    pred_dir = res / "predictions"
    fig4_peak_world_map(
        pred_dir, fig_dir,
        test_cases=[
            "0007_H_AO_H_3D_RIGID",
            "0017_H_AO_COA_3D_RIGID",
            "0020_H_AO_COA_3D_RIGID",
            "0225_H_AO_COA_3D_FSI_REST",
            "0226_H_AO_COA_3D_FSI_REST",
        ],
    )


if __name__ == "__main__":
    main()
