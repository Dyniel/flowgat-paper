# -*- coding: utf-8 -*-
"""
Peak-localization degeneracy check on dumped predictions.

CLAIM B HYPOTHESIS:
  noleak peak_loc < withleak peak_loc on test, hence "geometry-only model
  preserves peak localization".

DEGENERACY HYPOTHESIS (alternative):
  noleak model predicts a ~constant normalized peak position (e.g., always
  near the aortic root) regardless of input geometry. This coincidentally
  matches the true peak in some cases (rigid CoA with peak near root),
  fails on others, and produces small Euclidean distance only because the
  test set is biased toward root-peak cases.

This script tests the degeneracy hypothesis by:

  1. Computing the predicted peak position in three frames per case:
       (a) world frame (mm)               -- for cross-checking with eval CSV
       (b) bbox-normalized               -- in [0,1]^3, normalised by aorta AABB
       (c) PCA-axis-projected            -- scalar in [0,1] along principal axis

  2. Across ALL test cases for a given (variant, seed): if normalized peak
     positions cluster (small std), the model is producing a near-constant
     prediction independent of geometry.

  3. Comparing dispersion (std of normalized peak across cases) for
     withleak vs noleak. A non-degenerate model adapts predictions to each
     aorta's geometry → high std. A degenerate model → low std.

Output:
  results/diagnostics/peak_loc_diagnostic.csv     -- per (variant, seed, case)
  results/diagnostics/peak_loc_dispersion.csv     -- per (variant, seed)
  results/diagnostics/peak_loc_diagnostic.md      -- summary report
  results/figures/diag_peak_loc_normalized.png/pdf

Usage:
  python src/peak_loc_diagnostic.py \
      --predictions_dir results/predictions \
      --out_dir         results/diagnostics \
      --fig_dir         results/figures
"""

from __future__ import annotations
import os
import sys
import json
import glob
import argparse
from typing import Dict, List, Tuple

import numpy as np


def load_dump(path: str) -> Dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def peak_position_world(pred: np.ndarray, pos: np.ndarray, topk: int = 1) -> np.ndarray:
    """Predicted peak in world frame. Top-k centroid (matches evaluator)."""
    speed = np.linalg.norm(pred, axis=-1)
    if topk <= 1:
        i = int(np.argmax(speed))
        return pos[i]
    # top-k centroid weighted by speed
    idx = np.argpartition(-speed, topk - 1)[:topk]
    w = speed[idx]
    w = w / max(w.sum(), 1e-12)
    return (pos[idx] * w[:, None]).sum(axis=0)


def true_peak_world(y: np.ndarray, pos: np.ndarray) -> np.ndarray:
    speed = np.linalg.norm(y, axis=-1)
    i = int(np.argmax(speed))
    return pos[i]


def bbox_normalize(p: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Map a 3D point into the aorta's AABB frame, ∈ [0,1]^3 ideally."""
    pmin = pos.min(axis=0)
    pmax = pos.max(axis=0)
    extent = np.maximum(pmax - pmin, 1e-9)
    return (p - pmin) / extent


def pca_axis_project(p: np.ndarray, pos: np.ndarray) -> Tuple[float, float]:
    """
    Project a point onto the aorta's first principal axis (descending
    aorta is typically the elongated dimension). Return:
      - scalar coordinate normalized to [0,1] along the axis
      - aorta length along that axis (mm)
    """
    centroid = pos.mean(axis=0)
    centered = pos - centroid
    # Principal axis = top eigenvector of covariance
    # Use SVD on centered (more numerically stable than eig)
    # Subsample for speed if N very large
    if centered.shape[0] > 50000:
        idx = np.random.default_rng(0).choice(centered.shape[0], 50000, replace=False)
        c = centered[idx]
    else:
        c = centered
    u, s, vh = np.linalg.svd(c, full_matrices=False)
    axis = vh[0]
    # Project all nodes to find length along axis
    proj_all = centered @ axis
    pmin, pmax = proj_all.min(), proj_all.max()
    length = pmax - pmin
    # Project query point
    proj_p = (p - centroid) @ axis
    norm = (proj_p - pmin) / max(length, 1e-9)
    return float(norm), float(1000.0 * length)  # length in mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fig_dir", default=None)
    ap.add_argument("--splits", nargs="+", default=["test", "val"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.fig_dir:
        os.makedirs(args.fig_dir, exist_ok=True)

    # Discover (variant, seed) directories
    dump_root = args.predictions_dir
    variants = sorted(
        d for d in os.listdir(dump_root) if os.path.isdir(os.path.join(dump_root, d))
    )

    rows: List[Dict] = []   # per (variant, seed, case, split)
    for variant in variants:
        seeds = sorted(
            d.replace("seed_", "")
            for d in os.listdir(os.path.join(dump_root, variant))
            if d.startswith("seed_")
        )
        for seed in seeds:
            seed_dir = os.path.join(dump_root, variant, f"seed_{seed}")
            for split in args.splits:
                npz_files = sorted(
                    glob.glob(os.path.join(seed_dir, f"{split}_*.npz"))
                )
                for fpath in npz_files:
                    d = load_dump(fpath)
                    pred = d["pred"]
                    y = d["y"]
                    pos = d["pos"]
                    case_id = str(d["case_id"])

                    p_pred = peak_position_world(pred, pos, topk=1)
                    p_true = true_peak_world(y, pos)

                    # World-frame distance (mm)
                    dist_mm = float(1000.0 * np.linalg.norm(p_pred - p_true))

                    # Bbox-normalized
                    bn_pred = bbox_normalize(p_pred, pos)
                    bn_true = bbox_normalize(p_true, pos)

                    # PCA-axis projection
                    pca_pred, length_mm = pca_axis_project(p_pred, pos)
                    pca_true, _ = pca_axis_project(p_true, pos)

                    rows.append({
                        "variant": variant,
                        "seed": seed,
                        "split": split,
                        "case_id": case_id,
                        "n_nodes": int(pred.shape[0]),
                        "peak_loc_mm": dist_mm,
                        "pred_world_x": float(p_pred[0]),
                        "pred_world_y": float(p_pred[1]),
                        "pred_world_z": float(p_pred[2]),
                        "true_world_x": float(p_true[0]),
                        "true_world_y": float(p_true[1]),
                        "true_world_z": float(p_true[2]),
                        "pred_bbox_x": float(bn_pred[0]),
                        "pred_bbox_y": float(bn_pred[1]),
                        "pred_bbox_z": float(bn_pred[2]),
                        "true_bbox_x": float(bn_true[0]),
                        "true_bbox_y": float(bn_true[1]),
                        "true_bbox_z": float(bn_true[2]),
                        "pred_pca_norm": pca_pred,
                        "true_pca_norm": pca_true,
                        "aorta_length_mm": length_mm,
                    })

    # CSV per case
    import csv
    csv_path = os.path.join(args.out_dir, "peak_loc_diagnostic.csv")
    keys = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[diag] wrote {len(rows)} rows -> {csv_path}")

    # Dispersion analysis: per (variant, seed, split), how spread are the
    # normalized predicted peaks across cases?
    disp_rows: List[Dict] = []
    by_group: Dict[Tuple[str, str, str], List[Dict]] = {}
    for r in rows:
        key = (r["variant"], r["seed"], r["split"])
        by_group.setdefault(key, []).append(r)

    for (variant, seed, split), group in sorted(by_group.items()):
        if len(group) == 0:
            continue
        bn = np.array([[r["pred_bbox_x"], r["pred_bbox_y"], r["pred_bbox_z"]]
                       for r in group])
        pc = np.array([r["pred_pca_norm"] for r in group])
        bn_true = np.array([[r["true_bbox_x"], r["true_bbox_y"], r["true_bbox_z"]]
                            for r in group])
        pc_true = np.array([r["true_pca_norm"] for r in group])
        disp_rows.append({
            "variant": variant,
            "seed": seed,
            "split": split,
            "n_cases": len(group),
            # dispersion of predicted normalized peaks across cases
            "pred_bbox_std_x": float(bn[:, 0].std(ddof=0)),
            "pred_bbox_std_y": float(bn[:, 1].std(ddof=0)),
            "pred_bbox_std_z": float(bn[:, 2].std(ddof=0)),
            "pred_bbox_std_mean": float(bn.std(axis=0, ddof=0).mean()),
            "pred_pca_std": float(pc.std(ddof=0)),
            # for reference: dispersion of TRUE normalized peaks (geometry varies!)
            "true_bbox_std_mean": float(bn_true.std(axis=0, ddof=0).mean()),
            "true_pca_std": float(pc_true.std(ddof=0)),
            # ratio: pred / true. Values << 1 ⇒ model "ignores" geometry.
            "pred_over_true_bbox_std_ratio":
                float(bn.std(axis=0, ddof=0).mean()
                      / max(bn_true.std(axis=0, ddof=0).mean(), 1e-9)),
            "pred_over_true_pca_std_ratio":
                float(pc.std(ddof=0) / max(pc_true.std(ddof=0), 1e-9)),
            # mean peak loc
            "peak_loc_mm_mean":
                float(np.mean([r["peak_loc_mm"] for r in group])),
            "peak_loc_mm_median":
                float(np.median([r["peak_loc_mm"] for r in group])),
        })

    disp_csv = os.path.join(args.out_dir, "peak_loc_dispersion.csv")
    if disp_rows:
        with open(disp_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(disp_rows[0].keys()))
            w.writeheader()
            w.writerows(disp_rows)
        print(f"[diag] wrote {len(disp_rows)} dispersion rows -> {disp_csv}")

    # Markdown summary
    md_path = os.path.join(args.out_dir, "peak_loc_diagnostic.md")
    lines = []
    lines.append("# Peak-localization degeneracy diagnostic\n")
    lines.append(
        "Hypothesis: noleak model predicts a near-constant normalized peak\n"
        "position regardless of geometry. Test: dispersion of normalized\n"
        "predicted peaks across cases.\n"
    )
    lines.append("\n## Dispersion summary (per variant × seed × split)\n")
    lines.append(
        "| variant | seed | split | n | bbox-std | pca-std | "
        "pred/true bbox-std | pred/true pca-std | peak_loc median (mm) |\n"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
    for r in disp_rows:
        lines.append(
            f"| {r['variant']} | {r['seed']} | {r['split']} | "
            f"{r['n_cases']} | {r['pred_bbox_std_mean']:.3f} | "
            f"{r['pred_pca_std']:.3f} | "
            f"{r['pred_over_true_bbox_std_ratio']:.2f} | "
            f"{r['pred_over_true_pca_std_ratio']:.2f} | "
            f"{r['peak_loc_mm_median']:.1f} |\n"
        )

    # Decision rule
    lines.append("\n## Degeneracy verdict\n")
    lines.append(
        "- Rule of thumb: `pred/true bbox-std ratio` < 0.30 ⇒ model is\n"
        "  predicting a near-constant peak (degenerate).\n"
        "- Ratio between 0.30 and 0.70 ⇒ model partly adapts to geometry.\n"
        "- Ratio > 0.70 ⇒ geometry-aware peak prediction.\n\n"
    )
    for r in disp_rows:
        if r["split"] != "test":
            continue
        ratio = r["pred_over_true_bbox_std_ratio"]
        verdict = ("DEGENERATE" if ratio < 0.30 else
                   "PARTIAL" if ratio < 0.70 else "GEOMETRY-AWARE")
        lines.append(
            f"- {r['variant']} seed={r['seed']} (test): bbox-ratio={ratio:.2f} "
            f"⇒ **{verdict}**\n"
        )

    with open(md_path, "w") as f:
        f.writelines(lines)
    print(f"[diag] wrote summary -> {md_path}")

    # Optional figure: predicted vs true PCA-norm peak position per case,
    # variant comparison.
    if args.fig_dir is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            test_rows = [r for r in rows if r["split"] == "test"]
            if test_rows:
                cases = sorted({r["case_id"] for r in test_rows})
                variants_set = sorted({r["variant"] for r in test_rows})

                fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
                jitter = {v: i * 0.10 for i, v in enumerate(variants_set)}
                colors = {v: f"C{i}" for i, v in enumerate(variants_set)}
                for v in variants_set:
                    xs, ys, errs_lo, errs_hi = [], [], [], []
                    for ci, c in enumerate(cases):
                        seed_vals = [r["pred_pca_norm"] for r in test_rows
                                     if r["case_id"] == c and r["variant"] == v]
                        if not seed_vals:
                            continue
                        m = float(np.mean(seed_vals))
                        s = float(np.std(seed_vals, ddof=0))
                        xs.append(ci + jitter[v])
                        ys.append(m)
                        errs_lo.append(s)
                        errs_hi.append(s)
                    ax.errorbar(xs, ys, yerr=[errs_lo, errs_hi], fmt="o",
                                label=v, color=colors[v], capsize=3, alpha=0.8)
                # True peaks
                truth = []
                for ci, c in enumerate(cases):
                    vals = [r["true_pca_norm"] for r in test_rows
                            if r["case_id"] == c]
                    if vals:
                        truth.append((ci, float(np.mean(vals))))
                if truth:
                    tx, ty = zip(*truth)
                    ax.scatter(tx, ty, marker="x", s=80, color="black",
                               label="true peak", zorder=5)
                ax.set_xticks(range(len(cases)))
                ax.set_xticklabels(
                    [c.replace("_3D_RIGID", "").replace("_3D_FSI_REST", "")
                     for c in cases], rotation=30, ha="right", fontsize=8
                )
                ax.set_ylabel("normalized PCA-axis position (0=inlet, 1=outlet)")
                ax.set_ylim(-0.05, 1.05)
                ax.legend(loc="best", fontsize=8)
                ax.set_title("Predicted peak along principal axis (per test case, "
                             "errorbars = 3-seed std)")
                fig.tight_layout()
                fig.savefig(os.path.join(args.fig_dir, "diag_peak_loc_normalized.png"),
                            dpi=150)
                fig.savefig(os.path.join(args.fig_dir, "diag_peak_loc_normalized.pdf"))
                plt.close(fig)
                print(f"[diag] wrote figure -> {args.fig_dir}/diag_peak_loc_normalized.{{png,pdf}}")
        except Exception as e:
            print(f"[diag] figure failed: {e}")


if __name__ == "__main__":
    main()
