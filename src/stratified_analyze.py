# -*- coding: utf-8 -*-
"""
Per-pathology stratified aggregation of test results.

Test set (5 cases) split into pathology strata:

  healthy:    0007                    (n=1)
  CoA_rigid:  0017, 0020              (n=2)
  CoA_FSI:    0225, 0226              (n=2)

Pure-rigid: 0007, 0017, 0020 (combined healthy + CoA-rigid).
FSI:        0225, 0226.

For each (variant, stratum) report 3-seed mean ± std for headline metrics.
This addresses the reviewer-anticipated question: "is Claim B (peak_loc
preservation) consistent across pathologies?"

Inputs: results/per_seed/<variant>_test_seed<S>.csv (already exists).
Output:
  results/stratified/stratified_test.csv     -- per (variant, stratum, metric)
  results/stratified/stratified_test.md      -- human-readable summary

Usage:
  python src/stratified_analyze.py \
      --per_seed_dir results/per_seed \
      --out_dir      results/stratified
"""

from __future__ import annotations
import argparse
import os
import csv
import glob
from typing import Dict, List, Tuple

import numpy as np

# Pathology strata definitions (test set only)
STRATA: Dict[str, List[str]] = {
    "healthy":     ["0007_H_AO_H_3D_RIGID"],
    "CoA_rigid":   ["0017_H_AO_COA_3D_RIGID", "0020_H_AO_COA_3D_RIGID"],
    "CoA_FSI":     ["0225_H_AO_COA_3D_FSI_REST", "0226_H_AO_COA_3D_FSI_REST"],
    "rigid_all":   ["0007_H_AO_H_3D_RIGID",
                    "0017_H_AO_COA_3D_RIGID", "0020_H_AO_COA_3D_RIGID"],
    "all_test":    ["0007_H_AO_H_3D_RIGID",
                    "0017_H_AO_COA_3D_RIGID", "0020_H_AO_COA_3D_RIGID",
                    "0225_H_AO_COA_3D_FSI_REST", "0226_H_AO_COA_3D_FSI_REST"],
}

# Metrics to report (must exist as columns in per-seed CSVs)
METRICS = [
    "val/pp_at_0.100",
    "val/pp_at_0.050",
    "val/angle_err_he_mean_deg",
    "val/ewrmse_he",
    "clin/peak_loc_mm_mean",
    "clin/peak_mag_rel_mean",
    "clin/wss_mae_Pa_mean",
    "clin/wss_r2_mean",
]

# Renamed for human-readable headers
METRIC_LABELS = {
    "val/pp_at_0.100":           "PP@10",
    "val/pp_at_0.050":           "PP@5",
    "val/angle_err_he_mean_deg": "angle°",
    "val/ewrmse_he":             "ewRMSE",
    "clin/peak_loc_mm_mean":     "peak_loc_mm",
    "clin/peak_mag_rel_mean":    "peak_mag_rel",
    "clin/wss_mae_Pa_mean":      "WSS MAE Pa",
    "clin/wss_r2_mean":          "WSS R²",
}


def parse_seed(fname: str) -> int:
    base = os.path.basename(fname)
    return int(base.split("_seed")[-1].split(".")[0])


def parse_variant(fname: str) -> str:
    base = os.path.basename(fname)
    return base.split("_test_")[0] if "_test_" in base else base.split("_val_")[0]


def load_per_seed(per_seed_dir: str, split: str = "test") -> Dict[Tuple[str, int], Dict[str, Dict[str, float]]]:
    """
    Returns: {(variant, seed): {case_id: {metric: value}}}
    """
    out: Dict[Tuple[str, int], Dict[str, Dict[str, float]]] = {}
    for fpath in sorted(glob.glob(os.path.join(per_seed_dir, f"*_{split}_seed*.csv"))):
        if "_aggregate.json" in fpath:
            continue
        variant = parse_variant(fpath)
        seed = parse_seed(fpath)
        per_case: Dict[str, Dict[str, float]] = {}
        with open(fpath) as f:
            r = csv.DictReader(f)
            for row in r:
                cid = row["case_id"]
                vals = {}
                for m in METRICS:
                    if m in row and row[m] not in ("", "nan"):
                        try:
                            vals[m] = float(row[m])
                        except ValueError:
                            pass
                per_case[cid] = vals
        out[(variant, seed)] = per_case
    return out


def aggregate(
    data: Dict[Tuple[str, int], Dict[str, Dict[str, float]]]
) -> List[Dict]:
    """For each (variant, stratum, metric): 3-seed mean ± std of per-case mean."""
    rows = []
    variants = sorted({v for (v, s) in data.keys()})
    for variant in variants:
        for stratum, cases in STRATA.items():
            for metric in METRICS:
                # For each seed, compute mean over cases-in-stratum
                seed_means = []
                for (v, seed), per_case in data.items():
                    if v != variant:
                        continue
                    case_vals = []
                    for cid in cases:
                        if cid in per_case and metric in per_case[cid]:
                            x = per_case[cid][metric]
                            if not (isinstance(x, float) and (x != x)):
                                case_vals.append(x)
                    if case_vals:
                        seed_means.append(float(np.mean(case_vals)))
                if not seed_means:
                    continue
                arr = np.array(seed_means, dtype=float)
                rows.append({
                    "variant": variant,
                    "stratum": stratum,
                    "n_cases": len(cases),
                    "n_seeds": len(seed_means),
                    "metric": metric,
                    "metric_label": METRIC_LABELS.get(metric, metric),
                    "mean": float(arr.mean()),
                    "std":  float(arr.std(ddof=0)),
                })
    return rows


def write_csv(rows: List[Dict], path: str):
    if not rows:
        return
    keys = ["variant", "stratum", "n_cases", "n_seeds",
            "metric", "metric_label", "mean", "std"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_md(rows: List[Dict], path: str):
    """Pivoted markdown: one table per stratum, columns = variants."""
    variants = sorted({r["variant"] for r in rows})
    strata_order = ["healthy", "CoA_rigid", "rigid_all", "CoA_FSI", "all_test"]
    metrics_order = METRICS

    lines = ["# Per-pathology stratified test metrics (3-seed mean ± std)\n"]
    lines.append(
        "Strata:\n"
        "- **healthy** = {0007} (1 case)\n"
        "- **CoA_rigid** = {0017, 0020} (2 cases)\n"
        "- **rigid_all** = {0007, 0017, 0020} (3 cases, no FSI)\n"
        "- **CoA_FSI** = {0225, 0226} (2 cases)\n"
        "- **all_test** = all 5 test cases (matches headline)\n"
    )

    for stratum in strata_order:
        sub = [r for r in rows if r["stratum"] == stratum]
        if not sub:
            continue
        lines.append(f"\n## {stratum}\n")
        header = "| metric |"
        sep = "| --- |"
        for v in variants:
            header += f" {v} |"
            sep += " --- |"
        lines.append(header + "\n")
        lines.append(sep + "\n")
        for m in metrics_order:
            label = METRIC_LABELS.get(m, m)
            row = f"| {label} |"
            for v in variants:
                cell = next(
                    (r for r in sub if r["variant"] == v and r["metric"] == m),
                    None,
                )
                if cell is None:
                    row += " — |"
                else:
                    row += f" {cell['mean']:.4f} ± {cell['std']:.4f} |"
            lines.append(row + "\n")

    with open(path, "w") as f:
        f.writelines(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_seed_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = load_per_seed(args.per_seed_dir, split=args.split)
    print(f"[stratified] loaded {len(data)} (variant, seed) pairs")
    rows = aggregate(data)
    print(f"[stratified] aggregated {len(rows)} (variant, stratum, metric) rows")

    csv_path = os.path.join(args.out_dir, f"stratified_{args.split}.csv")
    md_path  = os.path.join(args.out_dir, f"stratified_{args.split}.md")
    write_csv(rows, csv_path)
    write_md(rows, md_path)
    print(f"[stratified] -> {csv_path}")
    print(f"[stratified] -> {md_path}")


if __name__ == "__main__":
    main()
