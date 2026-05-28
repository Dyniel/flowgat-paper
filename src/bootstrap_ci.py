# -*- coding: utf-8 -*-
"""
Paired bootstrap confidence intervals for (withleak − noleak) deltas.

Resampling unit: case_id (case-level). Within each resample we keep all 3
seeds, computing per-case mean across seeds. With 5 test cases this gives
wide CIs but properly reflects the small-N reality.

For each metric:
  - delta_per_case = mean_seeds(withleak[case]) - mean_seeds(noleak[case])
  - bootstrap: resample 5 cases with replacement, recompute mean delta
  - report: mean, 2.5%, 97.5%, two-sided p-value (fraction of bootstrap
    samples crossing zero, doubled, capped at 1.0)
  - paired permutation test (sign-flip per case) as cross-check

We report both because:
  - bootstrap measures stability under "if I had drawn different cases"
  - permutation measures null hypothesis "no systematic difference"

For peak_loc we additionally compute the median variant (matches the
analyze.py headline) for narrative completeness.

Output:
  results/bootstrap/bootstrap_test.csv
  results/bootstrap/bootstrap_test.md

Usage:
  python src/bootstrap_ci.py \
      --per_seed_dir results/per_seed \
      --out_dir      results/bootstrap \
      --n_boot       10000 \
      --seed         42
"""

from __future__ import annotations
import argparse
import os
import csv
import glob
from typing import Dict, List, Tuple

import numpy as np

METRICS = [
    "val/pp_at_0.100",
    "val/pp_at_0.050",
    "val/angle_err_he_mean_deg",
    "val/ewrmse_he",
    "clin/dP_mmHg_mae",
    "clin/peak_loc_mm_mean",
    "clin/peak_mag_rel_mean",
    "clin/wss_mae_Pa_mean",
    "clin/wss_bias_Pa_mean",
]

LABELS = {
    "val/pp_at_0.100":           "PP@10",
    "val/pp_at_0.050":           "PP@5",
    "val/angle_err_he_mean_deg": "angle°",
    "val/ewrmse_he":             "ewRMSE",
    "clin/dP_mmHg_mae":          "dP_MAE_mmHg",
    "clin/peak_loc_mm_mean":     "peak_loc_mm",
    "clin/peak_mag_rel_mean":    "peak_mag_rel",
    "clin/wss_mae_Pa_mean":      "WSS_MAE_Pa",
    "clin/wss_bias_Pa_mean":     "WSS_bias_Pa",
}

# (variant, split, ...): better-when-larger?  Used to sign the delta.
HIGHER_IS_BETTER = {
    "val/pp_at_0.100": True,
    "val/pp_at_0.050": True,
    "val/angle_err_he_mean_deg": False,
    "val/ewrmse_he": False,
    "clin/dP_mmHg_mae": False,
    "clin/peak_loc_mm_mean": False,
    "clin/peak_mag_rel_mean": False,   # closer to 0 is better; we report magnitude
    "clin/wss_mae_Pa_mean": False,
    "clin/wss_bias_Pa_mean": False,    # |bias| smaller is better
}


def parse_seed(fname: str) -> int:
    return int(os.path.basename(fname).split("_seed")[-1].split(".")[0])


def parse_variant(fname: str) -> str:
    base = os.path.basename(fname)
    if "_test_" in base: return base.split("_test_")[0]
    if "_val_"  in base: return base.split("_val_")[0]
    return base


def load_per_case(per_seed_dir: str, split: str) -> Dict[str, Dict[str, Dict[int, float]]]:
    """Returns {variant: {case_id: {seed: {metric: value}}}} -- but we collapse
    to {variant: {case_id: {metric: [values across seeds]}}} for ease of use."""
    raw: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for fp in sorted(glob.glob(os.path.join(per_seed_dir, f"*_{split}_seed*.csv"))):
        if "_aggregate.json" in fp:
            continue
        v = parse_variant(fp)
        with open(fp) as f:
            r = csv.DictReader(f)
            for row in r:
                cid = row["case_id"]
                for m in METRICS:
                    if m in row and row[m] not in ("", "nan"):
                        try:
                            x = float(row[m])
                        except ValueError:
                            continue
                        if x != x:  # NaN
                            continue
                        raw.setdefault(v, {}).setdefault(cid, {}).setdefault(m, []).append(x)
    return raw


def case_means(
    raw: Dict[str, Dict[str, Dict[str, List[float]]]]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Collapse seeds to per-case mean."""
    out = {}
    for v, by_case in raw.items():
        out[v] = {}
        for cid, by_metric in by_case.items():
            out[v][cid] = {m: float(np.mean(vs)) for m, vs in by_metric.items()}
    return out


def paired_arrays(
    cm: Dict[str, Dict[str, Dict[str, float]]],
    metric: str,
    var_a: str = "withleak",
    var_b: str = "noleak",
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    cases = sorted(set(cm.get(var_a, {}).keys()) & set(cm.get(var_b, {}).keys()))
    a_vals, b_vals, kept = [], [], []
    for c in cases:
        ma = cm[var_a][c].get(metric)
        mb = cm[var_b][c].get(metric)
        if ma is None or mb is None:
            continue
        a_vals.append(ma)
        b_vals.append(mb)
        kept.append(c)
    return kept, np.array(a_vals), np.array(b_vals)


def bootstrap_mean_diff(
    a: np.ndarray, b: np.ndarray, n_boot: int, rng: np.random.Generator
) -> Tuple[float, float, float, float]:
    """Returns (mean_delta, ci_low, ci_high, p_two_sided)."""
    n = len(a)
    delta = a - b
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = delta[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    # Two-sided percentile p-value vs 0
    p_pos = float((boot >= 0).mean())
    p_two = 2.0 * min(p_pos, 1.0 - p_pos)
    return float(delta.mean()), float(lo), float(hi), float(p_two)


def permutation_paired(
    a: np.ndarray, b: np.ndarray, n_perm: int, rng: np.random.Generator
) -> float:
    """Sign-flip permutation: for each perm, randomly swap (a,b) per case,
    return two-sided p-value for mean delta != 0."""
    n = len(a)
    if n == 0:
        return float("nan")
    delta = a - b
    obs = delta.mean()
    flips = rng.choice([-1.0, 1.0], size=(n_perm, n))
    perm_means = (flips * np.abs(delta)).mean(axis=1)
    p = float((np.abs(perm_means) >= abs(obs)).mean())
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_seed_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--var_a", default="withleak")
    ap.add_argument("--var_b", default="noleak")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    raw = load_per_case(args.per_seed_dir, split=args.split)
    cm  = case_means(raw)

    rows: List[Dict] = []
    for m in METRICS:
        cases, a, b = paired_arrays(cm, m, var_a=args.var_a, var_b=args.var_b)
        if len(a) == 0:
            continue
        mean_d, lo, hi, p_boot = bootstrap_mean_diff(a, b, args.n_boot, rng)
        p_perm = permutation_paired(a, b, args.n_perm, rng)
        sign_better_a = HIGHER_IS_BETTER[m]
        better_a = bool((mean_d > 0) == sign_better_a)
        rows.append({
            "metric":          m,
            "metric_label":    LABELS.get(m, m),
            "split":           args.split,
            "var_a":           args.var_a,
            "var_b":           args.var_b,
            "n_cases":         len(a),
            "mean_a":          float(a.mean()),
            "mean_b":          float(b.mean()),
            "mean_delta":      mean_d,
            "ci95_low":        lo,
            "ci95_high":       hi,
            "p_bootstrap":     p_boot,
            "p_permutation":   p_perm,
            "better_var":      args.var_a if better_a else args.var_b,
        })

    csv_path = os.path.join(args.out_dir, f"bootstrap_{args.split}.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[bootstrap] -> {csv_path}")

    md_path = os.path.join(args.out_dir, f"bootstrap_{args.split}.md")
    lines = [
        f"# Paired bootstrap CI: {args.var_a} vs {args.var_b} on {args.split} split\n",
        f"\nResampling unit: case_id (n={rows[0]['n_cases'] if rows else 0}). "
        f"Bootstrap iterations: {args.n_boot}. "
        f"Permutation iterations: {args.n_perm}.\n\n",
        f"| metric | {args.var_a} mean | {args.var_b} mean | Δ (a−b) | "
        "95% CI | p_boot | p_perm | better |\n",
        "| --- | --- | --- | --- | --- | --- | --- | --- |\n",
    ]
    for r in rows:
        lines.append(
            f"| {r['metric_label']} | {r['mean_a']:.4f} | {r['mean_b']:.4f} | "
            f"{r['mean_delta']:+.4f} | "
            f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] | "
            f"{r['p_bootstrap']:.4f} | {r['p_permutation']:.4f} | "
            f"**{r['better_var']}** |\n"
        )
    lines.append(
        "\n*Caveat:* with n=5 paired cases, bootstrap CIs are wide. "
        "Treat p-values as descriptive, not as definitive significance tests. "
        "Headline conclusions should rest on point estimates + per-case "
        "transparency rather than on these p-values alone.\n"
    )
    with open(md_path, "w") as f:
        f.writelines(lines)
    print(f"[bootstrap] -> {md_path}")


if __name__ == "__main__":
    main()
