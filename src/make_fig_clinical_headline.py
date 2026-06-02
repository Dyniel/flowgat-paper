# -*- coding: utf-8 -*-
"""
Generate the clinical-headline figure (proposed Fig 1) across four flow
domains: VMR aortas, Womersley pipe, Cosserat curved-tube sweep, and
synthetic U-bend CFD cohort.

Reads per-seed aggregate JSONs from results/per_seed/, computes the
mean and seed-to-seed standard deviation of four metrics per
(domain, variant), and renders a 1x4 grouped-bar layout (one panel
per metric, four bar groups per panel labelled by domain, four bars
per group labelled by variant).

Metrics:
  - val/angle_err_he_mean_deg   (direction quality)
  - clin/dP_mmHg_mae            (pressure-drop MAE)
  - clin/wss_mae_Pa_mean        (wall-shear-stress MAE)
  - clin/peak_loc_mm_mean       (peak-velocity localisation MAE)

Variant colour palette is fixed across panels so the reader can
trace a variant across metrics visually.

Output:
  results/figures/fig_clinical_headline.{pdf,png}
  results/figures/fig_clinical_headline_values.csv  (the underlying numbers)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

DOMAINS = [
    ("VMR aortas",     "vmr",       [
        "withleak", "leak_dir_only", "leak_mag_only", "noleak",
    ]),
    ("Womersley pipe", "womersley", [
        "womersley_withleak", "womersley_leak_dir_only",
        "womersley_leak_mag_only", "womersley_noleak",
    ]),
    ("Cosserat sweep", "cosserat",  [
        "cosserat_sweep_withleak", "cosserat_sweep_leak_dir_only",
        "cosserat_sweep_leak_mag_only", "cosserat_sweep_noleak",
    ]),
    ("U-bend CFD",     "subend",    [
        "subend_withleak", "subend_leak_dir_only",
        "subend_leak_mag_only", "subend_noleak",
    ]),
]

VARIANT_SHORT = {
    "withleak": "withleak", "leak_dir_only": "dir_only",
    "leak_mag_only": "mag_only", "noleak": "noleak",
    "womersley_withleak": "withleak", "womersley_leak_dir_only": "dir_only",
    "womersley_leak_mag_only": "mag_only", "womersley_noleak": "noleak",
    "cosserat_sweep_withleak": "withleak",
    "cosserat_sweep_leak_dir_only": "dir_only",
    "cosserat_sweep_leak_mag_only": "mag_only",
    "cosserat_sweep_noleak": "noleak",
    "subend_withleak": "withleak", "subend_leak_dir_only": "dir_only",
    "subend_leak_mag_only": "mag_only", "subend_noleak": "noleak",
}

PANELS = [
    ("angle ($^\\circ$)",  "val/angle_err_he_mean_deg", False),
    ("dP MAE [mmHg]",      "clin/dP_mmHg_mae",          False),
    ("WSS MAE [Pa]",       "clin/wss_mae_Pa_mean",      False),
    ("peak loc.\\ MAE [mm]", "clin/peak_loc_mm_mean",   False),
]

VARIANT_ORDER_SHORT = ["withleak", "dir_only", "mag_only", "noleak"]

VARIANT_COLORS = {
    "withleak":  "#1f77b4",
    "dir_only":  "#2ca02c",
    "mag_only":  "#ff7f0e",
    "noleak":    "#d62728",
}

DEFAULT_SEEDS = (1337, 2026, 777)


def parse_seed(path: Path) -> Optional[int]:
    stem = path.stem
    if "_seed" not in stem:
        return None
    tail = stem.rsplit("_seed", 1)[1]
    tail = tail.replace("_aggregate", "")
    try:
        return int(tail)
    except ValueError:
        return None


def load_metric(
    per_seed_dir: Path,
    variant: str,
    metric: str,
    split: str = "test",
    seeds: Optional[Sequence[int]] = DEFAULT_SEEDS,
) -> Optional[List[float]]:
    pattern = f"{variant}_{split}_seed*_aggregate.json"
    allowed = set(seeds) if seeds is not None else None
    values: List[float] = []
    for f in sorted(per_seed_dir.glob(pattern)):
        seed = parse_seed(f)
        if allowed is not None and seed not in allowed:
            continue
        try:
            with open(f) as fh:
                d = json.load(fh)
        except Exception:
            continue
        v = d.get(metric)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv != fv:  # NaN
            continue
        values.append(fv)
    return values or None


def build_table(per_seed_dir: Path, seeds: Optional[Sequence[int]] = DEFAULT_SEEDS) -> Dict:
    out: Dict = {}
    for domain_label, _, variants in DOMAINS:
        out[domain_label] = {}
        for variant in variants:
            short = VARIANT_SHORT.get(variant, variant)
            out[domain_label][short] = {}
            for _, metric, _ in PANELS:
                vals = load_metric(per_seed_dir, variant, metric, split="test", seeds=seeds)
                if vals is None:
                    out[domain_label][short][metric] = (float("nan"), float("nan"), 0)
                else:
                    arr = np.asarray(vals, dtype=np.float64)
                    mu = float(arr.mean())
                    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
                    out[domain_label][short][metric] = (mu, sd, int(arr.size))
    return out


def write_values_csv(table: Dict, out_csv: Path) -> None:
    rows: List[Dict] = []
    for domain_label in table:
        for variant_short in table[domain_label]:
            for _, metric, _ in PANELS:
                mu, sd, n = table[domain_label][variant_short][metric]
                rows.append({
                    "domain": domain_label,
                    "variant": variant_short,
                    "metric": metric,
                    "mean": mu,
                    "std": sd,
                    "n_seeds": n,
                })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"[fig-clinical] wrote {out_csv} ({len(rows)} rows)")


def plot(table: Dict, out_dir: Path, seed_label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(15.0, 4.2))

    domains = list(table.keys())
    n_domains = len(domains)
    n_variants = len(VARIANT_ORDER_SHORT)
    group_w = 0.8
    bar_w = group_w / n_variants
    x_centres = np.arange(n_domains, dtype=np.float64)

    for ax_idx, (panel_label, metric, _) in enumerate(PANELS):
        ax = axes[ax_idx]
        for i, variant_short in enumerate(VARIANT_ORDER_SHORT):
            xs = []
            ys = []
            errs = []
            for d_idx, domain in enumerate(domains):
                rec = table[domain].get(variant_short, {}).get(metric)
                if rec is None or not np.isfinite(rec[0]):
                    continue
                offset = (i - (n_variants - 1) / 2.0) * bar_w
                xs.append(x_centres[d_idx] + offset)
                ys.append(rec[0])
                errs.append(rec[1])
            ax.bar(xs, ys, width=bar_w * 0.95, color=VARIANT_COLORS[variant_short],
                   label=variant_short if ax_idx == 0 else None,
                   yerr=errs, capsize=2, error_kw={"linewidth": 0.6, "ecolor": "black"})
        ax.set_xticks(x_centres)
        ax.set_xticklabels(domains, rotation=20, ha="right", fontsize=8.5)
        ax.set_ylabel(panel_label)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=10)
    fig.suptitle(
        "Clinical-quantity headline across four flow domains "
        f"(test split, {seed_label}; error bars = seed-to-seed std)",
        y=1.10, fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = out_dir / f"fig_clinical_headline.{ext}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        print(f"[fig-clinical] wrote {path}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_seed_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--seeds",
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help=(
            "Comma-separated seeds to include, or 'all'. The paper headline "
            "uses the default three seeds: 1337,2026,777."
        ),
    )
    args = ap.parse_args()

    per_seed_dir = Path(args.per_seed_dir)
    out_dir = Path(args.out_dir)
    if args.seeds.lower() == "all":
        seeds = None
        seed_label = "all available seeds"
    else:
        seeds = tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip())
        seed_label = f"{len(seeds)} seeds ({','.join(str(s) for s in seeds)})"
    table = build_table(per_seed_dir, seeds=seeds)
    write_values_csv(table, out_dir / "fig_clinical_headline_values.csv")
    plot(table, out_dir, seed_label=seed_label)
    print("[fig-clinical] done")


if __name__ == "__main__":
    main()
