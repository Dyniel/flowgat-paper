# -*- coding: utf-8 -*-
"""
sUbend Phase E7 — investigate the dP_MAE inversion.

On sUbend the noleak variant shows LOWER dP_MAE (mean 2.2/4.8 mmHg val/test)
than withleak (4.6/8.3) — opposite of the pattern on VMR / Womersley /
Cosserat. Hypothesis: noleak predictions collapse in magnitude (small max
|u|), and the simplified Bernoulli dP = 4 * max(|u|)² happens to match the
true dP scale "for the wrong reason" on these synthetic cases.

This script measures the velocity-magnitude statistics that produce dP
under the existing clinical metric (see
`src/flowgnn_aorta/metrics/clinical.py:peak_pressure_drop_error`):

  dP_pred_mmHg  = 4 * (max ||u_pred||)²       # m/s assumed
  dP_gt_mmHg    = 4 * (max ||u_true||)²

For every prediction NPZ we record per-frame:
  - max ||u_pred||, max ||u_true||
  - mean ||u_pred||, mean ||u_true||
  - dP_pred_mmHg, dP_gt_mmHg, dP_abs_err_mmHg
  - magnitude_ratio = max||u_pred|| / max||u_true||  (collapse indicator)

We then aggregate per (variant, split) reporting:
  - mean / median magnitude_ratio
  - mean / std dP_pred, dP_gt, dP_abs_err
  - correlation(dP_pred, dP_gt) — does noleak track the true scale at all
    or is it just predicting a constant small value?

Output:
  results/diagnostics/subend/dp_investigation_per_frame.csv
  results/diagnostics/subend/dp_investigation_summary.csv
  results/diagnostics/subend/dp_investigation.md
  results/figures/subend_dp_scatter.{pdf,png}

CPU-only; no GPU required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

DEFAULT_VARIANTS = [
    "subend_withleak",
    "subend_leak_dir_only",
    "subend_leak_mag_only",
    "subend_noleak",
]

VARIANT_ORDER = {v: i for i, v in enumerate(DEFAULT_VARIANTS)}


def bernoulli_dp_mmHg(peak_speed_m_s: float) -> float:
    return 4.0 * peak_speed_m_s ** 2


def _split_of(stem: str) -> str:
    for prefix in ("test_", "val_"):
        if stem.startswith(prefix):
            return prefix.rstrip("_")
    return "unknown"


def _case_of(stem: str) -> str:
    for prefix in ("test_", "val_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def collect_per_frame(predictions_root: Path, variants: Sequence[str]) -> List[Dict]:
    rows: List[Dict] = []
    for variant in variants:
        v_dir = predictions_root / variant
        if not v_dir.is_dir():
            print(f"  [skip] {variant}: no predictions dir")
            continue
        for seed_dir in sorted(v_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            try:
                seed = int(seed_dir.name.split("_")[-1])
            except ValueError:
                continue
            for f in sorted(seed_dir.glob("*.npz")):
                stem = f.stem
                split = _split_of(stem)
                if split == "unknown":
                    continue
                case = _case_of(stem)
                d = np.load(f, allow_pickle=True)
                pred = np.asarray(d["pred"], dtype=np.float32)
                y = np.asarray(d["y"], dtype=np.float32)
                speed_pred = np.linalg.norm(pred, axis=-1)
                speed_true = np.linalg.norm(y, axis=-1)
                max_pred = float(speed_pred.max())
                max_true = float(speed_true.max())
                dp_pred = bernoulli_dp_mmHg(max_pred)
                dp_gt = bernoulli_dp_mmHg(max_true)
                rows.append({
                    "variant": variant,
                    "seed": seed,
                    "split": split,
                    "case": case,
                    "max_speed_pred": max_pred,
                    "max_speed_true": max_true,
                    "mean_speed_pred": float(speed_pred.mean()),
                    "mean_speed_true": float(speed_true.mean()),
                    "magnitude_ratio": (max_pred / max_true) if max_true > 0 else float("nan"),
                    "dP_pred_mmHg": dp_pred,
                    "dP_gt_mmHg": dp_gt,
                    "dP_abs_err_mmHg": abs(dp_pred - dp_gt),
                    "dP_signed_err_mmHg": dp_pred - dp_gt,
                })
    return rows


def _stats(xs: List[float]) -> Dict[str, float]:
    arr = np.asarray([x for x in xs if x is not None and np.isfinite(x)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "std": float("nan"), "p10": float("nan"), "p90": float("nan")}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "p10": float(np.quantile(arr, 0.10)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def summarize(rows: List[Dict], variants: Sequence[str]) -> List[Dict]:
    out: List[Dict] = []
    by_cond: Dict = {}
    for r in rows:
        by_cond.setdefault((r["variant"], r["split"]), []).append(r)
    for variant in variants:
        for split in ("val", "test"):
            group = by_cond.get((variant, split), [])
            if not group:
                continue
            n = len(group)
            max_pred = [r["max_speed_pred"] for r in group]
            max_true = [r["max_speed_true"] for r in group]
            ratio = [r["magnitude_ratio"] for r in group]
            dp_pred = [r["dP_pred_mmHg"] for r in group]
            dp_gt = [r["dP_gt_mmHg"] for r in group]
            dp_err = [r["dP_abs_err_mmHg"] for r in group]
            dp_signed = [r["dP_signed_err_mmHg"] for r in group]
            mp = np.asarray(max_pred, dtype=np.float64)
            mt = np.asarray(max_true, dtype=np.float64)
            dpp = np.asarray(dp_pred, dtype=np.float64)
            dpg = np.asarray(dp_gt, dtype=np.float64)
            corr_max = float(np.corrcoef(mp, mt)[0, 1]) if n > 1 else float("nan")
            corr_dp = float(np.corrcoef(dpp, dpg)[0, 1]) if n > 1 else float("nan")
            row = {
                "variant": variant,
                "split": split,
                "n_frames": n,
                "n_seeds": len({r["seed"] for r in group}),
                "n_cases": len({r["case"] for r in group}),
                "max_speed_pred_mean": _stats(max_pred)["mean"],
                "max_speed_true_mean": _stats(max_true)["mean"],
                "magnitude_ratio_mean": _stats(ratio)["mean"],
                "magnitude_ratio_median": _stats(ratio)["median"],
                "magnitude_ratio_p10": _stats(ratio)["p10"],
                "magnitude_ratio_p90": _stats(ratio)["p90"],
                "dP_pred_mean_mmHg": _stats(dp_pred)["mean"],
                "dP_gt_mean_mmHg": _stats(dp_gt)["mean"],
                "dP_abs_err_mean_mmHg": _stats(dp_err)["mean"],
                "dP_signed_err_mean_mmHg": _stats(dp_signed)["mean"],
                "corr_max_speed_pred_vs_true": corr_max,
                "corr_dP_pred_vs_gt": corr_dp,
            }
            out.append(row)
    return out


def write_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        print(f"[dp-inv] wrote {path} (empty)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[dp-inv] wrote {path} ({len(rows)} rows)")


def write_markdown(summary: List[Dict], out_md: Path) -> None:
    lines: List[str] = [
        "# sUbend ΔP investigation",
        "",
        "Tests whether the noleak variant's lower dP_MAE on sUbend is driven "
        "by magnitude collapse (predicted speeds shrink towards zero) rather "
        "than by accurate peak-velocity recovery.",
        "",
        "Bernoulli convention (used by `clinical.peak_pressure_drop_error`):",
        "  `dP[mmHg] = 4 * (max ||u||)^2`",
        "",
        "## Per-variant per-split summary",
        "",
        "| variant | split | n | max|u_pred| | max|u_true| | ratio (mean) | "
        "ratio (median, p10–p90) | dP_pred [mmHg] | dP_gt [mmHg] | "
        "|dP_pred−dP_gt| | signed err | corr(max_pred,max_true) | "
        "corr(dP_pred,dP_gt) |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| {r['variant']} | {r['split']} | {r['n_frames']} | "
            f"{r['max_speed_pred_mean']:.3f} | {r['max_speed_true_mean']:.3f} | "
            f"{r['magnitude_ratio_mean']:.3f} | "
            f"{r['magnitude_ratio_median']:.3f} "
            f"({r['magnitude_ratio_p10']:.3f}–{r['magnitude_ratio_p90']:.3f}) | "
            f"{r['dP_pred_mean_mmHg']:.2f} | {r['dP_gt_mean_mmHg']:.2f} | "
            f"{r['dP_abs_err_mean_mmHg']:.2f} | "
            f"{r['dP_signed_err_mean_mmHg']:+.2f} | "
            f"{r['corr_max_speed_pred_vs_true']:+.3f} | "
            f"{r['corr_dP_pred_vs_gt']:+.3f} |"
        )
    lines.extend([
        "",
        "## Interpretation key",
        "",
        "- `ratio` ≈ 1.0 → peak speed correctly recovered.",
        "- `ratio` ≪ 1.0 → magnitude collapse; predicted speeds shrunk.",
        "- `ratio` ≫ 1.0 → magnitude overshoot.",
        "- low `corr(dP_pred, dP_gt)` → predictions do not track per-case "
        "dP variation; dP_MAE may be a single-scale-match artefact.",
        "- signed err negative → predictions systematically *under*-predict dP.",
        "",
    ])
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))
    print(f"[dp-inv] wrote {out_md}")


def plot_scatter(rows: List[Dict], variants: Sequence[str], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[dp-inv] matplotlib unavailable, skipping scatter ({exc})")
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    by_variant: Dict[str, List[Dict]] = {}
    for r in rows:
        by_variant.setdefault(r["variant"], []).append(r)

    flat = axes.ravel()
    for ax, variant in zip(flat, variants):
        rs = by_variant.get(variant, [])
        if not rs:
            ax.set_title(f"{variant}\n(no data)")
            ax.set_axis_off()
            continue
        test_rs = [r for r in rs if r["split"] == "test"]
        val_rs = [r for r in rs if r["split"] == "val"]
        for subset, color, label in (
            (val_rs, "tab:blue", "val"),
            (test_rs, "tab:orange", "test"),
        ):
            if not subset:
                continue
            x = [r["dP_gt_mmHg"] for r in subset]
            y = [r["dP_pred_mmHg"] for r in subset]
            ax.scatter(x, y, s=8, alpha=0.5, color=color, label=label)
        lim_max = max(
            max((r["dP_gt_mmHg"] for r in rs), default=1.0),
            max((r["dP_pred_mmHg"] for r in rs), default=1.0),
        )
        ax.plot([0, lim_max], [0, lim_max], "k--", lw=1, alpha=0.6)
        ax.set_title(variant)
        ax.set_xlabel("dP_gt [mmHg]")
        ax.set_ylabel("dP_pred [mmHg]")
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("sUbend — predicted vs ground-truth Bernoulli ΔP", y=1.0)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = out_dir / f"subend_dp_scatter.{ext}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        print(f"[dp-inv] wrote {path}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--figures_dir", default=None)
    ap.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS)
    args = ap.parse_args()

    pred_root = Path(args.predictions_root)
    out_dir = Path(args.out_dir)
    figures_dir = Path(args.figures_dir) if args.figures_dir else out_dir
    variants = list(args.variants)
    rows = collect_per_frame(pred_root, variants)
    print(f"[dp-inv] collected {len(rows)} frame rows")
    write_csv(rows, out_dir / "dp_investigation_per_frame.csv")
    summary = summarize(rows, variants)
    write_csv(summary, out_dir / "dp_investigation_summary.csv")
    write_markdown(summary, out_dir / "dp_investigation.md")
    plot_scatter(rows, variants, figures_dir)
    print("[dp-inv] done")


if __name__ == "__main__":
    main()
