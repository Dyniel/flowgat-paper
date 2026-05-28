# -*- coding: utf-8 -*-
"""
Phase-aware diagnostic on Womersley (Phase E5 — Step 3 in STRATEGY_CP.md).

Hypothesis
----------
Variants without per-node direction leakage (``noleak``, ``leak_mag_only``)
should *flip* their predicted velocity sign when the true Womersley flow
reverses, because the model has only learned a fixed axial polarity from
the training-set distribution. Variants with direction leakage
(``leak_dir_only``, ``withleak``) carry the instantaneous direction in
their features and so should remain phase-invariant.

We test this by binning the test/val cases by their normalised phase
``φ ≡ (ω · t_phase) mod 2π`` and reporting:

  - median folded angle per phase bin per variant
  - median *signed* cosine per phase bin per variant
  - fraction of cases with cos_signed_median < 0   (flip rate)

Outputs
-------
  results/diagnostics/womersley/phase_per_case.csv
  results/diagnostics/womersley/phase_binned.csv
  results/diagnostics/womersley/phase_summary.md
  results/figures/womersley_phase_angle.{png,pdf}
  results/figures/womersley_phase_cos.{png,pdf}

This script consumes the per-case CSV produced by ``womersley_metrics.py``
— it does **not** re-load the .npz dumps, so it is cheap. Run after
``womersley_metrics.py``.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------

VARIANT_ORDER = [
    "womersley_leak_dir_only",
    "womersley_withleak",
    "womersley_noleak",
    "womersley_leak_mag_only",
]

VARIANT_COLORS = {
    "womersley_leak_dir_only": "#1f77b4",   # blue
    "womersley_withleak":      "#2ca02c",   # green
    "womersley_noleak":        "#d62728",   # red
    "womersley_leak_mag_only": "#9467bd",   # purple
}


def _read_per_case(per_case_csv: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(per_case_csv, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not r["variant"].startswith("womersley_"):
                continue
            for k, v in list(r.items()):
                if k in ("variant", "split", "case"):
                    continue
                try:
                    r[k] = float(v) if v not in ("", "nan") else float("nan")
                except ValueError:
                    pass
            rows.append(r)
    return rows


def _stats(vals: List[float]) -> Tuple[float, float, int]:
    clean = [v for v in vals
             if isinstance(v, (int, float))
             and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))]
    n = len(clean)
    if n == 0:
        return float("nan"), float("nan"), 0
    if n == 1:
        return float(clean[0]), 0.0, 1
    return float(mean(clean)), float(stdev(clean)), n


# ---------------------------------------------------------------------------
# Phase binning
# ---------------------------------------------------------------------------

def bin_by_phase(rows: List[Dict], n_bins: int = 4) -> List[Dict]:
    """Group rows into ``n_bins`` equal-width bins over [0, 2π).

    Bin 0 covers forward-peak forcing (cos > 0 strongest near φ=0),
    bin n_bins/2 covers reverse-peak (cos < 0 strongest near φ=π).
    """
    bins = np.linspace(0.0, 2.0 * math.pi, n_bins + 1)
    by_key: Dict[Tuple[str, int], List[Dict]] = {}
    for r in rows:
        ph = float(r.get("phase_norm", float("nan")))
        if math.isnan(ph):
            continue
        b = int(np.clip(np.searchsorted(bins, ph, side="right") - 1,
                        0, n_bins - 1))
        by_key.setdefault((r["variant"], b), []).append(r)

    out: List[Dict] = []
    for (variant, b), group in sorted(by_key.items()):
        rec = {
            "variant": variant,
            "phase_bin": b,
            "phase_lo": float(bins[b]),
            "phase_hi": float(bins[b + 1]),
            "n_cases": len(group),
        }
        for m in ("angle_median_deg", "cos_signed_median",
                  "mag_ratio_median", "frac_flipped",
                  "peak_mag_true", "Q_sign"):
            mu, sd, _ = _stats([r[m] for r in group])
            rec[f"{m}_mean"] = mu
            rec[f"{m}_std"] = sd
        # also: fraction of cases that *flipped overall* in this bin
        flips = [1.0 if (r.get("cos_signed_median", 0.0) < 0.0) else 0.0
                 for r in group]
        rec["case_flip_rate"] = float(np.mean(flips)) if flips else float("nan")
        out.append(rec)
    return out


def write_binned_csv(binned: List[Dict], out_csv: Path) -> None:
    if not binned:
        return
    keys = list(binned[0].keys())
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(binned)
    print(f"[wm-phase] binned -> {out_csv} ({len(binned)} rows)")


# ---------------------------------------------------------------------------
# Per-variant flip-rate summary
# ---------------------------------------------------------------------------

def per_variant_flip_rate(rows: List[Dict]) -> List[Dict]:
    """For each variant, count cases by 'expected reverse' (Q_sign < 0).

    Reports:
      - overall median signed cos
      - flip rate in forward half (Q_sign > 0)   — should be ~0 if model OK
      - flip rate in reverse half (Q_sign < 0)   — high here ⇒ model learned
                                                   fixed axial polarity
    """
    by_var: Dict[str, List[Dict]] = {}
    for r in rows:
        by_var.setdefault(r["variant"], []).append(r)

    out = []
    for variant, group in sorted(by_var.items()):
        fwd = [r for r in group if r.get("Q_sign", 1.0) > 0]
        rev = [r for r in group if r.get("Q_sign", 1.0) < 0]
        rec = {
            "variant": variant,
            "n_cases": len(group),
            "n_forward": len(fwd),
            "n_reverse": len(rev),
            "cos_signed_median_overall": float(np.median(
                [r["cos_signed_median"] for r in group
                 if not math.isnan(r.get("cos_signed_median", float("nan")))]
            )) if group else float("nan"),
            "angle_median_overall_deg": float(np.median(
                [r["angle_median_deg"] for r in group
                 if not math.isnan(r.get("angle_median_deg", float("nan")))]
            )) if group else float("nan"),
        }
        # Forward half stats
        if fwd:
            rec["cos_signed_median_forward"] = float(np.median(
                [r["cos_signed_median"] for r in fwd]
            ))
            rec["case_flip_rate_forward"] = float(np.mean(
                [1.0 if r["cos_signed_median"] < 0.0 else 0.0 for r in fwd]
            ))
            rec["angle_median_forward_deg"] = float(np.median(
                [r["angle_median_deg"] for r in fwd]
            ))
        else:
            rec["cos_signed_median_forward"] = float("nan")
            rec["case_flip_rate_forward"] = float("nan")
            rec["angle_median_forward_deg"] = float("nan")
        # Reverse half stats
        if rev:
            rec["cos_signed_median_reverse"] = float(np.median(
                [r["cos_signed_median"] for r in rev]
            ))
            rec["case_flip_rate_reverse"] = float(np.mean(
                [1.0 if r["cos_signed_median"] < 0.0 else 0.0 for r in rev]
            ))
            rec["angle_median_reverse_deg"] = float(np.median(
                [r["angle_median_deg"] for r in rev]
            ))
        else:
            rec["cos_signed_median_reverse"] = float("nan")
            rec["case_flip_rate_reverse"] = float("nan")
            rec["angle_median_reverse_deg"] = float("nan")
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_phase_vs_metric(
    rows: List[Dict],
    binned: List[Dict],
    metric: str,
    ylabel: str,
    fig_path_base: Path,
    *,
    ref_line: Optional[float] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    # Scatter: one point per case, x = phase_norm, y = metric
    for variant in VARIANT_ORDER:
        xs, ys = [], []
        for r in rows:
            if r["variant"] != variant:
                continue
            ph = r.get("phase_norm", float("nan"))
            y = r.get(metric, float("nan"))
            if isinstance(ph, float) and math.isnan(ph):
                continue
            if isinstance(y, float) and math.isnan(y):
                continue
            xs.append(ph)
            ys.append(y)
        if not xs:
            continue
        ax.scatter(xs, ys, s=24, alpha=0.45,
                   color=VARIANT_COLORS.get(variant, "k"),
                   label=variant.replace("womersley_", ""), edgecolor="none")

    # Lines: bin medians
    for variant in VARIANT_ORDER:
        seg = [r for r in binned if r["variant"] == variant]
        if not seg:
            continue
        seg.sort(key=lambda r: r["phase_bin"])
        xs = [0.5 * (r["phase_lo"] + r["phase_hi"]) for r in seg]
        ys = [r[f"{metric}_mean"] for r in seg]
        ax.plot(xs, ys, "-", color=VARIANT_COLORS.get(variant, "k"),
                linewidth=2.0, alpha=0.9)

    if ref_line is not None:
        ax.axhline(ref_line, color="k", linewidth=0.7, linestyle="--",
                   alpha=0.5)

    # Phase axis annotations
    ax.set_xlim(0.0, 2.0 * math.pi)
    ax.set_xticks([0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi])
    ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
    ax.axvspan(math.pi / 2, 3 * math.pi / 2, color="0.92", zorder=0,
               label="reverse-half cycle")
    ax.set_xlabel("normalised phase φ = (ω · t_phase) mod 2π")
    ax.set_ylabel(ylabel)
    if ymin is not None or ymax is not None:
        ax.set_ylim(bottom=ymin, top=ymax)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.set_title("Phase-aware diagnostic on Womersley benchmark", fontsize=11)
    fig.tight_layout()
    fig_path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path_base.with_suffix(".png"), dpi=150)
    fig.savefig(fig_path_base.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[wm-phase] figure -> {fig_path_base}.{{png,pdf}}")


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def _fmt(mu: float, sd: float, p: int = 2) -> str:
    if isinstance(mu, float) and math.isnan(mu):
        return "—"
    if sd is None or (isinstance(sd, float) and (math.isnan(sd) or sd == 0.0)):
        return f"{mu:.{p}f}"
    return f"{mu:.{p}f} ± {sd:.{p}f}"


def write_summary_md(
    binned: List[Dict],
    flip: List[Dict],
    out_md: Path,
) -> None:
    lines: List[str] = [
        "# Womersley phase-aware diagnostic",
        "",
        "**Hypothesis.** Variants without per-node direction leakage learn a "
        "fixed axial polarity from the training distribution. When tested at "
        "phases where the true Womersley flow has reversed "
        "(cos(ω·t_phase) < 0), they keep predicting the original polarity → "
        "their *signed* cosine flips negative even though the *folded* angle "
        "stays close to 0°. Direction-leakage variants carry the "
        "instantaneous sign in the feature, so they should be invariant to "
        "phase.",
        "",
        "## Per-variant flip rates (split by expected Q-sign)",
        "",
        "`forward` = cases with cos(ω·t_phase) > 0 (drive in +z); "
        "`reverse` = cases with cos(ω·t_phase) < 0 (drive in −z). "
        "If the model has learned a fixed axial polarity, "
        "`case_flip_rate_reverse` will be ≫ `case_flip_rate_forward`.",
        "",
        "| variant | n | fwd flip rate | rev flip rate | "
        "cos_signed (fwd) | cos_signed (rev) | "
        "angle° (fwd) | angle° (rev) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in flip:
        lines.append(
            "| " + " | ".join([
                r["variant"].replace("womersley_", ""),
                str(r["n_cases"]),
                _fmt(r["case_flip_rate_forward"], None, 2),
                _fmt(r["case_flip_rate_reverse"], None, 2),
                _fmt(r["cos_signed_median_forward"], None, 3),
                _fmt(r["cos_signed_median_reverse"], None, 3),
                _fmt(r["angle_median_forward_deg"], None, 1),
                _fmt(r["angle_median_reverse_deg"], None, 1),
            ]) + " |"
        )
    lines.append("")

    lines += [
        "## Phase-binned medians (4 bins over [0, 2π))",
        "",
        "Bins: 0=[0, π/2), 1=[π/2, π), 2=[π, 3π/2), 3=[3π/2, 2π). "
        "Reverse-flow half of the cycle is approximately bins 1+2.",
        "",
        "| variant | bin | n | angle° | cos_signed | case_flip_rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in binned:
        lines.append(
            "| " + " | ".join([
                r["variant"].replace("womersley_", ""),
                str(r["phase_bin"]),
                str(r["n_cases"]),
                _fmt(r["angle_median_deg_mean"], r["angle_median_deg_std"], 2),
                _fmt(r["cos_signed_median_mean"], r["cos_signed_median_std"], 3),
                _fmt(r["case_flip_rate"], None, 2),
            ]) + " |"
        )
    lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))
    print(f"[wm-phase] summary -> {out_md}")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_case_csv",
                    required=True,
                    help="results/diagnostics/womersley/per_case.csv")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fig_dir", required=True)
    ap.add_argument("--n_bins", type=int, default=4)
    args = ap.parse_args()

    rows = _read_per_case(Path(args.per_case_csv))
    print(f"[wm-phase] loaded {len(rows)} per-case rows")

    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Pass-through per_case dump (paper-side convenience)
    pc_out = out_dir / "phase_per_case.csv"
    if rows:
        with open(pc_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()),
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"[wm-phase] per-case -> {pc_out}")

    binned = bin_by_phase(rows, n_bins=int(args.n_bins))
    write_binned_csv(binned, out_dir / "phase_binned.csv")
    flip = per_variant_flip_rate(rows)
    # Also write per-variant flip table to CSV
    if flip:
        with open(out_dir / "phase_flip_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(flip[0].keys()),
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(flip)
        print(f"[wm-phase] flip table -> {out_dir / 'phase_flip_summary.csv'}")

    plot_phase_vs_metric(
        rows, binned, "angle_median_deg",
        ylabel="median angle ∠(u_pred, u_true) [deg]",
        fig_path_base=fig_dir / "womersley_phase_angle",
        ref_line=0.0, ymin=0.0,
    )
    plot_phase_vs_metric(
        rows, binned, "cos_signed_median",
        ylabel="median signed cosine of (u_pred, u_true)",
        fig_path_base=fig_dir / "womersley_phase_cos",
        ref_line=0.0, ymin=-1.05, ymax=1.05,
    )

    write_summary_md(binned, flip, out_dir / "phase_summary.md")
    print("[wm-phase] done.")


if __name__ == "__main__":
    main()
