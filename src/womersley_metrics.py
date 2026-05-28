# -*- coding: utf-8 -*-
"""
Womersley-specific metric repair (Phase E5 — Step 1 in STRATEGY_CP.md).

Why this script exists
----------------------
The headline PP@10 and WSS R² are *broken* on the cylindrical Womersley
benchmark, but for different reasons than originally diagnosed:

  - PP@10 is computed as per-node ``||u_pred - u_true|| / ||u_true||``. On
    Womersley, magnitude is wildly off (peak_mag_rel ≈ 3-5 for noleak/
    leak_dir_only) because magnitude is not derivable from geometry alone.
    Per-node relative error blows up → PP@10 ≈ 0 everywhere, even for
    variants whose *direction* is excellent (e.g. leak_dir_only with
    angle ≈ 14°).

  - WSS R² collapses to ~-10⁶ because the cylinder has a near-uniform
    wall shear stress (varies only weakly with R(s); here R is exactly
    constant per case). Var(WSS_true) ≈ 0 → R² = 1 - MSE/Var → -∞.
    The R² statistic is mathematically ill-defined in this regime; it is
    not a model failure mode.

What this script does
---------------------
For every dumped prediction in ``results/predictions/womersley_*`` it
computes three new families of metrics that *do* generalize to the
synthetic regime:

  PP_dir@θ   : % HE nodes with ∠(u_pred, u_true) ≤ θ°   (direction only)
  PP_peak@δ  : % HE nodes with ||u_pred-u_true|| / max(||u_true||) ≤ δ
               (peak-normalized vector error)
  cos_signed : median per-node cosine sim (sign-preserving — exposes
               directional flips that absolute-angle masks)
  mag_ratio  : median(|u_pred|/|u_true|)  ; >1 = overshoot, <1 = undershoot
  phase_norm : (omega·t_phase) mod (2π)   ; position in current cycle
  Q_sign     : sign(cos(omega·t_phase))   ; expected bulk-flow direction

Outputs
-------
  results/diagnostics/womersley/per_case.csv
  results/diagnostics/womersley/aggregate.csv     (mean ± std across seeds)
  results/diagnostics/womersley/summary.md        (human-readable, n=3)

Usage
-----
    python src/womersley_metrics.py \\
        --predictions_root results/predictions \\
        --data_root        data \\
        --out_dir          results/diagnostics/womersley
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WOMERSLEY_VARIANTS = [
    "womersley_leak_dir_only",
    "womersley_withleak",
    "womersley_noleak",
    "womersley_leak_mag_only",
]

HE_PERCENTILE = 0.80                  # match training/eval pipeline

ANGLE_THRESHOLDS_DEG = [5.0, 10.0, 15.0, 30.0]
PEAK_REL_THRESHOLDS = [0.10, 0.20, 0.50]


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def angle_deg(u_pred: np.ndarray, u_true: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-node angle in degrees in [0, 180]. Folded — sign is dropped."""
    pn = np.linalg.norm(u_pred, axis=-1).clip(min=eps)
    yn = np.linalg.norm(u_true, axis=-1).clip(min=eps)
    cos = np.clip((u_pred * u_true).sum(-1) / (pn * yn), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def cos_signed(u_pred: np.ndarray, u_true: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-node cosine similarity in [-1, 1]. Negative ⇒ flipped direction."""
    pn = np.linalg.norm(u_pred, axis=-1).clip(min=eps)
    yn = np.linalg.norm(u_true, axis=-1).clip(min=eps)
    return np.clip((u_pred * u_true).sum(-1) / (pn * yn), -1.0, 1.0)


def he_mask_from_speed(u_true: np.ndarray, pct: float) -> np.ndarray:
    """Top-(1 - pct) speed nodes — same convention as MetricAccumulator."""
    speed = np.linalg.norm(u_true, axis=-1)
    if speed.size == 0:
        return np.zeros_like(speed, dtype=bool)
    q = float(np.quantile(speed, pct))
    return speed >= q


# ---------------------------------------------------------------------------
# Per-case computation
# ---------------------------------------------------------------------------

def compute_case(
    pred: np.ndarray,
    y: np.ndarray,
    wall_mask: np.ndarray,
    meta: Optional[dict],
) -> Dict[str, float]:
    """Compute all Womersley-fix metrics for one case."""
    he = he_mask_from_speed(y, HE_PERCENTILE)
    if int(he.sum()) == 0:
        # Should not happen on Womersley (uniform tube), but be defensive.
        return {k: float("nan") for k in _ALL_METRIC_KEYS}

    u_pred_he = pred[he]
    u_true_he = y[he]
    speed_true = np.linalg.norm(y, axis=-1)
    speed_pred = np.linalg.norm(pred, axis=-1)
    peak_true = float(speed_true.max())

    ang = angle_deg(u_pred_he, u_true_he)
    cs = cos_signed(u_pred_he, u_true_he)

    out: Dict[str, float] = {}

    # PP_dir@θ — direction-only success rate
    for theta in ANGLE_THRESHOLDS_DEG:
        out[f"pp_dir_{theta:.0f}deg"] = float((ang <= theta).mean())

    # PP_peak@δ — peak-normalised vector error
    if peak_true > 1e-9:
        rel_peak = np.linalg.norm(u_pred_he - u_true_he, axis=-1) / peak_true
        for delta in PEAK_REL_THRESHOLDS:
            out[f"pp_peak_{delta:.2f}"] = float((rel_peak <= delta).mean())
    else:
        for delta in PEAK_REL_THRESHOLDS:
            out[f"pp_peak_{delta:.2f}"] = float("nan")

    # Signed cosine diagnostic (sign-flips invisible in folded angle)
    out["cos_signed_median"] = float(np.median(cs))
    out["cos_signed_mean"] = float(cs.mean())
    out["frac_flipped"] = float((cs < 0.0).mean())   # share of reversed nodes

    # Folded angle (matches existing physics_diagnostics convention)
    out["angle_median_deg"] = float(np.median(ang))
    out["angle_mean_deg"] = float(ang.mean())
    out["angle_p90_deg"] = float(np.percentile(ang, 90))

    # Magnitude ratio (HE only)
    rmag = speed_pred[he] / speed_true[he].clip(min=1e-9)
    out["mag_ratio_median"] = float(np.median(rmag))
    out["mag_ratio_mean"] = float(rmag.mean())

    # Peak-velocity quantities
    i_gt = int(np.argmax(speed_true))
    i_pr = int(np.argmax(speed_pred))
    out["peak_mag_true"] = float(speed_true[i_gt])
    out["peak_mag_pred"] = float(speed_pred[i_pr])
    out["peak_mag_rel"] = float(
        (speed_pred[i_pr] - speed_true[i_gt]) / max(speed_true[i_gt], 1e-12)
    )

    # Phase / cycle position from meta (if available)
    if meta is not None and "omega" in meta and "t_phase" in meta:
        phase = float(meta["omega"]) * float(meta["t_phase"])
        phase_norm = phase % (2.0 * math.pi)
        out["t_phase"] = float(meta["t_phase"])
        out["omega"] = float(meta["omega"])
        out["phase_norm"] = float(phase_norm)
        out["Q_sign"] = float(math.copysign(1.0, math.cos(phase)))
        out["alpha_womersley"] = float(meta.get("alpha_womersley", float("nan")))
        out["R_mm"] = float(meta.get("R", float("nan"))) * 1000.0
        out["L_mm"] = float(meta.get("L", float("nan"))) * 1000.0
        out["p_amp"] = float(meta.get("p_amp", float("nan")))
    else:
        for k in ("t_phase", "omega", "phase_norm", "Q_sign",
                  "alpha_womersley", "R_mm", "L_mm", "p_amp"):
            out[k] = float("nan")

    out["wall_frac"] = float(wall_mask.mean())
    return out


# Canonical column order, used both for CSV output and the empty-case fallback.
_ALL_METRIC_KEYS: List[str] = (
    [f"pp_dir_{t:.0f}deg" for t in ANGLE_THRESHOLDS_DEG]
    + [f"pp_peak_{d:.2f}" for d in PEAK_REL_THRESHOLDS]
    + [
        "cos_signed_median", "cos_signed_mean", "frac_flipped",
        "angle_median_deg", "angle_mean_deg", "angle_p90_deg",
        "mag_ratio_median", "mag_ratio_mean",
        "peak_mag_true", "peak_mag_pred", "peak_mag_rel",
        "t_phase", "omega", "phase_norm", "Q_sign",
        "alpha_womersley", "R_mm", "L_mm", "p_amp",
        "wall_frac",
    ]
)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def split_of(name: str) -> str:
    if name.startswith("test_"):
        return "test"
    if name.startswith("val_"):
        return "val"
    return "unknown"


def case_id_of(pred_filename: str) -> str:
    stem = Path(pred_filename).stem
    for prefix in ("test_", "val_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def load_data_meta(data_root: Path, variant: str, case_id: str) -> Optional[dict]:
    """The variant-specific data dir holds the raw npz with the cycle meta."""
    npz = data_root / f"npz_{variant}" / f"{case_id}.npz"
    if not npz.exists():
        # Fall back to base womersley dir; all variants share cycle params.
        npz = data_root / "npz_womersley" / f"{case_id}.npz"
    if not npz.exists():
        return None
    try:
        d = np.load(npz, allow_pickle=True)
        if "meta" not in d.files:
            return None
        return json.loads(str(d["meta"]))
    except Exception as exc:
        print(f"  [warn] could not load meta for {variant}/{case_id}: {exc}")
        return None


def collect_per_case(
    predictions_root: Path,
    data_root: Path,
    variants: List[str],
) -> List[Dict]:
    rows: List[Dict] = []
    for variant in variants:
        v_dir = predictions_root / variant
        if not v_dir.is_dir():
            print(f"  [skip] {variant} (no predictions dir)")
            continue
        for seed_dir in sorted(v_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            try:
                seed = int(seed_dir.name.split("_")[-1])
            except ValueError:
                continue
            for f in sorted(seed_dir.glob("*.npz")):
                split = split_of(f.name)
                if split == "unknown":
                    continue
                d = np.load(f, allow_pickle=True)
                pred = d["pred"].astype(np.float32)
                y = d["y"].astype(np.float32)
                wm = np.asarray(d["wall_mask"]).astype(bool)
                cid = case_id_of(f.name)
                meta = load_data_meta(data_root, variant, cid)
                row = {
                    "variant": variant,
                    "seed": seed,
                    "split": split,
                    "case": cid,
                }
                row.update(compute_case(pred, y, wm, meta))
                rows.append(row)
                print(
                    f"  {variant:30s} seed={seed:<5d} {split:4s} {cid[:30]:30s} "
                    f"ang={row['angle_median_deg']:6.2f}° "
                    f"PPdir10={row['pp_dir_10deg']:.3f} "
                    f"PPpeak20={row['pp_peak_0.20']:.3f} "
                    f"cos_s={row['cos_signed_median']:+.3f}"
                )
    return rows


def write_per_case(rows: List[Dict], out_csv: Path) -> None:
    if not rows:
        print(f"  [warn] no rows to write to {out_csv}")
        return
    keys = ["variant", "seed", "split", "case"] + _ALL_METRIC_KEYS
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[wm-metrics] per-case -> {out_csv} ({len(rows)} rows)")


def _stats(vals: List[float]) -> Tuple[float, float, int]:
    clean = [v for v in vals if v is not None
             and isinstance(v, (int, float))
             and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))]
    n = len(clean)
    if n == 0:
        return float("nan"), float("nan"), 0
    if n == 1:
        return float(clean[0]), 0.0, 1
    return float(mean(clean)), float(stdev(clean)), n


def aggregate(rows: List[Dict], out_csv: Path) -> List[Dict]:
    """Aggregate per-case rows to (variant, split): mean ± std over seeds.

    Each (variant, split, case) gets one value per seed. We average across
    cases *and* seeds (one pooled distribution per condition).
    """
    by_cond: Dict[Tuple[str, str], List[Dict]] = {}
    for r in rows:
        by_cond.setdefault((r["variant"], r["split"]), []).append(r)

    metric_keys = [k for k in _ALL_METRIC_KEYS
                   if k not in ("t_phase", "omega", "phase_norm",
                                "Q_sign", "alpha_womersley", "R_mm",
                                "L_mm", "p_amp", "wall_frac")]

    out_rows: List[Dict] = []
    for (variant, split), group in sorted(by_cond.items()):
        n_pred = len(group)
        seeds = sorted({int(r["seed"]) for r in group})
        cases = sorted({r["case"] for r in group})
        agg = {
            "variant": variant,
            "split": split,
            "n_predictions": n_pred,
            "n_seeds": len(seeds),
            "n_cases": len(cases),
        }
        for m in metric_keys:
            vals = [r[m] for r in group if m in r]
            mu, sd, _ = _stats(vals)
            agg[f"{m}_mean"] = mu
            agg[f"{m}_std"] = sd
        out_rows.append(agg)

    if not out_rows:
        return out_rows
    keys = ["variant", "split", "n_predictions", "n_seeds", "n_cases"]
    for m in metric_keys:
        keys += [f"{m}_mean", f"{m}_std"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)
    print(f"[wm-metrics] aggregate -> {out_csv} ({len(out_rows)} rows)")
    return out_rows


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def _fmt(mu: float, sd: float, p: int = 3) -> str:
    if isinstance(mu, float) and math.isnan(mu):
        return "—"
    if sd is None or (isinstance(sd, float) and (math.isnan(sd) or sd == 0.0)):
        return f"{mu:.{p}f}"
    return f"{mu:.{p}f} ± {sd:.{p}f}"


def write_summary_md(agg_rows: List[Dict], out_md: Path) -> None:
    if not agg_rows:
        out_md.write_text("# Womersley metrics — (no data)\n")
        return

    lines: List[str] = [
        "# Womersley metric repair — Phase E5",
        "",
        "WSS R² is **excluded** here by design — on a uniform-radius cylinder "
        "the variance of true WSS is ~0, so R² = 1 - MSE/Var collapses to "
        "≈ -10⁶ and is mathematically ill-defined. We report it in Limitations, "
        "not as a finding.",
        "",
        "PP@10 with per-node relative error is also excluded — it is "
        "dominated by magnitude error (peak_mag_rel ≈ 3-5 for noleak-family "
        "variants on Womersley), which buries direction quality.",
        "",
        "## Direction-only success rate (PP_dir@θ)",
        "",
        "Fraction of HE-mask (top-20%-speed) nodes with ∠(u_pred, u_true) ≤ θ.",
        "",
    ]
    cols = ["variant", "split", "n_cases", "n_seeds",
            "pp_dir_5deg", "pp_dir_10deg", "pp_dir_15deg", "pp_dir_30deg"]
    lines.append("| " + " | ".join(c for c in cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in agg_rows:
        row = [r["variant"], r["split"], str(r["n_cases"]), str(r["n_seeds"])]
        for m in ("pp_dir_5deg", "pp_dir_10deg",
                  "pp_dir_15deg", "pp_dir_30deg"):
            row.append(_fmt(r[f"{m}_mean"], r[f"{m}_std"], 3))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines += [
        "## Peak-normalised vector error (PP_peak@δ)",
        "",
        "Fraction of HE nodes with ||u_pred − u_true|| / max(||u_true||) ≤ δ. "
        "Uses *case-peak* as the normaliser, so it does not blow up when the "
        "model magnitude is off by a fixed multiplicative factor.",
        "",
    ]
    cols = ["variant", "split", "pp_peak_0.10", "pp_peak_0.20", "pp_peak_0.50"]
    lines.append("| " + " | ".join(c for c in cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in agg_rows:
        row = [r["variant"], r["split"]]
        for m in ("pp_peak_0.10", "pp_peak_0.20", "pp_peak_0.50"):
            row.append(_fmt(r[f"{m}_mean"], r[f"{m}_std"], 3))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines += [
        "## Signed-cosine + magnitude diagnostics",
        "",
        "`cos_signed_median` < 0 ⇒ majority of HE nodes have a flipped "
        "direction (typical when the model has learned a fixed axial "
        "polarity in training but the test phase is in the reverse half "
        "of the cycle). `frac_flipped` = share of HE nodes with cos < 0.",
        "",
    ]
    cols = ["variant", "split",
            "cos_signed_median", "frac_flipped",
            "angle_median_deg", "mag_ratio_median", "peak_mag_rel"]
    lines.append("| " + " | ".join(c for c in cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in agg_rows:
        row = [r["variant"], r["split"]]
        for m, p in (("cos_signed_median", 3), ("frac_flipped", 3),
                     ("angle_median_deg", 2), ("mag_ratio_median", 3),
                     ("peak_mag_rel", 3)):
            row.append(_fmt(r[f"{m}_mean"], r[f"{m}_std"], p))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))
    print(f"[wm-metrics] summary -> {out_md}")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions_root", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--variants", nargs="*", default=WOMERSLEY_VARIANTS)
    args = ap.parse_args()

    pred_root = Path(args.predictions_root)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_per_case(pred_root, data_root, list(args.variants))
    write_per_case(rows, out_dir / "per_case.csv")
    agg = aggregate(rows, out_dir / "aggregate.csv")
    write_summary_md(agg, out_dir / "summary.md")
    print("[wm-metrics] done.")


if __name__ == "__main__":
    main()
