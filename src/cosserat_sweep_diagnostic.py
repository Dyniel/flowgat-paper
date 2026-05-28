# -*- coding: utf-8 -*-
"""
Post-training diagnostic for the curved-tube Cosserat/Dean sweep.

The direction-identifiability metrics are intentionally delegated to
src/womersley_metrics.py so the Cosserat sweep uses the same definitions as
the existing unified Womersley/VMR analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from womersley_metrics import compute_case


VARIANTS = [
    "cosserat_sweep_withleak",
    "cosserat_sweep_leak_dir_only",
    "cosserat_sweep_leak_mag_only",
    "cosserat_sweep_noleak",
]

METRICS = [
    "pp_dir_10deg",
    "cos_signed_median",
    "frac_flipped",
    "angle_median_deg",
]


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


def _base_variant_dir(variant: str) -> str:
    if variant == "cosserat_sweep_noleak":
        return "npz_cosserat_sweep"
    return f"npz_{variant}"


def load_meta(data_root: Path, variant: str, case_id: str) -> Optional[dict]:
    paths = [
        data_root / _base_variant_dir(variant) / f"{case_id}.npz",
        data_root / "npz_cosserat_sweep" / f"{case_id}.npz",
    ]
    for npz in paths:
        if not npz.exists():
            continue
        try:
            d = np.load(npz, allow_pickle=True)
            if "meta" not in d.files:
                continue
            return json.loads(str(d["meta"]))
        except Exception as exc:
            print(f"  [warn] could not load meta from {npz}: {exc}")
    return None


def collect_per_case(
    predictions_root: Path,
    data_root: Path,
    variants: Sequence[str],
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
                cid = case_id_of(f.name)
                d = np.load(f, allow_pickle=True)
                pred = d["pred"].astype(np.float32)
                y = d["y"].astype(np.float32)
                wall_mask = np.asarray(d["wall_mask"]).astype(bool)
                meta = load_meta(data_root, variant, cid)
                metric_row = compute_case(pred, y, wall_mask, meta)
                row = {
                    "variant": variant,
                    "seed": seed,
                    "split": split,
                    "case": cid,
                }
                row.update(metric_row)
                if meta is not None:
                    row.update({
                        "R_mm": float(meta.get("R", float("nan"))) * 1000.0,
                        "L_mm": float(meta.get("L", float("nan"))) * 1000.0,
                        "alpha_womersley": float(meta.get("alpha_womersley", float("nan"))),
                        "kappa": float(meta.get("kappa", float("nan"))),
                        "eps": float(meta.get("eps", float("nan"))),
                        "De": float(meta.get("De", float("nan"))),
                    })
                else:
                    row.update({
                        "R_mm": float("nan"),
                        "L_mm": float("nan"),
                        "alpha_womersley": float("nan"),
                        "kappa": float("nan"),
                        "eps": float("nan"),
                        "De": float("nan"),
                    })
                rows.append(row)
                print(
                    f"  {variant:32s} seed={seed:<5d} {split:4s} {cid[:28]:28s} "
                    f"PPdir10={row['pp_dir_10deg']:.3f} "
                    f"cos_s={row['cos_signed_median']:+.3f} "
                    f"flip={row['frac_flipped']:.3f} "
                    f"De={row['De']:.3f} eps={row['eps']:.3f}"
                )
    return rows


def _clean(vals: Iterable[float]) -> List[float]:
    out: List[float] = []
    for v in vals:
        if v is None:
            continue
        if isinstance(v, (int, float)) and not math.isnan(float(v)) and not math.isinf(float(v)):
            out.append(float(v))
    return out


def _stats(vals: Iterable[float]) -> Tuple[float, float, int]:
    clean = _clean(vals)
    if not clean:
        return float("nan"), float("nan"), 0
    if len(clean) == 1:
        return clean[0], 0.0, 1
    return float(mean(clean)), float(stdev(clean)), len(clean)


def write_csv(rows: List[Dict], out_csv: Path, keys: Sequence[str]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(keys), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[cosserat-diag] wrote {out_csv} ({len(rows)} rows)")


def aggregate(rows: List[Dict], variants: Sequence[str]) -> List[Dict]:
    by_cond: Dict[Tuple[str, str], List[Dict]] = {}
    for r in rows:
        by_cond.setdefault((r["variant"], r["split"]), []).append(r)

    out: List[Dict] = []
    for variant in variants:
        for split in ("val", "test"):
            group = by_cond.get((variant, split), [])
            if not group:
                continue
            agg = {
                "variant": variant,
                "split": split,
                "n_predictions": len(group),
                "n_seeds": len({int(r["seed"]) for r in group}),
                "n_cases": len({r["case"] for r in group}),
            }
            for m in METRICS:
                mu, sd, _ = _stats(r.get(m, float("nan")) for r in group)
                agg[f"{m}_mean"] = mu
                agg[f"{m}_std"] = sd
            out.append(agg)
    return out


def _bin_edges(rows: List[Dict], key: str) -> np.ndarray:
    vals_by_case: Dict[str, float] = {}
    for r in rows:
        v = float(r.get(key, float("nan")))
        if math.isnan(v):
            continue
        vals_by_case.setdefault(str(r["case"]), v)
    vals = np.asarray(list(vals_by_case.values()), dtype=np.float64)
    if vals.size == 0:
        return np.asarray([float("nan")] * 4, dtype=np.float64)
    return np.quantile(vals, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])


def _bin_label(value: float, edges: np.ndarray) -> str:
    if not np.isfinite(value) or not np.all(np.isfinite(edges)):
        return "unknown"
    if value <= edges[1]:
        return "low"
    if value <= edges[2]:
        return "mid"
    return "high"


def stratify(rows: List[Dict], key: str, variants: Sequence[str]) -> List[Dict]:
    edges = _bin_edges(rows, key)
    if not np.all(np.isfinite(edges)):
        # All NaN — stratification not applicable (e.g. sUbend has no De/eps meta).
        return []
    labelled: Dict[Tuple[str, str, str], List[Dict]] = {}
    for r in rows:
        label = _bin_label(float(r.get(key, float("nan"))), edges)
        if label == "unknown":
            continue
        labelled.setdefault((r["variant"], r["split"], label), []).append(r)

    bounds = {
        "low": (edges[0], edges[1]),
        "mid": (edges[1], edges[2]),
        "high": (edges[2], edges[3]),
    }
    out: List[Dict] = []
    for variant in variants:
        for split in ("val", "test"):
            for label in ("low", "mid", "high"):
                group = labelled.get((variant, split, label), [])
                if not group:
                    continue
                lo, hi = bounds[label]
                row = {
                    "variant": variant,
                    "split": split,
                    "bin": label,
                    f"{key}_bin_low": float(lo),
                    f"{key}_bin_high": float(hi),
                    "n_predictions": len(group),
                    "n_seeds": len({int(r["seed"]) for r in group}),
                    "n_cases": len({r["case"] for r in group}),
                }
                for m in METRICS:
                    mu, sd, _ = _stats(r.get(m, float("nan")) for r in group)
                    row[f"{m}_mean"] = mu
                    row[f"{m}_std"] = sd
                out.append(row)
    return out


def _fmt(mu: float, sd: float, p: int = 3) -> str:
    if isinstance(mu, float) and math.isnan(mu):
        return "-"
    if sd is None or (isinstance(sd, float) and (math.isnan(sd) or sd == 0.0)):
        return f"{mu:.{p}f}"
    return f"{mu:.{p}f} +/- {sd:.{p}f}"


def _metric_table(rows: List[Dict], title: str) -> List[str]:
    lines = [f"## {title}", ""]
    cols = [
        "variant", "split", "n_cases", "n_seeds",
        "pp_dir_10deg", "cos_signed_median", "frac_flipped", "angle_median_deg",
    ]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        line = [
            str(r["variant"]),
            str(r["split"]),
            str(r["n_cases"]),
            str(r["n_seeds"]),
            _fmt(r["pp_dir_10deg_mean"], r["pp_dir_10deg_std"], 3),
            _fmt(r["cos_signed_median_mean"], r["cos_signed_median_std"], 3),
            _fmt(r["frac_flipped_mean"], r["frac_flipped_std"], 3),
            _fmt(r["angle_median_deg_mean"], r["angle_median_deg_std"], 2),
        ]
        lines.append("| " + " | ".join(line) + " |")
    lines.append("")
    return lines


def _de_flip_lines(strat_de: List[Dict], variants: Sequence[str]) -> List[str]:
    # No-direction-channel variants (the ones where Lemma 1 prediction applies):
    # names ending in `_leak_mag_only` or `_noleak`.
    no_dir_variants = [
        v for v in variants
        if v.endswith("_leak_mag_only") or v.endswith("_noleak")
    ]
    lines = [
        "## Dean-Bin Flip Check",
        "",
        "For no-direction-channel variants, Lemma 1 predicts that the sign "
        "flip rate should not systematically fall just because Dean number grows.",
        "",
        "| variant | split | low | mid | high | monotonic_decrease |",
        "|---|---|---|---|---|---|",
    ]
    for variant in no_dir_variants:
        for split in ("val", "test"):
            vals = {}
            for r in strat_de:
                if r["variant"] == variant and r["split"] == split:
                    vals[r["bin"]] = float(r["frac_flipped_mean"])
            if not vals:
                continue
            low = vals.get("low", float("nan"))
            mid = vals.get("mid", float("nan"))
            high = vals.get("high", float("nan"))
            mono = bool(np.isfinite([low, mid, high]).all() and low > mid > high)
            lines.append(
                f"| {variant} | {split} | {low:.3f} | {mid:.3f} | "
                f"{high:.3f} | {str(mono).lower()} |"
            )
    lines.append("")
    return lines


def write_summary_md(
    agg_rows: List[Dict],
    strat_de: List[Dict],
    out_md: Path,
    variants: Sequence[str],
) -> None:
    lines: List[str] = [
        "# Direction-Identifiability Diagnostic",
        "",
        "Direction-identifiability metrics reuse `src/womersley_metrics.py`.",
        "",
    ]
    if not agg_rows:
        lines.append("No prediction files were found.")
    else:
        lines.extend(_metric_table(agg_rows, "Aggregate"))
        if strat_de:
            lines.extend(_de_flip_lines(strat_de, variants))
        else:
            lines.append("## Dean-Bin Flip Check\n")
            lines.append(
                "_Skipped: dataset has no Dean number metadata "
                "(applicable to parametric sweeps only)._\n"
            )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))
    print(f"[cosserat-diag] wrote {out_md}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--variants", nargs="*", default=VARIANTS)
    args = ap.parse_args()

    pred_root = Path(args.predictions_root)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = list(args.variants)
    rows = collect_per_case(pred_root, data_root, variants)
    per_case_keys = [
        "variant", "seed", "split", "case",
        "R_mm", "L_mm", "alpha_womersley", "kappa", "eps", "De",
        *METRICS,
    ]
    write_csv(rows, out_dir / "per_case.csv", per_case_keys)

    agg_rows = aggregate(rows, variants)
    agg_keys = ["variant", "split", "n_predictions", "n_seeds", "n_cases"]
    for m in METRICS:
        agg_keys.extend([f"{m}_mean", f"{m}_std"])
    write_csv(agg_rows, out_dir / "aggregate.csv", agg_keys)

    strat_de = stratify(rows, "De", variants)
    strat_eps = stratify(rows, "eps", variants)
    strat_de_keys = ["variant", "split", "bin", "De_bin_low", "De_bin_high",
                     "n_predictions", "n_seeds", "n_cases"]
    strat_eps_keys = ["variant", "split", "bin", "eps_bin_low", "eps_bin_high",
                      "n_predictions", "n_seeds", "n_cases"]
    for m in METRICS:
        strat_de_keys.extend([f"{m}_mean", f"{m}_std"])
        strat_eps_keys.extend([f"{m}_mean", f"{m}_std"])
    write_csv(strat_de, out_dir / "stratified_by_de.csv", strat_de_keys)
    write_csv(strat_eps, out_dir / "stratified_by_eps.csv", strat_eps_keys)
    write_summary_md(agg_rows, strat_de, out_dir / "summary.md", variants)


if __name__ == "__main__":
    main()
