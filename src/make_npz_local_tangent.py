# -*- coding: utf-8 -*-
"""
Build NPZ variants for the *direction-prior ladder* in Phase E.

Variants produced:

  noleak_centerline : exactly like noleak, but x[:,3:6] is replaced by the
                      per-node iteratively-refined medial centerline tangent
                      (instead of global PCA tangent tiled per node).

We keep `noleak_pca` (the existing data/npz_noleak) as the global-PCA rung
of the ladder, and `leak_dir_only` (true unit(y)) as the ceiling.

Output:
  data/npz_noleak_centerline/<case>.npz + split.json (copied)
  data/npz_noleak_centerline/_centerline_diag.json   (angle vs true u, per case)

Usage:
  python src/make_npz_local_tangent.py \
      --src_dir   data/npz_noleak \
      --withleak_dir data/npz_withleak \
      --out_dir   data/npz_noleak_centerline \
      --n_bins 80 --n_iter 3 --smooth_sigma 3.0 --medial_quantile 0.85
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from centerline_tangent import compute_all_priors  # noqa: E402


def angle_deg_unsigned(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    cos = np.clip((a * b).sum(-1), -1.0, 1.0)
    return np.degrees(np.arccos(np.abs(cos)))


def process_case(
    case_id: str,
    src_dir: Path,
    withleak_dir: Path,
    out_dir: Path,
    *,
    n_bins: int,
    n_iter: int,
    smooth_sigma: float,
    medial_quantile: float,
) -> dict:
    src_path = src_dir / f"{case_id}.npz"
    wl_path = withleak_dir / f"{case_id}.npz"
    if not src_path.exists():
        raise FileNotFoundError(src_path)
    if not wl_path.exists():
        raise FileNotFoundError(wl_path)

    d = np.load(src_path, allow_pickle=False)
    arrays = {k: d[k] for k in d.files}
    d.close()

    pos = arrays["pos"].astype(np.float32)
    wall = arrays["wall_mask"].astype(bool)

    # compute priors
    t0 = time.time()
    priors = compute_all_priors(
        pos, wall,
        n_bins=int(n_bins),
        n_iter=int(n_iter),
        smooth_sigma=float(smooth_sigma),
        medial_quantile=float(medial_quantile),
        include_local_pca=False,
    )
    centerline_t = priors["centerline_tangent"].astype(np.float32)
    elapsed = time.time() - t0

    # replace x[:,3:6]
    x_new = arrays["x"].astype(np.float32).copy()
    if x_new.shape[1] < 6:
        raise ValueError(f"{case_id}: x has only {x_new.shape[1]} columns; expected ≥ 6")
    x_new[:, 3:6] = centerline_t
    arrays["x"] = x_new
    arrays["feature_mode"] = np.array("noleak_centerline_v1")

    # write
    out_path = out_dir / f"{case_id}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)

    # diagnostic: angle to true direction (from withleak NPZ, which has y)
    diag = {"case": case_id, "n_nodes": int(pos.shape[0]), "centerline_time_s": float(elapsed)}
    wl = np.load(wl_path, allow_pickle=False)
    y = wl["y"].astype(np.float32)
    wl.close()
    umag = np.linalg.norm(y, axis=1)
    moving = (~wall) & (umag > np.median(umag[~wall]))
    if int(moving.sum()) > 0:
        yhat = (y[moving] / np.linalg.norm(y[moving], axis=1, keepdims=True).clip(min=1e-8)).astype(np.float32)
        ang_global = angle_deg_unsigned(priors["global_pca_tangent"][moving], yhat)
        ang_cline = angle_deg_unsigned(centerline_t[moving], yhat)
        diag.update(
            moving_nodes=int(moving.sum()),
            angle_global_median=float(np.median(ang_global)),
            angle_global_mean=float(ang_global.mean()),
            angle_global_p90=float(np.percentile(ang_global, 90)),
            angle_centerline_median=float(np.median(ang_cline)),
            angle_centerline_mean=float(ang_cline.mean()),
            angle_centerline_p90=float(np.percentile(ang_cline, 90)),
        )
    return diag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True, help="Source NPZ dir (e.g., data/npz_noleak)")
    ap.add_argument("--withleak_dir", required=True, help="withleak dir (for y diagnostic)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_bins", type=int, default=80)
    ap.add_argument("--n_iter", type=int, default=3)
    ap.add_argument("--smooth_sigma", type=float, default=3.0)
    ap.add_argument("--medial_quantile", type=float, default=0.85)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src_dir = Path(args.src_dir)
    wl_dir = Path(args.withleak_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # copy split.json
    split_src = src_dir / "split.json"
    if not split_src.exists():
        raise SystemExit(f"split.json missing in {src_dir}")
    shutil.copy2(split_src, out_dir / "split.json")
    with open(split_src) as f:
        split = json.load(f)
    cases = sorted(set(sum((split.get(k, []) for k in ("train", "val", "test")), [])))
    print(f"[centerline] processing {len(cases)} cases")

    diags = []
    for i, case in enumerate(cases):
        out_path = out_dir / f"{case}.npz"
        if out_path.exists() and not args.force:
            print(f"  [{i+1}/{len(cases)}] {case}: SKIP (exists)")
            continue
        try:
            diag = process_case(
                case, src_dir, wl_dir, out_dir,
                n_bins=args.n_bins, n_iter=args.n_iter,
                smooth_sigma=args.smooth_sigma,
                medial_quantile=args.medial_quantile,
            )
            diags.append(diag)
            ang_g = diag.get("angle_global_median", float("nan"))
            ang_c = diag.get("angle_centerline_median", float("nan"))
            print(
                f"  [{i+1}/{len(cases)}] {case}: N={diag['n_nodes']:>7d} "
                f"t={diag['centerline_time_s']:5.1f}s  "
                f"angle_global={ang_g:5.1f}°  angle_centerline={ang_c:5.1f}°"
            )
        except Exception as e:
            print(f"  [ERR] {case}: {e}")
            raise

    diag_summary = {
        "config": dict(
            n_bins=int(args.n_bins),
            n_iter=int(args.n_iter),
            smooth_sigma=float(args.smooth_sigma),
            medial_quantile=float(args.medial_quantile),
        ),
        "n_cases": int(len(diags)),
        "median_angle_global": float(np.median([d["angle_global_median"] for d in diags if "angle_global_median" in d])) if diags else None,
        "median_angle_centerline": float(np.median([d["angle_centerline_median"] for d in diags if "angle_centerline_median" in d])) if diags else None,
        "cases": diags,
    }
    with open(out_dir / "_centerline_diag.json", "w") as f:
        json.dump(diag_summary, f, indent=2)
    print(f"[centerline] DONE: {len(diags)} cases written to {out_dir}")
    if diags:
        print(f"  median_angle_global = {diag_summary['median_angle_global']:.2f}°")
        print(f"  median_angle_centerline = {diag_summary['median_angle_centerline']:.2f}°")


if __name__ == "__main__":
    main()
