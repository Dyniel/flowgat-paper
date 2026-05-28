# -*- coding: utf-8 -*-
"""
Build a no-leak VMR NPZ dataset from an existing preprocessed NPZ directory.

The current v2 NPZ files contain node channels x[:, 3:6] derived from the
ground-truth CFD velocity direction, plus x[:, 8:10] derived from true velocity
statistics.  That is useful for diagnostics, but it is not submit-safe for a
predictive surrogate claim.

This script preserves mesh, labels, pressure gradients, wall masks, u_char, and
split metadata, but rewrites x to contain geometry-only features:

  [0] axial coordinate from PCA, normalized to [0, 1]
  [1] distance to nearest wall, normalized to [0, 1]
  [2] d_wall / local radius estimate, clipped to [0, 1]
  [3:6] global PCA vessel tangent, repeated per node
  [6] inlet proxy flag (lowest axial quantile)
  [7] outlet proxy flag (highest axial quantile)
  [8] local radius estimate, normalized to [0, 1]
  [9] distance from global PCA axis, normalized to [0, 1]
  [10] wall flag

Training may still use y for loss weighting or importance-centered sampling;
that is training-only supervision, not an inference-time input feature.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


FEATURE_NAMES = [
    "axial_pca_norm",
    "wall_distance_norm",
    "radial_position_local",
    "pca_tangent_x",
    "pca_tangent_y",
    "pca_tangent_z",
    "inlet_proxy",
    "outlet_proxy",
    "local_radius_norm",
    "distance_to_pca_axis_norm",
    "wall_flag",
]

MR_BASE_CASES = {
    "0001_H_AO_SVD_3D_RIGID",
    "0002_H_AO_SVD_3D_RIGID",
    "0004_H_AO_SVD_3D_RIGID",
    "0005_H_AO_SVD_3D_RIGID",
    "0006_H_AO_SVD_3D_RIGID",
    "0007_H_AO_H_3D_RIGID",
    "0010_H_AO_H_3D_RIGID",
    "0011_H_AO_H_3D_RIGID",
    "0012_H_AO_H_3D_RIGID",
    "0013_H_AO_COA_3D_RIGID",
    "0014_H_AO_COA_3D_RIGID",
    "0015_H_AO_COA_3D_RIGID",
    "0016_H_AO_COA_3D_RIGID",
    "0017_H_AO_COA_3D_RIGID",
    "0018_H_AO_COA_3D_RIGID",
    "0019_H_AO_COA_3D_RIGID",
    "0020_H_AO_COA_3D_RIGID",
    "0022_H_AO_MFS_3D_RIGID",
    "0023_H_AO_MFS_3D_RIGID",
    "0024_H_AO_H_3D_RIGID",
    "0225_H_AO_COA_3D_FSI_REST",
    "0226_H_AO_COA_3D_FSI_REST",
    "0232_H_AO_COA_3D_FSI_REST",
    "0234_H_AO_COA_3D_FSI_REST",
    "0241_H_AO_COA_3D_FSI_REST",
}


def infer_modality(case_id: str) -> str:
    return "MR" if case_id in MR_BASE_CASES else "CT"


def _normalize(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    mn = float(np.nanmin(a))
    mx = float(np.nanmax(a))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < 1.0e-12:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - mn) / (mx - mn)).astype(np.float32)


def _pca_axis(pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = pos - pos.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        axis = vt[0].astype(np.float32)
    except Exception:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    n = float(np.linalg.norm(axis))
    if n < 1.0e-12:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        axis = axis / n

    # Fix the arbitrary PCA sign for deterministic reruns.
    major = int(np.argmax(np.abs(axis)))
    if axis[major] < 0:
        axis = -axis

    axial_raw = centered @ axis
    axial = _normalize(axial_raw)
    return axis.astype(np.float32), axial_raw.astype(np.float32), axial.astype(np.float32)


def _wall_distance(pos: np.ndarray, wall_mask: np.ndarray) -> np.ndarray:
    wall_points = pos[wall_mask]
    if len(wall_points) < 8:
        return np.zeros(pos.shape[0], dtype=np.float32)
    try:
        from scipy.spatial import cKDTree
    except Exception as exc:
        raise RuntimeError("scipy is required for no-leak feature generation") from exc
    tree = cKDTree(wall_points)
    d_wall, _ = tree.query(pos, k=1)
    return d_wall.astype(np.float32)


def _local_radius(d_wall: np.ndarray, axial: np.ndarray, n_bins: int) -> np.ndarray:
    bins = np.floor(axial * (n_bins - 1)).astype(np.int64).clip(0, n_bins - 1)
    radius = np.zeros_like(d_wall, dtype=np.float32)
    for b in range(n_bins):
        mask = bins == b
        if np.any(mask):
            radius[mask] = float(np.max(d_wall[mask]))
    return radius


def build_noleak_x(
    pos: np.ndarray,
    wall_mask: np.ndarray,
    *,
    n_bins: int = 100,
    boundary_quantile: float = 0.02,
) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float32)
    wall_mask = np.asarray(wall_mask).astype(bool)
    n_nodes = pos.shape[0]

    axis, axial_raw, axial = _pca_axis(pos)
    d_wall = _wall_distance(pos, wall_mask)
    dist_wall = _normalize(d_wall)

    radius = _local_radius(d_wall, axial, int(n_bins))
    r_norm = (d_wall / np.maximum(radius, 1.0e-6)).clip(0.0, 1.0).astype(np.float32)
    radius_norm = _normalize(radius)

    centered = pos - pos.mean(axis=0, keepdims=True)
    proj = (centered @ axis)[:, None] * axis[None, :]
    axis_dist = np.linalg.norm(centered - proj, axis=1).astype(np.float32)
    axis_dist_norm = _normalize(axis_dist)

    q = float(boundary_quantile)
    q = min(max(q, 0.001), 0.20)
    lo = float(np.quantile(axial_raw, q))
    hi = float(np.quantile(axial_raw, 1.0 - q))
    inlet = (axial_raw <= lo).astype(np.float32)
    outlet = (axial_raw >= hi).astype(np.float32)

    tangent = np.repeat(axis[None, :], n_nodes, axis=0).astype(np.float32)

    x = np.stack(
        [
            axial,
            dist_wall,
            r_norm,
            tangent[:, 0],
            tangent[:, 1],
            tangent[:, 2],
            inlet,
            outlet,
            radius_norm,
            axis_dist_norm,
            wall_mask.astype(np.float32),
        ],
        axis=-1,
    ).astype(np.float32)
    return x


def _copy_jsons(src_dir: Path, out_dir: Path) -> None:
    for name in ("split.json",):
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)


def iter_npz(src_dir: Path, include_coronary: bool) -> Iterable[Path]:
    for path in sorted(src_dir.glob("*.npz")):
        if not include_coronary and "_CORO_" in path.name:
            continue
        yield path


def convert_one(path: Path, out_path: Path, *, n_bins: int, boundary_quantile: float) -> Dict:
    with np.load(path, allow_pickle=True) as z:
        arrays = {k: z[k] for k in z.files}

    required = {"pos", "wall_mask", "x"}
    missing = sorted(required - arrays.keys())
    if missing:
        raise ValueError(f"{path} missing required arrays: {missing}")

    old_x = np.asarray(arrays["x"], dtype=np.float32)
    new_x = build_noleak_x(
        arrays["pos"],
        arrays["wall_mask"],
        n_bins=n_bins,
        boundary_quantile=boundary_quantile,
    )
    arrays["x"] = new_x
    arrays["feature_mode"] = np.array("noleak_pca_axis_v1")
    arrays["feature_names"] = np.array(FEATURE_NAMES)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)

    old_dir_norm = float(np.mean(np.linalg.norm(old_x[:, 3:6], axis=1)))
    new_dir_norm = float(np.mean(np.linalg.norm(new_x[:, 3:6], axis=1)))
    return {
        "case": path.stem,
        "modality": infer_modality(path.stem),
        "n_nodes": int(new_x.shape[0]),
        "old_flow_dir_mean_norm": old_dir_norm,
        "new_tangent_mean_norm": new_dir_norm,
        "inlet_proxy_frac": float(new_x[:, 6].mean()),
        "outlet_proxy_frac": float(new_x[:, 7].mean()),
        "wall_frac": float(new_x[:, 10].mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default="/users/scratch1/dancies/data/vmr_npz_v2")
    ap.add_argument("--out-dir", default="/users/scratch1/dancies/data/vmr_npz_v8_noleak")
    ap.add_argument("--include-coronary", action="store_true",
                    help="Also convert *_CORO_* files. The active split can still exclude them.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n-bins", type=int, default=100)
    ap.add_argument("--boundary-quantile", type=float, default=0.02)
    args = ap.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    if not src_dir.is_dir():
        raise SystemExit(f"src-dir does not exist: {src_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    _copy_jsons(src_dir, out_dir)

    rows = []
    converted = skipped = 0
    for path in iter_npz(src_dir, include_coronary=bool(args.include_coronary)):
        out_path = out_dir / path.name
        if out_path.exists() and not args.force:
            skipped += 1
            continue
        info = convert_one(
            path,
            out_path,
            n_bins=int(args.n_bins),
            boundary_quantile=float(args.boundary_quantile),
        )
        rows.append(info)
        converted += 1
        print(
            f"[ok] {path.name}: N={info['n_nodes']} "
            f"inlet={info['inlet_proxy_frac']:.3f} "
            f"outlet={info['outlet_proxy_frac']:.3f} "
            f"wall={info['wall_frac']:.3f}",
            flush=True,
        )

    summary = {
        "feature_mode": "noleak_pca_axis_v1",
        "source_dir": str(src_dir),
        "out_dir": str(out_dir),
        "include_coronary": bool(args.include_coronary),
        "converted": converted,
        "skipped_existing": skipped,
        "total_npz": len(list(iter_npz(out_dir, include_coronary=True))),
        "modality_counts": {
            "MR": sum(1 for p in iter_npz(out_dir, include_coronary=True) if infer_modality(p.stem) == "MR"),
            "CT": sum(1 for p in iter_npz(out_dir, include_coronary=True) if infer_modality(p.stem) == "CT"),
        },
        "feature_names": FEATURE_NAMES,
        "cases": rows,
    }
    split_path = out_dir / "split.json"
    if split_path.exists():
        with open(split_path) as f:
            split = json.load(f)
        summary["split_counts"] = {k: len(v) for k, v in split.items()}
        summary["split_modality_counts"] = {
            k: {
                "MR": sum(1 for cid in v if infer_modality(cid) == "MR"),
                "CT": sum(1 for cid in v if infer_modality(cid) == "CT"),
            }
            for k, v in split.items()
        }
    with open(out_dir / "_noleak_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Done: converted={converted} skipped={skipped} out={out_dir}")


if __name__ == "__main__":
    main()
