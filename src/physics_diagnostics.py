# -*- coding: utf-8 -*-
"""
Quantitative physics diagnostics — Phase E2.

Converts an ML ablation into a *physical identifiability* measurement.
For each (variant, seed, case, split) prediction dump, compute:

  D1 — angular error             : per-node ∠(u_pred, u_true)
  D2 — magnitude scaling residual: |u_pred|/|u_true| vs local radius, and
                                   |u_true| vs local_radius (Poiseuille proxy)
  D3 — divergence residual       : local Jacobian trace of u (mass conservation
                                   proxy on graph)
  D4 — magnitude vs distance     : |u| as a function of arc-length distance
                                   from inlet (centerline-derived)

For each case we compute:
  - median/mean/p90 of D1
  - Spearman corr(|u_true|, radius) and corr(|u_pred|, radius)
  - mean abs divergence residual / mean |u| (normalized)

Output: results/diagnostics/physics/{variant}/{seed}/{case}_{split}.json
        results/diagnostics/physics_summary.csv (per (variant, seed, case))

Usage:
  python src/physics_diagnostics.py \
      --predictions_root results/predictions \
      --withleak_npz data/npz_withleak \
      --noleak_npz   data/npz_noleak \
      --out_dir results/diagnostics/physics \
      --summary_csv results/diagnostics/physics_summary.csv
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-node angle in degrees between vectors (sign-preserving, not folded)."""
    na = np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-8)
    nb = np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-8)
    cos = np.clip(((a / na) * (b / nb)).sum(-1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def wall_distance(pos: np.ndarray, wall_mask: np.ndarray) -> np.ndarray:
    if not wall_mask.any():
        return np.ones(pos.shape[0], dtype=np.float32)
    wall_pts = pos[wall_mask]
    d, _ = cKDTree(wall_pts).query(pos, k=1)
    return d.astype(np.float32)


def local_jacobian_trace(
    pos: np.ndarray,
    u: np.ndarray,
    *,
    k_neighbors: int = 16,
    sigma: float = 1.0,
) -> np.ndarray:
    """Estimate div(u) per node by least-squares Jacobian on kNN neighborhood.

    For each node i, find k nearest neighbors j. Solve for J (3x3) minimizing
      sum_j ||u_j - u_i - J (x_j - x_i)||^2
    weighted by gaussian over distance. Then div_i = tr(J).
    """
    n = pos.shape[0]
    tree = cKDTree(pos)
    dists, idxs = tree.query(pos, k=int(k_neighbors) + 1)
    # drop self
    dists = dists[:, 1:]
    idxs = idxs[:, 1:]

    div = np.zeros(n, dtype=np.float32)
    h = float(sigma) * np.median(dists[:, 0]).clip(min=1e-6)
    for i in range(n):
        nb = idxs[i]
        dx = (pos[nb] - pos[i]).astype(np.float32)  # (k,3)
        du = (u[nb] - u[i]).astype(np.float32)      # (k,3)
        # Gaussian weights
        d = dists[i]
        w = np.exp(-(d / h) ** 2).astype(np.float32)
        W = w[:, None]
        # Weighted least squares: dx.T W dx J^T = dx.T W du
        A = dx.T @ (W * dx)
        # regularization
        A = A + 1e-6 * np.eye(3, dtype=np.float32)
        B = dx.T @ (W * du)
        try:
            J_T = np.linalg.solve(A, B)  # solves A J^T = B → J^T = A^{-1} B
            J = J_T.T
            div[i] = float(np.trace(J))
        except np.linalg.LinAlgError:
            div[i] = 0.0
    return div


def local_jacobian_trace_vec(
    pos: np.ndarray,
    u: np.ndarray,
    *,
    k_neighbors: int = 16,
    sigma: float = 1.0,
) -> np.ndarray:
    """Vectorized version of local_jacobian_trace (much faster for large n)."""
    n = pos.shape[0]
    tree = cKDTree(pos)
    dists, idxs = tree.query(pos, k=int(k_neighbors) + 1)
    dists = dists[:, 1:]
    idxs = idxs[:, 1:]
    k = idxs.shape[1]

    dx = pos[idxs] - pos[:, None, :]      # (n, k, 3)
    du = u[idxs] - u[:, None, :]          # (n, k, 3)
    h = float(sigma) * np.median(dists[:, 0]).clip(min=1e-6)
    w = np.exp(-(dists / h) ** 2).astype(np.float32)  # (n, k)

    # A_i = dx_i^T W_i dx_i        shape (n,3,3)
    Wdx = dx * w[:, :, None]
    A = np.einsum("nki,nkj->nij", Wdx, dx)
    # regularize
    A = A + 1e-6 * np.eye(3, dtype=np.float32)[None]
    # B_i = dx_i^T W_i du_i        shape (n,3,3)
    B = np.einsum("nki,nkj->nij", Wdx, du)
    # Solve A_i J_i^T = B_i  →  J_i^T = A_i^{-1} B_i
    try:
        J_T = np.linalg.solve(A, B)
        # trace of J_i = trace of J_i^T
        div = np.einsum("nii->n", J_T).astype(np.float32)
    except np.linalg.LinAlgError:
        div = np.array([
            float(np.trace(np.linalg.solve(A[i] + 1e-3 * np.eye(3, dtype=np.float32), B[i]).T))
            for i in range(n)
        ], dtype=np.float32)
    return div


def diagnose_case(
    pred: np.ndarray,
    y: np.ndarray,
    pos: np.ndarray,
    wall_mask: np.ndarray,
    *,
    div_k: int = 16,
    moving_quantile: float = 0.20,
) -> Dict:
    """Compute physics diagnostics for one case."""
    n = pos.shape[0]
    u_pred = pred.astype(np.float32)
    u_true = y.astype(np.float32)
    u_pred_mag = np.linalg.norm(u_pred, axis=1)
    u_true_mag = np.linalg.norm(u_true, axis=1)

    # moving nodes (filter wall and very slow)
    thr = float(np.quantile(u_true_mag[~wall_mask], moving_quantile)) if (~wall_mask).any() else 0.0
    moving = (~wall_mask) & (u_true_mag > thr)
    if int(moving.sum()) < 100:
        moving = ~wall_mask

    # D1: angle
    ang = angle_deg(u_pred[moving], u_true[moving])

    # D2: magnitude vs radius (Poiseuille proxy)
    d_wall = wall_distance(pos, wall_mask)
    radius_norm = d_wall  # treat as proxy for local radius coordinate
    rho_true, p_true = spearmanr(u_true_mag[moving], radius_norm[moving])
    rho_pred, p_pred = spearmanr(u_pred_mag[moving], radius_norm[moving])

    # D2b: |u_pred|/|u_true| residual (clipped to avoid div-by-zero)
    rmag = u_pred_mag[moving] / u_true_mag[moving].clip(min=1e-6)
    # D2c: ratio percentiles (a clean diagnostic, not bounded)
    rmag = np.clip(rmag, 0.0, 10.0)

    # D3: divergence residual
    div_true = local_jacobian_trace_vec(pos, u_true, k_neighbors=int(div_k))
    div_pred = local_jacobian_trace_vec(pos, u_pred, k_neighbors=int(div_k))
    # normalize by characteristic |u|/L where L = median nearest-neighbor distance
    tree = cKDTree(pos)
    nn_d, _ = tree.query(pos, k=2)
    L = float(np.median(nn_d[:, 1])).clip(min=1e-8) if hasattr(float, "clip") else float(np.median(nn_d[:, 1]))
    L = max(L, 1e-8)
    U = float(u_true_mag[moving].mean()).clip(min=1e-8) if hasattr(float, "clip") else float(u_true_mag[moving].mean())
    U = max(U, 1e-8)
    norm = U / L
    div_true_n = np.abs(div_true[moving]) / norm
    div_pred_n = np.abs(div_pred[moving]) / norm

    return dict(
        n_nodes=int(n),
        n_moving=int(moving.sum()),
        # D1
        angle_median=float(np.median(ang)),
        angle_mean=float(ang.mean()),
        angle_p90=float(np.percentile(ang, 90)),
        # D2
        spearman_true_mag_radius=float(rho_true) if np.isfinite(rho_true) else float("nan"),
        spearman_pred_mag_radius=float(rho_pred) if np.isfinite(rho_pred) else float("nan"),
        # D2b
        mag_ratio_median=float(np.median(rmag)),
        mag_ratio_p90=float(np.percentile(rmag, 90)),
        # D3
        div_true_median=float(np.median(div_true_n)),
        div_true_mean=float(div_true_n.mean()),
        div_pred_median=float(np.median(div_pred_n)),
        div_pred_mean=float(div_pred_n.mean()),
        # context
        umean=float(u_true_mag[moving].mean()),
        wall_frac=float(wall_mask.mean()),
        mesh_L=float(L),
    )


def process_prediction_file(pred_path: Path) -> Dict:
    d = np.load(pred_path, allow_pickle=True)
    pred = d["pred"].astype(np.float32)
    y = d["y"].astype(np.float32)
    pos = d["pos"].astype(np.float32)
    wall_mask = np.asarray(d["wall_mask"]).astype(bool)
    case_id = str(d["case_id"]) if "case_id" in d.files else pred_path.stem
    variant = str(d["variant"]) if "variant" in d.files else pred_path.parent.parent.name
    seed = int(d["seed"]) if "seed" in d.files else int(pred_path.parent.name.split("_")[-1])
    split = str(d["split"]) if "split" in d.files else (
        "test" if "test_" in pred_path.name else ("val" if "val_" in pred_path.name else "unknown")
    )
    diag = diagnose_case(pred, y, pos, wall_mask)
    diag.update(case=case_id, variant=variant, seed=int(seed), split=split)
    return diag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--variants", nargs="*", default=None,
                    help="Restrict to these variants; default = all subdirs")
    ap.add_argument("--splits", nargs="*", default=["val", "test"])
    args = ap.parse_args()

    pred_root = Path(args.predictions_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.variants is None:
        variants = [p.name for p in pred_root.iterdir() if p.is_dir()]
    else:
        variants = list(args.variants)
    variants = sorted(variants)

    rows = []
    for variant in variants:
        v_dir = pred_root / variant
        for seed_dir in sorted(v_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            for f in sorted(seed_dir.glob("*.npz")):
                # filter by split prefix
                split_ok = any(f.name.startswith(s + "_") for s in args.splits)
                if not split_ok:
                    continue
                try:
                    diag = process_prediction_file(f)
                except Exception as e:
                    print(f"  [ERR] {f}: {e}")
                    continue
                # write per-case JSON
                rel = f"{variant}/{seed_dir.name}/{f.stem}.json"
                jpath = out_dir / rel
                jpath.parent.mkdir(parents=True, exist_ok=True)
                with open(jpath, "w") as jf:
                    json.dump(diag, jf, indent=2)
                rows.append(diag)
                print(
                    f"  {variant:24s} seed={diag['seed']:<5d} {diag['split']:5s} "
                    f"{diag['case'][:36]:36s}  angle={diag['angle_median']:6.2f}°  "
                    f"div_pred_n={diag['div_pred_mean']:8.3f}  "
                    f"div_true_n={diag['div_true_mean']:8.3f}"
                )

    # write summary CSV
    import csv
    if rows:
        keys = list(rows[0].keys())
        # put identifiers first
        ident = ["variant", "seed", "split", "case"]
        keys = ident + [k for k in keys if k not in ident]
        out_csv = Path(args.summary_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as cf:
            w = csv.DictWriter(cf, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"[diag] wrote {len(rows)} rows → {out_csv}")


if __name__ == "__main__":
    main()
