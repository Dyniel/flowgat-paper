# -*- coding: utf-8 -*-
"""
Synthetic Womersley pulsatile flow benchmark — Phase E3.

Generates NPZ cases with the same schema as VMR-derived data, but where
the ground-truth velocity field is *analytically known* from Womersley's
solution for pulsatile flow in a straight rigid cylindrical tube. This
gives us a controlled environment for our identifiability claim:

  - True axial direction = pure tube axis (no arch, no curvature)
  - True magnitude profile = Womersley parabolic-like radial profile,
    set by pressure gradient amplitude and Womersley number
  - All other geometric features (PCA tangent, centerline tangent,
    local radius) are exactly known

Falsification test the paper makes:

  H1 (identifiability of direction): on a straight tube, global PCA tangent
      should give ~0° angle to true direction. We predict the noleak variant
      should recover direction perfectly here — confirming that geometric
      identifiability is *complete* in the idealized regime, and the residual
      error on real aortas is *only* from imperfect centerline extraction.

  H2 (non-identifiability of magnitude): magnitude depends on the time
      phase / driving frequency / Womersley number — without access to those
      (or a proxy like inlet flow rate), no purely-geometric prior can
      recover it.

We generate cases with varied:
  - tube length L (axial) and radius R (cross-section)
  - Womersley number α = R sqrt(ω/ν)
  - dimensionless mean velocity Re_m = U_m R / ν
  - random rigid rotation in 3D (so PCA axis differs case-to-case)

The mesh is a structured cylindrical mesh (axial × radial × angular), but
written in unstructured NPZ form: pos, edge_index (kNN graph in 3D), wall_mask,
wall_normal, x (11 cols, schema-compatible), y (3D velocity), p_grad (axial
component of -∂p/∂x as a body force), u_char, length_char.

Output:
  data/npz_womersley_<TAG>/<case>.npz + split.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Womersley analytic solution
# ---------------------------------------------------------------------------

def womersley_velocity(r: np.ndarray, t: float, *,
                       R: float, omega: float, nu: float,
                       p_amp: float) -> np.ndarray:
    """Axial velocity in a straight rigid tube driven by p_x(t) = p_amp · cos(ω t).

    For r ∈ [0, R], we use the analytic Womersley solution:
      u(r,t) = Re{ (i p_amp / (ρ ω)) [ 1 - J0(αr/R √(i^3)) / J0(α √(i^3)) ] · exp(i ω t) }
    For simplicity we set ρ=1 (the same nondimensional convention as the rest
    of the codebase) and return a real-valued u.
    """
    from scipy.special import jv  # Bessel J0, J1

    if omega <= 0:
        # Steady Poiseuille fallback
        return (p_amp / (4.0 * nu)) * (R ** 2 - r ** 2)

    alpha = R * np.sqrt(omega / nu)
    # i^{3/2} = exp(i 3π/4)
    # Womersley argument uses complex Bessel J0 of i^{3/2} (alpha r / R)
    iota32 = np.exp(1j * 3.0 * np.pi / 4.0)
    arg_wall = alpha * iota32
    arg_r = (alpha * (r / max(R, 1e-9))) * iota32
    J0_wall = jv(0, arg_wall)
    J0_r = jv(0, arg_r)
    factor = 1.0 - (J0_r / J0_wall)
    coef = (1j * p_amp / max(omega, 1e-9)) * np.exp(1j * omega * t)
    u_c = coef * factor
    return np.real(u_c).astype(np.float32)


# ---------------------------------------------------------------------------
# Cylindrical mesh generation
# ---------------------------------------------------------------------------

def make_cylinder_mesh(
    *,
    R: float,
    L: float,
    n_axial: int,
    n_radial: int,
    n_angular: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pos, wall_mask, axial_idx_per_node) for a structured cylinder.

    Layout: for each axial slice (n_axial of them), and each radial ring
    (n_radial, where 0 is the center), and each angle (n_angular except at
    center which is single point), create a node. Wall = outermost radial ring.
    """
    pos = []
    wall = []
    axial_idx = []

    for ia in range(n_axial):
        z = (ia / max(n_axial - 1, 1)) * L
        for ir in range(n_radial + 1):
            r = (ir / n_radial) * R
            if ir == 0:
                pos.append([0.0, 0.0, z])
                wall.append(False)
                axial_idx.append(ia)
            else:
                for ja in range(n_angular):
                    theta = 2.0 * np.pi * ja / n_angular
                    x = r * np.cos(theta)
                    y = r * np.sin(theta)
                    pos.append([x, y, z])
                    wall.append(ir == n_radial)
                    axial_idx.append(ia)
    return (
        np.asarray(pos, dtype=np.float32),
        np.asarray(wall, dtype=bool),
        np.asarray(axial_idx, dtype=np.int64),
    )


def knn_edges(pos: np.ndarray, k: int) -> np.ndarray:
    from scipy.spatial import cKDTree
    tree = cKDTree(pos)
    _, idx = tree.query(pos, k=k + 1)
    src = np.repeat(np.arange(pos.shape[0]), k)
    dst = idx[:, 1:].reshape(-1)
    return np.stack([src, dst], axis=0).astype(np.int64)


def rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues rotation."""
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([
        [aa + bb - cc - dd, 2 * (bc + ad),     2 * (bd - ac)],
        [2 * (bc - ad),     aa + cc - bb - dd, 2 * (cd + ab)],
        [2 * (bd + ac),     2 * (cd - ab),     aa + dd - bb - cc],
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Build a single case
# ---------------------------------------------------------------------------

def build_one_case(
    case_id: str,
    *,
    R: float,
    L: float,
    n_axial: int,
    n_radial: int,
    n_angular: int,
    alpha_womersley: float,
    nu: float,
    p_amp: float,
    t_phase: float,
    rng: np.random.Generator,
    knn: int = 16,
    apply_rotation: bool = True,
) -> dict:
    pos_local, wall, axial_idx = make_cylinder_mesh(
        R=R, L=L, n_axial=n_axial, n_radial=n_radial, n_angular=n_angular,
    )
    n_nodes = pos_local.shape[0]

    # radial coord in local frame
    r_local = np.sqrt(pos_local[:, 0] ** 2 + pos_local[:, 1] ** 2).astype(np.float32)
    # omega from alpha = R sqrt(omega/nu) -> omega = alpha^2 * nu / R^2
    omega = (alpha_womersley ** 2) * nu / max(R ** 2, 1e-12)

    # axial velocity (real part of Womersley) — local frame, axis = z
    u_axial = womersley_velocity(r_local, t_phase, R=R, omega=omega, nu=nu, p_amp=p_amp)

    # local velocity vector: u along +z
    y_local = np.zeros_like(pos_local)
    y_local[:, 2] = u_axial
    # enforce no-slip exactly at wall
    y_local[wall] = 0.0

    # wall normal in local frame (radial direction outward)
    wn_local = np.stack([
        pos_local[:, 0] / r_local.clip(min=1e-8),
        pos_local[:, 1] / r_local.clip(min=1e-8),
        np.zeros_like(r_local),
    ], axis=1).astype(np.float32)
    wn_local[~wall] = 0.0  # zero out interior normals

    # Apply random rigid rotation to make PCA axis non-trivial
    if apply_rotation:
        ax = rng.normal(size=3).astype(np.float32)
        theta = float(rng.uniform(0.0, np.pi))
        Rmat = rotation_matrix(ax, theta)
    else:
        Rmat = np.eye(3, dtype=np.float32)
    pos = (pos_local @ Rmat.T).astype(np.float32)
    y = (y_local @ Rmat.T).astype(np.float32)
    wn = (wn_local @ Rmat.T).astype(np.float32)

    # graph edges (kNN in 3D, mirrors typical preprocessing)
    edge_index = knn_edges(pos, k=knn)
    # edge_attr: 5 cols, mimic VMR schema: dx, dy, dz, |d|, 1.0
    diff = pos[edge_index[1]] - pos[edge_index[0]]
    dist = np.linalg.norm(diff, axis=1, keepdims=True)
    edge_attr = np.concatenate([diff, dist, np.ones_like(dist)], axis=1).astype(np.float32)

    # Build x feature matrix (11 columns, same as noleak schema)
    # 0: axial_pca_norm
    # 1: wall_distance_norm
    # 2: radial_position_local (r/R)
    # 3-5: global PCA tangent (per-node tile)
    # 6: inlet proxy
    # 7: outlet proxy
    # 8: local_radius_norm
    # 9: distance_to_pca_axis_norm
    # 10: wall_flag
    from scipy.spatial import cKDTree
    wall_pts = pos[wall]
    if len(wall_pts) >= 8:
        d_wall, _ = cKDTree(wall_pts).query(pos, k=1)
    else:
        d_wall = np.zeros(n_nodes, dtype=np.float32)
    d_wall = d_wall.astype(np.float32)
    # PCA axis on positions
    centered = pos - pos.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    pca_axis = vt[0].astype(np.float32)
    if np.argmax(np.abs(pca_axis)) >= 0 and pca_axis[int(np.argmax(np.abs(pca_axis)))] < 0:
        pca_axis = -pca_axis
    axial_proj = centered @ pca_axis
    axial_norm = (axial_proj - axial_proj.min()) / max(axial_proj.max() - axial_proj.min(), 1e-9)
    dist_to_axis = np.linalg.norm(centered - axial_proj[:, None] * pca_axis[None, :], axis=1)
    dist_axis_norm = dist_to_axis / max(dist_to_axis.max(), 1e-9)
    dwall_norm = d_wall / max(d_wall.max(), 1e-9)
    r_per_R = (r_local / R).clip(0.0, 1.0).astype(np.float32)
    # inlet/outlet: 2% extremes on axial_proj
    lo, hi = np.quantile(axial_proj, [0.02, 0.98])
    inlet = (axial_proj <= lo).astype(np.float32)
    outlet = (axial_proj >= hi).astype(np.float32)
    # local radius norm: R (constant for cylinder), normalized to 1
    local_R_norm = np.ones(n_nodes, dtype=np.float32)
    # tangent (global PCA, tiled)
    tangent_tile = np.tile(pca_axis[None, :], (n_nodes, 1)).astype(np.float32)

    x = np.stack([
        axial_norm.astype(np.float32),
        dwall_norm.astype(np.float32),
        r_per_R,
        tangent_tile[:, 0], tangent_tile[:, 1], tangent_tile[:, 2],
        inlet, outlet,
        local_R_norm,
        dist_axis_norm.astype(np.float32),
        wall.astype(np.float32),
    ], axis=-1).astype(np.float32)

    # p_grad: pure axial body force in local frame ≈ -p_amp · cos(ω t) · ẑ, rotated
    pg_local = np.zeros_like(pos_local)
    pg_local[:, 2] = -p_amp * np.cos(omega * t_phase)
    p_grad = (pg_local @ Rmat.T).astype(np.float32)

    u_char = np.float32(max(np.abs(u_axial).max(), 1e-6))
    length_char = np.float32(2.0 * R)

    return dict(
        x=x,
        pos=pos,
        y=y,
        edge_index=edge_index,
        edge_attr=edge_attr,
        wall_mask=wall,
        wall_normal=wn,
        u_char=u_char,
        length_char=length_char,
        p_grad=p_grad,
        case_id=case_id,
        meta=dict(
            R=float(R), L=float(L),
            alpha_womersley=float(alpha_womersley),
            omega=float(omega),
            nu=float(nu),
            p_amp=float(p_amp),
            t_phase=float(t_phase),
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_train", type=int, default=20)
    ap.add_argument("--n_val", type=int, default=4)
    ap.add_argument("--n_test", type=int, default=4)
    ap.add_argument("--n_axial", type=int, default=80)
    ap.add_argument("--n_radial", type=int, default=12)
    ap.add_argument("--n_angular", type=int, default=24)
    ap.add_argument("--knn", type=int, default=16)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    rng = np.random.default_rng(int(args.seed))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    total = int(args.n_train + args.n_val + args.n_test)
    cases_by_split = {"train": [], "val": [], "test": []}

    nu = 3.3e-6  # m^2/s, same as base config
    for i in range(total):
        # vary geometry and flow parameters
        R = float(rng.uniform(0.008, 0.020))   # 8–20 mm radius (aortic range)
        L = float(rng.uniform(0.10, 0.30))     # 100–300 mm length
        alpha = float(rng.uniform(2.0, 12.0))  # aortic Womersley range
        # pressure-gradient amplitude — set so u_mean is O(0.1–1.0 m/s)
        # u_steady_max = p_amp R^2 / (4 nu) → p_amp ≈ u_max * 4 nu / R^2
        u_target = float(rng.uniform(0.2, 1.2))
        p_amp = u_target * 4.0 * nu / (R ** 2)
        t_phase = float(rng.uniform(0.0, 2.0 * np.pi / max(((alpha ** 2) * nu / R ** 2), 1e-6)))

        case_id = f"wm_{i:03d}_R{R*1000:04.1f}_L{L*1000:05.1f}_a{alpha:04.1f}".replace(".", "p")
        d = build_one_case(
            case_id,
            R=R, L=L,
            n_axial=int(args.n_axial),
            n_radial=int(args.n_radial),
            n_angular=int(args.n_angular),
            alpha_womersley=alpha,
            nu=nu,
            p_amp=p_amp,
            t_phase=t_phase,
            rng=rng,
            knn=int(args.knn),
        )
        path = out / f"{case_id}.npz"
        # NPZ: meta dict must be saved as flat fields
        meta = d.pop("meta")
        np.savez_compressed(path, **d, meta=json.dumps(meta))

        if i < args.n_train:
            cases_by_split["train"].append(case_id)
        elif i < args.n_train + args.n_val:
            cases_by_split["val"].append(case_id)
        else:
            cases_by_split["test"].append(case_id)
        print(f"  [{i+1}/{total}] {case_id}: N={d['pos'].shape[0]:>6d} α={alpha:5.1f} R={R*1000:.1f}mm L={L*1000:.0f}mm")

    with open(out / "split.json", "w") as f:
        json.dump(cases_by_split, f, indent=2)
    print(f"[womersley] DONE: {total} cases written to {out}")
    print(f"  split: train={len(cases_by_split['train'])} val={len(cases_by_split['val'])} test={len(cases_by_split['test'])}")


if __name__ == "__main__":
    main()
