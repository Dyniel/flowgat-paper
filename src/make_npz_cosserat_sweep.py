# -*- coding: utf-8 -*-
"""
Synthetic curved-tube Cosserat/Dean sweep.

This generator mirrors src/make_npz_womersley.py but sweeps a structured
disc mesh along a constant-curvature planar centerline and adds the specified
first-order Dean secondary-flow correction. The NPZ schema intentionally
matches the existing Womersley/VMR graph datasets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Womersley analytic solution (copied verbatim from make_npz_womersley.py)
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
# Mesh/graph helpers (copied verbatim from make_npz_womersley.py)
# ---------------------------------------------------------------------------

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
# Curved Cosserat tube geometry
# ---------------------------------------------------------------------------

def make_curved_centerline(
    L: float,
    kappa: float,
    n_axial: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Constant-curvature planar arc sampled by arc length.

    Args:
        L: arc length in metres.
        kappa: curvature in 1/metre.
        n_axial: number of centreline stations.

    Returns:
        gamma, T, N, B arrays, each sampled at n_axial stations. At s=0,
        T points along +z, N along +x, and B along +y.
    """
    s = np.linspace(0.0, L, int(n_axial), dtype=np.float64)
    if abs(float(kappa)) < 1e-12:
        gamma = np.stack([np.zeros_like(s), np.zeros_like(s), s], axis=1)
        T = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float64), (len(s), 1))
        N = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float64), (len(s), 1))
        B = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float64), (len(s), 1))
    else:
        phi = float(kappa) * s
        gamma = np.stack([
            (1.0 - np.cos(phi)) / float(kappa),
            np.zeros_like(s),
            np.sin(phi) / float(kappa),
        ], axis=1)
        T = np.stack([np.sin(phi), np.zeros_like(s), np.cos(phi)], axis=1)
        N = np.stack([np.cos(phi), np.zeros_like(s), -np.sin(phi)], axis=1)
        B = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float64), (len(s), 1))

    return (
        gamma.astype(np.float32),
        T.astype(np.float32),
        N.astype(np.float32),
        B.astype(np.float32),
    )


def make_curved_tube_mesh(
    centerline: np.ndarray,
    T: np.ndarray,
    N: np.ndarray,
    B: np.ndarray,
    R: float,
    n_axial: int,
    n_radial: int,
    n_angular: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sweep the same structured disc layout as make_cylinder_mesh along gamma."""
    pos: List[np.ndarray] = []
    wall: List[bool] = []
    axial_idx: List[int] = []
    frames: List[np.ndarray] = []

    for ia in range(int(n_axial)):
        g = centerline[ia]
        frame = np.stack([T[ia], N[ia], B[ia]], axis=0).astype(np.float32)
        for ir in range(int(n_radial) + 1):
            r = (ir / max(int(n_radial), 1)) * float(R)
            if ir == 0:
                pos.append(g.copy())
                wall.append(False)
                axial_idx.append(ia)
                frames.append(frame)
            else:
                for ja in range(int(n_angular)):
                    theta = 2.0 * np.pi * ja / int(n_angular)
                    p = g + r * np.cos(theta) * N[ia] + r * np.sin(theta) * B[ia]
                    pos.append(p.astype(np.float32))
                    wall.append(ir == int(n_radial))
                    axial_idx.append(ia)
                    frames.append(frame)

    return (
        np.asarray(pos, dtype=np.float32),
        np.asarray(wall, dtype=bool),
        np.asarray(axial_idx, dtype=np.int64),
        np.asarray(frames, dtype=np.float32),
    )


def dean_secondary(
    r_over_R: np.ndarray,
    theta: np.ndarray,
    De: float,
    u_z_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """First-order Dean vortex-pair secondary velocity components."""
    xi = np.asarray(r_over_R, dtype=np.float32).clip(0.0, 1.0)
    th = np.asarray(theta, dtype=np.float32)
    scale = np.float32(float(De) * float(u_z_max))

    xi2 = xi * xi
    u_N = (1.0 / 576.0) * (1.0 - xi2) * (4.0 - xi2) * np.cos(th)
    bracket = ((1.0 - 3.0 * xi2) * (4.0 - xi2)
               + xi * (1.0 - xi2) * (-2.0 * xi))
    u_B = -(1.0 / 576.0) * bracket * np.sin(th)
    return (scale * u_N).astype(np.float32), (scale * u_B).astype(np.float32)


# ---------------------------------------------------------------------------
# Build a single case
# ---------------------------------------------------------------------------

def build_one_curved_case(
    case_id: str,
    *,
    R: float,
    L: float,
    kappa: float,
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
    centerline, T, N, B = make_curved_centerline(
        L=L, kappa=kappa, n_axial=n_axial,
    )
    pos_local, wall, axial_idx, frame_at_node = make_curved_tube_mesh(
        centerline, T, N, B, R=R,
        n_axial=n_axial, n_radial=n_radial, n_angular=n_angular,
    )
    n_nodes = pos_local.shape[0]

    T_node = frame_at_node[:, 0, :]
    N_node = frame_at_node[:, 1, :]
    B_node = frame_at_node[:, 2, :]
    center_at_node = centerline[axial_idx]
    radial_vec = pos_local - center_at_node
    n_coord = np.sum(radial_vec * N_node, axis=1)
    b_coord = np.sum(radial_vec * B_node, axis=1)
    r_local = np.sqrt(n_coord ** 2 + b_coord ** 2).astype(np.float32)
    theta = np.arctan2(b_coord, n_coord).astype(np.float32)
    r_per_R = (r_local / max(R, 1e-12)).clip(0.0, 1.0).astype(np.float32)

    omega = (alpha_womersley ** 2) * nu / max(R ** 2, 1e-12)
    u_axial = womersley_velocity(
        r_local, t_phase, R=R, omega=omega, nu=nu, p_amp=p_amp,
    )
    u_z_max = float(np.max(np.abs(u_axial)))
    De = float(R * kappa)
    u_N, u_B = dean_secondary(r_per_R, theta, De=De, u_z_max=u_z_max)

    y_local = (
        u_axial[:, None] * T_node
        + u_N[:, None] * N_node
        + u_B[:, None] * B_node
    ).astype(np.float32)

    r_safe = np.maximum(r_local, 1e-8)
    wn_local = (
        (n_coord / r_safe)[:, None] * N_node
        + (b_coord / r_safe)[:, None] * B_node
    ).astype(np.float32)
    wn_local[~wall] = 0.0

    if apply_rotation:
        ax = rng.normal(size=3).astype(np.float32)
        theta_rot = float(rng.uniform(0.0, np.pi))
        Rmat = rotation_matrix(ax, theta_rot)
    else:
        Rmat = np.eye(3, dtype=np.float32)

    pos = (pos_local @ Rmat.T).astype(np.float32)
    y = (y_local @ Rmat.T).astype(np.float32)
    wn = (wn_local @ Rmat.T).astype(np.float32)
    T_rot_node = (T_node @ Rmat.T).astype(np.float32)

    edge_index = knn_edges(pos, k=knn)
    diff = pos[edge_index[1]] - pos[edge_index[0]]
    dist = np.linalg.norm(diff, axis=1, keepdims=True)
    edge_attr = np.concatenate([diff, dist, np.ones_like(dist)], axis=1).astype(np.float32)

    from scipy.spatial import cKDTree
    wall_pts = pos[wall]
    if len(wall_pts) >= 8:
        d_wall, _ = cKDTree(wall_pts).query(pos, k=1)
    else:
        d_wall = np.zeros(n_nodes, dtype=np.float32)
    d_wall = d_wall.astype(np.float32)

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
    lo, hi = np.quantile(axial_proj, [0.02, 0.98])
    inlet = (axial_proj <= lo).astype(np.float32)
    outlet = (axial_proj >= hi).astype(np.float32)
    local_R_norm = np.ones(n_nodes, dtype=np.float32)
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

    pg_mag = -p_amp * np.cos(omega * t_phase)
    p_grad = (pg_mag * T_rot_node).astype(np.float32)

    u_char = np.float32(max(np.linalg.norm(y, axis=1).max(), 1e-6))
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
            R=float(R),
            L=float(L),
            alpha_womersley=float(alpha_womersley),
            omega=float(omega),
            nu=float(nu),
            p_amp=float(p_amp),
            t_phase=float(t_phase),
            kappa=float(kappa / 1000.0),  # stored in 1/mm
            eps=float(R / L),
            De=float(De),
        ),
    )


def _sample_case_params(
    rng: np.random.Generator,
    *,
    nu: float,
    max_de: float,
    max_eps: float,
) -> Tuple[float, float, float, float, float, float, float]:
    """Uniform parameter draw with Dean/slenderness rejection."""
    while True:
        R = float(rng.uniform(0.008, 0.018))
        L = float(rng.uniform(0.100, 0.300))
        alpha = float(rng.uniform(2.0, 12.0))
        kappa_mm = float(rng.uniform(0.0, 0.04))
        kappa = kappa_mm * 1000.0
        eps = R / L
        De = R * kappa
        if De <= 0.0 or De > float(max_de) or eps > float(max_eps):
            continue

        omega = (alpha ** 2) * nu / max(R ** 2, 1e-12)
        t_phase = float(rng.uniform(0.0, 2.0 * np.pi / max(omega, 1e-12)))
        target_peak = float(rng.uniform(0.2, 1.5))
        r_probe = np.linspace(0.0, R, 512, dtype=np.float32)
        unit_resp = womersley_velocity(
            r_probe, t_phase, R=R, omega=omega, nu=nu, p_amp=1.0,
        )
        peak_unit = float(np.max(np.abs(unit_resp)))
        if not np.isfinite(peak_unit) or peak_unit <= 1e-12:
            continue
        p_amp = target_peak / peak_unit
        return R, L, alpha, kappa, p_amp, t_phase, target_peak


def _split_cases(case_ids: List[str], n_cases: int) -> dict:
    rng = np.random.default_rng(22)
    shuffled = list(case_ids)
    rng.shuffle(shuffled)
    if int(n_cases) == 300:
        n_train, n_val = 240, 24
    else:
        n_train = int(round(0.80 * int(n_cases)))
        n_val = int(round(0.08 * int(n_cases)))
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def _summary(meta_rows: List[dict]) -> str:
    cols = [
        ("R_mm", [m["R"] * 1000.0 for m in meta_rows]),
        ("L_mm", [m["L"] * 1000.0 for m in meta_rows]),
        ("alpha", [m["alpha_womersley"] for m in meta_rows]),
        ("kappa_1/mm", [m["kappa"] for m in meta_rows]),
        ("eps", [m["eps"] for m in meta_rows]),
        ("De", [m["De"] for m in meta_rows]),
    ]
    lines = ["parameter,min,median,max"]
    for name, vals in cols:
        a = np.asarray(vals, dtype=np.float64)
        lines.append(
            f"{name},{np.min(a):.6g},{np.median(a):.6g},{np.max(a):.6g}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_cases", type=int, default=300)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n_axial", type=int, default=80)
    ap.add_argument("--n_radial", type=int, default=10)
    ap.add_argument("--n_angular", type=int, default=32)
    ap.add_argument("--max_de", type=float, default=0.35)
    ap.add_argument("--max_eps", type=float, default=0.20)
    ap.add_argument("--knn", type=int, default=16)
    args = ap.parse_args()

    rng = np.random.default_rng(int(args.seed))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    nu = 3.3e-6
    case_ids: List[str] = []
    metas: List[dict] = []

    for i in range(int(args.n_cases)):
        R, L, alpha, kappa, p_amp, t_phase, target_peak = _sample_case_params(
            rng, nu=nu, max_de=float(args.max_de), max_eps=float(args.max_eps),
        )
        kappa_mm = kappa / 1000.0
        De = R * kappa
        eps = R / L
        case_id = (
            f"cs_{i:03d}_R{R*1000:04.1f}_L{L*1000:05.1f}_"
            f"a{alpha:04.1f}_k{kappa_mm:05.3f}"
        ).replace(".", "p")

        d = build_one_curved_case(
            case_id,
            R=R,
            L=L,
            kappa=kappa,
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
        meta = d.pop("meta")
        np.savez_compressed(out / f"{case_id}.npz", **d, meta=json.dumps(meta))
        case_ids.append(case_id)
        metas.append(meta)
        print(
            f"  [{i+1}/{int(args.n_cases)}] {case_id}: "
            f"N={d['pos'].shape[0]:>6d} R={R*1000:.1f}mm L={L*1000:.0f}mm "
            f"alpha={alpha:.2f} kappa={kappa_mm:.4f}/mm "
            f"eps={eps:.3f} De={De:.3f} peak={target_peak:.3f}m/s"
        )

    split = _split_cases(case_ids, int(args.n_cases))
    with open(out / "split.json", "w") as f:
        json.dump(split, f, indent=2)

    print(f"[cosserat-sweep] DONE: {len(case_ids)} cases written to {out}")
    print(
        f"  split: train={len(split['train'])} "
        f"val={len(split['val'])} test={len(split['test'])}"
    )
    print(_summary(metas))


if __name__ == "__main__":
    main()
