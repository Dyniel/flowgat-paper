# -*- coding: utf-8 -*-
"""
Preprocess Vascular Model Repository (VMR) cases into compact .npz files.

Input (per case):
    <vmr_root>/<case_id>/
        Meshes/<case_id>.vtu            volumetric mesh + CFD results (u, p)
        Meshes/<case_id>.vtp            surface (wall) mesh, optional but preferred
        Simulations/.../flow-files/     time-averaged field as backup
    OR simply a single .vtu with 'velocity' and 'pressure' point arrays.

Output (per case):
    <out_dir>/<case_id>.npz  with fields:
        x            [N, F]   normalised node features
        pos          [N, 3]   positions in metres
        y            [N, 3]   velocity in m/s
        edge_index   [2, E]
        edge_attr    [E, De]  [dx_hat, dy_hat, dz_hat, |r|, wall_edge]
        wall_mask    [N]
        wall_normal  [N, 3]   unit outward wall normal (zero on core nodes)
        p_grad       [N, 3]   ∇p per node (estimated by WLS on the fly)
        u_char, length_char   scalars (peak systolic speed, aortic diameter)

Also writes <out_dir>/split.json with a patient-level split (not random graph
split — we guarantee no case leakage between train/val/test).

Features in `x` (11 channels, last is the wall flag — the model assumes this):
    [0]   normalised distance to inlet, along the lumen
    [1]   normalised distance to the nearest wall
    [2]   local lumen radius (from distance transform)
    [3:6] local flow direction from a quick centerline fit (unit vector)
    [6]   inlet indicator                (1 at inlet face, 0 otherwise)
    [7]   outlet indicator               (1 at any outlet face, 0 otherwise)
    [8]   normalised inflow rate at this case (constant per graph; broadcast)
    [9]   normalised Reynolds number at this case (constant; broadcast)
    [10]  wall flag                      (MUST BE LAST)

All vector features follow the rotation convention in
`flowgnn_aorta/data/augment.py`: EDGE_VEC_SLICE = slice(0, 3).

IMPORTANT: reads require `pyvista` and `numpy`; install once with
    pip install pyvista numpy scipy
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np

# --- Optional deps imported lazily so the file can be READ without them ---
def _import_pv():
    # VTK 9.6 can hang on this cluster while resolving every shared-library
    # symlink during broad imports (`vtk` / `vtkmodules.all`).  PyVista only
    # needs the actual library path, so avoid pathlib.resolve() in VTK's hook.
    try:
        import vtkmodules
        from pathlib import Path

        def _fast_find_lib_path(paths, vtk_module_name):
            for ext in vtkmodules.LIB_EXTS:
                for base_path in paths or []:
                    for f in Path(base_path).glob(f"{vtk_module_name}[.-]*"):
                        if f.is_file() and ext in f.suffixes:
                            return str(f)
            return None

        vtkmodules.find_lib_path = _fast_find_lib_path
    except Exception:
        pass

    try:
        import pyvista as pv
        return pv
    except Exception as e:
        print("[preprocess_vmr] pyvista is required. "
              "Install with: pip install pyvista", file=sys.stderr)
        raise


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _vtu_has_velocity(path: str, velocity_names=("velocity", "u", "U", "Velocity", "vel", "Vel")) -> bool:
    """Quick header-only check (no pyvista) — returns True if any known velocity field found."""
    try:
        with open(path, "rb") as f:
            header = f.read(8000).decode("latin1", errors="replace")
        import re
        names = set(re.findall(r'Name="([^"]+)"', header))
        return bool(names & set(velocity_names))
    except Exception:
        return False


def _find_vtu(case_dir: str) -> Optional[str]:
    """
    Search strategy for VMR SimVascular cases:
      1. Prefer Simulations/*/mesh-complete/mesh-complete.mesh.vtu that has a
         velocity field (these are post-processed CFD results).
      2. Fall back to any other .vtu in the tree.
    """
    # Stage 1: SimVascular mesh-complete outputs with CFD data
    sims_dir = os.path.join(case_dir, "Simulations")
    if os.path.isdir(sims_dir):
        for sim_name in sorted(os.listdir(sims_dir)):
            candidate = os.path.join(
                sims_dir, sim_name, "mesh-complete", "mesh-complete.mesh.vtu"
            )
            if os.path.isfile(candidate) and _vtu_has_velocity(candidate):
                return candidate

    # Stage 2: any .vtu with velocity
    for root, _, files in os.walk(case_dir):
        for f in sorted(files):
            if f.endswith(".vtu"):
                p = os.path.join(root, f)
                if _vtu_has_velocity(p):
                    return p

    # Stage 3: any .vtu (mesh-only; will likely fail at velocity extraction)
    for root, _, files in os.walk(case_dir):
        for f in sorted(files):
            if f.endswith(".vtu"):
                return os.path.join(root, f)
    return None


def _find_all_sim_vtu(case_dir: str):
    """
    Returns list of (sim_name, vtu_path, vtp_path) for each SimVascular
    Simulations/* subdirectory that has a valid mesh-complete.mesh.vtu with
    a velocity field.  Used to produce one .npz per condition.
    """
    results = []
    sims_dir = os.path.join(case_dir, "Simulations")
    if not os.path.isdir(sims_dir):
        return results
    for sim_name in sorted(os.listdir(sims_dir)):
        mc_dir = os.path.join(sims_dir, sim_name, "mesh-complete")
        vtu = os.path.join(mc_dir, "mesh-complete.mesh.vtu")
        if not os.path.isfile(vtu) or not _vtu_has_velocity(vtu):
            continue
        # Prefer walls_combined.vtp for wall detection; fall back to exterior
        vtp = None
        for vtp_name in ("walls_combined.vtp", "mesh-complete.exterior.vtp"):
            cand = os.path.join(mc_dir, vtp_name)
            if os.path.isfile(cand):
                vtp = cand
                break
        results.append((sim_name, vtu, vtp))
    return results


def _find_vtp(case_dir: str) -> Optional[str]:
    """
    For wall surface: prefer walls_combined.vtp (SimVascular) over arbitrary .vtp.
    """
    for root, _, files in os.walk(case_dir):
        for fname in ("walls_combined.vtp", "mesh-complete.exterior.vtp"):
            p = os.path.join(root, fname)
            if os.path.isfile(p):
                return p
    # fallback: any vtp
    for root, _, files in os.walk(case_dir):
        for f in sorted(files):
            if f.endswith(".vtp"):
                return os.path.join(root, f)
    return None


def _build_edges_from_vtu(grid) -> np.ndarray:
    """
    Build an undirected edge list from tet/hex cell connectivity, then make
    it directed (both ways) for PyG's convention.  Returns [2, E] int64.
    """
    # pyvista gives us `cells` as a flat array with size-prefixes; the
    # simplest robust extraction is via `extract_all_edges`.
    edges_poly = grid.extract_all_edges()
    # `lines` is [n0, i0, i1, n1, i2, i3, ...]; we assume n=2 throughout.
    lines = np.asarray(edges_poly.lines).reshape(-1, 3)
    assert (lines[:, 0] == 2).all(), "unexpected edge polyline size"
    und = lines[:, 1:3].astype(np.int64)               # [E_und, 2]
    # Directed: both orientations
    dirc = np.concatenate([und, und[:, ::-1]], axis=0)
    # edge_index: [2, E]  with src = row 0, dst = row 1
    return dirc.T.copy()


def _unit(v: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)


def _wls_grad_py(values: np.ndarray,
                 pos: np.ndarray,
                 edge_index: np.ndarray,
                 eps_reg: float = 1.0e-4) -> np.ndarray:
    """
    Numpy re-implementation of the WLS gradient used for offline p_grad.
    values: [N] or [N, C]; returns [N, C, 3] (C defaults to 1 for scalars).
    """
    if values.ndim == 1:
        values = values[:, None]
    N, C = values.shape
    src, dst = edge_index[0], edge_index[1]
    dx = pos[src] - pos[dst]                              # [E, 3]
    dv = values[src] - values[dst]                        # [E, C]
    r2 = (dx * dx).sum(axis=-1, keepdims=True).clip(1.0e-12)
    w = 1.0 / r2                                          # [E, 1]

    A = np.zeros((N, 3, 3), dtype=np.float64)
    B = np.zeros((N, C, 3), dtype=np.float64)
    # accumulate
    outer_xx = dx[:, :, None] * dx[:, None, :] * w[:, :, None]    # [E,3,3]
    np.add.at(A, dst, outer_xx)
    outer_vx = dv[:, :, None] * dx[:, None, :] * w[:, :, None]    # [E,C,3]
    np.add.at(B, dst, outer_vx)

    A_reg = A + eps_reg * np.eye(3)[None]
    # G  s.t.  A G^T = B^T  =>  solve per-node
    G_T = np.linalg.solve(A_reg, np.swapaxes(B, -1, -2))         # [N, 3, C]
    G = np.swapaxes(G_T, -1, -2)                                 # [N, C, 3]
    return G.astype(np.float32)


def _normalize(a: np.ndarray, minv=None, maxv=None) -> np.ndarray:
    mn = a.min() if minv is None else minv
    mx = a.max() if maxv is None else maxv
    if mx - mn < 1.0e-12:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - mn) / (mx - mn)).astype(np.float32)


def _build_node_features(pos, y, wall_mask):
    """
    Compute the 11-channel node feature matrix shared across all processing paths.

    Channel layout (MUST match the comment in the file header and augment.py):
      [0]  dist_inlet  — axial position along vessel (PCA axis)
      [1]  dist_wall   — normalised distance to nearest wall (absolute)
      [2]  r_norm      — d_wall / R_local: radial position within cross-section
                         (≈0 at wall, ≈1 at centreline).  Previously was a
                         duplicate of ch1 — now carries distinct information
                         for Poiseuille-profile learning.
      [3–5] flow_dir   — unit flow-direction vector from CFD solution
      [6]  inlet       — inlet indicator (reserved, currently 0)
      [7]  outlet      — outlet indicator (reserved, currently 0)
      [8]  inflow_norm — u_mean / u_char (shape of velocity distribution)
      [9]  u_char_norm — per-case peak velocity / 2 m/s (case-level speed scale).
                         Replaces Re_norm which was always 1.0 for aortic flows.
      [10] wall_flag   — MUST BE LAST (model uses x[:,-1] as wall mask)

    Returns: x [N,11], u_char (float), length_char (float)
    """
    from scipy.spatial import cKDTree

    N = len(pos)

    # --- ch0: axial position via PCA ---
    ax = pos - pos.mean(axis=0, keepdims=True)
    try:
        _, _, Vt = np.linalg.svd(ax, full_matrices=False)
        vessel_axis = Vt[0]
        axial_proj = ax @ vessel_axis
    except Exception:
        axial_proj = ax[:, 0]
    dist_inlet = _normalize(axial_proj - axial_proj.min())

    # --- ch1: wall distance ---
    w_pts = pos[wall_mask]
    if len(w_pts) >= 8:
        wt = cKDTree(w_pts)
        d_wall, _ = wt.query(pos, k=1)
    else:
        d_wall = np.zeros(N, dtype=np.float32)
    dist_wall = _normalize(d_wall)

    # --- ch2: normalised radial position (d_wall / local vessel radius) ---
    # Local vessel radius = max d_wall within the same axial cross-section slice.
    n_bins = 100
    axial_bin = np.floor(dist_inlet * (n_bins - 1)).astype(int).clip(0, n_bins - 1)
    R_local = np.zeros(N, dtype=np.float32)
    for b in range(n_bins):
        mask_b = axial_bin == b
        if mask_b.sum() > 0:
            R_local[mask_b] = float(d_wall[mask_b].max())
    r_norm = (d_wall / np.maximum(R_local, 1.0e-6)).clip(0.0, 1.0).astype(np.float32)

    # --- ch3–5: flow direction (unit vector from CFD solution) ---
    y_masked = y * (1.0 - wall_mask[:, None]).astype(np.float32)
    flow_dir = _unit(y_masked.astype(np.float32) + 1.0e-8)

    # --- scalars ---
    u_char      = float(np.quantile(np.linalg.norm(y, axis=-1), 0.99))
    u_mean      = float(np.mean(np.linalg.norm(y, axis=-1)))
    length_char = float(np.ptp(pos, axis=0).max())
    inflow_norm = float(min(u_mean / max(u_char, 1.0e-6), 1.0))
    u_char_norm = float(min(u_char / 2.0, 1.0))   # normalised by 2 m/s reference

    x = np.stack([
        dist_inlet,                            # 0
        dist_wall,                             # 1
        r_norm,                                # 2  NEW: radial position
        flow_dir[:, 0],                        # 3
        flow_dir[:, 1],                        # 4
        flow_dir[:, 2],                        # 5
        np.zeros(N, dtype=np.float32),         # 6  inlet (reserved)
        np.zeros(N, dtype=np.float32),         # 7  outlet (reserved)
        np.full(N, inflow_norm, dtype=np.float32),  # 8
        np.full(N, u_char_norm, dtype=np.float32),  # 9  NEW: per-case speed scale
        wall_mask.astype(np.float32),          # 10 MUST BE LAST
    ], axis=-1).astype(np.float32)

    return x, u_char, length_char


# ---------------------------------------------------------------------------
#  Per-case processing
# ---------------------------------------------------------------------------

def process_case(
    case_dir: str,
    out_path: str,
    *,
    velocity_field: str = "velocity",
    pressure_field: str = "pressure",
    compute_p_grad: bool = True,
    skip_if_exists: bool = True,
    _override_vtu: Optional[str] = None,
    _override_vtp: Optional[str] = None,
    _override_case_id: Optional[str] = None,
) -> Optional[Dict]:
    """
    Returns a small dict summarising the case, or None on failure.
    _override_vtu / _override_vtp bypass the automatic search when the
    caller already knows the exact paths (e.g. per-simulation mode).
    """
    pv = _import_pv()
    case_id = _override_case_id or os.path.basename(case_dir.rstrip("/"))
    if skip_if_exists and os.path.exists(out_path):
        return {"case_id": case_id, "path": out_path, "status": "exists"}

    vtu_path = _override_vtu or _find_vtu(case_dir)
    if vtu_path is None:
        print(f"[preprocess_vmr][{case_id}] no .vtu found — skipping")
        return None

    grid = pv.read(vtu_path)
    pts = np.asarray(grid.points, dtype=np.float64)         # [N, 3] in mm or m

    # VMR geometry is typically in cm; convert to metres if so.  We detect by
    # the maximum extent — aortic arch is ~0.1 m; if values look like cm
    # (max > 1), scale down by /100.  If they look like mm (>>10), divide by 1000.
    extent = float(np.max(np.ptp(pts, axis=0)))
    if extent > 1.0 and extent < 50.0:
        pts = pts * 0.01                                    # cm → m
    elif extent >= 50.0:
        pts = pts * 1.0e-3                                  # mm → m
    pos = pts.astype(np.float32)

    # velocity field
    if velocity_field not in grid.point_data:
        # try common alternates
        for alt in ("u", "U", "Velocity", "vel", "Vel"):
            if alt in grid.point_data:
                velocity_field = alt
                break
        else:
            print(f"[preprocess_vmr][{case_id}] no velocity field in .vtu")
            return None
    y = np.asarray(grid.point_data[velocity_field], dtype=np.float32)
    if y.ndim != 2 or y.shape[1] != 3:
        print(f"[preprocess_vmr][{case_id}] velocity has wrong shape {y.shape}")
        return None

    # pressure field (optional; used for p_grad)
    p = None
    if pressure_field in grid.point_data:
        p = np.asarray(grid.point_data[pressure_field], dtype=np.float32)
    else:
        for alt in ("p", "P", "Pressure", "press"):
            if alt in grid.point_data:
                p = np.asarray(grid.point_data[alt], dtype=np.float32)
                break

    # Edges
    edge_index = _build_edges_from_vtu(grid)               # [2, E]

    # Edge features: unit direction + norm + wall_edge flag (set later)
    src, dst = edge_index[0], edge_index[1]
    r_vec = pos[src] - pos[dst]                            # [E, 3]
    r_norm = np.linalg.norm(r_vec, axis=-1, keepdims=True)
    r_hat = r_vec / np.clip(r_norm, 1.0e-12, None)

    # Wall identification via VTP if provided
    wall_mask = np.zeros(len(pos), dtype=bool)
    wall_normal = np.zeros((len(pos), 3), dtype=np.float32)
    vtp_path = _override_vtp or _find_vtp(case_dir)
    if vtp_path is not None:
        surf = pv.read(vtp_path)
        # Map surface points to volume points by nearest neighbour
        from scipy.spatial import cKDTree
        vol_tree = cKDTree(pos)
        surf_pts = np.asarray(surf.points, dtype=np.float64)
        if extent > 1.0 and extent < 50.0:
            surf_pts = surf_pts * 0.01
        elif extent >= 50.0:
            surf_pts = surf_pts * 1.0e-3
        dists, idx = vol_tree.query(surf_pts, k=1)
        wall_mask[idx] = True
        # surface normals
        surf = surf.compute_normals(point_normals=True, cell_normals=False,
                                    auto_orient_normals=True)
        if "Normals" in surf.point_data:
            n_surf = np.asarray(surf.point_data["Normals"], dtype=np.float32)
            wall_normal[idx] = n_surf
    else:
        print(f"[preprocess_vmr][{case_id}][warn] no .vtp found — "
              f"falling back to velocity-based wall estimate")
        # cheap fallback: near-zero velocity + boundary of the mesh
        sp = np.linalg.norm(y, axis=-1)
        thr = np.quantile(sp, 0.05)
        wall_mask = sp <= thr

    # Wall-edge flag: edge between two wall nodes
    wall_edge = (wall_mask[src] & wall_mask[dst]).astype(np.float32)[:, None]

    edge_attr = np.concatenate([r_hat.astype(np.float32),
                                r_norm.astype(np.float32),
                                wall_edge], axis=-1)        # [E, 5]

    # Pressure gradient (WLS)
    p_grad = None
    if compute_p_grad and p is not None:
        # scalar pressure: build [N, 1, 3] then squeeze
        pg = _wls_grad_py(p.astype(np.float64), pos.astype(np.float64),
                          edge_index, eps_reg=1.0e-4)
        p_grad = pg[:, 0, :].astype(np.float32)

    # --- Node features -----------------------------------------------------
    x, u_char, length_char = _build_node_features(pos, y, wall_mask)

    kw = dict(
        x=x, pos=pos, y=y,
        edge_index=edge_index,
        edge_attr=edge_attr,
        wall_mask=wall_mask,
        wall_normal=wall_normal,
        u_char=np.float32(u_char),
        length_char=np.float32(length_char),
    )
    if p_grad is not None:
        kw["p_grad"] = p_grad

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **kw)

    return {
        "case_id": case_id,
        "path": out_path,
        "N": int(N),
        "E": int(edge_index.shape[1]),
        "u_char": float(u_char),
        "length_char": float(length_char),
        "has_p_grad": p_grad is not None,
    }


# ---------------------------------------------------------------------------
#  Split generation
# ---------------------------------------------------------------------------

def make_split(
    case_ids: List[str],
    out_file: str,
    *,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 1337,
) -> Dict[str, List[str]]:
    """
    Deterministic hash-based split — reproducible across machines, avoids
    random-state dependencies, and guarantees case-level isolation.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1.0e-6

    def _bucket(cid: str) -> float:
        h = hashlib.sha1(f"{seed}:{cid}".encode()).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    tr, va, te = [], [], []
    for cid in sorted(case_ids):
        b = _bucket(cid)
        if b < train_frac:
            tr.append(cid)
        elif b < train_frac + val_frac:
            va.append(cid)
        else:
            te.append(cid)

    splits = {"train": tr, "val": va, "test": te}
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(splits, f, indent=2)
    return splits


def make_split_append_new_to_train(
    case_ids: List[str],
    out_file: str,
) -> Dict[str, List[str]]:
    """
    Preserve an existing split.json and append previously unseen cases to train.

    This is useful when extending the public VMR collection while keeping the
    historical validation/test cases fixed for apples-to-apples comparisons.
    If out_file does not exist yet, all cases are assigned to train.
    """
    old = {"train": [], "val": [], "test": []}
    if os.path.exists(out_file):
        with open(out_file, "r") as f:
            loaded = json.load(f)
        for split in old:
            old[split] = list(loaded.get(split, []))

    current = set(case_ids)
    splits = {"train": [], "val": [], "test": []}
    assigned = set()
    for split in ("train", "val", "test"):
        for cid in old[split]:
            if cid in current and cid not in assigned:
                splits[split].append(cid)
                assigned.add(cid)

    for cid in case_ids:
        if cid not in assigned:
            splits["train"].append(cid)
            assigned.add(cid)

    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(splits, f, indent=2)
    return splits


def _load_existing_summary(out_dir: str) -> Dict[str, Dict]:
    summary_path = os.path.join(out_dir, "_summary.json")
    if not os.path.exists(summary_path):
        return {}
    try:
        with open(summary_path, "r") as f:
            rows = json.load(f)
    except Exception:
        return {}
    if not isinstance(rows, list):
        return {}
    return {
        str(row["case_id"]): dict(row)
        for row in rows
        if isinstance(row, dict) and row.get("case_id")
    }


def _summarize_existing_npz(
    out_path: str,
    case_id: str,
    previous: Optional[Dict] = None,
) -> Dict:
    if previous is not None:
        info = dict(previous)
        info["case_id"] = case_id
        info["path"] = out_path
        return info

    info = {"case_id": case_id, "path": out_path, "status": "exists"}
    try:
        z = np.load(out_path, allow_pickle=False)
        try:
            if "pos" in z.files:
                info["N"] = int(z["pos"].shape[0])
            if "edge_index" in z.files:
                edge_index = z["edge_index"]
                if edge_index.ndim == 2:
                    info["E"] = int(edge_index.shape[1])
                else:
                    info["E"] = int(edge_index.shape[0])
            if "u_char" in z.files:
                info["u_char"] = float(np.asarray(z["u_char"]))
            if "length_char" in z.files:
                info["length_char"] = float(np.asarray(z["length_char"]))
            info["has_p_grad"] = "p_grad" in z.files
        finally:
            z.close()
    except Exception as e:
        info["warning"] = f"could not inspect existing npz: {e}"
    return info


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def process_flat_vtu(
    vtu_path: str,
    out_path: str,
    *,
    skip_if_exists: bool = True,
) -> Optional[Dict]:
    """
    Process a single projected VTU file (from project_0d_to_3d.py) that already
    contains 'velocity' and 'pressure' point arrays.  Wall detection uses the
    'rad' field (rad > 0.95 → wall) if present, otherwise falls back to velocity.
    """
    pv = _import_pv()
    sample_id = os.path.splitext(os.path.basename(vtu_path))[0]
    if skip_if_exists and os.path.exists(out_path):
        return {"case_id": sample_id, "path": out_path, "status": "exists"}

    grid = pv.read(vtu_path)
    pts  = np.asarray(grid.points, dtype=np.float64)

    extent = float(np.max(np.ptp(pts, axis=0)))
    if extent > 1.0 and extent < 50.0:
        pts = pts * 0.01
    elif extent >= 50.0:
        pts = pts * 1.0e-3
    pos = pts.astype(np.float32)

    # velocity / pressure
    y = None
    for fn in ("velocity", "u", "U", "Velocity", "vel"):
        if fn in grid.point_data:
            y = np.asarray(grid.point_data[fn], dtype=np.float32)
            break
    if y is None or y.ndim != 2 or y.shape[1] != 3:
        print(f"[preprocess_vmr][{sample_id}] no valid velocity field")
        return None

    p = None
    for fn in ("pressure", "p", "P", "Pressure"):
        if fn in grid.point_data:
            p = np.asarray(grid.point_data[fn], dtype=np.float32)
            break

    edge_index = _build_edges_from_vtu(grid)
    src, dst   = edge_index[0], edge_index[1]
    r_vec      = pos[src] - pos[dst]
    r_norm     = np.linalg.norm(r_vec, axis=-1, keepdims=True)
    r_hat      = r_vec / np.clip(r_norm, 1e-12, None)

    # Wall detection: prefer 'rad' field, fall back to velocity magnitude
    N = len(pos)
    wall_mask   = np.zeros(N, dtype=bool)
    wall_normal = np.zeros((N, 3), dtype=np.float32)
    if "rad" in grid.point_data:
        rad = np.asarray(grid.point_data["rad"], dtype=np.float32)
        wall_mask = rad > 0.92
        # Approximate outward normal: direction from centroid to node, projected
        # perpendicular to flow direction (CenterlineSectionNormal)
        if "CenterlineSectionNormal" in grid.point_data:
            flow_n = np.asarray(grid.point_data["CenterlineSectionNormal"],
                                dtype=np.float32)
            centroid = np.mean(pos, axis=0, keepdims=True)
            radial   = pos - centroid
            # remove flow component
            dot = (radial * flow_n).sum(axis=-1, keepdims=True)
            radial_perp = radial - dot * flow_n
            norm = np.linalg.norm(radial_perp, axis=-1, keepdims=True)
            wall_normal = (radial_perp / np.clip(norm, 1e-12, None)).astype(np.float32)
    else:
        sp  = np.linalg.norm(y, axis=-1)
        thr = np.quantile(sp, 0.05)
        wall_mask = sp <= thr

    # Hard no-slip on GT labels: wall nodes must have zero velocity in the
    # ground truth (consistent with the model's hard_no_slip projection)
    y = y.copy()
    y[wall_mask] = 0.0

    wall_edge = (wall_mask[src] & wall_mask[dst]).astype(np.float32)[:, None]
    edge_attr  = np.concatenate([r_hat.astype(np.float32),
                                  r_norm.astype(np.float32),
                                  wall_edge], axis=-1)

    # p_grad via WLS
    p_grad = None
    if p is not None:
        pg = _wls_grad_py(p.astype(np.float64), pos.astype(np.float64),
                          edge_index, eps_reg=1e-4)
        p_grad = pg[:, 0, :].astype(np.float32)

    # Node features
    x, u_char, length_char = _build_node_features(pos, y, wall_mask)

    kw = dict(x=x, pos=pos, y=y, edge_index=edge_index, edge_attr=edge_attr,
              wall_mask=wall_mask, wall_normal=wall_normal,
              u_char=np.float32(u_char), length_char=np.float32(length_char))
    if p_grad is not None:
        kw["p_grad"] = p_grad

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, **kw)
    return {
        "case_id": sample_id, "path": out_path,
        "N": int(N), "E": int(edge_index.shape[1]),
        "u_char": float(u_char), "length_char": float(length_char),
        "has_p_grad": p_grad is not None,
    }


def _timed_suffix_key(name: str) -> int:
    import re as _re
    m = _re.search(r'_(\d+)$', name)
    return int(m.group(1)) if m else -1


def _fsi_multistep_keys(point_data) -> Tuple[List[str], List[str]]:
    import re as _re
    all_keys = list(point_data.keys())
    vel_keys = [
        k for k in all_keys
        if _re.match(r'^velocity_\d+$', k, flags=_re.IGNORECASE)
    ]
    pres_keys = [
        k for k in all_keys
        if _re.match(r'^pressure_\d+$', k, flags=_re.IGNORECASE)
    ]
    vel_keys.sort(key=_timed_suffix_key)
    pres_keys.sort(key=_timed_suffix_key)
    return vel_keys, pres_keys


def _first_present_key(point_data, candidates: Tuple[str, ...]) -> Optional[str]:
    for key in candidates:
        if key in point_data:
            return key
    return None


def _derive_fsi_case_id(vtu_path: str) -> str:
    import re as _re
    stem = os.path.splitext(os.path.basename(vtu_path))[0]
    parts = vtu_path.split(os.sep)

    # Nested vascularmodel exports often include a directory named like
    # 0246_H_AO_AOD_3D_FSI_REST_VTU; use that over generic file names such
    # as Orig_Stiffness.vtu.
    for part in reversed(parts[:-1]):
        if _re.match(r'^\d{4}_H_.*_3D_.*_VTU$', part):
            return part[:-4]

    if _re.match(r'^result_\d+$', stem):
        parent = os.path.basename(os.path.dirname(vtu_path))
        return f"{parent}_3D_FSI"

    if stem == "Orig_Stiffness":
        for part in reversed(parts[:-1]):
            if _re.match(r'^\d{4}_H_', part):
                return f"{part}_3D_FSI_REST"

    return stem


def _is_result_vtu(vtu_path: str) -> bool:
    import re as _re
    stem = os.path.splitext(os.path.basename(vtu_path))[0]
    return _re.match(r'^result_\d+$', stem) is not None


def _is_noncanonical_stiffness_variant(vtu_path: str) -> bool:
    stem = os.path.splitext(os.path.basename(vtu_path))[0]
    return stem in {"10_Perc_Stiffness", "50_Perc_Stiffness"}


def process_fsi_multistep_vtu(
    vtu_path: str,
    out_path: str,
    *,
    case_id: Optional[str] = None,
    skip_if_exists: bool = True,
    existing_info: Optional[Dict] = None,
) -> Optional[Dict]:
    """
    Process a time-resolved FSI VTU from vascularmodel.com.

    The file stores one `velocity_NNNNN` / `pressure_NNNNN` array per saved
    timestep.  This function:
      1. Time-averages all timesteps (assumes they span ≥1 converged cardiac cycle).
      2. Converts cm/s → m/s and dyne/cm² → Pa.
      3. Identifies wall nodes via surface extraction (vtkOriginalPointIds);
         time-averaged FSI wall velocity ≈ 0 over a complete cycle.
      4. Saves an NPZ in the same schema as process_flat_vtu / process_case.
    """
    import re as _re
    pv = _import_pv()

    if case_id is None:
        case_id = os.path.splitext(os.path.basename(vtu_path))[0]
    if skip_if_exists and os.path.exists(out_path):
        return _summarize_existing_npz(out_path, case_id, existing_info)

    print(f"[fsi_multistep][{case_id}] reading {os.path.basename(vtu_path)} ...")
    grid = pv.read(vtu_path)
    N = grid.n_points

    pts = np.asarray(grid.points, dtype=np.float64)
    extent = float(np.max(np.ptp(pts, axis=0)))
    if extent > 1.0 and extent < 50.0:
        pts = pts * 0.01
    elif extent >= 50.0:
        pts = pts * 1.0e-3
    pos = pts.astype(np.float32)

    # Discover timestep arrays
    vel_keys, pres_keys = _fsi_multistep_keys(grid.point_data)
    if not vel_keys:
        print(f"[fsi_multistep][{case_id}] no velocity_NNNNN arrays found")
        return None
    n_steps = len(vel_keys)
    print(f"  {n_steps} velocity + {len(pres_keys)} pressure timesteps")

    # Incremental time-average (avoids loading all steps at once)
    vel_sum = np.zeros((N, 3), dtype=np.float64)
    for k in vel_keys:
        vel_sum += np.asarray(grid.point_data[k], dtype=np.float64)
    y = (vel_sum / n_steps).astype(np.float32) / 100.0   # cm/s → m/s

    p = None
    if pres_keys:
        pres_sum = np.zeros(N, dtype=np.float64)
        for k in pres_keys:
            pres_sum += np.asarray(grid.point_data[k], dtype=np.float64)
        p = ((pres_sum / len(pres_keys)) * 0.1).astype(np.float32)  # dyne/cm² → Pa

    # Wall detection: surface extraction gives boundary node IDs.
    # In FSI, time-averaged wall velocity ≈ 0 (net wall displacement = 0 per cycle).
    wall_mask   = np.zeros(N, dtype=bool)
    wall_normal = np.zeros((N, 3), dtype=np.float32)
    try:
        surf = grid.extract_surface(algorithm="dataset_surface")
        orig_ids = surf.point_data.get("vtkOriginalPointIds")
        if orig_ids is not None:
            wall_mask[np.unique(np.asarray(orig_ids))] = True
        surf = surf.compute_normals(point_normals=True, cell_normals=False,
                                    auto_orient_normals=True)
        if "Normals" in surf.point_data and orig_ids is not None:
            uniq, inv = np.unique(np.asarray(orig_ids), return_index=True)
            wall_normal[uniq] = np.asarray(surf.point_data["Normals"],
                                           dtype=np.float32)[inv]
        print(f"  wall nodes (surface): {wall_mask.sum()} / {N} "
              f"({100*wall_mask.mean():.1f}%)")
    except Exception as e:
        print(f"  [warn] surface extraction failed ({e}); falling back to vmag threshold")
        vmag = np.linalg.norm(y, axis=-1)
        thr  = float(np.quantile(vmag, 0.15))
        wall_mask = vmag <= thr

    # Hard no-slip on GT: time-averaged FSI wall motion should already be ~0;
    # enforce explicitly for consistency with model's hard_no_slip projection.
    y = y.copy()
    y[wall_mask] = 0.0

    # Edges and edge attributes
    edge_index = _build_edges_from_vtu(grid)
    src, dst   = edge_index[0], edge_index[1]
    r_vec  = pos[src] - pos[dst]
    r_norm = np.linalg.norm(r_vec, axis=-1, keepdims=True)
    r_hat  = r_vec / np.clip(r_norm, 1e-12, None)
    wall_edge = (wall_mask[src] & wall_mask[dst]).astype(np.float32)[:, None]
    edge_attr = np.concatenate([r_hat.astype(np.float32),
                                 r_norm.astype(np.float32),
                                 wall_edge], axis=-1)

    # Pressure gradient via WLS
    p_grad = None
    if p is not None:
        pg     = _wls_grad_py(p.astype(np.float64), pos.astype(np.float64),
                              edge_index, eps_reg=1e-4)
        p_grad = pg[:, 0, :].astype(np.float32)

    # Node features
    x, u_char, length_char = _build_node_features(pos, y, wall_mask)

    kw = dict(x=x, pos=pos, y=y, edge_index=edge_index, edge_attr=edge_attr,
              wall_mask=wall_mask, wall_normal=wall_normal,
              u_char=np.float32(u_char), length_char=np.float32(length_char))
    if p_grad is not None:
        kw["p_grad"] = p_grad

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, **kw)
    return {
        "case_id": case_id, "path": out_path,
        "N": int(N), "E": int(edge_index.shape[1]),
        "u_char": float(u_char), "length_char": float(length_char),
        "has_p_grad": p_grad is not None,
        "n_timesteps": n_steps,
    }


def process_fsi_series_vtu(
    vtu_paths: List[str],
    out_path: str,
    *,
    case_id: Optional[str] = None,
    skip_if_exists: bool = True,
    existing_info: Optional[Dict] = None,
) -> Optional[Dict]:
    """
    Process vascularmodel.com FSI exports stored as one VTU per saved timestep
    (`result_NNNNN.vtu`).  The per-file fields are usually `Velocity` and
    `Pressure`; this function time-averages the series into one graph, matching
    process_fsi_multistep_vtu's output schema.
    """
    pv = _import_pv()

    vtu_paths = sorted(vtu_paths)
    if not vtu_paths:
        return None
    if case_id is None:
        case_id = os.path.basename(os.path.dirname(vtu_paths[0]))
    if skip_if_exists and os.path.exists(out_path):
        return _summarize_existing_npz(out_path, case_id, existing_info)

    print(f"[fsi_series][{case_id}] reading {len(vtu_paths)} result_*.vtu files ...")
    grid0 = pv.read(vtu_paths[0])
    N = grid0.n_points

    pts = np.asarray(grid0.points, dtype=np.float64)
    extent = float(np.max(np.ptp(pts, axis=0)))
    if extent > 1.0 and extent < 50.0:
        pts = pts * 0.01
    elif extent >= 50.0:
        pts = pts * 1.0e-3
    pos = pts.astype(np.float32)

    vel_key0 = _first_present_key(grid0.point_data, ("Velocity", "velocity", "U", "u"))
    if vel_key0 is None:
        print(f"[fsi_series][{case_id}] no Velocity field found in first VTU")
        return None
    pres_key0 = _first_present_key(grid0.point_data, ("Pressure", "pressure", "P", "p"))

    vel_sum = np.zeros((N, 3), dtype=np.float64)
    pres_sum = np.zeros(N, dtype=np.float64) if pres_key0 is not None else None
    n_pres = 0

    for idx, path in enumerate(vtu_paths):
        grid = grid0 if idx == 0 else pv.read(path)
        if grid.n_points != N:
            raise ValueError(
                f"{case_id}: timestep {path} has {grid.n_points} points, expected {N}"
            )
        vel_key = _first_present_key(grid.point_data, ("Velocity", "velocity", "U", "u"))
        if vel_key is None:
            raise ValueError(f"{case_id}: no Velocity field in {path}")
        vel_sum += np.asarray(grid.point_data[vel_key], dtype=np.float64)

        pres_key = _first_present_key(grid.point_data, ("Pressure", "pressure", "P", "p"))
        if pres_sum is not None and pres_key is not None:
            pres_sum += np.asarray(grid.point_data[pres_key], dtype=np.float64)
            n_pres += 1

    n_steps = len(vtu_paths)
    y = (vel_sum / n_steps).astype(np.float32) / 100.0  # cm/s -> m/s
    p = None
    if pres_sum is not None and n_pres > 0:
        p = ((pres_sum / n_pres) * 0.1).astype(np.float32)  # dyne/cm^2 -> Pa

    wall_mask = np.zeros(N, dtype=bool)
    wall_normal = np.zeros((N, 3), dtype=np.float32)
    try:
        surf = grid0.extract_surface(algorithm="dataset_surface")
        orig_ids = surf.point_data.get("vtkOriginalPointIds")
        if orig_ids is not None:
            wall_mask[np.unique(np.asarray(orig_ids))] = True
        surf = surf.compute_normals(point_normals=True, cell_normals=False,
                                    auto_orient_normals=True)
        if "Normals" in surf.point_data and orig_ids is not None:
            uniq, inv = np.unique(np.asarray(orig_ids), return_index=True)
            wall_normal[uniq] = np.asarray(surf.point_data["Normals"],
                                           dtype=np.float32)[inv]
        print(f"  wall nodes (surface): {wall_mask.sum()} / {N} "
              f"({100*wall_mask.mean():.1f}%)")
    except Exception as e:
        print(f"  [warn] surface extraction failed ({e}); falling back to vmag threshold")
        vmag = np.linalg.norm(y, axis=-1)
        thr = float(np.quantile(vmag, 0.15))
        wall_mask = vmag <= thr

    y = y.copy()
    y[wall_mask] = 0.0

    edge_index = _build_edges_from_vtu(grid0)
    src, dst = edge_index[0], edge_index[1]
    r_vec = pos[src] - pos[dst]
    r_norm = np.linalg.norm(r_vec, axis=-1, keepdims=True)
    r_hat = r_vec / np.clip(r_norm, 1e-12, None)
    wall_edge = (wall_mask[src] & wall_mask[dst]).astype(np.float32)[:, None]
    edge_attr = np.concatenate([r_hat.astype(np.float32),
                                r_norm.astype(np.float32),
                                wall_edge], axis=-1)

    p_grad = None
    if p is not None:
        pg = _wls_grad_py(p.astype(np.float64), pos.astype(np.float64),
                          edge_index, eps_reg=1e-4)
        p_grad = pg[:, 0, :].astype(np.float32)

    x, u_char, length_char = _build_node_features(pos, y, wall_mask)

    kw = dict(x=x, pos=pos, y=y, edge_index=edge_index, edge_attr=edge_attr,
              wall_mask=wall_mask, wall_normal=wall_normal,
              u_char=np.float32(u_char), length_char=np.float32(length_char))
    if p_grad is not None:
        kw["p_grad"] = p_grad

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, **kw)
    return {
        "case_id": case_id, "path": out_path,
        "N": int(N), "E": int(edge_index.shape[1]),
        "u_char": float(u_char), "length_char": float(length_char),
        "has_p_grad": p_grad is not None,
        "n_timesteps": n_steps,
        "source": "result_vtu_series",
    }


def process_fsi_peaksystolic_vtu(
    vtu_path: str,
    out_path: str,
    *,
    case_id: Optional[str] = None,
    skip_if_exists: bool = True,
    existing_info: Optional[Dict] = None,
) -> Optional[Dict]:
    """
    Peak-systolic variant of process_fsi_multistep_vtu.

    Instead of time-averaging all velocity_NNNNN steps, pick the single
    timestep whose 99th-percentile velocity magnitude is maximal — i.e. the
    systolic peak frame.  Pressure is taken from the SAME timestep index
    (matched by trailing digit suffix).

    Output schema is identical to process_fsi_multistep_vtu so any downstream
    code (VMRGraphDataset, train_v*, evaluate.py) works unchanged.  u_char is
    naturally larger here (peak speed instead of cycle-average).
    """
    import re as _re
    pv = _import_pv()

    if case_id is None:
        case_id = os.path.splitext(os.path.basename(vtu_path))[0]
    if skip_if_exists and os.path.exists(out_path):
        return _summarize_existing_npz(out_path, case_id, existing_info)

    print(f"[peaksys][{case_id}] reading {os.path.basename(vtu_path)} ...")
    grid = pv.read(vtu_path)
    N = grid.n_points

    pts = np.asarray(grid.points, dtype=np.float64)
    extent = float(np.max(np.ptp(pts, axis=0)))
    if extent > 1.0 and extent < 50.0:
        pts = pts * 0.01
    elif extent >= 50.0:
        pts = pts * 1.0e-3
    pos = pts.astype(np.float32)

    vel_keys, pres_keys = _fsi_multistep_keys(grid.point_data)
    if not vel_keys:
        print(f"[peaksys][{case_id}] no velocity_NNNNN arrays found")
        return None
    n_steps = len(vel_keys)

    best_q = -1.0
    best_key = None
    best_v = None
    for k in vel_keys:
        v_cm_s = np.asarray(grid.point_data[k], dtype=np.float64)
        speed_cm_s = np.linalg.norm(v_cm_s, axis=-1)
        q99 = float(np.quantile(speed_cm_s, 0.99))
        if q99 > best_q:
            best_q = q99
            best_key = k
            best_v = v_cm_s
    print(f"  picked {best_key} of {n_steps} (||v||_99={best_q/100.0:.3f} m/s peak)")

    y = (best_v.astype(np.float32) / 100.0)  # cm/s → m/s

    p = None
    if pres_keys:
        suffix = best_key.split("_")[-1]
        match_p = [k for k in pres_keys if k.endswith(f"_{suffix}")]
        pres_key = match_p[0] if match_p else pres_keys[len(pres_keys) // 2]
        p = (np.asarray(grid.point_data[pres_key], dtype=np.float64) * 0.1).astype(np.float32)

    wall_mask = np.zeros(N, dtype=bool)
    wall_normal = np.zeros((N, 3), dtype=np.float32)
    try:
        surf = grid.extract_surface(algorithm="dataset_surface")
        orig_ids = surf.point_data.get("vtkOriginalPointIds")
        if orig_ids is not None:
            wall_mask[np.unique(np.asarray(orig_ids))] = True
        surf = surf.compute_normals(point_normals=True, cell_normals=False,
                                    auto_orient_normals=True)
        if "Normals" in surf.point_data and orig_ids is not None:
            uniq, inv = np.unique(np.asarray(orig_ids), return_index=True)
            wall_normal[uniq] = np.asarray(surf.point_data["Normals"],
                                           dtype=np.float32)[inv]
        print(f"  wall nodes (surface): {wall_mask.sum()} / {N} "
              f"({100*wall_mask.mean():.1f}%)")
    except Exception as e:
        print(f"  [warn] surface extraction failed ({e}); falling back to vmag threshold")
        vmag = np.linalg.norm(y, axis=-1)
        thr = float(np.quantile(vmag, 0.15))
        wall_mask = vmag <= thr

    y = y.copy()
    y[wall_mask] = 0.0

    edge_index = _build_edges_from_vtu(grid)
    src, dst = edge_index[0], edge_index[1]
    r_vec = pos[src] - pos[dst]
    r_norm = np.linalg.norm(r_vec, axis=-1, keepdims=True)
    r_hat = r_vec / np.clip(r_norm, 1e-12, None)
    wall_edge = (wall_mask[src] & wall_mask[dst]).astype(np.float32)[:, None]
    edge_attr = np.concatenate([r_hat.astype(np.float32),
                                r_norm.astype(np.float32),
                                wall_edge], axis=-1)

    p_grad = None
    if p is not None:
        pg = _wls_grad_py(p.astype(np.float64), pos.astype(np.float64),
                          edge_index, eps_reg=1e-4)
        p_grad = pg[:, 0, :].astype(np.float32)

    x, u_char, length_char = _build_node_features(pos, y, wall_mask)

    kw = dict(x=x, pos=pos, y=y, edge_index=edge_index, edge_attr=edge_attr,
              wall_mask=wall_mask, wall_normal=wall_normal,
              u_char=np.float32(u_char), length_char=np.float32(length_char))
    if p_grad is not None:
        kw["p_grad"] = p_grad

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, **kw)
    return {
        "case_id": case_id, "path": out_path,
        "N": int(N), "E": int(edge_index.shape[1]),
        "u_char": float(u_char), "length_char": float(length_char),
        "has_p_grad": p_grad is not None,
        "n_timesteps": n_steps,
        "peak_step": best_key,
        "peak_q99_m_s": best_q / 100.0,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Preprocess VMR vascular cases into .npz graph snapshots.")
    ap.add_argument("--vmr-root", default=None,
                    help="Directory containing one sub-directory per case.")
    ap.add_argument("--flat-vtu-dir", default=None,
                    help="Directory of projected VTU files from project_0d_to_3d.py "
                         "(each file = one sample; use instead of --vmr-root).")
    ap.add_argument("--fsi-vtu", default=None, nargs="+", metavar="VTU",
                    help="One or more time-resolved FSI VTU files from vascularmodel.com. "
                         "Each file is time-averaged and saved as one NPZ.")
    ap.add_argument("--peak-systolic", action="store_true", default=False,
                    help="With --fsi-vtu: instead of time-averaging, pick the "
                         "single timestep with max ||v||_99 (peak-systolic frame). "
                         "u_char becomes peak speed → larger absolute targets, "
                         "potentially better PP@10 ceiling for CoA jets.")
    ap.add_argument("--out-dir", required=True,
                    help="Where to write .npz files.")
    ap.add_argument("--only-aorta", action="store_true",
                    help="Filter to case IDs that contain 'aorta' in their name.")
    ap.add_argument("--split-file", default=None,
                    help="If given, also write a split.json with the same seed.")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac",   type=float, default=0.15)
    ap.add_argument("--test-frac",  type=float, default=0.15)
    ap.add_argument("--split-seed", type=int, default=1337)
    ap.add_argument("--append-new-to-train-split", action="store_true", default=False,
                    help="If split-file already exists, preserve current train/val/test "
                         "assignments and append unseen case IDs to train.")
    ap.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", "--force", dest="skip_existing", action="store_false",
                    help="Reprocess even if output NPZ already exists.")
    ap.add_argument("--per-simulation", action="store_true", default=False,
                    help="For SimVascular VMR cases: produce one .npz per "
                         "Simulations/* condition (e.g. LOOP_REST, LOOP_EXER_*).")
    args = ap.parse_args()

    if args.flat_vtu_dir is not None and args.vmr_root is None:
        # ----------------------------------------------------------------
        #  Flat-VTU mode: process projected VTU files directly
        # ----------------------------------------------------------------
        import glob as _glob
        vtu_files = sorted(_glob.glob(os.path.join(args.flat_vtu_dir, "*.vtu")))
        print(f"Flat-VTU mode: found {len(vtu_files)} VTU files in {args.flat_vtu_dir}")
        os.makedirs(args.out_dir, exist_ok=True)

        summary = []
        for vtu_path in vtu_files:
            sample_id = os.path.splitext(os.path.basename(vtu_path))[0]
            out_path  = os.path.join(args.out_dir, f"{sample_id}.npz")
            try:
                info = process_flat_vtu(vtu_path, out_path,
                                        skip_if_exists=args.skip_existing)
                if info is not None:
                    summary.append(info)
                    if isinstance(info.get("N"), int):
                        print(f"  [OK] {sample_id:<55s} "
                              f"N={info['N']:>7d}  "
                              f"u_char={info.get('u_char', float('nan')):.3f} m/s")
                    else:
                        print(f"  [OK] {sample_id} (exists)")
            except Exception as e:
                import traceback
                print(f"  [FAIL] {sample_id}: {e}")
                traceback.print_exc()

        ids_ok = [s["case_id"] for s in summary if "N" in s or "path" in s]
        if args.split_file is not None and ids_ok:
            # For flat VTU: split on patient ID (prefix before '__')
            patient_ids = sorted({cid.split("__")[0] for cid in ids_ok})
            # Assign each sample to same split as its patient
            import hashlib as _hs
            def _bucket(pid: str) -> float:
                h = _hs.sha1(f"{args.split_seed}:{pid}".encode()).hexdigest()
                return int(h[:8], 16) / 0xFFFFFFFF
            patient_split = {}
            for pid in patient_ids:
                b = _bucket(pid)
                if b < args.train_frac:
                    patient_split[pid] = "train"
                elif b < args.train_frac + args.val_frac:
                    patient_split[pid] = "val"
                else:
                    patient_split[pid] = "test"
            splits: dict = {"train": [], "val": [], "test": []}
            for cid in ids_ok:
                pid = cid.split("__")[0]
                splits[patient_split.get(pid, "test")].append(cid)
            os.makedirs(os.path.dirname(args.split_file) or ".", exist_ok=True)
            with open(args.split_file, "w") as f:
                json.dump(splits, f, indent=2)
            print(f"Split: train={len(splits['train'])}  val={len(splits['val'])}  "
                  f"test={len(splits['test'])}  → {args.split_file}")

        with open(os.path.join(args.out_dir, "_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=float)
        return   # <-- done with flat-VTU mode

    if args.fsi_vtu is not None:
        # ----------------------------------------------------------------
        #  FSI multi-timestep mode: process one or more FSI VTU files
        # ----------------------------------------------------------------
        os.makedirs(args.out_dir, exist_ok=True)
        summary = []
        previous_summary = _load_existing_summary(args.out_dir)
        proc_fn = (process_fsi_peaksystolic_vtu if args.peak_systolic
                   else process_fsi_multistep_vtu)
        if args.peak_systolic:
            print("[mode] PEAK-SYSTOLIC: each VTU → single peak-systolic snapshot")

        series_groups: Dict[str, List[str]] = {}
        single_vtu: List[str] = []
        skipped_variants: List[str] = []
        for vtu_path in sorted(args.fsi_vtu):
            if _is_noncanonical_stiffness_variant(vtu_path):
                skipped_variants.append(vtu_path)
                continue
            if _is_result_vtu(vtu_path):
                cid = _derive_fsi_case_id(vtu_path)
                series_groups.setdefault(cid, []).append(vtu_path)
            else:
                single_vtu.append(vtu_path)

        if skipped_variants:
            print("[mode] skipping noncanonical stiffness variants: "
                  f"{len(skipped_variants)} VTU files")

        for cid, vtu_paths in sorted(series_groups.items()):
            out_path = os.path.join(args.out_dir, f"{cid}.npz")
            try:
                info = process_fsi_series_vtu(vtu_paths, out_path,
                                              case_id=cid,
                                              skip_if_exists=args.skip_existing,
                                              existing_info=previous_summary.get(cid))
                if info is not None:
                    summary.append(info)
                    if isinstance(info.get("N"), int):
                        print(f"  [OK] {cid:<60s} N={info['N']:>7d}  "
                              f"E={info['E']:>9d}  u_char={info.get('u_char', float('nan')):.3f} m/s  "
                              f"steps={info.get('n_timesteps', '?')}")
                    else:
                        print(f"  [OK] {cid} (exists)")
            except Exception as e:
                import traceback
                print(f"  [FAIL] {cid}: {e}")
                traceback.print_exc()

        for vtu_path in single_vtu:
            cid = _derive_fsi_case_id(vtu_path)
            out_path = os.path.join(args.out_dir, f"{cid}.npz")
            try:
                info = proc_fn(vtu_path, out_path,
                               case_id=cid,
                               skip_if_exists=args.skip_existing,
                               existing_info=previous_summary.get(cid))
                if info is not None:
                    summary.append(info)
                    if isinstance(info.get("N"), int):
                        print(f"  [OK] {cid:<60s} N={info['N']:>7d}  "
                              f"E={info['E']:>9d}  u_char={info.get('u_char', float('nan')):.3f} m/s  "
                              f"steps={info.get('n_timesteps', '?')}")
                    else:
                        print(f"  [OK] {cid} (exists)")
            except Exception as e:
                import traceback
                print(f"  [FAIL] {cid}: {e}")
                traceback.print_exc()

        ids_ok = [s["case_id"] for s in summary if "N" in s or "path" in s]
        if args.split_file is not None and ids_ok:
            if args.append_new_to_train_split:
                splits = make_split_append_new_to_train(ids_ok, args.split_file)
            else:
                import hashlib as _hs
                def _bucket(pid: str) -> float:
                    h = _hs.sha1(f"{args.split_seed}:{pid}".encode()).hexdigest()
                    return int(h[:8], 16) / 0xFFFFFFFF
                splits: dict = {"train": [], "val": [], "test": []}
                for cid in ids_ok:
                    b = _bucket(cid)
                    if b < args.train_frac:
                        splits["train"].append(cid)
                    elif b < args.train_frac + args.val_frac:
                        splits["val"].append(cid)
                    else:
                        splits["test"].append(cid)
                os.makedirs(os.path.dirname(args.split_file) or ".", exist_ok=True)
                with open(args.split_file, "w") as f:
                    json.dump(splits, f, indent=2)
            print(f"Split: train={len(splits['train'])}  val={len(splits['val'])}  "
                  f"test={len(splits['test'])}  → {args.split_file}")

        with open(os.path.join(args.out_dir, "_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=float)
        return   # <-- done with fsi-vtu mode

    if args.vmr_root is None:
        print("ERROR: provide --vmr-root, --flat-vtu-dir, or --fsi-vtu", file=sys.stderr)
        sys.exit(1)

    # enumerate cases
    case_dirs = [
        d for d in sorted(os.listdir(args.vmr_root))
        if os.path.isdir(os.path.join(args.vmr_root, d))
    ]
    if args.only_aorta:
        case_dirs = [d for d in case_dirs if "aorta" in d.lower()]

    print(f"Found {len(case_dirs)} candidate cases.")
    os.makedirs(args.out_dir, exist_ok=True)

    summary = []
    for cid in case_dirs:
        case_dir = os.path.join(args.vmr_root, cid)

        if args.per_simulation:
            # One .npz per simulation condition (SimVascular structure)
            sim_entries = _find_all_sim_vtu(case_dir)
            if not sim_entries:
                # Maybe the case_dir itself is the VMR case (double-nested)
                inner = os.path.join(case_dir, cid)
                if os.path.isdir(inner):
                    sim_entries = _find_all_sim_vtu(inner)
            if not sim_entries:
                print(f"  [SKIP] {cid}: no Simulations/* VTU with velocity fields found")
                print(f"         (Raw VMR data needs svpost post-processing to produce "
                      f"VTU files with velocity/pressure fields)")
                continue
            for sim_name, vtu_path, vtp_path in sim_entries:
                sample_id = f"{cid}__{sim_name}"
                out_path = os.path.join(args.out_dir, f"{sample_id}.npz")
                try:
                    info = process_case(
                        case_dir=os.path.dirname(os.path.dirname(os.path.dirname(vtu_path))),
                        out_path=out_path,
                        skip_if_exists=args.skip_existing,
                        _override_vtu=vtu_path,
                        _override_vtp=vtp_path,
                        _override_case_id=sample_id,
                    )
                    if info is not None:
                        summary.append(info)
                        if isinstance(info.get("N"), int):
                            print(f"  [OK] {sample_id:<50s} "
                                  f"N={info['N']:>6d}  E={info['E']:>9d}  "
                                  f"u_char={info.get('u_char', float('nan')):.3f} m/s")
                        else:
                            print(f"  [OK] {sample_id} (exists)")
                except Exception as e:
                    print(f"  [FAIL] {sample_id}: {e}")
        else:
            out_path = os.path.join(args.out_dir, f"{cid}.npz")
            try:
                info = process_case(case_dir, out_path,
                                    skip_if_exists=args.skip_existing)
                if info is not None:
                    summary.append(info)
                    if isinstance(info.get("N"), int):
                        print(f"  [OK] {cid:<50s} "
                              f"N={info['N']:>6d}  E={info['E']:>9d}  "
                              f"u_char={info.get('u_char', float('nan')):.3f} m/s")
                    else:
                        print(f"  [OK] {cid} (exists)")
            except Exception as e:
                print(f"  [FAIL] {cid}: {e}")

    # Split
    ids_ok = [s["case_id"] for s in summary if "N" in s or "path" in s]
    if args.split_file is not None and ids_ok:
        splits = make_split(
            ids_ok, args.split_file,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.split_seed,
        )
        print(f"Wrote split: "
              f"train={len(splits['train'])}, "
              f"val={len(splits['val'])}, "
              f"test={len(splits['test'])}  →  {args.split_file}")

    # Summary json
    with open(os.path.join(args.out_dir, "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)


if __name__ == "__main__":
    main()
