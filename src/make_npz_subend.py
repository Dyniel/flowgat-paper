# -*- coding: utf-8 -*-
"""
Adapter: Suk-et-al. sUbend VTK frames -> project NPZ schema.

Input layout (after extract_subend.py finishes):
  data/external/newCFD_dataset/sUbend_<NNN>/CFD/Frame_<FF>.vtk     (25 frames each)

Mesh: VTK 5.1 UNSTRUCTURED_GRID, all VTK_HEXAHEDRON, ~1.13M nodes, ~1.10M cells.
Point data: `U` (N,3 double), `p` (N,) double. No wall_mask in source.

Output: data/npz_subend/<case>__f<FF>.npz with the same 11-column feature schema
as data/npz_womersley/ (noleak baseline). Wall mask is derived per-case as
the subset of surface nodes whose max |U| across the 25 frames stays below
WALL_SPEED_THRESHOLD; inlet/outlet planes therefore drop out automatically.

By default the mesh is stratified-subsampled to --max_nodes (50k by default),
keeping all wall nodes and randomly sampling interior nodes. The 11-column
feature matrix is then computed on the subsampled mesh and a kNN graph is
built in 3D (mirrors src/make_npz_womersley.py exactly).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Defaults — tuned for sUbend dataset
# --------------------------------------------------------------------------- #
WALL_SPEED_THRESHOLD = 0.05   # absolute fallback (m/s); see derive_wall_mask
DEFAULT_WALL_QUANTILE = 0.70  # surface nodes below this quantile of mean|U| are wall
DEFAULT_MAX_NODES = 50_000
DEFAULT_KNN = 16


# --------------------------------------------------------------------------- #
# Mesh helpers
# --------------------------------------------------------------------------- #
def knn_edges(pos: np.ndarray, k: int) -> np.ndarray:
    from scipy.spatial import cKDTree
    tree = cKDTree(pos)
    _, idx = tree.query(pos, k=k + 1)
    src = np.repeat(np.arange(pos.shape[0]), k)
    dst = idx[:, 1:].reshape(-1)
    return np.stack([src, dst], axis=0).astype(np.int64)


def extract_surface_ids(mesh) -> np.ndarray:
    """Return original-point-ids of surface (boundary) nodes."""
    import pyvista as pv
    m = mesh.copy()
    m["vtkOriginalPointIds"] = np.arange(m.n_points, dtype=np.int64)
    surf = m.extract_surface(algorithm="dataset_surface", pass_pointid=True)
    if "vtkOriginalPointIds" not in surf.point_data:
        raise RuntimeError("extract_surface did not return original point ids")
    return np.asarray(surf.point_data["vtkOriginalPointIds"], dtype=np.int64)


def read_frame(path: Path):
    import pyvista as pv
    return pv.read(str(path))


def list_frames(case_dir: Path) -> List[Path]:
    pat = re.compile(r"Frame_(\d+)\.vtk$")
    found = []
    for p in sorted(case_dir.glob("CFD/Frame_*.vtk")):
        m = pat.search(p.name)
        if m:
            found.append((int(m.group(1)), p))
    found.sort()
    return [p for _, p in found]


def derive_wall_mask(case_dir: Path, max_speed_threshold: float = WALL_SPEED_THRESHOLD,
                     wall_quantile: float = DEFAULT_WALL_QUANTILE
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pos, surface_ids, wall_mask_full).

    Wall detection on the boundary surface uses *cycle-mean* |U|:
      wall = surface_id where mean|U| < min(threshold, quantile cutoff).

    The quantile cutoff is the cycle-mean speed at the `wall_quantile`-th
    percentile of surface nodes; on a U-bend pipe the inlet/outlet planes
    sit in the high tail of surface speeds, so the bottom ~70% are wall.

    Loads only the first frame for pos+surface, then re-reads velocity per
    frame (memory-efficient).
    """
    frames = list_frames(case_dir)
    if not frames:
        raise FileNotFoundError(f"no frames under {case_dir}/CFD")
    first = read_frame(frames[0])
    pos = np.asarray(first.points, dtype=np.float32)
    surface_ids = extract_surface_ids(first)
    n = pos.shape[0]
    sum_speed = np.zeros(n, dtype=np.float64)

    for f in frames:
        m = read_frame(f)
        U = np.asarray(m.point_data["U"], dtype=np.float32)
        sum_speed += np.linalg.norm(U, axis=1).astype(np.float64)
    mean_speed = (sum_speed / len(frames)).astype(np.float32)

    surf_mean = mean_speed[surface_ids]
    q_cutoff = float(np.quantile(surf_mean, wall_quantile))
    cutoff = float(min(max_speed_threshold, q_cutoff))

    wall_mask_full = np.zeros(n, dtype=bool)
    wall_mask_full[surface_ids] = surf_mean < cutoff
    return pos, surface_ids, wall_mask_full


def stratified_subsample(pos: np.ndarray, wall_mask: np.ndarray, max_nodes: int,
                         rng: np.random.Generator) -> np.ndarray:
    """Return indices into pos producing approximately max_nodes nodes.

    We sample wall and interior independently with the SAME thinning rate, so
    the resulting wall fraction matches the source mesh's natural wall ratio
    (~3-5% for the sUbend volume meshes). This mirrors what the Womersley /
    Cosserat datasets look like at 8-10% wall.

    A minimum floor of 2000 wall nodes (or all wall nodes, whichever is
    smaller) keeps the wall-loss signal strong for the GNN.
    """
    n = pos.shape[0]
    if n <= max_nodes:
        return np.arange(n, dtype=np.int64)
    wall_idx = np.flatnonzero(wall_mask)
    interior_idx = np.flatnonzero(~wall_mask)
    rate = max_nodes / float(n)

    n_wall_keep = max(min(2000, wall_idx.size), int(round(wall_idx.size * rate)))
    n_wall_keep = min(n_wall_keep, wall_idx.size)
    n_int_keep = max(0, max_nodes - n_wall_keep)
    n_int_keep = min(n_int_keep, interior_idx.size)

    sel_wall = rng.choice(wall_idx, size=n_wall_keep, replace=False) if n_wall_keep < wall_idx.size else wall_idx
    sel_int = rng.choice(interior_idx, size=n_int_keep, replace=False)
    selected = np.concatenate([sel_wall, sel_int])
    selected.sort()
    return selected.astype(np.int64)


def make_features(pos: np.ndarray, wall: np.ndarray, R_char: float) -> Tuple[np.ndarray, np.ndarray]:
    """Build the 11-column feature matrix used by FlowGAT.

    Layout (mirrors src/make_npz_womersley.py):
      0  axial_pca_norm
      1  wall_distance_norm
      2  r_per_R_clipped
      3-5  global PCA tangent (tiled)
      6  inlet flag
      7  outlet flag
      8  local_radius_norm (default 1.0; populated by variant builders)
      9  distance_to_pca_axis_norm
     10  wall_flag
    """
    from scipy.spatial import cKDTree
    n_nodes = pos.shape[0]

    # Global PCA axis
    centered = pos - pos.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    pca_axis = vt[0].astype(np.float32)
    if pca_axis[int(np.argmax(np.abs(pca_axis)))] < 0:
        pca_axis = -pca_axis
    axial_proj = (centered @ pca_axis).astype(np.float32)
    axial_norm = (axial_proj - axial_proj.min()) / max(axial_proj.max() - axial_proj.min(), 1e-9)

    # Distance to PCA axis (for r_per_R and dist_axis_norm)
    perp = centered - axial_proj[:, None] * pca_axis[None, :]
    dist_to_axis = np.linalg.norm(perp, axis=1).astype(np.float32)
    dist_axis_norm = dist_to_axis / max(dist_to_axis.max(), 1e-9)
    r_per_R = (dist_to_axis / max(R_char, 1e-9)).clip(0.0, 1.0).astype(np.float32)

    # Wall distance
    wall_pts = pos[wall]
    if len(wall_pts) >= 8:
        d_wall, _ = cKDTree(wall_pts).query(pos, k=1)
    else:
        d_wall = np.zeros(n_nodes, dtype=np.float32)
    d_wall = d_wall.astype(np.float32)
    d_wall_norm = d_wall / max(d_wall.max(), 1e-9)

    # Inlet / outlet: 2% extremes on axial projection
    lo, hi = np.quantile(axial_proj, [0.02, 0.98])
    inlet = (axial_proj <= lo).astype(np.float32)
    outlet = (axial_proj >= hi).astype(np.float32)

    # local_radius_norm default = 1.0; the variant builder may overwrite to leak
    local_R_norm = np.ones(n_nodes, dtype=np.float32)

    tangent_tile = np.tile(pca_axis[None, :], (n_nodes, 1)).astype(np.float32)

    x = np.stack([
        axial_norm.astype(np.float32),
        d_wall_norm.astype(np.float32),
        r_per_R,
        tangent_tile[:, 0], tangent_tile[:, 1], tangent_tile[:, 2],
        inlet, outlet,
        local_R_norm,
        dist_axis_norm,
        wall.astype(np.float32),
    ], axis=-1).astype(np.float32)
    return x, pca_axis


def build_case_frames(case_dir: Path, *, max_nodes: int, knn: int,
                      rng: np.random.Generator, wall_threshold: float,
                      wall_quantile: float = DEFAULT_WALL_QUANTILE) -> List[Dict]:
    """Build one NPZ payload per frame in one case."""
    frames = list_frames(case_dir)
    if not frames:
        raise FileNotFoundError(f"no Frame_*.vtk under {case_dir}")

    pos_full, surface_ids, wall_full = derive_wall_mask(case_dir, wall_threshold, wall_quantile)
    n_full = pos_full.shape[0]

    # Subsample once per case so all frames share the same mesh
    sel = stratified_subsample(pos_full, wall_full, max_nodes, rng)
    pos = pos_full[sel].astype(np.float32)
    wall = wall_full[sel]

    # Build feature matrix and edges once per case
    bb = pos.max(0) - pos.min(0)
    R_char = float(0.5 * bb.min())  # half the shortest extent ~ tube radius
    L_char = float(bb.max())
    x_base, pca_axis = make_features(pos, wall, R_char)
    edge_index = knn_edges(pos, k=knn)
    diff = pos[edge_index[1]] - pos[edge_index[0]]
    dist = np.linalg.norm(diff, axis=1, keepdims=True).astype(np.float32)
    edge_attr = np.concatenate([diff.astype(np.float32), dist, np.ones_like(dist)], axis=1)

    # Wall normals: outward from PCA axis at wall nodes (rough approximation for U-bend)
    centered = pos - pos.mean(0, keepdims=True)
    perp = centered - (centered @ pca_axis)[:, None] * pca_axis[None, :]
    perp_norm = np.linalg.norm(perp, axis=1, keepdims=True).clip(min=1e-8)
    wn = (perp / perp_norm).astype(np.float32)
    wn[~wall] = 0.0

    payloads: List[Dict] = []
    pat = re.compile(r"Frame_(\d+)\.vtk$")
    for f in frames:
        frame_idx = int(pat.search(f.name).group(1))
        m = read_frame(f)
        U_full = np.asarray(m.point_data["U"], dtype=np.float32)
        p_full = np.asarray(m.point_data["p"], dtype=np.float32)
        y_full = U_full
        y = y_full[sel].astype(np.float32)
        p = p_full[sel].astype(np.float32)
        # Enforce no-slip on wall (consistent with CFD ground truth & other datasets)
        y[wall] = 0.0

        # p_grad: finite-difference of pressure along PCA axis as bulk forcing proxy
        axial_proj = (pos - pos.mean(0, keepdims=True)) @ pca_axis
        # Linear regression slope of p vs axial_proj
        s_var = float(np.var(axial_proj))
        if s_var > 1e-20:
            slope = float(np.cov(p, axial_proj, ddof=0)[0, 1] / s_var)
        else:
            slope = 0.0
        p_grad = np.tile((-slope * pca_axis)[None, :], (pos.shape[0], 1)).astype(np.float32)

        u_char = np.float32(max(np.linalg.norm(y_full, axis=1).max(), 1e-6))

        case_id = case_dir.name  # e.g. sUbend_012
        sample_id = f"{case_id}__f{frame_idx:02d}"
        payload = dict(
            x=x_base.copy(),
            pos=pos,
            y=y,
            edge_index=edge_index,
            edge_attr=edge_attr,
            wall_mask=wall,
            wall_normal=wn,
            u_char=u_char,
            length_char=np.float32(L_char),
            p_grad=p_grad,
            case_id=sample_id,
            meta=dict(
                source="subend",
                case=case_id,
                frame=int(frame_idx),
                n_frames=int(len(frames)),
                t_phase=float(frame_idx) / max(len(frames) - 1, 1),
                R=float(R_char),
                L=float(L_char),
                n_full=int(n_full),
                n_sampled=int(pos.shape[0]),
                bb_extent_x=float(bb[0]),
                bb_extent_y=float(bb[1]),
                bb_extent_z=float(bb[2]),
            ),
        )
        payloads.append(payload)
    return payloads


def save_payload(out_dir: Path, payload: Dict) -> Path:
    sid = payload["case_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{sid}.npz"
    meta = payload.pop("meta")
    np.savez_compressed(path, **payload, meta=json.dumps(meta))
    return path


def build_split(case_ids: Sequence[str], rng: np.random.Generator,
                n_train: int, n_val: int, n_test: int) -> Dict[str, List[str]]:
    """Case-level split (frames from the same case stay together)."""
    cases = sorted(case_ids)
    rng.shuffle(cases)
    assert len(cases) >= n_train + n_val + n_test, \
        f"need at least {n_train+n_val+n_test} cases, got {len(cases)}"
    return {
        "train": sorted(cases[:n_train]),
        "val":   sorted(cases[n_train:n_train+n_val]),
        "test":  sorted(cases[n_train+n_val:n_train+n_val+n_test]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", default="data/external/newCFD_dataset",
                    help="root containing sUbend_<NNN>/CFD/Frame_*.vtk")
    ap.add_argument("--out_dir", default="data/npz_subend")
    ap.add_argument("--max_nodes", type=int, default=DEFAULT_MAX_NODES)
    ap.add_argument("--knn", type=int, default=DEFAULT_KNN)
    ap.add_argument("--wall_speed_threshold", type=float, default=WALL_SPEED_THRESHOLD)
    ap.add_argument("--wall_quantile", type=float, default=DEFAULT_WALL_QUANTILE)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--cases", nargs="*", default=None,
                    help="optional whitelist of case dirs (basenames)")
    ap.add_argument("--n_train", type=int, default=9)
    ap.add_argument("--n_val", type=int, default=3)
    ap.add_argument("--n_test", type=int, default=3)
    args = ap.parse_args()

    src_root = Path(args.src_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted(p for p in src_root.glob("sUbend_*") if p.is_dir())
    if args.cases:
        wanted = set(args.cases)
        case_dirs = [p for p in case_dirs if p.name in wanted]
    if not case_dirs:
        raise SystemExit(f"no sUbend_* case dirs under {src_root}")

    rng = np.random.default_rng(int(args.seed))
    print(f"[subend] processing {len(case_dirs)} cases -> {out_dir}", flush=True)
    for ci, case_dir in enumerate(case_dirs):
        t0 = time.time()
        try:
            payloads = build_case_frames(
                case_dir,
                max_nodes=int(args.max_nodes),
                knn=int(args.knn),
                rng=rng,
                wall_threshold=float(args.wall_speed_threshold),
                wall_quantile=float(args.wall_quantile),
            )
        except Exception as exc:
            print(f"  [{ci+1}/{len(case_dirs)}] {case_dir.name}: ERROR {exc!r}", flush=True)
            continue
        for p in payloads:
            save_payload(out_dir, p)
        dt = time.time() - t0
        n_sampled = payloads[0]["pos"].shape[0] if payloads else -1
        print(f"  [{ci+1}/{len(case_dirs)}] {case_dir.name}: "
              f"{len(payloads)} frames, n_nodes={n_sampled}, walltime={dt:.1f}s", flush=True)

    # Build split.json at case granularity, then re-fan out to per-frame ids
    case_ids = [p.name for p in case_dirs]
    case_split = build_split(case_ids, rng, args.n_train, args.n_val, args.n_test)
    sample_split = {k: [] for k in case_split}
    for f in sorted(out_dir.glob("*.npz")):
        sid = f.stem  # e.g. sUbend_012__f04
        case = sid.split("__")[0]
        for split_name, members in case_split.items():
            if case in members:
                sample_split[split_name].append(sid)
                break

    with open(out_dir / "split.json", "w") as f:
        json.dump(sample_split, f, indent=2)
    print(f"[subend] split.json: "
          f"train={len(sample_split['train'])} val={len(sample_split['val'])} "
          f"test={len(sample_split['test'])}")
    print("[subend] DONE", flush=True)


if __name__ == "__main__":
    main()
