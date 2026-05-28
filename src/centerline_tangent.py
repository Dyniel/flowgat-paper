"""
Local centerline tangent utilities for Phase E (CP top-tier push).

Implements a *ladder* of geometric direction priors of increasing fidelity:

  1. global_pca_tangent     : one axis per case (same as existing noleak)
  2. local_pca_tangent      : per-node PCA on k-hop graph neighborhood
  3. centerline_tangent     : tangent of bin-centroid smoothed centerline,
                              evaluated at the centerline point closest to
                              each node
  4. frenet_tangent         : same tangent as (3), plus normal and binormal
                              for richer geometric features (optional)

The physical claim is that as the prior approaches the *true* local axial
direction, model performance converges to the leak_dir_only ceiling, while
direction-error (angle between prior and true CFD velocity) decreases
monotonically.

Dependencies: numpy, scipy. No VMTK / VTK required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Global PCA tangent (same as existing noleak — kept here for completeness)
# ---------------------------------------------------------------------------

def global_pca_tangent(pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (axis, axial_raw_coord) — global PCA of node positions."""
    centered = pos - pos.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0].astype(np.float32)
    n = float(np.linalg.norm(axis))
    if n < 1e-12:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        axis = axis / n
    # sign convention (same as make_npz_noleak.py)
    major = int(np.argmax(np.abs(axis)))
    if axis[major] < 0:
        axis = -axis
    axial_raw = (centered @ axis).astype(np.float32)
    return axis, axial_raw


# ---------------------------------------------------------------------------
# Local PCA tangent (per-node PCA on k-hop graph neighborhood)
# ---------------------------------------------------------------------------

def _khop_neighbors(edge_index: np.ndarray, n_nodes: int, k: int) -> list[np.ndarray]:
    """Return list of k-hop neighbor index arrays (each excludes self)."""
    # build adjacency once
    adj = [[] for _ in range(n_nodes)]
    src, dst = edge_index[0], edge_index[1]
    for s, d in zip(src.tolist(), dst.tolist()):
        adj[s].append(d)
    adj = [np.asarray(a, dtype=np.int64) for a in adj]

    out: list[np.ndarray] = []
    for v in range(n_nodes):
        frontier = {v}
        visited = {v}
        for _ in range(k):
            nxt: set[int] = set()
            for u in frontier:
                nxt.update(int(x) for x in adj[u])
            nxt -= visited
            visited |= nxt
            frontier = nxt
        visited.discard(v)
        out.append(np.fromiter(visited, dtype=np.int64))
    return out


def local_pca_tangent(
    pos: np.ndarray,
    edge_index: np.ndarray,
    *,
    k_hop: int = 2,
    fallback_axis: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-node local PCA top eigenvector.

    Uses k-hop graph neighborhood (default k=2 → ~50–150 neighbors at typical
    mesh resolution). Sign is aligned to the global PCA axis to keep tangent
    field continuous (no sign flips between neighboring nodes).
    """
    n = pos.shape[0]
    if fallback_axis is None:
        fallback_axis, _ = global_pca_tangent(pos)

    tangents = np.tile(fallback_axis[None, :], (n, 1)).astype(np.float32)

    # adjacency lists
    adj = [[] for _ in range(n)]
    for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        adj[s].append(d)

    for v in range(n):
        # k-hop BFS
        frontier = {v}
        visited = {v}
        for _ in range(int(k_hop)):
            nxt: set[int] = set()
            for u in frontier:
                nxt.update(int(x) for x in adj[u])
            nxt -= visited
            visited |= nxt
            frontier = nxt
        idx = np.fromiter(visited, dtype=np.int64)
        if idx.size < 6:
            continue
        P = pos[idx]
        P = P - P.mean(0, keepdims=True)
        try:
            _, _, vt = np.linalg.svd(P, full_matrices=False)
            t = vt[0]
        except np.linalg.LinAlgError:
            continue
        # align sign to fallback axis
        if t @ fallback_axis < 0:
            t = -t
        nrm = float(np.linalg.norm(t))
        if nrm > 1e-12:
            tangents[v] = (t / nrm).astype(np.float32)
    return tangents


def local_pca_tangent_kdtree(
    pos: np.ndarray,
    *,
    radius: Optional[float] = None,
    k: int = 32,
    fallback_axis: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-node local PCA top eigenvector using kNN ball (faster than k-hop BFS).

    Either pass `radius` (ball query) or `k` (k-nearest neighbors). kNN is
    deterministic and faster on large meshes.
    """
    if fallback_axis is None:
        fallback_axis, _ = global_pca_tangent(pos)

    n = pos.shape[0]
    tree = cKDTree(pos)
    if radius is not None:
        idx_lists = tree.query_ball_point(pos, r=float(radius))
    else:
        _, knn_idx = tree.query(pos, k=int(k) + 1)  # includes self
        idx_lists = [row[1:] for row in knn_idx]   # drop self

    tangents = np.tile(fallback_axis[None, :], (n, 1)).astype(np.float32)
    for v, idx in enumerate(idx_lists):
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size < 6:
            continue
        P = pos[idx] - pos[idx].mean(0, keepdims=True)
        try:
            _, _, vt = np.linalg.svd(P, full_matrices=False)
            t = vt[0]
        except np.linalg.LinAlgError:
            continue
        if t @ fallback_axis < 0:
            t = -t
        nrm = float(np.linalg.norm(t))
        if nrm > 1e-12:
            tangents[v] = (t / nrm).astype(np.float32)
    return tangents


# ---------------------------------------------------------------------------
# Bin-centroid centerline + per-node tangent at closest centerline point
# ---------------------------------------------------------------------------

@dataclass
class Centerline:
    points: np.ndarray       # (M, 3) smoothed polyline
    tangents: np.ndarray     # (M, 3) unit
    normals: np.ndarray      # (M, 3) unit (Frenet)
    binormals: np.ndarray    # (M, 3) unit (Frenet)
    curvature: np.ndarray    # (M,)


def _smooth_polyline(P: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return P.astype(np.float32)
    return np.stack(
        [gaussian_filter1d(P[:, d], sigma=float(sigma), mode="reflect") for d in range(3)],
        axis=1,
    ).astype(np.float32)


def extract_bin_centroid_centerline(
    pos: np.ndarray,
    wall_mask: np.ndarray,
    *,
    n_bins: int = 80,
    medial_quantile: float = 0.85,
    use_interior_only: bool = True,
    smooth_sigma: float = 3.0,
) -> Centerline:
    """Medial-axis-style centerline.

    For each axial slice (binned by global PCA axial coord), pick the nodes
    with highest distance-to-wall (top `1 - medial_quantile` quantile), and
    use their centroid as a centerline point. This is robust to branching
    (branch nodes near walls are filtered out by the d_wall criterion).

    Limitations: still uses global PCA axial as a slicing parameter, so
    aortic arch may collapse on itself (two slices on opposite sides of the
    arch share an axial bin). For aortas in our split this is acceptable;
    we mitigate by `smooth_sigma >= 3` and report the residual angle in
    Methods.
    """
    axis, axial_raw = global_pca_tangent(pos)
    mask = ~wall_mask if use_interior_only else np.ones(pos.shape[0], dtype=bool)
    if mask.sum() < n_bins * 2:
        mask = np.ones(pos.shape[0], dtype=bool)

    # wall distance for medial filtering
    wall_pts = pos[wall_mask]
    if len(wall_pts) >= 8:
        d_wall, _ = cKDTree(wall_pts).query(pos, k=1)
    else:
        d_wall = np.ones(pos.shape[0], dtype=np.float32)

    a_min = float(axial_raw[mask].min())
    a_max = float(axial_raw[mask].max())
    if a_max - a_min < 1e-6:
        a_max = a_min + 1.0
    bins = np.linspace(a_min, a_max, int(n_bins) + 1)

    centroids = []
    for i in range(int(n_bins)):
        in_bin = (axial_raw >= bins[i]) & (axial_raw < bins[i + 1]) & mask
        if in_bin.sum() < 5:
            continue
        nodes = np.where(in_bin)[0]
        dw = d_wall[nodes]
        thr = float(np.quantile(dw, float(medial_quantile)))
        keep = nodes[dw >= thr]
        if keep.size < 1:
            keep = nodes
        centroids.append(pos[keep].mean(0))
    centroids = np.asarray(centroids, dtype=np.float32)
    if centroids.shape[0] < 4:
        centroids = np.stack(
            [pos.mean(0) + t * axis for t in np.linspace(a_min, a_max, max(4, n_bins // 4))],
            axis=0,
        ).astype(np.float32)

    smoothed = _smooth_polyline(centroids, sigma=float(smooth_sigma))

    # Frenet frame via finite differences with arc-length parameterization
    diff = np.diff(smoothed, axis=0)
    seg_len = np.linalg.norm(diff, axis=1).clip(min=1e-8)
    # tangent at each polyline point (forward difference, with endpoint duplication)
    t_pts = np.empty_like(smoothed)
    t_pts[:-1] = diff / seg_len[:, None]
    t_pts[-1] = t_pts[-2]
    # tangent derivative w.r.t. arc length
    dt = np.empty_like(smoothed)
    dt[:-1] = np.diff(t_pts, axis=0) / seg_len[:, None]
    dt[-1] = dt[-2]
    curvature = np.linalg.norm(dt, axis=1)
    # normal: dt / |dt|, with fallback to a vector orthogonal to t
    n_pts = np.zeros_like(smoothed)
    nz = curvature > 1e-6
    n_pts[nz] = dt[nz] / curvature[nz, None]
    # for degenerate (straight) sections, pick any orthogonal direction
    if (~nz).any():
        # build any vector not parallel to t
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        for i in np.where(~nz)[0]:
            v = ref - (ref @ t_pts[i]) * t_pts[i]
            if np.linalg.norm(v) < 1e-6:
                v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                v = v - (v @ t_pts[i]) * t_pts[i]
            n_pts[i] = v / max(np.linalg.norm(v), 1e-8)
    b_pts = np.cross(t_pts, n_pts)
    b_nrm = np.linalg.norm(b_pts, axis=1, keepdims=True).clip(min=1e-8)
    b_pts = b_pts / b_nrm

    return Centerline(
        points=smoothed,
        tangents=t_pts.astype(np.float32),
        normals=n_pts.astype(np.float32),
        binormals=b_pts.astype(np.float32),
        curvature=curvature.astype(np.float32),
    )


def iterative_medial_centerline(
    pos: np.ndarray,
    wall_mask: np.ndarray,
    *,
    n_bins: int = 80,
    n_iter: int = 3,
    medial_quantile: float = 0.85,
    smooth_sigma: float = 3.0,
) -> Centerline:
    """Iteratively-refined medial centerline.

    Iteration 0 uses global-PCA axial coordinate for slicing (handles
    aortic arch poorly). Each subsequent iteration re-parametrizes nodes
    by their projected *arc length* on the current centerline, then
    re-bins and recomputes medial centroids. After 2-3 iterations, the
    centerline reliably follows the arch.
    """
    cl = extract_bin_centroid_centerline(
        pos, wall_mask, n_bins=n_bins, medial_quantile=medial_quantile,
        smooth_sigma=smooth_sigma,
    )
    if cl.points.shape[0] < 4:
        return cl

    # wall distance once
    if wall_mask.any():
        d_wall, _ = cKDTree(pos[wall_mask]).query(pos, k=1)
    else:
        d_wall = np.ones(pos.shape[0], dtype=np.float32)

    for _ in range(max(0, int(n_iter))):
        # arc-length per centerline point
        d = np.diff(cl.points, axis=0)
        seg = np.linalg.norm(d, axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float32)
        if arc[-1] < 1e-6:
            break

        # for each node, project to nearest cl point → node arc-length
        _, idx = cKDTree(cl.points).query(pos, k=1)
        node_arc = arc[idx]

        # re-bin by arc length, take medial centroids
        a_min, a_max = float(node_arc.min()), float(node_arc.max())
        if a_max - a_min < 1e-6:
            break
        bins = np.linspace(a_min, a_max, int(n_bins) + 1)
        centroids = []
        interior = ~wall_mask
        for i in range(int(n_bins)):
            in_bin = (node_arc >= bins[i]) & (node_arc < bins[i + 1]) & interior
            if in_bin.sum() < 5:
                continue
            nodes = np.where(in_bin)[0]
            dw = d_wall[nodes]
            thr = float(np.quantile(dw, float(medial_quantile)))
            keep = nodes[dw >= thr]
            if keep.size < 1:
                keep = nodes
            centroids.append(pos[keep].mean(0))
        if len(centroids) < 4:
            break
        centroids = np.asarray(centroids, dtype=np.float32)
        smoothed = _smooth_polyline(centroids, sigma=float(smooth_sigma))

        # rebuild Frenet
        diff = np.diff(smoothed, axis=0)
        seg_len = np.linalg.norm(diff, axis=1).clip(min=1e-8)
        t_pts = np.empty_like(smoothed)
        t_pts[:-1] = diff / seg_len[:, None]
        t_pts[-1] = t_pts[-2]
        dt = np.empty_like(smoothed)
        dt[:-1] = np.diff(t_pts, axis=0) / seg_len[:, None]
        dt[-1] = dt[-2]
        curv = np.linalg.norm(dt, axis=1)
        n_pts = np.zeros_like(smoothed)
        nz = curv > 1e-6
        n_pts[nz] = dt[nz] / curv[nz, None]
        if (~nz).any():
            ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            for i in np.where(~nz)[0]:
                v = ref - (ref @ t_pts[i]) * t_pts[i]
                if np.linalg.norm(v) < 1e-6:
                    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                    v = v - (v @ t_pts[i]) * t_pts[i]
                n_pts[i] = v / max(np.linalg.norm(v), 1e-8)
        b_pts = np.cross(t_pts, n_pts)
        b_nrm = np.linalg.norm(b_pts, axis=1, keepdims=True).clip(min=1e-8)
        b_pts = b_pts / b_nrm

        cl = Centerline(
            points=smoothed,
            tangents=t_pts.astype(np.float32),
            normals=n_pts.astype(np.float32),
            binormals=b_pts.astype(np.float32),
            curvature=curv.astype(np.float32),
        )
    return cl


def per_node_from_centerline(
    pos: np.ndarray,
    centerline: Centerline,
) -> dict:
    """Map per-centerline-point quantities to per-node by nearest neighbor."""
    tree = cKDTree(centerline.points)
    dist, idx = tree.query(pos, k=1)
    return dict(
        tangent=centerline.tangents[idx].astype(np.float32),
        normal=centerline.normals[idx].astype(np.float32),
        binormal=centerline.binormals[idx].astype(np.float32),
        curvature=centerline.curvature[idx].astype(np.float32),
        dist_to_centerline=dist.astype(np.float32),
        nearest_idx=idx.astype(np.int64),
    )


# ---------------------------------------------------------------------------
# Convenience: compute all four priors for a single case
# ---------------------------------------------------------------------------

def compute_all_priors(
    pos: np.ndarray,
    wall_mask: np.ndarray,
    edge_index: Optional[np.ndarray] = None,
    *,
    n_bins: int = 80,
    smooth_sigma: float = 3.0,
    n_iter: int = 3,
    medial_quantile: float = 0.85,
    include_local_pca: bool = False,
    local_pca_k: int = 32,
) -> dict:
    """Compute the ladder of geometric direction priors.

    Returns per-node:
      - global_pca_tangent   : single vector tiled per node
      - centerline_tangent   : iteratively-refined medial centerline tangent
                               (sign-aligned to global axis)
      - frenet_normal / binormal / curvature  : Frenet frame for richer features
      - dist_to_centerline   : per-node distance to centerline
      - (optional) local_pca_tangent : kept for completeness; not recommended
                                       for tubular geometry — picks cross-
                                       sectional axis at small k, chord at
                                       large k. Set include_local_pca=True
                                       explicitly to enable.
    """
    axis, _ = global_pca_tangent(pos)
    n_nodes = pos.shape[0]
    global_t = np.tile(axis[None, :], (n_nodes, 1)).astype(np.float32)

    cl = iterative_medial_centerline(
        pos, wall_mask,
        n_bins=int(n_bins), n_iter=int(n_iter),
        medial_quantile=float(medial_quantile),
        smooth_sigma=float(smooth_sigma),
    )
    per_node = per_node_from_centerline(pos, cl)
    centerline_t = per_node["tangent"]
    flip = (centerline_t @ axis) < 0
    centerline_t = np.where(flip[:, None], -centerline_t, centerline_t).astype(np.float32)
    normal_t = per_node["normal"]
    binormal_t = per_node["binormal"]

    out = dict(
        global_pca_tangent=global_t,
        centerline_tangent=centerline_t,
        frenet_normal=normal_t,
        frenet_binormal=binormal_t,
        frenet_curvature=per_node["curvature"],
        centerline_points=cl.points,
        dist_to_centerline=per_node["dist_to_centerline"],
    )
    if include_local_pca:
        out["local_pca_tangent"] = local_pca_tangent_kdtree(
            pos, k=int(local_pca_k), fallback_axis=axis
        )
    return out
