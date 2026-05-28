# -*- coding: utf-8 -*-
"""
Mesh-refinement convergence and Helmholtz-projection diagnostic for Womersley.

This script evaluates already-trained Womersley checkpoints on regenerated
1x/2x/4x meshes.  It intentionally does not train anything.

Outputs:
  results/diagnostics/mesh_refinement/parts/*.csv
  results/diagnostics/mesh_refinement/per_case.csv
  results/diagnostics/mesh_refinement/aggregate.csv
  results/diagnostics/mesh_refinement/summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
from scipy.spatial import cKDTree

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_SRC_DIR)
sys.path.insert(0, _SRC_DIR)
sys.path.insert(0, _REPO_DIR)

from physics_diagnostics import local_jacobian_trace_vec


VARIANTS = ("withleak", "leak_dir_only", "leak_mag_only", "noleak")
RESOLUTIONS = ("1x", "2x", "4x")
SEEDS = (1337, 2026, 777)
HE_PERCENTILE = 0.80
DEFAULT_CHUNK_NODES = 256_000

AGG_COLUMNS = [
    "variant", "resolution", "n_cases", "n_seeds",
    "div_pred_mean", "div_pred_p90",
    "div_true_mean", "div_true_p90",
    "div_analytical_mean", "div_analytical_p90",
    "helm_pred_mean", "helm_pred_p90",
    "helm_true_mean", "helm_true_p90",
]


def _finite(vals: Iterable[float]) -> List[float]:
    out: List[float] = []
    for v in vals:
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            out.append(x)
    return out


def _nanmean(vals: Iterable[float]) -> float:
    clean = _finite(vals)
    return float(np.mean(clean)) if clean else float("nan")


def _nanp(vals: Iterable[float], q: float) -> float:
    clean = _finite(vals)
    return float(np.percentile(clean, q)) if clean else float("nan")


def _weighted_mean(rows: Sequence[Dict[str, float]], key: str, weight_key: str) -> float:
    num = 0.0
    den = 0.0
    for r in rows:
        v = float(r.get(key, float("nan")))
        w = float(r.get(weight_key, 0.0))
        if math.isfinite(v) and math.isfinite(w) and w > 0.0:
            num += v * w
            den += w
    return num / den if den > 0.0 else float("nan")


def resolution_data_dir(data_root: Path, resolution: str) -> Path:
    return data_root / f"npz_womersley_meshref_{resolution}"


def checkpoint_path(ckpt_root: Path, variant: str, seed: int) -> Path:
    return ckpt_root / f"womersley_{variant}" / f"seed_{seed}" / "best.pt"


def config_path(config_root: Path, variant: str) -> Path:
    return config_root / f"womersley_{variant}.yaml"


def load_split_cases(data_dir: Path, split: str) -> List[str]:
    split_path = data_dir / "split.json"
    with open(split_path, "r") as f:
        split_obj = json.load(f)
    if split not in split_obj:
        raise KeyError(f"{split!r} not in {split_path}")
    return list(split_obj[split])


def apply_womersley_variant_features(
    x_base: np.ndarray,
    y: np.ndarray,
    wall_mask: np.ndarray,
    u_char: float,
    variant: str,
) -> np.ndarray:
    """Create the feature matrix expected by the trained leakage variant."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    x = x_base.astype(np.float32, copy=True)

    if variant in ("withleak", "leak_dir_only"):
        speed = np.linalg.norm(y, axis=1, keepdims=True).clip(min=1.0e-8)
        u_dir = (y / speed).astype(np.float32)
        u_dir[wall_mask] = 0.0
        x[:, 3:6] = u_dir

    if variant in ("withleak", "leak_mag_only"):
        speed = np.linalg.norm(y, axis=1).astype(np.float32)
        x[:, 8] = speed / max(float(u_char), 1.0e-8)

    return x


def contiguous_chunks(n_nodes: int, resolution: str, chunk_nodes: int) -> List[np.ndarray]:
    """4x defaults to sharded evaluation; smaller meshes stay whole."""
    if resolution != "4x" or n_nodes <= chunk_nodes:
        return [np.arange(n_nodes, dtype=np.int64)]
    chunks: List[np.ndarray] = []
    for start in range(0, n_nodes, int(chunk_nodes)):
        stop = min(start + int(chunk_nodes), n_nodes)
        chunks.append(np.arange(start, stop, dtype=np.int64))
    return chunks


def subset_edges_contiguous(
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    start: int,
    stop: int,
) -> Tuple[np.ndarray, np.ndarray]:
    src = edge_index[0]
    dst = edge_index[1]
    keep = (src >= start) & (src < stop) & (dst >= start) & (dst < stop)
    ei = edge_index[:, keep].astype(np.int64, copy=True)
    ei -= int(start)
    ea = edge_attr[keep].astype(np.float32, copy=False)
    return ei, ea


def median_nn_distance(pos: np.ndarray) -> float:
    tree = cKDTree(pos)
    d, _ = tree.query(pos, k=2)
    return max(float(np.median(d[:, 1])), 1.0e-12)


def he_mask(y: np.ndarray, wall_mask: np.ndarray) -> np.ndarray:
    speed = np.linalg.norm(y, axis=1)
    interior = ~wall_mask
    if int(interior.sum()) == 0:
        return np.ones(y.shape[0], dtype=bool)
    q = float(np.quantile(speed[interior], HE_PERCENTILE))
    mask = interior & (speed >= q)
    if int(mask.sum()) == 0:
        return interior
    return mask


def normalized_divergence(
    pos: np.ndarray,
    u: np.ndarray,
    y_for_scale: np.ndarray,
    wall_mask: np.ndarray,
    *,
    k_neighbors: int = 16,
) -> Tuple[float, float, float, int]:
    """Mean |trace(J)| * L / U, using the existing local-Jacobian estimator."""
    div = local_jacobian_trace_vec(pos, u, k_neighbors=int(k_neighbors))
    mask = ~wall_mask
    if int(mask.sum()) == 0:
        mask = np.ones(pos.shape[0], dtype=bool)
    L = median_nn_distance(pos)
    he = he_mask(y_for_scale, wall_mask)
    speed = np.linalg.norm(y_for_scale, axis=1)
    U = max(float(np.mean(speed[he])), 1.0e-12)
    val = float(np.mean(np.abs(div[mask])) * L / U)
    return val, L, U, int(mask.sum())


def graph_divergence(
    pos: np.ndarray,
    u: np.ndarray,
    *,
    k_neighbors: int = 16,
) -> np.ndarray:
    return local_jacobian_trace_vec(pos, u, k_neighbors=int(k_neighbors)).astype(np.float64)


def sym_graph_laplacian(pos: np.ndarray, edge_index: np.ndarray) -> sparse.csr_matrix:
    n = int(pos.shape[0])
    if edge_index.size == 0:
        return sparse.eye(n, format="csr", dtype=np.float64)

    src = edge_index[0].astype(np.int64, copy=False)
    dst = edge_index[1].astype(np.int64, copy=False)
    dx = pos[dst] - pos[src]
    dist2 = np.einsum("ij,ij->i", dx, dx).astype(np.float64)
    weights = 1.0 / np.maximum(dist2, 1.0e-18)

    w = sparse.coo_matrix((weights, (src, dst)), shape=(n, n), dtype=np.float64)
    w = (w + w.T).tocsr()
    w.sum_duplicates()
    degree = np.asarray(w.sum(axis=1)).ravel()
    lap = sparse.diags(degree, format="csr") - w
    return (0.5 * (lap + lap.T)).tocsr()


def solve_cg_mean_free(
    lap: sparse.csr_matrix,
    rhs: np.ndarray,
    *,
    tol: float = 1.0e-6,
    maxiter: int = 500,
) -> Tuple[Optional[np.ndarray], int]:
    n = lap.shape[0]
    if n == 0:
        return None, -1
    b = rhs.astype(np.float64, copy=True)
    b -= float(np.mean(b))
    if not np.isfinite(b).all():
        return None, -2
    if float(np.linalg.norm(b)) <= 1.0e-14:
        return np.zeros(n, dtype=np.float64), 0

    scale = float(np.mean(lap.diagonal())) if lap.shape[0] else 1.0
    reg = max(scale, 1.0) * 1.0e-10
    system = (lap + sparse.eye(n, format="csr", dtype=np.float64) * reg).tocsr()
    diag = system.diagonal()
    inv_diag = 1.0 / np.maximum(diag, 1.0e-30)
    precond = spla.LinearOperator(system.shape, matvec=lambda x: inv_diag * x)
    try:
        phi, info = spla.cg(system, b, rtol=tol, atol=0.0, maxiter=int(maxiter), M=precond)
    except TypeError:
        phi, info = spla.cg(system, b, tol=tol, maxiter=int(maxiter), M=precond)
    if info != 0 or phi is None or not np.isfinite(phi).all():
        return None, int(info)
    phi -= float(np.mean(phi))
    return phi.astype(np.float64), 0


def scalar_gradient_lstsq(
    pos: np.ndarray,
    phi: np.ndarray,
    *,
    k_neighbors: int = 16,
) -> np.ndarray:
    tree = cKDTree(pos)
    dists, idxs = tree.query(pos, k=int(k_neighbors) + 1)
    dists = dists[:, 1:]
    idxs = idxs[:, 1:]

    dx = pos[idxs] - pos[:, None, :]
    dphi = phi[idxs] - phi[:, None]
    h = max(float(np.median(dists[:, 0])), 1.0e-12)
    w = np.exp(-(dists / h) ** 2).astype(np.float64)
    wdx = dx.astype(np.float64) * w[:, :, None]
    a = np.einsum("nki,nkj->nij", wdx, dx.astype(np.float64))
    a = a + 1.0e-10 * np.eye(3, dtype=np.float64)[None]
    b = np.einsum("nki,nk->ni", wdx, dphi.astype(np.float64))
    try:
        grad = np.linalg.solve(a, b[..., None])[..., 0]
    except np.linalg.LinAlgError:
        grad = np.zeros((pos.shape[0], 3), dtype=np.float64)
        for i in range(pos.shape[0]):
            try:
                grad[i] = np.linalg.solve(a[i] + 1.0e-7 * np.eye(3), b[i])
            except np.linalg.LinAlgError:
                grad[i] = 0.0
    return grad.astype(np.float32)


def helmholtz_residual(
    pos: np.ndarray,
    u: np.ndarray,
    edge_index: np.ndarray,
    wall_mask: np.ndarray,
    *,
    k_neighbors: int = 16,
    tol: float = 1.0e-6,
    maxiter: int = 500,
) -> float:
    """Graph Helmholtz projection residual ||grad(phi)||_2 / ||u||_2.

    If CG fails, returns NaN and leaves the calling job alive.
    """
    try:
        rhs = graph_divergence(pos, u, k_neighbors=int(k_neighbors))
        lap = sym_graph_laplacian(pos, edge_index)
        phi, info = solve_cg_mean_free(lap, rhs, tol=float(tol), maxiter=int(maxiter))
        if phi is None or info != 0:
            return float("nan")
        grad_phi = scalar_gradient_lstsq(pos, phi, k_neighbors=int(k_neighbors))
        mask = ~wall_mask
        if int(mask.sum()) == 0:
            mask = np.ones(pos.shape[0], dtype=bool)
        num = float(np.linalg.norm(grad_phi[mask]))
        den = float(np.linalg.norm(u[mask]))
        return num / max(den, 1.0e-12)
    except Exception as exc:
        print(f"  [warn] helmholtz_residual failed: {exc}")
        return float("nan")


def analytic_divergence_zero() -> float:
    """Womersley u_z(r,t) has no z-dependence and zero radial/azimuthal flow."""
    return 0.0


def load_model(config_file: Path, ckpt_file: Path):
    import torch
    import yaml
    from flowgnn_aorta.models import FlowGAT
    from flowgnn_aorta.models.flow_sage import FlowSAGE

    model_registry = {"flowgat": FlowGAT, "flow_sage": FlowSAGE}
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obj = torch.load(ckpt_file, map_location=device)
    sd = obj["model"]
    equivariant_head = "head.lin_vec_edge.weight" in sd

    arch = str(cfg.get("model", {}).get("arch", "flowgat")).lower()
    model_cls = model_registry[arch]
    model = model_cls(
        node_in=int(cfg["model"].get("node_in") or 11),
        edge_in=int(cfg["model"].get("edge_in") or 5),
        hidden=int(cfg["model"]["hidden"]),
        heads=int(cfg["model"]["heads"]),
        layers=int(cfg["model"]["layers"]),
        dropout=0.0,
        attn_bias_beta=float(cfg["model"].get("attn_bias_beta", 1.5)),
        out=3,
        equivariant_head=equivariant_head,
        equivariant_basis=int(cfg["model"].get("equivariant_basis", 8)),
        hard_no_slip=bool(cfg["model"].get("hard_no_slip", True)),
        graph_stem_type=str(cfg["model"].get("graph_stem_type", "none")),
        graph_stem_layers=int(cfg["model"].get("graph_stem_layers", 0)),
        graph_stem_dropout=float(cfg["model"].get("graph_stem_dropout", 0.0)),
        graph_stem_residual=bool(cfg["model"].get("graph_stem_residual", True)),
    ).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, device


def make_data_object(
    x: np.ndarray,
    pos: np.ndarray,
    y: np.ndarray,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    wall_mask: np.ndarray,
    wall_normal: np.ndarray,
):
    import torch
    from flowgnn_aorta.data.vmr_loader import Data

    return Data(
        x=torch.from_numpy(x.astype(np.float32, copy=False)),
        pos=torch.from_numpy(pos.astype(np.float32, copy=False)),
        y=torch.from_numpy(y.astype(np.float32, copy=False)),
        edge_index=torch.from_numpy(edge_index.astype(np.int64, copy=False)),
        edge_attr=torch.from_numpy(edge_attr.astype(np.float32, copy=False)),
        wall_mask=torch.from_numpy(wall_mask.astype(np.bool_, copy=False)),
        wall_normal=torch.from_numpy(wall_normal.astype(np.float32, copy=False)),
    )


def forward_chunk(
    model,
    device,
    x: np.ndarray,
    pos: np.ndarray,
    y: np.ndarray,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    wall_mask: np.ndarray,
    wall_normal: np.ndarray,
) -> np.ndarray:
    import torch

    batch = make_data_object(x, pos, y, edge_index, edge_attr, wall_mask, wall_normal)
    batch = batch.to(device)
    with torch.no_grad():
        pred = model(batch)
    if not torch.isfinite(pred).all():
        raise FloatingPointError("non-finite model prediction")
    out = pred.detach().cpu().numpy().astype(np.float32)
    del batch, pred
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def diagnose_chunk(
    pos: np.ndarray,
    y: np.ndarray,
    pred: np.ndarray,
    edge_index: np.ndarray,
    wall_mask: np.ndarray,
    *,
    div_k: int,
    helm_k: int,
    helm_cg_tol: float,
    helm_cg_maxiter: int,
    skip_helmholtz: bool,
) -> Dict[str, float]:
    div_pred, L, U, n_mask = normalized_divergence(
        pos, pred, y, wall_mask, k_neighbors=int(div_k)
    )
    div_true, _, _, _ = normalized_divergence(
        pos, y, y, wall_mask, k_neighbors=int(div_k)
    )

    if skip_helmholtz:
        helm_pred = float("nan")
        helm_true = float("nan")
    else:
        helm_pred = helmholtz_residual(
            pos, pred, edge_index, wall_mask,
            k_neighbors=int(helm_k), tol=float(helm_cg_tol),
            maxiter=int(helm_cg_maxiter),
        )
        helm_true = helmholtz_residual(
            pos, y, edge_index, wall_mask,
            k_neighbors=int(helm_k), tol=float(helm_cg_tol),
            maxiter=int(helm_cg_maxiter),
        )

    return {
        "n_diag": float(n_mask),
        "mesh_L": float(L),
        "U_he": float(U),
        "div_pred": float(div_pred),
        "div_true": float(div_true),
        "div_analytical": float(analytic_divergence_zero()),
        "helm_pred": float(helm_pred),
        "helm_true": float(helm_true),
    }


def evaluate_case(
    case_path: Path,
    *,
    variant: str,
    resolution: str,
    seed: int,
    model,
    device,
    chunk_nodes: int,
    div_k: int,
    helm_k: int,
    helm_cg_tol: float,
    helm_cg_maxiter: int,
    skip_helmholtz: bool,
) -> Dict:
    z = np.load(case_path, allow_pickle=True)
    x_base = z["x"].astype(np.float32)
    pos = z["pos"].astype(np.float32)
    y = z["y"].astype(np.float32)
    edge_index = z["edge_index"].astype(np.int64)
    edge_attr = z["edge_attr"].astype(np.float32)
    wall_mask = z["wall_mask"].astype(bool)
    wall_normal = z["wall_normal"].astype(np.float32)
    u_char = float(z["u_char"])
    case_id = case_path.stem

    x = apply_womersley_variant_features(x_base, y, wall_mask, u_char, variant)
    chunks = contiguous_chunks(pos.shape[0], resolution, int(chunk_nodes))
    shard_rows: List[Dict[str, float]] = []

    print(
        f"  [case] {variant:13s} seed={seed:<5d} {resolution} {case_id} "
        f"N={pos.shape[0]} chunks={len(chunks)}"
    )
    for ci, idx in enumerate(chunks):
        start = int(idx[0])
        stop = int(idx[-1]) + 1
        ei, ea = subset_edges_contiguous(edge_index, edge_attr, start, stop)
        pred = forward_chunk(
            model, device,
            x[idx], pos[idx], y[idx], ei, ea, wall_mask[idx], wall_normal[idx],
        )
        diag = diagnose_chunk(
            pos[idx], y[idx], pred, ei, wall_mask[idx],
            div_k=int(div_k), helm_k=int(helm_k),
            helm_cg_tol=float(helm_cg_tol),
            helm_cg_maxiter=int(helm_cg_maxiter),
            skip_helmholtz=bool(skip_helmholtz),
        )
        diag["chunk"] = float(ci)
        diag["n_nodes"] = float(len(idx))
        diag["n_edges"] = float(ei.shape[1])
        shard_rows.append(diag)
        print(
            f"    chunk {ci+1:02d}/{len(chunks):02d}: "
            f"div_pred={diag['div_pred']:.6g} div_true={diag['div_true']:.6g} "
            f"helm_pred={diag['helm_pred']:.6g} helm_true={diag['helm_true']:.6g}"
        )

    row = {
        "variant": variant,
        "resolution": resolution,
        "seed": int(seed),
        "case": case_id,
        "n_nodes": int(pos.shape[0]),
        "n_chunks": int(len(chunks)),
        "mesh_L": _weighted_mean(shard_rows, "mesh_L", "n_diag"),
        "U_he": _weighted_mean(shard_rows, "U_he", "n_diag"),
        "div_pred": _weighted_mean(shard_rows, "div_pred", "n_diag"),
        "div_true": _weighted_mean(shard_rows, "div_true", "n_diag"),
        "div_analytical": _weighted_mean(shard_rows, "div_analytical", "n_diag"),
        "helm_pred": _weighted_mean(shard_rows, "helm_pred", "n_diag"),
        "helm_true": _weighted_mean(shard_rows, "helm_true", "n_diag"),
    }
    return row


def write_rows(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    keys = [
        "variant", "resolution", "seed", "case",
        "n_nodes", "n_chunks", "mesh_L", "U_he",
        "div_pred", "div_true", "div_analytical",
        "helm_pred", "helm_true",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[meshref] wrote {len(rows)} rows -> {path}")


def mode_eval(args: argparse.Namespace) -> None:
    data_dir = resolution_data_dir(Path(args.data_root), args.resolution)
    ckpt = checkpoint_path(Path(args.ckpt_root), args.variant, int(args.seed))
    cfg = config_path(Path(args.config_root), args.variant)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"missing data dir: {data_dir}")
    if not ckpt.exists():
        raise FileNotFoundError(f"missing checkpoint: {ckpt}")
    if not cfg.exists():
        raise FileNotFoundError(f"missing config: {cfg}")

    cases = load_split_cases(data_dir, args.split)
    if args.max_cases is not None:
        cases = cases[: int(args.max_cases)]

    model, device = load_model(cfg, ckpt)
    print(
        f"[meshref] eval variant={args.variant} seed={args.seed} "
        f"resolution={args.resolution} split={args.split} device={device}"
    )

    rows = []
    for case in cases:
        row = evaluate_case(
            data_dir / f"{case}.npz",
            variant=args.variant,
            resolution=args.resolution,
            seed=int(args.seed),
            model=model,
            device=device,
            chunk_nodes=int(args.chunk_nodes),
            div_k=int(args.div_k),
            helm_k=int(args.helm_k),
            helm_cg_tol=float(args.helm_cg_tol),
            helm_cg_maxiter=int(args.helm_cg_maxiter),
            skip_helmholtz=bool(args.skip_helmholtz),
        )
        rows.append(row)
        print(
            f"  -> {row['case']}: div_pred={row['div_pred']:.6g} "
            f"div_true={row['div_true']:.6g} helm_pred={row['helm_pred']:.6g}"
        )

    part = (
        Path(args.out_dir) / "parts" /
        f"{args.variant}_{args.resolution}_seed{int(args.seed)}.csv"
    )
    write_rows(rows, part)


def read_part_rows(out_dir: Path) -> List[Dict]:
    part_dir = out_dir / "parts"
    files = sorted(part_dir.glob("*.csv")) if part_dir.is_dir() else []
    if not files and (out_dir / "per_case.csv").exists():
        files = [out_dir / "per_case.csv"]

    rows_by_key: Dict[Tuple[str, str, int, str], Dict] = {}
    for f in files:
        with open(f, newline="") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                if not r:
                    continue
                for k, v in list(r.items()):
                    if k in ("variant", "resolution", "case"):
                        continue
                    try:
                        r[k] = int(v) if k in ("seed", "n_nodes", "n_chunks") else float(v)
                    except (TypeError, ValueError):
                        r[k] = float("nan")
                key = (str(r["variant"]), str(r["resolution"]), int(r["seed"]), str(r["case"]))
                rows_by_key[key] = r
    rows = list(rows_by_key.values())
    rows.sort(key=lambda r: (r["variant"], r["resolution"], int(r["seed"]), r["case"]))
    return rows


def aggregate_rows(rows: List[Dict]) -> List[Dict]:
    grouped: Dict[Tuple[str, str], List[Dict]] = {}
    for r in rows:
        grouped.setdefault((str(r["variant"]), str(r["resolution"])), []).append(r)

    out: List[Dict] = []
    for variant in VARIANTS:
        for resolution in RESOLUTIONS:
            group = grouped.get((variant, resolution), [])
            rec = {
                "variant": variant,
                "resolution": resolution,
                "n_cases": len({str(r["case"]) for r in group}),
                "n_seeds": len({int(r["seed"]) for r in group}),
            }
            for metric in ("div_pred", "div_true", "div_analytical", "helm_pred", "helm_true"):
                vals = [r.get(metric, float("nan")) for r in group]
                rec[f"{metric}_mean"] = _nanmean(vals)
                rec[f"{metric}_p90"] = _nanp(vals, 90.0)
            out.append(rec)
    return out


def write_aggregate(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AGG_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[meshref] aggregate -> {path} ({len(rows)} rows)")


def node_count_summary(per_case: List[Dict]) -> Dict[str, float]:
    by_res: Dict[str, List[float]] = {}
    for r in per_case:
        by_res.setdefault(str(r["resolution"]), []).append(float(r.get("n_nodes", float("nan"))))
    return {res: _nanmean(vals) for res, vals in by_res.items()}


def resolution_means(per_case: List[Dict], metric: str) -> Dict[str, float]:
    by_res: Dict[str, List[float]] = {}
    for r in per_case:
        by_res.setdefault(str(r["resolution"]), []).append(float(r.get(metric, float("nan"))))
    return {res: _nanmean(vals) for res, vals in by_res.items()}


def variant_lookup(agg: List[Dict], variant: str, resolution: str) -> Optional[Dict]:
    for r in agg:
        if r["variant"] == variant and r["resolution"] == resolution:
            return r
    return None


def format_float(x: float, digits: int = 4) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(v):
        return "nan"
    if abs(v) != 0.0 and (abs(v) < 1.0e-3 or abs(v) >= 1.0e3):
        return f"{v:.{digits}e}"
    return f"{v:.{digits}f}"


def write_summary(per_case: List[Dict], agg: List[Dict], path: Path) -> None:
    nodes = node_count_summary(per_case)
    div_true_by_res = resolution_means(per_case, "div_true")
    div_analytic_by_res = resolution_means(per_case, "div_analytical")
    mesh_l_by_res = resolution_means(per_case, "mesh_L")

    lines: List[str] = []
    lines.append("# Mesh-refinement diagnostic")
    lines.append("")
    lines.append("## Node counts")
    lines.append("")
    lines.append("| resolution | mean nodes/case | mean median-NN L |")
    lines.append("|---|---:|---:|")
    for res in RESOLUTIONS:
        lines.append(
            f"| {res} | {format_float(nodes.get(res, float('nan')), 1)} "
            f"| {format_float(mesh_l_by_res.get(res, float('nan')), 6)} |"
        )

    lines.append("")
    lines.append("## Estimator convergence")
    lines.append("")
    lines.append("| resolution | div_true_mean | div_analytical_mean |")
    lines.append("|---|---:|---:|")
    for res in RESOLUTIONS:
        lines.append(
            f"| {res} | {format_float(div_true_by_res.get(res, float('nan')), 6)} "
            f"| {format_float(div_analytic_by_res.get(res, float('nan')), 3)} |"
        )

    dt1 = div_true_by_res.get("1x", float("nan"))
    dt2 = div_true_by_res.get("2x", float("nan"))
    dt4 = div_true_by_res.get("4x", float("nan"))
    l1 = mesh_l_by_res.get("1x", float("nan"))
    l2 = mesh_l_by_res.get("2x", float("nan"))
    l4 = mesh_l_by_res.get("4x", float("nan"))
    if all(math.isfinite(v) and v > 0 for v in (dt1, dt2, dt4, l1, l2, l4)):
        lines.append("")
        lines.append(
            f"`div_true` ratios: 1x/2x={dt1/dt2:.2f}, 2x/4x={dt2/dt4:.2f}; "
            f"median-NN ratios: 1x/2x={l1/l2:.2f}, 2x/4x={l2/l4:.2f}."
        )

    lines.append("")
    lines.append("## Aggregate table")
    lines.append("")
    lines.append(
        "| variant | resolution | n_cases | n_seeds | div_pred_mean | "
        "div_true_mean | div_analytical_mean | helm_pred_mean | helm_true_mean |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in agg:
        lines.append(
            f"| {r['variant']} | {r['resolution']} | {r['n_cases']} | {r['n_seeds']} "
            f"| {format_float(r['div_pred_mean'], 6)} "
            f"| {format_float(r['div_true_mean'], 6)} "
            f"| {format_float(r['div_analytical_mean'], 3)} "
            f"| {format_float(r['helm_pred_mean'], 6)} "
            f"| {format_float(r['helm_true_mean'], 6)} |"
        )

    lines.append("")
    lines.append("## Criteria")
    lines.append("")
    stop = False
    lines.append("| variant | div plateau 4x vs 1x | status |")
    lines.append("|---|---:|---|")
    for variant in VARIANTS:
        r1 = variant_lookup(agg, variant, "1x")
        r4 = variant_lookup(agg, variant, "4x")
        if not r1 or not r4 or int(r1.get("n_cases", 0)) == 0 or int(r4.get("n_cases", 0)) == 0:
            rel = float("nan")
            status = "INCOMPLETE"
        else:
            base = float(r1["div_pred_mean"])
            fine = float(r4["div_pred_mean"])
            rel = abs(fine - base) / max(abs(base), 1.0e-12)
            if not math.isfinite(rel):
                status = "INCOMPLETE"
            else:
                status = "PASS" if rel <= 0.25 else "STOP"
            stop = stop or status == "STOP"
        lines.append(f"| {variant} | {format_float(rel, 4)} | {status} |")

    lines.append("")
    lines.append("| variant | helm_pred/helm_true at 4x | status |")
    lines.append("|---|---:|---|")
    for variant in ("withleak", "leak_dir_only"):
        r4 = variant_lookup(agg, variant, "4x")
        if not r4 or int(r4.get("n_cases", 0)) == 0:
            ratio = float("nan")
            status = "INCOMPLETE"
        else:
            ratio = float(r4["helm_pred_mean"]) / max(float(r4["helm_true_mean"]), 1.0e-12)
            if not math.isfinite(ratio):
                status = "INCOMPLETE"
            else:
                status = "PASS" if ratio >= 5.0 else "STOP"
            stop = stop or status == "STOP"
        lines.append(f"| {variant} | {format_float(ratio, 4)} | {status} |")

    if stop:
        lines.append("")
        lines.append("STOP: at least one hard-pass criterion failed; Section 3.8 needs re-framing.")
    else:
        lines.append("")
        lines.append("No STOP criterion was triggered by the available aggregate rows.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"[meshref] summary -> {path}")


def mode_aggregate(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    rows = read_part_rows(out_dir)
    write_rows(rows, out_dir / "per_case.csv")
    agg = aggregate_rows(rows)
    write_aggregate(agg, out_dir / "aggregate.csv")
    write_summary(rows, agg, out_dir / "summary.md")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("eval", "aggregate"), default="eval")
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--resolution", choices=RESOLUTIONS)
    ap.add_argument("--seed", type=int, choices=SEEDS)
    ap.add_argument("--split", default="test")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--ckpt_root", default="results/checkpoints")
    ap.add_argument("--config_root", default="configs")
    ap.add_argument("--out_dir", default="results/diagnostics/mesh_refinement")
    ap.add_argument("--chunk_nodes", type=int, default=DEFAULT_CHUNK_NODES)
    ap.add_argument("--div_k", type=int, default=16)
    ap.add_argument("--helm_k", type=int, default=16)
    ap.add_argument("--helm_cg_tol", type=float, default=1.0e-6)
    ap.add_argument("--helm_cg_maxiter", type=int, default=500)
    ap.add_argument("--skip_helmholtz", action="store_true")
    ap.add_argument("--max_cases", type=int, default=None)
    args = ap.parse_args()

    if args.mode == "eval":
        missing = [name for name in ("variant", "resolution", "seed") if getattr(args, name) is None]
        if missing:
            raise SystemExit(f"--mode eval requires: {', '.join('--' + m for m in missing)}")
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "eval":
        mode_eval(args)
    else:
        mode_aggregate(args)


if __name__ == "__main__":
    main()
