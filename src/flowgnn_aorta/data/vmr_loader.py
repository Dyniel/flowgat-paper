# -*- coding: utf-8 -*-
"""
VMR (Vascular Model Repository) preprocessed graph dataset.

This module does NOT parse raw VTP/VTU files — that is done once, offline,
by `scripts/preprocess_vmr.py`, which converts each case into a compact
`.npz` with a fixed schema (see below).  Keeping preprocessing separate
from training avoids repeated disk traversal, makes train/val/test splits
deterministic, and enables hermetic reruns.

Each `.npz` file must contain:
  - x            [N, F]   node features (normalised).  Last channel MUST
                          be the binary wall flag (0 core, 1 wall).  The
                          feature list expected is documented in
                          README.md §"Features".
  - pos          [N, 3]   node positions in metres
  - y            [N, 3]   ground-truth velocity in m/s
  - edge_index   [2, E]   PyG-style (src, dst) pairs
  - edge_attr    [E, De]  edge features (normalised)
  - wall_mask    [N]      bool, redundant with x[:,-1] > 0.5 (kept for safety)
  - wall_normal  [N, 3]   outward wall normals (unit, zero elsewhere)
  - p_grad       [N, 3]   OPTIONAL pressure gradient, used for the momentum
                          residual loss.  If absent, momentum loss is
                          disabled for that sample.
  - case_id      string   patient/case identifier (for split diagnostics)
  - u_char       float    characteristic speed used for HV/physics scaling
  - length_char  float    characteristic length (e.g. aortic diameter)

The dataset returns PyG `Data` objects, ready for batching with the PyG
DataLoader.  Split by `case_id` is enforced at construction time.
"""

from __future__ import annotations
import os
import glob
import json
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------------------
#  Spatial subgraph sampler
# --------------------------------------------------------------------------

def _spatial_subgraph(
    data: "Data",
    n_sample: int,
    importance_beta: float = 0.0,
) -> "Data":
    """
    Extract a spatially-contiguous subgraph of n_sample nodes.

    Seeds on a random lumen node, then takes the n_sample spatially nearest
    neighbours (O(N) via np.argpartition — no full sort needed).
    Edges where both endpoints fall inside the subset are retained.
    All scalar case-level metadata (u_char, length_char, case_id) is copied
    verbatim so HV-weighting and logging remain consistent with the full case.

    importance_beta > 0 enables velocity-magnitude weighted center sampling
    (FlowGAT-v5): center drawn with prob ∝ (||y||/u_char)^beta, restricted to
    lumen nodes. This biases training subgraphs toward jet / stenosis regions
    where PP@10 is bottlenecked, without leaking gradient information (the
    weights are precomputed from y but only used for sampling).
    """
    pos = data.pos.numpy()          # [N, 3]
    N = pos.shape[0]
    if N <= n_sample:
        return data

    wall = data.wall_mask.numpy()
    lumen_idx = np.where(~wall)[0]
    if len(lumen_idx) == 0:
        lumen_idx = np.arange(N)

    if importance_beta > 0.0 and hasattr(data, "y") and data.y is not None:
        y = data.y.numpy()
        speed = np.linalg.norm(y, axis=-1)
        u_char = float(getattr(data, "u_char", 1.0))
        if u_char <= 0.0:
            u_char = 1.0
        p = (speed[lumen_idx] / u_char) ** float(importance_beta)
        # Mix with uniform so even low-velocity centers are occasionally
        # explored (avoids overfitting to a small set of jets).
        p = 0.85 * p + 0.15 * np.ones_like(p) * float(p.mean() if p.sum() > 0 else 1.0)
        s = p.sum()
        if not np.isfinite(s) or s <= 0:
            center = int(np.random.choice(lumen_idx))
        else:
            center = int(np.random.choice(lumen_idx, p=p / s))
    else:
        center = int(np.random.choice(lumen_idx))

    dists = ((pos - pos[center]) ** 2).sum(axis=1)
    sub_idx = np.sort(np.argpartition(dists, n_sample)[:n_sample])

    node_map = np.full(N, -1, dtype=np.int64)
    node_map[sub_idx] = np.arange(n_sample, dtype=np.int64)

    ei = data.edge_index.numpy()    # [2, E]
    keep = (node_map[ei[0]] >= 0) & (node_map[ei[1]] >= 0)
    new_ei = torch.from_numpy(
        np.stack([node_map[ei[0, keep]], node_map[ei[1, keep]]])
    )

    t = torch.from_numpy(sub_idx)
    kw = dict(
        x=data.x[t], pos=data.pos[t], y=data.y[t],
        edge_index=new_ei,
        edge_attr=data.edge_attr[keep],
        wall_mask=data.wall_mask[t],
        wall_normal=data.wall_normal[t],
        u_char=data.u_char,
        length_char=data.length_char,
        case_id=data.case_id,
    )
    p_grad = getattr(data, "p_grad", None)
    kw["p_grad"] = data.p_grad[t] if p_grad is not None else None
    return Data(**kw)

try:
    from torch_geometric.data import Data
    _HAS_PYG = True
except Exception:
    _HAS_PYG = False
    class Data:  # type: ignore
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)
        def to(self, device):
            for k, v in vars(self).items():
                if hasattr(v, "to"):
                    setattr(self, k, v.to(device))
            return self


# --------------------------------------------------------------------------
#  Flow-axis subgraph sampler (global-coverage alternative to KNN ball)
# --------------------------------------------------------------------------

def _flow_axis_subgraph(data: "Data", n_sample: int, n_bins: int = 40) -> "Data":
    """
    Extract subgraph of n_sample nodes sampled uniformly along the aortic axis.

    x[:,0] is the normalized axial distance from inlet [0,1].  By sampling
    uniformly across axial bins we guarantee the subgraph spans the full
    vessel: inlet → stenosis/CoA jet → outlet.  This gives the GNN global
    context about the flow path without graph coarsening.

    Unlike _spatial_subgraph (random KNN ball), this sampler never misses
    the high-velocity jet because it always samples from the stenosis band.
    """
    axis = data.x[:, 0].numpy()     # axial distance from inlet, ≈[0,1]
    N = len(axis)
    if N <= n_sample:
        return data

    a_min, a_max = float(axis.min()), float(axis.max())
    bins = np.linspace(a_min, a_max, n_bins + 1)
    per_bin = max(1, n_sample // n_bins)

    sampled: List[int] = []
    for i in range(n_bins):
        in_bin = np.where((axis >= bins[i]) & (axis < bins[i + 1]))[0]
        if len(in_bin) == 0:
            continue
        k = min(per_bin, len(in_bin))
        sampled.extend(np.random.choice(in_bin, k, replace=False).tolist())

    sub_idx = np.unique(sampled)
    # Fill remaining slots with random nodes to reach n_sample
    if len(sub_idx) < n_sample:
        remain = np.setdiff1d(np.arange(N), sub_idx)
        extra_n = min(n_sample - len(sub_idx), len(remain))
        if extra_n > 0:
            extra = np.random.choice(remain, extra_n, replace=False)
            sub_idx = np.concatenate([sub_idx, extra])
    sub_idx = np.sort(sub_idx[:n_sample])

    node_map = np.full(N, -1, dtype=np.int64)
    node_map[sub_idx] = np.arange(len(sub_idx), dtype=np.int64)

    ei = data.edge_index.numpy()
    keep = (node_map[ei[0]] >= 0) & (node_map[ei[1]] >= 0)
    new_ei = torch.from_numpy(
        np.stack([node_map[ei[0, keep]], node_map[ei[1, keep]]])
    )

    t = torch.from_numpy(sub_idx)
    kw = dict(
        x=data.x[t], pos=data.pos[t], y=data.y[t],
        edge_index=new_ei,
        edge_attr=data.edge_attr[keep],
        wall_mask=data.wall_mask[t],
        wall_normal=data.wall_normal[t],
        u_char=data.u_char,
        length_char=data.length_char,
        case_id=data.case_id,
    )
    p_grad = getattr(data, "p_grad", None)
    kw["p_grad"] = data.p_grad[t] if p_grad is not None else None
    return Data(**kw)


# --------------------------------------------------------------------------
#  Main dataset class
# --------------------------------------------------------------------------

class VMRGraphDataset(Dataset):
    """
    Map-style PyTorch dataset over preprocessed VMR `.npz` files.

    Args:
        root:        directory containing the .npz files
        split_file:  optional path to a JSON file with
                       {"train": [case_id, ...], "val": [...], "test": [...]}
                     If provided AND `split` is one of those keys, only
                     files whose filename (without .npz) is in the list
                     are included.
        split:       "train" / "val" / "test" (used with split_file)
        strict:      if True, raise on any missing required field.
                     if False, log a warning and skip malformed files.
    """
    REQUIRED = [
        "x", "pos", "y", "edge_index", "edge_attr",
        "wall_mask", "wall_normal", "u_char", "length_char",
    ]

    def __init__(
        self,
        root: str,
        split_file: Optional[str] = None,
        split: Optional[str] = None,
        strict: bool = True,
        sample_nodes: Optional[int] = None,
        importance_beta: float = 0.0,
        subgraph_mode: str = "knn",
    ):
        super().__init__()
        self.root = root
        self.strict = strict
        self.importance_beta = float(importance_beta)
        if subgraph_mode not in ("knn", "axis"):
            raise ValueError(f"subgraph_mode must be 'knn' or 'axis', got {subgraph_mode!r}")
        self.subgraph_mode = subgraph_mode

        all_paths = sorted(glob.glob(os.path.join(root, "*.npz")))
        if split_file is not None and split is not None:
            with open(split_file, "r") as f:
                splits = json.load(f)
            if split not in splits:
                raise ValueError(f"split '{split}' not in split file {list(splits.keys())}")
            keep = set(splits[split])
            paths = [p for p in all_paths if os.path.splitext(os.path.basename(p))[0] in keep]
            if not paths:
                raise FileNotFoundError(
                    f"No .npz files in {root} match split '{split}' (split file: {split_file})."
                )
            self.paths = paths
        else:
            if not all_paths:
                raise FileNotFoundError(f"No .npz files found in {root}.")
            self.paths = all_paths

        self.sample_nodes = sample_nodes
        # Cache metadata shape on first sample for dim probing by the trainer
        self._probe: Optional[Dict[str, Any]] = None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        z = np.load(path, allow_pickle=False)

        for k in self.REQUIRED:
            if k not in z:
                msg = f"{path}: missing required field '{k}'"
                if self.strict:
                    raise KeyError(msg)
                print(f"[VMRGraphDataset][warn] {msg}; returning empty graph")
                return self._empty_graph()

        x          = torch.from_numpy(z["x"].astype(np.float32))
        pos        = torch.from_numpy(z["pos"].astype(np.float32))
        y          = torch.from_numpy(z["y"].astype(np.float32))
        edge_index = torch.from_numpy(z["edge_index"].astype(np.int64))
        edge_attr  = torch.from_numpy(z["edge_attr"].astype(np.float32))
        wall_mask  = torch.from_numpy(z["wall_mask"].astype(np.bool_))
        wall_normal = torch.from_numpy(z["wall_normal"].astype(np.float32))

        u_char = float(z["u_char"])
        length_char = float(z["length_char"])

        d = Data(
            x=x, pos=pos, y=y,
            edge_index=edge_index, edge_attr=edge_attr,
            wall_mask=wall_mask, wall_normal=wall_normal,
        )
        # attach scalars as attributes (PyG preserves them across batching if
        # wrapped in a 1-D tensor; keep as python floats here since batch
        # size is 1 for large graphs)
        d.u_char = u_char
        d.length_char = length_char
        d.case_id = os.path.splitext(os.path.basename(path))[0]

        if "p_grad" in z.files:
            d.p_grad = torch.from_numpy(z["p_grad"].astype(np.float32))
        else:
            d.p_grad = None

        # record feature dims once (from full graph, before any subsampling)
        if self._probe is None:
            self._probe = {
                "node_in": int(x.shape[1]),
                "edge_in": int(edge_attr.shape[1]) if edge_attr.ndim == 2 else 0,
            }

        if self.sample_nodes is not None and d.pos.shape[0] > self.sample_nodes:
            if self.subgraph_mode == "axis":
                d = _flow_axis_subgraph(d, self.sample_nodes)
            else:
                d = _spatial_subgraph(
                    d,
                    self.sample_nodes,
                    importance_beta=self.importance_beta,
                )

        return d

    def probe_dims(self) -> Dict[str, int]:
        if self._probe is None:
            _ = self[0]
        return self._probe or {"node_in": 0, "edge_in": 0}

    @staticmethod
    def _empty_graph():
        return Data(
            x=torch.zeros(1, 1, dtype=torch.float32),
            pos=torch.zeros(1, 3, dtype=torch.float32),
            y=torch.zeros(1, 3, dtype=torch.float32),
            edge_index=torch.zeros(2, 0, dtype=torch.int64),
            edge_attr=torch.zeros(0, 1, dtype=torch.float32),
            wall_mask=torch.zeros(1, dtype=torch.bool),
            wall_normal=torch.zeros(1, 3, dtype=torch.float32),
        )


# --------------------------------------------------------------------------
#  Paired dataset for Global-Local training
# --------------------------------------------------------------------------

class PairedVMRDataset(Dataset):
    """
    Yields (coarse_data, fine_data) pairs for the same anatomical case.

    coarse_data : full coarsened graph from coarse_root (e.g. vmr_npz_v3, ~20K nodes).
                  Provides global vessel topology and boundary condition context.
    fine_data   : subgraph from fine_root (e.g. vmr_npz_v2, full resolution).
                  Subsampled with flow-axis or KNN sampler during training;
                  returned at full resolution during val/test.

    Both roots must share the same split.json case IDs.  Cases present in
    only one root are silently dropped (logged once on first __getitem__).

    coarse_nn_idx is precomputed per sample: for each fine node, the index
    of the nearest coarse node (used for O(1) global feature interpolation
    during the forward pass — avoids scipy in the training loop).
    """

    def __init__(
        self,
        fine_root: str,
        coarse_root: str,
        split_file: str,
        split: str,
        fine_sample_nodes: Optional[int] = 100_000,
        sampler: str = "flow_axis",   # "flow_axis" | "spatial"
    ):
        super().__init__()
        self.fine_sample_nodes = fine_sample_nodes
        self.sampler = sampler

        fine_ds   = VMRGraphDataset(fine_root,   split_file, split, sample_nodes=None)
        coarse_ds = VMRGraphDataset(coarse_root, split_file, split, sample_nodes=None)

        fine_id_to_idx   = {os.path.splitext(os.path.basename(p))[0]: i
                            for i, p in enumerate(fine_ds.paths)}
        coarse_id_to_idx = {os.path.splitext(os.path.basename(p))[0]: i
                            for i, p in enumerate(coarse_ds.paths)}

        common = sorted(set(fine_id_to_idx) & set(coarse_id_to_idx))
        self.pairs = [(fine_id_to_idx[k], coarse_id_to_idx[k]) for k in common]
        self._fine_ds   = fine_ds
        self._coarse_ds = coarse_ds
        self._probe: Optional[Dict[str, Any]] = None

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        from scipy.spatial import cKDTree  # available in conda py312 env

        fine_i, coarse_i = self.pairs[idx]
        fine_data   = self._fine_ds[fine_i]
        coarse_data = self._coarse_ds[coarse_i]

        # Apply fine subgraph sampling (training only; val/test: fine_sample_nodes=None)
        if (self.fine_sample_nodes is not None
                and fine_data.pos.shape[0] > self.fine_sample_nodes):
            if self.sampler == "flow_axis":
                fine_data = _flow_axis_subgraph(fine_data, self.fine_sample_nodes)
            else:
                fine_data = _spatial_subgraph(fine_data, self.fine_sample_nodes)

        # Precompute nearest-coarse-node index for each fine node (CPU, O(N log M))
        tree = cKDTree(coarse_data.pos.numpy())
        _, nn_idx = tree.query(fine_data.pos.numpy(), k=1, workers=1)
        fine_data.coarse_nn_idx = torch.from_numpy(nn_idx.astype(np.int64))

        if self._probe is None:
            self._probe = self._fine_ds.probe_dims()

        return coarse_data, fine_data

    def probe_dims(self) -> Dict[str, int]:
        if self._probe is None:
            _ = self[0]
        return self._probe or {"node_in": 0, "edge_in": 0}
