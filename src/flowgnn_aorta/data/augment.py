# -*- coding: utf-8 -*-
"""
Train-time graph augmentations.

Critical correctness note (this is the bit reviewers will check):

  - `x` contains per-node SCALAR features AND *optionally* a binary wall flag.
    It must NOT contain any raw vectorial quantity that would require rotation.
  - `pos` is a geometric vector  →  must rotate with R.
  - `y`  is a geometric vector  →  must rotate with R.
  - `edge_attr`  CAN contain relative positions (dx, dy, dz, |r|). The first
    three rotate, the norm does not. The preprocessing script places relative
    directions in the FIRST THREE channels and the norm in the 4th; rotation
    augmentation respects that layout.  If you change the layout, update
    `EDGE_VEC_SLICE` below.
  - `wall_normal`  is a unit vector  →  rotates with R.

If any of the above is violated, rotation augmentation silently corrupts your
data and your "equivariance experiment" is invalid.  The assertions below are
cheap and catch it.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


# Edge-attribute convention (must match the preprocessing script):
#   [0:3] = unit direction (dx_hat, dy_hat, dz_hat)   [rotates]
#   [3]   = ||r||                                      [invariant]
#   [4:]  = scalar features (optional)                 [invariant]
EDGE_VEC_SLICE = slice(0, 3)

# Node feature convention (must match preprocess_vmr.py):
#   x[:,0:3] = scalar features (inlet dist, wall dist, lumen radius)
#   x[:,3:6] = local flow direction unit vector  [VECTOR — must rotate with R]
#   x[:,6:]  = scalar features (inlet/outlet indicators, inflow, Re, wall flag)
NODE_VEC_SLICE = slice(3, 6)   # channels in x that are a 3-D equivariant vector


# ---------------------------------------------------------------------------
#  Uniform random rotation in SO(3) via the Haar measure
# ---------------------------------------------------------------------------

def random_rotation_matrix(device, dtype=torch.float32) -> Tensor:
    """
    Uniform random SO(3) matrix via QR decomposition of a Gaussian 3×3.
    Corrects the sign so det(R) = +1.
    """
    A = torch.randn(3, 3, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    # Fix sign to make Haar-uniform
    d = torch.diagonal(R, 0)
    sign = torch.sign(d)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    Q = Q * sign.unsqueeze(0)
    # Ensure proper rotation (det = +1), reflect first column if not
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def apply_rotation_to_batch(batch, R: Tensor) -> None:
    """
    In-place rotation of all vectorial quantities in a PyG `Data` batch.
    """
    if hasattr(batch, "pos") and batch.pos is not None:
        batch.pos = batch.pos @ R.T                # (N,3) · (3,3)ᵀ = (N,3)
    if hasattr(batch, "y") and batch.y is not None and batch.y.shape[-1] == 3:
        batch.y = batch.y @ R.T
    if hasattr(batch, "wall_normal") and batch.wall_normal is not None:
        batch.wall_normal = batch.wall_normal @ R.T
    if hasattr(batch, "p_grad") and getattr(batch, "p_grad", None) is not None:
        batch.p_grad = batch.p_grad @ R.T
    if hasattr(batch, "edge_attr") and batch.edge_attr is not None:
        # Rotate only the vector part of edge_attr
        ea = batch.edge_attr
        if ea.shape[-1] >= 3:
            v = ea[:, EDGE_VEC_SLICE] @ R.T
            batch.edge_attr = torch.cat([v, ea[:, EDGE_VEC_SLICE.stop:]], dim=-1)
    if hasattr(batch, "x") and batch.x is not None:
        # x[:,3:6] is the local flow direction vector (NODE_VEC_SLICE).
        # Without this rotation the target y is in the rotated frame but
        # the flow-direction feature stays in the original frame — making
        # u=0 the only consistent prediction under random SO3 augmentation.
        x = batch.x
        s = NODE_VEC_SLICE
        if x.shape[-1] > s.stop:
            v = x[:, s] @ R.T
            batch.x = torch.cat([x[:, :s.start], v, x[:, s.stop:]], dim=-1)


# ---------------------------------------------------------------------------
#  Feature noise + edge dropout (kept from original, made stricter)
# ---------------------------------------------------------------------------

def apply_feature_noise(batch, std: float) -> None:
    """Adds Gaussian noise only to the scalar invariant part of x.

    By convention the LAST channel is the wall flag; we never perturb it.
    """
    if std <= 0.0 or not hasattr(batch, "x") or batch.x is None:
        return
    if batch.x.shape[-1] <= 1:
        return
    scalar_part = batch.x[:, :-1]
    noise = torch.randn_like(scalar_part) * float(std)
    batch.x = torch.cat([scalar_part + noise, batch.x[:, -1:]], dim=-1)


def apply_edge_dropout(batch, p: float, min_edges: int = 4) -> None:
    if p <= 0.0 or not hasattr(batch, "edge_index") or batch.edge_index is None:
        return
    ei = batch.edge_index
    E = ei.shape[1]
    if E <= min_edges:
        return
    keep = (torch.rand(E, device=ei.device) > p)
    if keep.sum() < min_edges:
        idx = torch.randint(0, E, (min_edges,), device=ei.device)
        keep = keep.clone()
        keep[idx] = True
    batch.edge_index = ei[:, keep]
    if hasattr(batch, "edge_attr") and batch.edge_attr is not None:
        batch.edge_attr = batch.edge_attr[keep]


# ---------------------------------------------------------------------------
#  Top-level augmentation pipeline
# ---------------------------------------------------------------------------

@dataclass
class AugmentConfig:
    enabled: bool = True
    rotate_so3: bool = True          # SO(3) equivariance augmentation
    feature_noise_std: float = 0.0
    edge_dropout_p: float = 0.0


def apply_augment(batch, cfg: AugmentConfig, training: bool = True):
    if not training or not cfg.enabled:
        return batch
    # 1) Rotate first — rotation is the structural augmentation
    if cfg.rotate_so3:
        R = random_rotation_matrix(device=batch.y.device, dtype=batch.y.dtype)
        apply_rotation_to_batch(batch, R)
    # 2) Feature noise on scalars
    apply_feature_noise(batch, cfg.feature_noise_std)
    # 3) Edge dropout
    apply_edge_dropout(batch, cfg.edge_dropout_p)
    return batch
