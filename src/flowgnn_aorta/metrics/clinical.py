# -*- coding: utf-8 -*-
"""
Clinical and technical evaluation metrics for predicted 3-D velocity fields
on aortic graphs.

Design principle: the main KPI is a *high-energy* (HE) success rate within
a relative tolerance — this matches the original project's rationale and is
what a cardiologist would read as "did we catch the fast jet correctly?".
Around that headline metric we add clinically interpretable diagnostics
that make the results meaningful outside an ML audience:

  1. PP@δ(HE) and AUC over tolerances  (existing)
  2. Strict tolerances {0.10, 0.05, 0.02, 0.01}
  3. Peak-velocity LOCATION error (mm)
  4. Peak-velocity MAGNITUDE error (relative)
  5. WSS magnitude error on wall nodes (Pa, R², bias)
  6. Pressure-drop estimate from the simplified Bernoulli  ΔP ≈ 4·v²
     (clinical convention) with error in mmHg
  7. Angular and speed-relative errors

All functions operate on torch tensors, are AMP-safe, and never silently
average over empty sets (they return NaN instead).
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import math
import torch
from torch import Tensor

from ..losses.physics import wls_velocity_gradient


# =============================================================================
#  HE mask (shared)
# =============================================================================

def he_mask_from_speed(y: Tensor, he_percentile: float) -> Tensor:
    speed = torch.linalg.vector_norm(y, dim=-1)
    if speed.numel() == 0:
        return torch.zeros_like(speed, dtype=torch.bool)
    q = torch.quantile(speed, float(he_percentile))
    return speed >= q


# =============================================================================
#  Per-node primitives
# =============================================================================

def relative_vector_error(
    pred: Tensor, y: Tensor, eps: float = 1.0e-6
) -> Tensor:
    diff = torch.linalg.vector_norm(pred - y, dim=-1)
    base = torch.linalg.vector_norm(y, dim=-1).clamp_min(eps)
    return diff / base


def speed_relative_error(
    pred: Tensor, y: Tensor, eps: float = 1.0e-6
) -> Tensor:
    sp = torch.linalg.vector_norm(pred, dim=-1)
    sy = torch.linalg.vector_norm(y, dim=-1).clamp_min(eps)
    return (sp - sy).abs() / sy


def angular_error_rad(
    pred: Tensor, y: Tensor, eps: float = 1.0e-6
) -> Tensor:
    dot = (pred * y).sum(dim=-1)
    pn = torch.linalg.vector_norm(pred, dim=-1)
    yn = torch.linalg.vector_norm(y, dim=-1)
    denom = (pn * yn).clamp_min(eps)
    c = (dot / denom).clamp(-1.0, 1.0)
    return torch.acos(c)


# =============================================================================
#  PP@δ and AUC over tolerances
# =============================================================================

def pp_at_tol(rel: Tensor, tol: float, mask: Optional[Tensor] = None) -> Tensor:
    if mask is None:
        return (rel <= tol).float().mean()
    m = mask if mask.dtype == torch.bool else (mask > 0.5)
    if m.sum() == 0:
        return rel.new_tensor(float("nan"))
    return (rel[m] <= tol).float().mean()


def pp_auc(
    pp_by_tol: Dict[float, Tensor],
    t_min: float,
    t_max: float,
) -> Tensor:
    tols = sorted([t for t in pp_by_tol.keys() if (t_min <= t <= t_max)])
    if len(tols) < 2:
        ref = next(iter(pp_by_tol.values()))
        return ref.new_tensor(float("nan"))
    area = pp_by_tol[tols[0]].new_zeros(())
    for a, b in zip(tols[:-1], tols[1:]):
        fa = pp_by_tol[a]
        fb = pp_by_tol[b]
        area = area + 0.5 * (fa + fb) * float(b - a)
    width = max(float(t_max - t_min), 1.0e-12)
    return area / width


# =============================================================================
#  Peak-velocity location & magnitude  (clinically important)
# =============================================================================

def peak_velocity_localisation(
    pred: Tensor,
    y: Tensor,
    pos: Tensor,
    topk: int = 1,
) -> Dict[str, float]:
    """
    Return a small dict describing how well the model localises the peak
    velocity.  `topk` lets you relax "predicted peak" to the centroid of the
    top-K predicted speeds, which is more stable for noisy predictions.

    Keys:
      - "peak_loc_mm"       Euclidean distance between predicted argmax
                            (or top-k centroid) and ground-truth argmax.
      - "peak_mag_rel"      (|û_peak| − |u_peak|) / |u_peak|   (signed)
      - "peak_mag_rel_abs"  |peak_mag_rel|
      - "peak_mag_gt"       |u_peak|     (for logging)
      - "peak_mag_pred"     |û_peak|
    """
    assert pred.shape == y.shape and pred.shape[-1] == 3
    assert pos.shape == y.shape
    if pred.shape[0] == 0:
        return {k: float("nan") for k in [
            "peak_loc_mm", "peak_mag_rel", "peak_mag_rel_abs",
            "peak_mag_gt", "peak_mag_pred",
        ]}

    sp_p = torch.linalg.vector_norm(pred, dim=-1)
    sp_y = torch.linalg.vector_norm(y, dim=-1)

    # Ground-truth peak
    i_gt = int(torch.argmax(sp_y).item())
    x_gt = pos[i_gt]

    # Predicted peak (top-k centroid of positions weighted by pred speed)
    k = max(1, min(int(topk), pred.shape[0]))
    vals, idx = torch.topk(sp_p, k=k)
    w = (vals / vals.sum().clamp_min(1.0e-12)).unsqueeze(-1)  # [k, 1]
    x_pred = (pos[idx] * w).sum(dim=0)                        # [3]

    # NOTE: pos is expected in metres; caller multiplies by 1000 if needed.
    peak_loc_m = torch.linalg.vector_norm(x_pred - x_gt).item()
    peak_loc_mm = 1000.0 * peak_loc_m

    peak_mag_gt = float(sp_y[i_gt].item())
    peak_mag_pred = float(sp_p[idx[0]].item())
    rel = (peak_mag_pred - peak_mag_gt) / max(peak_mag_gt, 1.0e-12)

    return {
        "peak_loc_mm": peak_loc_mm,
        "peak_mag_rel": rel,
        "peak_mag_rel_abs": abs(rel),
        "peak_mag_gt": peak_mag_gt,
        "peak_mag_pred": peak_mag_pred,
    }


# =============================================================================
#  Pressure drop from simplified Bernoulli (clinical convention)
# =============================================================================
#  In cardiology the modified Bernoulli equation
#
#        ΔP [mmHg]  ≈  4 · v² [m/s]
#
#  is the clinical default for estimating pressure drop across a stenosis
#  from peak velocity.  It is approximate but it is THE number clinicians
#  actually read — if our peak-velocity prediction is right, this number is
#  right, and that closes the loop to clinical practice.
# =============================================================================

def bernoulli_delta_p(peak_speed_m_s: float) -> float:
    return 4.0 * (peak_speed_m_s ** 2)


def peak_pressure_drop_error(pred: Tensor, y: Tensor) -> Dict[str, float]:
    """
    Returns the simplified Bernoulli ΔP (in mmHg) for predicted and ground-
    truth peak speed, along with their difference.  No pressure field is
    required — this is the *clinically relevant* derived quantity.
    """
    if pred.shape[0] == 0:
        return {k: float("nan") for k in
                ["dP_pred_mmHg", "dP_gt_mmHg", "dP_abs_err_mmHg", "dP_rel_err"]}
    v_gt   = float(torch.linalg.vector_norm(y,    dim=-1).max().item())
    v_pred = float(torch.linalg.vector_norm(pred, dim=-1).max().item())
    dP_gt   = bernoulli_delta_p(v_gt)
    dP_pred = bernoulli_delta_p(v_pred)
    abs_err = abs(dP_pred - dP_gt)
    rel_err = abs_err / max(dP_gt, 1.0e-6)
    return {
        "dP_pred_mmHg": dP_pred,
        "dP_gt_mmHg": dP_gt,
        "dP_abs_err_mmHg": abs_err,
        "dP_rel_err": rel_err,
    }


# =============================================================================
#  Wall shear stress (WSS) — τ_w  =  μ · (∂u_t/∂n)  at wall nodes
# =============================================================================
#  We approximate the velocity gradient tensor G at wall nodes via the WLS
#  estimator already used in the physics loss.  Then
#
#      τ_vec = μ · (G · n),                       full traction
#      τ_w   = τ_vec − (τ_vec · n) · n,           tangential part
#      |τ_w| = ||tangential traction||
#
#  where μ = ρ · ν is the dynamic viscosity.  For a faithful comparison we
#  compute τ_w from BOTH the predicted and the ground-truth velocity field
#  using the SAME estimator, so the comparison is unbiased by the finite-
#  difference operator.
# =============================================================================

def estimate_wss(
    u: Tensor,
    pos: Tensor,
    edge_index: Tensor,
    wall_mask: Tensor,
    wall_normals: Tensor,      # [N, 3], only used on wall nodes
    mu: float,                 # dynamic viscosity [Pa·s] = ρ·ν
) -> Tensor:
    """
    Returns  |τ_w|  at each wall node (others are 0).  Shape: [N].
    """
    if wall_mask.dtype != torch.bool:
        wall_mask = wall_mask > 0.5
    G = wls_velocity_gradient(u, pos, edge_index)     # [N, 3, 3]
    n = wall_normals                                   # [N, 3]
    n = n / torch.linalg.vector_norm(n, dim=-1, keepdim=True).clamp_min(1.0e-12)

    traction = torch.einsum("nab,nb->na", G, n) * float(mu)     # [N, 3]
    # tangential: τ_t = τ − (τ·n) n
    tn = (traction * n).sum(dim=-1, keepdim=True)               # [N, 1]
    tau_t = traction - tn * n                                   # [N, 3]
    mag = torch.linalg.vector_norm(tau_t, dim=-1)               # [N]
    mag = torch.where(wall_mask, mag, torch.zeros_like(mag))
    return mag


def wss_error_metrics(
    pred: Tensor, y: Tensor,
    pos: Tensor, edge_index: Tensor,
    wall_mask: Tensor, wall_normals: Tensor,
    mu: float,
) -> Dict[str, float]:
    """
    Compare |τ_w|(û) vs |τ_w|(u) on wall nodes.  Returns MAE (Pa), bias (Pa),
    NRMSE, R², and the global mean of ground-truth |τ_w| for scale.
    """
    if wall_mask.dtype != torch.bool:
        wall_mask = wall_mask > 0.5
    if wall_mask.sum() == 0:
        return {k: float("nan") for k in
                ["wss_mae_Pa", "wss_bias_Pa", "wss_nrmse", "wss_r2", "wss_mean_gt_Pa"]}

    tw_p = estimate_wss(pred, pos, edge_index, wall_mask, wall_normals, mu)[wall_mask]
    tw_y = estimate_wss(y,    pos, edge_index, wall_mask, wall_normals, mu)[wall_mask]

    diff = (tw_p - tw_y).float()
    mean_gt = float(tw_y.float().mean().item())
    mae  = float(diff.abs().mean().item())
    bias = float(diff.mean().item())
    nrmse = float(math.sqrt((diff ** 2).mean().item()) / max(mean_gt, 1.0e-9))

    y_mean = tw_y.float().mean()
    ss_res = ((tw_p - tw_y) ** 2).sum()
    ss_tot = ((tw_y - y_mean) ** 2).sum()
    r2 = float(1.0 - (ss_res / ss_tot.clamp_min(1.0e-12)).item())

    return {
        "wss_mae_Pa": mae,
        "wss_bias_Pa": bias,
        "wss_nrmse": nrmse,
        "wss_r2": r2,
        "wss_mean_gt_Pa": mean_gt,
    }
