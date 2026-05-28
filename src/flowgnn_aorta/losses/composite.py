# -*- coding: utf-8 -*-
"""
Composite training loss:

    L = λ_data · L_data(HV-weighted vector MSE)
      + λ_div  · L_div  (soft incompressibility)
      + λ_wall · L_wall (soft no-slip; complements the hard projection)
      + λ_mom  · L_mom  (steady momentum residual — optional)
      + λ_mag  · L_mag  (direct HE speed-relative error, for KPI alignment)

Each term is returned separately so the training loop can log them and run
ablations simply by zero-ing a coefficient in the config.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict

import torch
from torch import Tensor

from .physics import PhysicsTerms


# -----------------------------------------------------------------------------
#  HV (high-velocity) weights — same shape as the legacy formulation but
#  gathered in one place.  Weights depend on the ground-truth speed only,
#  so they do not leak prediction-dependent information into the loss.
# -----------------------------------------------------------------------------

def hv_weights(
    y: Tensor,
    u_char: float,
    p: float,
    w_min: float,
    w_max: float,
) -> Tensor:
    with torch.no_grad():
        speed = torch.linalg.vector_norm(y, ord=2, dim=-1)
        w = (speed / max(float(u_char), 1.0e-6)).pow(float(p))
        w = torch.clamp(w, min=float(w_min), max=float(w_max))
    return w


def weighted_vector_mse(pred: Tensor, target: Tensor, w: Tensor) -> Tensor:
    per_node = ((pred - target) ** 2).sum(dim=-1)             # [N]
    denom = w.sum().clamp_min(1.0e-8)
    return (per_node * w).sum() / denom


def he_speed_relative_loss(
    pred: Tensor,
    y: Tensor,
    he_mask: Tensor,
    eps: float = 1.0e-6,
) -> Tensor:
    """
    Direct optimisation pressure on the main KPI surrogate:
    speed-relative error on HE nodes only.
    """
    if he_mask.dtype != torch.bool:
        he_mask = he_mask > 0.5
    if he_mask.sum() == 0:
        return pred.new_zeros(())
    sp = torch.linalg.vector_norm(pred[he_mask], dim=-1)
    sy = torch.linalg.vector_norm(y[he_mask], dim=-1).clamp_min(eps)
    r = (sp - sy).abs() / sy
    return r.pow(2).mean()


# -----------------------------------------------------------------------------
#  Composite loss
# -----------------------------------------------------------------------------

@dataclass
class CompositeLossConfig:
    # data terms
    lambda_data: float = 1.0
    lambda_mag:  float = 0.0        # KPI-aligned aux term
    # physics terms
    lambda_div:  float = 1.0
    lambda_wall: float = 1.0        # on top of hard projection
    lambda_mom:  float = 0.0        # momentum residual (slower)
    # HV weighting
    hv_p:        float = 2.0
    hv_w_min:    float = 0.01
    hv_w_max:    float = 50.0
    near_wall_boost: float = 1.0
    # physical scales
    u_char:      float = 1.0
    length_char: float = 1.0
    nu:          float = 3.3e-6     # m²/s — blood at 37 °C
    rho:         float = 1060.0     # kg/m³
    # HE definition (used for mag-term only; evaluator uses its own)
    he_percentile: float = 0.80


class CompositeLoss:
    """
    Stateless callable that computes the composite loss and returns a dict
    of unweighted per-term values plus the weighted total.  Works with
    float32 and AMP.
    """

    def __init__(self, cfg: CompositeLossConfig):
        self.cfg = cfg
        self.physics = PhysicsTerms(
            u_char=cfg.u_char,
            length_char=cfg.length_char,
            nu=cfg.nu,
            rho=cfg.rho,
            compute_momentum=(cfg.lambda_mom > 0.0),
        )

    def __call__(
        self,
        pred: Tensor,
        y: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        wall_mask: Optional[Tensor] = None,
        p_grad: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        c = self.cfg
        out: Dict[str, Tensor] = {}

        # HV weights (may optionally boost wall)
        w = hv_weights(y, c.u_char, c.hv_p, c.hv_w_min, c.hv_w_max)
        if wall_mask is not None and c.near_wall_boost != 1.0:
            wb = wall_mask.to(w.dtype) if wall_mask.dtype != w.dtype else wall_mask.float()
            w = w * (1.0 + (c.near_wall_boost - 1.0) * wb)

        # Data term
        L_data = weighted_vector_mse(pred, y, w)
        out["data"] = L_data

        # Aux magnitude term on HE (cheap, stabilises the primary KPI)
        if c.lambda_mag > 0.0:
            with torch.no_grad():
                speed = torch.linalg.vector_norm(y, dim=-1)
                q = torch.quantile(speed, float(c.he_percentile))
                he_mask = speed >= q
            out["mag"] = he_speed_relative_loss(pred, y, he_mask)
        else:
            out["mag"] = pred.new_zeros(())

        # Physics terms
        phys = self.physics(
            pred, pos, edge_index,
            wall_mask=wall_mask, p_grad=p_grad, weight=w,
        )
        out.update(phys)   # "div", "wall", "mom"

        # Weighted total
        total = (
            c.lambda_data * out["data"]
            + c.lambda_mag  * out["mag"]
            + c.lambda_div  * out["div"]
            + c.lambda_wall * out["wall"]
            + c.lambda_mom  * out["mom"]
        )
        out["total"] = total
        return out
