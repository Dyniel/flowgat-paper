# -*- coding: utf-8 -*-
"""
Physics-informed loss terms for graph-discretised incompressible flow.

All operators are formulated directly on the graph (node positions + edges),
so no meshing/topology beyond edge_index / pos is required.  The divergence
and velocity-gradient estimator is a weighted moving least-squares (WLS)
fit of a local linear velocity field

    u(x)  ≈  u_i  +  G_i · (x - x_i)

around each node i, with weights w_ij = 1 / ||x_j - x_i||² over the graph
neighbourhood.  This is the standard first-order SPH-style gradient
estimator, consistent on unstructured meshes and numerically well-behaved
for scattered data.

The module deliberately returns each term separately (as plain scalars) so
the outer training loop can weight them, log them, and ablate them
independently.  It never modifies the prediction tensor in place.

References (physical background, not cited here):
  - Fatehi & Manzari, 2011, A remedy for numerical oscillations in
    weakly-compressible SPH (WLS gradient).
  - Temam, Navier-Stokes Equations (continuity, momentum).
"""

from __future__ import annotations
from typing import Optional, Dict

import torch
from torch import Tensor


# =============================================================================
#  Core: weighted least-squares velocity-gradient estimator on a graph.
# =============================================================================

def wls_velocity_gradient(
    u: Tensor,
    pos: Tensor,
    edge_index: Tensor,
    eps_reg: float = 1.0e-4,
) -> Tensor:
    """
    Estimate the 3×3 velocity gradient tensor G at every node by solving,
    per node i, the weighted least-squares problem

        G_i = argmin_G  Σ_{j∈N(i)}  w_ij  ||Δu_ij − G · Δx_ij||²

    where Δu_ij = u_j − u_i, Δx_ij = x_j − x_i, w_ij = 1 / ||Δx_ij||².

    Closed form:

        A_i = Σ_j  w_ij  Δx_ij  Δx_ijᵀ           (symmetric 3×3)
        B_i = Σ_j  w_ij  Δu_ij  Δx_ijᵀ           (general 3×3)
        G_i = B_i  A_i⁻¹

    A_i is regularised with `eps_reg · I` for numerical stability when
    neighbourhoods are nearly co-planar or degenerate.

    Args:
        u:          [N, 3] velocity per node.
        pos:        [N, 3] Cartesian position per node.
        edge_index: [2, E] graph edges.  By PyG convention edge_index[0] is
                    source and edge_index[1] is destination; we aggregate on
                    destination index, treating it as the centre node i and
                    source index as neighbour j.
        eps_reg:    Tikhonov regulariser for A.

    Returns:
        G: [N, 3, 3] with G[i, a, b] ≈ ∂u_a / ∂x_b at node i.
    """
    assert u.ndim == 2 and u.shape[-1] == 3, "u must be [N, 3]"
    assert pos.shape == u.shape, "pos must be [N, 3] matching u"
    assert edge_index.shape[0] == 2, "edge_index must be [2, E]"

    N = u.shape[0]
    src, dst = edge_index[0], edge_index[1]  # src = neighbour j, dst = centre i

    dx = pos[src] - pos[dst]                           # [E, 3]
    du = u[src]  - u[dst]                              # [E, 3]

    r2 = (dx * dx).sum(dim=-1, keepdim=True).clamp_min(1.0e-12)  # [E, 1]
    w  = 1.0 / r2                                      # [E, 1]

    # Outer products: we need shape [E, 3, 3]
    outer_xx = dx.unsqueeze(-1) * dx.unsqueeze(-2)     # [E, 3, 3]
    outer_ux = du.unsqueeze(-1) * dx.unsqueeze(-2)     # [E, 3, 3]

    # Per-edge contributions (weight broadcast over the 3×3)
    wA = w.unsqueeze(-1) * outer_xx                    # [E, 3, 3]
    wB = w.unsqueeze(-1) * outer_ux                    # [E, 3, 3]

    # Scatter-add into per-node A, B
    A = torch.zeros(N, 3, 3, device=u.device, dtype=u.dtype)
    B = torch.zeros(N, 3, 3, device=u.device, dtype=u.dtype)
    A.index_add_(0, dst, wA)
    B.index_add_(0, dst, wB)

    # Regularise and solve:  G = B · A⁻¹  ⇔  A Gᵀ = Bᵀ
    eye = torch.eye(3, device=u.device, dtype=u.dtype).expand(N, 3, 3)
    A_reg = A + eps_reg * eye
    G_T = torch.linalg.solve(A_reg, B.transpose(-1, -2))   # [N, 3, 3]
    G = G_T.transpose(-1, -2)
    return G


def divergence_from_grad(G: Tensor) -> Tensor:
    """∇·u = tr(G) = ∂u_x/∂x + ∂u_y/∂y + ∂u_z/∂z   →  [N]."""
    return G.diagonal(dim1=-2, dim2=-1).sum(dim=-1)


def laplacian_from_grad(
    u: Tensor,
    pos: Tensor,
    edge_index: Tensor,
    eps_reg: float = 1.0e-4,
) -> Tensor:
    """
    Component-wise graph Laplacian of the velocity field, ∇²u, per node.

    We approximate it by a WLS fit of a *local quadratic*, but in practice the
    cheap and numerically robust variant is a two-pass linear fit:
    first fit G = ∇u, then fit ∇G per-component, and take the trace.
    This is first-order accurate on a sufficiently rich neighbourhood and is
    mathematically equivalent to the divergence of the gradient.

    Returns: [N, 3]  with  Δu_a = ∂²u_a/∂x_b ∂x_b summed over b.
    """
    G = wls_velocity_gradient(u, pos, edge_index, eps_reg=eps_reg)   # [N, 3, 3]

    # For each velocity component a ∈ {0,1,2}, fit ∇(G[:, a, :]) and take trace.
    lap = torch.zeros_like(u)
    for a in range(3):
        g_a = G[:, a, :]                                              # [N, 3]
        H_a = wls_velocity_gradient(g_a, pos, edge_index, eps_reg=eps_reg)  # [N, 3, 3]
        lap[:, a] = H_a.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    return lap


# =============================================================================
#  Physics loss terms
# =============================================================================

def divergence_loss(
    u_pred: Tensor,
    pos: Tensor,
    edge_index: Tensor,
    weight: Optional[Tensor] = None,
    u_char: float = 1.0,
    length_char: float = 1.0,
    eps_reg: float = 1.0e-4,
) -> Tensor:
    """
    Soft incompressibility penalty:  L_div = mean( (L/U * ∇·û)² ).

    The (length_char / u_char) factor makes the term dimensionless, so λ
    has a physical interpretation ("how much worse than perfect dimensionless
    mass conservation is acceptable").  Pick:
      u_char   = characteristic velocity (peak systolic speed, m/s).
      length_char = characteristic length (aortic diameter, m).

    If `weight` is supplied (e.g. HV weights or HE mask as float), the mean
    is weighted.
    """
    G = wls_velocity_gradient(u_pred, pos, edge_index, eps_reg=eps_reg)
    div = divergence_from_grad(G)                         # [N]
    scale = float(length_char) / max(float(u_char), 1.0e-6)
    r = (scale * div).pow(2)                              # dimensionless

    if weight is None:
        return r.mean()
    w = weight.to(r.dtype).clamp_min(0.0)
    denom = w.sum().clamp_min(1.0e-8)
    return (r * w).sum() / denom


def no_slip_loss(
    u_pred: Tensor,
    wall_mask: Tensor,
    u_char: float = 1.0,
) -> Tensor:
    """
    Soft no-slip BC penalty:  L_wall = mean_{wall}( ||û/U||² ).

    Prefer combining this with the *hard* no-slip projection in
    `apply_hard_no_slip` below — a hard projection after forward is almost
    free, removes a whole class of "non-physical boundary leakage" critiques
    from reviewers, and this soft term then only has to discipline the first
    ring of core-flow nodes next to the wall.
    """
    if wall_mask.dtype != torch.bool:
        wall_mask = wall_mask > 0.5
    if wall_mask.sum() == 0:
        return u_pred.new_zeros(())
    u_w = u_pred[wall_mask] / max(float(u_char), 1.0e-6)
    return u_w.pow(2).sum(dim=-1).mean()


def apply_hard_no_slip(u_pred: Tensor, wall_mask: Tensor) -> Tensor:
    """
    Hard projection û[wall] ← 0.  Returns a *new* tensor (does not mutate
    the input).  Use this in the forward pass if you want to bake the BC
    into the prediction — it is differentiable via the identity on the
    complement.
    """
    if wall_mask.dtype != torch.bool:
        wall_mask = wall_mask > 0.5
    if wall_mask.sum() == 0:
        return u_pred
    mult = (~wall_mask).to(u_pred.dtype).unsqueeze(-1)    # [N, 1]
    return u_pred * mult


def momentum_residual_loss(
    u_pred: Tensor,
    pos: Tensor,
    edge_index: Tensor,
    *,
    nu: float,                      # kinematic viscosity  [m²/s]
    rho: float = 1060.0,            # blood density        [kg/m³]
    p_grad: Optional[Tensor] = None,# optional ∇p per node [N, 3]; None → drop
    weight: Optional[Tensor] = None,
    u_char: float = 1.0,
    length_char: float = 1.0,
    eps_reg: float = 1.0e-4,
    steady: bool = True,
) -> Tensor:
    """
    Residual of the steady incompressible Navier–Stokes momentum equation:

        r  =  (u · ∇) u  +  (1/ρ) ∇p  −  ν Δu       (steady)

    For a surrogate trained against converged CFD snapshots the steady
    version is the right target.  If `p_grad` is None we drop the pressure
    term (useful for pure "velocity consistency" regularisation — it won't
    punish the model for the unknown pressure field).

    The residual is non-dimensionalised by U²/L so λ is comparable across
    patients with different characteristic speeds.

    Returns:  a single scalar  mean( ||r̂||² )  (optionally weighted).
    """
    if not steady:
        raise NotImplementedError("Unsteady residual requires temporal data.")

    G   = wls_velocity_gradient(u_pred, pos, edge_index, eps_reg=eps_reg)   # [N, 3, 3]
    lap = laplacian_from_grad(u_pred, pos, edge_index, eps_reg=eps_reg)     # [N, 3]

    # Convective term (u·∇)u  =  G · u     (G[a,b] = ∂u_a/∂x_b)
    conv = torch.einsum("nab,nb->na", G, u_pred)                             # [N, 3]
    diff = float(nu) * lap                                                   # [N, 3]

    r = conv - diff                                                          # [N, 3]
    if p_grad is not None:
        r = r + p_grad / max(float(rho), 1.0e-12)

    # Non-dimensionalise by U²/L
    U = max(float(u_char),    1.0e-6)
    L = max(float(length_char), 1.0e-6)
    scale = L / (U * U)
    rn2 = (scale * r).pow(2).sum(dim=-1)                                      # [N]

    if weight is None:
        return rn2.mean()
    w = weight.to(rn2.dtype).clamp_min(0.0)
    return (rn2 * w).sum() / w.sum().clamp_min(1.0e-8)


# =============================================================================
#  Composite physics-term bag
# =============================================================================

class PhysicsTerms:
    """
    Lightweight container that evaluates all physics terms in one pass
    (re-using the gradient tensor where possible) and returns a dict of
    scalar losses.  The outer composite loss is responsible for weighting.

    Usage:
        pt = PhysicsTerms(u_char=1.2, length_char=0.025, nu=3.3e-6)
        losses = pt(u_pred, pos, edge_index, wall_mask=wall_mask)
        # losses["div"], losses["wall"], losses["mom"]
    """

    def __init__(
        self,
        u_char: float = 1.0,
        length_char: float = 1.0,
        nu: float = 3.3e-6,         # blood kin. visc. at 37°C (~3.3e-6 m²/s)
        rho: float = 1060.0,
        eps_reg: float = 1.0e-4,
        compute_momentum: bool = False,
    ):
        self.U   = float(u_char)
        self.L   = float(length_char)
        self.nu  = float(nu)
        self.rho = float(rho)
        self.eps = float(eps_reg)
        self.compute_momentum = bool(compute_momentum)

    def __call__(
        self,
        u_pred: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        *,
        wall_mask: Optional[Tensor] = None,
        p_grad: Optional[Tensor] = None,
        weight: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        out: Dict[str, Tensor] = {}

        # Gradient + divergence (shared)
        G = wls_velocity_gradient(u_pred, pos, edge_index, eps_reg=self.eps)
        div = divergence_from_grad(G)
        scale = self.L / max(self.U, 1.0e-6)
        div_loss = (scale * div).pow(2)
        if weight is None:
            out["div"] = div_loss.mean()
        else:
            w = weight.to(div_loss.dtype).clamp_min(0.0)
            out["div"] = (div_loss * w).sum() / w.sum().clamp_min(1.0e-8)

        # No-slip wall
        if wall_mask is not None:
            out["wall"] = no_slip_loss(u_pred, wall_mask, u_char=self.U)
        else:
            out["wall"] = u_pred.new_zeros(())

        # Momentum residual (optional, pricier)
        if self.compute_momentum:
            out["mom"] = momentum_residual_loss(
                u_pred, pos, edge_index,
                nu=self.nu, rho=self.rho, p_grad=p_grad,
                weight=weight, u_char=self.U, length_char=self.L,
                eps_reg=self.eps, steady=True,
            )
        else:
            out["mom"] = u_pred.new_zeros(())

        return out
