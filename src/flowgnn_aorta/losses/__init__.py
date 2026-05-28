# -*- coding: utf-8 -*-
from .physics import (
    wls_velocity_gradient,
    divergence_from_grad,
    laplacian_from_grad,
    divergence_loss,
    no_slip_loss,
    apply_hard_no_slip,
    momentum_residual_loss,
    PhysicsTerms,
)
from .composite import (
    hv_weights,
    weighted_vector_mse,
    he_speed_relative_loss,
    CompositeLoss,
    CompositeLossConfig,
)

__all__ = [
    "wls_velocity_gradient",
    "divergence_from_grad",
    "laplacian_from_grad",
    "divergence_loss",
    "no_slip_loss",
    "apply_hard_no_slip",
    "momentum_residual_loss",
    "PhysicsTerms",
    "hv_weights",
    "weighted_vector_mse",
    "he_speed_relative_loss",
    "CompositeLoss",
    "CompositeLossConfig",
]
