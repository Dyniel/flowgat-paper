# -*- coding: utf-8 -*-
"""
Global-Local FlowGAT — bidirectional two-scale architecture.

Motivation
----------
Standard subgraph sampling (KNN ball or flow-axis) gives local resolution but
loses global vessel context: a CoA jet depends on the stenosis-to-lumen area
ratio which spans the entire aorta.  Coarsening (v3) gives global context but
loses the spatial resolution needed to resolve the narrow jet.

This module implements a two-scale pipeline inspired by MultiScale MeshGraphNets
(Fortunato et al., 2022) and BSMS-GNN (Cao et al., 2023):

  Stage 1 — Global encoder   (coarse graph, ~20K nodes, full vessel)
    FlowGAT backbone → per-node hidden embeddings h_coarse [N_c, H_g]

  Stage 2 — Feature injection (nearest-neighbour interpolation)
    For each fine node, copy the embedding of its nearest coarse neighbour.
    Precomputed in the dataset as fine_data.coarse_nn_idx → O(1) at runtime.
    Conditioned fine features: [x_fine (11) | h_coarse_interp (H_g)] = 11+H_g.
    Wall flag stays as the LAST channel of x_fine → hard no-slip still works.

  Stage 3 — Local decoder    (fine subgraph, 100K nodes, flow-axis sampled)
    FlowGAT with extended node_in=11+H_g → velocity [N_f, 3]

Both stages are trained jointly end-to-end.  Gradients flow from the fine
prediction through NN-interpolation (index_select, differentiable) back into
the global encoder.
"""

from __future__ import annotations
import copy
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .gat_phys import FlowGAT


class GlobalLocalFlowGAT(nn.Module):
    """
    Two-scale surrogate: coarse global encoder + fine local decoder.

    Args
    ----
    node_in          : original node feature dimension (e.g. 11)
    edge_in          : edge feature dimension (e.g. 5)
    global_hidden    : hidden size for the global encoder
    global_heads     : attention heads for the global encoder
    global_layers    : GAT layers for the global encoder
    local_hidden     : hidden size for the local decoder
    local_heads      : attention heads for the local decoder
    local_layers     : GAT layers for the local decoder
    dropout          : dropout probability (shared)
    hard_no_slip     : zero wall-node predictions (applied externally to handle
                       the extended feature layout)
    """

    def __init__(
        self,
        node_in: int,
        edge_in: int,
        global_hidden: int = 128,
        global_heads: int = 4,
        global_layers: int = 6,
        local_hidden: int = 256,
        local_heads: int = 8,
        local_layers: int = 10,
        dropout: float = 0.0,
        attn_bias_beta: float = 1.5,
        hard_no_slip: bool = True,
        **kw,
    ):
        super().__init__()
        self.hard_no_slip = hard_no_slip
        self.global_hidden = global_hidden

        # Global encoder: standard FlowGAT backbone (no head needed)
        self.global_enc = FlowGAT(
            node_in=node_in,
            edge_in=edge_in,
            hidden=global_hidden,
            heads=global_heads,
            layers=global_layers,
            dropout=dropout,
            attn_bias_beta=attn_bias_beta,
            equivariant_head=False,
            hard_no_slip=False,
        )

        # Local decoder: FlowGAT with extended node input
        # Layout: [x_fine[:-1] (10 scalars) | h_coarse_interp (H_g) | wall_flag (1)]
        # = node_in + global_hidden features, wall flag still last.
        local_node_in = node_in + global_hidden
        self.local_dec = FlowGAT(
            node_in=local_node_in,
            edge_in=edge_in,
            hidden=local_hidden,
            heads=local_heads,
            layers=local_layers,
            dropout=dropout,
            attn_bias_beta=attn_bias_beta,
            equivariant_head=False,
            hard_no_slip=True,   # x[:, -1] is still the wall flag (see injection below)
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        coarse_batch,
        fine_batch,
        w: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args
        ----
        coarse_batch : PyG Data, full coarsened graph (~20K nodes)
        fine_batch   : PyG Data, fine subgraph (100K train / full at eval)
                       Must have attribute `coarse_nn_idx` [N_fine] (int64)
                       precomputed by PairedVMRDataset.
        w            : ignored (kept for API compatibility with train loop)
        """
        # 1. Global encoding
        h_coarse = self.global_enc.encode(coarse_batch)    # [N_c, H_g]

        # 2. NN interpolation: fine → nearest coarse embedding
        nn_idx = fine_batch.coarse_nn_idx                  # [N_f], int64
        h_interp = h_coarse[nn_idx]                        # [N_f, H_g]

        # 3. Inject global context, keeping wall flag as the LAST feature:
        #    [x_fine[:, :-1] | h_interp | x_fine[:, -1:]]
        x_f = fine_batch.x
        x_cond = torch.cat([x_f[:, :-1], h_interp, x_f[:, -1:]], dim=-1)

        # 4. Local decoding on conditioned subgraph (shallow-copy to avoid
        #    mutating the caller's batch in case it is reused)
        fine_cond = _shallow_copy_data(fine_batch)
        fine_cond.x = x_cond

        u = self.local_dec(fine_cond)                      # [N_f, 3]
        return u

    # ------------------------------------------------------------------
    def param_count(self) -> dict:
        g = sum(p.numel() for p in self.global_enc.parameters())
        l = sum(p.numel() for p in self.local_dec.parameters())
        return {"global": g, "local": l, "total": g + l}


# --------------------------------------------------------------------------
#  Tiny helper
# --------------------------------------------------------------------------

def _shallow_copy_data(data):
    """Return a shallow copy of a PyG Data object with independent __dict__."""
    new = object.__new__(type(data))
    new.__dict__.update(data.__dict__)
    return new
