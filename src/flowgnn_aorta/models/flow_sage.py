# -*- coding: utf-8 -*-
"""
FlowSAGE — vanilla GraphSAGE baseline for the *architecture-independence*
check in Phase E.

Same forward signature as `FlowGAT`, identical no-slip post-processing and
optional equivariant head. Differences:
  - replaces multi-head attention with SAGEConv (mean-aggregator GraphSAGE)
  - does NOT use edge_attr in message passing (clean separation from FlowGAT)
  - simpler, smaller parameter count for given (hidden, layers)

Intended use: train this on every variant in the leakage decomposition
ablation (withleak / leak_dir_only / leak_mag_only / noleak / noleak_centerline)
to show that the *qualitative pattern* of asymmetric direction/magnitude
identifiability is independent of the specific GNN architecture.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

try:
    from torch_geometric.nn import SAGEConv
    _HAS_PYG = True
except Exception:
    _HAS_PYG = False

from .gat_phys import EquivariantVectorHead


class FlowSAGE(nn.Module):
    """Vanilla GraphSAGE baseline with the same I/O contract as FlowGAT."""

    def __init__(
        self,
        node_in: int,
        edge_in: Optional[int] = None,  # accepted but ignored
        hidden: int = 128,
        heads: int = 4,                 # accepted but ignored
        layers: int = 8,
        dropout: float = 0.0,
        attn_bias_beta: float = 1.5,    # accepted but ignored
        out: int = 3,
        equivariant_head: bool = False,
        equivariant_basis: int = 8,
        hard_no_slip: bool = True,
        **kw,
    ):
        super().__init__()
        if not _HAS_PYG:
            raise RuntimeError("FlowSAGE requires torch_geometric.")
        assert out == 3, "FlowSAGE predicts a 3-vector per node."

        self.hidden = int(hidden)
        self.hard_no_slip = bool(hard_no_slip)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.convs = nn.ModuleList([
            SAGEConv(hidden, hidden, aggr="mean") for _ in range(int(layers))
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(int(layers))])
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.act = nn.SiLU()

        if equivariant_head:
            self.head = EquivariantVectorHead(hidden, basis_size=int(equivariant_basis))
            self._head_kind = "equivariant"
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, 3),
            )
            self._head_kind = "linear"

    def encode(self, batch) -> Tensor:
        x = batch.x
        edge_index = batch.edge_index
        h = self.node_encoder(x)
        for conv, ln in zip(self.convs, self.layer_norms):
            h_new = self.act(conv(h, edge_index))
            h_new = self.dropout(h_new)
            h = ln(h + h_new)
        return h

    def forward(self, batch, w: Optional[Tensor] = None) -> Tensor:
        x = batch.x
        edge_index = batch.edge_index
        pos = getattr(batch, "pos", None)
        h = self.encode(batch)
        if self._head_kind == "equivariant":
            if pos is None:
                raise ValueError("equivariant head requires batch.pos")
            u = self.head(h, pos, edge_index)
        else:
            u = self.head(h)
        if self.hard_no_slip and x.shape[-1] >= 1:
            wall = (x[:, -1] > 0.5).to(u.dtype).unsqueeze(-1)
            u = u * (1.0 - wall)
        return u
