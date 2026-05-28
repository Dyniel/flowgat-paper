# -*- coding: utf-8 -*-
"""
Flow-GAT model with an optional equivariant vector output head.

Design:
  - A standard multi-head attention-based message passing stack operates
    on *scalar* node features to produce latent representations h_i ∈ R^H.
  - The final prediction is a 3-vector per node.  Instead of a plain
    linear projection R^H → R^3 (which is NOT rotation-equivariant — it
    learns a fixed basis), we provide an equivariant head that combines
    a learnt scalar "gate" g_i with a *basis of geometric vectors*
    constructed from edge directions.  Concretely:

        û_i  =  Σ_k  g_i^{(k)} ·  v_i^{(k)}                       (1)

    where v_i^{(k)} are equivariant vector basis features computed from
    the graph and node positions (see `_vector_basis`), and g_i^{(k)}
    are scalar weights emitted by the invariant backbone.  Because
    v_i^{(k)} rotates with the input and g_i^{(k)} is invariant, û_i
    rotates with the input — the output is SO(3)-equivariant.

  - Edge features edge_attr are concatenated into the attention
    compatibility score as a bias — this is the "attn_bias" term in the
    original config, made explicit here.

  - You can request the plain (non-equivariant) head by setting
    `equivariant_head=False`; both are wired so rotation augmentation
    works either way, but the equivariant head dramatically improves
    rotation robustness on unseen orientations.

The call signature accepts `(batch, w)` or `(batch)` for compatibility
with the existing training loop; `w` is not used internally — it is left
as a hook in case you want to e.g. gate attention on HV later.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from torch_geometric.nn import MessagePassing, GCNConv, GraphConv, SAGEConv
    from torch_geometric.utils import softmax as pyg_softmax
    _HAS_PYG = True
except Exception:
    _HAS_PYG = False


# ---------------------------------------------------------------------------
#  A thin PyG-based GATv2-style layer with edge-feature bias
# ---------------------------------------------------------------------------

if _HAS_PYG:

    class LocalGraphStem(nn.Module):
        """
        Cheap local smoothing / propagation before the attention stack.

        The stem is intentionally simple and residual: it gives the model a
        low-variance local inductive bias without replacing the downstream
        edge-biased attention layers.  It is disabled by default so existing
        checkpoints remain compatible.
        """

        def __init__(
            self,
            hidden: int,
            kind: str = "none",
            layers: int = 0,
            dropout: float = 0.0,
            residual: bool = True,
        ):
            super().__init__()
            kind = (kind or "none").lower()
            self.kind = kind
            self.n_layers = int(layers)
            self.dropout = float(dropout)
            self.residual = bool(residual)

            if kind in ("none", "off", "") or self.n_layers <= 0:
                self.layers = nn.ModuleList()
                self.norms = nn.ModuleList()
                self.kind = "none"
                self.n_layers = 0
                return

            convs = []
            for _ in range(self.n_layers):
                if kind == "sage":
                    convs.append(SAGEConv(hidden, hidden))
                elif kind in ("graph", "graphconv"):
                    convs.append(GraphConv(hidden, hidden))
                elif kind == "gcn":
                    convs.append(GCNConv(hidden, hidden, add_self_loops=True, normalize=True))
                else:
                    raise ValueError(f"Unknown graph stem kind: {kind!r}")
            self.layers = nn.ModuleList(convs)
            self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(self.n_layers)])

        def forward(self, h: Tensor, edge_index: Tensor) -> Tensor:
            if self.kind == "none":
                return h
            for conv, ln in zip(self.layers, self.norms):
                h_new = conv(h, edge_index)
                h_new = F.silu(h_new)
                if self.dropout > 0.0:
                    h_new = F.dropout(h_new, p=self.dropout, training=self.training)
                h = ln(h + h_new) if self.residual else ln(h_new)
            return h

    class EdgeBiasedGATLayer(MessagePassing):
        """
        GATv2-style attention with an additive edge-feature bias:

            e_ij = LeakyReLU( W_q · h_i + W_k · h_j + W_e · f_ij )
            a_ij = softmax_j( e_ij · w_a )
            m_ij = W_v · h_j  +  W_e2 · f_ij
            h_i' = || Σ_j a_ij · m_ij

        where f_ij is the edge feature vector (e.g. [dx, dy, dz, |r|]).
        """

        def __init__(
            self,
            in_dim: int,
            out_dim: int,
            heads: int,
            edge_dim: int = 0,
            dropout: float = 0.0,
            attn_bias_beta: float = 1.5,
            concat: bool = True,
        ):
            super().__init__(aggr="add", node_dim=0)
            assert out_dim % heads == 0
            self.heads = heads
            self.per_head = out_dim // heads
            self.concat = concat
            self.attn_bias_beta = float(attn_bias_beta)
            self.edge_dim = edge_dim

            self.lin_q = nn.Linear(in_dim, heads * self.per_head, bias=False)
            self.lin_k = nn.Linear(in_dim, heads * self.per_head, bias=False)
            self.lin_v = nn.Linear(in_dim, heads * self.per_head, bias=False)
            self.lin_a = nn.Parameter(torch.empty(heads, self.per_head))
            nn.init.xavier_uniform_(self.lin_a)

            if edge_dim > 0:
                self.lin_e = nn.Linear(edge_dim, heads * self.per_head, bias=False)
                self.lin_e2 = nn.Linear(edge_dim, heads * self.per_head, bias=False)
            else:
                self.lin_e = None
                self.lin_e2 = None

            self.dropout = float(dropout)
            self.out_proj = nn.Linear(heads * self.per_head, out_dim)

        def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None) -> Tensor:
            q = self.lin_q(x).view(-1, self.heads, self.per_head)
            k = self.lin_k(x).view(-1, self.heads, self.per_head)
            v = self.lin_v(x).view(-1, self.heads, self.per_head)
            e_bias = None
            e_val  = None
            if self.lin_e is not None and edge_attr is not None:
                e_bias = self.lin_e(edge_attr).view(-1, self.heads, self.per_head)
                e_val  = self.lin_e2(edge_attr).view(-1, self.heads, self.per_head)
            out = self.propagate(edge_index, q=q, k=k, v=v, e_bias=e_bias, e_val=e_val)
            out = out.view(-1, self.heads * self.per_head)
            out = self.out_proj(out)
            return out

        def message(
            self,
            q_i: Tensor, k_j: Tensor, v_j: Tensor,
            e_bias: Optional[Tensor], e_val: Optional[Tensor],
            index: Tensor, ptr: Optional[Tensor], size_i: Optional[int],
        ) -> Tensor:
            x = q_i + k_j
            if e_bias is not None:
                x = x + self.attn_bias_beta * e_bias
            x = F.leaky_relu(x, negative_slope=0.2)
            # attention score per head
            alpha = (x * self.lin_a).sum(dim=-1)                # [E, heads]
            alpha = pyg_softmax(alpha, index=index, ptr=ptr, num_nodes=size_i)
            if self.dropout > 0.0:
                alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            m = v_j
            if e_val is not None:
                m = m + e_val
            return m * alpha.unsqueeze(-1)                      # [E, heads, per_head]

else:
    # PyG-less fallback: a minimal scatter-based GAT.  Only for smoke testing.
    class LocalGraphStem(nn.Module):  # type: ignore
        def __init__(self, *a, **k):
            super().__init__()
            if (k.get("kind") or "none") not in ("none", "off", "") and int(k.get("layers") or 0) > 0:
                raise RuntimeError(
                    "torch_geometric is required for graph_stem_type != 'none'."
                )

        def forward(self, h: Tensor, edge_index: Tensor) -> Tensor:
            return h

    class EdgeBiasedGATLayer(nn.Module):  # type: ignore
        def __init__(self, *a, **k):
            super().__init__()
            raise RuntimeError(
                "torch_geometric is required. Install with:\n"
                "  pip install torch-geometric"
            )


# ---------------------------------------------------------------------------
#  Equivariant vector basis and output head
# ---------------------------------------------------------------------------

def _node_mean_edge_direction(pos: Tensor, edge_index: Tensor) -> Tensor:
    """
    Mean outgoing edge direction at each node — one equivariant vector.
    Shape: [N, 3].  Deterministic, cheap, rotates with the input.
    """
    src, dst = edge_index[0], edge_index[1]
    d = pos[src] - pos[dst]                                      # [E, 3]
    d = d / torch.linalg.vector_norm(d, dim=-1, keepdim=True).clamp_min(1.0e-12)
    N = pos.shape[0]
    out = torch.zeros(N, 3, device=pos.device, dtype=pos.dtype)
    cnt = torch.zeros(N, 1, device=pos.device, dtype=pos.dtype)
    out.index_add_(0, dst, d)
    cnt.index_add_(0, dst, torch.ones(d.shape[0], 1, device=pos.device, dtype=pos.dtype))
    out = out / cnt.clamp_min(1.0)
    return out


def _vector_basis(
    h: Tensor,
    pos: Tensor,
    edge_index: Tensor,
    lin_vec_edge: nn.Linear,
) -> Tensor:
    """
    Build an equivariant vector basis of size K at each node by
    scalar-gating edge directions:

        V_i^{(k)}  =  Σ_j  (a_ij^{(k)}) · (x_j − x_i)

    where a_ij^{(k)} is a scalar computed from (h_i, h_j).  Each V^{(k)}
    rotates with the input — they form the basis vectors of the
    equivariant output head.  K is determined by lin_vec_edge.out_features.
    """
    src, dst = edge_index[0], edge_index[1]
    r = pos[src] - pos[dst]                                       # [E, 3]

    # Invariant gating signal:  concat of node hidden states (both endpoints)
    # is fed through a linear layer producing K coefficients per edge.
    h_pair = torch.cat([h[dst], h[src]], dim=-1)                  # [E, 2H]
    a = lin_vec_edge(h_pair)                                      # [E, K]
    K = a.shape[-1]
    # Broadcast: [E, K, 1] * [E, 1, 3]  →  [E, K, 3]
    contrib = a.unsqueeze(-1) * r.unsqueeze(-2)                   # [E, K, 3]

    N = pos.shape[0]
    V = torch.zeros(N, K, 3, device=pos.device, dtype=pos.dtype)
    V.index_add_(0, dst, contrib)
    return V                                                      # [N, K, 3]


class EquivariantVectorHead(nn.Module):
    """
    Produces an SO(3)-equivariant 3-vector per node from an invariant
    hidden state h_i and the graph geometry.

        û_i = Σ_k  g_i^{(k)} · V_i^{(k)}

    where g is an invariant MLP of h_i and V is the vector basis above.
    """
    def __init__(self, hidden: int, basis_size: int = 8):
        super().__init__()
        self.basis_size = int(basis_size)
        # Edge gating MLP that feeds the basis construction
        self.lin_vec_edge = nn.Linear(2 * hidden, basis_size, bias=True)
        # Invariant scalar gate g_i^{(k)} from node state
        self.gate = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, basis_size),
        )

    def forward(self, h: Tensor, pos: Tensor, edge_index: Tensor) -> Tensor:
        V = _vector_basis(h, pos, edge_index, self.lin_vec_edge)  # [N, K, 3]
        # Normalize each basis vector to unit length so that the gradient
        # through g (the gate) has full magnitude regardless of V's scale.
        # Without this, ||V|| ≈ 1e-4 m at init → ∂loss/∂g ≈ 0 → gate never learns.
        V = V / torch.linalg.vector_norm(V, dim=-1, keepdim=True).clamp_min(1e-8)
        g = self.gate(h)                                          # [N, K]
        u = (g.unsqueeze(-1) * V).sum(dim=-2)                     # [N, 3]
        return u


# ---------------------------------------------------------------------------
#  Full model
# ---------------------------------------------------------------------------

class FlowGAT(nn.Module):
    """
    Flow-GAT surrogate for node-wise 3-D velocity prediction on vascular
    graphs.  Supports an optional equivariant vector head.

    Forward signature accepts `(batch, w)` or `(batch)` — `w` is ignored
    at forward time (it is used in the loss).
    """
    def __init__(
        self,
        node_in: int,
        edge_in: Optional[int] = None,
        hidden: int = 128,
        heads: int = 4,
        layers: int = 8,
        dropout: float = 0.0,
        attn_bias_beta: float = 1.5,
        out: int = 3,                         # must be 3 for velocity
        equivariant_head: bool = True,
        equivariant_basis: int = 8,
        hard_no_slip: bool = True,
        graph_stem_type: str = "none",
        graph_stem_layers: int = 0,
        graph_stem_dropout: float = 0.0,
        graph_stem_residual: bool = True,
        **kw,                                 # swallow legacy consts kwargs
    ):
        super().__init__()
        assert out == 3, "FlowGAT predicts a 3-vector per node."
        self.hidden = hidden
        self.hard_no_slip = bool(hard_no_slip)

        self.node_encoder = nn.Sequential(
            nn.Linear(node_in, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.edge_encoder = (
            nn.Sequential(nn.Linear(edge_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            if edge_in else None
        )
        self.graph_stem = LocalGraphStem(
            hidden=hidden,
            kind=graph_stem_type,
            layers=graph_stem_layers,
            dropout=graph_stem_dropout,
            residual=graph_stem_residual,
        )

        e_dim = hidden if self.edge_encoder is not None else 0

        self.layers = nn.ModuleList([
            EdgeBiasedGATLayer(
                in_dim=hidden, out_dim=hidden, heads=heads,
                edge_dim=e_dim, dropout=dropout, attn_bias_beta=attn_bias_beta,
            )
            for _ in range(layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])

        if equivariant_head:
            self.head = EquivariantVectorHead(hidden, basis_size=equivariant_basis)
            self._head_kind = "equivariant"
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 3),
            )
            self._head_kind = "linear"

    # --------------------------------------------------------------
    def encode(self, batch) -> Tensor:
        """Return final hidden embeddings [N, hidden] without applying the prediction head."""
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = getattr(batch, "edge_attr", None)

        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr) if (self.edge_encoder is not None and edge_attr is not None) else None
        h = self.graph_stem(h, edge_index)

        for layer, ln in zip(self.layers, self.layer_norms):
            h_new = layer(h, edge_index, edge_attr=e)
            h = ln(h + h_new)
        return h

    def forward(self, batch, w: Optional[Tensor] = None) -> Tensor:
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = getattr(batch, "edge_attr", None)
        pos = getattr(batch, "pos", None)
        if pos is None:
            raise ValueError(
                "FlowGAT requires `batch.pos` (node positions in R^3). "
                "Populate it in the dataset."
            )

        h = self.encode(batch)

        if self._head_kind == "equivariant":
            u = self.head(h, pos, edge_index)                     # [N, 3]
        else:
            u = self.head(h)

        # Hard no-slip: zero out wall nodes if the last feature is wall flag.
        if self.hard_no_slip and x.shape[-1] >= 1:
            wall = (x[:, -1] > 0.5).to(u.dtype).unsqueeze(-1)     # [N, 1]
            u = u * (1.0 - wall)

        return u
