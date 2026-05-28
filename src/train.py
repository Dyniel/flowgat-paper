# -*- coding: utf-8 -*-
"""
TrainV6 — FlowGAT-v6 grid runner.

Extends train_v5.py:

  * Optional physics-informed losses: divergence, soft no-slip, momentum
    residual.  Lambdas are configurable; setting them all to 0 reduces the
    training objective to v5.

  * subgraph_mode {"knn", "axis"} switch — pass-through to VMRGraphDataset.

  * Optional equivariant_head + SO(3) augment (via existing FlowGAT code).

  * Efficiency telemetry written to <ckpt_dir>/efficiency.json:
      mean_epoch_sec, peak_gpu_mb, n_params, eval_time_s, train_time_s,
      best_epoch, best_pp10_val, total_epochs.

  * --override key.path=value CLI flag for in-place YAML overrides
    (used by the grid SLURM script to avoid duplicating 16 near-identical
    YAML files).

All other behaviour (KNN subgraph, focal HV, edge-pair diff, importance β,
HV-weighted MSE) is inherited verbatim from train_v5.py.
"""

from __future__ import annotations
import os, sys, time, math, json, random, argparse, warnings
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import yaml

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings("once", message="`torch.cuda.amp")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "flowgnn_aorta"))

from flowgnn_aorta.data.vmr_loader import VMRGraphDataset
from flowgnn_aorta.data.augment import AugmentConfig, apply_augment
from flowgnn_aorta.models.gat_phys import FlowGAT
from flowgnn_aorta.models.flow_sage import FlowSAGE


_MODEL_REGISTRY = {
    "flowgat": FlowGAT,
    "flow_sage": FlowSAGE,
}
from flowgnn_aorta.losses.physics import (
    divergence_loss, no_slip_loss, momentum_residual_loss,
)


# ---------------------------------------------------------------------------
#  Losses (verbatim from train_v5.py)
# ---------------------------------------------------------------------------

def _hv_base(y, u_char, p, w_min, w_max):
    with torch.no_grad():
        speed = torch.linalg.vector_norm(y, dim=-1)
        w = (speed / max(u_char, 1e-6)).pow(p).clamp(w_min, w_max)
    return w


def focal_hv_weights(pred, y, u_char, p, w_min, w_max,
                     focal_gamma, focal_tau, enabled):
    base = _hv_base(y, u_char, p, w_min, w_max)
    if not enabled or focal_gamma <= 0.0:
        return base
    with torch.no_grad():
        err = torch.linalg.vector_norm(pred.detach() - y, dim=-1)
        ratio = err / max(focal_tau * u_char, 1e-6)
        focal = (1.0 + ratio).pow(focal_gamma).clamp(max=10.0)
    return base * focal


def weighted_mse(pred, y, w):
    per_node = (pred - y).pow(2).sum(dim=-1)
    return (per_node * w).sum() / w.sum().clamp_min(1e-8)


def edge_pair_diff_loss(pred, y, edge_index, w, max_edges=0):
    src = edge_index[0]
    dst = edge_index[1]
    E = src.shape[0]
    if E == 0:
        return pred.new_zeros(())
    if max_edges > 0 and E > max_edges:
        idx = torch.randperm(E, device=src.device)[:max_edges]
        src = src[idx]; dst = dst[idx]
    dpred = pred[src] - pred[dst]
    dy    = y[src]    - y[dst]
    per_e = (dpred - dy).pow(2).sum(dim=-1)
    we    = 0.5 * (w[src] + w[dst])
    return (per_e * we).sum() / we.sum().clamp_min(1e-8)


def euclidean_pair_diff_loss(
    pred,
    y,
    pos,
    w,
    k=8,
    anchors=512,
    max_pairs=4096,
    length_char=0.025,
    normalize_by_distance=True,
    chunk_size=128,
):
    """
    Multi-anchor Euclidean pair loss.

    Unlike edge_pair_diff_loss, pairs are drawn from spatial KNN in Cartesian
    space, not only from mesh edges.  This gives a cheap pressure on jet
    boundaries at a slightly larger physical radius while avoiding an O(N^2)
    all-pairs distance matrix.
    """
    N = pred.shape[0]
    k = int(k)
    anchors = int(anchors)
    if N <= 1 or k <= 0 or anchors <= 0:
        return pred.new_zeros(())

    anchors = min(anchors, N)
    max_pairs = int(max_pairs)
    if max_pairs > 0:
        anchors = min(anchors, max(1, max_pairs // max(k, 1)))

    with torch.no_grad():
        prob = w.detach().float().clamp_min(0.0)
        if not torch.isfinite(prob).all() or float(prob.sum().item()) <= 0.0:
            anchor_idx = torch.randperm(N, device=pred.device)[:anchors]
        else:
            if int((prob > 0).sum().item()) < anchors:
                prob = prob + prob.mean().clamp_min(1e-8) * 0.05
            anchor_idx = torch.multinomial(prob / prob.sum(), anchors, replacement=False)

        pos_f = pos.detach().float()
        neigh_chunks = []
        dist_chunks = []
        for start in range(0, anchors, int(chunk_size)):
            a = anchor_idx[start:start + int(chunk_size)]
            d = torch.cdist(pos_f[a], pos_f)
            # Exclude the anchor itself when it appears in the candidate set.
            local = torch.arange(a.numel(), device=pred.device)
            d[local, a] = float("inf")
            vals, idx = torch.topk(d, k=min(k, max(1, N - 1)), dim=1, largest=False)
            neigh_chunks.append(idx.reshape(-1))
            dist_chunks.append(vals.reshape(-1))
        neigh_idx = torch.cat(neigh_chunks, dim=0)
        pair_dist = torch.cat(dist_chunks, dim=0).clamp_min(1e-6)
        anchor_rep = anchor_idx.repeat_interleave(min(k, max(1, N - 1)))

    dpred = pred[anchor_rep] - pred[neigh_idx]
    dy = y[anchor_rep] - y[neigh_idx]
    err = dpred - dy
    if normalize_by_distance:
        scale = (pair_dist / max(float(length_char), 1e-6)).to(err.dtype).unsqueeze(-1).clamp_min(1e-3)
        err = err / scale
    per_pair = err.pow(2).sum(dim=-1)
    we = 0.5 * (w[anchor_rep] + w[neigh_idx])
    return (per_pair * we).sum() / we.sum().clamp_min(1e-8)


def rel_err(pred, y, eps=1e-6):
    return (torch.linalg.vector_norm(pred - y, dim=-1) /
            torch.linalg.vector_norm(y, dim=-1).clamp_min(eps))


def speed_err(pred, y, eps=1e-6):
    return ((torch.linalg.vector_norm(pred, dim=-1) -
             torch.linalg.vector_norm(y, dim=-1)).abs() /
            torch.linalg.vector_norm(y, dim=-1).clamp_min(eps))


def _auc(pp_by_tol, t_min, t_max):
    tols = sorted(t for t in pp_by_tol if t_min <= t <= t_max)
    if len(tols) < 2:
        return float("nan")
    area = sum(0.5 * (pp_by_tol[a] + pp_by_tol[b]) * (b - a)
               for a, b in zip(tols[:-1], tols[1:]))
    return area / max(t_max - t_min, 1e-12)


# ---------------------------------------------------------------------------
#  DDP / W&B / utility helpers
# ---------------------------------------------------------------------------

def ddp_setup():
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return None
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group("nccl")
    torch.cuda.set_device(local)
    return rank, world, local


def ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        try: dist.barrier()
        except Exception: pass
        dist.destroy_process_group()


def allreduce_(t):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


class _Null:
    def log(self, *a, **k): pass
    def finish(self): pass


def init_wandb(cfg, main_rank):
    if not main_rank:
        return _Null()
    try:
        import wandb
        os.makedirs(cfg["logging"]["wandb_dir"], exist_ok=True)
        return wandb.init(project=cfg["logging"]["project"],
                          name=cfg["logging"]["run_name"],
                          config=cfg)
    except Exception as e:
        print(f"[wandb] disabled: {e}", flush=True)
        return _Null()


def seed_everything(seed, deterministic=False):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def _batch_scalar(batch, name: str, default: float) -> float:
    val = getattr(batch, name, None)
    if val is None:
        return float(default)
    try:
        if torch.is_tensor(val):
            return float(val.detach().flatten()[0].cpu().item())
        return float(val)
    except Exception:
        return float(default)


def save_ckpt(path, model, opt, sched, scaler, epoch, best_pp):
    m = model.module if hasattr(model, "module") else model
    torch.save({"epoch": epoch, "best_pp": best_pp,
                "model": m.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "scaler": scaler.state_dict()}, path)


def _set_nested(d: dict, dotted: str, value: Any):
    parts = dotted.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    # try to coerce the value
    try:
        if value.lower() in ("true", "false"):
            value = (value.lower() == "true")
        elif "." in value or "e" in value.lower():
            value = float(value)
        else:
            value = int(value)
    except (ValueError, AttributeError):
        pass
    cur[parts[-1]] = value


# ---------------------------------------------------------------------------
#  Training loop
# ---------------------------------------------------------------------------

def train(cfg: dict) -> Dict[str, Any]:
    rank_ws = ddp_setup()
    main_rank = (rank_ws is None) or (rank_ws[0] == 0)
    eff: Dict[str, Any] = {}

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg["training"].get("tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(cfg["training"].get("tf32", True))
        torch.set_float32_matmul_precision("high")

    seed = int(cfg["training"]["seed"]) + (rank_ws[0] if rank_ws else 0)
    seed_everything(seed, bool(cfg["training"].get("deterministic", False)))

    wb = init_wandb(cfg, main_rank)

    # ---- data ----------------------------------------------------------------
    data_root      = cfg["data"]["train_dir"]
    split_file     = cfg["data"].get("split_file")
    sample_nodes   = cfg["data"].get("sample_nodes", None)
    importance_beta = float(cfg["data"].get("importance_beta", 0.0))
    subgraph_mode  = str(cfg["data"].get("subgraph_mode", "knn"))

    train_ds = VMRGraphDataset(
        root=data_root, split_file=split_file, split="train",
        sample_nodes=sample_nodes,
        importance_beta=importance_beta,
        subgraph_mode=subgraph_mode,
    )
    val_ds = VMRGraphDataset(
        root=cfg["data"].get("val_dir", data_root),
        split_file=split_file, split="val",
    )
    if main_rank:
        print(f"Dataset: {len(train_ds)} train  {len(val_ds)} val "
              f"sample_nodes={sample_nodes} importance_beta={importance_beta} "
              f"subgraph_mode={subgraph_mode} data_root={data_root}",
              flush=True)

    try:
        from torch_geometric.loader import DataLoader
    except ImportError:
        from torch.utils.data import DataLoader

    nw = int(cfg["training"].get("num_workers", 4))
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=(nw > 0))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=max(1, nw // 2), pin_memory=True,
                            persistent_workers=(max(1, nw // 2) > 0))

    # ---- model ---------------------------------------------------------------
    probe = train_ds[0]
    mc = cfg["model"]
    if mc.get("node_in") is None:
        mc["node_in"] = int(probe.x.shape[1])
    if mc.get("edge_in") is None:
        mc["edge_in"] = int(probe.edge_attr.shape[1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if rank_ws:
        device = torch.device(f"cuda:{rank_ws[2]}")

    arch = str(mc.get("arch", "flowgat")).lower()
    if arch not in _MODEL_REGISTRY:
        raise ValueError(f"unknown model.arch={arch!r}; available: {sorted(_MODEL_REGISTRY)}")
    model_cls = _MODEL_REGISTRY[arch]
    # don't pass 'arch' to the constructor (it isn't a kwarg of the model)
    model = model_cls(**{k: v for k, v in mc.items() if v is not None and k != "arch"}).to(device)
    if rank_ws:
        model = DDP(model, device_ids=[rank_ws[2]], find_unused_parameters=False)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    eff["n_params"] = int(n_params)
    if main_rank:
        stem = f"{mc.get('graph_stem_type', 'none')}x{mc.get('graph_stem_layers', 0)}"
        print(f"[model] params={n_params/1e6:.2f}M equivariant_head={mc.get('equivariant_head', False)} "
              f"stem={stem}", flush=True)

    opt = AdamW(model.parameters(),
                lr=float(cfg["training"]["lr"]),
                weight_decay=float(cfg["training"]["weight_decay"]))
    t_max = int(cfg["training"].get("cosine_t_max",
                                    int(cfg["training"]["epochs"]) - 100))
    sched = CosineAnnealingLR(opt, T_max=t_max, eta_min=1e-6)
    amp = bool(cfg["training"].get("amp", True)) and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    autocast = lambda: torch.amp.autocast("cuda", enabled=amp)

    # ---- hyperparams ---------------------------------------------------------
    u_char  = float(cfg["data"]["u_char"])
    L_char  = float(cfg["data"].get("length_char", 0.025))
    nu_kin  = float(cfg["data"].get("nu", 3.3e-6))
    rho     = float(cfg["data"].get("rho", 1060.0))
    use_case_scales = bool(cfg["data"].get("use_case_scales", False))
    if main_rank and use_case_scales:
        print("[data] using per-case u_char/length_char for training loss scaling", flush=True)

    hv      = cfg["loss"]["hv"]
    hv_p    = float(hv["p"])
    hv_wmin = float(hv["min_weight"])
    hv_wmax = float(hv["max_weight"])
    nb      = float(hv.get("near_wall_boost", 1.0))
    he_pct  = float(cfg["loss"].get("he_percentile", 0.80))
    rel_tol = float(hv.get("rel_tol", 0.10))

    focal_cfg = cfg["loss"].get("focal", {}) or {}
    focal_enable    = bool(focal_cfg.get("enabled", False))
    focal_gamma     = float(focal_cfg.get("gamma", 1.0))
    focal_tau       = float(focal_cfg.get("tau",   0.10))
    focal_warmup_ep = int(focal_cfg.get("warmup_epochs", 50))

    pair_cfg = cfg["loss"].get("pair", {}) or {}
    lambda_pair    = float(pair_cfg.get("lambda", 0.0))
    pair_max_edges = int(pair_cfg.get("max_edges", 0))

    euc_pair_cfg = cfg["loss"].get("euclidean_pair", {}) or {}
    lambda_euc_pair = float(euc_pair_cfg.get("lambda", 0.0))
    euc_pair_k = int(euc_pair_cfg.get("k", 8))
    euc_pair_anchors = int(euc_pair_cfg.get("anchors", 512))
    euc_pair_max_pairs = int(euc_pair_cfg.get("max_pairs", 4096))
    euc_pair_norm = bool(euc_pair_cfg.get("normalize_by_distance", True))
    euc_pair_warmup = int(euc_pair_cfg.get("warmup_epochs", 0))
    euc_pair_chunk = int(euc_pair_cfg.get("chunk_size", 128))

    phys_cfg = cfg["loss"].get("physics", {}) or {}
    lambda_div  = float(phys_cfg.get("lambda_div",  0.0))
    lambda_mom  = float(phys_cfg.get("lambda_mom",  0.0))
    lambda_wall = float(phys_cfg.get("lambda_wall", 0.0))
    phys_warmup = int(phys_cfg.get("warmup_epochs", 0))
    eps_reg     = float(phys_cfg.get("eps_reg", 1.0e-4))
    use_phys    = (lambda_div + lambda_mom + lambda_wall) > 0.0
    if main_rank and use_phys:
        print(f"[physics] λ_div={lambda_div} λ_mom={lambda_mom} λ_wall={lambda_wall} "
              f"warmup={phys_warmup}ep", flush=True)

    aug_cfg = AugmentConfig(
        rotate_so3=bool(cfg["augment"].get("rotate_so3", True)),
        feature_noise_std=float(cfg["augment"].get("feature_noise_std", 0.003)),
        edge_dropout_p=float(cfg["augment"].get("edge_dropout_p", 0.02)),
    )

    strict_tols = [float(t) for t in cfg["eval"].get("strict_tols", [0.10, 0.05, 0.02, 0.01])]
    auc_min, auc_max = [float(x) for x in cfg["eval"].get("auc_range", [0.0, 0.10])]
    eval_every = int(cfg["eval"]["every"])
    full_every = int(cfg["eval"]["full_every"])
    max_vbatch = int(cfg["eval"]["max_batches"])

    epochs     = int(cfg["training"]["epochs"])
    ckpt_dir   = cfg["training"]["ckpt_dir"]
    save_every = int(cfg["training"].get("save_every", 50))
    keep_last  = int(cfg["training"].get("keep_last", 3))
    resume     = bool(cfg["training"].get("resume", False))
    patience   = int(cfg["goals"].get("patience", 30))
    target_pp  = float(cfg["goals"].get("target_recall_he", 0.95))
    max_nodes  = cfg["training"].get("max_nodes")
    max_edges  = cfg["training"].get("max_edges")

    os.makedirs(ckpt_dir, exist_ok=True)
    last_path = os.path.join(ckpt_dir, "last.pt")
    best_path = os.path.join(ckpt_dir, "best.pt")

    start_epoch = 0
    best_pp_val = -1.0
    best_epoch_val = 0
    bad_epochs  = 0

    if resume and os.path.exists(last_path):
        obj = torch.load(last_path, map_location="cpu", weights_only=False)
        m = model.module if hasattr(model, "module") else model
        m.load_state_dict(obj["model"])
        opt.load_state_dict(obj["opt"])
        sched.load_state_dict(obj["sched"])
        scaler.load_state_dict(obj["scaler"])
        start_epoch = int(obj["epoch"])
        best_pp_val = float(obj.get("best_pp", -1.0))
        if main_rank:
            print(f"[ckpt] resumed epoch={start_epoch}  best_pp={best_pp_val:.3f}", flush=True)

    n_train = len(train_ds)
    grad_accum = int(cfg["training"].get("grad_accum_steps", n_train))
    if grad_accum <= 0:
        grad_accum = n_train

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    train_t0 = time.time()
    epoch_secs: List[float] = []

    # ---- training loop -------------------------------------------------------
    for epoch in range(start_epoch + 1, epochs + 1):
        t0 = time.time()
        model.train()

        loss_sum  = torch.zeros(1, device=device)
        ldata_sum = torch.zeros(1, device=device)
        lpair_sum = torch.zeros(1, device=device)
        leucpair_sum = torch.zeros(1, device=device)
        ldiv_sum  = torch.zeros(1, device=device)
        lmom_sum  = torch.zeros(1, device=device)
        lwall_sum = torch.zeros(1, device=device)
        nodes     = torch.zeros(1, device=device)
        s_guard = s_oom = s_bad = 0

        opt.zero_grad(set_to_none=True)
        accum_count = 0

        focal_active_now = focal_enable and (epoch > focal_warmup_ep)
        phys_active_now  = use_phys and (epoch > phys_warmup)
        euc_pair_active_now = lambda_euc_pair > 0.0 and (epoch > euc_pair_warmup)

        batches = list(train_loader)
        for bi, batch in enumerate(batches):
            is_last = (bi == len(batches) - 1)

            if hasattr(batch, "to"):
                batch = batch.to(device)
            N = int(batch.y.shape[0])
            E = int(batch.edge_index.shape[1])
            if (max_nodes and N > max_nodes) or (max_edges and E > max_edges):
                s_guard += 1
                if is_last and accum_count > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   float(cfg["training"]["max_grad_norm"]))
                    scaler.step(opt); scaler.update()
                    opt.zero_grad(set_to_none=True); accum_count = 0
                continue

            batch = apply_augment(batch, aug_cfg)
            y = batch.y
            U_batch = _batch_scalar(batch, "u_char", u_char) if use_case_scales else u_char
            L_batch = _batch_scalar(batch, "length_char", L_char) if use_case_scales else L_char

            try:
                with autocast():
                    try:    pred = model(batch, None)
                    except TypeError: pred = model(batch)
                    if pred is None or not torch.isfinite(pred).all():
                        raise RuntimeError("non-finite pred")

                    w = focal_hv_weights(
                        pred, y, U_batch, hv_p, hv_wmin, hv_wmax,
                        focal_gamma=focal_gamma, focal_tau=focal_tau,
                        enabled=focal_active_now,
                    )
                    if nb != 1.0 and hasattr(batch, "wall_mask"):
                        w = w * (1.0 + (nb - 1.0) * batch.wall_mask.float())

                    L_data = weighted_mse(pred, y, w)
                    if lambda_pair > 0.0:
                        L_pair = edge_pair_diff_loss(
                            pred, y, batch.edge_index, w,
                            max_edges=pair_max_edges,
                        )
                    else:
                        L_pair = pred.new_zeros(())
                    if euc_pair_active_now:
                        L_euc_pair = euclidean_pair_diff_loss(
                            pred, y, batch.pos, w,
                            k=euc_pair_k,
                            anchors=euc_pair_anchors,
                            max_pairs=euc_pair_max_pairs,
                            length_char=L_batch,
                            normalize_by_distance=euc_pair_norm,
                            chunk_size=euc_pair_chunk,
                        )
                    else:
                        L_euc_pair = pred.new_zeros(())

                    L_div = pred.new_zeros(())
                    L_mom = pred.new_zeros(())
                    L_wall = pred.new_zeros(())
                    if phys_active_now:
                        # Physics losses run in fp32 for numerical stability of
                        # the WLS gradient solve.
                        with torch.amp.autocast("cuda", enabled=False):
                            pred_f = pred.float()
                            pos_f  = batch.pos.float()
                            ei     = batch.edge_index
                            if lambda_div > 0.0:
                                L_div = divergence_loss(
                                    pred_f, pos_f, ei,
                                    weight=w.float(), u_char=U_batch,
                                    length_char=L_batch, eps_reg=eps_reg,
                                )
                            if lambda_wall > 0.0 and hasattr(batch, "wall_mask"):
                                L_wall = no_slip_loss(
                                    pred_f, batch.wall_mask, u_char=U_batch,
                                )
                            if lambda_mom > 0.0:
                                pg = getattr(batch, "p_grad", None)
                                pg_f = pg.float() if (pg is not None and torch.is_tensor(pg)) else None
                                L_mom = momentum_residual_loss(
                                    pred_f, pos_f, ei,
                                    nu=nu_kin, rho=rho, p_grad=pg_f,
                                    weight=w.float(), u_char=U_batch,
                                    length_char=L_batch, eps_reg=eps_reg,
                                )

                    loss = (L_data
                            + lambda_pair * L_pair
                            + lambda_euc_pair * L_euc_pair
                            + lambda_div  * L_div
                            + lambda_mom  * L_mom
                            + lambda_wall * L_wall) / grad_accum
                    if not torch.isfinite(loss):
                        raise RuntimeError("non-finite loss")
                scaler.scale(loss).backward()
                accum_count += 1
            except torch.cuda.OutOfMemoryError:
                s_oom += 1; opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache(); accum_count = 0; continue
            except RuntimeError as e:
                msg = str(e)
                if "expandable_segment_" in msg or "INTERNAL ASSERT" in msg:
                    s_oom += 1; opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache(); accum_count = 0; continue
                s_bad += 1; opt.zero_grad(set_to_none=True); accum_count = 0; continue
            except Exception:
                s_bad += 1; opt.zero_grad(set_to_none=True); accum_count = 0; continue

            loss_sum  += loss.detach() * grad_accum * N
            ldata_sum += L_data.detach() * N
            lpair_sum += L_pair.detach() * N
            leucpair_sum += L_euc_pair.detach() * N
            ldiv_sum  += L_div.detach()  * N
            lmom_sum  += L_mom.detach()  * N
            lwall_sum += L_wall.detach() * N
            nodes     += N

            if accum_count >= grad_accum or is_last:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               float(cfg["training"]["max_grad_norm"]))
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True); accum_count = 0

        sched.step()
        allreduce_(loss_sum); allreduce_(nodes)
        allreduce_(ldata_sum); allreduce_(lpair_sum); allreduce_(leucpair_sum)
        allreduce_(ldiv_sum); allreduce_(lmom_sum); allreduce_(lwall_sum)
        denom = nodes.clamp_min(1.0)
        train_loss = float((loss_sum / denom).item())
        train_data = float((ldata_sum / denom).item())
        train_pair = float((lpair_sum / denom).item())
        train_euc_pair = float((leucpair_sum / denom).item())
        train_div  = float((ldiv_sum  / denom).item())
        train_mom  = float((lmom_sum  / denom).item())
        train_wall = float((lwall_sum / denom).item())

        # ---- eval ------------------------------------------------------------
        do_eval = (epoch % eval_every == 0) or (epoch == 1)
        do_full = (epoch % full_every == 0) or (epoch == 1)
        val_loss = None
        mlog: Dict = {}

        if do_eval:
            model.eval()
            vl_sum = torch.zeros(1, device=device)
            vn = torch.zeros(1, device=device)
            ok_by_tol = {t: 0.0 for t in strict_tols}
            he_tot_n = 0.0
            sp_sum = 0.0
            ewrmse_sum = 0.0
            n_he_sum = 0.0
            pp_frames: List[float] = []

            with torch.no_grad():
                for it, batch in enumerate(val_loader, 1):
                    if (not do_full) and max_vbatch > 0 and it > max_vbatch:
                        break
                    if hasattr(batch, "to"):
                        batch = batch.to(device)
                    y = batch.y
                    try:
                        try:    pred = model(batch, None)
                        except TypeError: pred = model(batch)
                    except Exception:
                        continue
                    if pred is None or not torch.isfinite(pred).all():
                        continue

                    U_eval = _batch_scalar(batch, "u_char", u_char) if use_case_scales else u_char
                    w_eval = _hv_base(y, U_eval, hv_p, hv_wmin, hv_wmax)
                    vl_sum += weighted_mse(pred, y, w_eval).detach() * y.shape[0]
                    vn     += y.shape[0]

                    pred_speed = torch.linalg.vector_norm(pred, dim=-1)
                    speed = torch.linalg.vector_norm(y, dim=-1)
                    if do_full and it == 1 and main_rank:
                        print(f"  [diag] pred speed: mean={pred_speed.mean():.4f} "
                              f"max={pred_speed.max():.4f} "
                              f"p80={torch.quantile(pred_speed, 0.8):.4f} | "
                              f"true speed: mean={speed.mean():.4f} "
                              f"max={speed.max():.4f}", flush=True)
                    q = torch.quantile(speed, he_pct)
                    he = (speed >= q)
                    if he.any():
                        r = rel_err(pred, y)[he].cpu().numpy()
                        sp = speed_err(pred, y)[he].cpu().numpy()
                        diff = (pred[he] - y[he]).pow(2).sum(dim=-1).sqrt().cpu().numpy()
                        he_tot_n += len(r)
                        ok_by_tol = {t: ok_by_tol[t] + float((r <= t).sum())
                                     for t in strict_tols}
                        sp_sum += float(sp.mean()) if len(sp) > 0 else 0.0
                        ewrmse_sum += float(diff.sum())
                        n_he_sum += float(len(diff))
                        pp_frames.append(float((r <= rel_tol).mean()))

            allreduce_(vl_sum); allreduce_(vn)
            val_loss = float((vl_sum / vn.clamp_min(1.0)).item())

            if he_tot_n > 0:
                pp_by_tol = {t: ok_by_tol[t] / he_tot_n for t in strict_tols}
                for t, v in pp_by_tol.items():
                    mlog[f"val/pp_at_rel_{t:.3f}"] = v
                auc = _auc(pp_by_tol, auc_min, auc_max)
                mlog[f"val/pp_auc_{auc_min:.3f}_{auc_max:.3f}"] = auc
                mlog["val/speed_err_he_mean"] = sp_sum / max(len(pp_frames), 1)
                mlog["val/ewrmse_he"] = ewrmse_sum / max(n_he_sum, 1.0)
            else:
                pp_by_tol = {t: 0.0 for t in strict_tols}
                auc = float("nan")

            if pp_frames:
                arr = np.array(pp_frames)
                mlog["diag/pp_frame_mean"] = float(arr.mean())
                mlog["diag/pp_frame_p50"]  = float(np.median(arr))
                mlog["diag/pp_frame_min"]  = float(arr.min())

        # ---- logging ---------------------------------------------------------
        t1 = time.time()
        epoch_secs.append(t1 - t0)
        if main_rank:
            log = {"epoch": epoch,
                   "train/loss": train_loss,
                   "train/data": train_data,
                   "train/pair": train_pair,
                   "train/euclidean_pair": train_euc_pair,
                   "train/div":  train_div,
                   "train/mom":  train_mom,
                   "train/wall": train_wall,
                   "train/focal_active": int(focal_active_now),
                   "train/phys_active":  int(phys_active_now),
                   "train/euclidean_pair_active": int(euc_pair_active_now),
                   "lr": float(sched.get_last_lr()[0]),
                   "time/epoch_sec": t1 - t0,
                   "train/skip_guard": s_guard,
                   "train/skip_oom":   s_oom,
                   "train/skip_bad":   s_bad}
            if val_loss is not None:
                log["val/loss"] = val_loss
            log.update(mlog)
            wb.log(log)

            if do_eval:
                pp10 = mlog.get("val/pp_at_rel_0.100", 0.0)
                pp05 = mlog.get("val/pp_at_rel_0.050", float("nan"))
                pp02 = mlog.get("val/pp_at_rel_0.020", float("nan"))
                ewr  = mlog.get("val/ewrmse_he", float("nan"))
                phy_str = (f" phys=d{train_div:.3e}/m{train_mom:.3e}/w{train_wall:.3e}"
                           if phys_active_now else "")
                euc_str = f" euc {train_euc_pair:.4e}" if euc_pair_active_now else ""
                print(
                    f"Epoch {epoch:04d} | train {train_loss:.4e} (data {train_data:.4e} "
                    f"pair {train_pair:.4e}{euc_str}{phy_str}) | "
                    f"val {val_loss:.4e} | "
                    f"PP@10 {pp10:.3f} | PP@5 {pp05:.3f} | PP@2 {pp02:.3f} | "
                    f"ewRMSE_HE {ewr:.4f} | "
                    f"focal={int(focal_active_now)} phys={int(phys_active_now)} | "
                    f"skips g={s_guard} o={s_oom} b={s_bad} | {t1-t0:.1f}s",
                    flush=True,
                )
            else:
                print(f"Epoch {epoch:04d} | train {train_loss:.4e} | "
                      f"focal={int(focal_active_now)} phys={int(phys_active_now)} | "
                      f"skips g={s_guard} o={s_oom} b={s_bad} | {t1-t0:.1f}s",
                      flush=True)

        # ---- checkpoint ------------------------------------------------------
        pp10_now = mlog.get("val/pp_at_rel_0.100", -1.0) if do_eval else -1.0
        if do_eval and main_rank:
            if pp10_now > best_pp_val:
                best_pp_val = pp10_now
                best_epoch_val = epoch
                bad_epochs = 0
                save_ckpt(best_path, model, opt, sched, scaler, epoch, best_pp_val)
            else:
                bad_epochs += 1

        if main_rank and epoch % save_every == 0:
            save_ckpt(last_path, model, opt, sched, scaler, epoch, best_pp_val)
            ep_path = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt")
            save_ckpt(ep_path, model, opt, sched, scaler, epoch, best_pp_val)
            old = sorted(f for f in os.listdir(ckpt_dir)
                         if f.startswith("epoch_") and f.endswith(".pt"))
            for f in old[:-keep_last]:
                try: os.remove(os.path.join(ckpt_dir, f))
                except Exception: pass

        # ---- stop criteria ---------------------------------------------------
        if do_eval and pp10_now >= target_pp:
            if main_rank:
                print(f"[STOP] target PP@10={target_pp} reached: {pp10_now:.3f}", flush=True)
            break
        if do_eval and bad_epochs >= patience:
            if main_rank:
                print(f"[STOP] no improvement for {patience} evals. best={best_pp_val:.3f}", flush=True)
            break

    train_t1 = time.time()
    if main_rank:
        eff["mean_epoch_sec"] = float(np.mean(epoch_secs)) if epoch_secs else 0.0
        eff["total_epochs"]   = int(len(epoch_secs))
        eff["best_epoch"]     = int(best_epoch_val)
        eff["best_pp10_val"]  = float(best_pp_val)
        eff["train_time_s"]   = float(train_t1 - train_t0)
        if device.type == "cuda":
            eff["peak_gpu_mb"] = float(torch.cuda.max_memory_allocated() / 1e6)
        eff_path = os.path.join(ckpt_dir, "efficiency.json")
        try:
            with open(eff_path, "w") as f:
                json.dump(eff, f, indent=2)
            print(f"[eff] {eff_path}: epoch_sec={eff['mean_epoch_sec']:.2f} "
                  f"peak_gpu_mb={eff.get('peak_gpu_mb', 0):.0f} "
                  f"params={eff['n_params']} best_pp10={eff['best_pp10_val']:.3f} "
                  f"@ep{eff['best_epoch']} ({eff['total_epochs']} ep total)",
                  flush=True)
        except Exception as e:
            print(f"[eff] failed to write {eff_path}: {e}", flush=True)

    try: wb.finish()
    except Exception: pass
    ddp_cleanup()
    return eff


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds",  default=None,
                    help="Comma-separated list of seeds (overrides config)")
    ap.add_argument("--smoke", action="store_true",
                    help="Run only 3 epochs as a smoke test")
    ap.add_argument("--override", action="append", default=[],
                    metavar="KEY.PATH=VALUE",
                    help="Override a YAML field, e.g. "
                         "training.epochs=500 or model.equivariant_head=true. "
                         "Can be passed multiple times.")
    ap.add_argument("--variant-name", default=None,
                    help="If set, used as the wandb run_name suffix and "
                         "prepended to the ckpt_dir leaf.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    for ov in args.override:
        if "=" not in ov:
            raise SystemExit(f"--override must be KEY.PATH=VALUE, got {ov!r}")
        key, value = ov.split("=", 1)
        _set_nested(cfg, key.strip(), value.strip())

    if args.smoke:
        cfg["training"]["epochs"] = 3
        cfg["eval"]["every"] = 1
        cfg["eval"]["full_every"] = 1
        cfg["eval"]["max_batches"] = 2
        cfg["goals"]["patience"] = 99
        cfg["training"]["save_every"] = 99

    seeds = ([int(s) for s in args.seeds.split(",")]
             if args.seeds else [int(cfg["training"]["seed"])])

    base_ckpt_dir = cfg["training"]["ckpt_dir"].rstrip("/")
    base_run_name = cfg["logging"].get("run_name", "flowgat_v6")
    if args.variant_name:
        base_run_name = f"{base_run_name}_{args.variant_name}"

    for i, s in enumerate(seeds, 1):
        print(f"\n===== Run {i}/{len(seeds)}  (seed={s}) =====", flush=True)
        cfg["training"]["seed"] = s
        cfg["logging"]["run_name"] = f"{base_run_name}_s{s}"
        cfg["training"]["ckpt_dir"] = f"{base_ckpt_dir}/seed_{s}"
        cfg["training"]["resume"] = False
        train(cfg)


if __name__ == "__main__":
    main()
