# -*- coding: utf-8 -*-
"""
Training engine for the Flow-GAT aortic velocity surrogate.

Compared to the legacy `train.py`:

  - Uses the composite physics + HV loss (`CompositeLoss`).
  - Uses `MetricAccumulator` for evaluation (DDP-aware).
  - Selects the best checkpoint by a COMPOSITE score
        S = 0.5·PP@0.10  +  0.5·AUC(0, 0.10)
    which is robust against KPI saturation and low-noise jitter.
  - Runs N seeds on request and aggregates mean ± std.
  - Logs per-term losses and clinical metrics to W&B.

The engine is agnostic to PyG: it only requires that batches expose
`x`, `pos`, `y`, `edge_index`, `edge_attr`, `wall_mask`, `wall_normal`.
"""

from __future__ import annotations
import os
import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..data import VMRGraphDataset, AugmentConfig, apply_augment
from ..models import FlowGAT
from ..losses import CompositeLoss, CompositeLossConfig
from ..metrics import MetricAccumulator
from .utils import (
    set_global_seed, setup_speed,
    ddp_setup, ddp_cleanup, ddp_all_reduce_,
    save_ckpt, load_ckpt, prune_ckpts,
)


# --------------------------------------------------------------------------
#  Small wrapper: safe W&B
# --------------------------------------------------------------------------

class _WandbNull:
    def init(self, *a, **k): return self
    def log(self, *a, **k): pass
    def finish(self): pass
    def __getattr__(self, _): return self


def _init_wandb(cfg: dict, main_rank: bool):
    if not main_rank:
        return _WandbNull()
    try:
        import wandb
        os.makedirs(cfg["logging"]["wandb_dir"], exist_ok=True)
        os.environ.setdefault("WANDB_DIR", cfg["logging"]["wandb_dir"])
        wb = wandb.init(
            project=cfg["logging"]["project"],
            name=cfg["logging"]["run_name"],
            config=cfg,
        )
        # Sweep-override hook
        wc = wb.config
        for key, path in [
            ("feature_noise_std", ("augment", "feature_noise_std")),
            ("edge_dropout_p",    ("augment", "edge_dropout_p")),
            ("rotate_so3",        ("augment", "rotate_so3")),
            ("lambda_div",        ("loss", "lambda_div")),
            ("lambda_wall",       ("loss", "lambda_wall")),
            ("lambda_mom",        ("loss", "lambda_mom")),
            ("lambda_mag",        ("loss", "lambda_mag")),
            ("seed",              ("training", "seed")),
        ]:
            if key in wc:
                cfg.setdefault(path[0], {})
                cfg[path[0]][path[1]] = wc[key]
        return wb
    except Exception as e:
        print(f"[wandb] disabled: {e}")
        return _WandbNull()


# --------------------------------------------------------------------------
#  DataLoader factory
# --------------------------------------------------------------------------

def _build_loaders(cfg: dict, rank_ws):
    try:
        from torch_geometric.loader import DataLoader
    except Exception:
        from torch.utils.data import DataLoader

    train_ds = VMRGraphDataset(
        cfg["data"]["train_dir"],
        split_file=cfg["data"].get("split_file"),
        split="train" if cfg["data"].get("split_file") else None,
    )
    val_ds = VMRGraphDataset(
        cfg["data"]["val_dir"],
        split_file=cfg["data"].get("split_file"),
        split="val" if cfg["data"].get("split_file") else None,
    )

    if rank_ws is not None:
        from torch.utils.data.distributed import DistributedSampler
        rank, world_size, _ = rank_ws
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank,
            shuffle=True, drop_last=True,
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank,
            shuffle=False, drop_last=False,
        )
        train_shuffle = False
    else:
        train_sampler = val_sampler = None
        train_shuffle = True

    nw = int(cfg["training"]["num_workers"])

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=train_shuffle,
        num_workers=nw,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(nw > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, nw // 2),
        sampler=val_sampler,
        pin_memory=True,
        persistent_workers=(max(1, nw // 2) > 0),
    )
    return train_loader, val_loader, train_ds, val_ds


# --------------------------------------------------------------------------
#  Composite selection score
# --------------------------------------------------------------------------

def _selection_score(metrics: Dict[str, float]) -> float:
    pp = float(metrics.get("val/pp_at_rel_0.100", 0.0) or 0.0)
    auc_key = [k for k in metrics if k.startswith("val/pp_auc_")]
    auc = float(metrics.get(auc_key[0], 0.0)) if auc_key else 0.0
    if math.isnan(pp) or math.isnan(auc):
        return -1.0
    return 0.5 * pp + 0.5 * auc


# --------------------------------------------------------------------------
#  Robust safety checks
# --------------------------------------------------------------------------

def _is_allocator_internal_assert(e: BaseException) -> bool:
    s = str(e)
    return ("expandable_segment_" in s) or ("CUDACachingAllocator.cpp" in s) or ("INTERNAL ASSERT FAILED" in s)


def _safe_isfinite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


# --------------------------------------------------------------------------
#  Main run
# --------------------------------------------------------------------------

def run(cfg: dict) -> Dict[str, float]:
    rank_ws = ddp_setup()
    main_rank = (rank_ws is None) or (rank_ws[0] == 0)

    setup_speed(bool(cfg["training"].get("tf32", True)))
    seed = int(cfg["training"].get("seed", 1337)) + (rank_ws[0] if rank_ws else 0)
    set_global_seed(seed, bool(cfg["training"].get("deterministic", False)))

    wandb = _init_wandb(cfg, main_rank=main_rank)
    train_loader, val_loader, train_ds, _ = _build_loaders(cfg, rank_ws)

    # --- model dims inferred from data if missing ------------------------
    probe = train_ds.probe_dims()
    node_in = cfg["model"].get("node_in") or probe["node_in"]
    edge_in = cfg["model"].get("edge_in") or probe["edge_in"]

    model = FlowGAT(
        node_in=node_in,
        edge_in=edge_in,
        hidden=int(cfg["model"]["hidden"]),
        heads=int(cfg["model"]["heads"]),
        layers=int(cfg["model"]["layers"]),
        dropout=float(cfg["model"].get("dropout", 0.0)),
        attn_bias_beta=float(cfg["model"].get("attn_bias_beta", 1.5)),
        out=3,
        equivariant_head=bool(cfg["model"].get("equivariant_head", True)),
        equivariant_basis=int(cfg["model"].get("equivariant_basis", 8)),
        hard_no_slip=bool(cfg["model"].get("hard_no_slip", True)),
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if rank_ws is not None:
        device = torch.device(f"cuda:{rank_ws[2]}")
    model = model.to(device)

    if rank_ws is not None:
        model = DDP(model, device_ids=[rank_ws[2]], output_device=rank_ws[2],
                    find_unused_parameters=False)

    # --- loss ------------------------------------------------------------
    _length_char = float(cfg["data"].get("length_char", 0.025))
    if main_rank:
        print(
            f"[warn] length_char={_length_char:.4f} m from config (global default); "
            f"per-case values in batch.length_char are NOT used by the loss. "
            f"Dataset range: 0.108–0.279 m."
        )
    loss_cfg = CompositeLossConfig(
        lambda_data=float(cfg["loss"]["lambda_data"]),
        lambda_mag=float(cfg["loss"]["lambda_mag"]),
        lambda_div=float(cfg["loss"]["lambda_div"]),
        lambda_wall=float(cfg["loss"]["lambda_wall"]),
        lambda_mom=float(cfg["loss"]["lambda_mom"]),
        hv_p=float(cfg["loss"]["hv"]["p"]),
        hv_w_min=float(cfg["loss"]["hv"]["min_weight"]),
        hv_w_max=float(cfg["loss"]["hv"]["max_weight"]),
        near_wall_boost=float(cfg["loss"]["hv"]["near_wall_boost"]),
        u_char=float(cfg["data"]["u_char"]),
        length_char=_length_char,
        nu=float(cfg["data"].get("nu", 3.3e-6)),
        rho=float(cfg["data"].get("rho", 1060.0)),
        he_percentile=float(cfg["loss"].get("he_percentile", 0.80)),
    )
    criterion = CompositeLoss(loss_cfg)

    # --- optim -----------------------------------------------------------
    opt = AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    sched = CosineAnnealingLR(
        opt,
        T_max=int(cfg["training"]["cosine_t_max"]),
        eta_min=float(cfg["training"].get("lr_min", 1.0e-6)),
    )

    amp_on = bool(cfg["training"].get("amp", True)) and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on)
    autocast = lambda: torch.amp.autocast("cuda", enabled=amp_on)

    grad_accum = int(cfg["training"].get("grad_accum_steps", 1))
    max_grad_norm = float(cfg["training"].get("max_grad_norm", 1.0))
    max_edges = cfg["training"].get("max_edges")
    max_nodes = cfg["training"].get("max_nodes")
    max_edges = int(max_edges) if max_edges is not None else None
    max_nodes = int(max_nodes) if max_nodes is not None else None

    # --- augmentation ----------------------------------------------------
    aug_cfg = AugmentConfig(
        enabled=bool(cfg["augment"]["enabled"]),
        rotate_so3=bool(cfg["augment"].get("rotate_so3", True)),
        feature_noise_std=float(cfg["augment"].get("feature_noise_std", 0.0)),
        edge_dropout_p=float(cfg["augment"].get("edge_dropout_p", 0.0)),
    )

    # --- eval config -----------------------------------------------------
    eval_every = int(cfg["eval"]["every"])
    full_every = int(cfg["eval"]["full_every"])
    max_val_batches = int(cfg["eval"]["max_batches"])
    strict_tols = [float(x) for x in cfg["eval"]["strict_tols"]]
    rel_thresholds = [float(x) for x in cfg["eval"]["rel_thresholds"]]
    auc_range = tuple(cfg["eval"].get("auc_range", [0.0, 0.10]))
    he_pct = float(cfg["loss"].get("he_percentile", 0.80))

    # --- checkpoint state -----------------------------------------------
    ckpt_dir = cfg["training"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    last_path = os.path.join(ckpt_dir, "last.pt")
    best_path = os.path.join(ckpt_dir, "best.pt")

    best_score = -1.0
    patience_bad = 0
    target_pp = float(cfg["goals"]["target_recall_he"])
    patience = int(cfg["goals"]["patience"])

    start_epoch = 0
    if bool(cfg["training"].get("resume", False)) and os.path.exists(last_path):
        obj = load_ckpt(last_path, model, opt, sched, scaler)
        start_epoch = int(obj.get("epoch", 0))
        best_score = float(obj.get("best_score", -1.0))
        if main_rank:
            print(f"[ckpt] resumed {last_path} epoch={start_epoch} best_score={best_score:.4f}")

    # =====================================================================
    #                        Training loop
    # =====================================================================
    for epoch in range(start_epoch + 1, int(cfg["training"]["epochs"]) + 1):
        t0 = time.time()
        model.train()
        if rank_ws is not None:
            try:
                train_loader.sampler.set_epoch(epoch)    # type: ignore
            except Exception:
                pass

        loss_sums = {k: torch.zeros(1, device=device) for k in
                     ("total", "data", "mag", "div", "wall", "mom")}
        train_nodes = torch.zeros(1, device=device)
        skip_guard = torch.zeros(1, device=device)
        skip_oom   = torch.zeros(1, device=device)
        skip_bad   = torch.zeros(1, device=device)

        opt.zero_grad(set_to_none=True)
        step_i = 0
        for batch in train_loader:
            step_i += 1
            if hasattr(batch, "to"):
                batch = batch.to(device)

            # size guard
            N = int(batch.y.shape[0])
            E = int(batch.edge_index.shape[1])
            if (max_edges is not None and E > max_edges) or (max_nodes is not None and N > max_nodes):
                skip_guard += 1.0
                continue

            # augment (in place on the moved-to-device batch)
            batch = apply_augment(batch, aug_cfg, training=True)

            try:
                with autocast():
                    try:
                        pred = model(batch)
                    except TypeError:
                        pred = model(batch, None)

                    if pred is None or (not _safe_isfinite(pred)):
                        raise RuntimeError("Non-finite pred")

                    losses = criterion(
                        pred=pred,
                        y=batch.y,
                        pos=batch.pos,
                        edge_index=batch.edge_index,
                        wall_mask=getattr(batch, "wall_mask", None),
                        p_grad=getattr(batch, "p_grad", None),
                    )
                    total = losses["total"] / float(max(1, grad_accum))
                    if not _safe_isfinite(total):
                        raise RuntimeError("Non-finite loss")

                scaler.scale(total).backward()

            except torch.cuda.OutOfMemoryError:
                skip_oom += 1.0
                opt.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            except RuntimeError as e:
                if _is_allocator_internal_assert(e):
                    skip_oom += 1.0
                    opt.zero_grad(set_to_none=True)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue
                skip_bad += 1.0
                opt.zero_grad(set_to_none=True)
                continue
            except Exception:
                skip_bad += 1.0
                opt.zero_grad(set_to_none=True)
                continue

            # accumulate stats
            for k in loss_sums:
                loss_sums[k] += losses[k].detach().float().view(1) * float(N)
            train_nodes += float(N)

            if (step_i % grad_accum) == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

        if (step_i % grad_accum) != 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        sched.step()

        for t in (*loss_sums.values(), train_nodes, skip_guard, skip_oom, skip_bad):
            ddp_all_reduce_(t)

        denom = train_nodes.clamp_min(1.0)
        train_losses_epoch = {k: float((v / denom).item()) for k, v in loss_sums.items()}
        s_guard = int(skip_guard.item())
        s_oom   = int(skip_oom.item())
        s_bad   = int(skip_bad.item())

        # =====================================================================
        #                      Evaluation
        # =====================================================================
        do_eval = (epoch % eval_every == 0) or (epoch == 1)
        do_full = (epoch % full_every == 0) or (epoch == 1)
        metrics: Dict[str, float] = {}
        val_loss = None

        if do_eval:
            model.eval()
            _mu_blood = float(cfg["data"]["nu"]) * float(cfg["data"].get("rho", 1060.0))
            acc = MetricAccumulator(
                strict_tols=strict_tols,
                rel_thresholds=rel_thresholds,
                he_percentile=he_pct,
                auc_range=auc_range,
                rel_tol_primary=0.10,
                compute_wss=bool(cfg["eval"].get("compute_wss", True)),
                mu_blood=_mu_blood,
            )
            val_loss_sum = torch.zeros(1, device=device)
            val_nodes    = torch.zeros(1, device=device)

            with torch.no_grad():
                for it, batch in enumerate(val_loader, start=1):
                    if (not do_full) and (max_val_batches > 0) and (it > max_val_batches):
                        break
                    if hasattr(batch, "to"):
                        batch = batch.to(device)

                    try:
                        pred = model(batch)
                    except TypeError:
                        pred = model(batch, None)
                    if pred is None or (not _safe_isfinite(pred)):
                        continue

                    losses_v = criterion(
                        pred=pred, y=batch.y, pos=batch.pos,
                        edge_index=batch.edge_index,
                        wall_mask=getattr(batch, "wall_mask", None),
                        p_grad=getattr(batch, "p_grad", None),
                    )
                    n = int(batch.y.shape[0])
                    val_loss_sum += losses_v["total"].detach().float().view(1) * float(n)
                    val_nodes    += float(n)

                    acc.update(
                        pred=pred.detach(),
                        y=batch.y.detach(),
                        pos=batch.pos.detach(),
                        edge_index=batch.edge_index.detach(),
                        wall_mask=getattr(batch, "wall_mask", None),
                        wall_normals=getattr(batch, "wall_normal", None),
                    )

            ddp_all_reduce_(val_loss_sum)
            ddp_all_reduce_(val_nodes)
            val_loss = float((val_loss_sum / val_nodes.clamp_min(1.0)).item())
            metrics = acc.finalize()

        # =====================================================================
        #                      Logging & ckpt
        # =====================================================================
        t1 = time.time()
        score = _selection_score(metrics) if do_eval else -1.0

        if main_rank:
            log = {
                "epoch": epoch,
                "time/epoch_sec": float(t1 - t0),
                "lr": float(sched.get_last_lr()[0]),
                "train/loss":      train_losses_epoch["total"],
                "train/loss_data": train_losses_epoch["data"],
                "train/loss_div":  train_losses_epoch["div"],
                "train/loss_wall": train_losses_epoch["wall"],
                "train/loss_mom":  train_losses_epoch["mom"],
                "train/loss_mag":  train_losses_epoch["mag"],
                "train/skip_guard": s_guard,
                "train/skip_oom": s_oom,
                "train/skip_bad": s_bad,
                "aug/rotate_so3": int(aug_cfg.rotate_so3),
                "aug/feature_noise_std": aug_cfg.feature_noise_std,
                "aug/edge_dropout_p": aug_cfg.edge_dropout_p,
            }
            if val_loss is not None:
                log["val/loss"] = val_loss
                log["val/selection_score"] = score
                log.update(metrics)
            try:
                wandb.log(log)
            except Exception:
                pass

            if do_eval:
                pp10 = metrics.get("val/pp_at_rel_0.100", float("nan"))
                auc_k = next((k for k in metrics if k.startswith("val/pp_auc_")), None)
                auc_v = metrics.get(auc_k, float("nan")) if auc_k else float("nan")
                peak_mm = metrics.get("clin/peak_loc_mm_mean", float("nan"))
                dp_mae  = metrics.get("clin/dP_mmHg_mae", float("nan"))
                print(
                    f"Epoch {epoch:04d} | train {train_losses_epoch['total']:.4e} | "
                    f"val {val_loss:.4e} | PP@10 {pp10:.3f} | AUC {auc_v:.3f} | "
                    f"peak {peak_mm:.1f}mm | dP_mae {dp_mae:.1f}mmHg | "
                    f"score {score:.4f} | skips g={s_guard} o={s_oom} b={s_bad} | "
                    f"{(t1 - t0):.1f}s"
                )
            else:
                print(
                    f"Epoch {epoch:04d} | train {train_losses_epoch['total']:.4e} | "
                    f"skips g={s_guard} o={s_oom} b={s_bad} | {(t1 - t0):.1f}s"
                )

        # best
        if do_eval and score > best_score and main_rank:
            best_score = score
            patience_bad = 0
            save_ckpt(best_path, model, opt, sched, scaler, epoch, best_score,
                      extra={"metrics": metrics})
        elif do_eval:
            patience_bad += 1

        # periodic last
        if main_rank and (epoch % int(cfg["training"].get("save_every", 50)) == 0):
            save_ckpt(last_path, model, opt, sched, scaler, epoch, best_score)
            ep_path = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt")
            save_ckpt(ep_path, model, opt, sched, scaler, epoch, best_score)
            prune_ckpts(ckpt_dir, prefix="epoch_",
                        keep_last=int(cfg["training"].get("keep_last", 3)))

        # stop criteria
        if do_eval and metrics.get("val/pp_at_rel_0.100", 0.0) >= target_pp:
            if main_rank:
                print(f"[STOP] target_recall_he reached: "
                      f"{metrics['val/pp_at_rel_0.100']:.3f}")
            break
        if do_eval and patience_bad >= patience:
            if main_rank:
                print(f"[STOP] early stopping: no improvement for {patience} evals. "
                      f"best_score={best_score:.4f}")
            break

    if main_rank:
        try:
            wandb.finish()
        except Exception:
            pass

    ddp_cleanup()
    return {"best_score": best_score}
