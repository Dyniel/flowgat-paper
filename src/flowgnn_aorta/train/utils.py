# -*- coding: utf-8 -*-
"""Training utilities: seeding, DDP helpers, checkpoint management."""

from __future__ import annotations
import os
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist


# --------------------------------------------------------------------------
#  Seeding / speed
# --------------------------------------------------------------------------

def set_global_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def setup_speed(tf32: bool) -> None:
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
            torch.backends.cudnn.allow_tf32 = bool(tf32)
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


# --------------------------------------------------------------------------
#  DDP
# --------------------------------------------------------------------------

def ddp_env() -> Optional[Tuple[int, int, int]]:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        return (
            int(os.environ["RANK"]),
            int(os.environ["WORLD_SIZE"]),
            int(os.environ.get("LOCAL_RANK", 0)),
        )
    return None


def ddp_setup() -> Optional[Tuple[int, int, int]]:
    env = ddp_env()
    if env is None:
        return None
    rank, world_size, local_rank = env
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def ddp_cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()


def ddp_all_reduce_(t: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=op)
    return t


# --------------------------------------------------------------------------
#  Checkpointing
# --------------------------------------------------------------------------

def save_ckpt(
    path: str,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    sched,
    scaler,
    epoch: int,
    best_score: float,
    extra: Optional[dict] = None,
) -> None:
    m = model.module if hasattr(model, "module") else model
    obj = {
        "epoch": int(epoch),
        "best_score": float(best_score),
        "model": m.state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict() if sched is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    if extra is not None:
        obj["extra"] = extra
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(obj, path)


def load_ckpt(
    path: str,
    model: torch.nn.Module,
    opt: Optional[torch.optim.Optimizer] = None,
    sched=None,
    scaler=None,
    map_location: str = "cpu",
) -> dict:
    obj = torch.load(path, map_location=map_location)
    m = model.module if hasattr(model, "module") else model
    m.load_state_dict(obj["model"], strict=True)
    if opt is not None and "opt" in obj:
        opt.load_state_dict(obj["opt"])
    if sched is not None and obj.get("sched") is not None:
        sched.load_state_dict(obj["sched"])
    if scaler is not None and obj.get("scaler") is not None:
        scaler.load_state_dict(obj["scaler"])
    return obj


def prune_ckpts(ckpt_dir: str, prefix: str = "epoch_", keep_last: int = 3) -> None:
    try:
        files = [f for f in os.listdir(ckpt_dir) if f.startswith(prefix) and f.endswith(".pt")]
        files.sort()
        while len(files) > keep_last:
            rm = files.pop(0)
            try:
                os.remove(os.path.join(ckpt_dir, rm))
            except Exception:
                pass
    except Exception:
        pass
