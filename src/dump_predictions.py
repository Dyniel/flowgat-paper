# -*- coding: utf-8 -*-
"""
Dump predicted velocity field, ground truth, and node coordinates for
each (variant, seed, case) on val + test splits, into compressed NPZ files.

These dumps are the input to:
  - peak_loc_diagnostic.py  (degeneracy check on noleak peak prediction)
  - stratified_analyze.py   (rigid vs FSI vs healthy)
  - bootstrap_ci.py         (paired bootstrap)
  - make_figures_v2.py      (Fig 4 peak-loc map overlay)

We dump the full predictions ONCE so all downstream analyses are deterministic
and do not require re-running GPU inference.

Output schema, one NPZ per (variant, seed, split, case):
  pred           [N, 3] float32  predicted velocity
  y              [N, 3] float32  ground-truth velocity
  pos            [N, 3] float32  node positions (metres)
  wall_mask      [N]    bool
  case_id        str
  variant        str
  seed           int
  split          str

Usage:
    python src/dump_predictions.py \
        --config configs/noleak.yaml \
        --ckpt   results/checkpoints/noleak/seed_1337/best.pt \
        --variant noleak --seed 1337 \
        --out_dir results/predictions
"""

from __future__ import annotations
import os
import sys
import argparse

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowgnn_aorta.data import VMRGraphDataset
from flowgnn_aorta.models import FlowGAT
from flowgnn_aorta.models.flow_sage import FlowSAGE

_MODEL_REGISTRY = {"flowgat": FlowGAT, "flow_sage": FlowSAGE}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    return ap.parse_args()


def _resolve_paths(cfg, cfg_dir):
    def _r(p):
        if p and not os.path.isabs(p):
            return os.path.join(cfg_dir, p)
        return p
    for k in ("train_dir", "val_dir", "test_dir", "split_file"):
        if k in cfg["data"]:
            cfg["data"][k] = _r(cfg["data"][k])
    return cfg


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg = _resolve_paths(cfg, os.path.dirname(os.path.abspath(args.config)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obj = torch.load(args.ckpt, map_location=device)
    sd = obj["model"]
    equivariant_head = "head.lin_vec_edge.weight" in sd

    out_root = os.path.join(args.out_dir, args.variant, f"seed_{args.seed}")
    os.makedirs(out_root, exist_ok=True)

    n_dumped = 0
    for split in args.splits:
        root = cfg["data"][f"{split}_dir"]
        ds = VMRGraphDataset(
            root, split_file=cfg["data"].get("split_file"), split=split
        )
        probe = ds.probe_dims()

        arch = str(cfg["model"].get("arch", "flowgat")).lower()
        model_cls = _MODEL_REGISTRY[arch]
        model = model_cls(
            node_in=cfg["model"].get("node_in") or probe["node_in"],
            edge_in=cfg["model"].get("edge_in") or probe["edge_in"],
            hidden=int(cfg["model"]["hidden"]),
            heads=int(cfg["model"]["heads"]),
            layers=int(cfg["model"]["layers"]),
            dropout=0.0,
            attn_bias_beta=float(cfg["model"].get("attn_bias_beta", 1.5)),
            out=3,
            equivariant_head=equivariant_head,
            equivariant_basis=int(cfg["model"].get("equivariant_basis", 8)),
            hard_no_slip=bool(cfg["model"].get("hard_no_slip", True)),
            graph_stem_type=str(cfg["model"].get("graph_stem_type", "none")),
            graph_stem_layers=int(cfg["model"].get("graph_stem_layers", 0)),
            graph_stem_dropout=float(cfg["model"].get("graph_stem_dropout", 0.0)),
            graph_stem_residual=bool(cfg["model"].get("graph_stem_residual", True)),
        ).to(device)
        model.load_state_dict(sd, strict=True)
        model.eval()

        with torch.no_grad():
            for i in range(len(ds)):
                batch = ds[i].to(device)
                pred = model(batch)
                if not torch.isfinite(pred).all():
                    print(f"  [WARN] non-finite pred for case "
                          f"{getattr(batch, 'case_id', i)}, skipping")
                    continue
                case_id = str(getattr(batch, "case_id", f"case_{i:04d}"))
                out_path = os.path.join(out_root, f"{split}_{case_id}.npz")
                np.savez_compressed(
                    out_path,
                    pred=pred.detach().cpu().numpy().astype(np.float32),
                    y=batch.y.detach().cpu().numpy().astype(np.float32),
                    pos=batch.pos.detach().cpu().numpy().astype(np.float32),
                    wall_mask=batch.wall_mask.detach().cpu().numpy(),
                    case_id=np.array(case_id),
                    variant=np.array(str(args.variant)),
                    seed=np.int64(args.seed),
                    split=np.array(split),
                )
                n_dumped += 1
                print(f"  -> {out_path}: N={pred.shape[0]}")

        # Free model + dataset before next split
        del model, ds
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"[dump_predictions] Wrote {n_dumped} NPZ files to {out_root}")


if __name__ == "__main__":
    main()
