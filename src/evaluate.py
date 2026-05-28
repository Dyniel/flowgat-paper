# -*- coding: utf-8 -*-
"""
Evaluate a trained checkpoint on a held-out dataset and dump a results CSV.

Usage:
    python scripts/evaluate.py \
        --config configs/vmr_physics.yaml \
        --ckpt runs/ckpts/seed_1337/best.pt \
        --split test \
        --out results/test_metrics.csv
"""

from __future__ import annotations
import os
import sys
import csv
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch

from flowgnn_aorta.data import VMRGraphDataset
from flowgnn_aorta.models import FlowGAT
from flowgnn_aorta.models.flow_sage import FlowSAGE

_MODEL_REGISTRY = {"flowgat": FlowGAT, "flow_sage": FlowSAGE}
from flowgnn_aorta.metrics import MetricAccumulator
from flowgnn_aorta.metrics.clinical import (
    peak_velocity_localisation, peak_pressure_drop_error, wss_error_metrics,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test",
                    choices=["train", "val", "test"])
    ap.add_argument("--out", default="results/metrics.csv")
    return ap.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve relative paths in config relative to the config file's directory,
    # so the script works regardless of the current working directory.
    cfg_dir = os.path.dirname(os.path.abspath(args.config))
    def _resolve(p):
        if p and not os.path.isabs(p):
            return os.path.join(cfg_dir, p)
        return p

    for key in ("train_dir", "val_dir", "test_dir", "split_file"):
        if key in cfg["data"]:
            cfg["data"][key] = _resolve(cfg["data"][key])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Dataset
    root = cfg["data"][f"{args.split}_dir"] if f"{args.split}_dir" in cfg["data"] \
        else cfg["data"]["val_dir"]
    split_file = cfg["data"].get("split_file")
    ds = VMRGraphDataset(root, split_file=split_file, split=args.split)
    probe = ds.probe_dims()

    # Load checkpoint first to infer architecture from saved weights.
    obj = torch.load(args.ckpt, map_location=device)
    sd = obj["model"]
    equivariant_head = "head.lin_vec_edge.weight" in sd

    # Model
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

    # Per-case and aggregate
    strict_tols   = [float(x) for x in cfg["eval"]["strict_tols"]]
    rel_thresholds = [float(x) for x in cfg["eval"]["rel_thresholds"]]
    he_pct = float(cfg["loss"].get("he_percentile", 0.80))
    auc_range = tuple(cfg["eval"].get("auc_range", [0.0, 0.10]))
    mu_blood = float(cfg["data"]["nu"]) * float(cfg["data"].get("rho", 1060.0))

    agg = MetricAccumulator(
        strict_tols=strict_tols, rel_thresholds=rel_thresholds,
        he_percentile=he_pct, auc_range=auc_range,
        rel_tol_primary=0.10, compute_wss=True, mu_blood=mu_blood,
    )

    rows = []
    with torch.no_grad():
        for i in range(len(ds)):
            batch = ds[i].to(device)

            pred = model(batch)
            if not torch.isfinite(pred).all():
                continue

            # per-case accumulator for clean CSV rows
            local = MetricAccumulator(
                strict_tols=strict_tols, rel_thresholds=rel_thresholds,
                he_percentile=he_pct, auc_range=auc_range,
                rel_tol_primary=0.10, compute_wss=True, mu_blood=mu_blood,
            )
            local.update(
                pred=pred, y=batch.y, pos=batch.pos,
                edge_index=batch.edge_index,
                wall_mask=getattr(batch, "wall_mask", None),
                wall_normals=getattr(batch, "wall_normal", None),
            )
            case_metrics = local.finalize()

            # also update the aggregate
            agg.update(
                pred=pred, y=batch.y, pos=batch.pos,
                edge_index=batch.edge_index,
                wall_mask=getattr(batch, "wall_mask", None),
                wall_normals=getattr(batch, "wall_normal", None),
            )

            row = {"case_id": getattr(batch, "case_id", f"case_{i:04d}")}
            row.update(case_metrics)
            rows.append(row)

    # Write CSV
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Write aggregate JSON
    agg_path = os.path.splitext(args.out)[0] + "_aggregate.json"
    with open(agg_path, "w") as f:
        json.dump(agg.finalize(), f, indent=2)

    print(f"Wrote {len(rows)} rows to {args.out}")
    print(f"Wrote aggregate to {agg_path}")


if __name__ == "__main__":
    main()
