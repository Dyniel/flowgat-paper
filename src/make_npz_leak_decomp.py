# -*- coding: utf-8 -*-
"""
Build the 2 feature-decomposition NPZ datasets for the controlled leakage
ablation, derived from the existing withleak + noleak NPZ files.

Schema definitions (only x[:,3:6] and x[:,8] differ between variants):

  withleak       : x[:,3:6] = unit(y),       x[:,8] = u_mean/u_char       (both leak)
  noleak         : x[:,3:6] = PCA tangent,   x[:,8] = local radius norm   (both geom)
  leak_dir_only  : x[:,3:6] = unit(y),       x[:,8] = local radius norm   (only dir leak)
  leak_mag_only  : x[:,3:6] = PCA tangent,   x[:,8] = u_mean/u_char       (only mag leak)

This script READS the existing NPZ files (which already have these columns
materialized) and WRITES new NPZ files with mixed columns. All other tensors
(pos, y, edge_index, edge_attr, wall_mask, wall_normal, p_grad, u_char,
length_char) are taken from withleak NPZ (verified identical to noleak NPZ
for non-x tensors — only x differs).

Output:
  data/npz_leak_dir_only/<case>.npz + split.json (copied from noleak/withleak)
  data/npz_leak_mag_only/<case>.npz + split.json

Verification: the script asserts shape consistency between sources and that
the resulting x columns 3:6 and 8 actually come from the intended source.

Usage:
  python src/make_npz_leak_decomp.py \
      --withleak_dir data/npz_withleak \
      --noleak_dir   data/npz_noleak \
      --out_dir_dir  data/npz_leak_dir_only \
      --out_dir_mag  data/npz_leak_mag_only
"""

from __future__ import annotations
import argparse
import os
import json
import shutil
from typing import Dict

import numpy as np


def load_npz(path: str) -> Dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def save_npz(path: str, payload: Dict[str, np.ndarray]):
    np.savez_compressed(path, **payload)


def build_one(case: str, wl_dir: str, nl_dir: str,
              out_dir_dir: str, out_dir_mag: str):
    wl_path = os.path.join(wl_dir, f"{case}.npz")
    nl_path = os.path.join(nl_dir, f"{case}.npz")
    if not os.path.exists(wl_path) or not os.path.exists(nl_path):
        raise FileNotFoundError(f"Missing NPZ: {wl_path} or {nl_path}")

    wl = load_npz(wl_path)
    nl = load_npz(nl_path)

    if wl["x"].shape != nl["x"].shape:
        raise ValueError(
            f"{case}: x shape mismatch wl={wl['x'].shape} nl={nl['x'].shape}"
        )
    if wl["x"].shape[1] < 9:
        raise ValueError(
            f"{case}: x has {wl['x'].shape[1]} cols, expected ≥ 9 (need cols 3:6 and 8)"
        )

    # Sanity: y, pos, edge_index identical between wl and nl
    for k in ("y", "pos"):
        if k in wl and k in nl:
            if not np.allclose(wl[k], nl[k], rtol=0, atol=0):
                # pos and y MUST be identical — if not, datasets diverged
                raise ValueError(f"{case}: {k} differs between withleak/noleak NPZ")

    # === leak_dir_only: base = withleak, replace x[:,8] with noleak's ===
    x_dir = wl["x"].copy()
    x_dir[:, 8] = nl["x"][:, 8]
    payload_dir = dict(wl)  # copy reference dict; will overwrite x
    payload_dir["x"] = x_dir

    # === leak_mag_only: base = noleak, replace x[:,3:6] with noleak's PCA;
    #                    keep noleak x[:,8] -> wait, we want WL's x[:,8] (mag leak).
    #     base = withleak, replace x[:,3:6] with noleak's PCA tangent.
    x_mag = wl["x"].copy()
    x_mag[:, 3:6] = nl["x"][:, 3:6]
    payload_mag = dict(wl)
    payload_mag["x"] = x_mag

    # write
    out_dir_path = os.path.join(out_dir_dir, f"{case}.npz")
    out_mag_path = os.path.join(out_dir_mag, f"{case}.npz")
    save_npz(out_dir_path, payload_dir)
    save_npz(out_mag_path, payload_mag)

    # post-write verification
    chk_dir = load_npz(out_dir_path)
    chk_mag = load_npz(out_mag_path)
    assert np.allclose(chk_dir["x"][:, 3:6], wl["x"][:, 3:6]), f"{case} dir x[:,3:6] != wl"
    assert np.allclose(chk_dir["x"][:, 8],   nl["x"][:, 8]),   f"{case} dir x[:,8] != nl"
    assert np.allclose(chk_mag["x"][:, 3:6], nl["x"][:, 3:6]), f"{case} mag x[:,3:6] != nl"
    assert np.allclose(chk_mag["x"][:, 8],   wl["x"][:, 8]),   f"{case} mag x[:,8] != wl"
    return out_dir_path, out_mag_path, x_dir.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--withleak_dir", required=True)
    ap.add_argument("--noleak_dir", required=True)
    ap.add_argument("--out_dir_dir", required=True)
    ap.add_argument("--out_dir_mag", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir_dir, exist_ok=True)
    os.makedirs(args.out_dir_mag, exist_ok=True)

    # split.json: copy from noleak (split is identical between wl/nl per manifest)
    split_src_candidates = [
        os.path.join(args.noleak_dir, "split.json"),
        os.path.join(args.withleak_dir, "split.json"),
    ]
    split_src = next((p for p in split_src_candidates if os.path.exists(p)), None)
    if split_src is None:
        raise FileNotFoundError("split.json not found in noleak or withleak dir")
    shutil.copy2(split_src, os.path.join(args.out_dir_dir, "split.json"))
    shutil.copy2(split_src, os.path.join(args.out_dir_mag, "split.json"))

    with open(split_src) as f:
        split = json.load(f)
    cases = []
    for k in ("train", "val", "test"):
        if k in split and isinstance(split[k], list):
            cases.extend(split[k])
    cases = sorted(set(cases))
    print(f"[decomp] processing {len(cases)} cases")

    for i, case in enumerate(cases):
        try:
            p_dir, p_mag, n = build_one(
                case, args.withleak_dir, args.noleak_dir,
                args.out_dir_dir, args.out_dir_mag,
            )
            if i % 10 == 0 or i == len(cases) - 1:
                print(f"  [{i+1}/{len(cases)}] {case}: N={n}")
        except Exception as e:
            print(f"  [ERR] {case}: {e}")
            raise

    print(f"[decomp] DONE: {len(cases)} cases × 2 variants")
    print(f"  -> {args.out_dir_dir}")
    print(f"  -> {args.out_dir_mag}")


if __name__ == "__main__":
    main()
