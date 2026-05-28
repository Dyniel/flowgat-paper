# -*- coding: utf-8 -*-
"""
Build leakage-variant Cosserat sweep NPZ datasets from data/npz_cosserat_sweep.

Reads the noleak Cosserat sweep (already global PCA tangent + unit local radius),
and produces:
  npz_cosserat_sweep_withleak       : x[:,3:6] = unit(y),  x[:,8] = u_mean/u_char
  npz_cosserat_sweep_leak_dir_only  : x[:,3:6] = unit(y),  x[:,8] = noleak local_R
  npz_cosserat_sweep_leak_mag_only  : x[:,3:6] = noleak PCA, x[:,8] = u_mean/u_char
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", default="data/npz_cosserat_sweep")
    ap.add_argument("--out_root", default="data")
    args = ap.parse_args()

    src = Path(args.src_dir)
    out_root = Path(args.out_root)
    targets = {
        "wl": out_root / "npz_cosserat_sweep_withleak",
        "do": out_root / "npz_cosserat_sweep_leak_dir_only",
        "mo": out_root / "npz_cosserat_sweep_leak_mag_only",
    }
    for p in targets.values():
        p.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / "split.json", p / "split.json")

    with open(src / "split.json") as f:
        split = json.load(f)
    cases = sorted(set(sum((split.get(k, []) for k in ("train", "val", "test")), [])))
    print(f"[cosserat-variants] processing {len(cases)} cases")

    for i, case in enumerate(cases):
        sp = src / f"{case}.npz"
        if not sp.exists():
            print(f"  [ERR] missing {sp}")
            continue
        d = np.load(sp, allow_pickle=True)
        arrays = {k: d[k] for k in d.files}
        d.close()

        y = arrays["y"].astype(np.float32)
        u_char = float(arrays["u_char"])
        wall = arrays["wall_mask"].astype(bool)

        # unit(y) -- direction prior, with safe fallback for tiny magnitudes
        umag = np.linalg.norm(y, axis=1, keepdims=True).clip(min=1e-8)
        u_dir = (y / umag).astype(np.float32)
        # zero out wall (no slip)
        u_dir[wall] = 0.0

        # u_mean / u_char as magnitude leak  (cf. preprocess_vmr convention)
        u_norm = np.linalg.norm(y, axis=1).astype(np.float32)
        u_leak_mag = (u_norm / max(u_char, 1e-8)).astype(np.float32)

        # base x (noleak)
        x_nl = arrays["x"].astype(np.float32)

        # withleak
        x_wl = x_nl.copy()
        x_wl[:, 3:6] = u_dir
        x_wl[:, 8] = u_leak_mag
        a_wl = dict(arrays); a_wl["x"] = x_wl
        a_wl["feature_mode"] = np.array("cosserat_sweep_withleak_v1")
        np.savez_compressed(targets["wl"] / f"{case}.npz", **a_wl)

        # leak_dir_only
        x_do = x_nl.copy()
        x_do[:, 3:6] = u_dir
        a_do = dict(arrays); a_do["x"] = x_do
        a_do["feature_mode"] = np.array("cosserat_sweep_leak_dir_only_v1")
        np.savez_compressed(targets["do"] / f"{case}.npz", **a_do)

        # leak_mag_only
        x_mo = x_nl.copy()
        x_mo[:, 8] = u_leak_mag
        a_mo = dict(arrays); a_mo["x"] = x_mo
        a_mo["feature_mode"] = np.array("cosserat_sweep_leak_mag_only_v1")
        np.savez_compressed(targets["mo"] / f"{case}.npz", **a_mo)

        if i % 5 == 0 or i == len(cases) - 1:
            print(f"  [{i+1}/{len(cases)}] {case}")

    print(f"[cosserat-variants] DONE; wrote to: {', '.join(str(p) for p in targets.values())}")


if __name__ == "__main__":
    main()
