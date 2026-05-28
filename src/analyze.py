#!/usr/bin/env python3
"""
Aggregate per-seed CSV/JSON evals into 3-seed mean ± std tables and the
controlled leakage-delta dataframe used by paper figures.

Inputs (assumed produced by jobs/01_train_seed.sh):
  results/per_seed/{variant}_{split}_seed{seed}.csv         (per-case rows)
  results/per_seed/{variant}_{split}_seed{seed}_aggregate.json  (whole-set)
  results/checkpoints/{variant}/seed_{seed}/efficiency.json

Outputs:
  results/aggregated/per_case_3seed.csv
  results/aggregated/aggregate_3seed.csv
  results/aggregated/leakage_delta.csv
  results/aggregated/efficiency.csv
  results/aggregated/summary.md
  results/manifest.json
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd


VARIANTS = ["withleak", "noleak"]
SEEDS_DEFAULT = [1337, 2026, 777]
SPLITS = ["val", "test"]

# Map short metric label -> JSON key in *_aggregate.json (whole-set agg)
JSON_KEY = {
    "pp10":         "val/pp_at_0.100",
    "pp5":          "val/pp_at_0.050",
    "pp2":          "val/pp_at_0.020",
    "pp_auc":       "val/pp_auc_0.000_0.100",
    "angle_err":    "val/angle_err_he_mean_deg",
    "ewrmse_he":    "val/ewrmse_he",
    "rel_err_he":   "val/rel_err_he_mean",
    "speed_err_he": "val/speed_err_he_mean",
    "peak_loc_mm":  "clin/peak_loc_mm_median",
    "peak_loc_mean": "clin/peak_loc_mm_mean",
    "peak_mag_rel": "clin/peak_mag_rel_mean",
    "wss_r2":       "clin/wss_r2_mean",
    "wss_mae":      "clin/wss_mae_Pa_mean",
}

# Map short metric label -> CSV column in per-case CSV
CSV_KEY = {
    "pp10":         "val/pp_at_0.100",
    "pp5":          "val/pp_at_0.050",
    "pp2":          "val/pp_at_0.020",
    "angle_err":    "val/angle_err_he_mean_deg",
    "ewrmse_he":    "val/ewrmse_he",
    "rel_err_he":   "val/rel_err_he_mean",
    "peak_loc_mm":  "clin/peak_loc_mm_median",
    "peak_mag_rel": "clin/peak_mag_rel_mean",
    "wss_r2":       "clin/wss_r2_mean",
}

# Metrics shown in the human summary (whole-set 3-seed table)
SUMMARY_METRICS = ["pp10", "pp5", "angle_err", "ewrmse_he",
                   "peak_loc_mm", "peak_mag_rel", "wss_r2"]
# Metrics aggregated per-case (5 test cases × 3 seeds = 15 obs/condition)
PER_CASE_METRICS = ["pp10", "pp5", "angle_err", "ewrmse_he",
                    "rel_err_he", "peak_loc_mm", "peak_mag_rel"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stats_3seed(values: list[float]) -> tuple[float, float, int]:
    vals = [v for v in values if v is not None
            and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), 0
    if n == 1:
        return float(vals[0]), 0.0, 1
    return float(mean(vals)), float(stdev(vals)), n


def load_per_case(per_seed_dir: Path, variant: str, split: str,
                  seed: int) -> pd.DataFrame | None:
    p = per_seed_dir / f"{variant}_{split}_seed{seed}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["variant"] = variant
    df["split"] = split
    df["seed"] = seed
    return df


def load_aggregate(per_seed_dir: Path, variant: str, split: str,
                   seed: int) -> dict | None:
    p = per_seed_dir / f"{variant}_{split}_seed{seed}_aggregate.json"
    if not p.exists():
        return None
    with open(p) as f:
        # Some entries can be NaN written as JSON literal `NaN` (non-standard)
        text = f.read()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Replace `NaN` -> `null` then parse (NaN is non-standard JSON)
        return json.loads(text.replace("NaN", "null"))


def aggregate_per_case(per_seed_dir: Path, seeds: list[int],
                       out_csv: Path) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        for split in SPLITS:
            frames = [load_per_case(per_seed_dir, variant, split, s)
                      for s in seeds]
            frames = [f for f in frames if f is not None]
            if not frames:
                continue
            df = pd.concat(frames, ignore_index=True)
            case_col = "case_id" if "case_id" in df.columns else "case"
            for case, g in df.groupby(case_col):
                row = {"variant": variant, "split": split, "case": case,
                       "n_seeds": len(g)}
                for short, csv_k in CSV_KEY.items():
                    if csv_k in g.columns:
                        m_mean, m_std, _ = stats_3seed(g[csv_k].tolist())
                        row[f"{short}_mean"] = m_mean
                        row[f"{short}_std"] = m_std
                rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(f"[analyze] per-case 3-seed -> {out_csv} ({len(out)} rows)")
    return out


def aggregate_whole_set(per_seed_dir: Path, seeds: list[int],
                        out_csv: Path) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        for split in SPLITS:
            agg_per_seed = []
            for s in seeds:
                a = load_aggregate(per_seed_dir, variant, split, s)
                if a is None:
                    continue
                agg_per_seed.append((s, a))
            if not agg_per_seed:
                continue
            row = {"variant": variant, "split": split,
                   "n_seeds": len(agg_per_seed)}
            for short, json_k in JSON_KEY.items():
                vals = [a.get(json_k) for _, a in agg_per_seed]
                m_mean, m_std, _ = stats_3seed(vals)
                row[f"{short}_mean"] = m_mean
                row[f"{short}_std"] = m_std
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(f"[analyze] whole-set 3-seed -> {out_csv} ({len(out)} rows)")
    return out


def leakage_delta(per_case: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        wl = per_case[(per_case.variant == "withleak") &
                      (per_case.split == split)].set_index("case")
        nl = per_case[(per_case.variant == "noleak") &
                      (per_case.split == split)].set_index("case")
        cases = sorted(set(wl.index) & set(nl.index))
        for case in cases:
            row = {"split": split, "case": case}
            for short in PER_CASE_METRICS:
                col = f"{short}_mean"
                if col in wl.columns and col in nl.columns:
                    a = float(wl.loc[case, col])
                    b = float(nl.loc[case, col])
                    row[f"{short}_withleak"] = a
                    row[f"{short}_noleak"] = b
                    row[f"{short}_delta"] = a - b
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(f"[analyze] leakage delta -> {out_csv} ({len(out)} rows)")
    return out


def collect_efficiency(ckpt_root: Path, seeds: list[int],
                       out_csv: Path) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        for s in seeds:
            p = ckpt_root / variant / f"seed_{s}" / "efficiency.json"
            if not p.exists():
                continue
            with open(p) as f:
                d = json.load(f)
            d["variant"] = variant
            d["seed"] = s
            rows.append(d)
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(f"[analyze] efficiency -> {out_csv} ({len(out)} rows)")
    return out


def df_to_md(df: pd.DataFrame, float_fmt: str = ".4f") -> str:
    """Tabulate-free Markdown renderer."""
    if df.empty:
        return "_(empty)_"
    cols = list(df.columns)
    def fmt(v):
        if isinstance(v, float):
            if np.isnan(v): return "—"
            return f"{v:{float_fmt}}"
        return str(v)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = "\n".join("| " + " | ".join(fmt(r[c]) for c in cols) + " |"
                     for _, r in df.iterrows())
    return "\n".join([head, sep, body])


def write_summary_md(agg_3seed: pd.DataFrame, leak_delta: pd.DataFrame,
                     eff: pd.DataFrame, out_md: Path,
                     n_seeds_expected: int = 3) -> None:
    lines = ["# Paper results — auto-generated by analyze.py", ""]

    # Headline numbers (test split, 3-seed mean ± std for SUMMARY_METRICS)
    lines += ["## 1. Headline 3-seed test-set metrics", ""]
    test = agg_3seed[agg_3seed.split == "test"]
    if not test.empty:
        rows = []
        for _, r in test.iterrows():
            row = {"variant": r["variant"], "n_seeds": int(r["n_seeds"])}
            for m in SUMMARY_METRICS:
                mean_c, std_c = f"{m}_mean", f"{m}_std"
                if mean_c in test.columns:
                    row[m] = (f"{r[mean_c]:.4f} ± {r[std_c]:.4f}"
                              if not np.isnan(r[mean_c]) else "—")
            rows.append(row)
        lines.append(df_to_md(pd.DataFrame(rows)))
    lines.append("")

    # Headline val
    lines += ["## 2. Headline 3-seed val-set metrics", ""]
    val = agg_3seed[agg_3seed.split == "val"]
    if not val.empty:
        rows = []
        for _, r in val.iterrows():
            row = {"variant": r["variant"], "n_seeds": int(r["n_seeds"])}
            for m in SUMMARY_METRICS:
                mean_c, std_c = f"{m}_mean", f"{m}_std"
                if mean_c in val.columns:
                    row[m] = (f"{r[mean_c]:.4f} ± {r[std_c]:.4f}"
                              if not np.isnan(r[mean_c]) else "—")
            rows.append(row)
        lines.append(df_to_md(pd.DataFrame(rows)))
    lines.append("")

    # Leakage delta (test only)
    lines += ["## 3. Per-case leakage delta (test set, withleak − noleak)", ""]
    if not leak_delta.empty:
        td = leak_delta[leak_delta.split == "test"].copy()
        # Show key delta columns
        keep = ["case"]
        for m in ["pp10", "angle_err", "peak_loc_mm", "ewrmse_he"]:
            for s in ["withleak", "noleak", "delta"]:
                k = f"{m}_{s}"
                if k in td.columns:
                    keep.append(k)
        td = td[keep]
        lines.append(df_to_md(td))
    lines.append("")

    # Efficiency
    lines += ["## 4. Training efficiency", ""]
    if not eff.empty:
        cols = ["variant", "seed", "best_epoch",
                "best_pp10_val", "total_epochs", "mean_epoch_sec",
                "train_time_s", "n_params", "peak_gpu_mb"]
        cols = [c for c in cols if c in eff.columns]
        lines.append(df_to_md(eff[cols]))
    lines.append("")

    # Status warnings
    lines += ["## 5. Run status", ""]
    if eff is not None and not eff.empty:
        ckpt_root = out_md.parent.parent / "checkpoints"
        for _, r in eff.iterrows():
            warn = []
            v = r["variant"]
            s = int(r["seed"])
            tot_ep = int(r.get("total_epochs", 0))
            best_ep = int(r.get("best_epoch", 0))
            train_t = float(r.get("train_time_s", 0.0))
            ckpt = ckpt_root / v / f"seed_{s}" / "best.pt"
            ckpt_age_s = (
                __import__("time").time() - ckpt.stat().st_mtime
                if ckpt.exists() else None
            )
            # If efficiency.json says short training but checkpoint is recent
            # and large, training likely hit a time limit and finalization
            # was skipped — telemetry is stale (smoke residue), but the
            # checkpoint is from a longer run.
            if tot_ep < 50 and train_t < 600 and ckpt.exists():
                ck_size_mb = ckpt.stat().st_size / 1e6
                warn.append(
                    f"⚠ stale efficiency.json (likely smoke residue); "
                    f"checkpoint exists ({ck_size_mb:.1f} MB), real training"
                    f" was longer — see SLURM log for actual epochs/best_ep"
                )
            elif tot_ep < 50:
                warn.append("⚠ smoke-test only (total_epochs < 50)")
            lines.append(f"- {v} seed={s}: "
                         f"best_ep={best_ep} total_ep={tot_ep} "
                         + (" ".join(warn) if warn else ""))
    lines.append("")

    out_md.write_text("\n".join(lines))
    print(f"[analyze] summary -> {out_md}")


def write_manifest(paper_dir: Path, out_json: Path,
                   seeds: list[int]) -> None:
    import datetime
    m: dict = {
        "paper_id": "flowgat-leakage-2026",
        "build_date_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "variants": VARIANTS,
        "seeds": seeds,
        "splits": SPLITS,
        "datasets": {},
        "checkpoints": [],
        "results": {},
    }
    for variant in VARIANTS:
        d = paper_dir / "data" / f"npz_{variant}"
        if d.exists():
            n_npz = len(list(d.glob("*.npz")))
            split_file = d / "split.json"
            m["datasets"][variant] = {
                "path": str(d.relative_to(paper_dir)),
                "n_cases": n_npz,
                "split_sha256": (sha256(split_file)
                                 if split_file.exists() else None),
            }

    ckpt_root = paper_dir / "results" / "checkpoints"
    for variant in VARIANTS:
        for s in seeds:
            ck = ckpt_root / variant / f"seed_{s}" / "best.pt"
            eff = ckpt_root / variant / f"seed_{s}" / "efficiency.json"
            entry = {"variant": variant, "seed": s,
                     "ckpt": (str(ck.relative_to(paper_dir))
                              if ck.exists() else None),
                     "ckpt_sha256": sha256(ck) if ck.exists() else None,
                     "efficiency": (json.load(open(eff))
                                    if eff.exists() else None)}
            m["checkpoints"].append(entry)

    agg = paper_dir / "results" / "aggregated"
    for f in ["per_case_3seed.csv", "aggregate_3seed.csv",
              "leakage_delta.csv", "efficiency.csv", "summary.md"]:
        p = agg / f
        if p.exists():
            m["results"][f] = {"path": str(p.relative_to(paper_dir)),
                               "sha256": sha256(p)}

    with open(out_json, "w") as f:
        json.dump(m, f, indent=2)
    print(f"[analyze] manifest -> {out_json}")


def main() -> None:
    global VARIANTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-dir",
                    default="/users/scratch1/dancies/flowgat_paper")
    ap.add_argument("--seeds", default="1337,2026,777")
    ap.add_argument("--variants", default=",".join(VARIANTS),
                    help="Comma-separated variant names (must match per_seed/ "
                         "filenames and checkpoints/<variant>/ dirs)")
    args = ap.parse_args()

    VARIANTS = [v.strip() for v in args.variants.split(",") if v.strip()]
    paper_dir = Path(args.paper_dir)
    seeds = [int(s) for s in args.seeds.split(",")]
    per_seed_dir = paper_dir / "results" / "per_seed"
    agg_dir = paper_dir / "results" / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root = paper_dir / "results" / "checkpoints"

    per_case = aggregate_per_case(per_seed_dir, seeds,
                                  agg_dir / "per_case_3seed.csv")
    whole = aggregate_whole_set(per_seed_dir, seeds,
                                agg_dir / "aggregate_3seed.csv")
    delta = leakage_delta(per_case, agg_dir / "leakage_delta.csv")
    eff = collect_efficiency(ckpt_root, seeds,
                             agg_dir / "efficiency.csv")
    write_summary_md(whole, delta, eff, agg_dir / "summary.md")
    write_manifest(paper_dir, paper_dir / "results" / "manifest.json", seeds)


if __name__ == "__main__":
    main()
