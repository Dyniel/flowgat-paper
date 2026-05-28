"""
Split fig4 (peak-velocity localisation on test cases) into independent
per-case PDFs with rasterised mesh background.

Why: the original fig4_peak_world_map.pdf bundles 5 cases x ~80k mesh
nodes as vector scatter primitives -> ~30 MB and slow Overleaf loads.
Each per-case PDF here uses rasterized=True on the gray point cloud,
which packs the mesh background into a single embedded PNG bitmap while
keeping the markers, labels and axes as vectors. Result: < 1 MB per
case PDF, fast Overleaf load, no visual degradation.

Outputs (into --outdir, default: results/figures/):
  fig4a_0007.pdf            # healthy
  fig4b_0017.pdf            # rigid coarctation 1
  fig4c_0020.pdf            # rigid coarctation 2
  fig4d_0225.pdf            # FSI coarctation 1
  fig4e_0226.pdf            # FSI coarctation 2

Run (from project root):
  /users/scratch1/dancies/conda_envs/py312/bin/python src/make_fig4_split.py
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       9.0,
    "axes.titlesize":  10.0,
    "axes.labelsize":  9.0,
    "axes.linewidth":  0.8,
    "savefig.dpi":     300,
    "figure.dpi":      150,
})

VARIANT_ORDER  = ["withleak", "leak_dir_only", "leak_mag_only", "noleak"]
VARIANT_LABELS = {
    "withleak":      "withleak",
    "leak_dir_only": "leak_dir_only",
    "leak_mag_only": "leak_mag_only",
    "noleak":        "noleak",
}
VARIANT_COLORS = {
    "withleak":      "#c0392b",
    "leak_dir_only": "#e67e22",
    "leak_mag_only": "#9b59b6",
    "noleak":        "#2980b9",
}

# stable per-case panel ordering and file labels
PANEL_ORDER = [
    ("0007", "a", "0007 — healthy"),
    ("0017", "b", "0017 — CoA rigid"),
    ("0020", "c", "0020 — CoA rigid"),
    ("0225", "d", "0225 — CoA FSI"),
    ("0226", "e", "0226 — CoA FSI"),
]


def _collect(predictions_dir: Path) -> Dict[str, Dict]:
    """Reproduce the aggregation from fig4_peak_world_map (v2)."""
    by_case: Dict[str, Dict] = {}
    variants = [d.name for d in sorted(predictions_dir.iterdir())
                if d.is_dir() and d.name in VARIANT_ORDER]
    if not variants:
        return by_case
    for v in variants:
        for seed_dir in sorted((predictions_dir / v).iterdir()):
            if not seed_dir.name.startswith("seed_"):
                continue
            for npz in sorted(seed_dir.glob("test_*.npz")):
                d = np.load(npz, allow_pickle=False)
                pred = d["pred"]; y = d["y"]; pos = d["pos"]
                cid = str(d["case_id"])
                speed_y = np.linalg.norm(y, axis=-1)
                speed_p = np.linalg.norm(pred, axis=-1)
                pt = pos[int(np.argmax(speed_y))]
                pp = pos[int(np.argmax(speed_p))]
                slot = by_case.setdefault(cid, {"pos": pos, "true": pt, "preds": {}})
                slot["preds"].setdefault(v, []).append(pp)
    return by_case


def _short_cid(cid: str) -> str:
    return cid.replace("_3D_RIGID", "").replace("_3D_FSI_REST", "")


def _render_panel(ax, slot: Dict, title: str, *, show_legend: bool):
    pos = slot["pos"]
    # rasterized=True is the critical change: packs the 80k-point mesh
    # background into a single embedded raster image inside the PDF.
    ax.scatter(pos[:, 0] * 1000, pos[:, 1] * 1000,
               s=0.05, c="#dddddd", alpha=0.6, marker=".",
               rasterized=True)
    for v, preds_list in slot["preds"].items():
        preds = np.array(preds_list)
        mean_p = preds.mean(axis=0)
        ax.scatter(mean_p[0] * 1000, mean_p[1] * 1000, s=130,
                   color=VARIANT_COLORS.get(v, "#888"),
                   label=VARIANT_LABELS.get(v, v),
                   edgecolors="black", linewidths=0.7, zorder=4)
        ax.scatter(preds[:, 0] * 1000, preds[:, 1] * 1000, s=22,
                   color=VARIANT_COLORS.get(v, "#888"), alpha=0.45,
                   zorder=3)
    pt = slot["true"]
    ax.scatter(pt[0] * 1000, pt[1] * 1000, s=240, marker="*",
               color="black", label="true peak", zorder=5)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    if show_legend:
        ax.legend(fontsize=7, loc="best", framealpha=0.85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions",
                    default="results/predictions",
                    help="root of per-variant prediction NPZ dumps")
    ap.add_argument("--outdir", default="results/figures",
                    help="output directory for fig4{a..e}_*.pdf")
    ap.add_argument("--raster-dpi", type=int, default=200,
                    help="DPI for the rasterised mesh background inside PDF")
    args = ap.parse_args()

    root  = Path(__file__).resolve().parent.parent
    pred  = (root / args.predictions).resolve()
    outd  = (root / args.outdir).resolve()
    outd.mkdir(parents=True, exist_ok=True)

    by_case = _collect(pred)
    if not by_case:
        raise SystemExit(f"no test predictions under {pred}")

    # match cases to PANEL_ORDER by 4-digit prefix
    short_to_full = {cid[:4]: cid for cid in by_case}

    for short, letter, title in PANEL_ORDER:
        full = short_to_full.get(short)
        if full is None:
            print(f"[fig4{letter}] missing case {short} — skip")
            continue
        slot = by_case[full]

        fig, ax = plt.subplots(1, 1, figsize=(4.2, 4.2))
        _render_panel(ax, slot, title, show_legend=(letter == "a"))
        fig.tight_layout()
        pdf = outd / f"fig4{letter}_{short}.pdf"
        fig.savefig(pdf, bbox_inches="tight", dpi=args.raster_dpi)
        plt.close(fig)
        size_kb = pdf.stat().st_size / 1024.0
        print(f"[fig4{letter}] -> {pdf}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
