#!/usr/bin/env python3
"""Repository-level reproducibility checks for the publication release.

This script is intentionally lightweight: it uses only the Python standard
library so that reviewers can run it before installing the training stack.
It checks the public-repo contract: small tracked files, no heavy datasets or
checkpoints, complete released CSV/JSON/figure artefacts, and a consistent
three-seed headline figure table.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "PUBLICATION_MANIFEST.md",
    "environment.yml",
    "environment.lock.yml",
    "requirements.txt",
    "pyproject.toml",
    "Makefile",
    "paper/main.tex",
    "paper/refs.bib",
    "src/train.py",
    "src/evaluate.py",
    "src/analyze.py",
    "src/make_fig_clinical_headline.py",
    "src/flowgnn_aorta/__init__.py",
    "configs/withleak.yaml",
    "configs/noleak.yaml",
    "jobs/SUBMIT_ALL_SR.sh",
    "results/manifest.json",
    "results/figures/fig_clinical_headline.pdf",
    "results/figures/fig_clinical_headline.png",
    "results/figures/fig_clinical_headline_values.csv",
    "results/diagnostics/subend/dp_investigation.md",
    "docs/REPRODUCIBILITY.md",
    "docs/DATA_ACCESS.md",
    "docs/RESULTS_INDEX.md",
]

FORBIDDEN_PREFIXES = [
    "data/",
    "logs/",
    "wandb/",
    "results/checkpoints/",
    "results/predictions/",
    "__pycache__/",
]

FORBIDDEN_SUFFIXES = [
    ".npz",
    ".pt",
    ".ckpt",
    ".pyc",
    ".vtu",
    ".vtk",
    ".h5",
    ".hdf5",
]


class CheckFailure(RuntimeError):
    pass


def run_git_ls_files() -> list[str]:
    try:
        out = subprocess.check_output(
            ["git", "ls-files"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [
            str(p.relative_to(ROOT))
            for p in ROOT.rglob("*")
            if p.is_file() and ".git" not in p.parts
        ]
    return [line.strip() for line in out.splitlines() if line.strip()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailure(message)


def check_required_paths() -> None:
    missing = [p for p in REQUIRED_PATHS if not (ROOT / p).exists()]
    require(not missing, "missing required paths: " + ", ".join(missing))


def check_tracked_file_policy(tracked: list[str], max_file_mb: float, max_repo_mb: float) -> None:
    forbidden = []
    total_bytes = 0
    too_large = []

    for rel in tracked:
        path = ROOT / rel
        if not path.exists() or not path.is_file():
            continue
        total_bytes += path.stat().st_size
        if any(rel.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            forbidden.append(rel)
        if any(rel.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
            forbidden.append(rel)
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > max_file_mb:
            too_large.append((rel, size_mb))

    require(not forbidden, "forbidden tracked artefacts: " + ", ".join(sorted(set(forbidden))[:25]))
    require(
        not too_large,
        "tracked files above limit: "
        + ", ".join(f"{rel} ({size_mb:.1f} MiB)" for rel, size_mb in too_large),
    )
    total_mb = total_bytes / (1024 * 1024)
    require(total_mb <= max_repo_mb, f"tracked payload is {total_mb:.1f} MiB > {max_repo_mb:.1f} MiB")


def check_manifest() -> None:
    manifest_path = ROOT / "results/manifest.json"
    with manifest_path.open() as fh:
        manifest = json.load(fh)
    for key in ["paper_id", "build_date_utc", "variants", "seeds", "splits", "datasets", "results"]:
        require(key in manifest, f"results/manifest.json missing key: {key}")
    require(set(["withleak", "leak_dir_only", "leak_mag_only", "noleak"]).issubset(manifest["variants"]),
            "manifest variants do not include the four leakage variants")
    require(set([1337, 2026, 777]).issubset(set(manifest["seeds"])),
            "manifest seeds do not include 1337, 2026, 777")


def check_result_counts() -> None:
    per_seed = list((ROOT / "results/per_seed").glob("*_aggregate.json"))
    figures = list((ROOT / "results/figures").glob("*"))
    diagnostics = list((ROOT / "results/diagnostics").rglob("*"))
    diagnostics = [p for p in diagnostics if p.is_file()]

    require(len(per_seed) >= 180, f"expected many per-seed aggregate JSONs, found {len(per_seed)}")
    require(len(figures) >= 20, f"expected publication figures, found {len(figures)}")
    require(len(diagnostics) >= 250, f"expected released diagnostics, found {len(diagnostics)}")


def check_clinical_headline_values() -> None:
    path = ROOT / "results/figures/fig_clinical_headline_values.csv"
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    require(len(rows) == 64, f"clinical headline table should have 64 rows, found {len(rows)}")
    bad_seed_rows = [r for r in rows if int(float(r["n_seeds"])) != 3]
    require(not bad_seed_rows, "clinical headline values must use exactly three seeds in every row")

    expected_domains = {"VMR aortas", "Womersley pipe", "Cosserat sweep", "U-bend CFD"}
    expected_variants = {"withleak", "dir_only", "mag_only", "noleak"}
    require({r["domain"] for r in rows} == expected_domains, "clinical headline domains changed unexpectedly")
    require({r["variant"] for r in rows} == expected_variants, "clinical headline variants changed unexpectedly")


def check_docs_have_links() -> None:
    manifest = (ROOT / "PUBLICATION_MANIFEST.md").read_text()
    readme = (ROOT / "README.md").read_text()
    require("https://github.com/Dyniel/flowgat-paper" in manifest, "manifest should name the GitHub repository")
    require("docs/REPRODUCIBILITY.md" in readme, "README should link the reproducibility guide")
    require("docs/DATA_ACCESS.md" in readme, "README should link the data-access guide")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-file-mb", type=float, default=40.0)
    parser.add_argument("--max-repo-mb", type=float, default=150.0)
    args = parser.parse_args(argv)

    checks = [
        ("required paths", lambda: check_required_paths()),
        ("tracked-file policy", lambda: check_tracked_file_policy(run_git_ls_files(), args.max_file_mb, args.max_repo_mb)),
        ("results manifest", check_manifest),
        ("result counts", check_result_counts),
        ("clinical headline seed policy", check_clinical_headline_values),
        ("documentation links", check_docs_have_links),
    ]

    for label, fn in checks:
        try:
            fn()
        except CheckFailure as exc:
            print(f"[verify] FAIL {label}: {exc}", file=sys.stderr)
            return 1
        print(f"[verify] PASS {label}")
    print("[verify] release reproducibility checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
