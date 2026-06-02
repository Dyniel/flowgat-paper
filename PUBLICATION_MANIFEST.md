# Publication manifest — what goes where

**Date:** 2026-06-02
**Target submission:** *Scientific Reports*
**Reproducibility target:** Nature Communications / Nature Portfolio reviewer-ready release.
**Repo size budget on GitHub:** ≤ 150 MB tracked payload (well under the 1 GB hard limit). Zenodo for the heavy data + checkpoints + predictions.

---

## A. GitHub repository (`release/`, reviewer-facing public repo)

URL: https://github.com/Dyniel/flowgat-paper

Everything needed for a reader to (a) read the paper, (b) audit the code,
(c) re-run the figure generation against released artefacts, and
(d) re-train the models if they download the Zenodo data deposit.

| Path in GitHub repo | Source path | Size | Notes |
|---|---|---|---|
| `README.md` | (new) | — | SR-style overview, clinical-triplet headline, four-domain Fig 1 inlined |
| `LICENSE` | (new) | — | MIT |
| `CITATION.cff` | (new) | — | Zenodo-compatible citation block |
| `.gitignore` | (new) | — | Excludes NPZs, checkpoints, predictions, logs, wandb, env caches |
| `.github/workflows/reproducibility.yml` | (new) | — | Lightweight CI: verifies release contract and re-renders headline figure |
| `Makefile` | (new) | — | One-command shortcuts for verify, figures, diagnostics, eval, train, Zenodo pack |
| `pyproject.toml` | (new) | — | Install metadata, lightweight dependencies, pytest config |
| `scripts/verify_release.py` | (new) | — | Standard-library release-integrity checker |
| `tests/test_release_integrity.py` | (new) | — | Pytest wrapper around the release-integrity checker |
| `environment.yml` | `env/environment.yml` | 4 KB | Conda env spec |
| `requirements.txt` | `env/requirements.txt` | 1 KB | pip lockfile cross-check |
| `paper/main.tex` | `paper/main.tex` | 88 KB | SR-reframe manuscript (1679 lines) |
| `paper/refs.bib` | `paper/refs.bib` | ~30 KB | Bibliography |
| `paper/cover_letter.tex` | `paper/cover_letter.tex` | 5 KB | SR cover letter draft |
| `paper/figures/` | `paper/figures/` | ≈ 0.4 MB | Schema/diagram figures (Fig 1 schema, etc.) |
| `src/` | `src/*.py`, `src/flowgnn_aorta/**/*.py` | ~600 KB | 46 .py files; `__pycache__/` excluded |
| `jobs/` | `jobs/*.sh` | ~112 KB | Slurm launcher scripts (1-34 + submit wrappers) |
| `configs/` | `configs/*.yaml` | ~77 KB | 32 YAML configs across VMR, Womersley, Cosserat, sUbend, SAGE, and no-BC variants |
| `results/manifest.json` | `results/manifest.json` | 8 KB | SHA-256 manifest of NPZ shards used in the paper |
| `results/per_seed/` | `results/per_seed/` | 3.0 MB | All per-seed CSV + JSON aggregates (404 files) |
| `results/diagnostics/` | `results/diagnostics/` | 2.3 MB | Cosserat/sUbend/mesh-refinement/SAGE/no-BC CSVs + MDs |
| `results/bootstrap/` | `results/bootstrap/` | 74 KB | Paired bootstrap CSVs + MDs |
| `results/stratified/` | `results/stratified/` | 23 KB | Per-pathology VMR breakdown |
| `results/figures/` | `results/figures/` | 32 MB | All publication figures (PDF + PNG) |
| `docs/REPRODUCIBILITY.md` | (new) | — | Four-level reviewer reproduction protocol |
| `docs/DATA_ACCESS.md` | (new) | — | GitHub vs Zenodo vs source-data policy |
| `docs/RESULTS_INDEX.md` | (new) | — | Claim-to-file map for released results |
| `docs/STRATEGY_SR.md` | `STRATEGY_SR.md` | 16 KB | Project narrative + decisions D1–D8 |
| `docs/archive/STRATEGY_CP_archived_20260519.md` | `docs/archive/` | 12 KB | CP-version archive |
| `PUBLICATION_MANIFEST.md` | (this file) | — | Self-describing |

**Total tracked payload remains below 150 MB** — large datasets,
checkpoints, prediction dumps, logs, and W&B telemetry stay outside git.

---

## B. Zenodo deposit (≈ 8 GB; one record)

Trained model artefacts that are too large for GitHub but are not derivable
from public sources without re-running compute.

| Path in Zenodo zip | Source path | Size | Notes |
|---|---|---|---|
| `checkpoints/` | `results/checkpoints/` | 1.7 GB | PyTorch state dicts for every (variant, seed) — 48 best.pt files (4 vars × 3 seeds × 4 domains) |
| `predictions/` | `results/predictions/` | 6.3 GB | Per-(variant, seed, split, case-frame) NPZ velocity dumps. Used by all downstream diagnostics. |
| `MANIFEST.txt` | (generated) | — | SHA-256 per file; cross-link to GitHub commit |
| `README.md` | (generated) | — | "How to use this deposit with the GitHub repo" |

Zenodo limit per record: 50 GB. We are at ~16 % of that budget — single
record is comfortable. DOI'd, citable, version-pinnable.

---

## C. NOT distributed — derivable from public sources

These are intentionally excluded from both GitHub and Zenodo. The build
recipe under `src/make_npz_*.py` regenerates them deterministically given
the listed public inputs.

| Excluded | Size | Public source / build script |
|---|---|---|
| `data/npz_{noleak,withleak,leak_dir_only,leak_mag_only}/` | 21 GB | VMR cohort (vascularmodel.com) + `src/preprocess_vmr.py` + `src/make_npz_*.py` |
| `data/npz_noleak_centerline/` | 5.3 GB | Same VMR + `src/centerline_tangent.py` + `make_npz_local_tangent.py` |
| `data/npz_womersley{,_*,_meshref_*}/` | 10 GB | Analytic — `src/make_npz_womersley.py` (deterministic from seed 22) |
| `data/npz_cosserat_sweep{,_*}/` | 7 GB | Analytic — `src/make_npz_cosserat_sweep{,_variants}.py` |
| `data/npz_subend{,_*}/` | 21 GB | newCFD_dataset (polybox.ethz.ch) + `src/make_npz_subend{,_variants}.py` |
| `data/external/newCFD_dataset/` | 43 GB | Original polybox release — see README for URL |
| `env/` (compiled venv) | — | Re-create via `environment.yml` |
| `wandb/` | — | Local training telemetry; not used downstream |
| `logs/` (Slurm) | — | Cluster-specific |
| `submission.zip`, `flowgat_paper_local_*.zip` | — | Old build snapshots |

---

## D. Pre-existing files that are NOT in the SR release

Conscious decisions to exclude (historical, off-narrative):

| File | Reason |
|---|---|
| `README.md` (root, CP-style) | Replaced by new SR-style README in `release/README.md`. |
| `tea_debug.log` | Empty placeholder, no value to readers |

---

## Build steps for the release directory

Driven by `release/.build.sh` (created in step 2). Idempotent: re-running
overwrites `release/` without touching source. See that script.

## Zenodo upload — manual

1. Visit zenodo.org → "New upload".
2. Set creators, title (match `main.tex`), keywords, version, license = CC-BY-4.0 (data) / MIT (code-only deposit if separate).
3. Upload `flowgat_paper_zenodo_<DATE>.zip` created by `make zenodo-pack`.
4. Reserve DOI, copy it into `CITATION.cff`, `README.md`, and this manifest, commit, push, then publish on Zenodo.

The order matters: Zenodo DOI → CITATION.cff → GitHub release tag → Zenodo
"link to GitHub release" (auto-fills version metadata).
