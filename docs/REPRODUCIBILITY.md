# Reproducibility guide

This repository is organised as a reviewer-facing release: small files in
GitHub, heavy artefacts in a versioned data deposit, and source datasets
regenerated from their original public distributions.

The four levels below are intentionally separated so that a reviewer can run
the fastest checks first and only move to GPU work if needed.

## Quick integrity check

Run this immediately after cloning:

```bash
python scripts/verify_release.py
```

Expected result:

```text
[verify] release reproducibility checks passed
```

The verifier checks that required files are present, the repository does not
track heavy datasets/checkpoints/predictions, the released results are
available, and the headline figure table uses exactly the three paper seeds:
`1337`, `2026`, and `777`.

## Environment

Preferred Conda setup:

```bash
conda env create -f environment.yml
conda activate flowgat-paper
```

For lightweight figure checks without CUDA:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For training or evaluation from checkpoints, install the full Conda
environment because the model stack uses CUDA PyTorch, PyTorch Geometric, VTK,
and PyVista.

The exact environment used on the production cluster is preserved in
`environment.lock.yml`; it is intentionally not the default because it contains
site-specific paths.

## Seed policy

The publication headline uses three seeds across all four domains:

```text
1337, 2026, 777
```

The VMR aorta folder also contains historical extra seeds (`42`, `3407`) for
auditability. Figure-generation code defaults to the three paper seeds and can
include every available seed with:

```bash
python src/make_fig_clinical_headline.py \
    --per_seed_dir results/per_seed \
    --out_dir results/figures \
    --seeds all
```

## Level 1: re-render figures from released tables

Cost: about 1 minute, CPU only, no `.npz` datasets, no checkpoints.

```bash
make verify
make figures
```

Equivalent explicit command:

```bash
python src/make_fig_clinical_headline.py \
    --per_seed_dir results/per_seed \
    --out_dir results/figures
```

Expected outputs:

```text
results/figures/fig_clinical_headline.pdf
results/figures/fig_clinical_headline.png
results/figures/fig_clinical_headline_values.csv
```

The values CSV has 64 rows: 4 domains x 4 leakage variants x 4 clinical
metrics.

## Level 2: re-run diagnostics from released predictions

Cost: about 30 minutes, CPU only.

Requires the heavy artefact deposit containing `results/predictions/`.
Extract it at repository root so that these folders exist:

```text
results/predictions/
results/checkpoints/
```

Then run:

```bash
make level2-diagnostics
```

This launches the Slurm wrappers for:

- U-bend direction-identifiability aggregation.
- U-bend pressure-drop / magnitude-collapse investigation.
- Paired bootstrap confidence intervals.
- Clinical-headline figure regeneration.

Primary outputs:

```text
results/diagnostics/subend/summary.md
results/diagnostics/subend/dp_investigation.md
results/bootstrap/*/bootstrap_*.md
results/figures/subend_dp_scatter.pdf
```

## Level 3: re-evaluate from released checkpoints

Cost: about 1 GPU-hour after source NPZ datasets are available.

Requires both:

- Heavy artefact deposit: `results/checkpoints/`.
- Regenerated source NPZ datasets under `data/npz_<variant>/`.

Run:

```bash
make level3-eval
```

Primary outputs:

```text
results/per_seed/*_aggregate.json
results/per_seed/*.csv
results/predictions/
```

After this, re-run Level 1 and Level 2.

## Level 4: full retraining from scratch

Cost: about 200 GPU-hours for the complete four-domain audit.

Run on the Slurm cluster after source datasets are available:

```bash
make level4-train
```

This submits the staged build, training, aggregation, and diagnostic jobs.
Individual launchers are kept in `jobs/` for inspection and for rerunning only
one section of the pipeline.

## Hardware notes

The production runs used A100 80 GB GPUs for training. The code does not depend
on A100-specific kernels, but memory and wall time were budgeted for that
hardware. CPU-only steps were run on login or batch CPU nodes.

## Determinism notes

The split files and seeds are fixed. GPU training can still differ at the last
decimal place across CUDA/cuDNN/PyTorch versions. The release therefore treats
the following as exact reproducibility artefacts:

- Published per-seed CSV and JSON tables in `results/per_seed/`.
- Published diagnostics in `results/diagnostics/`.
- Published figure-value CSVs in `results/figures/`.
- SHA-256 manifests in `results/manifest.json` and the Zenodo-side
  `MANIFEST.txt`.

Full retraining should reproduce the qualitative ordering and reported
confidence intervals, not byte-identical checkpoints.
