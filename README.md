# Direction is geometric, magnitude is not, and Bernoulli pressure drop alone deceives

**A four-variant clinical-quantity audit of graph neural network surrogates for vascular blood flow.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Submission: Scientific Reports](https://img.shields.io/badge/submission-Scientific%20Reports-blue.svg)](paper/main.tex)
[![DOI: pending](https://img.shields.io/badge/DOI-pending-lightgrey.svg)](#citation)

This repository accompanies a *Scientific Reports* submission that audits
graph neural network (GNN) surrogates for cardiovascular blood flow by
training four feature-leakage variants of the same backbone — direction
leaked, magnitude leaked, both, or neither — on four flow domains:
patient-specific aortas (Vascular Model Repository), an analytical
Womersley pipe, a parametric Cosserat curved-tube sweep, and synthetic
U-bend CFD cases. Across roughly 200 model–data combinations the
direction channel suffices to reproduce the full-leakage angular accuracy
while the magnitude channel does not, in every domain.

The repository additionally documents a methodological finding: the
clinically prominent simplified-Bernoulli pressure-drop metric
$\Delta P = 4\,v^{2}$ can be "won" on the U-bend cohort by a model that
collapses to a near-constant magnitude prediction whose per-frame
correlation with ground truth is statistically zero. We therefore propose
that vascular-flow GNN surrogate evaluations be reported on a
**clinical triplet** — pressure drop, wall shear stress, peak velocity
localisation — and never on pressure drop alone.

## TL;DR

| Metric (test split, mean over 3 seeds) | VMR (n=5) | Womersley (n=6) | Cosserat (n=36) | U-bend (n=75 frames) |
|---|---:|---:|---:|---:|
| Angular error (withleak / noleak) | 4.0° / 44.5° | 17.4° / 69.0° | 1.9° / 76.4° | 5.7° / 60.9° |
| WSS MAE [Pa] (withleak / noleak) | 13.5 / 15.7 | 0.06 / 0.04 | 0.50 / 0.80 | **0.23 / 0.75** |
| Peak loc. MAE [mm] (withleak / noleak) | 44.4 / 51.7 | – | – | **34.5 / 70.7** |
| dP MAE [mmHg] (withleak / noleak) | 22.6 / 22.3 | 0.20 / 0.19 | 2.52 / 3.14 | 8.34 / **4.79**\* |

\* The U-bend dP inversion (noleak "wins" by 3.55 mmHg, 95% CI [1.41, 5.57])
is the headline example of the magnitude-collapse failure mode that the
clinical triplet exposes. See `paper/main.tex` §"A clinical metric that
deceives" and `results/diagnostics/subend/dp_investigation.md` for the
per-frame mechanism.

The full clinical-headline panel is at
[`results/figures/fig_clinical_headline.pdf`](results/figures/fig_clinical_headline.pdf);
the pressure-drop scatter that exposes the magnitude collapse is at
[`results/figures/subend_dp_scatter.pdf`](results/figures/subend_dp_scatter.pdf).

## Repository layout

```
.
├── paper/                  Manuscript sources (main.tex, refs.bib, cover_letter.tex)
├── src/                    All Python — model, training, diagnostics, figure generators
├── jobs/                   Slurm launchers (1..34 + submit wrappers)
├── configs/                YAML configs (20 files: 4 variants × 5 datasets)
├── results/                Text-only diagnostics, CSVs, JSONs, figures
│   ├── manifest.json       SHA-256 of every NPZ shard used in the paper
│   ├── per_seed/           96 per-(variant, seed, split) aggregate JSON + CSV
│   ├── diagnostics/        Cosserat / sUbend / mesh-refinement diagnostics
│   ├── bootstrap/          Paired-bootstrap CIs (8 sUbend contrasts + 4 VMR)
│   ├── stratified/         Per-pathology VMR breakdown
│   └── figures/            All publication figures (PDF + PNG)
├── docs/
│   ├── STRATEGY_SR.md      Project narrative + decision log (D1–D8)
│   └── archive/            CP-version strategy snapshot (frozen 2026-05-19)
├── environment.yml         Conda env spec
├── requirements.txt        pip lockfile cross-check
├── LICENSE                 MIT
├── CITATION.cff            Zenodo-compatible citation
└── PUBLICATION_MANIFEST.md What is in this repo vs Zenodo vs derivable from sources
```

## Reproducing the analysis

The four data-deposit tiers from `PUBLICATION_MANIFEST.md` correspond to
four reproduction levels of increasing compute cost.

### Level 1 — re-render figures from released CSVs (≈ 1 minute, no GPU)

This is the cheapest re-run. Everything you need is in this repo.

```bash
conda env create -f environment.yml
conda activate flowgat-paper

# Regenerate the clinical-headline figure (proposed Fig 1):
python src/make_fig_clinical_headline.py \
    --per_seed_dir results/per_seed \
    --out_dir      results/figures

# Regenerate the U-bend dP-scatter diagnostic from released prediction NPZs
# (skip if you don't want to download the Zenodo deposit — the CSV +
# Markdown report are already in results/diagnostics/subend/):
python src/subend_dp_investigation.py \
    --predictions_root <path-to-Zenodo-predictions> \
    --out_dir          results/diagnostics/subend \
    --figures_dir      results/figures
```

### Level 2 — re-run the diagnostics from released predictions (≈ 30 min, no GPU)

Requires the Zenodo deposit (DOI: pending; ≈ 8 GB; `predictions/` and
`checkpoints/`). Extract it next to this repository and re-launch the
post-training diagnostics:

```bash
bash jobs/SUBMIT_E7_FOLLOWUP.sh   # jobs 25 + 26 + 27 + 28 in parallel
```

### Level 3 — re-evaluate from released checkpoints (≈ 1 hour, 1 GPU)

You also need the source NPZ datasets. The build recipe is in
`src/make_npz_*.py`; pointers to the public source data are in
`PUBLICATION_MANIFEST.md` section C. After regenerating
`data/npz_<variant>/`, run:

```bash
sbatch jobs/02_eval_all.sh   # re-evaluates every (variant, seed) ckpt
sbatch jobs/04_dump_predictions.sh
```

### Level 4 — full re-training from scratch (≈ 200 GPU-hours)

```bash
bash jobs/SUBMIT_ALL_SR.sh   # builds NPZs, trains all 48 models, aggregates
```

## Citation

See `CITATION.cff`. Once a Zenodo DOI is reserved, citations will resolve
to a versioned snapshot of this repository.

## License

MIT for code and documentation. The released CSVs and JSONs are also MIT.
Patient-specific aortic geometries are © Vascular Model Repository; the
U-bend CFD cohort is © the original newCFD\_dataset distribution
(polybox.ethz.ch). Trained model weights distributed via Zenodo are
released under CC-BY-4.0.

## Contributions and conflicts

See `paper/main.tex` §"Author contributions" and §"Competing interests".

## Contact

A. Daniel Cieslak — `cieslak.a.daniel@gmail.com`
