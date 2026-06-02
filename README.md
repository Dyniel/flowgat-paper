# Direction is geometric, magnitude is not, and Bernoulli pressure drop alone deceives

**A four-variant clinical-quantity audit of graph neural network surrogates for vascular blood flow.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Submission: Scientific Reports](https://img.shields.io/badge/submission-Scientific%20Reports-blue.svg)](paper/main.tex)
[![DOI: pending](https://img.shields.io/badge/DOI-pending-lightgrey.svg)](#citation)
[![Reproducibility checks](https://github.com/Dyniel/flowgat-paper/actions/workflows/reproducibility.yml/badge.svg)](https://github.com/Dyniel/flowgat-paper/actions/workflows/reproducibility.yml)

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

| Metric (test split, mean over seeds 1337/2026/777) | VMR (n=5) | Womersley (n=6) | Cosserat (n=36) | U-bend (n=75 frames) |
|---|---:|---:|---:|---:|
| Angular error (withleak / noleak) | 3.9° / 45.7° | 17.4° / 69.0° | 1.9° / 76.4° | 7.1° / 60.9° |
| WSS MAE [Pa] (withleak / noleak) | 14.3 / 13.9 | 0.06 / 0.04 | 0.50 / 0.80 | **0.23 / 0.75** |
| Peak loc. MAE [mm] (withleak / noleak) | 47.2 / 58.2 | – | – | **34.5 / 70.7** |
| dP MAE [mmHg] (withleak / noleak) | 22.69 / 22.59 | 0.20 / 0.19 | 2.52 / 3.14 | 8.34 / **4.79**\* |

\* The U-bend dP inversion (noleak "wins" by 3.55 mmHg, 95% CI [1.41, 5.57])
is the headline example of the magnitude-collapse failure mode that the
clinical triplet exposes. See `paper/main.tex` §"A clinical metric that
deceives" and `results/diagnostics/subend/dp_investigation.md` for the
per-frame mechanism.

The full clinical-headline panel is at
[`results/figures/fig_clinical_headline.pdf`](results/figures/fig_clinical_headline.pdf);
the pressure-drop scatter that exposes the magnitude collapse is at
[`results/figures/subend_dp_scatter.pdf`](results/figures/subend_dp_scatter.pdf).

## Reviewer quick start

```bash
python scripts/verify_release.py
make figures
```

The full reproducibility protocol is in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md), data and artefact access
is in [`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md), and the claim-to-file map
is in [`docs/RESULTS_INDEX.md`](docs/RESULTS_INDEX.md).

## Repository layout

```
.
├── paper/                  Manuscript sources (main.tex, refs.bib, cover_letter.tex)
├── src/                    All Python — model, training, diagnostics, figure generators
├── jobs/                   Slurm launchers (1..34 + submit wrappers)
├── configs/                YAML configs across domains, variants, and controls
├── results/                Text-only diagnostics, CSVs, JSONs, figures
│   ├── manifest.json       SHA-256 of every NPZ shard used in the paper
│   ├── per_seed/           Per-(variant, seed, split) aggregate JSON + CSV
│   ├── diagnostics/        Cosserat / sUbend / mesh-refinement diagnostics
│   ├── bootstrap/          Paired-bootstrap CIs (8 sUbend contrasts + 4 VMR)
│   ├── stratified/         Per-pathology VMR breakdown
│   └── figures/            All publication figures (PDF + PNG)
├── docs/
│   ├── REPRODUCIBILITY.md  Four-level reviewer protocol
│   ├── DATA_ACCESS.md      GitHub vs Zenodo vs source-data policy
│   ├── RESULTS_INDEX.md    Claim-to-file map for the released results
│   ├── STRATEGY_SR.md      Project narrative + decision log (D1–D8)
│   └── archive/            CP-version strategy snapshot (frozen 2026-05-19)
├── environment.yml         Conda env spec
├── pyproject.toml          Lightweight install metadata + test config
├── Makefile                Shortcuts for verify / figures / Slurm stages
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

python scripts/verify_release.py

# Regenerate the clinical-headline figure (proposed Fig 1):
make figures
```

### Level 2 — re-run the diagnostics from released predictions (≈ 30 min, no GPU)

Requires the Zenodo deposit (DOI: pending; ≈ 8 GB; `predictions/` and
`checkpoints/`). Extract it next to this repository and re-launch the
post-training diagnostics:

```bash
make level2-diagnostics
```

### Level 3 — re-evaluate from released checkpoints (≈ 1 hour, 1 GPU)

You also need the source NPZ datasets. The build recipe is in
`src/make_npz_*.py`; pointers to the public source data are in
`PUBLICATION_MANIFEST.md` section C. After regenerating
`data/npz_<variant>/`, run:

```bash
make level3-eval
```

### Level 4 — full re-training from scratch (≈ 200 GPU-hours)

```bash
make level4-train
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
