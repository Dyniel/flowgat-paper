# Results index

This page maps the main paper claims to the exact released files used to audit
them. All values below are from:

```text
results/figures/fig_clinical_headline_values.csv
```

The headline table uses the three paper seeds: `1337`, `2026`, and `777`.

## Clinical-headline values

| Metric, test split | VMR aortas | Womersley pipe | Cosserat sweep | U-bend CFD |
|---|---:|---:|---:|---:|
| Angle error, withleak / noleak | 3.9 / 45.7 deg | 17.4 / 69.0 deg | 1.9 / 76.4 deg | 7.1 / 60.9 deg |
| WSS MAE, withleak / noleak | 14.3 / 13.9 Pa | 0.06 / 0.04 Pa | 0.50 / 0.80 Pa | 0.23 / 0.75 Pa |
| Peak-location MAE, withleak / noleak | 47.2 / 58.2 mm | not used as headline | not used as headline | 34.5 / 70.7 mm |
| dP MAE, withleak / noleak | 22.69 / 22.59 mmHg | 0.20 / 0.19 mmHg | 2.52 / 3.14 mmHg | 8.34 / 4.79 mmHg |

The U-bend dP inversion is the main example showing why simplified Bernoulli
pressure drop should not be reported alone: the noleak variant has lower dP
MAE but fails direction and peak-localisation checks.

## Claim-to-file map

| Claim / audit | Primary released files |
|---|---|
| Four-domain clinical headline | `results/figures/fig_clinical_headline_values.csv`, `results/figures/fig_clinical_headline.pdf` |
| VMR pathology-stratified audit | `results/stratified/stratified_test.md`, `results/stratified/stratified_test.csv` |
| Womersley direction/magnitude separation | `results/diagnostics/womersley/summary.md`, `results/diagnostics/womersley/aggregate.csv` |
| Cosserat curvature sweep | `results/diagnostics/cosserat_sweep/summary.md`, `results/diagnostics/cosserat_sweep/aggregate.csv` |
| U-bend direction-identifiability | `results/diagnostics/subend/summary.md`, `results/diagnostics/subend/aggregate.csv` |
| U-bend pressure-drop collapse | `results/diagnostics/subend/dp_investigation.md`, `results/diagnostics/subend/dp_investigation_summary.csv`, `results/figures/subend_dp_scatter.pdf` |
| Bootstrap confidence intervals | `results/bootstrap/*/bootstrap_*.md`, `results/bootstrap/*/bootstrap_*.csv` |
| Mesh-refinement sensitivity | `results/diagnostics/mesh_refinement/summary.md`, `results/diagnostics/mesh_refinement/aggregate.csv` |
| Architecture follow-ups | `results/diagnostics/sage_cosserat/summary.md`, `results/diagnostics/sage_subend/summary.md`, `results/diagnostics/flowgat_nobc_cosserat/summary.md` |

## Recreating this index

Re-render the headline values and verify release integrity:

```bash
make figures
make verify
```

The figure generator writes both the figure and the CSV used above:

```text
results/figures/fig_clinical_headline.pdf
results/figures/fig_clinical_headline.png
results/figures/fig_clinical_headline_values.csv
```
