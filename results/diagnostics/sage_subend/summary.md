# Direction-Identifiability Diagnostic

Direction-identifiability metrics reuse `src/womersley_metrics.py`.

## Aggregate

| variant | split | n_cases | n_seeds | pp_dir_10deg | cos_signed_median | frac_flipped | angle_median_deg |
|---|---|---|---|---|---|---|---|
| sage_subend_withleak | val | 75 | 3 | 0.828 +/- 0.133 | 0.996 +/- 0.003 | 0.000 +/- 0.000 | 4.81 +/- 2.17 |
| sage_subend_withleak | test | 75 | 3 | 0.874 +/- 0.115 | 0.997 +/- 0.002 | 0.000 | 3.93 +/- 1.69 |
| sage_subend_leak_dir_only | val | 75 | 3 | 0.843 +/- 0.121 | 0.996 +/- 0.003 | 0.001 +/- 0.006 | 4.73 +/- 2.00 |
| sage_subend_leak_dir_only | test | 75 | 3 | 0.908 +/- 0.092 | 0.998 +/- 0.002 | 0.000 +/- 0.000 | 3.51 +/- 1.54 |
| sage_subend_leak_mag_only | val | 75 | 3 | 0.125 +/- 0.123 | 0.513 +/- 0.375 | 0.298 +/- 0.152 | 54.37 +/- 28.19 |
| sage_subend_leak_mag_only | test | 75 | 3 | 0.171 +/- 0.097 | 0.708 +/- 0.292 | 0.235 +/- 0.149 | 41.03 +/- 22.01 |
| sage_subend_noleak | val | 75 | 3 | 0.108 +/- 0.101 | 0.460 +/- 0.391 | 0.327 +/- 0.160 | 58.85 +/- 27.64 |
| sage_subend_noleak | test | 75 | 3 | 0.150 +/- 0.085 | 0.679 +/- 0.288 | 0.225 +/- 0.151 | 44.04 +/- 20.94 |

## Dean-Bin Flip Check

_Skipped: dataset has no Dean number metadata (applicable to parametric sweeps only)._
