# Direction-Identifiability Diagnostic

Direction-identifiability metrics reuse `src/womersley_metrics.py`.

## Aggregate

| variant | split | n_cases | n_seeds | pp_dir_10deg | cos_signed_median | frac_flipped | angle_median_deg |
|---|---|---|---|---|---|---|---|
| subend_withleak | val | 75 | 3 | 0.730 +/- 0.167 | 0.993 +/- 0.006 | 0.000 +/- 0.001 | 6.20 +/- 2.76 |
| subend_withleak | test | 75 | 3 | 0.755 +/- 0.181 | 0.994 +/- 0.006 | 0.000 +/- 0.000 | 5.73 +/- 2.71 |
| subend_leak_dir_only | val | 75 | 3 | 0.702 +/- 0.176 | 0.992 +/- 0.006 | 0.001 +/- 0.006 | 6.83 +/- 2.76 |
| subend_leak_dir_only | test | 75 | 3 | 0.771 +/- 0.182 | 0.994 +/- 0.006 | 0.000 +/- 0.001 | 5.65 +/- 2.79 |
| subend_leak_mag_only | val | 75 | 3 | 0.108 +/- 0.110 | 0.523 +/- 0.335 | 0.282 +/- 0.146 | 54.78 +/- 24.80 |
| subend_leak_mag_only | test | 75 | 3 | 0.161 +/- 0.093 | 0.736 +/- 0.268 | 0.202 +/- 0.143 | 39.04 +/- 20.26 |
| subend_noleak | val | 75 | 3 | 0.119 +/- 0.107 | 0.497 +/- 0.383 | 0.319 +/- 0.149 | 55.76 +/- 28.16 |
| subend_noleak | test | 75 | 3 | 0.178 +/- 0.098 | 0.675 +/- 0.312 | 0.282 +/- 0.147 | 43.67 +/- 23.10 |

## Dean-Bin Flip Check

_Skipped: dataset has no Dean number metadata (applicable to parametric sweeps only)._
