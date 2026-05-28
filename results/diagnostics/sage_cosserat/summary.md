# Direction-Identifiability Diagnostic

Direction-identifiability metrics reuse `src/womersley_metrics.py`.

## Aggregate

| variant | split | n_cases | n_seeds | pp_dir_10deg | cos_signed_median | frac_flipped | angle_median_deg |
|---|---|---|---|---|---|---|---|
| sage_cosserat_withleak | val | 24 | 3 | 1.000 | 1.000 +/- 0.000 | 0.000 | 1.49 +/- 0.48 |
| sage_cosserat_withleak | test | 36 | 3 | 0.998 +/- 0.013 | 1.000 +/- 0.000 | 0.000 | 1.60 +/- 0.67 |
| sage_cosserat_leak_dir_only | val | 24 | 3 | 1.000 | 1.000 +/- 0.000 | 0.000 | 1.26 +/- 0.41 |
| sage_cosserat_leak_dir_only | test | 36 | 3 | 0.999 +/- 0.009 | 1.000 +/- 0.000 | 0.000 | 1.38 +/- 0.63 |
| sage_cosserat_leak_mag_only | val | 24 | 3 | 0.088 +/- 0.205 | -0.030 +/- 0.724 | 0.528 +/- 0.414 | 90.50 +/- 52.63 |
| sage_cosserat_leak_mag_only | test | 36 | 3 | 0.033 +/- 0.066 | 0.209 +/- 0.614 | 0.383 +/- 0.363 | 75.20 +/- 41.89 |
| sage_cosserat_noleak | val | 24 | 3 | 0.067 +/- 0.168 | -0.066 +/- 0.720 | 0.550 +/- 0.412 | 92.99 +/- 52.03 |
| sage_cosserat_noleak | test | 36 | 3 | 0.029 +/- 0.061 | 0.249 +/- 0.597 | 0.363 +/- 0.353 | 72.98 +/- 40.61 |

## Dean-Bin Flip Check

For no-direction-channel variants, Lemma 1 predicts that the sign flip rate should not systematically fall just because Dean number grows.

| variant | split | low | mid | high | monotonic_decrease |
|---|---|---|---|---|---|
| sage_cosserat_leak_mag_only | val | 0.329 | 0.703 | 0.563 | false |
| sage_cosserat_leak_mag_only | test | 0.172 | 0.542 | 0.424 | false |
| sage_cosserat_noleak | val | 0.378 | 0.704 | 0.575 | false |
| sage_cosserat_noleak | test | 0.135 | 0.526 | 0.413 | false |
