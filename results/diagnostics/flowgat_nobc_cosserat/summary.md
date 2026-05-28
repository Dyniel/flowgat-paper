# Direction-Identifiability Diagnostic

Direction-identifiability metrics reuse `src/womersley_metrics.py`.

## Aggregate

| variant | split | n_cases | n_seeds | pp_dir_10deg | cos_signed_median | frac_flipped | angle_median_deg |
|---|---|---|---|---|---|---|---|
| cosserat_sweep_withleak_nobc | val | 24 | 3 | 1.000 +/- 0.001 | 0.999 +/- 0.000 | 0.000 | 1.76 +/- 0.64 |
| cosserat_sweep_withleak_nobc | test | 36 | 3 | 0.992 +/- 0.049 | 0.999 +/- 0.001 | 0.000 | 1.75 +/- 0.96 |
| cosserat_sweep_leak_dir_only_nobc | val | 24 | 3 | 1.000 | 1.000 +/- 0.000 | 0.000 | 1.48 +/- 0.43 |
| cosserat_sweep_leak_dir_only_nobc | test | 36 | 3 | 0.987 +/- 0.080 | 0.999 +/- 0.002 | 0.000 | 1.70 +/- 1.39 |
| cosserat_sweep_leak_mag_only_nobc | val | 24 | 3 | 0.075 +/- 0.167 | 0.006 +/- 0.762 | 0.503 +/- 0.432 | 88.54 +/- 55.19 |
| cosserat_sweep_leak_mag_only_nobc | test | 36 | 3 | 0.037 +/- 0.092 | 0.184 +/- 0.644 | 0.387 +/- 0.369 | 77.07 +/- 44.58 |
| cosserat_sweep_noleak_nobc | val | 24 | 3 | 0.073 +/- 0.161 | -0.025 +/- 0.749 | 0.528 +/- 0.426 | 90.33 +/- 54.37 |
| cosserat_sweep_noleak_nobc | test | 36 | 3 | 0.025 +/- 0.072 | 0.201 +/- 0.628 | 0.373 +/- 0.359 | 76.30 +/- 43.19 |

## Dean-Bin Flip Check

For no-direction-channel variants, Lemma 1 predicts that the sign flip rate should not systematically fall just because Dean number grows.

| variant | split | low | mid | high | monotonic_decrease |
|---|---|---|---|---|---|
