# Cosserat Sweep Diagnostic

Direction-identifiability metrics reuse `src/womersley_metrics.py`.

## Aggregate

| variant | split | n_cases | n_seeds | pp_dir_10deg | cos_signed_median | frac_flipped | angle_median_deg |
|---|---|---|---|---|---|---|---|
| cosserat_sweep_withleak | val | 24 | 3 | 1.000 +/- 0.000 | 0.999 +/- 0.000 | 0.000 | 1.75 +/- 0.64 |
| cosserat_sweep_withleak | test | 36 | 3 | 0.992 +/- 0.049 | 0.999 +/- 0.001 | 0.000 | 1.74 +/- 0.94 |
| cosserat_sweep_leak_dir_only | val | 24 | 3 | 1.000 | 1.000 +/- 0.000 | 0.000 | 1.47 +/- 0.42 |
| cosserat_sweep_leak_dir_only | test | 36 | 3 | 0.985 +/- 0.086 | 0.999 +/- 0.003 | 0.000 | 1.72 +/- 1.58 |
| cosserat_sweep_leak_mag_only | val | 24 | 3 | 0.083 +/- 0.178 | 0.009 +/- 0.758 | 0.502 +/- 0.430 | 88.33 +/- 54.81 |
| cosserat_sweep_leak_mag_only | test | 36 | 3 | 0.031 +/- 0.077 | 0.184 +/- 0.636 | 0.383 +/- 0.366 | 77.22 +/- 43.70 |
| cosserat_sweep_noleak | val | 24 | 3 | 0.073 +/- 0.174 | -0.022 +/- 0.743 | 0.525 +/- 0.429 | 90.11 +/- 53.70 |
| cosserat_sweep_noleak | test | 36 | 3 | 0.022 +/- 0.056 | 0.197 +/- 0.621 | 0.370 +/- 0.355 | 76.71 +/- 42.45 |

## Dean-Bin Flip Check

For no-direction-channel variants, Lemma 1 predicts that the sign flip rate should not systematically fall just because Dean number grows.

| variant | split | low | mid | high | monotonic_decrease |
|---|---|---|---|---|---|
| cosserat_sweep_leak_mag_only | val | 0.336 | 0.627 | 0.563 | false |
| cosserat_sweep_leak_mag_only | test | 0.179 | 0.535 | 0.423 | false |
| cosserat_sweep_noleak | val | 0.337 | 0.685 | 0.568 | false |
| cosserat_sweep_noleak | test | 0.138 | 0.537 | 0.422 | false |
