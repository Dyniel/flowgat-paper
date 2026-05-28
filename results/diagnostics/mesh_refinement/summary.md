# Mesh-refinement diagnostic

## Node counts

| resolution | mean nodes/case | mean median-NN L |
|---|---:|---:|
| 1x | 2.3e+04 | 0.001222 |
| 2x | 1.8e+05 | 6.111067e-04 |
| 4x | 1.5e+06 | 3.055524e-04 |

## Estimator convergence

| resolution | div_true_mean | div_analytical_mean |
|---|---:|---:|
| 1x | 9.864460e-05 | 0.000 |
| 2x | 6.967181e-06 | 0.000 |
| 4x | 1.302102e-06 | 0.000 |

`div_true` ratios: 1x/2x=14.16, 2x/4x=5.35; median-NN ratios: 1x/2x=2.00, 2x/4x=2.00.

## Aggregate table

| variant | resolution | n_cases | n_seeds | div_pred_mean | div_true_mean | div_analytical_mean | helm_pred_mean | helm_true_mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| withleak | 1x | 6 | 3 | 0.087290 | 9.864460e-05 | 0.000 | 0.028123 | 4.511621e-04 |
| withleak | 2x | 6 | 3 | 0.025562 | 6.967181e-06 | 0.000 | nan | nan |
| withleak | 4x | 6 | 3 | 0.006010 | 1.302102e-06 | 0.000 | nan | 1.563434e-05 |
| leak_dir_only | 1x | 6 | 3 | 0.073579 | 9.864460e-05 | 0.000 | 0.026481 | 4.511621e-04 |
| leak_dir_only | 2x | 6 | 3 | 0.019849 | 6.967181e-06 | 0.000 | nan | nan |
| leak_dir_only | 4x | 6 | 3 | 0.004637 | 1.302102e-06 | 0.000 | nan | 1.563434e-05 |
| leak_mag_only | 1x | 6 | 3 | 0.232077 | 9.864460e-05 | 0.000 | 0.053872 | 4.511621e-04 |
| leak_mag_only | 2x | 6 | 3 | 0.063819 | 6.967181e-06 | 0.000 | nan | nan |
| leak_mag_only | 4x | 6 | 3 | 0.012193 | 1.302102e-06 | 0.000 | nan | 1.563434e-05 |
| noleak | 1x | 6 | 3 | 0.072625 | 9.864460e-05 | 0.000 | 0.060808 | 4.511621e-04 |
| noleak | 2x | 6 | 3 | 0.020938 | 6.967181e-06 | 0.000 | nan | nan |
| noleak | 4x | 6 | 3 | 0.004432 | 1.302102e-06 | 0.000 | nan | 1.563434e-05 |

## Criteria

| variant | div plateau 4x vs 1x | status |
|---|---:|---|
| withleak | 0.9312 | STOP |
| leak_dir_only | 0.9370 | STOP |
| leak_mag_only | 0.9475 | STOP |
| noleak | 0.9390 | STOP |

| variant | helm_pred/helm_true at 4x | status |
|---|---:|---|
| withleak | nan | INCOMPLETE |
| leak_dir_only | nan | INCOMPLETE |

STOP: at least one hard-pass criterion failed; Section 3.8 needs re-framing.
