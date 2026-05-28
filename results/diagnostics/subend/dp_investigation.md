# sUbend ΔP investigation

Tests whether the noleak variant's lower dP_MAE on sUbend is driven by magnitude collapse (predicted speeds shrink towards zero) rather than by accurate peak-velocity recovery.

Bernoulli convention (used by `clinical.peak_pressure_drop_error`):
  `dP[mmHg] = 4 * (max ||u||)^2`

## Per-variant per-split summary

| variant | split | n | max|u_pred| | max|u_true| | ratio (mean) | ratio (median, p10–p90) | dP_pred [mmHg] | dP_gt [mmHg] | |dP_pred−dP_gt| | signed err | corr(max_pred,max_true) | corr(dP_pred,dP_gt) |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| subend_withleak | val | 225 | 1.194 | 0.553 | 3.736 | 3.017 (1.307–7.609) | 6.82 | 2.28 | 4.58 | +4.53 | +0.792 | +0.788 |
| subend_withleak | test | 225 | 1.553 | 0.817 | 3.246 | 3.049 (1.004–6.442) | 10.97 | 5.15 | 8.34 | +5.82 | +0.555 | +0.489 |
| subend_leak_dir_only | val | 225 | 1.442 | 0.553 | 5.017 | 4.317 (1.291–9.212) | 8.81 | 2.28 | 6.61 | +6.53 | +0.748 | +0.687 |
| subend_leak_dir_only | test | 225 | 1.586 | 0.817 | 3.920 | 3.148 (0.848–8.278) | 10.35 | 5.15 | 9.25 | +5.21 | +0.266 | +0.171 |
| subend_leak_mag_only | val | 225 | 1.050 | 0.553 | 3.478 | 2.622 (1.000–7.159) | 5.20 | 2.28 | 3.30 | +2.91 | +0.639 | +0.594 |
| subend_leak_mag_only | test | 225 | 1.371 | 0.817 | 2.961 | 2.548 (0.814–5.839) | 8.39 | 5.15 | 7.07 | +3.24 | +0.486 | +0.390 |
| subend_noleak | val | 225 | 0.407 | 0.553 | 1.612 | 1.304 (0.284–3.234) | 0.67 | 2.28 | 2.20 | -1.61 | +0.007 | +0.006 |
| subend_noleak | test | 225 | 0.419 | 0.817 | 1.093 | 0.848 (0.200–2.368) | 0.72 | 5.15 | 4.79 | -4.43 | -0.002 | -0.002 |

## Interpretation key

- `ratio` ≈ 1.0 → peak speed correctly recovered.
- `ratio` ≪ 1.0 → magnitude collapse; predicted speeds shrunk.
- `ratio` ≫ 1.0 → magnitude overshoot.
- low `corr(dP_pred, dP_gt)` → predictions do not track per-case dP variation; dP_MAE may be a single-scale-match artefact.
- signed err negative → predictions systematically *under*-predict dP.
