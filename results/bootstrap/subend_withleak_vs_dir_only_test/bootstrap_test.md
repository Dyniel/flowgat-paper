# Paired bootstrap CI: subend_withleak vs subend_leak_dir_only on test split

Resampling unit: case_id (n=75). Bootstrap iterations: 10000. Permutation iterations: 10000.

| metric | subend_withleak mean | subend_leak_dir_only mean | Δ (a−b) | 95% CI | p_boot | p_perm | better |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PP@10 | 0.0445 | 0.0315 | +0.0130 | [+0.0048, +0.0214] | 0.0018 | 0.0017 | **subend_withleak** |
| PP@5 | 0.0124 | 0.0076 | +0.0049 | [+0.0022, +0.0078] | 0.0002 | 0.0008 | **subend_withleak** |
| angle° | 7.0530 | 7.0500 | +0.0029 | [-0.1420, +0.1335] | 0.9366 | 0.9690 | **subend_leak_dir_only** |
| ewRMSE | 0.3381 | 0.3671 | -0.0289 | [-0.0472, -0.0099] | 0.0040 | 0.0046 | **subend_withleak** |
| dP_MAE_mmHg | 8.3390 | 9.2475 | -0.9085 | [-1.9749, +0.1174] | 0.0872 | 0.0934 | **subend_withleak** |
| peak_loc_mm | 34.5172 | 48.9992 | -14.4820 | [-18.1983, -10.9501] | 0.0000 | 0.0000 | **subend_withleak** |
| peak_mag_rel | 2.2880 | 3.0029 | -0.7148 | [-1.1950, -0.2762] | 0.0008 | 0.0023 | **subend_withleak** |
| WSS_MAE_Pa | 0.2328 | 0.3811 | -0.1483 | [-0.1893, -0.1069] | 0.0000 | 0.0000 | **subend_withleak** |
| WSS_bias_Pa | 0.0121 | 0.2558 | -0.2437 | [-0.2652, -0.2225] | 0.0000 | 0.0000 | **subend_withleak** |

*Caveat:* with n=5 paired cases, bootstrap CIs are wide. Treat p-values as descriptive, not as definitive significance tests. Headline conclusions should rest on point estimates + per-case transparency rather than on these p-values alone.
