# Paired bootstrap CI: subend_leak_dir_only vs subend_noleak on val split

Resampling unit: case_id (n=75). Bootstrap iterations: 10000. Permutation iterations: 10000.

| metric | subend_leak_dir_only mean | subend_noleak mean | Δ (a−b) | 95% CI | p_boot | p_perm | better |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PP@10 | 0.0259 | 0.0020 | +0.0239 | [+0.0151, +0.0341] | 0.0000 | 0.0000 | **subend_leak_dir_only** |
| PP@5 | 0.0068 | 0.0004 | +0.0064 | [+0.0037, +0.0096] | 0.0000 | 0.0000 | **subend_leak_dir_only** |
| angle° | 8.3894 | 66.2481 | -57.8587 | [-61.7229, -53.8464] | 0.0000 | 0.0000 | **subend_leak_dir_only** |
| ewRMSE | 0.2455 | 0.3492 | -0.1037 | [-0.1332, -0.0740] | 0.0000 | 0.0000 | **subend_leak_dir_only** |
| dP_MAE_mmHg | 6.6097 | 2.1975 | +4.4122 | [+3.3049, +5.4092] | 0.0000 | 0.0000 | **subend_noleak** |
| peak_loc_mm | 57.3389 | 75.2969 | -17.9580 | [-22.7406, -13.0519] | 0.0000 | 0.0000 | **subend_leak_dir_only** |
| peak_mag_rel | 4.0214 | 1.0467 | +2.9747 | [+2.3683, +3.6008] | 0.0000 | 0.0000 | **subend_noleak** |
| WSS_MAE_Pa | 0.3706 | 0.7067 | -0.3361 | [-0.3805, -0.2923] | 0.0000 | 0.0000 | **subend_leak_dir_only** |
| WSS_bias_Pa | 0.2929 | 0.6503 | -0.3574 | [-0.3967, -0.3176] | 0.0000 | 0.0000 | **subend_leak_dir_only** |

*Caveat:* with n=5 paired cases, bootstrap CIs are wide. Treat p-values as descriptive, not as definitive significance tests. Headline conclusions should rest on point estimates + per-case transparency rather than on these p-values alone.
