# Paired bootstrap CI: subend_leak_dir_only vs subend_noleak on test split

Resampling unit: case_id (n=75). Bootstrap iterations: 10000. Permutation iterations: 10000.

| metric | subend_leak_dir_only mean | subend_noleak mean | Δ (a−b) | 95% CI | p_boot | p_perm | better |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PP@10 | 0.0315 | 0.0045 | +0.0270 | [+0.0182, +0.0365] | 0.0000 | 0.0000 | **subend_leak_dir_only** |
| PP@5 | 0.0076 | 0.0008 | +0.0068 | [+0.0043, +0.0096] | 0.0000 | 0.0000 | **subend_leak_dir_only** |
| angle° | 7.0500 | 60.8844 | -53.8344 | [-57.5790, -50.1363] | 0.0000 | 0.0000 | **subend_leak_dir_only** |
| ewRMSE | 0.3671 | 0.4593 | -0.0923 | [-0.1285, -0.0554] | 0.0000 | 0.0000 | **subend_leak_dir_only** |
| dP_MAE_mmHg | 9.2475 | 4.7855 | +4.4620 | [+2.7325, +6.1000] | 0.0000 | 0.0000 | **subend_noleak** |
| peak_loc_mm | 48.9992 | 70.7295 | -21.7303 | [-25.3298, -18.1357] | 0.0000 | 0.0000 | **subend_leak_dir_only** |
| peak_mag_rel | 3.0029 | 0.6790 | +2.3239 | [+1.7611, +2.8983] | 0.0000 | 0.0000 | **subend_noleak** |
| WSS_MAE_Pa | 0.3811 | 0.7537 | -0.3726 | [-0.4189, -0.3255] | 0.0000 | 0.0000 | **subend_leak_dir_only** |
| WSS_bias_Pa | 0.2558 | 0.6780 | -0.4222 | [-0.4567, -0.3877] | 0.0000 | 0.0000 | **subend_leak_dir_only** |

*Caveat:* with n=5 paired cases, bootstrap CIs are wide. Treat p-values as descriptive, not as definitive significance tests. Headline conclusions should rest on point estimates + per-case transparency rather than on these p-values alone.
