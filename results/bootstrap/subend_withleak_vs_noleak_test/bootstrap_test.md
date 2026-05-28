# Paired bootstrap CI: subend_withleak vs subend_noleak on test split

Resampling unit: case_id (n=75). Bootstrap iterations: 10000. Permutation iterations: 10000.

| metric | subend_withleak mean | subend_noleak mean | Δ (a−b) | 95% CI | p_boot | p_perm | better |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PP@10 | 0.0445 | 0.0045 | +0.0400 | [+0.0295, +0.0514] | 0.0000 | 0.0000 | **subend_withleak** |
| PP@5 | 0.0124 | 0.0008 | +0.0117 | [+0.0085, +0.0152] | 0.0000 | 0.0000 | **subend_withleak** |
| angle° | 7.0530 | 60.8844 | -53.8315 | [-57.6289, -50.1115] | 0.0000 | 0.0000 | **subend_withleak** |
| ewRMSE | 0.3381 | 0.4593 | -0.1212 | [-0.1657, -0.0747] | 0.0000 | 0.0000 | **subend_withleak** |
| dP_MAE_mmHg | 8.3390 | 4.7855 | +3.5535 | [+1.4089, +5.5659] | 0.0012 | 0.0019 | **subend_noleak** |
| peak_loc_mm | 34.5172 | 70.7295 | -36.2123 | [-39.4110, -33.0003] | 0.0000 | 0.0000 | **subend_withleak** |
| peak_mag_rel | 2.2880 | 0.6790 | +1.6090 | [+1.1760, +2.0593] | 0.0000 | 0.0000 | **subend_noleak** |
| WSS_MAE_Pa | 0.2328 | 0.7537 | -0.5209 | [-0.5955, -0.4410] | 0.0000 | 0.0000 | **subend_withleak** |
| WSS_bias_Pa | 0.0121 | 0.6780 | -0.6659 | [-0.7016, -0.6284] | 0.0000 | 0.0000 | **subend_withleak** |

*Caveat:* with n=5 paired cases, bootstrap CIs are wide. Treat p-values as descriptive, not as definitive significance tests. Headline conclusions should rest on point estimates + per-case transparency rather than on these p-values alone.
