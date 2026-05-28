# Paired bootstrap CI: subend_withleak vs subend_leak_dir_only on val split

Resampling unit: case_id (n=75). Bootstrap iterations: 10000. Permutation iterations: 10000.

| metric | subend_withleak mean | subend_leak_dir_only mean | Δ (a−b) | 95% CI | p_boot | p_perm | better |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PP@10 | 0.0310 | 0.0259 | +0.0052 | [+0.0001, +0.0101] | 0.0468 | 0.0483 | **subend_withleak** |
| PP@5 | 0.0092 | 0.0068 | +0.0023 | [+0.0007, +0.0039] | 0.0048 | 0.0049 | **subend_withleak** |
| angle° | 7.4688 | 8.3894 | -0.9206 | [-1.2476, -0.6721] | 0.0000 | 0.0000 | **subend_withleak** |
| ewRMSE | 0.2375 | 0.2455 | -0.0080 | [-0.0201, +0.0038] | 0.1932 | 0.2004 | **subend_withleak** |
| dP_MAE_mmHg | 4.5835 | 6.6097 | -2.0262 | [-2.7289, -1.3233] | 0.0000 | 0.0000 | **subend_withleak** |
| peak_loc_mm | 20.4209 | 57.3389 | -36.9181 | [-43.2019, -30.4720] | 0.0000 | 0.0000 | **subend_withleak** |
| peak_mag_rel | 2.7383 | 4.0214 | -1.2832 | [-1.7608, -0.8287] | 0.0000 | 0.0000 | **subend_withleak** |
| WSS_MAE_Pa | 0.2525 | 0.3706 | -0.1181 | [-0.1474, -0.0881] | 0.0000 | 0.0000 | **subend_withleak** |
| WSS_bias_Pa | 0.1100 | 0.2929 | -0.1829 | [-0.2024, -0.1657] | 0.0000 | 0.0000 | **subend_withleak** |

*Caveat:* with n=5 paired cases, bootstrap CIs are wide. Treat p-values as descriptive, not as definitive significance tests. Headline conclusions should rest on point estimates + per-case transparency rather than on these p-values alone.
