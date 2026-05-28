# Paired bootstrap CI: subend_withleak vs subend_noleak on val split

Resampling unit: case_id (n=75). Bootstrap iterations: 10000. Permutation iterations: 10000.

| metric | subend_withleak mean | subend_noleak mean | Δ (a−b) | 95% CI | p_boot | p_perm | better |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PP@10 | 0.0310 | 0.0020 | +0.0291 | [+0.0198, +0.0391] | 0.0000 | 0.0000 | **subend_withleak** |
| PP@5 | 0.0092 | 0.0004 | +0.0088 | [+0.0058, +0.0119] | 0.0000 | 0.0000 | **subend_withleak** |
| angle° | 7.4688 | 66.2481 | -58.7793 | [-62.7011, -54.6938] | 0.0000 | 0.0000 | **subend_withleak** |
| ewRMSE | 0.2375 | 0.3492 | -0.1117 | [-0.1460, -0.0777] | 0.0000 | 0.0000 | **subend_withleak** |
| dP_MAE_mmHg | 4.5835 | 2.1975 | +2.3860 | [+1.4899, +3.2615] | 0.0000 | 0.0000 | **subend_noleak** |
| peak_loc_mm | 20.4209 | 75.2969 | -54.8761 | [-60.9580, -48.7421] | 0.0000 | 0.0000 | **subend_withleak** |
| peak_mag_rel | 2.7383 | 1.0467 | +1.6916 | [+1.2569, +2.1431] | 0.0000 | 0.0000 | **subend_noleak** |
| WSS_MAE_Pa | 0.2525 | 0.7067 | -0.4542 | [-0.5153, -0.3907] | 0.0000 | 0.0000 | **subend_withleak** |
| WSS_bias_Pa | 0.1100 | 0.6503 | -0.5403 | [-0.5770, -0.5030] | 0.0000 | 0.0000 | **subend_withleak** |

*Caveat:* with n=5 paired cases, bootstrap CIs are wide. Treat p-values as descriptive, not as definitive significance tests. Headline conclusions should rest on point estimates + per-case transparency rather than on these p-values alone.
