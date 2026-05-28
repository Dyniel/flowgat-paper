# Paired bootstrap CI: withleak vs noleak on test split

Resampling unit: case_id (n=5). Bootstrap iterations: 10000. Permutation iterations: 10000.

| metric | withleak mean | noleak mean | Δ (a−b) | 95% CI | p_boot | p_perm | better |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PP@10 | 0.1952 | 0.0091 | +0.1861 | [+0.0815, +0.2815] | 0.0000 | 0.0615 | **withleak** |
| PP@5 | 0.0482 | 0.0013 | +0.0468 | [+0.0168, +0.0761] | 0.0000 | 0.0628 | **withleak** |
| angle° | 3.8584 | 45.1365 | -41.2781 | [-52.4208, -30.1354] | 0.0000 | 0.0570 | **withleak** |
| ewRMSE | 0.1732 | 0.4670 | -0.2938 | [-0.3906, -0.2208] | 0.0000 | 0.0667 | **withleak** |
| peak_loc_mm | 47.2335 | 58.1923 | -10.9587 | [-53.0876, +31.0412] | 0.5972 | 0.6719 | **withleak** |
| peak_mag_rel | 0.5483 | 0.8262 | -0.2779 | [-0.8402, +0.1582] | 0.3096 | 0.4972 | **withleak** |
| WSS_MAE_Pa | 14.3241 | 13.8633 | +0.4607 | [-6.8277, +7.8879] | 0.8078 | 0.7552 | **noleak** |

*Caveat:* with n=5 paired cases, bootstrap CIs are wide. Treat p-values as descriptive, not as definitive significance tests. Headline conclusions should rest on point estimates + per-case transparency rather than on these p-values alone.
