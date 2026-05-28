# Paired bootstrap CI: withleak vs leak_dir_only on test split

Resampling unit: case_id (n=5). Bootstrap iterations: 10000. Permutation iterations: 10000.

| metric | withleak mean | leak_dir_only mean | Δ (a−b) | 95% CI | p_boot | p_perm | better |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PP@10 | 0.1952 | 0.2043 | -0.0090 | [-0.0355, +0.0175] | 0.4624 | 0.6246 | **leak_dir_only** |
| PP@5 | 0.0482 | 0.0612 | -0.0131 | [-0.0234, -0.0040] | 0.0010 | 0.1254 | **leak_dir_only** |
| angle° | 3.8584 | 3.1592 | +0.6992 | [+0.6277, +0.8053] | 0.0000 | 0.0570 | **leak_dir_only** |
| ewRMSE | 0.1732 | 0.1705 | +0.0027 | [-0.0111, +0.0198] | 0.7496 | 0.8157 | **leak_dir_only** |
| peak_loc_mm | 47.2335 | 54.3761 | -7.1425 | [-28.6810, +12.3405] | 0.4982 | 0.6856 | **withleak** |
| peak_mag_rel | 0.5483 | 0.5332 | +0.0151 | [+0.0044, +0.0265] | 0.0000 | 0.0652 | **leak_dir_only** |
| WSS_MAE_Pa | 14.3241 | 12.3001 | +2.0239 | [+0.8246, +3.3508] | 0.0000 | 0.0645 | **leak_dir_only** |

*Caveat:* with n=5 paired cases, bootstrap CIs are wide. Treat p-values as descriptive, not as definitive significance tests. Headline conclusions should rest on point estimates + per-case transparency rather than on these p-values alone.
