# Paired bootstrap CI: leak_dir_only vs leak_mag_only on test split

Resampling unit: case_id (n=5). Bootstrap iterations: 10000. Permutation iterations: 10000.

| metric | leak_dir_only mean | leak_mag_only mean | Δ (a−b) | 95% CI | p_boot | p_perm | better |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PP@10 | 0.2043 | 0.0002 | +0.2040 | [+0.0983, +0.3216] | 0.0000 | 0.0615 | **leak_dir_only** |
| PP@5 | 0.0612 | 0.0000 | +0.0612 | [+0.0259, +0.0979] | 0.0000 | 0.0628 | **leak_dir_only** |
| angle° | 3.1592 | 57.5154 | -54.3562 | [-76.9000, -40.6570] | 0.0000 | 0.0570 | **leak_dir_only** |
| ewRMSE | 0.1705 | 0.4965 | -0.3260 | [-0.3857, -0.2633] | 0.0000 | 0.0667 | **leak_dir_only** |
| peak_loc_mm | 54.3761 | 65.3092 | -10.9332 | [-35.6802, +12.4343] | 0.3820 | 0.4267 | **leak_dir_only** |
| peak_mag_rel | 0.5332 | 0.6895 | -0.1563 | [-0.2304, -0.0756] | 0.0000 | 0.0652 | **leak_dir_only** |
| WSS_MAE_Pa | 12.3001 | 9.0672 | +3.2330 | [+1.3483, +5.5201] | 0.0000 | 0.0645 | **leak_mag_only** |

*Caveat:* with n=5 paired cases, bootstrap CIs are wide. Treat p-values as descriptive, not as definitive significance tests. Headline conclusions should rest on point estimates + per-case transparency rather than on these p-values alone.
