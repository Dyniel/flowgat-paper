# Paired bootstrap CI: withleak vs leak_mag_only on test split

Resampling unit: case_id (n=5). Bootstrap iterations: 10000. Permutation iterations: 10000.

| metric | withleak mean | leak_mag_only mean | Δ (a−b) | 95% CI | p_boot | p_perm | better |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PP@10 | 0.1952 | 0.0002 | +0.1950 | [+0.0829, +0.2980] | 0.0000 | 0.0615 | **withleak** |
| PP@5 | 0.0482 | 0.0000 | +0.0482 | [+0.0171, +0.0788] | 0.0000 | 0.0628 | **withleak** |
| angle° | 3.8584 | 57.5154 | -53.6569 | [-76.2136, -39.9256] | 0.0000 | 0.0570 | **withleak** |
| ewRMSE | 0.1732 | 0.4965 | -0.3233 | [-0.3756, -0.2620] | 0.0000 | 0.0667 | **withleak** |
| peak_loc_mm | 47.2335 | 65.3092 | -18.0757 | [-46.3431, +2.8362] | 0.1354 | 0.3701 | **withleak** |
| peak_mag_rel | 0.5483 | 0.6895 | -0.1412 | [-0.2165, -0.0659] | 0.0000 | 0.0652 | **withleak** |
| WSS_MAE_Pa | 14.3241 | 9.0672 | +5.2569 | [+2.2157, +8.9615] | 0.0000 | 0.0645 | **leak_mag_only** |

*Caveat:* with n=5 paired cases, bootstrap CIs are wide. Treat p-values as descriptive, not as definitive significance tests. Headline conclusions should rest on point estimates + per-case transparency rather than on these p-values alone.
