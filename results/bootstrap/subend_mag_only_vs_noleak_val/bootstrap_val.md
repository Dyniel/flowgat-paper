# Paired bootstrap CI: subend_leak_mag_only vs subend_noleak on val split

Resampling unit: case_id (n=75). Bootstrap iterations: 10000. Permutation iterations: 10000.

| metric | subend_leak_mag_only mean | subend_noleak mean | Δ (a−b) | 95% CI | p_boot | p_perm | better |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PP@10 | 0.0053 | 0.0020 | +0.0034 | [+0.0005, +0.0063] | 0.0198 | 0.0250 | **subend_leak_mag_only** |
| PP@5 | 0.0007 | 0.0004 | +0.0003 | [-0.0003, +0.0009] | 0.3180 | 0.3448 | **subend_leak_mag_only** |
| angle° | 63.2796 | 66.2481 | -2.9685 | [-4.1885, -1.7439] | 0.0000 | 0.0000 | **subend_leak_mag_only** |
| ewRMSE | 0.3149 | 0.3492 | -0.0344 | [-0.0522, -0.0160] | 0.0002 | 0.0006 | **subend_leak_mag_only** |
| dP_MAE_mmHg | 3.3000 | 2.1975 | +1.1025 | [+0.0870, +2.0867] | 0.0334 | 0.0338 | **subend_noleak** |
| peak_loc_mm | 29.1259 | 75.2969 | -46.1710 | [-52.1780, -40.5307] | 0.0000 | 0.0000 | **subend_leak_mag_only** |
| peak_mag_rel | 2.5002 | 1.0467 | +1.4535 | [+1.0032, +1.9350] | 0.0000 | 0.0000 | **subend_noleak** |
| WSS_MAE_Pa | 0.3250 | 0.7067 | -0.3817 | [-0.4426, -0.3162] | 0.0000 | 0.0000 | **subend_leak_mag_only** |
| WSS_bias_Pa | 0.1035 | 0.6503 | -0.5469 | [-0.5653, -0.5284] | 0.0000 | 0.0000 | **subend_leak_mag_only** |

*Caveat:* with n=5 paired cases, bootstrap CIs are wide. Treat p-values as descriptive, not as definitive significance tests. Headline conclusions should rest on point estimates + per-case transparency rather than on these p-values alone.
