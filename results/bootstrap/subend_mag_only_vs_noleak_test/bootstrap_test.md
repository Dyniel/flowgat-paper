# Paired bootstrap CI: subend_leak_mag_only vs subend_noleak on test split

Resampling unit: case_id (n=75). Bootstrap iterations: 10000. Permutation iterations: 10000.

| metric | subend_leak_mag_only mean | subend_noleak mean | Δ (a−b) | 95% CI | p_boot | p_perm | better |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PP@10 | 0.0040 | 0.0045 | -0.0004 | [-0.0027, +0.0019] | 0.7078 | 0.7145 | **subend_noleak** |
| PP@5 | 0.0005 | 0.0008 | -0.0003 | [-0.0007, +0.0001] | 0.1604 | 0.1936 | **subend_noleak** |
| angle° | 51.7850 | 60.8844 | -9.0994 | [-11.5700, -6.4533] | 0.0000 | 0.0000 | **subend_leak_mag_only** |
| ewRMSE | 0.4251 | 0.4593 | -0.0342 | [-0.0653, -0.0016] | 0.0386 | 0.0404 | **subend_leak_mag_only** |
| dP_MAE_mmHg | 7.0706 | 4.7855 | +2.2850 | [+0.6245, +3.8613] | 0.0086 | 0.0083 | **subend_noleak** |
| peak_loc_mm | 27.8506 | 70.7295 | -42.8789 | [-48.8334, -37.3941] | 0.0000 | 0.0000 | **subend_leak_mag_only** |
| peak_mag_rel | 2.0456 | 0.6790 | +1.3666 | [+0.9820, +1.7777] | 0.0000 | 0.0000 | **subend_noleak** |
| WSS_MAE_Pa | 0.3526 | 0.7537 | -0.4011 | [-0.4646, -0.3319] | 0.0000 | 0.0000 | **subend_leak_mag_only** |
| WSS_bias_Pa | 0.1034 | 0.6780 | -0.5746 | [-0.5836, -0.5650] | 0.0000 | 0.0000 | **subend_leak_mag_only** |

*Caveat:* with n=5 paired cases, bootstrap CIs are wide. Treat p-values as descriptive, not as definitive significance tests. Headline conclusions should rest on point estimates + per-case transparency rather than on these p-values alone.
