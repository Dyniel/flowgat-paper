# Womersley metric repair — Phase E5

WSS R² is **excluded** here by design — on a uniform-radius cylinder the variance of true WSS is ~0, so R² = 1 - MSE/Var collapses to ≈ -10⁶ and is mathematically ill-defined. We report it in Limitations, not as a finding.

PP@10 with per-node relative error is also excluded — it is dominated by magnitude error (peak_mag_rel ≈ 3-5 for noleak-family variants on Womersley), which buries direction quality.

## Direction-only success rate (PP_dir@θ)

Fraction of HE-mask (top-20%-speed) nodes with ∠(u_pred, u_true) ≤ θ.

| variant | split | n_cases | n_seeds | pp_dir_5deg | pp_dir_10deg | pp_dir_15deg | pp_dir_30deg |
|---|---|---|---|---|---|---|---|
| womersley_leak_dir_only | test | 6 | 3 | 0.070 ± 0.148 | 0.309 ± 0.275 | 0.582 ± 0.351 | 0.948 ± 0.092 |
| womersley_leak_dir_only | val | 4 | 3 | 0.177 ± 0.229 | 0.478 ± 0.401 | 0.629 ± 0.376 | 0.862 ± 0.285 |
| womersley_leak_mag_only | test | 6 | 3 | 0.001 ± 0.003 | 0.008 ± 0.030 | 0.033 ± 0.086 | 0.186 ± 0.355 |
| womersley_leak_mag_only | val | 4 | 3 | 0.001 ± 0.002 | 0.001 ± 0.003 | 0.002 ± 0.005 | 0.006 ± 0.011 |
| womersley_noleak | test | 6 | 3 | 0.006 ± 0.021 | 0.041 ± 0.108 | 0.113 ± 0.240 | 0.236 ± 0.385 |
| womersley_noleak | val | 4 | 3 | 0.007 ± 0.018 | 0.018 ± 0.043 | 0.040 ± 0.077 | 0.148 ± 0.308 |
| womersley_withleak | test | 6 | 3 | 0.055 ± 0.129 | 0.231 ± 0.277 | 0.454 ± 0.330 | 0.923 ± 0.148 |
| womersley_withleak | val | 4 | 3 | 0.205 ± 0.267 | 0.391 ± 0.424 | 0.528 ± 0.413 | 0.842 ± 0.299 |

## Peak-normalised vector error (PP_peak@δ)

Fraction of HE nodes with ||u_pred − u_true|| / max(||u_true||) ≤ δ. Uses *case-peak* as the normaliser, so it does not blow up when the model magnitude is off by a fixed multiplicative factor.

| variant | split | pp_peak_0.10 | pp_peak_0.20 | pp_peak_0.50 |
|---|---|---|---|---|
| womersley_leak_dir_only | test | 0.003 ± 0.014 | 0.031 ± 0.086 | 0.155 ± 0.322 |
| womersley_leak_dir_only | val | 0.089 ± 0.146 | 0.240 ± 0.354 | 0.492 ± 0.510 |
| womersley_leak_mag_only | test | 0.000 ± 0.000 | 0.000 ± 0.001 | 0.030 ± 0.081 |
| womersley_leak_mag_only | val | 0.000 ± 0.001 | 0.001 ± 0.003 | 0.004 ± 0.009 |
| womersley_noleak | test | 0.000 ± 0.001 | 0.005 ± 0.013 | 0.044 ± 0.106 |
| womersley_noleak | val | 0.004 ± 0.010 | 0.016 ± 0.038 | 0.068 ± 0.166 |
| womersley_withleak | test | 0.000 ± 0.000 | 0.005 ± 0.019 | 0.178 ± 0.343 |
| womersley_withleak | val | 0.120 ± 0.206 | 0.273 ± 0.407 | 0.498 ± 0.504 |

## Signed-cosine + magnitude diagnostics

`cos_signed_median` < 0 ⇒ majority of HE nodes have a flipped direction (typical when the model has learned a fixed axial polarity in training but the test phase is in the reverse half of the cycle). `frac_flipped` = share of HE nodes with cos < 0.

| variant | split | cos_signed_median | frac_flipped | angle_median_deg | mag_ratio_median | peak_mag_rel |
|---|---|---|---|---|---|---|
| womersley_leak_dir_only | test | 0.963 ± 0.031 | 0.000 | 14.35 ± 6.29 | 2.153 ± 1.390 | 3.077 ± 2.962 |
| womersley_leak_dir_only | val | 0.956 ± 0.065 | 0.000 ± 0.001 | 13.92 ± 10.52 | 1.437 ± 0.545 | 1.461 ± 1.127 |
| womersley_leak_mag_only | test | 0.267 ± 0.546 | 0.242 ± 0.422 | 71.64 ± 37.57 | 5.603 ± 8.663 | 8.047 ± 11.463 |
| womersley_leak_mag_only | val | 0.002 ± 0.703 | 0.450 ± 0.495 | 93.38 ± 51.23 | 3.371 ± 4.066 | 4.740 ± 6.382 |
| womersley_noleak | test | 0.306 ± 0.603 | 0.331 ± 0.445 | 67.33 ± 41.93 | 1.284 ± 0.894 | 2.531 ± 2.329 |
| womersley_noleak | val | 0.067 ± 0.696 | 0.400 ± 0.433 | 88.26 ± 52.51 | 0.992 ± 0.532 | 1.289 ± 1.091 |
| womersley_withleak | test | 0.952 ± 0.038 | 0.000 | 16.44 ± 7.06 | 2.043 ± 1.350 | 3.373 ± 3.127 |
| womersley_withleak | val | 0.945 ± 0.067 | 0.000 ± 0.000 | 15.82 ± 11.29 | 1.372 ± 0.517 | 1.863 ± 1.470 |
