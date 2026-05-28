# Womersley metric repair — Phase E5

WSS R² is **excluded** here by design — on a uniform-radius cylinder the variance of true WSS is ~0, so R² = 1 - MSE/Var collapses to ≈ -10⁶ and is mathematically ill-defined. We report it in Limitations, not as a finding.

PP@10 with per-node relative error is also excluded — it is dominated by magnitude error (peak_mag_rel ≈ 3-5 for noleak-family variants on Womersley), which buries direction quality.

## Direction-only success rate (PP_dir@θ)

Fraction of HE-mask (top-20%-speed) nodes with ∠(u_pred, u_true) ≤ θ.

| variant | split | n_cases | n_seeds | pp_dir_5deg | pp_dir_10deg | pp_dir_15deg | pp_dir_30deg |
|---|---|---|---|---|---|---|---|
| leak_dir_only | test | 5 | 3 | 0.843 ± 0.100 | 0.988 ± 0.017 | 0.999 ± 0.001 | 1.000 ± 0.000 |
| leak_dir_only | val | 4 | 3 | 0.865 ± 0.089 | 0.988 ± 0.031 | 0.999 ± 0.002 | 1.000 ± 0.000 |
| leak_mag_only | test | 5 | 3 | 0.009 ± 0.012 | 0.051 ± 0.056 | 0.130 ± 0.123 | 0.385 ± 0.184 |
| leak_mag_only | val | 4 | 3 | 0.036 ± 0.068 | 0.099 ± 0.158 | 0.174 ± 0.193 | 0.436 ± 0.233 |
| noleak | test | 5 | 3 | 0.046 ± 0.047 | 0.136 ± 0.078 | 0.239 ± 0.086 | 0.515 ± 0.151 |
| noleak | val | 4 | 3 | 0.091 ± 0.129 | 0.216 ± 0.239 | 0.309 ± 0.250 | 0.598 ± 0.181 |
| noleak_centerline | test | 5 | 3 | 0.019 ± 0.011 | 0.087 ± 0.042 | 0.193 ± 0.086 | 0.478 ± 0.159 |
| noleak_centerline | val | 4 | 3 | 0.069 ± 0.083 | 0.188 ± 0.182 | 0.292 ± 0.216 | 0.525 ± 0.207 |
| sage_noleak | test | 5 | 3 | 0.038 ± 0.034 | 0.122 ± 0.088 | 0.225 ± 0.115 | 0.468 ± 0.152 |
| sage_noleak | val | 4 | 3 | 0.091 ± 0.115 | 0.219 ± 0.221 | 0.314 ± 0.222 | 0.579 ± 0.155 |
| sage_withleak | test | 5 | 3 | 0.782 ± 0.174 | 0.985 ± 0.019 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| sage_withleak | val | 4 | 3 | 0.861 ± 0.114 | 0.986 ± 0.017 | 0.999 ± 0.003 | 1.000 ± 0.000 |
| withleak | test | 5 | 3 | 0.768 ± 0.110 | 0.955 ± 0.038 | 0.995 ± 0.006 | 1.000 ± 0.000 |
| withleak | val | 4 | 3 | 0.819 ± 0.103 | 0.960 ± 0.056 | 0.995 ± 0.012 | 1.000 ± 0.000 |
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
| leak_dir_only | test | 0.681 ± 0.254 | 0.876 ± 0.128 | 0.997 ± 0.005 |
| leak_dir_only | val | 0.426 ± 0.276 | 0.677 ± 0.364 | 0.941 ± 0.107 |
| leak_mag_only | test | 0.124 ± 0.237 | 0.370 ± 0.437 | 0.715 ± 0.278 |
| leak_mag_only | val | 0.026 ± 0.040 | 0.124 ± 0.161 | 0.638 ± 0.183 |
| noleak | test | 0.186 ± 0.300 | 0.430 ± 0.352 | 0.760 ± 0.229 |
| noleak | val | 0.081 ± 0.116 | 0.230 ± 0.186 | 0.737 ± 0.143 |
| noleak_centerline | test | 0.157 ± 0.243 | 0.412 ± 0.370 | 0.739 ± 0.229 |
| noleak_centerline | val | 0.049 ± 0.064 | 0.183 ± 0.143 | 0.618 ± 0.186 |
| sage_noleak | test | 0.152 ± 0.269 | 0.454 ± 0.370 | 0.723 ± 0.264 |
| sage_noleak | val | 0.084 ± 0.110 | 0.225 ± 0.193 | 0.657 ± 0.169 |
| sage_withleak | test | 0.624 ± 0.292 | 0.866 ± 0.143 | 0.997 ± 0.005 |
| sage_withleak | val | 0.448 ± 0.258 | 0.681 ± 0.329 | 0.965 ± 0.066 |
| withleak | test | 0.654 ± 0.263 | 0.889 ± 0.118 | 0.998 ± 0.003 |
| withleak | val | 0.402 ± 0.258 | 0.670 ± 0.354 | 0.914 ± 0.168 |
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
| leak_dir_only | test | 0.999 ± 0.001 | 0.000 ± 0.000 | 2.72 ± 0.65 | 0.850 ± 0.103 | -0.533 ± 0.263 |
| leak_dir_only | val | 0.999 ± 0.000 | 0.000 | 2.35 ± 0.43 | 1.174 ± 0.324 | -0.185 ± 0.255 |
| leak_mag_only | test | 0.568 ± 0.537 | 0.240 ± 0.209 | 50.07 ± 35.89 | 0.515 ± 0.136 | -0.690 ± 0.195 |
| leak_mag_only | val | 0.795 ± 0.190 | 0.142 ± 0.122 | 34.15 ± 17.29 | 0.868 ± 0.388 | -0.248 ± 0.221 |
| noleak | test | 0.798 ± 0.177 | 0.159 ± 0.123 | 34.76 ± 14.85 | 0.533 ± 0.226 | 0.168 ± 0.950 |
| noleak | val | 0.897 ± 0.081 | 0.048 ± 0.022 | 23.83 ± 11.86 | 1.077 ± 0.217 | 0.288 ± 0.471 |
| noleak_centerline | test | 0.799 ± 0.131 | 0.136 ± 0.076 | 35.43 ± 11.98 | 0.497 ± 0.171 | 0.512 ± 1.061 |
| noleak_centerline | val | 0.825 ± 0.170 | 0.132 ± 0.132 | 30.62 ± 17.47 | 0.885 ± 0.274 | 2.391 ± 1.254 |
| sage_noleak | test | 0.700 ± 0.366 | 0.188 ± 0.174 | 41.50 ± 25.54 | 0.504 ± 0.180 | -0.009 ± 1.014 |
| sage_noleak | val | 0.905 ± 0.076 | 0.084 ± 0.075 | 23.06 ± 11.04 | 1.126 ± 0.383 | 0.215 ± 0.498 |
| sage_withleak | test | 0.998 ± 0.001 | 0.000 | 3.10 ± 1.10 | 0.866 ± 0.126 | -0.478 ± 0.338 |
| sage_withleak | val | 0.999 ± 0.001 | 0.000 | 2.56 ± 0.85 | 1.137 ± 0.294 | -0.198 ± 0.272 |
| withleak | test | 0.998 ± 0.001 | 0.000 ± 0.000 | 3.20 ± 0.88 | 0.874 ± 0.138 | -0.548 ± 0.261 |
| withleak | val | 0.999 ± 0.001 | 0.000 | 2.63 ± 0.84 | 1.202 ± 0.340 | -0.171 ± 0.272 |
| womersley_leak_dir_only | test | 0.963 ± 0.031 | 0.000 | 14.35 ± 6.29 | 2.153 ± 1.390 | 3.077 ± 2.962 |
| womersley_leak_dir_only | val | 0.956 ± 0.065 | 0.000 ± 0.001 | 13.92 ± 10.52 | 1.437 ± 0.545 | 1.461 ± 1.127 |
| womersley_leak_mag_only | test | 0.267 ± 0.546 | 0.242 ± 0.422 | 71.64 ± 37.57 | 5.603 ± 8.663 | 8.047 ± 11.463 |
| womersley_leak_mag_only | val | 0.002 ± 0.703 | 0.450 ± 0.495 | 93.38 ± 51.23 | 3.371 ± 4.066 | 4.740 ± 6.382 |
| womersley_noleak | test | 0.306 ± 0.603 | 0.331 ± 0.445 | 67.33 ± 41.93 | 1.284 ± 0.894 | 2.531 ± 2.329 |
| womersley_noleak | val | 0.067 ± 0.696 | 0.400 ± 0.433 | 88.26 ± 52.51 | 0.992 ± 0.532 | 1.289 ± 1.091 |
| womersley_withleak | test | 0.952 ± 0.038 | 0.000 | 16.44 ± 7.06 | 2.043 ± 1.350 | 3.373 ± 3.127 |
| womersley_withleak | val | 0.945 ± 0.067 | 0.000 ± 0.000 | 15.82 ± 11.29 | 1.372 ± 0.517 | 1.863 ± 1.470 |
