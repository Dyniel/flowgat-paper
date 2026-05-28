# Womersley phase-aware diagnostic

**Hypothesis.** Variants without per-node direction leakage learn a fixed axial polarity from the training distribution. When tested at phases where the true Womersley flow has reversed (cos(ω·t_phase) < 0), they keep predicting the original polarity → their *signed* cosine flips negative even though the *folded* angle stays close to 0°. Direction-leakage variants carry the instantaneous sign in the feature, so they should be invariant to phase.

## Per-variant flip rates (split by expected Q-sign)

`forward` = cases with cos(ω·t_phase) > 0 (drive in +z); `reverse` = cases with cos(ω·t_phase) < 0 (drive in −z). If the model has learned a fixed axial polarity, `case_flip_rate_reverse` will be ≫ `case_flip_rate_forward`.

| variant | n | fwd flip rate | rev flip rate | cos_signed (fwd) | cos_signed (rev) | angle° (fwd) | angle° (rev) |
|---|---:|---:|---:|---:|---:|---:|---:|
| leak_dir_only | 30 | 0.00 | 0.00 | 0.984 | 0.952 | 10.3 | 17.6 |
| leak_mag_only | 30 | 0.33 | 0.25 | 0.251 | 0.287 | 75.4 | 73.3 |
| noleak | 30 | 0.39 | 0.33 | 0.212 | 0.438 | 77.8 | 64.0 |
| withleak | 30 | 0.00 | 0.00 | 0.972 | 0.943 | 13.7 | 19.4 |

## Phase-binned medians (4 bins over [0, 2π))

Bins: 0=[0, π/2), 1=[π/2, π), 2=[π, 3π/2), 3=[3π/2, 2π). Reverse-flow half of the cycle is approximately bins 1+2.

| variant | bin | n | angle° | cos_signed | case_flip_rate |
|---|---:|---:|---:|---:|---:|
| leak_dir_only | 0 | 6 | 13.58 ± 7.83 | 0.964 ± 0.033 | 0.00 |
| leak_dir_only | 1 | 3 | 13.73 ± 5.56 | 0.968 ± 0.025 | 0.00 |
| leak_dir_only | 2 | 9 | 21.14 ± 9.08 | 0.922 ± 0.066 | 0.00 |
| leak_dir_only | 3 | 12 | 9.36 ± 3.65 | 0.985 ± 0.011 | 0.00 |
| leak_mag_only | 0 | 6 | 79.26 ± 50.44 | 0.139 ± 0.729 | 0.33 |
| leak_mag_only | 1 | 3 | 60.93 ± 5.80 | 0.484 ± 0.089 | 0.00 |
| leak_mag_only | 2 | 9 | 74.99 ± 38.58 | 0.195 ± 0.554 | 0.33 |
| leak_mag_only | 3 | 12 | 89.74 ± 51.44 | 0.066 ± 0.703 | 0.33 |
| noleak | 0 | 6 | 75.42 ± 52.38 | 0.178 ± 0.774 | 0.50 |
| noleak | 1 | 3 | 38.77 ± 18.34 | 0.753 ± 0.188 | 0.00 |
| noleak | 2 | 9 | 72.51 ± 39.92 | 0.223 ± 0.551 | 0.44 |
| noleak | 3 | 12 | 87.47 ± 52.55 | 0.081 ± 0.696 | 0.33 |
| withleak | 0 | 6 | 15.01 ± 9.31 | 0.955 ± 0.046 | 0.00 |
| withleak | 1 | 3 | 16.54 ± 7.79 | 0.953 ± 0.042 | 0.00 |
| withleak | 2 | 9 | 22.40 ± 9.26 | 0.914 ± 0.068 | 0.00 |
| withleak | 3 | 12 | 12.04 ± 6.40 | 0.972 ± 0.024 | 0.00 |
