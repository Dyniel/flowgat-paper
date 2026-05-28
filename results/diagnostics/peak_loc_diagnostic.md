# Peak-localization degeneracy diagnostic
Hypothesis: noleak model predicts a near-constant normalized peak
position regardless of geometry. Test: dispersion of normalized
predicted peaks across cases.

## Dispersion summary (per variant × seed × split)
| variant | seed | split | n | bbox-std | pca-std | pred/true bbox-std | pred/true pca-std | peak_loc median (mm) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| noleak | 1337 | test | 5 | 0.204 | 0.220 | 1.44 | 1.23 | 36.7 |
| noleak | 1337 | val | 4 | 0.143 | 0.443 | 0.62 | 2.20 | 122.6 |
| noleak | 2026 | test | 5 | 0.199 | 0.303 | 1.41 | 1.69 | 38.2 |
| noleak | 2026 | val | 4 | 0.134 | 0.124 | 0.58 | 0.61 | 68.6 |
| noleak | 777 | test | 5 | 0.177 | 0.229 | 1.25 | 1.28 | 37.2 |
| noleak | 777 | val | 4 | 0.159 | 0.126 | 0.69 | 0.63 | 77.5 |
| withleak | 1337 | test | 5 | 0.132 | 0.330 | 0.93 | 1.85 | 56.4 |
| withleak | 1337 | val | 4 | 0.129 | 0.228 | 0.56 | 1.13 | 30.0 |
| withleak | 2026 | test | 5 | 0.136 | 0.292 | 0.96 | 1.63 | 36.4 |
| withleak | 2026 | val | 4 | 0.229 | 0.243 | 0.99 | 1.20 | 68.8 |
| withleak | 777 | test | 5 | 0.145 | 0.307 | 1.02 | 1.71 | 56.8 |
| withleak | 777 | val | 4 | 0.152 | 0.240 | 0.66 | 1.19 | 40.1 |

## Degeneracy verdict
- Rule of thumb: `pred/true bbox-std ratio` < 0.30 ⇒ model is
  predicting a near-constant peak (degenerate).
- Ratio between 0.30 and 0.70 ⇒ model partly adapts to geometry.
- Ratio > 0.70 ⇒ geometry-aware peak prediction.

- noleak seed=1337 (test): bbox-ratio=1.44 ⇒ **GEOMETRY-AWARE**
- noleak seed=2026 (test): bbox-ratio=1.41 ⇒ **GEOMETRY-AWARE**
- noleak seed=777 (test): bbox-ratio=1.25 ⇒ **GEOMETRY-AWARE**
- withleak seed=1337 (test): bbox-ratio=0.93 ⇒ **GEOMETRY-AWARE**
- withleak seed=2026 (test): bbox-ratio=0.96 ⇒ **GEOMETRY-AWARE**
- withleak seed=777 (test): bbox-ratio=1.02 ⇒ **GEOMETRY-AWARE**
