# Strategia "CP-1000%" — Phase E5 status

**Cel:** Communications Physics submission, headline *physics-of-learning* — **identifiability of velocity field in tubular vascular geometries**.

**Aktualny stan (2026-05-15, post-Phase E5 + paper writing main pass):** wszystkie GPU runs Phase E zakończone, unified metrics policzone, **main paper text empirycznie kompletny** (`paper/main.tex` 1197 lines, ~5050 words main body). Abstract, Introduction, Results (7 podsekcji w tym L2/L3/L4/L5), Discussion (Helmholtz unification, sign as boundary quantity), Limitations (WSS R² + phase falsification), Methods (Dataset/Synthetic Womersley/Features/FlowGAT/SAGE/Centerline-tangent/Loss/Eval/Direction-only metrics/Physics diagnostics/Reproducibility) — wszystko gotowe. Brakuje: theoretical anchor (1-page derivation), reproducibility package, cover letter, internal review.

---

## 🏁 Final ML results

| variant | group | n | PP@10 | angle° | ewRMSE | rola |
|---|---|---:|---|---|---|---|
| `leak_dir_only` | VMR/FlowGAT | 5 | 0.237 ± 0.005 | 3.36 ± 0.34 | 0.153 ± 0.012 | ceiling |
| `withleak` | VMR/FlowGAT | 5 | 0.214 ± 0.030 | 4.04 ± 0.51 | 0.164 ± 0.015 | full leakage |
| `leak_mag_only` | VMR/FlowGAT | 5 | 0.0002 | 63.13 ± 2.98 | 0.473 | "ile bez dokąd" |
| `noleak_centerline` | VMR/FlowGAT | 3 | 0.0025 ± 0.0019 | 45.96 ± 4.02 | 0.467 ± 0.011 | refined geom prior |
| `noleak` | VMR/FlowGAT | 5 | 0.0091 ± 0.0045 | 44.47 ± 2.22 | 0.427 ± 0.010 | global PCA |
| `sage_withleak` | VMR/SAGE | 3 | 0.227 ± 0.057 | 3.50 ± 0.42 | 0.165 ± 0.018 | arch-indep ceiling |
| `sage_noleak` | VMR/SAGE | 3 | 0.0025 ± 0.0020 | 51.89 ± 8.28 | 0.460 ± 0.008 | arch-indep noleak |

**Unified direction-identifiability metrics (test, n=3 seeds × 5-6 cases) — all 4 variants × 2 domains:**

| variant | domain | PP_dir@10° | cos_signed | frac_flipped | angle°(med) |
|---|---|---|---|---|---|
| `leak_dir_only` | VMR | **0.988 ± 0.017** | **+0.999 ± 0.001** | **0.000** | 2.7 ± 0.6 |
| `leak_dir_only` | Womersley | 0.309 ± 0.275 | **+0.963 ± 0.031** | **0.000** | 14.4 ± 6.3 |
| `withleak` | VMR | **0.955 ± 0.038** | **+0.998 ± 0.001** | **0.000** | 3.2 ± 0.9 |
| `withleak` | Womersley | 0.231 ± 0.277 | **+0.952 ± 0.038** | **0.000** | 16.4 ± 7.1 |
| `noleak` | VMR | 0.136 ± 0.078 | +0.798 ± 0.177 | **0.159 ± 0.123** | 34.8 ± 14.9 |
| `noleak` | Womersley | 0.041 ± 0.108 | +0.306 ± 0.603 | **0.331 ± 0.445** | 67.3 ± 41.9 |
| `leak_mag_only` | VMR | 0.051 ± 0.056 | +0.568 ± 0.537 | **0.240 ± 0.209** | 50.1 ± 35.9 |
| `leak_mag_only` | Womersley | 0.008 ± 0.030 | +0.267 ± 0.546 | **0.242 ± 0.422** | 71.6 ± 37.6 |

**Plus L2/L3 confirmation:** sage_withleak cos_signed +0.998 ≈ FlowGAT withleak +0.998 (L2); noleak_centerline cos_signed +0.799 ≈ noleak +0.798 (L3 fully confirmed na new metrics).

Stare PP@10 i WSS R² na Womersley **omitted by design** — per-node relative error blow up gdy magnitude jest off (peak_mag_rel ≈ 3-5), WSS R² mathematically ill-defined gdy Var(WSS_true) ≈ 0 na uniform-radius cylinder.

**Niespodziankę z unified (2026-05-15):** VMR `noleak` ALSO ma flipy (15.9% nodes), choć mniej niż Womersley (33%) — aortic geometry łagodzi ale nie eliminuje sign degeneracy. Cross-domain pattern symetryczny. Dodatkowo: `leak_mag_only` jest *aktywnie szkodliwy* na VMR (cos +0.568 < noleak +0.798), magnitude-only feature myli direction learning.

### Physics diagnostics (per-node, n=15-18 cases × seeds)

| variant | n | angle° | mag_ratio | corr(\|u\|, r) | div_pred | div_true |
|---|---:|---|---|---|---|---|
| withleak (VMR) | 15 | 3.40 | 1.43 | 0.36 | 0.0027 | 0.0021 |
| noleak (VMR) | 15 | 36.31 | 0.63 | 0.15 | 0.0035 | 0.0021 |
| noleak_centerline | 15 | 37.78 | 0.72 | 0.16 | 0.0026 | 0.0014 |
| sage_withleak | 15 | 3.58 | 1.32 | **0.47** | 0.0013 | 0.0014 |
| sage_noleak | 15 | 39.04 | 0.71 | 0.19 | 0.0014 | 0.0014 |
| womersley_leak_dir_only | 18 | 14.01 | 2.94 | 0.49 | 0.108 | 0.0001 |
| womersley_withleak | 18 | 15.46 | 3.09 | 0.59 | 0.108 | 0.0001 |
| womersley_leak_mag_only | 18 | 71.21 | 2.17 | **0.68** | 0.109 | 0.0001 |
| womersley_noleak | 18 | 68.40 | 1.63 | 0.54 | 0.086 | 0.0001 |

**Pięć fizycznych findings:**

1. **Mass conservation + Poiseuille radial scaling są physics-essential channels** — model traci je bez direction leakage (corr 0.15 vs 0.36).
2. **Architecture-independence potwierdzona również na poziomie fizyki** — sage_withleak corr 0.47 vs FlowGAT 0.36 (SAGE *lepszy* na Poiseuille!); sage_noleak 0.19 ≈ FlowGAT noleak 0.15.
3. **noleak_centerline ≈ noleak w każdym wymiarze** — angle 37.78 vs 36.31, mag_ratio 0.72 vs 0.63, corr 0.16 vs 0.15. Confirms L3 na physics level.
4. **Womersley `leak_mag_only` ma najwyższy corr(\|u\|,r) = 0.68** — magnitude-leakage feature *jest* funkcją radiusa, mechanistic confirmation. `div_pred` ~100× > `div_true` we wszystkich Womersley variantach → modele **nie uczą się mass conservation na synthetic regime**.
5. **Direction degeneracy is per-case random, NOT phase-locked** (Phase E5 step 3, *new*) — no-direction-leakage variants flip sign w ~33-39% przypadków, niezależnie od position w cyklu Womersleya. Direction-leakage variants nigdy nie flippują (frac_flipped = 0). Sfalsyfikowana hipoteza "fixed axial polarity from training distribution", potwierdzony silniejszy fakt: bez direction leakage, sign recovery jest po prostu chaotyczny.

---

## 🎯 Pięć linii dowodu

| # | Linia | Status | Punktacja |
|---|---|---|---|
| **L1** | **Asymmetric leakage** — dir_only ≥ withleak ≫ mag_only ≈ noleak | ✅ **mocna** (n=5, low variance) | 8/10 |
| **L2** | **Architecture-independence** — SAGE replikuje VMR pattern (PP@10, angle, corr(\|u\|,r) 0.47 vs FlowGAT 0.36) | ✅ **mocna** (n=3 + physics diag) | 8.5/10 |
| **L3** | **Better geometric prior NIE pomaga** — noleak_centerline ≈ noleak w każdym wymiarze | ✅ **mocna i ciekawa** (counter-intuitive, n=3 + physics diag) | 8/10 |
| **L4** | **Cross-domain replication symmetric** — pattern utrzymuje się 1:1 na VMR i Womersley (frac_flipped 0 dla dir-leak vs >0 dla noleak na **obu** domenach); aortic geometry łagodzi sign degeneracy ale jej nie eliminuje | ✅ **mocna** (n=3 × 4 variants × 2 domains, full unified table) | **8/10** |
| **L5** | **Continuity learned as pattern, not as law** — VMR withleak ma div_pred ≈ div_true; Womersley wszystkie variants mają div_pred ~100× > div_true | ✅ **mocna** (kontrast across domains) | 7/10 |

L3, L4, L5 są strategicznymi aktywami:
- **L3**: "richer geometric ground truth as feature does not translate to better model output" — physics-of-learning finding mocniejszy niż prosta drabina priorów
- **L4 (nowa narracja)**: na geometrically-trivial cylinder z perfect axial PCA prior, no-direction-leakage variants nie odzyskują kierunku (cos_signed ≈ +0.3 z heavy bimodal distribution wokół ±1, 33% case-level flip rate). To replikuje VMR pattern (44° noleak vs 4° withleak), eliminując 'real-aorta complexity' jako wyjaśnienie i confirming że direction identifiability jest właściwością **per-node leakage**, nie geometrycznej dokładności
- **L5**: pozorne respektowanie continuity equation na VMR nie wynika z fizyki, tylko z wzorca w danych — when transferred to controlled synthetic (Womersley), model fails to conserve mass

---

## 🔧 Phase E5 status

| Krok | Effort | Status | Score impact |
|---|---|---|---|
| **1. Naprawa metryk** — PP_dir@θ + PP_peak@δ + cos_signed; WSS R² disabled na cylindrze | 1h CPU | ✅ **DONE** | +1.5 |
| **2. Re-framing narracji + writing** — H1 fail jest mocniejszym finding; Womersley section, Discussion update, Limitations, Methods (Womersley dataset) dodane do `paper/main.tex` | 2h writing | ✅ **DONE** | +1.0 |
| **3. Phase-aware diagnostic** — angle/cos vs phase_norm; **hipoteza H1 sfalsyfikowana** (fwd ≈ rev flip rate 0.39 vs 0.33), ale nowy fakt mocniejszy | 4h CPU | ✅ **DONE** | +1.0 |
| **5. Closing symmetry gap** *(new 2026-05-15)* — dump VMR `withleak`, `leak_dir_only`, `leak_mag_only` predictions, unified metrics na 4 variants × 2 domains | 0.5h GPU + CPU | ✅ **DONE** (job 18) | +0.5 (cross-domain confirm) |

**Wszystkie kroki Phase E5 zakończone, L4 = 8/10.**

### Nowa narracja Womersley (do papera, krok 2)

**Stare (sfalsyfikowane):** "Na cylindrze noleak flipuje sign w reverse half cycle bo learned fixed +z polarity."

**Nowe (mocniejsze):** *"Na geometrically-trivial straight tube z perfect axial PCA prior, no-direction-leakage variants nie odzyskują kierunku — cos_signed median +0.30 z bimodalną dystrybucją (przypadki klastrują przy +1 i -1), 33% case-level flip rate niezależnie od fazy. To NIE jest phase-locked artifact (fwd flip 0.39 ≈ rev flip 0.33). Direction-leakage variants są phase-invariant na perfect cos_signed +0.96, 0% flip rate. Replikuje VMR pattern na controlled regime, eliminując 'real-aorta complexity' jako wyjaśnienie. L4 + L3 łączą się: direction identifiability jest właściwością per-node leakage, nie geometrycznej dokładności."*

---

## 📍 CP positioning — claims dla abstract

| Claim | Co konkretnie | Status |
|---|---|---|
| **C1 — Methodology** | 4-variant feature-decomposition jako standardowy audit CFD-surrogate'ów | ✅ |
| **C2 — Physics finding** | Asymmetric identifiability: direction = geometry-redundant (L1), but practical geometric priors don't recover it (L3); magnitude = dynamics-essential (L1); sign recovery without direction leakage is case-level chaotic (L4) | ✅ |
| **C3 — Architecture independence** | Pattern przenosi się przez 2 niezależne architektury (FlowGAT, SAGE) na 2 domenach (VMR aorta, Womersley tube) | ✅ |
| **C4 — Honest baseline norm** | Per-stratum + per-case + bootstrap reporting na n=5; physics diagnostics quantitative; metryki świadomie omitted z uzasadnieniem (WSS R² na cylindrze) | ✅ |

### Cztery powody, dla których CP weźmie

1. **Methodological generalization** — 4-variant audit jako template który każdy autor CFD-surrogate może zastosować jutro.
2. **Reproducibility-crisis hook** — *Kapoor & Narayanan Patterns 2023* (294 papers, 8 leakage types).
3. **Physics interpretability** — "geometria nosi kierunek z degree-level precyzją, ale per-node leakage robi resztę; bez leakage sign degeneruje per-case" = *physics of learning*, fit pod CP "across all areas of physics".
4. **Safe-defence vs SOTA** — re-frame'ujemy headline numbers jako *diagnostic contribution*, trudno odrzucić.

### Decyzje narracyjne — LOCKED 2026-05-14

- **D1** Single full-stack paper (~6-8 stron CP letter + supplementary)
- **D2** "Physics of learning" framing — identifiability decomposition jako *physical statement*
- **D3** Peak_loc do supplementary (n=5, FSI stratification disclosure)
- **D4** WSS R² do Limitations — "open problem in this regime, separate from velocity-field finding"
- **D5** *(NEW 2026-05-15)* Phase hypothesis falsification → case-level direction degeneracy framing zamiast phase-locked

### Co NIE robimy

- ❌ AneuriskWeb / drugi dataset — future work
- ❌ Wskazywanie palcem konkretnych prac jako "zleakowane" — community-service framing
- ❌ Headline'owanie WSS

---

## 🗓️ Phase E5+ — manuscript path

| Krok | Effort | Status |
|---|---|---|
| Womersley metric fixes (kroki 1+3) | 1.5h CPU | ✅ DONE |
| VMR symmetry gap closing (krok 5) | 30 min | ✅ DONE |
| Womersley section + Discussion/Limitations/Methods update (Womersley dataset, Discussion Helmholtz unification) | 2h | ✅ DONE |
| Additional Results subsections (L2 SAGE, L3 noleak_centerline, L5 continuity) | 0.5 dnia | ✅ DONE |
| Additional Methods (SAGE backbone, centerline-tangent algorithm, PP_dir/PP_peak/cos_signed definitions, divergence estimator) | 0.5 dnia | ✅ DONE |
| Abstract refinement (5 findings, cross-domain, sign degeneracy) | 2h | ✅ DONE |
| Theoretical anchor (Cosserat rod + continuity) | 2 dni | **NEXT** |
| Reproducibility package (Zenodo DOI, README) | 1 dzień | TODO |
| Cover letter | 2h | TODO |
| Internal review + revision | 1 dzień | TODO |

**Realny czas do submission:** ~3-4 dni focused. Main text empirycznie kompletny (1197 lines, ~5050 words main body), brakuje: theoretical anchor (Methods derivation), Zenodo packaging, cover letter, peer review.

---

## 📦 Artefakty

**Phase E source code:**
- [src/centerline_tangent.py](src/centerline_tangent.py) — iteracyjny medial centerline + Frenet
- [src/physics_diagnostics.py](src/physics_diagnostics.py) — angle, divergence, Poiseuille scaling
- [src/make_npz_womersley.py](src/make_npz_womersley.py) — analytic pulsatile flow + 4 leak variants ([_variants](src/make_npz_womersley_variants.py))
- [src/flowgnn_aorta/models/flow_sage.py](src/flowgnn_aorta/models/flow_sage.py) — vanilla GraphSAGE second-arch
- `model.arch` dispatch w [src/train.py](src/train.py), [src/evaluate.py](src/evaluate.py), [src/dump_predictions.py](src/dump_predictions.py)

**Phase E5 source code (new, 2026-05-15):**
- [src/womersley_metrics.py](src/womersley_metrics.py) — PP_dir@θ, PP_peak@δ, cos_signed; works on VMR i Womersley (graceful fallback bez meta); WSS R² świadomie omitted
- [src/womersley_phase_analysis.py](src/womersley_phase_analysis.py) — phase-aware diag, 4-bin medianas, flip rate split by Q_sign, 2 figury

**Phase E5+ paper writing (new, 2026-05-15):**
- [paper/main.tex](paper/main.tex) — dodana sekcja `Cross-domain replication on a controlled Womersley benchmark` (Results), Methods/`Synthetic Womersley benchmark`, Discussion update (Helmholtz "geometry sets direction up to a sign"), Limitations (WSS R² disabled + phase hypothesis falsification)
- [paper/refs.bib](paper/refs.bib) — dodany `womersley1955`

**Data:** `data/npz_noleak_centerline/`, `data/npz_womersley{,_withleak,_leak_dir_only,_leak_mag_only}/`

**Configs:** `configs/{noleak_centerline, sage_*, womersley_*}.yaml`

**Jobs:**
- GPU: `jobs/{11,12,13}_train_*.sh` (orchestrator: `jobs/PHASE_E_SUBMIT.sh`). Recovery: `jobs/{15,16}_recover_*.sh`
- CPU: `jobs/17_womersley_fixes.sh` (Phase E5 step 1+3, ~1.5h, gpu:0)
- GPU+CPU: `jobs/18_dump_vmr_missing_and_unify.sh` (Phase E5 step 5 — closes symmetry gap, ~30 min)

**Results:**
- `results/checkpoints/{variant}/seed_*/best.pt`
- `results/predictions/{variant}/seed_<seed>/{val,test}_<case>.npz`
- `results/per_seed/{variant}_{val,test}_seed<seed>{,_aggregate.json}.csv`
- `results/diagnostics/physics/` + `physics_summary.csv`
- `results/diagnostics/womersley/{per_case,aggregate,phase_binned,phase_flip_summary}.csv` + `{summary,phase_summary}.md` *(Phase E5 step 1+3)*
- `results/diagnostics/all_variants/{per_case,aggregate}.csv` + `summary.md` *(Phase E5+ unified — 11 variants × 2 splits)*
- `results/predictions/{withleak,leak_dir_only,leak_mag_only}/seed_*/{val,test}_*.npz` *(new — closing symmetry gap)*
- `results/figures/womersley_phase_{angle,cos}.{png,pdf}` *(`cos` jest tą do papera — referencowana w main.tex jako `\Cref{fig:phase_cos}`)*

---

## 🧮 Theoretical anchor (na końcu — po empirical lock)

**Cel:** 1 strona derivation w Methods, pokazująca że asymmetric identifiability musi tak wyjść z continuity + tubular geometry.

**Plan szkicowy:**
1. Aproksymacja rury jako Cosserat rod (centerline arc-length `s`, lokalny radius `R(s)`, axial direction `T(s)`)
2. Z `∇·u = 0` → axial velocity `u_z(s, r, t) = (Q(t)/πR(s)²) · f(r/R)`
3. Lokalny kierunek redukuje się do `T(s) + O(Dean)` — tangent rzędu zerowego
4. Magnituda wymaga `Q(t)` — **niedostępna z czystej geometrii**
5. Identifiability warunki: (a) Dean << 1, (b) `Q(t)` = f(geometry) (Murray's law approx) — neither holds general
6. Wniosek: **direction id. geometryczna do O(1°) ± Dean correction; magnitude id. wymaga BC lub flow rate**

**Powiązanie z empiria:**
- VMR mediana angle 4° (Dean ~5) → consistent
- Magnitude failure (mag_ratio 0.63 dla noleak) → consistent
- noleak_centerline brak poprawy → consistent ("tangent już wyciągnięty z global PCA")
- Womersley cos_signed bimodality → consistent (without leakage, sign is undetermined → arbitrary case-level resolution)

**Effort:** 2 dni writing po empirical lock.

---

## 📚 Archive

**Phase B baseline (frozen 2026-05-13):** 3-seed leak_dir_only PP@10 0.238, angle 3.18° (ceiling). Artefakty w `results/{aggregated,bootstrap,stratified,figures}/`. Manifest sha256 w `results/manifest.json`.

**Phase D → Phase E pivot (2026-05-14):** Wprowadzony po krytyce zewnętrznej (local centerline tangent missing, Womersley falsification, architecture-independence). Phase E wykonana 2026-05-14 → 2026-05-15.

**Phase E5 (2026-05-15):** Metric repair + phase-aware diag. H1 (phase-locked flips) sfalsyfikowana, zastąpiona case-level chaos finding.

**Bibliography:** Kapoor & Narayanan *Patterns* 2023; Suk et al. *Comp Biol Med* 2024 (GEM-GCN); Suk et al. MICCAI 2024 (LaB-GATr); Tabe et al. *Sci Reports* 2026; Wang et al. *npj Digital Medicine* 2026; Pegolotti et al. *Front Cardiovasc Med* 2024; Ferdian et al. *J Biomech* 2020; *Phys Fluids* 2023 review; Alzheimer DL leakage scoping *Diagnostics* 2025.
