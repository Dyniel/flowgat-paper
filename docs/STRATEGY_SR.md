# Strategia SR-reframe — Scientific Reports submission

**Status: 2026-05-24, post Phase E8 (SAGE × {Cosserat, sUbend} + FlowGAT × Cosserat no-BC — all 3 stages DONE)**

**Update 2026-05-24 — Phase E8 closed. Pattern L1 replikuje się 1:1 na (FlowGAT/SAGE) × (with-BC/no-BC) × (Cosserat/sUbend); Limitations skraca się o dwa bullety ("architecture-specific", "BC-head-specific"). Pełny snapshot: [results/phase_E8_package/README.md](results/phase_E8_package/README.md). Patrz [§Phase E8 — SAGE + no-BC follow-up (NEW)](#phase-e8--sage--no-bc-follow-up-new) niżej.**

**Update 2026-05-22 — Phase E7 sUbend results dropped (training done 2026-05-20, analysis re-run pending after fix to `cosserat_sweep_diagnostic.py`). See [§Phase E7 — sUbend (NEW)](#phase-e7--subend-new) below.**

**Cel:** Scientific Reports submission (broad-audience), headline *clinical-quantity audit of vascular-flow GNN surrogates*. Pivot z Communications Physics, w którym headline było *physics of learning / identifiability decomposition*. Empirical core unchanged; tytuł, abstract, Results ordering, headline metrics changed.

**Archiwum:** poprzednia strategia (CP-1000%) jest w [docs/archive/STRATEGY_CP_archived_20260519.md](docs/archive/STRATEGY_CP_archived_20260519.md). Treść empiryczna z CP wciąż obowiązuje, tylko nadbudowa narracyjna inna.

---

## 🎯 Trzy zarzuty → trzy pivoty

| Zarzut (CP-style attack surface) | SR-style pivot |
|---|---|
| **"Test jest mały"** — n=5 VMR test cases | **Multi-domain × multi-resolution × multi-arch.** 3 datasety (VMR aorta, Womersley straight tube, **Cosserat curved tube** — NEW), 2 architektury (FlowGAT, SAGE), 3 mesh resolutions (1×, 2×, 4× — NEW), 4 feature variants × 3 seeds. ≈200 model–data combinations, jeden powtarzalny wzorzec. |
| **"Novelty trywialne"** — "data leakage to nie newsy" | **Cztery sfalsyfikowane wyjaśnienia + jedna positive structural claim.** Sfalsyfikowane: real-aorta-complexity (Womersley repl.), phase-locked sign artifact, richer-geometric-prior helps, Dean-curvature-anchors-sign (Cosserat repl.), continuity-learned-as-pattern (**NEW: mesh refinement** pokazuje że to był instrumentation artifact, nie ML failure). Pozytyw: continuity + tubular geometry → asymmetric identifiability, direction is geometric, magnitude requires BC. |
| **"Metryki z dupy — nic dla general public"** | **Lead with clinical interpretable units.** Pressure drop MAE [mmHg], peak velocity location [mm], wall shear stress MAE [Pa]. Per-seed JSONy już mają (`clin/dP_mmHg_mae`, `clin/wss_mae_Pa_mean`, `clin/peak_loc_mm_*`). PP_dir/cos_signed/frac_flipped → supplementary (technical companion). |

---

## Phase E8 — SAGE + no-BC follow-up (NEW)

**Cel:** zamknąć dwa reviewer-attack surfaces z Limitations: (a) architecture-specificity FlowGATa, (b) zależność wzorca od explicit no-slip-loss head. Trzy SLURM array jobs po 12 tasków (4 var × 3 seeds), trampoline orchestration → łącznie 36 GPU runs + 3 CPU agg jobs.

**Status:** ✅ wszystkie 3 stages DONE 2026-05-24. Pakiet wyników z opisem: [results/phase_E8_package/README.md](results/phase_E8_package/README.md).

### Headline — PP_dir@10° (test split, mean ± SD across 3 seeds)

| arch × domain × BC | withleak | leak_dir_only | leak_mag_only | noleak |
|---|---:|---:|---:|---:|
| SAGE × Cosserat (with BC) | 0.998 ± 0.013 | 0.999 ± 0.009 | **0.033 ± 0.066** | **0.029 ± 0.061** |
| SAGE × sUbend (with BC) | 0.874 ± 0.115 | 0.908 ± 0.092 | **0.171 ± 0.097** | **0.150 ± 0.085** |
| FlowGAT × Cosserat (**no BC**) | 0.992 ± 0.049 | 0.987 ± 0.080 | **0.037 ± 0.092** | **0.025 ± 0.072** |

Step function `withleak ≈ leak_dir_only ≫ leak_mag_only ≈ noleak` replikuje się identycznie we wszystkich trzech eksperymentach. Cosserat-collapse jest ostrzejszy (≈0.03) niż sUbend-collapse (≈0.15) — ten sam wzorzec jak w E7 (residual direction info bleeding through magnitude/geometry w realistycznej domenie).

### Dean-bin flip check (Lemma 1) — replicates on SAGE × Cosserat AND FlowGAT no-BC

Flip rate nie spada monotonicznie z De dla obu mag_only i noleak, na obu nowych architekturach/BC-konfiguracjach. Lemma 1 nie jest artefaktem FlowGATowego BC-head.

### Clinical-headline + L7 replication

| variant | split | dP_MAE [mmHg] | WSS_MAE [Pa] | peak_loc [mm] |
|---|---|---:|---:|---:|
| sage_subend_withleak | test | 9.54 | **0.24** | **34.0** |
| sage_subend_noleak | test | **4.77** | 1.08 | 60.6 |

L7 / D8 (Bernoulli dP-collapse trap) replikuje się też na SAGE — noleak ma niższy dP_MAE niż withleak mimo że WSS_MAE 4× gorsze i angle error 11× gorszy. Czyli to nie jest implementation quirk FlowGATa, tylko strukturalna pułapka Bernoulli-jako-single-scalar. **Wzmacnia argument za D8** (zawsze raportujemy 3 metryki razem).

### Cosmetic gotcha (nie wpływa na analizy)

`results/diagnostics/flowgat_nobc_cosserat/summary.md` — sekcja "Dean-Bin Flip Check" ma nagłówek tabeli ale puste wiersze. Dane są obecne w `stratified_by_de.csv` (24 wiersze, komplet). Najwyraźniej regression w skrypcie agg dla wariantów z suffixem `_nobc` w nazwie. **Do dorobienia ręcznie lub fix w `src/cosserat_sweep_diagnostic.py`** przed pisaniem rozdziału — porównaj z `sage_cosserat/summary.md` który ma 4 wiersze poprawnie.

### Operational note — Slurm AssocMaxSubmitJobLimit

Trampoline z agg 33 → train 31 + agg 34 padł raz na `AssocMaxSubmitJobLimit` (job submit limit). Stage 3 dosłany ręcznie. Jeśli E9+ wymaga takich łańcuchów, warto dodać retry-w-pętli w trampoline section agg-jobów (poll squeue, exponential backoff) zamiast jednorazowego sbatch.

### Co teraz dropuje się z Limitations / głównego paperu

| Wcześniejszy bullet w Limitations / open item | Status post-E8 |
|---|---|
| "Pattern może być artefaktem FlowGAT-specific message passing" | ❌ **dropped** — SAGE replikuje na 2 domenach |
| "Magnitude collapse może być forsowany przez explicit no-slip-loss head" | ❌ **dropped** — no-BC ablation replikuje na Cosserat |
| "L7 (dP-collapse) widziany tylko na FlowGAT × sUbend" | ✅ **wzmocniony** — replikuje na SAGE × sUbend, czyli prawdziwy strukturalny problem metryki |

---

## Phase E7 — sUbend (NEW)

**Training run:** job 24 array (4 variants × 3 seeds = 12 tasks), finished 2026-05-20.
**Dataset:** 15 synthetic U-bend cases (newCFD_dataset / polybox), 25 frames each, 50k nodes per graph, split 9/3/3 cases → 225 train + 75 val + 75 test frame-samples.
**Status:** unified PP_dir diagnostic (per_case.csv) computed correctly; `aggregate.csv` was empty due to a script bug (hardcoded VARIANTS list) — **fixed 2026-05-22**, re-run via `bash jobs/SUBMIT_E7_FOLLOWUP.sh` (jobs 25 + 26 + 27).

### sUbend — unified direction-identifiability (per_case n = 225 frames × var × split)

Computed from `results/diagnostics/subend/per_case.csv` (already populated; n_frames = 225 per cell):

| variant | split | PPdir@10° | cos_signed | frac_flipped | ang_median° |
|---|---|---:|---:|---:|---:|
| `subend_withleak`       | val  | **0.730** | 0.993 | 0.000 | 6.2 |
| `subend_leak_dir_only`  | val  | **0.702** | 0.992 | 0.001 | 6.8 |
| `subend_leak_mag_only`  | val  | 0.108 | 0.523 | 0.282 | 54.8 |
| `subend_noleak`         | val  | 0.119 | 0.497 | 0.319 | 55.8 |
| `subend_withleak`       | test | **0.755** | 0.994 | 0.000 | 5.7 |
| `subend_leak_dir_only`  | test | **0.771** | 0.994 | 0.000 | 5.6 |
| `subend_leak_mag_only`  | test | 0.161 | 0.736 | 0.202 | 39.0 |
| `subend_noleak`         | test | 0.178 | 0.675 | 0.282 | 43.7 |

Pattern `withleak ≈ dir_only ≫ mag_only ≈ noleak` replikuje się 1:1 na 4. domenie. Strongest replication so far in terms of n (15 cases × 25 frames × 3 seeds = 1125 per group).

### sUbend — clinical headline metrics (per-seed JSON aggregate)

| variant | dP_MAE [mmHg] val/test | WSS_MAE [Pa] val/test | WSS_bias [Pa] val/test | peak_loc [mm] val/test | ewRMSE val/test |
|---|---:|---:|---:|---:|---:|
| `subend_withleak`       | 4.58 / 8.34 | **0.25 / 0.23** | 0.11 / 0.01 | **20.4 / 34.5** | **0.26 / 0.42** |
| `subend_leak_dir_only`  | 6.61 / 9.25 | 0.37 / 0.38 | 0.29 / 0.26 | 57.3 / 49.0 | 0.28 / 0.45 |
| `subend_leak_mag_only`  | 3.30 / 7.07 | 0.33 / 0.35 | 0.10 / 0.10 | 29.1 / 27.9 | 0.35 / 0.51 |
| `subend_noleak`         | **2.20 / 4.79** | 0.71 / 0.75 | 0.65 / 0.68 | 75.3 / 70.7 | 0.40 / 0.59 |

**WSS, peak_loc, ewRMSE pattern matches expectation** (`noleak` worst, `withleak` best). **dP_MAE jest odwrócone** — noleak ma najniższe dP_MAE. Hipoteza: noleak collapsuje magnitudę → small max||u|| → small dP_pred via Bernoulli (`dP = 4·max||u||²`) → przypadkowo niski MAE bo true dP też low na większości frame'ów. **Investigation job 26 mierzy to bezpośrednio** (`max||u_pred||`, ratio vs `max||u_true||`, signed err, corr).

### Sukces narracyjny vs Limitations item

✅ **L1/L4 wzmocnione** — 4. domena (synthetic curved-tube real-CFD), pattern PP_dir replikuje się czysto, n znacznie większe niż 5-case VMR.
✅ **L2 (arch-independence)** — TBD czy puszczać SAGE na sUbend (decision in next sync); aktualnie pattern udowodniony na FlowGAT.
⚠️ **Open Limitations item (D8 — locked 2026-05-22):** raportujemy `dP_MAE` ZAWSZE razem z `WSS_MAE` i `peak_loc_mm` — sama pressure-drop może oszukiwać przy magnitude-collapsed predictions. Bernoulli dP jest single-scalar reduction całego pola — Methods note + Discussion sentence o tym ograniczeniu.

### Phase E7 — analysis pipeline status

| Krok | Status | Owner job |
|---|---|---|
| Trening 4×3 (job 24) | ✅ done 2026-05-20 | `jobs/24_train_subend.sh` |
| Per-case PP_dir CSV (1800 rows) | ✅ done | `jobs/25_subend_aggregate.sh` (per_case path) |
| Aggregate / stratify CSV | 🔧 fixed 2026-05-22, **re-run pending** | `jobs/25_subend_aggregate.sh` (post-fix to `src/cosserat_sweep_diagnostic.py`) |
| dP / magnitude-collapse investigation | 🔧 **new — pending submit** | `jobs/26_subend_dp_investigation.sh` (CPU, ~10 min) |
| Paired bootstrap (4 contrasts × 2 splits) | 🔧 **new — pending submit** | `jobs/27_subend_bootstrap.sh` (CPU, <1 min) |
| Update Fig 1 (clinical headline panel) | pending | `src/make_figures_v2.py` extension |
| Update Fig 3 (cross-domain panel: 4 domains) | pending | new figure |

Launcher: `bash jobs/SUBMIT_E7_FOLLOWUP.sh` submits 25 + 26 + 27 in parallel.

---

## 🆕 Nowe wyniki (Phase E6, 2026-05-18..19)

### Cosserat sweep — curved-tube replication, n=12 runs

4 variants × 3 seeds, mesh ~23k nodes, curved centerline z Dean number sweep (Cosserat-rod, first-order Dean secondary-flow correction). Same NPZ schema co Womersley.

| variant | n | angle° | ewRMSE | PP@10 | dP_MAE [mmHg] | WSS_MAE [Pa] |
|---|---:|---:|---:|---:|---:|---:|
| `cosserat_sweep_withleak` | 3 | 1.87 | 0.397 | 0.103 | 2.52 | 0.51 |
| `cosserat_sweep_leak_dir_only` | 3 | 1.91 | 0.400 | 0.067 | 2.47 | 0.64 |
| `cosserat_sweep_leak_mag_only` | 3 | 77.19 | 0.877 | 0.000 | 3.13 | 0.82 |
| `cosserat_sweep_noleak` | 3 | 76.45 | 0.875 | 0.000 | 3.14 | 0.80 |

Asymmetric leakage pattern przenosi się 1:1 z VMR i Womersley na curved tubes. **Dean curvature jako first-order rotational anchor NIE wystarcza** żeby model odzyskał sign bez direction-feature leakage — falsifikacja H_curvature-anchor. To bezpośrednio adresuje reviewer move "ale wasz Womersley to za prosty cylinder; curvature by zafiksowała sign". Unified diag (PP_dir, cos_signed, frac_flipped) — TBD after `jobs/22`.

### Mesh refinement — 1× / 2× / 4× × 4 variants × 3 seeds, n=36 runs

| variant | 1× | 2× | 4× | rate |
|---|---:|---:|---:|---|
| `withleak` | 0.087 | 0.026 | 0.006 | ×3.4 → ×4.3 |
| `leak_dir_only` | 0.074 | 0.020 | 0.005 | ×3.7 → ×4.3 |
| `noleak` | 0.073 | 0.021 | 0.004 | ×3.5 → ×4.8 |
| `leak_mag_only` | 0.232 | 0.064 | 0.012 | ×3.6 → ×5.2 |

`div_pred` ∝ h² → 0 z refinement we wszystkich wariantach. **L5 ("continuity learned as pattern, not as law") padło — to był O(h²) discretization artifact estymatora k-NN divergence, nie failure mass conservation.** Spójne z teorią: model uczy się ciągłości po prostu dlatego że wszystkie ground-truth velocity fields są solenoidalne; estymator pokazywał błąd O(h²) który skaluje się tak jak central differences. Czyste, broni się w peer review.

---

## 🎯 Updated 5-line evidence map (post-E6)

| # | Linia | Status | Roli w SR-paper |
|---|---|---|---|
| **L1** | Asymmetric leakage — dir_only ≈ withleak ≫ mag_only ≈ noleak | ✅ **bardzo mocna** (n=5 VMR, n=6 Womersley, n=~5 Cosserat, **n=15 sUbend ×25 frames**) | **headline** |
| **L2** | Architecture-independence — SAGE = FlowGAT pattern | ✅ **bardzo mocna** (VMR + Womersley + **Cosserat + sUbend (E8)**) | secondary, ale teraz "secondary mocne" |
| **L2b (NEW)** | BC-mechanism independence — no-slip-loss-head OFF replikuje 1:1 na Cosserat | ✅ **mocna** (E8 stage 3, n=12) | falsification 5 (closes "magnitude collapse forced by loss design") |
| **L3** | Better geometric prior nie pomaga — noleak_centerline ≈ noleak | ✅ **mocna i ciekawa** | falsification 1 |
| **L4** | Cross-domain replication — VMR + Womersley + Cosserat + **sUbend (NEW)** ten sam pattern | ✅ **bardzo mocna** (4 domeny) | falsification 2 (real-aorta), 3 (Dean-curvature), **4-bis (synthetic real-CFD curved tube)** |
| **~~L5~~** | ~~Continuity learned as pattern, not law~~ | ❌ **dropped** (mesh refinement: discretization artifact) | **wycięte z paper** |
| **L6** | Continuity convergence — div_pred ∝ h² → 0; model uczy się ciągłości, stary estymator miał O(h²) error | ✅ **mocna i pozytywna** (n=36) | falsification 4 + positive technical claim |
| **L7 (NEW, open)** | dP_MAE alone deceives — magnitude-collapsed predictions can produce low dP_MAE without recovering velocity field | 🔧 **investigation pending** (job 26) | Methods caveat + Limitations sentence |

L5 → L6 to **dobra zamiana** — z negative claim (model nie uczy się fizyki) na positive claim (uczy się, instrument miał błąd). Czyni paper safer w peer review.

L7 jest **defensive disclosure** — zgłaszamy własne ograniczenie zanim zrobi to reviewer; konwencjonalnie SR przyznaje high score reviewer pracom które proactywnie raportują ograniczenia metryk.

---

## 🧰 Headline metrics — co prowadzimy

Lead w abstract i Results:
- **Pressure drop MAE** [mmHg] — clinical sensible (kardiolog rozumie)
- **WSS MAE** [Pa] + WSS bias — clinical, Pa to standard
- **Peak velocity location** [mm] + relative magnitude error — clinical
- **3D vector angle error** [degrees] — proxy dla direction quality, deg jest intuitive

Companion (Methods + Supplementary):
- `PP_dir@10°` — strict per-node direction accuracy
- `cos_signed_median` — sign integrity
- `frac_flipped` — sign-degeneracy quantification
- `ewRMSE_he` — energy-weighted velocity error
- `corr(|u|, r)` — Poiseuille radial scaling
- `div_pred` vs h — continuity convergence

PP@10 (old VMR metric) → supplementary tylko.

---

## 🗺️ Kierunki work — co dalej

| Krok | Effort | Status | Notes |
|---|---|---|---|
| **0. E7 analysis follow-up** — fixed agg, dP investigation, bootstrap CIs | ~30 min CPU | **ready, pending submit** | `bash jobs/SUBMIT_E7_FOLLOWUP.sh` (25+26+27 in parallel) |
| **1. Cosserat + meshref aggregaty** (legacy from E6 — may already be done; verify) | 0.5h CPU | verify | `sbatch jobs/22_aggregate_cosserat_meshref.sh` if stale |
| **2. Re-skeleton paper/main.tex** — title, abstract, Intro, Results ordering pod clinical-first framing; usunięcie L5 sekcji; dodanie L6 + L7 (Limitations) + Cosserat + sUbend (4 domeny) | 1.5–2 dni | pending | bazujemy na obecnym 1197-line draft; ~70% treści zostaje, reordering + reframing |
| **3. Theoretical anchor (Cosserat rod + continuity)** | 2 dni writing | pending | sekcja Methods, derivation 1 strona; teraz spina się Cosserat + sUbend empirical perfectly |
| **4. Figures pass** — clinical-metric headline panel (Fig 1 nowy z 3 metrykami D8); leakage delta z dP_MAE+WSS_MAE+peak_loc; cross-domain panel z **4 datasetami** (VMR+Womersley+Cosserat+sUbend); mesh convergence panel; dP-collapse scatter (Fig SI z job 26) | 1 dzień | pending | reuse `results/figures/` + 3 nowe figury |
| **5. SR-format compliance** — ~~limit 4500 słów~~ SR ma "no upper word limit, but concise" od 2018; obecny main body ~5570 słów akceptowalny. Structured headings i formatting OK. | 0.5 dnia | mostly done | trim opcjonalny, nie wymagany |
| **6. Cover letter + reproducibility package (Zenodo)** | 1 dzień | **cover letter draft done** ([paper/cover_letter.tex](paper/cover_letter.tex)) | Zenodo upload pending |
| **7. Internal review + revision** | 1 dzień | pending | |

**Realny czas do submission:** ~5–7 dni focused, z czego ~50% to writing rather niż compute.

---

## 📍 SR positioning — claims dla abstract (locked-in candidates)

| Claim | Co konkretnie | Status |
|---|---|---|
| **S1 — Clinical** | GNN surrogate dla vascular blood flow osiąga ~2.5 mmHg dP_MAE i ~0.5 Pa WSS_MAE, ale **tylko kiedy ma dostępną informację o lokalnym kierunku przepływu**; bez niej dP_MAE rośnie do ~3.1 mmHg, WSS_MAE do ~0.8 Pa | ✅ |
| **S2 — Decomposition** | 4-variant feature-leakage audit jako reproducible community tool — direction-feature ≈ withleak, magnitude-feature ≈ noleak na 3 domenach × 2 arch × 3 mesh resolutions | ✅ |
| **S3 — Structural finding** | Direction recoverable z czystej geometrii (mean angle 2–4°), magnitude wymaga BC info — wynika z continuity + tubular geometry (Cosserat rod approximation) | ✅ |
| **S4 — Four falsifications** | Real-aorta complexity, phase-locked artifacts, richer-geometric-prior, Dean-curvature, continuity-not-learned — wszystkie 5 obalone | ✅ |
| **S5 — Reproducibility/honesty** | Per-stratum + per-case + bootstrap reporting; mesh-refinement evidence że divergence numbers w prior CFD-ML papers mogą być O(h²) artifacts | ✅ |

---

## 🚫 Czego NIE robimy w SR-pivot

- ❌ Nie wycinamy theoretical anchor — w SR jest jeszcze bardziej cenny niż w CP, bo daje paper structural respectability
- ❌ Nie szukamy WSS R² hero number — Limitations zostają jak były
- ❌ Nie wskazujemy palcem konkretnych prac — community-service framing
- ❌ Nie szufladkujemy 4-variant audit jako "ablation" — to *audit methodology*, framing musi być stand-alone tool

---

## 📚 Decyzje narracyjne — LOCKED 2026-05-19

- **D1 (SR)** Single full-stack paper (~4500 słów + supplementary)
- **D2 (SR)** Clinical-quantity-first framing — dP, WSS, peak velocity w abstract i Fig 1
- **D3 (SR)** Identifiability decomposition jako Methods + Discussion content, **nie** jako headline
- **D4 (SR)** Cosserat curved-tube sweep dołączony jako 3. domena replication (Phase E6)
- **D5 (SR)** L5 wycięte; zastąpione L6 (mesh-refinement convergence)
- **D6 (SR)** External polybox dataset (`newCFD_dataset.zip`) — przeznaczenie TBD po unzip; rozważyć jako 4. domain out-of-distribution test
- **D7 (SR, locked 2026-05-22)** sUbend = 4. domena replication (Phase E7). Trening 4×3 done, pattern L1/L4 replikuje się 1:1 → wchodzi do main paper jako "fourth domain". SAGE arch na sUbend = opcjonalne (jeśli czas pozwoli, jako Supplementary).
- **D8 (SR, locked 2026-05-22)** Clinical headline = TRZY metryki razem: `dP_MAE [mmHg]` + `WSS_MAE [Pa]` + `peak_loc [mm]`. Nigdy nie raportujemy dP_MAE samego — Bernoulli single-scalar może być deceiving (job 26 quantyfikuje). Discussion zawiera explicit sentence o tym.
- **D9 (SR, locked 2026-05-24)** Architecture- i BC-robustness do main paper jako pełny rezultat (nie tylko Supplementary). Po Phase E8 mamy 2 arch × 2 BC × 2 curved-domains z identycznym wzorcem — to przestaje być "robustness check" i wchodzi do Results jako struktural finding. Plan: nowa subsekcja w Results (~½ strony) + 3×4 supplementary panel z PP_dir box-plotami.
- **D10 (SR, locked 2026-05-24)** Phase E8 — żadne dalsze ablation runs przed submission. Lista jest zamknięta: VMR (5) + Womersley (6) + Cosserat (12+12+12) + sUbend (12+12) = 71 GPU runs jak w aggregowanej kolumnie evidence map. Każdy dodatkowy run = opóźnienie, a obecne pokrycie spełnia D9. Wszystko inne idzie do "Future Work" w Discussion.

---

## 📦 Nowe artefakty (post-CP archive)

**Code:**
- [src/make_npz_cosserat_sweep.py](src/make_npz_cosserat_sweep.py) — generator curved-tube mesh + Dean correction
- [src/make_npz_cosserat_sweep_variants.py](src/make_npz_cosserat_sweep_variants.py) — 4-variant fork
- [src/cosserat_sweep_diagnostic.py](src/cosserat_sweep_diagnostic.py) — agg + Dean/eps stratification (reuses womersley_metrics). **Patched 2026-05-22** to take variants list at runtime; previously hardcoded `cosserat_sweep_*` → aggregate.csv was empty for sUbend.
- [src/mesh_refinement_diagnostic.py](src/mesh_refinement_diagnostic.py) — eval + aggregate mode
- [src/make_npz_subend.py](src/make_npz_subend.py) — sUbend noleak NPZ builder (15 cases × 25 frames, 50k nodes)
- [src/make_npz_subend_variants.py](src/make_npz_subend_variants.py) — 3-variant fork (withleak, leak_dir_only, leak_mag_only)
- [src/subend_dp_investigation.py](src/subend_dp_investigation.py) — **NEW 2026-05-22**: magnitude-collapse / Bernoulli-dP diagnostic; outputs `dp_investigation_per_frame.csv`, `_summary.csv`, `.md`, and `figures/subend_dp_scatter.{pdf,png}`. Adresuje L7.
- [src/bootstrap_ci.py](src/bootstrap_ci.py) — **Patched 2026-05-22**: added `clin/dP_mmHg_mae` and `clin/wss_bias_Pa_mean` do METRICS list.
- [src/make_fig_clinical_headline.py](src/make_fig_clinical_headline.py) — **NEW 2026-05-22**: 4-domain × 4-variant clinical-headline grouped-bar generator (Fig 1 candidate). Output: `fig_clinical_headline.{pdf,png}` + `_values.csv`.
- [paper/cover_letter.tex](paper/cover_letter.tex) — **NEW 2026-05-22**: SR cover letter draft (1 page, suggested reviewers, COI placeholder).

**Configs:** `configs/cosserat_sweep_{withleak,leak_dir_only,leak_mag_only,noleak}.yaml`

**Jobs (E8 NEW, 2026-05-22..24):**
- `jobs/29_train_sage_cosserat.sh` — 4×3 = 12-task GPU array, SAGE arch on Cosserat sweep
- `jobs/30_train_sage_subend.sh` — 4×3 = 12-task GPU array, SAGE arch on sUbend
- `jobs/31_train_flowgat_nobc_cosserat.sh` — 4×3 = 12-task GPU array, FlowGAT with no-slip-loss head **disabled** on Cosserat
- `jobs/32_aggregate_sage_cosserat.sh`, `jobs/33_aggregate_sage_subend.sh`, `jobs/34_aggregate_flowgat_nobc_cosserat.sh` — agg jobs (CPU); 32/33 trampoline-submit next stage
- `jobs/SUBMIT_E8_SAGE_BC.sh` — orchestrator launcher (3-stage chain via trampoline pattern)

**Configs (E8 NEW):** `configs/sage_{cosserat,subend}_{withleak,leak_dir_only,leak_mag_only,noleak}.yaml`, `configs/cosserat_sweep_{withleak,leak_dir_only,leak_mag_only,noleak}_nobc.yaml`

**Jobs (E7 and earlier):**
- `jobs/19_train_cosserat_sweep.sh` — 4×3 = 12-task GPU array (~3–5h/task)
- `jobs/20_cosserat_sweep_pipeline.sh` — login-node orchestrator + diagnostic
- `jobs/21_mesh_refinement_eval.sh` — 4×3×3 = 36-task GPU array (eval-only on existing ckpts)
- `jobs/22_aggregate_cosserat_meshref.sh` — **CPU agg job**, runs both cosserat + meshref aggregation
- `jobs/23_subend_build_npz.sh` — sUbend dataset build (CPU)
- `jobs/24_train_subend.sh` — 4×3 = 12-task GPU array (sUbend)
- `jobs/25_subend_aggregate.sh` — sUbend direction-identifiability aggregate (CPU, **fixed 2026-05-22**)
- `jobs/26_subend_dp_investigation.sh` — **NEW 2026-05-22**: magnitude-collapse / Bernoulli-dP diagnostic (CPU)
- `jobs/27_subend_bootstrap.sh` — **NEW 2026-05-22**: paired bootstrap CIs for 4 contrasts × 2 splits (CPU)
- `jobs/28_clinical_headline_fig.sh` — **NEW 2026-05-22**: Fig 1 generator (clinical headline 4-domain × 4-variant grouped bar panel, CPU)
- `jobs/SUBMIT_E7_FOLLOWUP.sh` — **NEW 2026-05-22**: launcher dla 25 + 26 + 27 + 28 (independent parallel CPU jobs)

**Data:**
- `data/npz_cosserat_sweep{,_withleak,_leak_dir_only,_leak_mag_only}/`
- `data/npz_womersley_meshref_{1x,2x,4x}/`
- `data/npz_subend{,_withleak,_leak_dir_only,_leak_mag_only}/` — **built 2026-05-19** by job 23 (375 NPZ per variant × 4 = 1500 NPZs total). 15 cases × 25 frames, split 9/3/3.
- `data/external/newCFD_dataset.zip` — extracted 2026-05-19 → 15 sUbend cases ingested for E7. **Integration done**; raw .vtk frames can be deleted once npz dirs are verified (~75 GB savings on /scratch1 if needed).

**Results:**
- `results/predictions/cosserat_sweep_{withleak,leak_dir_only,leak_mag_only,noleak}/seed_*/{val,test}_*.npz`
- `results/predictions/subend_{withleak,leak_dir_only,leak_mag_only,noleak}/seed_*/{val,test}_sUbend_*__f*.npz` (~1800 files, ~3 GB)
- `results/diagnostics/mesh_refinement/parts/{variant}_{res}_seed{seed}.csv` (36 files)
- `results/diagnostics/subend/per_case.csv` (1800 rows; PP_dir etc. via unified `womersley_metrics`)
- `results/per_seed/cosserat_sweep_*_{val,test}_seed*{.csv,_aggregate.json}` (24 files × 2)
- `results/per_seed/subend_*_{val,test}_seed*{.csv,_aggregate.json}` (48 files; clinical metrics inside)
- **(E8 NEW)** `results/predictions/{sage_cosserat,sage_subend,cosserat_sweep}_*{,_nobc}/seed_*/{val,test}_*.npz`
- **(E8 NEW)** `results/per_seed/{sage_cosserat,sage_subend,cosserat_sweep}_*{,_nobc}_{val,test}_seed*{.csv,_aggregate.json}` — 144 files (3 runs × 4 var × 2 splits × 3 seeds × 2 formats)
- **(E8 NEW)** `results/diagnostics/{sage_cosserat,sage_subend,flowgat_nobc_cosserat}/{summary.md,aggregate.csv,per_case.csv,stratified_by_de.csv,stratified_by_eps.csv}`
- **(E8 NEW)** `results/phase_E8_package/` — curated snapshot with combined README + clinical_headline.csv + copies of all three diagnostics dirs (`sage_cosserat/`, `sage_subend/`, `flowgat_nobc_cosserat/`)

---

## 📅 Phase timeline

- Phase B (frozen 2026-05-13): VMR baseline
- Phase D → E pivot (2026-05-14): local centerline tangent, Womersley falsification, SAGE second-arch
- Phase E5 (2026-05-15): metric repair, phase-aware diagnostic
- Phase E6 (2026-05-18..19): curved-tube replication (Cosserat) + mesh refinement — closes "test too small" and "metrics from ass" attack surface
- Phase E7 (2026-05-19..20, analysis 05-22): sUbend 4th-domain replication. Pattern L1/L4 replikuje się; dP_MAE anomaly → L7 (Limitations item). Analysis re-run: `bash jobs/SUBMIT_E7_FOLLOWUP.sh` (25 + 26 + 27).
- **Phase E8 (2026-05-22..24): SAGE × {Cosserat, sUbend} + FlowGAT × Cosserat no-BC.** 36 GPU runs (3 stages × 12), all ✅ DONE. Closes "architecture-specificity" + "BC-head-specificity" attack surfaces. L7 replicates on SAGE × sUbend. Pakiet: [results/phase_E8_package/](results/phase_E8_package/). Launcher: `bash jobs/SUBMIT_E8_SAGE_BC.sh` (trampoline-chained — uwaga na AssocMaxSubmitJobLimit przy ręcznym recoverze).
- Phase E9 (planned): SR reskeleton + theoretical anchor + figures + cover letter. **Compute frozen** per D10.

---

## 📚 Bibliography note

Existing refs OK. Dodatki dla SR pivot:
- Cosserat rod theory ref (Antman, Rubin) — dla Methods derivation
- Mesh-convergence / divergence-estimator literature — dla L6 framing
- Polybox dataset citation — Suk et al. (TBD po unzip)
