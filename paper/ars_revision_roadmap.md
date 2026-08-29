# Revision Roadmap (Round 1)

Maps reviewer_report.md issues to fixes, current status, and evidence.
Status: ✅ done · 🔄 in progress · ⏳ pending

## CRITICAL

| Issue | Fix | Evidence | Status |
|---|---|---|---|
| C1 single-seed | 3-seed (42/43/44) mean±std for Table A/B | eval_multiseed.py; AE per-seed exposed | ✅ (post-refactor rerun pending) |
| C2 baseline coverage | tab:lit with PatchFormer/RUL-Mamba + 10 baselines (per-SP MAE/RMSE/R²) | paper table; sources cited | ✅ (numbers will change with per-window refactor) |
| C3 K=32 unexplained | fig_horizon per-horizon error; bounded-horizon-cost claim | make_horizon_analysis.py | ✅ (K=32 being removed as innovation; analysis kept) |
| C4 SP700 TRUL=0 | dropped PANASONIC SP700 row | Table A | ✅ |

## HIGH

| Issue | Fix | Status |
|---|---|---|
| H1 citation misplacement (LSTM) | moved to recurrent-network sentence | ✅ |
| H2 "first to…" overreach | hedged to one claim | ✅ |
| H3 physics evidence in noise | 3-seed + Ea vs literature; I3 ablation redesign in-flight (extrapolation/noise experiments) | 🔄 |
| H4 window size rationale | TJU aligned to W=64 with PatchFormer | 🔄 |
| H5 related-work gaps | model-based refs, iTransformer direct cite, grey 6/6 | ✅ |
| H6 quantile under-spec | P10/P50/P90 semantics added | ✅ |
| H7 GOTION exclusion | disclosed in Datasets + compare | ✅ |

## MEDIUM / LOW

| Issue | Status |
|---|---|
| M1 "single backbone" wording | ✅ |
| M3 Table B "0–1" | ✅ |
| M6 physics-regeneration tension | ✅ (r_pred form) |
| M2/M4/M5/M7, L1-L5 | ⏳ polish |

## Round-1-new items (post-review decisions, 2026-08-13)

| Item | Status |
|---|---|
| Normalization → per-window (align with baselines) | ✅ code done; full training running (main model seed 42) |
| Interaction → stage-query attention (V3, GDN state as query) | ✅ main-model component (stage 0.0066 vs multi 0.0076 vs single 0.0070) |
| K=32 removed as innovation (I4); Table B demoted | 🔄 paper edits pending |
| MIT → 8 train / 2 test subset | 🔄 training pending (in full-training queue) |
| Physics ablation redesign (extrapolation + noise robustness) | ✅ done: 9+ routes tested; winner = physics rate head (0.0042, extrap 0.7745, drop30 AE 23→17) — written as extension section |
| AE definition unified (crossing) + SP truncation | ✅ |
| RoPE removed entirely (class + MCU header + all references) | ✅ |
| Main model decision: multiscale + StageQuery + direct + z-score (no physics in main) | ✅ (2026-08-15, user directive) |
