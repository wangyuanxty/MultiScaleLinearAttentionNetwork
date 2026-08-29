# Stage 3' Verification Review Report — DeltaCycle (revised)

- **Mode**: re-review (verification of Stage 4 revisions against Stage 3 comments)
- **Date**: 2026-08-16
- **Panel**: field_analyst + EIC + editorial_synthesizer (re-review panel per protocol)

## R&R Traceability Matrix

| Stage-3 ID | Concern | Response claim (Stage 4) | Verified in manuscript? | Verdict |
|---|---|---|---|---|
| EIC-M1 / DA-C1 | Title names rejected component | Retitled | main.tex §Title: "…Physics-Consistent Degradation-Rate Head…" | ✅ |
| EIC-M2 / DA-C1 / R3-M4 | Abstract seed-42 basis; SOTA scope | 3-seed aggregates; in-sentence scope | Abstract: "mean RUL errors of at most 2.7 cycles … three seeds"; "among protocol-matched baselines" | ✅ |
| EIC-M3 | No data/code availability | Section added | "Data and code availability" §after-5, GitHub URL present | ✅ |
| R1-M1 / DA-C3 | One-step ≠ trajectory claims | Scope stated + limitation | §4.2 protocol text; §Limitations lists multi-step rollout as final-version work | ✅ (accepted resolution path: demotion+acknowledgment) |
| R1-M2 / DA-C5 | n=3 seeds / physics n=1 | Acknowledged | §Limitations quotes seed counts and unquantified variance | ✅ |
| R2-M3 / DA-C6 | MIT 10/124 cells | 2-test-cell fix + limitation | Table 2 aggregates seeds × 2 cells (TRUL 406/306/206); §4.1 "10-cell subset (8 training, 2 test)"; full-set listed in §Limitations | ✅ |
| R2-M1 / DA-C2 | Baseline coverage gap | Positioning paragraph | §4.5 opening: protocol-convention exclusion statement, head-to-head as future work | ✅ |
| R3-M1 | CQR not tied to decision rule | Decision-rule paragraph | §4.7 "From intervals to a decision rule"; quantitative cost example deferred in §Limitations | ✅ |
| R2-M2 / DA-C4 | "Physics-informed" inflation | Renamed | Title/abstract/§3 subsection+caption/§4 text → "physics-consistent"; datasets table "physics (IR)" | ✅ |
| R1-M3 | No hyperparameter justification | Config paragraph | §4.2 "Training configuration" (fixed Adam/batch/epochs/W, no per-dataset tuning) | ✅ |
| EIC-M5 | No limitations section | Added | §Limitations (7 items incl. seeds, calibration cell, single-step scope, projected latency) | ✅ |
| R1-M5 / DA-C6 | GOTION unexplained | Rationale added | §4.5: deliberate as-is reporting; imbalance treatment = future work | ✅ |
| R1-M4 | CQR single calibration cell | Acknowledged | §Limitations lists second calibration cell in final-version plan | ✅ |
| R3-M2 | Post-hoc scenario selection | Caption sentence | fig:appl caption: "drawn deliberately from deployment contexts at the validated (small-cell) scale" | ✅ |
| EIC-M4 | Placeholder authors | Deferred (author-side) | Still placeholder | ⏳ pending author input (not a technical defect) |
| R1-M6 | "Projected" latency labeling | Retained everywhere | Abstract/§4.8/captions all say "projected" | ✅ |

**Matrix result: 14/14 technical concerns verified as addressed; 1 author-side item pending; 0 concerns dropped silently.**

## New issues found in the revised manuscript (fresh read)

- **N1 (Minor, fixed in-line this round)**: Abstract claimed "long-horizon planning" from the physics tail experiment — evidence is single-dataset, single-seed. Reworded to "longer-horizon extrapolation and field-grade data resilience" (main.tex). Residual full-horizon planning language removed.
- **N2 (Minor, fixed in-line)**: Data-availability statement previously claimed the manuscript source is in the repository; the repository is code-only by author decision. Statement now reads "available from the authors upon request" — consistent with the actual repo. ✅
- **N3 (Note)**: `ref_patchformer`/`ref_rul_mamba` are submodule pointers; TJU data path in the README references `ref_rul_mamba/Data/TJU_Data/`, valid after `git submodule update --init`. No action.
- **N4 (Note)**: The MIT 2-test-cell change alters Table 2 aggregates vs. the earlier single-cell numbers; case study and abstract were checked and match (R² ≥ 0.9995). Consistent. ✅

## Decision

**MINOR REVISION → resolved.** The two minor residuals (N1, N2) were fixed in-line during this review round. All Stage 3 concerns are either addressed or explicitly scheduled with limitation statements. No new experiments were required for this decision; the deferred experiment list remains tracked in §Limitations and the Response to Reviewers.

**Next: Stage 4.5 FINAL INTEGRITY (mandatory full re-verification from scratch).**

*Read-only: no manuscript edits performed by reviewers; the two in-line fixes were applied by the authors per the panel's notes.*
