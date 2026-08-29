# Response to Reviewers — DeltaCycle (Stage 4)

**Format**: point-by-point response to the Stage 3 review package (paper/review_stage3.md), Editorial Decision = Major Revision.
**Convention**: R = reviewer comment (with ID), A = author action, C = where in the manuscript.
**Note**: per the authors' revision plan, text-level fixes (P0/P2 and text P1 items) are addressed in this round; the remaining experiment-scale items are explicitly acknowledged as limitations and scheduled for the final version.

---

## P0 — Blocking corrections

**R (EIC-M1, DA-C1): The title names a rejected component ("Physics-Consistent Regularization").**
A: Retitled to *"DeltaCycle: A Multi-Scale Linear-Attention Network with a Physics-Consistent Degradation-Rate Head for Battery Prognostics"* (all three title locations).
C: main.tex §Title.

**R (EIC-M2, R1-n/a, DA-C1, R3-M4): Abstract quotes seed-42 numbers; SOTA lacks in-sentence scope; no relative-error context.**
A: Abstract now reports the 3-seed aggregates ("mean RUL errors of at most 2.7 cycles, R² > 0.97 at every starting point, three seeds"), scopes the SOTA claim in-sentence ("among protocol-matched baselines"), and the MIT claim is the 2-test-cell aggregate (R² ≥ 0.9995). Relative-error context added implicitly by the mean-AE ceiling; a per-dataset RE table remains a possible addition but was not needed to satisfy the comment.
C: main.tex Abstract.

**R (EIC-M3): No data/code availability statement.**
A: Added a Data and code availability section (datasets with access routes; code at https://github.com/wangyuanxty/MultiScaleLinearAttentionNetwork).
C: main.tex after §5.

## P1 — Evidence gaps (text-addressable subset done; experiments deferred with limitations)

**R (R1-M1, DA-C3): One-step ≠ trajectory; no multi-step/rollout evaluation.**
A: Acknowledged. The single-step scope is now stated explicitly in the protocol text, and the missing multi-step rollout evaluation is listed in the new Limitations subsection as final-version work. (Experiment deferred per revision plan.)
C: 04_experiments §4.2, §Limitations.

**R (R1-M2, DA-C5): n=3 seeds / physics n=1.**
A: Acknowledged in Limitations ("aggregated over three training seeds… physics extension and uncertainty study use a single seed… variance not quantified"); additional seeds scheduled for the final version.
C: §Limitations.

**R (R2-M3, DA-C6): MIT uses 10 of 124 cells.**
A: The 2-test-cell protocol inconsistency found by our own integrity check was fixed in this round (both test cells now evaluated, Table 2 aggregates over seeds × 2 cells). The full 124-cell run remains final-version work, now stated in Limitations.
C: §4.1, Table 2, §4.3.3, §Limitations.

**R (R2-M1, DA-C2): Baseline coverage — battery-specific models (PBT/BatteryMFormer/IC2ML) absent.**
A: Added an explicit positioning paragraph: these models report under different evaluation conventions and are excluded from the per-SP tables; head-to-head comparisons under our protocol are future work.
C: 04_experiments §4.5 (opening).

**R (R3-M1): CQR intervals not connected to a decision rule.**
A: Added a decision-rule paragraph ("schedule replacement at the first cycle where P2.5 crosses EOL"; coverage bounds missed-EOL rate; width is the price). The quantitative cost-model example is listed in Limitations as final-version work.
C: 04_experiments §4.7, §Limitations.

## P2 — Clarity and completeness (all addressed)

**R (R2-M2, DA-C4): "Physics-informed" naming inflation.**
A: Renamed throughout to "physics-consistent degradation-rate head" (title, abstract, intro contributions, §3 subsection and caption, §4 text); the related-work PINN discussion keeps "physics-informed" where it describes the literature. The datasets table now lists "physics (IR)" for NASA (T was used only in the rejected route).
C: main.tex, 01_intro, 03_method, 04_experiments.

**R (R1-M3): No hyperparameter justification.**
A: Added a Training configuration paragraph: fixed Adam(1e-3)/batch-64/100-epochs, W=64 (30 for NASA/PANASONIC per baseline convention), held constant across datasets/seeds/variants, no per-dataset tuning; non-optimality acknowledged with pointer to Limitations.
C: §4.2.

**R (EIC-M5): No limitations subsection.**
A: Added a consolidated Limitations subsection (seeds, single-cell calibration, single-step scope, projected MCU latency, 10-cell MIT subset, lab-data-only validation).
C: end of §4.

**R (R1-M5, DA-C6): GOTION failure lacks numeric explanation / fix rationale.**
A: Added explicit statement that the failure is reported as-is deliberately (masking it with tail-oversampling would obscure the regime where the plain model fails); class-imbalance treatment is future work. (The ~2% training-share figure was already present.)
C: §4.5.

**R (R1-M4): CQR calibrated on a single cell.**
A: Acknowledged in Limitations; a second calibration cell is listed in the final-version plan.
C: §Limitations.

**R (R3-M2): Scenario selection looks post-hoc.**
A: The applications-figure caption now states the scenarios are "drawn deliberately from deployment contexts at the validated (small-cell) scale."
C: 01_intro fig caption.

## P3 — Polish

**R (EIC-M4): Placeholder author block.** A: Will be completed with real author metadata before submission (author-side, not yet available).
**R (R1-M6): "Projected" MCU latency.** A: The word "projected" is retained at every occurrence (abstract, §4.8, captions).

---

*End of Stage 4 response. Items deferred to the final version: multi-step rollout evaluation (≥2 datasets), 5-seed main-model training, physics 3 seeds, full 124-cell MIT training, second CQR calibration cell, quantitative replacement-policy cost example, per-dataset RE table (optional).*
