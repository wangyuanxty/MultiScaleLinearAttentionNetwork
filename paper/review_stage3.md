# Stage 3 Peer Review Package — DeltaCycle (v1.0 draft)

- **Mode**: full (5-person panel: EIC + 3 peer reviewers + Devil's Advocate)
- **Date**: 2026-08-16
- **Paper**: paper/main.tex (35 pp., Journal of Power Sources, elsarticle preprint)
- **Reviewer skill**: academic-paper-reviewer v1.10.0

---

## Phase 0 — Field Analysis & Reviewer Configuration

| Item | Determination |
|---|---|
| Primary discipline | Battery prognostics / data-driven RUL (engineering) |
| Secondary | Edge ML systems; management-science operations (reliability/maintenance) |
| Research paradigm | Empirical benchmarking + ablation, physics-informed model extension |
| Methodology type | Deep learning sequence model, single-step trajectory regression, conformal UQ, embedded deployment |
| Target journal tier | Journal of Power Sources (IF ~9, applied battery journal) |
| Paper maturity | Structurally complete; statistical depth (seed counts, protocol scope) is the weak axis |

**Reviewer panel:**

1. **EIC** — Editor of Journal of Power Sources, battery systems focus. Cares: fit, novelty positioning, completeness of validation, reproducibility statements.
2. **R1 (Methodology)** — Time-series forecasting statistician; cares: evaluation protocol rigor, seed discipline, statistical reporting, hyperparameter justification, reproducibility.
3. **R2 (Domain)** — Battery prognostics researcher (RUL baselines, physics-based+data-driven hybrids); cares: baseline coverage, physics claims vs actual mechanism, dataset conventions.
4. **R3 (Perspective)** — Operations/management science reviewer (maintenance scheduling, decision-making under uncertainty); cares: decision value, cost framing, deployment feasibility, honesty of outlook claims.
5. **DA (Devil's Advocate)** — Challenges core claims, finds strongest counter-arguments, overgeneralization, frame-lock.

---

## Phase 1 — Review Reports

### Report 1: Editor-in-Chief

**Summary of the paper.** DeltaCycle proposes a multi-scale linear-attention network (Gated DeltaNet-2 backbone, patch 2/4/8 branches, stage-query cross-scale exchange) for capacity trajectory prediction with a physics-informed degradation-rate head (IR-driven, absolute-space), conformalized quantile regression, and verified MCU deployment. Claims: SOTA on PANASONIC/TJU, R²≥0.9995 on MIT, physics head lifts tail-extrapolation R² 0.374→0.775, corruption robustness, 8 KB-state edge deployment with <3e-6 PyTorch agreement.

**Overall assessment.** The paper is technically competent, unusually honest about its failures (GOTION exclusion, CALCE behind PatchFormer, tested-and-rejected physics routes), and the four-pillar narrative (accuracy / physics / trust / deployability) is coherent for an applied journal. My concerns center on (i) a stale title, (ii) the abstract's reliance on single-seed numbers, and (iii) missing standard reporting items.

**Major issues.**

- **M1 (Major) — Title no longer matches the method.** The title reads "Physics-Consistent Regularization". The regularization route was tested and *rejected* by the authors themselves (§4.4: Arrhenius+IR prior worsens extrapolation R² 0.61→0.34). The shipped mechanism is a degradation-rate *head*, not regularization. A title describing a rejected component misleads at first contact. Retitle (e.g., "…with a Physics-Informed Degradation-Rate Head…").
- **M2 (Major) — Abstract quotes seed-42 numbers.** "RUL errors of 0–6 cycles with R² > 0.972 at every starting point (seed 42)" — the body's central table is 3-seed mean±std. Citing a single-seed range in the abstract while the body reports 3-seed aggregates invites "which number is real?" questions. Either quote the 3-seed aggregates or clearly justify the seed-42 basis.
- **M3 (Major) — No data/code availability statement.** JPS requires Data Availability; the manuscript has none (datasets are public but loaders/checkpoints/scripts are not referenced). Add a statement + repository reference.
- **M4 (Minor) — Author block is placeholder** ("Author One", example.com). Complete before submission.
- **M5 (Minor) — No explicit Limitations subsection.** Limitations are scattered (GOTION, seed counts, MCU scope); collect them into one subsection for reviewability.

**Novelty positioning (for the record).** The core recurrence (GDN-2) is NVIDIA's; stage-query exchange is architecturally close to OmniTIEFormer's TCEM; CQR is standard. The defensible novelty is the *synthesis*: a fully verified embedded deployment chain plus the absolute-space rate-head mechanism finding. The introduction should state this division of labor more sharply to preempt novelty challenges.

**Recommendation: Major Revision.**

---

### Report 2: R1 — Methodology

**Strengths.** Protocols are unusually carefully aligned across papers (normalization verified line-by-line against PatchFormer/RUL-Mamba preprocessing); internal consistency is high; the authors reproduce their own physics numbers on independent reruns; the flat-tail AE spread on CALCE is explained quantitatively (slope 0.0022/cycle).

**Major issues.**

- **M1 (Major) — Single-step protocol vs. trajectory claims.** All headline metrics are single-step (window→next value). The "decision-grade trajectory quality" and "long-horizon planning" claims (Case studies, §4.3; pillar ②) concern *multi-step* horizons, yet no multi-step/rollout experiment is reported anywhere for the main model (only the phys head's 10% tail). A reviewer will ask: what is the k-step-ahead error at k=50, 100? Either report multi-step errors or narrow the claims to what was measured.
- **M2 (Major) — Statistical basis.** n=3 seeds throughout; the physics section n=1. AE reported as mean over n=3 with spreads like 0/1/7 (CALCE). For a paper selling "decision-grade" intervals to operations audiences, 3 seeds is thin. Minimum: report all 5-seed plans as limitations and justify why 3 was chosen; consider bootstrap CIs for the AE mean.
- **M3 (Major) — No hyperparameter justification.** W=64, lr=1e-3, batch 64, 100 epochs, single-branch vs multi-branch configs — no search, no sensitivity. PatchFormer (the baseline) used TPE search; a one-config result invites "did you tune the baselines less than your model?" Add a sensitivity statement or a small lr/W sweep.
- **M4 (Medium) — CQR calibration single-cell.** q_adj calibrated on CS2_36 only, evaluated on CS2_35. Coverage 0.933 < 0.95 nominal is honestly admitted, but calibration with n=869 windows on one cell is fragile; report the calibration on a second cell or acknowledge fragility.
- **M5 (Medium) — GOTION AE=31 unexplained numerically.** The EOL-acceleration segment is "~2% of training data" — the authors should show the segment length and why 2% suffices to explain a 31-cycle error (vs. data imbalance fixes like oversampling that could have been tried).
- **M6 (Minor) — MCU numbers are projected.** "Projected 8–12 ms on STM32F4" — projected, not measured; x86/QEMU verification is real but the F4 number is arithmetic. Fine if labeled, but the abstract's "8–12 ms projected inference" should keep the word "projected" (it does — keep it everywhere).

**Recommendation: Major Revision.**

---

### Report 3: R2 — Domain (battery prognostics)

**Strengths.** Regeneration handling (PANASONIC zoom) is concrete; EOL conventions (70%/80%) are correct; the injection-route study (regularizer/feature/structural-head all failing under z-score) is a genuinely useful negative result the community needs; honest CALCE positioning.

**Major issues.**

- **M1 (Major) — Baseline coverage.** The lit tables cover PatchFormer, RUL-Mamba, general TSF models, and OmniTIEFormer (indirectly). Missing from the comparison: battery-specific 2024–2026 models the field would expect — PBT (pretrained battery transformer), BatteryMFormer, IC2ML, and the early-prediction line (Zhou et al. is cited for MIT but not compared on NASA/TJU). Add a paragraph positioning against these or justify exclusion.
- **M2 (Major) — "Physics-informed" naming.** The rate head solves no physics equations; it applies a monotonicity structural prior and reads a measured IR covariate. The authors' own ablation ([C,IR] input gives no gain; absolute-space control head 0.0097) shows the win is the *structure + objective space*, with IR as a driver. Calling this "physics-informed" is defensible but borderline; "physics-consistent structural head" or the paper's own "degradation-rate head driven by measured IR" is safer and preempts the PINN community's objection.
- **M3 (Medium) — MIT uses 10 of 124 cells.** "Full-set training planned for the final version" — reviewers strongly dislike planned work. With 124 cells available and training times of minutes, run the full set now or state a technical blocker.
- **M4 (Medium) — Temperature physics absent.** The title/abstract mention Arrhenius; the final head has no temperature term (tested-and-rejected). Ensure no lingering text implies temperature is used at inference (check §4.1 datasets table "physics (IR+T)" for NASA — clarify T is used only in the rejected route).
- **M5 (Minor) — The datasets table says NASA "physics (IR+T)"** — if T ended up unused in the surviving mechanism, the table overstates. Align.

**Recommendation: Major Revision.**

---

### Report 4: R3 — Perspective (operations / management science)

**Strengths.** Per-dataset case studies with explicit operational implications are exactly the right form for an applied audience; the applications figure maps scenarios→capabilities→decisions; the honesty markers ("deployment outlooks, not deployment reports", GOTION exclusion, CALCE behind PatchFormer) build credibility; the CALCE AE spread paragraph is the kind of explanation an operator actually needs.

**Major issues.**

- **M1 (Major) — Intervals are never connected to a decision rule.** CQR gives P2.5/P50/P97.5, but no section shows how an operator uses them (replacement threshold policy, cost model, expected-cost comparison vs. fixed-interval maintenance). One worked example (e.g., "replace when P2.5 crosses EOL" and its cost consequences on CALCE) would convert the UQ pillar from a metric to a decision tool. At minimum add a paragraph; ideally a small table.
- **M2 (Medium) — Scenario selection is post-hoc.** The six application scenarios are all small-cell/embedded contexts — precisely where the model was validated. A skeptical reader reads this as "scenarios chosen to fit the validation", not "validation chosen to fit the market". One sentence acknowledging the selection logic (validated-scale scenarios first; large-format future work) would defuse this — the intro already gestures at it; make it explicit.
- **M3 (Medium) — No total-cost framing for edge vs cloud.** The deployment pillar claims "near-zero marginal cost" in the ledger-level narrative but the paper itself doesn't quantify connectivity/TCO savings. A qualitative comparison suffices for JPS; add one short paragraph.
- **M4 (Minor) — "0–6 cycles RUL error" in the abstract without units of context** — 0–6 cycles on 33–577-cycle horizons; add relative error (RE) to the abstract for operators (e.g., "≤2% relative RUL error").

**Recommendation: Major Revision.**

---

### Report 5: Devil's Advocate

**Strongest counter-argument.** This paper's headline claims are built on cross-paper number transcription, and the paper itself demonstrates how fragile that substrate is. Its own comparison tables were corrected during preparation for a wrong-method transcription (RUL-Mamba's TJU row) and a mislabeled column (PANASONIC SP400); one of its physics numbers (0.447) turned out to be a stale code-state artifact. If the *authors'* transcription needed four rounds of auditing, how much weight can any "SOTA on PANASONIC/TJU" claim carry when the same fragility applies to every baseline number the comparison rests on — and when, on the one dataset where a self-run baseline exists (CALCE), the self-run PatchFormer *beats* the proposed model? The paper's real claim is not accuracy superiority; it is a deployment-synthesis claim. It should lead with that.

**Issue list.**

- **C1 (CRITICAL) — Title/abstract overclaim the method.** "Physics-Consistent Regularization" (title) names a rejected component; the abstract's seed-42 numbers contradict the body's 3-seed numbers. A reader who stops at the title+abstract carries a false picture of both the method and the evidence basis. This alone blocks acceptance.
- **C2 (CRITICAL) — Selective-SOTA framing.** "State-of-the-art on PANASONIC and TJU" is true *within the compared set*, but the compared set excludes the strongest recent battery-specific models (PBT, BatteryMFormer) and the paper loses on CALCE, is mixed on NASA, and reports no comparison at all for MIT (only a different-protocol R² citation). "SOTA" claims in the abstract should carry the scope qualifier *in the same sentence*, not in a later footnote.
- **C3 (MAJOR) — One-step ≠ trajectory.** Every headline number is one-step-ahead. The operational narrative (maintenance scheduling, long-horizon planning) presumes trajectory quality over hundreds of cycles. The only long-horizon evidence is a single-seed, single-dataset (CALCE) 10%-tail experiment. The mismatch between measured quantity and claimed use is the paper's biggest internal gap.
- **C4 (MAJOR) — Physics framing inflation.** The "physics" is a positivity-constrained linear readout on a measured IR — the authors' own control experiments (direct-abs head 0.0097; [C,IR] input no-gain) show the wins come from structure and objective space, not from physical modeling. Calling it physics-informed invites a category-error attack from the PINN community; the authors should rename to match what it is (a physics-consistent structural head).
- **C5 (MAJOR) — n=3 seeds, n=1 for physics.** Decision-grade claims on three seeds, with an AE whose mean swings from 1 to 2.7 depending on one seed's flat-tail crossing, is the weakest leg of the "trustworthy" pillar. Either add seeds (cheap — training is minutes) or demote the trust claims accordingly.
- **C6 (MINOR) — GOTION exclusion is also a missed opportunity.** An AE=31 failure mode *because* the tail is under-represented could have been fixed with oversampling or tail-weighted sampling in an afternoon; excluding it and citing "omni-targeted models mitigate with EOL-focused training" reads as citing a competitor's strength as a reason not to compete.

**Ignored alternative explanations.** The rate-head extrapolation win (0.374→0.775) is attributed to physics; an equally parsimonious explanation is that (a) the last-value anchor Q̂=Q_last−r prevents drift (structural), and (b) IR is simply a contemporaneous health covariate. The paper's own direct-abs control supports this reading. If both explanations are true, the contribution is "structural readout design", not "physics" — the authors should own whichever is the honest version.

**Missing stakeholder perspectives.** No BMS-engineer perspective on IR measurability in the field (is per-cycle IR available on commodity BMS hardware?); no operator perspective on false-alarm cost (what does a 5-cycle AE cost vs. a missed EOL?).

**Observations (non-defects).** The honesty about CALCE and GOTION is genuinely above the field's norm. The MCU verification chain is real engineering work and under-sold relative to the SOTA claims.

**Recommendation: Reject in current form; Major Revision with re-review.**

---

## Phase 2 — Editorial Synthesis

### Cross-reviewer matrix

| Issue | EIC | R1 | R2 | R3 | DA | Consensus |
|---|---|---|---|---|---|---|
| Title names rejected component | M1 | — | — | — | C1 | **Unanimous-adjacent (CRITICAL)** |
| Abstract seed-42 vs 3-seed basis | M2 | — | — | M4 | C1 | **Consensus (Major)** |
| No multi-step/rollout evaluation | — | M1 | — | — | C3 | **Consensus (Major)** |
| n=3 seeds / physics n=1 | — | M2 | — | — | C5 | **Consensus (Major)** |
| Physics naming inflation | — | — | M2 | — | C4 | Consensus (Major) |
| Selective-SOTA framing scope | — | — | — | — | C2 | DA-only but well-founded (Major) |
| Baseline coverage (PBT/BatteryMFormer/IC2ML) | — | — | M1 | — | C2 | R2+DA (Major) |
| Decision-rule connection for CQR | — | — | — | M1 | — | R3 (Major) |
| Data/code availability statement | M3 | — | — | — | — | EIC (Major, JPS mandatory) |
| Hyperparameter justification | — | M3 | — | — | — | R1 (Major) |
| MIT 10/124 cells | — | — | M3 | — | — | R2 (Medium→Major via DA framing) |
| CQR single calibration cell | — | M4 | — | — | — | R1 (Medium) |
| GOTION: numeric explanation / fix attempt | — | M5 | — | — | C6 | R1+DA (Medium) |
| Limitations section | M5 | — | — | — | — | EIC (Minor) |
| MCU "projected" labeling | — | M6 | — | — | — | R1 (Minor) |
| T in datasets table | — | — | M5 | — | — | R2 (Minor) |

### Arbitration

- The two CRITICAL items (title; abstract basis) are not in dispute and are cheap to fix — no arbitration needed.
- **Multi-step evaluation (R1-M1, DA-C3)**: The authors have deliberately avoided autoregressive rollout (error-accumulation narrative) — but the paper *consumes* long-horizon claims without measuring them. Arbitrated requirement: add a k-step/rollout experiment for the main model on ≥2 datasets, OR demote every long-horizon/decision claim to the one-step evidence. The pipeline can do the former cheaply.
- **Seeds (R1-M2, DA-C5)**: Training is minutes-scale; requiring 5 seeds is proportionate. Physics section ≥3 seeds.
- **Selective-SOTA (DA-C2)**: The scoping qualifier must move into the same sentence as the SOTA claim. Minimum viable fix.
- **Physics naming (R2-M2, DA-C4)**: rename to "physics-consistent structural head"; no experiments change.

### Editorial Decision

**Decision: MAJOR REVISION** (Devil's Advocate CRITICAL findings preclude Accept per iron rule; the fixes are substantial but none require new core research).

### Revision Roadmap (prioritized)

**P0 (blocking — correctness of claims):**
1. Retitle: replace "Physics-Consistent Regularization" with the surviving mechanism (e.g., "Physics-Consistent Degradation-Rate Head").
2. Abstract: replace seed-42 numbers with 3-seed aggregates; add RE to the RUL-error claim; scope "state-of-the-art" in-sentence ("among the compared protocol-matched baselines").
3. Add Data Availability + Code Availability statement (JPS mandatory).

**P1 (evidence gaps):**
4. Multi-step evaluation: rollout errors at k=25/50/100 for ≥2 datasets (CALCE + PANASONIC or TJU); if large, recalibrate the long-horizon narrative.
5. Seeds: extend main-model to 5 seeds (add 45/46 where missing); physics section to 3 seeds.
6. MIT: run the full 124-cell set (or state the technical blocker explicitly).
7. Baseline positioning: one paragraph vs. PBT/BatteryMFormer/IC2ML with justification of exclusion.
8. CQR: connect intervals to a decision rule (worked replacement-policy example or cost table).

**P2 (clarity & completeness):**
9. Rename physics mechanism per R2/DA (physics-consistent structural head) and align all text (incl. datasets table "IR+T").
10. Hyperparameter justification paragraph (fixed-config rationale or sensitivity sweep).
11. Limitations subsection consolidating: seeds, single-cell CQR calibration, projected MCU latency, MIT subset, lab-data-only validation.
12. GOTION: add numeric segment stats + state why no oversampling fix was attempted.
13. Second-cell CQR calibration check.
14. Applications figure caption: explicit scenario-selection logic sentence (already drafted in intro; make it unmistakable).

**P3 (polish):**
15. Author block completion.
16. MCU: keep "projected" labeling consistent everywhere.

---

*End of Stage 3 review package. Read-only: no manuscript edits performed in this stage.*
