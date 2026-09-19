# Reviewer Report — Gated DeltaFormer (v0 draft, 19 pages)

> **Fix status (2026-08-10 round 1)**: C2 ✅(tab:lit 文献对比), C4 ✅(SP700 行删除),
> H1 ✅, H2 ✅, H5 ✅(iTransformer 直接引用;model-based 引用线待补), H7 ✅(GOTION 披露),
> M1 ✅, M3 ✅(Table B PANASONIC=1). 未修: C1(多 seed), C3(K32 horizon 分析), H3(物理参数对照文献值),
> H4(W 敏感性), H6(quantile 实验描述), M2/M4/M5/M6/M7, L1–L5.

**Mode**: academic-paper-reviewer / full
**Target**: Journal of Power Sources
**Date**: 2026-08-10
**Verdict preview**: Major revision — the early-sensing + deployment story is
novel and well-executed, but statistical rigor, protocol transparency, and
baseline coverage need substantial work before submission.

---

## CRITICAL (must fix before submission)

### C1. Single-seed results with no significance testing
Every headline number (AE 0–2, PANASONIC xchg −53% MAE, R² 0.99+ table) comes
from one seed. The paper itself contrasts with literature reporting 10-run
averages. A reviewer will not accept "data-dependent architecture choice"
without error bars: the differences between single/multi/xchg on CALCE
(MAE 0.0092 vs 0.0131) are untestable as stated.
→ Fix: 3–5 seeds mean±std for (a) Table A all datasets, (b) ablation table,
(c) physics params. At minimum for the datasets backing the two main claims
(CALCE, PANASONIC, MIT).

### C2. Baseline coverage is a single model family
Table A compares only against PatchFormer (same group as RUL-Mamba, USTC).
RUL-Mamba, MambaLithium, LSTM, and general TSF baselines are mentioned in
Related Work but never evaluated. The Introduction claims RUL-Mamba is
SOTA; the Experiments never confront it.
→ Fix: add OmniTIEFormer (cite numbers, protocol caveat) + PatchFormer now;
RUL-Mamba/LSTM/TSF as soon as runs complete. Until then the "matches
state-of-the-art" claim is unsupported.

### C3. K=32 early sensing costs 1–17 cycles of accuracy vs K=1 — unexplained
K=1 AE is 0–2; K=32 AE is 0.2–16.9. The first few steps of a 32-step
one-shot should be nearly as accurate as K=1. If the model locates EOL
16.9 cycles off with a 31-cycle window, the reviewer will ask whether the
"early sensing" is sensing anything real, or whether the trajectory is
simply biased low in the EOL phase (the same under-learning PatchFormer is
accused of). Currently no analysis of *why* K=32 is 16.9 on CALCE but 0.2
on MIT.
→ Fix: show per-step accuracy of the K=32 window (error by horizon k), and
discuss the trade-off curve AE(K) explicitly.

### C4. PANASONIC SP=700 row has TRUL=0
TRUL=0 means the SP coincides with true EOL; AE=0 is vacuous. It inflates
"AE 0–2 across datasets" and the RE=0.000 entry looks suspicious.
→ Fix: drop the row or replace with SP=600 with a real RUL, and note the
EOL-adjacent SP policy.

---

## HIGH (should fix before submission)

### H1. Citation context error: LSTM classic cited for model-based methods
`ref_lstm_transformer` (Hochreiter & Schmidhuber 1997) is cited after
"Model-based approaches—equivalent-circuit models, extended/unscented Kalman
filters, particle filters—…" (01_intro L16–20). The LSTM paper is a data-driven
recurrent network, not a model-based method.
→ Fix: move citation to the recurrent-network sentence (L21–23); cite
ECM/EKF/UKF/PF properly (missing: 4–6 standard refs for the model-based
family — related work section also lacks them).

### H2. "First to…" claims overreach (three instances)
- "first battery RUL model with an end-to-end bit-exact MCU verification chain"
- "the first to adapt GDN-2 to battery prognostics"
- "the first to demonstrate bit-exact MCU deployment of a linear-attention battery model"
"To the best of our knowledge" only guards the first. All three are
unverifiable and a reviewer will push back.
→ Fix: keep one hedged claim ("to the best of our knowledge, no prior battery
RUL work reports bit-exact MCU verification"); drop the GDN-2-first claim
(it adds nothing).

### H3. Physics-consistency evidence is within seed noise
CALCE R² 0.9936→0.9960 (+0.002); NASA 0.9939→0.9917 (−0.002). With one seed
these are indistinguishable from noise; the abstract "zero accuracy cost"
is fine, but "physically correct parameters" rests only on sign of
γ_ir/γ_t and Ea ∈ [30,60] kJ/mol, an extremely wide band. Also: I3 is
evaluated only on CALCE/NASA (2 of 5 datasets) because PANASONIC/MIT/TJU
have no IR/T — the contribution's scope is smaller than the paper implies.
→ Fix: (a) multi-seed for the two physics datasets; (b) compare learned Ea
against a published value for LCO (not a range); (c) state explicitly that
the regularizer is demonstrated where IR/T exist.

### H4. Window size inconsistent without justification
W=64 for CALCE/MIT, W=30 for NASA/PANASONIC/TJU. Reviewers will ask why
and whether results are sensitive to W. No W-sensitivity study exists.
→ Fix: one sentence of justification (sequence lengths / last-SP distance);
optionally a W∈{30,64} robustness column for CALCE.

### H5. Related Work gaps
- No coverage of model-based methods (ECM, EKF/UKF/PF) despite Introduction
  L16–19 claiming to classify them — the classification sentence has no refs.
- iTransformer is named but not cited (02_related L15: cited only via
  ref_omnitie).
- Direct-RUL / foundation-model line (RUL-QMoE, BatteryGPT, IC2ML) from the
  literature notes is absent.
→ Fix: add 6–10 refs; cite iTransformer directly.

### H6. Quantile section under-specified
No dataset, window, or model config given for the quantile experiment; P50
AE=7 vs point model 6–7 is asserted without table; coverage 90.6% vs 80%
target on what test set?
→ Fix: report dataset (CALCE?), config, and a full table.

### H7. GOTION exclusion is silent
A 6th dataset was dropped for poor EOL-acceleration learning (AE=31). The
paper selects 5 of 6 datasets without disclosing selection. A reviewer
auditing data availability will see the checkpoint files.
→ Fix: one sentence in Datasets: "GOTION (27 Ah LFP) was excluded because
the EOL-acceleration segment is ~2% of data and the model failed to learn
it (AE=31); this failure mode is discussed in §Comparison."

---

## MEDIUM

### M1. "The single backbone serves both" (03_method §readout) — actually two
checkpoints
K=1 and K=32 are separately trained models (unified_*_K1.pt / K32.pt). The
sentence "the only difference is the output dimension" is correct at
architecture level but the text implies weight sharing.
→ Fix: "the same architecture and training setup; only the output dimension
differs."

### M2. MIT R²=1.0000 to 4 decimals
Perfect R² on all three SPs will trigger a data/protocol suspicion reflex.
Either report more decimals, note the smoothness of LFP fade + same-protocol
training cells, or report MAE only for MIT.

### M3. Table B "PANASONIC 0–1" indeterminate
A table entry "0–1" looks unmeasured. Report the exact value (0 or 1) or
explain the range (per-SP).

### M4. Deployment latency is projected, not measured
8–12 ms on STM32F4 is extrapolated from QEMU cycle counts. The text says
"projected" — keep it prominent and do not let "8–12 ms" appear bare in
abstract/table captions.

### M5. Ablation "systematically study" (Intro contribution 2) overstates
No patch-size sweep, no per-branch ablations, one W. "Systematically study
multi-scale patch decomposition" is too strong for 3 fixed patch sizes.

### M6. Physics-regeneration tension unaddressed
I2 handles regeneration (PANASONIC) but I3's r_phys ≥ β ≥ 0 is
nonnegative while regeneration gives negative r_pred. The text says the
long-term form is "regeneration-compatible" (validated on synthetic data)
but the paper never says this — a reviewer reading eq. (5) will ask how a
nonnegative physics target coexists with regeneration data.

### M7. Intro "four limitations remain" — (c) is a strong claim without
citation
"existing works either silently leak this information or restrict
evaluation to short horizons" — needs at least one example citation or
softening.

---

## LOW

### L1. hyperref Unicode warnings (4) — benign but clean up for camera-ready.
### L2. CRediT / Declaration / Acknowledgements placeholders empty.
### L3. Author names/affiliation placeholders ("Author One", "your
institution").
### L4. "Table A/Table B" naming — fine internally, but caption should carry
the full descriptive protocol names so the labels are self-contained.
### L5. Conclusion repeats "matches the state-of-the-art PatchFormer baseline"
— consistent with C2: match ≠ contribution; lead with early sensing +
deployment.

---

## What survives review well
- **Early-sensing protocol (Table B + Fig. k32)**: genuinely novel
  capability, AE=16.9 reproduces exactly between figure and table.
- **PatchFormer AR-32 stall analysis**: concrete mechanism (per-window
  renormalization, under-learned EOL segment), not hand-waving.
- **Coarse-branch stage separability (95.5%, 2.9× chance)**: real
  interpretability result.
- **Bit-exact QEMU verification chain**: unusual and credible; keep as the
  centerpiece of the deployment section.
- **Data-dependent architecture finding**: defensible once seeded.

## Priority order for revision
1. C1 (seeds) + C2 (baselines) — experimental blockers
2. C3 (K=32 analysis) — narrative blocker
3. H1/H2/H5 (citations + claims) — cheap, high-value fixes
4. C4/H3/H4/H7 (protocol transparency)
5. M1–M7, L1–L5 — polish
