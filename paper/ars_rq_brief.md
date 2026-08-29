# RQ Brief — DeltaCycle: Multi-Scale Linear Attention for Battery Prognostics

## Research Questions

**RQ1 (deployment)**: Can battery SOH/RUL prediction run on microcontrollers with
deterministic memory and verified bit-exact inference, rather than GPU-scale
models?

**RQ2 (architecture)**: How should multi-scale capacity representations interact
in a linear-attention backbone — and is the optimal interaction scheme
data-dependent across degradation regimes?

**RQ3 (physics)**: Can degradation physics (Arrhenius + internal resistance) be
injected through the training objective so the model stays pure-capacity at
inference, and does that injection provide measurable value (physical
consistency, extrapolation robustness)?

**RQ4 (protocol)**: What is the cleanest non-recursive evaluation protocol for
early RUL sensing, and how do normalization choices (per-window vs global)
affect comparability with existing SOTA baselines?

## Scope

- **In**: SOH trajectory + RUL (EOL threshold crossing) on 5 public datasets
  (CALCE, NASA, MIT, PANASONIC, TJU); pure-capacity input; MCU deployment
  (QEMU-verified).
- **Out**: raw voltage/current models, pack-level, real-time hardware power
  measurement, GOTION (excluded with disclosure).

## Deliverables

1. Method: GDN-2 linear attention + multi-scale branches + stage-query
   attention exchange + physics-in-objective + per-window normalization.
2. Evidence: Table A (non-recursive per-SP MAE/RMSE/R²/AE, 3 seeds),
   comparison table vs PatchFormer/RUL-Mamba/10 TSF baselines,
   deployment chain (INT8 + QEMU bit-exact).
3. Journal: Journal of Power Sources (elsarticle).

## Success Criteria

- AE ≤ 3 cycles mean across datasets (Table A, 3 seeds).
- R² ≥ 0.97 at all SPs.
- INT8 ≤ 1% AE degradation; QEMU bit-exact.
- I3: physics params in correct direction; extrapolation/noise value
  demonstrated or honestly scoped.

## Constraints

- Baselines from same-group papers (PatchFormer/RUL-Mamba) must be
  cross-checked with independent sources.
- All refs verified (Crossref/arXiv).
- Normalization and evaluation protocol must match baselines for fair tables.
