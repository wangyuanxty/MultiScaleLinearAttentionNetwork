# Synthesis Report — Literature Synthesis for DeltaCycle

## 1. Method families and our positioning

| Family | Key refs | Position vs us |
|---|---|---|
| Model-based (EKF/UPF/PF/ECM) | ref_ekf_yan, ref_upf_cong, ref_pf_echem, ref_pf_lui, ref_pf_rbf | Physics-first; parameter ID burden; no regeneration flexibility. We keep physics but in the objective. |
| Traditional ML (SVM/SVR/RVM) | ref_svm_klass, ref_svr_wang, ref_svm_patil, ref_rvm_guo | Fixed kernels; cannot extrapolate EOL acceleration. |
| Deep sequence (LSTM/GRU/TCN/CNN-BiLSTM) | ref_lstm_rul_zhang, ref_gru_ding, ref_tcn_lstm, ref_cnn_bilstm, ref_lstm_elman_li, ref_rnn_catelani, ref_gru_rnn | Long-term dependency limits; no deployment story. |
| Transformer for batteries | ref_trans_rul, ref_ceemdan_trans, ref_patchformer | Quadratic attention blocks MCU; PatchFormer is our main baseline. |
| General TSF (Autoformer/FEDformer/iTransformer/PatchTST/TimesNet/TimeMixer) | ref_autoformer, ref_fedformer, ref_itransformer, ref_patchtst, ref_timesnet, ref_timemixer | Strong accuracy, no battery-specific deployment. |
| SSM/Mamba for batteries | ref_mamba, ref_mamba_survey, ref_samamba_tsf, ref_mambalithium, ref_rulmamba | Linear-time but GPU-speed claims only; no MCU validation. |
| Grey system models | ref_grey_hybrid, ref_grey_frac, ref_grey_pemfc, ref_grey_nn, ref_grey_kalman, ref_grey_iter | Hand-crafted forms; small-data strengths; we are data-driven. |
| Physics-informed battery | ref_wang_pinn (Nat. Comm. 55-cell 6-protocol), ref_piddm (Arrhenius in objective) | **Our I3 paradigm matches PiDDM**; key difference: pure-capacity inference + deployment. |
| Linear attention / delta rule | ref_gdn, ref_gdn2 | We adapt GDN-2 to battery; first bit-exact MCU linear-attention battery model. |
| Direct-RUL / foundation | RUL-QMoE, BatteryGPT, PBT, IC2ML (cite-only) | Direct EOL output loses trajectory; we keep trajectory. |

## 2. Key comparison metrics (baselines, as reported)

| Baseline | NASA AR² | TJU AR² | CALCE AR² | Notes |
|---|---|---|---|---|
| PatchFormer | 0.9807 | 0.9997 | 0.9951 | 10-run; per-window normalization |
| RUL-Mamba | 0.9742 | 0.9987 (multivar) | — | no CALCE; TJU multivar only |
| iTransformer | 0.9730 | 0.9996 | 0.9839 | strongest general TSF |
| ModernTCN | 0.9644 | 0.9997 | 0.9878 | |
| Ours (pre-refactor, global norm) | 0.9810 | 0.9980 | 0.9917 | 3 seeds |

## 3. Protocol & normalization synthesis

- **Normalization divergence** (our investigation): PatchFormer/RUL-Mamba use
  pytorch_forecasting `EncoderNormalizer` (per-window z-score). Per-window
  collapses all cells into one distribution (data-efficiency) but breaks
  multi-step extrapolation (K=32 AE 15.9→25.3) and adds MCU cost. Decision:
  **switch to per-window** for strict baseline comparability; K=32 innovation
  dropped.
- **AE definition**: crossing-based (`seg[i] ≥ th > seg[i+1]`) unified across
  scripts; PatchFormer's single-point rule equivalent on smooth sequences.
- **SP truncation**: test sequence starts at SP−W (PatchFormer-style); verified
  numerically identical to full-sequence slicing.

## 4. Gaps we fill

1. First GDN-2 adaptation to battery prognostics with bit-exact MCU verification.
2. Systematic comparison of interaction mechanisms (scalar gate / state-query
   attention / no-exchange) under identical protocol.
3. Three-way physics-injection comparison (input / joint-output / objective)
   with Pareto conclusion.
4. Cross-normalization robustness study (per-window vs global) with empirical
   evidence for the K=32 breakage.

## 5. Open risks

- ~~I3 value evidence still in-flight~~ **Resolved (2026-08-15)**: 9+ injection-route
  experiments completed. Physics features (IR/T/EIS) are redundant with capacity
  history everywhere (objective penalty ❌, input concat ❌, structural head
  under z-score ❌ — sign-mismatch, absolute space unlocks it). Winner:
  physics rate head r = softplus(w·h) + softplus(γ)·IR, Q̂ = Q_last − r:
  MAE 0.0042 vs direct 0.0070; extrapolation 0.7745; robustness all 4 modes
  win (drop30 AE 23→17). Main model stays multiscale + StageQuery + direct +
  z-score (protocol-consistent); physics rate head becomes the extension/
  ablation section.
- Comparison numbers will shift with the refactor; full re-training in
  progress (main model, 5 datasets × 3 seeds, seed 42 first).
- MIT subset (8/2) vs other datasets' single-test protocol → document
  explicitly.
