# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Battery SOH/RUL prediction research for a Journal of Power Sources paper
(**DeltaCycle** — multi-scale linear-attention network on the Gated
DeltaNet-2 recurrence, with per-layer gated cross-scale exchange,
physics-consistent regularization). The paper
lives in `paper/` (LaTeX), experiment results in `checkpoints/` +
`results/`, and the literature library (PDFs + markdown conversions) in
`literature/`; upstream reference repos in `reference_repos/`. `paper_plan.md` tracks the narrative and all measured
numbers; `paper/reviewer_report.md` tracks reviewer issues and fixes.

## Environments

- **`py312`** (D:\anaconda\envs\py312, torch 2.13) — primary environment
  for training, evaluation, and figures. Always use this one.
- `patchformer` (torch 2.6 + pytorch_forecasting) — only for running
  PatchFormer baseline code in `reference_repos/ref_patchformer/`.
- GPU is available (`torch.cuda.is_available()` True); batched forwards
  are fast, per-window Python loops are slow (~76 ms/forward).

## Commands

```bash
# Train (per-SP protocol; seed 1..10 default 1).
python src/test_unified_train.py --dataset calce --k both --seed 43
#   datasets: calce | nasa | mit-subset | panasonic | tju | gotion
#   outputs: checkpoints/unified_{ds}_K{K}_seed{S}.pt
#            results/unified_train.json (metrics incl. table_a/table_b)

# Ablation runs (single/multi/xchg/physics configs)
python src/run_ablation.py --dataset calce
#   outputs: checkpoints/abl_{ds}_{config}.pt

# Evaluation (non-recursive protocols, no retraining)
python src/test_recompute_ae.py       # Table A: per-SP MAE/RMSE/R2/AE
python src/test_physics_ir_seeds.py   # physics rate-head 10-seed (std/extrap/robust/control)
python src/eval_multiseed.py          # mean±std over seeds 42/43/44

# Figures (matplotlib; all write paper/figures/*.pdf|png)
python src/make_figures.py            # traj/ablation/stages/k32/deploy/quantile
python src/make_figures_extra.py      # regen/compare
python src/make_figures_insight.py    # w_interp/state_evol
python src/make_horizon_analysis.py   # C3 per-horizon error
# Architecture/deployment figures via baoyu-image-gen (gpt-image-2),
# key in ~/.baoyu-skills/.env

# Paper compile (pdflatex + bibtex, run pdflatex TWICE to resolve \ref)
cd paper && pdflatex -interaction=nonstopmode main.tex && bibtex main \
  && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex

# MCU deployment verification (QEMU Cortex-M3, bit-exact vs PyTorch)
#   C sources in src/*.c (gdn2_mcu.c scan kernel, qemu_test.c harness)
```

## Architecture

- **`src/gdn_v2.py`** — GDN2Block core. Recurrence per head:
  `S_t = (I − k_t(b_t⊙k_t)ᵀ)diag(α_t)S_{t−1} + k_t(w_t⊙v_t)ᵀ`, `o_t = S_tᵀq_t`;
  state is fixed H·Dk·Dv·4 bytes = 8 KB per layer. `forward(..., return_state=True)`
  returns the final state; `_scan` is a per-step Python loop (B,H batched).
- **`src/gdn_model.py`** — GDNBatteryModel (the paper's model):
  three branches (patch 2/4/8), each = patch embed + 2 GDN-2 layers;
  per-layer gated CrossScaleExchange (scalar gate `σ(MLP([pool(a);pool(b)]))`);
  fused last-token readout `head_cap` (Linear(128)→GELU→Linear(K)); optional
  PhysicsRegularizer (Arrhenius + IR, loss-only). Build via `build_gdn_model(
  multiscale=True, cross_exchange=True, readout="last")`.
- **`src/load_datasets.py`** — dataset loaders (CALCE/NASA/MIT/PANASONIC/
  TJU/GOTION). `make_figures.load_series(ds)` returns the canonical
  `(caps, train_cells, test_cell, W, sps, eol_ah)` tuple used by every
  eval/figure script. EOL = rated × 70% (small cells) / 80% (large).
- **Normalization is critical**: inputs are `(cap − lo)/(hi − lo)` with
  lo/hi from TRAIN cells only. The PatchFormer repo's `x_min/x_max` differ
  from `load_series`'s — mixing them silently biases predictions. Always
  de-normalize with the same lo/hi the model was trained with.
- **Evaluation protocols** (paper §Experiments):
  Table A = non-recursive single-step trajectory (per-SP MAE/RMSE/R²/AE,
  AE via first threshold crossing). Architecture ablation (tab:ablation): two datasets at 10 seeds; physics rate-head, extrapolation, robustness, UQ/conformal in sec:phys/sec:conf.
- **`reference_repos/`** holds read-only upstream repos (PatchFormer,
  RUL-Mamba, GatedDeltaNet-2, etc.) with their own envs/configs; do not
  modify them. Their results/trajectories feed the comparison table.

## Gotchas

- **matplotlib 3.11 tight-bbox bug**: `plt.rcParams.update()` MERGES —
  importing `make_figures` leaves its `savefig.bbox='tight'` active, which
  with mathtext text produces a broken portrait canvas. New plotting
  scripts must set `"savefig.bbox": None` explicitly (see
  `make_figures_insight.py`).
- Training runs are long (100 epochs × 10 models ≈ hours): launch with
  run_in_background and verify checkpoints/timestamps before restarting;
  never kill tasks without user consent.
- `fig_w_interp` is saved as PNG (not PDF) for the bbox bug above.
- Paper tables use `\scriptsize`+`\tabcolsep=3pt`+numeric `r` columns
  (resizebox+multirow conflict caused column overlap — avoid that combo).
- Reference entries in `paper/refs.bib` are verified (Crossref/arXiv);
  add new ones only after verification.
