# DeltaCycle: A Multi-Scale Linear-Attention Network with a Physics-Consistent Degradation-Rate Head for Battery Prognostics

Official code for the DeltaCycle paper (battery SOH/RUL prediction).

## Overview

DeltaCycle is a multi-scale linear-attention architecture for lithium-ion battery
capacity trajectory and remaining-useful-life (RUL) prediction:

- **Backbone**: Gated DeltaNet-2 recurrence (fixed-size matrix state, linear time),
  three parallel branches over patch sizes 2/4/8.
- **Cross-scale exchange**: stage-query attention — the coarse branch's GDN state
  queries the fine/mid branches after every layer.
- **Physics-consistent rate head** (extension): `r = softplus(w·h) + softplus(γ)·IR`
  with γ ≥ 0, `Q̂_t = Q_{t-1} − r`, trained in absolute capacity space.
- **Uncertainty**: pinball-quantile regression with conformalized quantile
  regression (CQR), one forward pass, P2.5/P50/P97.5.
- **Deployment**: full C implementation of the GDN-2 scan matching PyTorch within
  floating-point rounding (< 3e-6), fixed 8 KB recurrent state, no dynamic
  allocation, INT8 at −75% weight memory.

## Repository layout

```
src/
  gdn_v2.py            GDN2Block core (recurrence, scan)
  gdn_model.py         DeltaCycle model + physics rate head + quantile/CQR
  load_datasets.py     dataset loaders (CALCE/NASA/MIT/PANASONIC/TJU/GOTION)
  test_full_train.py   main-model training (multiscale + stage-query, z-score protocol)
  test_eval_tableA.py  per-SP trajectory evaluation from checkpoints
  eval_mit_2test.py    MIT two-test-cell evaluation
  run_ablation.py      architecture/physics ablations
  test_ablation_robust.py  corrupted-input robustness (drop30/gauss/impulse)
  test_phys_irhead.py  physics rate-head training + unseen-tail extrapolation
  test_quantile_uq.py  quantile training + CQR calibration
  make_figures*.py     paper figure generation
  export_gdn_weights.py  weight export for the C deployment
  *.c / *.h            MCU deployment (gdn2_mcu.c scan kernel, verification harnesses)
```

Other `test_*.py` / `diag_*.py` files are one-off experiments kept for
reproducibility; the pipeline-relevant entry points are listed above.

## Setup

- Python 3.12, PyTorch ≥ 2.0 with CUDA (CPU works, slower).
- `pip install numpy pandas matplotlib scipy scikit-learn openpyxl h5py`

## Data

Public datasets used (download and place under `data/` as below; loaders expect
the standard formats):

| Dataset | Source | Loader |
|---|---|---|
| CALCE | Center for Advanced Life Cycle Engineering (request access) | `data/calce/<cell>/` xlsx files |
| NASA PCoE | [NASA Prognostics Data Repository](https://www.nasa.gov/prognostics-data-repository) | `data/nasa/` |
| MIT/Stanford | [data.matr.io/1](https://data.matr.io/1/) | `data/mit_stanford/` .mat (v7.3) |
| PANASONIC | as distributed by OmniTIEFormer / TJU sources | `data/panasonic/` |
| TJU | as distributed by PatchFormer / RUL-Mamba | `ref_rul_mamba/Data/TJU_Data/` |

## Reproduction

```bash
# Train the main model (multiscale + stage-query, z-score protocol)
python src/test_full_train.py --dataset calce --seed 42     # calce|nasa|mit|panasonic|tju

# Per-SP trajectory metrics (Table 2 in the paper) from saved checkpoints
python src/test_eval_tableA.py

# Physics extension: rate head + unseen-tail extrapolation + robustness
python src/test_phys_irhead.py
python src/test_ablation_robust.py

# Uncertainty quantification (quantiles + CQR)
python src/test_quantile_uq.py
```

Checkpoints and results are git-ignored (`checkpoints/`, `results/`).

## Citation

If you use this code, please cite the DeltaCycle paper (link added upon publication).
