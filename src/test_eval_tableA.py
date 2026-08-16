"""Table A per-SP evaluation from saved checkpoints (no retraining).

Loads checkpoints/full_{ds}_seed42.pt, rebuilds the model, evaluates
the PatchFormer-style non-recursive protocol per starting point:
  segment from SP-W onward; per-window z-score de-normalization;
  TRUL/PRUL via first EOL crossing; AE = |TRUL - PRUL|;
  trajectory MAE/RMSE/R^2 on the segment.

Output: results/table_a_seed42.json + printed table.
"""
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, '.')
from gdn_model import build_gdn_model
from make_figures import load_series
from eval_multiseed import true_rul

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-6


def load_model(ds, seed=42):
    ck = torch.load(f"../checkpoints/full_{ds}_seed{seed}.pt",
                    map_location=DEV, weights_only=False)
    cfg = ck["config"]
    model = build_gdn_model(
        multiscale=cfg["multiscale"], stage_query=cfg["stage_query"],
        input_dim=cfg["input_dim"], window_size=cfg["window_size"],
        output_len=cfg["output_len"], num_quantiles=cfg["num_quantiles"],
        readout=cfg["readout"],
    ).to(DEV)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def eval_dataset(ds, seed=42):
    model, ck = load_model(ds, seed)
    caps, train_cells, test_cell, W, sps, eol_ah = load_series(ds)
    lo, hi = ck["lo"], ck["hi"]
    tc = (caps[test_cell] - lo) / (hi - lo + EPS)
    th = (eol_ah - lo) / (hi - lo + EPS)

    seg_p = []
    with torch.no_grad():
        for i in range(W, len(tc)):
            win = tc[i - W:i, None]
            cin = torch.tensor(win, dtype=torch.float32).unsqueeze(0).to(DEV)
            wmean = float(win[:, 0].mean())
            wstd = float(win[:, 0].std()) + EPS
            seg_p.append(model(cin).item() * wstd + wmean)
    seg_p = np.array(seg_p)
    tv = tc[W:]

    rows = []
    for sp in sps:
        s = sp - W
        seg_t, seg_p_s = tv[s:], seg_p[s:]
        n = min(len(seg_t), len(seg_p_s))
        seg_t, seg_p_s = seg_t[:n], seg_p_s[:n]
        trul = true_rul(seg_t, th)
        prul = true_rul(seg_p_s, th)
        ae = abs(trul - prul)
        mae = float(np.mean(np.abs(seg_t - seg_p_s)))
        rmse = float(np.sqrt(np.mean((seg_t - seg_p_s) ** 2)))
        r2 = 1 - np.sum((seg_t - seg_p_s) ** 2) / (
            np.sum((seg_t - seg_t.mean()) ** 2) + EPS)
        rows.append({"SP": sp, "TRUL": trul, "PRUL": prul,
                     "MAE": round(mae, 4), "RMSE": round(rmse, 4),
                     "R2": round(float(r2), 4), "AE": ae})
        print(f"  {ds} SP{sp}: TRUL={trul} PRUL={prul} MAE={mae:.4f} "
              f"RMSE={rmse:.4f} R2={r2:.4f} AE={ae}", flush=True)
    return rows


def main():
    ds_list = ["calce", "nasa", "mit", "panasonic", "tju"]
    seeds = [42, 43, 44]
    all_rows = {ds: [] for ds in ds_list}
    for seed in seeds:
        for ds in ds_list:
            print(f"eval {ds} seed{seed} ...", flush=True)
            all_rows[ds].append(eval_dataset(ds, seed))

    # aggregate mean +/- std per SP
    agg = {}
    for ds in ds_list:
        sps = [r["SP"] for r in all_rows[ds][0]]
        out_rows = []
        for k, sp in enumerate(sps):
            row = {"SP": sp}
            for m in ["TRUL", "PRUL", "MAE", "RMSE", "R2", "AE"]:
                vals = [all_rows[ds][s][k][m] for s in range(len(seeds))]
                mu = float(np.mean(vals))
                sd = float(np.std(vals))
                row[m] = round(mu, 4)
                row[m + "_std"] = round(sd, 4)
            out_rows.append(row)
        agg[ds] = out_rows
        for r in out_rows:
            print(f"  {ds} SP{r['SP']}: TRUL={r['TRUL']:.0f} "
                  f"MAE={r['MAE']:.4f}±{r['MAE_std']:.4f} "
                  f"RMSE={r['RMSE']:.4f}±{r['RMSE_std']:.4f} "
                  f"R2={r['R2']:.4f}±{r['R2_std']:.4f} "
                  f"AE={r['AE']:.1f}±{r['AE_std']:.1f}", flush=True)
    os.makedirs("results", exist_ok=True)
    with open("results/table_a_3seed.json", "w") as f:
        json.dump(agg, f, indent=2)
    print("saved results/table_a_3seed.json", flush=True)


if __name__ == "__main__":
    main()
