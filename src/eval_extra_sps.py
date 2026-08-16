"""Extra per-SP evals for paper tables.

1. CALCE seeds 42/43/44/45 at SP 300/400/500 (decide seed set for
   Table A / lit after the seed-44 retrain).
2. PANASONIC seeds 42/43/44 at SP 300/400/500 (lit table uses
   300/400/500 to match OmniTIEFormer, Table A only reports 300/500).

Reads checkpoints/full_*_seed{S}.pt, no retraining.
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


def eval_sp(ds, seed, sp):
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
    return {"SP": sp, "TRUL": trul, "PRUL": prul,
            "MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "R2": round(float(r2), 4), "AE": ae}


def main():
    out = {}
    for ds, seeds in [("calce", [42, 43, 44, 45]),
                      ("panasonic", [42, 43, 44])]:
        sps = [300, 400, 500]
        out[ds] = {s: {} for s in seeds}
        for seed in seeds:
            for sp in sps:
                r = eval_sp(ds, seed, sp)
                out[ds][seed][sp] = r
                print(f"{ds} s{seed} SP{sp}: TRUL={r['TRUL']} "
                      f"PRUL={r['PRUL']} AE={r['AE']} MAE={r['MAE']} "
                      f"RMSE={r['RMSE']} R2={r['R2']}", flush=True)
        # aggregates per SP
        print(f"--- {ds} aggregates (seeds {seeds}):", flush=True)
        for sp in sps:
            aes = [out[ds][s][sp]["AE"] for s in seeds]
            maes = [out[ds][s][sp]["MAE"] for s in seeds]
            r2s = [out[ds][s][sp]["R2"] for s in seeds]
            rmses = [out[ds][s][sp]["RMSE"] for s in seeds]
            print(f"  SP{sp}: AE={sorted(aes)} median={np.median(aes):.1f} "
                  f"MAE={np.mean(maes):.4f}±{np.std(maes):.4f} "
                  f"RMSE={np.mean(rmses):.4f}±{np.std(rmses):.4f} "
                  f"R2={np.mean(r2s):.4f}±{np.std(r2s):.4f}", flush=True)
    os.makedirs("results", exist_ok=True)
    with open("results/extra_sps.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved results/extra_sps.json", flush=True)


if __name__ == "__main__":
    main()
