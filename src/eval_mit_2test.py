"""MIT evaluation over BOTH test cells (batch2_cell5, batch2_cell47).

The pipeline's load_series('mit') returns MIT_TEST_CELLS[0] only, so
Table A / case-study numbers were batch2_cell5-only. This script
evaluates both test cells from the existing checkpoints (no
retraining) and aggregates per-SP metrics over seeds x cells, to
support the "8 train / 2 test" protocol the paper describes.

Output: results/mit_2test.json + printed table.
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
from load_datasets import MIT_TEST_CELLS

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-6


def eval_cell(model, caps, cell, W, sps, eol_ah, lo, hi):
    tc = (caps[cell] - lo) / (hi - lo + EPS)
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
        rows.append({"SP": sp, "TRUL": trul, "PRUL": prul, "AE": ae,
                     "MAE": mae, "RMSE": rmse, "R2": r2})
    return rows


def main():
    caps, tr, _, W, sps, eol = load_series("mit")
    out = {}
    for seed in [42, 43, 44]:
        ck = torch.load(f"../checkpoints/full_mit_seed{seed}.pt",
                        map_location=DEV, weights_only=False)
        cfg = ck["config"]
        model = build_gdn_model(
            multiscale=cfg["multiscale"], stage_query=cfg["stage_query"],
            input_dim=cfg["input_dim"], window_size=cfg["window_size"],
            output_len=cfg["output_len"], num_quantiles=cfg["num_quantiles"],
            readout=cfg["readout"]).to(DEV)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        lo, hi = ck["lo"], ck["hi"]
        for cell in MIT_TEST_CELLS:
            rows = eval_cell(model, caps, cell, W, sps, eol, lo, hi)
            out.setdefault(cell, {}).setdefault(seed, rows)
            print(f"seed{seed} {cell}: " + " | ".join(
                f"SP{r['SP']} TRUL={r['TRUL']} PRUL={r['PRUL']} "
                f"AE={r['AE']} MAE={r['MAE']:.4f} R2={r['R2']:.4f}"
                for r in rows), flush=True)

    print("--- aggregates (seeds x cells) ---", flush=True)
    agg = {}
    for sp in sps:
        vals = {"MAE": [], "RMSE": [], "R2": [], "AE": [], "TRUL": []}
        for cell in MIT_TEST_CELLS:
            for seed in [42, 43, 44]:
                r = out[cell][seed][sps.index(sp)]
                vals["MAE"].append(r["MAE"]); vals["RMSE"].append(r["RMSE"])
                vals["R2"].append(r["R2"]); vals["AE"].append(r["AE"])
                vals["TRUL"].append(r["TRUL"])
        agg[sp] = {k: (float(np.mean(v)), float(np.std(v)))
                   for k, v in vals.items()}
        print(f"SP{sp}: TRUL={np.mean(vals['TRUL']):.0f} "
              f"MAE={np.mean(vals['MAE']):.4f}±{np.std(vals['MAE']):.4f} "
              f"RMSE={np.mean(vals['RMSE']):.4f}±{np.std(vals['RMSE']):.4f} "
              f"R2={np.mean(vals['R2']):.4f}±{np.std(vals['R2']):.4f} "
              f"AE={np.mean(vals['AE']):.2f} (values {sorted(set(round(a) for a in vals['AE']))})",
              flush=True)
    os.makedirs("results", exist_ok=True)
    with open("results/mit_2test.json", "w") as f:
        json.dump({sp: {k: v[0] for k, v in agg[sp].items()} for sp in agg},
                  f, indent=2)
    print("saved results/mit_2test.json", flush=True)


if __name__ == "__main__":
    main()
