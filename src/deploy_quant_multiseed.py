"""Ten-seed version of the deployment quantization experiment.

The paper's quantization numbers came from a SINGLE seed: running
verify_q8_trajectory.py on checkpoints/per_sp/calce/SP300_seed1.pt gives
fp32 R2=0.9955 MAE=0.0059, which is where "(R2=0.9955, MAE=0.00581)" comes
from.  One seed is not a result, and seed 1 happens to sit below the ten-seed
mean (Table 2 puts CALCE SP300 at 0.0065), so the section currently quotes a
favourable draw.

This runs the same three precisions over all ten seeds and reports the mean
and std.  It reuses verify_q8_trajectory's functions unchanged.

MAE and RMSE are reported in **Ah**, matching tab:tableA and tab:lit_all, by
multiplying the normalized value back by the train-cell range.  R^2 is
dimensionless and AE is in cycles, so neither moves.

Writes src/results/deploy_quant_10seed.json.  Reads only checkpoints.

Usage: python src/deploy_quant_multiseed.py [--sp 300] [--seeds 1-10]
"""
import argparse
import io
import json
import os
import sys

import numpy as np
import torch

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
sys.path.insert(0, _SRC)

from make_figures import load_series                                    # noqa: E402
from verify_q8_trajectory import (build_and_load, eval_ae,              # noqa: E402
                                  quantize_like_export)


def trajectory_predictions(model, tc, W):
    """Batched version of verify_q8_trajectory.trajectory_predictions.

    That one feeds the windows one at a time (`model(cin).item()` inside a
    Python loop), which costs ~4 minutes per model here.  There is no
    cross-window state in this path -- every window is an independent forward
    pass -- so the whole stack can go through in one call.

    Kept numerically identical to the original: same windows, same de-scaling
    with numpy's default (biased) std, model in eval mode.
    """
    wins = np.stack([tc[i - W:i] for i in range(W, len(tc))]).astype(np.float32)
    cin = torch.tensor(wins).unsqueeze(-1)
    model.eval()
    with torch.no_grad():
        pn = np.asarray(model(cin).cpu()).reshape(-1)
    return pn * wins.std(axis=1) + wins.mean(axis=1)

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="calce")
ap.add_argument("--sp", type=int, default=300)
ap.add_argument("--seeds", default="1-10")
ap.add_argument("--out", default=os.path.join(_SRC, "results",
                                              "deploy_quant_10seed.json"))
args = ap.parse_args()
lo_s, hi_s = (int(x) for x in args.seeds.split("-"))
SEEDS = list(range(lo_s, hi_s + 1))

caps, train_cells, test_cell, _W, _sps, eol_ah = load_series(args.dataset)
all_tr = np.concatenate([caps[c] for c in train_cells])
lo, hi = float(all_tr.min()), float(all_tr.max())
span = hi - lo

print(f"{args.dataset} {test_cell}  SP{args.sp}  range={span:.4f} Ah  "
      f"seeds {lo_s}-{hi_s}")
print(f"{'seed':>4s} {'precision':>9s} {'R2':>9s} {'AE':>6s} "
      f"{'MAE (Ah)':>10s} {'RMSE (Ah)':>10s}")

rows = {}
for seed in SEEDS:
    ck = os.path.join(_ROOT, "checkpoints", "per_sp", args.dataset,
                      f"SP{args.sp}_seed{seed}.pt")
    if not os.path.exists(ck):
        print(f"{seed:>4d}  MISSING {os.path.basename(ck)}")
        continue
    obj = torch.load(ck, map_location="cpu", weights_only=False)
    state = obj["state_dict"] if "state_dict" in obj else obj
    w = int(obj["W"]) if isinstance(obj, dict) and "W" in obj else _W
    # each checkpoint carries the lo/hi it was trained with; de-normalise with
    # those, not with a freshly computed range
    c_lo = float(obj["lo"]) if isinstance(obj, dict) and "lo" in obj else lo
    c_hi = float(obj["hi"]) if isinstance(obj, dict) and "hi" in obj else hi
    tc_s = (caps[test_cell] - c_lo) / (c_hi - c_lo)
    tv_s = tc_s[w:]
    eol_s = (eol_ah - c_lo) / (c_hi - c_lo)

    for label, st in (("fp32", state),
                      ("int8", quantize_like_export(state, 8)),
                      ("int4", quantize_like_export(state, 4))):
        model = build_and_load(st, "ms", w)
        pv = trajectory_predictions(model, tc_s, w)
        r2 = 1 - np.sum((tv_s - pv) ** 2) / np.sum((tv_s - tv_s.mean()) ** 2)
        _te, aes = eval_ae(pv, tv_s, eol_s, (args.sp,), w)
        mae = float(np.abs(tv_s - pv).mean()) * span
        rmse = float(np.sqrt(((tv_s - pv) ** 2).mean())) * span
        rows.setdefault(label, []).append(
            {"seed": seed, "R2": float(r2), "AE": float(aes[0]),
             "MAE_Ah": mae, "RMSE_Ah": rmse})
        print(f"{seed:>4d} {label:>9s} {r2:9.4f} {aes[0]:6.1f} "
              f"{mae:10.5f} {rmse:10.5f}", flush=True)

print(f"\n{'precision':>9s} {'R2':>17s} {'AE':>13s} {'MAE (Ah)':>19s}")
summary = {}
for label in ("fp32", "int8", "int4"):
    e = rows.get(label)
    if not e:
        continue
    r2 = np.array([x["R2"] for x in e])
    ae = np.array([x["AE"] for x in e])
    ma = np.array([x["MAE_Ah"] for x in e])
    summary[label] = {"R2_mean": float(r2.mean()), "R2_std": float(r2.std(ddof=1)),
                      "AE_mean": float(ae.mean()), "AE_std": float(ae.std(ddof=1)),
                      "MAE_Ah_mean": float(ma.mean()),
                      "MAE_Ah_std": float(ma.std(ddof=1)), "n": len(e)}
    print(f"{label:>9s} {r2.mean():9.4f}+-{r2.std(ddof=1):.4f} "
          f"{ae.mean():6.1f}+-{ae.std(ddof=1):.3f} "
          f"{ma.mean():9.5f}+-{ma.std(ddof=1):.5f}")

fp = summary.get("fp32", {}).get("MAE_Ah_mean")
for label in ("int8", "int4"):
    if fp and label in summary:
        d = summary[label]["MAE_Ah_mean"] / fp - 1
        print(f"\n{label} vs fp32: MAE {d * 100:+.1f}%   "
              f"R2 {summary[label]['R2_mean'] - summary['fp32']['R2_mean']:+.4f}")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
io.open(args.out, "w", encoding="utf-8").write(
    json.dumps({"dataset": args.dataset, "sp": args.sp, "seeds": SEEDS,
                "range_Ah": span, "per_seed": rows, "summary": summary},
               indent=1))
print(f"\nwrote {args.out}")
