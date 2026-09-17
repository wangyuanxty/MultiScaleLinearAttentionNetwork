"""Long-context per-SP trainer -- the same protocol at a longer input window.

Why this is the right shape for "carry the state across calls".  GDN-2 is a
linear recurrence, so feeding [history + predictions] in one pass and carrying
S across calls are the SAME computation: the state after k values is the state
after k values, however you got there.  Carrying the state is therefore not a
new information channel -- it is exactly the growing window, which was measured
and failed (PANASONIC k=50: MAE 0.0326 -> 0.1086).

But that failure was attributed to LENGTH EXTRAPOLATION: the model had only ever
seen 64-cycle inputs and the growing window reached 2.5-4x that.  Feeding a long
context at INFERENCE while training at 64 is the confounded experiment.
Training at the long context removes the confound, and it needs no architecture
change:

  * there is no positional encoding in gdn_model.py / gdn_v2.py -- grep for
    rope / positional / theta / pos_emb comes back empty
  * `window_size` is stored and used by no layer (only the __init__ signature)
  * verified: forward() accepts L = 64/128/256/512 and returns (B, 1) for all

So this is a thin wrapper: build_windows / train_one / eval_sp are imported from
train_per_sp.py verbatim, and only W differs.

Protocol: identical to train_per_sp.py, including the per-SP split and the
per-window z-score target.  The ONLY difference is the input length, so the
comparison against the W=64 checkpoints is a clean ablation of context length.

Outputs (new paths; nothing existing is touched):
  checkpoints/per_sp/{ds}/SP{sp}_seed{seed}_W{W}.pt
  results/per_sp_train_W{W}.json

    D:/anaconda/envs/py312/python.exe src/train_per_sp_w.py \
        --dataset calce --w 256 --seeds 1 --start-seed 1 --sps 500
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from make_figures import load_series                          # noqa: E402
from train_per_sp import (build_windows, train_one, eval_sp)  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--w", type=int, default=256,
                    help="input window length (dataset default is 64 or 30)")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--start-seed", type=int, default=1)
    ap.add_argument("--sps", type=int, nargs="+", default=None)
    args = ap.parse_args()

    ds = args.dataset
    caps, train_cells, test_cell, W_ds, sps, eol_ah = load_series(ds)
    if args.sps:
        sps = args.sps
    W = args.w
    caps = {c: caps[c].astype(np.float32) for c in caps}
    # load_series('mit') returns MIT_TEST_CELLS[0] only, while the protocol
    # evaluates on both; train_per_sp.py, eval_multiseed.py and
    # continue_per_sp_200.py each hard-code this, so mirror it here
    test_cells = [test_cell]
    if ds == "mit":
        from load_datasets import MIT_TEST_CELLS
        test_cells = list(MIT_TEST_CELLS)
    all_tr = np.concatenate([caps[c] for c in train_cells])
    lo, hi = float(all_tr.min()), float(all_tr.max())

    # the test cell only contributes cycles < SP, so an SP at or below W would
    # leave its windows with no usable context
    sps = [sp for sp in sps if sp > W]
    if not sps:
        raise SystemExit(f"every SP is <= W={W}; nothing to train")

    out_dir = os.path.join("..", "checkpoints", "per_sp", ds)
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"results/per_sp_train_W{W}.json"
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}

    print(f"=== {ds} W={W} (dataset default {W_ds})  seeds "
          f"{args.start_seed}..{args.start_seed + args.seeds - 1}  "
          f"SPs {sps} ===", flush=True)
    for sp in sps:
        Xtr, Ytr = build_windows(caps, train_cells, lo, hi, W)
        Xts, Yts = build_windows(caps, test_cells, lo, hi, W, max_cycle=sp)
        X = np.vstack([Xtr, Xts])
        Y = np.concatenate([Ytr, Yts])
        print(f"  SP{sp}: {len(X)} windows of {W} cycles "
              f"({len(Xtr)} from train cells + {len(Xts)} test-cell pre-SP)",
              flush=True)
        for seed in range(args.start_seed, args.start_seed + args.seeds):
            if str(seed) in out.setdefault(ds, {}).setdefault(str(sp), {}):
                print(f"    seed{seed}: SKIP (already in json)", flush=True)
                continue
            t0 = time.time()
            # train_one prints its own per-epoch progress.  save_prefix turns on
            # its PROG_SAVE checkpoints, so a 100-epoch run at a long W can be
            # evaluated part-way -- the question here is whether the rollout
            # improves, and that is answerable from a partly-trained model.
            ck_name = f"SP{sp}_seed{seed}_W{W}"
            model = train_one(
                seed, X, Y, W,
                save_prefix=os.path.join(out_dir, ck_name),
                save_meta=dict(lo=lo, hi=hi, sp=sp, eol_ah=eol_ah,
                               test_cells=list(test_cells),
                               train_cells=list(train_cells)))
            torch.save({"state_dict": model.state_dict(), "seed": seed,
                        "lo": lo, "hi": hi, "W": W, "sp": sp, "eol_ah": eol_ah,
                        "test_cells": test_cells, "train_cells": train_cells},
                       os.path.join(out_dir, f"SP{sp}_seed{seed}_W{W}.pt"))
            rows = eval_sp(model, caps, test_cells, lo, hi, W, sp, eol_ah)
            out[ds][str(sp)][str(seed)] = rows
            json.dump(out, open(out_path, "w"), indent=2)
            g = lambda f: float(np.mean([r[f] for r in rows]))  # noqa: E731
            print(f"    SP{sp} seed{seed}: MAE={g('MAE'):.5f} "
                  f"RMSE={g('RMSE'):.5f} R2={g('R2'):.4f} AE={g('AE'):.1f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
            del model
            torch.cuda.empty_cache()

    print("=== per-SP means over seeds ===", flush=True)
    for sp in sps:
        rs = [r for s in out[ds].get(str(sp), {})
              for r in out[ds][str(sp)][s]]
        if rs:
            print(f"  SP{sp}: MAE={np.mean([r['MAE'] for r in rs]):.5f} "
                  f"R2={np.mean([r['R2'] for r in rs]):.4f} "
                  f"AE={np.mean([r['AE'] for r in rs]):.2f}", flush=True)


if __name__ == "__main__":
    main()
