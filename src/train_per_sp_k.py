"""K-step per-SP trainer -- the multi-horizon counterpart of train_per_sp.py.

Same protocol, same architecture as the current main line (`multiscale=True,
stage_query=True, readout="last"`), same per-window z-score target.  The only
difference is `output_len=K`: one forward pass emits K steps instead of one.

Why a new file: every existing K-step script is a SUPERSEDED architecture.
test_recompute_ae_k32.py:49, test_k32_vs_pf.py:41,44, test_rul_compare.py and
test_rul_rollout_diag.py all build `cross_exchange=True` and load
`checkpoints/unified_*_K32*.pt`, whose weight keys are `cross.*` / `rope.*`.
Nothing in the current line (`stage_query=True`, `cross_stage.*`) has a K-step
trainer at all -- train_per_sp.py hard-codes output_len=1 -- so there is no way
to produce a K-step checkpoint that the current evaluators can read.

Two further problems in the old K=32 path, both avoided here:

  * its evaluation was RECURSIVE -- test_unified_train.eval_rul does
    `window = concat(window[K:], pred)`, feeding predictions back -- while its
    K=1 numbers came from a non-recursive path.  The published "K=1 vs K=32"
    comparison therefore mixed two protocols.
  * the true EOL used `argmax(raw_cap < eol)` (first cycle BELOW the threshold)
    while the predicted crossing used `pred[j] >= th > pred[j+1]` (last cycle
    at/above), a built-in ~1-cycle offset in AE.

Protocol here (identical to train_per_sp.py except for K):

  train  = every other cell in full + the test cell's cycles < SP
  eval   = NON-RECURSIVE, block-wise.  From cycle i, with a TRUE window, one
           forward gives cycles i..i+K-1; the next block starts at i+K, again
           from a true window.  Nothing is fed back, so this is the same family
           as Table A and as PatchFormer/RUL-Mamba's single `predict()` call.
  AE     = |TRUL - PRUL| in cycles, both via the repo's `true_rul` (last cycle
           at or above the threshold) so the two sides share a convention.
  metrics= MSE and AE.  No crossing rate: it carries one bit and hides how
           large the error is.

Outputs (new paths; nothing existing is touched):
  checkpoints/per_sp/{ds}/SP{sp}_seed{seed}_K{K}.pt
  results/per_sp_train_K{K}.json

    D:/anaconda/envs/py312/python.exe src/train_per_sp_k.py --dataset calce --k 32
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

from eval_multiseed import true_rul                 # noqa: E402
from gdn_model import build_gdn_model               # noqa: E402
from make_figures import load_series                # noqa: E402
from train_per_sp import (BATCH, EPOCHS, EPS, DEV,   # noqa: E402
                          build_windows)


def train_k(seed, X, XK, KM, W, K, progress_every=5):
    """Train one K-step model.

    Target is the per-window z-score for EACH of the K steps, normalized by the
    input window's own mean and std -- the same convention train_per_sp.py uses
    for its single step, extended to the K outputs.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_gdn_model(
        multiscale=True, stage_query=True, input_dim=1, window_size=W,
        output_len=K, readout="last").to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    N = len(X)
    t0w = time.time()
    last_t, last_ep = t0w, 0
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(N)
        tot, nb = 0.0, 0
        for s in range(0, N, BATCH):
            idx = perm[s:s + BATCH]
            x = torch.tensor(X[idx]).to(DEV)          # (B, W, 1)
            xk = torch.tensor(XK[idx]).to(DEV)        # (B, K) future truths
            km = torch.tensor(KM[idx]).to(DEV)        # (B, K) 1 where they exist
            opt.zero_grad()
            pred = model(x)                           # (B, K)
            wmean = x[:, :, 0].mean(dim=1, keepdim=True)
            wstd = x[:, :, 0].std(dim=1, keepdim=True) + EPS
            tgt = (xk - wmean) / wstd
            loss = (torch.abs(pred - tgt) * km).sum() / km.sum().clamp(min=1.0)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        if progress_every and (ep % progress_every == 0 or ep == EPOCHS - 1):
            now = time.time()
            per_ep = (now - last_t) / max(1, ep + 1 - last_ep)
            print(f"    [ep {ep + 1:>3}/{EPOCHS}] loss={tot / nb:.5f}"
                  f"  {now - t0w:.0f}s, {per_ep:.1f}s/ep", flush=True)
            last_t, last_ep = now, ep + 1
    return model


def eval_sp_k(model, caps, test_cells, lo, hi, W, K, sp, eol_ah):
    """Non-recursive block-wise trajectory, then MSE and AE.

    From cycle `sp`, take the TRUE window ending at `sp` and emit K steps; the
    next block starts at `sp + K` and again takes a true window.  Nothing is fed
    back, so the blocks tile the timeline and the result is one value per cycle.
    """
    th = (eol_ah - lo) / (hi - lo + EPS)
    rows = []
    with torch.no_grad():
        for tc in test_cells:
            seq = (caps[tc] - lo) / (hi - lo + EPS)
            blocks, i = [], sp
            while i < len(seq):
                win = seq[i - W:i]
                wm = float(win.mean())
                ws = float(win.std()) + EPS
                cin = torch.tensor(win[None, :, None], dtype=torch.float32)
                blocks.append(model(cin.to(DEV)).squeeze(0).cpu().numpy()
                              * ws + wm)
                i += K
            seg_p = np.concatenate(blocks)[:len(seq) - sp]
            tv = seq[sp:sp + len(seg_p)]
            # both sides through true_rul, so the crossing convention matches
            trul, prul = true_rul(tv, th), true_rul(seg_p, th)
            rows.append({
                "TRUL": int(trul), "PRUL": int(prul),
                "AE": abs(int(trul) - int(prul)),
                "MSE": float(np.mean((tv - seg_p) ** 2)),
                "RMSE": float(np.sqrt(np.mean((tv - seg_p) ** 2))),
                "MAE": float(np.mean(np.abs(tv - seg_p))),
                "R2": float(1 - np.sum((tv - seg_p) ** 2)
                            / (np.sum((tv - tv.mean()) ** 2) + EPS)),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--k", type=int, default=32, help="prediction horizon")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--start-seed", type=int, default=1)
    ap.add_argument("--sps", type=int, nargs="+", default=None)
    ap.add_argument("--progress-every", type=int, default=5)
    args = ap.parse_args()

    ds, K = args.dataset, args.k
    caps, train_cells, test_cell, W, sps, eol_ah = load_series(ds)
    if args.sps:
        sps = args.sps
    caps = {c: caps[c].astype(np.float32) for c in caps}
    # load_series('mit') returns MIT_TEST_CELLS[0] only, while the protocol
    # evaluates on both; the other per-SP scripts hard-code this, mirror it
    test_cells = [test_cell]
    if ds == "mit":
        from load_datasets import MIT_TEST_CELLS
        test_cells = list(MIT_TEST_CELLS)
    all_tr = np.concatenate([caps[c] for c in train_cells])
    lo, hi = float(all_tr.min()), float(all_tr.max())

    out_dir = os.path.join("..", "checkpoints", "per_sp", ds)
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"results/per_sp_train_K{K}.json"
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}

    print(f"=== {ds} K={K}  seeds {args.start_seed}.."
          f"{args.start_seed + args.seeds - 1}  SPs {sps}  W={W} ===",
          flush=True)
    for sp in sps:
        # identical split to train_per_sp.py.  XK/KM carry each window's next K
        # true values, capped at the same limit the windows obey, so the test
        # cell's windows never reach a held-out cycle.
        Xtr, _Ytr, XKtr, KMtr = build_windows(caps, train_cells, lo, hi, W,
                                              with_future=True, kmax=K)
        Xts, _Yts, XKts, KMts = build_windows(caps, test_cells, lo, hi, W,
                                              max_cycle=sp, with_future=True,
                                              kmax=K)
        X = np.vstack([Xtr, Xts])
        XK = np.vstack([XKtr, XKts])
        KM = np.vstack([KMtr, KMts])
        for seed in range(args.start_seed, args.start_seed + args.seeds):
            if str(seed) in out.setdefault(ds, {}).setdefault(str(sp), {}):
                print(f"  SP{sp} seed{seed}: SKIP (already in json)", flush=True)
                continue
            t0 = time.time()
            model = train_k(seed, X, XK, KM, W, K,
                            progress_every=args.progress_every)
            torch.save({"state_dict": model.state_dict(), "seed": seed,
                        "lo": lo, "hi": hi, "W": W, "sp": sp, "K": K,
                        "eol_ah": eol_ah, "test_cells": test_cells,
                        "train_cells": train_cells},
                       os.path.join(out_dir, f"SP{sp}_seed{seed}_K{K}.pt"))
            rows = eval_sp_k(model, caps, test_cells, lo, hi, W, K, sp, eol_ah)
            out[ds][str(sp)][str(seed)] = rows
            json.dump(out, open(out_path, "w"), indent=2)
            g = lambda f: float(np.mean([r[f] for r in rows]))  # noqa: E731
            print(f"  SP{sp} seed{seed}: MSE={g('MSE'):.3e} "
                  f"MAE={g('MAE'):.5f} AE={g('AE'):.1f} "
                  f"R2={g('R2'):.4f} [{time.time() - t0:.0f}s]", flush=True)
            del model
            torch.cuda.empty_cache()

    print("=== per-SP means over seeds ===", flush=True)
    for sp in sps:
        rs = [r for s in out[ds].get(str(sp), {})
              for r in out[ds][str(sp)][s]]
        if rs:
            print(f"  SP{sp}: MSE={np.mean([r['MSE'] for r in rs]):.3e} "
                  f"MAE={np.mean([r['MAE'] for r in rs]):.5f} "
                  f"AE={np.mean([r['AE'] for r in rs]):.2f}", flush=True)


if __name__ == "__main__":
    main()
