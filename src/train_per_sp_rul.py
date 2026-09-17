"""Per-SP trainer with a direct RUL head -- the only protocol here that
produces a real RUL number.

Why this exists.  Every other RUL number in this repo is read off a predicted
capacity *trajectory*: `true_rul(predicted_sequence, threshold)`.  That
trajectory is assembled from one-step predictions whose input windows are true
measurements, and the block-wise K-step variant re-anchors to measurements every
K cycles -- so the crossing point is computed with data from AFTER the launch
cycle.  It is not a RUL forecast, and the AE it yields is not a RUL error.
(The autoregressive variant IS a real forecast structurally, but all three
models -- ours, PatchFormer, RUL-Mamba -- fail under it; see
docs/ar_freeze_findings.md.)

`gdn_model.GDNBatteryModel.head_rul` (line 279) reads `fused[:, -1, :]` -- the
last timestep of the trunk, which depends only on the input window -- and emits
one scalar.  That is a genuine one-shot RUL: no rollout, no feedback, no future
data.  It has never been trained (`return_rul=True` has no caller anywhere in
the tree), so this file trains it.

Objective:
    loss = masked_mae(pred_cap, (y - wmean)/wstd)      unchanged, the main task
         + LAM_RUL * masked_mae(pred_rul, TRUL/scale)  the new head
The RUL target is the distance from the window's target cycle to the cell's true
EOL crossing (the last cycle at or above the threshold -- the repo's convention,
`eval_multiseed.true_rul`), normalized by the longest life among the TRAINING
cells so it lands in [0, 1].  Windows at or past the crossing have no remaining
life and are masked out of the RUL term; the capacity term keeps every window.

Evaluation: one forward pass on the window ending at SP gives PRUL directly.
    AE = |TRUL - PRUL| in cycles, no trajectory anywhere.
Both losses are printed every epoch -- a weight that is never seen is a weight
that silently does nothing (that mistake has been made twice in this repo).

Outputs (new paths; nothing existing is touched):
  checkpoints/per_sp/{ds}/SP{sp}_seed{seed}_rul.pt
  results/per_sp_train_rul.json

    D:/anaconda/envs/py312/python.exe src/train_per_sp_rul.py --dataset calce \
        --seeds 1 --start-seed 1 --sps 500
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

from gdn_model import build_gdn_model, masked_mae       # noqa: E402
from make_figures import load_series                    # noqa: E402
from train_per_sp import BATCH, EPOCHS, EPS, DEV, build_windows  # noqa: E402

LAM_RUL = float(os.environ.get("LAM_RUL", "1.0"))


def cross_index(raw, eol_ah):
    """0-based index of the last cycle at or above the threshold, or len(raw).

    Same rule as eval_multiseed.true_rul: `seq[i] >= th > seq[i+1]`.
    """
    below = np.asarray(raw) < eol_ah
    return int(np.argmax(below)) - 1 if below.any() else len(raw)


def build_rul(caps, cells, W, eol_ah, scale, max_cycle=None):
    """Per-window normalized remaining life + validity mask, aligned with
    build_windows' output order (window ending at cycle i -> target cycle i)."""
    R, M = [], []
    for c in cells:
        raw = np.asarray(caps[c], dtype=np.float64)
        cross = cross_index(raw, eol_ah)
        end = len(raw) if max_cycle is None else min(len(raw), max_cycle)
        for i in range(W, end):
            rul = cross - i
            R.append(rul / scale if rul >= 0 else 0.0)
            M.append(1.0 if rul >= 0 else 0.0)
    return np.asarray(R, dtype=np.float32), np.asarray(M, dtype=np.float32)


def train_rul(seed, X, Y, RUL, MR, W, progress_every=5):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_gdn_model(
        multiscale=True, stage_query=True, input_dim=1, window_size=W,
        output_len=1, readout="last").to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    N = len(X)
    t0w = time.time()
    last_t, last_ep = t0w, 0
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(N)
        sc_, sr_, nb = 0.0, 0.0, 0
        for s in range(0, N, BATCH):
            idx = perm[s:s + BATCH]
            x = torch.tensor(X[idx]).to(DEV)
            y = torch.tensor(Y[idx]).to(DEV)
            rt = torch.tensor(RUL[idx]).to(DEV)
            rm = torch.tensor(MR[idx]).to(DEV)
            opt.zero_grad()
            out_cap, out_rul = model(x, return_rul=True)
            wmean = x[:, :, 0].mean(dim=1)
            wstd = x[:, :, 0].std(dim=1) + EPS
            l_cap = masked_mae(out_cap.squeeze(-1), (y - wmean) / wstd,
                               torch.ones_like(y))
            # masked mean; clamp keeps a batch with no valid RUL from dividing
            # by zero (windows past the crossing are excluded)
            l_rul = ((out_rul.squeeze(-1) - rt).abs() * rm).sum() \
                / rm.sum().clamp(min=1.0)
            (l_cap + LAM_RUL * l_rul).backward()
            opt.step()
            sc_ += float(l_cap.detach())
            sr_ += float(l_rul.detach())
            nb += 1
        if progress_every and (ep % progress_every == 0 or ep == EPOCHS - 1):
            now = time.time()
            per_ep = (now - last_t) / max(1, ep + 1 - last_ep)
            print(f"    [ep {ep + 1:>3}/{EPOCHS}] cap={sc_ / nb:.5f} "
                  f"rul={sr_ / nb:.5f} (x{LAM_RUL})  {now - t0w:.0f}s, "
                  f"{per_ep:.1f}s/ep", flush=True)
            last_t, last_ep = now, ep + 1
    return model


def eval_rul_head(model, caps, test_cells, lo, hi, W, sp, eol_ah, scale):
    """One forward pass at SP -> PRUL.  No trajectory, no future data.

    eval() is not optional: train_rul leaves the model in train() mode, so
    without this the measurement runs with dropout active and moves between
    calls (observed 190.3 vs 181.8 for the same checkpoint and window).
    """
    model.eval()
    with torch.no_grad():
        tc = test_cells[0]
        raw = np.asarray(caps[tc], dtype=np.float64)
        cross = cross_index(raw, eol_ah)
        # the window seq[sp-W:sp] predicts cycle index sp, and the repo's TRUL
        # at SP is (crossing index - sp): CALCE SP500 gives 639 - 500 = 139,
        # matching per_sp_train.json's TRUL=139
        trul = cross - sp
        seq = (caps[tc] - lo) / (hi - lo + EPS)
        win = seq[sp - W:sp]
        cin = torch.tensor(win[None, :, None], dtype=torch.float32).to(DEV)
        _, out_rul = model(cin, return_rul=True)
        prul = float(out_rul.item()) * scale
    return {"SP": int(sp), "TRUL": int(trul), "PRUL": round(prul, 1),
            "AE": round(abs(trul - prul), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--start-seed", type=int, default=1)
    ap.add_argument("--sps", type=int, nargs="+", default=None)
    ap.add_argument("--progress-every", type=int, default=5)
    args = ap.parse_args()

    ds = args.dataset
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
    # normalize RUL by the longest life among the TRAINING cells, so the head
    # outputs something in [0, 1] on every dataset
    scale = float(max(cross_index(caps[c], eol_ah) for c in train_cells))

    out_dir = os.path.join("..", "checkpoints", "per_sp", ds)
    os.makedirs(out_dir, exist_ok=True)
    out_path = "results/per_sp_train_rul.json"
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}

    print(f"=== {ds} RUL head  seeds {args.start_seed}.."
          f"{args.start_seed + args.seeds - 1}  W={W}  eol={eol_ah} Ah"
          f"  rul_scale={scale:.0f} cycles  LAM_RUL={LAM_RUL} ===", flush=True)
    for sp in sps:
        Xtr, Ytr = build_windows(caps, train_cells, lo, hi, W)
        Xts, Yts = build_windows(caps, test_cells, lo, hi, W, max_cycle=sp)
        Rtr, Mtr = build_rul(caps, train_cells, W, eol_ah, scale)
        Rts, Mts = build_rul(caps, test_cells, W, eol_ah, scale, max_cycle=sp)
        X, Y = np.vstack([Xtr, Xts]), np.concatenate([Ytr, Yts])
        RUL, MR = np.concatenate([Rtr, Rts]), np.concatenate([Mtr, Mts])
        print(f"  SP{sp}: {len(X)} windows, {int(MR.sum())} with a valid RUL "
              f"({100 * MR.mean():.0f}%)", flush=True)
        for seed in range(args.start_seed, args.start_seed + args.seeds):
            if str(seed) in out.setdefault(ds, {}).setdefault(str(sp), {}):
                print(f"    seed{seed}: SKIP (already in json)", flush=True)
                continue
            t0 = time.time()
            model = train_rul(seed, X, Y, RUL, MR, W,
                              progress_every=args.progress_every)
            torch.save({"state_dict": model.state_dict(), "seed": seed,
                        "lo": lo, "hi": hi, "W": W, "sp": sp, "eol_ah": eol_ah,
                        "rul_scale": scale, "test_cells": test_cells,
                        "train_cells": train_cells},
                       os.path.join(out_dir, f"SP{sp}_seed{seed}_rul.pt"))
            row = eval_rul_head(model, caps, test_cells, lo, hi, W, sp,
                                eol_ah, scale)
            out[ds][str(sp)][str(seed)] = row
            json.dump(out, open(out_path, "w"), indent=2)
            print(f"    SP{sp} seed{seed}: TRUL={row['TRUL']} "
                  f"PRUL={row['PRUL']} AE={row['AE']} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
