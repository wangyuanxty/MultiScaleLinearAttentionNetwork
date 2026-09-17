"""Does the Yao et al. recipe actually roll out on OUR data?

Yao, Zhao & Kowal (J. Power Sources 660 (2025) 238569) get a recursive SOH
rollout working on CALCE while their baseline model fails outright.  Their
data is CALCE too -- the CX2 series (1.35 Ah); ours is the CS2 series
(1.13 Ah) -- so the question this answers is sharp:

    is recursive rollout blocked by OUR MODEL, or by something about the task?

The recipe, taken from their Section 2.1.2 and 3.2, is small enough to
reimplement exactly:

    * GRU encoder over the observed SOH history -> context vector
    * GRU decoder initialised from it; at each step the input is the previous
      SOH, predicted or ground truth with probability p (teacher forcing /
      scheduled sampling, Bengio et al. 2015).  p = 0.5.
    * hidden size 64, one layer each
    * NO normalisation: SOH already lives in [0, 1]

That is deliberately NOT our architecture: no GDN-2, no sliding window, no
multi-scale, no window-relative decode.  If this rolls out on CS2_35 and our
model does not, the gap is architectural and the fix is known.  If this also
fails, the gap is in the data or the protocol and no amount of decoder
tinkering will help.

Read-only w.r.t. the repo's checkpoints: writes only results/yao_baseline.json.

    cd src && D:/anaconda/envs/py312/python.exe yao_baseline.py --sp 300
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from make_figures import load_series                     # noqa: E402
from train_per_sp import DEV, EPS                        # noqa: E402

HID = 64          # Yao 3.2: "a hidden size of a moderate 64 ... layers of 1"
P_TF = 0.5        # Yao 3.2: "The ratio for teacher forcing is set to 0.5"
EPOCHS = 400
LR = 1e-3
SEED = 42
# Training rollouts are capped at this horizon.  The decoder is a recurrence --
# what it has to learn is the per-step dynamics, and the gradient through 600
# sequential steps costs 600 sequential kernel launches for no extra signal.
# Evaluation still rolls to the end of the series, uncapped.
MAX_H = 150


class EncDec(nn.Module):
    """GRU encoder -> context -> GRU decoder, per Yao Section 2.1.2."""

    def __init__(self, hid: int = HID):
        super().__init__()
        self.enc = nn.GRU(1, hid, batch_first=True)
        self.dec = nn.GRU(1, hid, batch_first=True)
        self.fc = nn.Linear(hid, 1)

    def forward(self, hist, n_future, target=None, p=0.0):
        """hist (B, L, 1) observed SOH; returns (B, n_future) rolled out.

        `target` supplies the ground truth used by the (1-p) branch of teacher
        forcing; without it the rollout is fully self-fed, which is exactly
        what inference does (Yao: "teacher forcing will be disabled").
        """
        _, h = self.enc(hist)
        inp = hist[:, -1:, :]              # (B, 1, 1) first input = last SOH
        outs = []
        for k in range(n_future):
            o, h = self.dec(inp, h)                        # o (B, 1, H)
            y = self.fc(o).reshape(-1)                     # (B,)
            outs.append(y)
            if target is not None and p < 1.0:
                # teacher forcing: ground truth with prob (1-p), else own
                # prediction.  The fed-back value is detached either way, so
                # each step's gradient comes only from that step's forward.
                use_true = torch.rand_like(y) >= p
                nxt = torch.where(use_true, target[:, k], y.detach())
            else:
                nxt = y.detach()                           # inference: self-fed
            inp = nxt.reshape(-1, 1, 1)                    # (B, 1, 1)
        return torch.stack(outs, dim=1)                    # (B, n_future)


def first_cross(series, thr) -> int:
    """First index i where series[i] >= thr > series[i+1]; -1 if never."""
    for i in range(len(series) - 1):
        if series[i] >= thr > series[i + 1]:
            return i
    return -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="calce")
    ap.add_argument("--sp", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    caps, train_cells, test_cell, _W, _sps, eol_ah = load_series(args.dataset)
    test_cells = [test_cell] if isinstance(test_cell, str) else list(test_cell)
    allc = np.concatenate([caps[c] for c in train_cells])
    lo, hi = float(allc.min()), float(allc.max())
    thr = (eol_ah - lo) / (hi - lo + EPS)

    def norm(c):
        return (np.asarray(caps[c], dtype=np.float64) - lo) / (hi - lo + EPS)

    print("=" * 92)
    print("  Yao et al. recipe reimplemented on %s (ours = CS2, theirs = CX2)"
          % args.dataset.upper())
    print("  train=%s  test=%s  lo=%.4f hi=%.4f  EOL(norm)=%.4f"
          % (train_cells, test_cells, lo, hi, thr))
    print("=" * 92, flush=True)

    tr = [norm(c) for c in train_cells]
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = EncDec().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # Each sample: encode a prefix of length L, roll out the remainder.  L
    # sweeps the whole range so the decoder meets every horizon, not just one.
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        losses, nb = 0.0, 0
        for s in tr:
            for L in range(50, len(s) - 2, 50):
                hist = torch.tensor(s[:L], dtype=torch.float32,
                                    device=DEV)[None, :, None]
                tgt = torch.tensor(s[L:], dtype=torch.float32, device=DEV)[None]
                if tgt.shape[1] < 2:
                    continue
                n_fut = min(tgt.shape[1], MAX_H)
                opt.zero_grad()
                out = model(hist, n_fut, target=tgt[:, :n_fut], p=P_TF)
                loss = (out - tgt[:, :n_fut]).abs().mean()
                loss.backward()
                opt.step()
                losses += float(loss)
                nb += 1
        if ep % 25 == 0 or ep == args.epochs - 1:
            print("    [ep %4d/%d] loss=%.5f  %.0fs"
                  % (ep + 1, args.epochs, losses / max(nb, 1),
                     time.time() - t0), flush=True)

    # ---- evaluation: the SAME recursive protocol our ar_probe uses ----------
    print()
    print("  recursive rollout from cycle %d (true history only, then self-fed)"
          % args.sp)
    print("  %-10s %8s %8s %9s %9s %8s %7s"
          % ("cell", "launch", "thr", "final", "trueEOL", "crossed", "AE"))
    out_rows = []
    model.eval()
    for tc in test_cells:
        s = norm(tc)
        eol = first_cross(s, thr) + 1
        if eol <= 0 or args.sp >= len(s):
            print("  %-10s skipped (no EOL or SP beyond series)" % tc)
            continue
        with torch.no_grad():
            hist = torch.tensor(s[:args.sp], dtype=torch.float32,
                                device=DEV)[None, :, None]
            n_fut = len(s) - args.sp
            full = model(hist, n_fut).cpu().numpy().ravel()
        xi = first_cross(np.concatenate([[s[args.sp - 1]], full]), thr)
        if xi < 0:
            ae_lbl = ">=%d" % (len(s) - 1 - eol)
        else:
            ae_lbl = "%d" % abs((args.sp + xi) - eol)
        # MAE over the span that has truth
        mae = float(np.mean(np.abs(full - s[args.sp:])))
        print("  %-10s %8d %8.4f %9.4f %9d %8s %7s"
              % (tc, args.sp, thr, float(full[-1]), eol,
                 "Y" if xi >= 0 else "n", ae_lbl))
        print("      MAE over the rollout %.5f   truth goes %.4f -> %.4f"
              % (mae, s[args.sp], s[-1]))
        out_rows.append(dict(cell=tc, launch=args.sp, thr=float(thr),
                             final=float(full[-1]), true_eol=int(eol),
                             crossed=bool(xi >= 0), ae=ae_lbl, mae=mae))
    res = dict(dataset=args.dataset, sp=args.sp, train_cells=list(train_cells),
               test_cells=test_cells, lo=lo, hi=hi, p_tf=P_TF, hid=HID,
               epochs=args.epochs, rows=out_rows)
    with open("results/yao_baseline.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\n  -> results/yao_baseline.json")


if __name__ == "__main__":
    main()
