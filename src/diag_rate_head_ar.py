"""Does the monotone rate head survive a SELF-FED rollout?

The claim under test is structural: `readout="phys_ir"` computes
`Q_hat = Q_last - softplus(...)`, so the trajectory can only fall, and the
quantity the model has to get right is the per-step rate (~7e-4) rather than
the window-relative offset our other decodes need (~-2.2e-2, 32x larger).  If
that is what makes it rollout-stable, it should show up here as a sane
declining trajectory instead of the plunge-to--0.16 or the 6x-too-slow drift
that the z-score and anchor decodes produce.

WHY THIS CHECKPOINT AND NOT THE PAPER'S:
    ../checkpoints/phys_figs_models.pt holds `rate_clean` (phys_ir) and
    `free_clean` (direct_z) trained by the SAME loop on the SAME data with
    only the readout changed -- a controlled pair, which is what a rollout
    comparison needs.  They are multiscale=False single-scale models, so this
    measures the HEAD's dynamics, not the paper model's.  Do not quote these
    numbers as the paper model's.

Self-fed means only the capacity channel is fed back; IR stays real, matching
make_figures_phys.predict, which treats IR as an exogenous measurement (a
deployed system measures it, it does not predict it).  The free head takes
capacity only, so it gets no such help.

Read-only: loads a checkpoint, prints.  Writes nothing.

    cd src && D:/anaconda/envs/py312/python.exe diag_rate_head_ar.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from gdn_model import build_gdn_model                    # noqa: E402
from make_figures_phys import EPS, IR_IDX, W, load_data  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = "../checkpoints/phys_figs_models.pt"
STEPS = 400


def build(kind, state):
    model = build_gdn_model(
        multiscale=False, input_dim=2 if kind == "phys_ir" else 1,
        window_size=W, output_len=1,
        readout="phys_ir" if kind == "phys_ir" else "last",
    ).to(DEV)
    model.load_state_dict(state)
    model.eval()
    return model


def rollout(model, kind, tc, ti, start, steps):
    """Feed the model its own capacity output; IR stays the real series.

    z-score decode is de-normalised per window, exactly as
    make_figures_phys.predict does -- but on the MODEL'S OWN window, which is
    the whole difference between that function and this one.
    """
    win = [float(v) for v in tc[start - W:start]]
    out = []
    with torch.no_grad():
        for k in range(steps):
            cyc = start + k
            ir = ti[cyc - W:cyc] if cyc <= len(ti) else ti[-W:]
            if kind == "phys_ir":
                w = np.stack([np.asarray(win[-W:], dtype=np.float32),
                              ir.astype(np.float32)], axis=1)
                cin = torch.tensor(w).unsqueeze(0).to(DEV)
                y = float(model(cin).item())          # already absolute
            else:
                x = np.asarray(win[-W:], dtype=np.float32)
                cin = torch.tensor(x[:, None]).unsqueeze(0).to(DEV)
                wm, ws = float(x.mean()), float(x.std()) + EPS
                y = float(model(cin).item()) * ws + wm
            out.append(y)
            win.append(y)
    return np.asarray(out)


def first_cross(series, thr):
    for i in range(len(series) - 1):
        if series[i] >= thr > series[i + 1]:
            return i
    return -1


def main() -> None:
    caps, feats, tr, tc, lc, hc, li, hi, eol = load_data()
    tc_n = (caps[tc] - lc) / (hc - lc + EPS)
    ti_n = (feats[tc][:, IR_IDX] - li) / (hi - li + EPS)
    thr = (eol - lc) / (hc - lc + EPS)
    true_eol = first_cross(tc_n, thr) + 1

    sd = torch.load(CKPT, map_location=DEV, weights_only=False)
    print("=" * 96)
    print("  CALCE  self-fed rollout   test=%s  W=%d  EOL=%d  thr=%.4f"
          % (tc, W, true_eol, thr))
    print("  NOTE single-scale models (multiscale=False) -- head dynamics, "
          "not the paper model")
    print("=" * 96)

    for start in (int(true_eol * 0.5), int(true_eol * 0.7)):
        print("\n  --- launch @ cycle %d   (true remaining life %d) ---"
              % (start, true_eol - start))
        print("  %-12s %9s %9s %9s %9s %8s %8s"
              % ("head", "start", "step50", "step200", "final", "crossed",
                 "AE"))
        for name, kind in (("free_clean", "direct_z"),
                           ("rate_clean", "phys_ir")):
            model = build(kind, sd[name])
            full = rollout(model, kind, tc_n, ti_n, start, STEPS)
            xi = first_cross(full, thr)
            if xi < 0:
                ae_lbl = ">=%d" % ((start + STEPS - 1) - true_eol)
            else:
                ae_lbl = str(abs(start + xi + 1 - true_eol))
            truth_end = tc_n[min(start + STEPS, len(tc_n) - 1)]
            print("  %-12s %9.4f %9.4f %9.4f %9.4f %8s %8s"
                  % (name, full[0], full[min(50, len(full) - 1)],
                     full[min(200, len(full) - 1)], full[-1],
                     "Y" if xi >= 0 else "n", ae_lbl))
            print("      truth over the same span: %.4f -> %.4f   (final "
                  "window std %.6f)" % (tc_n[start], truth_end,
                                        float(np.std(full[-W:]))))
            print("      per-step change: first10 %+.6f  last10 %+.6f"
                  % (np.mean(np.diff(full[:11])),
                     np.mean(np.diff(full[-11:]))))
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
