"""Non-retraining decode variants for the autoregressive freeze.

The doc's exclusion table records "freeze sigma at the launch window" as
ineffective in one line, with no script and no numbers behind it -- unlike the
growing-window and noise-injection entries, which have both.  This puts the
whole decode-side family on one comparable footing: same checkpoints, same
launch points, same rollout, only the decode differs.

Decode under test (the z-score rule is y_hat = z * sigma_window + mu_window):

  baseline      z * sigma_win  + mu_win     what the model was trained for
  freeze-sigma  z * sigma_0    + mu_win     restore the launch window's spread
  sigma-floor   z * max(sig_win, a*sig_0) + mu_win
  freeze-both   z * sigma_0    + mu_0       fully frozen decode (a bound, not
                                            a proposal: it predicts a constant)
  last-value    z * sigma_win  + last_win   reference point moved to the last
                                            input value
  sig-train(a)  z * sigma_typ(mu_win) + mu_win
                                            sigma for THIS level, from the
                                            training cells' own windows
  sig-slope(b)  z * d0*W/sqrt(12) + mu_win  sigma implied by the slope the
                                            LAUNCH window actually shows
  z-clamp       min(z, Z_CRIT) * sigma_win + mu_win
                                            leave z alone unless it has risen
                                            past the self-consistency value

z-clamp is the odd one out and the reason this file was reopened.  The other six
rescale the decode; this one constrains what z is allowed to say, and it comes
from the self-consistency condition rather than from a guess about sigma.  How
it behaves in each regime:

  * a healthy rollout emits z ~ -2.1, well past Z_CRIT -- the clamp is INERT,
    the rollout is untouched;
  * a freezing rollout walks z up toward 0 -- the clamp catches it at Z_CRIT
    and holds it there, which is exactly the step at which the decline would
    otherwise stop being self-sustaining.

The catch is that the clamp commands a drop of |Z_CRIT| * sigma_window, so once
the window has actually flattened (sigma at the cell's floor) it may not be able
to buy a whole cycle of decline.  headroom() below measures that a priori, and
the run stops short if the answer is no.

Why (a) and (b): everything that freezes sigma at one constant is a stopgap --
a cell's local spread grows as its decline accelerates, so the right sigma
moves.  (a) reads it off the level, (b) off the launch window's own slope;
both are non-retraining, and both are fixed by data the model already has.

A diagnostic is reported alongside each: the mean absolute z over the rollout.
A frozen-sigma variant that "works" while mean|z| collapses would be an
artefact; the measured value here is ~1.0-1.4, i.e. z does NOT decay to zero
on average, which is why this family was worth reopening at all.

Read-only: loads checkpoints and prints; writes nothing.

    D:/anaconda/envs/py312/python.exe src/ar_decode_variants.py --seeds 1 2 3
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import test_ar_rollout as tar          # noqa: E402
from make_figures import load_series   # noqa: E402

EPS = 1e-6
EOL = 640
THR = 0.77
KS = (50, 100)
SIG_FRAC = 0.5          # sigma-floor: never decode with less than this share
                        # of the launch window's spread

# z-clamp: the self-consistency value.  For the rollout to keep falling at some
# rate d, the new point has to sit one step below the window's newest value; in
# z-score terms that is  y - mu = -d*(W/2 + 1/2),  and a clean ramp of step d
# has sigma = d*sqrt((W^2-1)/12).  Dividing, d cancels and
#
#     Z_CRIT = -(W/2 + 1/2) / sqrt((W^2-1)/12)  =  -32.5 / 18.47295  =  -1.7593
#
# for W = 64 -- independent of how fast the cell is fading, which is what makes
# one constant usable for every cell.  (The doc quotes -1.73 from the slightly
# different -(W/2)*sqrt(12)/W; the 1.5% gap is a rounding of the same algebra
# and does not move any conclusion here, but the exact value costs nothing.)
Z_CRIT = -1.7593


def first_crossing(series, thr):
    for i in range(len(series) - 1):
        if series[i] >= thr > series[i + 1]:
            return i
    return -1


def sigma_table(caps, cells, lo, hi, W, nbins=20):
    """(centres, sigmas): the training cells' own window spread, by window mean.

    Built only from cells the model is allowed to see.  It answers "at this
    capacity level, how much does a REAL window vary?" -- which is exactly what
    the rollout's window stops telling us once it fills with the model's output.
    """
    means, sigs = [], []
    for c in cells:
        s = (np.asarray(caps[c]) - lo) / (hi - lo)
        for i in range(W, len(s)):
            w = s[i - W:i]
            means.append(float(w.mean()))
            sigs.append(float(w.std()))
    means = np.asarray(means)
    sigs = np.asarray(sigs)
    edges = np.linspace(means.min(), means.max(), nbins + 1)
    b = np.clip(np.digitize(means, edges) - 1, 0, nbins - 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    tab = np.array([sigs[b == j].mean() if (b == j).any() else np.nan
                    for j in range(nbins)])
    ok = ~np.isnan(tab)
    return centres[ok], tab[ok]


def headroom(seq, W, k):
    """Can the clamp still buy a cycle's worth of decline once the window flattens?

    The clamp commands  y - mu = Z_CRIT * sigma_window.  A frozen window sits at
    the cell's own sigma floor, so that product is the most decline the clamp can
    still force per step; the true per-cycle drop over the last k cycles before
    EOL is what it has to match.  Ratio >= 1 means the clamp can keep up even in
    the worst case; below 1 it cannot, and no amount of clamping will carry the
    rollout to the threshold on its own.

    Both quantities are in the same normalised space, so the ratio is invariant
    to the checkpoint's lo/hi and one number covers every seed.
    """
    s = np.asarray(seq, dtype=np.float64)
    floor = float(min(s[i - W:i].std() for i in range(W, len(s))))
    d_true = float(np.mean(np.abs(np.diff(s[EOL - k:EOL]))))
    return floor, d_true, abs(Z_CRIT) * floor / d_true


def rollout(model, seq, t0, W, steps, mode, ref):
    """Self-fed rollout from cycle index t0 with one of the decodes above.

    Returns (values, mean|z|, clamp binds).  The bind count is 0 for every mode
    but z-clamp, where it is the number of steps the clamp actually intervened.
    Warm-up is the W true cycles before t0; after that nothing real is fed,
    exactly as in test_ar_rollout.rollout.

    `ref` carries the launch-window quantities: mu0, sig0, and -- for (a)/(b) --
    the training sigma table and the launch window's own slope.
    """
    win = list(seq[t0 - W:t0].astype(np.float64))
    mu0, sig0 = float(np.mean(win)), float(np.std(win))
    # (b): the spread a window of this slope would have if it were a clean ramp,
    # sigma = d * W / sqrt(12).  d comes from the TRUE launch window.
    d0 = abs(float(np.polyfit(np.arange(W, dtype=np.float64),
                              np.asarray(win), 1)[0]))
    sig_b = d0 * W / np.sqrt(12.0)
    centres, tabs = ref["table"]
    out, zs = [], []
    nbind = 0                      # z-clamp only: steps where the clamp bit
    with torch.no_grad():
        for _ in range(steps):
            x = np.asarray(win[-W:], dtype=np.float32)
            wm = float(x.mean())
            ws = float(x.std()) + EPS
            z = float(model(torch.tensor(x[None, :, None], device=tar.DEV)))
            zs.append(abs(z))
            if mode == "baseline":
                y = z * ws + wm
            elif mode == "freeze-sigma":
                y = z * sig0 + wm
            elif mode == "sigma-floor":
                y = z * max(ws, SIG_FRAC * sig0) + wm
            elif mode == "freeze-both":
                y = z * sig0 + mu0
            elif mode == "last-value":
                y = z * ws + float(x[-1])
            elif mode == "sig-train":
                y = z * float(np.interp(wm, centres, tabs)) + wm
            elif mode == "sig-slope":
                y = z * sig_b + wm
            elif mode == "z-clamp":
                # inert while the rollout is healthy (z ~ -2.1 is already past
                # Z_CRIT); binds only once z has walked up past it
                if z > Z_CRIT:
                    z, nbind = Z_CRIT, nbind + 1
                y = z * ws + wm
            else:
                raise ValueError(mode)
            out.append(y)
            win.append(y)
    return np.array(out), float(np.mean(zs)), nbind


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="calce")
    ap.add_argument("--sp", type=int, default=500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--modes", nargs="+",
                    default=["baseline", "freeze-sigma", "sig-train",
                             "sig-slope", "z-clamp"])
    args = ap.parse_args()

    caps, _tr, te, W, _sps, _eol = load_series(args.dataset)
    caps32 = {c: caps[c].astype(np.float32) for c in caps}

    print("=" * 96)
    print(f"  {args.dataset.upper()} {te} SP{args.sp}  decode variants, "
          f"{len(args.seeds)} seeds, launch at true EOL - k")
    print(f"  z-score rule: y = z * sigma_window + mu_window;  "
          f"sigma-floor uses max(sigma_win, {SIG_FRAC}*sigma_launch)")
    print("=" * 96)
    if "z-clamp" in args.modes:
        # The go/no-go, read BEFORE the table: once the window has frozen its
        # sigma sits at the cell's floor, so |Z_CRIT| * sigma_floor is the most
        # decline the clamp can still force per step.  If that is under the true
        # per-cycle drop, the clamp alone cannot carry the rollout to the
        # threshold and the rows below will say so for a reason that is already
        # known here.  Raw Ah is fine -- the ratio is scale-invariant, so one
        # number covers every seed.
        floor_r, d_true, ratio = headroom(caps32[te], W, KS[0])
        verdict = "CAN keep up" if ratio >= 1 else "CANNOT keep up"
        print(f"  z-clamp headroom (a priori):")
        print(f"    |Z_CRIT| * sigma_floor = {abs(Z_CRIT) * floor_r:.5f}"
              f"   (sigma_floor {floor_r:.5f}, the flattest real window)")
        print(f"    true drop per cycle    = {d_true:.5f}"
              f"   (mean |diff| over the last {KS[0]} cycles before EOL)")
        print(f"    ratio {ratio:.2f}  ->  clamp {verdict} on its own")
        print("=" * 96)
    hdr = (f"  {'mode':<14}{'k':>4}{'MSE':>12}{'AE':>12}{'AE/k':>7}"
           f"{'final':>9}{'mean|z|':>10}{'sig0':>9}{'binds':>7}")
    print(hdr)
    print("-" * len(hdr))

    for mode in args.modes:
        for k in KS:
            mses, aes, finals, zs_, binds = [], [], [], [], []
            sig0 = float("nan")
            for seed in args.seeds:
                model, ck = tar.load_ckpt(args.dataset, args.sp, seed)
                seq = (caps32[te] - ck["lo"]) / (ck["hi"] - ck["lo"] + EPS)
                thr_c = (THR - ck["lo"]) / (ck["hi"] - ck["lo"] + EPS)
                t0 = EOL - k
                # (a) needs the training cells' sigma by level, in the SAME
                # normalisation the rollout uses -- hence ck's lo/hi
                table = sigma_table(caps32, ck["train_cells"], ck["lo"],
                                    ck["hi"], ck["W"])
                vals, zmean, nbind = rollout(model, seq, t0, ck["W"],
                                             len(seq) - t0, mode,
                                             {"table": table})
                truth = seq[t0:t0 + len(vals)]
                n = min(k, len(vals))
                mses.append(float(np.mean((vals[:n] - truth[:n]) ** 2)))
                finals.append(float(vals[-1]))
                zs_.append(zmean)
                binds.append(nbind)
                sig0 = float(np.std(seq[t0 - ck["W"]:t0]))
                i = first_crossing(vals, thr_c)
                # AE = |predicted crossing - true EOL| in cycles; a rollout
                # that never crosses contributes no AE and is reported as "--"
                if i >= 0:
                    aes.append(abs(int(t0 + i + 1) - EOL))
                del model
                torch.cuda.empty_cache()
            ae = (f"--/0" if not aes else
                  f"{np.mean(aes):.0f}"
                  + (f"+/-{np.std(aes, ddof=1):.0f}" if len(aes) > 1 else "")
                  + f"/{len(aes)}")
            aek = "--" if not aes else f"{np.mean(aes) / k:.2f}"
            print(f"  {mode:<14}{k:>4}{np.mean(mses):>12.2e}{ae:>12}{aek:>7}"
                  f"{np.median(finals):>9.4f}{np.mean(zs_):>10.4f}"
                  f"{sig0:>9.5f}{np.mean(binds):>7.0f}")
        print()

    print(f"  MSE over the k launch steps;  AE = |predicted crossing - true "
          f"EOL| in cycles ({len(args.seeds)} seeds);  AE/k is its ratio to")
    print("  the remaining life at launch, so an AE of k is a useless number.")
    print("  AE reads mean+/-std/N where N is how many of the seeds reached the")
    print("  threshold at all -- a rollout that never crosses contributes no AE,")
    print("  so N < seeds means the mean is over a self-selected subset.")
    print("  mean|z| is over the whole rollout and is the RAW z, before any")
    print("  clamp.  binds counts the steps z-clamp actually intervened on --")
    print("  0 means the clamp never fired and the row is baseline by another")
    print("  name; a large count means it was fighting the model the whole way.")


if __name__ == "__main__":
    main()
