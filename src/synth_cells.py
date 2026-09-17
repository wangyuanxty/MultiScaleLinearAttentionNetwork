"""Synthetic training cells with the RIGHT autocorrelation, plus the check that proves it.

The first augmentation attempt added i.i.d. Gaussian noise to the real training
cells.  It matched the VARIANCE of each cell's residual and nothing else.  The
measured lag-1 autocorrelation of those residuals is not zero:

    NASA  B0006 0.61   B0007 0.51   B0018 0.55
    CALCE CS2_36 0.30  CS2_37 0.24  CS2_38 0.37

and the per-cycle movement a series shows is Var(dx) = 2*sigma^2*(1 - rho_1).
At the same sigma, dropping rho_1 from 0.6 to 0 inflates that by 1.6x; the
measured effect was larger still -- the i.i.d. synthetic cells moved 0.0402 Ah
per cycle near their EOL against 0.0143 for the real cell they were copied from,
2.8x.  A series that jerks that hard cycle-to-cycle has no local slope left to
learn, and a model trained on it learns the wrong lesson for exactly the regime
the autoregressive rollout lives in: "a window with this much spread does not
predict its next value."

So the perturbation here is a phase-randomized surrogate of the real residual
(Theiler et al.): take the residual, FFT it, keep the magnitude spectrum,
randomize the phases, invert.  That preserves the autocorrelation EXACTLY --
whatever it happens to be for that cell, with no kernel width to pick -- while
producing a new realization.  Zero mean by construction (the DC bin is zeroed),
so the perturbation never shifts the level.

This file is the generator AND its acceptance test.  `main()` prints, per cell,
the three numbers the surrogate has to reproduce: residual sigma, the
autocorrelation at lags 1-3, and the per-cycle movement in the last 50 cycles
before that cell's EOL.  The i.i.d. version is printed alongside as the control
that failed.  Nothing is trained until those columns line up.

Read-only: loads series, prints.  Writes nothing.

    D:/anaconda/envs/py312/python.exe src/synth_cells.py --dataset calce
    D:/anaconda/envs/py312/python.exe src/synth_cells.py --dataset nasa --n-aug 9
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from make_figures import load_series                       # noqa: E402

DETREND_WIN = 21        # cycles; everything shorter than this counts as residual
NEAR_EOL = 50           # window over which per-cycle movement is measured


def detrend(s: np.ndarray, win: int = DETREND_WIN) -> np.ndarray:
    """Centred moving average, edges padded by repetition."""
    s = np.asarray(s, dtype=np.float64)
    pad = win // 2
    sp = np.pad(s, pad, mode="edge")
    return np.convolve(sp, np.ones(win) / win, mode="valid")[:len(s)]


def residual(s: np.ndarray, win: int = DETREND_WIN) -> np.ndarray:
    """What the trend does not explain.  Kept as measured -- no model assumed."""
    return np.asarray(s, dtype=np.float64) - detrend(s, win)


def surrogate(r: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A new realization with the same magnitude spectrum, hence the same ACF.

    Phase randomization: FFT, keep |F|, replace the phases with uniform draws,
    invert.  The DC bin is zeroed so the perturbation has exactly zero mean; the
    Nyquist bin is held real for an even-length series.
    """
    r = np.asarray(r, dtype=np.float64)
    r = r - r.mean()
    n = len(r)
    F = np.fft.rfft(r)
    mag = np.abs(F)
    mag[0] = 0.0
    ph = rng.uniform(0.0, 2.0 * np.pi, size=len(F))
    ph[0] = 0.0
    if n % 2 == 0:
        ph[-1] = 0.0
    return np.fft.irfft(mag * np.exp(1j * ph), n=n)


def acf(x: np.ndarray, maxlag: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    v = float(np.dot(x, x)) / n
    if v <= 0:
        return np.zeros(maxlag + 1)
    return np.array([float(np.dot(x[:n - k], x[k:])) / n / v
                     for k in range(maxlag + 1)])


def eol_cycle(s: np.ndarray, eol_ah: float) -> int:
    for i in range(len(s) - 1):
        if s[i] >= eol_ah > s[i + 1]:
            return i + 1
    return -1


def near_eol_move(s: np.ndarray, eol_ah: float, k: int = NEAR_EOL) -> float:
    """Mean |per-cycle change| over the last k cycles before this cell's EOL.

    This is the number the i.i.d. augmentation got wrong by 2.8x, and the one a
    usable synthetic cell has to match.
    """
    ec = eol_cycle(s, eol_ah)
    if ec < 0:
        return float("nan")
    seg = np.asarray(s, dtype=np.float64)[max(0, ec - k):ec]
    return float(np.mean(np.abs(np.diff(seg)))) if len(seg) > 1 else float("nan")


def envelope(r: np.ndarray, win: int = 31) -> np.ndarray:
    """Local scale of the residual (moving RMS, edges padded)."""
    pad = win // 2
    rp = np.pad(np.asarray(r, dtype=np.float64), pad, mode="edge")
    return np.sqrt(np.convolve(rp ** 2, np.ones(win) / win,
                               mode="valid")[:len(r)])


def build(caps, train_cells, n_aug, mult, seed=0, mode="surrogate",
          use_envelope=False):
    """Real cells plus `n_aug` perturbed copies each.  `caps` is left untouched.

    A synthetic cell is TREND + perturbation, not cell + perturbation.  The real
    cell already contains its own residual, so adding one on top doubles it --
    measured sigma went 0.01184 -> 0.01650, exactly the sqrt(2) that predicts.
    Replacing the residual is what makes the copy a different realization of the
    same process rather than a noisier version of the same trajectory.

    The residual is not stationary either: it is several times larger in the
    plateau than at the knee, so a globally-stationary surrogate is too jumpy
    near EOL, which is the one regime this whole exercise is about.  With
    use_envelope the SHAPE is randomised and the measured local scale is put
    back afterwards.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for c in train_cells:
        s = np.asarray(caps[c], dtype=np.float64)
        trend = detrend(s)
        r = s - trend
        env = None
        if use_envelope:
            env = envelope(r)
            r = np.divide(r, env, out=np.zeros_like(r), where=env > 1e-12)
        out[c] = s.copy()
        for k in range(n_aug):
            if mode == "surrogate":
                pert = surrogate(r, rng)
            elif mode == "iid":
                pert = rng.normal(0.0, r.std(), size=s.shape)
            else:
                raise ValueError(mode)
            if use_envelope:
                pert = pert / (pert.std() + 1e-12) * env
            out[f"{c}_a{k + 1:02d}"] = trend + mult * pert
    return out


def report(real, synth, eol_ah, n_show=3):
    """The acceptance test: sigma, ACF, and per-cycle movement, side by side."""
    hdr = (f"      {'cell':<12}{'sigma':>9}{'ACF1':>7}{'ACF2':>7}{'ACF3':>7}"
           f"{'drop/cyc':>10}")
    print(hdr)
    print("      " + "-" * (len(hdr) - 6))
    for c in sorted(real):
        s = np.asarray(real[c], dtype=np.float64)
        r = residual(s)
        a = acf(r)
        print(f"      {c:<12}{r.std():>9.5f}{a[1]:>7.2f}{a[2]:>7.2f}{a[3]:>7.2f}"
              f"{near_eol_move(s, eol_ah):>10.5f}   <- REAL")
        copies = sorted(k for k in synth if k.startswith(c + "_a"))
        for k in copies[:n_show]:
            sk = np.asarray(synth[k], dtype=np.float64)
            rk = residual(sk)
            ak = acf(rk)
            print(f"      {k:<12}{rk.std():>9.5f}{ak[1]:>7.2f}{ak[2]:>7.2f}"
                  f"{ak[3]:>7.2f}{near_eol_move(sk, eol_ah):>10.5f}")
        if len(copies) > n_show:
            print(f"      {f'({len(copies) - n_show} more)':<12}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="calce")
    ap.add_argument("--n-aug", type=int, default=3,
                    help="copies per real cell to show (the generator takes more)")
    ap.add_argument("--mult", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    caps, tr, te, W, _sps, eol_ah = load_series(args.dataset)
    real_tr = list(tr)

    print("=" * 78)
    print(f"  {args.dataset.upper()}  surrogate augmentation acceptance test")
    print(f"  test cell {te} held out;  detrend win={DETREND_WIN}, "
          f"drop/cyc measured over the last {NEAR_EOL} cycles before EOL")
    print("=" * 78)

    for mode, env, label in (
            ("iid", False, "i.i.d. noise (the first attempt: ACF destroyed)"),
            ("surrogate", False,
             "phase-randomized surrogate, globally stationary"),
            ("surrogate", True,
             "phase-randomized surrogate + measured local scale")):
        synth = build(caps, real_tr, args.n_aug, args.mult, args.seed, mode, env)
        print()
        print(f"  --- {label} ---")
        report(caps, synth, eol_ah)

    print()
    print("  PASS looks like: every synthetic row's ACF1 AND drop/cyc sitting on")
    print("  its REAL row.  drop/cyc is the one that matters most -- it is how")
    print("  hard the series moves per cycle in the last 50 before EOL, i.e. the")
    print("  slope the model has to read, and the quantity the i.i.d. attempt")
    print("  inflated 2-4x.")


if __name__ == "__main__":
    main()
