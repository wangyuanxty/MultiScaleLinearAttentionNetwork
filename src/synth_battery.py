r"""Fully synthetic degradation trajectories -- no real dataset anywhere in the loop.

Why a pure simulation.  The question under test is whether an autoregressive
rollout freezes because there are too few INDEPENDENT degradation trajectories
to learn from.  Real data cannot answer it: the libraries here top out at 43
cells (MIT), the per-SP protocol trains on 2-8 of them, and every earlier attempt
to stretch that -- noise augmentation, MIT 8->39, a longer window -- changed two
things at once or was measured with one seed.  A generator makes the trajectory
COUNT the free variable: the same distribution can be trained on 1 cell or 100,
with test cells held out and never touched.

The model
---------
    Q(n) = Q0 - a*sqrt(n) - b*max(0, n - n_k)^gamma + eps_n
            \_______/     \_______________________/     \___/
             SEI, sqrt-t      post-knee acceleration      AR(1) noise

The coefficients are NOT sampled.  (a, b, n_k, gamma) are strongly coupled --
near the first cycle a and b trade off almost exactly -- so a draw from them is
not a physically meaningful cell.  Instead four INTERPRETABLE quantities are
sampled and a, b are solved for:

    L         cycle life: the trend crosses EOL at exactly n = L
    n_k / L   where the knee starts, as a fraction of life
    f         share of the total loss that happened BEFORE the knee
    gamma     knee sharpness

Solving:  a*sqrt(n_k) = f*loss  gives  a = f*loss/sqrt(n_k);  substituting into
a*sqrt(L) + b*(L-n_k)^gamma = loss gives

    b = loss * (1 - f*sqrt(L/n_k)) / (L - n_k)^gamma

which needs f < sqrt(n_k/L) -- satisfied by construction, and asserted below.
Every generated cell therefore reaches EOL at exactly its own L, so no cell is
like NASA's B0007 (which never crossed and silently left the near-EOL regime with
2 trajectories instead of the nominal 3).

Noise is AR(1), not i.i.d.  Real capacity residuals have lag-1 autocorrelation
0.24-0.61; i.i.d. noise at the same sigma inflates per-cycle movement by 1.6x and
destroys the local slope the model has to read.  That mistake was made once
already with real data and is not repeated here.

Read-only.  Nothing is written, nothing existing is touched.

    D:/anaconda/envs/py312/python.exe src/synth_battery.py --check
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

EOL_FRAC = 0.80         # EOL at 80% of initial capacity
TAIL = 50               # cycles past EOL: makes the AR rollout exactly EOL-50 -> EOL+50
FLOOR_FRAC = 0.35       # the cell is "dead" here; the series ENDS rather than clipping


@dataclass(frozen=True)
class CellParams:
    """One synthetic cell, in interpretable units."""
    Q0: float           # initial capacity
    L: int              # cycle life: EOL is crossed at n = L
    nk_frac: float      # knee onset as a fraction of L
    f: float            # share of total loss suffered before the knee
    gamma: float        # knee sharpness
    snr: float          # noise sigma as a multiple of the trend's slope AT EOL
    phi: float          # noise lag-1 autocorrelation

    @property
    def n_k(self) -> float:
        return self.nk_frac * self.L

    @property
    def eol(self) -> float:
        return self.Q0 * EOL_FRAC

    @property
    def loss(self) -> float:
        return self.Q0 * (1.0 - EOL_FRAC)

    def coeffs(self) -> tuple[float, float]:
        """(a, b) solved so the trend crosses EOL at exactly n = L."""
        c = self.f * np.sqrt(self.L / self.n_k)     # share of loss the sqrt term
        if not c < 1.0:                             # carries at n = L
            raise ValueError(f"f={self.f} too large for nk_frac={self.nk_frac}")
        a = self.f * self.loss / np.sqrt(self.n_k)
        b = self.loss * (1.0 - c) / (self.L - self.n_k) ** self.gamma
        return float(a), float(b)

    @property
    def slope_eol(self) -> float:
        """|dQ/dn| of the noise-free trend where it crosses EOL, per cycle."""
        a, b = self.coeffs()
        return float(a / (2.0 * np.sqrt(self.L))
                     + b * self.gamma * (self.L - self.n_k) ** (self.gamma - 1.0))

    @property
    def sigma(self) -> float:
        """Noise sigma, in capacity units.

        Set from the EOL SLOPE rather than from Q0, because sigma is only
        meaningful against the signal it has to be read out of.  Sampling sigma
        as a fraction of Q0 (the obvious thing) let sigma/slope reach 11 against
        2.0 for a real CALCE cell, and at that ratio a single noise excursion
        moves the crossing by ten cycles -- the cell has no well-defined end of
        life at all.  Sampling the RATIO instead keeps every cell legible.
        """
        return self.snr * self.slope_eol


def sample_params(rng: np.random.Generator) -> CellParams:
    """One cell.  Ranges are the physically-plausible band, not fitted to any dataset.

    nk_frac starts at 0.65, not 0.50: a knee that begins at half of life spreads
    the post-knee loss over hundreds of cycles and leaves the trend almost flat
    where it crosses EOL.  Real cells put the knee at ~0.85 of life (CALCE
    CS2_35: onset ~550, EOL 640), so the last stretch is steep.
    """
    return CellParams(
        Q0=float(rng.uniform(0.95, 1.05)),
        L=int(round(rng.uniform(150, 450))),
        nk_frac=float(rng.uniform(0.65, 0.90)),
        f=float(rng.uniform(0.25, 0.50)),
        gamma=float(rng.uniform(2.0, 4.0)),
        snr=float(rng.uniform(0.8, 2.5)),      # CALCE CS2_35 measures 2.0
        phi=float(rng.uniform(0.20, 0.50)),
    )


def ar1_noise(n: int, sigma: float, phi: float,
              rng: np.random.Generator) -> np.ndarray:
    """Correlated noise with lag-1 autocorrelation exactly phi."""
    e = np.empty(n, dtype=np.float64)
    e[0] = rng.normal(0.0, sigma)
    step = np.sqrt(1.0 - phi * phi)
    for t in range(1, n):
        e[t] = phi * e[t - 1] + step * rng.normal(0.0, sigma)
    return e


def trend_of(p: CellParams, n: np.ndarray) -> np.ndarray:
    """The noise-free trend.  One definition, shared by the generator and the check."""
    a, b = p.coeffs()
    return p.Q0 - a * np.sqrt(n) - b * np.maximum(0.0, n - p.n_k) ** p.gamma


def make_cell(p: CellParams, rng: np.random.Generator, tail: int = TAIL):
    """The capacity series, in the same units as Q0.

    The series ENDS where the cell dies.  Past EOL the power law accelerates
    without bound, so the trajectory reaches the floor within ~20-30 cycles --
    and that is arithmetic, not a modelling artefact: a cell that crosses 80% and
    then fades at ~1%/cycle has only 20% of capacity left to spend, so a tail
    much longer than ~20-30 cycles would require capacity to go NEGATIVE.  Real
    cells behave the same way (MIT batch2_cell5: 18 cycles past EOL; NASA B0005:
    44).  Forcing a uniform long tail -- continuing linearly at the EOL slope, or
    dropping the floor -- makes unphysical trajectories, not better ones.

    A short tail weakens the AE lower bound on a rollout that never crosses
    (last cycle reached - EOL is all you can say).  That belongs in the
    MEASUREMENT, not here: roll out a FIXED number of steps instead of "until the
    series ends", so the crossing test uses the whole rollout while MSE uses only
    the part that overlaps real values.
    """
    n = np.arange(1, p.L + tail + 1, dtype=np.float64)
    trend = trend_of(p, n)
    dead = np.where(trend <= FLOOR_FRAC * p.Q0)[0]
    end = int(dead[0]) + 1 if len(dead) else len(trend)
    return trend[:end] + ar1_noise(end, p.sigma, p.phi, rng)


def make_pool(n_train: int, n_test: int, seed: int = 0):
    """(train, test, train_params, test_params).  Order is GENERATION order.

    Nesting depends on that: the caller takes prefixes train[:N], so any sort by
    L or n_k would make small-N draws systematically short-lived instead of a
    random sample of the same distribution.  Never sorted -- deliberately.
    """
    rng = np.random.default_rng(seed)
    test_params = [sample_params(rng) for _ in range(n_test)]     # held out first
    train_params = [sample_params(rng) for _ in range(n_train)]
    test = {f"T{i:02d}": make_cell(p, rng) for i, p in enumerate(test_params)}
    train = {f"C{i:03d}": make_cell(p, rng) for i, p in enumerate(train_params)}
    return train, test, train_params, test_params


def first_crossing(s: np.ndarray, thr: float) -> int:
    for i in range(len(s) - 1):
        if s[i] >= thr > s[i + 1]:
            return i + 1
    return -1


def acf1(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    v = float(np.dot(x, x)) / n
    return float(np.dot(x[:-1], x[1:])) / n / v if v > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="generate the pool and verify its invariants")
    ap.add_argument("--n-train", type=int, default=115)
    ap.add_argument("--n-test", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-show", type=int, default=8)
    args = ap.parse_args()

    train, test, tp, sp = make_pool(args.n_train, args.n_test, args.seed)
    print("=" * 96)
    print(f"  SYNTHETIC POOL  seed={args.seed}  {len(train)} train + {len(test)} test"
          f"  (test generated first, held out for every N)")
    print(f"  EOL_FRAC={EOL_FRAC}  TAIL={TAIL}  FLOOR={FLOOR_FRAC}  "
          f"L~U(150,450)  n_k/L~U(0.65,0.90)  f~U(0.25,0.50)  gamma~U(2.0,4.0)"
          f"  sig/d~U(0.8,2.5)")
    print("=" * 96)

    hdr = (f"  {'cell':<7}{'Q0':>7}{'L':>6}{'nk/L':>7}{'f':>6}{'gam':>6}"
           f"{'sig/d':>7}{'phi':>6}{'EOL@':>6}{'len':>5}{'slope':>9}{'drop/cyc':>10}{'acf1':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    rows = [("train", k, train[k], p) for k, p in zip(train, tp)][:args.n_show]
    rows += [("TEST", k, test[k], p) for k, p in zip(test, sp)]
    for tag, k, s, p in rows:
        eol = first_crossing(s, p.eol)
        seg = s[max(0, eol - 50):eol]
        drop = float(np.mean(np.abs(np.diff(seg)))) if len(seg) > 1 else float("nan")
        # residual acf1, against the noise-free TREND (we know it exactly here).
        # MUST be the same trend_of the generator used: computing the untruncated
        # power law here instead left the truncated tail as a systematic offset
        # and pushed acf1 to 0.82-0.94 against a nominal phi of 0.2-0.5.
        n = np.arange(1, len(s) + 1, dtype=np.float64)
        resid = s - trend_of(p, n)
        print(f"  {tag:<7}{p.Q0:>7.3f}{p.L:>6}{p.nk_frac:>7.2f}{p.f:>6.2f}"
              f"{p.gamma:>6.2f}{p.snr:>7.2f}{p.phi:>6.2f}"
              f"{eol if eol > 0 else -1:>6}{len(s):>5}{p.slope_eol:>9.5f}"
              f"{drop:>10.5f}{acf1(resid):>7.2f}")

    print()
    eols = [first_crossing(train[k], p.eol) for k, p in zip(train, tp)]
    dev = [abs(e - p.L) for e, p in zip(eols, tp)]
    print(f"  A. crossing vs nominal L:  max |dev| = {max(dev)} cycles, "
          f"mean {np.mean(dev):.1f}   (was 10 before sigma was tied to the EOL"
          f" slope)")
    tails = [len(np.asarray(train[k])) - p.L for k, p in zip(train, tp)]
    tt = [len(np.asarray(test[k])) - p.L for k, p in zip(test, sp)]
    print(f"  B. tail past EOL: train {min(tails)}-{max(tails)}, "
          f"TEST {min(tt)}-{max(tt)} cycles (one per test cell: {tt}).")
    print(f"     Short BY PHYSICS, not by construction: a cell at 80% fading "
          f"~1%/cycle has only ~20 cycles of capacity left to spend.  MIT's real")
    print(f"     batch2_cell5 had 18.  A long tail would need capacity to go "
          f"negative.  SEE run_synth.py for how the measurement handles it.")
    lens = [len(v) for v in train.values()]
    print(f"  C. series length: {min(lens)}-{max(lens)}   "
          f"train windows at W=64: {sum(max(0, n - 64) for n in lens)}"
          f"   at W=256: {sum(max(0, n - 256) for n in lens)}")
    print(f"  D. sig/d range {min(p.snr for p in tp):.2f}-"
          f"{max(p.snr for p in tp):.2f}   (real CALCE CS2_35 = 2.0)")
    print(f"  E. files written: none")


if __name__ == "__main__":
    main()
