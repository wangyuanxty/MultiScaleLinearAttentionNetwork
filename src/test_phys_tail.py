"""Physics tail evaluation — anchored protocol, v7 (multi-cell + IR).

Does the tail rate model need PHYSICS FEATURES (IR), or is the rate
concept alone sufficient? Per cell (all 4 CALCE cells as test, each
90/10 split), fit on the recent 30% of observed:

  local-rate: r = mean(recent rates)                    [no features]
  IR-linear : r = k0 + k1 * IR                          [physics feature]
  IR+n      : r = k0 + k1 * IR + k2 * n                 [feature + time]

Anchored eval at each tail step i (same protocol as GDN extrapolation):
  Q_hat_i = Q_{i-1} - r(IR_i, n_i)      with OBSERVED IR_i, n_i
Reference for CS2_35: GDN n-channel extR2=0.4471.
"""
import sys
import numpy as np

sys.path.insert(0, '.')
from make_figures import load_series
from load_datasets import load_calce_cells_multivar

EPS = 1e-6
W = 64
FIT_FRAC = 0.3
SMOOTH = 15
IR_IDX = 3


def smooth_rates(q):
    r = (q[:-5] - q[5:]) / 5.0
    r = np.concatenate([r, np.full(5, r[-1])])
    kernel = np.ones(SMOOTH) / SMOOTH
    return np.convolve(r, kernel, mode="same")


def eval_cell(q, ir, cell):
    cut = int(len(q) * 0.9)
    n_all = (np.arange(len(q)) + 1) / len(q)
    fit0 = int(cut * (1 - FIT_FRAC))
    r_all = smooth_rates(q)
    n_fit = n_all[fit0:cut]
    r_fit = r_all[fit0:cut]
    ir_fit = ir[fit0:cut]

    rows = {}

    # local-rate: constant, no features
    r_loc = float(np.mean(r_fit[-50:]))
    def rate_loc(n_, ir_):
        return r_loc

    # IR-linear
    k1, k0 = np.polyfit(ir_fit, r_fit, 1)
    def rate_ir(n_, ir_):
        return k0 + k1 * ir_

    # IR + n
    A = np.stack([np.ones_like(ir_fit), ir_fit, n_fit], axis=1)
    coef, *_ = np.linalg.lstsq(A, r_fit, rcond=None)
    def rate_irn(n_, ir_):
        return coef[0] + coef[1] * ir_ + coef[2] * n_

    def anchored(rate):
        p = []
        for i in range(cut, len(q)):
            p.append(q[i - 1] - max(rate(n_all[i], ir[i]), 0.0))
        p = np.clip(np.array(p), 0.0, None)
        seg_t = q[cut:]
        r2 = 1 - np.sum((seg_t - p) ** 2) / (np.sum((seg_t - seg_t.mean()) ** 2) + EPS)
        mae = np.mean(np.abs(seg_t - p))
        return r2, mae

    for name, rate, desc in [
        ("local-rate", rate_loc, f"r={r_loc:.5f}"),
        ("IR-linear", rate_ir, f"k0={k0:.5f} k1={k1:.5f}"),
        ("IR+n", rate_irn, f"c0={coef[0]:.5f} c1={coef[1]:.5f} c2={coef[2]:.5f}"),
    ]:
        r2, mae = anchored(rate)
        rows[name] = (r2, mae)
        print(f"  {cell} {name:10s} ({desc}): extR2={r2:.4f} extMAE={mae:.5f}",
              flush=True)
    return rows


def main():
    caps_all, feats_all, _ = load_calce_cells_multivar()
    cells = sorted(caps_all.keys())
    print(f"multi-cell anchored tail eval: {cells}", flush=True)
    agg = {}
    for cell in cells:
        q = caps_all[cell].astype(np.float64)
        ir = feats_all[cell][:, IR_IDX].astype(np.float64)
        rows = eval_cell(q, ir, cell)
        for k, v in rows.items():
            agg.setdefault(k, []).append(v)
    print("--- mean over cells ---", flush=True)
    for k, vs in agg.items():
        r2s = np.mean([v[0] for v in vs])
        maes = np.mean([v[1] for v in vs])
        print(f"  {k:10s}: extR2={r2s:.4f} extMAE={maes:.5f}", flush=True)
    print("reference (CS2_35): GDN n-channel 0.4471", flush=True)


if __name__ == "__main__":
    main()
