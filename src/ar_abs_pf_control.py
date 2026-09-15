"""Controls for the ABSOLUTE-TARGET PatchFormer AR rollout (ar_abs_pf_rollout.py).

The rollout's claims each get an independent test; the point of this file is
the third one, which the rollout's own gates do NOT cover.

  C1  decode        -- already gated in ar_abs_pf_rollout.py (G2): the hand
                       decode reproduces official predict() to 3e-08.
  C2  checkpoint    -- already gated there (G3): teacher-forced MAE/R2/AE
                       reproduce the recorded ABS run exactly.
  C3  plumbing      -- NOT covered by step 1 of the rollout.  d1 ~ 1e-06 only
                       proves the LAUNCH is right; it says nothing about the
                       window shift, the i_start index, or the
                       mm <-> rated <-> Ah round trip on the appended value.

C3 replaces the network with an oracle that returns the true capacity for the
cycle being predicted.  If the plumbing is correct the rollout must then
reproduce the true series exactly, and the value it reads back out of its own
window must equal the true value at that position -- that second part is what
exercises the append-side conversion, since with a pure pass-through oracle a
wrong append conversion would still leave the outputs correct.

Read-only: nothing is written.

    D:/anaconda/envs/patchformer/python.exe src/ar_abs_pf_control.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import ar_ktable_calce as K            # noqa: E402
import ar_rollout_baselines as arb     # noqa: E402
import run_pf_abs as R                 # noqa: E402  (chdirs into ref_patchformer)

ABS_JSON = os.path.join(SRC, "results", "ar_ktable_calce_pfabs.json")
TRAJ_ZSCORE = os.path.join(SRC, "results", "ar_traj_calce.npz")


class OracleNet:
    """Stands in for `model.network` inside ar_abs_pf_rollout.rollout_abs.

    `arb.net_out` calls `model.network(x)` and takes the raw output as a
    z-score in the globally normalised target space, so returning
    (true_rated - ts_c) / ts_s makes the rollout decode back to exactly the
    true capacity for the cycle being predicted.  Every call after the first
    also checks the value at the end of the window it was handed -- that is
    the value the rollout wrote there on the previous step, so checking it is
    what exercises the append-side conversion.
    """

    def __init__(self, cap_mm, cap_rated, i_start, sc_m, sc_s,
                 minv, maxv, ts_c, ts_s):
        self.cap_mm = cap_mm
        self.cap_rated = cap_rated
        self.i_start = i_start
        self.sc_m, self.sc_s = sc_m, sc_s
        self.minv, self.maxv = minv, maxv
        self.ts_c, self.ts_s = ts_c, ts_s
        self.n = 0
        self.window_err = 0.0

    def network(self, x):
        n = self.n
        if n > 0:
            # the rollout appended one value after the previous call; it must
            # read back as the true capacity of cycle i_start + n - 1.
            # mm -> rated is the INVERSE of (rated - minv)/(maxv - minv)
            w_last_mm = float(x[0, -1, 0]) * self.sc_s + self.sc_m
            w_last_rated = w_last_mm * (self.maxv - self.minv) + self.minv
            truth = self.cap_rated[self.i_start + n - 1]
            self.window_err = max(self.window_err, abs(w_last_rated - truth))
        self.n += 1
        z = (self.cap_rated[self.i_start + n] - self.ts_c) / self.ts_s
        return torch.tensor([[z]], dtype=torch.float32)


def control_c3() -> bool:
    print("=" * 78)
    print("  C3  plumbing: oracle rollout (true values fed instead of the net)")
    print("=" * 78)
    with open(ABS_JSON, encoding="utf-8") as f:
        doc = json.load(f)
    cfg = R.prepare_cfg("calce")
    cfg["_ds"] = "calce"
    series = arb.official_calce()
    _, _, df_all, minv, maxv = arb.build_frames(series, doc["sp"])
    cyc = df_all["Cycle"].values
    cap_mm = df_all["Capacity"].values.astype(np.float64)
    cap_rated = df_all["target"].values.astype(np.float64)
    ts_c = doc["diag"]["target_scale_centre"]
    ts_s = doc["diag"]["target_scale_scale"]
    sc_m = doc["diag"]["scaler_mean"]
    sc_s = doc["diag"]["scaler_scale"]

    import ar_abs_pf_rollout as A      # the rollout under test, not a rewrite

    ok_all = True
    for key in sorted(doc["points"], key=lambda s: int(s)):
        p = doc["points"][key]
        k, start = p["k"], p["first_predicted_cycle"]
        i_start = int(np.where(cyc == start)[0][0])
        steps = len(cap_mm) - i_start
        net = OracleNet(cap_mm, cap_rated, i_start, sc_m, sc_s,
                        minv, maxv, ts_c, ts_s)
        full = A.rollout_abs(net, cap_mm, i_start, cfg, ts_c, ts_s,
                             sc_m, sc_s, minv, maxv, max_steps=steps)
        truth_ah = cap_rated[i_start:i_start + steps] * cfg["rated"]
        err = float(np.abs(full - truth_ah).max())
        # the oracle is exact in float64 but hands its value back through a
        # float32 tensor, so allow float32 round-off on the append path only
        tol = 1e-5 * cfg["rated"]
        ok = err <= tol and net.window_err <= tol
        ok_all &= ok
        print(f"  k={k:>3} start={start} steps={steps:>3}: "
              f"max|oracle rollout - truth| = {err:.3e} Ah, "
              f"max window readback err = {net.window_err:.3e} -> "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"  -> C3 {'PASS' if ok_all else 'FAIL'}\n")
    return ok_all


def control_c4() -> None:
    """Re-derive the paired z-score baseline from the raw trajectories.

    The ABS row is quoted against the z-score PatchFormer run-1 MAE, which was
    first read out of ar_ktable_calce.json.  This recomputes it from the saved
    rollout arrays instead, so the quoted comparison is not a transcription.
    """
    print("=" * 78)
    print("  C4  paired z-score baseline, recomputed from the raw trajectories")
    print("=" * 78)
    z = np.load(TRAJ_ZSCORE)
    true = z["true"]
    with open(ABS_JSON, encoding="utf-8") as f:
        doc = json.load(f)
    span = doc["common_space"]["hi"] - doc["common_space"]["lo"]
    for k in (25, 50, 75, 100):
        a = z[f"pred_pf_k{k}_s1"]
        v = np.where(~np.isnan(a))[0]
        i0 = int(v[0])
        seg, tseg = a[i0:i0 + k], true[i0:i0 + k]
        n = min(len(seg), len(tseg))
        print(f"  k={k:>3}: z-score PF run1 MAE_common = "
              f"{float(np.mean(np.abs(seg[:n] - tseg[:n])) / span):.5f}")
    print("\n  quoted in docs section 7.1: 0.04016 / 0.02575 / 0.02918 / 0.03558\n")


def control_c5(sp: int = 500, run: int = 1) -> None:
    """The paired z-score baseline, measured live at the SAME SP.

    The `models.pf` block in src/results/ar_ktable_calce.json was written by a
    `--force-sp 300` run -- every k record carries `sp: 300` -- so it is not
    the SP500 counterpart of the ABS row, and the trajectory npz files on disk
    disagree with each other as well.  This runs the z-score rollout through
    ar_rollout_baselines (the established path, same frames, same truth
    series) at the requested SP and prints the MAE in the same space.

    Read-only: nothing is written or saved.
    """
    cfg = R.prepare_cfg("calce")
    cfg["rated"] = 1.1
    points = K.horizon_points((25, 50, 75, 100), 640)
    for key, p in points.items():
        if p["sp"] != sp:
            raise SystemExit(f"k={key} resolves to SP{p['sp']}, not SP{sp}")

    print("=" * 78)
    print(f"  C5  z-score PatchFormer, measured live at SP{sp} run{run}")
    print("=" * 78)
    model, ckpt = arb.load_pf(sp, run)
    os.chdir(ROOT)
    series = arb.official_calce()
    _, _, df_all, minv, maxv = arb.build_frames(series, sp)
    cyc = df_all["Cycle"].values
    cap_mm = df_all["Capacity"].values.astype(np.float64)
    i_sp = int(np.where(cyc == sp)[0][0])
    _, _, _, _, diag = arb.teacher_forced(model, df_all, minv, maxv, sp, i_sp,
                                          "pf")
    sc_m, sc_s = diag["scaler_mean"], diag["scaler_scale"]
    lo, hi = K.common_ref()
    thr_mm = (K.EOL_AH / cfg["rated"] - minv) / (maxv - minv)
    print(f"  ckpt = {os.path.relpath(ckpt, R.REPO)}")
    print(f"  teacher-forced decode err = {diag['max_decode_err']:.2e}\n")

    def to_ah(q_mm):
        return np.asarray(q_mm) * (maxv - minv) * cfg["rated"] \
            + minv * cfg["rated"]

    for key in sorted(points, key=lambda s: int(s)):
        p = points[key]
        k, start = p["k"], p["T"] + 1
        i_start = int(np.where(cyc == start)[0][0])
        full, _ = arb.rollout(model, df_all, minv, maxv, start, sc_m, sc_s,
                              max_steps=len(cap_mm) - i_start, name="pf")
        pv_c = (to_ah(full[:k]) - lo) / (hi - lo)
        tv_c = (to_ah(cap_mm[i_start:i_start + k]) - lo) / (hi - lo)
        cross = K.crossing_cycle(full, thr_mm, start)
        print(f"  k={k:>3} (T={p['T']}, start={start}): "
              f"MAE_common={float(np.mean(np.abs(pv_c - tv_c))):.5f} "
              f"cross={cross} last={float(full[-1]):.4f}Ah")
    print()


def control_c6(sp: int = 500, run: int = 1) -> None:
    """Which run do the stored PatchFormer trajectories actually come from?

    The two CALCE trajectory files disagree with each other and with the
    `models.pf` block of ar_ktable_calce.json, and the paper's AR table is
    computed from these arrays (src/ar_table_from_traj.py).  This re-runs the
    SP500 rollout live and reports the max deviation of each stored array from
    it, which identifies the stored artifact instead of guessing.

    Read-only: nothing is written.
    """
    print("=" * 78)
    print(f"  C6  identify the stored PF trajectories (vs a live SP{sp} run{run})")
    print("=" * 78)
    model, ckpt = arb.load_pf(sp, run)
    os.chdir(ROOT)
    series = arb.official_calce()
    _, _, df_all, minv, maxv = arb.build_frames(series, sp)
    cyc = df_all["Cycle"].values
    cap_mm = df_all["Capacity"].values.astype(np.float64)
    i_sp = int(np.where(cyc == sp)[0][0])
    _, _, _, _, diag = arb.teacher_forced(model, df_all, minv, maxv, sp, i_sp,
                                          "pf")
    sc_m, sc_s = diag["scaler_mean"], diag["scaler_scale"]
    print(f"  live ckpt = {os.path.relpath(ckpt, R.REPO)}\n")

    def to_ah(q_mm):
        return np.asarray(q_mm) * (maxv - minv) * 1.1 + minv * 1.1

    for k, t in ((25, 615), (100, 540)):
        start = t + 1
        i0 = int(np.where(cyc == start)[0][0])
        full, _ = arb.rollout(model, df_all, minv, maxv, start, sc_m, sc_s,
                              max_steps=len(cap_mm) - i0, name="pf")
        live = to_ah(full)
        for fn in ("ar_traj_calce.npz", "ar_traj_calce__sp500.npz"):
            path = os.path.join(SRC, "results", fn)
            if not os.path.exists(path):
                print(f"  k={k:>3} {fn:<28} (missing)")
                continue
            with np.load(path) as z:
                a = z[f"pred_pf_k{k}_s{run}"]
            v = np.where(~np.isnan(a))[0]
            j0 = int(v[0])
            n = min(len(live), len(v))
            d = float(np.abs(a[j0:j0 + n] - live[:n]).max())
            verdict = "SAME as live SP500" if d < 1e-6 else "DIFFERENT"
            print(f"  k={k:>3} {fn:<28} max|stored - live| = {d:.3e} Ah  "
                  f"{verdict}")
    print()


def main() -> None:
    c3 = control_c3()
    control_c4()
    control_c5()
    control_c6()
    print("=" * 78)
    print(f"  C3 plumbing {'PASS' if c3 else 'FAIL'}")
    print("=" * 78)
    if not c3:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
