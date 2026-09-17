"""TJU and NASA at their native W: Ours, PatchFormer and RUL-Mamba.

Why a separate file.  `ar_rollout_baselines.py` hardcodes CALCE -- TEST_CELL /
GID / RATED / SEQL / EOL_AH at lines 51-53, both series loaders at 70-107, both
checkpoint locators at 190-221.  Nothing here edits it: the module is imported
and those five globals are re-bound per dataset before any of its functions run
(Python resolves module globals at call time, so `build_frames`, `teacher_forced`
and `rollout` pick the new values up), and only the genuinely dataset-specific
parts are re-implemented -- the two series conventions and the two checkpoint
locators.

Both datasets are already trained for all three models; this is pure inference.

  ds     W    SP   test    EOL   series   PF root                          RM dir
  tju    64   200  CY25_1   778   886     results_TJU_RUL_prediction_sl_64  TJU
  nasa   30    50  B0005    124   168     results_RUL_prediction_sl_30      NASA

Two series conventions per dataset, kept separate exactly as on CALCE:
  * PF decodes in ITS OWN space (its own preprocessing -- TJU drops 3-sigma
    outliers, NASA keeps discharge cycles only) so it is decoded in the frame
    it was fitted in
  * RM and Ours run on OUR series (`load_series`)

Metrics are fixed here rather than left to each caller:
  * AE is the crossing-cycle error, always reported with AE/k.  A rollout that
    never crosses has NO AE -- it prints `-` and is never averaged in: a
    non-crossing "AE" is identically the launch-to-EOL distance and would be
    read as an accuracy change.
  * MAE / MSE are taken over the horizon the prediction is used on, launch to
    TRUE EOL (`*_eol`), not the whole 1000-step rollout -- past EOL the rollout
    is unconstrained and dilutes the number.  `*@50` is the fixed-horizon
    complement.
  * A launch point at or before SP is SKIPPED, not reported: `build_frames`
    puts the test cell's cycles < SP into training, so a warm-up window there
    would be reading training data.  This is what rules out EOL-100 on NASA
    (t0=24 < SP=50).

Read-only w.r.t. checkpoints.  Writes only results/ar_baselines_tju.json.

    D:/anaconda/envs/py312/python.exe src/ar_baselines_tju.py --ds tju --runs 1 2 3
    D:/anaconda/envs/py312/python.exe src/ar_baselines_tju.py --ds nasa --runs 1 2 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)
PROOT = os.path.dirname(SRC)

import ar_rollout_baselines as B                            # noqa: E402

OUT = os.path.join(SRC, "results", "ar_baselines_tju.json")

CFG = {
    "tju": dict(
        sp=200, test="CY25_1",
        cells={"CY25_1": 0, "CY25_2": 1, "CY25_3": 2},
        rated=2.5, seql=64, eol_ah=1.75, eol_frac=0.7,
        pf_root="results_TJU_RUL_prediction_sl_64", rm_ds="TJU",
        ks=(50, 100), start="CY25-05_1"),
    "nasa": dict(
        sp=50, test="B0005",
        cells={"B0005": 0, "B0006": 1, "B0007": 2, "B0018": 3},
        rated=2.0, seql=30, eol_ah=1.4, eol_frac=0.7,
        pf_root="results_RUL_prediction_sl_30", rm_ds="NASA",
        ks=(20, 30, 40, 50, 60), start=None),
    "calce": dict(
        sp=300, test="CS2_35",
        cells={"CS2_35": 0, "CS2_36": 1, "CS2_37": 2, "CS2_38": 3},
        rated=1.1, seql=64, eol_ah=0.77, eol_frac=0.7,
        pf_root="results_CALCE_RUL_prediction_sl_64", rm_ds="CALCE",
        ks=(50, 100), start=None),
    # MIT: PF and RM were both trained on OUR load_series split (see
    # run_pf_nasa_adapted.py -- there is no upstream MIT preprocessor), so both
    # use ours_series() here.  Only batch2_cell5 has baseline checkpoints;
    # batch2_cell47 is a second test cell for Ours alone.
    "mit": dict(
        sp=200, test="batch2_cell5", cells=None,
        rated=1.075, seql=64, eol_ah=0.86, eol_frac=0.8,
        pf_root="results_MIT_RUL_prediction_sl_64", rm_ds="MIT",
        ks=(50, 100), start=None),
}

# bound in configure(); score() reads EOL_AH, so it has to be module level
DS = "tju"
SP = 200
TEST_CELL = "CY25_1"
SEQL, RATED, EOL_AH = 64, 2.5, 1.75


def configure(ds: str) -> dict:
    """Bind the shared harness to `ds`, and the module globals score() uses."""
    global DS, SP, TEST_CELL, SEQL, RATED, EOL_AH
    c = CFG[ds]
    DS, SP, TEST_CELL = ds, c["sp"], c["test"]
    SEQL, RATED, EOL_AH = c["seql"], c["rated"], c["eol_ah"]
    cells = c["cells"]
    if cells is None:                       # derive from the series we will use
        from make_figures import load_series

        cells = {n: i for i, n in enumerate(sorted(load_series(ds)[0]))}
    B.TEST_CELL = c["test"]
    B.GID = cells
    B.RATED, B.SEQL, B.EOL_AH, B.EOL_FRAC = (c["rated"], c["seql"], c["eol_ah"],
                                             c["eol_frac"])
    return c


# ── the two series conventions ──────────────────────────────────────────
def _stub_assistant() -> None:
    """Neutralise the reference repo's import-time GPU probe.

    Both `TJUDataPreProcess` and `NASADataPreProcess` call
    `assistant.get_gpus_memory_info()` at IMPORT time to pick a device, and
    that parses `nvidia-smi` -- which reports one of the fields as `N/A` here,
    so the parse raises before any data is read.  The device is already chosen
    by this process; the only thing these paths need is the reader.
    """
    import types

    stub = types.ModuleType("assistant")
    stub.get_gpus_memory_info = lambda: (0, 0)
    sys.modules["assistant"] = stub


def official_tju():
    """The TJU series PatchFormer was trained on.

    `TJUDataPreProcess.BatteryDataRead` reads `data/TJU data/
    Dataset_3_NCM_NCA_battery/`, keeps CY25-05_1 #1..#3, and drops every row
    beyond 3 sigma on any column (`delete_3_sigma`), renumbering Cycle.  That
    outlier removal is why this is not the series our `load_series` returns,
    and PF must be decoded in the frame it was fitted in.  `args` is only
    stored by `DF`, never read on this path.
    """
    import argparse as _ap

    _stub_assistant()
    cwd = os.getcwd()
    os.chdir(B.REF_PF)
    sys.path.insert(0, B.REF_PF)
    try:
        import TJUDataPreProcess as mod
        data = mod.BatteryDataRead(_ap.Namespace(seq_len=SEQL))
        out = {n: g[["Cycle", "Capacity"]].to_numpy(dtype=np.float64)
               for n, g in data.items()}
    finally:
        os.chdir(cwd)
        if B.REF_PF in sys.path:
            sys.path.remove(B.REF_PF)
    if TEST_CELL not in out:
        raise SystemExit("official TJU series has no %s: %s"
                         % (TEST_CELL, sorted(out)))
    return out


def official_nasa():
    """The NASA series PatchFormer was trained on.

    `NASADataPreProcess.DataRead` keeps the discharge entries only and numbers
    them 1..N per cell.  It returns raw Ah, which is what `build_frames` wants
    (it divides by RATED itself).
    """
    import pandas as pd

    _stub_assistant()
    cwd = os.getcwd()
    saved_argv = sys.argv
    # NASADataPreProcess parses sys.argv at IMPORT time with its own module-level
    # parser, so our --ds/--models/... would abort it.  Hand it a bare argv; the
    # defaults it then captures (Battery_list, data_dir) are what we want, and
    # they stay on `mod.args` after argv is restored.
    sys.argv = saved_argv[:1]
    os.chdir(B.REF_PF)
    sys.path.insert(0, B.REF_PF)
    try:
        import NASADataPreProcess as mod
        raw = mod.DataRead(mod.args.Battery_list, mod.args.data_dir)
    finally:
        sys.argv = saved_argv
        os.chdir(cwd)
        if B.REF_PF in sys.path:
            sys.path.remove(B.REF_PF)
    df = pd.DataFrame(raw, columns=["BatteryName", "Cycle", "Capacity"])
    out = {n: g[["Cycle", "Capacity"]].to_numpy(dtype=np.float64)
           for n, g in df.groupby("BatteryName")}
    if TEST_CELL not in out:
        raise SystemExit("official NASA series has no %s: %s"
                         % (TEST_CELL, sorted(out)))
    return out


def official_series() -> dict:
    """PF's own frame for this dataset; CALCE's loader already lives in B.

    MIT has no upstream preprocessor: PF's MIT checkpoints were trained on our
    `load_series` split (run_pf_nasa_adapted.py), so PF uses the same series
    RM and Ours do.
    """
    if DS == "tju":
        return official_tju()
    if DS == "nasa":
        return official_nasa()
    if DS == "calce":
        return B.official_calce()
    return ours_series()


def ours_series() -> dict:
    """Our `load_series` series -- what RUL-Mamba and Ours use."""
    from make_figures import load_series

    caps, _tr, _te, _W, _sps, _eol = load_series(DS)
    return {n: np.stack([np.arange(1, len(a) + 1),
                         np.asarray(a, np.float64)], axis=1)
            for n, a in caps.items()}


# ── the two checkpoint locators ─────────────────────────────────────────
def load_pf(sp: int, run: int):
    import torch

    ckpt_dir = os.path.join(B.REF_PF, CFG[DS]["pf_root"], TEST_CELL,
                            "PatchFormer", "SP%d" % sp, "run%d" % run,
                            "checkpoints")
    ckpts = [os.path.join(ckpt_dir, f) for f in sorted(os.listdir(ckpt_dir))
             if f.endswith(".ckpt")]
    if not ckpts:
        raise FileNotFoundError(ckpt_dir)
    os.chdir(B.REF_PF)
    sys.path.insert(0, B.REF_PF)
    from ModelsModify.PatchFormer import PatchFormerNetModel
    model = PatchFormerNetModel.load_from_checkpoint(ckpts[0]).to(B.DEV).eval()
    torch.cuda.empty_cache()
    return model, ckpts[0]


def load_rm(sp: int, run: int):
    import torch

    ckpt_dir = os.path.join(B.REF_RM, "Outputs", CFG[DS]["rm_ds"],
                            "Univariable", "RULMamba", "Repeat_%d" % run,
                            "Start_Point_%d" % sp, "Checkpoints")
    ckpts = [os.path.join(ckpt_dir, f) for f in sorted(os.listdir(ckpt_dir))
             if f.endswith(".ckpt")]
    if not ckpts:
        raise FileNotFoundError(ckpt_dir)
    os.chdir(B.REF_RM)
    sys.path.insert(0, B.REF_RM)
    from Models.RULMamba import RULMambaNetModel
    model = RULMambaNetModel.load_from_checkpoint(ckpts[0]).to(B.DEV).eval()
    torch.cuda.empty_cache()
    return model, ckpts[0]


# ── metrics ─────────────────────────────────────────────────────────────
def launches(c: dict, eol: int):
    """[(label, launch index)] for this dataset.

    `launch="sp"` starts at the protocol's own SP -- the paper's evaluation
    point, and the setting a reviewer expects.  The default is EOL-k, which
    makes the horizon explicit and comparable across datasets but is OUR
    choice, not the paper's.
    """
    if c.get("launch") == "sp":
        return [("SP%d" % SP, SP)]
    return [("EOL-%d" % k, eol - k) for k in c["ks"]]


def cross_index(series: np.ndarray, thr: float) -> int:
    """First j with series[j] >= thr > series[j+1], else -1.

    Index j means the crossing happens between cycle `t0+j` and `t0+j+1`, so
    the crossing cycle is `t0+j+1` -- the ar_baselines_eolk convention.
    """
    for j in range(len(series) - 1):
        if series[j] >= thr > series[j + 1]:
            return j
    return -1


def score(pred_ah: np.ndarray, truth_ah: np.ndarray, t0: int, eol_cycle: int,
          k: int) -> dict:
    """All reported metrics for one rollout, capacities in Ah.

    `pred_ah[j]` / `truth_ah[j]` are cycle `t0+j` (0-based index into the
    series; cycle numbering is 1-based).
    """
    n = min(len(pred_ah), len(truth_ah))
    p, t = pred_ah[:n], truth_ah[:n]
    ci = cross_index(p, EOL_AH)
    if ci < 0 and n and p[0] < EOL_AH:
        # already below the threshold at the first predicted cycle: the
        # crossing happened at or before it.  `cross_index` cannot see this
        # (it needs series[j] >= thr), and recording it as "never crossed"
        # would read as a freeze when the rollout actually collapsed.
        ci = 0
    ti = cross_index(t, EOL_AH)
    if ci < 0:
        ae, crossed = None, False
    else:
        true_cycle = (t0 + ti + 1) if ti >= 0 else eol_cycle
        ae, crossed = float(abs((t0 + ci + 1) - true_cycle)), True
    n_eol = int(min(n, max(1, eol_cycle - t0)))        # launch .. true EOL
    n50 = min(50, n)
    # The predicted capacity AT the true EOL step.  This one cannot be gamed by
    # not moving: a frozen rollout's MAE over the horizon is pinned near half
    # the true decline (the floor), so it reads as "accurate" -- this column
    # shows what it costs at the point that actually matters.
    i_eol = int(min(n - 1, max(0, eol_cycle - t0)))
    return dict(
        ae=ae, ae_lbl=("-" if ae is None else "%.0f" % ae),
        ae_over_k=(None if ae is None else ae / k), crossed=crossed,
        final=float(p[-1]), cap_at_eol=float(p[i_eol]),
        true_at_eol=float(EOL_AH), n=int(n),
        mae_eol=float(np.mean(np.abs(p[:n_eol] - t[:n_eol]))),
        mse_eol=float(np.mean((p[:n_eol] - t[:n_eol]) ** 2)),
        mae50=float(np.mean(np.abs(p[:n50] - t[:n50]))),
        mse50=float(np.mean((p[:n50] - t[:n50]) ** 2)))


def hdr() -> None:
    print("  %-6s %-6s %-8s %6s %7s %8s %9s %9s %9s"
          % ("model", "run", "launch", "t0", "AE", "AE/k", "MAE_eol",
             "MSE_eol", "cap@EOL"))


def show(tag: str, run: int, label: str, t0: int, m: dict) -> None:
    print("  %-6s %-6d %-8s %6d %7s %8s %9.3e %9.3e %9.4f"
          % (tag, run, label, t0, m["ae_lbl"],
             ("%.2f" % m["ae_over_k"]) if m["ae_over_k"] is not None else "-",
             m["mae_eol"], m["mse_eol"], m["cap_at_eol"]), flush=True)


# ── the two baseline rollouts ───────────────────────────────────────────
def run_baseline(name: str, runs, ks, steps: int) -> dict:
    series = official_series() if name == "pf" else ours_series()
    loader = load_pf if name == "pf" else load_rm
    cap_cyc = series[TEST_CELL]
    eol = B.first_crossing(cap_cyc[:, 1], EOL_AH) + 1
    if eol <= 0:
        raise SystemExit("test cell never crosses the EOL threshold")
    print("\n  === %s   test=%s  EOL=%d  rated=%.2f  EOL_AH=%.2f  seql=%d ==="
          % (name.upper(), TEST_CELL, eol, RATED, EOL_AH, SEQL))
    hdr()
    out = {}
    for run in runs:
        try:
            model, _ck = loader(SP, run)
        except (FileNotFoundError, IndexError, OSError) as e:
            print("    run%d: NO CHECKPOINT (%s)" % (run, e))
            continue
        _df_tr, _df_te, df_all, minv, maxv = B.build_frames(series, SP)
        i_sp = int(np.where(df_all["Cycle"].values == SP)[0][0])
        _yt, _yp, _man, _off, diag = B.teacher_forced(
            model, df_all, minv, maxv, SP, i_sp, name)
        sc_m, sc_s = diag["scaler_mean"], diag["scaler_scale"]
        # `B.rollout` returns the model's OWN min--max space (the `p_mm` it
        # appends to the window), NOT Ah.  Invert the frame's map first, then
        # the per-rated scaling `build_frames` applied.
        def to_ah(v):
            return (np.asarray(v, dtype=np.float64) * (maxv - minv)
                    + minv) * RATED

        tv_all = to_ah(df_all["Capacity"].values)
        rows = []
        for label, t0 in launches(CFG[DS], eol):
            if t0 <= SEQL or t0 >= len(tv_all):
                print("    %s skipped (t0=%d outside series)" % (label, t0))
                continue
            if t0 < SP:
                print("    %s SKIPPED: t0=%d < SP=%d, the warm-up window "
                      "is training data" % (label, t0, SP))
                continue
            try:
                preds, i_launch = B.rollout(model, df_all, minv, maxv, t0,
                                            sc_m, sc_s, max_steps=steps,
                                            name=name)
            except ValueError as e:
                print("    %s skipped (%s)" % (label, e))
                continue
            p_ah = to_ah(preds)
            t_ah = tv_all[i_launch:i_launch + len(preds)]
            k = max(1, eol - t0)                    # horizon to EOL, for AE/k
            m = score(p_ah, t_ah, i_launch, eol, k)
            show(name, run, label, i_launch, m)
            rows.append(dict(launch_k=k, launch=label, **m))
        out["run%d" % run] = rows
        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
    return out


# ── ours ────────────────────────────────────────────────────────────────
def run_ours(seeds, ks, steps: int) -> dict:
    """Ours, rolled out through the same metric definition.

    `rollout_multi` and the checkpoint root come from ar_three_ds_neareol /
    ar_three_datasets, imported rather than copied so the rollout stays the one
    those results were produced with.
    """
    import torch
    from ar_three_datasets import pick_root
    from ar_three_ds_neareol import rollout_multi
    from gdn_model import build_gdn_model
    from make_figures import load_series
    from train_per_sp import DEV, EPS

    caps, _tr, _te, _W, _sps, eol_ah = load_series(DS)
    root = pick_root(DS, SP)
    print("\n  === OURS   test=%s  SP%d  W=64  root=%s ==="
          % (TEST_CELL, SP, root))
    hdr()
    out = {}
    for seed in seeds:
        path = os.path.join(root, DS, "SP%d_seed%d.pt" % (SP, seed))
        if not os.path.exists(path):
            print("    seed%d: NO CHECKPOINT (%s)" % (seed, path))
            continue
        ck = torch.load(path, map_location=DEV, weights_only=False)
        W, lo, hi = int(ck["W"]), float(ck["lo"]), float(ck["hi"])
        mode = ck.get("tgt_mode", "zscore")
        tcs = [c for c in ck.get("test_cells", []) if c in caps] or [TEST_CELL]
        seqs = [(np.asarray(caps[c], np.float64) - lo) / (hi - lo + EPS)
                for c in tcs]
        thr_n = (eol_ah - lo) / (hi - lo + EPS)
        eols = [cross_index(s, thr_n) + 1 for s in seqs]
        wins, meta = [], []
        for i, c in enumerate(tcs):
            for label, t0 in launches(CFG[DS], eols[i]):
                if t0 < W:
                    continue
                wins.append(seqs[i][t0 - W:t0])
                meta.append((c, label, t0, i))
        if not wins:
            print("    seed%d: no usable launch point" % seed)
            continue
        model = build_gdn_model(multiscale=True, stage_query=True, input_dim=1,
                                window_size=W, output_len=1,
                                readout="last").to(DEV)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        full = rollout_multi(model, wins, steps, mode)
        rows = []
        for j, (c, label, t0, i) in enumerate(meta):
            s = seqs[i]
            # back to Ah so every model in the table is in the same units
            p_ah = full[j] * (hi - lo + EPS) + lo
            n_true = int(min(steps, len(s) - t0))
            t_ah = s[t0:t0 + n_true] * (hi - lo + EPS) + lo
            k = max(1, eols[i] - t0)                # horizon to EOL, for AE/k
            m = score(p_ah, t_ah, t0, eols[i], k)
            show("ours", seed, label, t0, m)
            rows.append(dict(cell=c, launch_k=k, launch=label, **m))
        out["run%d" % seed] = rows
        del model
        torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="tju", choices=sorted(CFG))
    ap.add_argument("--models", nargs="+", default=["ours", "pf", "rm"],
                    choices=["ours", "pf", "rm"])
    ap.add_argument("--runs", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--ks", type=int, nargs="+", default=None)
    ap.add_argument("--steps", type=int, default=1000)
    args = ap.parse_args()

    c = configure(args.ds)
    ks = args.ks if args.ks else list(c["ks"])

    print("=" * 104)
    print("  %s  W=%d  SP=%d  test=%s  rated=%.2f Ah  EOL_AH=%.2f Ah"
          % (args.ds.upper(), SEQL, SP, TEST_CELL, RATED, EOL_AH))
    mode = ("SP%d" % SP if c.get("launch") == "sp"
            else "EOL-" + "/".join(str(k) for k in ks))
    print("  runs=%s  launch=%s  steps=%d" % (args.runs, mode, args.steps))
    print("  AE: a non-crossing rollout prints '-' and is NEVER averaged in")
    print("=" * 104, flush=True)

    res = {}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as fh:
            res = json.load(fh)
    got = {}
    if "ours" in args.models:
        got["ours"] = run_ours(args.runs, ks, args.steps)
    for name in ("pf", "rm"):
        if name in args.models:
            got[name] = run_baseline(name, args.runs, ks, args.steps)
    # merge, do not replace: Ours runs in py312 while the two baselines need the
    # `patchformer` env (pytorch_forecasting lives only there), so the two
    # halves arrive in separate invocations of this script.
    res.setdefault(args.ds, {}).update(got)

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print("\n  -> %s" % OUT)

    print("\n  === %s summary (mean over runs; cap@EOL true value = %.3f Ah) ==="
          % (args.ds.upper(), EOL_AH))
    print("  %-6s %-8s %7s %9s %9s %9s %8s"
          % ("model", "launch", "cross", "MAE_eol", "MSE_eol", "cap@EOL",
             "meanAE"))
    for name in ("ours", "pf", "rm"):
        labels = sorted({r.get("launch", "?")
                         for run in got.get(name, {}).values() for r in run})
        for label in labels:
            rs = [r for run in got.get(name, {}).values()
                  for r in run if r.get("launch") == label]
            if not rs:
                continue
            ncr = sum(1 for r in rs if r["crossed"])
            aes = [r["ae"] for r in rs if r["ae"] is not None]
            print("  %-6s %-8s %4d/%d %9.3e %9.3e %9.4f %8s"
                  % (name, label, ncr, len(rs),
                     float(np.mean([r["mae_eol"] for r in rs])),
                     float(np.mean([r["mse_eol"] for r in rs])),
                     float(np.mean([r["cap_at_eol"] for r in rs])),
                     ("%.1f" % float(np.mean(aes))) if aes else "-"))


if __name__ == "__main__":
    main()
