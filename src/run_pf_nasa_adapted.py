"""Self-run of the PatchFormer baseline (5 datasets) — official
pipeline composition.

For NASA and TJU the ingestion uses the upstream repo's official
readers (NASADataPreProcess / TJUDataPreProcess / CALCEDataPreProcess);
for PANASONIC and MIT the upstream repo ships no data, so our own
well-tested loaders (checkpoints/data_cache/load_series_*.pkl) provide
the per-cell capacity series — the training/evaluation pipeline is the
official one in every case (pytorch_forecasting TimeSeriesDataSet +
PatchFormerNetModel, per-window EncoderNormalizer targets, SMAPE,
lr 1e-3, early stopping, non-recursive full-trajectory evaluation,
official rul_value_error).

Per-SP protocol (uniform, matches the paper): train = other cells in
full + test-cell cycles before the start point; min--max from the
train part; df_test truncated at Cycle >= start - seq_len; actuals
measured from Cycle >= start.  The group_id column is added the way
CALCEDataPreProcess hardcodes it (TimeSeriesDataSet requires it).

Run with the `patchformer` conda env:

  python src/run_pf_nasa_adapted.py --dataset {nasa,tju,calce,panasonic,mit} --count 10

Outputs (in ref_patchformer, official layout):
  results_{out_dir}/<test>/PatchFormer/...  (per-run logs)
  results/pf_{ds}_selfrun_*.json            (per-SP metrics, resumable)
"""
import argparse
import functools
import json
import os
import pickle
import random
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

# same torch.load shim as the official NASA runner (pickled checkpoints)
_torch_original_load = torch.load


def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _torch_original_load(*a, **kw)


torch.load = _patched_load

REPO = r"D:\research\degradation_prognostics\Transformer_and_Multi_Scale_Models\reference_repos\ref_patchformer"
PROOT = r"D:\research\degradation_prognostics\Transformer_and_Multi_Scale_Models"
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, PROOT)

DATASETS = {
    "nasa": dict(
        rated=2.0, seql=30, batch=128, sps=[50, 70, 90], test="B0005",
        eol_frac=0.7, out_dir="results_RUL_prediction_sl_30",
        json_out="results/pf_nasa_selfrun_B0005.json",
        gid={"B0005": 0, "B0006": 1, "B0007": 2, "B0018": 3},
        batteries=["B0005", "B0006", "B0007", "B0018"],
    ),
    "tju": dict(
        rated=2.5, seql=64, batch=128, sps=[200, 300, 400], test="CY25_1",
        eol_frac=0.7, out_dir="results_TJU_RUL_prediction_sl_64",
        json_out="results/pf_tju_selfrun_CY25_1.json",
        gid={"CY25_1": 0, "CY25_2": 1, "CY25_3": 2},
        batteries=["CY25_1", "CY25_2", "CY25_3"],
    ),
    "calce": dict(
        rated=1.1, seql=64, batch=128, sps=[300, 400, 500], test="CS2_35",
        eol_frac=0.7, out_dir="results_CALCE_RUL_prediction_sl_64",
        json_out="results/pf_calce_selfrun_CS2_35.json",
        gid={"CS2_35": 0, "CS2_36": 1, "CS2_37": 2, "CS2_38": 3},
        batteries=["CS2_35", "CS2_36", "CS2_37", "CS2_38"],
    ),
    "panasonic": dict(
        rated=3.03, seql=30, batch=128, sps=[300, 400, 500], test="Cell03",
        eol_frac=0.7, out_dir="results_PANASONIC_RUL_prediction_sl_30",
        json_out="results/pf_panasonic_selfrun_Cell03.json",
        gid={}, rids=None, batteries=None,
    ),
    "mit": dict(
        rated=1.075, seql=64, batch=128, sps=[200, 300, 400], test="batch2_cell5",
        eol_frac=0.8, out_dir="results_MIT_RUL_prediction_sl_64",
        json_out="results/pf_mit_selfrun_batch2_cell5.json",
        gid={}, rids=None, batteries=None,
    ),
}


@functools.lru_cache(maxsize=None)
def _series_dict(ds):
    """{name: DataFrame(Cycle 1..N, Capacity in Ah)} per dataset."""
    if ds == "nasa":
        old = list(sys.argv)
        sys.argv = ["NASADataPreProcess.py"]
        try:
            import NASADataPreProcess as mod
        finally:
            sys.argv = old
        rows = pd.DataFrame(mod.BatteryData,
                            columns=["BatteryName", "Cycle", "Capacity"])
        return {n: g.reset_index(drop=True)[["Cycle", "Capacity"]]
                for n, g in rows.groupby("BatteryName")}
    if ds == "tju":
        import assistant
        assistant.get_gpus_memory_info = lambda: (0, None)
        import TJUDataPreProcess as mod
        Data = mod.BatteryDataRead(
            SimpleNamespace(Rated_Capacity=2.5, seq_len=64))
        return {n: g[["Cycle", "Capacity"]].reset_index(drop=True)
                for n, g in Data.items()}
    if ds == "calce":
        import CALCEDataPreProcess as mod
        Data = mod.BatteryDataRead(
            ["CS2_35", "CS2_36", "CS2_37", "CS2_38"], "data/CALCE data/")
        return {n: g[["Cycle", "Capacity"]].reset_index(drop=True)
                for n, g in Data.items()}
    # panasonic / mit: our load_series cache (per-cell Ah series)
    with open(os.path.join(PROOT, "checkpoints", "data_cache",
                           f"load_series_{ds}.pkl"), "rb") as f:
        caps, _tr, _te, _w, _sps, _eol = pickle.load(f)
    out = {}
    for n, arr in caps.items():
        df = pd.DataFrame({"Cycle": np.arange(1, len(arr) + 1),
                           "Capacity": arr.astype(np.float64)})
        out[n] = df
    return out


def prepare_cfg(ds):
    """Resolve gid/ratings from the series dict for pkl-based datasets."""
    cfg = dict(DATASETS[ds])
    if cfg.get("batteries") is None:
        s = _series_dict(ds)
        names = list(s.keys())
        cfg["batteries"] = names
        cfg["gid"] = {n: i for i, n in enumerate(names)}
    return cfg


def rul_value_error(y_test, y_predict, threshold):
    """Copied verbatim from RUL_Prediction_PatchFormer_NASA.py."""
    true_re, pred_re = len(y_test), 0
    for i in range(len(y_test) - 1):
        if y_test[i] <= threshold >= y_test[i + 1]:
            true_re = i - 1
            break
    for i in range(len(y_predict) - 1):
        if y_predict[i] <= threshold:
            pred_re = i - 1
            break
    rul_real = true_re + 1
    rul_pred = pred_re + 1
    ae_error = abs(true_re - pred_re)
    re_score = abs(true_re - pred_re) / true_re
    if re_score > 1:
        re_score = 1
    return rul_real, rul_pred, ae_error, re_score


def build_dfs(cfg, start_point):
    """Official DataProcess semantics + group_id (see module docstring)."""
    s = _series_dict(cfg["_ds"])
    base = pd.concat([pd.DataFrame(
        {"BatteryName": n, "Cycle": g["Cycle"], "Capacity": g["Capacity"]})
        for n, g in s.items()]).reset_index(drop=True)
    df = base.copy()
    df["Capacity"] /= cfg["rated"]
    df["target"] = df["Capacity"]
    df["constant"] = df["Capacity"] * 0

    in_test = df["BatteryName"] == cfg["test"]
    train_rows = df[(~in_test) | (in_test & (df["Cycle"] < start_point))]
    test_rows = df[in_test]
    minv = train_rows["Capacity"].min()
    maxv = train_rows["Capacity"].max()

    def prep(rows):
        r = rows[["Cycle", "Capacity", "target", "constant"]].copy()
        r["time_idx"] = r["Cycle"].map(lambda x: int(x - 1))
        r["group_id"] = rows["BatteryName"].map(cfg["gid"])
        r["Capacity"] = (r["Capacity"] - minv) / (maxv - minv)
        r["idx"] = [x for x in range(len(r))]
        r.set_index("idx", inplace=True)
        return r

    df_train = prep(train_rows)
    df_test_all = prep(test_rows)
    df_test = df_test_all.loc[
        df_test_all["Cycle"] >= start_point - cfg["seql"],
        ["time_idx", "group_id", "Cycle", "Capacity", "target", "constant"]]
    return df_train, df_test, df_test_all


def make_dataset(df, batch_size, shuffle, drop_last, seql):
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data.encoders import EncoderNormalizer

    t = TimeSeriesDataSet(
        df, time_idx="time_idx", target="target", group_ids=["group_id"],
        min_encoder_length=seql, max_encoder_length=seql,
        min_prediction_length=1, max_prediction_length=1,
        time_varying_known_reals=["Capacity"],
        time_varying_unknown_reals=["target"],
        target_normalizer=EncoderNormalizer(),
        add_encoder_length=False)
    return t, t.to_dataloader(train=shuffle, batch_size=batch_size,
                              shuffle=shuffle, num_workers=0,
                              drop_last=drop_last)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nasa", choices=list(DATASETS))
    ap.add_argument("--count", type=int, default=10,
                    help="independent runs (official seeds 1..count)")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--max_epochs", type=int, default=200)
    ap.add_argument("--suffix", type=str, default="",
                    help="suffix to isolate outputs (e.g. b16)")
    ap.add_argument("--check-data", action="store_true",
                    help="build datasets and stop before training")
    args = ap.parse_args()

    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping
    from pytorch_forecasting.metrics import SMAPE
    from ModelsModify.PatchFormer import PatchFormerNetModel

    cfg = prepare_cfg(args.dataset)
    cfg["_ds"] = args.dataset
    if args.batch_size is not None:
        cfg = dict(cfg, batch=args.batch_size)
    if args.suffix:
        cfg = dict(cfg,
                   json_out=cfg["json_out"].replace(".json", f"_{args.suffix}.json"),
                   out_dir=f'{cfg["out_dir"]}_{args.suffix}')
    for name in cfg["batteries"]:
        assert name in _series_dict(args.dataset), name + " missing"

    out_dir = os.path.join(cfg["out_dir"], cfg["test"], "PatchFormer")
    os.makedirs(out_dir, exist_ok=True)

    summary = {}
    if not args.check_data and os.path.exists(cfg["json_out"]):
        try:
            with open(cfg["json_out"], "r", encoding="utf-8") as f:
                summary = json.load(f)
        except (json.JSONDecodeError, ValueError):
            summary = {}
    for sp in cfg["sps"]:
        df_train, df_test, df_all = build_dfs(cfg, sp)
        mask_len = len(df_train)
        if args.check_data:
            print(f"SP{sp}: train={len(df_train)} test={len(df_test)} "
                  f"all={len(df_all)} gid={sorted(df_train['group_id'].unique())}")
            continue

        training, train_dl = make_dataset(df_train[0:int(0.8 * mask_len)],
                                          cfg["batch"], True, True, cfg["seql"])
        validing, val_dl = make_dataset(df_train[int(0.8 * mask_len):],
                                        cfg["batch"], False, False, cfg["seql"])
        testing, test_dl = make_dataset(df_test, cfg["batch"], False, False,
                                        cfg["seql"])

        sp_root = os.path.join(out_dir, f"SP{sp}")
        os.makedirs(sp_root, exist_ok=True)

        count = 0
        per_run = summary.get(f"SP{sp}", {}).get("runs", [])

        while count < args.count:
            count += 1
            seed = count
            if any(r.get("seed") == seed for r in per_run):
                print(f"[SP{sp} run{seed}] SKIP (already in json)", flush=True)
                continue
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            model = PatchFormerNetModel.from_dataset(
                training, patch_len=2, seq_len=cfg["seql"], pred_len=1,
                enc_in=1, d_model=16, learning_rate=0.001, loss=SMAPE())

            save_dir = os.path.join(sp_root, f"run{seed}")
            os.makedirs(save_dir, exist_ok=True)
            early_stop = EarlyStopping(monitor="val_loss", min_delta=1e-5,
                                       patience=10, verbose=False, mode="min")
            trainer = pl.Trainer(
                max_epochs=args.max_epochs, accelerator="gpu", devices=1,
                gradient_clip_val=0.2, callbacks=[early_stop],
                logger=False, default_root_dir=save_dir)
            trainer.fit(model, train_dataloaders=train_dl,
                        val_dataloaders=val_dl)

            best_path = trainer.checkpoint_callback.best_model_path
            print(f"[SP{sp} run{seed}] best: {best_path}", flush=True)
            best_model = PatchFormerNetModel.load_from_checkpoint(best_path).cuda()

            preds = best_model.predict(test_dl, batch_size=256)
            preds = preds.detach().cpu().numpy().reshape(-1)
            actuals_df = df_all.loc[df_all["Cycle"] >= sp, ["Cycle", "target"]]
            actuals = actuals_df["target"].values
            y_true = actuals * cfg["rated"]
            y_pred = preds * cfg["rated"]
            mask = y_true >= 0.

            from sklearn.metrics import r2_score
            mae = float(np.mean(np.abs(y_true[mask] - y_pred[mask])))
            rmse = float(np.sqrt(np.mean(np.square(y_true[mask] - y_pred[mask]))))
            r2 = float(r2_score(y_true[mask], y_pred[mask]))
            r_real, r_pred, ae, re = rul_value_error(
                y_true[mask], y_pred[mask],
                threshold=cfg["rated"] * cfg["eol_frac"])
            per_run.append({"seed": seed, "mae": mae, "rmse": rmse, "r2": r2,
                            "rul_real": int(r_real), "rul_pred": int(r_pred),
                            "ae": int(ae), "re": float(re)})
            print(f"[SP{sp} run{seed}] MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f} "
                  f"RUL={r_real}/{r_pred} AE={ae} RE={re:.4f}", flush=True)
            summary[f"SP{sp}"] = {"runs": per_run}
            with open(cfg["json_out"], "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            del model, best_model

        n_done = len(per_run)
        avg = {k: float(np.mean([r[k] for r in per_run]))
               for k in ("mae", "rmse", "r2", "ae", "re")}
        avg["rul_real"] = float(np.mean([r["rul_real"] for r in per_run]))
        avg["rul_pred"] = float(np.mean([r["rul_pred"] for r in per_run]))
        summary[f"SP{sp}"] = {"runs": per_run, "avg": avg}
        print(f"[SP{sp}] avg({n_done} runs) MAE={avg['mae']:.4f} "
              f"RMSE={avg['rmse']:.4f} R2={avg['r2']:.4f} "
              f"AE={avg['ae']:.2f}", flush=True)

    if not args.check_data:
        with open(cfg["json_out"], "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"saved {cfg['json_out']}", flush=True)


if __name__ == "__main__":
    main()
