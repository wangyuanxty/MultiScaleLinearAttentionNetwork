"""Self-run of the PatchFormer baseline (NASA B0005 / TJU CY25_1) —
official pipeline composition.

This driver composes the official code paths of the upstream
PatchFormer repo (ref_patchformer) verbatim:

  * data ingestion : NASADataPreProcess.py / TJUDataPreProcess.py —
    official readers (NASA .mat -> per-cycle discharge capacity;
    TJU csv -> CY25_1/2/3 capacities, 886/904/937 cycles), official
    train/test split & min--max normalization (train = other cells
    + test-cell cycles before the start point, exactly the NASA
    DataProcess rule; the TJU official process omits the per-cell
    prefix, and we add it back with the same rule for protocol
    uniformity across datasets).
  * model + training/eval : RUL_Prediction_PatchFormer_NASA.py —
    pytorch_forecasting TimeSeriesDataSet/PatchFormerNetModel with
    per-window EncoderNormalizer targets, SMAPE loss, lr=1e-3,
    early stopping, non-recursive full-trajectory evaluation and the
    official rul_value_error metric.

The group_id column is added the same way CALCEDataPreProcess
hardcodes it for CALCE (NASA's DataProcess drops BatteryName while
TimeSeriesDataSet requires group_ids=['group_id']); the per-patch
config follows each dataset's official NASA runner args
(seq_len 30/batch 128 NASA, seq_len 64 TJU).

Run with the `patchformer` conda env (pytorch_forecasting):

  D:\\anaconda\\envs\\patchformer\\python.exe src/run_pf_nasa_adapted.py --count 3
  D:\\anaconda\\envs\\patchformer\\python.exe src/run_pf_nasa_adapted.py --dataset tju --count 3

Outputs (in ref_patchformer, official layout):
  results_RUL_prediction_sl_*/<test>/PatchFormer/...  (per-run logs)
  results/pf_{nasa,tju}_selfrun_*.json               (per-SP metrics)
"""
import argparse
import json
import os
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
os.chdir(REPO)
sys.path.insert(0, REPO)

DATASETS = {
    "nasa": dict(
        rated=2.0, seql=30, batch=128, sps=[50, 70, 90], test="B0005",
        out_dir="results_RUL_prediction_sl_30",
        json_out="results/pf_nasa_selfrun_B0005.json",
        gid={"B0005": 0, "B0006": 1, "B0007": 2, "B0018": 3},
        batteries=["B0005", "B0006", "B0007", "B0018"],
    ),
    "tju": dict(
        rated=2.5, seql=64, batch=128, sps=[200, 300, 400], test="CY25_1",
        out_dir="results_TJU_RUL_prediction_sl_64",
        json_out="results/pf_tju_selfrun_CY25_1.json",
        gid={"CY25_1": 0, "CY25_2": 1, "CY25_3": 2},
        batteries=["CY25_1", "CY25_2", "CY25_3"],
    ),
}


def prep_import(ds):
    """Import the official data module for `ds`.

    NASA's module runs its parser + full DataRead at import time, so a
    fake argv takes the official defaults (B0005, [50,70,90], Rated 2.0).
    Both modules run assistant.get_gpus_memory_info() at import to pick
    the GPU; that helper crashes when nvidia-smi reports 'N/A' (e.g.
    while another job holds the GPU), so we stub it to device 0 first."""
    import assistant

    assistant.get_gpus_memory_info = lambda: (0, None)
    if ds == "nasa":
        old = list(sys.argv)
        sys.argv = ["NASADataPreProcess.py"]
        try:
            import NASADataPreProcess as mod
        finally:
            sys.argv = old
        return mod
    import TJUDataPreProcess as mod
    mod.BatteryData = mod.BatteryDataRead(  # official csv reader (runs only
        SimpleNamespace(Rated_Capacity=2.5, seq_len=64))   # in __main__)
    return mod


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


def base_frame(ds, mod):
    """Official per-cycle capacity table for the dataset.

    NASA: flat rows from NASADataPreProcess.DataRead (mat discharge
    cycles).  TJU: dict of per-battery frames from
    TJUDataPreProcess.BatteryDataRead (csv, 3-sigma cleaned).
    """
    if ds == "nasa":
        return pd.DataFrame(mod.BatteryData,
                            columns=["BatteryName", "Cycle", "Capacity"])
    parts = [mod.BatteryData[k][["BatteryName", "Cycle", "Capacity"]]
             for k in mod.BatteryData]
    return pd.concat(parts).reset_index(drop=True)


def build_dfs(ds, mod, cfg, start_point):
    """Official DataProcess semantics + group_id.

    Mirrors NASADataPreProcess.DataProcess (NASA) /
    TJUDataPreProcess.BatteryDataProcess (TJU): Capacity/Rated,
    train-test split by BatteryName, min--max from train only,
    target = Capacity; adds group_id the way
    CALCEDataPreProcess.BatteryDataProcess does, plus the runner's
    df_test truncation (Cycle >= start - seq_len).  The train split
    follows the NASA official rule (other cells + test cycles before
    start_point) for both datasets, so all per-SP rows share one
    protocol.
    """
    base = base_frame(ds, mod)
    df = base.copy()
    df["Capacity"] /= cfg["rated"]
    df["target"] = df["Capacity"]
    df["constant"] = df["Capacity"] * 0  # unused column kept by official proc

    in_test = df["BatteryName"] == cfg["test"]
    train_rows = df[(~in_test)
                    | (in_test & (df["Cycle"] < start_point))]
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
    df_test_all = prep(test_rows)             # full test battery (df_all)
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
    ap.add_argument("--count", type=int, default=3,
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

    cfg = DATASETS[args.dataset]
    if args.batch_size is not None:
        cfg = dict(cfg, batch=args.batch_size)
    if args.suffix:
        cfg = dict(cfg,
                   json_out=cfg["json_out"].replace(".json", f"_{args.suffix}.json"),
                   out_dir=f'{cfg["out_dir"]}_{args.suffix}')

    mod = prep_import(args.dataset)
    if args.dataset == "nasa":
        have = {r[0] for r in mod.BatteryData}
    else:
        have = set(mod.BatteryData.keys())
    for name in cfg["batteries"]:
        assert name in have, name + " missing"

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
        df_train, df_test, df_all = build_dfs(args.dataset, mod, cfg, sp)
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
            r_real, r_pred, ae, re = rul_value_error(y_true[mask], y_pred[mask],
                                                     threshold=cfg["rated"] * 0.7)
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
