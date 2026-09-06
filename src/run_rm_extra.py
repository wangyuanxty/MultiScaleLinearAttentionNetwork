"""RUL-Mamba self-run for datasets outside the upstream repo's
first-party data pipelines (CALCE / PANASONIC / MIT).

The upstream TJU trainer is configuration-driven: dataset cache path,
rated capacity, start points, window and model config all come from
the Base YAML it reads; its only dataset-specific binding is
`from Scripts.Data_Process.TJU_Data_Process import BatteryDataProcess`.
So a new dataset needs only (1) our own Base/model YAMLs (project
side), (2) a cache .npy with the official per-battery structure,
(3) the per-cell prefix rule (same as the official NASA process)
injected by wrapping the upstream TJU data process — exactly the
technique of run_rm_tju_prefix.py.  The model, trainer, early
stopping, and evaluation pipeline are the official ones.  The only
upstream metric deviation (threshold = rated * 0.7 in
evaluate_predictions) is corrected afterwards in our aggregation for
MIT, whose EOL is 80% of rated (Applies to AE/RUL only; MAE/RMSE/R$^2$
are threshold-independent).

Capacity series come from our own load_series caches.

Run with the `patchformer` conda env:

  python src/run_rm_extra.py --dataset calce   # panasonic | mit
"""
import argparse
import importlib
import os
import pickle
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

_orig_load = torch.load


def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)


torch.load = _patched_load

MODELS = {
    "calce": dict(
        rated=1.1, seql=64, sps=[300, 400, 500], test="CS2_35",
        gid={"CS2_35": 0, "CS2_36": 1, "CS2_37": 2, "CS2_38": 3},
        batteries=["CS2_35", "CS2_36", "CS2_37", "CS2_38"],
        name="CALCE", d_model=16, n_dec=2, pkl="calce"),
    "panasonic": dict(
        rated=3.03, seql=30, sps=[300, 400, 500], test="Cell03",
        gid=None, batteries=None, name="PANASONIC",
        d_model=48, n_dec=1, pkl="panasonic"),
    "mit": dict(
        rated=1.075, seql=64, sps=[200, 300, 400], test="batch2_cell5",
        gid=None, batteries=None, name="MIT",
        d_model=16, n_dec=2, pkl="mit"),
}

REF = r"D:\research\degradation_prognostics\Transformer_and_Multi_Scale_Models\reference_repos\ref_rul_mamba"
PROOT = r"D:\research\degradation_prognostics\Transformer_and_Multi_Scale_Models"
CFG_DIR = os.path.join(PROOT, "src", "baseline_cfgs")
CACHE_DIR = os.path.join(PROOT, "checkpoints", "data_cache")


def series(ds):
    with open(os.path.join(CACHE_DIR, f"load_series_{ds}.pkl"), "rb") as f:
        caps, _tr, _te, _w, _sps, _eol = pickle.load(f)
    return caps


def make_cache_npy(ds, cfg):
    caps = series(cfg["pkl"])
    out = {}
    for n, arr in caps.items():
        df = pd.DataFrame({"Cycle": np.arange(1, len(arr) + 1),
                           "Capacity": arr.astype(np.float64)})
        df["BatteryName"] = n
        out[n] = df
    path = os.path.join(PROOT, "checkpoints", f"rm_{ds}_cache.npy")
    np.save(path, np.array([out], dtype=object), allow_pickle=True)
    return path


def make_data_process(cfg):
    gid = cfg["gid"]
    if gid is None:
        caps = series(cfg["pkl"])
        gid = {n: i for i, n in enumerate(sorted(caps.keys()))}
    rated = cfg["rated"]
    seql = cfg["seql"]

    def DataProcess(BatteryData, test_name, start_point, args):
        """Official BatteryDataProcess semantics + per-cell prefix."""
        parts = []
        for name, df in BatteryData.items():
            r = df[["BatteryName", "Cycle", "Capacity"]].copy()
            if name == test_name:
                r = r[r["Cycle"] < start_point]
            parts.append(r)
        df_train = pd.concat(parts).reset_index(drop=True)
        df_test = BatteryData[test_name][["BatteryName", "Cycle",
                                          "Capacity"]].copy()
        for d in (df_train, df_test):
            d["Capacity"] /= rated
            d["target"] = d["Capacity"]
            d["time_idx"] = d["Cycle"].map(lambda x: int(x - 1))
            d["group_id"] = d["BatteryName"].map(gid)
        minv = df_train["Capacity"].min()
        maxv = df_train["Capacity"].max()
        for d in (df_train, df_test):
            d["Capacity"] = (d["Capacity"] - minv) / (maxv - minv)

        def fin(d):
            d = d.drop(["BatteryName"], axis=1)
            d["idx"] = [x for x in range(len(d))]
            d.set_index("idx", inplace=True)
            return d

        df_train = fin(df_train)
        df_test_all = fin(df_test)
        df_test = df_test_all.loc[
            df_test_all["Cycle"] >= start_point - seql,
            ["time_idx", "group_id", "Cycle", "Capacity", "target", "constant"]
            if "constant" in df_test_all.columns
            else ["time_idx", "group_id", "Cycle", "Capacity", "target"]]
        return df_train, df_test, df_test_all

    return DataProcess


def write_configs(ds, cfg, cache_path):
    os.makedirs(CFG_DIR, exist_ok=True)
    sps_str = "[" + ", ".join(str(s) for s in cfg["sps"]) + "]"
    base = f"""model: {{}}
dataset:
  name: {cfg['name']}
  input_mode: Univariable
  battery_cache_path: {cache_path}
  real_data_path_template: Results/Capacity_{{test_name}}_Real_Data.pth
  rated_capacity: {cfg['rated']}
  test_name: {cfg['test']}
  start_points: {sps_str}
features:
  time_varying_known_reals:
    - Capacity
  time_varying_unknown_reals:
    - target
window:
  seq_len: {cfg['seql']}
  label_len: 0
  pred_len: 1
train:
  count: 10
  batch_size: 128
  max_epochs: 200
  learning_rate: 0.001
  patience: 20
  gradient_clip_val: 0.2
  predict_batch_size: 256
  train_split_ratio: 0.8
  num_workers: 0
  shuffle_train: true
  drop_last_train: false
runtime:
  seed: 1
  auto_select_gpu: true
output:
  results_dir: Results
  logs_dir: Logs
  outputs_dir: Outputs
  results_filename_template: "{{dataset}}_{{input_mode}}_{{model}}_{{test_name}}.pth"
  log_filename_template: "{{dataset}}_{{input_mode}}_{{model}}_{{test_name}}.log"
  plot_dir_template: "Plots/{{dataset}}/{{input_mode}}/{{test_name}}/{{model}}"
"""
    bpath = os.path.join(CFG_DIR, f"{ds}_base.yaml")
    with open(bpath, "w", encoding="utf-8") as f:
        f.write(base)
    model = (f"model:\n  name: RULMamba\n"
             f"  class_path: Models.RULMamba.RULMambaNetModel\n"
             f"  build_args:\n"
             f"    seq_len: {cfg['seql']}\n    label_len: 0\n"
             f"    pred_len: 1\n    enc_in: 1\n    dec_in: 1\n    c_out: 1\n"
             f"    d_model: {cfg['d_model']}\n"
             f"    n_dec_layer: {cfg['n_dec']}\n    dropout: 0.1\n"
             f"    expand: 2\n")
    mpath = os.path.join(CFG_DIR, f"{ds}_rulmamba.yaml")
    with open(mpath, "w", encoding="utf-8") as f:
        f.write(model)
    return bpath, mpath


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(MODELS))
    ap.add_argument("--count", type=int, default=10)
    args = ap.parse_args()
    cfg = MODELS[args.dataset]

    cache_path = make_cache_npy(args.dataset, cfg)
    base_path, model_path = write_configs(args.dataset, cfg, cache_path)

    # Real_Data cache the trainer loads at startup (Ah series, test cell)
    caps = series(cfg["pkl"])
    os.makedirs(os.path.join(REF, "Results"), exist_ok=True)
    torch.save(np.asarray(caps[cfg["test"]], dtype=np.float64),
               os.path.join(REF, "Results",
                            f"Capacity_{cfg['test']}_Real_Data.pth"))

    os.chdir(REF)
    sys.path.insert(0, REF)
    # wrap the upstream TJU data process (per-cell prefix rule)
    td = importlib.import_module("Scripts.Data_Process.TJU_Data_Process")
    td.BatteryDataProcess = make_data_process(cfg)

    sys.argv = (["Train_extra.py", "--config", base_path,
                 "--model-config", model_path, "--model", "RULMamba",
                 "--test-name", cfg["test"],
                 "--start-points"] + [str(s) for s in cfg["sps"]] +
                ["--seed", "1", "--count", str(args.count)])
    runpy.run_path(os.path.join(
        REF, "Scripts", "TJU_Univariable_RUL_Prediction",
        "Train_TJU_Univariable.py"), run_name="__main__")


if __name__ == "__main__":
    main()
