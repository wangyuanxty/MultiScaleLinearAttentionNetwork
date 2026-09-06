"""Run the OFFICIAL RUL-Mamba TJU trainer with the per-cell prefix.

The repo's own TJU_BatteryDataProcess builds df_train from the other
cells only, while its NASA DataProcess (and our per-SP protocol)
include the test cell's cycles before the start point.  To keep the
TJU baseline rule-identical to NASA (and to our own models), we wrap
the official TJU DataProcess with those prefix rows and then execute
the official trainer (Train_TJU_Univariable.py) completely unchanged.

The trainer imports BatteryDataProcess at runtime
('from Scripts.Data_Process.TJU_Data_Process import ...'), so patching
the module attribute before runpy execution takes effect.  No file in
the upstream repo is modified.

Run with the `patchformer` conda env:

  D:\\anaconda\\envs\\patchformer\\python.exe src/run_rm_tju_prefix.py
"""
import os
import runpy
import sys

import torch

# torch 2.6 weights_only=True default breaks loading the repo's
# numpy-based .pth caches (same shim the official NASA runner applies)
_orig_load = torch.load


def _patched_load(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)


torch.load = _patched_load

REF = r"D:\research\degradation_prognostics\Transformer_and_Multi_Scale_Models\reference_repos\ref_rul_mamba"
os.chdir(REF)
sys.path.insert(0, REF)

import pandas as pd  # noqa: E402

from Scripts.Data_Process.TJU_Data_Process import BatteryDataProcess  # noqa: E402

_ORIG = BatteryDataProcess
_GID = {"CY25_1": 0, "CY25_2": 1, "CY25_3": 2}


def _with_prefix(BatteryData, test_name, start_point, args):
    """Official BatteryDataProcess + test-cell cycles < start_point."""
    df_train, df_test, df_all = _ORIG(BatteryData, test_name, start_point, args)
    pref = BatteryData[test_name][["BatteryName", "Cycle", "Capacity"]]
    pref = pref[pref["Cycle"] < start_point].copy()
    pref["Capacity"] /= args.Rated_Capacity
    pref["target"] = pref["Capacity"]
    pref["time_idx"] = pref["Cycle"].map(lambda x: int(x - 1))
    pref["group_id"] = pref["BatteryName"].map(_GID)
    pref = pref.drop(["BatteryName"], axis=1)
    pref["idx"] = [x for x in range(len(df_train), len(df_train) + len(pref))]
    pref.set_index("idx", inplace=True)
    df_train = pd.concat([df_train, pref])
    print(f"[prefix wrapper] {test_name} SP{start_point}: "
          f"added {len(pref)} prefix rows -> train={len(df_train)}", flush=True)
    return df_train, df_test, df_all


import Scripts.Data_Process.TJU_Data_Process as td  # noqa: E402

td.BatteryDataProcess = _with_prefix

sys.argv = ["Train_TJU_Univariable.py", "--model", "RULMamba",
            "--test-name", "CY25_1", "--start-points", "200", "300", "400",
            "--seed", "1", "--count", "10"]
runpy.run_path(os.path.join(REF, "Scripts", "TJU_Univariable_RUL_Prediction",
                            "Train_TJU_Univariable.py"), run_name="__main__")
