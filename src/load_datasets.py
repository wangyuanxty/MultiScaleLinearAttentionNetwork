"""Dataset loaders for CALCE and PANASONIC battery degradation data.
No per-cell normalization — raw capacity returned. Normalization handled by SlidingWindowBuilder.
CALCE: per-cycle discharge computed via Coulomb counting (current × time integration)
       on Step_Index == 7 (discharge phase), following XiuzeZhou's CALCE notebook.
PANASONIC: last column of IC curve = charge capacity per cycle.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import glob
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).parent.parent / "data"


# ─── CALCE ────────────────────────────────────────────

def load_calce_capacity(cell_name: str) -> np.ndarray:
    """Load CALCE with exact PatchFormer logic (CALCEDataPreProcess.py)."""
    import glob as gb
    cell_dir = DATA_DIR / "calce" / cell_name
    if not cell_dir.exists():
        raise FileNotFoundError(f"CALCE not found: {cell_dir}")

    path = gb.glob(str(cell_dir / "*.xlsx"))
    dates = []
    for p in path:
        try:
            df = pd.read_excel(p, sheet_name=1)
            if "~$" in str(p): continue
            dates.append(df["Date_Time"][0])
        except Exception:
            dates.append(pd.Timestamp.min)
    path_sorted = np.array(path)[np.argsort(dates)]

    count = 0; discharge_capacities = []
    for p in path_sorted:
        try: df = pd.read_excel(p, sheet_name=1)
        except Exception: continue
        for c in sorted(df["Cycle_Index"].unique()):
            df_d = df[(df["Cycle_Index"]==c) & (df["Step_Index"]==7)]
            d_c = df_d["Current(A)"]; d_t = df_d["Test_Time(s)"]
            if len(list(d_c))==0: continue
            time_diff = np.diff(list(d_t))
            d_c_arr = np.array(list(d_c))[1:]
            discharge_capacities.append(-1*np.sum(time_diff*d_c_arr/3600))
            count += 1

    discharge_capacities = np.array(discharge_capacities)

    # PF drop_outlier: np.arange(1,count,40), range_[:-1], 2-sigma
    if count > 80:
        range_ = np.arange(1, count, 40)
        keep_idx = []
        for i in range_[:-1]:  # PF skips last bin
            w = discharge_capacities[i:min(i+40,count)]
            mu,sg = w.mean(), w.std()
            idx = np.where((w<mu+2*sg)&(w>mu-2*sg))[0] + i
            keep_idx.extend(idx.tolist())
        discharge_capacities = discharge_capacities[np.array(sorted(set(keep_idx)))]

    capacities = discharge_capacities.astype(np.float32)
    logger.info(f"CALCE {cell_name}: {len(capacities)} cycles, [{capacities.min():.3f},{capacities.max():.3f}]")
    return capacities

def _calce_date(filepath: str) -> str:
    parts = Path(filepath).stem.split("_")
    return f"20{parts[4]}{parts[2].zfill(2)}{parts[3].zfill(2)}" if len(parts) >= 5 else filepath


# ─── PANASONIC ────────────────────────────────────────

def load_panasonic_capacity(sheet_name: Optional[str] = None,
                            dataset_path: str = "Dataset#3.xlsx") -> np.ndarray:
    path = DATA_DIR / "panasonic" / dataset_path
    df = pd.read_excel(path, sheet_name=sheet_name) if sheet_name else pd.read_excel(path)
    caps = pd.to_numeric(df[df.columns[-1]].iloc[1:], errors="coerce").dropna().values.astype(np.float32)
    logger.info(f"PANASONIC {dataset_path}/{sheet_name or 'default'}: {len(caps)} cycles")
    return caps


def load_panasonic_cells() -> Dict[str, np.ndarray]:
    path = DATA_DIR / "panasonic" / "Dataset#3.xlsx"
    return {s: load_panasonic_capacity(sheet_name=s) for s in pd.ExcelFile(path).sheet_names}


def load_gotion_cells() -> Dict[str, np.ndarray]:
    path = DATA_DIR / "panasonic" / "Dataset#5.xlsx"
    return {s: load_panasonic_capacity(sheet_name=s, dataset_path="Dataset#5.xlsx")
            for s in pd.ExcelFile(path).sheet_names}


# ─── MIT-Stanford ────────────────────────────────────

def load_mit_stanford() -> Dict[str, np.ndarray]:
    """Load MIT-Stanford fast-charging battery dataset.
    Uses h5py to read .mat v7.3 files.
    Extracts discharge capacity (QD) per cycle from summary.
    Filters batteries that never degrade below 80% of nominal.
    Merges overlapping batch1/batch2 cells.
    Returns: Dict[cell_name -> capacity array (Ah)]
    """
    import h5py
    data_dir = DATA_DIR / "mit_stanford"

    batches = {
        'batch1': '2017-05-12_batchdata_updated_struct_errorcorrect.mat',
        'batch2': '2017-06-30_batchdata_updated_struct_errorcorrect.mat',
        'batch3': '2018-04-12_batchdata_updated_struct_errorcorrect.mat',
    }

    all_cells = {}
    for batch_name, filename in batches.items():
        path = data_dir / filename
        if not path.exists():
            logger.warning(f"MIT-Stanford file not found: {path}")
            continue

        f = h5py.File(str(path), 'r')
        batch = f['batch']
        n_cells = batch['summary'].shape[0]

        for i in range(n_cells):
            # Extract discharge capacity per cycle
            qd_ref = batch['summary'][i, 0]
            qd = np.array(f[qd_ref]['QDischarge']).flatten().astype(np.float32)

            # Filter: capacity must reach below 80% of nominal (1.1Ah)
            if qd.max() < 0.01 or qd[0] < 0.01:
                continue
            eol_80 = qd[0] * 0.8
            if qd.min() > eol_80:
                continue  # never degraded to 80%

            cell_name = f'{batch_name}_cell{i}'
            all_cells[cell_name] = qd

        f.close()

    # Merge batch1 cells that continue in batch2
    # From reference: b1c0+b2c7, b1c1+b2c8, b1c2+b2c9, b1c3+b2c15, b1c4+b2c16
    merge_pairs = [
        ('batch1_cell0', 'batch2_cell7'),
        ('batch1_cell1', 'batch2_cell8'),
        ('batch1_cell2', 'batch2_cell9'),
        ('batch1_cell3', 'batch2_cell15'),
        ('batch1_cell4', 'batch2_cell16'),
    ]
    for a, b in merge_pairs:
        if a in all_cells and b in all_cells:
            all_cells[a] = np.concatenate([all_cells[a], all_cells[b]])
            del all_cells[b]
            logger.info(f"Merged {a} + {b}")

    logger.info(f"MIT-Stanford: {len(all_cells)} cells loaded")
    return all_cells


# ─── NASA ─────────────────────────────────────────────

def load_nasa_capacity(battery_name: str) -> np.ndarray:
    """Load per-cycle discharge capacity from NASA PCoE battery .mat files.

    Data structure (following PatchFormer NASADataPreProcess.py):
      data['B0005'][0,0]['cycle'] → (1, N) struct array
      Each cycle: fields = [type, ambient_temperature, time, data]
      Discharge data has 'Capacity' field → shape (1,1) scalar in Ah.

    Args:
        battery_name: e.g. 'B0005', 'B0006', 'B0007', 'B0018'

    Returns:
        Numpy array of discharge capacities in Ah (raw, NOT divided by rated 2.0Ah).
    """
    import scipy.io

    path = DATA_DIR / "nasa" / f"{battery_name}.mat"
    if not path.exists():
        raise FileNotFoundError(f"NASA file not found: {path}")

    data = scipy.io.loadmat(str(path))
    cycles = data[battery_name][0, 0]['cycle']  # shape (1, N)

    capacities = []
    for i in range(cycles.shape[1]):
        c = cycles[0, i]
        cycle_type = str(c['type'][0])
        if cycle_type != 'discharge':
            continue
        ds = c['data']
        if 'Capacity' not in ds.dtype.names:
            continue
        cap = float(ds[0, 0]['Capacity'][0, 0])
        if cap > 0.01:
            capacities.append(cap)

    capacities = np.array(capacities, dtype=np.float32)
    logger.info(
        f"NASA {battery_name}: {len(capacities)} discharge cycles, "
        f"[{capacities.min():.3f}, {capacities.max():.3f}] Ah"
    )
    return capacities


def load_nasa_multivar(battery_name: str) -> Dict[str, np.ndarray]:
    """Load per-cycle capacity + V/I/T/IR features from NASA .mat files.

    Returns dict with keys:
      capacity: discharge capacity per cycle (Ah)
      V_mean:   mean voltage during discharge (V)
      I_mean:   mean current during discharge (A)
      T_mean:   mean temperature during discharge (°C)
      T_max:    max temperature during discharge (°C)
      Re:       electrolyte resistance from EIS (Ohm), interpolated
      Rct:      charge transfer resistance from EIS (Ohm), interpolated

    EIS impedance measurements (Re, Rct) are collected every ~2 cycles.
    Linear interpolation aligns them to discharge cycle indices.
    """
    import scipy.io
    from scipy.interpolate import interp1d

    path = DATA_DIR / "nasa" / f"{battery_name}.mat"
    if not path.exists():
        raise FileNotFoundError(f"NASA file not found: {path}")

    data = scipy.io.loadmat(str(path))
    cycles = data[battery_name][0, 0]['cycle']  # shape (1, N)

    capacities, v_means, i_means, t_means, t_maxs = [], [], [], [], []
    eis_indices, re_vals, rct_vals = [], [], []

    for i in range(cycles.shape[1]):
        c = cycles[0, i]
        cycle_type = str(c['type'][0])
        ds = c['data']

        if cycle_type == 'discharge' and 'Capacity' in ds.dtype.names:
            cap = float(ds[0, 0]['Capacity'][0, 0])
            if cap < 0.01:
                continue
            v_arr = ds[0, 0]['Voltage_measured'][0]
            t_arr = ds[0, 0]['Temperature_measured'][0]
            i_arr = ds[0, 0]['Current_measured'][0]
            capacities.append(cap)
            v_means.append(float(np.mean(v_arr)))
            i_means.append(float(np.mean(i_arr)))
            t_means.append(float(np.mean(t_arr)))
            t_maxs.append(float(np.max(t_arr)))

        elif cycle_type == 'impedance':
            if 'Re' in ds.dtype.names and 'Rct' in ds.dtype.names:
                eis_indices.append(i)
                re_vals.append(float(ds[0, 0]['Re'][0, 0]))
                rct_vals.append(float(ds[0, 0]['Rct'][0, 0]))

    n_d = len(capacities)
    capacities = np.array(capacities, dtype=np.float32)
    v_means = np.array(v_means, dtype=np.float32)
    i_means = np.array(i_means, dtype=np.float32)
    t_means = np.array(t_means, dtype=np.float32)
    t_maxs = np.array(t_maxs, dtype=np.float32)

    # Build discharge cycle absolute indices (same loop pattern)
    d_indices = []
    for i in range(cycles.shape[1]):
        c = cycles[0, i]
        if (str(c['type'][0]) == 'discharge'
            and 'Capacity' in c['data'].dtype.names
            and float(c['data'][0, 0]['Capacity'][0, 0]) >= 0.01):
            d_indices.append(i)
    d_indices = np.array(d_indices[:n_d], dtype=np.float64)

    # Interpolate EIS to discharge cycle indices
    # Use clip-to-boundary: no extrapolation beyond EIS range
    if len(eis_indices) >= 2:
        re_interp = interp1d(eis_indices, re_vals, kind='linear',
                             bounds_error=False,
                             fill_value=(re_vals[0], re_vals[-1]))
        rct_interp = interp1d(eis_indices, rct_vals, kind='linear',
                              bounds_error=False,
                              fill_value=(rct_vals[0], rct_vals[-1]))
        re_at_d = re_interp(d_indices).astype(np.float32)
        rct_at_d = rct_interp(d_indices).astype(np.float32)
    else:
        re_at_d = np.full(n_d, np.nan, dtype=np.float32)
        rct_at_d = np.full(n_d, np.nan, dtype=np.float32)

        # Compute Δt (hours) between consecutive discharge cycles
    dt_hours = np.concatenate([[np.mean(np.diff(d_indices))], np.diff(d_indices)]).astype(np.float32)
    # Actual calendar gaps: compute from time field
    from datetime import datetime
    dts_abs = []
    for idx in d_indices.astype(int):
        tv = cycles[0, idx]['time'][0]
        dts_abs.append(datetime(int(tv[0]),int(tv[1]),int(tv[2]),int(tv[3]),int(tv[4]),int(tv[5])))
    dt_calendar = np.zeros(n_d, dtype=np.float32)
    for j in range(1, n_d):
        dt_calendar[j] = (dts_abs[j] - dts_abs[j-1]).total_seconds() / 3600
    dt_calendar[0] = dt_calendar[1:].mean() if n_d > 1 else 0

    logger.info(
        f"NASA {battery_name}: {n_d} cycles, V[{v_means.min():.2f},{v_means.max():.2f}], "
        f"T[{t_means.min():.1f},{t_maxs.max():.1f}]°C, "
        f"Re[{np.nanmin(re_at_d):.4f},{np.nanmax(re_at_d):.4f}]Ω, "
        f"Δt[{dt_calendar.min():.1f},{dt_calendar.max():.1f}]h"
    )

    return {
        'capacity': capacities,
        'V_mean': v_means,
        'I_mean': i_means,
        'T_mean': t_means,
        'T_max': t_maxs,
        'Re': re_at_d,
        'Rct': rct_at_d,
        'delta_t': dt_calendar,
    }


def load_nasa_cells_multivar() -> Dict[str, Dict[str, np.ndarray]]:
    """All NASA batteries with full multi-variable features."""
    return {b: load_nasa_multivar(b) for b in ['B0005', 'B0006', 'B0007', 'B0018']}


def load_nasa_cells() -> Dict[str, np.ndarray]:
    """Load all NASA batteries (B0005, B0006, B0007, B0018)."""
    return {b: load_nasa_capacity(b) for b in ['B0005', 'B0006', 'B0007', 'B0018']}


# ─── Test ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for cell in ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]:
        caps = load_calce_capacity(cell)
        eol = int(np.argmax(caps / caps[0] < 0.7))
        print(f"  {cell}: {len(caps)} cycles, EOL@70%={eol}")
    for name, caps in load_panasonic_cells().items():
        eol = int(np.argmax(caps / caps[0] < 0.7))
        print(f"  PANASONIC {name}: {len(caps)} cycles, EOL@70%={eol}")
    for name, caps in load_nasa_cells().items():
        eol = int(np.argmax(caps / 2.0 < 0.7))  # rated 2.0Ah → EOL @ 1.4Ah
        print(f"  NASA {name}: {len(caps)} cycles, EOL@1.4Ah={eol}")
# ─── MIT-Stanford multi-variable ──────────────────────

def load_mit_stanford_multivar():
    """Load MIT with per-cycle features from summary (per-cycle scalars, not raw time series).

    Returns (caps, feats, feat_dim) where:
      caps: Dict[cell_name -> (N,) float32]  — discharge capacity
      feats: Dict[cell_name -> (N, 6) float32]  — IR, QCharge, Tavg, Tmax, Tmin, chargetime
    """
    import h5py
    data_dir = DATA_DIR / 'mit_stanford'
    batches = {
        'batch1': '2017-05-12_batchdata_updated_struct_errorcorrect.mat',
        'batch2': '2017-06-30_batchdata_updated_struct_errorcorrect.mat',
        'batch3': '2018-04-12_batchdata_updated_struct_errorcorrect.mat',
    }
    feat_keys = ['IR', 'QCharge', 'Tavg', 'Tmax', 'Tmin', 'chargetime']
    all_caps, all_feats = {}, {}
    for batch_name, filename in batches.items():
        path = data_dir / filename
        if not path.exists(): continue
        f = h5py.File(str(path), 'r')
        summary = f['batch']['summary']
        for i in range(summary.shape[0]):
            cell = f[summary[i, 0]]
            qd = np.array(cell['QDischarge']).flatten().astype(np.float32)
            if qd.max() < 0.01 or qd[0] < 0.01: continue
            if qd.min() > qd[0] * 0.8: continue
            feats_list = [np.array(cell[k]).flatten().astype(np.float32) for k in feat_keys]
            n = min([len(qd)] + [len(x) for x in feats_list])
            cell_name = f'{batch_name}_cell{i}'
            all_caps[cell_name] = qd[:n]
            all_feats[cell_name] = np.stack([x[:n] for x in feats_list], axis=-1)
        f.close()
    merge = [('batch1_cell0','batch2_cell7'),('batch1_cell1','batch2_cell8'),
             ('batch1_cell2','batch2_cell9'),('batch1_cell3','batch2_cell15'),
             ('batch1_cell4','batch2_cell16')]
    for a,b in merge:
        if a in all_caps and b in all_caps:
            all_caps[a] = np.concatenate([all_caps[a], all_caps[b]])
            all_feats[a] = np.concatenate([all_feats[a], all_feats[b]])
            del all_caps[b]; del all_feats[b]
    return all_caps, all_feats, len(feat_keys)


# ─── CALCE multi-variable ─────────────────────────────

def load_calce_multivar(cell_name):
    """Load CALCE with per-cycle V/IR/dV/dt/Energy/AC + capacity."""
    cell_dir = DATA_DIR / 'calce' / cell_name
    import glob as gb
    files = gb.glob(str(cell_dir / '*.xlsx'))
    # Single pass: read each file once, extract date + data together
    filedata = []
    for f in files:
        if '~$' in str(f): continue
        try:
            xl = pd.ExcelFile(f)
            ch = [s for s in xl.sheet_names if s != 'Info'][0]
            df = pd.read_excel(f, sheet_name=ch)
            date0 = pd.to_datetime(df['Date_Time'].iloc[0])
            filedata.append((date0, f, df))
        except: continue
    filedata.sort(key=lambda x: x[0])
    records = []
    dts = []
    for date0, fpath, df in filedata:
        for ci in sorted(df['Cycle_Index'].unique()):
            d = df[(df['Cycle_Index']==ci) & (df['Step_Index']==7)]
            if len(d) < 2: continue
            t = pd.to_numeric(d['Test_Time(s)'], errors='coerce').values
            i = pd.to_numeric(d['Current(A)'], errors='coerce').values
            time_diff = np.diff(list(t)); d_c_arr = np.array(list(i))[1:]
            q = -np.sum(time_diff*d_c_arr/3600)
            if q <= 0.01: continue
            v = pd.to_numeric(d['Voltage(V)'], errors='coerce').values
            ir = pd.to_numeric(d['Internal_Resistance(Ohm)'], errors='coerce').values if 'Internal_Resistance(Ohm)' in d.columns else np.full(1,np.nan)
            dvdt = pd.to_numeric(d['dV/dt(V/s)'], errors='coerce').values if 'dV/dt(V/s)' in d.columns else np.full(1,np.nan)
            e_d = pd.to_numeric(d['Discharge_Energy(Wh)'], errors='coerce').values if 'Discharge_Energy(Wh)' in d.columns else np.full(1,np.nan)
            e_c_vals = pd.to_numeric(df[(df['Cycle_Index']==ci) & (df['Step_Index'].isin([2,4]))]['Charge_Energy(Wh)'], errors='coerce').values if 'Charge_Energy(Wh)' in d.columns else np.full(1,np.nan)
            ac = pd.to_numeric(d['AC_Impedance(Ohm)'], errors='coerce').values if 'AC_Impedance(Ohm)' in d.columns else np.full(1,np.nan)
            ph = pd.to_numeric(d['ACI_Phase_Angle(Deg)'], errors='coerce').values if 'ACI_Phase_Angle(Deg)' in d.columns else np.full(1,np.nan)
            records.append([float(q), float(np.mean(v)), float(np.min(v)), float(np.max(v)),
                float(np.mean(ir)), float(np.mean(np.abs(dvdt))),
                float(np.mean(e_d)), float(np.mean(e_c_vals)),
                float(np.mean(ac)), float(np.mean(ph))])
            dts.append(pd.to_datetime(d['Date_Time'].iloc[0]))
    df_out = pd.DataFrame(records, columns=['cap','V_mean','V_min','V_max','IR','dvdt','E_d','E_c','AC','Phase'])
    for col in ['IR','dvdt','AC','Phase']: df_out[col] = df_out[col].ffill().fillna(0)
    # Calendar Delta-t (timestamps already collected in main loop)
    dt_cal = np.zeros(len(records), dtype=np.float32)
    for j in range(1, len(dts)):
        dt_cal[j] = (dts[j] - dts[j-1]).total_seconds() / 3600
    dt_cal[0] = dt_cal[1:].mean() if len(dt_cal) > 1 else 0
    df_out['delta_t'] = dt_cal

    # PF drop_outlier on capacities, filter features with same indices
    caps_arr = df_out["cap"].values
    count_val = len(caps_arr)
    if count_val > 80:
        range_ = np.arange(1, count_val, 40)
        keep = []
        for ii in range_[:-1]:
            w = caps_arr[ii:min(ii+40, count_val)]
            mu, sg = w.mean(), w.std()
            idx = np.where((w < mu + 2*sg) & (w > mu - 2*sg))[0] + ii
            keep.extend(idx.tolist())
        df_out = df_out.iloc[np.array(sorted(set(keep)))].reset_index(drop=True)

    feat_cols = ['V_mean','V_min','V_max','IR','dvdt','E_d','E_c','AC','Phase','delta_t']
    return df_out["cap"].values.astype(np.float32), df_out[feat_cols].values.astype(np.float32)


def load_calce_cells_multivar():
    """All CALCE cells with multi-variable features."""
    cells = ['CS2_35','CS2_36','CS2_37','CS2_38']
    caps, feats = {}, {}
    for c in cells:
        cap, feat = load_calce_multivar(c)
        caps[c] = cap; feats[c] = feat
    return caps, feats, 9


def load_tju_cells():
    """TJU (Tongji) cells — capacity only. Rated 2.5Ah, EOL = 1.75Ah (70%)."""
    import numpy as np
    path = Path(__file__).parent.parent / 'ref_rul_mamba' / 'Data' / 'TJU_Data' / 'Dataset_3_NCM_NCA_Battery_1C.npy'
    d = np.load(str(path), allow_pickle=True).item()
    caps = {}
    for k in ['CY25_1', 'CY25_2', 'CY25_3']:
        df = d[k]
        caps[k] = df['Capacity'].values.astype(np.float32)
    return caps


# MIT/Stanford subset: 8 train / 2 test (protocol-consistent subset)
MIT_TEST_CELLS = [
    'batch2_cell5', 'batch2_cell47',
]
MIT_TRAIN_CELLS = [
    'batch2_cell0', 'batch2_cell11', 'batch2_cell26', 'batch2_cell32',
    'batch2_cell36', 'batch2_cell37', 'batch2_cell42', 'batch2_cell46',
]
