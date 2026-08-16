"""Recompute AE (PatchFormer/RUL-Mamba protocol) from saved checkpoints — no retraining.
Loads unified_*.pt K=1 models, runs non-recursive eval with rul_value_error."""
import sys, numpy as np, torch, os
sys.path.insert(0, '.')
from gdn_model import build_gdn_model
from load_datasets import load_calce_cells_multivar, load_nasa_multivar, load_mit_stanford, load_panasonic_cells, load_gotion_cells

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = 'D:/research/degradation_prognostics/Transformer_and_Multi_Scale_Models/checkpoints'

def load_data(ds):
    if ds == 'calce':
        caps_all, _, _ = load_calce_cells_multivar()
        caps = {c: caps_all[c].copy().astype(np.float32) for c in caps_all}
        return caps, ['CS2_36','CS2_37','CS2_38'], 'CS2_35', 64, [300,400,500], 0.77
    elif ds == 'nasa':
        caps = {}
        for b in ['B0005','B0006','B0007','B0018']:
            caps[b] = load_nasa_multivar(b)['capacity'].astype(np.float32)
        return caps, ['B0006','B0007','B0018'], 'B0005', 30, [50,70,90], 1.40
    elif ds == 'mit':
        caps_all = load_mit_stanford()
        tcells = ['batch2_cell11','batch2_cell26','batch2_cell32','batch2_cell36','batch2_cell37','batch2_cell42','batch2_cell46']
        caps = {c: caps_all[c].copy().astype(np.float32) for c in tcells + ['batch2_cell5']}
        return caps, tcells, 'batch2_cell5', 64, [200,300,400], 0.86
    elif ds == 'panasonic':
        caps_all = load_panasonic_cells()
        caps = {c: caps_all[c].copy().astype(np.float32) for c in caps_all}
        cells = sorted(caps.keys())
        return caps, cells[:-1], cells[-1], 30, [300,500,700], 2.12
    elif ds == 'gotion':
        caps_all = load_gotion_cells()
        caps = {c: caps_all[c].copy().astype(np.float32) for c in ['Cell01','Cell02','Cell03']}
        return caps, ['Cell02','Cell03'], 'Cell01', 30, [500,800,1100], 21.60

def recompute(ds):
    caps, train_cells, test_cell, W, sps, eol_ah = load_data(ds)
    all_tr = np.concatenate([caps[c] for c in train_cells])
    lo, hi = all_tr.min(), all_tr.max()
    tc = (caps[test_cell] - lo) / (hi - lo + 1e-8)

    ckpt = f'{CKPT}/unified_{ds}_K1.pt'
    if not os.path.exists(ckpt):
        print(f'{ds}: no checkpoint {ckpt}')
        return
    model = build_gdn_model(multiscale=True, cross_exchange=True, input_dim=1,
                            window_size=W, output_len=1, readout="last").to(DEV)
    model.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=True))
    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(W, len(tc)):
            cin = torch.tensor(tc[i-W:i], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
            preds.append(model(cin).item())
    pv = np.array(preds)[:len(tc)-W]; tv = tc[W:]
    threshold_n = (eol_ah - lo) / (hi - lo + 1e-8)
    print(f'\n=== {ds} (threshold {eol_ah}Ah, norm {threshold_n:.4f}) ===')
    print('  {:>5} {:>6} {:>6} {:>8} {:>8} {:>8} {:>5} {:>6}'.format('SP','TRUL','PRUL','MAE','RMSE','R2','AE','RE'))
    for sp in sps:
        seg_p = pv[sp-W:]; seg_t = tc[sp:]
        n = min(len(seg_p), len(seg_t))
        seg_p, seg_t = seg_p[:n], seg_t[:n]
        mae = np.mean(np.abs(seg_p - seg_t))
        rmse = np.sqrt(np.mean((seg_p - seg_t) ** 2))
        r2 = 1 - np.sum((seg_t - seg_p)**2) / (np.sum((seg_t - seg_t.mean())**2) + 1e-8)
        true_re, pred_re = len(seg_t), 0
        for i in range(len(seg_t)-1):
            if seg_t[i] >= threshold_n > seg_t[i+1]:
                true_re = i; break
        for i in range(len(seg_p)-1):
            if seg_p[i] >= threshold_n > seg_p[i+1]:
                pred_re = i; break
        tru = true_re + 1
        pru = pred_re + 1
        ae = abs(tru - pru)
        re = ae / tru if tru > 0 else 0
        print(f'  {sp:>5} {tru:>6} {pru:>6} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f} {ae:>5} {re:>6.4f}')

for ds in ['calce', 'nasa', 'mit', 'panasonic', 'gotion']:
    recompute(ds)
