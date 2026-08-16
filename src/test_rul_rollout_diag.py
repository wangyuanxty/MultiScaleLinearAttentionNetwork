"""Diagnostic: print K=32 rollout trajectory per SP, check EOL crossing."""
import sys, numpy as np, torch, os
sys.path.insert(0, '.')
from gdn_model import build_gdn_model
from load_datasets import load_calce_cells_multivar, load_mit_stanford

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = 'D:/research/degradation_prognostics/Transformer_and_Multi_Scale_Models/checkpoints'
K = 32

def diag(name, raw_test, lo, hi, W, sps, eol_ah, ckpt_key):
    tc = (raw_test - lo) / (hi - lo + 1e-8)
    eol_n = (eol_ah - lo) / (hi - lo + 1e-8)
    te = int(np.argmax(raw_test < eol_ah))
    print(f'\n=== {name}: {len(tc)} cycles, EOL@{te} ===')

    ckpt = f'{CKPT}/unified_{ckpt_key}_K32.pt'
    if not os.path.exists(ckpt):
        ckpt = f'{CKPT}/abl_{ckpt_key}_multi-cap-traj32.pt'
    if not os.path.exists(ckpt):
        print(f'No checkpoint: {ckpt}')
        return
    model = build_gdn_model(multiscale=True, cross_exchange=True, input_dim=1,
                            window_size=W, output_len=32, readout="last").to(DEV)
    model.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=True))
    model.eval()

    for sp in sps:
        if sp >= len(tc) - 2: continue
        print(f'\nSP={sp} (true RUL={te-sp})')
        window = tc[sp-W:sp].copy()
        traj = []
        with torch.no_grad():
            for chunk in range(50):
                cin = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
                pred = model(cin).squeeze(0).cpu().numpy()
                traj.extend(pred.tolist())
                if chunk == 0:
                    print(f'  Chunk0 pred: {[f"{x:.4f}" for x in pred[:3]]}...{[f"{x:.4f}" for x in pred[-3:]]}')
                window = np.concatenate([window[K:], pred])
                if pred[-1] < eol_n:
                    print(f'  Crossed at chunk {chunk}')
                    break
            else:
                print(f'  NEVER crossed. Last={traj[-1]:.4f} > eol_n={eol_n:.4f}')
        traj = np.array(traj)
        for j in range(len(traj)-1):
            if traj[j] >= eol_n > traj[j+1]:
                frac = (eol_n - traj[j]) / (traj[j+1] - traj[j] + 1e-8)
                pe = sp + j + frac
                print(f'  j={j} [{traj[j]:.4f} > {eol_n:.4f} > {traj[j+1]:.4f}] pe={pe:.1f} RUL_MAE={abs(te-pe):.1f}')
                break
            elif traj[j] < eol_n:
                print(f'  j={j} [{traj[j]:.4f} < {eol_n:.4f}] pe={sp+j:.1f} RUL_MAE={abs(te-(sp+j)):.1f}')
                break

# MIT-subset
caps_all = load_mit_stanford()
tcells = ['batch2_cell11','batch2_cell26','batch2_cell32','batch2_cell36','batch2_cell37','batch2_cell42','batch2_cell46']
test_c = 'batch2_cell5'
train_caps = np.concatenate([caps_all[c] for c in tcells])
raw_t = caps_all[test_c].astype(np.float32)
diag('MIT-subset', raw_t, train_caps.min(), train_caps.max(), 64, [200,300,400], 0.86, 'mit')

# CALCE
caps_c, _, _ = load_calce_cells_multivar()
tc_c = ['CS2_36','CS2_37','CS2_38']; test_c = 'CS2_35'
train_c = np.concatenate([caps_c[c] for c in tc_c])
raw_tc = caps_c[test_c].astype(np.float32)
diag('CALCE', raw_tc, train_c.min(), train_c.max(), 64, [300,400,500], 0.77, 'calce')
