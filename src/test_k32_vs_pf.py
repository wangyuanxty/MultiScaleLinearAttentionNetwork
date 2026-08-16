"""CALCE 段内 AR 32 步对比: PatchFormer AR vs Ours K=1 AR vs Ours K=32 一次预测.
协议: 每段从真实窗口起步, 段内自回归滚 32 步(预测喂回);
段末穿 EOL → 插值穿越 → AE; 未穿则真实窗口右移一步重新滚.
PatchFormer 手动调用: per-window scale = 窗口 target 的 mean/std (已验证=官方 scale)."""
import sys, torch, numpy as np, glob, os, pandas as pd
sys.path.insert(0, '.')
sys.path.insert(0, r'D:\research\degradation_prognostics\Transformer_and_Multi_Scale_Models\ref_patchformer')

torch.serialization.add_safe_globals([
    'pytorch_forecasting.data.encoders.EncoderNormalizer',
    'pytorch_forecasting.data.encoders.NaNLabelEncoder',
    'pytorch_forecasting.data.encoders.MultiNormalizer',
    'pytorch_forecasting.data.encoders.TorchNormalizer',
    'pytorch_lightning.callbacks.early_stopping.EarlyStopping',
    'numpy.core.multiarray.scalar',
])
_torch_load = torch.load
def _patched_load(*a, **kw):
    kw['weights_only'] = False
    return _torch_load(*a, **kw)
torch.load = _patched_load

from gdn_model import build_gdn_model
from load_datasets import load_calce_cells_multivar
from ModelsModify.PatchFormer import PatchFormerNetModel

DEV = torch.device('cuda')
W, EOL_AH = 64, 0.77
SPs = [300, 400, 500]

caps_all, _, _ = load_calce_cells_multivar()
train_caps = np.concatenate([caps_all[c] for c in ['CS2_36','CS2_37','CS2_38']])
raw_test = caps_all['CS2_35'].astype(np.float32)
lo, hi = train_caps.min(), train_caps.max()
tc = (raw_test - lo) / (hi - lo + 1e-8)
eol_n = (EOL_AH - lo) / (hi - lo + 1e-8)
te = int(np.argmax(raw_test < EOL_AH))
print(f'CALCE: {len(tc)} cycles, EOL@{te}, threshold_n={eol_n:.4f}')

# 我们的模型
m1 = build_gdn_model(multiscale=True, cross_exchange=True, input_dim=1, window_size=W, output_len=1, readout='last').to(DEV)
m1.load_state_dict(torch.load('../checkpoints/unified_calce_K1.pt', map_location=DEV, weights_only=True))
m1.eval()
m32 = build_gdn_model(multiscale=True, cross_exchange=True, input_dim=1, window_size=W, output_len=32, readout='last').to(DEV)
m32.load_state_dict(torch.load('../checkpoints/unified_calce_K32.pt', map_location=DEV, weights_only=True))
m32.eval()

# PatchFormer models per SP
pf_models = {}
for sp in SPs:
    ckpts = glob.glob(os.path.join(
        r'D:\research\degradation_prognostics\Transformer_and_Multi_Scale_Models\ref_patchformer',
        f'results_CALCE_RUL_prediction_sl_64/CS2_35/PatchFormer/SP{sp}/*/PatchFormerNetModel/checkpoints/*.ckpt'))
    if ckpts:
        pf_models[sp] = PatchFormerNetModel.load_from_checkpoint(ckpts[0]).to(DEV).eval()
        pf_models[sp].network = pf_models[sp].network.cuda().eval()

def pf_predict(model, window):
    """PatchFormer 手动调用: per-window scale = 窗口 mean/std."""
    scale = torch.tensor([[[window.mean()], [window.std() + 1e-8]]], dtype=torch.float32, device=DEV)
    cin = torch.tensor(window, dtype=torch.float32).unsqueeze(-1).unsqueeze(0).to(DEV)
    with torch.no_grad():
        raw = model.network(cin)
        return model.transform_output(raw, target_scale=scale).squeeze().item()

def ar_segment(model, t, seg_len=32, is_pf=False):
    """段内 AR: 真实窗口起步, 滚 seg_len 步."""
    window = tc[t-W:t].copy()
    traj = []
    with torch.no_grad():
        for _ in range(seg_len):
            if is_pf:
                p = pf_predict(model, window)
            else:
                cin = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
                p = model(cin).squeeze(0).cpu().numpy()[0]
            traj.append(p)
            window = np.concatenate([window[1:], [p]])
    return np.array(traj)

def segment_eval(model, sp, is_pf=False):
    """步进分段: t 右移, 每段滚 32 步; 段末穿 EOL → 段内插值穿越 → AE."""
    mae = rmse = r2 = None
    ae = None
    for t in range(sp, te):
        traj32 = ar_segment(model, t, 32, is_pf)
        if t == sp:
            true = tc[sp:sp+32]
            n = min(len(traj32), len(true))
            p, q = traj32[:n], true[:n]
            mae = np.mean(np.abs(p - q))
            rmse = np.sqrt(np.mean((p - q) ** 2))
            r2 = 1 - np.sum((q - p)**2) / (np.sum((q - q.mean())**2) + 1e-8)
        if traj32[-1] < eol_n:
            for j in range(len(traj32)-1):
                if traj32[j] >= eol_n > traj32[j+1]:
                    frac = (eol_n - traj32[j]) / (traj32[j+1] - traj32[j] + 1e-8)
                    pe = t + j + frac
                    break
            else:
                pe = t + len(traj32) - 1
            ae = abs(te - pe)
            break
    return mae, rmse, r2, ae

print(f'\n=== CALCE 段内 AR 32 步对比 ===')
print(f'{"SP":>4} {"method":<16} {"trajMAE":>8} {"trajRMSE":>8} {"trajR2":>7} {"AE":>6}')
for sp in SPs:
    if sp in pf_models:
        mae_p, rmse_p, r2_p, ae_p = segment_eval(pf_models[sp], sp, is_pf=True)
        print(f'{sp:>4} {"PatchFormer AR":<16} {mae_p:>8.4f} {rmse_p:>8.4f} {r2_p:>7.4f} {ae_p if ae_p is not None else -1:>6.1f}')
    mae_1, rmse_1, r2_1, ae_1 = segment_eval(m1, sp, is_pf=False)
    print(f'{sp:>4} {"Ours K=1 AR":<16} {mae_1:>8.4f} {rmse_1:>8.4f} {r2_1:>7.4f} {ae_1 if ae_1 is not None else -1:>6.1f}')
    # Ours K=32: stride-1 非递归提前感知 (已有协议)
    ae_32 = None
    with torch.no_grad():
        for t in range(sp, te):
            window = tc[t-W:t]
            cin = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
            pred = m32(cin).squeeze(0).cpu().numpy()
            if t == sp:
                mae_32 = np.mean(np.abs(pred[:32] - tc[sp:sp+32]))
            if pred[-1] < eol_n:
                for j in range(len(pred)-1):
                    if pred[j] >= eol_n > pred[j+1]:
                        frac = (eol_n - pred[j]) / (pred[j+1] - pred[j] + 1e-8)
                        pe = t + j + frac
                        break
                else:
                    pe = t + len(pred) - 1
                ae_32 = abs(te - pe)
                break
    print(f'{sp:>4} {"Ours K=32":<16} {mae_32:>8.4f} {"-":>8} {"-":>7} {ae_32 if ae_32 is not None else -1:>6.1f}')
