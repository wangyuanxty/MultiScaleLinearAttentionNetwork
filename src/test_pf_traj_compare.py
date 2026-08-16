"""PatchFormer TRUE autoregressive 32-step trajectory vs our non-autoregressive.

Same protocol: window 64, SP=300/400/500, normalized capacity.
  PatchFormer: roll 32 steps (feed predictions back)
  Ours (K=32 non-AR): one-shot trajectory (from test_seq2vec_traj: MAE 0.005-0.018)
"""
import sys, torch, numpy as np, glob, os

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

pf_repo = r'D:\research\degradation_prognostics\Transformer_and_Multi_Scale_Models\ref_patchformer'
sys.path.insert(0, pf_repo)

d = np.load(os.path.join(pf_repo, r'data\CALCE data\CALCE_Data.npy'), allow_pickle=True).item()
test_caps = d['CS2_35']['Capacity'].values.astype(np.float32)
train_caps = np.concatenate([d[c]['Capacity'].values for c in ['CS2_36', 'CS2_37', 'CS2_38']])
x_min, x_max = train_caps.min(), train_caps.max()
test_norm = (test_caps - x_min) / (x_max - x_min)

from ModelsModify.PatchFormer import PatchFormerNetModel
from pytorch_forecasting import TimeSeriesDataSet

W, K = 64, 32


def load_pf(sp):
    ckpts = []
    base = os.path.join(pf_repo, 'results_CALCE_RUL_prediction_sl_64', 'CS2_35', 'PatchFormer', f'SP{sp}')
    for root, dirs, files in os.walk(base):
        for f in files:
            if f.endswith('.ckpt'):
                ckpts.append(os.path.join(root, f))
    m = PatchFormerNetModel.load_from_checkpoint(ckpts[0]).cuda()
    m.eval()
    return m.network.cuda()


print(f"{'SP':>4} {'method':<14} {'MAE':>8} {'RMSE':>8}  first6 pred vs targ")
for sp in [300, 400, 500]:
    net = load_pf(sp)
    tgt = test_norm[sp:sp + K]
    window = test_norm[sp - W:sp]
    w = torch.tensor(window, dtype=torch.float32).unsqueeze(-1).unsqueeze(0).cuda()
    preds = []
    with torch.no_grad():
        for _ in range(K):
            p = net(w).squeeze().item()
            preds.append(p)
            w = torch.cat([w[:, 1:, :], torch.tensor([[[p]]], dtype=torch.float32, device='cuda')], dim=1)
    pred = np.array(preds)
    mae = np.mean(np.abs(pred - tgt))
    rmse = np.sqrt(np.mean((pred - tgt) ** 2))
    print(f"{sp:>4} {'PatchFormer-AR':<14} {mae:>8.4f} {rmse:>8.4f}  {np.round(pred[:6], 3)} vs {np.round(tgt[:6], 3)}")

print("\nOurs (K=32 non-AR, from test_seq2vec_traj):")
print("  SP=300: MAE=0.008  SP=400: MAE=0.005  SP=500: MAE=0.018")
print("done")
