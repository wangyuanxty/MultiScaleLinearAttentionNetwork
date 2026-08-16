"""Test PatchFormer with TRUE autoregressive/recursive EOL prediction."""
import sys, torch, numpy as np, glob, os

# Fix PyTorch 2.6 weights_only issue
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

# Point to PatchFormer repo
pf_repo = r'D:\research\degradation_prognostics\Transformer_and_Multi_Scale_Models\ref_patchformer'
sys.path.insert(0, pf_repo)

# Load PatchFormer's preprocessed data
d = np.load(os.path.join(pf_repo, r'data\CALCE data\CALCE_Data.npy'), allow_pickle=True).item()
test_caps = d['CS2_35']['Capacity'].values.astype(np.float32)
train_caps = np.concatenate([d[c]['Capacity'].values for c in ['CS2_36','CS2_37','CS2_38']])

# Normalize same as training
x_min, x_max = train_caps.min(), train_caps.max()
eol_norm = (0.77 - x_min) / (x_max - x_min)
test_norm = (test_caps - x_min) / (x_max - x_min)
true_eol = int(np.argmax(test_caps < 0.77))

# Find the trained PatchFormer checkpoint (recursively)
ckpts = []
for root, dirs, files in os.walk(os.path.join(pf_repo, 'results_CALCE_RUL_prediction_sl_64', 'CS2_35', 'PatchFormer', 'SP300')):
    for f in files:
        if f.endswith('.ckpt'):
            ckpts.append(os.path.join(root, f))
if not ckpts:
    print("No checkpoint found! Run PatchFormer training first.")
    sys.exit(1)
ckpt = ckpts[0]
print(f'Using checkpoint: {os.path.basename(ckpt)}')

# Import PatchFormer model class
from ModelsModify.PatchFormer import PatchFormerNetModel
from pytorch_forecasting import TimeSeriesDataSet

# Load model
model = PatchFormerNetModel.load_from_checkpoint(ckpt).cuda()
model.eval()
W = 64

print(f'PatchFormer CS2_35: {len(test_caps)} cycles, EOL@0.77Ah={true_eol}')
print(f'Normalization: min={x_min:.4f}, max={x_max:.4f}, EOL threshold={eol_norm:.4f}')
print()

# Use the raw network (bypass pytorch_forecasting wrapper)
raw_net = model.network.cuda()

for SP in [300, 400, 500]:
    window = test_norm[SP-W:SP]
    w = torch.tensor(window, dtype=torch.float32).unsqueeze(-1).unsqueeze(0).cuda()

    # True autoregressive (K=1): predict, feed back, repeat
    preds = []
    for step in range(800):
        with torch.no_grad():
            p = raw_net(w).squeeze().item()
        preds.append(p)
        w = torch.cat([w[:, 1:, :], torch.tensor([[[p]]], dtype=torch.float32, device='cuda')], dim=1)
        if p < eol_norm:
            break

    n_steps = len(preds)
    if n_steps < 800 and n_steps >= 2:
        j = n_steps - 2
        if preds[j] >= eol_norm > preds[j+1]:
            frac = (eol_norm - preds[j]) / (preds[j+1] - preds[j] + 1e-8)
            pe = SP + j + frac
        else:
            pe = SP + j
    elif n_steps < 800:
        pe = SP + n_steps
    else:
        pe = -1
    ae = round(abs(true_eol - pe), 1) if pe >= 0 else -1
    print(f'SP={SP}: AE={ae} cycles, pred_EOL={pe:.1f}, true_EOL={true_eol}, steps={n_steps}')
