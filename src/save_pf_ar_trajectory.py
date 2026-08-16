"""Save PatchFormer true-AR rollout trajectory (CALCE CS2_35, SP=300)
to checkpoints/pf_ar_sp300.npz for figure generation."""
import sys
import os
import numpy as np
import torch

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
CKPT = r'D:\research\degradation_prognostics\Transformer_and_Multi_Scale_Models\checkpoints'

d = np.load(os.path.join(pf_repo, r'data\CALCE data\CALCE_Data.npy'), allow_pickle=True).item()
test_caps = d['CS2_35']['Capacity'].values.astype(np.float32)
train_caps = np.concatenate([d[c]['Capacity'].values for c in ['CS2_36', 'CS2_37', 'CS2_38']])
x_min, x_max = train_caps.min(), train_caps.max()
eol_norm = (0.77 - x_min) / (x_max - x_min)
test_norm = (test_caps - x_min) / (x_max - x_min)

ckpts = []
for root, dirs, files in os.walk(os.path.join(pf_repo, 'results_CALCE_RUL_prediction_sl_64', 'CS2_35', 'PatchFormer', 'SP300')):
    for f in files:
        if f.endswith('.ckpt'):
            ckpts.append(os.path.join(root, f))
if not ckpts:
    print('No PatchFormer checkpoint found')
    sys.exit(1)

from ModelsModify.PatchFormer import PatchFormerNetModel

model = PatchFormerNetModel.load_from_checkpoint(ckpts[0])
model.eval()
raw_net = model.network.cuda()
W = 64
SP = 300

window = test_norm[SP - W:SP]
w = torch.tensor(window, dtype=torch.float32).unsqueeze(-1).unsqueeze(0).cuda()
preds = []
for step in range(800):
    with torch.no_grad():
        p = raw_net(w).squeeze().item()
    preds.append(p)
    w = torch.cat([w[:, 1:, :], torch.tensor([[[p]]], dtype=torch.float32, device='cuda')], dim=1)
    if p < eol_norm:
        break

np.savez(os.path.join(CKPT, 'pf_ar_sp300.npz'),
         preds_norm=np.array(preds, dtype=np.float32),
         true_eol=int(np.argmax(test_caps < 0.77)),
         x_min=x_min, x_max=x_max, sp=SP, n_steps=len(preds))
print(f'saved: {len(preds)} steps, last pred {preds[-1]:.4f} (EOL norm {eol_norm:.4f})')
