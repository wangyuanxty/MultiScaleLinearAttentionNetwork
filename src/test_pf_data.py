"""Train GDN on PatchFormer's preprocessed CALCE data for fair comparison."""
import sys, torch, numpy as np
sys.path.insert(0, 'D:/research/degradation_prognostics/Transformer_and_Multi_Scale_Models/src')
from gdn_model import build_gdn_model
from data_pipeline import SlidingWindowBuilder, BatteryDegradationDataset
from torch.utils.data import DataLoader

# Load PatchFormer data
d = np.load(r'D:\research\degradation_prognostics\Transformer_and_Multi_Scale_Models\ref_patchformer\data\CALCE data\CALCE_Data.npy', allow_pickle=True).item()
cells = {k: d[k]['Capacity'].values.astype(np.float32) for k in ['CS2_35','CS2_36','CS2_37','CS2_38']}

W = 64
device = torch.device('cuda')
builder = SlidingWindowBuilder(window_size=W, stride=1, normalize='per_dataset')
train_ds, val_ds, test_ds = builder.build_cell_disjoint(cells, test_cell='CS2_35')
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

model = build_gdn_model(patch_size=2, window_size=W).to(device)
opt = torch.optim.Adam(model.parameters(), lr=5e-4)
criterion = torch.nn.L1Loss()

best_val = float('inf')
patience, wait = 40, 0
for epoch in range(200):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device).unsqueeze(-1)
        opt.zero_grad(); loss = criterion(model(x), y); loss.backward(); opt.step()
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device).unsqueeze(-1)
            val_loss += criterion(model(x), y).item() * x.size(0)
    val_loss /= len(val_ds)
    if val_loss < best_val: best_val = val_loss; wait = 0; torch.save(model.state_dict(), 'gdn_pf_data.pt')
    else: wait += 1
    if wait >= patience: print(f'Early stop at {epoch}'); break
    if epoch % 20 == 0: print(f'{epoch}: val={val_loss:.6f}')

# Evaluate EOL with SAME normalization as training
model.load_state_dict(torch.load('gdn_pf_data.pt'))
model.eval()
test_caps = cells['CS2_35']
# Apply same per_dataset normalization
train_caps = np.concatenate([cells[c] for c in ['CS2_36','CS2_37','CS2_38']])
x_min, x_max = train_caps.min(), train_caps.max()
test_norm = (test_caps - x_min) / (x_max - x_min)
eol_norm = (0.77 - x_min) / (x_max - x_min)  # threshold in normalized space
true_eol = int(np.argmax(test_caps < 0.77))

print(f'min={x_min:.4f}, max={x_max:.4f}, EOL threshold: 0.77Ah -> {eol_norm:.4f} norm')
print(f'Test cell CS2_35: {len(test_caps)} cycles, EOL@0.77Ah={true_eol}')

for SP in [300, 400, 500]:
    start = max(0, SP - W)
    window = test_norm[start:SP]
    w = torch.tensor(window, dtype=torch.float32).view(1,W,1).to(device)
    preds = []
    for step in range(600):
        p = model(w).item()
        preds.append(p)
        w = torch.cat([w[:,1:,:], torch.tensor([[[p]]], dtype=torch.float32, device=device)], dim=1)
        if p < eol_norm: break
    n_steps = len(preds)
    pred_eol = SP + n_steps
    ae = abs(pred_eol - true_eol) if n_steps < 600 else -1
    print(f'SP={SP}: reached={n_steps<600}, AE={ae} cycles, steps={n_steps}')
