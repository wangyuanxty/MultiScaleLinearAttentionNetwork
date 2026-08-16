"""NASA physics regularization: IR (Re) + T (T_mean) + Arrhenius.
Trains single-branch model with physics regularizer, outputs learned
gamma_ir, gamma_t, Ea, and SOH metrics."""
import sys, numpy as np, torch, torch.nn as nn, os, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gdn_model import GDNBatteryModel, PhysicsRegularizer, masked_mae
from load_datasets import load_nasa_multivar, load_nasa_capacity

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
W, SEED = 30, 42
torch.manual_seed(SEED); np.random.seed(SEED)

# ─── Load NASA data ──────────────────────────────────────────
batteries = ['B0005', 'B0006', 'B0007', 'B0018']
test_bat = 'B0005'
train_bats = [b for b in batteries if b != test_bat]

# Feature layout: [C, V, I, Tmean, Tmax, Re, Rct] → 7 channels
# Physics: ir_ch=5 (Re), t_ch=3 (Tmean)
IR_CH, T_CH = 5, 3

def build_sequences(bat_list):
    xs, ys, physs = [], [], []
    for bat in bat_list:
        data = load_nasa_multivar(bat)
        cap = data['capacity']
        n = len(cap)
        feats = np.stack([
            cap,
            data['V_mean'], data['I_mean'],
            data['T_mean'], data['T_max'],
            data['Re'], data['Rct'],
        ], axis=-1).astype(np.float32)
        cmin, cmax = cap.min(), cap.max()
        cap_n = (cap - cmin) / (cmax - cmin + 1e-6)
        feats[:, 0] = cap_n
        for c in range(1, feats.shape[1]):
            m, s = feats[:, c].mean(), feats[:, c].std()
            if s > 1e-6:
                feats[:, c] = (feats[:, c] - m) / s
        phys = feats[:, [IR_CH, T_CH]].copy()
        for i in range(W, n):
            xs.append(feats[i-W:i])
            ys.append(cap_n[i])
            physs.append(phys[i-1])
    return (torch.tensor(np.stack(xs), dtype=torch.float32),
            torch.tensor(np.array(ys), dtype=torch.float32).unsqueeze(-1),
            torch.tensor(np.stack(physs), dtype=torch.float32))

x_tr, y_tr, p_tr = build_sequences(train_bats)
x_te, y_te, p_te = build_sequences([test_bat])

print(f"Train: {x_tr.shape}, Test: {x_te.shape}")
print(f"Physics dim: {p_tr.shape[-1]} (IR=Re, T=Tmean)")

# ─── Build models ────────────────────────────────────────────
def build_model(use_phys):
    return GDNBatteryModel(
        patch_size=1, multiscale=False, d_model=64, num_layers=2,
        num_heads=4, head_dim=16, expand_v=2.0, conv_size=4, dropout=0.1,
        input_dim=7, window_size=W, output_len=1, num_quantiles=1,
        use_physics=use_phys, ir_ch=IR_CH, t_ch=T_CH, readout="last",
    ).to(DEV)

# Train baseline
print("\n=== Baseline (no physics) ===")
model_base = build_model(False)
opt = torch.optim.AdamW(model_base.parameters(), lr=1e-3, weight_decay=1e-4)
for ep in range(60):
    model_base.train()
    idx = torch.randperm(len(x_tr))
    loss_sum = 0.0
    for i in range(0, len(x_tr), 128):
        b_idx = idx[i:i+128]
        bx, by = x_tr[b_idx].to(DEV), y_tr[b_idx].to(DEV)
        pred = model_base(bx)
        loss = nn.L1Loss()(pred[:, 0], by.squeeze(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += loss.item()
    if ep % 20 == 0:
        model_base.eval()
        with torch.no_grad():
            pred_te = model_base(x_te.to(DEV))
            mae = nn.L1Loss()(pred_te[:, 0], y_te.to(DEV).squeeze(-1)).item()
            ss_res = ((pred_te[:, 0] - y_te.to(DEV).squeeze(-1))**2).sum()
            ss_tot = ((y_te.to(DEV).squeeze(-1) - y_te.mean())**2).sum()
            r2 = 1 - ss_res.item() / (ss_tot.item() + 1e-8)
        print(f"  E{ep}: loss={loss_sum:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

model_base.eval()
with torch.no_grad():
    pred_b = model_base(x_te.to(DEV))
    mae_b = nn.L1Loss()(pred_b[:, 0], y_te.to(DEV).squeeze(-1)).item()
    ss_res = ((pred_b[:, 0] - y_te.to(DEV).squeeze(-1))**2).sum()
    ss_tot = ((y_te.to(DEV).squeeze(-1) - y_te.mean())**2).sum()
    r2_b = 1 - ss_res.item() / (ss_tot.item() + 1e-8)

# Train with physics
print("\n=== With physics regularizer ===")
model_phys = build_model(True)
phys_reg = PhysicsRegularizer(lambda_=0.1).to(DEV)
params = list(model_phys.parameters()) + list(phys_reg.parameters())
opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)

for ep in range(60):
    model_phys.train()
    idx = torch.randperm(len(x_tr))
    l_sum, p_sum = 0.0, 0.0
    for i in range(0, len(x_tr), 128):
        b_idx = idx[i:i+128]
        bx, by, bp = x_tr[b_idx].to(DEV), y_tr[b_idx].to(DEV), p_tr[b_idx].to(DEV)
        pred = model_phys(bx, phys=None)
        l_mae = nn.L1Loss()(pred[:, 0], by.squeeze(-1))
        l_phys = phys_reg(pred[:, 0:1], bx[:, :, 0:1], bp)
        loss = l_mae + l_phys
        opt.zero_grad(); loss.backward(); opt.step()
        l_sum += l_mae.item(); p_sum += l_phys.item()
    if ep % 20 == 0:
        model_phys.eval()
        with torch.no_grad():
            pred_te = model_phys(x_te.to(DEV))
            mae = nn.L1Loss()(pred_te[:, 0], y_te.to(DEV).squeeze(-1)).item()
            ss_res = ((pred_te[:, 0] - y_te.to(DEV).squeeze(-1))**2).sum()
            ss_tot = ((y_te.to(DEV).squeeze(-1) - y_te.mean())**2).sum()
            r2 = 1 - ss_res.item() / (ss_tot.item() + 1e-8)
        g_ir = nn.functional.softplus(phys_reg.gamma_ir).item()
        g_t  = nn.functional.softplus(phys_reg.gamma_t).item()
        ea   = torch.exp(phys_reg.Ea_log).item() * 1000.0
        print(f"  E{ep}: mae={l_sum:.4f} phys={p_sum:.4f}  MAE={mae:.4f} R2={r2:.4f}")
        print(f"       g_ir={g_ir:.4f} g_t={g_t:.4f} Ea={ea:.0f}J/mol base={phys_reg.base.item():.4f}")

model_phys.eval()
with torch.no_grad():
    pred_p = model_phys(x_te.to(DEV))
    mae_p = nn.L1Loss()(pred_p[:, 0], y_te.to(DEV).squeeze(-1)).item()
    ss_res = ((pred_p[:, 0] - y_te.to(DEV).squeeze(-1))**2).sum()
    ss_tot = ((y_te.to(DEV).squeeze(-1) - y_te.mean())**2).sum()
    r2_p = 1 - ss_res.item() / (ss_tot.item() + 1e-8)

g_ir = nn.functional.softplus(phys_reg.gamma_ir).item()
g_t  = nn.functional.softplus(phys_reg.gamma_t).item()
ea   = torch.exp(phys_reg.Ea_log).item() * 1000.0

print(f"\n=== Results ===")
print(f"Baseline:  R2={r2_b:.4f}  MAE={mae_b:.4f}")
print(f"+Physics:  R2={r2_p:.4f}  MAE={mae_p:.4f}")
print(f"  gamma_ir={g_ir:.6f} (should be >0)")
print(f"  gamma_t ={g_t:.6f} (should be >0)")
print(f"  Ea      ={ea:.0f} J/mol")
print(f"  base    ={phys_reg.base.item():.6f}")

torch.save(model_phys.state_dict(), '../checkpoints/best_nasa_physics.pt')
print("Saved to ../checkpoints/best_nasa_physics.pt")
