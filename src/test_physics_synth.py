"""Physics gate mechanism validation on synthetic data with known ground truth.

Generate capacity sequences whose decay is EXACTLY driven by the physics gate:
  C_{t+1} = C_t * exp(-(base + gamma_ir_true*IR_t + gamma_t_true*arrhenius(T_t)))
  arrhenius(T) = exp(-Ea_true/R * (1/T - 1/298))   [relative to 298K]

Train the physics-gated model on [C, IR, T], then check whether the learned
gamma_ir, gamma_t, Ea recover the true values. If yes, the gate is learnable;
real-data failures are then a data problem, not a mechanism problem.
"""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gdn_model import build_gdn_model, masked_mae
from data_pipeline import Seq2VecDataset
from torch.utils.data import DataLoader
import logging; logging.basicConfig(level=logging.WARNING)

DEV = torch.device("cuda")
W, OUT, BATCH, EPOCHS = 64, 1, 64, 150
SEED = 7

# ─── ground truth physics (same as test_physics_reg.py) ───────────
GAMMA_IR_TRUE = 0.01     # IR modulation strength
GAMMA_T_TRUE = 0.003     # Arrhenius modulation strength
EA_TRUE = 50000.0        # J/mol (~50 kJ/mol, typical SEI growth)
BASE_RATE = 0.0003       # base per-step decay


def arrhenius_rel(t_c, ea):
    """Relative Arrhenius factor vs 298K. ~1 at 25C, ~3.5 at 45C for Ea=50kJ."""
    return np.exp(-ea / 8.314 * (1.0 / (t_c + 273.15) - 1.0 / 298.0))


def gen_cell(seq_len=240, seed=0, gamma_ir=GAMMA_IR_TRUE, gamma_t=GAMMA_T_TRUE,
             temp_base=25.0, temp_amp=6.0, ea=EA_TRUE):
    rng = np.random.default_rng(seed)
    t = np.arange(seq_len)
    ir = 0.02 + 0.13 * t / seq_len + 0.005 * rng.standard_normal(seq_len)  # 0.02 -> 0.15
    ir = np.clip(ir, 0.01, None)
    temp = temp_base + temp_amp * np.sin(2 * np.pi * t / 60.0) + 1.5 * rng.standard_normal(seq_len)
    cap = np.zeros(seq_len)
    cap[0] = 1.0
    for i in range(1, seq_len):
        rate = BASE_RATE + gamma_ir * ir[i] + gamma_t * arrhenius_rel(temp[i], ea)
        cap[i] = cap[i - 1] * np.exp(-rate) + 0.0002 * rng.standard_normal()
    return cap.astype(np.float32), ir.astype(np.float32), temp.astype(np.float32)


def build_loader(caps, feats):
    samples = []
    for c, p in zip(caps, feats):
        for i in range(len(c) - W):
            samples.append((c[i:i + W, None], p[i:i + W], c[i + W]))
    return DataLoader(samples, BATCH, shuffle=True)


def train_model(ld, use_physics, label):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = build_gdn_model(
        multiscale=False, input_dim=1, window_size=W, output_len=OUT,
        use_physics=use_physics, readout="last",
    ).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"[{label}] {len(ld.dataset)} samples, "
          f"{sum(p.numel() for p in model.parameters()):,} params", flush=True)
    for ep in range(EPOCHS):
        model.train()
        for cap, phys, tgt in ld:
            cap, tgt = cap.to(DEV), tgt.to(DEV)
            phys = phys.to(DEV) if use_physics else None
            opt.zero_grad()
            loss = masked_mae(model(cap, phys=phys).unsqueeze(-1), tgt.unsqueeze(-1))
            loss.backward()
            opt.step()
        if ep % 50 == 0:
            print(f"  E{ep} L={loss.item():.4f}", flush=True)
    return model


def eval_ood(model, use_physics, lo, hi, label):
    """Out-of-distribution cells: stronger IR, hotter temperature."""
    model.eval()
    r2s, maes = [], []
    with torch.no_grad():
        for seed in range(4):
            c, ir, tp = gen_cell(seq_len=240, seed=100 + seed,
                                 gamma_ir=0.03, gamma_t=0.010, temp_base=38.0, temp_amp=7.0)
            cs = (c - lo) / (hi - lo)
            fh = np.stack([ir, tp], axis=1)
            preds = []
            for i in range(W, len(cs)):
                cin = torch.tensor(cs[i - W:i], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
                fin = torch.tensor(fh[i - W:i], dtype=torch.float32).unsqueeze(0).to(DEV)
                preds.append(model(cin, phys=fin if use_physics else None).item())
            pv = np.array(preds)[:len(cs) - W]; tv = cs[W:]
            r2 = 1 - np.sum((tv - pv) ** 2) / np.sum((tv - tv.mean()) ** 2)
            mae = np.mean(np.abs(pv - tv))
            r2s.append(r2); maes.append(mae)
    print(f"[{label}] OOD R2={np.mean(r2s):.4f} MAE={np.mean(maes):.4f} (mean of 4 OOD cells)")
    return np.mean(maes)


def main():
    # in-distribution training cells (mild physics)
    caps, irs, temps = [], [], []
    for i in range(8):
        c, ir, tp = gen_cell(seq_len=240, seed=i)
        caps.append(c); irs.append(ir); temps.append(tp)
    lo, hi = min(c.min() for c in caps), max(c.max() for c in caps)
    caps_scaled = [(c - lo) / (hi - lo) for c in caps]
    feats = [np.stack([ir, tp], axis=1) for ir, tp in zip(irs, temps)]

    ld = build_loader(caps_scaled, feats)
    model_plain = train_model(ld, use_physics=False, label="no-gate")
    model_phys = train_model(ld, use_physics=True, label="physics-gate")

    print("\n=== out-of-distribution: stronger IR + hotter temp ===")
    mae_plain = eval_ood(model_plain, False, lo, hi, "no-gate  ")
    mae_phys = eval_ood(model_phys, True, lo, hi, "physics-gate")
    print(f"\nOOD MAE: no-gate={mae_plain:.4f}  physics-gate={mae_phys:.4f}  "
          f"improvement={100 * (mae_plain - mae_phys) / max(mae_plain, 1e-9):.1f}%")

    g = model_phys.branches[0].layers[0]['gdn']
    gi = float(torch.nn.functional.softplus(g.gamma_ir).item())
    gt = float(torch.nn.functional.softplus(g.gamma_t).item())
    ea = float(np.exp(g.Ea_log.item()))
    print("\n=== gate parameter recovery ===")
    print(f"gamma_ir: true={GAMMA_IR_TRUE}  learned={gi:.4f}")
    print(f"gamma_t:  true={GAMMA_T_TRUE}  learned={gt:.4f}")
    print(f"Ea:       true={EA_TRUE:.0f}  learned={ea * 1000:.0f}")
    print("done")


if __name__ == "__main__":
    main()
