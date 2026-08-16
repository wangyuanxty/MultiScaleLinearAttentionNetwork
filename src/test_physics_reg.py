"""Physics regularization validation on synthetic data.

Physics regularizer: L = L_mae + lambda*|r_pred - r_phys|, where
  r_pred = -log(c_{t+1}/c_t) is the model-implied decay rate,
  r_phys = base + gamma_ir*IR + gamma_t*arrhenius(T) is the Arrhenius rate.

IR/T ARE regular input features (no info removal). The constraint is on the
OUTPUT behavior. Key questions:
  1. Do gamma_ir/gamma_t/Ea recover true values? (regularizer must see the signal)
  2. Does regularization help OUT-OF-DISTRIBUTION extrapolation
     (stronger IR + hotter temp than training)?

Configs: A. no-reg (plain MAE), B. reg (MAE + physics reg), both 3ch input [cap, IR, T].
"""
import sys, numpy as np, torch, torch.nn as nn
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gdn_model import build_gdn_model, masked_mae, PhysicsRegularizer
from torch.utils.data import DataLoader
import logging; logging.basicConfig(level=logging.WARNING)

DEV = torch.device("cuda")
W, OUT, BATCH, EPOCHS = 64, 1, 64, 150
SEED = 7

GAMMA_IR_TRUE = 0.01
GAMMA_T_TRUE = 0.003
EA_TRUE = 50000.0
BASE_RATE = 0.0003


def arrhenius_rel(t_c, ea):
    return np.exp(-ea / 8.314 * (1.0 / (t_c + 273.15) - 1.0 / 298.0))


def gen_cell(seq_len=240, seed=0, gamma_ir=GAMMA_IR_TRUE, gamma_t=GAMMA_T_TRUE,
             temp_base=25.0, temp_amp=6.0, ea=EA_TRUE, regen=False):
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
        # capacity regeneration: occasional recovery, e.g. after rest (every ~30 cycles)
        if regen and i % 30 == 0:
            cap[i] *= 1.008  # +0.8% rebound, short-lived
    return cap.astype(np.float32), ir.astype(np.float32), temp.astype(np.float32)


def build_loader(caps, feats):
    samples = []
    for c, p in zip(caps, feats):
        for i in range(len(c) - W):
            samples.append((c[i:i + W, None], p[i:i + W], c[i + W]))
    return DataLoader(samples, BATCH, shuffle=True)


def train_model(ld, use_reg, label):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = build_gdn_model(
        multiscale=False, input_dim=3, window_size=W, output_len=OUT, readout="last",
    ).to(DEV)
    reg = PhysicsRegularizer(base_init=BASE_RATE, lambda_=0.01).to(DEV) if use_reg else None
    params = list(model.parameters()) + (list(reg.parameters()) if reg else [])
    opt = torch.optim.Adam(params, lr=1e-3)
    print(f"[{label}] model={sum(p.numel() for p in model.parameters()):,} "
          f"reg={sum(p.numel() for p in reg.parameters()) if reg else 0}", flush=True)
    for ep in range(EPOCHS):
        model.train()
        for cap, phys, tgt in ld:
            cap, tgt = cap.to(DEV), tgt.to(DEV)
            phys = phys.to(DEV)
            x = torch.cat([cap, phys], dim=-1)
            opt.zero_grad()
            pred = model(x)  # (B, 1)
            loss = masked_mae(pred, tgt.unsqueeze(-1))
            if reg:
                loss = loss + reg(pred, cap, phys[:, -1:, :].squeeze(1))
            loss.backward()
            opt.step()
        if ep % 50 == 0:
            extra = ""
            if reg:
                gi = float(torch.nn.functional.softplus(reg.gamma_ir).item())
                gt = float(torch.nn.functional.softplus(reg.gamma_t).item())
                ea = float(np.exp(reg.Ea_log.item()))
                extra = f" gamma_ir={gi:.3f} gamma_t={gt:.3f} Ea={ea:.0f}"
            print(f"  E{ep} L={loss.item():.4f}{extra}", flush=True)
    return model, reg


def eval_ood(model, lo, hi, label):
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
                xin = torch.cat([cin, fin], dim=-1)
                preds.append(model(xin).item())
            pv = np.array(preds)[:len(cs) - W]; tv = cs[W:]
            r2 = 1 - np.sum((tv - pv) ** 2) / np.sum((tv - tv.mean()) ** 2)
            mae = np.mean(np.abs(pv - tv))
            r2s.append(r2); maes.append(mae)
    print(f"[{label}] OOD R2={np.mean(r2s):.4f} MAE={np.mean(maes):.4f}")
    return np.mean(maes)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-a", action="store_true", help="skip no-reg baseline (already trained)")
    ap.add_argument("--regen", action="store_true", help="add capacity regeneration to synthetic cells")
    args = ap.parse_args()

    caps, irs, temps = [], [], []
    for i in range(8):
        c, ir, tp = gen_cell(seq_len=240, seed=i, regen=args.regen)
        caps.append(c); irs.append(ir); temps.append(tp)
    lo, hi = min(c.min() for c in caps), max(c.max() for c in caps)
    caps_scaled = [(c - lo) / (hi - lo) for c in caps]
    feats = [np.stack([ir, tp], axis=1) for ir, tp in zip(irs, temps)]
    ld = build_loader(caps_scaled, feats)

    need_a = args.skip_a and not args.regen  # regeneration data changes A too
    model_a, _ = (None, None) if need_a else train_model(ld, use_reg=False, label="A: no-reg")
    model_b, reg_b = train_model(ld, use_reg=True, label="B: phys-reg (lam=0.01)")

    if reg_b is not None:
        ea = float(np.exp(reg_b.Ea_log.item()))
        gi = float(torch.nn.functional.softplus(reg_b.gamma_ir).item())
        gt = float(torch.nn.functional.softplus(reg_b.gamma_t).item())
        print(f"\n=== reg parameter recovery ===")
        print(f"gamma_ir: true={GAMMA_IR_TRUE}  learned={gi:.3f}")
        print(f"gamma_t:  true={GAMMA_T_TRUE}  learned={gt:.3f}")
        print(f"Ea:       true={EA_TRUE:.0f}  learned={ea:.0f}")

    print("\n=== OOD: stronger IR + hotter temp ===")
    mae_a = eval_ood(model_a, lo, hi, "A: no-reg   ") if model_a is not None else float("nan")
    mae_b = eval_ood(model_b, lo, hi, "B: phys-reg ")
    print(f"\nOOD MAE: no-reg={mae_a:.4f}  phys-reg={mae_b:.4f}  "
          f"improvement={100 * (mae_a - mae_b) / max(mae_a, 1e-9):.1f}%")

    # regeneration compatibility: holdout cells WITH regen, in-distribution physics
    print("\n=== regen compatibility (holdout cells with capacity regeneration) ===")
    for tag, m in (("A: no-reg  ", model_a), ("B: phys-reg", model_b)):
        if m is None:
            continue
        m.eval()
        maes = []
        with torch.no_grad():
            for seed in range(4):
                c, ir, tp = gen_cell(seq_len=240, seed=200 + seed, regen=True)
                cs = (c - lo) / (hi - lo)
                fh = np.stack([ir, tp], axis=1)
                preds = []
                for i in range(W, len(cs)):
                    cin = torch.tensor(cs[i - W:i], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
                    fin = torch.tensor(fh[i - W:i], dtype=torch.float32).unsqueeze(0).to(DEV)
                    xin = torch.cat([cin, fin], dim=-1)
                    preds.append(m(xin).item())
                pv = np.array(preds)[:len(cs) - W]; tv = cs[W:]
                maes.append(np.mean(np.abs(pv - tv)))
        print(f"[{tag}] regen-holdout MAE={np.mean(maes):.4f}")
    print("done")


if __name__ == "__main__":
    main()
