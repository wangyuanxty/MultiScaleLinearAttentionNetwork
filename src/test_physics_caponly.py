"""Pure-capacity model + physics regularizer (training-target injection).
Input: [C] window only. Physics (IR/T) used ONLY in loss via r_phys.
Verify gamma_ir/gamma_t/Ea still learn correct directions on CALCE + NASA."""
import sys, numpy as np, torch, torch.nn as nn
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gdn_model import build_gdn_model, masked_mae, PhysicsRegularizer
from load_datasets import load_calce_cells_multivar, load_nasa_multivar

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH, SEED, EPOCHS = 64, 42, 100


def make_data(ds_name):
    """Return caps dict, phys dict (IR[,T]), train/test lists, W, lo, hi."""
    if ds_name == 'calce':
        caps_all, feats_all, _ = load_calce_cells_multivar()
        cells = list(caps_all.keys())
        caps = {c: caps_all[c].copy().astype(np.float32) for c in cells}
        # CALCE feature col 3 = IR (from keep=[0,1,2,3,9])
        phys = {c: feats_all[c][:, 3:4].copy().astype(np.float32) for c in cells}
        train_cells, test_cell = [c for c in cells if c != 'CS2_35'], 'CS2_35'
        W = 64
    else:
        cells = ['B0005','B0006','B0007','B0018']
        caps, phys = {}, {}
        for bat in cells:
            d = load_nasa_multivar(bat)
            caps[bat] = d['capacity'].astype(np.float32)
            phys[bat] = np.stack([d['Re'], d['T_mean']], axis=-1).astype(np.float32)
        train_cells, test_cell = ['B0006','B0007','B0018'], 'B0005'
        W = 30
    all_tr = np.concatenate([caps[c] for c in train_cells])
    lo, hi = all_tr.min(), all_tr.max()
    return caps, phys, train_cells, test_cell, W, lo, hi


def train_and_eval(ds_name, use_phys):
    torch.manual_seed(SEED); np.random.seed(SEED)
    caps, phys_raw, train_cells, test_cell, W, lo, hi = make_data(ds_name)
    def scale(seqs): return [(s-lo)/(hi-lo) for s in seqs]

    # z-score physics per battery
    phys = {}
    for bat in train_cells + [test_cell]:
        f = phys_raw[bat].copy()
        for c in range(f.shape[1]):
            f[:,c] = (f[:,c]-f[:,c].mean())/(f[:,c].std()+1e-6)
        phys[bat] = f

    # Build samples: (cap_window, phys_at_window_end, target)
    samples = []
    for bat in train_cells:
        cap = scale([caps[bat]])[0]
        pf = phys[bat]
        for i in range(W, len(cap)):
            samples.append((cap[i-W:i].copy(), pf[i-1].copy(), cap[i]))

    model = build_gdn_model(multiscale=False, input_dim=1, window_size=W,
                            output_len=1, readout="last").to(DEV)
    phys_reg = PhysicsRegularizer(lambda_=0.1).to(DEV) if use_phys else None
    params = list(model.parameters())
    if phys_reg: params += list(phys_reg.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)

    for ep in range(EPOCHS):
        model.train()
        np.random.shuffle(samples)
        for i in range(0, len(samples), BATCH):
            batch = samples[i:i+BATCH]
            cap = torch.tensor([s[0] for s in batch], dtype=torch.float32).unsqueeze(-1).to(DEV)
            pf  = torch.tensor([s[1] for s in batch], dtype=torch.float32).to(DEV)
            tgt = torch.tensor([s[2] for s in batch], dtype=torch.float32).unsqueeze(-1).to(DEV)
            pred = model(cap)
            loss = masked_mae(pred, tgt, torch.ones_like(tgt))
            if phys_reg:
                loss = loss + phys_reg(pred, cap, pf)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 25 == 0:
            print(f"  E{ep} L={loss.item():.4f}", flush=True)

    model.eval()
    tc = scale([caps[test_cell]])[0]
    pv = []
    with torch.no_grad():
        for i in range(W, len(tc)):
            cin = torch.tensor(tc[i-W:i], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
            pv.append(model(cin).item())
    pv = np.array(pv)[:len(tc)-W]; tv = tc[W:]
    r2 = 1 - np.sum((tv-pv)**2)/np.sum((tv-tv.mean())**2)
    out = {"r2": r2}
    if phys_reg:
        out["gamma_ir"] = nn.functional.softplus(phys_reg.gamma_ir).item()
        out["gamma_t"] = nn.functional.softplus(phys_reg.gamma_t).item()
        out["ea"] = torch.exp(phys_reg.Ea_log).item()*1000
        out["base"] = phys_reg.base.item()
    return out


print("=== CALCE (physics: IR only) ===", flush=True)
r_b = train_and_eval('calce', False)
r_p = train_and_eval('calce', True)
print(f"  baseline R2={r_b['r2']:.4f}")
print(f"  +physics R2={r_p['r2']:.4f}  gamma_ir={r_p['gamma_ir']:.6f}  "
      f"gamma_t={r_p['gamma_t']:.6f}  Ea={r_p['ea']:.0f}  base={r_p['base']:.6f}")

print("\n=== NASA (physics: IR + T) ===", flush=True)
n_b = train_and_eval('nasa', False)
n_p = train_and_eval('nasa', True)
print(f"  baseline R2={n_b['r2']:.4f}")
print(f"  +physics R2={n_p['r2']:.4f}  gamma_ir={n_p['gamma_ir']:.6f}  "
      f"gamma_t={n_p['gamma_t']:.6f}  Ea={n_p['ea']:.0f}  base={n_p['base']:.6f}")

print(f"\n=== FINAL ===")
print(f"CALCE: {r_b['r2']:.4f} -> {r_p['r2']:.4f}  g_ir={r_p['gamma_ir']:.4f}")
print(f"NASA:  {n_b['r2']:.4f} -> {n_p['r2']:.4f}  g_ir={n_p['gamma_ir']:.4f}  g_t={n_p['gamma_t']:.4f}  Ea={n_p['ea']:.0f}")
