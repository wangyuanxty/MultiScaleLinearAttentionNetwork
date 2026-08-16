"""Test variable-length inference with NASA model."""
import sys, numpy as np, torch
sys.path.insert(0, '.')
from gdn_model import build_gdn_model
from load_datasets import load_nasa_capacity

DEV = torch.device('cpu')
model = build_gdn_model(multiscale=False, input_dim=1, window_size=30,
                        output_len=1, readout="last").to(DEV)
ckpt = '../checkpoints/best_nasa_B0005.pt'
sd = torch.load(ckpt, map_location=DEV, weights_only=True)
model.load_state_dict(sd)
model.eval()
print(f"Loaded {ckpt}")

# Use B0005 capacity data
caps = load_nasa_capacity('B0005')
cap_t = torch.tensor(caps, dtype=torch.float32).unsqueeze(-1).to(DEV)
print(f"B0005: {len(caps)} cycles")

# Normalize: simple min-max based on visible range
cap_min, cap_max = float(cap_t.min()), float(cap_t.max())
cap_n = (cap_t - cap_min) / (cap_max - cap_min + 1e-6)

for sp in [50, 70, 90]:
    print(f"\nSP={sp} true={caps[sp]:.4f}")
    with torch.no_grad():
        for wname, ws in [('W=30(training)',30), ('W=60',60), ('W=full',sp)]:
            start = max(0, sp - ws)
            x = cap_n[start:sp].unsqueeze(0)
            pred_n = model(x).item()
            pred = pred_n * (cap_max - cap_min) + cap_min
            print(f"  {wname:16s} L={x.shape[1]:2d} pred={pred:.4f} diff={abs(pred-caps[sp]):.4f}")
