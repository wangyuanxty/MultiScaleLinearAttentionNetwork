"""Dump PyTorch intermediates for staged C-vs-PyTorch comparison."""
import numpy as np
import torch
from gdn_model import build_gdn_model

m = build_gdn_model(multiscale=False, input_dim=1, window_size=64,
                    output_len=1, readout="last")
m.load_state_dict(torch.load("gdn_weights.pt", map_location="cpu",
                             weights_only=False))
m.eval()
x = np.loadtxt("test_input.csv", dtype=np.float32)
cin = torch.tensor(x).unsqueeze(0).unsqueeze(-1)

caps = {}

def hk(name):
    def hook(module, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        caps[name] = o.detach().numpy()
    return hook

handles = [
    m.branches[0].layers[0]["gdn"].register_forward_hook(hk("l0gdn")),
    m.branches[0].layers[1]["gdn"].register_forward_hook(hk("l1gdn")),
]
with torch.no_grad():
    pred = m(cin).item()
for h in handles:
    h.remove()

emb = m.branches[0].init_proj(cin).detach().numpy()
l0 = m.branches[0].layers[0]["norm"](
    torch.tensor(caps["l0gdn"]) + torch.tensor(emb)).detach().numpy()
l1 = m.branches[0].layers[1]["norm"](
    torch.tensor(caps["l1gdn"]) + torch.tensor(l0)).detach().numpy()

np.set_printoptions(precision=6, suppress=True, linewidth=200)
print("pred   =", pred)
print("emb    =", emb[0, :4])
print("l0gdn  =", caps["l0gdn"][0, :4])
print("l0norm =", l0[0, :4])
print("l1gdn  =", caps["l1gdn"][0, :4])
print("l1norm =", l1[0, :4])
# head input = last token (position 63) of l1norm
print("head_in=", l1[0, -1, :4])
