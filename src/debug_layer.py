"""Full-layer numpy replica vs PyTorch layer-0 output (float32)."""
import torch
import numpy as np
import torch.nn.functional as F
from gdn_model import build_gdn_model

m = build_gdn_model(multiscale=False, input_dim=1, window_size=64,
                    output_len=1, readout="last")
st = torch.load("gdn_weights.pt", map_location="cpu", weights_only=False)
m.load_state_dict(st)
m.eval()
x = np.loadtxt("test_input.csv", dtype=np.float32)
cin = torch.tensor(x).unsqueeze(0).unsqueeze(-1)
gdn = m.branches[0].layers[0]["gdn"]

with torch.no_grad():
    emb = m.branches[0].init_proj(cin)
    out_py = gdn(emb)                       # (1, 32, 64)
    q = F.normalize(gdn.q_conv(gdn.q_proj(emb)), p=2, dim=-1)
    k = F.normalize(gdn.k_conv(gdn.k_proj(emb)), p=2, dim=-1)
    v = gdn.v_conv(gdn.v_proj(emb))
    raw = gdn.f_proj(emb).view(1, 32, 4, 16)
    g = -gdn.A_log.exp().view(1, 1, 4, 1) * F.softplus(
        raw + gdn.dt_bias.view(1, 1, 4, 16))
    b = torch.sigmoid(gdn.b_proj(emb))
    w = torch.sigmoid(gdn.w_proj(emb))
    g_out = gdn.g_proj(emb).view(1, 32, 4, 32)
    Dk, Dv = 16, 32
    q2 = q.view(1, 32, 4, Dk).transpose(1, 2)
    k2 = k.view(1, 32, 4, Dk).transpose(1, 2)
    v2 = v.view(1, 32, 4, Dv).transpose(1, 2)
    g2 = g.transpose(1, 2)
    b2 = b.view(1, 32, 4, Dk).transpose(1, 2)
    w2 = w.view(1, 32, 4, Dv).transpose(1, 2)
    o, S = gdn._scan(q2, k2, v2, b2, w2, g2, return_state=True)

qn = q2[0].numpy(); kn = k2[0].numpy(); vn = v2[0].numpy()
bn = b2[0].numpy(); wn = w2[0].numpy(); dn = np.exp(g2[0].numpy())
L, H = 32, 4
S = np.zeros((H, Dk, Dv), dtype=np.float32)
o_np = np.zeros((H, L, Dv), dtype=np.float32)
for t in range(L):
    for h in range(H):
        qt = qn[h, t]; kt = kn[h, t]; vt = vn[h, t]
        bt = bn[h, t]; wt = wn[h, t]; dt = dn[h, t]
        ke = kt * bt
        Sh = S[h]
        Sn = np.zeros((Dk, Dv), dtype=np.float32)
        for i in range(Dk):
            for j in range(Dv):
                es = np.float32(0)
                for p in range(Dk):
                    es += np.float32(
                        (np.float32(1.0 if p == i else 0.0) - ke[i] * kt[p])
                        * (dt[p] * Sh[p, j]))
                Sn[i, j] = np.float32(es + kt[i] * (vt[j] * wt[j]))
        S[h] = Sn
        for j in range(Dv):
            o_np[h, t, j] = np.float32((Sn[:, j] * qt).sum())

on_w = gdn.out_norm.weight.detach().numpy()
go = g_out[0].detach().numpy()
o_n = np.zeros_like(o_np)
for h in range(H):
    for t in range(L):
        xh = o_np[h, t]
        inv = 1.0 / np.sqrt((xh * xh).sum() / Dv + 1e-5)
        silu = go[t, h] / (1.0 + np.exp(-go[t, h]))
        o_n[h, t] = (xh * inv * on_w) * silu

out_w = gdn.out_proj.weight.detach().numpy()
flat = o_n.reshape(L, -1)
out_np = (out_w @ flat.T).T.astype(np.float32)
nw = m.branches[0].layers[0]["norm"].weight.detach().numpy()
res = out_np + emb[0].detach().numpy()
inv = 1.0 / np.sqrt((res * res).sum(axis=1) / 64 + 1e-5)
res = res * inv[:, None] * nw

np.set_printoptions(precision=6, suppress=True)
print("py layer0 out[0,:4] =", out_py[0, 0, :4].numpy())
print("np layer0 out[0,:4] =", res[0, :4])
print("py layer0 out[5,:4] =", out_py[0, 5, :4].numpy())
print("np layer0 out[5,:4] =", res[5, :4])
print("max |diff| =", np.abs(out_py[0].numpy() - res).max())
