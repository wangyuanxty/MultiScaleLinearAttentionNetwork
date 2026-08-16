"""Quick validation: (1) 2/3/5ch R2 comparison; (2) T+Re joint predictability.
If T+Re(~IR) keep R2 close to full 5ch AND are jointly predictable,
a single 3ch model outputs [C_next, T_next, Re_next] solving autoregression."""
import sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, '.')
from gdn_model import build_gdn_model, masked_mae
from load_datasets import load_nasa_multivar
from data_pipeline import Seq2VecDataset, collate_seq2vec

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH, W, SEED = 64, 30, 42
TRAIN, TEST = ["B0006","B0007","B0018"], "B0005"

def load_nasa_stack(feat_keys):
    caps, feats = {}, {}
    for bat in TRAIN + [TEST]:
        data = load_nasa_multivar(bat)
        caps[bat] = data['capacity'].astype(np.float32)
        if feat_keys:
            feats[bat] = np.stack([data[k] for k in feat_keys], axis=-1).astype(np.float32)
        else:
            feats[bat] = None
    all_cap = np.concatenate([caps[c] for c in TRAIN])
    return caps, feats, all_cap.min(), all_cap.max()

def train_nasa(tag, feat_keys, epochs=30):
    torch.manual_seed(SEED); np.random.seed(SEED)
    caps, feats, lo, hi = load_nasa_stack(feat_keys)
    def scale(seqs): return [(s-lo)/(hi-lo) for s in seqs]
    model = build_gdn_model(
        multiscale=False, input_dim=1+len(feat_keys), window_size=W,
        output_len=1, readout="last",
    ).to(DEV)
    tr = Seq2VecDataset(scale([caps[c] for c in TRAIN]), W, 1, 1,
                        [feats[c] for c in TRAIN] if feat_keys else None)
    ld = torch.utils.data.DataLoader(tr, BATCH, shuffle=True, collate_fn=collate_seq2vec)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        model.train()
        for cap, feat, tgt, msk in ld:
            cap, tgt, msk = cap.to(DEV), tgt.to(DEV), msk.to(DEV)
            fdev = feat.to(DEV) if feat is not None else None
            x = torch.cat([cap, fdev], dim=-1) if fdev is not None else cap
            opt.zero_grad()
            loss = masked_mae(model(x), tgt, msk)
            loss.backward(); opt.step()
        if ep % 10 == 0: print(f"  [{tag}] E{ep} L={loss.item():.4f}", flush=True)
    model.eval()
    tc_test = scale([caps[TEST]])[0]
    ftest = feats[TEST] if feat_keys else None
    preds = []
    with torch.no_grad():
        for i in range(W, len(tc_test)):
            cin = torch.tensor(tc_test[i-W:i], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
            if ftest is not None:
                fin = torch.tensor(ftest[i-W:i], dtype=torch.float32).unsqueeze(0).to(DEV)
                cin = torch.cat([cin, fin], dim=-1)
            preds.append(model(cin).item())
    pv = np.array(preds)[:len(tc_test)-W]; tv = tc_test[W:]
    r2 = 1 - np.sum((tv-pv)**2) / np.sum((tv-tv.mean())**2)
    sps = [50,70,90]
    parts = []
    for sp in sps:
        si, ei = sp-W, min(sp-W+30, len(pv))
        if ei>si: parts.append(f"SP{sp}={np.mean(np.abs(pv[si:ei]-tv[sp:min(sp+30,len(tv))])):.4f}")
    return r2, "; ".join(parts)

# ─── Test 1: input features vs R2 (NASA, 30ep, output=cap only) ──
print("=== Test 1: input features vs R2 (NASA, 30ep, out=C) ===")
feat_sets = [
    ("in[C,V,I,T,Re,Rct]->C", ['V_mean','I_mean','T_mean','Re','Rct']),
    ("in[C,T,Re]->C",         ['T_mean','Re']),
    ("in[C,T]->C",            ['T_mean']),
    ("in[C]->C",              []),
    ("cap-only",         []),
]
for tag, keys in feat_sets:
    r2, mae = train_nasa(tag, keys, 40)
    print(f"  -> {tag}: R2={r2:.4f}  {mae}")

# ─── Test 2: joint [C, T, Re] prediction ──────────────────────
print("\n=== Test 2: joint [C, T, Re] prediction ===")
torch.manual_seed(SEED); np.random.seed(SEED)
caps, feats, lo, hi = load_nasa_stack(['T_mean','Re'])
def scale(seqs): return [(s-lo)/(hi-lo) for s in seqs]
# z-score T and Re
fs = {}
for bat in TRAIN+[TEST]:
    f = feats[bat]; c = caps[bat]
    f[:,0] = (f[:,0]-f[:,0].mean())/(f[:,0].std()+1e-6)
    f[:,1] = (f[:,1]-f[:,1].mean())/(f[:,1].std()+1e-6)
    fs[bat] = f

from torch.utils.data import Dataset as TD
class Joint3ch(TD):
    def __init__(self, cap_seqs, feat_seqs, win):
        self.s = []
        for cap,f in zip(cap_seqs, feat_seqs):
            cap=np.asarray(cap,np.float32); f=np.asarray(f,np.float32)
            for i in range(len(cap)-win):
                self.s.append((cap[i:i+win],f[i:i+win],cap[i+win],f[i+win,0],f[i+win,1]))
    def __len__(self): return len(self.s)
    def __getitem__(self,i):
        c,f,tc,tt,tr=self.s[i]
        return(torch.tensor(c).unsqueeze(-1),torch.tensor(f),torch.tensor([tc,tt,tr]))

tr_j=Joint3ch(scale([caps[c] for c in TRAIN]),[fs[c] for c in TRAIN],W)
ld_j=torch.utils.data.DataLoader(tr_j,BATCH,shuffle=True)
m_j=build_gdn_model(multiscale=False,input_dim=3,window_size=W,output_len=3,readout="last").to(DEV)
opt_j=torch.optim.Adam(m_j.parameters(),lr=1e-3)
for ep in range(30):
    m_j.train()
    for cap,feat,tgt in ld_j:
        cap,feat,tgt=cap.to(DEV),feat.to(DEV),tgt.to(DEV)
        pred=m_j(torch.cat([cap,feat],dim=-1))
        loss=nn.L1Loss()(pred,tgt)
        opt_j.zero_grad();loss.backward();opt_j.step()
    if ep%10==0: print(f"  [joint3ch] E{ep} L={loss.item():.4f}",flush=True)

m_j.eval()
tc_t=scale([caps[TEST]])[0]; ft_t=fs[TEST]
eC,eT,eR=[],[],[]
with torch.no_grad():
    for i in range(W,len(tc_t)):
        cin=torch.tensor(tc_t[i-W:i],dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
        fin=torch.tensor(ft_t[i-W:i],dtype=torch.float32).unsqueeze(0).to(DEV)
        p=m_j(torch.cat([cin,fin],dim=-1)).squeeze(0)
        eC.append(abs(p[0].item()-tc_t[i]))
        eT.append(abs(p[1].item()-ft_t[i,0]))
        eR.append(abs(p[2].item()-ft_t[i,1]))
eC=np.array(eC);eT=np.array(eT);eR=np.array(eR)
# R2 of C channel from joint model
pv_c = []
with torch.no_grad():
    for i in range(W,len(tc_t)):
        cin=torch.tensor(tc_t[i-W:i],dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEV)
        fin=torch.tensor(ft_t[i-W:i],dtype=torch.float32).unsqueeze(0).to(DEV)
        p=m_j(torch.cat([cin,fin],dim=-1)).squeeze(0)
        pv_c.append(p[0].item())
pv_c=np.array(pv_c)[:len(tc_t)-W]; tv_c=tc_t[W:]
r2_joint_c = 1 - np.sum((tv_c-pv_c)**2)/np.sum((tv_c-tv_c.mean())**2)
print(f"\nJoint [C,T,Re]->[C,T,Re]: C-channel R2={r2_joint_c:.4f}")
print(f"  C MAE={eC.mean():.4f}  T MAE={eT.mean():.4f}  Re MAE={eR.mean():.4f}")
t_ref=np.array([ft_t[i,0] for i in range(W,len(tc_t))])
r_ref=np.array([ft_t[i,1] for i in range(W,len(tc_t))])
print(f"  T  MAE/std={eT.mean()/t_ref.std():.3f}")
print(f"  Re MAE/std={eR.mean()/r_ref.std():.3f}")
print(f"\n>>> vs cap-only [C]->[C]: R2 from Test 1 above")
