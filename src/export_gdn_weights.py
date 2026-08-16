"""Export single-branch GDN-2 weights to C header (v3, current pipeline).

Trains the single-branch model (readout="last") with per-window z-score
targets on CALCE (the current protocol), then exports all tensors to
gdn_weights.h + test input/reference for the C verification chain.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from gdn_model import build_gdn_model, masked_mae
from make_figures import load_series

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W, BATCH, EPOCHS, SEED = 64, 64, 100, 42
EPS = 1e-6


def write_array(f, name, arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:  # depthwise conv weight (C,1,K) -> (C,K)
        arr = arr.squeeze(1)
    if arr.ndim == 1:
        f.write(f"#define {name}_size {len(arr)}\n")
        f.write(f"const float {name}[{len(arr)}] = {{\n")
    elif arr.ndim == 2:
        f.write(f"#define {name}_rows {arr.shape[0]}\n")
        f.write(f"#define {name}_cols {arr.shape[1]}\n")
        f.write(f"const float {name}[{arr.shape[0] * arr.shape[1]}] = {{\n")
    flat = arr.flatten()
    for i, v in enumerate(flat):
        f.write(f" {v:.7f}f," if i % 8 else f"    {v:.7f}f,")
        if (i + 1) % 8 == 0:
            f.write("\n")
    f.write("\n};\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gdn_weights.h")
    ap.add_argument("--from-ckpt", default=None,
                    help="re-export the header from a saved state_dict "
                         "(skips training)")
    args = ap.parse_args()

    if args.from_ckpt:
        state = torch.load(args.from_ckpt, map_location="cpu",
                           weights_only=False)
        with open(args.out, "w") as f:
            f.write("// Auto-generated GDN-2 weights for MCU deployment (v3)\n")
            f.write("// single-branch, patch=2, d_model=64, 2 layers\n")
            f.write("#ifndef GDN_WEIGHTS_H\n#define GDN_WEIGHTS_H\n\n")
            for name, tensor in sorted(state.items()):
                write_array(f, name.replace(".", "_"), tensor.numpy())
            f.write("#endif // GDN_WEIGHTS_H\n")
        print(f"re-exported {len(state)} arrays to {args.out}", flush=True)
        return

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    caps, train_cells, test_cell, W, sps, eol_ah = load_series("calce")
    all_tr = np.concatenate([caps[c] for c in train_cells])
    lo, hi = all_tr.min(), all_tr.max()

    model = build_gdn_model(multiscale=False, input_dim=1, window_size=W,
                            output_len=1, readout="last").to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    X, Y = [], []
    for c in train_cells:
        seq = (caps[c] - lo) / (hi - lo + EPS)
        for i in range(W, len(seq)):
            X.append(seq[i - W:i, None])
            Y.append(seq[i])
    X = np.stack(X).astype(np.float32)
    Y = np.array(Y, dtype=np.float32)
    N = len(X)
    print(f"[train] single-branch z-score (N={N}) ...", flush=True)
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(N)
        for s in range(0, N, BATCH):
            idx = perm[s:s + BATCH]
            x = torch.tensor(X[idx]).to(DEV)
            y = torch.tensor(Y[idx]).to(DEV)
            opt.zero_grad()
            pred = model(x).squeeze(-1)
            wmean = x[:, :, 0].mean(dim=1)
            wstd = x[:, :, 0].std(dim=1) + EPS
            tgt = (y - wmean) / wstd
            loss = masked_mae(pred, tgt, torch.ones_like(y))
            loss.backward()
            opt.step()
        if ep % 25 == 0:
            print(f"    ep{ep} loss={loss.item():.4f}", flush=True)

    state = model.cpu().state_dict()
    with open(args.out, "w") as f:
        f.write("// Auto-generated GDN-2 weights for MCU deployment (v3)\n")
        f.write(f"// single-branch, patch=2, d_model=64, 2 layers, "
                f"params={sum(p.numel() for p in model.parameters()):,}\n")
        f.write("#ifndef GDN_WEIGHTS_H\n#define GDN_WEIGHTS_H\n\n")
        for name, tensor in sorted(state.items()):
            cname = name.replace(".", "_")
            write_array(f, cname, tensor.numpy())
        f.write("#endif // GDN_WEIGHTS_H\n")
    print(f"exported to {args.out}: {len(state)} weight arrays", flush=True)

    # checkpoint for verification
    ckpt = Path(args.out).with_suffix(".pt")
    torch.save(state, ckpt)
    print(f"saved checkpoint: {ckpt}", flush=True)

    # test input + PyTorch reference (raw model output, z-space)
    tc = (caps[test_cell] - lo) / (hi - lo + EPS)
    np.savetxt(Path(args.out).parent / "test_input.csv", tc[0:W], fmt="%.8f")
    model.eval()
    with torch.no_grad():
        cin = torch.tensor(tc[0:W], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        pred = model(cin).item()
    (Path(args.out).parent / "test_py_out.txt").write_text(f"{pred:.10f}")
    print(f"test: PyTorch pred on window[0:64]={pred:.10f}", flush=True)


if __name__ == "__main__":
    main()
