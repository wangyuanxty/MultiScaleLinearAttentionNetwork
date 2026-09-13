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


def _emit_floats(f, values) -> None:
    """Write floats eight per line, each with an `f` suffix."""
    for i, v in enumerate(values):
        f.write(f" {v}f," if i % 8 else f"\n    {v}f,")
    f.write("\n")


def _emit_int8(f, values, per_line: int = 16) -> None:
    for i, v in enumerate(values):
        f.write(f" {v}," if i % per_line else f"\n    {v},")
    f.write("\n")


def write_array(f, name, arr, quant: str = "fp32") -> None:
    """Emit one tensor as C arrays.

    2-D weights follow PyTorch's dynamic-quantization convention: in int8
    mode the matrix is stored as signed bytes plus one fp32 scale per
    output channel (symmetric, s_i = max_j|W_ij| / 127). 1-D tensors
    (biases, RMSNorm weights, A_log, dt) stay fp32 in both modes -- they
    amount to a few KB, and PyTorch keeps biases fp32 as well.
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:  # depthwise conv weight (C,1,K) -> (C,K)
        arr = arr.squeeze(1)
    if arr.ndim == 1:
        f.write(f"#define {name}_size {len(arr)}\n")
        f.write(f"const float {name}[{len(arr)}] = {{")
        _emit_floats(f, [f"{v:.7f}" for v in arr])
        f.write("};\n\n")
        return

    rows, cols = arr.shape
    f.write(f"#define {name}_rows {rows}\n")
    f.write(f"#define {name}_cols {cols}\n")
    if quant == "int8":
        scale = np.abs(arr).max(axis=1) / 127.0
        scale = np.where(scale == 0.0, 1.0, scale)  # all-zero row guard
        q = np.clip(np.round(arr / scale[:, None]), -127, 127).astype(np.int8)
        f.write(f"const signed char {name}_q[{rows * cols}] = {{")
        _emit_int8(f, q.flatten().tolist())
        f.write("};\n")
        f.write(f"const float {name}_s[{rows}] = {{")
        _emit_floats(f, [f"{v:.9g}" for v in scale])
        f.write("};\n\n")
        return

    f.write(f"const float {name}[{rows * cols}] = {{")
    _emit_floats(f, [f"{v:.7f}" for v in arr.flatten()])
    f.write("};\n\n")


def write_header(out_path, state, quant: str, banner: str) -> int:
    """Write every tensor in `state` to `out_path`; returns the array count."""
    with open(out_path, "w") as f:
        f.write("// Auto-generated GDN-2 weights for MCU deployment (v3)\n")
        f.write(f"// {banner}\n")
        f.write(f"// quantization: {quant}\n")
        f.write("#ifndef GDN_WEIGHTS_H\n#define GDN_WEIGHTS_H\n\n")
        for name, tensor in sorted(state.items()):
            write_array(f, name.replace(".", "_"), tensor.numpy(), quant)
        f.write("#endif // GDN_WEIGHTS_H\n")
    return len(state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="output header (default: gdn_weights.h for fp32, "
                         "gdn_weights_q8.h for int8)")
    ap.add_argument("--quant", choices=("fp32", "int8"), default="fp32",
                    help="weight storage; int8 keeps 1-D tensors in fp32")
    ap.add_argument("--from-ckpt", default=None,
                    help="re-export the header from a saved state_dict "
                         "(skips training)")
    args = ap.parse_args()
    # Distinct defaults keep the committed fp32 header from being clobbered
    # by an int8 run (and vice versa).
    if args.out is None:
        args.out = "gdn_weights_q8.h" if args.quant == "int8" else "gdn_weights.h"

    if args.from_ckpt:
        state = torch.load(args.from_ckpt, map_location="cpu",
                           weights_only=False)
        n = write_header(args.out, state, args.quant,
                         "single-branch, patch=2, d_model=64, 2 layers")
        print(f"re-exported {n} arrays to {args.out} ({args.quant})",
              flush=True)
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
    n = write_header(
        args.out, state, args.quant,
        f"single-branch, patch=2, d_model=64, 2 layers, "
        f"params={sum(p.numel() for p in model.parameters()):,}")
    print(f"exported to {args.out}: {n} weight arrays ({args.quant})",
          flush=True)

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
