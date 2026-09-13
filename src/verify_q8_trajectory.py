"""Trajectory-level check for the int8 weight path used by the C deployment.

A single-window comparison cannot see AE: AE is a property of the whole
predicted trajectory (where it first crosses the EOL threshold). This
script loads the exact checkpoint that export_gdn_weights.py hands to the
C code -- single-branch, patch=2, per-window z-score targets -- and
evaluates the CALCE per-SP trajectory twice:

    fp32  : the exported weights as they are
    q8    : the same weights put through the int8 round trip that
            export_gdn_weights.py --quant int8 applies (per-output-channel
            symmetric int8, then dequantized back to fp32)

The q8 variant mirrors the C path (int8 weights, fp32 arithmetic), so the
delta between the two rows is what the C deployment actually costs. The
absolute values are NOT comparable to tab:deploy, which was measured on a
different model (absolute-space targets, multiscale backbone).
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from gdn_model import build_gdn_model  # noqa: E402
from make_figures import load_series  # noqa: E402

CKPT = Path(__file__).parent / "gdn_weights.pt"
SPS = (300, 400, 500)


def quantize_like_export(state):
    """Apply the int8 round trip of export_gdn_weights.write_array.

    2-D tensors (Linear weights, and 3-D depthwise-conv weights squeezed to
    2-D) get one symmetric scale per output channel; everything 1-D stays
    fp32. Returns a new state dict.
    """
    out = {}
    for name, tensor in state.items():
        w = tensor.float()
        squeezed = w.squeeze(1) if w.dim() == 3 else w
        if squeezed.dim() != 2:
            out[name] = w  # biases, RMSNorm weights, A_log, dt
            continue
        scale = squeezed.abs().amax(dim=1, keepdim=True) / 127.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        q = torch.clamp(torch.round(squeezed / scale), -127, 127)
        deq = q * scale
        out[name] = deq.unsqueeze(1) if w.dim() == 3 else deq
    return out


def build_and_load(state):
    model = build_gdn_model(multiscale=False, input_dim=1, window_size=64,
                            output_len=1, readout="last")
    model.load_state_dict(state)
    model.eval()
    return model


def trajectory_predictions(model, tc, W):
    """Non-recursive single step over the whole test cell.

    Each step is fed the true history window; the head emits a per-window
    z-score, which is mapped back to capacity with that window's statistics.
    """
    preds = []
    with torch.no_grad():
        for i in range(W, len(tc)):
            win = tc[i - W:i]
            cin = torch.tensor(win, dtype=torch.float32).view(1, W, 1)
            preds.append(model(cin).item())
    pv = np.array(preds)
    windows = np.stack([tc[i - W:i] for i in range(W, len(tc))])
    return pv * windows.std(axis=1) + windows.mean(axis=1)


def eval_ae(pv, tv, eol_n, sps, W):
    """AE in cycles per starting point, via first threshold crossing.

    `tv` starts at cycle W, so the true crossing index has to be offset by
    W before it can be compared with the absolute cycle numbers below.
    """
    true_eol = int(np.argmax(tv < eol_n)) + W
    aes = []
    for sp in sps:
        seg = pv[sp - 64:]
        pred_eol = -1
        for j in range(len(seg) - 1):
            if seg[j] >= eol_n > seg[j + 1]:
                frac = (eol_n - seg[j]) / (seg[j + 1] - seg[j] + 1e-8)
                pred_eol = sp + j + frac
                break
            if seg[j] < eol_n:
                pred_eol = sp + j
                break
        aes.append(abs(true_eol - pred_eol) if pred_eol >= 0 else float("nan"))
    return true_eol, np.array(aes)


def main():
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    caps, train_cells, test_cell, W, _sps, eol_ah = load_series("calce")
    all_tr = np.concatenate([caps[c] for c in train_cells])
    lo, hi = all_tr.min(), all_tr.max()
    tc = (caps[test_cell] - lo) / (hi - lo + 1e-8)
    eol_n = (eol_ah - lo) / (hi - lo + 1e-8)
    tv = tc[W:]

    print(f"checkpoint : {CKPT.name}  ({len(state)} tensors)")
    print(f"test cell  : {test_cell}  ({len(tc)} cycles, "
          f"EOL threshold {eol_ah} Ah)")
    print()

    for label, st in (("fp32", state), ("int8", quantize_like_export(state))):
        model = build_and_load(st)
        pv = trajectory_predictions(model, tc, W)
        r2 = 1 - np.sum((tv - pv) ** 2) / np.sum((tv - tv.mean()) ** 2)
        true_eol, aes = eval_ae(pv, tv, eol_n, SPS, W)
        print(f"{label:5s}  R2={r2:.4f}  "
              f"AE={'/'.join(f'{a:.1f}' for a in aes)}  "
              f"MAE={np.abs(tv - pv).mean():.4f}")

    print()
    print(f"true EOL at cycle {true_eol}")


if __name__ == "__main__":
    main()
