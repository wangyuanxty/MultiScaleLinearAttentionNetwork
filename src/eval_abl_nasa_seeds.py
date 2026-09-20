"""Re-evaluate the NASA ablation from its stored checkpoints.

The paper's NASA row (single 0.0087+/-0.0002, multi 0.0098+/-0.0014,
main 0.0084+/-0.0004) does not match what is stored in
src/results/ablation_fullseq_nasa.json (0.0088+/-0.0008, 0.0091+/-0.0013,
0.0091+/-0.0012) -- the means differ and so does the ordering.  The
checkpoints themselves are still on disk, so this re-runs the evaluation
directly and reports what the stored weights actually give.

eval_fullseq() is the parent script's own evaluator, imported rather than
reimplemented, so the protocol matches by construction.

Reads only: checkpoints/abl_seed/nasa/{single,multi,xchg}/seed{1..10}.pt
Writes: nothing.

Usage: cd src && python eval_abl_nasa_seeds.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train_ablation_fullseq as T          # noqa: E402

DS = "nasa"
SEEDS = tuple(range(1, 11))
CONFIGS = {
    "single": {"multiscale": False, "stage_query": False},
    "multi": {"multiscale": True, "stage_query": False},
    "xchg": {"multiscale": True, "stage_query": True},
}
ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints", "abl_seed", DS,
)
# Paper, Table 4 (tab:ablation)
PAPER = {
    "single": (0.0087, 0.0002, 0.9932, 1.8),
    "multi": (0.0098, 0.0014, 0.9930, 1.8),
    "xchg": (0.0084, 0.0004, 0.9934, 0.5),
}


def eval_one(path, cfg, caps, test_cell, W, lo, hi, eol_ah):
    """Load one checkpoint and run the parent script's own evaluator."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    model = T.build_gdn_model(
        multiscale=cfg["multiscale"], stage_query=cfg["stage_query"],
        input_dim=1, window_size=W, output_len=1, readout="last",
    ).to(T.DEV)
    model.load_state_dict(sd)
    return T.eval_fullseq(model, caps, test_cell, W, lo, hi, eol_ah)


def main() -> None:
    caps, train_cells, test_cell, W, sps, eol_ah = T.load_series(DS)
    caps = {c: np.asarray(caps[c], dtype=np.float32) for c in caps}
    all_tr = np.concatenate([caps[c] for c in train_cells])
    lo, hi = float(all_tr.min()), float(all_tr.max())
    print(f"{DS}: W={W} test_cell={test_cell} eol_ah={eol_ah} lo={lo:.4f} hi={hi:.4f}\n")

    print(f"{'arm':>8} {'n':>3} | {'re-evaluated from ckpt':>40} | {'paper (Table 4)':>26}")
    for name, cfg in CONFIGS.items():
        ms = []
        for seed in SEEDS:
            p = os.path.join(ROOT, name, f"seed{seed}.pt")
            if not os.path.exists(p):
                continue
            ms.append(eval_one(p, cfg, caps, test_cell, W, lo, hi, eol_ah))
        if not ms:
            print(f"{name:>8}   no checkpoints")
            continue
        mae = np.array([m["mae"] for m in ms])
        r2 = np.array([m["r2"] for m in ms])
        ae = np.array([m["ae"] for m in ms])
        p = PAPER[name]
        print(f"{name:>8} {len(ms):>3} | "
              f"{mae.mean():.4f}+/-{mae.std(ddof=0):.4f} R2 {r2.mean():.4f} AE {ae.mean():.2f} | "
              f"{p[0]:.4f}+/-{p[1]:.4f} R2 {p[2]:.4f} AE {p[3]:.1f}")

    # The directory also holds continuation checkpoints (further training on
    # top of seeds 1-3).  Try each continuation length and see whether any
    # reproduces the paper's row, which differs from the plain seeds above.
    print("\n--- continuation checkpoints (3 seeds each) vs paper ---")
    print(f"{'arm':>8} {'cont':>5} | {'MAE':>18} {'R2':>8} {'AE':>5} | {'paper MAE/R2/AE':>22}")
    for name, cfg in CONFIGS.items():
        p = PAPER[name]
        for cont in (100, 200, 400):
            ms = []
            for seed in (1, 2, 3):
                f = os.path.join(ROOT, name, f"seed{seed}_cont{cont}.pt")
                if not os.path.exists(f):
                    continue
                ms.append(eval_one(f, cfg, caps, test_cell, W, lo, hi, eol_ah))
            if not ms:
                continue
            mae = np.array([m["mae"] for m in ms])
            r2 = np.array([m["r2"] for m in ms])
            ae = np.array([m["ae"] for m in ms])
            print(f"{name:>8} {cont:>5} | "
                  f"{mae.mean():.4f}+/-{mae.std(ddof=0):.4f} {r2.mean():>8.4f} {ae.mean():>5.2f} | "
                  f"{p[0]:.4f} / {p[2]:.4f} / {p[3]:.1f}")


if __name__ == "__main__":
    main()
