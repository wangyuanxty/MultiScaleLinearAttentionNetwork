"""Point every remaining per-window decode at train_per_sp.window_std.

The window-std convention drifted into two camps: the batched forward paths all
used torch tensors (`x[:, :, 0].std(dim=1)` -- UNBIASED, which is what training
does) while the per-window Python loops slid numpy arrays and called `.std()` --
biased, 0.78% smaller at W=64, straight into `y = z*wstd + wmean`.

The first four files were fixed by hand (train_per_sp.eval_sp, ar_probe,
eval_per_sp_existing, eval_multiseed).  This does the rest mechanically, and is
deliberately conservative: it only rewrites the exact call shapes it recognises,
prints every change, and refuses to touch a file whose shape it does not match
-- a silent miss here is a wrong number later, which is the whole reason the
conventions diverged in the first place.

    python fix_eval_std.py            # dry run: show what would change
    python fix_eval_std.py --apply
"""
from __future__ import annotations

import argparse
import os
import re

SRC = os.path.dirname(os.path.abspath(__file__))

# (file, exact old substring, exact new substring).  `window_std` is imported
# from train_per_sp and calls torch.std on a float32 copy -- the same operator
# and dtype training's target scale came from.
SUBS = [
    ("run_ablation.py",
     "wstd = float(win.std()) + 1e-6", "wstd = window_std(win) + 1e-6"),
    ("test_ablation_multi.py",
     "wstd = float(win[:, 0].std()) + EPS", "wstd = window_std(win[:, 0]) + EPS"),
    ("test_ablation_phys.py",
     "wstd = float(win[:, 0].std()) + EPS", "wstd = window_std(win[:, 0]) + EPS"),
    ("test_ablation_robust.py",
     "wstd = float(win[:, 0].std()) + EPS", "wstd = window_std(win[:, 0]) + EPS"),
    ("make_figures_v2.py",
     "wstd = float(win[:, 0].std()) + EPS", "wstd = window_std(win[:, 0]) + EPS"),
    ("make_figures_phys.py",
     "wstd = float(win[:, 0].std()) + EPS", "wstd = window_std(win[:, 0]) + EPS"),
    ("test_physics_extrapolation.py",
     "wstd = float(win[:, 0].std()) + EPS", "wstd = window_std(win[:, 0]) + EPS"),
    ("test_physics_ir_seeds.py",
     "wstd = float(win.std()) + EPS", "wstd = window_std(win) + EPS"),
    ("test_phys_structural.py",
     "wstd = float(win[:, 0].std()) + EPS", "wstd = window_std(win[:, 0]) + EPS"),
    ("test_phys_dropfill.py",
     "wstd = float(win[:, 0].std()) + EPS", "wstd = window_std(win[:, 0]) + EPS"),
    ("test_quantile_uq.py",
     "wstd = float(win[:, 0].std()) + EPS", "wstd = window_std(win[:, 0]) + EPS"),
    ("test_n_channel.py",
     "wstd = float(win[:, 0].std()) + EPS", "wstd = window_std(win[:, 0]) + EPS"),
    ("test_noise_robustness.py",
     "wstd = float(win.std()) + EPS", "wstd = window_std(win) + EPS"),
]

# These two are batched (N, W) numpy arrays, so they need the torch call inline
# rather than the 1-D window_std helper.
BATCHED = [
    ("make_figures.py",
     "wstd = windows.std(axis=1, keepdims=True) + 1e-6",
     "wstd = torch.as_tensor(windows, dtype=torch.float32).std(dim=1, "
     "keepdim=True).numpy() + 1e-6"),
    ("ar_ae_horizon.py",
     "wstd = x[:, :, 0].std(axis=1) + EPS",
     "wstd = torch.as_tensor(x[:, :, 0], dtype=torch.float32).std(dim=1)"
     ".numpy() + EPS"),
]

IMPORT_NEEDED = {
    "train_per_sp": "from train_per_sp import window_std",
    "torch": "import torch",
}


def ensure_import(src: str, need: str, stmt: str) -> tuple:
    """Add `stmt` if `need` is not already imported. Returns (src, added)."""
    if re.search(r"^import %s$|^import %s\b|from %s import" % (need, need, need),
                 src, re.M):
        return src, False
    # insert after the last top-level import line
    lines = src.split("\n")
    last = max(i for i, l in enumerate(lines)
               if re.match(r"^(import |from )\S", l))
    lines.insert(last + 1, stmt)
    return "\n".join(lines), True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    missed = []
    for fname, old, new in SUBS + BATCHED:
        path = os.path.join(SRC, fname)
        if not os.path.exists(path):
            missed.append((fname, old, "FILE MISSING"))
            continue
        src = open(path, encoding="utf-8").read()
        n = src.count(old)
        if n == 0:
            # already converted, or the shape differs -- either way, say so
            if new in src:
                print("  %-32s already done" % fname)
            else:
                missed.append((fname, old, "PATTERN NOT FOUND"))
            continue
        out = src.replace(old, new)
        need = "torch" if (fname, old, new) in BATCHED else "train_per_sp"
        stmt = IMPORT_NEEDED[need]
        out, added = ensure_import(out, need, stmt)
        print("  %-32s %d site(s) rewritten%s"
              % (fname, n, "  + added `%s`" % stmt if added else ""))
        if args.apply:
            open(path, "w", encoding="utf-8").write(out)
    if missed:
        print()
        print("  !! NOT MODIFIED -- check these by hand:")
        for f, o, why in missed:
            print("     %-32s %s   (%s)" % (f, o, why))
    if not args.apply:
        print()
        print("  dry run -- re-run with --apply to write")


if __name__ == "__main__":
    main()
