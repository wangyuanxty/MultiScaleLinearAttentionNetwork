"""Capacity-matched single-branch control for the CALCE per-SP ablation.

Why this exists
---------------
The per-SP ablation compares three configs that differ enormously in size:

    single @ d_model=64    113,674 params
    multi  @ d_model=64    340,506 params   (3.00x single, no cross-exchange)
    xchg   @ d_model=64    483,866 params   (4.26x single, stage-query exchange)

So the comparison is not "multi-scale vs single-scale" but "4.3x the
parameters vs single".  If single at matched capacity matches or beats xchg,
then the multi-scale architecture is not earning its parameters and the
"it just needs more capacity" explanation is dead.  If xchg still wins, the
architecture has real value at equal budget.

Why grow single rather than shrink multi
----------------------------------------
The multi-scale model has a floor: at d_model=24 it is still 289,026 params,
because the per-branch embedding and readout are fixed costs that do not
scale with width.  It cannot be brought down to single's 113,674.  The single
branch has no such ceiling, so the meeting point has to be raised.  Matching
at xchg's existing operating point (483,866) also means the multi-scale side
needs no retraining -- only one new config.

d_model=304 was chosen by sweeping: it gives 483,514 params, within 352
(0.07%) of xchg@64's 483,866.

How
---
train_one() calls the module-global build_gdn_model without a d_model, so
the width is injected by patching that name.  Everything else -- the training
loop, build_windows, eval_sp, the z-score target, the unbiased torch std in
the decode -- is the parent script's own code, unchanged.

train_per_sp_ablation.py itself is not modified.

Outputs (new config dir; nothing existing is overwritten):
    checkpoints/per_sp_abl/calce/single_wide/SP{sp}_seed{seed}.pt
    results/per_sp_ablation_calce_single_wide.json   <- written relative to cwd,
                                                        so it lands in src/results/

Usage: cd src && python train_per_sp_ablation_wide.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train_per_sp_ablation as A          # noqa: E402

D_MODEL = 304
# The parent SKIPs any (SP, seed) already present in its results JSON, so
# raising this from 3 to 10 only trains seeds 4..10 and leaves 1..3 alone.
SEEDS = 10

_orig_build = A.build_gdn_model


def _build_with_width(*args, **kwargs):
    kwargs["d_model"] = D_MODEL
    return _orig_build(*args, **kwargs)


def main(ds: str = "calce") -> None:
    A.build_gdn_model = _build_with_width
    # A new key, so the parent's CONFIGS still describe the d_model=64 arms.
    A.CONFIGS["single_wide"] = {"multiscale": False, "stage_query": False}
    sys.argv = [
        "train_per_sp_ablation.py",
        "--dataset", ds,
        "--config", "single_wide",
        "--seeds", str(SEEDS),
    ]
    A.main()


if __name__ == "__main__":
    # Dataset comes from the command line, read before main() replaces argv.
    main(sys.argv[1] if len(sys.argv) > 1 else "calce")
