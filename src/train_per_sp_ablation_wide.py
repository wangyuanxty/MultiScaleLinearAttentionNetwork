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

# Capacity-matched controls, both sized against xchg's 483,866 params:
#   single_wide  1 branch,  no exchange, d_model=304 -> 483,514  (-352)
#   multi_wide   3 branches, no exchange, d_model=95 -> ~484,067 (+201)
#
# multi_wide is what isolates the exchange mechanism: comparing it with xchg
# differs only in the cross_stage module, at equal budget.  Comparing multi
# (340,506) with xchg conflates the mechanism with +42% parameters.
# (config, d_model, window_override).  Modes ending in _w64 re-run at W=64
# instead of the dataset default: PANASONIC's trajectories are 920 cycles, but
# the baseline convention sets W=30, which leaves the patch-8 branch only 3-4
# tokens -- the coarse scale the multi-scale design exists to provide is
# degenerate there.  They use distinct config names so their checkpoints and
# results JSON never mix with the W=30 arms.
MODES = {
    # name: (cfg, d_model, window_override, head_dim_override)
    "single_wide":     ({"multiscale": False, "stage_query": False}, 304, None, None),
    "multi_wide":      ({"multiscale": True,  "stage_query": False}, 95,  None, None),
    "single_wide_w64": ({"multiscale": False, "stage_query": False}, 304, 64,   None),
    "multi_wide_w64":  ({"multiscale": True,  "stage_query": False}, 95,  64,   None),
    "xchg_w64":        ({"multiscale": True,  "stage_query": True},  64,  64,   None),
    # State-grown controls: same ~484k budget, but spent on the recurrent
    # state (head_dim) rather than on the in/out projections.  single_state
    # -> 483,778 params, 105 KB/layer; multi_state -> 490,278, 16.5 KB per
    # branch per layer (three branches).  xchg keeps 8 KB/layer.
    "single_state":    ({"multiscale": False, "stage_query": False}, 64,  None, 58),
    "multi_state":     ({"multiscale": True,  "stage_query": False}, 64,  None, 23),
}
# The parent SKIPs any (SP, seed) already present in its results JSON, so
# raising this from 3 to 10 only trains seeds 4..10 and leaves 1..3 alone.
SEEDS = 10

_orig_build = A.build_gdn_model
_orig_load = A.load_series
_TAG = {"d_model": 64, "window": None, "head_dim": None}


def _build_with_width(*args, **kwargs):
    kwargs["d_model"] = _TAG["d_model"]
    if _TAG["head_dim"]:
        kwargs["head_dim"] = _TAG["head_dim"]
    return _orig_build(*args, **kwargs)


def _load_with_window(*args, **kwargs):
    caps, tr, te, W, sps, eol = _orig_load(*args, **kwargs)
    if _TAG["window"]:
        W = _TAG["window"]
    return caps, tr, te, W, sps, eol


def main(ds: str = "calce", mode: str = "single_wide", seeds: int = SEEDS) -> None:
    cfg, d_model, window, head_dim = MODES[mode]
    _TAG["d_model"] = d_model
    _TAG["window"] = window
    _TAG["head_dim"] = head_dim
    A.build_gdn_model = _build_with_width
    A.load_series = _load_with_window
    # A distinct key per mode, so the parent's CONFIGS keep describing the
    # d_model=64 / default-window arms.
    A.CONFIGS[mode] = dict(cfg)
    sys.argv = [
        "train_per_sp_ablation.py",
        "--dataset", ds,
        "--config", mode,
        "--seeds", str(seeds),
    ]
    A.main()


if __name__ == "__main__":
    # Arguments are read before main() replaces argv.
    main(sys.argv[1] if len(sys.argv) > 1 else "calce",
         sys.argv[2] if len(sys.argv) > 2 else "single_wide",
         int(sys.argv[3]) if len(sys.argv) > 3 else SEEDS)
