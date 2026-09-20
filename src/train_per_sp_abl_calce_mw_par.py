"""CALCE per-SP multi_wide, split across two concurrent processes.

Two things have to be arranged that the parent does not do on its own.

Model width.  train_per_sp_ablation.py builds its model through the
module-global build_gdn_model with default width, so d_model=95 (the
capacity-matched point, 484,067 params against xchg's 483,866) is injected
by patching that name, exactly as train_per_sp_ablation_wide.py does.

Concurrent output.  The parent loads the results JSON once at startup and
rewrites the whole dict after every finished (SP, seed), so two processes
sharing one file would clobber each other -- the later writer silently
drops the earlier one's rows.  Each process therefore gets its own file,
selected by the PAR_TAG environment variable.  os.path.exists is redirected
alongside open() so that a restart with the same tag resumes (the parent's
SKIP check reads that path) instead of retraining.

Checkpoints need no redirection: they are per (SP, seed) files, and the two
processes are given disjoint seed ranges, so they never collide.

Nothing in train_per_sp_ablation.py is modified, and no existing path is
written: the tagged JSONs are new names.

Usage (two processes, disjoint seed ranges):
    PAR_TAG=a python train_per_sp_abl_calce_mw_par.py \
        --dataset calce --config multi_wide --start-seed 1 --seeds 5
    PAR_TAG=b python train_per_sp_abl_calce_mw_par.py \
        --dataset calce --config multi_wide --start-seed 6 --seeds 5

Merge the two tagged JSONs into results/per_sp_ablation_calce_multi_wide.json
once both finish.
"""
import builtins
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train_per_sp_ablation as A          # noqa: E402

D_MODEL = 95          # 484,067 params; xchg at d_model=64 is 483,866 (+0.04%)
PREFIX = "results/per_sp_ablation_"
TAG = os.environ.get("PAR_TAG", "a")

# --- capacity-matched width ------------------------------------------------
_orig_build = A.build_gdn_model


def _build_with_width(*args, **kwargs):
    kwargs["d_model"] = D_MODEL
    return _orig_build(*args, **kwargs)


A.build_gdn_model = _build_with_width
A.CONFIGS["multi_wide"] = {"multiscale": True, "stage_query": False}

# --- per-process results file ----------------------------------------------
_orig_open = builtins.open
_orig_exists = os.path.exists


def _redirect(path):
    if isinstance(path, str) and path.startswith(PREFIX) and path.endswith(".json"):
        return f"{path[:-5]}_{TAG}.json"
    return path


def _open(file, *args, **kwargs):
    return _orig_open(_redirect(file), *args, **kwargs)


def _exists(path):
    return _orig_exists(_redirect(path))


builtins.open = _open
os.path.exists = _exists


if __name__ == "__main__":
    A.main()
