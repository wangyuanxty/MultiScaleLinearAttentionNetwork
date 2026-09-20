"""Full-sequence ablation on TJU and GOTION.

train_ablation_fullseq.py restricts --dataset to calce/nasa/panasonic.  That
list is the only per-dataset code in the file: everything below the flag
resolves the split, window length and EOL threshold through
make_figures.load_series, which already handles TJU and GOTION, and the
training loop, evaluator and output paths are dataset-agnostic.

This wrapper widens the choices in memory and then calls the parent's main(),
so the protocol is the parent's by construction -- including the resume path
that reuses an existing checkpoint and the SKIP for (config, seed) pairs
already in the results JSON.  train_ablation_fullseq.py itself is not
modified, and its existing calce/nasa/panasonic artefacts are untouched
because the output paths carry the dataset name.

Outputs (new paths only):
    checkpoints/abl_seed/{tju,gotion}/{config}/seed{seed}.pt
    results/ablation_fullseq_{tju,gotion}.json   <- relative to cwd (src/results/)

Usage: cd src && python train_ablation_fullseq_more.py \
           --dataset gotion --config single multi xchg --seeds 10
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train_ablation_fullseq as T          # noqa: E402

EXTRA_DATASETS = ["tju", "gotion"]

# The parent calls argparse.ArgumentParser.add_argument("--dataset",
# choices=[...]) at parse time, so widening the stored choices is enough --
# no edit to the parent's source.
_orig_add_argument = argparse.ArgumentParser.add_argument


def _add_argument(self, *names, **kwargs):
    if "--dataset" in names and "choices" in kwargs:
        kwargs["choices"] = list(kwargs["choices"]) + [
            d for d in EXTRA_DATASETS if d not in kwargs["choices"]
        ]
    return _orig_add_argument(self, *names, **kwargs)


argparse.ArgumentParser.add_argument = _add_argument


if __name__ == "__main__":
    T.main()
