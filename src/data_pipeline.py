"""
Sliding-window dataset for seq2vec battery capacity prediction.

Produces (cap_x, feat_x, target_y, mask) where:
  cap_x:   (window_size, 1)        — capacity input
  feat_x:  (window_size, C)        — feature input (None if no features)
  target_y: (output_len,)          — strided future capacity values
  mask:    (output_len,)           — 1.0 for valid positions, 0.0 beyond EOL
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class Seq2VecDataset(Dataset):
    """Seq2vec dataset: sliding window → multi-step strided output."""

    def __init__(
        self,
        cap_sequences: List[np.ndarray],
        window_size: int = 300,
        output_len: int = 128,
        stride: int = 8,
        feat_sequences: Optional[List[np.ndarray]] = None,
    ):
        self.window_size = window_size
        self.output_len = output_len
        self.stride = stride
        self.has_features = feat_sequences is not None
        self.samples = []

        for idx, cap in enumerate(cap_sequences):
            cap = np.asarray(cap, dtype=np.float32)
            n = len(cap)
            if n <= window_size:
                continue

            feat = np.asarray(feat_sequences[idx], dtype=np.float32) if feat_sequences else None

            for i in range(0, n - window_size, 1):
                cap_x = cap[i : i + window_size]
                feat_x = feat[i : i + window_size] if feat is not None else None
                target_start = i + window_size
                target_end = min(n, target_start + output_len * stride)
                future_cap = cap[target_start:target_end]
                target_y = np.full(output_len, np.nan, dtype=np.float32)
                mask = np.zeros(output_len, dtype=np.float32)
                valid_len = min(output_len, max(0, (n - target_start + stride - 1) // stride))
                for j in range(valid_len):
                    t_idx = target_start + j * stride
                    if t_idx < n:
                        target_y[j] = cap[t_idx]
                        mask[j] = 1.0
                self.samples.append((cap_x, feat_x, target_y, mask))

        logger.info(
            f"Seq2Vec: {len(self.samples)} samples from {len(cap_sequences)} seqs "
            f"(win={window_size}, out={output_len}, stride={stride})"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cap_x, feat_x, target_y, mask = self.samples[idx]
        cap_t = torch.tensor(cap_x, dtype=torch.float32).unsqueeze(-1)
        feat_t = torch.tensor(feat_x, dtype=torch.float32) if feat_x is not None else torch.zeros(1, 1)
        tgt_t = torch.tensor(target_y, dtype=torch.float32)
        msk_t = torch.tensor(mask, dtype=torch.float32)
        tgt_t = torch.nan_to_num(tgt_t, nan=0.0)
        return cap_t, feat_t, tgt_t, msk_t


def collate_seq2vec(batch):
    """Collate for Seq2VecDataset. Returns (cap, feat|None, target, mask)."""
    cap, feat, target, mask = zip(*batch)
    cap = torch.stack(cap)
    target = torch.stack(target)
    mask = torch.stack(mask)
    if feat[0].shape[-1] == 1 and feat[0].shape[-2] == 1:
        return cap, None, target, mask
    return cap, torch.stack(feat), target, mask


class SlidingWindowBuilder:
    """Builds seq2vec datasets with leave-one-cell-out split."""

    def __init__(self, window_size=300, output_len=128, stride=8, normalize="per_dataset"):
        self.window_size = window_size
        self.output_len = output_len
        self.stride = stride
        self.normalize = normalize

    def build_cell_disjoint(
        self,
        cell_capacities: Dict[str, np.ndarray],
        test_cell: str,
        cell_features: Optional[Dict[str, np.ndarray]] = None,
        train_ratio: float = 0.8,
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """Train/val/test split: leave-one-cell-out."""
        train_val_cells = [c for c in cell_capacities if c != test_cell]
        train_seqs, val_seqs = [], []
        train_feats, val_feats = [], []

        for cell_name in train_val_cells:
            seq = np.asarray(cell_capacities[cell_name], dtype=np.float32)
            feat = np.asarray(cell_features[cell_name], dtype=np.float32) if cell_features else None
            split = int(len(seq) * train_ratio)
            train_seqs.append(seq[:split])
            val_seqs.append(seq[split:])
            if cell_features and feat is not None:
                train_feats.append(feat[:split])
                val_feats.append(feat[split:])

        test_seqs = [np.asarray(cell_capacities[test_cell], dtype=np.float32)]
        test_feats = [np.asarray(cell_features[test_cell], dtype=np.float32)] if cell_features else None

        if self.normalize == "per_dataset":
            all_train = np.concatenate(train_seqs)
            x_min, x_max = all_train.min(), all_train.max()
            def scale(seqs):
                return [(s - x_min) / (x_max - x_min) if x_max > x_min else np.zeros_like(s) for s in seqs]
            train_seqs = scale(train_seqs)
            val_seqs = scale(val_seqs)
            test_seqs = scale(test_seqs)

        tf = train_feats if cell_features else None
        vf = val_feats if cell_features else None
        tsf = test_feats if cell_features else None

        train_ds = Seq2VecDataset(train_seqs, self.window_size, self.output_len, self.stride, tf)
        val_ds = Seq2VecDataset(val_seqs, self.window_size, self.output_len, self.stride, vf)
        test_ds = Seq2VecDataset(test_seqs, self.window_size, self.output_len, self.stride, tsf)

        logger.info(f"Train: {len(train_ds)} Val: {len(val_ds)} Test: {len(test_ds)}")
        return train_ds, val_ds, test_ds
