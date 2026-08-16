"""
Training script for battery capacity prediction baselines.

Usage:
    python train.py --model transformer --dataset calce --test_cell CS2_35 --window 64 --epochs 200
    python train.py --model transformer --dataset panasonic --window 64 --epochs 200
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
import json
from datetime import datetime

from data_pipeline import SlidingWindowBuilder, BatteryDegradationDataset
from baseline_transformer import build_transformer_baseline
from gdn_model import build_gdn_model
from load_datasets import load_calce_capacity, load_panasonic_cells, load_gotion_cells, load_mit_stanford

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_trajectory(model, initial_window, eol_threshold, max_steps=2000):
    """
    Recursively predict future capacity until EOL.

    Args:
        model: trained model
        initial_window: (L,) numpy array of last known capacities
        eol_threshold: stop when predicted capacity < this (e.g., 0.7)
        max_steps: safety limit

    Returns:
        pred_trajectory: (T,) predicted future capacities
        eol_cycle: cycle index (from start of prediction) where EOL is reached
    """
    model.eval()
    window = torch.tensor(initial_window, dtype=torch.float32).unsqueeze(-1).unsqueeze(0).to(DEVICE)
    preds = []

    for step in range(max_steps):
        pred = model(window).item()
        preds.append(pred)
        # Slide window: remove first, append prediction
        window = torch.cat([window[:, 1:, :],
                            torch.tensor([[[pred]]], dtype=torch.float32, device=DEVICE)], dim=1)
        if pred < eol_threshold:
            break

    preds = np.array(preds)
    eol_idx = np.argmax(preds < eol_threshold)
    eol_cycle = int(eol_idx) if preds[eol_idx] < eol_threshold else -1

    return preds, eol_cycle


@torch.no_grad()
def evaluate_eol(model, test_capacities, window_size, eol_threshold):
    """
    Evaluate EOL prediction accuracy.

    For each test cell, takes the first 'window_size' known cycles
    as initial context, then recursively predicts until EOL.
    Compares predicted EOL to true EOL.

    Args:
        model: trained model
        test_capacities: Dict[str, np.ndarray] of test cell capacity sequences
        window_size: initial context length
        eol_threshold: capacity threshold for EOL (e.g., 0.7)

    Returns:
        dict with eol_errors (cycles), relative_errors, and trajectory preds
    """
    model.eval()
    eol_errors = []
    relative_errors = []

    for cell_name, caps in test_capacities.items():
        caps = np.asarray(caps, dtype=np.float32)
        true_eol = int(np.argmax(caps < eol_threshold))
        if true_eol == 0 and caps[0] >= eol_threshold:
            continue  # this cell never reached EOL

        # Use first 'window_size' cycles as initial context
        start_idx = max(window_size, 0)
        initial_window = caps[start_idx - window_size : start_idx]

        pred_traj, pred_eol = predict_trajectory(model, initial_window, eol_threshold)
        pred_eol_abs = start_idx + pred_eol if pred_eol >= 0 else -1

        if pred_eol >= 0 and true_eol > 0:
            ae = abs(true_eol - pred_eol_abs)
            re = ae / true_eol
            eol_errors.append(ae)
            relative_errors.append(re)
            logger.info(f"  {cell_name}: true_EOL={true_eol}, pred_EOL={pred_eol_abs}, AE={ae}, RE={re:.4f}")
        else:
            logger.warning(f"  {cell_name}: pred_EOL not reached")

    if eol_errors:
        return {
            "eol_mae": float(np.mean(eol_errors)),
            "eol_rmse": float(np.sqrt(np.mean(np.array(eol_errors) ** 2))),
            "eol_mre": float(np.mean(relative_errors)),
        }
    return {"eol_mae": -1, "eol_rmse": -1, "eol_mre": -1}


def train_epoch_multistep(model, loader, optimizer, device, max_steps=20, teacher_ratio=0.5):
    """
    Multi-step recursive training with scheduled sampling.

    For each sample, unrolls K steps:
      - predict next value
      - with prob teacher_ratio: use ground truth for next input
      - otherwise: use own prediction (free running)
      - accumulate loss over all K steps

    Args:
        max_steps: maximum rollout steps per sample
        teacher_ratio: probability of using ground truth (1.0 = full teacher forcing)
    """
    model.train()
    criterion = nn.L1Loss()
    total_loss, total_samples = 0.0, 0

    for batch_x, batch_y in loader:
        B, L, _ = batch_x.shape
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # For each sample in the batch, find the start of test data
        # We use the sliding window: first L points are context, then rollout
        window = batch_x.clone()  # (B, L, 1) — initial context window
        loss_sum = 0.0

        for step in range(max_steps):
            pred = model(window)  # (B, 1)
            true_next = batch_y  # (B,) — next-step target from loader

            # Loss for this step
            loss_sum += criterion(pred.squeeze(-1), true_next)

            # Slide window: teacher forcing or free running
            if torch.rand(1).item() < teacher_ratio:
                next_val = true_next.view(B, 1, 1)
            else:
                next_val = pred.detach().view(B, 1, 1)  # detach: no grad through time

            window = torch.cat([window[:, 1:, :], next_val], dim=1)

        # Average loss over steps
        loss = loss_sum / max_steps
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B
        total_samples += B

    return total_loss / total_samples


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device).unsqueeze(-1)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).squeeze(-1)
        total_loss += criterion(pred, y).item() * x.size(0)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y.cpu().numpy())
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    return {
        "loss": total_loss / len(loader.dataset),
        "mae": float(np.mean(np.abs(preds - targets))),
        "rmse": float(np.sqrt(np.mean((preds - targets) ** 2))),
        "r2": float(1 - np.sum((targets - preds) ** 2) / np.sum((targets - targets.mean()) ** 2)),
    }


def build_dataloaders_calce(test_cell: str, window_size: int, batch_size: int):
    """Load CALCE data with cell-disjoint split: 3 cells train, 1 test."""
    cells = ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]
    if test_cell not in cells:
        raise ValueError(f"test_cell must be one of {cells}")

    capacities = {}
    for cell in cells:
        caps = load_calce_capacity(cell)
        capacities[cell] = caps

    builder = SlidingWindowBuilder(window_size=window_size, stride=1, normalize="per_dataset")
    train_ds, val_ds, test_ds = builder.build_cell_disjoint(capacities, test_cell=test_cell)

    logger.info(f"CALCE cell-disjoint | test={test_cell} | train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def build_dataloaders_panasonic(test_cell: str, window_size: int, batch_size: int):
    """Load PANASONIC with cell-disjoint split: 2 cells train/val, 1 test."""
    cells = load_panasonic_cells()
    valid_cells = list(cells.keys())
    if test_cell not in valid_cells:
        raise ValueError(f"test_cell must be one of {valid_cells}")

    builder = SlidingWindowBuilder(window_size=window_size, stride=1, normalize="per_dataset")
    train_ds, val_ds, test_ds = builder.build_cell_disjoint(cells, test_cell=test_cell)

    logger.info(f"PANASONIC cell-disjoint | test={test_cell} | train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="transformer", choices=["transformer","gdn"])
    parser.add_argument("--dataset", type=str, default="calce", choices=["calce", "panasonic", "gotion", "mit"])
    parser.add_argument("--test_cell", type=str, default="CS2_35")
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--patch_size", type=int, default=2, help="Patch size (default 2 = PatchFormer-style)")
    parser.add_argument("--multiscale", action="store_true", help="Enable 3-branch multi-scale")
    parser.add_argument("--cross_exchange", action="store_true", help="Enable cross-scale exchange")
    parser.add_argument("--learnable_ps", action="store_true", help="Enable learnable patch sizes")
    parser.add_argument("--eol_threshold", type=float, default=0.7, help="EOL capacity threshold")
    args = parser.parse_args()

    logger.info(f"Device: {DEVICE}")

    # ─── Data ───
    test_capacities = {}  # raw test cell data for EOL evaluation

    if args.dataset == "calce":
        train_caps = {}
        for cell in ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]:
            train_caps[cell] = load_calce_capacity(cell)

        builder = SlidingWindowBuilder(window_size=args.window, stride=1, normalize="per_dataset")
        train_ds, val_ds, test_ds = builder.build_cell_disjoint(train_caps, test_cell=args.test_cell)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        exp_name = f"{args.model}_{args.dataset}_{args.test_cell}"
        # Normalize test cell with SAME params as training
        train_seq = np.concatenate([train_caps[c] for c in train_caps if c != args.test_cell])
        x_min, x_max = train_seq.min(), train_seq.max()
        test_caps_raw = load_calce_capacity(args.test_cell)
        test_capacities[args.test_cell] = (test_caps_raw - x_min) / (x_max - x_min) if x_max > x_min else test_caps_raw
        # EOL threshold in normalized space: 70% of NOMINAL capacity
        eol_raw = 0.77  # 70% of 1.1Ah rated capacity for CALCE
        eol_normalized = (eol_raw - x_min) / (x_max - x_min)
        args.eol_threshold = eol_normalized
        logger.info(f"EOL threshold: raw={eol_raw}Ah, normalized={eol_normalized:.4f} (min={x_min:.4f}, max={x_max:.4f})")

    elif args.dataset == "panasonic":
        train_loader, val_loader, test_loader = build_dataloaders_panasonic(
            args.test_cell, args.window, args.batch_size
        )
        exp_name = f"{args.model}_{args.dataset}_{args.test_cell}"
        cells = load_panasonic_cells()
        test_capacities[args.test_cell] = cells[args.test_cell]

    elif args.dataset == "mit":
        cells = load_mit_stanford()
        # Use first half of cells for training, last for testing
        cell_names = sorted(cells.keys())
        test_cell = cell_names[-1]
        builder = SlidingWindowBuilder(window_size=args.window, stride=1, normalize="per_dataset")
        train_ds, val_ds, test_ds = builder.build_cell_disjoint(cells, test_cell=test_cell)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        exp_name = f"{args.model}_{args.dataset}_{test_cell}"
        train_seq = np.concatenate([cells[c] for c in cell_names if c != test_cell])
        x_min, x_max = train_seq.min(), train_seq.max()
        test_caps_raw = cells[test_cell]
        test_capacities[test_cell] = (test_caps_raw - x_min) / (x_max - x_min)
        eol_raw = 0.88  # 80% of 1.1Ah nominal capacity
        eol_normalized = (eol_raw - x_min) / (x_max - x_min)
        args.eol_threshold = eol_normalized
        logger.info(f"MIT cell-disjoint | test={test_cell} | {len(cells)} cells total")
        logger.info(f"EOL threshold: raw={eol_raw}Ah, normalized={eol_normalized:.4f}")

    else:  # gotion
        cells = load_gotion_cells()
        builder = SlidingWindowBuilder(window_size=args.window, stride=1, normalize="per_dataset")
        train_ds, val_ds, test_ds = builder.build_cell_disjoint(cells, test_cell=args.test_cell)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        exp_name = f"{args.model}_{args.dataset}_{args.test_cell}"
        test_capacities[args.test_cell] = cells[args.test_cell]
        logger.info(f"GOTION cell-disjoint | test={args.test_cell}")

    # ─── Model ───
    if args.model == "transformer":
        model = build_transformer_baseline(window_size=args.window)
    elif args.model == "gdn":
        model = build_gdn_model(
            patch_size=args.patch_size, multiscale=args.multiscale,
            cross_exchange=args.cross_exchange, learnable_ps=args.learnable_ps,
            window_size=args.window)
    model = model.to(DEVICE)
    logger.info(f"Model: {exp_name} | params: {sum(p.numel() for p in model.parameters()):,}")

    # ─── Training ───
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_rmse": []}

    pbar = tqdm(range(args.epochs), desc=exp_name)
    for epoch in pbar:
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_rmse"].append(val_metrics["rmse"])

        pbar.set_postfix(
            train=f"{train_loss:.4f}",
            val=f"{val_metrics['loss']:.4f}",
            mae=f"{val_metrics['mae']:.4f}",
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT_DIR / f"{exp_name}_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # ─── Test (single-step) ───
    model.load_state_dict(torch.load(CHECKPOINT_DIR / f"{exp_name}_best.pt"))
    test_metrics = evaluate(model, test_loader, criterion, DEVICE)
    logger.info(f"Single-step | MAE={test_metrics['mae']:.6f} RMSE={test_metrics['rmse']:.6f} R2={test_metrics['r2']:.6f}")

    # ─── EOL (recursive rollout) ───
    logger.info("EOL prediction (recursive rollout):")
    eol_metrics = evaluate_eol(model, test_capacities, args.window, args.eol_threshold)
    if eol_metrics["eol_mae"] >= 0:
        logger.info(f"EOL | MAE={eol_metrics['eol_mae']:.1f} cycles  RMSE={eol_metrics['eol_rmse']:.1f}  MRE={eol_metrics['eol_mre']:.4f}")
    else:
        logger.info("EOL | failed to predict")

    # ─── Save ───
    results = {
        "experiment": exp_name,
        "single_step_mae": test_metrics["mae"],
        "single_step_rmse": test_metrics["rmse"],
        "single_step_r2": test_metrics["r2"],
        "eol_mae_cycles": eol_metrics.get("eol_mae", -1),
        "eol_rmse_cycles": eol_metrics.get("eol_rmse", -1),
        "eol_mre": eol_metrics.get("eol_mre", -1),
        "params": sum(p.numel() for p in model.parameters()),
        "best_val_loss": best_val_loss,
        "history": {k: [float(v) for v in vals] for k, vals in history.items()},
        "timestamp": datetime.now().isoformat(),
    }
    out_path = CHECKPOINT_DIR / f"{exp_name}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
