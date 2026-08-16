"""EOL evaluation: load trained model checkpoint, run recursive rollout."""
import argparse, torch, numpy as np, logging
from pathlib import Path
from baseline_transformer import build_transformer_baseline
from gdn_model import build_gdn_model
from load_datasets import load_calce_capacity, load_panasonic_cells

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def rollout(model, window, threshold, max_steps=2000):
    model.eval()
    w = torch.tensor(window, dtype=torch.float32).unsqueeze(-1).unsqueeze(0).to(DEVICE)
    preds = []
    for _ in range(max_steps):
        p = model(w).item()
        preds.append(p)
        w = torch.cat([w[:, 1:, :], torch.tensor([[[p]]], dtype=torch.float32, device=DEVICE)], dim=1)
        if p < threshold:
            break
    preds = np.array(preds)
    idx = np.argmax(preds < threshold)
    return preds, int(idx) if preds[idx] < threshold else -1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model_type", choices=["transformer","gdn"], default="transformer")
    p.add_argument("--patch_size", type=int, default=2)
    p.add_argument("--multiscale", action="store_true")
    p.add_argument("--cross_exchange", action="store_true")
    p.add_argument("--dataset", choices=["calce","panasonic"], required=True)
    p.add_argument("--test_cell", required=True)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--eol", type=float, default=0.7)
    args = p.parse_args()

    builders = {"transformer": build_transformer_baseline, "gdn": build_gdn_model}
    model = builders[args.model_type](window_size=args.window) if args.model_type == "transformer" \
        else builders[args.model_type](patch_size=args.patch_size, multiscale=args.multiscale,
                                        cross_exchange=args.cross_exchange, window_size=args.window)
    model = model.to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.eval()

    cells = {"calce": lambda: {args.test_cell: load_calce_capacity(args.test_cell)},
             "panasonic": load_panasonic_cells}[args.dataset]()

    for name, caps in cells.items():
        caps = np.asarray(caps, dtype=np.float32)
        true_eol = int(np.argmax(caps < args.eol))
        start = max(args.window, 0)
        _, pred_eol = rollout(model, caps[start - args.window : start], args.eol)
        pred_abs = start + pred_eol if pred_eol >= 0 else -1
        logger.info(f"{name}: true={true_eol}, pred={pred_abs}")
        if pred_eol >= 0:
            ae, re = abs(true_eol - pred_abs), abs(true_eol - pred_abs) / true_eol
            logger.info(f"  AE={ae} cycles, RE={re:.4f} ({re*100:.1f}%)")


if __name__ == "__main__":
    main()
