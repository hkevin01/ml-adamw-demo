from __future__ import annotations

import argparse
from pathlib import Path

from data import DataConfig, get_dataloaders
from model import MLPClassifier
from train import TrainConfig, fit
from utils import get_device, plot_history, save_history, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch AdamW training demo")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-samples", type=int, default=6000)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--n-classes", type=int, default=2)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=("none", "cosine", "onecycle", "linear"),
        help="Learning-rate scheduler type",
    )
    parser.add_argument(
        "--scheduler-t-max",
        type=int,
        default=None,
        help="Cosine scheduler T_max (defaults to epochs)",
    )
    parser.add_argument(
        "--onecycle-pct-start",
        type=float,
        default=0.3,
        help="OneCycleLR pct_start",
    )
    parser.add_argument(
        "--onecycle-div-factor",
        type=float,
        default=25.0,
        help="OneCycleLR div_factor",
    )
    parser.add_argument(
        "--onecycle-final-div-factor",
        type=float,
        default=1e4,
        help="OneCycleLR final_div_factor",
    )
    parser.add_argument("--disable-scheduler", action="store_true")
    parser.add_argument("--no-amp", action="store_true")

    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="If set, best-val-loss model weights are saved to <dir>/best_model.pt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    data_config = DataConfig(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_classes=args.n_classes,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    train_loader, val_loader = get_dataloaders(data_config)

    model = MLPClassifier(
        input_dim=args.n_features,
        num_classes=args.n_classes,
        hidden_dims=(128, 64),
        dropout=0.1,
    )

    scheduler_name = "none" if args.disable_scheduler else args.scheduler
    scheduler_t_max = args.scheduler_t_max if args.scheduler_t_max is not None else args.epochs
    train_config = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_amp=not args.no_amp,
        scheduler=scheduler_name,
        scheduler_t_max=scheduler_t_max,
        min_lr=args.min_lr,
        onecycle_pct_start=args.onecycle_pct_start,
        onecycle_div_factor=args.onecycle_div_factor,
        onecycle_final_div_factor=args.onecycle_final_div_factor,
        checkpoint_dir=args.checkpoint_dir,
    )

    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=device,
    )

    out_dir = Path(args.artifacts_dir)
    history_path = out_dir / "history.json"
    curves_path = out_dir / "curves.png"
    save_history(history, history_path)
    plot_history(history, curves_path)

    best_val_acc = max(history["val_acc"]) if history["val_acc"] else 0.0
    best_val_loss = min(history["val_loss"]) if history["val_loss"] else float("inf")

    print(f"Saved metrics to: {history_path}")
    print(f"Saved plot to: {curves_path}")
    print(f"Best val_acc: {best_val_acc:.4f}")
    print(f"Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
