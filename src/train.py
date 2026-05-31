from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from evaluate import validate
from utils import accuracy_from_logits


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-2
    use_amp: bool = True
    scheduler: str = "cosine"
    scheduler_t_max: int = 30
    min_lr: float = 1e-5
    onecycle_pct_start: float = 0.3
    onecycle_div_factor: float = 25.0
    onecycle_final_div_factor: float = 1e4


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    steps_per_epoch: int,
) -> tuple[torch.optim.lr_scheduler.LRScheduler | None, bool]:
    scheduler_name = config.scheduler.lower()

    if scheduler_name == "none":
        return None, False

    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, config.scheduler_t_max),
            eta_min=config.min_lr,
        )
        return scheduler, False

    if scheduler_name == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.lr,
            epochs=max(1, config.epochs),
            steps_per_epoch=max(1, steps_per_epoch),
            pct_start=config.onecycle_pct_start,
            div_factor=config.onecycle_div_factor,
            final_div_factor=config.onecycle_final_div_factor,
        )
        return scheduler, True

    if scheduler_name == "linear":
        # Keep start_factor fixed and decay to a bounded end factor across epochs.
        end_factor = config.min_lr / config.lr if config.lr > 0 else 1.0
        end_factor = min(1.0, max(0.0, end_factor))
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=end_factor,
            total_iters=max(1, config.epochs),
        )
        return scheduler, False

    raise ValueError(
        f"Unsupported scheduler '{config.scheduler}'. "
        "Expected one of: none, cosine, onecycle, linear."
    )


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scheduler_step_on_batch: bool,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    model.train()
    loss_sum = 0.0
    acc_sum = 0.0
    num_batches = 0

    amp_enabled = use_amp and device.type == "cuda"
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(x)
            loss = criterion(logits, y)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler is not None and scheduler_step_on_batch:
            scheduler.step()

        loss_sum += float(loss.item())
        acc_sum += accuracy_from_logits(logits, y)
        num_batches += 1

    if num_batches == 0:
        return 0.0, 0.0
    return loss_sum / num_batches, acc_sum / num_batches


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
) -> dict[str, list[float]]:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler, scheduler_step_on_batch = _build_scheduler(
        optimizer=optimizer,
        config=config,
        steps_per_epoch=len(train_loader),
    )

    amp_enabled = config.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = _train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_step_on_batch=scheduler_step_on_batch,
            scaler=scaler,
            device=device,
            use_amp=config.use_amp,
        )
        val_loss, val_acc = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=config.use_amp,
        )
        if scheduler is not None and not scheduler_step_on_batch:
            scheduler.step()

        current_lr = float(optimizer.param_groups[0]["lr"])
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch:03d}/{config.epochs:03d} "
            f"| train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"| lr={current_lr:.6f}"
        )

    return history
