from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils import accuracy_from_logits


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    num_batches = 0

    amp_enabled = use_amp and device.type == "cuda"
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(x)
            loss = criterion(logits, y)

        loss_sum += float(loss.item())
        acc_sum += accuracy_from_logits(logits, y)
        num_batches += 1

    # Postcondition: restore training mode so the caller's model state is unchanged.
    model.train()

    if num_batches == 0:
        return 0.0, 0.0
    return loss_sum / num_batches, acc_sum / num_batches
