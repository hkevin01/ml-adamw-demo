from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, random_split


@dataclass(frozen=True)
class DataConfig:
    n_samples: int = 6000
    n_features: int = 20
    n_classes: int = 2
    val_ratio: float = 0.2
    batch_size: int = 128
    num_workers: int = 0
    seed: int = 42


def _make_gaussian_classification_data(
    n_samples: int,
    n_features: int,
    n_classes: int,
    seed: int,
) -> Tuple[Tensor, Tensor]:
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2")

    g = torch.Generator().manual_seed(seed)
    samples_per_class = n_samples // n_classes
    remainder = n_samples % n_classes

    features = []
    labels = []
    for class_idx in range(n_classes):
        class_count = samples_per_class + (1 if class_idx < remainder else 0)

        center = torch.randn(n_features, generator=g) * 3.0
        scale = 0.8 + 0.4 * torch.rand(1, generator=g).item()
        class_x = center + scale * torch.randn(class_count, n_features, generator=g)
        class_y = torch.full((class_count,), class_idx, dtype=torch.long)

        features.append(class_x)
        labels.append(class_y)

    x = torch.cat(features, dim=0)
    y = torch.cat(labels, dim=0)

    perm = torch.randperm(x.size(0), generator=g)
    x = x[perm]
    y = y[perm]

    x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)
    return x, y


def get_dataloaders(config: DataConfig) -> Tuple[DataLoader, DataLoader]:
    if not 0.0 < config.val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0, 1)")

    x, y = _make_gaussian_classification_data(
        n_samples=config.n_samples,
        n_features=config.n_features,
        n_classes=config.n_classes,
        seed=config.seed,
    )
    dataset = TensorDataset(x, y)

    val_size = int(len(dataset) * config.val_ratio)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(config.seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader
