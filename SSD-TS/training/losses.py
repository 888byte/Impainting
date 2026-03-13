"""Training loss helpers."""
from __future__ import annotations

import torch


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    diff = torch.abs(pred - target) * mask
    return diff.sum() / (mask.sum() + eps)
