"""Distillation helpers for posterior bridge training."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def posterior_kl_loss(student_logits: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(student_logits, dim=-1)
    return F.kl_div(log_probs, teacher_probs, reduction='batchmean')


def embedding_distillation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target.detach()) ** 2)
