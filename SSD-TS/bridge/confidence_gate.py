"""Confidence gating for posterior and retrieval fusion."""
from __future__ import annotations

from typing import Dict

import torch


class ConfidenceGate:
    def __call__(self, posterior_info: Dict[str, torch.Tensor], retrieval_info: Dict[str, torch.Tensor]) -> torch.Tensor:
        post_conf = posterior_info.get('confidence', None)
        ret_conf = retrieval_info.get('confidence', None)
        if post_conf is None and ret_conf is None:
            raise ValueError('At least one confidence tensor is required')
        if post_conf is None:
            return torch.zeros_like(ret_conf)
        if ret_conf is None:
            return torch.ones_like(post_conf)
        denom = (post_conf + ret_conf).clamp_min(1e-8)
        return (post_conf / denom).clamp(0.0, 1.0)


def posterior_confidence(entropy: torch.Tensor, num_prototypes: int) -> torch.Tensor:
    if num_prototypes <= 1:
        return torch.ones_like(entropy)
    return (1.0 - entropy / torch.log(torch.tensor(float(num_prototypes), device=entropy.device))).clamp(0.0, 1.0)
