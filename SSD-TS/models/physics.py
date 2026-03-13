# -*- coding: utf-8 -*-
"""pigment_task.physics.fading_forward_lab

Physics-inspired forward model + cycle-consistency regularizer for pigment fading.

We work in *normalized Lab* space:
  - L_norm in [0,1] (denorm: *100)
  - a_norm, b_norm in roughly [-1,1] (denorm: *128)

Forward model (simple, differentiable):
    c(d) = c_inf + (c0 - c_inf) * exp(-k * d)

where:
  - c0: predicted original color (t0)
  - d: normalized dose/time in [0,1] (we use index/(L-1) by default)
  - k: fading rate (>=0), optionally dependent on condition embedding
  - c_inf: asymptotic color, optionally dependent on condition embedding

This is NOT a classic PDE-PINN; it's a differentiable dynamics prior (PINN-style constraint)
that integrates smoothly into your diffusion training as an extra loss term.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def warmup_weight(step: int, warmup_steps: int) -> float:
    """Linear warmup from 0->1."""
    if warmup_steps <= 0:
        return 1.0
    return float(min(1.0, max(0.0, step / float(warmup_steps))))


@dataclass
class PhysicsCfg:
    # master switch
    enable: bool = False

    # loss weights / scheduling
    lambda_cycle: float = 0.2
    warmup_steps: int = 2000
    t_max: int = 30  # only apply cycle loss when diffusion timestep <= t_max
    exclude_t0: bool = True

    # parameterization choices
    cond_dependent: bool = True
    cond_hidden: int = 128

    per_channel_k: bool = True  # True: k_L,k_a,k_b; False: one k for all
    learn_c_inf: bool = True    # learn asymptotic color

    # initialization
    init_k: float = 1.0  # softplus(k_raw) ~ init_k


def _expand_batch(x: torch.Tensor, B: int) -> torch.Tensor:
    if x.shape[0] == B:
        return x
    if x.shape[0] == 1:
        return x.expand(B, *x.shape[1:])
    raise ValueError(f"Cannot expand tensor of shape {tuple(x.shape)} to batch {B}")


class FadingForwardModelLab(nn.Module):
    """A tiny differentiable forward model in normalized Lab space."""

    def __init__(self, cond_dim: int, cfg: Optional[PhysicsCfg] = None) -> None:
        super().__init__()
        self.cfg = cfg or PhysicsCfg()
        self.cond_dim = int(cond_dim)

        k_dim = 3 if self.cfg.per_channel_k else 1
        self.k_base_raw = nn.Parameter(torch.zeros(k_dim))

        # Initialize k so that softplus(k_raw) ~ init_k
        with torch.no_grad():
            init = float(self.cfg.init_k)
            # inverse softplus approx: log(exp(k)-1)
            self.k_base_raw.copy_(torch.log(torch.expm1(torch.tensor(init)) + 1e-6).expand_as(self.k_base_raw))

        if self.cfg.learn_c_inf:
            self.cinf_base_raw = nn.Parameter(torch.zeros(3))  # to be squashed to valid range
        else:
            self.register_buffer("cinf_base_raw", torch.zeros(3))

        # cond -> deltas
        self.cond_mlp: Optional[nn.Module]
        if self.cfg.cond_dependent and self.cond_dim > 0:
            out_dim = k_dim + (3 if self.cfg.learn_c_inf else 0)
            self.cond_mlp = nn.Sequential(
                nn.Linear(self.cond_dim, int(self.cfg.cond_hidden)),
                nn.GELU(),
                nn.Linear(int(self.cfg.cond_hidden), out_dim),
            )
        else:
            self.cond_mlp = None

    def _decode_params(self, B: int, cond: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # base
        k_raw = self.k_base_raw.view(1, -1).expand(B, -1)  # (B,k_dim)
        if self.cfg.learn_c_inf:
            cinf_raw = self.cinf_base_raw.view(1, 3).expand(B, 3)  # (B,3)
        else:
            cinf_raw = torch.zeros((B, 3), device=k_raw.device, dtype=k_raw.dtype)

        # condition deltas
        if self.cond_mlp is not None and cond is not None:
            cond = _expand_batch(cond, B)
            delta = self.cond_mlp(cond)
            k_dim = k_raw.shape[1]
            k_raw = k_raw + delta[:, :k_dim]
            if self.cfg.learn_c_inf:
                cinf_raw = cinf_raw + delta[:, k_dim:k_dim + 3]

        # k: positive
        k = F.softplus(k_raw) + 1e-6
        if k.shape[1] == 1:
            k = k.expand(B, 3)

        # cinf: clamp to Lab norm ranges: L in [0,1], ab in [-1,1]
        cinf_L = torch.sigmoid(cinf_raw[:, 0:1])
        cinf_ab = torch.tanh(cinf_raw[:, 1:3])
        cinf = torch.cat([cinf_L, cinf_ab], dim=-1)  # (B,3)

        return k, cinf

    def forward(self, c0: torch.Tensor, dose: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        c0:   (B,3) normalized Lab
        dose: (B,L,1) or (1,L,1) in [0,1]
        return: (B,L,3)
        """
        if c0.ndim != 2 or c0.shape[-1] != 3:
            raise ValueError(f"c0 must be (B,3), got {tuple(c0.shape)}")

        B = c0.shape[0]
        if dose.ndim == 1:
            dose = dose.view(1, -1, 1)
        if dose.ndim != 3 or dose.shape[-1] != 1:
            raise ValueError(f"dose must be (*,L,1), got {tuple(dose.shape)}")

        dose = _expand_batch(dose, B).clamp(0.0, 1.0)

        k, cinf = self._decode_params(B, cond)  # (B,3),(B,3)

        c0_seq = c0.unsqueeze(1)      # (B,1,3)
        cinf_seq = cinf.unsqueeze(1)  # (B,1,3)
        k_seq = k.unsqueeze(1)        # (B,1,3)

        expo = (-k_seq * dose).clamp(min=-50.0, max=0.0)
        w = torch.exp(expo)           # (B,L,3)
        out = cinf_seq + (c0_seq - cinf_seq) * w
        return out

    def cycle_loss(
        self,
        x0_pred: torch.Tensor,
        x0_true: torch.Tensor,
        mask: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Physics cycle consistency:
          c0_hat = x0_pred[:,0]
          c_hat_seq = forward(c0_hat, dose)
          compare only on observed entries (mask==1). Optionally exclude t0 frame.

        If t is provided, only apply to samples with t<=cfg.t_max (stability).
        """
        if x0_true.ndim != 3 or x0_true.shape[-1] != 3:
            raise ValueError(f"x0_true must be (B,L,3), got {tuple(x0_true.shape)}")

        B, L, _ = x0_true.shape
        c0_hat = x0_pred[:, 0, :]  # (B,3)

        # default dose: index/(L-1)
        idx = torch.arange(L, device=x0_true.device, dtype=x0_true.dtype)
        dose = (idx / float(max(L - 1, 1))).view(1, L, 1).expand(B, L, 1)

        pred_seq = self.forward(c0_hat, dose, cond=cond)  # (B,L,3)

        mask_use = mask
        if self.cfg.exclude_t0 and L > 0:
            mask_use = mask_use.clone()
            mask_use[:, 0, :] = 0.0

        diff = (pred_seq - x0_true).abs() * mask_use
        num = diff.sum(dim=(1, 2))
        den = mask_use.sum(dim=(1, 2)).clamp_min(1.0)
        per = num / den  # (B,)

        if t is not None:
            gate = (t <= int(self.cfg.t_max)).float().view(B)
            if gate.sum() > 0:
                return (per * gate).sum() / gate.sum()
            return per.mean()

        return per.mean()
