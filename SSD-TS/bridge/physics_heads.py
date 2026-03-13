"""Lightweight heads for physics-informed auxiliary constraints."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SpecColorHeadConfig:
    in_dim: int
    hidden_dim: int = 128
    n_layers: int = 2
    dropout: float = 0.0
    out_dim: int = 3


@dataclass
class DamageHeadConfig:
    in_dim: int
    hidden_dim: int = 128
    n_layers: int = 2
    dropout: float = 0.0
    out_dim: int = 1


def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int, dropout: float) -> nn.Sequential:
    layers = []
    dim = in_dim
    for _ in range(max(int(n_layers) - 1, 0)):
        layers.append(nn.Linear(dim, hidden_dim))
        layers.append(nn.SiLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class SpecColorHead(nn.Module):
    def __init__(self, cfg: SpecColorHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = _make_mlp(
            in_dim=int(cfg.in_dim),
            hidden_dim=int(cfg.hidden_dim),
            out_dim=int(cfg.out_dim),
            n_layers=int(cfg.n_layers),
            dropout=float(cfg.dropout),
        )

    def forward(self, pseudo_cond: torch.Tensor) -> torch.Tensor:
        if pseudo_cond.ndim != 2:
            raise ValueError(f"SpecColorHead expects (B,D), got {tuple(pseudo_cond.shape)}")
        return self.net(pseudo_cond)


class DamageHead(nn.Module):
    def __init__(self, cfg: DamageHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = _make_mlp(
            in_dim=int(cfg.in_dim),
            hidden_dim=int(cfg.hidden_dim),
            out_dim=int(cfg.out_dim),
            n_layers=int(cfg.n_layers),
            dropout=float(cfg.dropout),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"DamageHead expects (B,D), got {tuple(features.shape)}")
        return self.net(features).squeeze(-1)
