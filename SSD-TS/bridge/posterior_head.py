"""Posterior head for prototype assignment from RGB/color embeddings."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PosteriorHeadConfig:
    in_dim: int
    num_prototypes: int
    hidden_dim: int = 256
    n_layers: int = 2
    dropout: float = 0.0


class PosteriorHead(nn.Module):
    def __init__(self, cfg: PosteriorHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        layers = []
        dim = cfg.in_dim
        for _ in range(int(cfg.n_layers) - 1):
            layers.append(nn.Linear(dim, cfg.hidden_dim))
            layers.append(nn.SiLU())
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            dim = cfg.hidden_dim
        layers.append(nn.Linear(dim, cfg.num_prototypes))
        self.net = nn.Sequential(*layers)

    def forward(self, z_color: torch.Tensor) -> torch.Tensor:
        return self.net(z_color)
