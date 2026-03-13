"""
Predict missing modality embeddings (Raman/XRD) from the color embedding.

Design choice:
- We predict **embeddings** (not raw spectra) for the missing modalities, which is a common and practical approach in
  missing-modality multimodal learning (predict in representation space, then feed downstream model).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


@dataclass
class CondPredictorConfig:
    in_dim: int = 128
    d_model: int = 128
    use_raman: bool = True
    use_xrd: bool = False
    hidden_dim: int = 256
    n_layers: int = 2
    dropout: float = 0.0


def _make_mlp(in_dim: int, out_dim: int, hidden_dim: int, n_layers: int, dropout: float) -> nn.Sequential:
    layers = []
    dim = in_dim
    for i in range(int(n_layers) - 1):
        layers.append(nn.Linear(dim, hidden_dim))
        layers.append(nn.SiLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class ColorToSpecPredictor(nn.Module):
    """
    Input: color embedding z_c (B,in_dim)
    Output: dict with keys:
      - "raman_emb": (B,d_model) if use_raman
      - "xrd_emb":   (B,d_model) if use_xrd
    """
    def __init__(self, cfg: CondPredictorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.raman_head = _make_mlp(cfg.in_dim, cfg.d_model, cfg.hidden_dim, cfg.n_layers, cfg.dropout) if cfg.use_raman else None
        self.xrd_head = _make_mlp(cfg.in_dim, cfg.d_model, cfg.hidden_dim, cfg.n_layers, cfg.dropout) if cfg.use_xrd else None

    def forward(self, z_color: torch.Tensor) -> Dict[str, torch.Tensor]:
        if z_color.dim() != 2 or z_color.size(-1) != self.cfg.in_dim:
            raise ValueError(f"ColorToSpecPredictor expects (B,{self.cfg.in_dim}), got {tuple(z_color.shape)}")
        out: Dict[str, torch.Tensor] = {}
        if self.raman_head is not None:
            out["raman_emb"] = self.raman_head(z_color)
        if self.xrd_head is not None:
            out["xrd_emb"] = self.xrd_head(z_color)
        return out
