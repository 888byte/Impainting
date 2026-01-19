"""
Mamba-based encoder for large 1D spectra (Raman/XRD).

Input: (B, L_spec) float32
Output embedding: (B, d_model) float32

We then concatenate embeddings across modalities and optionally fuse (Linear+Norm).

Patch v2 additions:
- Optional peak feature projection (raman_peaks/xrd_peaks) added to embeddings as an auxiliary feature stream.
- Optionally return per-modality embeddings for missing-modality training/inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from .mamba_wrappers import SequenceMambaBlock


class MambaSpectralEncoder(nn.Module):
    def __init__(
        self,
        spec_len: int,
        d_model: int = 128,
        n_layers: int = 4,
        dropout: float = 0.0,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.spec_len = int(spec_len)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.dropout = float(dropout)
        self.pooling = pooling

        self.in_proj = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList([SequenceMambaBlock(d_model=d_model, dropout=dropout) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        spec: (B, L_spec)
        returns: (B, d_model)
        """
        if spec.ndim != 2:
            raise ValueError(f"spec must be (B,L), got {tuple(spec.shape)}")
        x = spec.unsqueeze(-1)  # (B,L,1)
        x = self.in_proj(x)     # (B,L,D)
        for blk in self.blocks:
            x = blk(x)
        x = self.out_norm(x)
        if self.pooling == "mean":
            return x.mean(dim=1)
        else:
            return x[:, -1, :]


@dataclass
class ConditionerConfig:
    use_raman: bool = False
    use_xrd: bool = False
    raman_len: int = 1024
    xrd_len: int = 2048
    d_model: int = 128
    n_layers: int = 4
    dropout: float = 0.0

    # Optional engineered features (peak tables)
    raman_peak_dim: int = 0   # e.g., 2*top_k
    xrd_peak_dim: int = 0

    # Whether to apply a small fusion MLP+LN after concatenation
    use_fuse: bool = True


class MultimodalConditioner(nn.Module):
    """
    Convert Raman/XRD spectra into a condition vector.

    - For each enabled modality, we encode it to (B,d_model).
    - Optionally add a projected peak feature vector into the same embedding.
    - Concatenate across modalities -> (B, cond_dim).
    - Optionally apply fuse (Linear+SiLU+LayerNorm).
    """
    def __init__(self, cfg: ConditionerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.raman_enc = MambaSpectralEncoder(
            spec_len=cfg.raman_len, d_model=cfg.d_model, n_layers=cfg.n_layers, dropout=cfg.dropout
        ) if cfg.use_raman else None
        self.xrd_enc = MambaSpectralEncoder(
            spec_len=cfg.xrd_len, d_model=cfg.d_model, n_layers=cfg.n_layers, dropout=cfg.dropout
        ) if cfg.use_xrd else None

        self.raman_peak_proj = None
        if cfg.use_raman and int(cfg.raman_peak_dim) > 0:
            self.raman_peak_proj = nn.Sequential(
                nn.Linear(int(cfg.raman_peak_dim), cfg.d_model),
                nn.SiLU(),
                nn.LayerNorm(cfg.d_model),
            )

        self.xrd_peak_proj = None
        if cfg.use_xrd and int(cfg.xrd_peak_dim) > 0:
            self.xrd_peak_proj = nn.Sequential(
                nn.Linear(int(cfg.xrd_peak_dim), cfg.d_model),
                nn.SiLU(),
                nn.LayerNorm(cfg.d_model),
            )

        cond_dim = 0
        if self.raman_enc is not None:
            cond_dim += cfg.d_model
        if self.xrd_enc is not None:
            cond_dim += cfg.d_model
        self.cond_dim = cond_dim

        if cond_dim == 0 or not cfg.use_fuse:
            self.fuse = nn.Identity()
        else:
            self.fuse = nn.Sequential(
                nn.Linear(cond_dim, cond_dim),
                nn.SiLU(),
                nn.LayerNorm(cond_dim),
            )

    def encode_modalities(
        self,
        raman: Optional[torch.Tensor],
        xrd: Optional[torch.Tensor],
        raman_peaks: Optional[torch.Tensor] = None,
        xrd_peaks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return per-modality embeddings (before concatenation/fuse)."""
        embeds: Dict[str, torch.Tensor] = {}
        if self.raman_enc is not None:
            if raman is None:
                raise ValueError("use_raman=True but raman is None")
            z = self.raman_enc(raman)
            if self.raman_peak_proj is not None and raman_peaks is not None:
                z = z + self.raman_peak_proj(raman_peaks)
            embeds["raman"] = z
        if self.xrd_enc is not None:
            if xrd is None:
                raise ValueError("use_xrd=True but xrd is None")
            z = self.xrd_enc(xrd)
            if self.xrd_peak_proj is not None and xrd_peaks is not None:
                z = z + self.xrd_peak_proj(xrd_peaks)
            embeds["xrd"] = z
        return embeds

    def forward(
        self,
        raman: Optional[torch.Tensor],
        xrd: Optional[torch.Tensor],
        raman_peaks: Optional[torch.Tensor] = None,
        xrd_peaks: Optional[torch.Tensor] = None,
        return_embeds: bool = False,
    ) -> Union[Optional[torch.Tensor], Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]]:
        embeds = self.encode_modalities(raman, xrd, raman_peaks=raman_peaks, xrd_peaks=xrd_peaks)

        if not embeds:
            return (None, {}) if return_embeds else None

        feats = []
        if "raman" in embeds:
            feats.append(embeds["raman"])
        if "xrd" in embeds:
            feats.append(embeds["xrd"])
        cond_cat = torch.cat(feats, dim=-1)
        cond = self.fuse(cond_cat)

        if return_embeds:
            return cond, embeds
        return cond
