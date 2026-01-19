# pixel_brushnet.py
# Pixel-space BrushNet-style dual-branch residual injector (NO VAE)
# - input: color_prior (B,3,H,W) + confidence (B,1,H,W)
# - output: multi-scale residuals to be added into the texture UNet
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _zero_init_conv(conv: nn.Conv2d) -> nn.Conv2d:
    nn.init.zeros_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return conv


class ZeroConv1x1(nn.Module):
    """Zero-initialized 1x1 conv so that injecting this branch starts as no-op."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = _zero_init_conv(nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1, groups: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)
        # groups 需要能整除 out_ch；不整除就降到 16/8/4/1
        g = groups
        while out_ch % g != 0 and g > 1:
            g //= 2
        self.gn = nn.GroupNorm(g, out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class Downsample(nn.Module):
    """Simple strided conv downsample."""
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


@dataclass
class PixelBrushNetConfig:
    in_ch: int = 4                     # 3(color_prior)+1(confidence)
    base_ch: int = 64                  # should match UNet nf (recommended)
    ch_mult: Tuple[int, ...] = (1, 2, 4, 8)
    num_blocks_per_level: int = 2      # lightweight
    conditioning_scale: float = 1.0    # like BrushNet control scale w


class PixelBrushNet(nn.Module):
    """
    Pixel-space BrushNet branch:
    - Extracts multi-scale features from (color_prior, confidence)
    - Produces zero-conv residuals for each resolution level + mid
    """
    def __init__(self, cfg: PixelBrushNetConfig, out_ch_per_level: List[int]):
        super().__init__()
        self.cfg = cfg
        self.out_ch_per_level = out_ch_per_level
        assert len(out_ch_per_level) == len(cfg.ch_mult), \
            f"out_ch_per_level({len(out_ch_per_level)}) must match len(ch_mult)({len(cfg.ch_mult)})"

        self.stem = ConvGNAct(cfg.in_ch, cfg.base_ch)

        # feature extractor
        feats = []
        downs = []
        zeros = []

        ch = cfg.base_ch
        for li, mult in enumerate(cfg.ch_mult):
            level_out = cfg.base_ch * mult
            blocks = [ConvGNAct(ch, level_out)]
            for _ in range(cfg.num_blocks_per_level - 1):
                blocks.append(ConvGNAct(level_out, level_out))
            feats.append(nn.Sequential(*blocks))

            # residual projector to match UNet channels at this level
            zeros.append(ZeroConv1x1(level_out, out_ch_per_level[li]))

            ch = level_out
            if li != len(cfg.ch_mult) - 1:
                downs.append(Downsample(ch))

        self.level_blocks = nn.ModuleList(feats)
        self.downs = nn.ModuleList(downs)
        self.zero_projs = nn.ModuleList(zeros)

        # mid
        self.mid_block = nn.Sequential(
            ConvGNAct(ch, ch),
            ConvGNAct(ch, ch),
        )
        self.mid_zero = ZeroConv1x1(ch, ch)

    @staticmethod
    def _downsample_conf(conf: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        # conf: (B,1,H,W) -> (B,1,h,w)
        return F.interpolate(conf, size=target_hw, mode="bilinear", align_corners=False)

    def forward(
        self,
        color_prior: torch.Tensor,
        confidence: torch.Tensor,
        *,
        conditioning_scale: Optional[float] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Returns:
          - level_residuals: list length L, each residual matches UNet feature channels at that level
          - mid_residual: residual for mid block
        """
        if conditioning_scale is None:
            conditioning_scale = float(self.cfg.conditioning_scale)

        # safety
        assert color_prior.dim() == 4 and confidence.dim() == 4
        assert color_prior.size(1) == 3, f"color_prior should be (B,3,H,W), got {tuple(color_prior.shape)}"
        assert confidence.size(1) == 1, f"confidence should be (B,1,H,W), got {tuple(confidence.shape)}"

        x = torch.cat([color_prior, confidence], dim=1)  # (B,4,H,W)
        x = self.stem(x)

        level_residuals: List[torch.Tensor] = []
        for li, block in enumerate(self.level_blocks):
            x = block(x)

            # per-level confidence gating (downsample to current resolution)
            conf_l = self._downsample_conf(confidence, (x.shape[-2], x.shape[-1]))
            # residual = zero_conv(feat) * conf * scale
            res = self.zero_projs[li](x) * conf_l * conditioning_scale
            level_residuals.append(res)

            if li < len(self.downs):
                x = self.downs[li](x)

        x = self.mid_block(x)
        conf_m = self._downsample_conf(confidence, (x.shape[-2], x.shape[-1]))
        mid_residual = self.mid_zero(x) * conf_m * conditioning_scale

        return level_residuals, mid_residual
