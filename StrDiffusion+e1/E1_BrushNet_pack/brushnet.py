"""
E1 BrushNet conditioning module (generic PyTorch implementation)

Goal:
- Add a "Brush Encoder" branch that extracts multi-scale features from masked input
  and injects them into an existing Texture U-Net via ZeroConv residual injection.

This file is repo-agnostic: you will still need to wire it into your StrDiffusion
Texture U-Net forward pass.

Reference (your plan):
- Brush Encoder input: [I_deg*(1-M), M], hole region is 0, mask indicates missing.
- Multi-scale inject into Texture U-Net (ZeroConv residual injection, init=0, stable).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroConv2d(nn.Conv2d):
    """
    1x1 conv initialized to all zeros (weights and bias).
    Common trick to make injection start as no-op and stabilize training.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.gn = nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch, eps=1e-6)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


@dataclass
class BrushEncoderConfig:
    in_channels: int = 4              # [RGB(3) + M(1)] by default
    base_channels: int = 64
    num_scales: int = 5               # produce feats at 1,1/2,1/4,1/8,1/16
    max_channels: int = 512


class BrushEncoder(nn.Module):
    """
    Simple pyramid encoder returning multi-scale features.

    Output order:
      feats[0] -> same resolution as input (1x)
      feats[1] -> /2
      feats[2] -> /4
      ...
    """
    def __init__(self, cfg: BrushEncoderConfig):
        super().__init__()
        self.cfg = cfg

        chs: List[int] = []
        c = cfg.base_channels
        for _ in range(cfg.num_scales):
            chs.append(min(c, cfg.max_channels))
            c *= 2

        self.stem = ConvBlock(cfg.in_channels, chs[0], stride=1)
        blocks = []
        in_c = chs[0]
        for i in range(1, cfg.num_scales):
            out_c = chs[i]
            blocks.append(nn.Sequential(
                ConvBlock(in_c, out_c, stride=2),   # downsample
                ConvBlock(out_c, out_c, stride=1),
            ))
            in_c = out_c
        self.down_blocks = nn.ModuleList(blocks)

    @property
    def out_channels_per_scale(self) -> List[int]:
        chs: List[int] = []
        c = self.cfg.base_channels
        for _ in range(self.cfg.num_scales):
            chs.append(min(c, self.cfg.max_channels))
            c *= 2
        return chs

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats: List[torch.Tensor] = []
        h = self.stem(x)
        feats.append(h)
        for blk in self.down_blocks:
            h = blk(h)
            feats.append(h)
        return feats


class BrushInjector(nn.Module):
    """
    Creates one ZeroConv per scale to map Brush features into the target channel dim.

    You must tell it what the target channel count is per UNet injection site.
    For example, if you inject into UNet hidden states at 5 resolutions, pass those 5 channel numbers.

    It will resize brush_feats[k] spatially to match the UNet tensor and add a residual.
    """
    def __init__(self, brush_channels: Sequence[int], target_channels: Sequence[int]):
        super().__init__()
        if len(brush_channels) < len(target_channels):
            raise ValueError(f"brush_channels ({len(brush_channels)}) < target_channels ({len(target_channels)})")
        self.num_sites = len(target_channels)
        self.proj = nn.ModuleList([
            ZeroConv2d(int(brush_channels[i]), int(target_channels[i]))
            for i in range(self.num_sites)
        ])

    def inject(self, h: torch.Tensor, brush_feat: torch.Tensor, site_idx: int) -> torch.Tensor:
        if site_idx >= self.num_sites:
            raise IndexError(site_idx)
        if brush_feat.shape[-2:] != h.shape[-2:]:
            brush_feat = F.interpolate(brush_feat, size=h.shape[-2:], mode="bilinear", align_corners=False)
        return h + self.proj[site_idx](brush_feat)
