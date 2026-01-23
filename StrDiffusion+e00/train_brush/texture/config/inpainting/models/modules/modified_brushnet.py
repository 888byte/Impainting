# -*- coding: utf-8 -*-
"""
ModifiedBrushNet：面向 Pixel-space StrDiffusion 的 BrushNet 改造版（无 VAE、直接处理 RGB）
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .module_util import (
    ResBlock, Downsample, SinusoidalPosEmb,
    LinearAttention, PreNorm, Residual, default_conv
)
from .DenoisingUNet_arch import ConditionalUNet


class ZeroConv2d(nn.Module):
    """Zero-Conv：1x1 Conv，权重和 bias 全部初始化为 0，用于“安全注入”特征。"""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, 1, 0)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ModifiedBrushNet(nn.Module):
    """Pixel-space BrushNet Encoder（只做下采样编码，不做解码）"""
    def __init__(self, in_nc: int = 8, nf: int = 64, depth: int = 4, time_emb_dim_mult: int = 4) -> None:
        super().__init__()
        self.in_nc, self.nf, self.depth = in_nc, nf, depth

        time_dim = nf * time_emb_dim_mult
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(nf),
            nn.Linear(nf, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_conv = nn.Conv2d(in_nc, nf, 7, padding=3)

        dims = [nf * (2 ** i) for i in range(depth)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.downs.append(nn.ModuleList([
                ResBlock(default_conv, dim_in, dim_in, time_emb_dim=time_dim, groups=8),
                ResBlock(default_conv, dim_in, dim_in, time_emb_dim=time_dim, groups=8),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else default_conv(dim_in, dim_out),
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResBlock(default_conv, mid_dim, mid_dim, time_emb_dim=time_dim, groups=8)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ResBlock(default_conv, mid_dim, mid_dim, time_emb_dim=time_dim, groups=8)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        if time.dim() == 2 and time.shape[1] == 1:
            time = time[:, 0]
        t = self.time_mlp(time)

        h = self.init_conv(x)
        feats: List[torch.Tensor] = []
        for b1, b2, attn, down in self.downs:
            h = b1(h, t)
            h = b2(h, t)
            h = attn(h)
            feats.append(h)    # 对齐 UNet 每层 attn 输出
            h = down(h)

        h = self.mid_block1(h, t)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t)
        return feats, h


class ConditionalUNetWithBrush(nn.Module):
    """ConditionalUNet + BrushNet feature injection（pixel-space）"""
    def __init__(
        self,
        in_nc: int = 3, out_nc: int = 3, nf: int = 64, depth: int = 4, guide_dim: int = 64,
        brush_setting: Optional[Dict[str, Any]] = None,
        use_conf_gate: bool = True,
    ) -> None:
        super().__init__()
        self.unet = ConditionalUNet(in_nc=in_nc, out_nc=out_nc, nf=nf, depth=depth, guide_dim=guide_dim)

        bs = brush_setting or {}
        self.brush = ModifiedBrushNet(
            in_nc=int(bs.get("in_nc", 8)),
            nf=int(bs.get("nf", nf)),
            depth=int(bs.get("depth", depth)),
        )

        dims = [nf * (2 ** i) for i in range(depth)]
        self.zero_down = nn.ModuleList([ZeroConv2d(d, d) for d in dims[:-1]])
        self.zero_mid = ZeroConv2d(dims[-1], dims[-1])

        self.use_conf_gate = use_conf_gate
        self._inj_down: Optional[List[torch.Tensor]] = None
        self._inj_mid: Optional[torch.Tensor] = None
        self._conf_map: Optional[torch.Tensor] = None

        self._register_hooks()

    def _register_hooks(self) -> None:
        for i, layer in enumerate(self.unet.downs):
            layer[2].register_forward_hook(self._make_down_hook(i))  # attn
        self.unet.mid_attn.register_forward_hook(self._mid_hook)

    def _make_down_hook(self, idx: int):
        def _hook(m, inp, out):
            if self._inj_down is None or idx >= len(self._inj_down):
                return out
            inj = self._inj_down[idx]
            if inj is None:
                return out
            if inj.shape[-2:] != out.shape[-2:]:
                inj = F.interpolate(inj, out.shape[-2:], mode="bilinear", align_corners=False)
            if self.use_conf_gate and self._conf_map is not None:
                conf = self._conf_map
                if conf.shape[-2:] != out.shape[-2:]:
                    conf = F.interpolate(conf, out.shape[-2:], mode="bilinear", align_corners=False)
                inj = inj * conf
            return out + inj
        return _hook

    def _mid_hook(self, m, inp, out):
        if self._inj_mid is None:
            return out
        inj = self._inj_mid
        if inj.shape[-2:] != out.shape[-2:]:
            inj = F.interpolate(inj, out.shape[-2:], mode="bilinear", align_corners=False)
        if self.use_conf_gate and self._conf_map is not None:
            conf = self._conf_map
            if conf.shape[-2:] != out.shape[-2:]:
                conf = F.interpolate(conf, out.shape[-2:], mode="bilinear", align_corners=False)
            inj = inj * conf
        return out + inj

    def forward(
        self,
        xt: torch.Tensor,
        cond: torch.Tensor,
        time: torch.Tensor = None,
        S: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        color_prior: Optional[torch.Tensor] = None,
        conf_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # time 统一为 (B,)
        if time is None:
            time = torch.zeros((xt.shape[0],), device=xt.device)
        if not isinstance(time, torch.Tensor):
            time = torch.full((xt.shape[0],), float(time), device=xt.device)

        if (mask is not None) and (color_prior is not None) and (conf_map is not None):
            brush_in = torch.cat([xt, mask, color_prior, conf_map], dim=1)  # (B,8,H,W)
            feats, mid = self.brush(brush_in, time)

            inj_down: List[torch.Tensor] = []
            for i in range(len(self.zero_down)):
                inj_down.append(self.zero_down[i](feats[i]))
            self._inj_down = inj_down
            self._inj_mid = self.zero_mid(mid)
            self._conf_map = conf_map
        else:
            self._inj_down, self._inj_mid, self._conf_map = None, None, None

        out = self.unet(xt, cond, time=time, S=S)

        # 清理缓存
        self._inj_down, self._inj_mid, self._conf_map = None, None, None
        return out
