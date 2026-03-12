# -*- coding: utf-8 -*-
"""BrushNet wrapper for the active StrDiffusion texture generator.

This wrapper keeps BrushNet responsible for multi-scale conditioning and adds
an optional bottleneck-only MGLC-Tex block. The MGLC block does not create a
second conditioning path; it only refines bottleneck features after BrushNet
mid-feature fusion.
"""

import functools
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.mglc_block import MGLCBlock
from .modules.module_util import (
    LinearAttention,
    NonLinearity,
    PreNorm,
    ResBlock,
    Residual,
    SinusoidalPosEmb,
    Upsample,
    Downsample,
    default_conv,
)
from .pixel_brushnet import PixelBrushNet, PixelBrushNetLite


class ConditionalUNetWithBrushNet(nn.Module):
    """Active generator used by the BrushNet training config."""

    def __init__(
        self,
        in_nc: int = 3,
        out_nc: int = 3,
        nf: int = 64,
        depth: int = 4,
        brushnet_in_nc: int = 8,
        brushnet_enabled: bool = True,
        brushnet_lite: bool = False,
        texture_core_opt: Optional[dict] = None,
    ) -> None:
        super().__init__()

        self.depth = depth
        self.nf = nf
        self.brushnet_enabled = brushnet_enabled

        texture_core_opt = texture_core_opt or {}
        self.texture_core_enabled = bool(
            texture_core_opt.get("enabled", False)
            and texture_core_opt.get("insert_mid", True)
        )

        block_class = functools.partial(
            ResBlock, conv=default_conv, act=NonLinearity()
        )

        time_dim = nf * 4
        sinu_pos_emb = SinusoidalPosEmb(nf)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(nf, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_conv = default_conv(in_nc * 2, nf, 7)

        self.downs = nn.ModuleList([])
        for i in range(depth):
            dim_in = nf * int(math.pow(2, i))
            dim_out = nf * int(math.pow(2, i + 1))
            self.downs.append(
                nn.ModuleList(
                    [
                        block_class(
                            dim_in=dim_in,
                            dim_out=dim_in,
                            time_emb_dim=time_dim,
                        ),
                        block_class(
                            dim_in=dim_in,
                            dim_out=dim_in,
                            time_emb_dim=time_dim,
                        ),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if i != (depth - 1)
                        else default_conv(dim_in, dim_out),
                    ]
                )
            )

        self.ups = nn.ModuleList([])
        for i in range(depth):
            dim_in = nf * int(math.pow(2, i))
            dim_out = nf * int(math.pow(2, i + 1))
            self.ups.insert(
                0,
                nn.ModuleList(
                    [
                        block_class(
                            dim_in=dim_out + dim_in,
                            dim_out=dim_out,
                            time_emb_dim=time_dim,
                        ),
                        block_class(
                            dim_in=dim_out + dim_in,
                            dim_out=dim_out,
                            time_emb_dim=time_dim,
                        ),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if i != 0
                        else default_conv(dim_out, dim_in),
                    ]
                ),
            )

        mid_dim = nf * int(math.pow(2, depth))
        self.mid_block1 = block_class(
            dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim
        )
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = block_class(
            dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim
        )

        self.mglc_mid = None
        if self.texture_core_enabled:
            self.mglc_mid = MGLCBlock(
                channels=mid_dim,
                backend=texture_core_opt.get("backend", "conv_surrogate"),
                use_mask_gate=texture_core_opt.get("use_mask_gate", True),
                gate_hidden=texture_core_opt.get("gate_hidden", 16),
                boundary_width=texture_core_opt.get("boundary_width", 3),
                zero_init_last=texture_core_opt.get("zero_init_last", True),
            )

        self.final_res_block = block_class(
            dim_in=nf * 2, dim_out=nf, time_emb_dim=time_dim
        )
        self.final_conv = nn.Conv2d(nf, out_nc, 3, 1, 1)

        if brushnet_enabled:
            if brushnet_lite:
                self.brushnet = PixelBrushNetLite(
                    in_nc=brushnet_in_nc, nf=nf, depth=depth
                )
            else:
                self.brushnet = PixelBrushNet(
                    in_nc=brushnet_in_nc, nf=nf, depth=depth
                )
            print(
                f"[ConditionalUNetWithBrushNet] BrushNet enabled (lite={brushnet_lite})"
            )
        else:
            self.brushnet = None
            print("[ConditionalUNetWithBrushNet] BrushNet disabled")

    def check_image_size(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Pad ``x`` so height/width are divisible by the UNet scale factor."""
        scale = int(math.pow(2, self.depth))
        mod_pad_h = (scale - h % scale) % scale
        mod_pad_w = (scale - w % scale) % scale
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")

    def forward(
        self,
        xt: torch.Tensor,
        cond: torch.Tensor,
        time: Union[int, float, torch.Tensor],
        S: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        color_prior: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the active generator.

        Args:
            xt: [B, 3, H, W] noisy state.
            cond: [B, 3, H, W] diffusion condition.
            time: scalar or [B] timestep tensor.
            S: reserved structure-guide slot kept for call compatibility.
            mask: [B, 1, H, W] repair-region mask in BrushNet semantics.
            color_prior: [B, 3, H, W] color prior.
            confidence: [B, 1, H, W] confidence map.

        Returns:
            Tuple[Tensor, Tensor], both [B, 3, H, W].
        """
        del S

        if isinstance(time, (int, float)):
            time = torch.tensor([time], device=xt.device)

        x = xt - cond
        x = torch.cat([x, cond], dim=1)  # [B, 6, H, W]

        H, W = x.shape[2:]
        x = self.check_image_size(x, H, W)
        t = self.time_mlp(time)

        brushnet_features = None
        brushnet_mid = None
        if (
            self.brushnet_enabled
            and self.brushnet is not None
            and mask is not None
            and color_prior is not None
            and confidence is not None
        ):
            mask = self.check_image_size(mask, H, W)
            color_prior = self.check_image_size(color_prior, H, W)
            confidence = self.check_image_size(confidence, H, W)

            bn_output = self.brushnet(xt, mask, color_prior, confidence, time)
            brushnet_features = bn_output["down_features"]
            brushnet_mid = bn_output["mid_feature"]

        x = self.init_conv(x)
        x_res = x.clone()

        skips = []
        brushnet_idx = 0
        for blocks in self.downs:
            b1, b2, attn, downsample = blocks

            x = b1(x, t)
            if brushnet_features is not None and brushnet_idx < len(brushnet_features):
                x = x + brushnet_features[brushnet_idx]
                brushnet_idx += 1
            skips.append(x)

            x = b2(x, t)
            x = attn(x)
            if brushnet_features is not None and brushnet_idx < len(brushnet_features):
                x = x + brushnet_features[brushnet_idx]
                brushnet_idx += 1
            skips.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        if brushnet_mid is not None:
            x = x + brushnet_mid

        # MGLC-Tex does not duplicate BrushNet conditioning. It only refines
        # bottleneck features after BrushNet fusion.
        if self.texture_core_enabled and self.mglc_mid is not None:
            x = self.mglc_mid(x, mask)

        for blocks in self.ups:
            b1, b2, attn, upsample = blocks

            x = torch.cat([x, skips.pop()], dim=1)
            x = b1(x, t)

            x = torch.cat([x, skips.pop()], dim=1)
            x = b2(x, t)
            x = attn(x)
            x = upsample(x)

        x = torch.cat([x, x_res], dim=1)
        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        x = x[..., :H, :W]
        return x, x


def create_brushnet_unet(opt: dict) -> nn.Module:
    """Factory helper for the active BrushNet generator."""
    network_opt = opt.get("network_G", {}).get("setting", {})
    brushnet_opt = opt.get("brushnet", {})
    texture_core_opt = opt.get("texture_core", {})

    return ConditionalUNetWithBrushNet(
        in_nc=network_opt.get("in_nc", 3),
        out_nc=network_opt.get("out_nc", 3),
        nf=network_opt.get("nf", 64),
        depth=network_opt.get("depth", 4),
        brushnet_in_nc=brushnet_opt.get("in_nc", 8),
        brushnet_enabled=brushnet_opt.get("enabled", True),
        brushnet_lite=brushnet_opt.get("lite", False),
        texture_core_opt=texture_core_opt,
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConditionalUNetWithBrushNet(
        in_nc=3,
        out_nc=3,
        nf=64,
        depth=4,
        brushnet_in_nc=8,
        brushnet_enabled=True,
        texture_core_opt={"enabled": True, "backend": "conv_surrogate"},
    ).to(device)

    batch_size = 2
    height, width = 256, 256
    xt = torch.randn(batch_size, 3, height, width, device=device)
    cond = torch.randn(batch_size, 3, height, width, device=device)
    time = torch.randint(0, 1000, (batch_size,), device=device).float()
    mask = torch.randint(0, 2, (batch_size, 1, height, width), device=device).float()
    color_prior = torch.randn(batch_size, 3, height, width, device=device)
    confidence = torch.rand(batch_size, 1, height, width, device=device)

    with torch.no_grad():
        output, _ = model(
            xt, cond, time, mask=mask, color_prior=color_prior, confidence=confidence
        )

    print(f"Input: xt={xt.shape}, cond={cond.shape}")
    print(f"Output: {output.shape}")
