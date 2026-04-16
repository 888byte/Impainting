# -*- coding: utf-8 -*-
"""BrushNet wrapper for the active StrDiffusion texture generator.

This wrapper keeps BrushNet responsible for multi-scale conditioning and adds
an optional MGLC-Tex feature enhancement path. The texture core refines
features after BrushNet fusion and can be inserted at bottleneck and decoder
stages. Optional ``restore_S_guidance`` restores the legacy SPADE guidance as a
baseline-compatibility feature.
"""

import functools
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.DenoisingUNet_arch import SPADEBlock
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
        brushnet_prior_dropout_prob: float = 0.0,
        brushnet_feature_scale: float = 0.10,
        brushnet_use_spatial_gate: bool = True,
        texture_core_opt: Optional[dict] = None,
        main_guidance_opt: Optional[dict] = None,
        restore_S_guidance: bool = False,
    ) -> None:
        super().__init__()

        self.depth = depth
        self.nf = nf
        self.brushnet_enabled = brushnet_enabled
        self.restore_S_guidance = restore_S_guidance
        self.brushnet_prior_dropout_prob = float(brushnet_prior_dropout_prob)
        self.brushnet_feature_scale = float(brushnet_feature_scale)
        self.brushnet_use_spatial_gate = bool(brushnet_use_spatial_gate)
        if not 0.0 <= self.brushnet_prior_dropout_prob <= 1.0:
            raise ValueError("brushnet_prior_dropout_prob must be in [0, 1]")

        texture_core_opt = texture_core_opt or {}
        main_guidance_opt = main_guidance_opt or {}
        insert_mid = texture_core_opt.get("insert_mid", True)
        insert_dec = texture_core_opt.get("insert_dec", False)
        texture_core_enabled = bool(texture_core_opt.get("enabled", False))
        if texture_core_enabled and not (insert_mid or insert_dec):
            raise ValueError(
                "texture_core.enabled is true, but no insertion position is enabled"
            )

        self.texture_core_enabled = texture_core_enabled
        self.texture_core_insert_mid = bool(texture_core_enabled and insert_mid)
        self.texture_core_insert_dec = bool(texture_core_enabled and insert_dec)

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
        self.main_guidance_proj = None
        self.main_guidance_use_observed = bool(
            main_guidance_opt.get("use_observed_input", True)
        )
        self.main_guidance_use_mask = bool(main_guidance_opt.get("use_mask", True))
        self.main_guidance_enabled = bool(main_guidance_opt.get("enabled", False))
        if self.main_guidance_enabled:
            guidance_in_nc = 0
            if self.main_guidance_use_observed:
                guidance_in_nc += in_nc
            if self.main_guidance_use_mask:
                guidance_in_nc += 1
            if guidance_in_nc <= 0:
                raise ValueError(
                    "main_guidance.enabled is true, but no guidance input is enabled"
                )
            self.main_guidance_proj = default_conv(guidance_in_nc, nf, 3)
            if main_guidance_opt.get("zero_init", True):
                nn.init.zeros_(self.main_guidance_proj.weight)
                if self.main_guidance_proj.bias is not None:
                    nn.init.zeros_(self.main_guidance_proj.bias)

        self.downs = nn.ModuleList([])
        for i in range(depth):
            dim_in = nf * int(math.pow(2, i))
            dim_out = nf * int(math.pow(2, i + 1))
            down_block = [
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if i != (depth - 1) else default_conv(dim_in, dim_out),
            ]
            if self.restore_S_guidance:
                # Keep SPADEBlock as downs[i][4] to match the pretrained weight
                # key layout (downs.0.4.*, downs.1.4.*, ...) from the original
                # ConditionalUNet checkpoint.  This allows direct weight reuse
                # without any key remapping.
                down_block.append(SPADEBlock(dim_out, dim_out, 1))
            self.downs.append(nn.ModuleList(down_block))

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

        block_kwargs = {
            "backend": texture_core_opt.get("backend", "conv_surrogate"),
            "branch_mode": texture_core_opt.get("branch_mode", "both"),
            "use_mask_gate": texture_core_opt.get("use_mask_gate", True),
            "gate_hidden": texture_core_opt.get("gate_hidden", 16),
            "boundary_width": texture_core_opt.get("boundary_width", 3),
            "zero_init_last": texture_core_opt.get("zero_init_last", True),
        }

        self.mglc_mid = None
        if self.texture_core_insert_mid:
            self.mglc_mid = MGLCBlock(channels=mid_dim, **block_kwargs)

        self.mglc_dec = None
        if self.texture_core_insert_dec:
            self.mglc_dec = MGLCBlock(channels=nf, **block_kwargs)

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

    def _apply_brushnet_feature_gate(
        self,
        feature: torch.Tensor,
        mask: Optional[torch.Tensor],
        confidence: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Keep BrushNet as a weak hole-only reference branch.

        BrushNet features are auxiliary guidance. They must not overwrite the
        original StrDiffusion repair path globally.  We therefore inject them
        only inside the hole mask and scale them by confidence plus a small
        global feature scale.  This preserves the prior-free main trunk on known
        pixels and prevents an imperfect color prior from dominating the score.
        """
        gated = feature * self.brushnet_feature_scale
        if not self.brushnet_use_spatial_gate or mask is None:
            return gated

        gate = mask
        if confidence is not None:
            gate = gate * confidence.clamp(0.0, 1.0)
        if gate.shape[-2:] != feature.shape[-2:]:
            gate = F.interpolate(gate, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        return gated * gate.clamp(0.0, 1.0)

    def forward(
        self,
        xt: torch.Tensor,
        cond: torch.Tensor,
        time: Union[int, float, torch.Tensor],
        S: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        color_prior: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
        observed_degraded: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the active generator.

        Args:
            xt: [B, 3, H, W] noisy state.
            cond: [B, 3, H, W] diffusion condition.
            time: scalar or [B] timestep tensor.
            S: optional structure guide used only when ``restore_S_guidance`` is on.
            mask: [B, 1, H, W] repair-region mask in BrushNet semantics.
            color_prior: [B, 3, H, W] color prior.
            confidence: [B, 1, H, W] confidence map.
            observed_degraded: [B, 3, H, W] observed damaged input fed to the
                main trunk so the trunk does not rely on color_prior alone.

        Returns:
            Tuple[Tensor, Tensor], both [B, 3, H, W].
        """
        if isinstance(time, (int, float)):
            time = torch.tensor([time], device=xt.device)

        x = xt - cond
        x = torch.cat([x, cond], dim=1)  # [B, 6, H, W]

        H, W = x.shape[2:]
        x = self.check_image_size(x, H, W)
        t = self.time_mlp(time)
        mask_padded = self.check_image_size(mask, H, W) if mask is not None else None
        observed_padded = (
            self.check_image_size(observed_degraded, H, W)
            if observed_degraded is not None
            else None
        )

        brushnet_features = None
        brushnet_mid = None
        if (
            self.brushnet_enabled
            and self.brushnet is not None
            and mask is not None
            and color_prior is not None
            and confidence is not None
        ):
            color_prior = self.check_image_size(color_prior, H, W)
            confidence = self.check_image_size(confidence, H, W)
            # During training, randomly skip BrushNet for individual samples.
            # This forces the main trunk to learn to repair without color guidance,
            # preventing over-reliance on the prior and improving robustness.
            # Per-sample dropout: each sample in the batch independently decides
            # whether to run BrushNet, keeping gradient variance low.
            run_brushnet = True
            if self.training and self.brushnet_prior_dropout_prob > 0.0:
                batch = color_prior.shape[0]
                # [B, 1, 1, 1] boolean mask: True = run BrushNet for this sample
                keep = (
                    torch.rand(batch, 1, 1, 1, device=color_prior.device)
                    >= self.brushnet_prior_dropout_prob
                )
                if not keep.any():
                    run_brushnet = False
                elif not keep.all():
                    # Zero out dropped samples' prior so BrushNet sees no signal
                    # for them. The feature gate will suppress injection anyway,
                    # but zeroing the input is cleaner.
                    keep_f = keep.float()
                    color_prior = color_prior * keep_f
                    confidence = confidence * keep_f

            if run_brushnet:
                bn_output = self.brushnet(xt, mask_padded, color_prior, confidence, time)
                brushnet_features = bn_output["down_features"]
                brushnet_mid = bn_output["mid_feature"]

        x = self.init_conv(x)
        if self.main_guidance_proj is not None:
            guidance_inputs = []
            if self.main_guidance_use_observed and observed_padded is not None:
                guidance_inputs.append(observed_padded)
            if self.main_guidance_use_mask and mask_padded is not None:
                guidance_inputs.append(mask_padded)
            if guidance_inputs:
                x = x + self.main_guidance_proj(torch.cat(guidance_inputs, dim=1))
        x_res = x.clone()

        skips = []
        brushnet_idx = 0
        for idx, blocks in enumerate(self.downs):
            b1, b2, attn, downsample = blocks

            x = b1(x, t)
            if brushnet_features is not None and brushnet_idx < len(brushnet_features):
                x = x + self._apply_brushnet_feature_gate(
                    brushnet_features[brushnet_idx], mask_padded, confidence
                )
                brushnet_idx += 1
            skips.append(x)

            x = b2(x, t)
            x = attn(x)
            if brushnet_features is not None and brushnet_idx < len(brushnet_features):
                x = x + self._apply_brushnet_feature_gate(
                    brushnet_features[brushnet_idx], mask_padded, confidence
                )
                brushnet_idx += 1
            skips.append(x)

            x = downsample(x)

            # Baseline compatibility fix only; not counted as the texture-core
            # innovation. This restores the legacy S guidance when requested.
            if self.restore_S_guidance and S is not None and len(blocks) > 4:
                x = blocks[4](x, S)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        if brushnet_mid is not None:
            x = x + self._apply_brushnet_feature_gate(
                brushnet_mid, mask_padded, confidence
            )

        # MGLC-Tex does not duplicate BrushNet conditioning. It only refines
        # features after BrushNet fusion.
        if self.mglc_mid is not None:
            x = self.mglc_mid(x, mask_padded)

        for blocks in self.ups:
            b1, b2, attn, upsample = blocks

            x = torch.cat([x, skips.pop()], dim=1)
            x = b1(x, t)

            x = torch.cat([x, skips.pop()], dim=1)
            x = b2(x, t)
            x = attn(x)
            x = upsample(x)

        if self.mglc_dec is not None:
            x = self.mglc_dec(x, mask_padded)

        x = torch.cat([x, x_res], dim=1)
        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        x = x[..., :H, :W]
        return x, x

    def _apply_prior_dropout(
        self,
        color_prior: torch.Tensor,
        confidence: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Drop hole-region prior content on random samples during training.

        This keeps color_prior as a color guide instead of the only content
        provider for the repair region.
        """
        if mask is None:
            return color_prior, confidence

        batch = color_prior.shape[0]
        drop_flag = (
            torch.rand(batch, 1, 1, 1, device=color_prior.device)
            < self.brushnet_prior_dropout_prob
        ).float()
        keep_mask = 1.0 - drop_flag * mask
        return color_prior * keep_mask, confidence * keep_mask


def create_brushnet_unet(opt: dict) -> nn.Module:
    """Factory helper for the active BrushNet generator."""
    network_opt = opt.get("network_G", {}).get("setting", {})
    brushnet_opt = opt.get("brushnet", {})
    texture_core_opt = opt.get("texture_core", {})
    main_guidance_opt = opt.get("main_guidance", {})
    restore_S_guidance = opt.get("restore_S_guidance", False)

    return ConditionalUNetWithBrushNet(
        in_nc=network_opt.get("in_nc", 3),
        out_nc=network_opt.get("out_nc", 3),
        nf=network_opt.get("nf", 64),
        depth=network_opt.get("depth", 4),
        brushnet_in_nc=brushnet_opt.get("in_nc", 8),
        brushnet_enabled=brushnet_opt.get("enabled", True),
        brushnet_lite=brushnet_opt.get("lite", False),
        brushnet_prior_dropout_prob=brushnet_opt.get("prior_dropout_prob", 0.0),
        brushnet_feature_scale=brushnet_opt.get("feature_scale", 0.10),
        brushnet_use_spatial_gate=brushnet_opt.get("use_spatial_gate", True),
        texture_core_opt=texture_core_opt,
        main_guidance_opt=main_guidance_opt,
        restore_S_guidance=restore_S_guidance,
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
        texture_core_opt={
            "enabled": True,
            "insert_mid": True,
            "insert_dec": True,
            "backend": "sem_lite",
            "branch_mode": "both",
            "use_mask_gate": True,
        },
        restore_S_guidance=True,
    ).to(device)

    batch_size = 2
    height, width = 256, 256
    xt = torch.randn(batch_size, 3, height, width, device=device)
    cond = torch.randn(batch_size, 3, height, width, device=device)
    time = torch.randint(0, 1000, (batch_size,), device=device).float()
    mask = torch.randint(0, 2, (batch_size, 1, height, width), device=device).float()
    color_prior = torch.randn(batch_size, 3, height, width, device=device)
    confidence = torch.rand(batch_size, 1, height, width, device=device)
    structure = torch.rand(batch_size, 1, height, width, device=device)

    with torch.no_grad():
        output, _ = model(
            xt,
            cond,
            time,
            S=structure,
            mask=mask,
            color_prior=color_prior,
            confidence=confidence,
        )

    print(f"Input: xt={xt.shape}, cond={cond.shape}")
    print(f"Output: {output.shape}")
