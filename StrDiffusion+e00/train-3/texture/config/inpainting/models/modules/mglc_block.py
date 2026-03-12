"""MGLC-Tex MVP block for bottleneck-only texture enhancement.

This module does not create a second conditioning branch. BrushNet still owns
multi-scale conditioning. MGLCBlock only refines bottleneck features that have
already fused BrushNet hints.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_group_count(num_channels: int, max_groups: int = 32) -> int:
    """Return a GroupNorm group count that divides ``num_channels``."""
    num_groups = min(max_groups, num_channels)
    while num_groups > 1 and num_channels % num_groups != 0:
        num_groups -= 1
    return num_groups


def _zero_init_conv(conv: nn.Conv2d) -> None:
    """Zero-initialize the given convolution layer in-place."""
    nn.init.zeros_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


class _ConvSurrogateBranch(nn.Module):
    """A local or context texture branch.

    Input:
        feat: [B, C, h, w]
    Output:
        feat_out: [B, C, h, w]
    """

    def __init__(self, channels: int, branch_type: str, zero_init_last: bool) -> None:
        super().__init__()
        if branch_type == "local":
            layers = [
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
                nn.GELU(),
                nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels),
                nn.Conv2d(channels, channels, kernel_size=1),
            ]
        elif branch_type == "context":
            layers = [
                nn.Conv2d(channels, channels, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=(1, 7),
                    padding=(0, 3),
                    groups=channels,
                ),
                nn.GELU(),
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=(7, 1),
                    padding=(3, 0),
                    groups=channels,
                ),
                nn.GELU(),
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=3,
                    padding=3,
                    dilation=3,
                    groups=channels,
                ),
                nn.Conv2d(channels, channels, kernel_size=1),
            ]
        else:
            raise ValueError(f"Unsupported branch_type: {branch_type}")

        self.net = nn.Sequential(*layers)
        if zero_init_last:
            _zero_init_conv(self.net[-1])

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Apply the branch to ``feat`` shaped [B, C, h, w]."""
        return self.net(feat)


class MaskGate(nn.Module):
    """Predict two spatial gates from mask and boundary cues.

    Inputs:
        mask: [B, 1, h, w]
        boundary_band: [B, 1, h, w]
    Outputs:
        g_local: [B, 1, h, w]
        g_ctx: [B, 1, h, w]
    """

    def __init__(self, hidden_channels: int = 16) -> None:
        super().__init__()
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")

        self.net = nn.Sequential(
            nn.Conv2d(2, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 2, kernel_size=1),
        )

    def forward(
        self, mask: torch.Tensor, boundary_band: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return local/context gates for tensors shaped [B, 1, h, w]."""
        gate_logits = self.net(torch.cat([mask, boundary_band], dim=1))
        gate_weights = torch.softmax(gate_logits, dim=1)
        return gate_weights[:, 0:1], gate_weights[:, 1:2]


class MGLCBlock(nn.Module):
    """Mask-Gated Locality-Continuity texture block for bottleneck features.

    BrushNet remains responsible for multi-scale conditioning. This block only
    enhances bottleneck features after BrushNet fusion.

    Inputs:
        feat: [B, C, h, w]
        mask: [B, 1, H, W] with BrushNet semantics, where 1 means repair region
    Output:
        feat_out: [B, C, h, w]
    """

    def __init__(
        self,
        channels: int,
        backend: str = "conv_surrogate",
        use_mask_gate: bool = True,
        gate_hidden: int = 16,
        boundary_width: int = 3,
        zero_init_last: bool = True,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if boundary_width < 0:
            raise ValueError("boundary_width must be non-negative")
        if backend != "conv_surrogate":
            raise NotImplementedError(
                f"Unsupported texture_core backend: {backend}"
            )

        self.backend = backend
        self.use_mask_gate = use_mask_gate
        self.boundary_width = boundary_width

        self.norm = nn.GroupNorm(_resolve_group_count(channels), channels)
        self.local_branch = _ConvSurrogateBranch(
            channels=channels,
            branch_type="local",
            zero_init_last=zero_init_last,
        )
        self.context_branch = _ConvSurrogateBranch(
            channels=channels,
            branch_type="context",
            zero_init_last=zero_init_last,
        )
        self.gate = MaskGate(gate_hidden) if use_mask_gate else None

    def _resize_mask(
        self, mask: torch.Tensor, spatial_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Resize ``mask`` to [B, 1, h, w] using nearest-neighbor sampling."""
        return F.interpolate(mask.float(), size=spatial_size, mode="nearest")

    def _build_boundary_band(self, mask: torch.Tensor) -> torch.Tensor:
        """Approximate a boundary band from a resized mask.

        Input:
            mask: [B, 1, h, w]
        Output:
            boundary_band: [B, 1, h, w]
        """
        if self.boundary_width == 0:
            return torch.zeros_like(mask)

        kernel_size = 2 * self.boundary_width + 1
        dilated = F.max_pool2d(
            mask, kernel_size=kernel_size, stride=1, padding=self.boundary_width
        )
        eroded = 1.0 - F.max_pool2d(
            1.0 - mask,
            kernel_size=kernel_size,
            stride=1,
            padding=self.boundary_width,
        )
        return torch.clamp(dilated - eroded, min=0.0, max=1.0)

    def forward(
        self, feat: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Refine bottleneck features.

        Args:
            feat: [B, C, h, w]
            mask: [B, 1, H, W] or ``None``

        Returns:
            Tensor shaped [B, C, h, w]
        """
        if feat.dim() != 4:
            raise ValueError(f"feat must be 4D, got shape {tuple(feat.shape)}")
        if mask is None:
            return feat
        if mask.dim() != 4 or mask.shape[1] != 1:
            raise ValueError(f"mask must be [B, 1, H, W], got {tuple(mask.shape)}")

        mask_resized = self._resize_mask(mask, feat.shape[-2:])
        feat_norm = self.norm(feat)
        feat_local = self.local_branch(feat_norm)
        feat_ctx = self.context_branch(feat_norm)

        if not self.use_mask_gate:
            return feat + feat_local + feat_ctx

        boundary_band = self._build_boundary_band(mask_resized)
        gate_local, gate_ctx = self.gate(mask_resized, boundary_band)
        return feat + gate_local * feat_local + gate_ctx * feat_ctx
