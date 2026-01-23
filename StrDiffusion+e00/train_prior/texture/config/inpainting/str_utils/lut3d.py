import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class LutMeta:
    size: int
    step: float


class Lut3D(nn.Module):
    """
    3D LUT (trilinear) implemented in PyTorch.

    npz must contain:
      - grid: (N,) float32 in [0,255]
      - lut_rgb: (N,N,N,3) uint8
      - lut_conf: (N,N,N) float32 (optional)

    apply(image):
      image: (B,3,H,W) float in [0,1] or uint8 in [0,255]
      returns:
        recolored: (B,3,H,W) float [0,1]
        conf: (B,1,H,W) float [0,1] (ones if no lut_conf)
    """

    def __init__(self, npz_path: str, device: Optional[torch.device] = None):
        super().__init__()
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"LUT npz not found: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)
        grid = data["grid"].astype(np.float32)
        lut_rgb = data["lut_rgb"]
        lut_conf = data["lut_conf"] if "lut_conf" in data.files else None

        n = int(grid.shape[0])
        step = float((grid[-1] - grid[0]) / (n - 1))

        lut_rgb_f = torch.from_numpy(lut_rgb.astype(np.float32) / 255.0)  # (N,N,N,3)
        self.register_buffer("lut_rgb", lut_rgb_f, persistent=False)

        if lut_conf is not None:
            conf_f = torch.from_numpy(lut_conf.astype(np.float32))  # (N,N,N)
            self.register_buffer("lut_conf", conf_f, persistent=False)
        else:
            self.lut_conf = None

        self.meta = LutMeta(size=n, step=step)

        if device is not None:
            self.to(device)

    @staticmethod
    def _as_float01(x: torch.Tensor) -> torch.Tensor:
        if x.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            return x.float() / 255.0
        return x.float().clamp(0.0, 1.0)

    def _trilinear(self, lut_flat: torch.Tensor, rgb01: torch.Tensor, out_channels: int) -> torch.Tensor:
        B, _, H, W = rgb01.shape
        n = self.meta.size

        pos = rgb01 * (n - 1)
        r = pos[:, 0, :, :]
        g = pos[:, 1, :, :]
        b = pos[:, 2, :, :]

        r0 = torch.floor(r).long().clamp(0, n - 1)
        g0 = torch.floor(g).long().clamp(0, n - 1)
        b0 = torch.floor(b).long().clamp(0, n - 1)

        r1 = (r0 + 1).clamp(0, n - 1)
        g1 = (g0 + 1).clamp(0, n - 1)
        b1 = (b0 + 1).clamp(0, n - 1)

        wr = (r - r0.float()).clamp(0.0, 1.0)
        wg = (g - g0.float()).clamp(0.0, 1.0)
        wb = (b - b0.float()).clamp(0.0, 1.0)

        def lin(ri, gi, bi):
            return (ri * (n * n) + gi * n + bi).view(B, -1)

        idx000 = lin(r0, g0, b0)
        idx001 = lin(r0, g0, b1)
        idx010 = lin(r0, g1, b0)
        idx011 = lin(r0, g1, b1)
        idx100 = lin(r1, g0, b0)
        idx101 = lin(r1, g0, b1)
        idx110 = lin(r1, g1, b0)
        idx111 = lin(r1, g1, b1)

        def gather(idx):
            return lut_flat[idx]  # (B, HW, C)

        c000 = gather(idx000)
        c001 = gather(idx001)
        c010 = gather(idx010)
        c011 = gather(idx011)
        c100 = gather(idx100)
        c101 = gather(idx101)
        c110 = gather(idx110)
        c111 = gather(idx111)

        wr = wr.view(B, -1, 1)
        wg = wg.view(B, -1, 1)
        wb = wb.view(B, -1, 1)
        one = 1.0

        w000 = (one - wr) * (one - wg) * (one - wb)
        w001 = (one - wr) * (one - wg) * wb
        w010 = (one - wr) * wg * (one - wb)
        w011 = (one - wr) * wg * wb
        w100 = wr * (one - wg) * (one - wb)
        w101 = wr * (one - wg) * wb
        w110 = wr * wg * (one - wb)
        w111 = wr * wg * wb

        out = (w000 * c000 + w001 * c001 + w010 * c010 + w011 * c011 +
               w100 * c100 + w101 * c101 + w110 * c110 + w111 * c111)

        out = out.view(B, H, W, out_channels).permute(0, 3, 1, 2).contiguous()
        return out

    @torch.no_grad()
    def apply(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb = self._as_float01(image)
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"Expected (B,3,H,W), got {tuple(rgb.shape)}")

        n = self.meta.size
        lut_rgb_flat = self.lut_rgb.view(n * n * n, 3)
        out_rgb = self._trilinear(lut_rgb_flat, rgb, out_channels=3).clamp(0.0, 1.0)

        if self.lut_conf is None:
            conf = torch.ones((rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]), device=rgb.device, dtype=rgb.dtype)
        else:
            lut_conf_flat = self.lut_conf.view(n * n * n, 1)
            conf = self._trilinear(lut_conf_flat, rgb, out_channels=1).clamp(0.0, 1.0)

        return out_rgb, conf
