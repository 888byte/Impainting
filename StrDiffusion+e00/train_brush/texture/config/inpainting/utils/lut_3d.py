# -*- coding: utf-8 -*-
"""
3D 颜色 LUT 读取与三线性插值（Trilinear Interpolation）

本模块用于读取你提供的 pigment_lut33.npz，并对输入 RGB 像素执行 3D LUT 映射。
严禁最近邻（Nearest Neighbor），否则会出现严重色带/断层（Posterization）。

LUT 文件结构（来自你的颜色推理模型输出）：
- grid:      (G,)   0~255 的等距采样点（G=33）
- lut_rgb:   (G,G,G,3)  目标 RGB（通常是“复原后颜色”）
- lut_conf:  (G,G,G)    置信度（0~1）
- lut_std:   (G,G,G) 或 (G,G,G,3)  (可选) 颜色不确定性/方差等
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


def _to_255_float(img: np.ndarray) -> np.ndarray:
    """把输入图像统一转换到 float32 的 [0, 255] 取值域。"""
    if img.dtype == np.uint8:
        return img.astype(np.float32)
    img_f = img.astype(np.float32)
    if img_f.max() <= 1.5:  # 看起来是[0,1]
        img_f = img_f * 255.0
    return img_f


@dataclass
class PigmentLUT3D:
    """3D LUT + 三线性插值采样器。"""
    lut_rgb: np.ndarray
    lut_conf: Optional[np.ndarray] = None
    lut_std: Optional[np.ndarray] = None
    grid: Optional[np.ndarray] = None

    @classmethod
    def from_npz(cls, npz_path: str) -> "PigmentLUT3D":
        data = np.load(npz_path)
        lut_rgb = data["lut_rgb"].astype(np.float32)
        lut_conf = data["lut_conf"].astype(np.float32) if "lut_conf" in data.files else None
        lut_std = data["lut_std"].astype(np.float32) if "lut_std" in data.files else None
        grid = data["grid"].astype(np.float32) if "grid" in data.files else None
        return cls(lut_rgb=lut_rgb, lut_conf=lut_conf, lut_std=lut_std, grid=grid)

    @property
    def grid_size(self) -> int:
        return int(self.lut_rgb.shape[0])

    def _trilinear(self, lut: np.ndarray, rgb_255: np.ndarray) -> np.ndarray:
        """
        对任意 LUT（shape=(G,G,G,*)）进行三线性插值采样。
        约定 LUT 下标顺序为 (R, G, B)。
        输入 rgb_255: (H,W,3) float32 in [0,255]
        返回: (H,W,*) float32
        """
        G = self.grid_size
        scale = (G - 1) / 255.0

        pos = np.clip(rgb_255 * scale, 0.0, float(G - 1))     # (H,W,3)
        i0 = np.floor(pos).astype(np.int32)                   # (H,W,3)
        i1 = np.clip(i0 + 1, 0, G - 1).astype(np.int32)       # (H,W,3)
        w = (pos - i0.astype(np.float32)).astype(np.float32)  # (H,W,3)

        H, W, _ = rgb_255.shape
        N = H * W

        r0 = i0[..., 0].reshape(N)
        g0 = i0[..., 1].reshape(N)
        b0 = i0[..., 2].reshape(N)
        r1 = i1[..., 0].reshape(N)
        g1 = i1[..., 1].reshape(N)
        b1 = i1[..., 2].reshape(N)

        wr = w[..., 0].reshape(N, 1)
        wg = w[..., 1].reshape(N, 1)
        wb = w[..., 2].reshape(N, 1)

        c000 = lut[r0, g0, b0]
        c001 = lut[r0, g0, b1]
        c010 = lut[r0, g1, b0]
        c011 = lut[r0, g1, b1]
        c100 = lut[r1, g0, b0]
        c101 = lut[r1, g0, b1]
        c110 = lut[r1, g1, b0]
        c111 = lut[r1, g1, b1]

        c00 = c000 * (1 - wb) + c001 * wb
        c01 = c010 * (1 - wb) + c011 * wb
        c10 = c100 * (1 - wb) + c101 * wb
        c11 = c110 * (1 - wb) + c111 * wb

        c0 = c00 * (1 - wg) + c01 * wg
        c1 = c10 * (1 - wg) + c11 * wg

        c = c0 * (1 - wr) + c1 * wr

        out_shape = (H, W) + c.shape[1:]
        return c.reshape(out_shape).astype(np.float32)

    def apply_rgb(
        self,
        img_rgb: np.ndarray,
        return_conf: bool = True,
        return_std: bool = False,
        out_uint8: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """对输入 RGB 图像做 LUT 映射（Trilinear）。"""
        rgb_255 = np.clip(_to_255_float(img_rgb), 0.0, 255.0)
        out_rgb = np.clip(self._trilinear(self.lut_rgb, rgb_255), 0.0, 255.0)

        conf = None
        if return_conf and self.lut_conf is not None:
            conf = self._trilinear(self.lut_conf, rgb_255)
            if conf.ndim == 3 and conf.shape[-1] == 1:
                conf = conf[..., 0]
            conf = np.clip(conf.astype(np.float32), 0.0, 1.0)

        std = None
        if return_std and self.lut_std is not None:
            std = self._trilinear(self.lut_std, rgb_255).astype(np.float32)

        if out_uint8:
            return np.round(out_rgb).astype(np.uint8), conf, std
        return out_rgb.astype(np.float32), conf, std
