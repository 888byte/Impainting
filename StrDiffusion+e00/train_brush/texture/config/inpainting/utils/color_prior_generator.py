# -*- coding: utf-8 -*-
"""
第一阶段：颜色先验图（Color_Prior）与置信度图（Confidence_Map）生成

流程：
1) 对 mask_keep=1 的像素执行 LUT（三线性）得到 mapped_rgb + conf_lut
2) mask_hole 区域用传统 inpaint 填充（默认 cv2.inpaint + 大孔洞多尺度策略）
3) 置信度融合：Conf_final = alpha * Conf_LUT + beta * Conf_Inpaint
   其中 Conf_Inpaint 在 hole 区域显著低于 known，并随距离边界深入而衰减
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import cv2
import torch

from .lut_3d import PigmentLUT3D


@dataclass
class Stage1Config:
    inpaint_backend: str = "auto"       # auto|cv2|patchmatch|xphoto
    inpaint_method: str = "telea"       # telea|ns
    inpaint_radius: int = 3
    large_mask_ratio: float = 0.30
    large_mask_downsample: float = 0.5
    alpha: float = 0.7
    beta: float = 0.3
    inpaint_base_conf: float = 0.15
    inpaint_decay_strength: float = 1.0


class ColorPriorGenerator:
    def __init__(self, lut_path: str, cfg: Optional[Stage1Config] = None) -> None:
        self.lut = PigmentLUT3D.from_npz(lut_path)
        self.cfg = cfg or Stage1Config()

        # patchmatch-cython（可选）
        self._pm_available = False
        try:
            import patchmatch  # patchmatch-cython: patchmatch.inpaint_pyramid(image, mask)
            self._patchmatch = patchmatch
            self._pm_available = True
        except Exception:
            self._patchmatch = None
            self._pm_available = False

        # opencv-contrib xphoto（可选）
        self._xphoto_available = hasattr(cv2, "xphoto") and hasattr(cv2.xphoto, "inpaint")

    @torch.no_grad()
    def generate_torch(self, img_rgb: torch.Tensor, mask_keep: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        img_rgb:   (B,3,H,W) float [0,1] (RGB)
        mask_keep: (B,1,H,W) float {0,1}  1=已知/保留 0=缺失
        """
        device = img_rgb.device
        B, _, H, W = img_rgb.shape
        priors, confs = [], []

        for b in range(B):
            im = img_rgb[b].detach().float().cpu().clamp(0, 1)
            mk = (mask_keep[b, 0].detach().float().cpu().numpy() > 0.5).astype(np.float32)

            img_u8 = (im.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
            prior_u8, conf_f = self.generate_np(img_u8, mk)

            priors.append(torch.from_numpy(prior_u8.astype(np.float32) / 255.0).permute(2, 0, 1))
            confs.append(torch.from_numpy(conf_f.astype(np.float32))[None, ...])

        color_prior = torch.stack(priors, 0).to(device)
        conf_map = torch.stack(confs, 0).to(device)
        return color_prior, conf_map

    def generate_np(self, img_rgb_u8: np.ndarray, mask_keep_01: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        H, W, _ = img_rgb_u8.shape
        mask_keep = (mask_keep_01 > 0.5).astype(np.float32)
        mask_hole = 1.0 - mask_keep

        mapped_u8, conf_lut, _ = self.lut.apply_rgb(img_rgb_u8, return_conf=True, out_uint8=True)
        if conf_lut is None:
            conf_lut = np.ones((H, W), np.float32)
        conf_lut = conf_lut.astype(np.float32) * mask_keep

        hole_mask_u8 = (mask_hole * 255.0).astype(np.uint8)
        mapped_known = (mapped_u8.astype(np.float32) * mask_keep[..., None]).astype(np.uint8)

        inpaint_u8 = self._inpaint_rgb(mapped_known, hole_mask_u8, mask_keep)

        color_prior = (
            mapped_u8.astype(np.float32) * mask_keep[..., None]
            + inpaint_u8.astype(np.float32) * mask_hole[..., None]
        ).round().astype(np.uint8)

        conf_inpaint = self._build_inpaint_conf(mask_keep, hole_mask_u8)

        cfg = self.cfg
        conf_final = np.clip(cfg.alpha * conf_lut + cfg.beta * conf_inpaint, 0.0, 1.0).astype(np.float32)
        return color_prior, conf_final

    # ---------- Inpaint 后端 ----------
    def _inpaint_rgb(self, rgb_known_u8: np.ndarray, hole_mask_u8: np.ndarray, mask_keep: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        hole_ratio = float((hole_mask_u8 > 0).mean())

        backend = cfg.inpaint_backend.lower()
        if backend == "auto":
            if self._pm_available and hole_ratio > 0.10:
                backend = "patchmatch"
            elif self._xphoto_available and hole_ratio > 0.30:
                backend = "xphoto"
            else:
                backend = "cv2"

        if backend == "patchmatch" and self._pm_available:
            return self._inpaint_patchmatch(rgb_known_u8, hole_mask_u8)
        if backend == "xphoto" and self._xphoto_available:
            return self._inpaint_xphoto(rgb_known_u8, mask_keep)
        return self._inpaint_cv2_multiscale(rgb_known_u8, hole_mask_u8)

    def _inpaint_cv2_multiscale(self, rgb_known_u8: np.ndarray, hole_mask_u8: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        method = cv2.INPAINT_TELEA if cfg.inpaint_method.lower() == "telea" else cv2.INPAINT_NS
        hole_ratio = float((hole_mask_u8 > 0).mean())

        if hole_ratio <= cfg.large_mask_ratio:
            return cv2.inpaint(rgb_known_u8, hole_mask_u8, cfg.inpaint_radius, method)

        H, W = hole_mask_u8.shape
        ds = float(np.clip(cfg.large_mask_downsample, 0.1, 1.0))
        nh, nw = max(16, int(H * ds)), max(16, int(W * ds))

        small_img = cv2.resize(rgb_known_u8, (nw, nh), interpolation=cv2.INTER_AREA)
        small_msk = cv2.resize(hole_mask_u8, (nw, nh), interpolation=cv2.INTER_NEAREST)
        small_out = cv2.inpaint(small_img, small_msk, cfg.inpaint_radius, method)
        up = cv2.resize(small_out, (W, H), interpolation=cv2.INTER_LINEAR)
        refine = cv2.inpaint(up, hole_mask_u8, max(1, cfg.inpaint_radius // 2), method)
        return refine

    def _inpaint_patchmatch(self, rgb_known_u8: np.ndarray, hole_mask_u8: np.ndarray) -> np.ndarray:
        # patchmatch-cython：mask=255 表示 known，0 表示 hole【:contentReference[oaicite:3]{index=3}】
        keep_mask = (hole_mask_u8 == 0).astype(np.uint8) * 255
        img_f = rgb_known_u8.astype(np.float32) / 255.0
        out = self._patchmatch.inpaint_pyramid(img_f, keep_mask)  # 返回 float [0,1]
        return np.clip(out * 255.0, 0, 255).round().astype(np.uint8)

    def _inpaint_xphoto(self, rgb_known_u8: np.ndarray, mask_keep: np.ndarray) -> np.ndarray:
        # xphoto.inpaint：mask 非 0 表示有效区域，0 表示需要重建区域【:contentReference[oaicite:4]{index=4}】
        keep = (mask_keep > 0.5).astype(np.uint8) * 255
        return cv2.xphoto.inpaint(rgb_known_u8, keep, cv2.xphoto.INPAINT_FSR_FAST)

    # ---------- 置信度 ----------
    def _build_inpaint_conf(self, mask_keep: np.ndarray, hole_mask_u8: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        hole = (hole_mask_u8 > 0).astype(np.uint8)
        dist = cv2.distanceTransform(hole, cv2.DIST_L2, 3).astype(np.float32)
        maxd = float(dist.max()) + 1e-6
        dist_norm = dist / maxd
        decay = np.exp(-cfg.inpaint_decay_strength * dist_norm).astype(np.float32)

        hole_conf = np.clip(cfg.inpaint_base_conf * decay, 0.0, 1.0)
        conf = mask_keep.astype(np.float32) * 1.0 + (1.0 - mask_keep.astype(np.float32)) * hole_conf
        return np.clip(conf, 0.0, 1.0).astype(np.float32)

    # Debug：全图 LUT 映射（Transformed_GT 用）
    def apply_lut_full_np(self, img_rgb_u8: np.ndarray) -> np.ndarray:
        mapped, _, _ = self.lut.apply_rgb(img_rgb_u8, return_conf=False, out_uint8=True)
        return mapped
