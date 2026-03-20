# -*- coding: utf-8 -*-
"""壁画推理数据集。

用法:
    在 ``options/test/ir-sde-brushnet.yml`` 中设置:
        datasets:
          test:
            mode: mural_inference
            dataroot_degraded: /path/to/degraded
            dataroot_mask: /path/to/mask
            dataroot_GT: /path/to/gt    # 可选

Mask 语义:
    - 输入 mask 白色/255 表示待修复区域
    - 输出 mask_hole: 1 表示待修复区域
    - 输出 mask_known: 1 表示已知区域
"""

import os
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.utils.data as data


IMG_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".JPG",
    ".JPEG",
    ".PNG",
    ".BMP",
    ".TIF",
    ".TIFF",
}


def _list_image_paths(root: Optional[str]) -> List[str]:
    if not root or not os.path.isdir(root):
        return []
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for filename in sorted(filenames):
            ext = os.path.splitext(filename)[1]
            if ext in IMG_EXTENSIONS:
                paths.append(os.path.join(dirpath, filename))
    return sorted(paths)


def _build_stem_map(paths: List[str]) -> Dict[str, str]:
    mapping = {}
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        mapping[stem] = path
    return mapping


def _load_rgb_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0


def _load_mask(path: str, target_hw) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"无法读取掩码: {path}")
    if mask.shape[:2] != target_hw:
        mask = cv2.resize(mask, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.float32)
    return mask


def _load_confidence(path: str, target_hw) -> np.ndarray:
    confidence = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if confidence is None:
        raise FileNotFoundError(f"无法读取置信度图: {path}")
    if confidence.shape[:2] != target_hw:
        confidence = cv2.resize(
            confidence,
            (target_hw[1], target_hw[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    return confidence.astype(np.float32) / 255.0


class MuralInferenceDataset(data.Dataset):
    """用于最终增强版推理的真实数据集。"""

    def __init__(self, opt: dict):
        super().__init__()
        self.opt = opt
        self.gt_mode = opt.get("gt_mode", "partial")

        self.degraded_paths = _list_image_paths(opt.get("dataroot_degraded"))
        if not self.degraded_paths:
            raise FileNotFoundError("dataroot_degraded 下没有可用图像。")

        self.mask_map = _build_stem_map(_list_image_paths(opt.get("dataroot_mask")))
        if not self.mask_map:
            raise FileNotFoundError("dataroot_mask 下没有可用掩码。")

        self.gt_map = _build_stem_map(_list_image_paths(opt.get("dataroot_GT")))
        self.color_prior_map = _build_stem_map(
            _list_image_paths(opt.get("dataroot_color_prior"))
        )
        self.confidence_map = _build_stem_map(
            _list_image_paths(opt.get("dataroot_confidence"))
        )

    def __len__(self) -> int:
        return len(self.degraded_paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        degraded_path = self.degraded_paths[index]
        stem = os.path.splitext(os.path.basename(degraded_path))[0]
        if stem not in self.mask_map:
            raise KeyError(f"缺少同名掩码: {stem}")

        degraded = _load_rgb_image(degraded_path)
        height, width = degraded.shape[:2]
        mask_hole = _load_mask(self.mask_map[stem], (height, width))
        mask_known = 1.0 - mask_hole

        sample = {
            "degraded": torch.from_numpy(np.transpose(degraded, (2, 0, 1))).float(),
            "mask_hole": torch.from_numpy(mask_hole[None, ...]).float(),
            "mask_known": torch.from_numpy(mask_known[None, ...]).float(),
            "degraded_path": degraded_path,
            "stem": stem,
            "gt_mode": self.gt_mode,
        }

        if stem in self.gt_map:
            gt = _load_rgb_image(self.gt_map[stem])
            sample["GT"] = torch.from_numpy(np.transpose(gt, (2, 0, 1))).float()
            sample["GT_path"] = self.gt_map[stem]

        if stem in self.color_prior_map:
            color_prior = _load_rgb_image(self.color_prior_map[stem])
            sample["color_prior"] = torch.from_numpy(
                np.transpose(color_prior, (2, 0, 1))
            ).float()

        if stem in self.confidence_map:
            confidence = _load_confidence(self.confidence_map[stem], (height, width))
            sample["confidence"] = torch.from_numpy(confidence[None, ...]).float()

        return sample
