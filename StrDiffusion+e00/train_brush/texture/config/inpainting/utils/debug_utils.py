# -*- coding: utf-8 -*-
"""
Debug 可视化工具：把训练/推理中的 Tensor 保存成图片，便于排查数据对齐、Mask、先验图是否正确。
"""

from __future__ import annotations
import os
import numpy as np
import torch
import cv2


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def tensor_to_uint8_rgb(t: torch.Tensor) -> np.ndarray:
    if t.dim() == 4:
        t = t[0]
    t = t.detach().float().cpu()
    if t.min() < -0.1:
        t = (t + 1.0) * 0.5
    t = t.clamp(0, 1)
    return (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def tensor_to_uint8_gray(t: torch.Tensor) -> np.ndarray:
    if t.dim() == 4:
        t = t[0]
    if t.dim() == 3 and t.shape[0] == 1:
        t = t[0]
    t = t.detach().float().cpu()
    if t.min() < -0.1:
        t = (t + 1.0) * 0.5
    t = t.clamp(0, 1)
    return (t.numpy() * 255.0).round().astype(np.uint8)


def save_rgb(t: torch.Tensor, save_path: str) -> None:
    _ensure_dir(os.path.dirname(save_path))
    rgb = tensor_to_uint8_rgb(t)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, bgr)


def save_mask(t: torch.Tensor, save_path: str) -> None:
    _ensure_dir(os.path.dirname(save_path))
    gray = tensor_to_uint8_gray(t)
    cv2.imwrite(save_path, gray)


def save_debug_pack(
    debug_dir: str,
    prefix: str,
    input_img: torch.Tensor = None,
    transformed_gt: torch.Tensor = None,
    prior: torch.Tensor = None,
    conf: torch.Tensor = None,
    masked_input: torch.Tensor = None,
) -> None:
    _ensure_dir(debug_dir)
    if input_img is not None:
        save_rgb(input_img, os.path.join(debug_dir, f"{prefix}_Input_Image.png"))
    if transformed_gt is not None:
        save_rgb(transformed_gt, os.path.join(debug_dir, f"{prefix}_Transformed_GT.png"))
    if prior is not None:
        save_rgb(prior, os.path.join(debug_dir, f"{prefix}_Generated_Prior.png"))
    if conf is not None:
        save_mask(conf, os.path.join(debug_dir, f"{prefix}_Confidence_Map.png"))
    if masked_input is not None:
        save_rgb(masked_input, os.path.join(debug_dir, f"{prefix}_Masked_Input.png"))
