#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""基于模型恢复调色板并生成颜色先验图与置信图。

流程：
1. 使用 Telea 做结构修补，保留纹理和亮度信息。
2. 从掩膜外区域提取上下文调色板。
3. 用当前统一推理链恢复调色板颜色。
4. 对掩膜区域按最近上下文调色板做重着色，并输出置信图。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.distance import cdist

try:
    from inference.pipeline import (
        _confidence_or_default,
        _fuse_confidence,
        _resolve_condition,
        _stabilize_single_rgb_prediction,
        load_checkpoint,
    )
    from training.diffusion import p_sample_loop
    from utils.color_utils import LabNorm, lab_to_rgb, rgb_to_lab
except ImportError as e:
    print(f"[Error] Core modules not found: {e}")
    raise SystemExit(1)


def get_spatial_confidence(mask_u8: np.ndarray) -> np.ndarray:
    """Boundary pixels keep high confidence, center pixels decay gradually."""
    dist_map = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    max_dist = dist_map.max() + 1e-8
    norm_dist = dist_map / max_dist
    spatial_conf = 1.0 - norm_dist
    spatial_conf = np.clip(spatial_conf, 0.1, 1.0)
    spatial_conf[mask_u8 == 0] = 1.0
    return spatial_conf.astype(np.float32)


@torch.no_grad()
def batch_restore_palette(rgb_centers: np.ndarray, bundle, args, device: torch.device):
    """Restore a batch of palette colors and return Lab, RGB and confidence."""
    denoiser = bundle["denoiser"]
    schedule = bundle["schedule"]
    lab_norm = LabNorm()

    batch_size = len(rgb_centers)
    lab_centers = rgb_to_lab(rgb_centers.astype(np.float32))
    x_curr = lab_norm.normalize(lab_centers).astype(np.float32)
    x_curr_t = torch.from_numpy(x_curr).to(device)

    x0 = np.stack([lab_centers, lab_centers], axis=1)
    x0n = lab_norm.normalize(x0).astype(np.float32)
    mask = np.zeros((batch_size, 2, 3), dtype=np.float32)
    mask[:, 1, :] = 1.0

    x0_t = torch.from_numpy(x0n).to(device)
    mask_t = torch.from_numpy(mask).to(device)

    cond, info = _resolve_condition(
        bundle=bundle,
        batch=None,
        x_curr=x_curr_t,
        device=device,
        cond_method=args.cond_method,
        library_npz=args.library_npz if args.library_npz else None,
        retrieval_k=int(args.retrieval_k),
        retrieval_temp=float(args.retrieval_temp),
    )

    print(f"Step 3: Restoring {batch_size} palette colors (samples={args.num_samples})...")
    samples = []
    for i in range(int(args.num_samples)):
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  sample {i + 1}/{args.num_samples}")
        x_s = p_sample_loop(denoiser, schedule, x_obs=x0_t * mask_t, obs_mask=mask_t, cond=cond)
        samples.append(x_s[:, 0, :].detach().cpu().numpy())

    arr = np.stack(samples, axis=0)
    mean_norm = np.mean(arr, axis=0)
    std_norm = np.std(arr, axis=0)

    pred_lab = lab_norm.denormalize(mean_norm)
    std_scalar = np.linalg.norm(std_norm, axis=1)
    diff_conf = np.exp(-std_scalar).astype(np.float32)

    bridge_conf = None
    if isinstance(info.get("confidence", None), torch.Tensor):
        bridge_conf = info["confidence"].detach().cpu().numpy().reshape(-1).astype(np.float32)

    retrieval_conf = None
    retrieval_info = info.get("retrieval", None)
    if isinstance(retrieval_info, dict) and isinstance(retrieval_info.get("confidence", None), torch.Tensor):
        retrieval_conf = retrieval_info["confidence"].detach().cpu().numpy().reshape(-1).astype(np.float32)
    elif args.cond_method == "retrieval" and isinstance(info.get("confidence", None), torch.Tensor):
        retrieval_conf = info["confidence"].detach().cpu().numpy().reshape(-1).astype(np.float32)

    infer_cfg = bundle["cfg"].get("inference", {})
    confidences = []
    for i in range(batch_size):
        fused = _fuse_confidence(
            float(diff_conf[i]),
            float(std_scalar[i]),
            None if retrieval_conf is None else float(retrieval_conf[i]),
        )
        base_conf = _confidence_or_default(
            fused,
            None if bridge_conf is None else float(bridge_conf[i]),
            float(diff_conf[i]),
        )
        if bool(infer_cfg.get("stabilize_single_rgb", True)) and args.cond_method != "true":
            pred_lab[i], eff_conf = _stabilize_single_rgb_prediction(
                lab_centers[i],
                pred_lab[i],
                base_conf,
                infer_cfg,
            )
        else:
            eff_conf = base_conf
        confidences.append(float(eff_conf))

    pred_rgb = lab_to_rgb(pred_lab)
    return pred_lab, pred_rgb, np.asarray(confidences, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成颜色先验图和置信图")
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--mask_path", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results_priors_recolor")
    parser.add_argument("--prototype_bank", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_colors", type=int, default=32)
    parser.add_argument("--inpaint_radius", type=int, default=3)
    parser.add_argument(
        "--cond_method",
        type=str,
        default="auto",
        choices=["auto", "pred", "retrieval", "posterior", "posterior_retrieval"],
    )
    parser.add_argument("--library_npz", type=str, default="data/standard_alignment/library_embeddings.npz")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--retrieval_k", type=int, default=5)
    parser.add_argument("--retrieval_temp", type=float, default=0.07)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.ckpt}")
    bundle = load_checkpoint(args.ckpt, device, prototype_bank_path=args.prototype_bank)

    img = cv2.imread(args.img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {args.img_path}")
    mask = cv2.imread(args.mask_path, 0)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {args.mask_path}")
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    print("Step 1: Structure inpainting...")
    inpainted_img = cv2.inpaint(img, mask, args.inpaint_radius, cv2.INPAINT_TELEA)
    cv2.imwrite(str(output_dir / "01_structure.png"), inpainted_img)

    print("Step 2: Building context palette...")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    valid_pixels = img_rgb[mask == 0]
    if len(valid_pixels) == 0:
        raise ValueError("Mask covers the whole image; no context pixels are available.")
    if len(valid_pixels) > 30000:
        indices = np.random.choice(len(valid_pixels), 30000, replace=False)
        sample_pixels = valid_pixels[indices]
    else:
        sample_pixels = valid_pixels

    pixel_values = np.float32(sample_pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, faded_palette_rgb = cv2.kmeans(pixel_values, args.n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    restored_palette_lab, restored_palette_rgb, palette_conf = batch_restore_palette(
        faded_palette_rgb,
        bundle,
        args,
        device,
    )

    print("Step 4: Recoloring the masked region...")
    inpainted_rgb = cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2RGB)
    inpainted_lab = rgb_to_lab(inpainted_rgb)

    hole_indices = np.where(mask == 255)
    hole_pixels_rgb = inpainted_rgb[hole_indices].astype(np.float32)

    if len(hole_pixels_rgb) > 0:
        dists = cdist(hole_pixels_rgb, faded_palette_rgb)
        nearest_cluster_indices = np.argmin(dists, axis=1)

        current_L = inpainted_lab[hole_indices][:, 0]
        target_lab_full = restored_palette_lab[nearest_cluster_indices]

        new_pixels_lab = np.zeros_like(target_lab_full)
        new_pixels_lab[:, 0] = current_L
        new_pixels_lab[:, 1] = target_lab_full[:, 1]
        new_pixels_lab[:, 2] = target_lab_full[:, 2]

        hole_restored_rgb = lab_to_rgb(new_pixels_lab)
        hole_confidences = palette_conf[nearest_cluster_indices]

        color_prior_map = inpainted_rgb.copy()
        color_prior_map[hole_indices] = hole_restored_rgb

        h, w = img.shape[:2]
        spatial_conf = get_spatial_confidence(mask)
        color_conf_map = np.ones((h, w), dtype=np.float32)
        color_conf_map[hole_indices] = hole_confidences.astype(np.float32).reshape(-1)
        final_conf = np.clip(spatial_conf * color_conf_map, 0.0, 1.0)
    else:
        color_prior_map = inpainted_rgb
        final_conf = np.ones(img.shape[:2], dtype=np.float32)

    color_prior_bgr = cv2.cvtColor(color_prior_map, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_dir / "color_prior_recolor.png"), color_prior_bgr)

    conf_vis = np.clip(final_conf * 255.0, 0, 255).astype(np.uint8)
    conf_heat = cv2.applyColorMap(conf_vis, cv2.COLORMAP_JET)
    cv2.imwrite(str(output_dir / "confidence_map.png"), conf_vis)
    cv2.imwrite(str(output_dir / "confidence_heatmap.png"), conf_heat)

    print(f"Saved prior image to: {output_dir / 'color_prior_recolor.png'}")
    print(f"Saved confidence map to: {output_dir / 'confidence_map.png'}")


if __name__ == "__main__":
    main()
