#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""整图颜色替换测试脚本。

流程：
1. 对整张图做 KMeans 聚类，得到当前褪色调色板。
2. 使用当前统一推理链批量恢复每个调色板颜色。
3. 用恢复后的调色板重建整张图。

这是一个可视化实验脚本，不会修改训练数据或 checkpoint。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

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


def perform_clustering(image_rgb: np.ndarray, k: int = 8):
    """Run KMeans on the full image to find dominant colors."""
    print(f"[1/4] Running KMeans with {k} clusters...")
    h, w, _ = image_rgb.shape
    pixel_values = image_rgb.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return labels.flatten().reshape((h, w)), centers


@torch.no_grad()
def batch_restore_palette(rgb_centers: np.ndarray, bundle, args, device: torch.device):
    """Restore a batch of palette colors using the current unified inference path."""
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

    print(f"[2/4] Restoring {batch_size} palette colors (samples={args.num_samples})...")
    samples = []
    for i in range(int(args.num_samples)):
        if (i + 1) % 5 == 0 or i == 0:
            print(f"       sample {i + 1}/{args.num_samples}")
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
    return pred_rgb, confidences


def save_palette_preview(input_palette: np.ndarray, output_palette: np.ndarray, confidences, path: Path) -> None:
    """Save a simple palette strip for quick inspection."""
    n = len(input_palette)
    tile_h = 48
    tile_w = 80
    canvas = np.ones((tile_h * 2, tile_w * n, 3), dtype=np.uint8) * 255
    for i in range(n):
        x0 = i * tile_w
        canvas[:tile_h, x0 : x0 + tile_w] = input_palette[i]
        canvas[tile_h:, x0 : x0 + tile_w] = output_palette[i]
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main() -> None:
    parser = argparse.ArgumentParser(description="整图颜色替换测试脚本")
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--output_image", type=str, default="restored_batch.png")
    parser.add_argument("--palette_preview", type=str, default="")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--prototype_bank", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_colors", type=int, default=12, help="KMeans 聚类数")
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

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.ckpt}")
    bundle = load_checkpoint(args.ckpt, device, prototype_bank_path=args.prototype_bank)

    img_bgr = cv2.imread(args.input_image)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {args.input_image}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    labels, centers = perform_clustering(img_rgb, k=int(args.n_colors))
    restored_palette, confidences = batch_restore_palette(centers, bundle, args, device)

    print("[3/4] Reconstructing image from restored palette...")
    restored_img_rgb = restored_palette[labels]

    print("[4/4] Saving outputs...")
    out_path = Path(args.output_image)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(restored_img_rgb, cv2.COLOR_RGB2BGR))

    if args.palette_preview:
        preview_path = Path(args.palette_preview)
    else:
        preview_path = out_path.with_name(out_path.stem + "_palette.png")
    save_palette_preview(centers.astype(np.uint8), restored_palette.astype(np.uint8), confidences, preview_path)

    print(f"Saved recolored image to: {out_path}")
    print(f"Saved palette preview to: {preview_path}")


if __name__ == "__main__":
    main()
