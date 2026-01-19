#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate Color Prior & Confidence Maps (Luminance-Preserving Recolor Version)
METHOD: "Keep the Texture, Force-Swap the Color"
1. Inpaint structure (Telea) -> Extract Luminance (L) channel (Texture).
2. Analyze Context -> Predict Restored Colors (a, b) channels.
3. For every hole pixel, find the closest context color, and FORCE replace its (a,b) channels 
   with the restored version, while keeping the original L channel.

 python t3.py \
  --img_path /home/610-wws/Impainting/dataset/裁剪的图片/test/cropped_images/42-0-1_bottom.jpg \
  --mask_path /home/610-wws/Impainting/dataset/裁剪的图片/test/output_masks/42-0-1_bottom_mask.png \
  --ckpt ckpt/pigment_lab_raman_xrd/best_model.pt  \
  --n_colors 64 \
  --cond_method pred
"""
import argparse
import os
import sys
import numpy as np
import cv2
import torch
from scipy.spatial.distance import cdist

sys.path.append(os.getcwd())

try:
    from pigment_task.infer_pigment import (
        _load_ckpt,
        _predict_embeds_from_rgb,
        _build_cond_from_pred_embeds,
        _retrieval_raman_embed,
        _load_library_npz,
        _sample_with_confidence
    )
    from pigment_task.color_utils import LabNorm, rgb_to_lab, lab_to_rgb
except ImportError as e:
    print(f"[Error] Core modules not found: {e}")
    sys.exit(1)

def get_spatial_confidence(mask):
    """Spatial confidence: 1.0 at boundary, dropping to 0.1 at center."""
    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_dist = dist_map.max() + 1e-8
    norm_dist = dist_map / max_dist
    spatial_conf = 1.0 - norm_dist
    spatial_conf = np.clip(spatial_conf, 0.1, 1.0)
    spatial_conf[mask == 0] = 1.0 
    return spatial_conf

def batch_inference_with_conf(rgb_centers, model_components, args, device, lib_raman, lab_norm):
    """Infer restored colors for the palette."""
    cfg, denoiser, conditioner, schedule, color_encoder, cond_predictor = model_components
    K = len(rgb_centers)
    
    # Preprocess
    lab_centers = rgb_to_lab(rgb_centers)
    x_curr = lab_norm.normalize(lab_centers).astype(np.float32)
    x_curr_t = torch.from_numpy(x_curr).to(device)
    
    # Dummy inputs
    x0 = np.stack([lab_centers, lab_centers], axis=1)
    x0n = lab_norm.normalize(x0).astype(np.float32)
    mask = np.zeros((K, 2, 3), dtype=np.float32)
    mask[:, 1, :] = 1.0 
    
    x0_t = torch.from_numpy(x0n).to(device)
    mask_t = torch.from_numpy(mask).to(device)

    # Condition
    cond = None
    if args.cond_method == "retrieval":
        zc = color_encoder(x_curr_t)
        raman_emb, _, _ = _retrieval_raman_embed(zc, lib_raman, top_k=args.retrieval_k, temp=args.retrieval_temp)
        embeds_pred = {"raman": raman_emb}
        if conditioner.xrd_enc:
            embeds_pred["xrd"] = cond_predictor(zc)["xrd_emb"] if cond_predictor else torch.zeros_like(raman_emb)
        cond = _build_cond_from_pred_embeds(conditioner, embeds_pred)
    else:
        embeds_pred = _predict_embeds_from_rgb(x_curr_t, conditioner, color_encoder, cond_predictor)
        cond = _build_cond_from_pred_embeds(conditioner, embeds_pred)

    # Sampling - 修改这里以适配实际返回值数量
    result = _sample_with_confidence(
        denoiser, schedule, x0_t, mask_t, cond, num_samples=args.num_samples
    )
    
    # 根据实际返回值数量处理
    if isinstance(result, tuple) and len(result) >= 2:
        pred_lab0_batch, model_conf_ret = result[0], result[1]
    else:
        # 如果返回值不符合预期，抛出错误
        raise ValueError(f"_sample_with_confidence returned unexpected format: {result}")
    
    # Extract Confidence Safely
    conf_vals = None
    if isinstance(model_conf_ret, dict):
        for k in ["confidence_diffusion", "confidence", "std"]:
            if k in model_conf_ret:
                conf_vals = model_conf_ret[k]
                break
        if conf_vals is None: conf_vals = list(model_conf_ret.values())[0]
    else:
        conf_vals = model_conf_ret

    if isinstance(conf_vals, torch.Tensor):
        conf_vals = conf_vals.detach().cpu().numpy()
    if np.ndim(conf_vals) == 0:
        conf_vals = np.full(K, float(conf_vals))
    elif len(conf_vals) != K:
        conf_vals = np.resize(conf_vals, K)

    return pred_lab0_batch, conf_vals

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--mask_path', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default="results_priors_recolor")
    
    parser.add_argument('--n_colors', type=int, default=32)
    parser.add_argument('--inpaint_radius', type=int, default=3)
    parser.add_argument('--cond_method', type=str, default='pred')
    parser.add_argument('--library_npz', type=str, default='data/standard_alignment/library_embeddings.npz')
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--retrieval_k', type=int, default=5)
    parser.add_argument('--retrieval_temp', type=float, default=0.07)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load & Preprocess
    print(f"Loading image: {args.img_path}")
    img = cv2.imread(args.img_path)
    mask = cv2.imread(args.mask_path, 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 2. Traditional Inpainting (Geometry Only)
    print("Step 1: Structure Inpainting...")
    inpainted_img = cv2.inpaint(img, mask, args.inpaint_radius, cv2.INPAINT_TELEA)
    cv2.imwrite(os.path.join(args.output_dir, "01_structure.png"), inpainted_img)

    # 3. Analyze Context (Outside Mask)
    print("Step 2: Analyzing Context Palette...")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    valid_pixels = img_rgb[mask == 0]
    
    if len(valid_pixels) > 30000:
        indices = np.random.choice(len(valid_pixels), 30000, replace=False)
        sample_pixels = valid_pixels[indices]
    else:
        sample_pixels = valid_pixels

    # K-Means on Context
    pixel_values = np.float32(sample_pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, faded_palette_rgb = cv2.kmeans(pixel_values, args.n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # 4. Infer Restored Palette
    print("Step 3: Predicting Restored Palette...")
    model_components = _load_ckpt(args.ckpt, device)
    lab_norm = LabNorm()
    
    lib_raman = None
    if args.cond_method == 'retrieval':
        try:
            lib = _load_library_npz(args.library_npz)
            lib_raman = torch.from_numpy(lib['raman_emb' if 'raman_emb' in lib else 'embeddings']).float().to(device)
        except:
            args.cond_method = 'pred'

    # Get Restored Palette in LAB
    restored_palette_lab, palette_conf = batch_inference_with_conf(
        faded_palette_rgb, model_components, args, device, lib_raman, lab_norm
    )
    
    # 5. Luminance-Guided Recolor
    print("Step 4: Luminance-Guided Recoloring...")
    
    # Convert Inpainted Image to LAB
    inpainted_rgb = cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2RGB)
    inpainted_lab = rgb_to_lab(inpainted_rgb) # (H, W, 3)
    
    # Extract Hole Pixels
    hole_indices = np.where(mask == 255)
    hole_pixels_rgb = inpainted_rgb[hole_indices].astype(np.float32)
    
    if len(hole_pixels_rgb) > 0:
        # Match each hole pixel to nearest Faded Palette Color (based on RGB similarity)
        dists = cdist(hole_pixels_rgb, faded_palette_rgb)
        nearest_cluster_indices = np.argmin(dists, axis=1)
        
        # --- KEY LOGIC: FUSION ---
        # 1. Take L (Lightness) from Inpainted Image (Preserves structure/texture)
        # 2. Take a, b (Color) from Restored Palette (Fixes color)
        
        # Get the original L channel of the hole pixels
        current_L = inpainted_lab[hole_indices][:, 0]
        
        # Get the targeted Restored Color (L, a, b)
        target_lab_full = restored_palette_lab[nearest_cluster_indices]
        
        # Construct New Color:
        # L = current_L (Keep texture from inpainting)
        # a = target_lab_full[:, 1] (Use predicted color)
        # b = target_lab_full[:, 2] (Use predicted color)
        # Optional: Blend L slightly if restoration implies significant brightening
        
        new_pixels_lab = np.zeros_like(target_lab_full)
        new_pixels_lab[:, 0] = current_L 
        new_pixels_lab[:, 1] = target_lab_full[:, 1]
        new_pixels_lab[:, 2] = target_lab_full[:, 2]
        
        hole_restored_rgb = lab_to_rgb(new_pixels_lab)
        hole_confidences = palette_conf[nearest_cluster_indices]
        
        # 6. Reconstruct
        color_prior_map = inpainted_rgb.copy()
        color_prior_map[hole_indices] = hole_restored_rgb
        
        # Confidence
        h, w = img.shape[:2]
        spatial_conf = get_spatial_confidence(mask)
        color_conf_map = np.ones((h, w), dtype=np.float32)
        
        # 修复：确保hole_confidences是一维数组，形状与hole_indices[0]相同
        if hole_confidences.ndim > 1:
            # 如果是多维数组，取第一个值或平均值
            hole_confidences = np.mean(hole_confidences, axis=-1) if hole_confidences.ndim > 1 else hole_confidences.flatten()
        
        color_conf_map[hole_indices] = hole_confidences
        final_conf = spatial_conf * color_conf_map
    else:
        color_prior_map = inpainted_rgb
        final_conf = np.ones(img.shape[:2])

    # 7. Save
    color_prior_bgr = cv2.cvtColor(color_prior_map, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(args.output_dir, "color_prior_recolor.png"), color_prior_bgr)
    
    conf_vis = (final_conf * 255).astype(np.uint8)
    conf_heat = cv2.applyColorMap(conf_vis, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(args.output_dir, "confidence_map.png"), conf_vis)
    cv2.imwrite(os.path.join(args.output_dir, "confidence_heatmap.png"), conf_heat)
    
    print(f"✅ Recolor Finished!")
    print(f"-> Prior: {os.path.join(args.output_dir, 'color_prior_recolor.png')}")

if __name__ == "__main__":
    main()