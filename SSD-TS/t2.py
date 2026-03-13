#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
整图颜色替换
python t2.py \
  --ckpt ckpt/pigment_lab_raman_xrd/best_model.pt \
  --input_image /home/610-wws/Impainting/dataset/裁剪的图片/test/cropped_images/42-0-1_bottom.jpg \
  --output_image test_restored_block.png \
  --n_colors 32 \
  --cond_method pred\
  --library_npz data/standard_alignment/library_embeddings.npz \
  --num_samples 30

  python t2.py \
  --ckpt ckpt/pigment_lab_raman_xrd/best_model.pt \
  --input_image /home/610-wws/Impainting/SSD-TS/1.png \
  --output_image test_restored_block.png \
  --n_colors 32 \
  --cond_method pred\
  --library_npz data/standard_alignment/library_embeddings.npz \
  --num_samples 30
"""
import argparse
import os
import numpy as np
import cv2
import torch
from tqdm import tqdm

try:
    from bridge.condition_builder import (
        build_cond_from_pred_embeds as _build_cond_from_pred_embeds,
        load_library_npz as _load_library_npz,
        predict_embeds_from_rgb as _predict_embeds_from_rgb,
        retrieval_raman_embed as _retrieval_raman_embed,
    )
    from inference.pipeline import load_checkpoint as _load_ckpt
    from inference.uncertainty import sample_with_confidence as _sample_with_confidence
    from legacy.experiment_compat import unpack_model_components as _unpack_model_components
    from utils.color_utils import LabNorm, rgb_to_lab, lab_to_rgb
except ImportError as e:
    print(f"[Error] Core modules not found: {e}")
    raise SystemExit(1)


def perform_clustering(image, k=8):
    """
    K-Means clustering to find dominant colors.
    """
    print(f"[1/4] Identifying {k} dominant color regions...")
    h, w, c = image.shape
    pixel_values = image.reshape((-1, 3)).astype(np.float32)

    # K-Means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    
    # Reshape labels to (H, W)
    labels = labels.flatten().reshape((h, w))
    return labels, centers # centers is (k, 3) float32

def batch_inference(rgb_centers, model_components, args, device, lib_raman, lab_norm):
    """
    Run diffusion on all centers in parallel.
    rgb_centers: (K, 3) numpy array
    """
    print(f"[2/4] Preparing batch inference for {len(rgb_centers)} colors...")
    cfg, denoiser, conditioner, schedule, color_encoder, cond_predictor = _unpack_model_components(model_components)
    
    K = len(rgb_centers)
    
    # 1. Preprocess Batch: RGB -> Lab -> Normalize
    # rgb_centers shape: (K, 3)
    lab_centers = rgb_to_lab(rgb_centers) # (K, 3)
    
    # x_curr: (K, 3) -> Normalized Lab
    x_curr = lab_norm.normalize(lab_centers).astype(np.float32)
    x_curr_t = torch.from_numpy(x_curr).to(device)
    
    # x0: (K, 2, 3) -> Sequence input for diffusion [target, current]
    # We use x_curr as placeholder for target (index 0), it will be masked anyway
    x0 = np.stack([lab_centers, lab_centers], axis=1) 
    x0n = lab_norm.normalize(x0).astype(np.float32)
    
    # mask: (K, 2, 3) -> 0 for target, 1 for current
    mask = np.zeros((K, 2, 3), dtype=np.float32)
    mask[:, 1, :] = 1.0
    
    x0_t = torch.from_numpy(x0n).to(device)
    mask_t = torch.from_numpy(mask).to(device)

    # 2. Build Batch Condition
    print("[3/4] Running Diffusion Model (Parallel)...")
    cond = None
    if args.cond_method == "retrieval":
        # Retrieval for the whole batch at once
        zc = color_encoder(x_curr_t) # (K, d)
        
        # Batch retrieval
        raman_emb, w, _ = _retrieval_raman_embed(
            zc, lib_raman, top_k=int(args.retrieval_k), temp=float(args.retrieval_temp)
        )
        
        embeds_pred = {}
        if conditioner.raman_enc: 
            embeds_pred["raman"] = raman_emb
        if conditioner.xrd_enc:
            if cond_predictor:
                tmp = cond_predictor(zc)
                embeds_pred["xrd"] = tmp["xrd_emb"]
            else:
                embeds_pred["xrd"] = torch.zeros_like(raman_emb)
                
        cond = _build_cond_from_pred_embeds(conditioner, embeds_pred)
        
    elif args.cond_method == "pred":
        embeds_pred = _predict_embeds_from_rgb(x_curr_t, conditioner, color_encoder, cond_predictor)
        cond = _build_cond_from_pred_embeds(conditioner, embeds_pred)

    # 3. Parallel Sampling
    # _sample_with_confidence runs the diffusion loop `num_samples` times.
    # Inside each loop, it processes the entire batch of K items.
    pred_lab0_batch, _, _ = _sample_with_confidence(
        denoiser, schedule, x0_t, mask_t, cond, num_samples=args.num_samples
    )
    # pred_lab0_batch shape: (K, 3), denormalized

    # 4. Postprocess Batch: Lab -> RGB
    pred_rgb0_batch = lab_to_rgb(pred_lab0_batch) # (K, 3) uint8
    
    # Simple clip logic (optional, lab_to_rgb usually handles it)
    return pred_rgb0_batch

def main():
    parser = argparse.ArgumentParser(description="Parallel Pigment Restoration")
    parser.add_argument('--input_image', type=str, required=True)
    parser.add_argument('--output_image', type=str, default="restored_batch.png")
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--n_colors', type=int, default=12, help="Number of color clusters")
    
    # Inference config
    parser.add_argument('--cond_method', type=str, default='retrieval', choices=['retrieval', 'pred'])
    parser.add_argument('--library_npz', type=str, default='data/standard_alignment/library_embeddings.npz')
    parser.add_argument('--num_samples', type=int, default=30)
    parser.add_argument('--retrieval_k', type=int, default=5)
    parser.add_argument('--retrieval_temp', type=float, default=0.07)
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    print(f"Loading model: {args.ckpt}")
    model_components = _load_ckpt(args.ckpt, device)
    
    # Load Library (if needed)
    lib_raman = None
    if args.cond_method == "retrieval":
        try:
            lib = _load_library_npz(args.library_npz)
            key = 'raman_emb' if 'raman_emb' in lib else 'embeddings'
            if key in lib:
                lib_raman = torch.from_numpy(lib[key].astype(np.float32)).to(device)
            else:
                args.cond_method = "pred"
        except:
            args.cond_method = "pred"

    lab_norm = LabNorm()

    # Read Image
    img_bgr = cv2.imread(args.input_image)
    if img_bgr is None: raise ValueError("Image not found")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 1. K-Means Clustering
    labels, centers = perform_clustering(img_rgb, k=args.n_colors)
    
    # 2. Batch Inference (ALL centers at once)
    restored_palette = batch_inference(
        centers, model_components, args, device, lib_raman, lab_norm
    )
    
    # 3. Apply Colors Back
    print("[4/4] Reconstructing image...")
    # Map label indices to restored RGB values
    restored_img_rgb = restored_palette[labels]
    
    # 4. Save
    restored_img_bgr = cv2.cvtColor(restored_img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output_image, restored_img_bgr)
    print(f"✅ Done! Saved to: {args.output_image}")

if __name__ == "__main__":
    main()