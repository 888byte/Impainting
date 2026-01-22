#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate Color Prior & Confidence Maps (LUT-driven, Palette + Luminance-Preserving Recolor)

This is a drop-in conceptual replacement for your original t3.py: it keeps the
"context palette -> recolor hole by nearest palette" logic, but replaces the diffusion/
embedding model with a precomputed 3D LUT from an npz file (e.g. pigment_lut33.npz).

Pipeline (same spirit as original):
1) Inpaint structure (Telea) -> keep texture/L.
2) Extract outside-mask pixels -> KMeans -> faded_palette_rgb.
3) Restore palette via LUT interpolation (lut_lab or lut_rgb) + palette confidence.
4) For each hole pixel: find nearest faded palette color in RGB; swap (a,b) to restored palette,
   optionally keeping L from inpaint.
5) Confidence = spatial_confidence(mask) * palette_conf (from LUT).

Example:
python t5.py \
  --img_path /home/610-wws/Impainting/dataset/裁剪的图片/test/cropped_images/42-0-1_bottom.jpg \
  --mask_path /home/610-wws/Impainting/dataset/裁剪的图片/test/output_masks/42-0-1_bottom_mask.png \
  --lut_npz /home/610-wws/Impainting/SSD-TS/pigment_lut33.npz \
  --n_colors 64 \
  --output_dir results_priors_lut
"""

import argparse
import os
import sys
import numpy as np
import cv2

sys.path.append(os.getcwd())

try:
    # Must match the Lab definition used to generate lut_lab.
    from pigment_task.color_utils import rgb_to_lab, lab_to_rgb
except ImportError as e:
    print(f"[Error] pigment_task.color_utils not found: {e}")
    sys.exit(1)


def get_spatial_confidence(mask_u8: np.ndarray) -> np.ndarray:
    """Spatial confidence: 1.0 at boundary, dropping to 0.1 at center (inside hole)."""
    dist_map = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    max_dist = float(dist_map.max()) + 1e-8
    norm_dist = dist_map / max_dist
    spatial_conf = 1.0 - norm_dist
    spatial_conf = np.clip(spatial_conf, 0.1, 1.0)
    spatial_conf[mask_u8 == 0] = 1.0
    return spatial_conf.astype(np.float32)


def _prepare_indices_and_weights(vals: np.ndarray, grid: np.ndarray):
    vals = np.clip(vals.astype(np.float32), float(grid[0]), float(grid[-1]))
    i0 = np.searchsorted(grid, vals, side="right") - 1
    i0 = np.clip(i0, 0, len(grid) - 2)
    i1 = i0 + 1
    g0 = grid[i0].astype(np.float32)
    g1 = grid[i1].astype(np.float32)
    t = (vals - g0) / (g1 - g0 + 1e-8)
    t = np.clip(t, 0.0, 1.0)
    return i0, i1, t


def lut_trilinear(points_rgb: np.ndarray,
                  grid: np.ndarray,
                  lut: np.ndarray,
                  lut_order: str = "rgb") -> np.ndarray:
    """Trilinear interpolation for a 3D LUT."""
    if points_rgb.ndim != 2 or points_rgb.shape[1] != 3:
        raise ValueError(f"points_rgb must be (N,3), got {points_rgb.shape}")

    if lut_order.lower() not in {"rgb", "bgr"}:
        raise ValueError("lut_order must be 'rgb' or 'bgr'")

    pts = points_rgb.astype(np.float32)
    if lut_order.lower() == "bgr":
        pts = pts[:, ::-1]

    r = pts[:, 0]
    g = pts[:, 1]
    b = pts[:, 2]

    r0, r1, tr = _prepare_indices_and_weights(r, grid)
    g0, g1, tg = _prepare_indices_and_weights(g, grid)
    b0, b1, tb = _prepare_indices_and_weights(b, grid)

    c000 = lut[r0, g0, b0].astype(np.float32)
    c001 = lut[r0, g0, b1].astype(np.float32)
    c010 = lut[r0, g1, b0].astype(np.float32)
    c011 = lut[r0, g1, b1].astype(np.float32)
    c100 = lut[r1, g0, b0].astype(np.float32)
    c101 = lut[r1, g0, b1].astype(np.float32)
    c110 = lut[r1, g1, b0].astype(np.float32)
    c111 = lut[r1, g1, b1].astype(np.float32)

    if c000.ndim == 1:
        trb, tgb, tbb = tr, tg, tb
    else:
        trb, tgb, tbb = tr[:, None], tg[:, None], tb[:, None]

    c00 = c000 * (1 - tbb) + c001 * tbb
    c01 = c010 * (1 - tbb) + c011 * tbb
    c10 = c100 * (1 - tbb) + c101 * tbb
    c11 = c110 * (1 - tbb) + c111 * tbb

    c0 = c00 * (1 - tgb) + c01 * tgb
    c1 = c10 * (1 - tgb) + c11 * tgb

    c = c0 * (1 - trb) + c1 * trb
    return c


def load_lut_npz(path: str):
    data = np.load(path, allow_pickle=True)
    required = ["grid", "lut_lab", "lut_rgb", "lut_conf"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"LUT npz missing keys: {missing}. Found keys: {list(data.keys())}")

    grid = data["grid"].astype(np.float32)
    lut_lab = data["lut_lab"].astype(np.float32)
    lut_rgb = data["lut_rgb"]
    lut_conf = data["lut_conf"].astype(np.float32)

    return grid, lut_lab, lut_rgb, lut_conf, data


def nearest_palette_indices(hole_rgb: np.ndarray, palette_rgb: np.ndarray, chunk: int = 200000) -> np.ndarray:
    """Find nearest palette color for each hole pixel using chunked squared L2 in RGB."""
    N = hole_rgb.shape[0]
    K = palette_rgb.shape[0]
    out = np.empty(N, dtype=np.int64)
    pal = palette_rgb.astype(np.float32)

    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        x = hole_rgb[s:e].astype(np.float32)  # (M,3)
        # (M,K,3) -> (M,K)
        d2 = np.sum((x[:, None, :] - pal[None, :, :]) ** 2, axis=2)
        out[s:e] = np.argmin(d2, axis=1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--mask_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results_priors_lut')

    parser.add_argument('--n_colors', type=int, default=32)
    parser.add_argument('--inpaint_radius', type=int, default=3)

    # LUT
    parser.add_argument('--lut_npz', type=str, default='pigment_lut33.npz')
    parser.add_argument('--lut_order', type=str, default='rgb', choices=['rgb', 'bgr'])
    parser.add_argument('--use_lut', type=str, default='lab', choices=['lab', 'rgb'],
                        help='Restore palette using lut_lab or lut_rgb.')

    # Luminance
    g = parser.add_mutually_exclusive_group()
    g.add_argument('--keep_luminance', dest='keep_luminance', action='store_true',
                   help='Keep L from inpainted image, only swap (a,b). (default)')
    g.add_argument('--no_keep_luminance', dest='keep_luminance', action='store_false',
                   help='Do not preserve L; use LUT Lab (L,a,b) directly.')
    parser.set_defaults(keep_luminance=True)

    # Confidence
    parser.add_argument('--conf_key', type=str, default='lut_conf',
                        help='Key in npz to use as confidence base (default: lut_conf).')

    # Speed/memory
    parser.add_argument('--nn_chunk', type=int, default=200000,
                        help='Chunk size for nearest-palette search (trade speed vs memory).')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load
    print(f"Loading image: {args.img_path}")
    img_bgr = cv2.imread(args.img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {args.img_path}")

    mask_u8 = cv2.imread(args.mask_path, 0)
    if mask_u8 is None:
        raise FileNotFoundError(f"Cannot read mask: {args.mask_path}")
    _, mask_u8 = cv2.threshold(mask_u8, 127, 255, cv2.THRESH_BINARY)

    # 2) Inpaint structure
    print("Step 1: Structure Inpainting (Telea)...")
    inpainted_bgr = cv2.inpaint(img_bgr, mask_u8, args.inpaint_radius, cv2.INPAINT_TELEA)
    cv2.imwrite(os.path.join(args.output_dir, "01_structure.png"), inpainted_bgr)

    # Convert for processing
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    inpainted_lab = rgb_to_lab(inpainted_rgb)

    # 3) Build palette from outside mask (context)
    print("Step 2: Analyzing Context Palette (KMeans)...")
    valid_pixels = img_rgb[mask_u8 == 0]
    if valid_pixels.size == 0:
        raise ValueError("No context pixels found (mask covers whole image?)")

    if len(valid_pixels) > 30000:
        idx = np.random.choice(len(valid_pixels), 30000, replace=False)
        sample_pixels = valid_pixels[idx]
    else:
        sample_pixels = valid_pixels

    pixel_values = np.float32(sample_pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, _, faded_palette_rgb = cv2.kmeans(pixel_values, args.n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 4) Restore palette with LUT
    print(f"Step 3: Restoring Palette via LUT ({args.lut_npz})...")
    grid, lut_lab, lut_rgb, lut_conf_default, lut_all = load_lut_npz(args.lut_npz)

    if args.conf_key in lut_all:
        lut_conf_src = lut_all[args.conf_key].astype(np.float32)
        if lut_conf_src.shape != (len(grid), len(grid), len(grid)):
            print(f"[Warn] conf_key={args.conf_key} shape={lut_conf_src.shape} not (33,33,33); using lut_conf")
            lut_conf_src = lut_conf_default
    else:
        print(f"[Warn] conf_key={args.conf_key} not found; using lut_conf")
        lut_conf_src = lut_conf_default

    palette_conf = lut_trilinear(faded_palette_rgb, grid, lut_conf_src, lut_order=args.lut_order)
    palette_conf = np.clip(palette_conf.astype(np.float32), 0.0, 1.0)

    if args.use_lut == 'lab':
        restored_palette_lab = lut_trilinear(faded_palette_rgb, grid, lut_lab, lut_order=args.lut_order)
        if restored_palette_lab.shape[-1] != 3:
            raise ValueError(f"lut_lab interpolation output shape unexpected: {restored_palette_lab.shape}")
    else:
        restored_palette_rgb = lut_trilinear(faded_palette_rgb, grid, lut_rgb, lut_order=args.lut_order)
        restored_palette_rgb = np.clip(restored_palette_rgb, 0, 255).astype(np.uint8)
        restored_palette_lab = rgb_to_lab(restored_palette_rgb)

    # 5) Recolor hole by nearest palette
    print("Step 4: Luminance-Guided Recoloring...")
    hole_indices = np.where(mask_u8 == 255)
    if hole_indices[0].size > 0:
        hole_rgb = inpainted_rgb[hole_indices].astype(np.float32)
        nn_idx = nearest_palette_indices(hole_rgb, faded_palette_rgb, chunk=args.nn_chunk)

        current_L = inpainted_lab[hole_indices][:, 0]
        target_lab = restored_palette_lab[nn_idx]

        if args.keep_luminance:
            new_lab = target_lab.copy()
            new_lab[:, 0] = current_L
        else:
            new_lab = target_lab

        hole_restored_rgb = lab_to_rgb(new_lab)
        hole_conf = palette_conf[nn_idx]

        color_prior_map = inpainted_rgb.copy()
        color_prior_map[hole_indices] = hole_restored_rgb

        spatial_conf = get_spatial_confidence(mask_u8)
        color_conf_map = np.ones(mask_u8.shape, dtype=np.float32)
        color_conf_map[hole_indices] = hole_conf
        final_conf = spatial_conf * color_conf_map

    else:
        print("[Info] Empty hole region (mask has no 255).")
        color_prior_map = inpainted_rgb
        final_conf = np.ones(mask_u8.shape, dtype=np.float32)

    # 6) Save
    color_prior_bgr = cv2.cvtColor(color_prior_map, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(args.output_dir, "color_prior_lut_palette.png"), color_prior_bgr)

    conf_vis = np.clip(final_conf * 255.0, 0, 255).astype(np.uint8)
    conf_heat = cv2.applyColorMap(conf_vis, cv2.COLORMAP_JET)

    cv2.imwrite(os.path.join(args.output_dir, "confidence_map.png"), conf_vis)
    cv2.imwrite(os.path.join(args.output_dir, "confidence_heatmap.png"), conf_heat)

    print("✅ LUT Palette Recolor Finished!")
    print(f"-> Prior: {os.path.join(args.output_dir, 'color_prior_lut_palette.png')}")


if __name__ == "__main__":
    main()
