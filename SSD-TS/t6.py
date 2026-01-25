#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full-Image Replacement (NO mask, NO confidence)

Pipeline:
1) Read image.
2) Sample pixels from whole image -> KMeans -> faded_palette_rgb.
3) Restore palette via 3D LUT (lut_lab or lut_rgb).
4) For ALL pixels: find nearest faded palette color in RGB; map to restored palette in Lab,
   optionally keeping per-pixel L from original image.
5) Save recolored full image.

Example:
python t6.py \
  --img_path 1.png \
  --lut_npz pigment_lut33.npz \
  --n_colors 128 \
  --output_dir results
"""

import argparse
import os
import sys
import numpy as np
import cv2

sys.path.append(os.getcwd())

try:
    from pigment_task.color_utils import rgb_to_lab, lab_to_rgb
except ImportError as e:
    print(f"[Error] pigment_task.color_utils not found: {e}")
    sys.exit(1)


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


def lut_trilinear(points_rgb: np.ndarray, grid: np.ndarray, lut: np.ndarray, lut_order: str = "rgb") -> np.ndarray:
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
    required = ["grid", "lut_lab", "lut_rgb"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"LUT npz missing keys: {missing}. Found keys: {list(data.keys())}")

    grid = data["grid"].astype(np.float32)
    lut_lab = data["lut_lab"].astype(np.float32)
    lut_rgb = data["lut_rgb"]
    return grid, lut_lab, lut_rgb


def nearest_palette_indices(pixels_rgb: np.ndarray, palette_rgb: np.ndarray, chunk: int = 200000) -> np.ndarray:
    """Nearest palette color index for each pixel (chunked squared L2 in RGB)."""
    N = pixels_rgb.shape[0]
    out = np.empty(N, dtype=np.int64)
    pal = palette_rgb.astype(np.float32)

    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        x = pixels_rgb[s:e].astype(np.float32)  # (M,3)
        d2 = np.sum((x[:, None, :] - pal[None, :, :]) ** 2, axis=2)  # (M,K)
        out[s:e] = np.argmin(d2, axis=1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results_full_nomask')

    parser.add_argument('--n_colors', type=int, default=32)
    parser.add_argument('--sample_max', type=int, default=30000,
                        help='Max pixels sampled for KMeans from whole image.')

    # LUT
    parser.add_argument('--lut_npz', type=str, default='pigment_lut33.npz')
    parser.add_argument('--lut_order', type=str, default='rgb', choices=['rgb', 'bgr'])
    parser.add_argument('--use_lut', type=str, default='lab', choices=['lab', 'rgb'],
                        help='Restore palette using lut_lab or lut_rgb.')

    # Luminance
    g = parser.add_argument_group('Luminance options')
    g.add_argument('--keep_luminance', dest='keep_luminance', action='store_true',
                   help='Keep per-pixel L from original image, only swap (a,b). (default)')
    g.add_argument('--no_keep_luminance', dest='keep_luminance', action='store_false',
                   help='Do not preserve L; use LUT Lab (L,a,b) directly.')
    parser.set_defaults(keep_luminance=True)

    # Speed/memory
    parser.add_argument('--nn_chunk', type=int, default=200000)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load image
    print(f"Loading image: {args.img_path}")
    img_bgr = cv2.imread(args.img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {args.img_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]
    img_lab = rgb_to_lab(img_rgb)  # (H,W,3)

    # 2) KMeans palette from whole image
    print("Step 1: KMeans palette from whole image...")
    all_pixels = img_rgb.reshape(-1, 3)
    if len(all_pixels) > args.sample_max:
        idx = np.random.choice(len(all_pixels), args.sample_max, replace=False)
        sample_pixels = all_pixels[idx]
    else:
        sample_pixels = all_pixels

    pixel_values = np.float32(sample_pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, _, faded_palette_rgb = cv2.kmeans(
        pixel_values, args.n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )  # (K,3), float32 in [0,255]

    # 3) Restore palette via LUT
    print(f"Step 2: Restoring palette via LUT ({args.lut_npz})...")
    grid, lut_lab, lut_rgb = load_lut_npz(args.lut_npz)

    if args.use_lut == 'lab':
        restored_palette_lab = lut_trilinear(faded_palette_rgb, grid, lut_lab, lut_order=args.lut_order)
        if restored_palette_lab.shape[-1] != 3:
            raise ValueError(f"lut_lab interpolation output shape unexpected: {restored_palette_lab.shape}")
    else:
        restored_palette_rgb = lut_trilinear(faded_palette_rgb, grid, lut_rgb, lut_order=args.lut_order)
        restored_palette_rgb = np.clip(restored_palette_rgb, 0, 255).astype(np.uint8)
        restored_palette_lab = rgb_to_lab(restored_palette_rgb)

    # 4) Full image replace
    print("Step 3: Full-image palette replace...")
    all_rgb = img_rgb.reshape(-1, 3).astype(np.float32)     # (N,3)
    all_lab = img_lab.reshape(-1, 3).astype(np.float32)     # (N,3)

    nn_idx = nearest_palette_indices(all_rgb, faded_palette_rgb, chunk=args.nn_chunk)  # (N,)
    target_lab = restored_palette_lab[nn_idx].astype(np.float32)  # (N,3)

    if args.keep_luminance:
        new_lab = target_lab.copy()
        new_lab[:, 0] = all_lab[:, 0]
    else:
        new_lab = target_lab

    recolored_rgb = lab_to_rgb(new_lab)
    if recolored_rgb.dtype != np.uint8:
        recolored_rgb = np.clip(recolored_rgb, 0, 255).astype(np.uint8)

    recolored_map = recolored_rgb.reshape(H, W, 3)

    # 5) Save
    out_bgr = cv2.cvtColor(recolored_map, cv2.COLOR_RGB2BGR)
    out_path = os.path.join(args.output_dir, "color_prior_full_replace.png")
    cv2.imwrite(out_path, out_bgr)

    print("✅ Full-Image LUT Palette Replace Finished!")
    print(f"-> Output: {out_path}")


if __name__ == "__main__":
    main()
