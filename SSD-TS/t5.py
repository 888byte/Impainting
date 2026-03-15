#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LUT 驱动的掩膜区域颜色先验图脚本。

流程：
1. 用 Telea 做结构修补。
2. 从掩膜外区域提取上下文调色板。
3. 通过 LUT 恢复调色板颜色与调色板置信度。
4. 对掩膜区域做软分配重着色，并输出颜色先验图与置信图。

示例：
  python t5.py \
    --img_path demo.png \
    --mask_path demo_mask.png \
    --lut_npz pigment_lut33.npz \
    --n_colors 64 \
    --output_dir results
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

try:
    from utils.color_utils import rgb_to_lab, lab_to_rgb
except ImportError as e:
    print(f"[Error] utils.color_utils not found: {e}")
    raise SystemExit(1)


def get_spatial_confidence(mask_u8: np.ndarray) -> np.ndarray:
    """Boundary pixels keep high confidence, center pixels decay gradually."""
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


def lut_trilinear(points_rgb: np.ndarray, grid: np.ndarray, lut: np.ndarray, lut_order: str = "rgb") -> np.ndarray:
    """Trilinear interpolation for a 3D LUT. points_rgb: (N,3) in [0,255]."""
    if points_rgb.ndim != 2 or points_rgb.shape[1] != 3:
        raise ValueError(f"points_rgb must be (N,3), got {points_rgb.shape}")
    if lut_order.lower() not in {"rgb", "bgr"}:
        raise ValueError("lut_order must be 'rgb' or 'bgr'")

    pts = points_rgb.astype(np.float32)
    if lut_order.lower() == "bgr":
        pts = pts[:, ::-1]

    r, g, b = pts[:, 0], pts[:, 1], pts[:, 2]
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
    return c0 * (1 - trb) + c1 * trb


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


def kmeans_palette(img_rgb: np.ndarray, mask_u8: np.ndarray, n_colors: int, sample_max: int = 60000):
    """Build palette from outside-mask pixels."""
    valid_pixels = img_rgb[mask_u8 == 0]
    if valid_pixels.size == 0:
        raise ValueError("No context pixels found (mask covers the whole image?)")

    if len(valid_pixels) > sample_max:
        idx = np.random.choice(len(valid_pixels), sample_max, replace=False)
        sample_pixels = valid_pixels[idx]
    else:
        sample_pixels = valid_pixels

    pixel_values = np.float32(sample_pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, _, centers = cv2.kmeans(pixel_values, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return centers


def soft_palette_target_lab(
    pixels_rgb: np.ndarray,
    palette_rgb: np.ndarray,
    restored_palette_lab: np.ndarray,
    palette_conf: np.ndarray,
    sigma: float = 25.0,
    topk: int = 6,
    chunk: int = 120000,
):
    """Soft-assign each pixel to the nearest palette colors in RGB space."""
    num_points = pixels_rgb.shape[0]
    num_palette = palette_rgb.shape[0]
    out_lab = np.empty((num_points, 3), dtype=np.float32)
    out_conf = np.empty((num_points,), dtype=np.float32)

    pal = palette_rgb.astype(np.float32)
    pal_lab = restored_palette_lab.astype(np.float32)
    pal_conf = palette_conf.astype(np.float32).reshape(-1)

    sig2 = max(1e-6, float(sigma) ** 2)
    kk = min(max(1, int(topk)), num_palette)

    for s in range(0, num_points, chunk):
        e = min(s + chunk, num_points)
        x = pixels_rgb[s:e].astype(np.float32)

        d2 = np.sum((x[:, None, :] - pal[None, :, :]) ** 2, axis=2)
        idx = np.argpartition(d2, kth=kk - 1, axis=1)[:, :kk]
        d2_top = np.take_along_axis(d2, idx, axis=1)

        w = np.exp(-d2_top / (2.0 * sig2)).astype(np.float32)
        w_sum = np.sum(w, axis=1, keepdims=True) + 1e-8
        w = w / w_sum

        lab_top = pal_lab[idx]
        conf_top = pal_conf[idx]

        out_lab[s:e] = np.sum(lab_top * w[:, :, None], axis=1)
        out_conf[s:e] = np.sum(conf_top * w, axis=1)

    return out_lab, out_conf


def _has_guided_filter() -> bool:
    return hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter")


def masked_edge_aware_smooth(delta: np.ndarray, mask01: np.ndarray, guide: np.ndarray, radius: int = 16, eps: float = 0.01, method: str = "guided") -> np.ndarray:
    """Smooth delta inside mask using normalized filtering."""
    if method == "none":
        return delta * mask01

    m = mask01.astype(np.float32)
    num_in = (delta * m).astype(np.float32)
    den_in = m

    if method == "guided" and _has_guided_filter():
        num = cv2.ximgproc.guidedFilter(guide=guide, src=num_in, radius=radius, eps=eps, dDepth=-1)
        den = cv2.ximgproc.guidedFilter(guide=guide, src=den_in, radius=radius, eps=eps, dDepth=-1)
    else:
        num = cv2.bilateralFilter(num_in, d=0, sigmaColor=eps * 255.0, sigmaSpace=radius)
        den = cv2.bilateralFilter(den_in, d=0, sigmaColor=eps * 255.0, sigmaSpace=radius)

    return (num / (den + 1e-6)) * m


def multiscale_normalized_smooth(delta: np.ndarray, mask01: np.ndarray, down: int = 1, sigma: float = 2.0) -> np.ndarray:
    """Multiscale normalized smoothing inside mask."""
    if down <= 0:
        return delta * mask01

    m = mask01.astype(np.float32)
    num = (delta * m).astype(np.float32)
    den = m.astype(np.float32)

    for _ in range(down):
        num = cv2.pyrDown(num)
        den = cv2.pyrDown(den)

    num = cv2.GaussianBlur(num, (0, 0), sigmaX=sigma, sigmaY=sigma)
    den = cv2.GaussianBlur(den, (0, 0), sigmaX=sigma, sigmaY=sigma)

    for _ in range(down):
        num = cv2.pyrUp(num)
        den = cv2.pyrUp(den)

    h, w = delta.shape[:2]
    num = num[:h, :w]
    den = den[:h, :w]
    return (num / (den + 1e-6)) * m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--mask_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results_priors_lut_mural_opt")
    parser.add_argument("--n_colors", type=int, default=64)
    parser.add_argument("--kmeans_samples", type=int, default=60000)
    parser.add_argument("--inpaint_radius", type=int, default=3)
    parser.add_argument("--lut_npz", type=str, default="pigment_lut33.npz")
    parser.add_argument("--lut_order", type=str, default="rgb", choices=["rgb", "bgr"])
    parser.add_argument("--use_lut", type=str, default="lab", choices=["lab", "rgb"])
    parser.add_argument("--keep_luminance", dest="keep_luminance", action="store_true")
    parser.add_argument("--no_keep_luminance", dest="keep_luminance", action="store_false")
    parser.set_defaults(keep_luminance=True)
    parser.add_argument("--soft_sigma", type=float, default=25.0)
    parser.add_argument("--soft_topk", type=int, default=6)
    parser.add_argument("--delta_smooth", type=str, default="guided", choices=["none", "guided", "bilateral"])
    parser.add_argument("--gf_radius", type=int, default=16)
    parser.add_argument("--gf_eps", type=float, default=0.01)
    parser.add_argument("--ms_down", type=int, default=1)
    parser.add_argument("--ms_sigma", type=float, default=2.0)
    parser.add_argument("--chunk", type=int, default=120000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading image: {args.img_path}")
    img_bgr = cv2.imread(args.img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {args.img_path}")

    mask_u8 = cv2.imread(args.mask_path, 0)
    if mask_u8 is None:
        raise FileNotFoundError(f"Cannot read mask: {args.mask_path}")
    _, mask_u8 = cv2.threshold(mask_u8, 127, 255, cv2.THRESH_BINARY)

    h, w = mask_u8.shape[:2]
    hole_mask01 = (mask_u8 == 255).astype(np.float32)

    print("Step 1: Structure inpainting (Telea)...")
    inpainted_bgr = cv2.inpaint(img_bgr, mask_u8, args.inpaint_radius, cv2.INPAINT_TELEA)
    cv2.imwrite(os.path.join(args.output_dir, "01_structure.png"), inpainted_bgr)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    inpainted_lab = rgb_to_lab(inpainted_rgb).astype(np.float32)

    print("Step 2: Analyzing context palette (KMeans)...")
    faded_palette_rgb = kmeans_palette(img_rgb, mask_u8, args.n_colors, sample_max=args.kmeans_samples)

    print(f"Step 3: Restoring palette via LUT ({args.lut_npz})...")
    grid, lut_lab, lut_rgb, lut_conf_default, _ = load_lut_npz(args.lut_npz)
    palette_conf = lut_trilinear(faded_palette_rgb, grid, lut_conf_default, lut_order=args.lut_order)
    palette_conf = np.clip(palette_conf.astype(np.float32).reshape(-1), 0.0, 1.0)

    if args.use_lut == "lab":
        restored_palette_lab = lut_trilinear(faded_palette_rgb, grid, lut_lab, lut_order=args.lut_order).astype(np.float32)
        if restored_palette_lab.shape[-1] != 3:
            raise ValueError(f"lut_lab interpolation output shape unexpected: {restored_palette_lab.shape}")
    else:
        restored_palette_rgb = lut_trilinear(faded_palette_rgb, grid, lut_rgb, lut_order=args.lut_order)
        restored_palette_rgb = np.clip(restored_palette_rgb, 0, 255).astype(np.uint8)
        restored_palette_lab = rgb_to_lab(restored_palette_rgb).astype(np.float32)

    print("Step 4: Recoloring masked region...")
    hole_yx = np.where(mask_u8 == 255)
    if hole_yx[0].size == 0:
        print("[Info] Empty hole region; saving the inpainted image directly.")
        color_prior_map = inpainted_rgb
        hole_conf_map = np.ones((h, w), dtype=np.float32)
    else:
        hole_rgb = inpainted_rgb[hole_yx].astype(np.float32)
        current_lab = inpainted_lab[hole_yx].astype(np.float32)

        target_lab, hole_conf = soft_palette_target_lab(
            pixels_rgb=hole_rgb,
            palette_rgb=faded_palette_rgb,
            restored_palette_lab=restored_palette_lab,
            palette_conf=palette_conf,
            sigma=args.soft_sigma,
            topk=args.soft_topk,
            chunk=args.chunk,
        )

        if args.keep_luminance:
            target_lab[:, 0] = current_lab[:, 0]

        delta_a = np.zeros((h, w), dtype=np.float32)
        delta_b = np.zeros((h, w), dtype=np.float32)
        delta_a[hole_yx] = target_lab[:, 1] - current_lab[:, 1]
        delta_b[hole_yx] = target_lab[:, 2] - current_lab[:, 2]

        L = inpainted_lab[..., 0].astype(np.float32)
        guide = (L / (L.max() + 1e-6)).astype(np.float32)

        if args.ms_down > 0:
            delta_a = multiscale_normalized_smooth(delta_a, hole_mask01, down=args.ms_down, sigma=args.ms_sigma)
            delta_b = multiscale_normalized_smooth(delta_b, hole_mask01, down=args.ms_down, sigma=args.ms_sigma)

        delta_a_s = masked_edge_aware_smooth(delta_a, hole_mask01, guide, radius=args.gf_radius, eps=args.gf_eps, method=args.delta_smooth)
        delta_b_s = masked_edge_aware_smooth(delta_b, hole_mask01, guide, radius=args.gf_radius, eps=args.gf_eps, method=args.delta_smooth)

        out_lab = inpainted_lab.copy()
        out_lab[..., 1] = out_lab[..., 1] + delta_a_s
        out_lab[..., 2] = out_lab[..., 2] + delta_b_s

        out_rgb = lab_to_rgb(out_lab.reshape(-1, 3)).reshape(h, w, 3)
        if out_rgb.dtype != np.uint8:
            out_rgb = np.clip(out_rgb, 0, 255).astype(np.uint8)

        color_prior_map = out_rgb
        hole_conf_map = np.ones((h, w), dtype=np.float32)
        hole_conf_map[hole_yx] = np.clip(hole_conf, 0.0, 1.0)

    spatial_conf = get_spatial_confidence(mask_u8)
    final_conf = np.clip(spatial_conf * hole_conf_map, 0.0, 1.0)

    out_prior = os.path.join(args.output_dir, "color_prior_lut_mural_opt.png")
    cv2.imwrite(out_prior, cv2.cvtColor(color_prior_map, cv2.COLOR_RGB2BGR))

    conf_vis = np.clip(final_conf * 255.0, 0, 255).astype(np.uint8)
    conf_heat = cv2.applyColorMap(conf_vis, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(args.output_dir, "confidence_map.png"), conf_vis)
    cv2.imwrite(os.path.join(args.output_dir, "confidence_heatmap.png"), conf_heat)

    print("Finished LUT prior generation.")
    print(f"-> Prior: {out_prior}")
    print(f"-> Conf : {os.path.join(args.output_dir, 'confidence_map.png')}")


if __name__ == "__main__":
    main()
