#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
mask区域内根据lut进行颜色替换
python t7.py \
  --img_path /home/610-wws/Impainting/dataset/裁剪的图片/test/cropped_images/42-0-1_bottom.jpg \
  --mask_path /home/610-wws/Impainting/dataset/裁剪的图片/test/output_masks/000098_right_mask.png \
  --lut_npz pigment_lut33.npz \
  --use_lut lab --keep_luminance \
  --delta_smooth guided --gf_radius 16 --gf_eps 0.01 \
  --ms_down 1 --ms_sigma 2.0 \
  --mask_feather 7

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


def read_mask_float(mask_path: str, invert: bool = False, feather: int = 7) -> np.ndarray:
    """
    Read mask as float in [0,1]. 1 means "apply replacement".
    feather: Gaussian feather radius (pixels). 0 => hard mask.
    """
    m = cv2.imread(mask_path, 0)
    if m is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    if invert:
        m = 255 - m

    if feather and feather > 0:
        k = 2 * feather + 1
        m = cv2.GaussianBlur(m, (k, k), sigmaX=0, sigmaY=0)

    m = (m.astype(np.float32) / 255.0)
    m = np.clip(m, 0.0, 1.0)
    return m


def smooth_delta_multiscale(delta: np.ndarray, down: int = 1, sigma: float = 2.0) -> np.ndarray:
    """Downsample -> Gaussian -> Upsample for large-area unification."""
    x = delta
    for _ in range(max(0, down)):
        x = cv2.pyrDown(x)
    x = cv2.GaussianBlur(x, (0, 0), sigmaX=sigma, sigmaY=sigma)
    for _ in range(max(0, down)):
        x = cv2.pyrUp(x)
    return x


def _filter_single_channel(field: np.ndarray, guide: np.ndarray, method: str, radius: int, eps: float) -> np.ndarray:
    """
    field: (H,W) float32
    guide: (H,W) float32
    method: 'guided' or 'bilateral'
    """
    field = field.astype(np.float32)
    guide = guide.astype(np.float32)

    if method == "guided":
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
            return cv2.ximgproc.guidedFilter(guide=guide, src=field, radius=radius, eps=eps, dDepth=-1)
        # fallback if ximgproc is not available
        return cv2.bilateralFilter(field, d=0, sigmaColor=eps * 255.0, sigmaSpace=radius)

    if method == "bilateral":
        return cv2.bilateralFilter(field, d=0, sigmaColor=eps * 255.0, sigmaSpace=radius)

    raise ValueError("method must be guided or bilateral")


def masked_smooth(field: np.ndarray, mask01: np.ndarray, guide: np.ndarray,
                  method: str, radius: int, eps: float) -> np.ndarray:
    """
    Mask-normalized smoothing:
        smooth(field*mask) / smooth(mask)
    prevents outside region from influencing inside.
    """
    num = _filter_single_channel(field * mask01, guide, method, radius, eps)
    den = _filter_single_channel(mask01, guide, method, radius, eps)
    return num / (den + 1e-6)


def apply_lut_to_lab(img_rgb: np.ndarray, grid: np.ndarray, lut_lab: np.ndarray,
                     lut_order: str, chunk: int) -> np.ndarray:
    """Map RGB -> Lab via LUT (continuous)."""
    H, W = img_rgb.shape[:2]
    flat_rgb = img_rgb.reshape(-1, 3).astype(np.float32)
    out_lab = np.empty((flat_rgb.shape[0], 3), dtype=np.float32)
    for s in range(0, flat_rgb.shape[0], chunk):
        e = min(s + chunk, flat_rgb.shape[0])
        out_lab[s:e] = lut_trilinear(flat_rgb[s:e], grid, lut_lab, lut_order=lut_order)
    return out_lab.reshape(H, W, 3).astype(np.float32)


def apply_lut_to_rgb(img_rgb: np.ndarray, grid: np.ndarray, lut_rgb: np.ndarray,
                     lut_order: str, chunk: int) -> np.ndarray:
    """Map RGB -> RGB via LUT (continuous)."""
    H, W = img_rgb.shape[:2]
    flat_rgb = img_rgb.reshape(-1, 3).astype(np.float32)
    out_rgb = np.empty_like(flat_rgb, dtype=np.float32)
    for s in range(0, flat_rgb.shape[0], chunk):
        e = min(s + chunk, flat_rgb.shape[0])
        out_rgb[s:e] = lut_trilinear(flat_rgb[s:e], grid, lut_rgb, lut_order=lut_order)
    out_rgb = np.clip(out_rgb, 0, 255).astype(np.uint8)
    return out_rgb.reshape(H, W, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--lut_npz", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results_mural_lut")

    # Optional mask => only replace mask region
    parser.add_argument("--mask_path", type=str, default=None, help="If provided: only replace mask region.")
    parser.add_argument("--mask_invert", action="store_true")
    parser.add_argument("--mask_feather", type=int, default=7, help="Feather radius (px). 0 => hard edge.")

    parser.add_argument("--lut_order", type=str, default="rgb", choices=["rgb", "bgr"])
    parser.add_argument("--use_lut", type=str, default="lab", choices=["lab", "rgb"])
    parser.add_argument("--chunk", type=int, default=400000)

    # Murals usually look better keeping luminance from original
    parser.add_argument("--keep_luminance", action="store_true")
    parser.add_argument("--no_keep_luminance", dest="keep_luminance", action="store_false")
    parser.set_defaults(keep_luminance=True)

    # Smooth delta_ab for region feel
    parser.add_argument("--delta_smooth", type=str, default="guided",
                        choices=["none", "guided", "bilateral"])
    parser.add_argument("--gf_radius", type=int, default=16)
    parser.add_argument("--gf_eps", type=float, default=0.01)

    # Multiscale pre-smooth for large painted areas
    parser.add_argument("--ms_down", type=int, default=1)
    parser.add_argument("--ms_sigma", type=float, default=2.0)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    img_bgr = cv2.imread(args.img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {args.img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)

    grid, lut_lab, lut_rgb = load_lut_npz(args.lut_npz)

    orig_lab = rgb_to_lab(img_rgb).astype(np.float32)
    L_orig = orig_lab[..., 0].astype(np.float32)
    guide = (L_orig / (L_orig.max() + 1e-6)).astype(np.float32)

    # 1) Continuous LUT mapping (no palette quantization)
    if args.use_lut == "rgb":
        mapped_rgb = apply_lut_to_rgb(img_rgb, grid, lut_rgb, args.lut_order, args.chunk)
        mapped_lab = rgb_to_lab(mapped_rgb).astype(np.float32)
    else:
        mapped_lab = apply_lut_to_lab(img_rgb, grid, lut_lab, args.lut_order, args.chunk)

    if args.keep_luminance:
        mapped_lab[..., 0] = L_orig

    # 2) delta on chroma only
    da = (mapped_lab[..., 1] - orig_lab[..., 1]).astype(np.float32)
    db = (mapped_lab[..., 2] - orig_lab[..., 2]).astype(np.float32)

    # If mask-only: we will do masked normalized smoothing and then blend
    if args.mask_path is not None:
        m = read_mask_float(args.mask_path, invert=args.mask_invert, feather=args.mask_feather)  # (H,W) in [0,1]

        # Multiscale smoothing in masked-normalized way
        if args.ms_down > 0:
            da_num = smooth_delta_multiscale(da * m, down=args.ms_down, sigma=args.ms_sigma)
            db_num = smooth_delta_multiscale(db * m, down=args.ms_down, sigma=args.ms_sigma)
            m_den  = smooth_delta_multiscale(m,      down=args.ms_down, sigma=args.ms_sigma)
            da = da_num / (m_den + 1e-6)
            db = db_num / (m_den + 1e-6)

        # Edge-aware masked smoothing (recommended)
        if args.delta_smooth != "none":
            da = masked_smooth(da, m, guide, method=args.delta_smooth, radius=args.gf_radius, eps=args.gf_eps)
            db = masked_smooth(db, m, guide, method=args.delta_smooth, radius=args.gf_radius, eps=args.gf_eps)

        # Build recolored (full) in Lab
        new_lab = orig_lab.copy()
        new_lab[..., 0] = L_orig if args.keep_luminance else mapped_lab[..., 0]
        new_lab[..., 1] = orig_lab[..., 1] + da
        new_lab[..., 2] = orig_lab[..., 2] + db

        recolor_rgb = lab_to_rgb(new_lab.reshape(-1, 3)).reshape(img_rgb.shape)
        recolor_rgb = np.clip(recolor_rgb, 0, 255).astype(np.uint8)

        # Blend only in mask region (feathered)
        m3 = m[..., None].astype(np.float32)
        out_rgb = (img_rgb.astype(np.float32) * (1.0 - m3) + recolor_rgb.astype(np.float32) * m3)
        out_rgb = np.clip(out_rgb, 0, 255).astype(np.uint8)

        out_name = "mural_lut_mask_only.png"
    else:
        # Full image mode (no mask)
        if args.ms_down > 0:
            da = smooth_delta_multiscale(da, down=args.ms_down, sigma=args.ms_sigma)
            db = smooth_delta_multiscale(db, down=args.ms_down, sigma=args.ms_sigma)

        if args.delta_smooth != "none":
            da = _filter_single_channel(da, guide, method=args.delta_smooth, radius=args.gf_radius, eps=args.gf_eps)
            db = _filter_single_channel(db, guide, method=args.delta_smooth, radius=args.gf_radius, eps=args.gf_eps)

        new_lab = orig_lab.copy()
        new_lab[..., 0] = L_orig if args.keep_luminance else mapped_lab[..., 0]
        new_lab[..., 1] = orig_lab[..., 1] + da
        new_lab[..., 2] = orig_lab[..., 2] + db

        out_rgb = lab_to_rgb(new_lab.reshape(-1, 3)).reshape(img_rgb.shape)
        out_rgb = np.clip(out_rgb, 0, 255).astype(np.uint8)

        out_name = "mural_lut_full.png"

    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    out_path = os.path.join(args.output_dir, out_name)
    cv2.imwrite(out_path, out_bgr)
    print(f"✅ Saved: {out_path}")


if __name__ == "__main__":
    main()
