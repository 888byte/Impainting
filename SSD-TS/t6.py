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
  python t6.py --img_path 1.png --lut_npz pigment_lut33.npz \
  --use_lut lab --keep_luminance \
  --delta_smooth guided --gf_radius 16 --gf_eps 0.01 \
  --ms_down 1 --ms_sigma 2.0
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


def guided_filter_if_available(src: np.ndarray, guide: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """
    If cv2.ximgproc.guidedFilter exists, use it for better region consistency on murals.
    Otherwise fallback to bilateral (still OK).
    src: single channel float32
    guide: single channel float32 (e.g., L)
    """
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
        # guidedFilter expects guide to be 8U/32F; src can be 8U/32F.
        return cv2.ximgproc.guidedFilter(guide=guide, src=src, radius=radius, eps=eps, dDepth=-1)
    # fallback: bilateral (not joint, but still edge-preserving-ish for chroma deltas)
    return cv2.bilateralFilter(src, d=0, sigmaColor=eps * 255.0, sigmaSpace=radius)


def smooth_delta_multiscale(delta: np.ndarray, down: int = 1, ksize: int = 0, sigma: float = 2.0) -> np.ndarray:
    """
    Multiscale smoothing: downsample -> Gaussian -> upsample.
    For murals, this helps unify large regions without killing texture.
    delta: (H,W) float32
    """
    orig_shape = delta.shape
    x = delta
    for _ in range(max(0, down)):
        x = cv2.pyrDown(x)
    x = cv2.GaussianBlur(x, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    for _ in range(max(0, down)):
        x = cv2.pyrUp(x)
    
    # Ensure the output shape matches the input shape
    if x.shape != orig_shape:
        x = cv2.resize(x, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR)
    
    return x


def apply_lut_mural(img_rgb: np.ndarray,
                    grid: np.ndarray,
                    lut_lab: np.ndarray,
                    lut_rgb: np.ndarray,
                    use_lut: str,
                    lut_order: str,
                    keep_luminance: bool,
                    chunk: int,
                    delta_smooth: str,
                    gf_radius: int,
                    gf_eps: float,
                    ms_down: int,
                    ms_sigma: float) -> np.ndarray:
    """
    Core:
    1) LUT continuous mapping
    2) Compute delta_ab
    3) Smooth delta_ab (guided if available / bilateral / multiscale)
    4) new_ab = orig_ab + smooth(delta_ab), L from orig (recommended)
    """
    H, W = img_rgb.shape[:2]
    orig_lab = rgb_to_lab(img_rgb).astype(np.float32)
    flat_rgb = img_rgb.reshape(-1, 3).astype(np.float32)

    if use_lut == "rgb":
        # RGB LUT mapping, then convert to Lab to do delta in ab (still works)
        out_rgb = np.empty_like(flat_rgb, dtype=np.float32)
        for s in range(0, flat_rgb.shape[0], chunk):
            e = min(s + chunk, flat_rgb.shape[0])
            out_rgb[s:e] = lut_trilinear(flat_rgb[s:e], grid, lut_rgb, lut_order=lut_order)
        mapped_rgb = np.clip(out_rgb, 0, 255).astype(np.uint8).reshape(H, W, 3)
        mapped_lab = rgb_to_lab(mapped_rgb).astype(np.float32)
    else:
        # Lab LUT mapping
        out_lab = np.empty((flat_rgb.shape[0], 3), dtype=np.float32)
        for s in range(0, flat_rgb.shape[0], chunk):
            e = min(s + chunk, flat_rgb.shape[0])
            out_lab[s:e] = lut_trilinear(flat_rgb[s:e], grid, lut_lab, lut_order=lut_order)
        mapped_lab = out_lab.reshape(H, W, 3).astype(np.float32)

    # If you don't keep luminance, you can still do delta smoothing, but murals usually look better keeping L.
    L_orig = orig_lab[..., 0].astype(np.float32)
    if keep_luminance:
        mapped_lab[..., 0] = L_orig

    # delta on chroma only
    da = (mapped_lab[..., 1] - orig_lab[..., 1]).astype(np.float32)
    db = (mapped_lab[..., 2] - orig_lab[..., 2]).astype(np.float32)

    # Extra multiscale smoothing to unify big painted regions
    if ms_down > 0:
        da = smooth_delta_multiscale(da, down=ms_down, sigma=ms_sigma)
        db = smooth_delta_multiscale(db, down=ms_down, sigma=ms_sigma)

    # Edge-aware smoothing (best: guided filter if available)
    guide = (L_orig / (L_orig.max() + 1e-6)).astype(np.float32)  # normalize for stability

    if delta_smooth == "none":
        da_s, db_s = da, db
    elif delta_smooth == "guided":
        da_s = guided_filter_if_available(da, guide, radius=gf_radius, eps=gf_eps)
        db_s = guided_filter_if_available(db, guide, radius=gf_radius, eps=gf_eps)
    elif delta_smooth == "bilateral":
        # bilateral on delta (simple, dependency-free)
        da_s = cv2.bilateralFilter(da, d=0, sigmaColor=gf_eps * 255.0, sigmaSpace=gf_radius)
        db_s = cv2.bilateralFilter(db, d=0, sigmaColor=gf_eps * 255.0, sigmaSpace=gf_radius)
    else:
        raise ValueError("delta_smooth must be one of: none, guided, bilateral")

    new_lab = orig_lab.copy()
    new_lab[..., 0] = L_orig if keep_luminance else mapped_lab[..., 0]
    new_lab[..., 1] = orig_lab[..., 1] + da_s
    new_lab[..., 2] = orig_lab[..., 2] + db_s

    out_rgb = lab_to_rgb(new_lab.reshape(-1, 3)).reshape(H, W, 3)
    out_rgb = np.clip(out_rgb, 0, 255).astype(np.uint8)
    return out_rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--lut_npz", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results_mural_lut")

    parser.add_argument("--lut_order", type=str, default="rgb", choices=["rgb", "bgr"])
    parser.add_argument("--use_lut", type=str, default="lab", choices=["lab", "rgb"])
    parser.add_argument("--chunk", type=int, default=400000)

    # murals usually look better keeping luminance
    parser.add_argument("--keep_luminance", action="store_true")
    parser.add_argument("--no_keep_luminance", dest="keep_luminance", action="store_false")
    parser.set_defaults(keep_luminance=True)

    # delta smoothing (region-wise feel)
    parser.add_argument("--delta_smooth", type=str, default="guided",
                        choices=["none", "guided", "bilateral"],
                        help="Smooth chroma delta (da/db). guided is best if available.")
    parser.add_argument("--gf_radius", type=int, default=16, help="Larger => bigger regions unify more.")
    parser.add_argument("--gf_eps", type=float, default=0.01, help="Smaller => stronger edge preserving.")

    # multiscale pre-smooth (helps murals unify large flat painted areas)
    parser.add_argument("--ms_down", type=int, default=1, help="0 disable; 1-2 recommended for murals.")
    parser.add_argument("--ms_sigma", type=float, default=2.0)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    img_bgr = cv2.imread(args.img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {args.img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    grid, lut_lab, lut_rgb = load_lut_npz(args.lut_npz)

    out_rgb = apply_lut_mural(
        img_rgb=img_rgb,
        grid=grid,
        lut_lab=lut_lab,
        lut_rgb=lut_rgb,
        use_lut=args.use_lut,
        lut_order=args.lut_order,
        keep_luminance=args.keep_luminance,
        chunk=args.chunk,
        delta_smooth=args.delta_smooth,
        gf_radius=args.gf_radius,
        gf_eps=args.gf_eps,
        ms_down=args.ms_down,
        ms_sigma=args.ms_sigma
    )

    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    out_path = os.path.join(args.output_dir, "mural_lut_full.png")
    cv2.imwrite(out_path, out_bgr)
    print(f"✅ Saved: {out_path}")


if __name__ == "__main__":
    main()
