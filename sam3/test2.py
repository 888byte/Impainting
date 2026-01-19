#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_color_masks_v5_auto.py

面向：3~40+ 种颜色、几千张壁画图片，“不可能每个颜色单独手调参数”的场景。

核心改进：**全自动阈值 + 区域级生长（superpixel graph grow）**
- 用 SLIC 超像素把图像切成很多小区域（更符合“一个色块一片区域”）
- 每个颜色在每张图上，自动找到最像它的“最小色差 superpixel”作为 presence 判断
- seed/grow 阈值按 d_min 自动生成：seed= d_min+seed_margin，grow=d_min+grow_margin（再做上限截断）
- 在 superpixel 邻接图上 BFS 生长，避免像素级噪声/纹理碎片
- 可选 Hue Gate（由表格 RGB 自动算 HSV，不需要人为为每种颜色配置），能显著减少“红棕 vs 肤色/背景”这类误吸
- 可选 dataset 级相机偏色校准（--calibrate）：自动微调每个颜色的 Lab 中心，不需要人工逐色调

输出：
out_dir/颜色名/<image>_mask.png
out_dir/颜色名/<image>_compare.png (可选：原图|叠加|mask)
out_dir/summary.csv (每张图每色像素数)

依赖：
pip install opencv-python numpy pandas openpyxl tqdm scikit-image

推荐先小规模调试：
--include_names 黑色 红棕色  --max_images 50  --save_preview --preview_dir ...
确认 OK 后去掉 include_names / max_images 跑全量。
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import cv2

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    from skimage.segmentation import slic
except Exception:
    slic = None


# ------------------------- IO utils -------------------------

INVALID_FS_CHARS = r'[\x00-\x1f\\/:*?"<>|]'

def sanitize_name(name: str) -> str:
    s = str(name).strip()
    s = re.sub(INVALID_FS_CHARS, "_", s)
    s = re.sub(r"\s+", " ", s)
    return s if s else "unnamed"

def list_images(images_dir: str, exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")) -> List[str]:
    paths = []
    for root, _, files in os.walk(images_dir):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in exts:
                paths.append(os.path.join(root, fn))
    return sorted(paths)

def imread_unicode(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img

def imwrite_unicode(path: str, img: np.ndarray) -> None:
    ext = os.path.splitext(path)[1].lower()
    if not ext:
        ext = ".png"
        path += ext
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"Failed to encode image for writing: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    buf.tofile(path)


# ------------------------- color utils -------------------------

def cv_lab_to_cie(lab_cv: np.ndarray) -> np.ndarray:
    """OpenCV LAB -> CIE Lab(D65)"""
    lab = lab_cv.astype(np.float32)
    L = lab[..., 0] * (100.0 / 255.0)
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    return np.stack([L, a, b], axis=-1)

def rgb_to_cie_lab(rgb: Tuple[int, int, int]) -> np.ndarray:
    r, g, b = rgb
    arr = np.array([[[b, g, r]]], dtype=np.uint8)
    lab_cv = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
    return cv_lab_to_cie(lab_cv)[0, 0, :].astype(np.float32)

def rgb_to_hsv_cv(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Return OpenCV HSV: H [0,179], S,V [0,255]"""
    r, g, b = rgb
    arr = np.array([[[b, g, r]]], dtype=np.uint8)
    hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)[0, 0, :]
    return float(hsv[0]), float(hsv[1]), float(hsv[2])

def parse_triplet(value) -> Optional[Tuple[float, float, float]]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return float(value[0]), float(value[1]), float(value[2])
    s = str(value).strip()
    if not s:
        return None
    parts = re.split(r"[,\s]+", s)
    parts = [p for p in parts if p != ""]
    if len(parts) < 3:
        return None
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except Exception:
        return None

@dataclass
class TargetColor:
    name: str
    lab_cie: np.ndarray  # (3,)
    rgb: Optional[Tuple[int, int, int]] = None
    hsv: Optional[Tuple[float, float, float]] = None  # OpenCV HSV

def load_colors_from_table(
    color_table: str,
    sheet: Optional[str],
    name_col_candidates=("颜色(中文)", "颜色", "name", "Name"),
    rgb_col_candidates=("RGB", "rgb", "R,G,B", "R G B"),
    lab_col_candidates=("Lab (D65)", "Lab", "lab", "L*a*b*", "L a b"),
) -> List[TargetColor]:
    if color_table.lower().endswith(".csv"):
        df = pd.read_csv(color_table)
    else:
        df = pd.read_excel(color_table, sheet_name=sheet) if sheet else pd.read_excel(color_table)
    cols = list(df.columns)

    def pick_col(cands):
        for c in cands:
            if c in cols:
                return c
        for c in cols:
            for cand in cands:
                if cand.lower() in str(c).lower():
                    return c
        return None

    name_col = pick_col(name_col_candidates)
    rgb_col = pick_col(rgb_col_candidates)
    lab_col = pick_col(lab_col_candidates)

    if not name_col:
        raise ValueError(f"Cannot find name column. Existing columns: {cols}")

    colors: List[TargetColor] = []
    for _, row in df.iterrows():
        name = row.get(name_col, None)
        if name is None or (isinstance(name, float) and np.isnan(name)):
            continue
        name = sanitize_name(str(name))

        lab_cie = None
        rgb = None
        hsv = None

        if lab_col and row.get(lab_col, None) is not None:
            t = parse_triplet(row.get(lab_col))
            if t is not None:
                lab_cie = np.array(t, dtype=np.float32)

        if rgb_col and row.get(rgb_col, None) is not None:
            t = parse_triplet(row.get(rgb_col))
            if t is not None:
                r, g, b = [int(round(x)) for x in t]
                r = int(np.clip(r, 0, 255)); g = int(np.clip(g, 0, 255)); b = int(np.clip(b, 0, 255))
                rgb = (r, g, b)
                hsv = rgb_to_hsv_cv(rgb)
                if lab_cie is None:
                    lab_cie = rgb_to_cie_lab(rgb)

        if lab_cie is None:
            continue

        colors.append(TargetColor(name=name, lab_cie=lab_cie, rgb=rgb, hsv=hsv))

    if not colors:
        raise ValueError("No usable colors found in table (need at least Lab or RGB).")
    return colors


# ------------------------- superpixel helpers -------------------------

def slic_labels(img_bgr: np.ndarray, n_segments: int, compactness: float) -> np.ndarray:
    if slic is None:
        raise RuntimeError("scikit-image not installed. Please: pip install scikit-image")
    lab_cv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    labels = slic(lab_cv, n_segments=n_segments, compactness=compactness,
                  start_label=0, channel_axis=-1)
    return labels.astype(np.int32)

def sp_means_from_labels(values: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """values: (H,W,C) float32; labels: (H,W) int32 -> (means: (N,C), counts: (N,))"""
    flat = labels.reshape(-1)
    n = int(flat.max()) + 1
    counts = np.bincount(flat, minlength=n).astype(np.float32)
    counts[counts == 0] = 1.0
    means = []
    for c in range(values.shape[2]):
        v = values[..., c].reshape(-1).astype(np.float32)
        s = np.bincount(flat, weights=v, minlength=n).astype(np.float32)
        means.append(s / counts)
    return np.stack(means, axis=1), counts

def sp_adjacency(labels: np.ndarray) -> List[List[int]]:
    """Build adjacency list of superpixel graph by scanning 4-neighborhood boundaries."""
    h, w = labels.shape
    n = int(labels.max()) + 1
    adj = [set() for _ in range(n)]
    # right neighbors
    a = labels[:, :-1]
    b = labels[:, 1:]
    diff = a != b
    if np.any(diff):
        pa = a[diff].reshape(-1)
        pb = b[diff].reshape(-1)
        for u, v in zip(pa.tolist(), pb.tolist()):
            adj[u].add(v)
            adj[v].add(u)
    # down neighbors
    a = labels[:-1, :]
    b = labels[1:, :]
    diff = a != b
    if np.any(diff):
        pa = a[diff].reshape(-1)
        pb = b[diff].reshape(-1)
        for u, v in zip(pa.tolist(), pb.tolist()):
            adj[u].add(v)
            adj[v].add(u)
    return [sorted(list(s)) for s in adj]


# ------------------------- scoring / gating -------------------------

def hue_diff_cv(h1: np.ndarray, h2: float) -> np.ndarray:
    """OpenCV hue in [0,179], circular diff in [0,90]."""
    d = np.abs(h1 - h2)
    return np.minimum(d, 180.0 - d)

def weighted_deltaE76(sp_lab: np.ndarray, tgt_lab: np.ndarray, wL: float) -> np.ndarray:
    """sp_lab: (N,3), tgt_lab: (3,) -> (N,)"""
    dL = (sp_lab[:, 0] - tgt_lab[0]) * wL
    da = sp_lab[:, 1] - tgt_lab[1]
    db = sp_lab[:, 2] - tgt_lab[2]
    return np.sqrt(dL*dL + da*da + db*db).astype(np.float32)

def choose_wL(tgt_lab: np.ndarray, wL_lowchroma: float, chroma_thr: float) -> float:
    chroma = float(np.sqrt(tgt_lab[1]**2 + tgt_lab[2]**2))
    return wL_lowchroma if chroma <= chroma_thr else 1.0


# ------------------------- main segmentation -------------------------

def region_grow_on_graph(
    adj: List[List[int]],
    seeds: np.ndarray,          # bool (N,)
    allowed: np.ndarray,        # bool (N,)
    sp_lab: np.ndarray,         # (N,3)
    block_de: float
) -> np.ndarray:
    """BFS grow; also blocks crossing if neighbor differs too much from current segment mean (Lab)."""
    n = seeds.shape[0]
    sel = np.zeros(n, dtype=bool)
    q = []
    idx = np.where(seeds)[0]
    sel[idx] = True
    q.extend(idx.tolist())

    while q:
        u = q.pop()
        for v in adj[u]:
            if sel[v]:
                continue
            if not allowed[v]:
                continue
            # block crossing strong color boundary between adjacent superpixels
            du = sp_lab[u] - sp_lab[v]
            de_uv = float(np.sqrt(np.sum(du * du)))
            if de_uv > block_de:
                continue
            sel[v] = True
            q.append(v)
    return sel


def make_masks_auto(
    img_bgr: np.ndarray,
    colors: List[TargetColor],
    args
) -> Dict[str, np.ndarray]:
    # optional blur for color stability
    img_work = cv2.GaussianBlur(img_bgr, (0, 0), args.blur) if args.blur > 0 else img_bgr

    labels = slic_labels(img_work, n_segments=args.sp_n, compactness=args.sp_compact)
    adj = sp_adjacency(labels)

    lab_cie = cv_lab_to_cie(cv2.cvtColor(img_work, cv2.COLOR_BGR2LAB))
    sp_lab, sp_counts = sp_means_from_labels(lab_cie, labels)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    sp_hsv, _ = sp_means_from_labels(hsv, labels)  # H in [0,179], S,V in [0,255]
    sp_h = sp_hsv[:, 0]; sp_s = sp_hsv[:, 1]

    masks: Dict[str, np.ndarray] = {}
    N = sp_lab.shape[0]

    for c in colors:
        wL = choose_wL(c.lab_cie, args.wL_lowchroma, args.chroma_thr)
        d = weighted_deltaE76(sp_lab, c.lab_cie, wL=wL)  # (N,)
        dmin = float(np.min(d))

        # presence check: if even closest superpixel is far, consider absent
        if dmin > args.presence_de:
            masks[c.name] = np.zeros(labels.shape, np.uint8)
            continue

        seed_thr = min(args.seed_max, dmin + args.seed_margin)
        grow_thr = min(args.grow_max, dmin + args.grow_margin)

        allowed = d <= grow_thr
        seeds = d <= seed_thr

        # hue gating (auto from RGB) to reduce confusing colors
        if args.hue_gate and c.hsv is not None:
            ht, st, _ = c.hsv
            if st >= args.hue_gate_min_sat:
                hd = hue_diff_cv(sp_h, ht)
                allowed = allowed & (hd <= args.hue_tol) & (sp_s >= args.hue_min_seg_sat)
                seeds = seeds & allowed

        # lightness gate (auto): very dark colors shouldn't grow into bright areas, etc.
        if args.light_gate:
            Lt = float(c.lab_cie[0])
            if Lt <= args.dark_L:
                allowed = allowed & (sp_lab[:, 0] <= args.dark_L_allow)
                seeds = seeds & allowed
            elif Lt >= args.light_L:
                allowed = allowed & (sp_lab[:, 0] >= args.light_L_allow)
                seeds = seeds & allowed

        # avoid accidental single-superpixel "presence"
        seed_pix = float(np.sum(sp_counts[seeds])) if np.any(seeds) else 0.0
        if seed_pix < args.min_seed_pixels:
            masks[c.name] = np.zeros(labels.shape, np.uint8)
            continue

        sel = region_grow_on_graph(adj, seeds, allowed, sp_lab, block_de=args.block_de)

        mask = np.isin(labels, np.where(sel)[0]).astype(np.uint8) * 255

        # per-color auto min_area: dark colors keep thinner details
        min_area = args.min_area_dark if float(c.lab_cie[0]) <= args.dark_L_for_small else args.min_area

        if args.close_iter > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=args.close_iter)
        if args.dilate_iter > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.dilate(mask, k, iterations=args.dilate_iter)

        mask = filter_small_components(mask, min_area=min_area)
        masks[c.name] = mask

    return masks


def filter_small_components(mask_u8: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask_u8
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    out = np.zeros_like(mask_u8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def make_compare_image(img_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    mask3 = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
    overlay = img_bgr.copy()
    green = np.zeros_like(img_bgr)
    green[..., 1] = 255
    alpha = 0.45
    m = mask_u8 > 0
    overlay[m] = cv2.addWeighted(img_bgr[m], 1 - alpha, green[m], alpha, 0)
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(overlay, cnts, -1, (0, 0, 255), 1)
    return np.concatenate([img_bgr, overlay, mask3], axis=1)


# ------------------------- optional dataset calibration -------------------------

def calibrate_palette(img_paths: List[str], colors: List[TargetColor], args) -> None:
    """
    自动校准：扫描若干张图的 superpixel，收集“非常接近该颜色”的区域，
    用这些区域的均值来微调每个颜色中心，抵消整体色偏。
    不需要人为逐色调参。
    """
    if args.calib_max_images <= 0:
        return
    use_paths = img_paths[:args.calib_max_images]

    K = len(colors)
    sum_lab = np.zeros((K, 3), dtype=np.float64)
    sum_w = np.zeros((K,), dtype=np.float64)

    iterator = tqdm(use_paths, desc="Calibrate") if tqdm else use_paths
    for p in iterator:
        img = imread_unicode(p)
        img_work = cv2.GaussianBlur(img, (0, 0), args.blur) if args.blur > 0 else img

        labels = slic_labels(img_work, n_segments=args.sp_n, compactness=args.sp_compact)
        lab_cie = cv_lab_to_cie(cv2.cvtColor(img_work, cv2.COLOR_BGR2LAB))
        sp_lab, sp_counts = sp_means_from_labels(lab_cie, labels)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        sp_hsv, _ = sp_means_from_labels(hsv, labels)
        sp_h = sp_hsv[:, 0]; sp_s = sp_hsv[:, 1]

        for k, c in enumerate(colors):
            wL = choose_wL(c.lab_cie, args.wL_lowchroma, args.chroma_thr)
            d = weighted_deltaE76(sp_lab, c.lab_cie, wL=wL)
            ok = d <= args.calib_de

            if args.hue_gate and c.hsv is not None:
                ht, st, _ = c.hsv
                if st >= args.hue_gate_min_sat:
                    hd = hue_diff_cv(sp_h, ht)
                    ok = ok & (hd <= args.hue_tol) & (sp_s >= args.hue_min_seg_sat)

            if not np.any(ok):
                continue
            w = sp_counts[ok].astype(np.float64)
            sum_lab[k] += np.sum(sp_lab[ok].astype(np.float64) * w.reshape(-1, 1), axis=0)
            sum_w[k] += np.sum(w)

    for k, c in enumerate(colors):
        if sum_w[k] <= 0:
            continue
        est = (sum_lab[k] / sum_w[k]).astype(np.float32)
        # blend with prior to avoid drifting
        c.lab_cie = (args.calib_alpha * c.lab_cie + (1.0 - args.calib_alpha) * est).astype(np.float32)


# ------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--color_table", required=True)
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--exclude_names", nargs="*", default=[])
    ap.add_argument("--include_names", nargs="*", default=None)

    ap.add_argument("--max_images", type=int, default=0, help=">0 时只处理前 N 张图（测试用）。")

    # superpixel params
    ap.add_argument("--sp_n", type=int, default=2800, help="SLIC 超像素数量：2000~4500 常用")
    ap.add_argument("--sp_compact", type=float, default=8.0, help="越小越贴合边界（壁画常用 6~10）")

    # auto threshold params (global, applies to all colors)
    ap.add_argument("--presence_de", type=float, default=22.0, help="最接近的 superpixel ΔE > 此值 -> 认为该颜色不存在")
    ap.add_argument("--seed_margin", type=float, default=4.0, help="seed_thr = dmin + seed_margin")
    ap.add_argument("--grow_margin", type=float, default=16.0, help="grow_thr = dmin + grow_margin")
    ap.add_argument("--seed_max", type=float, default=16.0, help="seed_thr 上限（越小越精准）")
    ap.add_argument("--grow_max", type=float, default=40.0, help="grow_thr 上限（越大越宽松）")

    ap.add_argument("--block_de", type=float, default=18.0, help="邻接超像素之间 ΔE>block_de 则阻止跨越（防泄漏）")

    # smoothing / cleanup
    ap.add_argument("--blur", type=float, default=0.7, help="颜色计算前模糊 sigma（抑制纹理）")
    ap.add_argument("--close", dest="close_iter", type=int, default=1)
    ap.add_argument("--dilate", dest="dilate_iter", type=int, default=0)

    ap.add_argument("--min_area", type=int, default=180, help="普通颜色的最小连通域面积")
    ap.add_argument("--min_area_dark", type=int, default=25, help="深色（如黑色描边）保留更小连通域")
    ap.add_argument("--dark_L_for_small", type=float, default=30.0, help="目标颜色 Lt<=此值 -> 用 min_area_dark")

    # weighting for low-chroma colors (gray/black/white)
    ap.add_argument("--wL_lowchroma", type=float, default=1.7, help="低饱和(低chroma)颜色对L通道加权")
    ap.add_argument("--chroma_thr", type=float, default=10.0)

    # hue gate (auto from RGB, no per-color manual tuning)
    ap.add_argument("--hue_gate", action="store_true", help="启用 hue 门控（推荐，尤其红/棕等易混色）")
    ap.add_argument("--hue_tol", type=float, default=18.0, help="Hue 允许差（OpenCV H*2 ≈ 度数）")
    ap.add_argument("--hue_gate_min_sat", type=float, default=35.0, help="目标颜色 S>=此值才启用 hue gate")
    ap.add_argument("--hue_min_seg_sat", type=float, default=15.0, help="superpixel S>=此值才允许")

    # light gate (auto from target L, no per-color manual tuning)
    ap.add_argument("--light_gate", action="store_true", help="启用明度门控（对黑/白更稳）")
    ap.add_argument("--dark_L", type=float, default=22.0)
    ap.add_argument("--dark_L_allow", type=float, default=45.0, help="目标很暗时，允许的区域 L<=此值")
    ap.add_argument("--light_L", type=float, default=80.0)
    ap.add_argument("--light_L_allow", type=float, default=60.0, help="目标很亮时，允许的区域 L>=此值")

    # seed existence
    ap.add_argument("--min_seed_pixels", type=int, default=800, help="种子像素总和小于此值 -> 认为不存在，防误检")

    # output
    ap.add_argument("--keep_empty", action="store_true", help="仍输出全黑 mask（默认跳过）")
    ap.add_argument("--save_preview", action="store_true")
    ap.add_argument("--preview_dir", default=None)

    # calibration (optional)
    ap.add_argument("--calibrate", action="store_true", help="先扫一部分图片自动校准颜色中心（推荐）")
    ap.add_argument("--calib_max_images", type=int, default=300, help="校准使用前 N 张图")
    ap.add_argument("--calib_de", type=float, default=18.0, help="校准时只用 ΔE<=此值的超像素")
    ap.add_argument("--calib_alpha", type=float, default=0.5, help="与表格颜色中心的混合比例：越大越信表格")

    args = ap.parse_args()

    colors = load_colors_from_table(args.color_table, args.sheet)

    if args.exclude_names:
        excl = set(map(str, args.exclude_names))
        colors = [c for c in colors if c.name not in excl]
    if args.include_names is not None:
        incl = set(map(str, args.include_names))
        colors = [c for c in colors if c.name in incl]

    if not colors:
        raise SystemExit("No colors left after filtering include/exclude.")

    img_paths = list_images(args.images_dir)
    if not img_paths:
        raise SystemExit(f"No images found in {args.images_dir}")

    if args.max_images and args.max_images > 0:
        img_paths = img_paths[:args.max_images]

    os.makedirs(args.out_dir, exist_ok=True)
    if args.preview_dir:
        os.makedirs(args.preview_dir, exist_ok=True)

    if args.calibrate:
        calibrate_palette(img_paths, colors, args)

    summary_rows = []
    iterator = tqdm(img_paths, desc="Images") if tqdm else img_paths

    for p in iterator:
        img = imread_unicode(p)
        masks = make_masks_auto(img, colors, args)

        base = os.path.splitext(os.path.basename(p))[0]
        for cname, mask in masks.items():
            pix = int(np.count_nonzero(mask))
            summary_rows.append({"image": os.path.basename(p), "color": cname, "mask_pixels": pix})

            if (not args.keep_empty) and pix == 0:
                continue

            cdir = os.path.join(args.out_dir, cname)
            os.makedirs(cdir, exist_ok=True)

            out_mask = os.path.join(cdir, f"{base}_mask.png")
            imwrite_unicode(out_mask, mask)

            if args.save_preview:
                cmp_img = make_compare_image(img, mask)
                out_cmp = os.path.join(cdir, f"{base}_compare.png")
                imwrite_unicode(out_cmp, cmp_img)
                if args.preview_dir:
                    out_flat = os.path.join(args.preview_dir, f"{base}__{cname}.png")
                    imwrite_unicode(out_flat, cmp_img)

    try:
        pd.DataFrame(summary_rows).to_csv(os.path.join(args.out_dir, "summary.csv"),
                                         index=False, encoding="utf-8-sig")
    except Exception:
        pass

    print("Done:", args.out_dir)
    if args.preview_dir:
        print("Previews:", args.preview_dir)


if __name__ == "__main__":
    main()
