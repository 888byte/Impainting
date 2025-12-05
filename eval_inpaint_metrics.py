# eval_inpaint_metrics.py
"""
统一评估三种 inpainting 结果的 PSNR / SSIM / ΔE 指标，并导出 CSV。

目录约定（可用命令行参数覆盖）：
  GT 图像:       ./datasets/mural_inpaint/images
  掩膜:         ./datasets/mural_inpaint/masks
  StableDiff:   ./results/sd_mural
  LDM:          ./results/ldm_mural
  RePaint:      ./results/repaint_mural

运行示例：
  python eval_inpaint_metrics.py \
      --gt_dir ./datasets/mural_inpaint/images \
      --mask_dir ./datasets/mural_inpaint/masks \
      --method_dir sd=./results/sd_mural \
      --method_dir ldm=./results/ldm_mural \
      --method_dir repaint=./results/repaint_mural \
      --out_csv ./eval_inpaint_metrics.csv

"""

import os
import glob
import csv
import argparse
import numpy as np
from PIL import Image

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import rgb2lab


def load_image(path, expected_size=None):
    """加载 RGB 图像为 float32 数组，范围 [0,1]，形状 (H, W, 3)。"""
    img = Image.open(path).convert("RGB")
    if expected_size is not None:
        img = img.resize(expected_size, Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def load_mask(path, expected_size=None):
    """加载掩膜为 bool 数组，True 表示需修复区域。"""
    m = Image.open(path).convert("L")
    if expected_size is not None:
        m = m.resize(expected_size, Image.NEAREST)
    arr = np.asarray(m).astype(np.float32)
    mask = arr > 127.5
    return mask


def compute_bbox_from_mask(mask):
    """从 mask 中计算非零区域的 bounding box，返回 (y_min, y_max, x_min, x_max)，若 mask 全 0，返回 None。"""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    return y_min, y_max, x_min, x_max


def compute_metrics(gt_img, pred_img, mask):
    """
    计算 PSNR / SSIM / ΔE：
      - PSNR: 在掩膜区域
      - SSIM: 在 mask 的 bounding box 区域
      - ΔE: 在掩膜区域 (CIE76)

    gt_img, pred_img: (H, W, 3), float32, [0,1]
    mask: (H, W), bool
    """
    assert gt_img.shape == pred_img.shape, "GT 与预测图尺寸不一致"

    # 若 mask 全 0，则全图计算 PSNR / SSIM / ΔE
    if not mask.any():
        roi_mask = np.ones(mask.shape, dtype=bool)
    else:
        roi_mask = mask

    # --- PSNR（掩膜/ROI 区域）---
    diff = gt_img - pred_img
    mse_roi = np.mean((diff[roi_mask]) ** 2)
    if mse_roi == 0:
        psnr_roi = float("inf")
    else:
        psnr_roi = 10 * np.log10(1.0 / mse_roi)

    # --- SSIM（在 mask 的 bounding box 上计算）---
    bbox = compute_bbox_from_mask(roi_mask)
    if bbox is None:
        # 没有 ROI，就对整图算
        y_min, y_max, x_min, x_max = 0, gt_img.shape[0] - 1, 0, gt_img.shape[1] - 1
    else:
        y_min, y_max, x_min, x_max = bbox

    gt_crop = gt_img[y_min:y_max+1, x_min:x_max+1, :]
    pr_crop = pred_img[y_min:y_max+1, x_min:x_max+1, :]

    # 转灰度用于 SSIM
    gt_gray = 0.299 * gt_crop[..., 0] + 0.587 * gt_crop[..., 1] + 0.114 * gt_crop[..., 2]
    pr_gray = 0.299 * pr_crop[..., 0] + 0.587 * pr_crop[..., 1] + 0.114 * pr_crop[..., 2]

    try:
        ssim_roi = structural_similarity(
            gt_gray,
            pr_gray,
            data_range=1.0,
        )
    except Exception as e:
        print("SSIM 计算失败，返回 NaN：", e)
        ssim_roi = float("nan")

    # --- ΔE (Lab 色差，掩膜区域) ---
    # 转 Lab：范围大致 L in [0,100], a,b in [-128,127]
    gt_lab = rgb2lab(gt_img)
    pr_lab = rgb2lab(pred_img)

    delta = gt_lab - pr_lab
    delta_e = np.sqrt(np.sum(delta ** 2, axis=2))  # CIE76

    delta_e_roi = delta_e[roi_mask]
    if delta_e_roi.size == 0:
        delta_e_mean = float("nan")
        delta_e_std = float("nan")
    else:
        delta_e_mean = float(delta_e_roi.mean())
        delta_e_std = float(delta_e_roi.std())

    return psnr_roi, ssim_roi, delta_e_mean, delta_e_std


def parse_method_dirs(method_dir_args):
    """
    将命令行的 --method_dir sd=./results/sd_mural
    解析成 { 'sd': './results/sd_mural', ... }
    """
    method_dirs = {}
    for s in method_dir_args:
        if "=" not in s:
            raise ValueError(f"--method_dir 参数格式错误，应为 name=path，收到：{s}")
        name, path = s.split("=", 1)
        method_dirs[name.strip()] = path.strip()
    return method_dirs


def main():
    parser = argparse.ArgumentParser(description="评估 inpainting PSNR/SSIM/ΔE 指标并导出 CSV。")
    parser.add_argument("--gt_dir", type=str, default="./datasets/mural_inpaint/images",
                        help="GT 图像目录（原始完整图像）")
    parser.add_argument("--mask_dir", type=str, default="./datasets/mural_inpaint/masks",
                        help="掩膜目录（白=修复区域，黑=保持）")
    parser.add_argument("--method_dir", type=str, action="append", required=True,
                        help="方法名=结果目录，例如：sd=./results/sd_mural，可重复多次")
    parser.add_argument("--out_csv", type=str, default="./eval_inpaint_metrics.csv",
                        help="输出 CSV 文件路径")
    parser.add_argument("--img_suffix", type=str, default=".png",
                        help="GT 图像后缀，默认 .png")
    parser.add_argument("--mask_suffix", type=str, default="_mask.png",
                        help="掩膜文件名后缀，默认 _mask.png")
    args = parser.parse_args()

    gt_dir = args.gt_dir
    mask_dir = args.mask_dir
    method_dirs = parse_method_dirs(args.method_dir)
    out_csv = args.out_csv

    print("GT 目录:", gt_dir)
    print("Mask 目录:", mask_dir)
    print("方法及结果目录:")
    for name, path in method_dirs.items():
        print(f"  {name} -> {path}")

    # 遍历 GT 图像
    gt_paths = sorted(glob.glob(os.path.join(gt_dir, f"*{args.img_suffix}")))
    print(f"发现 {len(gt_paths)} 张 GT 图像。")

    rows = []
    for gt_path in gt_paths:
        base = os.path.basename(gt_path)
        stem = base[:-len(args.img_suffix)] if args.img_suffix and base.endswith(args.img_suffix) else os.path.splitext(base)[0]

        mask_path = os.path.join(mask_dir, f"{stem}{args.mask_suffix}")
        if not os.path.exists(mask_path):
            print(f"[警告] 找不到 mask，跳过：{mask_path}")
            continue

        # 载入 GT 和 mask
        gt_img = load_image(gt_path, expected_size=None)  # 保留原尺寸
        H, W, _ = gt_img.shape
        mask = load_mask(mask_path, expected_size=(W, H))
        # 注意 PIL resize 是 (width, height)，我们 load_image 没变尺寸，所以这里转置一下顺序
        # 前面 expected_size=(W, H) 已经按宽高来

        for method_name, method_dir in method_dirs.items():
            pred_path = os.path.join(method_dir, f"{stem}.png")
            if not os.path.exists(pred_path):
                print(f"[提示] 方法 {method_name} 对 {stem} 没有结果，跳过。")
                continue

            pred_img = load_image(pred_path, expected_size=(W, H))

            # 计算指标
            psnr, ssim, de_mean, de_std = compute_metrics(gt_img, pred_img, mask)

            row = {
                "filename": stem,
                "method": method_name,
                "psnr_mask": psnr,
                "ssim_bbox": ssim,
                "deltaE_mask_mean": de_mean,
                "deltaE_mask_std": de_std,
            }
            rows.append(row)

    # 写 CSV
    fieldnames = ["filename", "method", "psnr_mask", "ssim_bbox", "deltaE_mask_mean", "deltaE_mask_std"]
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"评估完成，共写入 {len(rows)} 行结果。CSV 保存到：{out_csv}")


if __name__ == "__main__":
    main()
