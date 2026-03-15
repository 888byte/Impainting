#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
颜色对比测试图。

用途：
- 批量生成测试颜色
- 调用当前统一推理链恢复颜色
- 输出“左侧输入 / 右侧恢复”的对比图

示例：
python t1.py \
  --ckpt ckpt/lab_raman_xrd/best_model.pt \
  --cond_method auto \
  --prototype_bank data/pigment_npz/prototype_bank.npz \
  --num_samples 30 \
  --palette hsv_fps \
  --n_test_colors 144 \
  --min_lab_dist 10 \
  --output_image batch_test_144_auto.png

python t1.py \
  --ckpt ckpt/lab_raman_xrd/best_model.pt \
  --cond_method posterior_retrieval \
  --prototype_bank data/pigment_npz/prototype_bank.npz \
  --library_npz data/standard_alignment/library_embeddings.npz \
  --num_samples 30 \
  --palette hsv_fps \
  --n_test_colors 144 \
  --min_lab_dist 10 \
  --output_image batch_test_144_postret.png
"""

from __future__ import annotations

import argparse
import math
import random
import colorsys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

try:
    from inference.pipeline import (
        _confidence_or_default,
        _fuse_confidence,
        _resolve_condition,
        _stabilize_single_rgb_prediction,
        load_checkpoint as _load_ckpt,
    )
    from training.diffusion import p_sample_loop
    from utils.color_utils import LabNorm, lab_to_rgb, rgb_to_lab
except ImportError as e:
    print(f"[Error] Core modules not found: {e}")
    raise SystemExit(1)


# 预定义颜色，方便快速观察常见颜色行为。
NAMED_COLORS_DICT = {
    "Black (Line)": [0, 0, 0],
    "Dark Grey": [50, 50, 50],
    "White (Paper)": [255, 255, 255],
    "Red": [255, 0, 0],
    "Green": [0, 255, 0],
    "Blue": [0, 0, 255],
    "Cyan": [0, 255, 255],
    "Magenta": [255, 0, 255],
    "Yellow": [255, 255, 0],
    "Dark Red": [139, 0, 0],
    "Ochre": [204, 119, 34],
    "Malachite": [11, 218, 81],
    "Azurite": [0, 127, 255],
    "Skin Tone": [255, 224, 189],
    "Brown": [165, 42, 42],
}


def _clamp_u8(x: float) -> int:
    return int(0 if x < 0 else 255 if x > 255 else round(x))


def _rgb_list_unique(rgb_list, min_dist_rgb: float = 2.0):
    """按 RGB 欧氏距离做粗去重，避免候选池里出现大量几乎重复的颜色。"""
    uniq = []
    for c in rgb_list:
        keep = True
        for u in uniq:
            d = ((c[0] - u[0]) ** 2 + (c[1] - u[1]) ** 2 + (c[2] - u[2]) ** 2) ** 0.5
            if d < min_dist_rgb:
                keep = False
                break
        if keep:
            uniq.append(c)
    return uniq


def generate_candidate_palette_hsv():
    """生成覆盖较全面的候选颜色池。"""
    candidates = []

    # 灰阶，额外照顾暗色区域。
    gray_vals = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]
    for v in gray_vals:
        candidates.append([v, v, v])

    # HSV 网格：兼顾暗色、低饱和、高饱和和彩色区域。
    hues = np.linspace(0.0, 1.0, 24, endpoint=False)
    sats = [0.25, 0.5, 0.75, 1.0]
    vals = [0.15, 0.30, 0.50, 0.70, 0.90, 1.0]

    for h in hues:
        for s in sats:
            for v in vals:
                r, g, b = colorsys.hsv_to_rgb(float(h), float(s), float(v))
                candidates.append([_clamp_u8(r * 255), _clamp_u8(g * 255), _clamp_u8(b * 255)])

    return _rgb_list_unique(candidates, min_dist_rgb=1.5)


def farthest_point_sampling_lab(rgb_candidates, n_select: int, seed: int = 7, min_lab_dist: float = 10.0, force_include=None):
    """
    在候选颜色中做 Lab 空间的最远点采样，尽量让测试颜色彼此拉开。

    参数：
    - min_lab_dist: 允许的最小 Lab 间隔
    - force_include: 需要优先纳入的颜色列表
    """
    rng = random.Random(seed)

    rgb_arr = np.array(rgb_candidates, dtype=np.float32)
    lab_arr = rgb_to_lab(rgb_arr)

    selected_idx = []
    if force_include:
        for fc in force_include:
            fc_lab = rgb_to_lab(np.array([fc], dtype=np.float32))[0]
            d = np.linalg.norm(lab_arr - fc_lab[None, :], axis=1)
            selected_idx.append(int(np.argmin(d)))

    selected_idx = list(dict.fromkeys(selected_idx))

    if not selected_idx:
        selected_idx = [rng.randrange(len(rgb_candidates))]

    sel_lab = lab_arr[selected_idx]
    min_d = np.min(np.linalg.norm(lab_arr[:, None, :] - sel_lab[None, :, :], axis=2), axis=1)

    while len(selected_idx) < n_select:
        i = int(np.argmax(min_d))
        if min_d[i] < min_lab_dist:
            break
        selected_idx.append(i)
        d_new = np.linalg.norm(lab_arr - lab_arr[i][None, :], axis=1)
        min_d = np.minimum(min_d, d_new)

    return [rgb_candidates[i] for i in selected_idx[:n_select]]


def build_test_palette(args):
    """
    根据配置生成测试颜色。

    - named_only: 只使用预定义颜色
    - hsv_pool: 从 HSV 候选池直接取前 N 个
    - hsv_fps: 从 HSV 候选池做 Lab 最远点采样
    """
    names = []
    rgbs = []

    named_names = list(NAMED_COLORS_DICT.keys())
    named_rgbs = list(NAMED_COLORS_DICT.values())

    if args.palette == "named_only":
        return named_names, named_rgbs

    candidates = generate_candidate_palette_hsv()

    if args.include_named:
        rgbs = [list(x) for x in named_rgbs]
        names = list(named_names)

    remain = max(0, args.n_test_colors - len(rgbs))
    if remain <= 0:
        return names[:args.n_test_colors], rgbs[:args.n_test_colors]

    if args.palette == "hsv_pool":
        for i, c in enumerate(candidates[:remain]):
            rgbs.append(c)
            names.append(f"HSV_pool_{i}")
        return names, rgbs

    if args.palette == "hsv_fps":
        force = named_rgbs if args.include_named else None
        picked = farthest_point_sampling_lab(
            rgb_candidates=candidates,
            n_select=remain,
            seed=args.seed,
            min_lab_dist=args.min_lab_dist,
            force_include=force,
        )
        for i, c in enumerate(picked):
            rgbs.append(c)
            names.append(f"Diverse_{i}")
        return names, rgbs

    raise ValueError(f"Unknown palette: {args.palette}")


@torch.no_grad()
def batch_inference_with_per_sample_confidence(rgb_list, bundle, args, device, lab_norm):
    """批量推理测试颜色，并输出每个颜色的恢复结果与置信度。"""
    denoiser = bundle["denoiser"]
    schedule = bundle["schedule"]
    batch_size = len(rgb_list)

    rgb_arr = np.array(rgb_list, dtype=np.float32)
    lab_arr = rgb_to_lab(rgb_arr)

    x_curr = lab_norm.normalize(lab_arr).astype(np.float32)
    x_curr_t = torch.from_numpy(x_curr).to(device)

    x0 = np.stack([lab_arr, lab_arr], axis=1)
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

    print(f"Running diffusion for batch of {batch_size} colors (samples={args.num_samples})...")
    samples = []
    for i in range(int(args.num_samples)):
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Sample {i + 1}/{args.num_samples}... (approx {(i + 1) / args.num_samples * 100:.1f}%)")
        x_s = p_sample_loop(denoiser, schedule, x_obs=x0_t * mask_t, obs_mask=mask_t, cond=cond)
        samples.append(x_s[:, 0, :].detach().cpu().numpy())

    arr = np.stack(samples, axis=0)
    mean_norm = np.mean(arr, axis=0)
    std_norm = np.std(arr, axis=0)

    pred_lab_batch = lab_norm.denormalize(mean_norm)
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
            pred_lab_batch[i], eff_conf = _stabilize_single_rgb_prediction(
                lab_arr[i],
                pred_lab_batch[i],
                base_conf,
                infer_cfg,
            )
        else:
            eff_conf = base_conf
        confidences.append(float(eff_conf))

    pred_rgb_batch = lab_to_rgb(pred_lab_batch)
    return pred_rgb_batch, confidences


def plot_results(names, inputs, outputs, confidences, output_file, max_cols: int = 6):
    """绘制输入/输出对比图。"""
    n = len(names)
    cols = min(max_cols, max(3, int(round(math.sqrt(n)))))
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 2.8))
    fig.suptitle("Pigment Restoration Batch Test\n(Left: Input/Faded -> Right: Output/Restored)", fontsize=16)

    axes = np.array(axes).reshape(-1) if isinstance(axes, np.ndarray) else [axes]

    for i in range(n):
        ax = axes[i]
        name = names[i]
        rgb_in = inputs[i]
        rgb_out = outputs[i]
        conf_value = confidences[i]

        ax.add_patch(Rectangle((0, 0), 1, 1, color=np.array(rgb_in) / 255.0))
        ax.add_patch(Rectangle((1, 0), 1, 1, color=np.array(rgb_out) / 255.0))
        ax.plot([1, 1], [0, 1], color="white", linewidth=2, linestyle="--")

        ax.set_xlim(0, 2)
        ax.set_ylim(0, 1)
        ax.axis("off")

        lum_in = 0.299 * rgb_in[0] + 0.587 * rgb_in[1] + 0.114 * rgb_in[2]
        c_in = "white" if lum_in < 128 else "black"
        lum_out = 0.299 * rgb_out[0] + 0.587 * rgb_out[1] + 0.114 * rgb_out[2]
        c_out = "white" if lum_out < 128 else "black"

        ax.text(0.5, 0.5, f"IN\n{rgb_in}", ha="center", va="center", color=c_in, fontweight="bold", fontsize=9)
        ax.text(1.5, 0.5, f"OUT\n{list(map(int, rgb_out))}\nConf: {conf_value:.3f}", ha="center", va="center", color=c_out, fontweight="bold", fontsize=9)
        ax.set_title(name, fontsize=9, pad=4)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="批量可视化颜料颜色恢复效果")
    parser.add_argument("--ckpt", type=str, default="ckpt/pigment_lab_raman_xrd/best_model.pt")
    parser.add_argument("--library_npz", type=str, default="data/standard_alignment/library_embeddings.npz")
    parser.add_argument("--prototype_bank", type=str, default="")
    parser.add_argument("--cond_method", type=str, default="auto", choices=["auto", "pred", "retrieval", "posterior", "posterior_retrieval"])
    parser.add_argument("--num_samples", type=int, default=30, help="每个颜色的 diffusion 采样次数")
    parser.add_argument("--output_image", type=str, default="batch_pigment_test.png")
    parser.add_argument("--retrieval_k", type=int, default=5)
    parser.add_argument("--retrieval_temp", type=float, default=0.07)
    parser.add_argument("--palette", type=str, default="hsv_fps", choices=["named_only", "hsv_pool", "hsv_fps"])
    parser.add_argument("--n_test_colors", type=int, default=96, help="要测试的颜色总数")
    parser.add_argument("--include_named", action="store_true", default=True, help="是否包含预定义常见颜色")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min_lab_dist", type=float, default=12.0, help="候选颜色之间的最小 Lab 距离")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from {args.ckpt}...")
    bundle = _load_ckpt(args.ckpt, device, prototype_bank_path=args.prototype_bank)
    print("Model loaded successfully.")

    lab_norm = LabNorm()

    print("Generating test colors...")
    names, rgb_inputs = build_test_palette(args)
    print(f"Generated {len(rgb_inputs)} test colors (palette={args.palette}, include_named={args.include_named})")

    print("Starting batch inference...")
    print(f"Estimated time: {len(rgb_inputs) * args.num_samples * 200 / 500:.0f} seconds (approx)")
    restored_rgbs, confidences = batch_inference_with_per_sample_confidence(
        rgb_inputs,
        bundle,
        args,
        device,
        lab_norm,
    )
    print("Batch inference completed.")

    plot_results(names, rgb_inputs, restored_rgbs, confidences, args.output_image)


if __name__ == "__main__":
    main()
