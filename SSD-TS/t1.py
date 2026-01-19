#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
颜色对比图片（支持更全面的测试色覆盖 + 避免相近重复）

示例：
python t1.py \
  --ckpt ckpt/pigment_lab_raman_xrd_v2/best_model.pt \
  --cond_method pred \
  --num_samples 30 \
  --palette hsv_fps \
  --n_test_colors 96 \
  --output_image batch_test_pred_96.png


python t1.py --ckpt ckpt/pigment_lab_raman_xrd_v2/best_model.pt \
  --cond_method pred --num_samples 30 \
  --palette hsv_fps --n_test_colors 144 --min_lab_dist 10 \
  --output_image batch_test_144.png

"""
import argparse
import os
import sys
import math
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import colorsys

# 确保能导入 pigment_task
sys.path.append(os.getcwd())

try:
    from pigment_task.infer_pigment import (
        _load_ckpt,
        _predict_embeds_from_rgb,
        _build_cond_from_pred_embeds,
        _retrieval_raman_embed,
        _load_library_npz,
    )
    from pigment_task.diffusion import p_sample_loop
    from pigment_task.color_utils import LabNorm, rgb_to_lab, lab_to_rgb
except ImportError as e:
    print("【错误】无法导入 pigment_task 模块。请确保脚本在项目根目录下运行。")
    print(f"详细错误: {e}")
    sys.exit(1)

# ==============================================================================
# 1) 你原来的“命名颜色”（保留）
# ==============================================================================
NAMED_COLORS_DICT = {
    "Black (Line)":      [0, 0, 0],
    "Dark Grey":         [50, 50, 50],
    "White (Paper)":     [255, 255, 255],
    "Red":               [255, 0, 0],
    "Green":             [0, 255, 0],
    "Blue":              [0, 0, 255],
    "Cyan":              [0, 255, 255],
    "Magenta":           [255, 0, 255],
    "Yellow":            [255, 255, 0],
    "Dark Red":          [139, 0, 0],
    "Ochre":             [204, 119, 34],
    "Malachite":         [11, 218, 81],
    "Azurite":           [0, 127, 255],
    "Skin Tone":         [255, 224, 189],
    "Brown":             [165, 42, 42]
}

def _clamp_u8(x: float) -> int:
    return int(0 if x < 0 else 255 if x > 255 else round(x))

def _rgb_list_unique(rgb_list, min_dist_rgb=2.0):
    """简单 RGB 距离去重（粗过滤）"""
    uniq = []
    for c in rgb_list:
        keep = True
        for u in uniq:
            d = ((c[0]-u[0])**2 + (c[1]-u[1])**2 + (c[2]-u[2])**2) ** 0.5
            if d < min_dist_rgb:
                keep = False
                break
        if keep:
            uniq.append(c)
    return uniq

def generate_candidate_palette_hsv():
    """
    生成一个覆盖全面的候选池：
    - 灰度：从 0 到 255（包含大量 <50）
    - HSV：多 hue + 多 saturation + 多 value（包含暗色、低饱和、高饱和）
    """
    candidates = []

    # 灰度（特别保证 <50 的覆盖）
    gray_vals = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]
    for v in gray_vals:
        candidates.append([v, v, v])

    # HSV 网格（可以按需调整密度）
    hues = np.linspace(0.0, 1.0, 24, endpoint=False)          # 24 个色相
    sats = [0.25, 0.5, 0.75, 1.0]                             # 低饱和到高饱和
    vals = [0.15, 0.30, 0.50, 0.70, 0.90, 1.0]                # 很暗到很亮（0.15 会产生大量 <50）

    for h in hues:
        for s in sats:
            for v in vals:
                r, g, b = colorsys.hsv_to_rgb(float(h), float(s), float(v))
                candidates.append([_clamp_u8(r*255), _clamp_u8(g*255), _clamp_u8(b*255)])

    candidates = _rgb_list_unique(candidates, min_dist_rgb=1.5)
    return candidates

def farthest_point_sampling_lab(rgb_candidates, n_select, seed=7, min_lab_dist=10.0, force_include=None):
    """
    从候选颜色里挑 N 个彼此尽量不相近的颜色（Lab 空间 FPS）。
    - min_lab_dist：强制最小间距（太大可能挑不满）
    - force_include：优先纳入的颜色列表（例如你的 NAMED_COLORS）
    """
    rng = random.Random(seed)

    rgb_arr = np.array(rgb_candidates, dtype=np.float32)
    lab_arr = rgb_to_lab(rgb_arr)  # (M,3)

    selected_idx = []
    if force_include:
        # 把 force_include 里最接近的候选点加入（避免因离散候选池不包含而漏掉）
        for fc in force_include:
            fc_lab = rgb_to_lab(np.array([fc], dtype=np.float32))[0]
            d = np.linalg.norm(lab_arr - fc_lab[None, :], axis=1)
            selected_idx.append(int(np.argmin(d)))

    # 去重
    selected_idx = list(dict.fromkeys(selected_idx))

    if not selected_idx:
        selected_idx = [rng.randrange(len(rgb_candidates))]

    # 维护每个点到“已选集合”的最近距离
    sel_lab = lab_arr[selected_idx]
    min_d = np.min(np.linalg.norm(lab_arr[:, None, :] - sel_lab[None, :, :], axis=2), axis=1)

    # FPS 迭代
    while len(selected_idx) < n_select:
        # 按“离已选集合最远”选择
        i = int(np.argmax(min_d))
        if min_d[i] < min_lab_dist:
            # 已经无法再找到足够远的点
            break
        selected_idx.append(i)

        # 更新 min_d
        d_new = np.linalg.norm(lab_arr - lab_arr[i][None, :], axis=1)
        min_d = np.minimum(min_d, d_new)

    selected_rgb = [rgb_candidates[i] for i in selected_idx[:n_select]]
    return selected_rgb

def build_test_palette(args):
    """
    根据 args.palette 生成测试色：
    - named_only: 只用 NAMED_COLORS
    - hsv_pool: 直接用 HSV 候选池前 N 个（不保证不相近）
    - hsv_fps: HSV 候选池 + Lab FPS（推荐）
    """
    names = []
    rgbs = []

    named_names = list(NAMED_COLORS_DICT.keys())
    named_rgbs = list(NAMED_COLORS_DICT.values())

    if args.palette == "named_only":
        names = named_names
        rgbs = named_rgbs
        return names, rgbs

    candidates = generate_candidate_palette_hsv()

    if args.include_named:
        # 先把 named 放进来
        rgbs = [list(x) for x in named_rgbs]
        names = list(named_names)

    remain = max(0, args.n_test_colors - len(rgbs))
    if remain <= 0:
        return names[:args.n_test_colors], rgbs[:args.n_test_colors]

    if args.palette == "hsv_pool":
        # 直接补齐（可能有相近色）
        for i, c in enumerate(candidates[:remain]):
            rgbs.append(c)
            names.append(f"HSV_pool_{i}")
        return names, rgbs

    if args.palette == "hsv_fps":
        # FPS 选出“尽量不相近”的补充色
        force = named_rgbs if args.include_named else None
        picked = farthest_point_sampling_lab(
            rgb_candidates=candidates,
            n_select=remain,
            seed=args.seed,
            min_lab_dist=args.min_lab_dist,
            force_include=force
        )
        for i, c in enumerate(picked):
            rgbs.append(c)
            names.append(f"Diverse_{i}")
        return names, rgbs

    raise ValueError(f"Unknown palette: {args.palette}")

# ==============================================================================
# 2) 推理：批量推理 + 向量化置信度
# ==============================================================================
@torch.no_grad()
def batch_inference_with_per_sample_confidence(rgb_list, model_components, args, device, lib_raman, lab_norm):
    cfg, denoiser, conditioner, schedule, color_encoder, cond_predictor = model_components
    B = len(rgb_list)

    rgb_arr = np.array(rgb_list, dtype=np.float32)
    lab_arr = rgb_to_lab(rgb_arr)  # (B,3)

    x_curr = lab_norm.normalize(lab_arr).astype(np.float32)
    x_curr_t = torch.from_numpy(x_curr).to(device)

    x0 = np.stack([lab_arr, lab_arr], axis=1)  # (B,2,3)
    x0n = lab_norm.normalize(x0).astype(np.float32)

    mask = np.zeros((B, 2, 3), dtype=np.float32)
    mask[:, 1, :] = 1.0

    x0_t = torch.from_numpy(x0n).to(device)
    mask_t = torch.from_numpy(mask).to(device)

    # 条件生成
    if args.cond_method == "retrieval":
        if color_encoder is None or lib_raman is None:
            raise ValueError("Retrieval mode requires color_encoder and library!")

        zc = color_encoder(x_curr_t)  # (B,d)
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

    else:  # pred
        embeds_pred = _predict_embeds_from_rgb(x_curr_t, conditioner, color_encoder, cond_predictor)
        cond = _build_cond_from_pred_embeds(conditioner, embeds_pred)

    # 多次采样：每次对整个 batch 一起做
    print(f"Running diffusion for batch of {B} colors (samples={args.num_samples})...")
    samples = []
    for _ in range(int(args.num_samples)):
        x_s = p_sample_loop(denoiser, schedule, x_obs=x0_t * mask_t, obs_mask=mask_t, cond=cond)
        samples.append(x_s[:, 0, :].detach().cpu().numpy())  # (B,3) normalized

    arr = np.stack(samples, axis=0)  # (S,B,3) normalized

    mean_norm = np.mean(arr, axis=0)  # (B,3)
    std_norm = np.std(arr, axis=0)    # (B,3)

    pred_lab0_batch = lab_norm.denormalize(mean_norm)  # (B,3) denorm
    pred_rgb0_batch = lab_to_rgb(pred_lab0_batch)      # (B,3) uint8

    # 向量化置信度：每个样本一个
    std_scalar = np.linalg.norm(std_norm, axis=1)        # (B,)
    confidences = np.exp(-std_scalar).astype(float).tolist()

    return pred_rgb0_batch, confidences

# ==============================================================================
# 3) 绘图：自动适配更多格子
# ==============================================================================
def plot_results(names, inputs, outputs, confidences, output_file, max_cols=6):
    n = len(names)
    cols = min(max_cols, max(3, int(round(math.sqrt(n)))))
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 2.8))
    fig.suptitle('Pigment Restoration Batch Test\n(Left: Input/Faded -> Right: Output/Restored)', fontsize=16)

    axes = np.array(axes).reshape(-1) if isinstance(axes, np.ndarray) else [axes]

    for i in range(n):
        ax = axes[i]
        name = names[i]
        rgb_in = inputs[i]
        rgb_out = outputs[i]
        conf_value = confidences[i]

        ax.add_patch(Rectangle((0, 0), 1, 1, color=np.array(rgb_in)/255.0))
        ax.add_patch(Rectangle((1, 0), 1, 1, color=np.array(rgb_out)/255.0))
        ax.plot([1, 1], [0, 1], color='white', linewidth=2, linestyle='--')

        ax.set_xlim(0, 2)
        ax.set_ylim(0, 1)
        ax.axis('off')

        lum_in = 0.299*rgb_in[0] + 0.587*rgb_in[1] + 0.114*rgb_in[2]
        c_in = 'white' if lum_in < 128 else 'black'
        lum_out = 0.299*rgb_out[0] + 0.587*rgb_out[1] + 0.114*rgb_out[2]
        c_out = 'white' if lum_out < 128 else 'black'

        ax.text(0.5, 0.5, f"IN\n{rgb_in}", ha='center', va='center', color=c_in, fontweight='bold', fontsize=9)
        ax.text(1.5, 0.5, f"OUT\n{list(map(int, rgb_out))}\nConf: {conf_value:.3f}",
                ha='center', va='center', color=c_out, fontweight='bold', fontsize=9)
        ax.set_title(name, fontsize=9, pad=4)

    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.savefig(output_file, dpi=150)
    print(f"\n✅ 可视化图表已保存至: {output_file}")

# ==============================================================================
# main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Batch Visualize Pigment Restoration (More Colors)")
    parser.add_argument('--ckpt', type=str, default="ckpt/pigment_lab_raman_xrd_v2/best_model.pt")
    parser.add_argument('--library_npz', type=str, default='data/standard_alignment/library_embeddings.npz')
    parser.add_argument('--cond_method', type=str, default='retrieval', choices=['retrieval', 'pred'])
    parser.add_argument('--num_samples', type=int, default=30, help="Diffusion samples per color (higher=stable)")
    parser.add_argument('--output_image', type=str, default='batch_pigment_test.png')
    parser.add_argument('--retrieval_k', type=int, default=5)
    parser.add_argument('--retrieval_temp', type=float, default=0.07)

    # ✅ 新增：更全面的颜色测试
    parser.add_argument('--palette', type=str, default='hsv_fps', choices=['named_only', 'hsv_pool', 'hsv_fps'])
    parser.add_argument('--n_test_colors', type=int, default=96, help="总共测试多少种颜色")
    parser.add_argument('--include_named', action='store_true', default=True, help="是否包含原来的命名颜色")
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--min_lab_dist', type=float, default=12.0, help="避免相近颜色的最小 Lab 距离")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型
    print(f"Loading model from {args.ckpt}...")
    model_components = _load_ckpt(args.ckpt, device)

    lib_raman = None
    if args.cond_method == "retrieval":
        print(f"Loading library from {args.library_npz}...")
        try:
            lib = _load_library_npz(args.library_npz)
            key = 'raman_emb' if 'raman_emb' in lib else 'embeddings'
            if key in lib:
                lib_raman = torch.from_numpy(lib[key].astype(np.float32)).to(device)
            else:
                print("Warning: Library key not found, fallback to pred mode.")
                args.cond_method = "pred"
        except Exception as e:
            print(f"Library load failed: {e}. Fallback to pred.")
            args.cond_method = "pred"

    lab_norm = LabNorm()

    # ✅ 生成更全面且不相近的测试色
    names, rgb_inputs = build_test_palette(args)
    print(f"[INFO] Testing colors: {len(rgb_inputs)} (palette={args.palette}, include_named={args.include_named})")

    # 批量推理
    restored_rgbs, confidences = batch_inference_with_per_sample_confidence(
        rgb_inputs, model_components, args, device, lib_raman, lab_norm
    )

    # 绘图
    plot_results(names, rgb_inputs, restored_rgbs, confidences, args.output_image)

if __name__ == "__main__":
    main()
