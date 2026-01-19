#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_sam3_and_save.py (color-gated union)

想更容易命中相似色：--delta_e_points 18 --delta_e_gate 28
想更少串色：--min_overlap 0.25（更严格） 或 --color_gate_dilate 21（gate 别太宽）

python run_sam3_and_save.py \
  --excel "/home/lab610/Impainting/sam3/拉曼物质颜色汇总.xlsx" \
  --sheet "颜色映射表" \
  --images_dir "/home/lab610/Impainting/dataset/裁剪的图片/test1/cropped_images" \
  --out_dir "/home/lab610/Impainting/dataset/裁剪的图片/test1/color_mask" \
  --checkpoint "/home/lab610/Impainting/sam3/sam3/sam3.pt" \
  --bpe "/home/lab610/Impainting/sam3/sam3/bpe_simple_vocab_16e6.txt.gz" \
  --delta_e 18 \
  --morph_kernel 7 \
  --min_area 20 \
  --max_points_per_color 60 \
  --color_gate_dilate 31 \
  --min_overlap 0.20 \
  --multimask_output \
  --post_min_component_area 200 \
  --post_keep_topk 0 \
  --merge_close_kernel 5 \
  --save_debug_points



- Excel(颜色映射表): 颜色(中文), HEX, Lab (D65)
- Lab 近似(ΔE 欧氏)得到颜色候选区域 C，并由 C 的连通域质心生成点提示
- SAM3 点分割得到 M
- 关键：用颜色候选区域对 M 做约束（过滤+裁剪），再 union 成 merged
- 保存：
    masks/    : individual masks（可选）
    merged/   : union 后的 merged mask（单通道 0/255）
    meta/     : jsonl 记录

依赖：
  pip install opencv-python pillow numpy pandas openpyxl torch
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image

import cv2
import torch

# sam3 import（兼容不同安装方式）
try:
    from sam3 import build_sam3_image_model
except Exception:
    from sam3.model_builder import build_sam3_image_model  # type: ignore

from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ----------------------------
# 数据结构
# ----------------------------
@dataclass(frozen=True)
class ColorSpec:
    name_cn: str
    hex_rgb: str
    lab_l: float
    lab_a: float
    lab_b: float


@dataclass
class MaskRecord:
    kind: str  # "individual" | "merged"
    image_path: str
    hint_path: str
    color_name: str
    color_hex: str
    lab: Tuple[float, float, float]
    point_xy: Optional[Tuple[int, int]]  # merged=None
    score: Optional[float]               # merged=None
    mask_path: str


# ----------------------------
# 工具函数
# ----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def norm_hex(s: object) -> str:
    s = str(s).strip()
    if s.startswith("#"):
        s = s[1:]
    if s.lower().startswith("0x"):
        s = s[2:]
    s = s.upper()
    if len(s) != 6 or not re.fullmatch(r"[0-9A-F]{6}", s):
        raise ValueError(f"HEX 格式不对（需要 6 位 RRGGBB）：{s}")
    return s


def parse_lab_field(v: object) -> Tuple[float, float, float]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        raise ValueError("Lab 字段为空")
    if isinstance(v, (list, tuple, np.ndarray)):
        if len(v) < 3:
            raise ValueError(f"Lab 列表长度不足：{v}")
        return (float(v[0]), float(v[1]), float(v[2]))
    s = str(v).strip()
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if len(nums) < 3:
        raise ValueError(f"无法从 Lab 字符串解析出3个数：{s}")
    return (float(nums[0]), float(nums[1]), float(nums[2]))


def read_colors_from_excel(
    excel_path: Path,
    sheet: str = "颜色映射表",
    col_name_cn: str = "颜色(中文)",
    col_hex: str = "HEX",
    col_lab: str = "Lab (D65)",
) -> List[ColorSpec]:
    df = pd.read_excel(excel_path, sheet_name=sheet)
    missing = [c for c in [col_name_cn, col_hex, col_lab] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Excel sheet '{sheet}' 缺少列：{missing}\n当前列：{list(df.columns)}"
        )

    colors: List[ColorSpec] = []
    for _, r in df.iterrows():
        name_cn = str(r[col_name_cn]).strip()
        if not name_cn or name_cn.lower() == "nan":
            continue
        try:
            hx = norm_hex(r[col_hex])
            L, a, b = parse_lab_field(r[col_lab])
        except Exception as e:
            print(f"[WARN] 跳过颜色行（{name_cn}）：{e}")
            continue

        colors.append(ColorSpec(name_cn=name_cn, hex_rgb=hx, lab_l=L, lab_a=a, lab_b=b))

    if not colors:
        raise ValueError(f"没有从 {excel_path} / {sheet} 读到任何有效颜色记录")
    return colors


def iter_images(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(root)
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    files.sort()
    return files


def lab_to_opencv(lab: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """CIELAB -> OpenCV Lab（L 0..255, a/b 偏移128）"""
    L, a, b = lab
    return (float(L) * 255.0 / 100.0, float(a) + 128.0, float(b) + 128.0)


def rgb_to_lab_opencv(rgb_u8: np.ndarray) -> np.ndarray:
    """RGB uint8 -> OpenCV Lab uint8"""
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)


def build_color_candidate_mask(
    lab_hint_f32: np.ndarray,
    target_lab_cielab: Tuple[float, float, float],
    delta_e: float,
    morph_kernel: int,
) -> np.ndarray:
    """
    返回候选区域 C：uint8 0/1
    - dist(L2) <= delta_e
    - 可选开闭运算去噪
    """
    tL, ta, tb = lab_to_opencv(target_lab_cielab)
    target = np.array([tL, ta, tb], dtype=np.float32)[None, None, :]
    dist = np.linalg.norm(lab_hint_f32 - target, axis=2)
    C = (dist <= float(delta_e)).astype(np.uint8)

    if morph_kernel and morph_kernel >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        C = cv2.morphologyEx(C, cv2.MORPH_OPEN, k)
        C = cv2.morphologyEx(C, cv2.MORPH_CLOSE, k)
    return C  # 0/1


def points_from_candidate_mask(C01: np.ndarray, min_area: int, max_points: int) -> List[Tuple[int, int]]:
    """对候选区域做连通域，取质心点"""
    num, _labels, stats, centroids = cv2.connectedComponentsWithStats(C01, connectivity=8)
    pts: List[Tuple[int, int, int]] = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        cx, cy = centroids[i]
        pts.append((area, int(round(cx)), int(round(cy))))
    pts.sort(key=lambda t: t[0], reverse=True)
    pts = pts[: int(max_points)]
    return [(x, y) for _, x, y in pts]


def dilate_mask01(mask01: np.ndarray, k: int) -> np.ndarray:
    """uint8 0/1 -> uint8 0/1"""
    if k <= 0:
        return mask01
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    d = cv2.dilate(mask01 * 255, kern)
    return (d > 0).astype(np.uint8)


def save_mask_u8(mask_bool: np.ndarray, out_path: Path) -> None:
    m = (mask_bool.astype(np.uint8) * 255)
    Image.fromarray(m, mode="L").save(out_path)


def make_individual_name(orig_stem: str, color_hex: str, point_idx: int, mask_idx: int) -> str:
    return f"{orig_stem}__{color_hex}__p{point_idx:03d}__m{mask_idx:02d}.png"


def make_merged_name(orig_stem: str, color_hex: str) -> str:
    return f"{orig_stem}__{color_hex}__merged.png"


def union_update(dst_union: np.ndarray, mask_bool: np.ndarray) -> None:
    np.logical_or(dst_union, mask_bool, out=dst_union)


def keep_top_components(mask_bool: np.ndarray, min_area: int, topk: int) -> np.ndarray:
    """
    对 bool mask 做连通域过滤：
    - 去掉 area < min_area
    - 保留面积最大的 topk 个（topk<=0 表示不限制）
    """
    u8 = mask_bool.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    comps = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= int(min_area):
            comps.append((area, i))
    if not comps:
        return np.zeros_like(mask_bool, dtype=bool)
    comps.sort(reverse=True)
    if topk > 0:
        keep_ids = {i for _, i in comps[:topk]}
    else:
        keep_ids = {i for _, i in comps}
    out = np.zeros_like(mask_bool, dtype=bool)
    for i in keep_ids:
        out |= (labels == i)
    return out


def maybe_close(mask_bool: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return mask_bool
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    u8 = mask_bool.astype(np.uint8) * 255
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, kern)
    return (u8 > 0)


# ----------------------------
# SAM3 封装
# ----------------------------
class Sam3Runner:
    def __init__(
        self,
        checkpoint_path: Path,
        bpe_path: Path,
        device: str,
        confidence_threshold: float,
        enable_inst_interactivity: bool = True,
        use_bfloat16_autocast: bool = True,
    ) -> None:
        self.device = device
        self.use_autocast = use_bfloat16_autocast and device.startswith("cuda")

        if device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        kwargs = dict(
            bpe_path=str(bpe_path),
            checkpoint_path=str(checkpoint_path),
            device=device,
            enable_inst_interactivity=enable_inst_interactivity,
        )
        try:
            self.model = build_sam3_image_model(**kwargs, load_from_HF=False)  # type: ignore
        except TypeError:
            self.model = build_sam3_image_model(**kwargs)  # type: ignore

        self.processor = Sam3Processor(self.model, confidence_threshold=confidence_threshold, device=device)

    @torch.no_grad()
    def set_image(self, image_pil: Image.Image):
        return self.processor.set_image(image_pil)

    @torch.no_grad()
    def segment_one_point(
        self,
        inference_state,
        x: int,
        y: int,
        multimask_output: bool,
    ) -> List[Tuple[np.ndarray, float]]:
        self.processor.reset_all_prompts(inference_state)

        point_coords = np.array([[x, y]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)

        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.use_autocast
            else contextlib.nullcontext()
        )

        with autocast_ctx:
            masks, scores, _logits = self.model.predict_inst(
                inference_state,
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=bool(multimask_output),
            )

        masks_np = np.array(masks)
        scores_np = np.array(scores, dtype=np.float32)

        out: List[Tuple[np.ndarray, float]] = []
        if masks_np.ndim == 2:
            out.append((masks_np.astype(bool), float(scores_np.item() if scores_np.size else 0.0)))
        else:
            for k in range(masks_np.shape[0]):
                sc = float(scores_np[k]) if scores_np.size else 0.0
                out.append((masks_np[k].astype(bool), sc))
        return out


# ----------------------------
# 主流程
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--excel", type=str, required=True)
    ap.add_argument("--sheet", type=str, default="颜色映射表")
    ap.add_argument("--images_dir", type=str, required=True)
    ap.add_argument("--hint_dir", type=str, default="")
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--bpe", type=str, required=True)

    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--confidence", type=float, default=0.3)

    ap.add_argument("--delta_e", type=float, default=10.0)
    ap.add_argument("--min_area", type=int, default=6)
    ap.add_argument("--max_points_per_color", type=int, default=50)
    ap.add_argument("--morph_kernel", type=int, default=3)

    ap.add_argument("--multimask_output", action="store_true")
    ap.add_argument("--save_all_multimasks", action="store_true")

    ap.add_argument("--save_debug_points", action="store_true")
    ap.add_argument("--limit_images", type=int, default=0)

    # 保存策略
    ap.add_argument("--only_merged", action="store_true", help="只保存 merged，不保存 individual")

    # ✅ 颜色约束（核心）
    ap.add_argument("--color_gate_dilate", type=int, default=7,
                    help="对颜色候选区域 C 做膨胀后再裁剪 mask，建议 5~11")
    ap.add_argument("--min_overlap", type=float, default=0.25,
                    help="过滤阈值：overlap(M, Cg)/area(M) < min_overlap 的 mask 丢弃")

    # ✅ merged 去噪
    ap.add_argument("--post_min_component_area", type=int, default=200,
                    help="merged 后去掉小连通域（像素）")
    ap.add_argument("--post_keep_topk", type=int, default=3,
                    help="merged 后只保留面积最大的 topk 个连通域（<=0 表示不限制）")
    ap.add_argument("--merge_close_kernel", type=int, default=0,
                    help="merged 最后做一次 CLOSE 填洞（0=不做，建议 5/7）")

    args = ap.parse_args()

    excel_path = Path(args.excel)
    images_dir = Path(args.images_dir)
    hint_dir = Path(args.hint_dir) if args.hint_dir else images_dir
    out_dir = Path(args.out_dir)

    checkpoint_path = Path(args.checkpoint)
    bpe_path = Path(args.bpe)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在：{checkpoint_path}")
    if not bpe_path.exists():
        raise FileNotFoundError(f"bpe 不存在：{bpe_path}")

    device = args.device.strip() or ("cuda" if torch.cuda.is_available() else "cpu")

    colors = read_colors_from_excel(excel_path, sheet=args.sheet)
    runner = Sam3Runner(
        checkpoint_path=checkpoint_path,
        bpe_path=bpe_path,
        device=device,
        confidence_threshold=float(args.confidence),
    )

    img_paths = iter_images(images_dir)
    if args.limit_images and args.limit_images > 0:
        img_paths = img_paths[: int(args.limit_images)]

    records: List[MaskRecord] = []

    for img_path in img_paths:
        rel = img_path.relative_to(images_dir)
        hint_path = hint_dir / rel
        if not hint_path.exists():
            alt = hint_dir / img_path.name
            if alt.exists():
                hint_path = alt
            else:
                print(f"[WARN] 找不到提示图：{hint_path}（跳过 {img_path.name}）")
                continue

        image_pil = Image.open(img_path).convert("RGB")
        hint_pil = Image.open(hint_path).convert("RGB")
        hint_rgb = np.array(hint_pil, dtype=np.uint8)

        # 预计算 hint 的 Lab（节省每个颜色重复转换）
        lab_hint = rgb_to_lab_opencv(hint_rgb).astype(np.float32)

        orig_stem = img_path.stem
        W, H = image_pil.size
        inference_state = runner.set_image(image_pil)

        for color in colors:
            target_lab = (color.lab_l, color.lab_a, color.lab_b)

            # 1) 颜色候选区域 C + 点
            C01 = build_color_candidate_mask(
                lab_hint_f32=lab_hint,
                target_lab_cielab=target_lab,
                delta_e=float(args.delta_e),
                morph_kernel=int(args.morph_kernel),
            )
            pts = points_from_candidate_mask(C01, min_area=int(args.min_area), max_points=int(args.max_points_per_color))
            if not pts:
                continue

            # 2) 颜色约束区域 Cg（膨胀）
            Cg01 = dilate_mask01(C01, int(args.color_gate_dilate)).astype(bool)

            color_dir = out_dir / f"{color.hex_rgb}_{color.name_cn}"
            masks_dir = color_dir / "masks"
            merged_dir = color_dir / "merged"
            meta_dir = color_dir / "meta"
            dbg_dir = color_dir / "debug"
            ensure_dir(meta_dir)
            ensure_dir(merged_dir)
            if not args.only_merged:
                ensure_dir(masks_dir)
            if args.save_debug_points:
                ensure_dir(dbg_dir)

            if args.save_debug_points:
                dbg = hint_rgb.copy()
                for (x, y) in pts:
                    cv2.circle(dbg, (x, y), 3, (255, 0, 0), -1)
                Image.fromarray(dbg).save(dbg_dir / f"{orig_stem}__{color.hex_rgb}__points.png")
                # 也把 C01 保存出来方便你看阈值是否过松/过紧
                Image.fromarray((C01 * 255).astype(np.uint8), mode="L").save(
                    dbg_dir / f"{orig_stem}__{color.hex_rgb}__C01.png"
                )

            # 3) union
            union_mask = np.zeros((H, W), dtype=bool)
            saved_individual: List[MaskRecord] = []

            for pi, (x, y) in enumerate(pts):
                per_point = runner.segment_one_point(
                    inference_state=inference_state,
                    x=x,
                    y=y,
                    multimask_output=bool(args.multimask_output),
                )
                if not per_point:
                    continue

                if args.multimask_output and args.save_all_multimasks:
                    to_use = list(enumerate(per_point))
                else:
                    best_mask, best_score = max(per_point, key=lambda t: t[1])
                    to_use = [(0, (best_mask, best_score))]

                for mi, (M, score) in to_use:
                    M = M.astype(bool)

                    area_M = float(M.sum())
                    if area_M <= 0:
                        continue

                    # ✅ 过滤：mask 必须“主要落在颜色候选附近”
                    overlap = float((M & Cg01).sum())
                    if overlap / area_M < float(args.min_overlap):
                        continue

                    # ✅ 过滤：主要落在颜色区域附近才要
                    overlap = float((M & Cg01).sum())
                    if overlap / area_M < float(args.min_overlap):
                        continue

                    # ✅ 不裁剪：保留 SAM3 的原始 mask，避免被 Cg 剪碎
                    M_gated = M

                    if not M_gated.any():
                        continue

                    union_update(union_mask, M_gated)

                    if not args.only_merged:
                        out_path = masks_dir / make_individual_name(orig_stem, color.hex_rgb, pi, mi)
                        save_mask_u8(M_gated, out_path)
                        rec = MaskRecord(
                            kind="individual",
                            image_path=str(img_path),
                            hint_path=str(hint_path),
                            color_name=color.name_cn,
                            color_hex=color.hex_rgb,
                            lab=target_lab,
                            point_xy=(x, y),
                            score=float(score),
                            mask_path=str(out_path),
                        )
                        records.append(rec)
                        saved_individual.append(rec)

            if not union_mask.any():
                continue

            # 4) merged 后处理：去碎点/保留大组件/填洞
            union_mask = keep_top_components(
                union_mask,
                min_area=int(args.post_min_component_area),
                topk=int(args.post_keep_topk),
            )
            union_mask = maybe_close(union_mask, int(args.merge_close_kernel))

            merged_path = merged_dir / make_merged_name(orig_stem, color.hex_rgb)
            save_mask_u8(union_mask, merged_path)

            merged_rec = MaskRecord(
                kind="merged",
                image_path=str(img_path),
                hint_path=str(hint_path),
                color_name=color.name_cn,
                color_hex=color.hex_rgb,
                lab=target_lab,
                point_xy=None,
                score=None,
                mask_path=str(merged_path),
            )
            records.append(merged_rec)

            meta_path = meta_dir / f"{orig_stem}__{color.hex_rgb}__meta.jsonl"
            with meta_path.open("a", encoding="utf-8") as f:
                for r in saved_individual:
                    f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
                f.write(json.dumps(asdict(merged_rec), ensure_ascii=False) + "\n")

        print(f"[OK] {img_path.name}")

    ensure_dir(out_dir)
    index_path = out_dir / "mask_index.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, ensure_ascii=False, indent=2)

    print(f"\nDone. Total records: {len(records)}")
    print(f"Index saved: {index_path}")


if __name__ == "__main__":
    main()
