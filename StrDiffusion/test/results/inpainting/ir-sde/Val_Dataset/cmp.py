#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_sam3_and_save.py

根据颜色表的 Lab (D65) 在 hint 图中找点 -> 用 SAM3 点分割生成 mask
并对同一张图/同一颜色的所有 mask 做 union 得到 merged。

为了解决“相似颜色没选中”，本版做了：
1) 默认 delta_e 更宽
2) 自动扩阈值找点（找不到就扩大再试）
3) gate 更宽并保存 debug（__C01 / __gate / __points）

Excel sheet 列：
- 颜色(中文)
- HEX
- Lab (D65)

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
    point_xy: Optional[Tuple[int, int]]
    score: Optional[float]
    mask_path: str
    used_delta_e_points: float
    used_delta_e_gate: float


# ----------------------------
# 基础工具
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
    sheet: str,
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


# ----------------------------
# Lab / mask 相关
# ----------------------------
def cielab_to_opencv_lab(lab: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    CIELAB (L 0..100, a/b -128..127) -> OpenCV Lab（L 0..255, a/b 偏移128）
    """
    L, a, b = lab
    return (float(L) * 255.0 / 100.0, float(a) + 128.0, float(b) + 128.0)


def rgb_to_lab_opencv(rgb_u8: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return lab


def lab_distance_map(
    lab_img_f32: np.ndarray,
    target_lab_cielab: Tuple[float, float, float],
    wL: float,
    wa: float,
    wb: float,
) -> np.ndarray:
    """
    加权 L2 距离：sqrt(((dL)/wL)^2 + ((da)/wa)^2 + ((db)/wb)^2)
    w 越大，越“宽容”对应通道差异。
    """
    tL, ta, tb = cielab_to_opencv_lab(target_lab_cielab)
    tgt = np.array([tL, ta, tb], dtype=np.float32)[None, None, :]
    d = lab_img_f32 - tgt
    dL = d[..., 0] / float(wL)
    da = d[..., 1] / float(wa)
    db = d[..., 2] / float(wb)
    return np.sqrt(dL * dL + da * da + db * db)


def build_candidate_mask01(
    lab_img_f32: np.ndarray,
    target_lab_cielab: Tuple[float, float, float],
    delta_e: float,
    morph_kernel: int,
    wL: float,
    wa: float,
    wb: float,
) -> np.ndarray:
    """
    返回 uint8 0/1 的候选掩码（用于找点）
    """
    dist = lab_distance_map(lab_img_f32, target_lab_cielab, wL=wL, wa=wa, wb=wb)
    C = (dist <= float(delta_e)).astype(np.uint8)

    if morph_kernel and morph_kernel >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        C = cv2.morphologyEx(C, cv2.MORPH_OPEN, k)
        C = cv2.morphologyEx(C, cv2.MORPH_CLOSE, k)

    return C  # 0/1


def points_from_mask01(mask01: np.ndarray, min_area: int, max_points: int) -> List[Tuple[int, int]]:
    num, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask01, connectivity=8)
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


def dilate01(mask01: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return mask01
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    d = cv2.dilate(mask01 * 255, kern)
    return (d > 0).astype(np.uint8)


def close01(mask01: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return mask01
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    u8 = (mask01.astype(np.uint8) * 255)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, kern)
    return (u8 > 0).astype(np.uint8)


def save_mask_u8(mask_bool: np.ndarray, out_path: Path) -> None:
    out = (mask_bool.astype(np.uint8) * 255)
    Image.fromarray(out, mode="L").save(out_path)


def union_update(dst_union: np.ndarray, mask_bool: np.ndarray) -> None:
    np.logical_or(dst_union, mask_bool, out=dst_union)


def keep_components(mask_bool: np.ndarray, min_area: int, topk: int) -> np.ndarray:
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


def close_bool(mask_bool: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return mask_bool
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    u8 = (mask_bool.astype(np.uint8) * 255)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, kern)
    return (u8 > 0)


# ----------------------------
# 输出命名
# ----------------------------
def make_individual_name(orig_stem: str, hex_rgb: str, point_idx: int, mask_idx: int) -> str:
    # 包含原文件名（orig_stem）
    return f"{orig_stem}__{hex_rgb}__p{point_idx:03d}__m{mask_idx:02d}.png"


def make_merged_name(orig_stem: str, hex_rgb: str) -> str:
    return f"{orig_stem}__{hex_rgb}__merged.png"


# ----------------------------
# SAM3 封装
# ----------------------------
class Sam3Runner:
    def __init__(self, checkpoint: Path, bpe: Path, device: str, confidence: float) -> None:
        self.device = device
        self.use_autocast = device.startswith("cuda")

        if device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        kwargs = dict(
            bpe_path=str(bpe),
            checkpoint_path=str(checkpoint),
            device=device,
            enable_inst_interactivity=True,
        )
        try:
            self.model = build_sam3_image_model(**kwargs, load_from_HF=False)  # type: ignore
        except TypeError:
            self.model = build_sam3_image_model(**kwargs)  # type: ignore

        self.processor = Sam3Processor(self.model, confidence_threshold=float(confidence), device=device)

    @torch.no_grad()
    def set_image(self, image_pil: Image.Image):
        return self.processor.set_image(image_pil)

    @torch.no_grad()
    def predict_one_point(
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
            masks, scores, _ = self.model.predict_inst(
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
            for i in range(masks_np.shape[0]):
                out.append((masks_np[i].astype(bool), float(scores_np[i]) if scores_np.size else 0.0))
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

    # ✅ 默认更宽：点阈值 + gate 阈值分离
    ap.add_argument("--delta_e_points", type=float, default=15.0, help="用于找点的 ΔE（更宽更容易命中相似颜色）")
    ap.add_argument("--delta_e_gate", type=float, default=22.0, help="用于 gate 的 ΔE（通常应 >= delta_e_points）")

    # ✅ 自动扩阈值：找不到点时自动扩大 delta_e_points 再试
    ap.add_argument("--auto_widen_steps", type=int, default=2, help="找点失败时自动扩阈值的次数（0=关闭）")
    ap.add_argument("--auto_widen_factor", type=float, default=1.25, help="每次扩阈值的倍率（如 1.25）")

    ap.add_argument("--morph_kernel", type=int, default=7)
    ap.add_argument("--min_area", type=int, default=20)
    ap.add_argument("--max_points_per_color", type=int, default=60)

    # ✅ gate 更宽
    ap.add_argument("--color_gate_dilate", type=int, default=31, help="gate 膨胀核大小（宽一点不容易剪碎）")
    ap.add_argument("--gate_close_kernel", type=int, default=7, help="gate 膨胀后再 CLOSE（让 gate 连贯）")

    # ✅ overlap 过滤（防跑偏），然后 soft-crop（防串色）
    ap.add_argument("--min_overlap", type=float, default=0.15, help="overlap(M, gate)/area(M) 的最小值，越大越严格")
    ap.add_argument("--soft_crop", action="store_true", help="开启软裁剪：M_final = M & gate（更少串色）")
    ap.set_defaults(soft_crop=True)

    ap.add_argument("--multimask_output", action="store_true")
    ap.add_argument("--save_all_multimasks", action="store_true")

    ap.add_argument("--only_merged", action="store_true", help="只保存 merged，不保存 individual")
    ap.add_argument("--save_debug_points", action="store_true")
    ap.add_argument("--limit_images", type=int, default=0)

    # ✅ 通道权重（让“相似颜色更容易命中”）：w 越大越宽容
    ap.add_argument("--lab_wL", type=float, default=1.2, help="L 通道宽容度（>1 更宽）")
    ap.add_argument("--lab_wa", type=float, default=1.0, help="a 通道宽容度（>1 更宽）")
    ap.add_argument("--lab_wb", type=float, default=1.0, help="b 通道宽容度（>1 更宽）")

    # merged 后处理
    ap.add_argument("--post_min_component_area", type=int, default=200)
    ap.add_argument("--post_keep_topk", type=int, default=0)
    ap.add_argument("--merge_close_kernel", type=int, default=5)

    args = ap.parse_args()

    excel_path = Path(args.excel)
    images_dir = Path(args.images_dir)
    hint_dir = Path(args.hint_dir) if args.hint_dir else images_dir
    out_dir = Path(args.out_dir)

    checkpoint = Path(args.checkpoint)
    bpe = Path(args.bpe)
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint 不存在：{checkpoint}")
    if not bpe.exists():
        raise FileNotFoundError(f"bpe 不存在：{bpe}")

    device = args.device.strip() or ("cuda" if torch.cuda.is_available() else "cpu")

    colors = read_colors_from_excel(excel_path, sheet=args.sheet)
    runner = Sam3Runner(checkpoint=checkpoint, bpe=bpe, device=device, confidence=args.confidence)

    img_paths = iter_images(images_dir)
    if args.limit_images and args.limit_images > 0:
        img_paths = img_paths[: int(args.limit_images)]

    all_records: List[MaskRecord] = []

    for img_path in img_paths:
        rel = img_path.relative_to(images_dir)
        hint_path = hint_dir / rel
        if not hint_path.exists():
            alt = hint_dir / img_path.name
            if alt.exists():
                hint_path = alt
            else:
                print(f"[WARN] 找不到 hint：{hint_path}（跳过 {img_path.name}）")
                continue

        image_pil = Image.open(img_path).convert("RGB")
        hint_rgb = np.array(Image.open(hint_path).convert("RGB"), dtype=np.uint8)

        # 预计算 hint Lab（OpenCV）
        lab_hint = rgb_to_lab_opencv(hint_rgb).astype(np.float32)

        W, H = image_pil.size
        inference_state = runner.set_image(image_pil)
        orig_stem = img_path.stem

        for color in colors:
            target_lab = (color.lab_l, color.lab_a, color.lab_b)

            # ---------- 1) 生成候选 C01 + 自动扩阈值找点 ----------
            used_de_points = float(args.delta_e_points)
            C01 = build_candidate_mask01(
                lab_img_f32=lab_hint,
                target_lab_cielab=target_lab,
                delta_e=used_de_points,
                morph_kernel=int(args.morph_kernel),
                wL=float(args.lab_wL),
                wa=float(args.lab_wa),
                wb=float(args.lab_wb),
            )
            pts = points_from_mask01(C01, min_area=int(args.min_area), max_points=int(args.max_points_per_color))

            if not pts and int(args.auto_widen_steps) > 0:
                for step in range(1, int(args.auto_widen_steps) + 1):
                    used_de_points = float(args.delta_e_points) * (float(args.auto_widen_factor) ** step)
                    C01 = build_candidate_mask01(
                        lab_img_f32=lab_hint,
                        target_lab_cielab=target_lab,
                        delta_e=used_de_points,
                        morph_kernel=int(args.morph_kernel),
                        wL=float(args.lab_wL),
                        wa=float(args.lab_wa),
                        wb=float(args.lab_wb),
                    )
                    pts = points_from_mask01(C01, min_area=int(args.min_area), max_points=int(args.max_points_per_color))
                    if pts:
                        break

            if not pts:
                continue

            # ---------- 2) 生成更宽的 gate（用 delta_e_gate） ----------
            used_de_gate = float(args.delta_e_gate)
            gate01 = build_candidate_mask01(
                lab_img_f32=lab_hint,
                target_lab_cielab=target_lab,
                delta_e=used_de_gate,
                morph_kernel=int(args.morph_kernel),
                wL=float(args.lab_wL),
                wa=float(args.lab_wa),
                wb=float(args.lab_wb),
            )
            gate01 = dilate01(gate01, int(args.color_gate_dilate))
            gate01 = close01(gate01, int(args.gate_close_kernel))
            gate_bool = gate01.astype(bool)

            # 输出目录
            color_dir = out_dir / f"{color.hex_rgb}_{color.name_cn}"
            masks_dir = color_dir / "masks"
            merged_dir = color_dir / "merged"
            meta_dir = color_dir / "meta"
            dbg_dir = color_dir / "debug"
            ensure_dir(merged_dir)
            ensure_dir(meta_dir)
            if not args.only_merged:
                ensure_dir(masks_dir)
            if args.save_debug_points:
                ensure_dir(dbg_dir)

            # debug：点 + C01 + gate
            if args.save_debug_points:
                dbg = hint_rgb.copy()
                for (x, y) in pts:
                    cv2.circle(dbg, (x, y), 3, (255, 0, 0), -1)
                Image.fromarray(dbg).save(dbg_dir / f"{orig_stem}__{color.hex_rgb}__points.png")
                Image.fromarray((C01 * 255).astype(np.uint8), mode="L").save(
                    dbg_dir / f"{orig_stem}__{color.hex_rgb}__C01.png"
                )
                Image.fromarray((gate01 * 255).astype(np.uint8), mode="L").save(
                    dbg_dir / f"{orig_stem}__{color.hex_rgb}__gate.png"
                )

            # ---------- 3) SAM3 分割 + union ----------
            union_mask = np.zeros((H, W), dtype=bool)
            saved_individual: List[MaskRecord] = []

            for pi, (x, y) in enumerate(pts):
                per_point = runner.predict_one_point(
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

                    overlap = float((M & gate_bool).sum())
                    if overlap / area_M < float(args.min_overlap):
                        continue

                    if args.soft_crop:
                        M_final = (M & gate_bool)  # 软裁剪：减少串色
                    else:
                        M_final = M  # 只过滤不裁剪：区域更大但更易串色

                    if not M_final.any():
                        continue

                    union_update(union_mask, M_final)

                    if not args.only_merged:
                        out_path = masks_dir / make_individual_name(orig_stem, color.hex_rgb, pi, mi)
                        save_mask_u8(M_final, out_path)
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
                            used_delta_e_points=float(used_de_points),
                            used_delta_e_gate=float(used_de_gate),
                        )
                        all_records.append(rec)
                        saved_individual.append(rec)

            if not union_mask.any():
                continue

            # ---------- 4) merged 后处理 ----------
            union_mask = keep_components(
                union_mask,
                min_area=int(args.post_min_component_area),
                topk=int(args.post_keep_topk),
            )
            union_mask = close_bool(union_mask, int(args.merge_close_kernel))

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
                used_delta_e_points=float(used_de_points),
                used_delta_e_gate=float(used_de_gate),
            )
            all_records.append(merged_rec)

            meta_path = meta_dir / f"{orig_stem}__{color.hex_rgb}__meta.jsonl"
            with meta_path.open("a", encoding="utf-8") as f:
                for r in saved_individual:
                    f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
                f.write(json.dumps(asdict(merged_rec), ensure_ascii=False) + "\n")

        print(f"[OK] {img_path.name}")

    ensure_dir(out_dir)
    index_path = out_dir / "mask_index.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in all_records], f, ensure_ascii=False, indent=2)

    print(f"\nDone. Total records: {len(all_records)}")
    print(f"Index saved: {index_path}")


if __name__ == "__main__":
    main()
