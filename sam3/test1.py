#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SAM3 auto mask generator (Color seeds -> SAM3 point(inst) -> semantic label map)

Fixes:
- Some SAM3 forks have image_model.predict_inst, but inst_interactive_predictor is None unless enabled/initialized.
- This script builds the model with inst interactivity ON (best-effort across forks) and initializes predictor if needed.
- Point prompt segmentation is done via image_model.predict_inst(...) (NOT Sam3Processor.set_point_prompt).

Outputs:
- <stem>_label.png   uint16 single-channel label mask (0=bg, 1..K class_id)
- <stem>_preview.png optional RGB preview using palette RGB
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch

# Optional deps
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

# SAM3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ---------------------------
# CLI
# ---------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--images", type=str, required=True,
                   help="Input image path OR a directory containing images.")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Output directory for masks.")

    p.add_argument("--excel", type=str, default="./拉曼物质颜色汇总.xlsx",
                   help="Excel file with colors (RGB/Lab).")
    p.add_argument("--sheet", type=str, default="颜色汇总(含RGB_Lab)",
                   help="Sheet name.")
    p.add_argument("--max_classes", type=int, default=44,
                   help="Keep first N classes (44 default).")

    p.add_argument("--weights", type=str, default="./sam3/sam3.pt",
                   help="SAM3 checkpoint path (sam3.pt).")
    p.add_argument("--bpe", type=str, default="./sam3/bpe_simple_vocab_16e6.txt.gz",
                   help="BPE vocab path (bpe_simple_vocab_16e6.txt.gz).")

    p.add_argument("--device", type=str, default="cuda",
                   help="cuda / cpu")
    p.add_argument("--fp16", action="store_true",
                   help="Use autocast fp16 on CUDA.")

    # seed mining
    p.add_argument("--seed_downscale", type=int, default=4,
                   help="Downscale factor for seed mining (>=1).")
    p.add_argument("--deltaE_pos", type=float, default=10.0,
                   help="Positive Lab distance threshold for region candidates.")
    p.add_argument("--deltaE_neg", type=float, default=18.0,
                   help="Negative Lab distance threshold to sample background points.")
    p.add_argument("--min_area", type=int, default=40,
                   help="Min component area (on downscaled image).")
    p.add_argument("--max_seeds_per_class", type=int, default=6,
                   help="Max regions (seeds) per class per image.")

    # filter / merge
    p.add_argument("--min_score", type=float, default=0.20,
                   help="Ignore masks whose score < min_score.")
    p.add_argument("--save_preview", action="store_true",
                   help="Save a colorized preview PNG.")
    p.add_argument("--label_mode", type=str, default="mineral",
                   choices=["mineral", "color"],
                   help="mineral: each row is a class; color: group by 颜色(中文).")

    return p.parse_args()


# ---------------------------
# Basic utilities
# ---------------------------
def list_images(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return [p for p in sorted(path.rglob("*")) if p.suffix.lower() in exts]


def resize_rgb(rgb: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return rgb
    h, w = rgb.shape[:2]
    nh, nw = max(1, h // scale), max(1, w // scale)
    if cv2 is not None:
        return cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    return np.array(Image.fromarray(rgb).resize((nw, nh), resample=Image.BILINEAR))


def rgb_to_lab_image(rgb_uint8: np.ndarray) -> np.ndarray:
    """
    RGB uint8 [H,W,3] -> Lab float32:
      L* ~ [0,100], a*,b* ~ [-128,127]
    """
    if cv2 is not None:
        lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
        L = lab[..., 0] * (100.0 / 255.0)
        a = lab[..., 1] - 128.0
        b = lab[..., 2] - 128.0
        return np.stack([L, a, b], axis=-1).astype(np.float32)

    # fallback: scikit-image
    try:
        from skimage.color import rgb2lab  # type: ignore
        return rgb2lab(rgb_uint8.astype(np.float32) / 255.0).astype(np.float32)
    except Exception as e:
        raise RuntimeError(
            "Need opencv-python or scikit-image for RGB->Lab conversion.\n"
            "Install one:\n  pip install opencv-python\nor\n  pip install scikit-image"
        ) from e


def connected_components(binary: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    binary: bool/uint8 [H,W]
    return: (num_components_including_bg, labels_int32)
    """
    b = (binary > 0).astype(np.uint8)
    if cv2 is not None:
        num, lbl = cv2.connectedComponents(b, connectivity=8)
        return num, lbl.astype(np.int32)

    # fallback: scipy
    try:
        import scipy.ndimage as ndi  # type: ignore
        lbl, num = ndi.label(b)
        return num + 1, lbl.astype(np.int32)
    except Exception as e:
        raise RuntimeError(
            "Need opencv-python or scipy for connected components.\n"
            "Install one:\n  pip install opencv-python\nor\n  pip install scipy"
        ) from e


def sample_negative_points(
    dist_map: np.ndarray,
    comp_mask: np.ndarray,
    num_neg: int,
    deltaE_neg: float,
) -> List[Tuple[int, int]]:
    cand = (dist_map > deltaE_neg) & (~comp_mask)
    ys, xs = np.where(cand)
    if len(xs) == 0:
        return []
    idx = np.random.choice(len(xs), size=min(num_neg, len(xs)), replace=False)
    return [(int(xs[i]), int(ys[i])) for i in idx]


# ---------------------------
# Palette (Excel)
# ---------------------------
def extract_palette(excel_path: str, sheet: str, label_mode: str, max_classes: int) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=sheet)

    # Lab columns
    if {"Lab_L*", "Lab_a*", "Lab_b*"}.issubset(df.columns):
        Lcol, acol, bcol = "Lab_L*", "Lab_a*", "Lab_b*"
    elif "Lab (D65)" in df.columns:
        lab_split = df["Lab (D65)"].astype(str).str.split(",", expand=True)
        df["Lab_L*"] = pd.to_numeric(lab_split[0], errors="coerce")
        df["Lab_a*"] = pd.to_numeric(lab_split[1], errors="coerce")
        df["Lab_b*"] = pd.to_numeric(lab_split[2], errors="coerce")
        Lcol, acol, bcol = "Lab_L*", "Lab_a*", "Lab_b*"
    else:
        raise ValueError("Excel中找不到 Lab_L*/Lab_a*/Lab_b* 或 Lab (D65) 列。")

    # RGB columns
    if {"RGB_R", "RGB_G", "RGB_B"}.issubset(df.columns):
        rcol, gcol, bcol_rgb = "RGB_R", "RGB_G", "RGB_B"
    elif "RGB" in df.columns:
        rgb_split = df["RGB"].astype(str).str.split(",", expand=True)
        df["RGB_R"] = pd.to_numeric(rgb_split[0], errors="coerce")
        df["RGB_G"] = pd.to_numeric(rgb_split[1], errors="coerce")
        df["RGB_B"] = pd.to_numeric(rgb_split[2], errors="coerce")
        rcol, gcol, bcol_rgb = "RGB_R", "RGB_G", "RGB_B"
    else:
        raise ValueError("Excel中找不到 RGB_R/G/B 或 RGB 列。")

    if label_mode == "mineral":
        # 更合理的优先级：优先矿物名，不要 Sheet3/Sheet4
        label_col = None
        for c in ["物质(展示)", "RRUFF名称", "页签名", "颜色(中文)"]:
            if c in df.columns:
                label_col = c
                break
        if label_col is None:
            label_col = df.columns[0]
        df["label"] = df[label_col].astype(str)
        out = df[["label", rcol, gcol, bcol_rgb, Lcol, acol, bcol]].copy()
    else:
        if "颜色(中文)" not in df.columns:
            raise ValueError("label_mode=color 需要存在 ‘颜色(中文)’ 列。")
        g = df.groupby("颜色(中文)", dropna=False)
        out = g[[rcol, gcol, bcol_rgb, Lcol, acol, bcol]].mean().reset_index()
        out = out.rename(columns={"颜色(中文)": "label"})

    out = out.rename(columns={
        rcol: "r", gcol: "g", bcol_rgb: "b",
        Lcol: "L", acol: "a", bcol: "bb",
    })
    out = out.dropna(subset=["r", "g", "b", "L", "a", "bb"]).reset_index(drop=True)

    out["r"] = out["r"].round().astype(int).clip(0, 255)
    out["g"] = out["g"].round().astype(int).clip(0, 255)
    out["b"] = out["b"].round().astype(int).clip(0, 255)
    out[["L", "a", "bb"]] = out[["L", "a", "bb"]].astype(float)

    if max_classes and max_classes > 0:
        out = out.iloc[:max_classes].reset_index(drop=True)

    return out


# ---------------------------
# Build SAM3 with inst interactivity (THE FIX)
# ---------------------------
def _try_call(fn, *args):
    try:
        fn(*args)
        return True
    except TypeError:
        return False
    except Exception:
        return False


def build_sam3_with_inst(checkpoint_path: str, bpe_path: str, device: torch.device):
    """
    Build SAM3 image model and ensure inst_interactive_predictor is initialized.

    This function is written to be robust across different forks:
    - Try to pass an "enable inst interactivity" argument if builder supports it.
    - If still None, try common init methods on the model.
    - If still None, raise a clear error with debug info.
    """
    sig = inspect.signature(build_sam3_image_model)
    kwargs = {
        "checkpoint_path": checkpoint_path,
        "bpe_path": bpe_path,
        "load_from_HF": False,
    }

    # Some forks accept device in builder
    if "device" in sig.parameters:
        kwargs["device"] = str(device)

    # Try common arg names to enable inst interactivity
    for k in [
        "enable_inst_interactivity",
        "enable_inst_interactive",
        "enable_interactivity",
        "inst_interactivity",
        "enable_inst",
        "use_inst_interactivity",
    ]:
        if k in sig.parameters:
            kwargs[k] = True
            break

    model = build_sam3_image_model(**kwargs)

    # Move to device if possible
    try:
        model = model.to(device)
    except Exception:
        pass
    model.eval()

    if not hasattr(model, "predict_inst"):
        raise RuntimeError("模型没有 predict_inst()，说明不是 inst 交互版本。")

    # If predictor is None, try to initialize
    if getattr(model, "inst_interactive_predictor", None) is None:
        # Try explicit method names first
        candidates = [
            "init_inst_interactive_predictor",
            "build_inst_interactive_predictor",
            "setup_inst_interactive_predictor",
            "setup_inst_interactivity",
            "enable_inst_interactivity",
            "enable_interactivity",
            "_init_inst_interactive_predictor",
            "_build_inst_interactive_predictor",
        ]
        for name in candidates:
            if hasattr(model, name) and callable(getattr(model, name)):
                fn = getattr(model, name)
                if _try_call(fn) or _try_call(fn, True):
                    if getattr(model, "inst_interactive_predictor", None) is not None:
                        break

    # Last resort: scan for any callable with both "inst" and "predictor"/"interactive" keywords
    if getattr(model, "inst_interactive_predictor", None) is None:
        for name in dir(model):
            lname = name.lower()
            if ("inst" in lname) and (("predictor" in lname) or ("interactive" in lname)) and callable(getattr(model, name)):
                fn = getattr(model, name)
                if _try_call(fn) or _try_call(fn, True):
                    if getattr(model, "inst_interactive_predictor", None) is not None:
                        break

    if getattr(model, "inst_interactive_predictor", None) is None:
        # Give a super clear error
        raise RuntimeError(
            "predict_inst 存在，但 inst_interactive_predictor 仍然是 None。\n"
            "说明 build_sam3_image_model 没有正确开启 inst interactivity，或你的 fork 使用了不同的初始化函数名。\n"
            "建议：在 sam3/model/sam3_image.py 搜索 'inst_interactive_predictor' 看它如何初始化。"
        )

    return model


# ---------------------------
# predict_inst wrapper (handles different param names)
# ---------------------------
def predict_inst(image_model, state, points_xy: List[Tuple[int, int]], labels_01: List[int]):
    """
    Robustly call image_model.predict_inst across forks with different argument names.
    """
    point_coords = np.array(points_xy, dtype=np.float32)  # [N,2] (x,y)
    point_labels = np.array(labels_01, dtype=np.int32)    # [N]

    fn = image_model.predict_inst
    psig = inspect.signature(fn)
    params = psig.parameters

    kwargs = {}

    # coords name
    if "point_coords" in params:
        kwargs["point_coords"] = point_coords
    elif "points" in params:
        kwargs["points"] = point_coords
    elif "point_coordinates" in params:
        kwargs["point_coordinates"] = point_coords
    else:
        # fallback: try common
        kwargs["point_coords"] = point_coords

    # labels name
    if "point_labels" in params:
        kwargs["point_labels"] = point_labels
    elif "labels" in params:
        kwargs["labels"] = point_labels
    elif "point_label" in params:
        kwargs["point_label"] = point_labels
    else:
        kwargs["point_labels"] = point_labels

    # multimask
    if "multimask_output" in params:
        kwargs["multimask_output"] = True

    out = fn(state, **kwargs)

    # output variants: dict or tuple/list
    if isinstance(out, dict):
        masks = out.get("masks", None)
        scores = out.get("scores", None)
        logits = out.get("logits", None)
    elif isinstance(out, (tuple, list)) and len(out) >= 2:
        masks, scores = out[0], out[1]
        logits = out[2] if len(out) > 2 else None
    else:
        raise RuntimeError(f"Unexpected predict_inst output type: {type(out)}")

    if masks is None or scores is None:
        raise RuntimeError("predict_inst output missing masks/scores.")

    if torch.is_tensor(masks):
        masks = masks.detach().cpu().numpy()
    else:
        masks = np.asarray(masks)

    if torch.is_tensor(scores):
        scores = scores.detach().cpu().numpy()
    else:
        scores = np.asarray(scores)

    return masks, scores, logits


# ---------------------------
# Per-image pipeline
# ---------------------------
def run_on_one_image(
    img_path: Path,
    out_dir: Path,
    processor: Sam3Processor,
    image_model,
    palette: pd.DataFrame,
    device: torch.device,
    fp16: bool,
    seed_downscale: int,
    deltaE_pos: float,
    deltaE_neg: float,
    min_area: int,
    max_seeds_per_class: int,
    min_score: float,
    save_preview: bool,
):
    img = Image.open(img_path).convert("RGB")
    rgb = np.array(img, dtype=np.uint8)
    H, W = rgb.shape[:2]

    # Output label map: 0=background, 1..K=class id
    label_map = np.zeros((H, W), dtype=np.uint16)
    best_score_map = np.zeros((H, W), dtype=np.float32)

    with torch.inference_mode():
        state = processor.set_image(img)

    rgb_small = resize_rgb(rgb, seed_downscale)
    lab_small = rgb_to_lab_image(rgb_small)
    h, w = lab_small.shape[:2]

    use_autocast = fp16 and (device.type == "cuda")
    autocast_ctx = torch.cuda.amp.autocast if use_autocast else None

    # Reset API may or may not exist in some forks
    has_reset = hasattr(processor, "reset_all_prompts") and callable(getattr(processor, "reset_all_prompts"))

    for class_idx, row in palette.iterrows():
        cls_id = int(class_idx) + 1
        lab_ref = np.array([row["L"], row["a"], row["bb"]], dtype=np.float32).reshape(1, 1, 3)

        dist = np.linalg.norm(lab_small - lab_ref, axis=2)
        pos_mask = (dist <= deltaE_pos)

        num_cc, lbl = connected_components(pos_mask.astype(np.uint8))
        if num_cc <= 1:
            continue

        comps: List[Tuple[int, int, int, np.ndarray]] = []
        for cc_id in range(1, num_cc):
            comp = (lbl == cc_id)
            area = int(comp.sum())
            if area < min_area:
                continue
            ys, xs = np.where(comp)
            cx = int(xs.mean())
            cy = int(ys.mean())
            comps.append((cx, cy, area, comp))

        if not comps:
            continue

        comps.sort(key=lambda x: x[2], reverse=True)
        comps = comps[:max_seeds_per_class]

        for (cx_s, cy_s, _area, comp_mask) in comps:
            x = int((cx_s + 0.5) * seed_downscale)
            y = int((cy_s + 0.5) * seed_downscale)
            x = int(np.clip(x, 0, W - 1))
            y = int(np.clip(y, 0, H - 1))

            neg_s = sample_negative_points(dist, comp_mask, num_neg=3, deltaE_neg=deltaE_neg)

            points = [(x, y)]
            labels = [1]
            for (nx_s, ny_s) in neg_s:
                nx = int((nx_s + 0.5) * seed_downscale)
                ny = int((ny_s + 0.5) * seed_downscale)
                nx = int(np.clip(nx, 0, W - 1))
                ny = int(np.clip(ny, 0, H - 1))
                points.append((nx, ny))
                labels.append(0)

            with torch.inference_mode():
                if has_reset:
                    processor.reset_all_prompts(state)

                if autocast_ctx is not None:
                    with autocast_ctx():
                        masks, scores, _ = predict_inst(image_model, state, points, labels)
                else:
                    masks, scores, _ = predict_inst(image_model, state, points, labels)

            if len(scores) == 0:
                continue

            best_i = int(np.argmax(scores))
            best_score = float(scores[best_i])
            if best_score < min_score:
                continue

            m = masks[best_i]
            m_bin = (m >= 0.5) if (m.dtype != np.bool_) else m.astype(bool)

            upd = m_bin & (best_score > best_score_map)
            if np.any(upd):
                label_map[upd] = cls_id
                best_score_map[upd] = best_score

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem

    Image.fromarray(label_map, mode="I;16").save(out_dir / f"{stem}_label.png")

    if save_preview:
        pal_rgb = palette[["r", "g", "b"]].to_numpy(dtype=np.uint8)
        color = np.zeros((H, W, 3), dtype=np.uint8)
        for cid in range(1, len(palette) + 1):
            color[label_map == cid] = pal_rgb[cid - 1]
        Image.fromarray(color).save(out_dir / f"{stem}_preview.png")

    print(f"[OK] {img_path.name} -> {out_dir / (stem + '_label.png')}")


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    np.random.seed(0)

    images_path = Path(args.images)
    out_dir = Path(args.out_dir)

    palette = extract_palette(args.excel, args.sheet, args.label_mode, args.max_classes)
    print(f"Loaded palette: {len(palette)} classes from {args.excel} / {args.sheet}")
    print("First 5 labels:", palette["label"].head(5).tolist())

    dev = args.device
    if dev != "cpu" and not torch.cuda.is_available():
        dev = "cpu"
    device = torch.device(dev)
    print("Device:", device)

    # === FIXED BUILD ===
    image_model = build_sam3_with_inst(
        checkpoint_path=args.weights,
        bpe_path=args.bpe,
        device=device,
    )

    print("has predict_inst:", hasattr(image_model, "predict_inst"))
    print("inst_interactive_predictor is None:",
          getattr(image_model, "inst_interactive_predictor", None) is None)

    processor = Sam3Processor(image_model)

    imgs = list_images(images_path)
    if not imgs:
        raise FileNotFoundError(f"No images found under: {images_path}")

    for img_path in imgs:
        run_on_one_image(
            img_path=img_path,
            out_dir=out_dir,
            processor=processor,
            image_model=image_model,
            palette=palette,
            device=device,
            fp16=args.fp16,
            seed_downscale=args.seed_downscale,
            deltaE_pos=args.deltaE_pos,
            deltaE_neg=args.deltaE_neg,
            min_area=args.min_area,
            max_seeds_per_class=args.max_seeds_per_class,
            min_score=args.min_score,
            save_preview=args.save_preview,
        )


if __name__ == "__main__":
    main()
