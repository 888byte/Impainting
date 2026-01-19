#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""active_color_eval.py

改进(2)：主动评估/挖掘高风险颜色区域。

这个脚本做两件事之一：
A) 全局 palette 扫描（无 GT）：在 Lab 空间用 FPS(最远点采样)生成覆盖更均匀的颜色集合，
   然后用模型做一次批量推理，导出每个颜色的输出、扩散不确定性、（可选）检索置信度。
   你可以用不确定性最高的那些颜色，回到你的“颜色序列增强”管线里做重点增强。

B) 从有 GT 的 test_npz 中挖掘：计算每个样本 t0 的 DeltaE2000，把当前颜色（最后观测）
   按 HSV/Lab 分箱，输出 error 最大的 bins & 样本列表。

Usage examples:
  # A) 生成 256 个覆盖均匀的颜色并评估
  python active_color_eval.py --ckpt ckpt/.../best_model.pt --cond_method pred \
    --mode palette --n_colors 256 --num_samples 20 --out_csv palette_eval.csv

  # B) 挖掘 test_npz 上 error 最大的样本
  python active_color_eval.py --ckpt ckpt/.../best_model.pt --cond_method pred \
    --mode mine --test_npz data/pigment_npz_v2/test.npz --num_samples 10 --out_csv mine.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from pigment_task.color_utils import LabNorm, lab_to_rgb, rgb_to_lab, delta_e2000
from pigment_task.diffusion import p_sample_loop
from pigment_task.infer_pigment import (
    _load_ckpt,
    _predict_embeds_from_rgb,
    _build_cond_from_pred_embeds,
    _retrieval_raman_embed,
    _retrieval_confidence,
    _load_library_npz,
    _sample_with_confidence,
)
from pigment_task.dataset_pigment import PigmentNPZDataset


def fps_lab_palette(n: int, seed: int = 0, cand_n: int = 20000) -> np.ndarray:
    """Farthest point sampling in Lab.

    We sample candidate Lab points uniformly in a bounded box and FPS-select n points.
    Bounds are conservative to stay in-gamut for sRGB conversion (approx):
      L in [0,100], a,b in [-110,110]
    """
    rng = np.random.default_rng(seed)
    cand = np.empty((cand_n, 3), dtype=np.float32)
    cand[:, 0] = rng.uniform(0, 100, size=cand_n)
    cand[:, 1] = rng.uniform(-110, 110, size=cand_n)
    cand[:, 2] = rng.uniform(-110, 110, size=cand_n)

    # Initialize with a random point
    sel = np.empty((n, 3), dtype=np.float32)
    sel[0] = cand[rng.integers(0, cand_n)]
    # Maintain min distance to selected
    d2 = np.sum((cand - sel[0]) ** 2, axis=1)
    for i in range(1, n):
        idx = int(np.argmax(d2))
        sel[i] = cand[idx]
        d2 = np.minimum(d2, np.sum((cand - sel[i]) ** 2, axis=1))
    return sel


@torch.no_grad()
def infer_rgb_batch(
    rgbs: np.ndarray,
    model_components,
    device: torch.device,
    cond_method: str,
    library_npz: Optional[str],
    retrieval_k: int,
    retrieval_temp: float,
    num_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Infer original color for a batch of current rgbs.

    Returns:
      pred_rgb0: (B,3) uint8
      conf_diff: (B,) float
      conf_ret: (B,) float (nan if not retrieval)
    """
    cfg, denoiser, conditioner, schedule, color_encoder, cond_predictor = model_components
    B = int(rgbs.shape[0])
    lab_norm = LabNorm()

    lab = rgb_to_lab(rgbs.astype(np.float32))  # (B,3)

    # Build dummy sequence L=2: [t0 unknown, t1 observed]
    x0 = np.stack([lab, lab], axis=1).astype(np.float32)  # (B,2,3)
    x0n = lab_norm.normalize(x0).astype(np.float32)
    mask = np.zeros((B, 2, 3), dtype=np.float32)
    mask[:, 1, :] = 1.0

    x0_t = torch.from_numpy(x0n).to(device)
    mask_t = torch.from_numpy(mask).to(device)
    x_curr_t = torch.from_numpy(lab_norm.normalize(lab).astype(np.float32)).to(device)

    # condition
    lib_raman = None
    if cond_method == "retrieval":
        if not library_npz:
            raise ValueError("retrieval needs --library_npz")
        lib = _load_library_npz(library_npz)
        lib_raman = torch.from_numpy(lib["raman_emb"].astype(np.float32)).to(device)

    if conditioner.cond_dim == 0:
        cond = None
        conf_ret = np.full((B,), np.nan, dtype=np.float32)
    elif cond_method == "pred":
        if color_encoder is None or cond_predictor is None:
            raise ValueError("pred requires ckpt with color_encoder+cond_predictor")
        embeds_pred = _predict_embeds_from_rgb(x_curr_t, conditioner, color_encoder, cond_predictor)
        cond = _build_cond_from_pred_embeds(conditioner, embeds_pred)
        conf_ret = np.full((B,), np.nan, dtype=np.float32)
    elif cond_method == "retrieval":
        if color_encoder is None:
            raise ValueError("retrieval requires ckpt with color_encoder")
        zc = color_encoder(x_curr_t)
        raman_emb, w, _ = _retrieval_raman_embed(zc, lib_raman, top_k=int(retrieval_k), temp=float(retrieval_temp))
        embeds_pred: Dict[str, torch.Tensor] = {}
        if conditioner.raman_enc is not None:
            embeds_pred["raman"] = raman_emb
        if conditioner.xrd_enc is not None:
            if cond_predictor is None:
                embeds_pred["xrd"] = torch.zeros_like(raman_emb)
            else:
                tmp = cond_predictor(zc)
                embeds_pred["xrd"] = tmp["xrd_emb"]
        cond = _build_cond_from_pred_embeds(conditioner, embeds_pred)
        conf_ret = _retrieval_confidence(w).detach().cpu().numpy().astype(np.float32)
    else:
        raise ValueError("palette mode only supports pred/retrieval")

    if int(num_samples) <= 1:
        x_obs = x0_t * mask_t
        x_s = p_sample_loop(denoiser, schedule, x_obs=x_obs, obs_mask=mask_t, cond=cond)
        pred_lab0 = lab_norm.denormalize(x_s[:, 0, :].detach().cpu().numpy())
        conf_diff = np.full((B,), np.nan, dtype=np.float32)
    else:
        pred_lab0, info = _sample_with_confidence(denoiser, schedule, x0_t, mask_t, cond, num_samples=int(num_samples))
        conf_diff = np.full((B,), float(info.get("conf_diffusion", np.nan)), dtype=np.float32)

    pred_rgb0 = lab_to_rgb(pred_lab0)
    return pred_rgb0, conf_diff, conf_ret


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--cond_method", type=str, default="pred", choices=["pred", "retrieval"])
    ap.add_argument("--library_npz", type=str, default="")
    ap.add_argument("--retrieval_k", type=int, default=5)
    ap.add_argument("--retrieval_temp", type=float, default=0.07)
    ap.add_argument("--num_samples", type=int, default=20)

    ap.add_argument("--mode", type=str, default="palette", choices=["palette", "mine"])
    ap.add_argument("--n_colors", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--test_npz", type=str, default="")
    ap.add_argument("--max_rows", type=int, default=20000)

    ap.add_argument("--out_csv", type=str, required=True)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_components = _load_ckpt(args.ckpt, device)

    if args.mode == "palette":
        lab = fps_lab_palette(int(args.n_colors), seed=int(args.seed))
        rgbs = lab_to_rgb(lab)
        pred_rgb0, conf_diff, conf_ret = infer_rgb_batch(
            rgbs,
            model_components,
            device=device,
            cond_method=str(args.cond_method),
            library_npz=args.library_npz if args.library_npz else None,
            retrieval_k=int(args.retrieval_k),
            retrieval_temp=float(args.retrieval_temp),
            num_samples=int(args.num_samples),
        )

        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "in_r", "in_g", "in_b",
                "out_r", "out_g", "out_b",
                "conf_diffusion", "conf_retrieval",
            ])
            for i in range(rgbs.shape[0]):
                w.writerow([
                    int(rgbs[i, 0]), int(rgbs[i, 1]), int(rgbs[i, 2]),
                    int(pred_rgb0[i, 0]), int(pred_rgb0[i, 1]), int(pred_rgb0[i, 2]),
                    float(conf_diff[i]) if np.isfinite(conf_diff[i]) else "",
                    float(conf_ret[i]) if np.isfinite(conf_ret[i]) else "",
                ])
        print(json.dumps({"saved": args.out_csv, "n": int(rgbs.shape[0])}, ensure_ascii=False))
        return

    # mine mode
    if not args.test_npz:
        raise ValueError("--mode mine requires --test_npz")

    ds = PigmentNPZDataset(args.test_npz)
    dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False, num_workers=0, drop_last=False)
    cfg, denoiser, conditioner, schedule, color_encoder, cond_predictor = model_components

    lab_norm = LabNorm()
    rows: List[Dict] = []
    seen = 0

    for batch in dl:
        if seen >= int(args.max_rows):
            break
        x0 = batch["x0"].to(device)
        mask = batch["mask"].to(device)

        # last observed as current
        obs_any = (mask.mean(dim=-1) > 0.5).long()
        idx_range = torch.arange(mask.size(1), device=device).view(1, -1)
        last_idx = torch.max(idx_range * obs_any, dim=1).values
        x_curr = x0[torch.arange(x0.size(0), device=device), last_idx]

        # cond (pred only, for mining)
        if conditioner.cond_dim == 0:
            cond = None
        else:
            if args.cond_method != "pred":
                raise ValueError("mine mode currently supports --cond_method pred (to be consistent with RGB-only inference)")
            if color_encoder is None or cond_predictor is None:
                raise ValueError("pred requires ckpt with color_encoder+cond_predictor")
            embeds_pred = _predict_embeds_from_rgb(x_curr, conditioner, color_encoder, cond_predictor)
            cond = _build_cond_from_pred_embeds(conditioner, embeds_pred)

        pred_lab0, info = _sample_with_confidence(denoiser, schedule, x0, mask, cond, num_samples=int(args.num_samples))
        gt_lab0 = lab_norm.denormalize(x0[:, 0, :].detach().cpu().numpy())
        curr_lab = lab_norm.denormalize(x_curr.detach().cpu().numpy())

        pred_rgb0 = lab_to_rgb(pred_lab0)
        curr_rgb = lab_to_rgb(curr_lab)

        for i in range(pred_lab0.shape[0]):
            de = float(delta_e2000(pred_lab0[i], gt_lab0[i]))
            rows.append({
                "deltaE2000": de,
                "curr_rgb": curr_rgb[i].tolist(),
                "pred_rgb0": pred_rgb0[i].tolist(),
                "gt_rgb0": lab_to_rgb(gt_lab0[i][None, :])[0].tolist(),
                "conf_diffusion": float(info.get("conf_diffusion", np.nan)),
            })

        seen += int(x0.shape[0])

    # sort by deltaE desc and dump
    rows.sort(key=lambda d: d["deltaE2000"], reverse=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["deltaE2000", "curr_rgb", "pred_rgb0", "gt_rgb0", "conf_diffusion"])
        for r in rows:
            w.writerow([r["deltaE2000"], r["curr_rgb"], r["pred_rgb0"], r["gt_rgb0"], r["conf_diffusion"]])

    top = rows[:10]
    print(json.dumps({"saved": args.out_csv, "rows": len(rows), "top10": top}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
