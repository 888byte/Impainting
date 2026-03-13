#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pretrain cross-modal alignment between:
- color (Lab derived from RGB) embeddings
- Raman spectrum embeddings

Why:
- To enable *RGB-only* inference when Raman/XRD are missing by retrieving a plausible Raman embedding
  from your 40+ standard Raman spectra library (CLIP-style cross-modal retrieval).

Inputs (your uploaded files):
- 拉曼汇总.xlsx : standard Raman spectra (one sheet per substance)
- 拉曼物质颜色汇总.xlsx : table mapping sheet name -> RGB/Lab

Outputs:
- alignment_ckpt.pt : {"color_encoder": ..., "raman_encoder": ..., "cfg": ...}
- library_embeddings.npz : raman_emb (N,d), lab (N,3), sheet_name (N,)

Example:
  python pigment_task/pretrain_raman_color_alignment.py \
    --raman_sum_xlsx "/mnt/data/拉曼汇总.xlsx" \
    --color_map_xlsx "/mnt/data/拉曼物质颜色汇总.xlsx" \
    --out_dir "data/standard_alignment" \
    --d_model 128 --raman_len 1024
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from utils.color_utils import LabNorm
from utils.io_utils import load_rruff_raman_sheet
from models.color_encoder import ColorEncoder, ColorEncoderConfig
from models.spectral_encoder import MambaSpectralEncoder


class RamanColorPairs(Dataset):
    def __init__(self, specs: np.ndarray, labs_norm: np.ndarray):
        self.specs = specs.astype(np.float32)
        self.labs = labs_norm.astype(np.float32)

    def __len__(self) -> int:
        return int(self.specs.shape[0])

    def __getitem__(self, idx: int):
        return {
            "spec": torch.from_numpy(self.specs[idx]),
            "lab": torch.from_numpy(self.labs[idx]),
        }


def clip_loss(z_img: torch.Tensor, z_txt: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
    """
    Symmetric InfoNCE loss.
    """
    z1 = F.normalize(z_img, dim=-1)
    z2 = F.normalize(z_txt, dim=-1)
    logits = (z1 @ z2.T) / float(temp)
    labels = torch.arange(z1.size(0), device=z1.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_i2t + loss_t2i)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raman_sum_xlsx", type=str, required=True)
    ap.add_argument("--color_map_xlsx", type=str, required=True)
    ap.add_argument("--color_sheet", type=str, default="颜色汇总(含RGB_Lab)")
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--raman_len", type=int, default=1024)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--temp", type=float, default=0.07)

    ap.add_argument("--aux_reg", type=float, default=0.1, help="Optional auxiliary Raman->Lab MSE weight")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_excel(args.color_map_xlsx, sheet_name=args.color_sheet)
    # robust column names
    if "页签名" not in df.columns:
        raise ValueError("color_map_xlsx must contain column '页签名' (sheet name)")
    for col in ["Lab_L*", "Lab_a*", "Lab_b*"]:
        if col not in df.columns:
            raise ValueError(f"color_map_xlsx missing column {col}")

    rows = df.dropna(subset=["页签名", "Lab_L*", "Lab_a*", "Lab_b*"]).copy()
    sheet_names: List[str] = [str(s).strip() for s in rows["页签名"].tolist() if str(s).strip()]
    labs = rows[["Lab_L*", "Lab_a*", "Lab_b*"]].to_numpy(dtype=np.float32)

    lab_norm = LabNorm()
    labs_norm = lab_norm.normalize(labs).astype(np.float32)

    specs = []
    valid_names = []
    valid_labs_norm = []
    for name, labn in zip(sheet_names, labs_norm):
        try:
            spec = load_rruff_raman_sheet(args.raman_sum_xlsx, name, new_len=int(args.raman_len))
            specs.append(spec.astype(np.float32))
            valid_names.append(name)
            valid_labs_norm.append(labn.astype(np.float32))
        except Exception as e:
            print(f"[WARN] skip sheet '{name}': {e}")
            continue

    specs = np.stack(specs, axis=0).astype(np.float32)
    valid_labs_norm = np.stack(valid_labs_norm, axis=0).astype(np.float32)
    print(f"[INFO] Loaded {specs.shape[0]} Raman-color pairs")

    ds = RamanColorPairs(specs, valid_labs_norm)
    dl = DataLoader(ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0, drop_last=True)

    raman_enc = MambaSpectralEncoder(
        spec_len=int(args.raman_len),
        d_model=int(args.d_model),
        n_layers=int(args.n_layers),
        dropout=float(args.dropout),
        pooling="mean",
    ).to(device)

    color_enc = ColorEncoder(ColorEncoderConfig(in_dim=3, d_model=int(args.d_model), hidden_dim=256, n_layers=2, dropout=0.0)).to(device)

    # optional auxiliary regression head
    reg_head = nn.Linear(int(args.d_model), 3).to(device)

    opt = torch.optim.AdamW(
        list(raman_enc.parameters()) + list(color_enc.parameters()) + list(reg_head.parameters()),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    best = 1e9
    for ep in range(1, int(args.epochs) + 1):
        raman_enc.train()
        color_enc.train()
        reg_head.train()

        losses = []
        for batch in dl:
            spec = batch["spec"].to(device)
            lab = batch["lab"].to(device)

            zr = raman_enc(spec)
            zc = color_enc(lab)
            loss_align = clip_loss(zc, zr, temp=float(args.temp))

            loss_aux = torch.tensor(0.0, device=device)
            if float(args.aux_reg) > 0:
                pred_lab = reg_head(zr)
                loss_aux = torch.mean((pred_lab - lab) ** 2)

            loss = loss_align + float(args.aux_reg) * loss_aux

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses.append(float(loss.item()))

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        if ep % 50 == 0 or ep == 1:
            print(f"[ep {ep:04d}] loss={mean_loss:.6f}")

        if mean_loss < best:
            best = mean_loss
            # save best
            ckpt = {
                "cfg": {
                    "raman_len": int(args.raman_len),
                    "d_model": int(args.d_model),
                    "n_layers": int(args.n_layers),
                    "temp": float(args.temp),
                },
                "color_encoder": color_enc.state_dict(),
                "raman_encoder": raman_enc.state_dict(),
            }
            torch.save(ckpt, os.path.join(args.out_dir, "alignment_ckpt.pt"))

    # Build library embeddings
    raman_enc.eval()
    color_enc.eval()

    with torch.no_grad():
        zr_all = []
        zc_all = []
        for i in range(specs.shape[0]):
            spec = torch.from_numpy(specs[i:i+1]).to(device)
            lab = torch.from_numpy(valid_labs_norm[i:i+1]).to(device)
            zr_all.append(raman_enc(spec).cpu().numpy()[0])
            zc_all.append(color_enc(lab).cpu().numpy()[0])
        zr_all = np.stack(zr_all, axis=0).astype(np.float32)
        zc_all = np.stack(zc_all, axis=0).astype(np.float32)

    np.savez_compressed(
        os.path.join(args.out_dir, "library_embeddings.npz"),
        raman_emb=zr_all,
        color_emb=zc_all,
        lab=lab_norm.denormalize(valid_labs_norm),
        sheet_name=np.asarray(valid_names, dtype=object),
    )
    meta = {
        "N": int(specs.shape[0]),
        "out_dir": args.out_dir,
        "raman_len": int(args.raman_len),
        "d_model": int(args.d_model),
        "n_layers": int(args.n_layers),
        "note": "raman_emb/color_emb are NOT L2-normalized; normalize before cosine similarity",
    }
    with open(os.path.join(args.out_dir, "library_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
