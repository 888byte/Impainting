
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optional: Pretrain Raman Mamba encoder using your 40+ standard Raman spectra.

Input files you said you have:
- 拉曼汇总.xlsx: many sheets, each is a Raman spectrum (RRUFF style)
- 拉曼物质颜色汇总.xlsx: sheet "颜色汇总(含RGB_Lab)" provides (页签名 -> Lab)

This script trains:
  Raman spectrum -> Lab (normalized)

Then you can reuse encoder weights in the multimodal fading model.

Usage:
  python pigment_task/pretrain_raman_encoder.py \
    --rruff_excel "/path/拉曼汇总.xlsx" \
    --color_excel "/path/拉曼物质颜色汇总.xlsx" \
    --out_ckpt "ckpt/raman_pretrain.pt"
"""


from __future__ import annotations

import argparse
import os
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils.color_utils import LabNorm, delta_e2000
from utils.io_utils import load_rruff_raman_sheet
from models.spectral_encoder import ConditionerConfig, MultimodalConditioner


class RamanColorDataset(Dataset):
    def __init__(self, specs: np.ndarray, labs: np.ndarray):
        self.specs = specs.astype(np.float32)
        self.labs = labs.astype(np.float32)

    def __len__(self) -> int:
        return self.specs.shape[0]

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.specs[idx]), torch.from_numpy(self.labs[idx])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def eval_encoder(encoder: MultimodalConditioner, head: nn.Module, dl: DataLoader, device: torch.device, lab_norm: LabNorm):
    encoder.eval()
    head.eval()
    des = []
    maes = []
    for spec, lab in dl:
        spec = spec.to(device)
        lab = lab.to(device)
        cond = encoder(spec, None)  # (B,d)
        pred = head(cond)           # (B,3)
        maes.append(torch.mean(torch.abs(pred - lab)).item())
        gt = lab_norm.denormalize(lab.cpu().numpy())
        pr = lab_norm.denormalize(pred.cpu().numpy())
        des.append(float(np.mean(delta_e2000(gt, pr))))
    return float(np.mean(maes)), float(np.mean(des))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rruff_excel", type=str, required=True)
    ap.add_argument("--color_excel", type=str, required=True)
    ap.add_argument("--out_ckpt", type=str, required=True)

    ap.add_argument("--raman_len", type=int, default=1024)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    lab_norm = LabNorm()

    # Read label table
    df = pd.read_excel(args.color_excel, sheet_name="颜色汇总(含RGB_Lab)")
    rows = []
    for _, r in df.iterrows():
        sheet = str(r.get("页签名", "")).strip()
        if not sheet or sheet.lower() == "nan":
            continue
        try:
            lab = np.array([float(r["Lab_L*"]), float(r["Lab_a*"]), float(r["Lab_b*"])], dtype=np.float32)
        except Exception:
            continue
        rows.append((sheet, lab))

    specs: List[np.ndarray] = []
    labs: List[np.ndarray] = []
    for sheet, lab in rows:
        try:
            spec = load_rruff_raman_sheet(args.rruff_excel, sheet_name=sheet, new_len=args.raman_len)
        except Exception as e:
            print(f"[WARN] skip {sheet}: {e}")
            continue
        specs.append(spec)
        labs.append(lab_norm.normalize(lab))

    if len(specs) < 10:
        raise RuntimeError("Too few usable Raman spectra after parsing. Check excel sheets / names.")

    specs_arr = np.stack(specs, axis=0)
    labs_arr = np.stack(labs, axis=0)

    # split
    N = specs_arr.shape[0]
    idx = np.arange(N)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    n_val = max(1, int(N * 0.1))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    ds_train = RamanColorDataset(specs_arr[train_idx], labs_arr[train_idx])
    ds_val = RamanColorDataset(specs_arr[val_idx], labs_arr[val_idx])
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Encoder as conditioner (raman only)
    cond_cfg = ConditionerConfig(
        use_raman=True,
        use_xrd=False,
        raman_len=args.raman_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )
    encoder = MultimodalConditioner(cond_cfg).to(device)

    # Small head to predict Lab
    head = nn.Sequential(
        nn.Linear(encoder.cond_dim, encoder.cond_dim),
        nn.SiLU(),
        nn.Linear(encoder.cond_dim, 3),
    ).to(device)

    opt = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_de = 1e9
    for ep in range(1, args.epochs + 1):
        encoder.train()
        head.train()
        for spec, lab in dl_train:
            spec = spec.to(device)
            lab = lab.to(device)
            cond = encoder(spec, None)
            pred = head(cond)
            loss = loss_fn(pred, lab)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        if ep % 10 == 0 or ep == 1:
            mae, de = eval_encoder(encoder, head, dl_val, device, lab_norm)
            print(f"[val] ep={ep} mae_norm={mae:.6f} deltaE2000={de:.3f}")
            if de < best_de:
                best_de = de
                os.makedirs(os.path.dirname(args.out_ckpt) or ".", exist_ok=True)
                torch.save({"conditioner": encoder.state_dict(), "head": head.state_dict(), "cfg": vars(args)}, args.out_ckpt)
                print(f"[OK] saved best to {args.out_ckpt}")

    print("[DONE]")


if __name__ == "__main__":
    main()