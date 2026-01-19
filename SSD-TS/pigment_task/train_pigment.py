#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""pigment_task.train_pigment

训练：根据褪色颜色序列预测原始颜色 (t0)。

基于 hotfix2 改进：
- 修复“best_model 永远在第一轮”的根因：原先只用 val_loss(true_cond) 作为 best/early-stop 指标。
- 新增 train.monitor_metric：pred_cond / true_cond / mixed（默认 missing_modality.enable 时用 pred_cond）
- 同时保存 best_true_model.pt 与 best_pred_model.pt，避免混淆
- 兼容 diffusion.beta_start/beta_end 与 color_aug.p

用法：
  python -m pigment_task.train_pigment --config pigment_task/configs/pigment_lab_raman_xrd_v2.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pigment_task.dataset_pigment import PigmentNPZDataset
from pigment_task.diffusion import DiffusionConfig, DiffusionSchedule, diffusion_loss, p_sample_loop
from pigment_task.color_utils import delta_e2000
from pigment_task.models.color_encoder import ColorEncoder, ColorEncoderConfig
from pigment_task.models.cond_predictor import ColorToSpecPredictor, CondPredictorConfig
from pigment_task.models.pigment_denoiser import DenoiserConfig, MambaDenoiser
from pigment_task.models.spectral_encoder import ConditionerConfig, MultimodalConditioner


# ------------------------------ utils ------------------------------

def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _denorm_lab_torch(x_norm: torch.Tensor) -> torch.Tensor:
    """(B,L,3) normalized -> Lab"""
    x = x_norm.clone()
    x[..., 0] = x[..., 0] * 100.0
    x[..., 1] = x[..., 1] * 128.0
    x[..., 2] = x[..., 2] * 128.0
    return x


def _norm_lab_torch(x_lab: torch.Tensor) -> torch.Tensor:
    x = x_lab.clone()
    x[..., 0] = x[..., 0] / 100.0
    x[..., 1] = x[..., 1] / 128.0
    x[..., 2] = x[..., 2] / 128.0
    return x


def _apply_color_aug_selected(
    x0_norm: torch.Tensor,
    selected: torch.Tensor,
    cfg: Dict,
) -> torch.Tensor:
    """对 selected==True 的样本做 Lab 域随机化。x0_norm: (B,L,3)"""
    if not bool(cfg.get("enable", False)):
        return x0_norm
    if selected is None:
        return x0_norm

    device = x0_norm.device
    B, L, _ = x0_norm.shape
    selected = selected.to(device).view(B)
    if selected.sum().item() == 0:
        return x0_norm

    # 兼容 p / prob
    prob = cfg.get("prob", cfg.get("p", 0.0))
    prob = float(prob)
    if prob <= 0:
        return x0_norm

    gate = (torch.rand(B, device=device) < prob) & selected
    if gate.sum().item() == 0:
        return x0_norm

    idx = torch.nonzero(gate, as_tuple=False).squeeze(1)
    x = x0_norm.clone()
    xb = _denorm_lab_torch(x[idx])  # (n,L,3)

    n = xb.shape[0]
    # sample params
    L_scale = float(cfg.get("L_scale", 0.0))
    L_shift = float(cfg.get("L_shift", 0.0))
    ab_scale = float(cfg.get("ab_scale", 0.0))
    ab_rot = float(cfg.get("ab_rotate_deg", 0.0))
    noise_std = float(cfg.get("noise_std", 0.0))

    if L_scale != 0.0:
        sL = 1.0 + (torch.rand(n, 1, 1, device=device) * 2.0 - 1.0) * L_scale
        xb[..., 0:1] = xb[..., 0:1] * sL
    if L_shift != 0.0:
        bL = (torch.rand(n, 1, 1, device=device) * 2.0 - 1.0) * L_shift
        xb[..., 0:1] = xb[..., 0:1] + bL

    if ab_scale != 0.0:
        sab = 1.0 + (torch.rand(n, 1, 1, device=device) * 2.0 - 1.0) * ab_scale
        xb[..., 1:3] = xb[..., 1:3] * sab

    if ab_rot != 0.0:
        theta = (torch.rand(n, 1, 1, device=device) * 2.0 - 1.0) * (ab_rot * np.pi / 180.0)
        c = torch.cos(theta)
        s = torch.sin(theta)
        a = xb[..., 1:2]
        b = xb[..., 2:3]
        a2 = c * a - s * b
        b2 = s * a + c * b
        xb[..., 1:2] = a2
        xb[..., 2:3] = b2

    if noise_std != 0.0:
        xb = xb + torch.randn_like(xb) * noise_std

    xb[..., 0] = xb[..., 0].clamp(0.0, 100.0)
    xb[..., 1] = xb[..., 1].clamp(-128.0, 128.0)
    xb[..., 2] = xb[..., 2].clamp(-128.0, 128.0)

    x[idx] = _norm_lab_torch(xb)
    return x


def _last_observed_index(mask: torch.Tensor) -> torch.Tensor:
    """mask: (B,L,3) -> (B,) last index where observed==1"""
    obs = (mask.mean(dim=-1) > 0.5).to(torch.long)  # (B,L)
    B, L = obs.shape
    idx_range = torch.arange(L, device=mask.device).view(1, L).expand(B, L)
    return torch.max(idx_range * obs, dim=1).values  # (B,)


def _gather_last_observed(x0: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return x_curr: (B,3) normalized Lab from last observed step."""
    idx = _last_observed_index(mask)  # (B,)
    b = torch.arange(x0.shape[0], device=x0.device)
    return x0[b, idx, :]


@torch.no_grad()
def build_cond_true(
    conditioner: MultimodalConditioner,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
    """返回 cond + 原始 embeds（用于 pred loss）。"""
    if conditioner.cond_dim == 0:
        return None, {}

    raman = batch.get("raman", None)
    xrd = batch.get("xrd", None)
    raman_peaks = batch.get("raman_peaks", None)
    xrd_peaks = batch.get("xrd_peaks", None)

    cond, embeds = conditioner(
        raman.to(device) if raman is not None else None,
        xrd.to(device) if xrd is not None else None,
        raman_peaks=raman_peaks.to(device) if raman_peaks is not None else None,
        xrd_peaks=xrd_peaks.to(device) if xrd_peaks is not None else None,
        return_embeds=True,
    )
    return cond, embeds


def build_cond_from_pred_embeds(
    conditioner: MultimodalConditioner,
    embeds_pred: Dict[str, torch.Tensor],
) -> Optional[torch.Tensor]:
    if conditioner.cond_dim == 0:
        return None

    feats = []
    if conditioner.raman_enc is not None:
        r = embeds_pred.get("raman_emb", embeds_pred.get("raman", None))
        if r is None:
            raise KeyError("Missing predicted Raman embedding")
        feats.append(r)
    if conditioner.xrd_enc is not None:
        x = embeds_pred.get("xrd_emb", embeds_pred.get("xrd", None))
        if x is None:
            raise KeyError("Missing predicted XRD embedding")
        feats.append(x)

    cond_cat = torch.cat(feats, dim=-1)
    return conditioner.fuse(cond_cat)


def _get_embed(embeds: Dict[str, torch.Tensor], keys: Tuple[str, ...]) -> Optional[torch.Tensor]:
    """Robustly fetch an embedding tensor from a dict with unknown key naming."""
    if embeds is None:
        return None
    for k in keys:
        v = embeds.get(k, None)
        if v is not None:
            return v
    return None


@dataclass
class MissingModalityCfg:
    enable: bool = False
    drop_prob: float = 0.3
    lambda_pred: float = 0.1
    freeze_color_encoder: bool = False
    freeze_conditioner: bool = False


@torch.no_grad()
def eval_sampling_metrics(
    denoiser: nn.Module,
    conditioner: MultimodalConditioner,
    schedule: DiffusionSchedule,
    dl_val: DataLoader,
    device: torch.device,
    num_batches: int = 1,
    cond_override: Optional[str] = None,
    color_encoder: Optional[ColorEncoder] = None,
    cond_predictor: Optional[ColorToSpecPredictor] = None,
) -> Dict[str, float]:
    """用采样评估 t0 的 DeltaE（较慢）。"""
    denoiser.eval()
    conditioner.eval()
    if color_encoder is not None:
        color_encoder.eval()
    if cond_predictor is not None:
        cond_predictor.eval()

    all_de = []
    for i, batch in enumerate(dl_val):
        if i >= num_batches:
            break
        x0 = batch["x0"].to(device)
        mask = batch["mask"].to(device)

        if cond_override == "pred":
            if color_encoder is None or cond_predictor is None:
                raise ValueError("cond_override='pred' but color_encoder/cond_predictor missing")
            x_curr = _gather_last_observed(x0, mask)
            zc = color_encoder(x_curr)
            embeds_pred = cond_predictor(zc)
            cond = build_cond_from_pred_embeds(conditioner, embeds_pred)
        else:
            cond, _ = build_cond_true(conditioner, batch, device)

        x_obs = x0 * mask
        x_sample = p_sample_loop(denoiser, schedule, x_obs=x_obs, obs_mask=mask, cond=cond)

        pred0 = _denorm_lab_torch(x_sample[:, 0, :]).detach().cpu().numpy()
        gt0 = _denorm_lab_torch(x0[:, 0, :]).detach().cpu().numpy()
        for p, g in zip(pred0, gt0):
            all_de.append(float(delta_e2000(p, g)))

    if not all_de:
        return {"deltaE2000_mean": float("nan"), "deltaE2000_std": float("nan")}
    return {"deltaE2000_mean": float(np.mean(all_de)), "deltaE2000_std": float(np.std(all_de))}


@torch.no_grad()
def eval_val_loss(
    denoiser: nn.Module,
    conditioner: MultimodalConditioner,
    schedule: DiffusionSchedule,
    dl_val: DataLoader,
    device: torch.device,
    num_batches: int = 10,
    cond_override: Optional[str] = None,
    color_encoder: Optional[ColorEncoder] = None,
    cond_predictor: Optional[ColorToSpecPredictor] = None,
) -> float:
    denoiser.eval()
    conditioner.eval()
    if color_encoder is not None:
        color_encoder.eval()
    if cond_predictor is not None:
        cond_predictor.eval()

    losses = []
    for i, batch in enumerate(dl_val):
        if i >= num_batches:
            break
        x0 = batch["x0"].to(device)
        mask = batch["mask"].to(device)

        if cond_override == "pred":
            if color_encoder is None or cond_predictor is None:
                raise ValueError("cond_override='pred' but color_encoder/cond_predictor missing")
            x_curr = _gather_last_observed(x0, mask)
            zc = color_encoder(x_curr)
            embeds_pred = cond_predictor(zc)
            cond = build_cond_from_pred_embeds(conditioner, embeds_pred)
        else:
            cond, _ = build_cond_true(conditioner, batch, device)

        loss = diffusion_loss(denoiser, schedule, x0=x0, obs_mask=mask, cond=cond)
        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else float("nan")


def _monitor_value(
    monitor_metric: str,
    val_true: float,
    val_pred: Optional[float],
    mixed_alpha: float,
) -> float:
    m = (monitor_metric or "").lower()
    if m == "true_cond":
        return float(val_true)
    if m == "pred_cond":
        if val_pred is None or (isinstance(val_pred, float) and np.isnan(val_pred)):
            return float("inf")
        return float(val_pred)
    if m == "mixed":
        if val_pred is None or (isinstance(val_pred, float) and np.isnan(val_pred)):
            return float(val_true)
        a = float(mixed_alpha)
        a = max(0.0, min(1.0, a))
        return float(a * val_pred + (1.0 - a) * val_true)
    # fallback
    return float(val_true)


def _save_ckpt(
    path: str,
    cfg: Dict,
    ep: int,
    global_step: int,
    monitor_val: float,
    denoiser: nn.Module,
    conditioner: nn.Module,
    color_encoder: Optional[nn.Module],
    cond_predictor: Optional[nn.Module],
) -> None:
    ckpt = {
        "cfg": cfg,
        "epoch": ep,
        "global_step": global_step,
        "val_loss": float(monitor_val),
        "denoiser": denoiser.state_dict(),
        "conditioner": conditioner.state_dict(),
    }
    if color_encoder is not None:
        ckpt["color_encoder"] = color_encoder.state_dict()
    if cond_predictor is not None:
        ckpt["cond_predictor"] = cond_predictor.state_dict()
    torch.save(ckpt, path)


# ------------------------------ main ------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)

    train_npz = cfg["data"]["train_npz"]
    val_npz = cfg["data"]["val_npz"]

    ds_train = PigmentNPZDataset(train_npz)
    ds_val = PigmentNPZDataset(val_npz)

    tr_cfg = cfg.get("train", {})
    batch_size = int(tr_cfg.get("batch_size", 64))
    num_workers = int(tr_cfg.get("num_workers", 0))

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=int(tr_cfg.get("eval_batch_size", batch_size)), shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)

    device = torch.device(tr_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    _set_seed(int(tr_cfg.get("seed", 42)))

    # diffusion schedule (兼容 beta_start/beta_end)
    diff_cfg_raw = cfg.get("diffusion", {}) or {}
    beta_0 = diff_cfg_raw.get("beta_0", diff_cfg_raw.get("beta_start", 1e-4))
    beta_T = diff_cfg_raw.get("beta_T", diff_cfg_raw.get("beta_end", 0.02))
    diff_cfg = DiffusionConfig(
        T=int(diff_cfg_raw.get("T", 200)),
        beta_0=float(beta_0),
        beta_T=float(beta_T),
    )
    schedule = DiffusionSchedule(diff_cfg, device=device)

    # conditioner
    mod_cfg = cfg.get("modality", {})
    cond_cfg = ConditionerConfig(
        use_raman=bool(mod_cfg.get("use_raman", False)),
        use_xrd=bool(mod_cfg.get("use_xrd", False)),
        raman_len=int(mod_cfg.get("raman_len", 1024)),
        xrd_len=int(mod_cfg.get("xrd_len", 2048)),
        d_model=int(mod_cfg.get("spec_d_model", 128)),
        n_layers=int(mod_cfg.get("spec_n_layers", 4)),
        dropout=float(mod_cfg.get("spec_dropout", 0.0)),
        raman_peak_dim=int(mod_cfg.get("raman_peak_dim", 0)),
        xrd_peak_dim=int(mod_cfg.get("xrd_peak_dim", 0)),
        use_fuse=bool(mod_cfg.get("use_fuse", True)),
    )
    conditioner = MultimodalConditioner(cond_cfg).to(device)

    # denoiser
    model_cfg = cfg.get("model", {})
    den_cfg = DenoiserConfig(
        in_channels=int(model_cfg.get("in_channels", 3)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        n_layers=int(model_cfg.get("n_layers", 4)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        cond_dim=int(conditioner.cond_dim),
    )
    denoiser = MambaDenoiser(den_cfg).to(device)

    # missing-modality modules (optional)
    mm_cfg_raw = cfg.get("missing_modality", {}) or {}
    mm_cfg = MissingModalityCfg(
        enable=bool(mm_cfg_raw.get("enable", False)),
        drop_prob=float(mm_cfg_raw.get("drop_prob", 0.3)),
        lambda_pred=float(mm_cfg_raw.get("lambda_pred", 0.1)),
        freeze_color_encoder=bool(mm_cfg_raw.get("freeze_color_encoder", False)),
        freeze_conditioner=bool(mm_cfg_raw.get("freeze_conditioner", False)),
    )

    color_encoder: Optional[ColorEncoder] = None
    cond_predictor: Optional[ColorToSpecPredictor] = None
    if mm_cfg.enable and conditioner.cond_dim > 0:
        ce_cfg = ColorEncoderConfig(
            in_dim=3,
            d_model=int(mm_cfg_raw.get("color_d_model", cond_cfg.d_model)),
            hidden_dim=int(mm_cfg_raw.get("color_hidden_dim", 256)),
            n_layers=int(mm_cfg_raw.get("color_n_layers", 2)),
            dropout=float(mm_cfg_raw.get("color_dropout", 0.0)),
        )
        color_encoder = ColorEncoder(ce_cfg).to(device)

        cp_cfg = CondPredictorConfig(
            in_dim=ce_cfg.d_model,
            d_model=cond_cfg.d_model,
            use_raman=bool(cond_cfg.use_raman),
            use_xrd=bool(cond_cfg.use_xrd),
            hidden_dim=int(mm_cfg_raw.get("pred_hidden_dim", 256)),
            n_layers=int(mm_cfg_raw.get("pred_n_layers", 2)),
            dropout=float(mm_cfg_raw.get("pred_dropout", 0.0)),
        )
        cond_predictor = ColorToSpecPredictor(cp_cfg).to(device)

        if mm_cfg.freeze_color_encoder:
            for p in color_encoder.parameters():
                p.requires_grad = False
        if mm_cfg.freeze_conditioner:
            for p in conditioner.parameters():
                p.requires_grad = False

    # load pretrained weights (optional)
    pre_cfg = cfg.get("pretrained", {}) or {}
    if pre_cfg.get("alignment_ckpt", ""):
        ckpt_path = pre_cfg["alignment_ckpt"]
        sd = torch.load(ckpt_path, map_location="cpu")
        if color_encoder is not None and isinstance(sd, dict) and "color_encoder" in sd:
            color_encoder.load_state_dict(sd["color_encoder"], strict=False)
            print(f"[INFO] Loaded color_encoder from {ckpt_path}")
        if conditioner.raman_enc is not None and isinstance(sd, dict) and "raman_encoder" in sd:
            conditioner.raman_enc.load_state_dict(sd["raman_encoder"], strict=False)
            print(f"[INFO] Loaded raman_encoder from {ckpt_path}")

    if pre_cfg.get("raman_encoder_ckpt", ""):
        ckpt_path = pre_cfg["raman_encoder_ckpt"]
        sd = torch.load(ckpt_path, map_location="cpu")
        if isinstance(sd, dict) and "conditioner" in sd:
            conditioner.load_state_dict(sd["conditioner"], strict=False)
        else:
            conditioner.load_state_dict(sd, strict=False)
        print(f"[INFO] Loaded pretrained conditioner from {ckpt_path}")

    # optimizer
    params = list(denoiser.parameters()) + list(conditioner.parameters())
    if color_encoder is not None:
        params += list(color_encoder.parameters())
    if cond_predictor is not None:
        params += list(cond_predictor.parameters())

    opt = torch.optim.AdamW(
        params,
        lr=float(tr_cfg.get("lr", 1e-3)),
        weight_decay=float(tr_cfg.get("weight_decay", 0.0)),
    )

    grad_clip = float(tr_cfg.get("grad_clip", 1.0))
    epochs = int(tr_cfg.get("epochs", 200))
    log_every = int(tr_cfg.get("log_every", 50))
    eval_every = int(tr_cfg.get("eval_every", 1))
    eval_num_batches = int(tr_cfg.get("eval_num_batches", 2))
    save_every = int(tr_cfg.get("save_every", 10))
    save_dir = tr_cfg.get("save_dir", "ckpt/pigment_v2")

    early_stopping_patience = int(tr_cfg.get("early_stopping_patience", 100))
    early_stopping_min_delta = float(tr_cfg.get("early_stopping_min_delta", 0.001))

    os.makedirs(save_dir, exist_ok=True)

    # color aug cfg
    color_aug_cfg = cfg.get("color_aug", {}) or {}

    # monitor metric selection
    monitor_metric = str(tr_cfg.get("monitor_metric", "")).strip().lower()
    mixed_alpha = float(tr_cfg.get("monitor_mixed_alpha", 0.7))
    if not monitor_metric:
        monitor_metric = "pred_cond" if (mm_cfg.enable and conditioner.cond_dim > 0) else "true_cond"
    print(f"[INFO] monitor_metric = {monitor_metric} (mixed_alpha={mixed_alpha})")

    best_monitor = float("inf")
    best_true = float("inf")
    best_pred = float("inf")
    no_improve_count = 0
    global_step = 0

    for ep in range(1, epochs + 1):
        denoiser.train()
        conditioner.train()
        if color_encoder is not None:
            color_encoder.train()
        if cond_predictor is not None:
            cond_predictor.train()

        for batch in dl_train:
            x0 = batch["x0"].to(device)
            mask = batch["mask"].to(device)

            # true cond + embeds
            cond_true, embeds_true = build_cond_true(conditioner, batch, device)
            if global_step == 0 and mm_cfg.enable and conditioner.cond_dim > 0:
                try:
                    print(f"[DEBUG] embeds_true keys: {list(embeds_true.keys())}")
                except Exception:
                    pass

            B = x0.shape[0]
            use_pred = torch.zeros(B, device=device, dtype=torch.bool)

            has_raman = batch.get("has_raman", None)
            has_xrd = batch.get("has_xrd", None)
            if has_raman is not None:
                has_raman = has_raman.to(device).view(B).long()
            if has_xrd is not None:
                has_xrd = has_xrd.to(device).view(B).long()

            # Robustly infer availability from spectrum tensors
            raman_raw = batch.get("raman", None)
            if raman_raw is not None:
                raman_raw = raman_raw.to(device)
                has_raman_inf = (raman_raw.abs().sum(dim=1) > 1e-6).long()
                has_raman = has_raman_inf if has_raman is None else torch.maximum(has_raman, has_raman_inf)

            xrd_raw = batch.get("xrd", None)
            if xrd_raw is not None:
                xrd_raw = xrd_raw.to(device)
                has_xrd_inf = (xrd_raw.abs().sum(dim=1) > 1e-6).long()
                has_xrd = has_xrd_inf if has_xrd is None else torch.maximum(has_xrd, has_xrd_inf)

            if global_step == 0:
                try:
                    hr = float(has_raman.float().mean().item()) if has_raman is not None else None
                    hx = float(has_xrd.float().mean().item()) if has_xrd is not None else None
                    print(f"[DEBUG] has_raman_ratio={hr}  has_xrd_ratio={hx}")
                except Exception:
                    pass

            # choose cond_in
            if conditioner.cond_dim == 0 or (not mm_cfg.enable) or color_encoder is None or cond_predictor is None:
                cond_in = cond_true if conditioner.cond_dim > 0 else None
                drop_eff = 0.0
            else:
                # missing modality -> force pred
                if conditioner.raman_enc is not None and has_raman is not None:
                    use_pred |= (has_raman == 0)
                if conditioner.xrd_enc is not None and has_xrd is not None:
                    use_pred |= (has_xrd == 0)

                drop_p = float(mm_cfg.drop_prob)
                warmup_ep = int(mm_cfg_raw.get("drop_prob_warmup_epochs", 10))
                allow_all_pred = bool(mm_cfg_raw.get("allow_all_pred", False))
                drop_cap = float(mm_cfg_raw.get("drop_prob_cap", 0.8))
                if not allow_all_pred:
                    drop_p = min(drop_p, drop_cap)
                if warmup_ep > 0:
                    drop_eff = drop_p * min(1.0, float(ep) / float(warmup_ep))
                else:
                    drop_eff = drop_p

                forced_pred = use_pred.clone()

                if cond_true is not None and drop_eff > 0:
                    use_pred |= (torch.rand(B, device=device) < drop_eff)
                elif cond_true is None:
                    use_pred[:] = True

                # 额外安全：尽量保证每个 batch 至少有一定比例走 true_cond（如果存在可用真谱）
                min_true_frac = float(mm_cfg_raw.get("min_true_frac", 0.2))
                if (not allow_all_pred) and (cond_true is not None) and (min_true_frac > 0):
                    cur_true = float((~use_pred).float().mean().item())
                    if cur_true < min_true_frac:
                        candidates = torch.nonzero(~forced_pred, as_tuple=False).view(-1)
                        if candidates.numel() > 0:
                            need = int(np.ceil((min_true_frac - cur_true) * B))
                            need = max(1, need)
                            take = min(need, int(candidates.numel()))
                            sel = candidates[torch.randperm(candidates.numel(), device=device)[:take]]
                            use_pred[sel] = False

                # build pred cond
                x_curr = _gather_last_observed(x0, mask)
                zc = color_encoder(x_curr)
                embeds_pred = cond_predictor(zc)
                cond_pred = build_cond_from_pred_embeds(conditioner, embeds_pred)

                if cond_true is None:
                    cond_in = cond_pred
                else:
                    cond_in = torch.where(use_pred.view(B, 1), cond_pred, cond_true)

            # color augmentation (only pred-cond samples by default)
            x0_use = _apply_color_aug_selected(x0, selected=use_pred, cfg=color_aug_cfg)

            # diffusion loss
            loss = diffusion_loss(denoiser, schedule, x0=x0_use, obs_mask=mask, cond=cond_in)

            # predicted-embedding supervision
            loss_pred = torch.tensor(0.0, device=device)
            if mm_cfg.enable and conditioner.cond_dim > 0 and color_encoder is not None and cond_predictor is not None:
                x_curr2 = _gather_last_observed(x0_use, mask)
                zc2 = color_encoder(x_curr2)
                embeds_pred2 = cond_predictor(zc2)

                pred_losses = []
                true_r = _get_embed(embeds_true, ("raman", "raman_emb", "raman_feat", "raman_z"))
                true_x = _get_embed(embeds_true, ("xrd", "xrd_emb", "xrd_feat", "xrd_z"))
                pred_r = _get_embed(embeds_pred2, ("raman_emb", "raman", "raman_feat"))
                pred_x = _get_embed(embeds_pred2, ("xrd_emb", "xrd", "xrd_feat"))

                if conditioner.raman_enc is not None and (true_r is not None) and (pred_r is not None):
                    diff = (pred_r - true_r.detach()) ** 2
                    if has_raman is not None:
                        valid = (has_raman == 1)
                        if valid.any():
                            pred_losses.append(diff[valid].mean())
                    else:
                        pred_losses.append(diff.mean())

                if conditioner.xrd_enc is not None and (true_x is not None) and (pred_x is not None):
                    diff = (pred_x - true_x.detach()) ** 2
                    if has_xrd is not None:
                        valid = (has_xrd == 1)
                        if valid.any():
                            pred_losses.append(diff[valid].mean())
                    else:
                        pred_losses.append(diff.mean())

                if pred_losses:
                    loss_pred = torch.stack(pred_losses).mean()
                    loss = loss + float(mm_cfg.lambda_pred) * loss_pred

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()

            global_step += 1
            if global_step % log_every == 0:
                msg = f"[ep {ep:03d} step {global_step:06d}] loss={loss.item():.6f}"
                if mm_cfg.enable and conditioner.cond_dim > 0:
                    msg += f"  pred_mse={loss_pred.item():.6f}"
                msg += f"  use_pred={use_pred.float().mean().item():.2f}"
                msg += f"  drop_eff={float(drop_eff):.2f}"
                print(msg)

        # ---- eval ----
        if ep % eval_every == 0:
            val_loss_true = eval_val_loss(denoiser, conditioner, schedule, dl_val, device, num_batches=eval_num_batches)
            print(f"[ep {ep:03d}] val_loss(true_cond)={val_loss_true:.6f}")

            val_loss_pred: Optional[float] = None
            if mm_cfg.enable and conditioner.cond_dim > 0 and color_encoder is not None and cond_predictor is not None:
                val_loss_pred = eval_val_loss(
                    denoiser,
                    conditioner,
                    schedule,
                    dl_val,
                    device,
                    num_batches=eval_num_batches,
                    cond_override="pred",
                    color_encoder=color_encoder,
                    cond_predictor=cond_predictor,
                )
                print(f"[ep {ep:03d}] val_loss(pred_cond)={val_loss_pred:.6f}")

            # save best_true / best_pred always
            if val_loss_true < best_true - early_stopping_min_delta:
                best_true = float(val_loss_true)
                path = os.path.join(save_dir, "best_true_model.pt")
                _save_ckpt(path, cfg, ep, global_step, best_true, denoiser, conditioner, color_encoder, cond_predictor)
                print(f"[SAVE BEST TRUE] {path}  val_loss_true={best_true:.6f}")

            if val_loss_pred is not None and val_loss_pred < best_pred - early_stopping_min_delta:
                best_pred = float(val_loss_pred)
                path = os.path.join(save_dir, "best_pred_model.pt")
                _save_ckpt(path, cfg, ep, global_step, best_pred, denoiser, conditioner, color_encoder, cond_predictor)
                print(f"[SAVE BEST PRED] {path}  val_loss_pred={best_pred:.6f}")

            # monitor value for best_model / early stopping
            monitor_val = _monitor_value(monitor_metric, val_loss_true, val_loss_pred, mixed_alpha)
            improved = monitor_val < best_monitor - early_stopping_min_delta
            if improved:
                best_monitor = float(monitor_val)
                no_improve_count = 0
                best_path = os.path.join(save_dir, "best_model.pt")
                _save_ckpt(best_path, cfg, ep, global_step, best_monitor, denoiser, conditioner, color_encoder, cond_predictor)
                print(f"[SAVE BEST] {best_path}  monitor({monitor_metric})={best_monitor:.6f}")
            else:
                no_improve_count += 1

            if no_improve_count >= early_stopping_patience:
                print(
                    f"Early stopping after {no_improve_count} epochs without improvement. "
                    f"Best monitor({monitor_metric})={best_monitor:.6f} | best_true={best_true:.6f} | best_pred={best_pred:.6f}"
                )
                break

            if bool(tr_cfg.get("eval_use_sampling", False)):
                sm = eval_sampling_metrics(denoiser, conditioner, schedule, dl_val, device, num_batches=int(tr_cfg.get("eval_sampling_num_batches", 1)))
                print(f"[ep {ep:03d}] sampling true_cond: {sm}")
                if mm_cfg.enable and conditioner.cond_dim > 0 and color_encoder is not None and cond_predictor is not None:
                    sm2 = eval_sampling_metrics(
                        denoiser,
                        conditioner,
                        schedule,
                        dl_val,
                        device,
                        num_batches=int(tr_cfg.get("eval_sampling_num_batches", 1)),
                        cond_override="pred",
                        color_encoder=color_encoder,
                        cond_predictor=cond_predictor,
                    )
                    print(f"[ep {ep:03d}] sampling pred_cond: {sm2}")

        # save periodic
        if ep % save_every == 0 or ep == epochs:
            ckpt = {
                "cfg": cfg,
                "epoch": ep,
                "global_step": global_step,
                "best_monitor": float(best_monitor),
                "best_true": float(best_true),
                "best_pred": float(best_pred),
                "denoiser": denoiser.state_dict(),
                "conditioner": conditioner.state_dict(),
            }
            if color_encoder is not None:
                ckpt["color_encoder"] = color_encoder.state_dict()
            if cond_predictor is not None:
                ckpt["cond_predictor"] = cond_predictor.state_dict()
            out_path = os.path.join(save_dir, f"ckpt_ep{ep}.pt")
            torch.save(ckpt, out_path)
            print(f"[SAVE] {out_path}")


if __name__ == "__main__":
    main()
