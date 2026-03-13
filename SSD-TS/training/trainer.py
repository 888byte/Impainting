from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bridge.condition_builder import (
    build_cond_from_pred_embeds,
    build_true_condition,
    gather_last_observed,
    teacher_posterior,
)
from bridge.distill import embedding_distillation_loss, posterior_kl_loss
from bridge.posterior_head import PosteriorHead, PosteriorHeadConfig
from bridge.prototype_bank import PrototypeBank, build_prototype_bank
from data.dataset import PigmentNPZDataset
from models.color_encoder import ColorEncoder, ColorEncoderConfig
from models.cond_predictor import ColorToSpecPredictor, CondPredictorConfig
from models.denoiser import DenoiserConfig, MambaDenoiser
from models.physics import PhysicsCfg, FadingForwardModelLab, warmup_weight
from models.spectral_encoder import ConditionerConfig, MultimodalConditioner
from training.diffusion import DiffusionConfig, DiffusionSchedule, diffusion_loss, p_sample_loop
from training.samplers import build_parent_sampler
from utils.color_utils import delta_e2000
from utils.config_utils import load_config
from utils.seed import set_seed


@dataclass
class MissingModalityCfg:
    enable: bool = False
    drop_prob: float = 0.3
    lambda_pred: float = 0.1
    freeze_color_encoder: bool = False
    freeze_conditioner: bool = False


def _denorm_lab_torch(x_norm: torch.Tensor) -> torch.Tensor:
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


def _apply_color_aug_selected(x0_norm: torch.Tensor, selected: Optional[torch.Tensor], cfg: Dict[str, Any]) -> torch.Tensor:
    if not bool(cfg.get('enable', False)) or selected is None:
        return x0_norm
    device = x0_norm.device
    batch_size = x0_norm.shape[0]
    selected = selected.to(device).view(batch_size)
    if selected.sum().item() == 0:
        return x0_norm

    prob = float(cfg.get('prob', cfg.get('p', 0.0)))
    if prob <= 0:
        return x0_norm
    gate = (torch.rand(batch_size, device=device) < prob) & selected
    if gate.sum().item() == 0:
        return x0_norm

    idx = torch.nonzero(gate, as_tuple=False).squeeze(1)
    x = x0_norm.clone()
    xb = _denorm_lab_torch(x[idx])

    n = xb.shape[0]
    L_scale = float(cfg.get('L_scale', 0.0))
    L_shift = float(cfg.get('L_shift', 0.0))
    ab_scale = float(cfg.get('ab_scale', 0.0))
    ab_rot = float(cfg.get('ab_rotate_deg', 0.0))
    noise_std = float(cfg.get('noise_std', 0.0))

    if L_scale != 0.0:
        xb[..., 0:1] = xb[..., 0:1] * (1.0 + (torch.rand(n, 1, 1, device=device) * 2.0 - 1.0) * L_scale)
    if L_shift != 0.0:
        xb[..., 0:1] = xb[..., 0:1] + (torch.rand(n, 1, 1, device=device) * 2.0 - 1.0) * L_shift
    if ab_scale != 0.0:
        xb[..., 1:3] = xb[..., 1:3] * (1.0 + (torch.rand(n, 1, 1, device=device) * 2.0 - 1.0) * ab_scale)
    if ab_rot != 0.0:
        theta = (torch.rand(n, 1, 1, device=device) * 2.0 - 1.0) * (ab_rot * np.pi / 180.0)
        c = torch.cos(theta)
        s = torch.sin(theta)
        a = xb[..., 1:2]
        b = xb[..., 2:3]
        xb[..., 1:2] = c * a - s * b
        xb[..., 2:3] = s * a + c * b
    if noise_std != 0.0:
        xb = xb + torch.randn_like(xb) * noise_std

    xb[..., 0] = xb[..., 0].clamp(0.0, 100.0)
    xb[..., 1] = xb[..., 1].clamp(-128.0, 128.0)
    xb[..., 2] = xb[..., 2].clamp(-128.0, 128.0)
    x[idx] = _norm_lab_torch(xb)
    return x


def _get_embed(embeds: Dict[str, torch.Tensor], keys: Tuple[str, ...]) -> Optional[torch.Tensor]:
    for key in keys:
        value = embeds.get(key, None)
        if value is not None:
            return value
    return None


@torch.no_grad()
def eval_val_loss(
    denoiser: nn.Module,
    conditioner: MultimodalConditioner,
    schedule: DiffusionSchedule,
    loader: DataLoader,
    device: torch.device,
    num_batches: int = 2,
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
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= int(num_batches):
            break
        x0 = batch['x0'].to(device)
        mask = batch['mask'].to(device)
        if cond_override == 'pred':
            if color_encoder is None or cond_predictor is None:
                raise ValueError("cond_override='pred' but color_encoder/cond_predictor missing")
            x_curr = gather_last_observed(x0, mask)
            zc = color_encoder(x_curr)
            cond = build_cond_from_pred_embeds(conditioner, cond_predictor(zc))
        else:
            cond, _ = build_true_condition(conditioner, batch, device)
        losses.append(float(diffusion_loss(denoiser, schedule, x0=x0, obs_mask=mask, cond=cond).item()))
    return float(np.mean(losses)) if losses else float('nan')


@torch.no_grad()
def eval_sampling_metrics(
    denoiser: nn.Module,
    conditioner: MultimodalConditioner,
    schedule: DiffusionSchedule,
    loader: DataLoader,
    device: torch.device,
    num_batches: int = 1,
    cond_override: Optional[str] = None,
    color_encoder: Optional[ColorEncoder] = None,
    cond_predictor: Optional[ColorToSpecPredictor] = None,
) -> Dict[str, float]:
    denoiser.eval()
    conditioner.eval()
    if color_encoder is not None:
        color_encoder.eval()
    if cond_predictor is not None:
        cond_predictor.eval()

    all_de = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= int(num_batches):
            break
        x0 = batch['x0'].to(device)
        mask = batch['mask'].to(device)
        if cond_override == 'pred':
            if color_encoder is None or cond_predictor is None:
                raise ValueError("cond_override='pred' but color_encoder/cond_predictor missing")
            x_curr = gather_last_observed(x0, mask)
            zc = color_encoder(x_curr)
            cond = build_cond_from_pred_embeds(conditioner, cond_predictor(zc))
        else:
            cond, _ = build_true_condition(conditioner, batch, device)
        x_sample = p_sample_loop(denoiser, schedule, x_obs=x0 * mask, obs_mask=mask, cond=cond)
        pred0 = _denorm_lab_torch(x_sample[:, 0, :]).detach().cpu().numpy()
        gt0 = _denorm_lab_torch(x0[:, 0, :]).detach().cpu().numpy()
        for pred, gt in zip(pred0, gt0):
            all_de.append(float(delta_e2000(pred, gt)))
    return {
        'deltaE2000_mean': float(np.mean(all_de)) if all_de else float('nan'),
        'deltaE2000_std': float(np.std(all_de)) if all_de else float('nan'),
    }


def _monitor_value(monitor_metric: str, val_true: float, val_pred: Optional[float], mixed_alpha: float) -> float:
    monitor_metric = (monitor_metric or '').lower()
    if monitor_metric == 'pred_cond':
        if val_pred is None or np.isnan(val_pred):
            return float('inf')
        return float(val_pred)
    if monitor_metric == 'mixed':
        if val_pred is None or np.isnan(val_pred):
            return float(val_true)
        alpha = max(0.0, min(1.0, float(mixed_alpha)))
        return float(alpha * val_pred + (1.0 - alpha) * val_true)
    return float(val_true)


@torch.no_grad()
def _load_or_build_prototype_bank(
    cfg: Dict[str, Any],
    conditioner: MultimodalConditioner,
    device: torch.device,
) -> Optional[PrototypeBank]:
    bridge_cfg = cfg.get('bridge', {})
    proto_cfg = bridge_cfg.get('prototype_bank', {})
    bank_path = str(proto_cfg.get('path', '') or '')
    if bank_path and os.path.exists(bank_path):
        return PrototypeBank.load(bank_path)
    train_npz = str(cfg.get('data', {}).get('train_npz', '') or '')
    if not train_npz or not os.path.exists(train_npz):
        return None
    was_training = conditioner.training
    conditioner.eval()
    bank = build_prototype_bank(
        npz_path=train_npz,
        conditioner=conditioner,
        device=device,
        index_csv=str(cfg.get('data', {}).get('train_index', '') or ''),
        batch_size=int(bridge_cfg.get('prototype_batch_size', 256)),
    )
    if was_training:
        conditioner.train()
    if bank_path:
        bank.save(bank_path)
    return bank


def _save_ckpt(
    path: str,
    cfg: Dict[str, Any],
    ep: int,
    global_step: int,
    monitor_val: float,
    denoiser: nn.Module,
    conditioner: nn.Module,
    color_encoder: Optional[nn.Module],
    cond_predictor: Optional[nn.Module],
    fading_model: Optional[nn.Module] = None,
    posterior_head: Optional[nn.Module] = None,
) -> None:
    ckpt = {
        'cfg': cfg,
        'epoch': ep,
        'global_step': global_step,
        'val_loss': float(monitor_val),
        'denoiser': denoiser.state_dict(),
        'conditioner': conditioner.state_dict(),
    }
    if color_encoder is not None:
        ckpt['color_encoder'] = color_encoder.state_dict()
    if cond_predictor is not None:
        ckpt['cond_predictor'] = cond_predictor.state_dict()
    if fading_model is not None:
        ckpt['fading_model'] = fading_model.state_dict()
    if posterior_head is not None:
        ckpt['posterior_head'] = posterior_head.state_dict()
    torch.save(ckpt, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tr_cfg = cfg.get('train', {})
    device = torch.device(tr_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    set_seed(int(tr_cfg.get('seed', 42)))

    train_index = str(cfg.get('data', {}).get('train_index', '') or '')
    val_index = str(cfg.get('data', {}).get('val_index', '') or '')
    train_npz = cfg['data']['train_npz']
    val_npz = cfg['data']['val_npz']
    ds_train = PigmentNPZDataset(train_npz, index_csv=train_index)
    ds_val = PigmentNPZDataset(val_npz, index_csv=val_index)

    bridge_cfg = cfg.get('bridge', {})
    sampler = build_parent_sampler(ds_train.sample_index) if bool(bridge_cfg.get('use_group_sampler', False)) else None
    batch_size = int(tr_cfg.get('batch_size', 64))
    num_workers = int(tr_cfg.get('num_workers', 0))
    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=int(tr_cfg.get('eval_batch_size', batch_size)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    diff_cfg_raw = cfg.get('diffusion', {}) or {}
    diff_cfg = DiffusionConfig(
        T=int(diff_cfg_raw.get('T', 200)),
        beta_0=float(diff_cfg_raw.get('beta_0', diff_cfg_raw.get('beta_start', 1e-4))),
        beta_T=float(diff_cfg_raw.get('beta_T', diff_cfg_raw.get('beta_end', 0.02))),
    )
    schedule = DiffusionSchedule(diff_cfg, device=device)

    mod_cfg = cfg.get('modality', {})
    cond_cfg = ConditionerConfig(
        use_raman=bool(mod_cfg.get('use_raman', False)),
        use_xrd=bool(mod_cfg.get('use_xrd', False)),
        raman_len=int(mod_cfg.get('raman_len', 1024)),
        xrd_len=int(mod_cfg.get('xrd_len', 2048)),
        d_model=int(mod_cfg.get('spec_d_model', 128)),
        n_layers=int(mod_cfg.get('spec_n_layers', 4)),
        dropout=float(mod_cfg.get('spec_dropout', 0.0)),
        raman_peak_dim=int(mod_cfg.get('raman_peak_dim', 0)),
        xrd_peak_dim=int(mod_cfg.get('xrd_peak_dim', 0)),
        use_fuse=bool(mod_cfg.get('use_fuse', True)),
    )
    conditioner = MultimodalConditioner(cond_cfg).to(device)

    model_cfg = cfg.get('model', {})
    denoiser = MambaDenoiser(
        DenoiserConfig(
            in_channels=int(model_cfg.get('in_channels', 3)),
            hidden_dim=int(model_cfg.get('hidden_dim', 128)),
            n_layers=int(model_cfg.get('n_layers', 4)),
            dropout=float(model_cfg.get('dropout', 0.0)),
            cond_dim=int(conditioner.cond_dim),
        )
    ).to(device)

    mm_cfg_raw = cfg.get('missing_modality', {}) or {}
    mm_cfg = MissingModalityCfg(
        enable=bool(mm_cfg_raw.get('enable', False)),
        drop_prob=float(mm_cfg_raw.get('drop_prob', 0.3)),
        lambda_pred=float(mm_cfg_raw.get('lambda_pred', 0.1)),
        freeze_color_encoder=bool(mm_cfg_raw.get('freeze_color_encoder', False)),
        freeze_conditioner=bool(mm_cfg_raw.get('freeze_conditioner', False)),
    )

    bridge_mode = str(bridge_cfg.get('mode', 'pred'))
    needs_color_branch = bool(mm_cfg.enable or (bool(bridge_cfg.get('enable', False)) and bridge_mode in {'posterior', 'posterior_retrieval'}))
    needs_predictor = bool(mm_cfg.enable or (bool(bridge_cfg.get('enable', False)) and bridge_mode in {'posterior_retrieval'}))

    color_encoder: Optional[ColorEncoder] = None
    cond_predictor: Optional[ColorToSpecPredictor] = None
    if conditioner.cond_dim > 0 and needs_color_branch:
        ce_cfg = ColorEncoderConfig(
            in_dim=3,
            d_model=int(mm_cfg_raw.get('color_d_model', cond_cfg.d_model)),
            hidden_dim=int(mm_cfg_raw.get('color_hidden_dim', 256)),
            n_layers=int(mm_cfg_raw.get('color_n_layers', 2)),
            dropout=float(mm_cfg_raw.get('color_dropout', 0.0)),
        )
        color_encoder = ColorEncoder(ce_cfg).to(device)
        if needs_predictor:
            cp_cfg = CondPredictorConfig(
                in_dim=ce_cfg.d_model,
                d_model=cond_cfg.d_model,
                use_raman=bool(cond_cfg.use_raman),
                use_xrd=bool(cond_cfg.use_xrd),
                hidden_dim=int(mm_cfg_raw.get('pred_hidden_dim', 256)),
                n_layers=int(mm_cfg_raw.get('pred_n_layers', 2)),
                dropout=float(mm_cfg_raw.get('pred_dropout', 0.0)),
            )
            cond_predictor = ColorToSpecPredictor(cp_cfg).to(device)
        if mm_cfg.freeze_color_encoder:
            for p in color_encoder.parameters():
                p.requires_grad = False
        if mm_cfg.freeze_conditioner:
            for p in conditioner.parameters():
                p.requires_grad = False

    pretrained_cfg = cfg.get('pretrained', {}) or {}
    alignment_ckpt = str(pretrained_cfg.get('alignment_ckpt', '') or '')
    if alignment_ckpt and os.path.exists(alignment_ckpt):
        sd = torch.load(alignment_ckpt, map_location='cpu')
        if color_encoder is not None and isinstance(sd, dict) and 'color_encoder' in sd:
            color_encoder.load_state_dict(sd['color_encoder'], strict=False)
        if conditioner.raman_enc is not None and isinstance(sd, dict) and 'raman_encoder' in sd:
            conditioner.raman_enc.load_state_dict(sd['raman_encoder'], strict=False)

    raman_encoder_ckpt = str(pretrained_cfg.get('raman_encoder_ckpt', '') or '')
    if raman_encoder_ckpt and os.path.exists(raman_encoder_ckpt):
        sd = torch.load(raman_encoder_ckpt, map_location='cpu')
        conditioner.load_state_dict(sd.get('conditioner', sd), strict=False)

    phys_raw = cfg.get('physics', {}) or {}
    phys_cfg = PhysicsCfg(
        enable=bool(phys_raw.get('enable', False)),
        lambda_cycle=float(phys_raw.get('lambda_cycle', 0.2)),
        warmup_steps=int(phys_raw.get('warmup_steps', 2000)),
        t_max=int(phys_raw.get('t_max', 30)),
        exclude_t0=bool(phys_raw.get('exclude_t0', True)),
        cond_dependent=bool(phys_raw.get('cond_dependent', True)),
        cond_hidden=int(phys_raw.get('cond_hidden', 128)),
        per_channel_k=bool(phys_raw.get('per_channel_k', True)),
        learn_c_inf=bool(phys_raw.get('learn_c_inf', True)),
        init_k=float(phys_raw.get('init_k', 1.0)),
    )
    fading_model: Optional[nn.Module] = None
    if phys_cfg.enable:
        fading_model = FadingForwardModelLab(cond_dim=int(conditioner.cond_dim), cfg=phys_cfg).to(device)

    prototype_bank: Optional[PrototypeBank] = None
    posterior_head: Optional[PosteriorHead] = None
    bridge_enabled = bool(bridge_cfg.get('enable', False)) and conditioner.cond_dim > 0 and color_encoder is not None
    if bridge_enabled and bridge_mode in {'posterior', 'posterior_retrieval'}:
        prototype_bank = _load_or_build_prototype_bank(cfg, conditioner, device)
        if prototype_bank is not None:
            posterior_head = PosteriorHead(
                PosteriorHeadConfig(
                    in_dim=color_encoder.cfg.d_model,
                    num_prototypes=prototype_bank.num_prototypes,
                    hidden_dim=int(bridge_cfg.get('hidden_dim', 256)),
                    n_layers=int(bridge_cfg.get('n_layers', 2)),
                    dropout=float(bridge_cfg.get('dropout', 0.0)),
                )
            ).to(device)

    params = list(denoiser.parameters()) + list(conditioner.parameters())
    if color_encoder is not None:
        params += list(color_encoder.parameters())
    if cond_predictor is not None:
        params += list(cond_predictor.parameters())
    if fading_model is not None:
        params += list(fading_model.parameters())
    if posterior_head is not None:
        params += list(posterior_head.parameters())

    opt = torch.optim.AdamW(
        params,
        lr=float(tr_cfg.get('lr', 1e-3)),
        weight_decay=float(tr_cfg.get('weight_decay', 0.0)),
    )

    grad_clip = float(tr_cfg.get('grad_clip', 1.0))
    epochs = int(tr_cfg.get('epochs', 200))
    log_every = int(tr_cfg.get('log_every', 50))
    eval_every = int(tr_cfg.get('eval_every', 1))
    eval_num_batches = int(tr_cfg.get('eval_num_batches', 2))
    save_every = int(tr_cfg.get('save_every', 10))
    save_dir = str(tr_cfg.get('save_dir', 'ckpt/default'))
    os.makedirs(save_dir, exist_ok=True)

    color_aug_cfg = cfg.get('color_aug', {}) or {}
    monitor_metric = str(tr_cfg.get('monitor_metric', '')).strip().lower() or ('pred_cond' if mm_cfg.enable and cond_predictor is not None else 'true_cond')
    mixed_alpha = float(tr_cfg.get('monitor_mixed_alpha', 0.7))
    early_stopping_patience = int(tr_cfg.get('early_stopping_patience', 100))
    early_stopping_min_delta = float(tr_cfg.get('early_stopping_min_delta', 0.001))

    best_monitor = float('inf')
    best_true = float('inf')
    best_pred = float('inf')
    no_improve_count = 0
    global_step = 0

    print(f'[INFO] monitor_metric={monitor_metric} bridge_mode={bridge_mode} group_sampler={sampler is not None}')

    for ep in range(1, epochs + 1):
        if bridge_enabled and bool(bridge_cfg.get('refresh_each_epoch', False)) and posterior_head is not None and ep > 1:
            prototype_bank = _load_or_build_prototype_bank(cfg, conditioner, device)

        denoiser.train()
        conditioner.train()
        if color_encoder is not None:
            color_encoder.train()
        if cond_predictor is not None:
            cond_predictor.train()
        if posterior_head is not None:
            posterior_head.train()
        if fading_model is not None:
            fading_model.train()

        for batch in dl_train:
            x0 = batch['x0'].to(device)
            mask = batch['mask'].to(device)
            cond_true, embeds_true = build_true_condition(conditioner, batch, device)

            batch_size_now = x0.shape[0]
            use_pred = torch.zeros(batch_size_now, device=device, dtype=torch.bool)
            has_raman = batch.get('has_raman', None)
            has_xrd = batch.get('has_xrd', None)
            if has_raman is not None:
                has_raman = has_raman.to(device).view(batch_size_now).long()
            if has_xrd is not None:
                has_xrd = has_xrd.to(device).view(batch_size_now).long()
            if 'raman' in batch:
                inferred = (batch['raman'].to(device).abs().sum(dim=1) > 1e-6).long()
                has_raman = inferred if has_raman is None else torch.maximum(has_raman, inferred)
            if 'xrd' in batch:
                inferred = (batch['xrd'].to(device).abs().sum(dim=1) > 1e-6).long()
                has_xrd = inferred if has_xrd is None else torch.maximum(has_xrd, inferred)

            x_curr = gather_last_observed(x0, mask)
            zc = color_encoder(x_curr) if color_encoder is not None else None
            embeds_pred = cond_predictor(zc) if (zc is not None and cond_predictor is not None) else None
            cond_pred = build_cond_from_pred_embeds(conditioner, embeds_pred) if embeds_pred is not None else None

            if cond_true is None:
                use_pred[:] = True
            elif mm_cfg.enable and cond_pred is not None:
                if conditioner.raman_enc is not None and has_raman is not None:
                    use_pred |= (has_raman == 0)
                if conditioner.xrd_enc is not None and has_xrd is not None:
                    use_pred |= (has_xrd == 0)
                drop_p = float(mm_cfg.drop_prob)
                warmup_ep = int(mm_cfg_raw.get('drop_prob_warmup_epochs', 10))
                allow_all_pred = bool(mm_cfg_raw.get('allow_all_pred', False))
                drop_cap = float(mm_cfg_raw.get('drop_prob_cap', 0.8))
                if not allow_all_pred:
                    drop_p = min(drop_p, drop_cap)
                drop_eff = drop_p * min(1.0, float(ep) / float(max(warmup_ep, 1))) if warmup_ep > 0 else drop_p
                forced_pred = use_pred.clone()
                use_pred |= (torch.rand(batch_size_now, device=device) < drop_eff)
                min_true_frac = float(mm_cfg_raw.get('min_true_frac', 0.2))
                if cond_true is not None and not allow_all_pred and min_true_frac > 0:
                    cur_true = float((~use_pred).float().mean().item())
                    if cur_true < min_true_frac:
                        candidates = torch.nonzero(~forced_pred, as_tuple=False).view(-1)
                        if candidates.numel() > 0:
                            need = max(1, int(np.ceil((min_true_frac - cur_true) * batch_size_now)))
                            sel = candidates[torch.randperm(candidates.numel(), device=device)[: min(need, int(candidates.numel()))]]
                            use_pred[sel] = False
            else:
                drop_eff = 0.0

            cond_post = None
            loss_bridge = torch.tensor(0.0, device=device)
            loss_distill = torch.tensor(0.0, device=device)
            if posterior_head is not None and prototype_bank is not None and zc is not None and cond_true is not None:
                posterior_temp = float(bridge_cfg.get('posterior_temp', 0.07))
                teacher_temp = float(bridge_cfg.get('teacher_temp', posterior_temp))
                logits = posterior_head(zc)
                logits_scaled = logits / posterior_temp
                teacher_probs = teacher_posterior(
                    cond_true.detach(),
                    prototype_bank,
                    device=device,
                    temp=teacher_temp,
                    normalize=bool(bridge_cfg.get('prototype_bank', {}).get('normalize', True)),
                )
                weights = torch.softmax(logits_scaled, dim=-1)
                top_k = int(bridge_cfg.get('prototype_top_k', 0))
                if top_k > 0 and top_k < weights.shape[-1]:
                    topv, topi = torch.topk(weights, k=top_k, dim=-1)
                    sparse = torch.zeros_like(weights)
                    sparse.scatter_(1, topi, topv)
                    weights = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                cond_post = prototype_bank.aggregate(weights, device=device)
                loss_bridge = posterior_kl_loss(logits_scaled, teacher_probs)
                if bool(bridge_cfg.get('use_distill', False)):
                    loss_distill = embedding_distillation_loss(cond_post, cond_true.detach())

            pseudo_cond = cond_pred
            if cond_post is not None and bridge_mode in {'posterior', 'posterior_retrieval'}:
                pseudo_cond = cond_post

            if cond_true is None:
                cond_in = pseudo_cond
            elif pseudo_cond is None:
                cond_in = cond_true
            else:
                cond_in = torch.where(use_pred.view(batch_size_now, 1), pseudo_cond, cond_true)

            x0_use = _apply_color_aug_selected(x0, selected=use_pred if pseudo_cond is not None else None, cfg=color_aug_cfg)

            loss_phys = torch.tensor(0.0, device=device)
            if fading_model is not None and phys_cfg.enable:
                diff_loss, x0_pred, t = diffusion_loss(
                    denoiser,
                    schedule,
                    x0=x0_use,
                    obs_mask=mask,
                    cond=cond_in,
                    return_x0_pred=True,
                    return_t=True,
                )
                loss_phys = fading_model.cycle_loss(x0_pred=x0_pred, x0_true=x0_use, mask=mask, cond=cond_in, t=t)
                loss = diff_loss + float(phys_cfg.lambda_cycle) * warmup_weight(global_step, phys_cfg.warmup_steps) * loss_phys
            else:
                loss = diffusion_loss(denoiser, schedule, x0=x0_use, obs_mask=mask, cond=cond_in)

            loss_pred = torch.tensor(0.0, device=device)
            if mm_cfg.enable and embeds_pred is not None and cond_true is not None:
                pred_losses = []
                true_r = _get_embed(embeds_true, ('raman', 'raman_emb', 'raman_feat', 'raman_z'))
                true_x = _get_embed(embeds_true, ('xrd', 'xrd_emb', 'xrd_feat', 'xrd_z'))
                pred_r = _get_embed(embeds_pred, ('raman_emb', 'raman', 'raman_feat'))
                pred_x = _get_embed(embeds_pred, ('xrd_emb', 'xrd', 'xrd_feat'))
                if conditioner.raman_enc is not None and true_r is not None and pred_r is not None:
                    diff = (pred_r - true_r.detach()) ** 2
                    pred_losses.append(diff[(has_raman == 1)] .mean() if has_raman is not None and (has_raman == 1).any() else diff.mean())
                if conditioner.xrd_enc is not None and true_x is not None and pred_x is not None:
                    diff = (pred_x - true_x.detach()) ** 2
                    pred_losses.append(diff[(has_xrd == 1)] .mean() if has_xrd is not None and (has_xrd == 1).any() else diff.mean())
                if pred_losses:
                    loss_pred = torch.stack(pred_losses).mean()
                    loss = loss + float(mm_cfg.lambda_pred) * loss_pred

            if posterior_head is not None and prototype_bank is not None:
                loss = loss + float(bridge_cfg.get('loss_weight', 1.0)) * loss_bridge
                if bool(bridge_cfg.get('use_distill', False)):
                    loss = loss + float(bridge_cfg.get('distill_weight', 0.1)) * loss_distill

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            global_step += 1

            if global_step % log_every == 0:
                msg = f"[ep {ep:03d} step {global_step:06d}] loss={loss.item():.6f}"
                if mm_cfg.enable and cond_predictor is not None:
                    msg += f" pred_mse={loss_pred.item():.6f}"
                if posterior_head is not None and prototype_bank is not None:
                    msg += f" posterior_kl={loss_bridge.item():.6f}"
                    if bool(bridge_cfg.get('use_distill', False)):
                        msg += f" distill={loss_distill.item():.6f}"
                if fading_model is not None and phys_cfg.enable:
                    msg += f" phys={loss_phys.item():.6f}"
                msg += f" use_pred={use_pred.float().mean().item():.2f}"
                print(msg)

        if ep % eval_every == 0:
            val_loss_true = eval_val_loss(denoiser, conditioner, schedule, dl_val, device, num_batches=eval_num_batches)
            print(f'[ep {ep:03d}] val_loss(true_cond)={val_loss_true:.6f}')
            val_loss_pred = None
            if cond_predictor is not None and color_encoder is not None:
                val_loss_pred = eval_val_loss(
                    denoiser,
                    conditioner,
                    schedule,
                    dl_val,
                    device,
                    num_batches=eval_num_batches,
                    cond_override='pred',
                    color_encoder=color_encoder,
                    cond_predictor=cond_predictor,
                )
                print(f'[ep {ep:03d}] val_loss(pred_cond)={val_loss_pred:.6f}')

            if val_loss_true < best_true - early_stopping_min_delta:
                best_true = float(val_loss_true)
                _save_ckpt(os.path.join(save_dir, 'best_true_model.pt'), cfg, ep, global_step, best_true, denoiser, conditioner, color_encoder, cond_predictor, fading_model, posterior_head)
            if val_loss_pred is not None and val_loss_pred < best_pred - early_stopping_min_delta:
                best_pred = float(val_loss_pred)
                _save_ckpt(os.path.join(save_dir, 'best_pred_model.pt'), cfg, ep, global_step, best_pred, denoiser, conditioner, color_encoder, cond_predictor, fading_model, posterior_head)

            monitor_val = _monitor_value(monitor_metric, val_loss_true, val_loss_pred, mixed_alpha)
            if monitor_val < best_monitor - early_stopping_min_delta:
                best_monitor = float(monitor_val)
                no_improve_count = 0
                _save_ckpt(os.path.join(save_dir, 'best_model.pt'), cfg, ep, global_step, best_monitor, denoiser, conditioner, color_encoder, cond_predictor, fading_model, posterior_head)
            else:
                no_improve_count += 1

            if bool(tr_cfg.get('eval_use_sampling', False)):
                stats = eval_sampling_metrics(denoiser, conditioner, schedule, dl_val, device, num_batches=int(tr_cfg.get('eval_sampling_num_batches', 1)))
                print(f'[ep {ep:03d}] sampling true_cond: {stats}')
                if cond_predictor is not None and color_encoder is not None:
                    stats_pred = eval_sampling_metrics(
                        denoiser,
                        conditioner,
                        schedule,
                        dl_val,
                        device,
                        num_batches=int(tr_cfg.get('eval_sampling_num_batches', 1)),
                        cond_override='pred',
                        color_encoder=color_encoder,
                        cond_predictor=cond_predictor,
                    )
                    print(f'[ep {ep:03d}] sampling pred_cond: {stats_pred}')

            if no_improve_count >= early_stopping_patience:
                print(f'Early stopping after {no_improve_count} epochs without improvement.')
                break

        if ep % save_every == 0 or ep == epochs:
            _save_ckpt(
                os.path.join(save_dir, f'ckpt_ep{ep}.pt'),
                cfg,
                ep,
                global_step,
                best_monitor,
                denoiser,
                conditioner,
                color_encoder,
                cond_predictor,
                fading_model,
                posterior_head,
            )


if __name__ == '__main__':
    main()
