"""Inference pipeline and evaluation entrypoints for pigment restoration."""
from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Dict, Optional

import numpy as np
import torch

from bridge.condition_builder import (
    build_posterior_condition,
    build_posterior_retrieval_condition,
    build_pred_condition,
    build_retrieval_condition,
    build_true_condition,
    gather_last_observed,
)
from bridge.physics_heads import DamageHead, DamageHeadConfig, SpecColorHead, SpecColorHeadConfig
from bridge.posterior_head import PosteriorHead, PosteriorHeadConfig
from bridge.prototype_bank import PrototypeBank
from data.dataset import PigmentNPZDataset
from inference.uncertainty import sample_with_confidence
from models.color_encoder import ColorEncoder, ColorEncoderConfig
from models.cond_predictor import ColorToSpecPredictor, CondPredictorConfig
from models.denoiser import DenoiserConfig, MambaDenoiser
from models.spectral_encoder import ConditionerConfig, MultimodalConditioner
from training.diffusion import DiffusionConfig, DiffusionSchedule, p_sample_loop
from utils.color_utils import LabNorm, delta_e2000, lab_to_rgb, rgb_to_lab
from utils.config_utils import normalize_config

CONF_BETA = 4.0
CONF_SOFT_MIN = 0.35
CONF_RET_BASE = 0.85


def _rts_smoother_random_walk(y: np.ndarray, r: np.ndarray, q: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64).reshape(3,)
    length = int(y.shape[0])
    x_f = np.zeros((length, 3), dtype=np.float64)
    p_f = np.zeros((length, 3), dtype=np.float64)
    x = y[0].copy()
    p = np.maximum(r[0].copy(), 1e-6)
    for t in range(length):
        if t > 0:
            p = p + q
        s = p + np.maximum(r[t], 1e-6)
        k = p / s
        x = x + k * (y[t] - x)
        p = (1.0 - k) * p
        x_f[t] = x
        p_f[t] = p
    x_s = x_f.copy()
    p_s = p_f.copy()
    for t in range(length - 2, -1, -1):
        denom = p_f[t] + q
        c = np.where(denom > 1e-12, p_f[t] / denom, 0.0)
        x_s[t] = x_f[t] + c * (x_s[t + 1] - x_f[t])
        p_s[t] = p_f[t] + c * (p_s[t + 1] - (p_f[t] + q))
    return x_s


def _scalar_from_info(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().view(-1)[0].item())
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return float(value.reshape(-1)[0])
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_retrieval_confidence(info: Dict[str, object]) -> Optional[float]:
    if isinstance(info.get('retrieval', None), dict):
        return _scalar_from_info(info['retrieval'].get('confidence', None))
    return _scalar_from_info(info.get('confidence', None))


def _fuse_confidence(c_diff: Optional[float], std_norm: Optional[float], c_ret: Optional[float]) -> Optional[float]:
    if c_diff is None:
        return None
    s = max(0.0, float(std_norm or 0.0))
    pen = np.exp(-CONF_BETA * s)
    pen = CONF_SOFT_MIN + (1.0 - CONF_SOFT_MIN) * pen
    if c_ret is None:
        ret_factor = 1.0
    else:
        c_ret = max(0.0, min(1.0, float(c_ret)))
        ret_factor = CONF_RET_BASE + (1.0 - CONF_RET_BASE) * c_ret
    return float(max(0.0, min(1.0, float(c_diff) * pen * ret_factor)))




def _build_damage_features(bundle: Dict[str, object], x_curr: torch.Tensor, cond: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    color_encoder = bundle.get('color_encoder', None)
    conditioner = bundle['conditioner']
    if color_encoder is None:
        return None
    zc = color_encoder(x_curr)
    cond_dim = int(conditioner.cond_dim)
    if cond_dim <= 0:
        return zc
    if cond is None:
        cond = torch.zeros(zc.shape[0], cond_dim, device=zc.device, dtype=zc.dtype)
    return torch.cat([zc, cond], dim=-1)


def _physics_diagnostics(
    bundle: Dict[str, object],
    x_curr: torch.Tensor,
    cond: Optional[torch.Tensor],
    pred_lab0: np.ndarray,
    info: Dict[str, object],
) -> Dict[str, np.ndarray]:
    diagnostics: Dict[str, np.ndarray] = {}
    entropy = info.get('entropy', None)
    if entropy is None and isinstance(info.get('posterior', None), dict):
        entropy = info['posterior'].get('entropy', None)
    if isinstance(entropy, torch.Tensor):
        diagnostics['posterior_entropy'] = entropy.detach().cpu().numpy()

    confidence = info.get('confidence', None)
    if isinstance(confidence, torch.Tensor):
        diagnostics['bridge_confidence'] = confidence.detach().cpu().numpy()

    spec_color_head = bundle.get('spec_color_head', None)
    if spec_color_head is not None and cond is not None:
        lab_norm = LabNorm()
        pred_norm = torch.from_numpy(lab_norm.normalize(np.asarray(pred_lab0, dtype=np.float32)).astype(np.float32)).to(x_curr.device)
        aux_color = spec_color_head(cond)
        agreement = torch.exp(-torch.abs(aux_color - pred_norm).mean(dim=-1)).detach().cpu().numpy()
        diagnostics['spec_color_agreement'] = agreement

    damage_head = bundle.get('damage_head', None)
    if damage_head is not None:
        features = _build_damage_features(bundle, x_curr, cond)
        if features is not None:
            diagnostics['damage_score'] = damage_head(features).detach().cpu().numpy()

    return diagnostics


def load_checkpoint(ckpt_path: str, device: torch.device, prototype_bank_path: str = "") -> Dict[str, object]:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    cfg = normalize_config(ckpt.get('cfg', {}))

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
    if 'conditioner' in ckpt:
        conditioner.load_state_dict(ckpt['conditioner'], strict=False)
    conditioner.eval()

    den_cfg = DenoiserConfig(
        in_channels=int(cfg.get('model', {}).get('in_channels', 3)),
        hidden_dim=int(cfg.get('model', {}).get('hidden_dim', 128)),
        n_layers=int(cfg.get('model', {}).get('n_layers', 4)),
        dropout=float(cfg.get('model', {}).get('dropout', 0.0)),
        cond_dim=int(conditioner.cond_dim),
    )
    denoiser = MambaDenoiser(den_cfg).to(device)
    if 'denoiser' in ckpt:
        denoiser.load_state_dict(ckpt['denoiser'], strict=False)
    denoiser.eval()

    color_encoder = None
    cond_predictor = None
    missing_cfg = cfg.get('missing_modality', {})
    if 'color_encoder' in ckpt and 'cond_predictor' in ckpt:
        ce_cfg = ColorEncoderConfig(
            in_dim=3,
            d_model=int(missing_cfg.get('color_d_model', cond_cfg.d_model)),
            hidden_dim=int(missing_cfg.get('color_hidden_dim', 256)),
            n_layers=int(missing_cfg.get('color_n_layers', 2)),
            dropout=float(missing_cfg.get('color_dropout', 0.0)),
        )
        color_encoder = ColorEncoder(ce_cfg).to(device)
        color_encoder.load_state_dict(ckpt['color_encoder'], strict=False)
        color_encoder.eval()

        cp_cfg = CondPredictorConfig(
            in_dim=ce_cfg.d_model,
            d_model=cond_cfg.d_model,
            use_raman=bool(cond_cfg.use_raman),
            use_xrd=bool(cond_cfg.use_xrd),
            hidden_dim=int(missing_cfg.get('pred_hidden_dim', 256)),
            n_layers=int(missing_cfg.get('pred_n_layers', 2)),
            dropout=float(missing_cfg.get('pred_dropout', 0.0)),
        )
        cond_predictor = ColorToSpecPredictor(cp_cfg).to(device)
        cond_predictor.load_state_dict(ckpt['cond_predictor'], strict=False)
        cond_predictor.eval()

    diff_cfg = DiffusionConfig(
        T=int(cfg.get('diffusion', {}).get('T', 200)),
        beta_0=float(cfg.get('diffusion', {}).get('beta_0', 1e-4)),
        beta_T=float(cfg.get('diffusion', {}).get('beta_T', 0.02)),
    )
    schedule = DiffusionSchedule(diff_cfg, device=device)

    bridge_cfg = cfg.get('bridge', {})
    bank_path = prototype_bank_path or bridge_cfg.get('prototype_bank', {}).get('path', '')
    prototype_bank = PrototypeBank.load(bank_path) if bank_path and os.path.exists(bank_path) else None
    posterior_head = None
    if color_encoder is not None and prototype_bank is not None and (('posterior_head' in ckpt) or bridge_cfg.get('enable', False)):
        ph_cfg = PosteriorHeadConfig(
            in_dim=color_encoder.cfg.d_model,
            num_prototypes=prototype_bank.num_prototypes,
            hidden_dim=int(bridge_cfg.get('hidden_dim', 256)),
            n_layers=int(bridge_cfg.get('n_layers', 2)),
            dropout=float(bridge_cfg.get('dropout', 0.0)),
        )
        posterior_head = PosteriorHead(ph_cfg).to(device)
        if 'posterior_head' in ckpt:
            posterior_head.load_state_dict(ckpt['posterior_head'], strict=False)
        posterior_head.eval()

    physics_cfg = cfg.get('physics', {})
    physics_enabled = bool(physics_cfg.get('enable', False))

    spec_color_head = None
    if physics_enabled and bool(physics_cfg.get('use_spec_color_consistency', False)):
        if 'spec_color_head' in ckpt and conditioner.cond_dim > 0:
            spec_color_head = SpecColorHead(
                SpecColorHeadConfig(
                    in_dim=int(conditioner.cond_dim),
                    hidden_dim=int(physics_cfg.get('cond_hidden', 128)),
                    n_layers=2,
                    dropout=0.0,
                )
            ).to(device)
            spec_color_head.load_state_dict(ckpt['spec_color_head'], strict=False)
            spec_color_head.eval()
        else:
            warnings.warn('physics.use_spec_color_consistency is enabled but checkpoint has no spec_color_head; disabling diagnostics.')

    damage_head = None
    if physics_enabled and bool(physics_cfg.get('use_damage_constraint', False)):
        if color_encoder is None:
            warnings.warn('physics.use_damage_constraint is enabled but checkpoint has no color_encoder; disabling diagnostics.')
        elif 'damage_head' in ckpt:
            damage_head = DamageHead(
                DamageHeadConfig(
                    in_dim=int(color_encoder.cfg.d_model) + int(conditioner.cond_dim),
                    hidden_dim=int(physics_cfg.get('cond_hidden', 128)),
                    n_layers=2,
                    dropout=0.0,
                )
            ).to(device)
            damage_head.load_state_dict(ckpt['damage_head'], strict=False)
            damage_head.eval()
        else:
            warnings.warn('physics.use_damage_constraint is enabled but checkpoint has no damage_head; disabling diagnostics.')

    return {
        'cfg': cfg,
        'denoiser': denoiser,
        'conditioner': conditioner,
        'schedule': schedule,
        'color_encoder': color_encoder,
        'cond_predictor': cond_predictor,
        'posterior_head': posterior_head,
        'prototype_bank': prototype_bank,
        'spec_color_head': spec_color_head,
        'damage_head': damage_head,
    }


def _resolve_condition(bundle: Dict[str, object], batch: Optional[Dict[str, torch.Tensor]], x_curr: torch.Tensor, device: torch.device, cond_method: str, library_npz: Optional[str], retrieval_k: int, retrieval_temp: float):
    conditioner = bundle['conditioner']
    color_encoder = bundle['color_encoder']
    cond_predictor = bundle['cond_predictor']
    posterior_head = bundle['posterior_head']
    prototype_bank = bundle['prototype_bank']
    bridge_cfg = bundle['cfg'].get('bridge', {})

    info: Dict[str, object] = {}
    if conditioner.cond_dim == 0:
        return None, info
    if cond_method == 'auto':
        cond_method = bridge_cfg.get('mode', 'pred') if bridge_cfg.get('enable', False) else 'pred'
    if cond_method == 'true':
        if batch is None:
            raise ValueError('true condition requires batch spectra')
        cond, _ = build_true_condition(conditioner, batch, device)
        return cond, info
    if cond_method == 'pred':
        if color_encoder is None or cond_predictor is None:
            raise ValueError('pred condition requires color_encoder and cond_predictor')
        cond, embeds = build_pred_condition(x_curr, conditioner, color_encoder, cond_predictor)
        info['embeds'] = embeds
        return cond, info
    if cond_method == 'retrieval':
        if color_encoder is None or not library_npz:
            raise ValueError('retrieval condition requires color_encoder and library_npz')
        return build_retrieval_condition(x_curr, conditioner, color_encoder, cond_predictor, library_npz, device, retrieval_k=retrieval_k, retrieval_temp=retrieval_temp)
    if cond_method == 'posterior':
        if color_encoder is None or posterior_head is None or prototype_bank is None:
            raise ValueError('posterior condition requires color_encoder, posterior_head, and prototype_bank')
        return build_posterior_condition(x_curr, color_encoder, posterior_head, prototype_bank, device, top_k=int(bridge_cfg.get('prototype_top_k', 0)), temp=float(bridge_cfg.get('posterior_temp', 0.07)))
    if cond_method == 'posterior_retrieval':
        if color_encoder is None or posterior_head is None or prototype_bank is None or not library_npz:
            raise ValueError('posterior_retrieval requires color_encoder, posterior_head, prototype_bank, and library_npz')
        return build_posterior_retrieval_condition(
            x_curr_norm_lab=x_curr,
            conditioner=conditioner,
            color_encoder=color_encoder,
            cond_predictor=cond_predictor,
            posterior_head=posterior_head,
            prototype_bank=prototype_bank,
            library_npz=library_npz,
            device=device,
            retrieval_k=retrieval_k,
            retrieval_temp=retrieval_temp,
            top_k=int(bridge_cfg.get('prototype_top_k', 0)),
            temp=float(bridge_cfg.get('posterior_temp', 0.07)),
        )
    raise ValueError(f'Unknown cond_method: {cond_method}')


@torch.no_grad()
def evaluate_test(bundle: Dict[str, object], test_npz: str, device: torch.device, cond_method: str, library_npz: Optional[str] = None, retrieval_k: int = 5, retrieval_temp: float = 0.07, num_samples: int = 1, max_batches: int = 50, kalman_refine: bool = False, kalman_meas_std_lab: float = 1.0, kalman_process_std_lab: float = 2.0) -> Dict[str, float]:
    ds = PigmentNPZDataset(test_npz)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, drop_last=False)
    lab_norm = LabNorm()
    de_list = []
    conf_list = []
    diag_lists: Dict[str, list] = {}

    for bi, batch in enumerate(dl):
        if bi >= int(max_batches):
            break
        x0 = batch['x0'].to(device)
        mask = batch['mask'].to(device)
        x_curr = gather_last_observed(x0, mask)
        cond, info = _resolve_condition(bundle, batch, x_curr, device, cond_method, library_npz, retrieval_k, retrieval_temp)
        effective_num_samples = int(num_samples)
        if kalman_refine and effective_num_samples <= 1:
            effective_num_samples = 8
        sample_info = {'conf_diffusion': None, 'diffusion_std_norm_meanL2': None, 'num_samples': effective_num_samples}
        if effective_num_samples <= 1:
            x_obs = x0 * mask
            x_sample = p_sample_loop(bundle['denoiser'], bundle['schedule'], x_obs=x_obs, obs_mask=mask, cond=cond)
            pred0 = lab_norm.denormalize(x_sample[:, 0, :].detach().cpu().numpy())
            gt0 = lab_norm.denormalize(x0[:, 0, :].detach().cpu().numpy())
        else:
            pred0, std0, sample_info = sample_with_confidence(bundle['denoiser'], bundle['schedule'], x0, mask, cond, num_samples=effective_num_samples)
            gt0 = lab_norm.denormalize(x0[:, 0, :].detach().cpu().numpy())
            if kalman_refine:
                x0_den = lab_norm.denormalize(x0.detach().cpu().numpy())
                mask_np = mask.detach().cpu().numpy()
                batch_size, length, _ = x0_den.shape
                q = np.asarray([kalman_process_std_lab ** 2] * 3, dtype=np.float64)
                refined = []
                for i in range(batch_size):
                    y = x0_den[i].copy()
                    y[0] = pred0[i]
                    r = np.ones((length, 3), dtype=np.float64) * (kalman_meas_std_lab ** 2)
                    r[0] = np.maximum(std0[i] ** 2, 1e-6)
                    unobs = mask_np[i].mean(axis=-1) < 0.5
                    r[unobs] = 1e6
                    refined.append(_rts_smoother_random_walk(y, r, q)[0])
                pred0 = np.stack(refined, axis=0).astype(np.float32)
            diff_conf = _scalar_from_info(sample_info.get('conf_diffusion', None))
            diff_std = _scalar_from_info(sample_info.get('diffusion_std_norm_meanL2', None))
            retrieval_conf = _extract_retrieval_confidence(info)
            fused_conf = _fuse_confidence(diff_conf, diff_std, retrieval_conf)
            if diff_conf is not None:
                diag_lists.setdefault('cdiff', []).extend([diff_conf] * pred0.shape[0])
                conf_list.extend([diff_conf] * pred0.shape[0])
            if diff_std is not None:
                diag_lists.setdefault('std', []).extend([diff_std] * pred0.shape[0])
            if retrieval_conf is not None:
                diag_lists.setdefault('cret', []).extend([retrieval_conf] * pred0.shape[0])
            if fused_conf is not None:
                diag_lists.setdefault('conf', []).extend([fused_conf] * pred0.shape[0])
        for p, g in zip(pred0, gt0):
            de_list.append(float(delta_e2000(p, g)))
        diagnostics = _physics_diagnostics(bundle, x_curr, cond, pred0, info)
        for key, value in diagnostics.items():
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            diag_lists.setdefault(key, []).extend(arr.tolist())
        if 'confidence' in info:
            conf_value = info['confidence']
            if isinstance(conf_value, torch.Tensor):
                conf_list.extend(conf_value.detach().cpu().numpy().tolist())
    out = {
        'deltaE2000_mean': float(np.mean(de_list)) if de_list else float('nan'),
        'deltaE2000_std': float(np.std(de_list)) if de_list else float('nan'),
    }
    if conf_list:
        out['confidence_mean'] = float(np.mean(conf_list))
        out['confidence_std'] = float(np.std(conf_list))
    for key, values in diag_lists.items():
        if values:
            out[f'{key}_mean'] = float(np.mean(values))
            out[f'{key}_std'] = float(np.std(values))
    return out


def _single_rgb(
    bundle: Dict[str, object],
    rgb: np.ndarray,
    device: torch.device,
    cond_method: str,
    library_npz: Optional[str],
    retrieval_k: int,
    retrieval_temp: float,
    num_samples: int,
    kalman_refine: bool = False,
    kalman_meas_std_lab: float = 1.0,
    kalman_process_std_lab: float = 2.0,
):
    lab = rgb_to_lab(rgb[None, :])[0]
    lab_norm = LabNorm()
    x_curr = lab_norm.normalize(lab[None, :]).astype(np.float32)
    x0 = np.stack([lab, lab], axis=0)[None, :, :]
    x0n = lab_norm.normalize(x0).astype(np.float32)
    mask = np.array([[[0, 0, 0], [1, 1, 1]]], dtype=np.float32)
    x0_t = torch.from_numpy(x0n).to(device)
    mask_t = torch.from_numpy(mask).to(device)
    x_curr_t = torch.from_numpy(x_curr).to(device)
    cond, info = _resolve_condition(bundle, None, x_curr_t, device, cond_method, library_npz, retrieval_k, retrieval_temp)

    effective_num_samples = max(int(num_samples), 8)
    pred_lab0_b, std_lab0_b, sample_info = sample_with_confidence(
        bundle['denoiser'],
        bundle['schedule'],
        x0_t,
        mask_t,
        cond,
        num_samples=effective_num_samples,
    )
    pred_lab0 = pred_lab0_b[0]
    std_lab0 = std_lab0_b[0]

    if kalman_refine:
        y = np.stack([pred_lab0, lab.astype(np.float32)], axis=0)
        r = np.ones((2, 3), dtype=np.float64) * (float(kalman_meas_std_lab) ** 2)
        r[0] = np.maximum(std_lab0.astype(np.float64) ** 2, 1e-6)
        q = np.asarray([float(kalman_process_std_lab) ** 2] * 3, dtype=np.float64)
        pred_lab0 = _rts_smoother_random_walk(y, r, q)[0].astype(np.float32)

    pred_rgb0 = lab_to_rgb(pred_lab0[None, :])[0]
    bridge_conf = _scalar_from_info(info.get('confidence', None))
    retrieval_conf = _extract_retrieval_confidence(info)
    diff_conf = _scalar_from_info(sample_info.get('conf_diffusion', None))
    diff_std = _scalar_from_info(sample_info.get('diffusion_std_norm_meanL2', None))
    fused_conf = _fuse_confidence(diff_conf, diff_std, retrieval_conf)

    diagnostics = _physics_diagnostics(bundle, x_curr_t, cond, pred_lab0[None, :], info)
    out = {
        'rgb': pred_rgb0.tolist(),
        'lab': pred_lab0.tolist(),
        'conf': fused_conf,
        'std': diff_std,
        'cdiff': diff_conf,
        'cret': retrieval_conf if retrieval_conf is not None else bridge_conf,
    }
    for key in ('posterior_entropy', 'confidence_bridge', 'spec_color_agreement', 'damage_score'):
        if key == 'confidence_bridge':
            value = bridge_conf
        else:
            value = diagnostics.get(key, None)
            if value is not None:
                value = float(np.asarray(value).reshape(-1)[0])
        if value is not None:
            out[key] = value
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--test_npz', type=str, default='')
    ap.add_argument('--max_batches', type=int, default=50)
    ap.add_argument('--rgb', type=str, default='')
    ap.add_argument('--cond_method', type=str, default='auto', choices=['auto', 'true', 'pred', 'retrieval', 'posterior', 'posterior_retrieval'])
    ap.add_argument('--library_npz', type=str, default='')
    ap.add_argument('--prototype_bank', type=str, default='')
    ap.add_argument('--retrieval_k', type=int, default=5)
    ap.add_argument('--retrieval_temp', type=float, default=0.07)
    ap.add_argument('--num_samples', type=int, default=1)
    ap.add_argument('--kalman_refine', action='store_true')
    ap.add_argument('--kalman_rts', action='store_true')
    ap.add_argument('--kalman_meas_std_lab', type=float, default=1.0)
    ap.add_argument('--kalman_process_std_lab', type=float, default=2.0)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    bundle = load_checkpoint(args.ckpt, device, prototype_bank_path=args.prototype_bank)

    if args.test_npz:
        stats = evaluate_test(
            bundle=bundle,
            test_npz=args.test_npz,
            device=device,
            cond_method=args.cond_method,
            library_npz=args.library_npz if args.library_npz else None,
            retrieval_k=int(args.retrieval_k),
            retrieval_temp=float(args.retrieval_temp),
            num_samples=int(args.num_samples),
            max_batches=int(args.max_batches),
            kalman_refine=bool(args.kalman_refine or args.kalman_rts),
            kalman_meas_std_lab=float(args.kalman_meas_std_lab),
            kalman_process_std_lab=float(args.kalman_process_std_lab),
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return

    if args.rgb:
        rgb = np.array([float(x) for x in args.rgb.split(',')], dtype=np.float32)
        if rgb.size != 3:
            raise ValueError("--rgb must be 'R,G,B'")
        out = _single_rgb(
            bundle=bundle,
            rgb=rgb,
            device=device,
            cond_method=args.cond_method,
            library_npz=args.library_npz if args.library_npz else None,
            retrieval_k=int(args.retrieval_k),
            retrieval_temp=float(args.retrieval_temp),
            num_samples=int(args.num_samples),
            kalman_refine=bool(args.kalman_refine or args.kalman_rts),
            kalman_meas_std_lab=float(args.kalman_meas_std_lab),
            kalman_process_std_lab=float(args.kalman_process_std_lab),
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    raise ValueError('Provide either --test_npz or --rgb')


if __name__ == '__main__':
    main()

_load_ckpt = load_checkpoint
_sample_with_confidence = sample_with_confidence


