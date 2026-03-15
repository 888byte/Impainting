from __future__ import annotations

import warnings

import numpy as np
import torch

from bridge.physics_heads import DamageHead, DamageHeadConfig, SpecColorHead, SpecColorHeadConfig
from inference.pipeline import _single_rgb, _stabilize_single_rgb_prediction, load_checkpoint
from models.color_encoder import ColorEncoder, ColorEncoderConfig
from models.cond_predictor import ColorToSpecPredictor, CondPredictorConfig
from models.denoiser import DenoiserConfig, MambaDenoiser
from models.spectral_encoder import ConditionerConfig, MultimodalConditioner
from training.physics_losses import (
    compute_aug_consistency_loss,
    compute_damage_losses,
    compute_parent_consistency_loss,
    compute_spec_color_loss,
)
from utils.config_utils import normalize_config


def test_normalize_config_adds_physics_defaults():
    cfg = normalize_config({})
    physics = cfg['physics']
    assert physics['enable'] is False
    assert physics['use_cycle_model'] == 'auto'
    assert physics['use_spec_color_consistency'] is False
    assert physics['use_parent_consistency'] is False
    assert physics['use_aug_consistency'] is False
    assert physics['use_damage_constraint'] is False
    assert physics['lambda_spec_pred_consistency'] == 0.0
    assert physics['side_consistency_scale'] == 0.25
    assert physics['low_confidence_skip_physics'] is True
    inference = cfg['inference']
    assert inference['stabilize_single_rgb'] is True
    assert inference['stabilize_min_strength'] == 0.15
    assert inference['stabilize_drift_scale_L'] == 18.0
    assert inference['stabilize_ab_cap_gain'] == 28.0


def test_single_rgb_stabilizer_pulls_low_conf_prediction_towards_input():
    cfg = normalize_config({})['inference']
    current = np.array([55.0, 0.0, 0.0], dtype=np.float32)
    predicted = np.array([82.0, 48.0, 42.0], dtype=np.float32)
    stabilized, eff_conf = _stabilize_single_rgb_prediction(current, predicted, 0.05, cfg)
    assert np.linalg.norm(stabilized - current) < np.linalg.norm(predicted - current)
    assert eff_conf <= 0.05
    assert abs(float(stabilized[1] - current[1])) < abs(float(predicted[1] - current[1]))



def test_spec_color_loss_uses_true_x0_and_detached_pred_target():
    torch.manual_seed(0)
    head = SpecColorHead(SpecColorHeadConfig(in_dim=4, hidden_dim=8, n_layers=2))
    pseudo_cond = torch.randn(3, 4, requires_grad=True)
    x0_true = torch.zeros(3, 2, 3)
    x0_pred = torch.full((3, 2, 3), 5.0, requires_grad=True)

    loss_no_pred, _ = compute_spec_color_loss(
        spec_color_head=head,
        pseudo_cond=pseudo_cond,
        x0_true=x0_true,
        x0_pred=x0_pred,
        lambda_pred_consistency=0.0,
    )
    loss_ref, _ = compute_spec_color_loss(
        spec_color_head=head,
        pseudo_cond=pseudo_cond,
        x0_true=x0_true,
        x0_pred=None,
        lambda_pred_consistency=0.0,
    )
    assert torch.allclose(loss_no_pred, loss_ref)

    loss_with_pred, _ = compute_spec_color_loss(
        spec_color_head=head,
        pseudo_cond=pseudo_cond,
        x0_true=x0_true,
        x0_pred=x0_pred,
        lambda_pred_consistency=0.5,
    )
    loss_with_pred.backward()
    assert torch.isfinite(loss_with_pred)
    assert pseudo_cond.grad is not None
    assert x0_pred.grad is None


def test_parent_consistency_auto_falls_back_to_side_with_downweight():
    logits = torch.tensor(
        [
            [2.0, 0.0],
            [1.8, 0.2],
            [0.0, 2.0],
            [0.2, 1.8],
        ],
        dtype=torch.float32,
    )
    batch = {'side': ['left', 'left', 'right', 'right']}
    loss_side_full, level_full = compute_parent_consistency_loss(
        batch=batch,
        level='side',
        posterior_logits=logits,
        pseudo_cond=None,
        side_consistency_scale=1.0,
    )
    loss_side_auto, level_auto = compute_parent_consistency_loss(
        batch=batch,
        level='auto',
        posterior_logits=logits,
        pseudo_cond=None,
        side_consistency_scale=0.25,
    )
    assert level_full == 'side'
    assert level_auto == 'side'
    assert torch.isfinite(loss_side_auto)
    assert torch.allclose(loss_side_auto, loss_side_full * 0.25, atol=1e-6)


def test_parent_and_aug_consistency_skip_without_valid_groups():
    latent = torch.randn(3, 5)
    zero_parent, level = compute_parent_consistency_loss(
        batch={},
        level='auto',
        posterior_logits=None,
        pseudo_cond=latent,
    )
    zero_small_group, _ = compute_parent_consistency_loss(
        batch={'side': ['left', 'right', 'solo']},
        level='auto',
        posterior_logits=None,
        pseudo_cond=latent,
    )
    zero_aug = compute_aug_consistency_loss(
        batch={'augmentation_parent_id': ['a', 'b', 'c']},
        posterior_logits=None,
        pseudo_cond=latent,
    )
    assert level is None
    assert float(zero_parent.item()) == 0.0
    assert float(zero_small_group.item()) == 0.0
    assert float(zero_aug.item()) == 0.0


def test_damage_losses_skip_without_order_and_run_with_sequence_time():
    damage_score = torch.tensor([0.1, 0.4, 0.7], dtype=torch.float32)
    mono_zero, smooth_zero = compute_damage_losses(batch={}, damage_score=damage_score, requires_order=True)
    assert float(mono_zero.item()) == 0.0
    assert float(smooth_zero.item()) == 0.0

    batch = {
        'sequence_parent_id': ['seq', 'seq', 'seq'],
        't': torch.tensor([1, 2, 3], dtype=torch.long),
    }
    mono, smooth = compute_damage_losses(batch=batch, damage_score=damage_score, requires_order=True)
    assert torch.isfinite(mono)
    assert torch.isfinite(smooth)
    assert mono.item() == 0.0


def _build_minimal_ckpt(path, include_heads: bool) -> None:
    device = torch.device('cpu')
    cond_cfg = ConditionerConfig(use_raman=True, use_xrd=True, raman_len=16, xrd_len=24, d_model=8, n_layers=1, use_fuse=True)
    conditioner = MultimodalConditioner(cond_cfg).to(device)
    denoiser = MambaDenoiser(DenoiserConfig(in_channels=3, hidden_dim=16, n_layers=1, dropout=0.0, cond_dim=conditioner.cond_dim)).to(device)
    color_encoder = ColorEncoder(ColorEncoderConfig(in_dim=3, d_model=8, hidden_dim=16, n_layers=2)).to(device)
    cond_predictor = ColorToSpecPredictor(
        CondPredictorConfig(in_dim=8, d_model=8, use_raman=True, use_xrd=True, hidden_dim=16, n_layers=2)
    ).to(device)

    ckpt = {
        'cfg': {
            'modality': {
                'use_raman': True,
                'use_xrd': True,
                'raman_len': 16,
                'xrd_len': 24,
                'spec_d_model': 8,
                'spec_n_layers': 1,
                'spec_dropout': 0.0,
                'use_fuse': True,
            },
            'model': {
                'in_channels': 3,
                'hidden_dim': 16,
                'n_layers': 1,
                'dropout': 0.0,
            },
            'diffusion': {
                'T': 2,
                'beta_0': 0.0001,
                'beta_T': 0.01,
            },
            'missing_modality': {
                'enable': True,
                'color_d_model': 8,
                'color_hidden_dim': 16,
                'color_n_layers': 2,
                'pred_hidden_dim': 16,
                'pred_n_layers': 2,
            },
            'bridge': {
                'enable': False,
                'mode': 'pred',
                'prototype_bank': {'path': ''},
            },
            'physics': {
                'enable': True,
                'use_cycle_model': False,
                'use_spec_color_consistency': True,
                'use_damage_constraint': True,
                'cond_hidden': 16,
            },
        },
        'conditioner': conditioner.state_dict(),
        'denoiser': denoiser.state_dict(),
        'color_encoder': color_encoder.state_dict(),
        'cond_predictor': cond_predictor.state_dict(),
    }
    if include_heads:
        spec_head = SpecColorHead(SpecColorHeadConfig(in_dim=conditioner.cond_dim, hidden_dim=16, n_layers=2)).to(device)
        damage_head = DamageHead(DamageHeadConfig(in_dim=color_encoder.cfg.d_model + conditioner.cond_dim, hidden_dim=16, n_layers=2)).to(device)
        ckpt['spec_color_head'] = spec_head.state_dict()
        ckpt['damage_head'] = damage_head.state_dict()
    torch.save(ckpt, path)


def test_load_checkpoint_warns_and_disables_missing_physics_heads(scratch_dir):
    ckpt_path = scratch_dir / 'no_heads.pt'
    _build_minimal_ckpt(ckpt_path, include_heads=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        bundle = load_checkpoint(str(ckpt_path), torch.device('cpu'))
    messages = [str(item.message) for item in caught]
    assert any('spec_color_head' in msg for msg in messages)
    assert any('damage_head' in msg for msg in messages)
    assert bundle['spec_color_head'] is None
    assert bundle['damage_head'] is None


def test_single_rgb_outputs_physics_diagnostics_when_weights_exist(scratch_dir):
    ckpt_path = scratch_dir / 'with_heads.pt'
    _build_minimal_ckpt(ckpt_path, include_heads=True)
    bundle = load_checkpoint(str(ckpt_path), torch.device('cpu'))
    out = _single_rgb(
        bundle=bundle,
        rgb=np.array([120.0, 80.0, 60.0], dtype=np.float32),
        device=torch.device('cpu'),
        cond_method='pred',
        library_npz=None,
        retrieval_k=1,
        retrieval_temp=0.1,
        num_samples=1,
        kalman_refine=True,
    )
    assert out['spec_color_agreement'] is not None
    assert out['damage_score'] is not None
    assert out['rgb'] is not None
    assert out['lab'] is not None
    assert out['conf'] is not None
    assert out['std'] is not None
    assert out['cdiff'] is not None
    assert 'cret' in out
    assert 'pred_rgb_original' not in out
    assert 'pred_lab_original' not in out
    assert 'confidence_diffusion' not in out
    assert 'diffusion_std_norm_meanL2' not in out
    assert 'confidence_retrieval' not in out
    assert 'input_rgb_current' not in out
    assert 'num_samples_used' not in out
    assert 'kalman_refined' not in out

