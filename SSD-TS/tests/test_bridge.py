from __future__ import annotations

import numpy as np
import torch

from bridge.condition_builder import (
    build_posterior_condition,
    build_posterior_retrieval_condition,
    build_pred_condition,
    build_retrieval_condition,
    build_true_condition,
    teacher_posterior,
)
from bridge.posterior_head import PosteriorHead, PosteriorHeadConfig
from bridge.prototype_bank import PrototypeBank
from models.color_encoder import ColorEncoder, ColorEncoderConfig
from models.cond_predictor import ColorToSpecPredictor, CondPredictorConfig
from models.spectral_encoder import ConditionerConfig, MultimodalConditioner


def test_bridge_modes_shape(scratch_dir):
    device = torch.device('cpu')
    batch_size = 4
    conditioner = MultimodalConditioner(
        ConditionerConfig(
            use_raman=True,
            use_xrd=True,
            raman_len=16,
            xrd_len=24,
            d_model=8,
            n_layers=1,
            use_fuse=True,
        )
    ).to(device)
    color_encoder = ColorEncoder(ColorEncoderConfig(in_dim=3, d_model=8, hidden_dim=16, n_layers=2)).to(device)
    cond_predictor = ColorToSpecPredictor(
        CondPredictorConfig(in_dim=8, d_model=8, use_raman=True, use_xrd=True, hidden_dim=16, n_layers=2)
    ).to(device)

    batch = {
        'raman': torch.randn(batch_size, 16),
        'xrd': torch.randn(batch_size, 24),
    }
    x_curr = torch.randn(batch_size, 3)

    cond_true, embeds_true = build_true_condition(conditioner, batch, device)
    assert cond_true.shape == (batch_size, conditioner.cond_dim)
    assert 'raman' in embeds_true and 'xrd' in embeds_true

    cond_pred, embeds_pred = build_pred_condition(x_curr, conditioner, color_encoder, cond_predictor)
    assert cond_pred.shape == cond_true.shape
    assert 'raman' in embeds_pred and 'xrd' in embeds_pred

    lib_path = scratch_dir / 'library.npz'
    np.savez_compressed(lib_path, raman_emb=np.random.randn(6, 8).astype(np.float32))
    cond_ret, info_ret = build_retrieval_condition(
        x_curr_norm_lab=x_curr,
        conditioner=conditioner,
        color_encoder=color_encoder,
        cond_predictor=cond_predictor,
        library_npz=str(lib_path),
        device=device,
        retrieval_k=3,
        retrieval_temp=0.1,
    )
    assert cond_ret.shape == cond_true.shape
    assert info_ret['weights'].shape == (batch_size, 3)
    assert info_ret['confidence'].shape == (batch_size,)

    bank = PrototypeBank(
        prototype_ids=['p0', 'p1', 'p2', 'p3'],
        cond_vectors=np.random.randn(4, conditioner.cond_dim).astype(np.float32),
        metadata=[{'id': f'p{i}'} for i in range(4)],
    )
    posterior_head = PosteriorHead(PosteriorHeadConfig(in_dim=8, num_prototypes=4, hidden_dim=16, n_layers=2)).to(device)
    cond_post, info_post = build_posterior_condition(
        x_curr_norm_lab=x_curr,
        color_encoder=color_encoder,
        posterior_head=posterior_head,
        prototype_bank=bank,
        device=device,
        top_k=2,
        temp=0.1,
    )
    assert cond_post.shape == cond_true.shape
    assert info_post['weights'].shape == (batch_size, 4)
    assert info_post['confidence'].shape == (batch_size,)

    teacher = teacher_posterior(cond_true, bank, device=device, temp=0.1)
    assert teacher.shape == (batch_size, 4)
    assert torch.allclose(teacher.sum(dim=-1), torch.ones(batch_size), atol=1e-5)

    cond_joint, info_joint = build_posterior_retrieval_condition(
        x_curr_norm_lab=x_curr,
        conditioner=conditioner,
        color_encoder=color_encoder,
        cond_predictor=cond_predictor,
        posterior_head=posterior_head,
        prototype_bank=bank,
        library_npz=str(lib_path),
        device=device,
        retrieval_k=3,
        retrieval_temp=0.1,
        top_k=2,
        temp=0.1,
    )
    assert cond_joint.shape == cond_true.shape
    assert info_joint['alpha'].shape == (batch_size, 1)

