"""Unified bridge API for true, pred, retrieval, posterior, and posterior+retrieval conditions."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from bridge.confidence_gate import ConfidenceGate, posterior_confidence
from bridge.prototype_bank import PrototypeBank


def last_observed_index(mask: torch.Tensor) -> torch.Tensor:
    obs = (mask.mean(dim=-1) > 0.5).to(torch.long)
    batch_size, length = obs.shape
    idx_range = torch.arange(length, device=mask.device).view(1, length).expand(batch_size, length)
    return torch.max(idx_range * obs, dim=1).values


def gather_last_observed(x0: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    idx = last_observed_index(mask)
    return x0[torch.arange(x0.shape[0], device=x0.device), idx, :]


@torch.no_grad()
def build_true_condition(conditioner, batch: Dict[str, torch.Tensor], device: torch.device) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
    if conditioner.cond_dim == 0:
        return None, {}
    raman = batch.get('raman', None)
    xrd = batch.get('xrd', None)
    cond, embeds = conditioner(
        raman.to(device) if raman is not None else None,
        xrd.to(device) if xrd is not None else None,
        raman_peaks=batch.get('raman_peaks', None).to(device) if 'raman_peaks' in batch else None,
        xrd_peaks=batch.get('xrd_peaks', None).to(device) if 'xrd_peaks' in batch else None,
        return_embeds=True,
    )
    return cond, embeds


def build_cond_from_pred_embeds(conditioner, embeds_pred: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    if conditioner.cond_dim == 0:
        return None
    feats = []
    if conditioner.raman_enc is not None:
        r = embeds_pred.get('raman_emb', embeds_pred.get('raman', None))
        if r is None:
            raise KeyError('Missing predicted Raman embedding')
        feats.append(r)
    if conditioner.xrd_enc is not None:
        x = embeds_pred.get('xrd_emb', embeds_pred.get('xrd', None))
        if x is None:
            raise KeyError('Missing predicted XRD embedding')
        feats.append(x)
    return conditioner.fuse(torch.cat(feats, dim=-1))


@torch.no_grad()
def predict_embeds_from_rgb(x_curr_norm_lab: torch.Tensor, conditioner, color_encoder, cond_predictor) -> Dict[str, torch.Tensor]:
    zc = color_encoder(x_curr_norm_lab)
    pred = cond_predictor(zc)
    out: Dict[str, torch.Tensor] = {}
    if conditioner.raman_enc is not None:
        out['raman'] = pred['raman_emb']
    if conditioner.xrd_enc is not None:
        out['xrd'] = pred['xrd_emb']
    return out


@torch.no_grad()
def build_pred_condition(x_curr_norm_lab: torch.Tensor, conditioner, color_encoder, cond_predictor):
    embeds_pred = predict_embeds_from_rgb(x_curr_norm_lab, conditioner, color_encoder, cond_predictor)
    return build_cond_from_pred_embeds(conditioner, embeds_pred), embeds_pred


def load_library_npz(path: str) -> Dict[str, np.ndarray]:
    lib = np.load(path, allow_pickle=True)
    return {k: lib[k] for k in lib.files}


@torch.no_grad()
def retrieval_raman_embed(z_color: torch.Tensor, lib_raman_emb: torch.Tensor, top_k: int = 5, temp: float = 0.07):
    zq = F.normalize(z_color, dim=-1)
    zk = F.normalize(lib_raman_emb, dim=-1)
    sim = zq @ zk.T
    k = min(int(top_k), sim.size(1))
    topv, topi = torch.topk(sim, k=k, dim=-1)
    weights = F.softmax(topv / float(temp), dim=-1)
    gathered = lib_raman_emb[topi]
    return torch.sum(gathered * weights.unsqueeze(-1), dim=1), weights, topi


def retrieval_confidence(weights: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    w = torch.clamp(weights, eps, 1.0)
    ent = -torch.sum(w * torch.log(w), dim=-1)
    return (1.0 - ent / np.log(weights.size(-1) + eps)).clamp(0.0, 1.0)


@torch.no_grad()
def build_retrieval_condition(
    x_curr_norm_lab: torch.Tensor,
    conditioner,
    color_encoder,
    cond_predictor,
    library_npz: str,
    device: torch.device,
    retrieval_k: int = 5,
    retrieval_temp: float = 0.07,
):
    lib = load_library_npz(library_npz)
    if 'raman_emb' not in lib:
        raise ValueError("library_npz must contain 'raman_emb'")
    lib_raman = torch.from_numpy(lib['raman_emb'].astype(np.float32)).to(device)
    zc = color_encoder(x_curr_norm_lab)
    raman_emb, weights, topi = retrieval_raman_embed(zc, lib_raman, top_k=int(retrieval_k), temp=float(retrieval_temp))
    embeds_pred: Dict[str, torch.Tensor] = {}
    if conditioner.raman_enc is not None:
        embeds_pred['raman'] = raman_emb
    if conditioner.xrd_enc is not None:
        if cond_predictor is None:
            embeds_pred['xrd'] = torch.zeros_like(raman_emb)
        else:
            embeds_pred['xrd'] = cond_predictor(zc)['xrd_emb']
    cond = build_cond_from_pred_embeds(conditioner, embeds_pred)
    info = {
        'weights': weights,
        'top_index': topi,
        'confidence': retrieval_confidence(weights),
    }
    return cond, info


def teacher_posterior(cond_true: torch.Tensor, bank: PrototypeBank, device: torch.device, temp: float = 0.07, normalize: bool = True) -> torch.Tensor:
    prototypes = bank.to_tensor(device=device, normalize=normalize)
    query = F.normalize(cond_true, dim=-1) if normalize else cond_true
    logits = query @ prototypes.T / float(temp)
    return F.softmax(logits, dim=-1)


@torch.no_grad()
def build_posterior_condition(
    x_curr_norm_lab: torch.Tensor,
    color_encoder,
    posterior_head,
    prototype_bank: PrototypeBank,
    device: torch.device,
    top_k: int = 0,
    temp: float = 0.07,
):
    zc = color_encoder(x_curr_norm_lab)
    logits = posterior_head(zc)
    weights = F.softmax(logits / float(temp), dim=-1)
    if int(top_k) > 0 and int(top_k) < weights.size(-1):
        topv, topi = torch.topk(weights, k=int(top_k), dim=-1)
        sparse = torch.zeros_like(weights)
        sparse.scatter_(1, topi, topv)
        weights = sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    cond = prototype_bank.aggregate(weights, device=device)
    entropy = -torch.sum(weights * torch.log(weights.clamp_min(1e-8)), dim=-1)
    confidence = posterior_confidence(entropy, prototype_bank.num_prototypes)
    info = {
        'logits': logits,
        'weights': weights,
        'entropy': entropy,
        'confidence': confidence,
    }
    return cond, info


@torch.no_grad()
def build_posterior_retrieval_condition(
    x_curr_norm_lab: torch.Tensor,
    conditioner,
    color_encoder,
    cond_predictor,
    posterior_head,
    prototype_bank: PrototypeBank,
    library_npz: str,
    device: torch.device,
    retrieval_k: int = 5,
    retrieval_temp: float = 0.07,
    top_k: int = 0,
    temp: float = 0.07,
    gate: Optional[ConfidenceGate] = None,
):
    posterior_cond, posterior_info = build_posterior_condition(
        x_curr_norm_lab=x_curr_norm_lab,
        color_encoder=color_encoder,
        posterior_head=posterior_head,
        prototype_bank=prototype_bank,
        device=device,
        top_k=top_k,
        temp=temp,
    )
    retrieval_cond, retrieval_info = build_retrieval_condition(
        x_curr_norm_lab=x_curr_norm_lab,
        conditioner=conditioner,
        color_encoder=color_encoder,
        cond_predictor=cond_predictor,
        library_npz=library_npz,
        device=device,
        retrieval_k=retrieval_k,
        retrieval_temp=retrieval_temp,
    )
    gate = gate or ConfidenceGate()
    alpha = gate(posterior_info, retrieval_info).view(-1, 1)
    cond = alpha * posterior_cond + (1.0 - alpha) * retrieval_cond
    info = {
        'alpha': alpha,
        'posterior': posterior_info,
        'retrieval': retrieval_info,
        'confidence': alpha.view(-1),
    }
    return cond, info
