"""Helpers for minimal physics-informed soft constraints."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def _to_list(values) -> Optional[List[str]]:
    if values is None:
        return None
    if isinstance(values, (list, tuple)):
        return [str(v) for v in values]
    return None


def _valid_groups(group_ids: Sequence[str]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, group_id in enumerate(group_ids):
        if group_id:
            groups[str(group_id)].append(idx)
    return {key: idxs for key, idxs in groups.items() if len(idxs) >= 2}


def has_repeated_group(group_ids: Optional[Sequence[str]]) -> bool:
    if not group_ids:
        return False
    counts = Counter(str(v) for v in group_ids if v)
    return any(count >= 2 for count in counts.values())


def extract_group_ids(batch: Dict[str, object], key: str) -> Optional[List[str]]:
    return _to_list(batch.get(key, None))


def select_parent_group_ids(batch: Dict[str, object], level: str = 'auto') -> Tuple[Optional[List[str]], Optional[str]]:
    order = [str(level)]
    if str(level) == 'auto':
        order = ['spectral', 'sequence', 'side']
    key_map = {
        'aug': 'augmentation_parent_id',
        'spectral': 'spectral_parent_id',
        'sequence': 'sequence_parent_id',
        'side': 'side',
    }
    for item in order:
        key = key_map.get(item, item)
        ids = extract_group_ids(batch, key)
        if has_repeated_group(ids):
            return ids, item
    if str(level) == 'side':
        ids = extract_group_ids(batch, 'side')
        return (ids, 'side') if ids else (None, None)
    return None, None


def build_physics_weight(confidence: Optional[torch.Tensor], enabled: bool = True, ref_tensor: Optional[torch.Tensor] = None):
    if not enabled or confidence is None:
        if ref_tensor is None:
            return None
        return torch.ones(ref_tensor.shape[0], device=ref_tensor.device, dtype=ref_tensor.dtype)
    if not isinstance(confidence, torch.Tensor):
        if ref_tensor is None:
            return None
        return torch.full((ref_tensor.shape[0],), float(confidence), device=ref_tensor.device, dtype=ref_tensor.dtype)
    weight = confidence.detach().float()
    if weight.ndim == 0:
        weight = weight.view(1)
    return weight.clamp(0.0, 1.0)


def _weighted_mean(values: torch.Tensor, weight: Optional[torch.Tensor]) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_tensor(0.0)
    if weight is None:
        return values.mean()
    weight = weight.to(values.device, values.dtype).view(-1)
    return (values.view(-1) * weight).sum() / weight.sum().clamp_min(1e-8)


def compute_spec_color_loss(
    spec_color_head,
    pseudo_cond: Optional[torch.Tensor],
    x0_true: torch.Tensor,
    confidence: Optional[torch.Tensor] = None,
    x0_pred: Optional[torch.Tensor] = None,
    lambda_pred_consistency: float = 0.0,
):
    if spec_color_head is None or pseudo_cond is None:
        return x0_true.new_tensor(0.0), None
    aux_color = spec_color_head(pseudo_cond)
    target = x0_true[:, 0, :]
    per_sample = F.smooth_l1_loss(aux_color, target, reduction='none').mean(dim=-1)
    weight = build_physics_weight(confidence, enabled=True, ref_tensor=aux_color)
    loss = _weighted_mean(per_sample, weight)
    if x0_pred is not None and float(lambda_pred_consistency) > 0:
        pred_target = x0_pred[:, 0, :].detach()
        per_pred = F.smooth_l1_loss(aux_color, pred_target, reduction='none').mean(dim=-1)
        loss = loss + float(lambda_pred_consistency) * _weighted_mean(per_pred, weight)
    return loss, aux_color


def _posterior_group_loss(logits: torch.Tensor, groups: Dict[str, List[int]], sample_weight: Optional[torch.Tensor]) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    losses = []
    weights = []
    for idxs in groups.values():
        idx = torch.tensor(idxs, device=logits.device, dtype=torch.long)
        target = probs[idx].mean(dim=0, keepdim=True).detach().expand(len(idxs), -1)
        kl = F.kl_div(log_probs[idx], target, reduction='none').sum(dim=-1)
        losses.append(kl.mean())
        if sample_weight is not None:
            weights.append(sample_weight[idx].mean())
    if not losses:
        return logits.new_tensor(0.0)
    loss_tensor = torch.stack(losses)
    weight_tensor = torch.stack(weights) if weights else None
    return _weighted_mean(loss_tensor, weight_tensor)


def _latent_group_loss(latent: torch.Tensor, groups: Dict[str, List[int]], sample_weight: Optional[torch.Tensor]) -> torch.Tensor:
    losses = []
    weights = []
    for idxs in groups.values():
        idx = torch.tensor(idxs, device=latent.device, dtype=torch.long)
        values = latent[idx]
        mean_value = values.mean(dim=0, keepdim=True).detach().expand_as(values)
        cosine = 1.0 - F.cosine_similarity(values, mean_value, dim=-1)
        l2 = F.smooth_l1_loss(values, mean_value, reduction='none').mean(dim=-1)
        losses.append((cosine + l2).mean())
        if sample_weight is not None:
            weights.append(sample_weight[idx].mean())
    if not losses:
        return latent.new_tensor(0.0)
    loss_tensor = torch.stack(losses)
    weight_tensor = torch.stack(weights) if weights else None
    return _weighted_mean(loss_tensor, weight_tensor)


def compute_parent_consistency_loss(
    batch: Dict[str, object],
    level: str,
    posterior_logits: Optional[torch.Tensor],
    pseudo_cond: Optional[torch.Tensor],
    confidence: Optional[torch.Tensor] = None,
    side_consistency_scale: float = 0.25,
):
    group_ids, resolved_level = select_parent_group_ids(batch, level)
    if not group_ids:
        ref = posterior_logits if posterior_logits is not None else pseudo_cond
        if ref is None:
            return torch.tensor(0.0), None
        return ref.new_tensor(0.0), None
    groups = _valid_groups(group_ids)
    ref = posterior_logits if posterior_logits is not None else pseudo_cond
    if not groups or ref is None:
        return ref.new_tensor(0.0) if ref is not None else torch.tensor(0.0), resolved_level
    weight = build_physics_weight(confidence, enabled=True, ref_tensor=ref)
    if posterior_logits is not None:
        loss = _posterior_group_loss(posterior_logits, groups, weight)
    else:
        loss = _latent_group_loss(pseudo_cond, groups, weight)
    if resolved_level == 'side':
        loss = loss * float(side_consistency_scale)
    return loss, resolved_level


def compute_aug_consistency_loss(
    batch: Dict[str, object],
    posterior_logits: Optional[torch.Tensor],
    pseudo_cond: Optional[torch.Tensor],
    confidence: Optional[torch.Tensor] = None,
):
    group_ids = extract_group_ids(batch, 'augmentation_parent_id')
    ref = posterior_logits if posterior_logits is not None else pseudo_cond
    if not group_ids or ref is None:
        if ref is None:
            return torch.tensor(0.0)
        return ref.new_tensor(0.0)
    groups = _valid_groups(group_ids)
    if not groups:
        return ref.new_tensor(0.0)
    weight = build_physics_weight(confidence, enabled=True, ref_tensor=ref)
    if posterior_logits is not None:
        return _posterior_group_loss(posterior_logits, groups, weight)
    return _latent_group_loss(pseudo_cond, groups, weight)


def compute_damage_losses(
    batch: Dict[str, object],
    damage_score: Optional[torch.Tensor],
    confidence: Optional[torch.Tensor] = None,
    requires_order: bool = True,
):
    if damage_score is None:
        zero = torch.tensor(0.0)
        return zero, zero
    seq_ids = extract_group_ids(batch, 'sequence_parent_id')
    t = batch.get('t', None)
    if requires_order and (seq_ids is None or t is None):
        zero = damage_score.new_tensor(0.0)
        return zero, zero
    if seq_ids is None or t is None:
        zero = damage_score.new_tensor(0.0)
        return zero, zero
    if isinstance(t, torch.Tensor):
        t_tensor = t.to(damage_score.device).view(-1).long()
    else:
        zero = damage_score.new_tensor(0.0)
        return zero, zero
    groups = _valid_groups(seq_ids)
    if not groups:
        zero = damage_score.new_tensor(0.0)
        return zero, zero
    weight = build_physics_weight(confidence, enabled=True, ref_tensor=damage_score)
    mono_losses = []
    smooth_losses = []
    mono_weights = []
    smooth_weights = []
    for idxs in groups.values():
        idx = torch.tensor(idxs, device=damage_score.device, dtype=torch.long)
        times = t_tensor[idx]
        if torch.unique(times).numel() < 2:
            continue
        order = torch.argsort(times)
        idx = idx[order]
        times = times[order].float()
        scores = damage_score[idx]
        dt = (times[1:] - times[:-1]).clamp_min(1.0)
        slopes = (scores[1:] - scores[:-1]) / dt
        mono_losses.append(torch.relu(-slopes).mean())
        if weight is not None:
            mono_weights.append(weight[idx].mean())
        if slopes.numel() > 1:
            smooth_losses.append(torch.abs(slopes[1:] - slopes[:-1]).mean())
            if weight is not None:
                smooth_weights.append(weight[idx].mean())
    mono = _weighted_mean(torch.stack(mono_losses), torch.stack(mono_weights) if mono_weights else None) if mono_losses else damage_score.new_tensor(0.0)
    smooth = _weighted_mean(torch.stack(smooth_losses), torch.stack(smooth_weights) if smooth_weights else None) if smooth_losses else damage_score.new_tensor(0.0)
    return mono, smooth
