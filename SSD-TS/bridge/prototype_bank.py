"""Prototype bank persistence and train-fold aggregation helpers."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import PigmentNPZDataset
from data.index.sample_index import load_sample_index


@dataclass
class PrototypeBank:
    prototype_ids: List[str]
    cond_vectors: np.ndarray
    metadata: List[Dict[str, str]]

    @property
    def num_prototypes(self) -> int:
        return int(self.cond_vectors.shape[0])

    @property
    def cond_dim(self) -> int:
        return int(self.cond_vectors.shape[1]) if self.cond_vectors.ndim == 2 else 0

    def to_tensor(self, device: torch.device, normalize: bool = True) -> torch.Tensor:
        vec = torch.from_numpy(self.cond_vectors.astype(np.float32)).to(device)
        if normalize:
            vec = torch.nn.functional.normalize(vec, dim=-1)
        return vec

    def aggregate(self, weights: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or weights.device
        bank = torch.from_numpy(self.cond_vectors.astype(np.float32)).to(device)
        return weights @ bank

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        np.savez_compressed(path, prototype_ids=np.asarray(self.prototype_ids, dtype=object), cond_vectors=self.cond_vectors.astype(np.float32))
        meta_path = os.path.splitext(path)[0] + '.json'
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({'prototype_ids': self.prototype_ids, 'metadata': self.metadata}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> 'PrototypeBank':
        arr = np.load(path, allow_pickle=True)
        meta_path = os.path.splitext(path)[0] + '.json'
        metadata: List[Dict[str, str]] = []
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            metadata = payload.get('metadata', [])
        return cls(
            prototype_ids=[str(x) for x in arr['prototype_ids'].tolist()],
            cond_vectors=arr['cond_vectors'].astype(np.float32),
            metadata=metadata,
        )


def _default_group_id(batch: Dict[str, object], sample_index: Optional[List[Dict[str, str]]], idx: int) -> str:
    if sample_index is not None and idx < len(sample_index):
        row = sample_index[idx]
        if row.get('spectral_parent_id'):
            return row['spectral_parent_id']
    exp_id = int(batch['exp_id']) if 'exp_id' in batch else -1
    patch_id = int(batch['patch_id']) if 'patch_id' in batch else -1
    return f'{exp_id}:{patch_id}'


@torch.no_grad()
def build_prototype_bank(
    npz_path: str,
    conditioner,
    device: torch.device,
    index_csv: str = '',
    batch_size: int = 256,
) -> PrototypeBank:
    dataset = PigmentNPZDataset(npz_path=npz_path, index_csv=index_csv)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    sample_index = dataset.sample_index

    buckets: Dict[str, List[np.ndarray]] = {}
    metadata: Dict[str, Dict[str, str]] = {}
    row_offset = 0
    for batch in loader:
        raman = batch.get('raman', None)
        xrd = batch.get('xrd', None)
        if conditioner.cond_dim == 0:
            raise ValueError('Cannot build prototype bank when conditioner.cond_dim == 0')
        cond = conditioner(
            raman.to(device) if raman is not None else None,
            xrd.to(device) if xrd is not None else None,
            raman_peaks=batch.get('raman_peaks', None).to(device) if 'raman_peaks' in batch else None,
            xrd_peaks=batch.get('xrd_peaks', None).to(device) if 'xrd_peaks' in batch else None,
            return_embeds=False,
        )
        cond_np = cond.detach().cpu().numpy()
        batch_size_now = cond_np.shape[0]
        for local_idx in range(batch_size_now):
            global_idx = row_offset + local_idx
            pseudo = {
                'exp_id': batch['exp_id'][local_idx].item() if 'exp_id' in batch else -1,
                'patch_id': batch['patch_id'][local_idx].item() if 'patch_id' in batch else -1,
            }
            group_id = _default_group_id(pseudo, sample_index, global_idx)
            buckets.setdefault(group_id, []).append(cond_np[local_idx])
            if group_id not in metadata:
                metadata[group_id] = sample_index[global_idx] if sample_index is not None and global_idx < len(sample_index) else {
                    'exp_id': str(pseudo['exp_id']),
                    'patch_id': str(pseudo['patch_id']),
                }
        row_offset += batch_size_now

    prototype_ids = sorted(buckets.keys())
    cond_vectors = np.stack([np.mean(np.stack(buckets[key], axis=0), axis=0) for key in prototype_ids], axis=0).astype(np.float32)
    return PrototypeBank(prototype_ids=prototype_ids, cond_vectors=cond_vectors, metadata=[metadata[key] for key in prototype_ids])
