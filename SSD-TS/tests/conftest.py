from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from uuid import uuid4
from typing import Dict

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def write_sample_index(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'sample_id',
                'npz_row',
                'source_log',
                'exp_id',
                'exp_tag',
                'side',
                'patch_id',
                'time_index',
                'sequence_parent_id',
                'spectral_parent_id',
                'augmentation_parent_id',
                'is_augmented',
                'split_group_id',
                'has_raman',
                'has_xrd',
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def make_npz(path: Path, n: int, raman_len: int = 16, xrd_len: int = 24, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0.0, 0.2, size=(n, 2, 3)).astype(np.float32)
    x0[:, 0, :] = rng.uniform(low=[0.2, -0.2, -0.2], high=[0.8, 0.2, 0.2], size=(n, 3)).astype(np.float32)
    x0[:, 1, :] = (x0[:, 0, :] + rng.normal(0.02, 0.03, size=(n, 3))).astype(np.float32)
    mask = np.zeros((n, 2, 3), dtype=np.float32)
    mask[:, 1, :] = 1.0
    raman = rng.normal(size=(n, raman_len)).astype(np.float32)
    xrd = rng.normal(size=(n, xrd_len)).astype(np.float32)
    np.savez_compressed(
        path,
        x0=x0,
        mask=mask,
        raman=raman,
        xrd=xrd,
        meta_patch_id=np.tile(np.array([1, 2], dtype=np.int64), int(np.ceil(n / 2)))[:n],
        meta_t=np.tile(np.array([1, 2], dtype=np.int64), int(np.ceil(n / 2)))[:n],
        meta_exp_id=np.zeros((n,), dtype=np.int64),
        meta_exp_tag=np.full((n,), 66, dtype=np.int64),
        meta_has_raman=np.ones((n,), dtype=np.int64),
        meta_has_xrd=np.ones((n,), dtype=np.int64),
        has_raman=np.ones((n,), dtype=np.int64),
        has_xrd=np.ones((n,), dtype=np.int64),
    )
    rows = []
    for idx in range(n):
        patch_id = int(1 + (idx % 2))
        side = 'left' if idx % 2 == 0 else 'right'
        rows.append(
            {
                'sample_id': f'sample-{idx:04d}',
                'npz_row': idx,
                'source_log': f'{side}_{idx}.txt',
                'exp_id': 0,
                'exp_tag': 66,
                'side': side,
                'patch_id': patch_id,
                'time_index': int(1 + (idx % 3)),
                'sequence_parent_id': f'{side}:0:{patch_id}',
                'spectral_parent_id': f'66:{patch_id}',
                'augmentation_parent_id': f'{side}:0:{patch_id}',
                'is_augmented': 0,
                'split_group_id': f'66:{patch_id}',
                'has_raman': 1,
                'has_xrd': 1,
            }
        )
    write_sample_index(path.with_name(path.stem + '_index.csv'), rows)
    return path


@pytest.fixture()
def scratch_dir() -> Path:
    base = REPO_ROOT / 'tests_runtime'
    base.mkdir(exist_ok=True)
    path = base / f'case-{uuid4().hex}'
    path.mkdir(parents=True, exist_ok=False)
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture()
def synthetic_project(scratch_dir: Path) -> Dict[str, Path]:
    data_dir = scratch_dir / 'data'
    ckpt_dir = scratch_dir / 'ckpt'
    data_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_npz = make_npz(data_dir / 'train.npz', n=8, seed=1)
    val_npz = make_npz(data_dir / 'val.npz', n=4, seed=2)
    test_npz = make_npz(data_dir / 'test.npz', n=4, seed=3)
    cfg = {
        'data': {
            'train_npz': str(train_npz),
            'val_npz': str(val_npz),
            'test_npz': str(test_npz),
            'train_index': str(train_npz.with_name('train_index.csv')),
            'val_index': str(val_npz.with_name('val_index.csv')),
            'test_index': str(test_npz.with_name('test_index.csv')),
        },
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
            'T': 4,
            'beta_0': 0.0001,
            'beta_T': 0.01,
        },
        'missing_modality': {
            'enable': True,
            'drop_prob': 0.2,
            'lambda_pred': 0.05,
            'color_d_model': 8,
            'color_hidden_dim': 16,
            'color_n_layers': 2,
            'pred_hidden_dim': 16,
            'pred_n_layers': 2,
        },
        'bridge': {
            'enable': True,
            'mode': 'posterior',
            'use_gate': False,
            'use_distill': True,
            'use_group_sampler': True,
            'loss_weight': 0.1,
            'distill_weight': 0.05,
            'posterior_temp': 0.1,
            'teacher_temp': 0.1,
            'prototype_bank': {
                'path': str(data_dir / 'prototype_bank.npz'),
                'normalize': True,
            },
        },
        'train': {
            'device': 'cpu',
            'seed': 7,
            'batch_size': 2,
            'eval_batch_size': 2,
            'epochs': 1,
            'lr': 0.001,
            'weight_decay': 0.0,
            'grad_clip': 1.0,
            'log_every': 1,
            'eval_every': 1,
            'eval_num_batches': 1,
            'save_every': 1,
            'save_dir': str(ckpt_dir),
            'early_stopping_patience': 5,
        },
        'color_aug': {'enable': False},
        'physics': {'enable': False},
    }
    config_path = scratch_dir / 'config.json'
    config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    return {
        'root': REPO_ROOT,
        'tmp_path': scratch_dir,
        'config': config_path,
        'train_npz': train_npz,
        'val_npz': val_npz,
        'test_npz': test_npz,
        'prototype_bank': data_dir / 'prototype_bank.npz',
        'ckpt_dir': ckpt_dir,
    }


@pytest.fixture()
def python_exe() -> str:
    return sys.executable

