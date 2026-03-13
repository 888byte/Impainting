from __future__ import annotations

import json
import subprocess

import numpy as np
import pandas as pd

from data.dataset import PigmentNPZDataset


def _write_rgb_log(path, humidity: float, t_steps: int = 3):
    lines = []
    for t in range(1, t_steps + 1):
        lines.append(f'{t} Temperature: 25 C Humidity: {humidity}')
        for patch in range(1, 10):
            base = 20 * patch + t
            lines.append(f'NO.{patch} R:{base} G:{base + 1} B:{base + 2}')
    path.write_text('\n'.join(lines), encoding='utf-8')


def _write_wide_workbook(path, names):
    rows = []
    header = []
    sub = []
    for name in names:
        header.extend([name, name])
        sub.extend(['x', 'y'])
    rows.append(header)
    rows.append(sub)
    x = np.linspace(100.0, 400.0, num=16)
    for i in range(len(x)):
        row = []
        for idx, _name in enumerate(names):
            y = np.sin(x / (40.0 + idx * 5.0)) + idx
            row.extend([float(x[i]), float(y[i])])
        rows.append(row)
    pd.DataFrame(rows).to_excel(path, sheet_name='Sheet1', header=False, index=False)


def test_preprocess_wide_sheet_and_sidecar(scratch_dir, python_exe, synthetic_project):
    left_log = scratch_dir / '1-3_Left.txt'
    right_log = scratch_dir / '1-3_Right.txt'
    _write_rgb_log(left_log, humidity=66.0)
    _write_rgb_log(right_log, humidity=66.0)

    raman_xlsx = scratch_dir / 'raman.xlsx'
    xrd_xlsx = scratch_dir / 'xrd.xlsx'
    names = ['铅丹+铅白 66', '密陀僧+铅白 66']
    _write_wide_workbook(raman_xlsx, names)
    _write_wide_workbook(xrd_xlsx, names)

    meta = {
        'raman_len': 32,
        'xrd_len': 48,
        'experiments': {
            '66': {
                'patch_to_raman_sheet': {'1': '66 铅丹+铅白', '2': '66 密陀僧+铅白'},
                'patch_to_xrd_sheet': {'1': '66 铅丹+铅白', '2': '66 密陀僧+铅白'},
            }
        },
    }
    meta_json = scratch_dir / 'meta.json'
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    out_dir = scratch_dir / 'out'

    cmd = [
        python_exe,
        'preprocess.py',
        '--rgb_logs',
        f'{left_log},{right_log}',
        '--exp_tags',
        '66,66',
        '--use_patches',
        '1-2',
        '--meta_json',
        str(meta_json),
        '--raman_excel',
        str(raman_xlsx),
        '--xrd_excel',
        str(xrd_xlsx),
        '--output_dir',
        str(out_dir),
        '--split_mode',
        'group_patch',
        '--seed',
        '1',
        '--val_ratio',
        '0.25',
        '--test_ratio',
        '0.25',
    ]
    res = subprocess.run(cmd, cwd=synthetic_project['root'], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr or res.stdout

    train_npz = out_dir / 'train.npz'
    assert train_npz.exists()
    assert (out_dir / 'sample_index.csv').exists()
    assert (out_dir / 'train_index.csv').exists()
    data = np.load(train_npz, allow_pickle=True)
    assert data['x0'].shape[1] == 2
    assert 'meta_t' in data
    assert 'raman' in data and data['raman'].shape[1] == 32
    assert 'xrd' in data and data['xrd'].shape[1] == 48

    ds = PigmentNPZDataset(str(train_npz), index_csv=str(out_dir / 'train_index.csv'))
    sample = ds[0]
    assert 'side' in sample
    assert 'spectral_parent_id' in sample
    assert sample['x0'].shape == (2, 3)

