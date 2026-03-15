from __future__ import annotations

import os
import subprocess
from pathlib import Path

import cv2
import numpy as np


def _run(cmd, cwd, env=None):
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=run_env)


def _write_demo_image(path: Path) -> Path:
    h, w = 48, 48
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack(
        [
            xx / max(1, w - 1) * 255.0,
            yy / max(1, h - 1) * 255.0,
            (xx + yy) / max(1, h + w - 2) * 255.0,
        ],
        axis=-1,
    ).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path


def _write_demo_mask(path: Path) -> Path:
    mask = np.zeros((48, 48), dtype=np.uint8)
    cv2.rectangle(mask, (14, 14), (33, 33), 255, thickness=-1)
    cv2.imwrite(str(path), mask)
    return path


def _write_demo_lut(path: Path) -> Path:
    from utils.color_utils import rgb_to_lab

    grid = np.array([0.0, 255.0], dtype=np.float32)
    lut_rgb = np.zeros((2, 2, 2, 3), dtype=np.uint8)
    for i, r in enumerate(grid.astype(np.uint8)):
        for j, g in enumerate(grid.astype(np.uint8)):
            for k, b in enumerate(grid.astype(np.uint8)):
                lut_rgb[i, j, k] = np.array([r, g, b], dtype=np.uint8)

    lut_lab = rgb_to_lab(lut_rgb.reshape(-1, 3).astype(np.float32)).reshape(2, 2, 2, 3).astype(np.float32)
    lut_conf = np.full((2, 2, 2), 0.8, dtype=np.float32)
    lut_std = np.full((2, 2, 2), 0.1, dtype=np.float32)
    lut_cdiff = np.full((2, 2, 2), 0.9, dtype=np.float32)
    lut_cret = np.full((2, 2, 2), 0.7, dtype=np.float32)
    np.savez_compressed(
        path,
        grid=grid,
        lut_rgb=lut_rgb,
        lut_lab=lut_lab,
        lut_conf=lut_conf,
        lut_std=lut_std,
        lut_cdiff=lut_cdiff,
        lut_cret=lut_cret,
    )
    return path


def test_train_infer_evaluate_and_build_prototypes_smoke(synthetic_project, python_exe):
    root = synthetic_project['root']
    config = synthetic_project['config']

    train_res = _run([python_exe, 'train.py', '--config', str(config)], cwd=root)
    assert train_res.returncode == 0, train_res.stderr or train_res.stdout

    ckpt = synthetic_project['ckpt_dir'] / 'ckpt_ep1.pt'
    assert ckpt.exists()
    assert synthetic_project['prototype_bank'].exists()

    infer_pred = _run(
        [
            python_exe,
            'infer.py',
            '--ckpt',
            str(ckpt),
            '--test_npz',
            str(synthetic_project['test_npz']),
            '--device',
            'cpu',
            '--cond_method',
            'pred',
            '--max_batches',
            '1',
        ],
        cwd=root,
    )
    assert infer_pred.returncode == 0, infer_pred.stderr or infer_pred.stdout
    assert 'deltaE2000_mean' in infer_pred.stdout

    infer_post = _run(
        [
            python_exe,
            'infer.py',
            '--ckpt',
            str(ckpt),
            '--test_npz',
            str(synthetic_project['test_npz']),
            '--device',
            'cpu',
            '--cond_method',
            'posterior',
            '--prototype_bank',
            str(synthetic_project['prototype_bank']),
            '--max_batches',
            '1',
        ],
        cwd=root,
    )
    assert infer_post.returncode == 0, infer_post.stderr or infer_post.stdout
    assert 'deltaE2000_mean' in infer_post.stdout

    eval_res = _run(
        [
            python_exe,
            'evaluate.py',
            '--mode',
            'test',
            '--ckpt',
            str(ckpt),
            '--test_npz',
            str(synthetic_project['test_npz']),
            '--device',
            'cpu',
            '--cond_method',
            'pred',
            '--max_batches',
            '1',
        ],
        cwd=root,
    )
    assert eval_res.returncode == 0, eval_res.stderr or eval_res.stdout
    assert 'deltaE2000_mean' in eval_res.stdout

    bank_out = synthetic_project['tmp_path'] / 'rebuilt_bank.npz'
    build_res = _run(
        [
            python_exe,
            'build_prototypes.py',
            '--config',
            str(config),
            '--device',
            'cpu',
            '--output',
            str(bank_out),
        ],
        cwd=root,
    )
    assert build_res.returncode == 0, build_res.stderr or build_res.stdout
    assert bank_out.exists()

    legacy_infer = _run(
        [
            python_exe,
            '-m',
            'pigment_task.infer_pigment',
            '--ckpt',
            str(ckpt),
            '--test_npz',
            str(synthetic_project['test_npz']),
            '--device',
            'cpu',
            '--cond_method',
            'pred',
            '--max_batches',
            '1',
        ],
        cwd=root,
    )
    assert legacy_infer.returncode == 0, legacy_infer.stderr or legacy_infer.stdout
    assert 'deltaE2000_mean' in legacy_infer.stdout


def test_legacy_wrappers_help(synthetic_project, python_exe):
    root = synthetic_project['root']
    for cmd in (
        [python_exe, '-m', 'pigment_task.preprocess_pigment', '--help'],
        [python_exe, '-m', 'pigment_task.train_pigment', '--help'],
    ):
        res = _run(cmd, cwd=root)
        assert res.returncode == 0, res.stderr or res.stdout


def test_t1_smoke_generates_image(synthetic_project, python_exe):
    root = synthetic_project['root']
    config = synthetic_project['config']
    train_res = _run([python_exe, 'train.py', '--config', str(config)], cwd=root)
    assert train_res.returncode == 0, train_res.stderr or train_res.stdout

    ckpt = synthetic_project['ckpt_dir'] / 'ckpt_ep1.pt'
    assert ckpt.exists()

    out_img = synthetic_project['tmp_path'] / 't1_smoke.png'
    res = _run(
        [
            python_exe,
            't1.py',
            '--ckpt',
            str(ckpt),
            '--cond_method',
            'pred',
            '--num_samples',
            '1',
            '--palette',
            'named_only',
            '--output_image',
            str(out_img),
        ],
        cwd=root,
        env={'KMP_DUPLICATE_LIB_OK': 'TRUE'},
    )
    assert res.returncode == 0, res.stderr or res.stdout
    assert out_img.exists()


def test_t2_and_t3_smoke_generate_outputs(synthetic_project, python_exe):
    root = synthetic_project['root']
    config = synthetic_project['config']
    train_res = _run([python_exe, 'train.py', '--config', str(config)], cwd=root)
    assert train_res.returncode == 0, train_res.stderr or train_res.stdout

    ckpt = synthetic_project['ckpt_dir'] / 'ckpt_ep1.pt'
    assert ckpt.exists()

    img_path = _write_demo_image(synthetic_project['tmp_path'] / 'demo.png')
    mask_path = _write_demo_mask(synthetic_project['tmp_path'] / 'demo_mask.png')

    out_img = synthetic_project['tmp_path'] / 't2_restored.png'
    preview = synthetic_project['tmp_path'] / 't2_palette.png'
    t2_res = _run(
        [
            python_exe,
            't2.py',
            '--input_image',
            str(img_path),
            '--ckpt',
            str(ckpt),
            '--device',
            'cpu',
            '--cond_method',
            'pred',
            '--num_samples',
            '1',
            '--n_colors',
            '4',
            '--output_image',
            str(out_img),
            '--palette_preview',
            str(preview),
        ],
        cwd=root,
        env={'KMP_DUPLICATE_LIB_OK': 'TRUE'},
    )
    assert t2_res.returncode == 0, t2_res.stderr or t2_res.stdout
    assert out_img.exists()
    assert preview.exists()

    t3_out = synthetic_project['tmp_path'] / 't3_results'
    t3_res = _run(
        [
            python_exe,
            't3.py',
            '--img_path',
            str(img_path),
            '--mask_path',
            str(mask_path),
            '--ckpt',
            str(ckpt),
            '--device',
            'cpu',
            '--cond_method',
            'pred',
            '--num_samples',
            '1',
            '--n_colors',
            '4',
            '--output_dir',
            str(t3_out),
        ],
        cwd=root,
        env={'KMP_DUPLICATE_LIB_OK': 'TRUE'},
    )
    assert t3_res.returncode == 0, t3_res.stderr or t3_res.stdout
    assert (t3_out / '01_structure.png').exists()
    assert (t3_out / 'color_prior_recolor.png').exists()
    assert (t3_out / 'confidence_map.png').exists()


def test_t4_to_t7_lut_scripts_smoke(synthetic_project, python_exe):
    root = synthetic_project['root']
    img_path = _write_demo_image(synthetic_project['tmp_path'] / 'lut_demo.png')
    mask_path = _write_demo_mask(synthetic_project['tmp_path'] / 'lut_mask.png')
    lut_path = _write_demo_lut(synthetic_project['tmp_path'] / 'demo_lut.npz')

    t4_res = _run([python_exe, 't4.py', str(lut_path)], cwd=root)
    assert t4_res.returncode == 0, t4_res.stderr or t4_res.stdout
    assert 'lut_rgb' in t4_res.stdout

    t5_out = synthetic_project['tmp_path'] / 't5_results'
    t5_res = _run(
        [
            python_exe,
            't5.py',
            '--img_path',
            str(img_path),
            '--mask_path',
            str(mask_path),
            '--lut_npz',
            str(lut_path),
            '--n_colors',
            '4',
            '--output_dir',
            str(t5_out),
        ],
        cwd=root,
    )
    assert t5_res.returncode == 0, t5_res.stderr or t5_res.stdout
    assert (t5_out / 'color_prior_lut_mural_opt.png').exists()
    assert (t5_out / 'confidence_map.png').exists()

    t6_out = synthetic_project['tmp_path'] / 't6_results'
    t6_res = _run(
        [
            python_exe,
            't6.py',
            '--img_path',
            str(img_path),
            '--lut_npz',
            str(lut_path),
            '--output_dir',
            str(t6_out),
        ],
        cwd=root,
    )
    assert t6_res.returncode == 0, t6_res.stderr or t6_res.stdout
    assert (t6_out / 'mural_lut_full.png').exists()

    t7_out = synthetic_project['tmp_path'] / 't7_results'
    t7_res = _run(
        [
            python_exe,
            't7.py',
            '--img_path',
            str(img_path),
            '--mask_path',
            str(mask_path),
            '--lut_npz',
            str(lut_path),
            '--output_dir',
            str(t7_out),
        ],
        cwd=root,
    )
    assert t7_res.returncode == 0, t7_res.stderr or t7_res.stdout
    assert (t7_out / 'mural_lut_mask_only.png').exists()
