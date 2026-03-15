from __future__ import annotations

import os
import subprocess

def _run(cmd, cwd, env=None):
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=run_env)


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
    assert out_img.exists()

