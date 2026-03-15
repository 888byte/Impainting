# -*- coding: utf-8 -*-
"""Build a 3D LUT from the current RGB inference pipeline.

Default behavior uses a fast in-process batch engine:
- load checkpoint once
- evaluate many RGB grid points per GPU batch
- keep resume support via out_npz / done_npy

Legacy subprocess mode is still available for debugging.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from inference.pipeline import (
    _confidence_or_default,
    _fuse_confidence,
    _resolve_condition,
    _rts_smoother_random_walk,
    _stabilize_single_rgb_prediction,
    load_checkpoint,
)
from training.diffusion import p_sample_loop
from utils.color_utils import LabNorm, lab_to_rgb, rgb_to_lab

_JSON_START_PAT = re.compile(r"\{")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Build LUT from the current infer.py RGB pipeline')
    ap.add_argument('--ckpt', type=str, default='ckpt/lab_raman_xrd/best_model.pt')
    ap.add_argument('--library_npz', type=str, default='data/standard_alignment/library_embeddings.npz')
    ap.add_argument('--prototype_bank', type=str, default='data/pigment_npz/prototype_bank.npz')
    ap.add_argument(
        '--cond_method',
        type=str,
        default='auto',
        choices=['auto', 'pred', 'retrieval', 'posterior', 'posterior_retrieval'],
    )
    ap.add_argument('--num_samples', type=int, default=14)
    ap.add_argument('--grid_size', type=int, default=33)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--out_npz', type=str, default='pigment_lut33.npz')
    ap.add_argument('--done_npy', type=str, default='')
    ap.add_argument('--save_every', type=int, default=300)
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--heartbeat_sec', type=int, default=30)
    ap.add_argument('--retries', type=int, default=5)
    ap.add_argument('--timeout_sec', type=int, default=90)
    ap.add_argument('--python_exe', type=str, default='python')
    ap.add_argument('--verbose_cmd', action='store_true')
    ap.add_argument('--kalman_rts', action='store_true')
    ap.add_argument('--lut_order', type=str, default='RGB', choices=['RGB'])
    ap.add_argument('--retrieval_k', type=int, default=5)
    ap.add_argument('--retrieval_temp', type=float, default=0.07)

    ap.add_argument('--engine', type=str, default='batch', choices=['batch', 'subprocess'])
    ap.add_argument('--batch_size', type=int, default=2048)
    ap.add_argument('--min_batch_size', type=int, default=128)
    ap.add_argument('--sample_log_every', type=int, default=0)

    ap.add_argument('--max_workers', type=int, default=40)
    ap.add_argument('--max_inflight', type=int, default=200)
    return ap.parse_args()


def _parse_last_json(stdout: str) -> dict:
    if not stdout:
        raise ValueError('empty stdout')
    starts = [m.start() for m in _JSON_START_PAT.finditer(stdout)]
    if not starts:
        raise ValueError("no '{' found in stdout")
    candidate = stdout[starts[-1]:].strip()
    try:
        return json.loads(candidate)
    except Exception:
        end = candidate.rfind('}')
        if end == -1:
            raise
        return json.loads(candidate[: end + 1])


def _atomic_save_npz(path: Path, **kwargs) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp.npz')
    np.savez_compressed(tmp, **kwargs)
    os.replace(tmp, path)


def _build_infer_command(args: argparse.Namespace, rgb: tuple[float, float, float]) -> list[str]:
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        args.python_exe,
        str(repo_root / 'infer.py'),
        '--ckpt',
        args.ckpt,
        '--device',
        args.device,
        '--rgb',
        f'{rgb[0]:.5g},{rgb[1]:.5g},{rgb[2]:.5g}',
        '--cond_method',
        args.cond_method,
        '--num_samples',
        str(args.num_samples),
        '--retrieval_k',
        str(args.retrieval_k),
        '--retrieval_temp',
        str(args.retrieval_temp),
    ]
    if args.library_npz:
        cmd.extend(['--library_npz', args.library_npz])
    if args.prototype_bank:
        cmd.extend(['--prototype_bank', args.prototype_bank])
    if args.kalman_rts:
        cmd.append('--kalman_rts')
    return cmd


def run_infer_subprocess(args: argparse.Namespace, rgb: tuple[float, float, float]) -> dict:
    cmd = _build_infer_command(args, rgb)
    if args.verbose_cmd:
        print(' '.join(cmd))

    last_err = None
    for attempt in range(1, int(args.retries) + 1):
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=int(args.timeout_sec),
            )
        except subprocess.TimeoutExpired as exc:
            last_err = RuntimeError(f'infer timeout {attempt}/{args.retries}: {exc}')
            continue

        if proc.returncode != 0:
            stderr_tail = (proc.stderr or '').strip()[-400:]
            stdout_tail = (proc.stdout or '').strip()[-400:]
            last_err = RuntimeError(
                f'infer failed attempt {attempt}/{args.retries}: code={proc.returncode} '
                f'stderr={stderr_tail!r} stdout={stdout_tail!r}'
            )
            continue

        try:
            return _parse_last_json(proc.stdout)
        except Exception as exc:
            last_err = RuntimeError(
                f'json parse error attempt {attempt}/{args.retries}: {str(exc)[:200]}'
            )
            continue

    raise last_err if last_err else RuntimeError('infer failed with unknown error')


def load_resume_arrays(args: argparse.Namespace, grid: np.ndarray):
    n = len(grid)
    lut_rgb = np.zeros((n, n, n, 3), dtype=np.uint8)
    lut_lab = np.zeros((n, n, n, 3), dtype=np.float32)
    lut_conf = np.zeros((n, n, n), dtype=np.float32)
    lut_std = np.zeros((n, n, n), dtype=np.float32)
    lut_cdiff = np.zeros((n, n, n), dtype=np.float32)
    lut_cret = np.zeros((n, n, n), dtype=np.float32)

    out_npz = Path(args.out_npz)
    done_npy = Path(args.done_npy) if args.done_npy else out_npz.with_name(out_npz.stem + '_done.npy')

    if out_npz.exists():
        prev = np.load(out_npz, allow_pickle=True)
        try:
            for key, arr in {
                'lut_rgb': lut_rgb,
                'lut_lab': lut_lab,
                'lut_conf': lut_conf,
                'lut_std': lut_std,
                'lut_cdiff': lut_cdiff,
                'lut_cret': lut_cret,
            }.items():
                if key in prev and prev[key].shape == arr.shape:
                    arr[:] = prev[key]
            if 'grid' in prev:
                gprev = prev['grid'].astype(np.float32)
                if gprev.shape == grid.shape and np.allclose(gprev, grid):
                    print('[Resume] Loaded existing LUT arrays (grid matches).')
                else:
                    print('[Warn] Existing LUT grid differs from current grid.')
        finally:
            prev.close()

    if done_npy.exists():
        done = np.load(done_npy)
        if done.shape != (n, n, n):
            print('[Warn] done marker shape mismatch, resetting it.')
            done = np.zeros((n, n, n), dtype=np.uint8)
    else:
        done = np.zeros((n, n, n), dtype=np.uint8)

    return {
        'out_npz': out_npz,
        'done_npy': done_npy,
        'lut_rgb': lut_rgb,
        'lut_lab': lut_lab,
        'lut_conf': lut_conf,
        'lut_std': lut_std,
        'lut_cdiff': lut_cdiff,
        'lut_cret': lut_cret,
        'done': done,
    }


def save_all(args: argparse.Namespace, grid: np.ndarray, state: dict) -> None:
    np.save(state['done_npy'], state['done'])
    _atomic_save_npz(
        state['out_npz'],
        grid=grid,
        lut_rgb=state['lut_rgb'],
        lut_lab=state['lut_lab'],
        lut_conf=state['lut_conf'],
        lut_std=state['lut_std'],
        lut_cdiff=state['lut_cdiff'],
        lut_cret=state['lut_cret'],
        meta=dict(
            axis_order=args.lut_order,
            ckpt=args.ckpt,
            library_npz=args.library_npz,
            prototype_bank=args.prototype_bank,
            cond_method=args.cond_method,
            num_samples=int(args.num_samples),
            grid_size=int(args.grid_size),
            device=args.device,
            kalman_rts=bool(args.kalman_rts),
            engine=args.engine,
            batch_size=int(args.batch_size),
            min_batch_size=int(args.min_batch_size),
            max_workers=int(args.max_workers),
            max_inflight=int(args.max_inflight),
            save_every=int(args.save_every),
            log_every=int(args.log_every),
            heartbeat_sec=int(args.heartbeat_sec),
            retrieval_k=int(args.retrieval_k),
            retrieval_temp=float(args.retrieval_temp),
            retries=int(args.retries),
            timeout_sec=int(args.timeout_sec),
        ),
    )


def pending_points(grid: np.ndarray, done: np.ndarray):
    for i, r in enumerate(grid):
        for j, g in enumerate(grid):
            for k, b in enumerate(grid):
                if done[i, j, k]:
                    continue
                yield i, j, k, float(r), float(g), float(b)


def take_points(point_iter: Iterator[Tuple[int, int, int, float, float, float]], batch_size: int):
    pts = []
    for _ in range(int(batch_size)):
        try:
            pts.append(next(point_iter))
        except StopIteration:
            break
    return pts


def _tensor_to_numpy_1d(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().reshape(-1).astype(np.float32)
    if isinstance(value, np.ndarray):
        return value.reshape(-1).astype(np.float32)
    try:
        return np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None


def _extract_conf_arrays(info: Dict[str, object], batch_size: int):
    bridge_conf = _tensor_to_numpy_1d(info.get('confidence', None))
    retrieval_conf = None
    retrieval_info = info.get('retrieval', None)
    if isinstance(retrieval_info, dict):
        retrieval_conf = _tensor_to_numpy_1d(retrieval_info.get('confidence', None))
    elif retrieval_info is None and info.get('weights', None) is not None and info.get('top_index', None) is not None:
        retrieval_conf = _tensor_to_numpy_1d(info.get('confidence', None))

    if bridge_conf is not None and bridge_conf.size == 1 and batch_size > 1:
        bridge_conf = np.full((batch_size,), float(bridge_conf[0]), dtype=np.float32)
    if retrieval_conf is not None and retrieval_conf.size == 1 and batch_size > 1:
        retrieval_conf = np.full((batch_size,), float(retrieval_conf[0]), dtype=np.float32)
    return bridge_conf, retrieval_conf


@torch.no_grad()
def infer_batch(bundle: Dict[str, object], args: argparse.Namespace, rgb_batch: np.ndarray, device: torch.device):
    lab_norm = LabNorm()
    rgb_batch = np.asarray(rgb_batch, dtype=np.float32)
    batch_size = int(rgb_batch.shape[0])

    curr_lab = rgb_to_lab(rgb_batch).astype(np.float32)
    x_curr = lab_norm.normalize(curr_lab).astype(np.float32)
    x0 = np.stack([curr_lab, curr_lab], axis=1).astype(np.float32)
    x0n = lab_norm.normalize(x0).astype(np.float32)
    mask = np.zeros((batch_size, 2, 3), dtype=np.float32)
    mask[:, 1, :] = 1.0

    x_curr_t = torch.from_numpy(x_curr).to(device)
    x0_t = torch.from_numpy(x0n).to(device)
    mask_t = torch.from_numpy(mask).to(device)
    cond, info = _resolve_condition(
        bundle=bundle,
        batch=None,
        x_curr=x_curr_t,
        device=device,
        cond_method=args.cond_method,
        library_npz=args.library_npz if args.library_npz else None,
        retrieval_k=int(args.retrieval_k),
        retrieval_temp=float(args.retrieval_temp),
    )

    effective_num_samples = max(int(args.num_samples), 8)
    x_obs = x0_t * mask_t
    samples = []
    sample_log_every = max(0, int(args.sample_log_every))
    for sample_idx in range(effective_num_samples):
        if sample_log_every > 0 and ((sample_idx + 1) == 1 or (sample_idx + 1) % sample_log_every == 0):
            print(f'  [Sample] batch={batch_size} sample={sample_idx + 1}/{effective_num_samples}')
        x_s = p_sample_loop(bundle['denoiser'], bundle['schedule'], x_obs=x_obs, obs_mask=mask_t, cond=cond)
        samples.append(x_s[:, 0, :].detach().cpu().numpy())

    arr = np.stack(samples, axis=0)
    mean_norm = np.mean(arr, axis=0)
    std_norm = np.std(arr, axis=0)
    pred_lab = lab_norm.denormalize(mean_norm).astype(np.float32)
    std_lab = (std_norm * np.asarray([lab_norm.L_scale, lab_norm.ab_scale, lab_norm.ab_scale], dtype=np.float32)).astype(np.float32)
    std_scalar = np.linalg.norm(std_norm, axis=-1).astype(np.float32)
    diff_conf = np.exp(-std_scalar).astype(np.float32)

    if args.kalman_rts:
        q = np.asarray([2.0 ** 2] * 3, dtype=np.float64)
        refined = []
        for i in range(batch_size):
            y = np.stack([pred_lab[i], curr_lab[i]], axis=0)
            r = np.ones((2, 3), dtype=np.float64)
            r[0] = np.maximum(std_lab[i].astype(np.float64) ** 2, 1e-6)
            refined.append(_rts_smoother_random_walk(y, r, q)[0])
        pred_lab = np.stack(refined, axis=0).astype(np.float32)

    bridge_conf, retrieval_conf = _extract_conf_arrays(info, batch_size)
    infer_cfg = bundle['cfg'].get('inference', {})
    conf_out = np.zeros((batch_size,), dtype=np.float32)
    cret_out = np.zeros((batch_size,), dtype=np.float32)

    for i in range(batch_size):
        bridge_i = None if bridge_conf is None else float(bridge_conf[i])
        retrieval_i = None if retrieval_conf is None else float(retrieval_conf[i])
        fused = _fuse_confidence(float(diff_conf[i]), float(std_scalar[i]), retrieval_i)
        base_conf = _confidence_or_default(fused, bridge_i, float(diff_conf[i]))
        if bool(infer_cfg.get('stabilize_single_rgb', True)) and args.cond_method != 'true':
            pred_lab[i], eff_conf = _stabilize_single_rgb_prediction(curr_lab[i], pred_lab[i], base_conf, infer_cfg)
            conf_out[i] = np.float32(eff_conf)
        else:
            conf_out[i] = np.float32(base_conf if fused is None else fused)
        cret_out[i] = np.float32(np.nan if retrieval_i is None and bridge_i is None else (retrieval_i if retrieval_i is not None else bridge_i))

    pred_rgb = lab_to_rgb(pred_lab).astype(np.uint8)
    return {
        'rgb': pred_rgb,
        'lab': pred_lab,
        'conf': conf_out,
        'std': std_scalar,
        'cdiff': diff_conf,
        'cret': cret_out,
    }


def _is_retryable_batch_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    keywords = [
        'out of memory',
        'cuda',
        'cublas',
        'cudnn',
        'resource exhausted',
        'resource temporarily unavailable',
    ]
    return any(key in msg for key in keywords)


def _write_results(points, result, state):
    for idx, (i, j, k, _r, _g, _b) in enumerate(points):
        state['lut_rgb'][i, j, k] = result['rgb'][idx]
        state['lut_lab'][i, j, k] = result['lab'][idx]
        state['lut_conf'][i, j, k] = result['conf'][idx]
        state['lut_std'][i, j, k] = result['std'][idx]
        state['lut_cdiff'][i, j, k] = result['cdiff'][idx]
        state['lut_cret'][i, j, k] = result['cret'][idx]
        state['done'][i, j, k] = 1


def run_batch_engine(args: argparse.Namespace, grid: np.ndarray, state: dict) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    bundle = load_checkpoint(args.ckpt, device, prototype_bank_path=args.prototype_bank)

    total = int(args.grid_size) ** 3
    completed_this_run = 0
    failures = 0
    started = time.time()
    point_iter = pending_points(grid, state['done'])
    batch_size = max(1, int(args.batch_size))
    min_batch_size = max(1, int(args.min_batch_size))
    log_every = max(1, int(args.log_every))

    print(
        f'[Mode=batch] device={device} | batch_size={batch_size} | min_batch_size={min_batch_size} '
        f'| num_samples={int(args.num_samples)} | cond_method={args.cond_method}'
    )

    while True:
        points = take_points(point_iter, batch_size)
        if not points:
            break

        queue: List[List[Tuple[int, int, int, float, float, float]]] = [points]
        while queue:
            chunk = queue.pop(0)
            rgb_batch = np.asarray([pt[3:] for pt in chunk], dtype=np.float32)
            pending_total = total - int(state['done'].sum())
            print(
                f'[BatchStart] size={len(chunk)} | total_done={int(state["done"].sum())}/{total} '
                f'| pending={pending_total}'
            )
            try:
                result = infer_batch(bundle, args, rgb_batch, device)
                _write_results(chunk, result, state)
                completed_this_run += len(chunk)
                elapsed = (time.time() - started) / 60.0
                print(
                    f'[BatchDone] size={len(chunk)} | total_done={int(state["done"].sum())}/{total} '
                    f'| completed_this_run={completed_this_run} | failures={failures} | elapsed={elapsed:.1f} min'
                )
                if completed_this_run > 0 and completed_this_run % log_every == 0:
                    print(
                        f'[Progress] total_done={int(state["done"].sum())}/{total} '
                        f'| completed_this_run={completed_this_run} | failures={failures} | elapsed={elapsed:.1f} min'
                    )
                if completed_this_run > 0 and completed_this_run % int(args.save_every) == 0:
                    save_all(args, grid, state)
                    print(f'[Save] total_done={int(state["done"].sum())}/{total}')
            except Exception as exc:
                retryable = _is_retryable_batch_error(exc)
                if retryable and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if len(chunk) > min_batch_size and (retryable or len(chunk) > 1):
                    mid = len(chunk) // 2
                    left = chunk[:mid]
                    right = chunk[mid:]
                    print(
                        f'[Split] size={len(chunk)} -> {len(left)} + {len(right)} '
                        f'| err={str(exc)[:220]}'
                    )
                    if right:
                        queue.insert(0, right)
                    if left:
                        queue.insert(0, left)
                    continue

                failures += len(chunk)
                for pt in chunk:
                    print(f'[Fail] idx={pt[:3]} rgb={pt[3:]} err={str(exc)[:220]}')

    elapsed = (time.time() - started) / 60.0
    save_all(args, grid, state)
    print(f'[Done] saved to {state["out_npz"]}')
    print(
        f'       done={int(state["done"].sum())}/{total}, completed_this_run={completed_this_run}, failures={failures}'
    )
    print(f'       elapsed={elapsed:.1f} min')


def task_subprocess(args: argparse.Namespace, i: int, j: int, k: int, r: float, g: float, b: float):
    obj = run_infer_subprocess(args, (r, g, b))

    rgb_pred = np.clip(np.array(obj['rgb'], dtype=np.float32), 0.0, 255.0).round().astype(np.uint8)
    lab_pred = np.array(obj['lab'], dtype=np.float32)
    conf = np.float32(float(obj.get('conf') or 0.0))
    cdiff = np.float32(float(obj.get('cdiff') or 0.0))
    std = np.float32(float(obj.get('std') or 0.0))
    c_ret_raw = obj.get('cret', None)
    cret = np.float32(np.nan if c_ret_raw is None else float(c_ret_raw))
    return i, j, k, rgb_pred, lab_pred, conf, cdiff, std, cret


def run_subprocess_engine(args: argparse.Namespace, grid: np.ndarray, state: dict) -> None:
    total = int(args.grid_size) ** 3
    completed_this_run = 0
    failures = 0
    started = time.time()
    point_iter = pending_points(grid, state['done'])
    inflight: dict = {}
    initial_submitted = 0
    log_every = max(1, int(args.log_every))
    heartbeat_sec = max(1, int(args.heartbeat_sec))

    print(
        f'[Mode=subprocess] workers={int(args.max_workers)} | inflight_cap={int(args.max_inflight)} '
        f'| num_samples={int(args.num_samples)} | timeout={int(args.timeout_sec)}s'
    )

    try:
        with ThreadPoolExecutor(max_workers=int(args.max_workers)) as executor:
            while len(inflight) < int(args.max_inflight):
                try:
                    pt = next(point_iter)
                except StopIteration:
                    break
                fut = executor.submit(task_subprocess, args, *pt)
                inflight[fut] = pt
                initial_submitted += 1

            print(
                f'[Launch] submitted initial={initial_submitted} | workers={int(args.max_workers)} '
                f'| inflight_cap={int(args.max_inflight)} | save_every={int(args.save_every)} '
                f'| log_every={log_every} | heartbeat={heartbeat_sec}s'
            )

            while inflight:
                done_futs, _ = wait(inflight.keys(), timeout=float(heartbeat_sec), return_when=FIRST_COMPLETED)
                if not done_futs:
                    elapsed = (time.time() - started) / 60.0
                    print(
                        f'[Heartbeat] total_done={int(state["done"].sum())}/{total} '
                        f'| completed_this_run={completed_this_run} | inflight={len(inflight)} '
                        f'| failures={failures} | elapsed={elapsed:.1f} min'
                    )
                    continue

                for fut in done_futs:
                    pt = inflight.pop(fut)
                    try:
                        i, j, k, rgb_u8, lab_f32, conf, cdiff, std, cret = fut.result()
                        state['lut_rgb'][i, j, k] = rgb_u8
                        state['lut_lab'][i, j, k] = lab_f32
                        state['lut_conf'][i, j, k] = conf
                        state['lut_cdiff'][i, j, k] = cdiff
                        state['lut_std'][i, j, k] = std
                        state['lut_cret'][i, j, k] = cret
                        state['done'][i, j, k] = 1
                        completed_this_run += 1
                        if completed_this_run == 1 or completed_this_run % log_every == 0:
                            elapsed = (time.time() - started) / 60.0
                            print(
                                f'[Progress] total_done={int(state["done"].sum())}/{total} '
                                f'| completed_this_run={completed_this_run} | inflight={len(inflight)} '
                                f'| failures={failures} | elapsed={elapsed:.1f} min'
                            )
                    except Exception as exc:
                        failures += 1
                        print(f'[Fail] idx={pt[:3]} rgb={pt[3:]} err={str(exc)[:240]}')

                    if completed_this_run > 0 and completed_this_run % int(args.save_every) == 0:
                        save_all(args, grid, state)
                        elapsed = (time.time() - started) / 60.0
                        print(
                            f'[Save] +{completed_this_run} this run | total_done={int(state["done"].sum())}/{total} '
                            f'| failures={failures} | elapsed={elapsed:.1f} min'
                        )

                    while len(inflight) < int(args.max_inflight):
                        try:
                            pt2 = next(point_iter)
                        except StopIteration:
                            break
                        fut2 = executor.submit(task_subprocess, args, *pt2)
                        inflight[fut2] = pt2
    finally:
        save_all(args, grid, state)
        elapsed = (time.time() - started) / 60.0
        print(f'[Done] saved to {state["out_npz"]}')
        print(
            f'       done={int(state["done"].sum())}/{total}, completed_this_run={completed_this_run}, failures={failures}'
        )
        print(f'       elapsed={elapsed:.1f} min')


def main() -> None:
    args = parse_args()
    grid = np.linspace(0.0, 255.0, int(args.grid_size), dtype=np.float32)
    state = load_resume_arrays(args, grid)

    total = int(args.grid_size) ** 3
    done_count = int(state['done'].sum())
    print(f'[Init] done={done_count}/{total}, pending={total - done_count}')

    if args.engine == 'subprocess':
        run_subprocess_engine(args, grid, state)
    else:
        run_batch_engine(args, grid, state)


if __name__ == '__main__':
    main()
