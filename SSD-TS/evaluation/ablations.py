"""Palette scans and mining utilities for evaluation/ablation."""
from __future__ import annotations

import csv
import json
from typing import Dict, List, Optional

import numpy as np
import torch

from bridge.condition_builder import gather_last_observed
from evaluation.protocols import fps_lab_palette
from inference.pipeline import _resolve_condition, load_checkpoint
from inference.uncertainty import sample_with_confidence
from data.dataset import PigmentNPZDataset
from training.diffusion import p_sample_loop
from utils.color_utils import LabNorm, delta_e2000, lab_to_rgb, rgb_to_lab


@torch.no_grad()
def infer_rgb_batch(rgbs: np.ndarray, bundle: Dict[str, object], device: torch.device, cond_method: str, library_npz: Optional[str], retrieval_k: int, retrieval_temp: float, num_samples: int):
    lab_norm = LabNorm()
    lab = rgb_to_lab(rgbs.astype(np.float32))
    x0 = np.stack([lab, lab], axis=1).astype(np.float32)
    x0n = lab_norm.normalize(x0).astype(np.float32)
    mask = np.zeros((rgbs.shape[0], 2, 3), dtype=np.float32)
    mask[:, 1, :] = 1.0
    x0_t = torch.from_numpy(x0n).to(device)
    mask_t = torch.from_numpy(mask).to(device)
    x_curr_t = torch.from_numpy(lab_norm.normalize(lab).astype(np.float32)).to(device)
    cond, info = _resolve_condition(bundle, None, x_curr_t, device, cond_method, library_npz, retrieval_k, retrieval_temp)
    if int(num_samples) <= 1:
        x_s = p_sample_loop(bundle['denoiser'], bundle['schedule'], x_obs=x0_t * mask_t, obs_mask=mask_t, cond=cond)
        pred_lab0 = lab_norm.denormalize(x_s[:, 0, :].detach().cpu().numpy())
        conf_diff = np.full((rgbs.shape[0],), np.nan, dtype=np.float32)
    else:
        pred_lab0, _, sample_info = sample_with_confidence(bundle['denoiser'], bundle['schedule'], x0_t, mask_t, cond, num_samples=int(num_samples))
        conf_diff = np.full((rgbs.shape[0],), float(sample_info.get('conf_diffusion', np.nan)), dtype=np.float32)
    pred_rgb0 = lab_to_rgb(pred_lab0)
    conf_bridge = np.full((rgbs.shape[0],), np.nan, dtype=np.float32)
    if 'confidence' in info:
        conf_value = info['confidence']
        if isinstance(conf_value, torch.Tensor):
            conf_bridge = conf_value.detach().cpu().numpy().astype(np.float32)
    return pred_rgb0, conf_diff, conf_bridge


def run_palette_scan(bundle: Dict[str, object], device: torch.device, cond_method: str, n_colors: int, num_samples: int, out_csv: str, seed: int = 0, library_npz: Optional[str] = None, retrieval_k: int = 5, retrieval_temp: float = 0.07):
    lab = fps_lab_palette(int(n_colors), seed=int(seed))
    rgbs = lab_to_rgb(lab)
    pred_rgb0, conf_diff, conf_bridge = infer_rgb_batch(rgbs, bundle, device, cond_method, library_npz, retrieval_k, retrieval_temp, num_samples)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['in_r', 'in_g', 'in_b', 'out_r', 'out_g', 'out_b', 'conf_diffusion', 'conf_bridge'])
        for i in range(rgbs.shape[0]):
            w.writerow([int(rgbs[i, 0]), int(rgbs[i, 1]), int(rgbs[i, 2]), int(pred_rgb0[i, 0]), int(pred_rgb0[i, 1]), int(pred_rgb0[i, 2]), float(conf_diff[i]) if np.isfinite(conf_diff[i]) else '', float(conf_bridge[i]) if np.isfinite(conf_bridge[i]) else ''])
    return {'saved': out_csv, 'n': int(rgbs.shape[0])}


def run_mine(bundle: Dict[str, object], test_npz: str, device: torch.device, out_csv: str, cond_method: str = 'pred', num_samples: int = 20, max_rows: int = 20000, library_npz: Optional[str] = None, retrieval_k: int = 5, retrieval_temp: float = 0.07):
    ds = PigmentNPZDataset(test_npz)
    dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False, num_workers=0, drop_last=False)
    rows: List[Dict[str, object]] = []
    seen = 0
    lab_norm = LabNorm()
    for batch in dl:
        if seen >= int(max_rows):
            break
        x0 = batch['x0'].to(device)
        mask = batch['mask'].to(device)
        x_curr = gather_last_observed(x0, mask)
        cond, _ = _resolve_condition(bundle, batch, x_curr, device, cond_method, library_npz, retrieval_k, retrieval_temp)
        pred_lab0, _, info = sample_with_confidence(bundle['denoiser'], bundle['schedule'], x0, mask, cond, num_samples=int(num_samples))
        gt_lab0 = lab_norm.denormalize(x0[:, 0, :].detach().cpu().numpy())
        curr_lab = lab_norm.denormalize(x_curr.detach().cpu().numpy())
        pred_rgb0 = lab_to_rgb(pred_lab0)
        curr_rgb = lab_to_rgb(curr_lab)
        gt_rgb0 = lab_to_rgb(gt_lab0)
        for i in range(pred_lab0.shape[0]):
            rows.append({
                'deltaE2000': float(delta_e2000(pred_lab0[i], gt_lab0[i])),
                'curr_rgb': curr_rgb[i].tolist(),
                'pred_rgb0': pred_rgb0[i].tolist(),
                'gt_rgb0': gt_rgb0[i].tolist(),
                'conf_diffusion': float(info.get('conf_diffusion', np.nan)),
            })
        seen += int(x0.shape[0])
    rows.sort(key=lambda item: item['deltaE2000'], reverse=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['deltaE2000', 'curr_rgb', 'pred_rgb0', 'gt_rgb0', 'conf_diffusion'])
        for row in rows:
            w.writerow([row['deltaE2000'], row['curr_rgb'], row['pred_rgb0'], row['gt_rgb0'], row['conf_diffusion']])
    return {'saved': out_csv, 'rows': len(rows), 'top10': rows[:10]}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--mode', type=str, default='palette', choices=['palette', 'mine'])
    ap.add_argument('--cond_method', type=str, default='pred')
    ap.add_argument('--library_npz', type=str, default='')
    ap.add_argument('--prototype_bank', type=str, default='')
    ap.add_argument('--retrieval_k', type=int, default=5)
    ap.add_argument('--retrieval_temp', type=float, default=0.07)
    ap.add_argument('--num_samples', type=int, default=20)
    ap.add_argument('--n_colors', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--test_npz', type=str, default='')
    ap.add_argument('--max_rows', type=int, default=20000)
    ap.add_argument('--out_csv', type=str, required=True)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    bundle = load_checkpoint(args.ckpt, device, prototype_bank_path=args.prototype_bank)
    if args.mode == 'palette':
        out = run_palette_scan(bundle, device, args.cond_method, args.n_colors, args.num_samples, args.out_csv, seed=args.seed, library_npz=args.library_npz if args.library_npz else None, retrieval_k=args.retrieval_k, retrieval_temp=args.retrieval_temp)
    else:
        if not args.test_npz:
            raise ValueError('--mode mine requires --test_npz')
        out = run_mine(bundle, args.test_npz, device, args.out_csv, cond_method=args.cond_method, num_samples=args.num_samples, max_rows=args.max_rows, library_npz=args.library_npz if args.library_npz else None, retrieval_k=args.retrieval_k, retrieval_temp=args.retrieval_temp)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
