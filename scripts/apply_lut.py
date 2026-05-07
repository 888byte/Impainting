#!/usr/bin/env python3
"""
Apply 3D LUT from a .npz file to images in a folder.
Saves outputs with suffix _lut.png next to inputs (lossless PNG to preserve clarity).
"""
import argparse
import os
import sys
import numpy as np
from PIL import Image


def load_lut(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    # prefer lut_rgb if present
    files = list(d.files)
    if 'lut_rgb' in d.files:
        lut = d['lut_rgb']
    else:
        # try first array with 4 dims and last dim 3
        lut = None
        for k in d.files:
            arr = d[k]
            if getattr(arr, 'ndim', 0) == 4 and arr.shape[-1] == 3:
                lut = arr
                break
        if lut is None:
            raise ValueError('No suitable 3D RGB LUT found in npz')
    grid = d['grid'] if 'grid' in d.files else np.linspace(0,255, lut.shape[0])
    return lut, grid


def apply_lut_to_array(img_arr, lut, grid):
    # img_arr: H,W,3 uint8
    H, W = img_arr.shape[:2]
    N = lut.shape[0]
    # flatten
    flat = img_arr.reshape(-1, 3).astype(np.float32)
    # positions along LUT axis
    positions = np.arange(N, dtype=np.float32)
    # map channel values (0..255) to fractional index along grid
    idxs = np.empty_like(flat)
    for c in range(3):
        idxs[:, c] = np.interp(flat[:, c], grid, positions)
    r_idx = idxs[:, 0]
    g_idx = idxs[:, 1]
    b_idx = idxs[:, 2]
    r0 = np.floor(r_idx).astype(np.int32)
    g0 = np.floor(g_idx).astype(np.int32)
    b0 = np.floor(b_idx).astype(np.int32)
    r1 = np.clip(r0 + 1, 0, N - 1)
    g1 = np.clip(g0 + 1, 0, N - 1)
    b1 = np.clip(b0 + 1, 0, N - 1)
    wr = (r_idx - r0).astype(np.float32)
    wg = (g_idx - g0).astype(np.float32)
    wb = (b_idx - b0).astype(np.float32)
    lutf = lut.astype(np.float32)
    # gather corner samples
    c000 = lutf[r0, g0, b0]
    c100 = lutf[r1, g0, b0]
    c010 = lutf[r0, g1, b0]
    c110 = lutf[r1, g1, b0]
    c001 = lutf[r0, g0, b1]
    c101 = lutf[r1, g0, b1]
    c011 = lutf[r0, g1, b1]
    c111 = lutf[r1, g1, b1]
    # interpolate
    c00 = c000 * (1.0 - wr)[:, None] + c100 * wr[:, None]
    c01 = c001 * (1.0 - wr)[:, None] + c101 * wr[:, None]
    c10 = c010 * (1.0 - wr)[:, None] + c110 * wr[:, None]
    c11 = c011 * (1.0 - wr)[:, None] + c111 * wr[:, None]
    c0 = c00 * (1.0 - wg)[:, None] + c10 * wg[:, None]
    c1 = c01 * (1.0 - wg)[:, None] + c11 * wg[:, None]
    c = c0 * (1.0 - wb)[:, None] + c1 * wb[:, None]
    c = np.clip(c, 0, 255)
    out = c.reshape(H, W, 3).astype(np.uint8)
    return out


def process_file(in_path, out_path, lut, grid):
    im = Image.open(in_path).convert('RGBA' if im_has_alpha(in_path) else 'RGB')
    arr = np.array(im)
    alpha = None
    if arr.shape[2] == 4:
        alpha = arr[:, :, 3].copy()
        rgb = arr[:, :, :3]
    else:
        rgb = arr
    out_rgb = apply_lut_to_array(rgb, lut, grid)
    if alpha is not None:
        out_arr = np.dstack([out_rgb, alpha])
        out_im = Image.fromarray(out_arr, mode='RGBA')
    else:
        out_im = Image.fromarray(out_rgb, mode='RGB')
    out_im.save(out_path, format='PNG')


def im_has_alpha(p):
    try:
        from PIL import Image
        im = Image.open(p)
        return im.mode in ('LA', 'RGBA', 'PA')
    except Exception:
        return False


def find_images(folder, limit=2):
    exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    files = [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts]
    files.sort()
    return files[:limit]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--lut', default=os.path.join('cmp','pigment_lut49_xhq.npz'))
    p.add_argument('--infiles', nargs='*', default=None)
    p.add_argument('--outdir', default=None)
    args = p.parse_args(argv)
    lut_path = args.lut
    if not os.path.exists(lut_path):
        raise FileNotFoundError(lut_path)
    lut, grid = load_lut(lut_path)
    lut = np.asarray(lut)
    # pick images
    if args.infiles:
        files = args.infiles
    else:
        # images in same folder as lut
        folder = os.path.dirname(lut_path) or '.'
        files = find_images(folder, limit=2)
        files = [os.path.join(folder, f) for f in files]
    outdir = args.outdir if args.outdir is not None else os.path.dirname(files[0])
    os.makedirs(outdir, exist_ok=True)
    for inf in files:
        inf_path = inf
        base = os.path.splitext(os.path.basename(inf_path))[0]
        out_path = os.path.join(outdir, base + '_lut.png')
        print('Processing', inf_path, '->', out_path)
        process_file(inf_path, out_path, lut, grid)
    print('Done')

if __name__ == '__main__':
    main()
