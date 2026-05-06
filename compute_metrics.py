#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute PSNR, SSIM, LPIPS, FID for each method folder against `gt`.
Usage: python compute_metrics.py

If required packages are missing the script will print pip install commands.

Outputs a CSV `metrics_results.csv` in the current folder.
"""
import os
import sys
import shutil
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

base = Path(r"d:\code\ky\bihua\Impainting\cmp\新建文件夹\新建文件夹")
gt_dir = base / 'gt'
if not gt_dir.exists():
    print('ERROR: gt folder not found at', gt_dir)
    sys.exit(1)

method_folders = [p.name for p in base.iterdir() if p.is_dir() and p.name not in ('gt','mask','mask_merge')]
print('Found method folders:', method_folders)

# Helper: require modules or print install hints
missing = []
try:
    from skimage.metrics import peak_signal_noise_ratio as psnr_sk
    from skimage.metrics import structural_similarity as ssim_sk
except Exception:
    missing.append('scikit-image')
try:
    import torch
except Exception:
    missing.append('torch')
# LPIPS
have_lpips = True
try:
    import lpips
except Exception:
    have_lpips = False
    missing.append('lpips')
# pytorch-fid (for FID)
have_pytorch_fid = True
try:
    from pytorch_fid import fid_score
except Exception:
    have_pytorch_fid = False
    missing.append('pytorch-fid')

if missing:
    print('\nMissing packages detected:')
    for m in sorted(set(missing)):
        print('  -', m)
    print('\nInstall with pip (recommended in your venv):')
    print('pip install --upgrade pip')
    pkgs = []
    if 'scikit-image' in missing:
        pkgs.append('scikit-image')
    if 'torch' in missing:
        pkgs.append('torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117')
    if 'lpips' in missing:
        pkgs.append('lpips')
    if 'pytorch-fid' in missing:
        pkgs.append('pytorch-fid')
    print('pip install ' + ' '.join(pkgs))
    print('\nAfter installation, re-run: python compute_metrics.py')
    # still proceed to compute PSNR/SSIM only if scikit-image available
    if 'scikit-image' in missing:
        sys.exit(1)

# Function to load image as ndarray (RGB) normalized [0,255]
def load_img(path):
    try:
        im = Image.open(path)
        im.load()
        im = im.convert('RGB')
        return np.array(im)
    except Exception as e:
        print(f"  [WARN] Failed to load image {path}: {e}")
        return None

# Function to compute LPIPS between two RGB uint8 images using lpips model
def compute_lpips(img1, img2, lpips_model):
    # img: HWC uint8 0-255
    import torch
    to_tensor = lambda x: (torch.from_numpy(x).permute(2,0,1).float() / 127.5 - 1.0).unsqueeze(0).to(device)
    a = to_tensor(img1)
    b = to_tensor(img2)
    with torch.no_grad():
        val = lpips_model(a, b)
    return val.item()

# compute PSNR & SSIM per pair
import csv
results = []
# device for torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)

lpips_model = None
if have_lpips:
    try:
        # use 'alex' trunk to avoid large VGG download issues
        lpips_model = lpips.LPIPS(net='alex').to(device)
    except Exception as e:
        print('Failed to load LPIPS model:', e)
        have_lpips = False

for method in method_folders:
    method_path = base / method
    print('\nProcessing method:', method)
    # find matching files with gt by basename (without extension)
    gt_files = sorted([p for p in gt_dir.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg')])
    # map basename -> path for method
    method_map = {p.stem: p for p in method_path.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg')}

    paired = []
    for g in gt_files:
        key = g.stem
        if key in method_map:
            paired.append((g, method_map[key]))
        else:
            # try alternate: gt may be .jpg and method .png but stem same; above covers stem
            pass
    print(f'  Found {len(paired)} paired images out of {len(gt_files)}')
    if len(paired) == 0:
        print('  No pairs found, skipping')
        continue

    psnrs = []
    ssims = []
    lpips_vals = []

    for gpath, mpath in paired:
        g = load_img(gpath)
        m = load_img(mpath)
        if g is None or m is None:
            # skip corrupted/unreadable images
            continue
        # if sizes differ, resize method to gt
        if g.shape != m.shape:
            from PIL import Image
            m = np.array(Image.fromarray(m).resize((g.shape[1], g.shape[0]), resample=Image.BILINEAR))
        # PSNR
        try:
            p = psnr_sk(g, m, data_range=255)
        except Exception:
            # fallback
            mse = np.mean((g.astype(np.float32)-m.astype(np.float32))**2)
            p = 20 * np.log10(255.0 / np.sqrt(mse)) if mse!=0 else 100.0
        psnrs.append(p)
        # SSIM (multichannel)
        try:
            # try standard call (newer skimage uses channel_axis)
            try:
                s = ssim_sk(g, m, data_range=255, channel_axis=2)
            except TypeError:
                # fallback for older versions
                s = ssim_sk(g, m, data_range=255, multichannel=True)
        except ValueError as e:
            # handle small images where default win_size is too large
            try:
                h, w = g.shape[0], g.shape[1]
                # choose largest odd win_size <= min(h,w)
                max_side = min(h, w)
                win = max_side if (max_side % 2 == 1) else max_side - 1
                if win < 3:
                    win = 3
                try:
                    s = ssim_sk(g, m, data_range=255, channel_axis=2, win_size=win)
                except TypeError:
                    s = ssim_sk(g, m, data_range=255, multichannel=True, win_size=win)
            except Exception:
                # as a last resort, compute SSIM on grayscale with minimal window
                try:
                    from skimage.color import rgb2gray
                    g_gray = (rgb2gray(g) * 255).astype(np.uint8)
                    m_gray = (rgb2gray(m) * 255).astype(np.uint8)
                    s = ssim_sk(g_gray, m_gray, data_range=255)
                except Exception:
                    s = 0.0
        ssims.append(s)
        # LPIPS
        if have_lpips and lpips_model is not None:
            try:
                val = compute_lpips(g, m, lpips_model)
                lpips_vals.append(val)
            except Exception as e:
                # on error, disable lpips
                print('    LPIPS error:', e)
                have_lpips = False
                lpips_vals = []

    avg_psnr = float(np.mean(psnrs)) if psnrs else None
    avg_ssim = float(np.mean(ssims)) if ssims else None
    avg_lpips = float(np.mean(lpips_vals)) if lpips_vals else None

    # FID: if pytorch_fid available, compute using temporary directories with paired images
    fid_val = None
    if have_pytorch_fid:
        try:
            tmp_gt = Path(tempfile.mkdtemp())
            tmp_md = Path(tempfile.mkdtemp())
            # copy paired images to tmp dirs with same names
            for i,(gpath, mpath) in enumerate(paired):
                shutil.copy(gpath, tmp_gt / (gpath.name))
                shutil.copy(mpath, tmp_md / (gpath.name))
            # calculate fid via CLI to avoid multiprocessing issues on Windows
            try:
                import subprocess, re
                # try to locate pytorch-fid CLI in user scripts folder
                import pathlib
                user_scripts = pathlib.Path.home() / 'AppData' / 'Roaming' / 'Python' / f'Python{sys.version_info.major}{sys.version_info.minor}' / 'Scripts'
                cli = user_scripts / 'pytorch-fid.exe'
                if cli.exists():
                    cmd = [str(cli), str(tmp_gt), str(tmp_md), "--device", device]
                else:
                    cmd = ["pytorch-fid", str(tmp_gt), str(tmp_md), "--device", device]
                # set env var to avoid OpenMP duplicate lib crash on Windows
                env = os.environ.copy()
                env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
                out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, env=env)
                # try to parse a float from output
                m = re.search(r"FID\s*[:=]\s*([0-9]+\.?[0-9]*)", out)
                if not m:
                    # fallback: any float
                    m = re.search(r"([-+]?[0-9]*\.?[0-9]+)", out)
                if m:
                    fid_val = float(m.group(1))
                else:
                    fid_val = None
            except Exception as e:
                print('  FID computation failed:', e)
                fid_val = None
        except Exception as e:
            print('  FID setup failed:', e)
            fid_val = None
        finally:
            try:
                shutil.rmtree(tmp_gt)
                shutil.rmtree(tmp_md)
            except Exception:
                pass

    results.append({
        'method': method,
        'pairs': len(paired),
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'lpips': avg_lpips,
        'fid': fid_val,
    })

# save CSV
out_csv = Path('metrics_results.csv')
with out_csv.open('w', newline='', encoding='utf-8') as f:
    import csv
    w = csv.DictWriter(f, fieldnames=['method','pairs','psnr','ssim','lpips','fid'])
    w.writeheader()
    for row in results:
        w.writerow(row)

print('\nDone. Results written to', out_csv)
print('Summary:')
for r in results:
    print(f"{r['method']}: pairs={r['pairs']}, PSNR={r['psnr']}, SSIM={r['ssim']}, LPIPS={r['lpips']}, FID={r['fid']}")
