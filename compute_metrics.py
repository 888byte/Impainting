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
import re

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

# Allow skipping heavy model loads via environment (useful for debugging)
if os.environ.get('SKIP_LPIPS', '').lower() in ('1', 'true', 'yes'):
    have_lpips = False
if os.environ.get('SKIP_FID', '').lower() in ('1', 'true', 'yes'):
    have_pytorch_fid = False

# 若要暂时跳过 LPIPS，可设置环境变量 SKIP_LPIPS=1

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


# Function to compute mask coverage (percentage of white pixels in mask image)
def mask_coverage(mask_path):
    try:
        m = Image.open(mask_path).convert('L')
        arr = np.array(m)
        white = int((arr > 128).sum())
        total = int(arr.size)
        if total == 0:
            return None
        return float(white) * 100.0 / float(total)
    except Exception as e:
        print(f"  [WARN] Failed to load mask {mask_path}: {e}")
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
    # list method image files
    method_files = [p for p in method_path.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg')]
    method_map = {p.stem: p for p in method_files}

    paired = []
    # handle RFR-style img_N mapping separately
    img_style_files = []
    for p in method_files:
        m = re.match(r'^img_(\d+)$', p.stem)
        if m:
            img_style_files.append((int(m.group(1)), p))

    if img_style_files:
        # sort by numeric index
        img_style_files.sort(key=lambda x: x[0])
        img_paths = [p for (_, p) in img_style_files]
        pair_count = min(len(img_paths), len(gt_files))
        for i in range(pair_count):
            g = gt_files[i]
            mfile = img_paths[i]
            # mask percent lookup by gt filename
            mpercent = None
            mask_dir = base / 'mask'
            if mask_dir.exists():
                for mf in mask_dir.iterdir():
                    if not mf.is_file():
                        continue
                    if mf.stem.startswith(g.stem):
                        mpercent = mask_coverage(mf)
                        break
            paired.append((g, mfile, mpercent))
    else:
        # general matching: exact stem match or method file stem startswith gt stem
        for g in gt_files:
            key = g.stem
            found = None
            if key in method_map:
                found = method_map[key]
            else:
                # prefix match (e.g., LRDiff: 000098_bottom_f startswith 000098_bottom)
                for mf in method_files:
                    if mf.stem.startswith(key):
                        found = mf
                        break
            mpercent = None
            mask_dir = base / 'mask'
            if mask_dir.exists():
                for mf in mask_dir.iterdir():
                    if not mf.is_file():
                        continue
                    if mf.stem.startswith(key):
                        mpercent = mask_coverage(mf)
                        break
            if found is not None:
                paired.append((g, found, mpercent))

    print(f'  Found {len(paired)} paired images out of {len(gt_files)}')

    psnrs = []
    ssims = []
    lpips_vals = []

    # keep only successfully loaded pairs for FID
    valid_pairs = []

    # prepare mask bins and store file-pairs per bin for per-bin FID
    bins = {'10-20': [], '20-30': [], '30+': []}
    bin_pairs = {'10-20': [], '20-30': [], '30+': []}

    for gpath, mpath, mpercent in paired:
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

        # record this valid pair for later FID computation
        valid_pairs.append((gpath, mpath))

        # append to bin if mask percentage known (store metrics and file pairs)
        try:
            if mpercent is not None:
                lpval = lpips_vals[-1] if lpips_vals else None
                if 10.0 <= mpercent < 20.0:
                    bins['10-20'].append((p, s, lpval))
                    bin_pairs['10-20'].append((gpath, mpath))
                elif 20.0 <= mpercent < 30.0:
                    bins['20-30'].append((p, s, lpval))
                    bin_pairs['20-30'].append((gpath, mpath))
                elif mpercent >= 30.0:
                    bins['30+'].append((p, s, lpval))
                    bin_pairs['30+'].append((gpath, mpath))
        except Exception:
            pass

    avg_psnr = float(np.mean(psnrs)) if psnrs else None
    avg_ssim = float(np.mean(ssims)) if ssims else None
    avg_lpips = float(np.mean(lpips_vals)) if lpips_vals else None

    # FID: if pytorch_fid available, compute using temporary directories with successfully loaded paired images
    fid_val = None
    if have_pytorch_fid:
        if len(valid_pairs) > 0:
            try:
                tmp_gt = Path(tempfile.mkdtemp())
                tmp_md = Path(tempfile.mkdtemp())
                # copy paired images to tmp dirs with same names
                for gpath, mpath in valid_pairs:
                    shutil.copy(gpath, tmp_gt / gpath.name)
                    shutil.copy(mpath, tmp_md / gpath.name)

                # try API call first (use num_workers=0 on Windows to avoid multiprocessing)
                try:
                    old_kmp = os.environ.get('KMP_DUPLICATE_LIB_OK')
                    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
                    # batch_size 50 is reasonable; num_workers=0 for Win
                    fid_value = fid_score.calculate_fid_given_paths([str(tmp_gt), str(tmp_md)], batch_size=50, device=device, dims=2048, num_workers=0)
                    fid_val = float(fid_value)
                except Exception as e_api:
                    print('  FID API failed:', e_api)
                    # fallback to CLI if API fails
                    try:
                        import subprocess, re, pathlib
                        user_scripts = pathlib.Path.home() / 'AppData' / 'Roaming' / 'Python' / f'Python{sys.version_info.major}{sys.version_info.minor}' / 'Scripts'
                        cli = user_scripts / 'pytorch-fid.exe'
                        if cli.exists():
                            cmd = [str(cli), str(tmp_gt), str(tmp_md), "--device", device]
                        else:
                            cmd = ["pytorch-fid", str(tmp_gt), str(tmp_md), "--device", device]
                        env = os.environ.copy()
                        env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
                        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, env=env)
                        m = re.search(r"FID\s*[:=]\s*([0-9]+\.?[0-9]*)", out)
                        if not m:
                            m = re.search(r"([-+]?[0-9]*\.?[0-9]+)", out)
                        if m:
                            fid_val = float(m.group(1))
                        else:
                            fid_val = None
                    except Exception as e_cli:
                        print('  FID CLI failed:', e_cli)
                        fid_val = None
                finally:
                    # restore KMP env
                    if old_kmp is None:
                        os.environ.pop('KMP_DUPLICATE_LIB_OK', None)
                    else:
                        os.environ['KMP_DUPLICATE_LIB_OK'] = old_kmp
            except Exception as e:
                print('  FID setup failed:', e)
                fid_val = None
            finally:
                try:
                    shutil.rmtree(tmp_gt)
                    shutil.rmtree(tmp_md)
                except Exception:
                    pass
        else:
            fid_val = None

    # compute per-bin FID (may be slow); require at least 2 samples per bin to attempt
    bin_fids = {'10-20': None, '20-30': None, '30+': None}
    if have_pytorch_fid:
        for bname, pairs_list in bin_pairs.items():
            if not pairs_list or len(pairs_list) < 2:
                bin_fids[bname] = None
                continue
            try:
                tmp_gt_b = Path(tempfile.mkdtemp())
                tmp_md_b = Path(tempfile.mkdtemp())
                for gpath, mpath in pairs_list:
                    try:
                        shutil.copy(gpath, tmp_gt_b / gpath.name)
                        shutil.copy(mpath, tmp_md_b / gpath.name)
                    except Exception:
                        # skip problematic files
                        pass
                try:
                    old_kmp = os.environ.get('KMP_DUPLICATE_LIB_OK')
                    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
                    batch = min(50, max(1, len(list(tmp_gt_b.iterdir()))))
                    val = fid_score.calculate_fid_given_paths([str(tmp_gt_b), str(tmp_md_b)], batch_size=batch, device=device, dims=2048, num_workers=0)
                    bin_fids[bname] = float(val)
                except Exception as e_api:
                    print(f'  Bin FID API failed for {method} {bname}:', e_api)
                    # try CLI fallback
                    try:
                        import subprocess, re, pathlib
                        user_scripts = pathlib.Path.home() / 'AppData' / 'Roaming' / 'Python' / f'Python{sys.version_info.major}{sys.version_info.minor}' / 'Scripts'
                        cli = user_scripts / 'pytorch-fid.exe'
                        if cli.exists():
                            cmd = [str(cli), str(tmp_gt_b), str(tmp_md_b), "--device", device]
                        else:
                            cmd = ["pytorch-fid", str(tmp_gt_b), str(tmp_md_b), "--device", device]
                        env = os.environ.copy()
                        env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
                        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, env=env)
                        m = re.search(r"FID\s*[:=]\s*([0-9]+\.?[0-9]*)", out)
                        if not m:
                            m = re.search(r"([-+]?[0-9]*\.?[0-9]+)", out)
                        if m:
                            bin_fids[bname] = float(m.group(1))
                        else:
                            bin_fids[bname] = None
                    except Exception as e_cli:
                        print(f'  Bin FID CLI failed for {method} {bname}:', e_cli)
                        bin_fids[bname] = None
                finally:
                    if old_kmp is None:
                        os.environ.pop('KMP_DUPLICATE_LIB_OK', None)
                    else:
                        os.environ['KMP_DUPLICATE_LIB_OK'] = old_kmp
            except Exception as e:
                print(f'  Bin FID setup failed for {method} {bname}:', e)
                bin_fids[bname] = None
            finally:
                try:
                    shutil.rmtree(tmp_gt_b)
                    shutil.rmtree(tmp_md_b)
                except Exception:
                    pass

    # compute per-bin stats
    bin_stats = {}
    for bname, items in bins.items():
        if not items:
            bin_stats[bname] = {'count': 0, 'psnr': None, 'ssim': None, 'lpips': None, 'fid': None}
        else:
            ps = [it[0] for it in items]
            ss = [it[1] for it in items]
            lp = [it[2] for it in items if it[2] is not None]
            bin_stats[bname] = {
                'count': len(items),
                'psnr': float(np.mean(ps)) if ps else None,
                'ssim': float(np.mean(ss)) if ss else None,
                'lpips': float(np.mean(lp)) if lp else None,
                'fid': float(bin_fids.get(bname)) if bin_fids.get(bname) is not None else None,
            }

    results.append({
        'method': method,
        'pairs': len(valid_pairs),
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'lpips': avg_lpips,
        'fid': fid_val,
        'bins': bin_stats,
    })

# save CSV
out_csv = Path('metrics_results.csv')
with out_csv.open('w', newline='', encoding='utf-8') as f:
    import csv
    w = csv.DictWriter(f, fieldnames=['method','pairs','psnr','ssim','lpips','fid'])
    w.writeheader()
    for row in results:
        # write only top-level fields
        w.writerow({
            'method': row.get('method'),
            'pairs': row.get('pairs'),
            'psnr': row.get('psnr'),
            'ssim': row.get('ssim'),
            'lpips': row.get('lpips'),
            'fid': row.get('fid'),
        })

# write per-bin stats to JSON and CSV for detailed inspection
import json
bins_out = Path('metrics_bins.json')
with bins_out.open('w', encoding='utf-8') as f:
    json.dump({r['method']: r.get('bins') for r in results}, f, ensure_ascii=False, indent=2)

bins_csv = Path('metrics_bins.csv')
with bins_csv.open('w', newline='', encoding='utf-8') as f:
    import csv
    writer = csv.DictWriter(f, fieldnames=['method','bin','count','psnr','ssim','lpips','fid'])
    writer.writeheader()
    for r in results:
        bins = r.get('bins') or {}
        for bname, stats in bins.items():
            writer.writerow({
                'method': r.get('method'),
                'bin': bname,
                'count': stats.get('count'),
                'psnr': stats.get('psnr'),
                'ssim': stats.get('ssim'),
                'lpips': stats.get('lpips'),
                'fid': stats.get('fid'),
            })

print('\nDone. Results written to', out_csv, ',', bins_out, 'and', bins_csv)
print('Summary:')
for r in results:
    print(f"{r['method']}: pairs={r['pairs']}, PSNR={r['psnr']}, SSIM={r['ssim']}, LPIPS={r['lpips']}, FID={r['fid']}")
    print('  Bin stats:', r.get('bins'))
