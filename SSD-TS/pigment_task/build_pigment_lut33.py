# -*- coding: utf-8 -*-
"""pigment_lut_build.py

python  pigment_task/build_pigment_lut33.py 

Generate a 3D LUT (current RGB -> predicted original RGB/Lab + confidence) by
calling `python infer.py` on a regular RGB grid.

==============================
IMPORTANT NOTES / PITFALLS
==============================
1) LUT axis order (very important)
   This script stores LUT values as lut[i, j, k] == lut[R_index, G_index, B_index]
   (i.e., the first dimension corresponds to R, second to G, third to B).

   Your APPLY code MUST use the same convention when doing trilinear interpolation.
   If your apply uses lut[B, G, R] (common in some .cube readers), colors will be
   severely wrong (channel/axis swap).

2) `infer.py` already returns the LUT-facing fields we need
   This script now reads `rgb/lab/conf/std/cdiff/cret` directly instead of
   recomputing confidence from legacy fields.

3) Missing optional diagnostics are tolerated
   `cret` may still be missing for some condition paths. In that case we store NaN.

4) Performance / stability
   - Spawning a Python process per grid point is inherently expensive.
   - Too many workers will thrash CPU/GPU/IO and may hang.
   Defaults below are set for stability; tune upward carefully.

5) Data types
   - `lut_rgb` is stored as uint8 (0..255) for compactness.
   - `lut_lab` and confidence-related arrays are float32.

Output
------
An .npz file containing:
  grid, lut_rgb, lut_lab, lut_conf, lut_std, lut_cdiff, lut_cret, meta
and a .npy "done" marker for resume.
"""

import os
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import numpy as np

# =========================
# User config
# =========================
CKPT = "/home/610-wws/Impainting/SSD-TS/ckpt/pigment_lab_raman_xrd/best_model.pt"
LIB = "data/standard_alignment/library_embeddings.npz"
COND_METHOD = "pred"
NUM_SAMPLES = 14  # 8~16 is usually enough for LUT
N = 33

OUT_NPZ = "pigment_lut33.npz"
DONE_NPY = "pigment_lut33_done.npy"  # resume marker

# Axis order used by this file (see docstring). Keep consistent with apply.
AXIS_ORDER = "RGB"  # stored as [R, G, B] in lut[r_idx, g_idx, b_idx]

# Parallel settings (stable defaults)
MAX_WORKERS = 40
MAX_INFLIGHT = 200  # ~2-3x workers is enough
SAVE_EVERY = 300
RETRIES = 5
SUBPROCESS_TIMEOUT_SEC = 90

# Logging
VERBOSE_CMD = True  # printing every command is extremely slow

# Use current env python (recommended). If you insist, change to "python".
PYTHON = "python"

# `infer.py --rgb` already returns the fused confidence (`conf`) plus the
# component diagnostics that should be persisted into the LUT.

# =========================
# Grid
# =========================
grid = np.linspace(0.0, 255.0, N, dtype=np.float32)

# =========================
# LUT arrays
# =========================
lut_rgb = np.zeros((N, N, N, 3), dtype=np.uint8)
# Lab is continuous; keep float32
lut_lab = np.zeros((N, N, N, 3), dtype=np.float32)
# confidence / diagnostics
lut_conf = np.zeros((N, N, N), dtype=np.float32)
# diffusion_std_norm_meanL2
lut_std = np.zeros((N, N, N), dtype=np.float32)
# confidence_diffusion
lut_cdiff = np.zeros((N, N, N), dtype=np.float32)
# confidence_retrieval (may be missing)
lut_cret = np.zeros((N, N, N), dtype=np.float32)

# =========================
# Resume: load existing LUT + done
# =========================
if os.path.exists(OUT_NPZ):
    prev = np.load(OUT_NPZ, allow_pickle=True)
    if "lut_rgb" in prev and prev["lut_rgb"].shape == lut_rgb.shape:
        lut_rgb[:] = prev["lut_rgb"]
    if "lut_lab" in prev and prev["lut_lab"].shape == lut_lab.shape:
        lut_lab[:] = prev["lut_lab"]
    if "lut_conf" in prev and prev["lut_conf"].shape == lut_conf.shape:
        lut_conf[:] = prev["lut_conf"]
    if "lut_std" in prev and prev["lut_std"].shape == lut_std.shape:
        lut_std[:] = prev["lut_std"]
    if "lut_cdiff" in prev and prev["lut_cdiff"].shape == lut_cdiff.shape:
        lut_cdiff[:] = prev["lut_cdiff"]
    if "lut_cret" in prev and prev["lut_cret"].shape == lut_cret.shape:
        lut_cret[:] = prev["lut_cret"]

    if "grid" in prev:
        gprev = prev["grid"].astype(np.float32)
        if gprev.shape == grid.shape and np.allclose(gprev, grid):
            print("[Resume] Loaded LUT arrays from existing npz (grid matches).")
        else:
            print("[Warn] Existing npz grid differs from current grid. Proceed with caution.")

if os.path.exists(DONE_NPY):
    done = np.load(DONE_NPY)
    if done.shape != (N, N, N):
        print("[Warn] DONE marker shape mismatch; resetting done marker.")
        done = np.zeros((N, N, N), dtype=np.uint8)
else:
    done = np.zeros((N, N, N), dtype=np.uint8)

# =========================
# Infer + parse
# =========================
# More robust than r"\{.*\}" which is greedy and may grab logs.
# We look for the LAST JSON object in stdout by finding the last '{' and parsing.
# This assumes infer_pigment prints a JSON dict at the end (as in your example).

_json_start_pat = re.compile(r"\{")


def _parse_last_json(stdout: str) -> dict:
    if not stdout:
        raise ValueError("empty stdout")
    # Find last '{'
    starts = [m.start() for m in _json_start_pat.finditer(stdout)]
    if not starts:
        raise ValueError("no '{' found in stdout")
    last = starts[-1]
    s = stdout[last:].strip()
    # Try direct parse; if fails, try to trim trailing noise
    try:
        return json.loads(s)
    except Exception:
        # fallback: find last '}' and parse substring
        end = s.rfind("}")
        if end == -1:
            raise
        return json.loads(s[: end + 1])


def run_infer(rgb, retries=RETRIES):
    """Call infer_pigment for one RGB point.

    rgb: tuple(float r, float g, float b) in [0,255]
    """
    rgb_str = f"{rgb[0]:.5g},{rgb[1]:.5g},{rgb[2]:.5g}"
    cmd = [
        PYTHON,
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "infer.py"),
        "--ckpt",
        CKPT,
        "--rgb",
        rgb_str,
        "--cond_method",
        COND_METHOD,
        "--library_npz",
        LIB,
        "--num_samples",
        str(NUM_SAMPLES),
        "--kalman_rts",
    ]
    if VERBOSE_CMD:
        print(" ".join(cmd))

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            p = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=SUBPROCESS_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired as e:
            last_err = RuntimeError(f"infer timeout {attempt}/{retries}: {e}")
            continue

        if p.returncode != 0:
            last_err = RuntimeError(
                f"infer failed attempt {attempt}/{retries}: {p.stderr[:300]}"
            )
            continue

        try:
            return _parse_last_json(p.stdout)
        except Exception as e:
            last_err = RuntimeError(
                f"json parse error attempt {attempt}/{retries}: {str(e)[:200]}"
            )
            continue

    raise last_err if last_err else RuntimeError("infer failed with unknown error")




def _atomic_save_npz(path: str, **kwargs):
    # NOTE: numpy appends ".npz" automatically if the filename doesn't end with it.
    # So the temp filename must end with ".npz", otherwise os.replace() will fail.
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **kwargs)
    os.replace(tmp, path)


def save_all():
    np.save(DONE_NPY, done)
    _atomic_save_npz(
        OUT_NPZ,
        grid=grid,
        lut_rgb=lut_rgb,
        lut_lab=lut_lab,
        lut_conf=lut_conf,
        lut_std=lut_std,
        lut_cdiff=lut_cdiff,
        lut_cret=lut_cret,
        meta=dict(
            axis_order=AXIS_ORDER,
            ckpt=CKPT,
            lib=LIB,
            cond_method=COND_METHOD,
            num_samples=NUM_SAMPLES,
            N=N,
            max_workers=MAX_WORKERS,
            max_inflight=MAX_INFLIGHT,
            save_every=SAVE_EVERY,
            retries=RETRIES,
            timeout_sec=SUBPROCESS_TIMEOUT_SEC,
        ),
    )


# =========================
# Worker task
# =========================

def task(i, j, k, r, g, b):
    obj = run_infer((r, g, b))

    rgb_pred = obj["rgb"]
    lab_pred = obj["lab"]
    c = float(obj.get("conf") or 0.0)
    c_diff = float(obj.get("cdiff") or 0.0)
    s = float(obj.get("std") or 0.0)

    c_ret_raw = obj.get("cret", None)
    c_ret = float("nan") if c_ret_raw is None else float(c_ret_raw)

    rgb_u8 = np.clip(np.array(rgb_pred, dtype=np.float32), 0.0, 255.0).round().astype(
        np.uint8
    )
    lab_f32 = np.array(lab_pred, dtype=np.float32)

    return (
        i,
        j,
        k,
        rgb_u8,
        lab_f32,
        np.float32(c),
        np.float32(c_diff),
        np.float32(s),
        np.float32(c_ret),
    )


# =========================
# Build pending iterator
# =========================

def pending_points():
    for i, r in enumerate(grid):
        for j, g in enumerate(grid):
            for k, b in enumerate(grid):
                if done[i, j, k]:
                    continue
                yield (i, j, k, float(r), float(g), float(b))


total = N * N * N
done_count = int(done.sum())
print(f"[Init] done={done_count}/{total}, pending={total - done_count}")

# =========================
# Parallel loop
# =========================
completed_this_run = 0
failures = 0
t0 = time.time()

point_iter = pending_points()
inflight = {}  # future -> (i,j,k,r,g,b)

try:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        # prefill inflight
        while len(inflight) < MAX_INFLIGHT:
            try:
                pt = next(point_iter)
            except StopIteration:
                break
            fut = ex.submit(task, *pt)
            inflight[fut] = pt

        while inflight:
            done_futs, _ = wait(inflight.keys(), return_when=FIRST_COMPLETED)

            for fut in done_futs:
                pt = inflight.pop(fut)

                try:
                    (
                        i,
                        j,
                        k,
                        rgb_u8,
                        lab_f32,
                        c,
                        c_diff,
                        s,
                        c_ret,
                    ) = fut.result()

                    lut_rgb[i, j, k] = rgb_u8
                    lut_lab[i, j, k] = lab_f32
                    lut_conf[i, j, k] = c
                    lut_std[i, j, k] = s
                    lut_cdiff[i, j, k] = c_diff
                    lut_cret[i, j, k] = c_ret

                    done[i, j, k] = 1
                    completed_this_run += 1

                except Exception as e:
                    failures += 1
                    # do NOT mark done, it will be retried on next run
                    print(f"[Fail] idx={pt[:3]} rgb={pt[3:]} err={str(e)[:200]}")

                # periodic save
                if completed_this_run > 0 and (completed_this_run % SAVE_EVERY == 0):
                    save_all()
                    elapsed = (time.time() - t0) / 60.0
                    print(
                        f"[Save] +{completed_this_run} this run | total_done={int(done.sum())}/{total} "
                        f"| failures={failures} | elapsed={elapsed:.1f} min"
                    )

                # refill inflight
                while len(inflight) < MAX_INFLIGHT:
                    try:
                        pt2 = next(point_iter)
                    except StopIteration:
                        break
                    fut2 = ex.submit(task, *pt2)
                    inflight[fut2] = pt2

finally:
    # final save even if interrupted
    save_all()
    elapsed = (time.time() - t0) / 60.0
    print(f"[Done] saved to {OUT_NPZ}")
    print(
        f"       done={int(done.sum())}/{total}, completed_this_run={completed_this_run}, failures={failures}"
    )
    print(f"       elapsed={elapsed:.1f} min")
