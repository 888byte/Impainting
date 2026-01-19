#根据训练好的权重生成lut（直接映射之前颜色和现在颜色的表）
"""

"""

import os, json, math, re, subprocess, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

CKPT = "ckpt/pigment_lab_raman_xrd_v2/best_model.pt"
LIB  = "data/standard_alignment/library_embeddings.npz"
COND_METHOD = "pred"
NUM_SAMPLES = 14          # 8~16 is usually enough for LUT
BETA = 8.0
N = 33

OUT_NPZ  = "pigment_lut33.npz"
DONE_NPY = "pigment_lut33_done.npy"  # resume marker

# Parallel settings
MAX_WORKERS  = 45   # you want 10
MAX_INFLIGHT = 225   # keep ~5x workers tasks inflight to avoid huge futures list
SAVE_EVERY   = 300  # save every N completed points
RETRIES      = 3    # retry per point if infer fails

# Use current env python (recommended). If you insist, change to "python".
PYTHON = "python"

# =========================
# Grid
# =========================
grid = np.linspace(0.0, 255.0, N, dtype=np.float32)

# =========================
# LUT arrays
# =========================
lut_rgb   = np.zeros((N, N, N, 3), dtype=np.uint8)
lut_lab   = np.zeros((N, N, N, 3), dtype=np.float32)
lut_conf  = np.zeros((N, N, N), dtype=np.float32)
lut_std   = np.zeros((N, N, N), dtype=np.float32)
lut_cdiff = np.zeros((N, N, N), dtype=np.float32)
lut_cret  = np.zeros((N, N, N), dtype=np.float32)

# =========================
# Resume: load existing LUT + done
# =========================
if os.path.exists(OUT_NPZ):
    prev = np.load(OUT_NPZ, allow_pickle=True)
    if "lut_rgb" in prev:   lut_rgb[:]   = prev["lut_rgb"]
    if "lut_lab" in prev:   lut_lab[:]   = prev["lut_lab"]
    if "lut_conf" in prev:  lut_conf[:]  = prev["lut_conf"]
    if "lut_std" in prev:   lut_std[:]   = prev["lut_std"]
    if "lut_cdiff" in prev: lut_cdiff[:] = prev["lut_cdiff"]
    if "lut_cret" in prev:  lut_cret[:]  = prev["lut_cret"]
    if "grid" in prev:
        gprev = prev["grid"].astype(np.float32)
        if gprev.shape == grid.shape and np.allclose(gprev, grid):
            print("[Resume] Loaded LUT arrays from existing npz (grid matches).")
        else:
            print("[Warn] Existing npz grid differs from current grid. Be careful.")

if os.path.exists(DONE_NPY):
    done = np.load(DONE_NPY)
    assert done.shape == (N, N, N)
else:
    done = np.zeros((N, N, N), dtype=np.uint8)

# =========================
# Infer + parse
# =========================
json_pat = re.compile(r"\{.*\}", re.DOTALL)

def run_infer(rgb, retries=RETRIES):
    # rgb: (r,g,b) floats
    rgb_str = f"{rgb[0]:.5g},{rgb[1]:.5g},{rgb[2]:.5g}"
    cmd = [
        PYTHON, "-m", "pigment_task.infer_pigment",
        "--ckpt", CKPT,
        "--rgb", rgb_str,
        "--cond_method", COND_METHOD,
        "--library_npz", LIB,
        "--num_samples", str(NUM_SAMPLES),
        "--kalman_rts",
    ]
    print(" ".join(cmd))
    last_err = None
    for attempt in range(1, retries + 1):
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            last_err = RuntimeError(f"infer failed attempt {attempt}/{retries}: {p.stderr[:300]}")
            continue

        m = list(json_pat.finditer(p.stdout))
        if not m:
            last_err = RuntimeError(f"no json found attempt {attempt}/{retries}: {p.stdout[:300]}")
            continue

        try:
            return json.loads(m[-1].group(0))
        except Exception as e:
            last_err = RuntimeError(f"json parse error attempt {attempt}/{retries}: {e}")
            continue

    raise last_err if last_err else RuntimeError("infer failed with unknown error")

def fuse_conf(obj):
    c_diff = float(obj.get("confidence_diffusion", 0.0))
    s      = float(obj.get("diffusion_std_norm_meanL2", 0.0))
    c_ret  = float(obj.get("confidence_retrieval", 0.0))
    c = c_diff * math.exp(-BETA * s) * (0.5 + 0.5 * c_ret)
    c = max(0.0, min(1.0, c))
    return c, c_diff, s, c_ret

def save_all():
    np.save(DONE_NPY, done)
    np.savez_compressed(
        OUT_NPZ,
        grid=grid,
        lut_rgb=lut_rgb,
        lut_lab=lut_lab,
        lut_conf=lut_conf,
        lut_std=lut_std,
        lut_cdiff=lut_cdiff,
        lut_cret=lut_cret,
        meta=dict(
            ckpt=CKPT, lib=LIB, cond_method=COND_METHOD,
            num_samples=NUM_SAMPLES, beta=BETA, N=N,
            max_workers=MAX_WORKERS, max_inflight=MAX_INFLIGHT,
            save_every=SAVE_EVERY, retries=RETRIES
        )
    )

# =========================
# Worker task
# =========================
def task(i, j, k, r, g, b):
    obj = run_infer((r, g, b))
    rgb_pred = obj["pred_rgb_original"]  # [R,G,B]
    lab_pred = obj["pred_lab_original"]  # [L,a,b]
    c, c_diff, s, c_ret = fuse_conf(obj)
    return i, j, k, rgb_pred, lab_pred, c, c_diff, s, c_ret

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
print(f"[Init] done={int(done.sum())}/{total}, pending={total - int(done.sum())}")

# =========================
# Parallel loop (10 workers)
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
                    i, j, k, rgb_pred, lab_pred, c, c_diff, s, c_ret = fut.result()

                    lut_rgb[i, j, k]   = np.array(rgb_pred, dtype=np.uint8)
                    lut_lab[i, j, k]   = np.array(lab_pred, dtype=np.float32)
                    lut_conf[i, j, k]  = np.float32(c)
                    lut_std[i, j, k]   = np.float32(s)
                    lut_cdiff[i, j, k] = np.float32(c_diff)
                    lut_cret[i, j, k]  = np.float32(c_ret)

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
                    print(f"[Save] +{completed_this_run} this run | total_done={int(done.sum())}/{total} "
                          f"| failures={failures} | elapsed={elapsed:.1f} min")

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
    print(f"       done={int(done.sum())}/{total}, completed_this_run={completed_this_run}, failures={failures}")
    print(f"       elapsed={elapsed:.1f} min")
