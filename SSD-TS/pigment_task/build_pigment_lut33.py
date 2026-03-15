# -*- coding: utf-8 -*-
"""Build a 3D LUT by querying the current `infer.py --rgb` pipeline.

The generated `.npz` stores:
- `grid`: sampling grid in [0, 255]
- `lut_rgb`: predicted restored RGB for each grid point
- `lut_lab`: predicted restored Lab for each grid point
- `lut_conf`: fused confidence from the current inference pipeline
- `lut_std`: diffusion uncertainty scalar
- `lut_cdiff`: diffusion confidence term
- `lut_cret`: retrieval/bridge confidence term

This script is resume-safe. If `out_npz` or `done_npy` already exists, finished grid
points will be skipped automatically.
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

import numpy as np

_JSON_START_PAT = re.compile(r"\{")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build LUT from the current infer.py RGB pipeline")
    ap.add_argument("--ckpt", type=str, default="ckpt/lab_raman_xrd/best_model.pt")
    ap.add_argument("--library_npz", type=str, default="data/standard_alignment/library_embeddings.npz")
    ap.add_argument("--prototype_bank", type=str, default="data/pigment_npz/prototype_bank.npz")
    ap.add_argument(
        "--cond_method",
        type=str,
        default="auto",
        choices=["auto", "pred", "retrieval", "posterior", "posterior_retrieval"],
    )
    ap.add_argument("--num_samples", type=int, default=14)
    ap.add_argument("--grid_size", type=int, default=33)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_npz", type=str, default="pigment_lut33.npz")
    ap.add_argument("--done_npy", type=str, default="")
    ap.add_argument("--max_workers", type=int, default=40)
    ap.add_argument("--max_inflight", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=300)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--heartbeat_sec", type=int, default=30)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--timeout_sec", type=int, default=90)
    ap.add_argument("--python_exe", type=str, default="python")
    ap.add_argument("--verbose_cmd", action="store_true")
    ap.add_argument("--kalman_rts", action="store_true")
    ap.add_argument("--lut_order", type=str, default="RGB", choices=["RGB"])
    return ap.parse_args()


def _parse_last_json(stdout: str) -> dict:
    if not stdout:
        raise ValueError("empty stdout")
    starts = [m.start() for m in _JSON_START_PAT.finditer(stdout)]
    if not starts:
        raise ValueError("no '{' found in stdout")
    candidate = stdout[starts[-1] :].strip()
    try:
        return json.loads(candidate)
    except Exception:
        end = candidate.rfind("}")
        if end == -1:
            raise
        return json.loads(candidate[: end + 1])


def _atomic_save_npz(path: Path, **kwargs) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(tmp, **kwargs)
    os.replace(tmp, path)


def _build_infer_command(args: argparse.Namespace, rgb: tuple[float, float, float]) -> list[str]:
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        args.python_exe,
        str(repo_root / "infer.py"),
        "--ckpt",
        args.ckpt,
        "--device",
        args.device,
        "--rgb",
        f"{rgb[0]:.5g},{rgb[1]:.5g},{rgb[2]:.5g}",
        "--cond_method",
        args.cond_method,
        "--num_samples",
        str(args.num_samples),
    ]
    if args.library_npz:
        cmd.extend(["--library_npz", args.library_npz])
    if args.prototype_bank:
        cmd.extend(["--prototype_bank", args.prototype_bank])
    if args.kalman_rts:
        cmd.append("--kalman_rts")
    return cmd


def run_infer(args: argparse.Namespace, rgb: tuple[float, float, float]) -> dict:
    cmd = _build_infer_command(args, rgb)
    if args.verbose_cmd:
        print(" ".join(cmd))

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
            last_err = RuntimeError(f"infer timeout {attempt}/{args.retries}: {exc}")
            continue

        if proc.returncode != 0:
            last_err = RuntimeError(
                f"infer failed attempt {attempt}/{args.retries}: {proc.stderr[:300]}"
            )
            continue

        try:
            return _parse_last_json(proc.stdout)
        except Exception as exc:
            last_err = RuntimeError(
                f"json parse error attempt {attempt}/{args.retries}: {str(exc)[:200]}"
            )
            continue

    raise last_err if last_err else RuntimeError("infer failed with unknown error")


def load_resume_arrays(args: argparse.Namespace, grid: np.ndarray):
    n = len(grid)
    lut_rgb = np.zeros((n, n, n, 3), dtype=np.uint8)
    lut_lab = np.zeros((n, n, n, 3), dtype=np.float32)
    lut_conf = np.zeros((n, n, n), dtype=np.float32)
    lut_std = np.zeros((n, n, n), dtype=np.float32)
    lut_cdiff = np.zeros((n, n, n), dtype=np.float32)
    lut_cret = np.zeros((n, n, n), dtype=np.float32)

    out_npz = Path(args.out_npz)
    done_npy = Path(args.done_npy) if args.done_npy else out_npz.with_name(out_npz.stem + "_done.npy")

    if out_npz.exists():
        prev = np.load(out_npz, allow_pickle=True)
        try:
            for key, arr in {
                "lut_rgb": lut_rgb,
                "lut_lab": lut_lab,
                "lut_conf": lut_conf,
                "lut_std": lut_std,
                "lut_cdiff": lut_cdiff,
                "lut_cret": lut_cret,
            }.items():
                if key in prev and prev[key].shape == arr.shape:
                    arr[:] = prev[key]
            if "grid" in prev:
                gprev = prev["grid"].astype(np.float32)
                if gprev.shape == grid.shape and np.allclose(gprev, grid):
                    print("[Resume] Loaded existing LUT arrays (grid matches).")
                else:
                    print("[Warn] Existing LUT grid differs from current grid.")
        finally:
            prev.close()

    if done_npy.exists():
        done = np.load(done_npy)
        if done.shape != (n, n, n):
            print("[Warn] done marker shape mismatch, resetting it.")
            done = np.zeros((n, n, n), dtype=np.uint8)
    else:
        done = np.zeros((n, n, n), dtype=np.uint8)

    return {
        "out_npz": out_npz,
        "done_npy": done_npy,
        "lut_rgb": lut_rgb,
        "lut_lab": lut_lab,
        "lut_conf": lut_conf,
        "lut_std": lut_std,
        "lut_cdiff": lut_cdiff,
        "lut_cret": lut_cret,
        "done": done,
    }


def save_all(args: argparse.Namespace, grid: np.ndarray, state: dict) -> None:
    np.save(state["done_npy"], state["done"])
    _atomic_save_npz(
        state["out_npz"],
        grid=grid,
        lut_rgb=state["lut_rgb"],
        lut_lab=state["lut_lab"],
        lut_conf=state["lut_conf"],
        lut_std=state["lut_std"],
        lut_cdiff=state["lut_cdiff"],
        lut_cret=state["lut_cret"],
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
            max_workers=int(args.max_workers),
            max_inflight=int(args.max_inflight),
            save_every=int(args.save_every),
            log_every=int(args.log_every),
            heartbeat_sec=int(args.heartbeat_sec),
            retries=int(args.retries),
            timeout_sec=int(args.timeout_sec),
        ),
    )


def task(args: argparse.Namespace, i: int, j: int, k: int, r: float, g: float, b: float):
    obj = run_infer(args, (r, g, b))

    rgb_pred = np.clip(np.array(obj["rgb"], dtype=np.float32), 0.0, 255.0).round().astype(np.uint8)
    lab_pred = np.array(obj["lab"], dtype=np.float32)
    conf = np.float32(float(obj.get("conf") or 0.0))
    cdiff = np.float32(float(obj.get("cdiff") or 0.0))
    std = np.float32(float(obj.get("std") or 0.0))

    c_ret_raw = obj.get("cret", None)
    cret = np.float32(np.nan if c_ret_raw is None else float(c_ret_raw))

    return i, j, k, rgb_pred, lab_pred, conf, cdiff, std, cret


def pending_points(grid: np.ndarray, done: np.ndarray):
    for i, r in enumerate(grid):
        for j, g in enumerate(grid):
            for k, b in enumerate(grid):
                if done[i, j, k]:
                    continue
                yield i, j, k, float(r), float(g), float(b)


def main() -> None:
    args = parse_args()
    grid = np.linspace(0.0, 255.0, int(args.grid_size), dtype=np.float32)
    state = load_resume_arrays(args, grid)

    total = int(args.grid_size) ** 3
    done_count = int(state["done"].sum())
    print(f"[Init] done={done_count}/{total}, pending={total - done_count}")

    completed_this_run = 0
    failures = 0
    started = time.time()

    point_iter = pending_points(grid, state["done"])
    inflight: dict = {}
    initial_submitted = 0
    log_every = max(1, int(args.log_every))
    heartbeat_sec = max(1, int(args.heartbeat_sec))
    last_heartbeat = time.time()

    try:
        with ThreadPoolExecutor(max_workers=int(args.max_workers)) as executor:
            while len(inflight) < int(args.max_inflight):
                try:
                    pt = next(point_iter)
                except StopIteration:
                    break
                fut = executor.submit(task, args, *pt)
                inflight[fut] = pt
                initial_submitted += 1

            print(
                f"[Launch] submitted initial={initial_submitted} | workers={int(args.max_workers)} "
                f"| inflight_cap={int(args.max_inflight)} | save_every={int(args.save_every)} "
                f"| log_every={log_every} | heartbeat={heartbeat_sec}s"
            )

            while inflight:
                done_futs, _ = wait(
                    inflight.keys(),
                    timeout=float(heartbeat_sec),
                    return_when=FIRST_COMPLETED,
                )
                if not done_futs:
                    elapsed = (time.time() - started) / 60.0
                    print(
                        f"[Heartbeat] total_done={int(state['done'].sum())}/{total} "
                        f"| completed_this_run={completed_this_run} | inflight={len(inflight)} "
                        f"| failures={failures} | elapsed={elapsed:.1f} min"
                    )
                    last_heartbeat = time.time()
                    continue
                for fut in done_futs:
                    pt = inflight.pop(fut)
                    try:
                        i, j, k, rgb_u8, lab_f32, conf, cdiff, std, cret = fut.result()
                        state["lut_rgb"][i, j, k] = rgb_u8
                        state["lut_lab"][i, j, k] = lab_f32
                        state["lut_conf"][i, j, k] = conf
                        state["lut_cdiff"][i, j, k] = cdiff
                        state["lut_std"][i, j, k] = std
                        state["lut_cret"][i, j, k] = cret
                        state["done"][i, j, k] = 1
                        completed_this_run += 1
                        if completed_this_run == 1 or completed_this_run % log_every == 0:
                            elapsed = (time.time() - started) / 60.0
                            print(
                                f"[Progress] total_done={int(state['done'].sum())}/{total} "
                                f"| completed_this_run={completed_this_run} | inflight={len(inflight)} "
                                f"| failures={failures} | elapsed={elapsed:.1f} min"
                            )
                    except Exception as exc:
                        failures += 1
                        print(f"[Fail] idx={pt[:3]} rgb={pt[3:]} err={str(exc)[:200]}")

                    if completed_this_run > 0 and completed_this_run % int(args.save_every) == 0:
                        save_all(args, grid, state)
                        elapsed = (time.time() - started) / 60.0
                        print(
                            f"[Save] +{completed_this_run} this run | total_done={int(state['done'].sum())}/{total} "
                            f"| failures={failures} | elapsed={elapsed:.1f} min"
                        )

                    while len(inflight) < int(args.max_inflight):
                        try:
                            pt2 = next(point_iter)
                        except StopIteration:
                            break
                        fut2 = executor.submit(task, args, *pt2)
                        inflight[fut2] = pt2
                if time.time() - last_heartbeat >= heartbeat_sec:
                    elapsed = (time.time() - started) / 60.0
                    print(
                        f"[Heartbeat] total_done={int(state['done'].sum())}/{total} "
                        f"| completed_this_run={completed_this_run} | inflight={len(inflight)} "
                        f"| failures={failures} | elapsed={elapsed:.1f} min"
                    )
                    last_heartbeat = time.time()
    finally:
        save_all(args, grid, state)
        elapsed = (time.time() - started) / 60.0
        print(f"[Done] saved to {state['out_npz']}")
        print(
            f"       done={int(state['done'].sum())}/{total}, completed_this_run={completed_this_run}, failures={failures}"
        )
        print(f"       elapsed={elapsed:.1f} min")


if __name__ == "__main__":
    main()
