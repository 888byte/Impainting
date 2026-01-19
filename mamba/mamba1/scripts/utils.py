\
import re
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any

import numpy as np
import yaml
from tqdm import tqdm

# Optional: scikit-image for rgb2lab
from skimage import color


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_color_log_records(txt_path: str) -> List[dict]:
    """
    Parse color log like:
      01Temperature: 23.97 C       Humidity: 62.69 %RH
      NO.1   R:196  G:242  B:242
      ...
      NO.9 ...
    We only keep records that contain all 9 NO lines right after the temperature line.
    """
    lines = Path(txt_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    recs = []
    i = 0
    temp_re = re.compile(r"(\d+)\s*Temperature:\s*([0-9.]+)\s*C\s*Humidity:\s*([0-9.]+)")
    rgb_re = re.compile(r"NO\.(\d+)\s+R:(\d+)\s+G:(\d+)\s+B:(\d+)")
    while i < len(lines):
        m = temp_re.match(lines[i].strip())
        if m:
            hour_idx = int(m.group(1))
            temp = float(m.group(2))
            hum = float(m.group(3))
            rgbs = {}
            ok = True
            for j in range(1, 10):
                if i + j >= len(lines):
                    ok = False
                    break
                m2 = rgb_re.match(lines[i + j].strip())
                if not m2:
                    ok = False
                    break
                no = int(m2.group(1))
                rgbs[no] = (int(m2.group(2)), int(m2.group(3)), int(m2.group(4)))
            if ok and len(rgbs) == 9:
                rec = {"hour": hour_idx, "temp": temp, "hum": hum}
                for no in range(1, 10):
                    rec[f"rgb{no}"] = rgbs[no]
                recs.append(rec)
                i += 10
                continue
        i += 1
    return recs


def make_monotonic_time(hours: List[int], eps: float = 0.01) -> np.ndarray:
    """
    Reconstruct a monotonic time axis from the log's hour-like index.
    Heuristic:
      - if hour decreases, we assume day rollover (+24)
      - if hour repeats, add small epsilon to keep strict monotonicity
    """
    times = []
    day_offset = 0.0
    prev = None
    same = 0
    for h in hours:
        if prev is None:
            prev = h
            same = 0
        else:
            if h < prev:
                day_offset += 24.0
                same = 0
            elif h == prev:
                same += 1
            else:
                same = 0
        times.append(day_offset + float(h) + same * eps)
        prev = h
    return np.asarray(times, dtype=np.float32)


def rgb_series_to_lab(rgb_series_uint8: np.ndarray) -> np.ndarray:
    """
    rgb_series_uint8: (T,3) in [0..255]
    return lab: (T,3) with L in [0..100], a,b in ~[-128..127]
    """
    rgb = rgb_series_uint8.astype(np.float32) / 255.0
    lab = color.rgb2lab(rgb.reshape(-1, 1, 3), illuminant="D65", observer="2").reshape(-1, 3).astype(np.float32)
    return lab


def lab_norm(lab: np.ndarray) -> np.ndarray:
    """
    Normalize Lab for stable learning.
    L: /100
    a,b: /128
    """
    out = lab.copy().astype(np.float32)
    out[:, 0] /= 100.0
    out[:, 1:] /= 128.0
    return out


def lab_unnorm(lab_n: np.ndarray) -> np.ndarray:
    out = lab_n.copy().astype(np.float32)
    out[:, 0] *= 100.0
    out[:, 1:] *= 128.0
    return out


# ---- Excel fast reader (open workbook once) ----
def open_workbook(path: str):
    import openpyxl
    return openpyxl.load_workbook(path, read_only=True, data_only=True)


def read_two_columns_numeric(wb, sheet_name: str, col0=1, col1=2):
    """
    Read numeric rows from two columns (1-indexed) in an Excel sheet.
    Returns (x, y) float32 arrays.
    """
    ws = wb[sheet_name]
    xs = []
    ys = []
    for row in ws.iter_rows(min_row=1, values_only=True):
        a = row[col0 - 1] if len(row) >= col0 else None
        b = row[col1 - 1] if len(row) >= col1 else None
        try:
            xa = float(a)
            yb = float(b)
        except (TypeError, ValueError):
            continue
        if math.isfinite(xa) and math.isfinite(yb):
            xs.append(xa)
            ys.append(yb)
    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)
    if x.size == 0:
        raise ValueError(f"No numeric spectrum found in sheet: {sheet_name}")
    # sort by x
    order = np.argsort(x)
    return x[order], y[order]


def interp_to_grid(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    # out-of-range -> 0
    return np.interp(grid, x, y, left=0.0, right=0.0).astype(np.float32)


def normalize_spec(y: np.ndarray) -> np.ndarray:
    y = y.astype(np.float32)
    y = y - float(np.min(y))
    n = float(np.linalg.norm(y)) + 1e-8
    return (y / n).astype(np.float32)


def power_iteration_L(D: np.ndarray, n_iter: int = 5) -> float:
    # approximate largest eigenvalue of D^T D
    K = D.shape[1]
    v = np.random.randn(K).astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-8)
    for _ in range(n_iter):
        v = D.T @ (D @ v)
        v /= (np.linalg.norm(v) + 1e-8)
    return float(v.T @ (D.T @ (D @ v)))


def fista_nonneg_lasso(D: np.ndarray, r: np.ndarray, lam: float, n_iter: int = 250) -> np.ndarray:
    """
    Solve: min_{a>=0} 0.5||r - D a||^2 + lam ||a||_1
    using FISTA with nonnegative L1 proximal: a = max(0, x - lam*eta)
    """
    Dt = D.T
    L = power_iteration_L(D, n_iter=5)
    eta = 1.0 / (L + 1e-8)

    K = D.shape[1]
    a = np.zeros(K, dtype=np.float32)
    y = a.copy()
    t = 1.0

    for _ in range(n_iter):
        grad = Dt @ (D @ y - r)  # (K,)
        x = y - eta * grad
        a_next = np.maximum(0.0, x - lam * eta).astype(np.float32)
        t_next = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = a_next + ((t - 1.0) / t_next) * (a_next - a)
        a, t = a_next, t_next
    return a
